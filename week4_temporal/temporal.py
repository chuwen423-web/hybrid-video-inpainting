"""Week 4：低成本去闪烁、去光晕、逐物体修补（只用 OpenCV + 每段一两次 LaMa，不下载新模型）。

三个低成本办法（输入沿用第三周的抽帧和 SAM 2 mask：week3_video_pipeline/work/<clip>/）：
  ③ 稳定 mask + 去光晕（Zhihan）：杯子 mask 加上贴着杯子的阴影 → 闭运算 → 外扩 → 前后 2 帧取并集（mask 边缘不再逐帧抖动）
  ① 背景板（Mingran）：用 ORB 特征 + 单应矩阵把每帧对齐到参考帧，杯子不在的帧露出的真实背景取中位数拼成一块“背景板”；
     所有帧都没露出过的地方，只在背景板上用 LaMa 补一次。之后每帧把背景板按镜头运动变形回来填进 mask，
     再按 mask 外一圈的亮度做增益补偿（时间上平滑）。补丁每帧都是同一块 → 不闪；杯子被拿走时填的是真实背景
  ② 光流时间平滑（Mingran，对照组）：第三周的逐帧 LaMa 结果，mask 内和“按光流对齐的上一帧结果”加权平均
逐物体修补（Zhihan，Chen 的建议）：clip_C 的杯子和纸袋各用自己的 mask、各自补洞、各自做增益补偿，和合并成一个大 mask 对比。

输出：work/<clip>/out_<method>/、results/<clip>__*.mp4（H.264，浏览器能直接播放）、results/metrics.csv、对比图。
用法：python temporal.py [clip_A clip_C clip_D]（用第二周的 .venv 运行）
"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
W3 = os.path.join(PROJ, "week3_video_pipeline")
sys.path.insert(0, W3)
from pipeline import imread, imwrite, inpaint_crop, mask_ops  # noqa: E402  第三周的读写和 LaMa 修补
from evaluate_video import flicker  # noqa: E402  第三周的闪烁指标（保证前后可比）

sys.path.insert(0, os.path.join(PROJ, "week1_image_validation"))
from tools.metrics import seam_score  # noqa: E402  第一周的接缝指标（光晕的代理指标）

CLIPS = ["clip_A", "clip_C", "clip_D"]
ALIGN_MAX = 6.0  # 背景板和当前帧在 mask 外一圈的平均差（灰度级）超过这个值，说明对不齐，改用逐帧方法
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
LABEL = {"original": "original", "w3_lama": "Week 3: per-frame LaMa", "lama_flow": "(2) LaMa + flow smoothing",
         "plate": "(1) background plate", "w4": "Week 4: (1)+(3) final", "w4_book": "Week 4 + book"}


def w3(cid, *a):
    return os.path.join(W3, "work", cid, *a)


def wdir(cid, *a):
    return os.path.join(ROOT, "work", cid, *a)


def load_seq(folder, n, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return [imread(os.path.join(folder, f"{i:05d}.png"), flag) for i in range(n)]


def write_mp4(path, frames, fps):
    """H.264（浏览器能直接播放）：用 Windows 自带的 Media Foundation 编码，不需要额外下载；不可用时退回 mp4v。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp = os.path.join(tempfile.gettempdir(), f"w4_{os.getpid()}.mp4")  # Media Foundation 打不开含中文的路径
    vw = cv2.VideoWriter(tmp, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, (w, h))
    if not vw.isOpened():
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    shutil.move(tmp, path)


# ---------- ③ 稳定 mask + 阴影 ----------
def shadow_mask(img, m, ring=45, ratio=0.8):
    """贴着物体、比周围桌面明显暗、而且在物体下半部分的像素（物体在桌面上的阴影 / 接触暗边）。"""
    if not m.any():
        return m
    ys, xs = np.nonzero(m)
    pad = ring + 40  # 只在物体附近的小区域里算，比整帧快很多
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad + 1, max(0, xs.min() - pad), xs.max() + pad + 1
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = _shadow(img[y0:y1, x0:x1], m[y0:y1, x0:x1], ring, ratio)
    return out


def _shadow(img, m, ring, ratio):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    u = m.astype(np.uint8)
    d = cv2.distanceTransform(1 - u, cv2.DIST_L2, 3)  # 到物体的距离；比大核膨胀快得多
    near = (d > 2) & (d <= ring)
    outer = (d > ring) & (d <= ring + 35)
    ys = np.nonzero(m)[0]
    lower = np.zeros_like(m)
    lower[int(np.percentile(ys, 50)):, :] = True
    cand = near & lower & (g < ratio * np.median(g[outer]))
    n, lab = cv2.connectedComponents(cand.astype(np.uint8))
    touch = np.unique(lab[(cv2.dilate(u, K(4)) > 0) & cand])
    sh = np.isin(lab, touch[touch > 0])
    return sh if sh.sum() < 0.6 * m.sum() else np.zeros_like(m)  # 太大说明碰到的是深色物体而不是阴影


def stable_masks(frames, masks, dilate_px=12, win=2, shadow=True):
    base = []
    for img, m in zip(frames, masks):
        if not m.any():
            base.append(np.zeros(m.shape, np.uint8))
            continue
        u = ((m | shadow_mask(img, m)) if shadow else m).astype(np.uint8) * 255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, K(6))
        base.append(cv2.dilate(u, K(dilate_px)))
    n = len(base)
    return [np.max(base[max(0, t - win):t + win + 1], axis=0) for t in range(n)]  # 前后 win 帧取并集


# ---------- ① 背景板 ----------
def homographies(frames, excl, idxs, ref, scale=0.5):
    """每帧 → 参考帧的单应矩阵（只用 mask 以外的背景特征）。直接匹配失败时沿用相邻帧链式累乘。"""
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    S = np.diag([scale, scale, 1.0])
    feats = {}
    for t in idxs:
        g = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        e = cv2.resize(cv2.dilate(excl[t], K(15)), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        feats[t] = orb.detectAndCompute(g, (e == 0).astype(np.uint8) * 255)

    def match(a, b):
        (k1, d1), (k2, d2) = feats[a], feats[b]
        if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
            return None, 0
        m = bf.match(d1, d2)
        if len(m) < 30:
            return None, 0
        Hm, inl = cv2.findHomography(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k2[x.trainIdx].pt for x in m]),
                                     cv2.RANSAC, 2.0)
        return (np.linalg.inv(S) @ Hm @ S, int(inl.sum())) if Hm is not None else (None, 0)

    H, stats = {ref: np.eye(3)}, {"direct": 0, "chained": 0}
    order = sorted(idxs, key=lambda t: abs(t - ref))  # 从参考帧向两边推
    for t in order[1:]:
        Hd, n = match(t, ref)
        if Hd is not None and n >= 60:
            H[t] = Hd
            stats["direct"] += 1
            continue
        # 靠近参考帧一侧、已经算过的最近一帧（中间可能有物体不在画面里、没有 mask 的帧）
        nb = max(k for k in H if k < t) if t > ref else min(k for k in H if k > t)
        Hn, _ = match(t, nb)
        H[t] = H[nb] @ Hn if Hn is not None else H[nb]
        stats["chained"] += 1
    return smooth_homographies(H, frames[0].shape), stats


def smooth_homographies(H, shape, win=7):
    """逐帧估计的单应矩阵有细小抖动，会让补丁跟着抖（闪烁）。把画面四个角的轨迹在时间上取中位数平滑，
    再反算单应矩阵：保留真实的镜头运动，去掉估计噪声。"""
    h, w = shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    ts = sorted(H)
    pts = np.stack([cv2.perspectiveTransform(src[None], H[t])[0] for t in ts])
    out = {}
    for k, t in enumerate(ts):
        seg = pts[max(0, k - win):k + win + 1]
        out[t] = cv2.getPerspectiveTransform(src, np.median(seg, 0).astype(np.float32))
    return out


def warp_region(img, M, size, interp=cv2.INTER_LINEAR):
    return cv2.warpPerspective(img, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def median_plate(frames, samples, H, T, size, excl):
    """把 samples 这些帧对齐到背景板坐标，mask 外（真实背景）的像素取中位数。返回 (背景板, 有效样本数)。"""
    stack = []
    for t in samples:
        M = T @ H[t]
        img = warp_region(frames[t], M, size).astype(np.float32)
        ok = warp_region(((excl[t] == 0) * 255).astype(np.uint8), M, size, cv2.INTER_NEAREST) > 0
        img[~ok] = np.nan
        stack.append(img)
    stack = np.stack(stack)
    count = np.isfinite(stack[..., 0]).sum(0)
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.nanmedian(stack, axis=0)).astype(np.uint8), count


def build_plate(frames, idxs, H, holes, excl, fps, pad=80, max_samples=60, win_s=1.5, stride_s=1.5):
    """参考帧坐标下 holes（要填的区域）外接框 + pad 的背景板。
    整段一块全局板（找出从没露出过的洞）+ 每 stride_s 秒一块局部板（只用前后 win_s 秒的帧）：
    纸袋被推动这类背景本身在变的情况，局部板跟得上；局部板里没露出的地方用全局板补。
    返回 (全局板, 平移矩阵 T, 从未露出的洞, [(中心帧, 局部板, 局部有效)])。"""
    h, w = frames[0].shape[:2]
    corners = []
    for t in idxs:
        ys, xs = np.nonzero(holes[t])
        if len(xs):
            pts = np.float32([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]])[None]
            corners.append(cv2.perspectiveTransform(pts, H[t])[0])
    c = np.concatenate(corners)
    x0, y0 = np.floor(c.min(0) - pad).astype(int)
    x1, y1 = np.ceil(c.max(0) + pad).astype(int)
    x0, y0, x1, y1 = max(x0, -w // 2), max(y0, -h // 2), min(x1, w + w // 2), min(y1, h + h // 2)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (x1 - x0, y1 - y0)
    plate, count = median_plate(frames, idxs[::max(1, len(idxs) // max_samples)], H, T, size, excl)
    win, stride = round(win_s * fps), max(1, round(stride_s * fps))
    local = []
    for cen in range(idxs[0], idxs[-1] + stride, stride):
        samp = [t for t in idxs if abs(t - cen) <= win][::3]
        if samp:
            p, cnt = median_plate(frames, samp, H, T, size, excl)
            local.append((min(cen, idxs[-1]), p, cnt >= 2))
    return plate, T, count < 3, local


def fill_plate(plate, never):
    """背景板上从未露出的洞用 LaMa 补一次（设置同第三周：裁出洞周围、最长边 768）。"""
    if not never.any():
        return plate, 0
    return inpaint_crop(plate, cv2.dilate(never.astype(np.uint8) * 255, K(3)), "lama"), 1


def local_plates(plate, never, local, max_gap=2, min_px=200, still_thr=6.0):
    """背景本身没变（局部板和全局板几乎一样，固定场景 / 只是镜头在动）→ 这一块直接用全局板，不再补洞。
    背景在变（例如 clip_C 纸袋被推动）→ 用局部板，其中没露出的像素：
      1. 先向前后 max_gap 块内的局部板借（只借时间上接近的）；
      2. 整段都没露出过的（never）→ 用全局板那一次 LaMa 的结果；
      3. 别的时间露出过、但离得太远的 → 在这块局部板上单独补一次 LaMa（例如纸袋被推过来之前的桌面）。
    返回 ([(中心帧, 背景板)], 额外的 LaMa 次数, 用了局部板的块数)。"""
    out, calls, used = [], 0, 0
    for k, (c, p, ok) in enumerate(local):
        both = ok & ~never
        diff = np.abs(p.astype(np.int16) - plate.astype(np.int16)).mean(2)[both]
        if not diff.size or np.median(diff) < still_thr:
            out.append((c, plate))
            continue
        used += 1
        img, miss = p.copy(), ~ok
        for d in range(1, max_gap + 1):
            for j in (k - d, k + d):
                if 0 <= j < len(local) and miss.any():
                    take = miss & local[j][2]
                    img[take] = local[j][1][take]
                    miss &= ~take
        img[miss & never] = plate[miss & never]
        rest = miss & ~never
        if rest.sum() >= min_px:
            img = inpaint_crop(img, cv2.dilate(rest.astype(np.uint8) * 255, K(3)), "lama")
            calls += 1
        elif rest.any():
            img[rest] = plate[rest]
        out.append((c, img))
    return out, calls, used


def plate_remove(frames, idxs, H, masks, excl, keep, fps):
    """① 背景板删除（一个物体或一组 mask）：建板 → 补洞 → 局部板 → 逐帧贴回。"""
    t0 = time.perf_counter()
    plate, T, never, local = build_plate(frames, idxs, H, masks, excl, fps)
    plate, calls = fill_plate(plate, never)
    plates, extra, used = local_plates(plate, never, local)
    outs, err = render_plate(frames, idxs, H, plates, T, masks, keep=keep)
    return outs, {"lama_calls": calls + extra, "never_seen_pct": round(100 * float(never.mean()), 1), "align_err": err,
                  "local_plates_used": f"{used}/{len(local)}",
                  "seconds": round(time.perf_counter() - t0, 1)}, plate


def plate_at(t, plates):
    """t 时刻的背景板：相邻两块局部板按时间线性过渡，避免切换时跳一下。"""
    cs = [c for c, _ in plates]
    k = int(np.searchsorted(cs, t))
    if k == 0 or len(plates) == 1:
        return plates[0][1]
    if k >= len(plates):
        return plates[-1][1]
    (c0, p0), (c1, p1) = plates[k - 1], plates[k]
    a = (t - c0) / max(1, c1 - c0)
    return cv2.addWeighted(p0, 1 - a, p1, a, 0)


def ring_gain(frame, fill, m, inner=6, outer=20):
    """mask 外一圈：原帧 / 填充内容 的每通道亮度比（补偿自动曝光和光照变化，去掉亮度不一致造成的光晕）。"""
    u = (m > 127).astype(np.uint8)
    ring = (cv2.dilate(u, K(outer)) > 0) & ~(cv2.dilate(u, K(inner)) > 0) & (fill.sum(2) > 0)
    if ring.sum() < 200:
        return None
    return np.clip(frame[ring].mean(0) / (fill[ring].mean(0) + 1e-3), 0.7, 1.4)


def smooth_gains(gains, win=9):
    g = np.array([x if x is not None else [np.nan] * 3 for x in gains], np.float64)
    out = np.ones_like(g)
    for t in range(len(g)):
        seg = g[max(0, t - win // 2):t + win // 2 + 1]
        seg = seg[np.isfinite(seg[:, 0])]
        out[t] = np.median(seg, 0) if len(seg) else 1.0
    return out


def render_plate(frames, idxs, H, plates, T, masks, keep=None, feather_px=6):
    """每帧：当时的背景板按镜头运动变形回当前帧 → 增益补偿（时间平滑）→ 只在 mask 内羽化贴回。
    keep：要保留在前面的物体（手），这些像素不替换。"""
    h, w = frames[0].shape[:2]
    out = [f.copy() for f in frames]
    fills, gains, errs = {}, [], []
    for t in idxs:
        fills[t] = cv2.warpPerspective(plate_at(t, plates), np.linalg.inv(T @ H[t]), (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        gains.append(ring_gain(frames[t], fills[t], masks[t]) if masks[t].any() else None)
        if masks[t].any() and t % 5 == 0:  # 对齐误差：mask 外一圈，背景板和当前帧差多少（灰度级）
            ring = (cv2.dilate(masks[t], K(20)) > 0) & ~(cv2.dilate(masks[t], K(6)) > 0)
            errs.append(float(np.abs(frames[t].astype(np.int16) - fills[t].astype(np.int16))[ring].mean()))
    align_err = round(float(np.median(errs)), 1) if errs else 0.0
    gains = smooth_gains(gains)
    for t, gn in zip(idxs, gains):
        if not masks[t].any():
            continue
        a = mask_ops.feather(masks[t], feather_px)
        if keep is not None and keep[t].any():
            a = a * (cv2.dilate(keep[t].astype(np.uint8), K(2)) == 0)
        a = a[..., None]
        fill = np.clip(fills[t].astype(np.float32) * gn, 0, 255)
        out[t] = (a * fill + (1 - a) * frames[t]).astype(np.uint8)
    return out, align_err


# ---------- ② 光流时间平滑（对照组） ----------
def flow_smooth(frames, lama, masks, alpha=0.75, scale=0.5):
    """mask 内：alpha × 按光流对齐的上一帧结果 + (1-alpha) × 当前帧 LaMa。mask 内的光流（杯子自己的运动）
    换成 mask 外一圈背景光流的中位数。"""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    h, w = frames[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = [lama[0].copy()]
    prev_g = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    for t in range(1, len(frames)):
        g = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        m = masks[t] > 127
        if not m.any() or not (masks[t - 1] > 127).any():
            out.append(lama[t].copy())
            prev_g = g
            continue
        f = cv2.resize(dis.calc(g, prev_g, None), (w, h)) / scale  # 当前帧每个像素在上一帧的位置
        ring = (cv2.dilate(masks[t], K(30)) > 0) & ~m
        f[m] = np.median(f[ring], 0) if ring.any() else 0
        prev = cv2.remap(out[-1], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = mask_ops.feather(masks[t], 6)[..., None] * alpha
        out.append((a * prev + (1 - a) * lama[t]).astype(np.uint8))
        prev_g = g
    return out


# ---------- 指标 ----------
def metrics(orig, outs, eval_mask, own_masks, idxs):
    """闪烁：统一用 eval_mask（各方法 mask 的并集）保证可比；接缝（光晕代理）：各方法自己的 mask 边界。"""
    og = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in orig]
    rows = {}
    for name, seq in outs.items():
        g = og if name == "original" else [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in seq]
        r, j = flicker(g, eval_mask)
        seam = [seam_score(seq[t], own_masks[name][t])["seam_ratio"] for t in idxs[::5] if own_masks[name][t].any()]
        rows[name] = {"flicker_med": round(float(np.nanmedian(r)), 3), "flicker_p90": round(float(np.nanpercentile(r, 90)), 3),
                      "max_jump": round(float(np.nanmax(j)), 2), "seam_ratio": round(float(np.nanmedian(seam)), 3) if seam else "",
                      "curve": r}
    return rows


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (25, 25, 25), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compare_video(path, seqs, names, fps, size=(360, 640)):
    rows = []
    for t in range(len(seqs[0])):
        rows.append(np.hstack([label(cv2.resize(s[t], size, interpolation=cv2.INTER_AREA), f"{LABEL.get(n, n)}  {t / fps:4.1f}s")
                               for s, n in zip(seqs, names)]))
    write_mp4(path, rows, fps)
    return rows


def crop_sheet(path, seqs, names, masks, picks, r=190):
    """放大看杯子原位置：每行一个时刻，每列一种方法。"""
    rows = []
    for t in picks:
        ys, xs = np.nonzero(masks[t] > 127)
        cx, cy = int(xs.mean()), int(ys.mean())
        h, w = seqs[0][t].shape[:2]
        x0, y0 = min(max(0, cx - r), w - 2 * r), min(max(0, cy - r), h - 2 * r)
        rows.append(np.hstack([label(cv2.resize(s[t][y0:y0 + 2 * r, x0:x0 + 2 * r], (300, 300)), LABEL.get(n, n)[:30])
                               for s, n in zip(seqs, names)]))
    imwrite(path, np.vstack(rows))


# ---------- clip_C：换书 ----------
def composite_book(base, orig, bag, hand, cup, patch, key=0):
    """第三周 replace_book.py 的逐帧合成（书跟着纸袋的运动走、手在前面），底图换成 Week 4 去掉杯子的结果。
    纸袋被推动后露出书外的部分，和第三周一样逐帧用 LaMa 补（这块面积小）。"""
    from replace_book import homography, object_motion  # 第三周的对齐函数
    region0 = cv2.dilate(bag[key].astype(np.uint8), K(12)) > 0
    orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    key_g = cv2.cvtColor(base[key], cv2.COLOR_BGR2GRAY)
    key_excl = cv2.dilate((region0 | hand[key] | cup[key]).astype(np.uint8), K(15)) > 0
    alpha0 = mask_ops.feather(region0.astype(np.uint8) * 255, 5)
    bag_key = cv2.dilate(bag[key].astype(np.uint8), K(4)) > 0
    Hprev, outs, regions = np.eye(3), [], []
    h, w = base[0].shape[:2]
    for t in range(len(base)):
        if t == key:
            Hm = np.eye(3)
        else:
            Hm, ni = object_motion(orig[key], orig[t], bag_key & ~cup[key], (cv2.dilate(bag[t].astype(np.uint8), K(4)) > 0) & ~hand[t] & ~cup[t], orb, bf)
            if Hm is None or ni < 20:
                excl = cv2.dilate((bag[t] | hand[t] | cup[t]).astype(np.uint8), K(15)) > 0
                Hm, ni = homography(key_g, cv2.cvtColor(base[t], cv2.COLOR_BGR2GRAY), key_excl, excl, orb, bf)
            if Hm is None or ni < 25:
                Hm = Hprev
        Hprev = Hm
        book = cv2.warpPerspective(patch, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpPerspective(alpha0, Hm, (w, h), flags=cv2.INTER_LINEAR)
        a *= 1 - (cv2.dilate(hand[t].astype(np.uint8), K(3)) > 0)  # 手在书前面
        out = (a[..., None] * book + (1 - a[..., None]) * base[t]).astype(np.uint8)
        leftover = (cv2.dilate(bag[t].astype(np.uint8), K(6)) > 0) & (a < 0.5) & ~hand[t]
        if leftover.sum() > 150:
            out = inpaint_crop(out, mask_ops.dilate(leftover.astype(np.uint8) * 255, 4), "lama")
        outs.append(out)
        regions.append(((a > 0.05) | leftover).astype(np.uint8) * 255)
    return outs, regions


# ---------- 主流程 ----------
def run_clip(cid, plot=True):
    t_all = time.perf_counter()
    info = json.load(open(w3(cid, "info.json"), encoding="utf-8"))
    n, fps = info["frames"], info["fps"]
    frames = [imread(os.path.join(w3(cid, "frames"), f"{i:05d}.jpg")) for i in range(n)]
    zero = np.zeros(frames[0].shape[:2], bool)

    def obj(k):
        d = w3(cid, "masks", f"obj{k}")
        return [m > 127 for m in load_seq(d, n, gray=True)] if os.path.isdir(d) else [zero] * n

    cup, bag, hand = obj(1), obj(2), obj(3)
    old = [cv2.dilate(cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, K(4)), K(8)) for m in cup]  # 第三周的 mask（外扩 8 px）
    log = {"clip": cid, "frames": n, "fps": fps}
    res = os.path.join(ROOT, "results")
    os.makedirs(res, exist_ok=True)

    # ③ 稳定 mask + 阴影
    t0 = time.perf_counter()
    stab = stable_masks(frames, cup)
    log["ms_masks"] = round((time.perf_counter() - t0) / n * 1000, 1)
    idxs = [t for t in range(n) if stab[t].any()]
    ref = idxs[len(idxs) // 2]
    hand_u = [cv2.dilate(h.astype(np.uint8) * 255, K(10)) for h in hand]
    excl = [np.maximum(s, hd) for s, hd in zip(stab, hand_u)]

    # ① 背景板
    t0 = time.perf_counter()
    H, hstats = homographies(frames, excl, idxs, ref)
    t_h = time.perf_counter() - t0

    def plate_method(masks, ex):
        outs, inf, plate = plate_remove(frames, idxs, H, masks, ex, hand, fps)
        inf["ms_per_frame"] = round((inf.pop("seconds") + t_h) / len(idxs) * 1000, 1)
        return outs, inf, plate

    old_excl = [np.maximum(o, hd) for o, hd in zip(old, hand_u)]
    outs = {"original": frames, "w3_lama": load_seq(w3(cid, "out_lama"), n)}
    t0 = time.perf_counter()
    outs["lama_flow"] = flow_smooth(frames, outs["w3_lama"], old)
    log["lama_flow_ms_per_frame"] = round((time.perf_counter() - t0) / n * 1000 + info["ms_per_frame"]["lama"], 1)
    outs["plate"], log["plate"], _ = plate_method(old, old_excl)
    outs["w4"], log["w4"], plate_img = plate_method(stab, excl)
    log["w4"]["mode"] = "background plate"
    if log["w4"]["align_err"] > ALIGN_MAX:
        # 镜头大幅运动 + 前后景视差：背景板对不齐 → 改用 ③ 的新 mask 逐帧 LaMa + 轻度光流平滑（光晕照样去掉）
        t0 = time.perf_counter()
        lama_new = [inpaint_crop(f, m, "lama") if m.any() else f for f, m in zip(frames, stab)]
        outs["w4"] = flow_smooth(frames, lama_new, stab, alpha=0.5)
        for t in idxs:  # 手保持在前面
            if hand[t].any():
                hk = cv2.dilate(hand[t].astype(np.uint8), K(2)) > 0
                outs["w4"][t][hk] = frames[t][hk]
        log["w4"] = {"mode": "per-frame LaMa + flow (plate rejected)", "plate_align_err": log["w4"]["align_err"],
                     "lama_calls": len(idxs), "ms_per_frame": round((time.perf_counter() - t0) / len(idxs) * 1000, 1)}
    log["w4"]["ms_per_frame"] = round(log["w4"]["ms_per_frame"] + log["ms_masks"], 1)  # 最终方法的耗时包含 ③ 稳定 mask
    log["homography"] = hstats
    log["ref_frame"] = ref
    log["w3_lama_ms_per_frame"] = info["ms_per_frame"]["lama"]
    imwrite(os.path.join(res, f"{cid}_plate.jpg"), plate_img)

    own = {"original": stab, "w3_lama": old, "lama_flow": old, "plate": old, "w4": stab}
    eval_mask = [np.maximum(a, b) for a, b in zip(old, stab)]
    rows = metrics(frames, outs, eval_mask, own, idxs)

    if cid == "clip_C":  # 逐物体修补（杯子 + 纸袋）对比合并 mask；再在最终去杯子的结果上换书
        bagm = [cv2.dilate(cv2.morphologyEx(b.astype(np.uint8) * 255, cv2.MORPH_CLOSE, K(6)), K(10)) for b in bag]
        union = [np.maximum(a, b) for a, b in zip(stab, bagm)]
        ex2 = [np.maximum(u, hd) for u, hd in zip(union, hand_u)]
        ids2 = [t for t in range(n) if union[t].any()]
        H2, _ = homographies(frames, ex2, ids2, ids2[len(ids2) // 2])
        # 一个大 mask：杯子和纸袋一起，一块背景板、一次补洞、一个增益
        uni, iu, _ = plate_remove(frames, ids2, H2, union, ex2, hand, fps)
        # 逐物体：先用纸袋自己的背景板去掉纸袋，再在去掉纸袋的画面上用杯子自己的背景板去掉杯子
        idb = [t for t in range(n) if bagm[t].any()]
        per, ib, _ = plate_remove(frames, idb, H2, bagm, ex2, hand, fps)
        exc = [np.maximum(s, hd) for s, hd in zip(stab, hand_u)]
        per, ic, _ = plate_remove(per, [t for t in ids2 if stab[t].any()], H2, stab, exc, hand, fps)
        r2 = metrics(frames, {"union": uni, "per_object": per}, union, {"union": union, "per_object": union}, ids2)
        for m, nm in ((stab, "cup"), (bagm, "bag")):
            for name, seq in (("union", uni), ("per_object", per)):
                s = [seam_score(seq[t], m[t])["seam_ratio"] for t in ids2[::5] if m[t].any()]
                r2[name][f"seam_{nm}"] = round(float(np.nanmedian(s)), 3)
        r2["union"].update(lama_calls=iu["lama_calls"], seconds=iu["seconds"])
        r2["per_object"].update(lama_calls=ib["lama_calls"] + ic["lama_calls"], seconds=round(ib["seconds"] + ic["seconds"], 1))
        log["per_object"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in r2.items()}
        compare_video(os.path.join(res, "clip_C__per_object_compare.mp4"), [frames, uni, per], ["original", "cup+bag: one mask", "cup+bag: per object"], fps)
        crop_sheet(os.path.join(res, "clip_C_per_object_crops.jpg"), [frames, uni, per], ["original", "one mask (union)", "per object"],
                   union, [ids2[k] for k in np.linspace(5, len(ids2) - 6, 3).astype(int)], r=230)

        patch = imread(os.path.join(w3(cid, "out_lama+book"), "00000.png"))  # 第三周 SDXL 生成、选定的书（第 0 帧）
        book, breg = composite_book(outs["w4"], frames, bag, hand, cup, patch)  # 底图：Week 4 最终的去杯子结果
        old_book = load_seq(w3(cid, "out_lama+book"), n)
        bmask = [np.maximum(a, b) for a, b in zip(union, breg)]
        rb = metrics(frames, {"w3_book": old_book, "w4_book": book}, bmask, {"w3_book": bmask, "w4_book": bmask}, ids2)
        log["book"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in rb.items()}
        outs["w4_book"] = book
        write_mp4(os.path.join(res, "clip_C__w4_book.mp4"), book, fps)
        write_mp4(os.path.join(res, "clip_C__w3_book.mp4"), old_book, fps)  # 第三周结果转成 H.264，演示界面能播放
        compare_video(os.path.join(res, "clip_C__book_compare.mp4"), [frames, old_book, book], ["original", "Week 3: LaMa + book", "w4_book"], fps)

    # 输出
    for name in ("original", "w3_lama", "lama_flow", "plate", "w4"):
        write_mp4(os.path.join(res, f"{cid}__{name}.mp4"), outs[name], fps)
    compare_video(os.path.join(res, f"{cid}__compare_w4.mp4"), [frames, outs["w3_lama"], outs["lama_flow"], outs["w4"]],
                  ["original", "w3_lama", "lama_flow", "w4"], fps)
    picks = [idxs[k] for k in np.linspace(3, len(idxs) - 4, 3).astype(int)]
    crop_sheet(os.path.join(res, f"{cid}_crops.jpg"), [frames, outs["w3_lama"], outs["lama_flow"], outs["plate"], outs["w4"]],
               ["original", "w3_lama", "lama_flow", "plate", "w4"], eval_mask, picks)
    ov = [mask_ops.overlay(mask_ops.overlay(frames[t], stab[t], color=(0, 200, 255)), old[t]) for t in picks]
    imwrite(os.path.join(res, f"{cid}_masks.jpg"),
            np.hstack([label(cv2.resize(o, (360, 640)), f"red: w3  orange: w4 mask  {t / fps:.1f}s") for o, t in zip(ov, picks)]))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 2.8))
        tt = np.arange(1, n) / fps
        for name, r in rows.items():
            plt.plot(tt, r["curve"], lw=1.6 if name == "w4" else 1.0, ls="--" if name == "original" else "-", label=LABEL[name])
        plt.yscale("log")
        plt.xlabel("seconds")
        plt.ylabel("flicker ratio")
        plt.title(f"{cid}: flicker of the edited region (close to original = natural)", fontsize=10)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(res, f"{cid}_flicker_w4.png"), dpi=110)
        plt.close()
    log["metrics"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in rows.items()}
    log["seconds_total"] = round(time.perf_counter() - t_all, 1)
    json.dump(log, open(os.path.join(res, f"{cid}_w4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{cid}] 完成 {log['seconds_total']} s；闪烁中位数 " +
          "，".join(f"{k} {v['flicker_med']}" for k, v in log["metrics"].items()), flush=True)
    return log


def write_csv(logs):
    path = os.path.join(ROOT, "results", "metrics.csv")
    keys = ["clip", "method", "flicker_med", "flicker_p90", "max_jump", "seam_ratio", "ms_per_frame", "lama_calls"]
    old = {}
    if os.path.exists(path):  # 只重跑部分片段时保留其他片段的结果
        with open(path, encoding="utf-8-sig") as f:
            old = {(r["clip"], r["method"]): r for r in csv.DictReader(f)}
    for lg in logs:
        old = {k: v for k, v in old.items() if k[0] != lg["clip"]}
        for name, m in lg["metrics"].items():
            ms = {"w3_lama": lg["w3_lama_ms_per_frame"], "lama_flow": lg["lama_flow_ms_per_frame"],
                  "plate": lg["plate"]["ms_per_frame"], "w4": lg["w4"]["ms_per_frame"]}.get(name, "")
            calls = {"w3_lama": "1 per frame", "lama_flow": "1 per frame", "plate": lg["plate"]["lama_calls"], "w4": lg["w4"]["lama_calls"]}.get(name, "")
            old[(lg["clip"], name)] = {"clip": lg["clip"], "method": name, **{k: m[k] for k in keys[2:6]}, "ms_per_frame": ms, "lama_calls": calls}
        for name, m in lg.get("book", {}).items():
            old[(lg["clip"], name)] = {"clip": lg["clip"], "method": name, **{k: m[k] for k in keys[2:6]}, "ms_per_frame": "", "lama_calls": ""}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: (r["clip"], r["method"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=CLIPS)
    a = ap.parse_args()
    logs = [run_clip(c) for c in a.clips]
    write_csv(logs)


if __name__ == "__main__":
    main()
