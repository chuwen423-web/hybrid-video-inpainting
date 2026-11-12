"""视频管线（调整后的 Week 3）：检测 → 确认 → SAM 2 跟踪 → 逐帧删除（Telea / LaMa）→ 贴回原帧 → 输出视频。

  1. 抽帧          work/<clip>/frames/00000.jpg ...（SAM 2 直接读这个目录）
  2. 检测          YOLOX（OpenCV dnn）逐帧检测目标类别 → work/<clip>/detections.json
  3. 片段 + 关键帧  检测到目标的帧合并成片段（clip_D：杯子只在中段出现）；挑置信度最高、互相间隔 ≥1 秒的帧作关键帧
  4. 确认          results/<clip>_confirm.png：关键帧检测框 + 时间轴。目标按 requirements.txt 确认（config 里可改）
  5. SAM 2 跟踪     关键帧上的检测框作为提示，正向 + 反向传播；片段（±边距）以外强制为空
                   需要用户补充的目标（clip_C 的纸袋不是 COCO 类别）用 config 里的 manual 提示
  6. 逐帧删除      mask 外扩 → 只裁 mask 周围的区域修补（Telea / LaMa）→ 羽化后贴回原帧（mask 外与原视频完全一致）
  7. 输出          results/<clip>__telea.mp4 / __lama.mp4 / __compare.mp4（原视频 | mask | Telea | LaMa）

用法：python pipeline.py [clip_A clip_C ...] [--skip-sam2]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
WEEK2 = os.path.join(PROJ, "week2_mask_tracking")
sys.path.append(os.path.join(PROJ, "week1_image_validation"))
from tools import mask_ops  # noqa: E402  第一周的 feather / overlay

from detect import YOLOX, find_segments  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")


# ---------- 读写（兼容中文路径） ----------
def imread(p, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(p, np.uint8), flags)
    if img is None:
        raise FileNotFoundError(p)
    return img


def imwrite(p, img, q=95):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ext = os.path.splitext(p)[1].lower()
    cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else [])[1].tofile(p)


def wdir(cid, *a):
    return os.path.join(ROOT, "work", cid, *a)


def load_cfg():
    with open(os.path.join(ROOT, "config", "clips.json"), encoding="utf-8") as f:
        return json.load(f)


def frame_paths(cid):
    d = wdir(cid, "frames")
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


def write_video(path, frames, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


# ---------- 1. 抽帧 ----------
def extract(cid, c, cfg):
    d = wdir(cid, "frames")
    src = os.path.join(ROOT, cfg["source_dir"], c["video"])
    cap = cv2.VideoCapture(src)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.isdir(d) and len(os.listdir(d)) >= n - 2:
        return fps
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        imwrite(os.path.join(d, f"{i:05d}.jpg"), f, 95)
        i += 1
    print(f"[{cid}] 抽帧 {i} 帧 @ {fps:.1f} fps")
    return fps


# ---------- 2-4. 检测、片段、关键帧、确认图 ----------
def detect_all(cid, det):
    """逐帧检测全部 COCO 类别并缓存；返回 [[{"cls","score","box"}...] 每帧]"""
    p = wdir(cid, "detections.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    dets = []
    for fp in frame_paths(cid):
        dets.append([{"cls": c, "score": round(s, 3), "box": [round(float(v), 1) for v in b]} for c, s, b in det(imread(fp))])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dets, f)
    return dets


def of_class(dets, cls):
    return [sorted((d for d in ds if d["cls"] == cls), key=lambda d: -d["score"]) for ds in dets]


def pick_keyframes(dets, segs, fps, k=6, min_conf=0.6):
    cand = sorted(((d[0]["score"], i) for i, d in enumerate(dets) if d and d[0]["score"] >= min_conf), reverse=True)
    keys = []
    for s, i in cand:
        if any(a <= i <= b for a, b in segs) and all(abs(i - j) >= fps for j in keys):
            keys.append(i)
        if len(keys) >= k * max(1, len(segs)):
            break
    for a, b in segs:  # 每个片段至少一个关键帧
        if not any(a <= j <= b for j in keys):
            best = max((dets[i][0]["score"], i) for i in range(a, b + 1) if dets[i]) if any(dets[i] for i in range(a, b + 1)) else None
            if best:
                keys.append(best[1])
    return sorted(keys)


def confirm_sheet(cid, dets, segs, keys, fps, target, manual):
    frames = frame_paths(cid)
    n = len(frames)
    tiles = []
    for i in keys[:6]:
        img = imread(frames[i])
        for d in dets[i]:
            x0, y0, x1, y1 = map(int, d["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 4)
            cv2.putText(img, f"{target} {d['score']:.2f}", (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        for m in manual:
            if m["frame"] == i:
                if m.get("box"):
                    x0, y0, x1, y1 = m["box"]
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 160, 0), 4)
                for (px, py), lab in zip(m.get("points", []), m.get("labels", [])):
                    cv2.circle(img, (px, py), 14, (0, 220, 0) if lab else (0, 0, 255), -1)
        img = cv2.resize(img, (270, 480))
        cv2.rectangle(img, (0, 0), (270, 26), (25, 25, 25), -1)
        cv2.putText(img, f"key frame {i} ({i / fps:.1f}s)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    row = np.hstack(tiles)
    bar = np.full((60, row.shape[1], 3), 245, np.uint8)
    for i, d in enumerate(dets):
        if d:
            x = int(i / n * row.shape[1])
            cv2.line(bar, (x, 8), (x, 30), (60, 170, 60), 1)
    for a, b in segs:
        cv2.rectangle(bar, (int(a / n * row.shape[1]), 34), (int(b / n * row.shape[1]), 50), (230, 120, 40), -1)
    cv2.putText(bar, f"green: frames with '{target}' detected   blue: segments processed   ({n} frames, {n / fps:.1f}s)",
                (6, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
    imwrite(os.path.join(ROOT, "results", f"{cid}_confirm.png"), np.vstack([row, bar]))


# ---------- 5. SAM 2 ----------
_SAM2 = {}


def sam2_predictor():
    if "p" not in _SAM2:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        with open(os.path.join(WEEK2, "config", "clips.json"), encoding="utf-8") as f:
            m = json.load(f)["sam2_models"]["large"]
        _SAM2["p"] = build_sam2_video_predictor(m["config"], os.path.join(WEEK2, m["checkpoint"]),
                                                device="cuda" if torch.cuda.is_available() else "cpu")
    return _SAM2["p"]


def run_sam2(cid, prompts, n, span_by_obj):
    """prompts: {obj_id: [{"frame", "box"?, "points"?, "labels"?}]}；返回 {obj_id: [bool HxW]*n}"""
    import torch
    pred = sam2_predictor()
    out = {oid: [None] * n for oid in prompts}
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(video_path=wdir(cid, "frames"), offload_video_to_cpu=True, offload_state_to_cpu=n > 500)
        first = min(p["frame"] for ps in prompts.values() for p in ps)
        for oid, ps in prompts.items():
            for p in ps:
                kw = {}
                if p.get("box") is not None:
                    kw["box"] = np.array(p["box"], np.float32)
                if p.get("points"):
                    kw["points"] = np.array(p["points"], np.float32)
                    kw["labels"] = np.array(p["labels"], np.int32)
                pred.add_new_points_or_box(state, frame_idx=p["frame"], obj_id=oid, **kw)
        for reverse in (False, True):
            for fi, oids, logits in pred.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
                for j, oid in enumerate(oids):
                    out[oid][fi] = (logits[j, 0] > 0).cpu().numpy()
        pred.reset_state(state)
    dt = time.perf_counter() - t0
    shape = next(m.shape for ms in out.values() for m in ms if m is not None)
    for oid, ms in out.items():
        a, b = span_by_obj[oid]
        out[oid] = [m if (m is not None and a <= i <= b) else np.zeros(shape, bool) for i, m in enumerate(ms)]
    return out, dt


# ---------- 6. 逐帧删除 ----------
_LAMA = {}


def lama_model():
    if "m" not in _LAMA:
        import torch
        import io
        with open(os.path.join(ROOT, "models", "big-lama.pt"), "rb") as f:  # 经内存读：torch.jit.load 不支持含中文的路径
            _LAMA["m"] = torch.jit.load(io.BytesIO(f.read()), map_location="cuda").eval()
    return _LAMA["m"]


def crop_box(mask, shape, ratio=0.6, min_pad=48):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(max(min_pad, ratio * max(x1 - x0, y1 - y0)))
    H, W = shape[:2]
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad + 1), min(H, y1 + pad + 1)


def inpaint_crop(img, mask, method, max_side=768):
    """只修补 mask 周围的裁剪区域，再羽化贴回原图；mask 为 0/255 uint8（已外扩）。"""
    if not mask.any():
        return img.copy()
    x0, y0, x1, y1 = crop_box(mask, img.shape)
    crop, cm = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    s = min(1.0, max_side / max(crop.shape[:2]))
    if s < 1:
        crop_s = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cm_s = cv2.resize(cm, (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        crop_s, cm_s = crop, cm
    if method == "telea":
        fill = cv2.inpaint(crop_s, cm_s, 7, cv2.INPAINT_TELEA)
    else:
        import torch
        h, w = crop_s.shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        im = cv2.copyMakeBorder(crop_s, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        mm = cv2.copyMakeBorder(cm_s, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().cuda() / 255
        m = torch.from_numpy((mm > 127).astype(np.float32))[None, None].cuda()
        with torch.inference_mode():
            r = lama_model()(t, m)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        fill = cv2.cvtColor((r * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)[:h, :w]
    if s < 1:
        fill = cv2.resize(fill, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    a = mask_ops.feather(cm, 4)[..., None]
    out[y0:y1, x0:x1] = (a * fill + (1 - a) * crop).astype(np.uint8)
    return out


def prep_mask(m, dilate_px):
    u = m.astype(np.uint8) * 255
    if not u.any():
        return u
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask_ops.dilate(u, dilate_px)


# ---------- 主流程 ----------
def process(cid, c, cfg, det, skip_sam2=False):
    fps = extract(cid, c, cfg)
    frames = frame_paths(cid)
    n = len(frames)
    target = c["remove"]["target"]
    all_dets = detect_all(cid, det)
    dets = of_class(all_dets, target)
    present = [bool(d) for d in dets]
    segs = find_segments(present, min_len=5, max_gap=8, margin=int(fps * 0.3))
    span = (segs[0][0], segs[-1][1]) if segs else (0, -1)  # 首尾片段之间都交给 SAM 2 判断（检测器会漏检俯视等角度）
    keys = pick_keyframes(dets, segs, fps)
    manual = c.get("manual_prompts", [])
    confirm_sheet(cid, dets, segs, keys, fps, target, manual)
    info = {"clip": cid, "frames": n, "fps": fps, "target": target, "detected_frames": sum(present),
            "segments": segs, "span": list(span), "keyframes": keys,
            "confirmed_by": c["remove"].get("confirmed_by", "test_clips/requirements.txt")}
    print(f"[{cid}] 检测到 '{target}' {sum(present)}/{n} 帧；片段 {segs}；关键帧 {keys}")

    mask_dir = wdir(cid, "masks")
    if not skip_sam2 or not os.path.isdir(mask_dir):
        prompts = {1: [{"frame": i, "box": dets[i][0]["box"]} for i in keys]}
        spans = {1: span}
        for m in manual:  # 用户补充的目标（例如纸袋），obj_id 从 2 开始
            prompts.setdefault(m["obj_id"], []).append(m)
            spans[m["obj_id"]] = (0, n - 1)
        for t in c.get("track_classes", []):  # 额外跟踪的检测类别（例如 clip_C 的手 = person），供替换时处理遮挡
            td = of_class(all_dets, t["class"])
            tsegs = find_segments([bool(d) for d in td], min_len=5, max_gap=8, margin=int(fps * 0.3))
            if tsegs:
                tkeys = pick_keyframes(td, tsegs, fps, k=3, min_conf=0.5)
                prompts[t["obj_id"]] = [{"frame": i, "box": td[i][0]["box"]} for i in tkeys]
                spans[t["obj_id"]] = (tsegs[0][0], tsegs[-1][1])
                info.setdefault("extra_tracks", {})[t["class"]] = {"segments": tsegs, "keyframes": tkeys}
        masks, dt = run_sam2(cid, prompts, n, spans)
        info["sam2_seconds"] = round(dt, 1)
        for oid, ms in masks.items():
            for i, m in enumerate(ms):
                imwrite(os.path.join(mask_dir, f"obj{oid}", f"{i:05d}.png"), m.astype(np.uint8) * 255)
        print(f"[{cid}] SAM 2：{len(prompts)} 个目标，{n} 帧，{dt:.0f} s")

    rm = [imread(os.path.join(mask_dir, "obj1", f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in range(n)]
    info["masked_frames"] = int(sum(m.any() for m in rm))
    vids = {"telea": [], "lama": []}
    times = {}
    comp = []
    for method in vids:
        t0 = time.perf_counter()
        for i, fp in enumerate(frames):
            img = imread(fp)
            out = inpaint_crop(img, prep_mask(rm[i], cfg["dilate_px"]), method)
            imwrite(wdir(cid, f"out_{method}", f"{i:05d}.png"), out)
            vids[method].append(out)
        times[method] = round((time.perf_counter() - t0) / n * 1000, 1)
    info["ms_per_frame"] = times
    for i, fp in enumerate(frames):
        img = imread(fp)
        ov = mask_ops.overlay(img, prep_mask(rm[i], cfg["dilate_px"]))
        row = [img, ov, vids["telea"][i], vids["lama"][i]]
        row = [cv2.resize(x, (360, 640), interpolation=cv2.INTER_AREA) for x in row]
        for x, t in zip(row, ("original", "mask (SAM 2)", "OpenCV Telea", "LaMa")):
            cv2.rectangle(x, (0, 0), (360, 28), (25, 25, 25), -1)
            cv2.putText(x, f"{t}  {i / fps:4.1f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        comp.append(np.hstack(row))
    res = os.path.join(ROOT, "results")
    for method, fs in vids.items():
        write_video(os.path.join(res, f"{cid}__{method}.mp4"), fs, fps)
    write_video(os.path.join(res, f"{cid}__compare.mp4"), comp, fps)
    with open(wdir(cid, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"[{cid}] 删除完成：Telea {times['telea']} ms/帧，LaMa {times['lama']} ms/帧；有 mask 的帧 {info['masked_frames']}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--skip-sam2", action="store_true", help="已有 mask 时跳过 SAM 2")
    args = ap.parse_args()
    cfg = load_cfg()
    det = YOLOX(os.path.join(ROOT, "models", "yolox_s.onnx"), conf=0.25)
    for cid, c in cfg["clips"].items():
        if args.clips and cid not in args.clips:
            continue
        process(cid, c, cfg, det, args.skip_sam2)


if __name__ == "__main__":
    main()
