"""视频结果评测：时间闪烁、保持度、检测 / 片段统计，输出表格、曲线和抽帧对比图。

指标（每段视频 × 每种方法）：
  flicker_med / flicker_p90  编辑区域的帧间变化 ÷ 周围环带的帧间变化（逐帧取比值后的中位数 / 90 分位数）。
                             ≈ 原视频的值：编辑区域和周围一样自然地随画面变化；明显更大：闪烁 / 跳变（Chen 提到的问题）
  max_jump                   编辑区域单帧最大的平均灰度变化（找最明显的一次跳变）
  outside_psnr / changed%    mask 外扩 10 px 以外区域和原视频的差异（贴回原帧后应为 inf / 0）
输出：results/metrics.csv、results/<clip>_flicker.png、results/<clip>_frames.png
"""
import csv
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import ROOT, frame_paths, imread, imwrite, load_cfg, prep_mask, wdir  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731


def load_seq(folder, n, gray=False):
    out = []
    for i in range(n):
        p = os.path.join(folder, f"{i:05d}.png")
        out.append(imread(p, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR) if os.path.exists(p) else None)
    return out


def flicker(frames_gray, masks):
    ratios, jumps = [], []
    for t in range(1, len(frames_gray)):
        E = (masks[t] > 0) | (masks[t - 1] > 0)
        if E.sum() < 200:
            ratios.append(np.nan)
            jumps.append(np.nan)
            continue
        ring = (cv2.dilate(E.astype(np.uint8), K(25)) > 0) & ~E
        d = cv2.absdiff(frames_gray[t], frames_gray[t - 1]).astype(np.float32)
        de, dr = d[E].mean(), d[ring].mean()
        ratios.append(de / max(dr, 0.5))
        jumps.append(de)
    return np.array(ratios), np.array(jumps)


def preservation(orig, out, masks):
    sse = cnt = changed = 0
    for o, r, m in zip(orig, out, masks):
        outside = cv2.dilate(m, K(10)) == 0
        d = cv2.absdiff(o, r)[outside].astype(np.float64)
        sse += (d ** 2).sum()
        cnt += d.size
        changed += (d.max(axis=1) > 10).sum()
    mse = sse / max(cnt, 1)
    return (float("inf") if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)), round(100 * changed / max(cnt / 3, 1), 3)


def main():
    cfg = load_cfg()
    rows = []
    for cid in cfg["clips"]:
        ip = wdir(cid, "info.json")
        if not os.path.exists(ip):
            continue
        info = json.load(open(ip, encoding="utf-8"))
        n, fps = info["frames"], info["fps"]
        orig = [imread(p) for p in frame_paths(cid)]
        orig_g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in orig]
        rm = [prep_mask(imread(os.path.join(wdir(cid, "masks", "obj1"), f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127,
                        cfg["dilate_px"]) for i in range(n)]
        methods = {"original": orig}
        for m in ("telea", "lama", "lama+book"):
            d = wdir(cid, f"out_{m}")
            if os.path.isdir(d):
                methods[m] = load_seq(d, n)
        masks_for = {k: rm for k in methods}
        if "lama+book" in methods:  # 替换任务：编辑区域 = 杯子 mask ∪ 书的合成区域
            bm = load_seq(wdir(cid, "book_region"), n, gray=True)
            masks_for["lama+book"] = [np.maximum(a, b if b is not None else 0) for a, b in zip(rm, bm)]
        curves = {}
        for name, seq in methods.items():
            g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in seq] if name != "original" else orig_g
            r, j = flicker(g, masks_for[name])
            curves[name] = r
            psnr, ch = preservation(orig, seq, masks_for[name]) if name != "original" else ("-", "-")
            rows.append({"clip": cid, "method": name, "frames": n, "masked_frames": info["masked_frames"],
                         "detected_frames": info["detected_frames"], "segments": info["segments"],
                         "flicker_med": round(float(np.nanmedian(r)), 3) if np.isfinite(r).any() else "",
                         "flicker_p90": round(float(np.nanpercentile(r, 90)), 3) if np.isfinite(r).any() else "",
                         "max_jump": round(float(np.nanmax(j)), 2) if np.isfinite(j).any() else "",
                         "outside_psnr": psnr, "outside_changed_pct": ch,
                         "ms_per_frame": info.get("ms_per_frame", {}).get(name.split("+")[0], "")})
        # 闪烁曲线
        plt.figure(figsize=(8, 2.8))
        tt = np.arange(1, n) / fps
        for name, r in curves.items():
            plt.plot(tt, r, lw=1.6 if name == "lama" else 1.0, label=name, ls="--" if name == "original" else "-")
        for a, b in info["segments"]:
            plt.axvspan(a / fps, b / fps, color="0.92", zorder=0)
        plt.yscale("log"); plt.xlabel("秒"); plt.ylabel("编辑区域帧间变化 ÷ 周围")
        plt.title(f"{cid}：闪烁（≈ 原视频为自然；灰色 = 检测到目标的片段）", fontsize=10)
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "results", f"{cid}_flicker.png"), dpi=110); plt.close()
        # 抽帧对比：只取有 mask 的帧
        idx = [i for i in range(n) if rm[i].any()]
        picks = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, 4)] if idx else []
        rows_img = []
        for i in picks:
            tiles = [orig[i]] + [methods[m][i] for m in ("telea", "lama", "lama+book") if m in methods]
            tiles = [cv2.resize(x, (270, 480), interpolation=cv2.INTER_AREA) for x in tiles]
            for x, lab in zip(tiles, ["original"] + [m for m in ("telea", "lama", "lama+book") if m in methods]):
                cv2.rectangle(x, (0, 0), (270, 24), (25, 25, 25), -1)
                cv2.putText(x, f"{lab}  {i / fps:.1f}s", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows_img.append(np.hstack(tiles))
        if rows_img:
            imwrite(os.path.join(ROOT, "results", f"{cid}_frames.png"), np.vstack(rows_img))
    keys = ["clip", "method", "frames", "masked_frames", "detected_frames", "segments", "flicker_med", "flicker_p90",
            "max_jump", "outside_psnr", "outside_changed_pct", "ms_per_frame"]
    with open(os.path.join(ROOT, "results", "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f"{'clip':8s} {'method':10s} {'mask帧':>7s} {'闪烁中位':>8s} {'闪烁p90':>8s} {'最大跳变':>8s} {'外PSNR':>8s} {'外变化%':>8s}")
    for r in rows:
        print(f"{r['clip']:8s} {r['method']:10s} {r['masked_frames']:>7} {r['flicker_med']:>8} {r['flicker_p90']:>8} "
              f"{r['max_jump']:>8} {str(r['outside_psnr']):>8} {str(r['outside_changed_pct']):>8}")


if __name__ == "__main__":
    main()
