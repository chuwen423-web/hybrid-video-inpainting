"""clip_B：MiniMax H3（ref2v，纸袋 → 参考图里的书）结果的对照评估，给演示界面用。

H3 是整帧重新生成的（30 fps → 24 fps），没有我们的 SAM 2 mask，所以：
1. 按时间把原视频的帧对齐到 H3 的每一帧（24 fps）；
2. 编辑区域 = 原帧和 H3 帧差别很大的连通区域（纸袋 / 书），前后 ±2 帧取并集；
3. 用和 clip_A/C/D 同一套指标：闪烁（第三周 flicker）、接缝（第一周 seam_score），
   再加 mask 外的保真度（第三周 preservation：PSNR 和改动像素比例），看 H3 有没有动不该动的地方。
输出 results/clip_B_h3.json 和几张图（close-ups / mask / 闪烁曲线），只读 H3 结果，不改它们。

用法：python eval_h3.py
"""
import json
import os
import sys

import cv2
import numpy as np

from temporal import K, ROOT, flicker, imread, imwrite, label, seam_score

sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "week3_video_pipeline"))
from evaluate_video import preservation  # noqa: E402

RES = os.path.join(ROOT, "results")
CID = "clip_B"
DIFF_THR = 40      # 灰度级：原帧和 H3 帧差这么多才算“被改了”
MIN_AREA = 3000    # 像素：小于这个的差异块当作重新生成带来的噪声


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def edit_masks(orig, out):
    masks = []
    for o, r in zip(orig, out):
        d = cv2.absdiff(cv2.GaussianBlur(o, (7, 7), 0), cv2.GaussianBlur(r, (7, 7), 0)).max(2)
        m = cv2.morphologyEx((d > DIFF_THR).astype(np.uint8), cv2.MORPH_OPEN, K(3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K(15))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(m)
        if cs:  # 只留最大的一块（纸袋 → 书）并填洞；别处的差异（如桌边漂移）算作 mask 外的改动
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) >= MIN_AREA:
                cv2.drawContours(keep, [c], -1, 1, -1)
        masks.append(cv2.dilate(keep, K(8)))
    u = [np.max(masks[max(0, t - 2):t + 3], axis=0) for t in range(len(masks))]  # 时间上取并集，mask 不抖
    return [(m * 255).astype(np.uint8) for m in u]


def main():
    out, fps_out = read_video(os.path.join(RES, f"{CID}__book.mp4"))
    orig_all, fps_in = read_video(os.path.join(RES, f"{CID}__original.mp4"))
    idx = [min(len(orig_all) - 1, round(k / fps_out * fps_in)) for k in range(len(out))]
    orig = [orig_all[i] for i in idx]  # 原视频按时间重采样到 24 fps，两边逐帧可比
    masks = edit_masks(orig, out)
    area = [float((m > 0).mean() * 100) for m in masks]

    rows = {}
    for name, seq in (("original", orig), ("h3", out)):
        r, j = flicker([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in seq], masks)
        rows[name] = {"flicker_med": round(float(np.nanmedian(r)), 3), "flicker_p90": round(float(np.nanpercentile(r, 90)), 3),
                      "max_jump": round(float(np.nanmax(j)), 2), "curve": r}
    seam = [seam_score(out[t], masks[t])["seam_ratio"] for t in range(0, len(out), 5) if masks[t].any()]
    rows["h3"]["seam_ratio"] = round(float(np.nanmedian(seam)), 3)
    rows["original"]["seam_ratio"] = ""
    psnr, changed = preservation(orig, out, masks)

    # 图：闪烁曲线、close-ups（原视频 | H3 | 编辑区域）、mask 叠加
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    tt = np.arange(1, len(out)) / fps_out
    plt.plot(tt, rows["original"]["curve"], label="original", color="#888")
    plt.plot(tt, rows["h3"]["curve"], label="MiniMax H3", color="#b42318")
    plt.xlabel("time (s)"); plt.ylabel("flicker ratio"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RES, f"{CID}_flicker_h3.png"), dpi=110); plt.close()

    picks = [0, len(out) // 2, len(out) - 1]
    sheet = []
    for t in picks:
        ov = out[t].copy()
        ov[masks[t] > 0] = (0.5 * ov[masks[t] > 0] + [0, 80, 200]).clip(0, 255).astype(np.uint8)
        sz = (360, 640)
        sheet.append(np.hstack([label(cv2.resize(orig[t], sz), f"original {t / fps_out:.1f}s"),
                                label(cv2.resize(out[t], sz), "MiniMax H3"),
                                label(cv2.resize(ov, sz), "edited region (diff)")]))
    imwrite(os.path.join(RES, f"{CID}_crops.jpg"), np.vstack(sheet))
    ref = imread(os.path.join(RES, f"{CID}_Book.png"))
    ref = cv2.resize(ref, (640 * ref.shape[1] // ref.shape[0], 640))
    imwrite(os.path.join(RES, f"{CID}_masks.jpg"), np.hstack([label(ref, "reference image (prompt)"),
                                                            label(cv2.resize(orig[0], (360, 640)), "original"),
                                                            label(cv2.resize(out[0], (360, 640)), "MiniMax H3")]))

    info = {"clip": CID, "frames": len(out), "fps": fps_out, "source_fps": fps_in, "source_frames": len(orig_all),
            "edited_area_pct": round(float(np.median(area)), 1),
            "h3": {"mode": "MiniMax H3 ref2v (whole video regenerated from a reference image)",
                   "psnr_outside": psnr, "changed_outside_pct": changed, "lama_calls": 0},
            "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in rows.items()}}
    with open(os.path.join(RES, f"{CID}_h3.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
