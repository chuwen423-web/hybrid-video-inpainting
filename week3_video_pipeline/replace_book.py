"""clip_C 替换任务：删除咖啡杯之后，把桌上的纸袋替换成一本黑色的书。

做法（关键帧生成 + 逐帧对齐合成，对应 Proposal 的“生成内容只取 mask 区域再合成”）：
  1. 关键帧（默认第 0 帧：纸袋完整、没有手）取 LaMa 删除杯子后的画面，纸袋 mask 外扩后交给 SDXL inpaint，
     生成 4 个候选（results/clip_C_book_candidates.png），按 config 里的 book_seed 选一个（用户可改）
  2. 书跟着纸袋走：每帧用纸袋区域内的 ORB 特征估计纸袋相对关键帧的平移 + 旋转 + 缩放（纸袋被手推动时书也跟着动）；
     纸袋区域特征不够时，退回用背景特征估计的相机运动（单应矩阵）
  3. 手（SAM 2 跟踪的 person）在书前面时保留手；纸袋被碰动后露出书区域外的部分，再用 LaMa 补掉
需要 ComfyUI 在 http://127.0.0.1:8188 运行（week3_sdxl_inpaint/start_comfyui.bat）。
"""
import json
import os
import sys

import cv2
import numpy as np

from pipeline import PROJ, ROOT, frame_paths, imread, imwrite, inpaint_crop, load_cfg, mask_ops, wdir, write_video

sys.path.insert(0, os.path.join(PROJ, "week3_sdxl_inpaint"))
import sdxl_inpaint as sx  # noqa: E402  复用第三周的 ComfyUI 客户端

CID = "clip_C"
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
NEG = "paper bag, brown paper, text, logo, letters, watermark, blurry, low quality, distorted, hand, person"


def seq(folder, n):
    return [imread(os.path.join(folder, f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127 if os.path.exists(os.path.join(folder, f"{i:05d}.png"))
            else None for i in range(n)]


def generate_book(key_img, region, prompt, seeds, choose, neg=NEG):
    """在关键帧上用 SDXL 生成书；返回 (整帧补丁, 候选图列表)。"""
    x0, y0, x1, y1 = cv2.boundingRect(region.astype(np.uint8))
    x1, y1 = x0 + x1, y0 + y1
    pad = int(0.5 * max(x1 - x0, y1 - y0))
    H, W = key_img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop, cm = key_img[cy0:cy1, cx0:cx1], region[cy0:cy1, cx0:cx1].astype(np.uint8) * 255
    w2, h2 = sx.model_size(crop.shape[0], crop.shape[1], 1024)
    img_name = sx.upload(cv2.resize(crop, (w2, h2), interpolation=cv2.INTER_LANCZOS4), "clipC_book_image.png")
    mask_name = sx.upload(cv2.cvtColor(cv2.resize(cm, (w2, h2), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR), "clipC_book_mask.png")
    s = {"steps": 30, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}
    cands = {}
    for seed in seeds:
        out, dt = sx.run_workflow(sx.build_workflow(img_name, mask_name, prompt, neg, seed, s, f"week3_video/clipC_book_s{seed}"))
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
        full = key_img.copy()
        a = mask_ops.feather(cm, 5)[..., None]
        full[cy0:cy1, cx0:cx1] = (a * out + (1 - a) * crop).astype(np.uint8)
        cands[seed] = full
        print(f"[book] seed {seed}: {w2}x{h2}, {dt:.1f}s")
    return cands[choose], cands, (cx0, cy0, cx1, cy1)


def homography(key_g, cur_g, key_excl, cur_excl, orb, bf):
    k1, d1 = orb.detectAndCompute(key_g, (~key_excl).astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(cur_g, (~cur_excl).astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None, 0
    m = bf.match(d1, d2)
    if len(m) < 20:
        return None, 0
    src = np.float32([k1[x.queryIdx].pt for x in m])
    dst = np.float32([k2[x.trainIdx].pt for x in m])
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return Hm, int(inl.sum()) if inl is not None else 0


def object_motion(key_img, cur_img, key_m, cur_m, orb, bf):
    """纸袋本身相对关键帧的相似变换（2x3 → 3x3）。"""
    g1, g2 = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, key_m.astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(g2, cur_m.astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 15 or len(k2) < 15:
        return None, 0
    m = bf.match(d1, d2)
    if len(m) < 15:
        return None, 0
    A, inl = cv2.estimateAffinePartial2D(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k2[x.trainIdx].pt for x in m]),
                                         method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if A is None:
        return None, 0
    return np.vstack([A, [0, 0, 1]]), int(inl.sum())


def main():
    cfg = load_cfg()
    c = cfg["clips"][CID]
    rep = c["replace"]
    info = json.load(open(wdir(CID, "info.json"), encoding="utf-8"))
    n, fps = info["frames"], info["fps"]
    key = rep.get("key_frame", 0)
    lama = [imread(os.path.join(wdir(CID, "out_lama"), f"{i:05d}.png")) for i in range(n)]
    orig = [imread(p) for p in frame_paths(CID)]
    bag = seq(wdir(CID, "masks", "obj2"), n)
    hand = seq(wdir(CID, "masks", "obj3"), n) if os.path.isdir(wdir(CID, "masks", "obj3")) else [None] * n
    cup = seq(wdir(CID, "masks", "obj1"), n)
    zero = np.zeros(lama[0].shape[:2], bool)
    bag = [b if b is not None else zero for b in bag]
    hand = [h if h is not None else zero for h in hand]

    region0 = cv2.dilate(bag[key].astype(np.uint8), K(12)) > 0
    prompt = rep["with"] + ", top view, photo, natural light, sharp"
    neg = ", ".join(x for x in (rep.get("negative"), NEG) if x)
    patch, cands, _ = generate_book(lama[key], region0, prompt, rep.get("seeds", [0, 1, 2, 3]), rep.get("book_seed", 0), neg)
    tiles = [("key frame (cup removed)", lama[key])] + [(f"SDXL seed {s}", im) for s, im in cands.items()]
    sheet = [cv2.resize(im, (225, 400)) for _, im in tiles]
    for im, (t, _) in zip(sheet, tiles):
        cv2.rectangle(im, (0, 0), (225, 24), (25, 25, 25), -1)
        cv2.putText(im, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite(os.path.join(ROOT, "results", "clip_C_book_candidates.png"), np.hstack(sheet))

    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    key_g = cv2.cvtColor(lama[key], cv2.COLOR_BGR2GRAY)
    key_excl = cv2.dilate((region0 | hand[key] | cup[key]).astype(np.uint8), K(15)) > 0
    alpha0 = mask_ops.feather(region0.astype(np.uint8) * 255, 5)
    Hprev = np.eye(3)
    outs, regions, inliers, modes = [], [], [], []
    bag_key = cv2.dilate(bag[key].astype(np.uint8), K(4)) > 0
    for t in range(n):
        g = cv2.cvtColor(lama[t], cv2.COLOR_BGR2GRAY)
        excl = cv2.dilate((bag[t] | hand[t] | cup[t]).astype(np.uint8), K(15)) > 0
        if t == key:
            Hm, ni, mode = np.eye(3), 999, "key"
        else:
            Hm, ni = object_motion(orig[key], orig[t], bag_key & ~cup[key], (cv2.dilate(bag[t].astype(np.uint8), K(4)) > 0) & ~hand[t] & ~cup[t], orb, bf)
            mode = "object"
            if Hm is None or ni < 20:
                Hm, ni = homography(key_g, g, key_excl, excl, orb, bf)
                mode = "camera"
            if Hm is None or ni < 25:
                Hm, mode = Hprev, "hold"
        Hprev = Hm
        inliers.append(ni)
        modes.append(mode)
        h, w = g.shape
        book = cv2.warpPerspective(patch, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpPerspective(alpha0, Hm, (w, h), flags=cv2.INTER_LINEAR)
        a *= 1 - (cv2.dilate(hand[t].astype(np.uint8), K(3)) > 0)  # 手在书前面
        out = (a[..., None] * book + (1 - a[..., None]) * lama[t]).astype(np.uint8)
        leftover = (cv2.dilate(bag[t].astype(np.uint8), K(6)) > 0) & (a < 0.5) & ~hand[t]
        if leftover.sum() > 150:  # 纸袋被碰动后露出的部分
            out = inpaint_crop(out, mask_ops.dilate(leftover.astype(np.uint8) * 255, 4), "lama")
        reg = ((a > 0.05) | leftover).astype(np.uint8) * 255
        imwrite(os.path.join(wdir(CID, "out_lama+book"), f"{t:05d}.png"), out)
        imwrite(os.path.join(wdir(CID, "book_region"), f"{t:05d}.png"), reg)
        outs.append(out)
        regions.append(reg)
    res = os.path.join(ROOT, "results")
    write_video(os.path.join(res, "clip_C__lama+book.mp4"), outs, fps)
    comp = []
    for t in range(n):
        row = [cv2.resize(x, (360, 640), interpolation=cv2.INTER_AREA) for x in (orig[t], lama[t], outs[t])]
        for x, lab in zip(row, ("original", "cup removed (LaMa)", "+ bag -> black book")):
            cv2.rectangle(x, (0, 0), (360, 28), (25, 25, 25), -1)
            cv2.putText(x, f"{lab}  {t / fps:4.1f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        comp.append(np.hstack(row))
    write_video(os.path.join(res, "clip_C__replace_compare.mp4"), comp, fps)
    ok = [x for x in inliers if x >= 20]
    from collections import Counter
    info["replace"] = {"key_frame": key, "seed": rep.get("book_seed", 0), "prompt": prompt,
                       "alignment_modes": dict(Counter(modes)), "median_inliers": int(np.median(ok)) if ok else 0}
    json.dump(info, open(wdir(CID, "info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[book] 完成：{n} 帧；对齐方式 {info['replace']['alignment_modes']}（中位内点 {info['replace']['median_inliers']}）")


if __name__ == "__main__":
    main()
