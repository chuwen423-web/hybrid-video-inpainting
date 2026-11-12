"""Mask 生成与处理（传统方法：定位 + mask 构建）。

只依赖 OpenCV + NumPy。约定：
- 二值 mask: uint8, 0 或 255
- alpha:     float32, 0.0 ~ 1.0
"""
import cv2
import numpy as np


def box_mask(shape, box):
    """直接用矩形框作为 mask（最粗糙的定位方式，作为对照）。box = [x, y, w, h]"""
    x, y, w, h = box
    m = np.zeros(shape[:2], np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


def grabcut_mask(img, box, iters=8, fg_points=None, bg_points=None, point_radius=6):
    """用 GrabCut 从矩形框里分割出目标。

    fg_points / bg_points: 可选的 [x, y] 点列表，分别强制标为前景 / 背景，
    用来修正 GrabCut 分错的地方（相当于 SAM 的点提示的传统版）。
    """
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], np.uint8)  # GC_BGD
    mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
    for pts, label in ((fg_points, cv2.GC_FGD), (bg_points, cv2.GC_BGD)):
        for px, py in pts or []:
            cv2.circle(mask, (int(px), int(py)), point_radius, int(label), -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def clean_mask(mask, open_px=3, close_px=7, keep_largest=True, fill_holes=True):
    """形态学清理：开运算去噪点，闭运算补缝，保留最大连通域，填洞。"""
    m = mask.copy()
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.where(labels == idx, 255, 0).astype(np.uint8)
    if fill_holes:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(m)
        cv2.drawContours(m, contours, -1, 255, -1)
    return m


def dilate(mask, px):
    """外扩 mask。修复/替换时一般要外扩几个像素，盖住物体的边缘和阴影。"""
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def feather(mask, px):
    """把二值 mask 变成边缘柔和的 alpha（0~1）。

    做法：先对 mask 求距离变换，在边界内侧 px 像素内从 0 线性过渡到 1，
    这样 mask 内部完全使用新内容，过渡带只落在 mask 内侧，不会侵入 mask 外的原图。
    """
    if px <= 0:
        return (mask > 127).astype(np.float32)
    inside = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(inside / float(px), 0.0, 1.0).astype(np.float32)


def seam_band(mask, width):
    """mask 边界两侧各 width 像素的环带（用于接缝修复和接缝指标）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    outer = cv2.dilate(mask, k)
    inner = cv2.erode(mask, k)
    return cv2.subtract(outer, inner)


def overlay(img, mask, color=(0, 0, 255), alpha=0.45, contour=True):
    """把 mask 半透明叠加在图上，便于检查 mask 质量。"""
    vis = img.copy()
    sel = mask > 127
    vis[sel] = (vis[sel] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour:
        cs, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, color, 2)
    return vis
