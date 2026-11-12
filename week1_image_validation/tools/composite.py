"""合成工具：混合管线的核心（原图 + 生成内容 → 只替换 mask 区域）。"""
import cv2
import numpy as np

from . import mask_ops


def alpha_composite(original, generated, alpha):
    """out = alpha * generated + (1 - alpha) * original。alpha 为 HxW 的 float(0~1)。"""
    a = alpha[..., None].astype(np.float32)
    out = a * generated.astype(np.float32) + (1.0 - a) * original.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def match_size(generated, original):
    """全局模型输出的分辨率经常和原图不同，先统一尺寸。"""
    if generated.shape[:2] != original.shape[:2]:
        generated = cv2.resize(generated, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    return generated


def align_to_original(generated, original, mask, max_iter=200, eps=1e-6):
    """用 ECC 把生成图对齐到原图（只用 mask 外的区域估计变换）。

    全局模型有时会把画面轻微平移、缩放或裁剪。直接按 mask 抠图会错位，
    所以合成前先估计一个仿射变换把它对齐回来。
    返回 (对齐后的图, 2x3 变换矩阵, ECC 相关系数)。失败时返回原输入。
    """
    generated = match_size(generated, original)
    g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g2 = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    outside = (mask_ops.dilate(mask, 15) == 0).astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps),
                                        outside, 5)
    except cv2.error:
        return generated, np.eye(2, 3, dtype=np.float32), float("nan")
    h, w = original.shape[:2]
    aligned = cv2.warpAffine(generated, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned, warp, float(cc)


def color_match_in_band(original, generated, mask, band_px=12):
    """在 mask 外侧的一圈环带里比较原图和生成图的颜色均值/方差，
    把生成内容的颜色校正到原图的色调（全局模型常带整体色偏）。"""
    ring = cv2.subtract(mask_ops.dilate(mask, band_px), mask) > 0
    if ring.sum() < 50:
        return generated
    o = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        mo, so = o[..., c][ring].mean(), o[..., c][ring].std() + 1e-6
        mg, sg = g[..., c][ring].mean(), g[..., c][ring].std() + 1e-6
        g[..., c] = (g[..., c] - mg) * (so / sg) + mo
    g = np.clip(g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_LAB2BGR)


def poisson_composite(original, generated, mask, mode=cv2.NORMAL_CLONE):
    """泊松融合：把生成图里 mask 区域的梯度贴进原图，边界颜色自动过渡。"""
    m = (mask > 127).astype(np.uint8) * 255
    # seamlessClone 要求 mask 不能碰到图像边缘
    m[:2, :] = 0; m[-2:, :] = 0; m[:, :2] = 0; m[:, -2:] = 0
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return original.copy()
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    center = (int((x0 + x1 + 1) // 2), int((y0 + y1 + 1) // 2))
    src = generated[y0:y1 + 1, x0:x1 + 1]
    sm = m[y0:y1 + 1, x0:x1 + 1]
    return cv2.seamlessClone(src, original, sm, center, mode)
