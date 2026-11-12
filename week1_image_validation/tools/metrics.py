"""评测指标（对应项目三条标准里可量化的部分）。

- 保持度 Preservation：mask（外扩后）以外区域与原图的差异 → PSNR / SSIM / 平均绝对差 / 变化像素占比
- 自然融合 Integration：接缝环带的梯度突变程度（越接近 1 越自然）
- 目标完成度 Completion：需要人工打分；这里只给一个辅助量：mask 内与原图的差异（删除了东西应该变化明显）

只依赖 OpenCV + NumPy（SSIM 自己实现，不依赖 scikit-image）。
"""
import cv2
import numpy as np

from . import mask_ops


def _psnr(a, b, sel):
    d = (a.astype(np.float64) - b.astype(np.float64))[sel]
    mse = float(np.mean(d ** 2)) if d.size else 0.0
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_map(a, b):
    """灰度 SSIM 图（Wang et al. 2004，高斯窗 11x11, sigma=1.5）。"""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a ** 2
    sbb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))


def preservation(original, result, mask, margin_px=10, change_thresh=10):
    """mask 外扩 margin_px 后，计算外部区域与原图的一致性。"""
    outside = mask_ops.dilate(mask, margin_px) == 0
    diff = np.abs(original.astype(np.int16) - result.astype(np.int16)).max(axis=2)
    smap = ssim_map(original, result)
    return {
        "outside_psnr": round(_psnr(original, result, outside), 2),
        "outside_ssim": round(float(smap[outside].mean()), 4),
        "outside_mad": round(float(diff[outside].mean()), 3),
        "outside_changed_pct": round(100.0 * float((diff[outside] > change_thresh).mean()), 3),
    }


def inside_change(original, result, mask):
    """mask 内的平均变化（辅助判断目标是否真的被改动；不能替代人工判断）。"""
    inside = mask > 127
    diff = np.abs(original.astype(np.int16) - result.astype(np.int16)).mean(axis=2)
    return {"inside_mad": round(float(diff[inside].mean()), 2)}


def seam_score(result, mask, width=3, ref_width=12):
    """接缝自然度：接缝带的平均梯度 ÷ 外围参考带的平均梯度。

    ≈1 表示接缝处纹理强度和周围一致；明显 >1 说明有硬边/色差；明显 <1 说明糊了。
    这是一个粗略的代理指标，最终仍以人工打分为准。
    """
    g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    seam = mask_ops.seam_band(mask, width) > 0
    ref = (mask_ops.dilate(mask, width + ref_width) > 0) & ~(mask_ops.dilate(mask, width) > 0)
    if seam.sum() == 0 or ref.sum() == 0:
        return {"seam_ratio": float("nan")}
    return {"seam_ratio": round(float(mag[seam].mean() / (mag[ref].mean() + 1e-6)), 3)}


def diff_heatmap(original, result, mask=None, gain=4.0):
    """差异热力图：亮 = 改动大。叠加 mask 轮廓（白线）方便看改动是否越界。"""
    diff = np.abs(original.astype(np.int16) - result.astype(np.int16)).max(axis=2).astype(np.float32)
    heat = cv2.applyColorMap(np.clip(diff * gain, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if mask is not None:
        cs, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heat, cs, -1, (255, 255, 255), 1)
    return heat


def all_metrics(original, result, mask):
    m = {}
    m.update(preservation(original, result, mask))
    m.update(inside_change(original, result, mask))
    m.update(seam_score(result, mask))
    return m
