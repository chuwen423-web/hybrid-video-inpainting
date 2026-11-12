"""可视化：带标题的对比网格图。"""
import cv2
import numpy as np


def label(img, text, h=28):
    bar = np.full((h, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid(items, cols=3, cell_w=360, pad=6):
    """items: [(标题, BGR 图)]。标题只用英文/数字（OpenCV 不能直接画中文）。"""
    cells = []
    for title, im in items:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        s = cell_w / im.shape[1]
        im = cv2.resize(im, (cell_w, int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        cells.append(label(im, title))
    ch = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.full((ch - c.shape[0], cell_w, 3), 255, np.uint8)]) for c in cells]
    while len(cells) % cols:
        cells.append(np.full((ch, cell_w, 3), 255, np.uint8))
    rows = []
    for r in range(0, len(cells), cols):
        row = cells[r:r + cols]
        sep = np.full((ch, pad, 3), 255, np.uint8)
        rows.append(np.hstack([x for c in row for x in (c, sep)][:-1]))
    hsep = np.full((pad, rows[0].shape[1], 3), 255, np.uint8)
    return np.vstack([x for r in rows for x in (r, hsep)][:-1])
