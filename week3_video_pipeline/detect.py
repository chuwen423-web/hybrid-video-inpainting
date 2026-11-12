"""目标检测：YOLOX-S（OpenCV Model Zoo，Apache-2.0），用 OpenCV dnn 运行。

对应两位教授 10/22 的建议：
  - Chen：先试 OpenCV 里的模型；用检测结果让用户确认目标，再交给后续处理（半自动）
  - Ranga：长视频里先用检测找出目标出现的片段，只处理这些帧

预处理 / 后处理和 opencv_zoo/models/object_detection_yolox/yolox.py 一致：
  RGB、letterbox 到 640x640（灰色 114 填充，图像放左上角），网格解码 → NMS。
"""
import cv2
import numpy as np

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"]


class YOLOX:
    def __init__(self, model_path, input_size=640, conf=0.35, nms=0.5):
        # 从内存读：OpenCV dnn 不支持含中文的路径
        self.net = cv2.dnn.readNetFromONNX(np.fromfile(model_path, np.uint8))
        self.size = input_size
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def __call__(self, bgr):
        """返回 [(类别名, 置信度, (x0, y0, x1, y1))]，坐标为原图像素。"""
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:int(h * r), :int(w * r)] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (int(w * r), int(h * r)))
        self.net.setInput(canvas.transpose(2, 0, 1)[None].astype(np.float32))
        d = self.net.forward()[0]
        xy = (d[:, :2] + self.grids) * self.strides
        wh = np.exp(d[:, 2:4]) * self.strides
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        scores = d[:, 4:5] * d[:, 5:]
        cls = scores.argmax(1)
        best = scores.max(1)
        sel = best > self.conf
        if not sel.any():
            return []
        b, s, c = boxes[sel], best[sel], cls[sel]
        keep = cv2.dnn.NMSBoxesBatched([[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in b], s.tolist(), c.tolist(),
                                       self.conf, self.nms)
        out = []
        for i in np.array(keep).reshape(-1):
            x0, y0, x1, y1 = b[i]
            out.append((COCO[c[i]], float(s[i]), (max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1))))
        return out


def find_segments(present, min_len=5, max_gap=8, margin=6):
    """present: 每帧是否检测到目标（bool 列表）。合并短间断、去掉过短片段、两端各留 margin 帧。返回 [(start, end)]（含 end）。"""
    n = len(present)
    segs, start, gap = [], None, 0
    for i, p in enumerate(present):
        if p:
            if start is None:
                start = i
            gap = 0
            last = i
        elif start is not None:
            gap += 1
            if gap > max_gap:
                segs.append((start, last))
                start = None
    if start is not None:
        segs.append((start, last))
    return [(max(0, s - margin), min(n - 1, e + margin)) for s, e in segs if e - s + 1 >= min_len]
