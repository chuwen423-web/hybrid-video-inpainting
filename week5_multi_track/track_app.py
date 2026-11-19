"""多目标识别与追踪界面：上传视频 → YOLOX 逐帧检测 → IoU 关联成一个个"实例"（同一类的多个物体分开编号）
→ 用户勾选要追踪的实例 → SAM 2 同时追踪全部实例，得到逐像素 mask → 彩色叠加视频、时间轴、逐帧轨迹表。

用法：双击 start_tracker.bat（或 python track_app.py），浏览器打开 http://127.0.0.1:8768
只监听本机；不装新库（第二周的 .venv）。复用：
  任务状态 / 后台队列 / 文件服务     week4_video_editor/editor_app.py（GPU 同时只跑一个任务）
  YOLOX、SAM 2、读写图片            week3_video_pipeline/detect.py、pipeline.py
每个任务一个文件夹：jobs/<id>/，可以随时整个删掉。
"""
import json
import os
import shutil
import sys
import tempfile
import time
import webbrowser
from http.server import ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(PROJ, "week4_video_editor"))
import editor_app as ea  # noqa: E402  （它会把 week3 / week4_temporal 加进 sys.path）
from editor_app import YOLOX, find_segments, imread, imwrite, jdir, load_job, log, new_job, run_bg, update  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")
ea.JOBS = os.path.join(ROOT, "jobs")  # 任务放在本文件夹，不和编辑器混在一起
PORT = 8768
MAX_OBJ = 16  # 一次最多追踪的实例数（SAM 2 显存和时间随实例数增长）
# 网页 "Test results" 标签页展示的最终测试结果（只读：不能在这些任务上重新追踪）
SHOWCASE = [
    {"clip": "clip_C", "id": "4d06cc02",
     "note": "Hand-held cup: all 13 detected instances tracked. The cup gets 5 detector numbers (it leaves the frame and comes back; "
             "the hand covers its middle) — SAM 2 joins them into one object, #1."},
    {"clip": "clip_D", "id": "ef36290b",
     "note": "Camera pans across an office: 15 chairs + 1 cup tracked at the same time (16 instances = the limit). "
             "Result: 8 chairs and the cup, each with its own number."},
]
SHOW_IDS = {s["id"] for s in SHOWCASE}
# 20 种容易区分的颜色（Trubetskoy 色表，越靠前越好区分），网页用同样的顺序
HEX = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
       "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"]
PALETTE = [(int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)) for h in HEX]  # BGR


def color(i):
    return PALETTE[(i - 1) % len(PALETTE)]


class VideoOut:
    """边算边写的 H.264 视频（不必把所有帧留在内存里）。和 temporal.write_mp4 一样：先写到英文临时路径再移过去。"""

    def __init__(self, path, fps, size):
        self.path, self.tmp = path, os.path.join(tempfile.gettempdir(), f"w5_{os.getpid()}_{time.time_ns()}.mp4")
        self.vw = cv2.VideoWriter(self.tmp, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, size)
        if not self.vw.isOpened():
            self.vw = cv2.VideoWriter(self.tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, f):
        self.vw.write(f)

    def close(self):
        self.vw.release()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        shutil.move(self.tmp, self.path)


def frame_path(jid, i):
    return jdir(jid, "frames", f"{i:05d}.jpg")


def label_path(jid, i):
    return jdir(jid, "labels", f"{i:05d}.png")


# ---------- 1. 检测 + 关联成实例 ----------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def link_tracks(dets, iou_thr=0.3, max_gap=8, min_len=5, new_conf=0.4, min_best=0.5):
    """逐帧把检测框连成轨迹（同类别、IoU 贪心匹配；按上一段速度预测位置，允许中间漏检 max_gap 帧）。
    同一类有好几个物体（例如三个杯子）时各自成一条轨迹，这是"多目标"的关键：编辑器只按类别分组，这里按实例分。
    低于 new_conf 的框只能延长已有轨迹，不能新开（减少误检带来的碎轨迹）。"""
    tracks = []
    for t, ds in enumerate(dets):
        cand = []
        for k, tr in enumerate(tracks):
            gap = t - tr["last"]
            if gap > max_gap + 1:
                continue
            pb = np.array(tr["boxes"][tr["last"]]) + tr["vel"] * gap
            for j, d in enumerate(ds):
                if d["cls"] == tr["cls"]:
                    v = iou(pb, d["box"])
                    if v >= iou_thr:
                        cand.append((v, k, j))
        used_t, used_d = set(), set()
        for v, k, j in sorted(cand, reverse=True):
            if k in used_t or j in used_d:
                continue
            tr, d = tracks[k], ds[j]
            nv = (np.array(d["box"]) - np.array(tr["boxes"][tr["last"]])) / (t - tr["last"])
            tr["vel"] = 0.5 * tr["vel"] + 0.5 * nv
            tr["boxes"][t], tr["scores"][t], tr["last"] = d["box"], d["score"], t
            used_t.add(k)
            used_d.add(j)
        for j, d in enumerate(ds):
            if j not in used_d and d["score"] >= new_conf:
                tracks.append({"cls": d["cls"], "boxes": {t: d["box"]}, "scores": {t: d["score"]}, "last": t, "vel": np.zeros(4)})
    return [tr for tr in tracks if len(tr["boxes"]) >= min_len and max(tr["scores"].values()) >= min_best]


def prompt_frame(tr, dets):
    """给 SAM 2 的提示帧：置信度高、而且和同一帧里其他框重叠少的一帧（重叠多时框可能包住两个物体）。"""
    def score(t):
        others = [iou(tr["boxes"][t], d["box"]) for d in dets[t] if d["box"] != tr["boxes"][t]]
        return tr["scores"][t] - 0.8 * max(others, default=0.0)
    return max(tr["boxes"], key=score)


def prepare(jid, src):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    det = ea._DET.get("det") or ea._DET.setdefault("det", YOLOX(os.path.join(ea.W3, "models", "yolox_s.onnx"), conf=0.25))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    dets, size = [], None
    vo = None
    while n < int(fps * ea.MAX_SECONDS):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(1.0, ea.MAX_SIDE / max(h, w))
        if s < 1:
            f = cv2.resize(f, (round(w * s / 2) * 2, round(h * s / 2) * 2), interpolation=cv2.INTER_AREA)
        if vo is None:
            size = (f.shape[1], f.shape[0])
            vo = VideoOut(jdir(jid, "original.mp4"), fps, size)
        vo.write(f)
        imwrite(frame_path(jid, n), f, 95)
        dets.append([{"cls": c, "score": round(sc, 3), "box": [round(float(v), 1) for v in b]} for c, sc, b in det(f)])
        n += 1
        if n % 20 == 0:
            update(jid, progress=int(90 * n / min(total, fps * ea.MAX_SECONDS)))
    cap.release()
    if vo:
        vo.close()
    if n < 10:
        raise ValueError("could not read the video (need at least 10 frames)")
    update(jid, frames=n, fps=round(fps, 3), width=size[0], height=size[1])
    log(jid, f"{n} frames, {size[0]}x{size[1]} @ {fps:.1f} fps; YOLOX boxes: {sum(map(len, dets))}")
    with open(jdir(jid, "detections.json"), "w", encoding="utf-8") as fh:
        json.dump(dets, fh)

    tracks = link_tracks(dets)
    tracks.sort(key=lambda tr: min(tr["boxes"]))
    objs, boxes_by_frame = [], [[] for _ in range(n)]
    for k, tr in enumerate(tracks, start=1):
        pf = prompt_frame(tr, dets)
        present = [t in tr["boxes"] for t in range(n)]
        segs = find_segments(present, min_len=1, max_gap=8, margin=0)
        img = imread(frame_path(jid, pf))
        x0, y0, x1, y1 = tr["boxes"][pf]
        px, py = (x1 - x0) * 0.15 + 4, (y1 - y0) * 0.15 + 4
        crop = img[int(max(0, y0 - py)):int(min(size[1], y1 + py)), int(max(0, x0 - px)):int(min(size[0], x1 + px))]
        imwrite(jdir(jid, "out", f"thumb_{k}.jpg"), cv2.resize(crop, (round(80 * crop.shape[1] / crop.shape[0]), 80)) if crop.size else img)
        objs.append({"id": k, "cls": tr["cls"], "color": HEX[(k - 1) % len(HEX)], "yolo_frames": len(tr["boxes"]),
                     "yolo_segments": segs, "best_score": round(max(tr["scores"].values()), 3),
                     "prompt_frame": int(pf), "box": tr["boxes"][pf], "thumb": f"thumb_{k}.jpg"})
        for t, b in tr["boxes"].items():
            boxes_by_frame[t].append((k, tr["cls"], b))
    update(jid, objects=objs, stage="detected", result=None, labels={}, labels_drawn={})
    # YOLO + IoU 关联的结果视频（只有框），用来和 SAM 2 的结果对照
    write_boxes_video(jid, boxes_by_frame, load_job(jid))
    log(jid, f"found {len(objs)} object instances: " + (", ".join(f"#{o['id']} {o['cls']}" for o in objs) or "nothing"))


# ---------- 自定义名称：视频里显示的名字 ----------
# 名字只允许英文（OpenCV 的 putText 画不了中文，会变成问号）：字母、数字、空格和少量符号
NAME_RE = __import__("re").compile(r"^[A-Za-z0-9 ._()'&+/-]*$")


def display_name(job, k, cls):
    """和网页的 nameOf 一样：先看 k 所属的整个物体（合并后的编号）的名字，再看合并进来的各个检测编号；都没有就用原类别。"""
    labels = job.get("labels") or {}
    tracks = (job.get("result") or {}).get("tracks") or []
    t = next((t for t in tracks if t["id"] == k or k in t.get("merged_from", [])), None)
    ids = [t["id"], *t.get("merged_from", [])] if t else [k]
    hit = next((labels[str(i)] for i in ids if str(i) in labels), None)
    return hit if hit is not None else cls


def write_boxes_video(jid, boxes_by_frame, job):
    fps, size = job["fps"], (job["width"], job["height"])
    vo = VideoOut(jdir(jid, "out", "boxes.mp4"), fps, size)
    lw = max(2, size[0] // 400)
    for t in range(job["frames"]):
        f = imread(frame_path(jid, t))
        for k, cls, b in boxes_by_frame[t]:
            draw_tag(f, k, display_name(job, k, cls), [int(v) for v in b], color(k), lw)
        vo.write(f)
    vo.close()


def render_boxes(jid):
    """重画检测预览视频。检测框没有单独保存，从 detections.json 重新关联：link_tracks 是确定性的，编号和第一次完全一样。"""
    job = load_job(jid)
    with open(jdir(jid, "detections.json"), encoding="utf-8") as fh:
        dets = json.load(fh)
    tracks = link_tracks(dets)
    tracks.sort(key=lambda tr: min(tr["boxes"]))
    if len(tracks) != len([o for o in job.get("objects", []) if not o.get("manual")]):  # 手动框选的物体没有检测框，不画
        raise RuntimeError("detections changed since this job was created")
    boxes_by_frame = [[] for _ in range(job["frames"])]
    for k, tr in enumerate(tracks, start=1):
        for t, b in tr["boxes"].items():
            boxes_by_frame[t].append((k, tr["cls"], b))
    write_boxes_video(jid, boxes_by_frame, job)


def add_manual(jid, name, frame, box):
    """手动框选检测器不认识的物体（COCO 80 类以外，例如扑克牌、纸袋）。
    和检测到的实例一样进列表：提示帧 = 框选的那一帧，SAM 2 从这个框开始向前、向后追踪。编号接在已有编号后面。"""
    job = load_job(jid)
    objs = job.get("objects", [])
    k = max([o["id"] for o in objs], default=0) + 1
    img = imread(frame_path(jid, frame))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    crop = img[y0:y1, x0:x1]
    imwrite(jdir(jid, "out", f"thumb_{k}.jpg"), cv2.resize(crop, (max(8, round(80 * crop.shape[1] / crop.shape[0])), 80)))
    objs.append({"id": k, "cls": name, "color": HEX[(k - 1) % len(HEX)], "yolo_frames": 0, "yolo_segments": [],
                 "best_score": None, "prompt_frame": int(frame), "box": [float(x0), float(y0), float(x1), float(y1)],
                 "thumb": f"thumb_{k}.jpg", "manual": True})
    update(jid, objects=objs)
    log(jid, f"added by hand: #{k} {name} (frame {frame}, box {x0},{y0},{x1},{y1})")
    return k


def relabel(jid):
    """改名后只重画视频和轨迹表（不重新检测、不重新跑 SAM 2）。"""
    job = load_job(jid)
    render_boxes(jid)
    if job.get("stage") == "tracked" and job.get("result"):
        update(jid, progress=40)
        render_tracked(jid, 40, 95)
    update(jid, labels_drawn=dict(job.get("labels") or {}))
    log(jid, "videos redrawn with the new names: " + (", ".join(f"#{k} {v}" for k, v in (job.get("labels") or {}).items()) or "original classes"))


def draw_tag(img, k, cls, box, col, lw):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), col, lw)
    txt = f"#{k} {cls}"
    fs = max(0.5, img.shape[1] / 1600)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    ty = y0 - 4 if y0 - th - 8 > 0 else y0 + th + 8
    cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), col, -1)
    ink = (0, 0, 0) if 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2] > 150 else (255, 255, 255)  # 浅色底用黑字
    cv2.putText(img, txt, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, ink, 2, cv2.LINE_AA)


# ---------- 2. SAM 2 同时追踪多个实例 ----------
def track(jid, ids):
    """每个实例在它的提示帧上给一个框，SAM 2 一次性向后、再向前传播所有实例。
    每帧只存一张"标签图"（像素值 = 实例编号，0 = 背景）：几个 mask 重叠时取 SAM 2 打分最高的那个，
    这样同一个像素只属于一个物体，也不用把 N×帧数 张 mask 都放在内存里。
    同时累计两两 mask 的重叠面积：两个编号其实是同一个物体（检测轨迹断开后又接上）时把它们合并。"""
    import torch
    job = load_job(jid)
    n, fps = job["frames"], job["fps"]
    objs = [o for o in job["objects"] if o["id"] in ids]
    pred = ea.sam2_predictor()
    k = len(objs)
    inter = torch.zeros(k, k, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    co = torch.zeros_like(inter)
    pos = {o["id"]: i for i, o in enumerate(objs)}
    seen, done = set(), 0
    shutil.rmtree(jdir(jid, "labels"), ignore_errors=True)
    os.makedirs(jdir(jid, "labels"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(video_path=jdir(jid, "frames"), offload_video_to_cpu=True,
                                offload_state_to_cpu=n * k > 3000)
        for o in objs:
            pred.add_new_points_or_box(state, frame_idx=o["prompt_frame"], obj_id=o["id"], box=np.array(o["box"], np.float32))
        first = min(o["prompt_frame"] for o in objs)
        for reverse in (False, True):
            for fi, oids, logits in pred.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
                lg = logits[:, 0].float()
                if fi not in seen:
                    with torch.autocast(lg.device.type, enabled=False):  # 面积计数要精确，不用 bfloat16
                        m = (lg > 0).flatten(1).float()
                        idx = torch.tensor([pos[i] for i in oids], device=inter.device)
                        ix = (idx[:, None], idx[None, :])
                        inter[ix] += (m @ m.T).double().to(inter.device)
                        area = m.sum(1).double().to(inter.device)
                        co[ix] += area[:, None] * (area[None, :] > 0)  # co[i, j]：i 在 j 也出现的帧里的面积
                    seen.add(fi)
                mx, am = lg.max(0)
                lab = torch.where(mx > 0, torch.tensor(list(oids), device=lg.device)[am], 0).to(torch.uint8).cpu().numpy()
                imwrite(label_path(jid, fi), lab)
                done += 1
                if done % 15 == 0:
                    update(jid, progress=min(85, int(85 * done / max(1, n + first))))
        pred.reset_state(state)

    # 合并重复：两个编号在都出现的帧里 mask 的 IoU > 0.5 → 同一个物体（例如杯子出画后又回来，检测器给了新编号），保留编号小的。
    # 用 IoU 而不是"小的被大的包含"：桌子上的杯子、人手里的杯子是不同物体，不能合并
    inter, co = inter.cpu().numpy(), co.cpu().numpy()
    merge = {o["id"]: o["id"] for o in objs}
    for i in range(k):
        for j in range(i + 1, k):
            union = co[i, j] + co[j, i] - inter[i, j]
            inside = inter[i, j] / max(1.0, min(co[i, j], co[j, i]))  # 小的那个有多少落在大的里面
            # 同一类、而且一个几乎完全落在另一个里面：检测器只框到了物体的一部分（例如手挡住杯子中间，杯子下半截单独一个框）
            if union > 0 and (inter[i, j] / union > 0.5 or (objs[i]["cls"] == objs[j]["cls"] and inside > 0.8)):
                root = merge[objs[i]["id"]]
                for key, v in merge.items():
                    if v == objs[j]["id"]:
                        merge[key] = root
    lut = np.arange(256, dtype=np.uint8)
    for a, b in merge.items():
        lut[a] = b
    merged = {a: b for a, b in merge.items() if a != b}
    if merged:
        log(jid, "merged duplicates (same object, two detector tracks): " + ", ".join(f"#{a}→#{b}" for a, b in merged.items()))

    # 逐帧统计 + 叠加视频 + 轨迹表
    finals = sorted(set(merge.values()))
    info = {o["id"]: dict(o) for o in objs}
    for i in finals:  # 合并后的类别取检测帧数最多的那个（YOLOX 偶尔把同一物体认成别的类，例如隔板认成 laptop）
        votes = {}
        for a, b in merge.items():
            if b == i:
                votes[info[a]["cls"]] = votes.get(info[a]["cls"], 0) + info[a]["yolo_frames"]
        info[i]["cls"] = max(votes, key=votes.get)
    stats = {i: {"frames": 0, "area": 0.0, "present": [False] * n} for i in finals}
    size = (job["width"], job["height"])
    # 结果视频按最终物体的顺序重新配色（前几种颜色最好区分），编号不变
    for r, i in enumerate(finals):
        info[i]["color"] = HEX[r % len(HEX)]
    for t in range(n):
        # 合并编号；中值滤波去掉两个 mask 打分接近处的椒盐状碎点（标签图里只有少数几个值，中值滤波相当于局部多数投票）
        lab = cv2.medianBlur(lut[imread(label_path(jid, t), cv2.IMREAD_UNCHANGED)], 5)
        imwrite(label_path(jid, t), lab)
        for i in np.unique(lab[lab > 0]):
            area = int(np.count_nonzero(lab == i))
            if area < 30:
                continue
            st = stats[int(i)]
            st["frames"] += 1
            st["area"] += area
            st["present"][t] = True

    tracked = []
    for i in finals:
        st = stats[i]
        o = info[i]
        tracked.append({"id": i, "cls": o["cls"], "color": o["color"], "thumb": o["thumb"], "prompt_frame": o["prompt_frame"],
                        "yolo_frames": sum(info[a]["yolo_frames"] for a, b in merge.items() if b == i),
                        "yolo_segments": o["yolo_segments"] + [s for a, b in merge.items() if b == i and a != i for s in info[a]["yolo_segments"]],
                        "sam_frames": st["frames"], "sam_segments": find_segments(st["present"], min_len=1, max_gap=2, margin=0),
                        "mean_area_pct": round(100 * st["area"] / max(1, st["frames"]) / (size[0] * size[1]), 2),
                        "merged_from": [a for a, b in merge.items() if b == i and a != i]})
    update(jid, result={"tracks": tracked, "selected": len(objs)})
    job = load_job(jid)
    render_tracked(jid, 85, 99)  # 视频里用改过的名字（第 2 步追踪前改的名字也算）
    if (job.get("labels") or {}) != (job.get("labels_drawn") or {}):
        render_boxes(jid)  # 追踪前改过名字、还没重画：检测预览视频一起更新
    update(jid, stage="tracked", labels_drawn=dict(job.get("labels") or {}))
    log(jid, "tracked: " + ", ".join(f"#{o['id']} {display_name(job, o['id'], o['cls'])} ({o['sam_frames']} frames)" for o in tracked))


def render_tracked(jid, p0=40, p1=95):
    """按 labels/ 里已经合并好的标签图画追踪结果视频、并排对比视频、轨迹表和 mask 压缩包。
    追踪结束时调用一次；之后改名只调用这一步（几秒到十几秒），不重跑 SAM 2。"""
    job = load_job(jid)
    n, fps = job["frames"], job["fps"]
    tracks = job["result"]["tracks"]
    size = (job["width"], job["height"])
    half = (360, round(360 * size[1] / size[0] / 2) * 2)
    lw = max(2, size[0] // 400)
    fcol = {t["id"]: (int(t["color"][5:7], 16), int(t["color"][3:5], 16), int(t["color"][1:3], 16)) for t in tracks}  # BGR
    names = {t["id"]: display_name(job, t["id"], t["cls"]) for t in tracks}
    colors_lut = np.zeros((256, 3), np.uint8)
    for i, c in fcol.items():
        colors_lut[i] = c
    vo, vc = VideoOut(jdir(jid, "out", "tracked.mp4"), fps, size), VideoOut(jdir(jid, "out", "compare.mp4"), fps, (720, half[1]))
    rows = ["frame,time_s,id,class,name,x,y,w,h,mask_area_px"]  # class = YOLOX 原类别，name = 显示的名字（没改过就等于 class）
    for t in range(n):
        f = imread(frame_path(jid, t))
        lab = imread(label_path(jid, t), cv2.IMREAD_UNCHANGED)
        vis = f.copy()
        sel = lab > 0
        vis[sel] = (f[sel] * 0.5 + colors_lut[lab[sel]] * 0.5).astype(np.uint8)
        for i in np.unique(lab[sel]) if sel.any() else []:
            i = int(i)
            if i not in fcol:
                continue
            m = (lab == i).astype(np.uint8)
            area = int(cv2.countNonZero(m))
            if area < 30:
                continue
            x, y, w, h = cv2.boundingRect(m)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, fcol[i], lw)
            draw_tag(vis, i, names[i], [x, y, x + w, y + h], fcol[i], 1)
            cls = next(tr["cls"] for tr in tracks if tr["id"] == i)
            rows.append(f"{t},{t / fps:.3f},{i},{cls},{names[i].replace(',', ' ')},{x},{y},{w},{h},{area}")
        cv2.putText(vis, f"{t / fps:5.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vo.write(vis)
        vc.write(np.hstack([cv2.resize(f, half, interpolation=cv2.INTER_AREA), cv2.resize(vis, half, interpolation=cv2.INTER_AREA)]))
        if t % 20 == 0:
            update(jid, progress=p0 + int((p1 - p0) * t / n))
    vo.close()
    vc.close()
    with open(jdir(jid, "out", "tracks.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    shutil.make_archive(jdir(jid, "out", "label_masks"), "zip", jdir(jid, "labels"))


# ---------- HTTP ----------
class Handler(ea.Handler):
    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        parts = p.strip("/").split("/")
        if p in ("/", "/index.html"):
            return self.send_file(os.path.join(ROOT, "web", "index.html"))
        if p == "/api/samples":
            return self.send_json(sorted(f for f in os.listdir(ea.SAMPLES) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))))
        if p == "/api/showcase":
            return self.send_json([s for s in SHOWCASE if os.path.isfile(jdir(s["id"], "job.json"))])
        if len(parts) >= 3 and parts[:2] == ["api", "jobs"] and ea.ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            jid = parts[2]
            if len(parts) == 3:
                return self.send_json(load_job(jid))
            if len(parts) == 5 and parts[3] == "frame" and parts[4].isdigit():  # 手动框选用的单帧画面
                return self.send_file(frame_path(jid, int(parts[4])))
            if len(parts) == 5 and parts[3] in ("file", "download") and ea.NAME_RE.match(parts[4]):
                path = jdir(jid, "original.mp4") if parts[4] == "original.mp4" else jdir(jid, "out", parts[4])
                return self.send_file(path, download=f"{jid}_{parts[4]}" if parts[3] == "download" else None)
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        parts = p.strip("/").split("/")
        if p in ("/api/upload", "/api/sample"):
            if p == "/api/upload":  # 原始字节上传，文件名放在 X-Filename
                name = os.path.basename(unquote(self.headers.get("X-Filename", "video.mp4")))
                data = self.body()
            else:
                name = json.loads(self.body() or b"{}").get("name", "")
                if name not in os.listdir(ea.SAMPLES):
                    return self.send_json({"error": "unknown sample"}, 400)
                data = None
            jid = new_job(name)
            src = jdir(jid, "source" + os.path.splitext(name)[1].lower())
            if data is None:
                shutil.copyfile(os.path.join(ea.SAMPLES, name), src)
            else:
                with open(src, "wb") as fh:
                    fh.write(data)
            run_bg(jid, "detect", prepare, src)
            return self.send_json({"id": jid})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "label" and ea.ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            # 自定义物体名称：只改显示用的名字（job.json 的 labels），编号、mask 和 YOLOX 原类别都不变；空字符串 = 恢复原类别
            jid = parts[2]
            if jid in SHOW_IDS:
                return self.send_json({"error": "this is a saved test result (read-only)"}, 403)
            job = load_job(jid)
            if job.get("busy"):
                return self.send_json({"error": f"still running: {job['busy']}"}, 409)
            body = json.loads(self.body() or b"{}")
            try:
                oid = int(body.get("id"))
            except (TypeError, ValueError):
                return self.send_json({"error": "bad object id"}, 400)
            if oid not in {o["id"] for o in job.get("objects", [])}:
                return self.send_json({"error": "unknown object"}, 400)
            name = " ".join(str(body.get("name", "")).split())[:40]
            if not NAME_RE.match(name):  # 视频里的标签用 OpenCV 画，只能显示英文
                return self.send_json({"error": "names can use English letters, digits, spaces and . _ - ( ) ' & + / only"}, 400)
            labels = dict(job.get("labels") or {})
            for other in body.get("same", []):  # 合并进这个物体的其他编号：名字统一存在物体编号上
                labels.pop(str(other), None)
            if name:
                labels[str(oid)] = name
            else:
                labels.pop(str(oid), None)
            update(jid, labels=labels)
            return self.send_json({"labels": labels})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in ("manual", "manual_del") and ea.ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            # 手动框选物体 / 删掉手动框选的物体（检测到的实例不能删，只能不勾选）
            jid = parts[2]
            if jid in SHOW_IDS:
                return self.send_json({"error": "this is a saved test result (read-only)"}, 403)
            job = load_job(jid)
            if job.get("busy"):
                return self.send_json({"error": f"still running: {job['busy']}"}, 409)
            if job.get("stage") not in ("detected", "tracked"):
                return self.send_json({"error": "wait for the detection to finish"}, 400)
            body = json.loads(self.body() or b"{}")
            if parts[3] == "manual":
                name = " ".join(str(body.get("name", "")).split())[:40] or "object"
                if not NAME_RE.match(name):
                    return self.send_json({"error": "names can use English letters, digits, spaces and . _ - ( ) ' & + / only"}, 400)
                try:
                    frame, box = int(body["frame"]), [float(v) for v in body["box"]]
                except (KeyError, TypeError, ValueError):
                    return self.send_json({"error": "need a frame and a box"}, 400)
                if len(box) != 4 or not (0 <= frame < job.get("frames", 0)) or box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    return self.send_json({"error": "drag a box around the object first"}, 400)
                return self.send_json({"id": add_manual(jid, name, frame, box)})
            try:
                oid = int(body.get("id"))
            except (TypeError, ValueError):
                return self.send_json({"error": "bad object id"}, 400)
            objs = job.get("objects", [])
            if not any(o["id"] == oid and o.get("manual") for o in objs):
                return self.send_json({"error": "only objects added by hand can be removed"}, 400)
            labels = {k: v for k, v in (job.get("labels") or {}).items() if k != str(oid)}
            drawn = {k: v for k, v in (job.get("labels_drawn") or {}).items() if k != str(oid)}
            update(jid, objects=[o for o in objs if o["id"] != oid], labels=labels, labels_drawn=drawn)
            log(jid, f"removed #{oid} (added by hand)")
            return self.send_json({"ok": True})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "relabel" and ea.ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            # 只按新名字重画视频和轨迹表，不重新检测、不重跑 SAM 2
            jid = parts[2]
            if jid in SHOW_IDS:
                return self.send_json({"error": "this is a saved test result (read-only)"}, 403)
            job = load_job(jid)
            if job.get("busy"):
                return self.send_json({"error": f"still running: {job['busy']}"}, 409)
            if job.get("stage") not in ("detected", "tracked"):
                return self.send_json({"error": "nothing to redraw yet"}, 400)
            run_bg(jid, "rename", relabel)
            return self.send_json({"ok": True})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "track" and ea.ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            jid = parts[2]
            if jid in SHOW_IDS:
                return self.send_json({"error": "this is a saved test result (read-only) — upload the clip again to track it"}, 403)
            job = load_job(jid)
            if job.get("busy"):
                return self.send_json({"error": f"still running: {job['busy']}"}, 409)
            ids = [int(i) for i in json.loads(self.body() or b"{}").get("ids", [])]
            valid = {o["id"] for o in job.get("objects", [])}
            ids = [i for i in ids if i in valid]
            if not ids:
                return self.send_json({"error": "choose at least one object"}, 400)
            if len(ids) > MAX_OBJ:
                return self.send_json({"error": f"choose at most {MAX_OBJ} objects"}, 400)
            run_bg(jid, "track", track, ids)
            return self.send_json({"ok": True})
        self.send_error(404)


def main():
    os.makedirs(ea.JOBS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Multi-object tracker: {url}  (Ctrl+C to stop)", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
