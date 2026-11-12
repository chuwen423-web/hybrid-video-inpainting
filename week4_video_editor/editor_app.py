"""视频编辑界面（完整流程，可处理新视频）：上传 → 检测 → 勾选物体 / 手动补框 → SAM 2 跟踪 → 删除 / 替换 → 播放和下载。

用法：双击 start_editor.bat（或 python editor_app.py），浏览器打开 http://127.0.0.1:8767
只监听本机；不需要联网，不装新库（用第二周的 .venv：PyTorch + SAM 2 + OpenCV）。
复用已经验证过的代码：
  检测 / 片段 / 关键帧 / LaMa     week3_video_pipeline/pipeline.py、detect.py
  删除（去闪烁、去光晕）           week4_temporal/temporal.py（稳定 mask + 背景板，对不齐时自动改逐帧 LaMa + 光流平滑）
  替换                             上传的图片按原物体自身的运动逐帧贴上去（week3 replace_book.py 的对齐函数），手保持在前面；
                                   原物体先用上面的删除方法去掉。不调用 SDXL，不需要 ComfyUI
每个任务一个文件夹：jobs/<id>/（抽帧、检测、mask、结果视频），可以随时删掉整个文件夹。
"""
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
W3 = os.path.join(PROJ, "week3_video_pipeline")
sys.path.insert(0, os.path.join(PROJ, "week4_temporal"))
sys.path.insert(0, W3)
import temporal as tp  # noqa: E402  第四周：稳定 mask、背景板、光流平滑、H.264 输出、闪烁指标
from detect import COCO, YOLOX, find_segments  # noqa: E402
from pipeline import imread, imwrite, inpaint_crop, mask_ops, pick_keyframes, sam2_predictor  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")
PORT = 8767
JOBS = os.path.join(ROOT, "jobs")
SAMPLES = os.path.join(PROJ, "test_clips")
MAX_SIDE, MAX_SECONDS = 1280, 30
ID_RE = re.compile(r"^[a-z0-9]{8}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,80}$")
COLORS = [(0, 0, 255), (0, 200, 255), (255, 120, 0), (0, 200, 0), (200, 0, 200), (255, 255, 0)]
K = tp.K
WORK = threading.Lock()  # GPU 一次只跑一个任务
_DET = {}


# ---------- 任务状态 ----------
def jdir(jid, *a):
    return os.path.join(JOBS, jid, *a)


def load_job(jid):
    with open(jdir(jid, "job.json"), encoding="utf-8") as f:
        return json.load(f)


def save_job(job):
    tmp = jdir(job["id"], "job.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=1)
    os.replace(tmp, jdir(job["id"], "job.json"))


def update(jid, **kw):
    job = load_job(jid)
    job.update(kw)
    save_job(job)
    return job


def log(jid, msg):
    job = load_job(jid)
    job["log"] = (job.get("log", []) + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-60:]
    save_job(job)


def run_bg(jid, step, fn, *args):
    """后台执行一步；GPU 同时只跑一个任务，排队的任务显示 waiting。"""
    update(jid, busy=step, error=None, progress=0, step_started=time.time())

    def work():
        log(jid, f"{step}: waiting for the GPU" if WORK.locked() else f"{step}: started")
        with WORK:
            try:
                t0 = time.time()
                fn(jid, *args)
                log(jid, f"{step}: done in {time.time() - t0:.0f} s")
                update(jid, busy=None, progress=100)
            except Exception as e:  # 把错误显示在网页上，而不是只在命令行里
                traceback.print_exc()
                log(jid, f"{step}: ERROR {e}")
                update(jid, busy=None, error=f"{step}: {e}")
    threading.Thread(target=work, daemon=True).start()


# ---------- 1. 抽帧 + 检测 ----------
def prepare(jid, src):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < int(fps * MAX_SECONDS):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1:
            f = cv2.resize(f, (round(w * s / 2) * 2, round(h * s / 2) * 2), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if len(frames) < 10:
        raise ValueError("could not read the video (need at least 10 frames)")
    n = len(frames)
    for i, f in enumerate(frames):
        imwrite(jdir(jid, "frames", f"{i:05d}.jpg"), f, 95)
    tp.write_mp4(jdir(jid, "original.mp4"), frames, fps)
    update(jid, frames=n, fps=round(fps, 3), width=frames[0].shape[1], height=frames[0].shape[0])
    log(jid, f"{n} frames, {frames[0].shape[1]}x{frames[0].shape[0]} @ {fps:.1f} fps")

    if "det" not in _DET:
        _DET["det"] = YOLOX(os.path.join(W3, "models", "yolox_s.onnx"), conf=0.25)
    dets = []
    for i, f in enumerate(frames):
        dets.append([{"cls": c, "score": round(s, 3), "box": [round(float(v), 1) for v in b]} for c, s, b in _DET["det"](f)])
        if i % 20 == 0:
            update(jid, progress=int(100 * i / n))
    with open(jdir(jid, "detections.json"), "w", encoding="utf-8") as f:
        json.dump(dets, f)
    classes = []
    for cls in COCO:
        per = [sorted((d for d in ds if d["cls"] == cls), key=lambda d: -d["score"]) for ds in dets]
        present = [bool(p) for p in per]
        if sum(present) < 5:
            continue
        segs = find_segments(present, min_len=5, max_gap=8, margin=int(fps * 0.3))
        if not segs:
            continue
        keys = pick_keyframes(per, segs, fps, k=4, min_conf=0.5)
        if not keys:
            continue
        best = max(keys, key=lambda i: per[i][0]["score"])
        img = frames[best].copy()
        x0, y0, x1, y1 = map(int, per[best][0]["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), max(2, img.shape[1] // 240))
        th = cv2.resize(img, (240, round(240 * img.shape[0] / img.shape[1])))
        imwrite(jdir(jid, "out", f"det_{cls.replace(' ', '_')}.jpg"), th)
        classes.append({"cls": cls, "frames": sum(present), "segments": segs, "keyframes": keys,
                        "best_score": per[best][0]["score"], "thumb": f"det_{cls.replace(' ', '_')}.jpg"})
    classes.sort(key=lambda c: -c["frames"])
    update(jid, classes=classes, stage="detected")
    log(jid, "detected: " + (", ".join(f"{c['cls']} ({c['frames']} frames)" for c in classes) or "nothing"))


# ---------- 2. SAM 2 跟踪 ----------
def track(jid, objects):
    job = load_job(jid)
    n, fps = job["frames"], job["fps"]
    with open(jdir(jid, "detections.json"), encoding="utf-8") as f:
        dets = json.load(f)
    prompts, spans = {}, {}
    for k, o in enumerate(objects, start=1):
        if o["source"] == "class":
            c = next(c for c in job["classes"] if c["cls"] == o["cls"])
            per = [sorted((d for d in ds if d["cls"] == o["cls"]), key=lambda d: -d["score"]) for ds in dets]
            prompts[k] = [{"frame": i, "box": per[i][0]["box"]} for i in c["keyframes"]]
            spans[k] = (c["segments"][0][0], c["segments"][-1][1])
        else:
            prompts[k] = [p for p in o["prompts"] if p.get("box") or p.get("points")]
            spans[k] = (0, n - 1)
        if not prompts[k]:
            raise ValueError(f"object '{o['name']}' has no box or click")
    import torch
    pred = sam2_predictor()
    masks = {k: [None] * n for k in prompts}
    done = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(video_path=jdir(jid, "frames"), offload_video_to_cpu=True, offload_state_to_cpu=n > 500)
        first = min(p["frame"] for ps in prompts.values() for p in ps)
        for k, ps in prompts.items():
            for p in ps:
                kw = {}
                if p.get("box"):
                    kw["box"] = np.array(p["box"], np.float32)
                if p.get("points"):
                    kw["points"] = np.array(p["points"], np.float32)
                    kw["labels"] = np.array(p["labels"], np.int32)
                pred.add_new_points_or_box(state, frame_idx=p["frame"], obj_id=k, **kw)
        for reverse in (False, True):
            for fi, oids, logits in pred.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
                for j, oid in enumerate(oids):
                    masks[oid][fi] = (logits[j, 0] > 0).cpu().numpy()
                done += 1
                if done % 15 == 0:
                    update(jid, progress=min(99, int(100 * done / max(1, n + first))))
        pred.reset_state(state)
    shape = (job["height"], job["width"])
    frames = [imread(jdir(jid, "frames", f"{i:05d}.jpg")) for i in range(n)]
    stats = []
    for k, o in enumerate(objects, start=1):
        a, b = spans[k]
        ms = [m if (m is not None and a <= i <= b) else np.zeros(shape, bool) for i, m in enumerate(masks[k])]
        for i, m in enumerate(ms):
            imwrite(jdir(jid, "masks", f"obj{k}", f"{i:05d}.png"), m.astype(np.uint8) * 255)
        stats.append({**o, "id": k, "masked_frames": int(sum(m.any() for m in ms))})
        for i, m in enumerate(ms):
            if m.any():
                frames[i] = mask_ops.overlay(frames[i], m.astype(np.uint8) * 255, color=COLORS[(k - 1) % len(COLORS)])
    tp.write_mp4(jdir(jid, "out", "masks.mp4"), frames, fps)
    update(jid, objects=stats, stage="tracked", result=None)
    log(jid, "tracked: " + ", ".join(f"{s['name']} ({s['action']}, {s['masked_frames']} frames)" for s in stats))


# ---------- 3. 删除 / 替换 ----------
def remove_auto(jid, frames, raw, keep, fps):
    """第四周的删除方法：稳定 mask（含阴影）+ 背景板；背景板对不齐时改用新 mask 逐帧 LaMa + 光流平滑。"""
    stab = tp.stable_masks(frames, raw)
    idxs = [t for t in range(len(frames)) if stab[t].any()]
    if not idxs:
        return [f.copy() for f in frames], stab, {"mode": "nothing to remove"}
    keep_u = [cv2.dilate(k.astype(np.uint8) * 255, K(10)) for k in keep]
    excl = [np.maximum(s, ku) for s, ku in zip(stab, keep_u)]
    log(jid, f"remove: aligning {len(idxs)} frames to build a background plate")
    H, _ = tp.homographies(frames, excl, idxs, idxs[len(idxs) // 2])
    outs, info, _ = tp.plate_remove(frames, idxs, H, stab, excl, keep, fps)
    info["mode"] = "background plate"
    if info["align_err"] > tp.ALIGN_MAX:
        log(jid, f"remove: plate alignment error {info['align_err']} > {tp.ALIGN_MAX} → per-frame LaMa + flow smoothing")
        lama = []
        for i, (f, m) in enumerate(zip(frames, stab)):
            lama.append(inpaint_crop(f, m, "lama") if m.any() else f)
            if i % 20 == 0:
                update(jid, progress=40 + int(40 * i / len(frames)))
        outs = tp.flow_smooth(frames, lama, stab, alpha=0.5)
        for t in idxs:
            if keep[t].any():
                hk = cv2.dilate(keep[t].astype(np.uint8), K(2)) > 0
                outs[t][hk] = frames[t][hk]
        info = {"mode": "per-frame LaMa + flow (plate could not be aligned)", "plate_align_err": info["align_err"], "lama_calls": len(idxs)}
    return outs, stab, info


def place_asset(asset, mask, shape):
    """把替换图片放到原物体在关键帧上的位置：按外接框缩放（保持比例）、底边对齐、水平居中。返回整帧的 (图像, alpha)。"""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgb, a = asset[..., :3], (asset[..., 3].astype(np.float32) / 255 if asset.shape[2] == 4 else np.ones(asset.shape[:2], np.float32))
    s = min((x1 - x0) / rgb.shape[1], (y1 - y0) / rgb.shape[0])
    w, h = max(2, round(rgb.shape[1] * s)), max(2, round(rgb.shape[0] * s))
    rgb, a = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    H, W = shape[:2]
    img, alpha = np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.float32)
    cx, by = (x0 + x1) // 2, y1
    px, py = cx - w // 2, by - h
    sx0, sy0, sx1, sy1 = max(0, -px), max(0, -py), min(w, W - px), min(h, H - py)
    img[py + sy0:py + sy1, px + sx0:px + sx1] = rgb[sy0:sy1, sx0:sx1]
    alpha[py + sy0:py + sy1, px + sx0:px + sx1] = a[sy0:sy1, sx0:sx1]
    return img, alpha


def follow_paste(base, frames, obj, keep, asset):
    """替换：在关键帧（物体最完整、没有被手挡住的一帧）放好图片，之后每帧按原物体自身的运动（物体区域内的 ORB → 相似变换）
    跟着移动；物体特征不够时用背景估计的镜头运动；手保持在前面。"""
    from replace_book import homography, object_motion  # 第三周的对齐函数
    area = [m.sum() - (m & keep[i]).sum() * 4 for i, m in enumerate(obj)]
    key = int(np.argmax(area))
    img0, a0 = place_asset(asset, obj[key], frames[0].shape)
    orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    obj_key = cv2.dilate(obj[key].astype(np.uint8), K(4)) > 0
    key_g = cv2.cvtColor(base[key], cv2.COLOR_BGR2GRAY)
    key_excl = cv2.dilate((obj[key] | keep[key]).astype(np.uint8), K(15)) > 0
    h, w = frames[0].shape[:2]
    out, Hprev, region = [], np.eye(3), []
    for t in range(len(frames)):
        if not obj[t].any():  # 物体不在画面里（例如被拿出画面）→ 不贴
            out.append(base[t])
            region.append(np.zeros((h, w), np.uint8))
            continue
        if t == key:
            Hm = np.eye(3)
        else:
            Hm, ni = object_motion(frames[key], frames[t], obj_key, (cv2.dilate(obj[t].astype(np.uint8), K(4)) > 0) & ~keep[t], orb, bf)
            if Hm is None or ni < 20:
                excl = cv2.dilate((obj[t] | keep[t]).astype(np.uint8), K(15)) > 0
                Hm, ni = homography(key_g, cv2.cvtColor(base[t], cv2.COLOR_BGR2GRAY), key_excl, excl, orb, bf)
            if Hm is None or ni < 25:
                Hm = Hprev
        Hprev = Hm
        im = cv2.warpPerspective(img0, Hm, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(a0, Hm, (w, h), flags=cv2.INTER_LINEAR)
        a *= 1 - (cv2.dilate(keep[t].astype(np.uint8), K(3)) > 0)
        out.append((a[..., None] * im + (1 - a[..., None]) * base[t]).astype(np.uint8))
        region.append((a > 0.05).astype(np.uint8) * 255)
    return out, region, key


def edit(jid):
    job = load_job(jid)
    n, fps = job["frames"], job["fps"]
    frames = [imread(jdir(jid, "frames", f"{i:05d}.jpg")) for i in range(n)]
    zero = np.zeros(frames[0].shape[:2], bool)
    objs = {o["id"]: [m > 127 for m in tp.load_seq(jdir(jid, "masks", f"obj{o['id']}"), n, gray=True)] for o in job["objects"]}
    keep = [zero.copy() for _ in range(n)]
    raw = [zero.copy() for _ in range(n)]
    for o in job["objects"]:
        for i in range(n):
            if o["action"] == "keep":
                keep[i] |= objs[o["id"]][i]
            else:  # 删除和替换都先把原物体去掉；紧挨着的物体合成一个 mask 一起去（ADR-005）
                raw[i] |= objs[o["id"]][i]
    for i in range(n):
        raw[i] &= ~keep[i]
    t0 = time.time()
    update(jid, progress=10)
    out, stab, info = remove_auto(jid, frames, raw, keep, fps)
    edited = [s.copy() for s in stab]
    for o in job["objects"]:
        if o["action"] == "replace":
            asset = imread(jdir(jid, "assets", o["asset"]), cv2.IMREAD_UNCHANGED)
            if asset.ndim == 2:
                asset = cv2.cvtColor(asset, cv2.COLOR_GRAY2BGR)
            out, region, key = follow_paste(out, frames, objs[o["id"]], keep, asset)
            edited = [np.maximum(e, r) for e, r in zip(edited, region)]
            log(jid, f"replace '{o['name']}': placed on frame {key}, following the object's motion")
    update(jid, progress=90)
    tp.write_mp4(jdir(jid, "out", "result.mp4"), out, fps)
    rows = []
    for t in range(n):
        tiles = [cv2.resize(x, (360, round(360 * x.shape[0] / x.shape[1])), interpolation=cv2.INTER_AREA) for x in (frames[t], out[t])]
        rows.append(np.hstack([tp.label(tiles[0], f"original  {t / fps:4.1f}s"), tp.label(tiles[1], "edited")]))
    tp.write_mp4(jdir(jid, "out", "compare.mp4"), rows, fps)
    idxs = [t for t in range(n) if edited[t].any()]
    m = tp.metrics(frames, {"original": frames, "result": out}, edited, {"original": edited, "result": edited}, idxs) if idxs else {}
    result = {"mode": info.get("mode"), "info": {k: v for k, v in info.items() if k != "mode"}, "seconds": round(time.time() - t0, 1),
              "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in m.items()}}
    update(jid, stage="edited", result=result)
    log(jid, f"edit: {result['mode']}; flicker original {m.get('original', {}).get('flicker_med')} → result {m.get('result', {}).get('flicker_med')}")


# ---------- HTTP ----------
def new_job(name):
    jid = uuid.uuid4().hex[:8]
    os.makedirs(jdir(jid, "out"), exist_ok=True)
    save_job({"id": jid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "stage": "new", "log": [], "busy": None})
    return jid


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, download=None):
        if not os.path.isfile(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        parts = p.strip("/").split("/")
        if p in ("/", "/index.html"):
            return self.send_file(os.path.join(ROOT, "web", "index.html"))
        if p == "/api/samples":
            return self.send_json(sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))))
        if p == "/api/sample_asset":
            return self.send_file(os.path.join(ROOT, "assets", "black_book.png"))
        if len(parts) >= 3 and parts[:2] == ["api", "jobs"] and ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            jid = parts[2]
            if len(parts) == 3:
                return self.send_json(load_job(jid))
            if len(parts) == 5 and parts[3] == "frame" and parts[4].isdigit():
                return self.send_file(jdir(jid, "frames", f"{int(parts[4]):05d}.jpg"))
            if len(parts) == 5 and parts[3] == "asset" and NAME_RE.match(parts[4]):
                return self.send_file(jdir(jid, "assets", parts[4]))
            if len(parts) == 5 and parts[3] in ("file", "download") and NAME_RE.match(parts[4]):
                path = jdir(jid, "original.mp4") if parts[4] == "original.mp4" else jdir(jid, "out", parts[4])
                return self.send_file(path, download=f"{jid}_{parts[4]}" if parts[3] == "download" else None)
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        parts = p.strip("/").split("/")
        if p == "/api/upload":  # 原始字节上传，文件名放在 X-Filename
            name = os.path.basename(unquote(self.headers.get("X-Filename", "video.mp4")))
            data = self.body()
            jid = new_job(name)
            src = jdir(jid, "source" + os.path.splitext(name)[1].lower())
            with open(src, "wb") as f:
                f.write(data)
            run_bg(jid, "prepare", prepare, src)
            return self.send_json({"id": jid})
        if p == "/api/sample":
            name = json.loads(self.body() or b"{}").get("name", "")
            if name not in os.listdir(SAMPLES):
                return self.send_json({"error": "unknown sample"}, 400)
            jid = new_job(name)
            src = jdir(jid, "source" + os.path.splitext(name)[1].lower())
            shutil.copyfile(os.path.join(SAMPLES, name), src)
            run_bg(jid, "prepare", prepare, src)
            return self.send_json({"id": jid})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and ID_RE.match(parts[2]) and os.path.isdir(jdir(parts[2])):
            jid, action = parts[2], parts[3]
            job = load_job(jid)
            if job.get("busy"):
                return self.send_json({"error": f"still running: {job['busy']}"}, 409)
            if action == "asset":  # 替换用的图片（PNG 可以带透明背景）
                data = self.body()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    return self.send_json({"error": "not an image"}, 400)
                name = f"asset_{uuid.uuid4().hex[:6]}.png"
                imwrite(jdir(jid, "assets", name), img)
                return self.send_json({"asset": name})
            if action == "sample_asset":
                os.makedirs(jdir(jid, "assets"), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, "assets", "black_book.png"), jdir(jid, "assets", "black_book.png"))
                return self.send_json({"asset": "black_book.png"})
            if action == "track":
                objects = json.loads(self.body())["objects"]
                for o in objects:
                    if o.get("action") not in ("remove", "replace", "keep"):
                        return self.send_json({"error": "bad action"}, 400)
                    if o["action"] == "replace" and not (o.get("asset") and NAME_RE.match(o["asset"])):
                        return self.send_json({"error": f"choose a replacement image for '{o['name']}'"}, 400)
                if not any(o["action"] in ("remove", "replace") for o in objects):
                    return self.send_json({"error": "choose at least one object to remove or replace"}, 400)
                run_bg(jid, "track", track, objects)
                return self.send_json({"ok": True})
            if action == "edit":
                if job.get("stage") not in ("tracked", "edited"):
                    return self.send_json({"error": "track the objects first"}, 400)
                run_bg(jid, "edit", edit)
                return self.send_json({"ok": True})
        self.send_error(404)


def main():
    os.makedirs(JOBS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Video editor: {url}  (Ctrl+C to stop)", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Editor stopped.")


if __name__ == "__main__":
    main()
