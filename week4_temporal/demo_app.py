"""Week 4 演示界面（原型演示用）：选视频 → 看检测确认 → 勾选要删 / 要换的物体 → 选方法 → 并排同步播放 + 指标。

用法：双击 start_demo.bat（或 python demo_app.py），浏览器打开 http://127.0.0.1:8766
只监听本机；不需要联网，不装新库（用第二周的 .venv）。演示时播放已经算好的结果；
“Re-run”按钮会在后台重新跑 temporal.py（只用缓存的 SAM 2 mask，每段约 1–3 分钟）。
"""
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
W3 = os.path.join(os.path.dirname(ROOT), "week3_video_pipeline")
CLIPS = ["clip_A", "clip_B", "clip_C", "clip_D"]
H3_CLIPS = {"clip_B"}  # 只有 MiniMax H3 结果的对照组（eval_h3.py 算指标），不能在这里重跑
SCENES = {"clip_A": "Fixed camera: paper cup + paper bag",
          "clip_B": "Handheld, 4 s excerpt: paper bag → book from a reference image (MiniMax H3)",
          "clip_C": "A hand picks up the cup, puts it back and pushes the bag (bag → black book)",
          "clip_D": "Camera pans across an office; the cup is only visible mid-video"}
PORT = 8766
FILE_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,80}$")
JOB = {"running": False, "clip": None, "log": [], "code": None}
LOCK = threading.Lock()


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def h3_info(cid):
    h3 = read_json(os.path.join(ROOT, "results", f"{cid}_h3.json")) or {}
    res = os.path.join(ROOT, "results")
    return {"id": cid, "scene": SCENES[cid], "fps": h3.get("fps"), "frames": h3.get("frames"), "h3_only": True,
            "source_fps": h3.get("source_fps"), "edited_area_pct": h3.get("edited_area_pct"),
            "objects": [{"id": "book", "label": "Replace 'paper bag' with the book in the reference image", "action": "replace",
                         "default": True, "fixed": True,
                         "source": "MiniMax H3 ref2v: reference image + text prompt, the whole clip is regenerated (no mask)"}],
            "metrics": h3.get("metrics"), "w4": {"h3": h3.get("h3")}, "sheet": f"{cid}_masks.jpg",
            "videos": sorted(f for f in os.listdir(res) if f.startswith(cid + "__") and f.endswith(".mp4"))}


def clip_info(cid):
    if cid in H3_CLIPS:
        return h3_info(cid)
    cfg = read_json(os.path.join(W3, "config", "clips.json"))["clips"][cid]
    w3 = read_json(os.path.join(W3, "work", cid, "info.json")) or {}
    w4 = read_json(os.path.join(ROOT, "results", f"{cid}_w4.json")) or {}
    objects = [{"id": "cup", "label": f"Remove '{cfg['remove']['target']}'", "source": "YOLOX detection (OpenCV dnn) → SAM 2 tracking",
                "frames": w3.get("masked_frames"), "detected": w3.get("detected_frames"), "action": "remove", "default": True}]
    if "replace" in cfg:
        objects.append({"id": "bag", "label": "Replace 'paper bag' with a black book", "action": "replace", "default": True,
                        "source": "not a COCO class → user prompt: 1 box + 3 clicks on frame 0 → SAM 2"})
    if cfg.get("track_classes"):
        objects.append({"id": "hand", "label": "Keep the hand in front (occlusion)", "action": "keep", "default": True,
                        "source": "YOLOX 'person' → SAM 2", "fixed": True})
    res = os.path.join(ROOT, "results")
    return {"id": cid, "scene": SCENES[cid], "fps": w3.get("fps"), "frames": w3.get("frames"),
            "segments": w3.get("segments"), "keyframes": w3.get("keyframes"), "objects": objects,
            "metrics": w4.get("metrics"), "w4": {k: w4.get(k) for k in ("plate", "w4", "homography", "ms_masks", "seconds_total",
                                                                       "lama_flow_ms_per_frame", "w3_lama_ms_per_frame", "per_object", "book")},
            "videos": sorted(f for f in os.listdir(res) if f.startswith(cid + "__") and f.endswith(".mp4")) if os.path.isdir(res) else []}


def run_job(cid):
    with LOCK:
        JOB.update(running=True, clip=cid, log=[f"$ python temporal.py {cid}"], code=None)
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "temporal.py"), cid], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"})
    for line in p.stdout:
        line = line.rstrip()
        if line and not line.startswith(("[ERROR", "[libopen", "OpenCV", "Failed to load OpenH264", "\tPlease")):
            with LOCK:
                JOB["log"] = (JOB["log"] + [line])[-200:]
    with LOCK:
        JOB.update(running=False, code=p.wait())


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

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):  # 视频拖动进度条需要 Range
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
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

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.send_file(os.path.join(ROOT, "web", "index.html"))
        if path == "/api/clips":
            return self.send_json([clip_info(c) for c in CLIPS])
        if path == "/api/status":
            with LOCK:
                return self.send_json(dict(JOB))
        for prefix, base in (("/w4/", os.path.join(ROOT, "results")), ("/w3/", os.path.join(W3, "results"))):
            if path.startswith(prefix):
                name = path[len(prefix):]
                if not FILE_RE.match(name):  # 只允许结果文件夹里的文件名，不允许子路径
                    return self.send_error(400)
                return self.send_file(os.path.join(base, name))
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if path == "/api/run":
            cid = body.get("clip")
            if cid not in CLIPS:
                return self.send_json({"error": "unknown clip"}, 400)
            if cid in H3_CLIPS:
                return self.send_json({"error": "MiniMax H3 results are pre-computed; they cannot be re-run here"}, 400)
            with LOCK:
                if JOB["running"]:
                    return self.send_json({"error": f"{JOB['clip']} is still running"}, 409)
            threading.Thread(target=run_job, args=(cid,), daemon=True).start()
            return self.send_json({"ok": True})
        self.send_error(404)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Week 4 demo: {url}  (Ctrl+C to stop)", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:  # Ctrl+C 正常退出，不打印 Traceback
        print("Demo stopped.")


if __name__ == "__main__":
    main()
