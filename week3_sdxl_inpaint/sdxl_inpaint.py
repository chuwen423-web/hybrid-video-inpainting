"""路线 2：通过 ComfyUI 跑 SDXL Inpainting（diffusers/stable-diffusion-xl-1.0-inpainting-0.1）。

前提：ComfyUI 已经在 http://127.0.0.1:8188 运行（双击 start_comfyui.bat）。

流程（每个 case）：
  1. 读第一周 route2 导出的模型输入：week1_image_validation/outputs/model_inputs/<id>_image.png / <id>_mask.png
     （mask 白色 = 要重画的区域，已经按 cases.json 外扩过）
  2. 放大到最长边 1024（SDXL 的原生分辨率，边长取 8 的倍数），上传到 ComfyUI
  3. 提交工作流：UNETLoader + DualCLIPLoader(sdxl) + VAELoader → InpaintModelConditioning → KSampler → VAEDecode
  4. 每个种子一张结果，缩回原图尺寸，存到 outputs/<id>/seed<k>.png，并生成多种子对比图 results/<id>_seeds.png
  删除任务可选 prefill 变体（config 里的 prefill）：输入图先换成第一周路线 1 的 Telea 修补结果，
  再用较低 denoise 精修——目标已经被抹掉，模型就不会按 mask 的轮廓把物体“补回来”。
  5. 第一个种子的结果交给第一周统一评测（每个变体各两张）：
       week1 outputs/route2_inpaint/<id>__sdxl_inpaint.png            模型原始输出（mask 外也会被 VAE 轻微改动）
       week1 outputs/route2_inpaint/<id>__sdxl_inpaint+pasteback.png  只取 mask 内的生成内容贴回原图（第一周的羽化合成）

用法：python sdxl_inpaint.py                   # config/sdxl_cases.json 里的全部 case
     python sdxl_inpaint.py coffee_replace_spoon
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WEEK1 = os.path.join(os.path.dirname(ROOT), "week1_image_validation")
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
sys.path.append(WEEK1)
from tools import composite, mask_ops, viz  # noqa: E402  第一周的合成 / 可视化工具

MODELS = {"unet": "sdxl_inpaint_0.1_unet.fp16.safetensors", "clip_l": "sdxl_clip_l.fp16.safetensors",
          "clip_g": "sdxl_clip_g.fp16.safetensors", "vae": "sdxl_inpaint_0.1_vae.safetensors"}


def read_img(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(path)


def api(path, data=None, raw=False):
    req = urllib.request.Request(COMFY + path, data=data,
                                 headers={"Content-Type": "application/json"} if data is not None and not raw else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    return body if raw else json.loads(body)


def upload(img, name):
    """multipart 上传到 ComfyUI 的 input 目录（同名覆盖）。"""
    boundary = uuid.uuid4().hex
    png = cv2.imencode(".png", img)[1].tobytes()
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
             f'Content-Type: image/png\r\n\r\n'.encode() + png + b"\r\n",
             f"--{boundary}--\r\n".encode()]
    req = urllib.request.Request(COMFY + "/upload/image", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def build_workflow(image_name, mask_name, prompt, negative, seed, s, prefix):
    """ComfyUI API 格式的工作流（节点 id → {class_type, inputs}）。"""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": MODELS["clip_l"], "clip_name2": MODELS["clip_g"], "type": "sdxl"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "8": {"class_type": "InpaintModelConditioning", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["3", 0], "pixels": ["4", 0], "mask": ["5", 0], "noise_mask": True}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": s["steps"], "cfg": s["cfg"], "sampler_name": s["sampler_name"],
            "scheduler": s["scheduler"], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["8", 2], "denoise": s["denoise"]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def run_workflow(wf):
    cid = uuid.uuid4().hex
    pid = api("/prompt", json.dumps({"prompt": wf, "client_id": cid}).encode())["prompt_id"]
    t0 = time.perf_counter()
    while True:
        hist = api(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages", []))[:2000])
            imgs = [im for out in h.get("outputs", {}).values() for im in out.get("images", [])]
            if imgs:
                break
        if time.perf_counter() - t0 > 600:
            raise TimeoutError(pid)
        time.sleep(0.5)
    im = imgs[0]
    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
    data = api(f"/view?{q}", raw=True)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), time.perf_counter() - t0


def model_size(h, w, long_side):
    s = long_side / max(h, w)
    return int(round(w * s / 8)) * 8, int(round(h * s / 8)) * 8


def main():
    with open(os.path.join(ROOT, "config", "sdxl_cases.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    only = set(sys.argv[1:])
    api("/system_stats")  # ComfyUI 没启动时这里直接报错
    rows = []
    for cid, c in cfg["cases"].items():
        if only and cid not in only:
            continue
        img_p = os.path.join(WEEK1, "outputs", "model_inputs", f"{cid}_image.png")
        mask_p = os.path.join(WEEK1, "outputs", "model_inputs", f"{cid}_mask.png")
        if not os.path.exists(img_p):
            print(f"[sdxl] {cid}: 找不到 {img_p}，先在第一周运行 run_all.py")
            continue
        img, mask = read_img(img_p), read_img(mask_p, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        w2, h2 = model_size(H, W, cfg["long_side"])
        img_up = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_LANCZOS4)
        mask_up = cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_NEAREST)
        mask_name = upload(cv2.cvtColor(mask_up, cv2.COLOR_GRAY2BGR), f"{cid}_mask.png")
        negative = ", ".join(x for x in (c.get("negative"), cfg["negative_common"]) if x)

        variants = [("sdxl_inpaint", img, cfg["sampler"]["denoise"])]
        if c.get("prefill"):
            variants.append(("sdxl_inpaint_prefill", read_img(os.path.join(WEEK1, c["prefill"])), c.get("prefill_denoise", 0.75)))
        tiles = [("original", img), ("mask", mask_ops.overlay(img, mask))]
        if c.get("prefill"):
            tiles.append(("prefill: OpenCV Telea", variants[1][1]))
        for vname, src, denoise in variants:
            src_up = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_LANCZOS4)
            img_name = upload(src_up, f"{cid}_{vname}_image.png")
            s = dict(cfg["sampler"], denoise=denoise)
            for k, seed in enumerate(cfg["seeds"]):
                wf = build_workflow(img_name, mask_name, c["prompt"], negative, seed, s, f"week3/{cid}_{vname}_s{seed}")
                if k == 0:
                    os.makedirs(os.path.join(ROOT, "workflows"), exist_ok=True)
                    with open(os.path.join(ROOT, "workflows", f"{cid}__{vname}_api.json"), "w", encoding="utf-8") as f:
                        json.dump(wf, f, ensure_ascii=False, indent=2)
                out, dt = run_workflow(wf)
                out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
                write_img(os.path.join(ROOT, "outputs", cid, f"{vname}_seed{seed}.png"), out)
                label = "SDXL" if vname == "sdxl_inpaint" else f"prefill+SDXL d{denoise}"
                tiles.append((f"{label} seed {seed}", out))
                rows.append({"case": cid, "variant": vname, "seed": seed, "denoise": denoise, "size": f"{w2}x{h2}", "seconds": round(dt, 2)})
                print(f"[sdxl] {cid} {vname} seed {seed}: {w2}x{h2}, denoise {denoise}, {dt:.1f} s")
                if k == 0:  # 交给第一周统一评测
                    r2 = os.path.join(WEEK1, "outputs", "route2_inpaint")
                    write_img(os.path.join(r2, f"{cid}__{vname}.png"), out)
                    pasted = composite.alpha_composite(img, out, mask_ops.feather(mask, 6))
                    write_img(os.path.join(r2, f"{cid}__{vname}+pasteback.png"), pasted)
        cols = 3 if len(tiles) % 3 == 0 or len(tiles) < 7 else 4
        write_img(os.path.join(ROOT, "results", f"{cid}_seeds.png"), viz.grid(tiles, cols=cols, cell_w=360))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "timing.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["case", "variant", "seed", "denoise", "size", "seconds"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
