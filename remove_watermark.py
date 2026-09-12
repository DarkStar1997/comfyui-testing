#!/usr/bin/env python3
"""Watermark removal for single-image PDFs.

Extracts the certificate image from the PDF, serves a browser masking UI on
localhost, inpaints the painted region with LaMa (inside the comfyui
container), and rebuilds the PDF with the cleaned image. Iterative: after each
pass the cleaned image becomes the new working base.

Usage: python remove_watermark.py FILE.pdf [--port 8788] [--backend lama|flux]
"""
import argparse
import base64
import io
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_fetch import ensure_models

REPO = Path(__file__).resolve().parent
CONTAINER = "comfyui"
LAMA_MODELS = [("inpaint",
                 "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
                 "big-lama.pt")]

FLUX_FILES = ["models/diffusion_models/flux1-fill-dev.safetensors",
              "models/text_encoders/t5xxl_fp8_e4m3fn.safetensors",
              "models/text_encoders/clip_l.safetensors",
              "models/vae/ae.safetensors"]
COMFY = "http://localhost:8188"
FLUX_PROMPT = ("a clean scan of an official certificate document on white paper, "
               "crisp black printed text, seals and signatures intact, no watermark")

LAMA_RUNNER = r"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

img = Image.open(os.environ["IN"]).convert("RGB")
mask = Image.open(os.environ["MASK"]).convert("L")
w, h = img.size
W, H = (w + 7) // 8 * 8, (h + 7) // 8 * 8

im = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 255
ms = torch.from_numpy(np.asarray(mask)).float()[None, None] / 255
im = F.pad(im, (0, W - w, 0, H - h), "reflect")
ms = F.pad(ms, (0, W - w, 0, H - h), "constant", 1.0)

model = torch.jit.load(os.environ["MODEL"], map_location="cuda")
with torch.no_grad():
    out = model(im.cuda(), ms.cuda())
res = (out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")[:h, :w]
Image.fromarray(res).save(os.environ["OUT"])
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Watermark remover</title><style>
body{font-family:system-ui;margin:16px;background:#1e1e1e;color:#ddd}
#wrap{position:relative;display:inline-block;line-height:0;box-shadow:0 0 12px #000}
canvas{position:absolute;top:0;left:0;max-width:100%;pointer-events:none}
#base{position:relative;max-width:100%;display:block;cursor:crosshair}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
button,select,input{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 10px}
button{cursor:pointer} button.active{background:#0a6} #finish{background:#0a6d54;border-color:#0a6}
#status{margin-left:8px}
</style></head><body>
<div class="bar">
 <select id="tool"><option value="brush">Brush</option><option value="rect">Rectangle</option>
 <option value="eraser">Eraser</option></select>
 <label>Size <input id="size" type="range" min="6" max="140" value="40"> <span id="szv">40</span></label>
 <button id="undo">Undo</button> <button id="clear">Clear</button>
 <button id="finish">Remove watermark &amp; rebuild PDF</button>
 <span id="status"></span>
</div>
<div id="wrap"><img id="base" src="/base.png"><canvas id="mask"></canvas><canvas id="view"></canvas></div>
<script>
const base=document.getElementById('base'), mask=document.getElementById('mask'),
      view=document.getElementById('view'), ctxM=mask.getContext('2d'),
      ctxV=view.getContext('2d'), status=document.getElementById('status');
let strokes=[], drawing=false, sx=0, sy=0, tool='brush', size=40, rect=null;
base.onload=()=>{mask.width=view.width=base.naturalWidth;
                 mask.height=view.height=base.naturalHeight;ctxV.clearRect(0,0,view.width,view.height);
 fetch('/mask.png').then(r=>r.ok?r.blob():null).then(b=>{if(!b)return;
   const i=new Image();i.onload=()=>{ctxM.drawImage(i,0,0);redraw();};i.src=URL.createObjectURL(b);});};
base.src='/base.png?'+Date.now();
document.getElementById('tool').onchange=e=>tool=e.target.value;
document.getElementById('size').oninput=e=>{size=+e.target.value;document.getElementById('szv').textContent=size;};
function pos(e){const r=base.getBoundingClientRect();
  return [(e.clientX-r.left)*base.naturalWidth/r.width,(e.clientY-r.top)*base.naturalHeight/r.height];}
function dot(c,x,y,erase){c.globalCompositeOperation=erase?'destination-out':'source-over';
  c.fillStyle=erase?'rgba(0,0,0,1)':(c===ctxM?'#fff':'rgba(255,40,40,.55)');c.beginPath();c.arc(x,y,size/2,0,7);c.fill();}
function line(x1,y1,x2,y2,erase){for(const c of [ctxM,ctxV]){
  c.save();c.globalCompositeOperation=erase?'destination-out':'source-over';
  c.strokeStyle=erase?'rgba(0,0,0,1)':(c===ctxM?'#fff':'rgba(255,40,40,.55)');c.lineWidth=size;c.lineCap='round';c.lineJoin='round';
  c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();c.restore();}}
function redraw(){ctxV.clearRect(0,0,view.width,view.height);ctxV.drawImage(mask,0,0);}
base.onpointerdown=e=>{e.preventDefault();base.setPointerCapture(e.pointerId);drawing=true;
  [sx,sy]=pos(e);if(tool==='brush'||tool==='eraser'){strokes.push(ctxM.getImageData(0,0,mask.width,mask.height));
  if(strokes.length>12)strokes.shift();line(sx,sy,sx+0.01,sy+0.01,tool==='eraser');}
  else{rect=[sx,sy,sx,sy];}};
base.onpointermove=e=>{if(!drawing)return;const[x,y]=pos(e);
  if(tool==='brush'||tool==='eraser')line(sx,sy,x,y,tool==='eraser'),[sx,sy]=[x,y];
  else{rect[2]=x;rect[3]=y;redraw();ctxV.strokeStyle='rgba(255,40,40,.6)';ctxV.lineWidth=2;
   ctxV.strokeRect(...rect.map((v,i)=>i<2?v:Math.max(v,0)).slice(0,2),Math.abs(rect[2]-rect[0]),Math.abs(rect[3]-rect[1]));}};
base.onpointerup=e=>{if(!drawing)return;drawing=false;
  if(tool==='rect'&&rect){strokes.push(ctxM.getImageData(0,0,mask.width,mask.height));
   if(strokes.length>12)strokes.shift();ctxM.fillStyle='#fff';
   ctxM.fillRect(Math.min(rect[0],rect[2]),Math.min(rect[1],rect[3]),Math.abs(rect[2]-rect[0]),Math.abs(rect[3]-rect[1]));redraw();}};
document.getElementById('undo').onclick=()=>{const s=strokes.pop();if(s)ctxM.putImageData(s,0,0);redraw();};
document.getElementById('clear').onclick=()=>{ctxM.clearRect(0,0,mask.width,mask.height);redraw();};
document.getElementById('finish').onclick=async()=>{
  const d=ctxM.getImageData(0,0,mask.width,mask.height);let painted=0;
  for(let i=3;i<d.data.length;i+=400)if(d.data[i]>10){painted=1;break;}
  if(!painted){status.textContent='paint the watermark first';return;}
  status.textContent='inpainting...';document.getElementById('finish').disabled=true;
  const t=document.createElement('canvas');t.width=mask.width;t.height=mask.height;
  t.getContext('2d').fillStyle='#000';t.getContext('2d').fillRect(0,0,t.width,t.height);
  t.getContext('2d').drawImage(mask,0,0);
  const r=await fetch('/finish',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mask:t.toDataURL('image/png')})});
  const j=await r.json();
  status.textContent=j.ok?('done: '+j.pdf+' — paint more if needed, then finish again'):('error: '+j.error);
  document.getElementById('finish').disabled=false;
  if(j.ok){strokes=[];ctxM.clearRect(0,0,mask.width,mask.height);base.src='/base.png?'+Date.now();}
};
</script></body></html>"""


class State:
    def __init__(self, pdf_path, work):
        self.pdf = pdf_path
        self.work = work
        self.lock = threading.Lock()
        self.base = work / "base.png"
        self.out_pdf = pdf_path.with_name(pdf_path.stem + "_clean.pdf")
        self.xref = None


def run_lama(st):
    env = {"IN": "/ComfyUI/" + str(st.base.relative_to(REPO)),
           "MASK": "/ComfyUI/" + str((st.work / "mask.png").relative_to(REPO)),
           "OUT": "/ComfyUI/" + str((st.work / "clean.png").relative_to(REPO)),
           "MODEL": "/ComfyUI/" + str(Path("models/inpaint/big-lama.pt"))}
    cmd = ["docker", "exec", "-i"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [CONTAINER, "python", "-"]
    r = subprocess.run(cmd, input=LAMA_RUNNER, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])


def _api(path, payload=None):
    req = urllib.request.Request(
        COMFY + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_flux(st, size=1440, steps=20):
    base = Image.open(st.base).convert("RGB")
    mask = Image.open(st.work / "mask.png").convert("L")
    w, h = base.size
    s = size / max(w, h)
    fw, fh = max(16, round(w * s / 16) * 16), max(16, round(h * s / 16) * 16)

    small = base.resize((fw, fh), Image.LANCZOS)
    small_m = mask.resize((fw, fh), Image.LANCZOS)
    rgba = small.convert("RGBA")
    rgba.putalpha(small_m.point(lambda v: 255 - v))  # painted -> transparent
    rel = st.work.relative_to(REPO / "input")
    rgba.save(REPO / "input" / rel / "flux_in.png")

    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-fill-dev.safetensors",
                         "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                         "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": str(rel / "flux_in.png")}},
        "5": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": FLUX_PROMPT}},
        "7": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 30}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "9": {"class_type": "InpaintModelConditioning",
              "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["3", 0],
                         "pixels": ["4", 0], "mask": ["4", 1], "noise_mask": True}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["5", 0], "positive": ["9", 0], "negative": ["9", 1],
                          "latent_image": ["9", 2], "seed": random.randint(0, 2**48),
                          "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "normal", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": str(rel / "flux_out")}},
    }
    pid = _api("/prompt", {"prompt": wf, "client_id": "wm"})["prompt_id"]

    for _ in range(600):
        time.sleep(3)
        hist = _api(f"/history/{pid}").get(pid)
        if not hist:
            continue
        if hist.get("status", {}).get("status_str") == "error":
            msgs = [m for n, m in hist.get("status", {}).get("messages", []) if n == "execution_error"]
            raise RuntimeError(f"ComfyUI error: {msgs[-1] if msgs else hist['status']}"[:600])
        outs = hist.get("outputs", {})
        if outs:
            fname = next(iter(outs.values()))["images"][0]["filename"]
            sub = next(iter(outs.values()))["images"][0].get("subfolder", "")
            src = REPO / "output" / sub / fname
            full = Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)
            m = mask.point(lambda v: v > 0 and 255)
            Image.composite(full, base, m).save(st.work / "clean.png")
            return
    raise RuntimeError("flux timeout (30min)")


class Handler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        st = self.state
        if self.path.startswith("/base.png"):
            self._send(200, st.base.read_bytes(), "image/png")
        elif self.path.startswith("/mask.png"):
            p = st.work / "mask.png"
            self._send(200, p.read_bytes(), "image/png") if p.exists() else self._send(404, b"")
        else:
            self._send(200, PAGE.encode(), "text/html")

    def do_POST(self):
        st = self.state
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n))
        try:
            with st.lock:
                raw = base64.b64decode(data["mask"].split(",", 1)[1])
                mask = Image.open(io.BytesIO(raw)).convert("L")
                mask = mask.filter(ImageFilter.MaxFilter(7))
                mask = mask.point(lambda v: 255 if v > 10 else 0)
                mask.save(st.work / "mask.png")
                if self.backend == "flux":
                    run_flux(st, self.flux_size)
                else:
                    run_lama(st)
                img = Image.open(st.work / "clean.png").convert("RGB")
                img.save(st.work / "clean.jpg", quality=94)
                doc = pymupdf.open(st.pdf)
                doc[0].replace_image(st.xref, filename=str(st.work / "clean.jpg"))
                doc.save(st.out_pdf, garbage=3, deflate=True)
                doc.close()
                img.save(st.base)
            self._send(200, {"ok": True, "pdf": str(st.out_pdf)})
        except Exception as e:
            self._send(200, {"ok": False, "error": str(e)[-400:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--backend", choices=["lama", "flux"], default="flux")
    ap.add_argument("--flux-size", type=int, default=1440)
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    doc = pymupdf.open(pdf)
    images = doc[0].get_images(full=True)
    if len(images) != 1:
        sys.exit(f"expected exactly 1 image on page 1, found {len(images)}")
    xref = images[0][0]
    info = doc.extract_image(xref)
    doc.close()

    work = REPO / "input" / "wm_work" / pdf.stem
    work.mkdir(parents=True, exist_ok=True)
    (work / "base.png").write_bytes(info["image"]) if info["ext"] == "png" \
        else Image.open(io.BytesIO(info["image"])).convert("RGB").save(work / "base.png")

    if args.backend == "flux":
        missing = [f for f in FLUX_FILES if not (REPO / f).is_file()]
        if missing:
            sys.exit("missing flux models:\n  " + "\n  ".join(missing))
    else:
        ensure_models(LAMA_MODELS)

    Handler.state = State(pdf, work)
    Handler.state.xref = xref
    Handler.backend = args.backend
    Handler.flux_size = args.flux_size
    url = f"http://127.0.0.1:{args.port}"
    print(f"masking UI: {url}  ->  output: {Handler.state.out_pdf}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
