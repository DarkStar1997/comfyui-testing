#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
INPUT_DIR = REPO_DIR / "input"
OUTPUT_DIR = REPO_DIR / "output" / "video"
HOST = "http://localhost:8188"
CHUNK_SECONDS = 10
WIDTH, HEIGHT = 1344, 768

TEMPLATE = json.loads(r"""{
 "92": {
  "inputs": {
   "filename_prefix": "video/MiniMax_H3",
   "format": "auto",
   "format.codec": "auto",
   "codec": "auto",
   "video": [
    "130",
    0
   ]
  },
  "class_type": "SaveVideo",
  "_meta": {
   "title": "Save Video"
  }
 },
 "115": {
  "inputs": {
   "aspect_ratio": "16:9 (Widescreen)",
   "megapixels": 0.4,
   "multiple": 32
  },
  "class_type": "ResolutionSelector",
  "_meta": {
   "title": "Resolution Selector (Size)"
  }
 },
 "119": {
  "inputs": {
   "vae_name": "minimax_h3_video_vae_fp16.safetensors"
  },
  "class_type": "VAELoader",
  "_meta": {
   "title": "Load VAE"
  }
 },
 "120": {
  "inputs": {
   "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
  },
  "class_type": "VAELoader",
  "_meta": {
   "title": "Load VAE"
  }
 },
 "121": {
  "inputs": {
   "samples": [
    "125",
    0
   ],
   "vae": [
    "120",
    0
   ]
  },
  "class_type": "VAEDecodeAudio",
  "_meta": {
   "title": "VAE Decode Audio"
  }
 },
 "122": {
  "inputs": {
   "samples": [
    "125",
    0
   ],
   "vae": [
    "119",
    0
   ]
  },
  "class_type": "VAEDecode",
  "_meta": {
   "title": "VAE Decode"
  }
 },
 "123": {
  "inputs": {
   "sampler_name": "res_multistep"
  },
  "class_type": "KSamplerSelect",
  "_meta": {
   "title": "KSamplerSelect"
  }
 },
 "124": {
  "inputs": {
   "scheduler": "simple",
   "steps": [
    "142",
    0
   ],
   "denoise": 1,
   "model": [
    "127",
    0
   ]
  },
  "class_type": "BasicScheduler",
  "_meta": {
   "title": "BasicScheduler"
  }
 },
 "125": {
  "inputs": {
   "noise": [
    "129",
    0
   ],
   "guider": [
    "126",
    0
   ],
   "sampler": [
    "123",
    0
   ],
   "sigmas": [
    "124",
    0
   ],
   "latent_image": [
    "136",
    1
   ]
  },
  "class_type": "SamplerCustomAdvanced",
  "_meta": {
   "title": "SamplerCustomAdvanced"
  }
 },
 "126": {
  "inputs": {
   "model": [
    "141",
    0
   ],
   "conditioning": [
    "136",
    0
   ]
  },
  "class_type": "BasicGuider",
  "_meta": {
   "title": "Basic Guider"
  }
 },
 "127": {
  "inputs": {
   "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
   "weight_dtype": "default"
  },
  "class_type": "UNETLoader",
  "_meta": {
   "title": "Load Diffusion Model"
  }
 },
 "128": {
  "inputs": {
   "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
   "type": "minimax",
   "device": "default"
  },
  "class_type": "CLIPLoader",
  "_meta": {
   "title": "Load CLIP"
  }
 },
 "129": {
  "inputs": {
   "noise_seed": 261662374822964
  },
  "class_type": "RandomNoise",
  "_meta": {
   "title": "RandomNoise"
  }
 },
 "130": {
  "inputs": {
   "fps": 24,
   "bit_depth": 8,
   "color_space": "sRGB",
   "images": [
    "122",
    0
   ],
   "audio": [
    "121",
    0
   ]
  },
  "class_type": "CreateVideo",
  "_meta": {
   "title": "Create Video"
  }
 },
 "131": {
  "inputs": {
   "expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
   "values.a": [
    "132",
    0
   ]
  },
  "class_type": "ComfyMathExpression",
  "_meta": {
   "title": "Math Expression"
  }
 },
 "132": {
  "inputs": {
   "value": 5
  },
  "class_type": "PrimitiveFloat",
  "_meta": {
   "title": "Float (Duration)"
  }
 },
 "136": {
  "inputs": {
   "prompt": [
    "138",
    0
   ],
   "width": [
    "115",
    0
   ],
   "height": [
    "115",
    1
   ],
   "length": [
    "131",
    1
   ],
   "ref_image_size": "match",
   "clip": [
    "128",
    0
   ],
   "vae": [
    "119",
    0
   ],
   "audio_vae": [
    "120",
    0
   ],
   "ref_images.ref_image_0": [
    "137",
    0
   ],
   "ref_images.ref_image_1": [
    "139",
    0
   ]
  },
  "class_type": "MiniMaxH3ReferenceToVideo",
  "_meta": {
   "title": "MiniMax H3 Reference to Video"
  }
 },
 "137": {
  "inputs": {
   "image": "red_superboy_on_city_roof.png"
  },
  "class_type": "LoadImage",
  "_meta": {
   "title": "Load Image"
  }
 },
 "138": {
  "inputs": {
   "value": "Bold comic-book ink style, heavy linework, red and blue-black palette, night city. Use <Picture 2> and <Picture 1> as reference frames and <Audio 1> exactly as it is.\nCUT 1: top-down view of the little boy superhero on the rooftop \u2014 red cape fluttering in the wind, hands planted on his hips, freckles and a cocky grin as he looks straight up into the camera. The camera slowly descends toward him as he delivers his line \u2014 as he speaks, comic-book graphic overlay text word by word in sync with his voice: \"GET READY TO\" - \"MEET\" \u2014 \"YOUR\" \u2014 \"MAKER\" \u2014 huge jagged comic lettering, white with heavy black outlines and red drop shadows, tilted at scrappy angles, until the three words hang stacked in the air above him between his face and the lens.\nTRANSITION: a violent WHIP PAN off the rooftop that SMEARS the floating words away with it, motion-streaked \u2014\nCUT 2: low hero angle on the colossal black mech-kaiju towering over the skyline as it rears back and unleashes a GIANT terrifying ROAR \u2014 jaws wide with fangs, red eyes and chest-core flaring blinding bright, blue lightning arcing off its head, the roar's shockwave rippling dust and rattling windows down the buildings, comic-style speed-lines and ink splatter bursting from the impact of the sound. It leans INTO the camera as the roar peaks. Hold on the roar."
  },
  "class_type": "PrimitiveStringMultiline",
  "_meta": {
   "title": "Input Text (Prompt)"
  }
 },
 "139": {
  "inputs": {
   "image": "mecha_dragon_lightning.png"
  },
  "class_type": "LoadImage",
  "_meta": {
   "title": "Load Image"
  }
 },
 "141": {
  "inputs": {
   "switch": [
    "146",
    0
   ],
   "on_false": [
    "127",
    0
   ],
   "on_true": [
    "145",
    0
   ]
  },
  "class_type": "ComfySwitchNode",
  "_meta": {
   "title": "If/Else Switch (model)"
  }
 },
 "142": {
  "inputs": {
   "switch": [
    "146",
    0
   ],
   "on_false": [
    "143",
    0
   ],
   "on_true": [
    "144",
    0
   ]
  },
  "class_type": "ComfySwitchNode",
  "_meta": {
   "title": "If/Else Switch (Steps)"
  }
 },
 "143": {
  "inputs": {
   "value": 20
  },
  "class_type": "PrimitiveInt",
  "_meta": {
   "title": "Int (Full)"
  }
 },
 "144": {
  "inputs": {
   "value": 4
  },
  "class_type": "PrimitiveInt",
  "_meta": {
   "title": "Int (Lightning LoRA)"
  }
 },
 "145": {
  "inputs": {
   "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
   "strength_model": 1,
   "model": [
    "127",
    0
   ]
  },
  "class_type": "LoraLoaderModelOnly",
  "_meta": {
   "title": "Load LoRA"
  }
 },
 "146": {
  "inputs": {
   "value": false
  },
  "class_type": "PrimitiveBoolean",
  "_meta": {
   "title": "Boolean (Enable Lightning LoRA)"
  }
 }
}""")


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def docker(*args):
    return subprocess.run(["docker", "exec", "comfyui", *args],
                          capture_output=True, text=True)


def find_output(prefix):
    files = sorted(OUTPUT_DIR.glob(prefix + "_*.mp4"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def extract_last_frame(mp4_name, png_name):
    r = docker("ffmpeg", "-y", "-sseof", "-0.05", "-i", f"/ComfyUI/output/video/{mp4_name}",
               "-update", "1", "-frames:v", "1", f"/ComfyUI/input/{png_name}")
    if r.returncode != 0:
        sys.exit(f"last-frame extraction failed: {r.stderr[-500:]}")


def crop_to_canvas(src_name, dst_name):
    r = docker("ffmpeg", "-y", "-i", f"/ComfyUI/input/{src_name}",
               "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}",
               "-frames:v", "1", f"/ComfyUI/input/{dst_name}")
    if r.returncode != 0:
        sys.exit(f"crop failed: {r.stderr[-500:]}")


def queueAndWait(prompt, name):
    pid = api("/prompt", {"prompt": prompt})["prompt_id"]
    print(f"[{name}] queued: {pid}", flush=True)
    deadline = time.time() + 3600
    empty_and_idle = 0
    while time.time() < deadline:
        time.sleep(10)
        try:
            hist = api(f"/history/{pid}")
        except urllib.error.URLError:
            hist = {}
        if hist:
            status = hist[pid].get("status", {})
            st = status.get("status_str")
            if st == "success":
                print(f"[{name}] success ({status.get('messages', []) and ''})"
                      f" elapsed ok", flush=True)
                return
            msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
            sys.exit(f"[{name}] failed: {st} {json.dumps(msgs)[:800]}")
        try:
            idle = api("/prompt")["exec_info"]["queue_remaining"] == 0
        except urllib.error.URLError:
            idle = False
        empty_and_idle = empty_and_idle + 1 if idle else 0
        if empty_and_idle >= 6:
            sys.exit(f"[{name}] lost: history empty and queue idle (server restart?)")
    sys.exit(f"[{name}] timed out")


def buildPrompt(idx, total, cfg, refs):
    p = dict(TEMPLATE)
    p = {k: dict(v) for k, v in p.items()}
    p["115"]["inputs"]["megapixels"] = 0.98
    p["132"]["inputs"]["value"] = CHUNK_SECONDS
    p["129"]["inputs"]["noise_seed"] = cfg["base_seed"] + idx
    p["145"]["inputs"]["lora_name"] = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    p["92"]["inputs"]["filename_prefix"] = f'video/{cfg["name"]}_chunk{idx:02d}'

    user_prompt = cfg["prompts"][min(idx, len(cfg["prompts"]) - 1)]
    if idx == 0:
        tags = " ".join(f"<Picture {i + 1}> shows the main character."
                        for i in range(len(refs) - 1, 0, -1))
        if cfg["start_image"]:
            p["137"]["inputs"]["image"] = cfg["start_image"]
            p["139"]["inputs"]["image"] = refs[-1]
            text = f"The scene begins from <Picture 1>. {tags} {user_prompt}"
        else:
            p["137"]["inputs"]["image"] = refs[0]
            p["139"]["inputs"]["image"] = refs[-1]
            text = f"{tags} {user_prompt}"
    else:
        p["137"]["inputs"]["image"] = f'{cfg["name"]}_chunk{idx - 1:02d}_last.png'
        p["139"]["inputs"]["image"] = refs[0]
        text = (f"Continue the exact scene from <Picture 1> - same character, wardrobe, "
                f"location, lighting and camera style, motion flowing seamlessly from that "
                f"exact frame. <Picture 2> shows the main character for identity. {user_prompt}")
    p["138"]["inputs"]["value"] = text

    if cfg["chain_mode"] == "frame":
        p["127"]["inputs"]["unet_name"] = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        first = p["137"]["inputs"]["image"] if idx > 0 or cfg["start_image"] else None
        if idx > 0 or cfg["start_image"]:
            p["136"] = {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["128", 0], "vae": ["119", 0], "prompt": ["138", 0],
                    "width": ["115", 0], "height": ["115", 1], "length": ["131", 1],
                    "first_frame": ["137", 0],
                },
            }
        else:
            sys.exit("frame chain-mode requires a start image for chunk 0")
    return p


def main():
    parser = argparse.ArgumentParser(description="Generate arbitrary-duration H3 videos in 10s chunks")
    parser.add_argument("--chain-mode", choices=["r2v", "frame"], default="r2v",
                        help="r2v: reference-conditioned continuation (default); frame: first-frame chaining")
    args = parser.parse_args()

    cfg = {
        "name": "coffee20",
        "total_duration_s": 20,
        "base_seed": 20260911,
        "prompts": [
            ("Photorealistic scene at a calm, unhurried pace in a small restaurant kitchen. A young "
             "chef in a white apron works at a relaxed, deliberate speed, each movement slow and "
             "controlled. The chef is COMPLETELY ALONE in the kitchen - exactly one person in the "
             "entire scene, no other chefs, no background staff, no customers, nobody else visible "
             "anywhere. Every hand movement is anatomically correct and physically plausible - "
             "natural fingers, wrists bending the right way, realistic grasp when holding pans and "
             "utensils. 0-4s: he calmly stirs a pan and checks a pot, composed and focused. "
             "4-7s: stepping across the kitchen he plants his foot on a wet mop and slips, falling "
             "gently onto his back. 7-10s: a slice of bread pops from the toaster with a soft ding "
             "as he props himself up. Camera: smooth, very slow tracking shot, fixed during the "
             "slip. Audio: soft kitchen ambience, light sizzling, a soft thud, toaster ding."),
            ("The chef remains COMPLETELY ALONE - exactly one person in frame, no other chefs, no "
             "background staff, nobody else appears. All movements stay anatomically correct and "
             "physically realistic - hands and wrists bend naturally, plates and cups are held with "
             "a correct, comfortable grip. 0-5s: the chef gets to his feet, dusts himself off and "
             "pours a fresh cup of coffee. 5-10s: he leans against the counter, sips slowly and "
             "relaxes with a satisfied smile as the morning rush settles around him. Camera: gentle "
             "medium close-up. Audio: soft kitchen ambience and a contented sip."),
        ],
        "ref_images": ["coffee_chef_still.png"],
        "start_image": "coffee_open_still.png",
        "chain_mode": args.chain_mode,
    }

    chunks = -(-cfg["total_duration_s"] // CHUNK_SECONDS)
    print(f"{cfg['name']}: {chunks} chunks of {CHUNK_SECONDS}s (mode {cfg['chain_mode']})", flush=True)

    for i, ref in enumerate(cfg["ref_images"]):
        src = REPO_DIR / ref
        if src.is_file():
            shutil.copy(src, INPUT_DIR / src.name)

    outputs = []
    for i in range(chunks):
        prefix = f'{cfg["name"]}_chunk{i:02d}'
        existing = find_output(prefix)
        if existing:
            print(f"[chunk {i}] exists: {existing.name} (skip)", flush=True)
        else:
            refs = cfg["ref_images"]
            prompt = buildPrompt(i, chunks, cfg, refs)
            queueAndWait(prompt, f"chunk {i}")
            existing = find_output(prefix)
            if not existing:
                sys.exit(f"[chunk {i}] no output found for prefix {prefix}")
        outputs.append(existing.name)
        last_png = f'{cfg["name"]}_chunk{i:02d}_last.png'
        if not (INPUT_DIR / last_png).is_file():
            extract_last_frame(existing.name, last_png)

    list_file = f'{cfg["name"]}_list.txt'
    entries = "\n".join(f"file '{o}'" for o in outputs) + "\n"
    r = subprocess.run(["docker", "exec", "-i", "comfyui", "sh", "-c",
                        f"cat > /ComfyUI/output/video/{list_file}"],
                       input=entries, text=True, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"write list failed: {r.stderr[-500:]}")
    r = subprocess.run(["docker", "exec", "comfyui", "sh", "-c",
                        f"cd /ComfyUI/output/video && ffmpeg -y -f concat -safe 0 -i {list_file} "
                        f"-c copy {cfg['name']}.mp4 && rm {list_file}"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"concat failed: {r.stderr[-500:]}")
    print(f"DONE: output/video/{cfg['name']}.mp4 ({chunks} x {CHUNK_SECONDS}s chunks)", flush=True)


if __name__ == "__main__":
    main()
