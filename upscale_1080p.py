#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER = "http://localhost:8188"
CONTAINER = "comfyui"
UNET = "seedvr2_7b_int8_convrot.safetensors"
VAE = "seedvr2_ema_vae_fp16.safetensors"
SEED = 959948902156062


def api(path, data=None):
    if data is None:
        with urllib.request.urlopen(SERVER + path) as r:
            return json.load(r)
    req = urllib.request.Request(SERVER + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def segment_prompt(video: str, start: float, duration: float, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "Video Slice", "inputs": {
            "video": ["1", 0], "start_time": start, "duration": duration,
            "strict_duration": True}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "4": {"class_type": "ResizeImageMaskNode", "inputs": {
            "resize_type": "scale dimensions", "resize_type.width": 1920,
            "resize_type.height": 1080, "resize_type.crop": "center",
            "scale_method": "lanczos", "input": ["3", 0]}},
        "5": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["4", 0]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "8": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64,
            "temporal_overlap": 8, "pixels": ["5", 0], "vae": ["6", 0]}},
        "9": {"class_type": "SeedVR2TemporalChunk", "inputs": {
            "temporal_overlap": 0, "chunking_mode": "auto", "latent": ["8", 0]}},
        "10": {"class_type": "SeedVR2Conditioning", "inputs": {
            "model": ["7", 0], "vae_conditioning": ["9", 0]}},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": SEED, "steps": 1, "cfg": 1, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1, "model": ["7", 0],
            "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["9", 0]}},
        "12": {"class_type": "SeedVR2TemporalMerge", "inputs": {
            "latents": ["11", 0], "temporal_overlap": ["9", 1]}},
        "13": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64,
            "temporal_overlap": 8, "samples": ["12", 0], "vae": ["6", 0]}},
        "14": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "none", "images": ["13", 0],
            "original_resized_images": ["4", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0], "audio": ["3", 1], "fps": ["3", 2],
            "bit_depth": ["3", 3], "color_space": "sRGB"}},
        "16": {"class_type": "SaveVideo", "inputs": {
            "filename_prefix": f"video/{prefix}", "format": "auto", "video": ["15", 0]}},
    }


def wait_done(pid: str, timeout: float) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = api(f"/history/{pid}")
        except Exception:
            time.sleep(5)
            continue
        if pid in h:
            return h[pid]["status"]["status_str"]
        time.sleep(5)
    return "timeout"


def main() -> int:
    parser = argparse.ArgumentParser(description="SeedVR2 1080p upscale with temporal segmentation")
    parser.add_argument("video", help="filename inside ComfyUI input dir")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--prefix", default="seedvr2_seg")
    parser.add_argument("--out", default="Upscaled_seedVR2_1080p.mp4")
    args = parser.parse_args()

    probe = subprocess.run(
        ["docker", "exec", CONTAINER, "ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", f"/ComfyUI/input/{args.video}"],
        capture_output=True, text=True)
    total = float(json.loads(probe.stdout)["format"]["duration"])
    seg_len = total / args.segments
    print(f"input: {args.video} ({total:.3f}s) -> {args.segments} segments of {seg_len:.3f}s")

    parts = []
    for i in range(args.segments):
        start = round(i * seg_len, 3)
        prefix = f"{args.prefix}{i}"
        found = subprocess.run(
            ["docker", "exec", CONTAINER, "sh", "-c",
             f"ls /ComfyUI/output/video/{prefix}_*.mp4 2>/dev/null | head -1"],
            capture_output=True, text=True).stdout.strip()
        if found:
            print(f"segment {i + 1}/{args.segments}: already done ({found})")
            parts.append(found)
            continue
        pid = api("/prompt", {"prompt": segment_prompt(args.video, start, seg_len, prefix)})["prompt_id"]
        print(f"segment {i + 1}/{args.segments}: start={start}s queued ({pid})")
        status = wait_done(pid, 3600)
        print(f"segment {i + 1}: {status}")
        if status != "success":
            return 1
        found = subprocess.run(
            ["docker", "exec", CONTAINER, "sh", "-c",
             f"ls /ComfyUI/output/video/{prefix}_*.mp4 2>/dev/null | head -1"],
            capture_output=True, text=True).stdout.strip()
        if not found:
            print(f"segment {i + 1}: no output file")
            return 1
        parts.append(found)
        print(f"segment {i + 1}: {found}")

    lst = "\n".join(f"file '{Path(p).name}'" for p in parts)
    subprocess.run(["docker", "exec", CONTAINER, "sh", "-c",
                    f"printf '%s\\n' \"{lst}\" > /ComfyUI/output/video/concat_list.txt"], check=True)
    subprocess.run(["docker", "exec", CONTAINER, "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", "/ComfyUI/output/video/concat_list.txt", "-c", "copy",
                    f"/ComfyUI/output/video/{args.out}"], check=True)
    print(f"done: output/video/{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
