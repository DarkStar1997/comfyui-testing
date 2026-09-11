import sys

from huggingface_hub import hf_hub_download

FILES = [
    ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors"),
    ("Comfy-Org/MiniMax-H3", "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_int8_convrot.safetensors"),
    ("Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors"),
]

MODEL_DIR = "/ComfyUI/models"

failed = []
for repo_id, filename in FILES:
    print(f"==> {repo_id}/{filename}", flush=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODEL_DIR)
        print(f"    done: {path}", flush=True)
    except Exception as e:
        failed.append((repo_id, filename))
        print(f"    FAILED: {e}", flush=True)

if failed:
    print(f"\n{len(failed)} file(s) failed:", flush=True)
    for repo_id, filename in failed:
        print(f"  {repo_id}/{filename}", flush=True)
    sys.exit(1)
print("\nAll model files present.", flush=True)
