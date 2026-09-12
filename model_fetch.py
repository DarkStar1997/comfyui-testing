"""Shared per-script model fetcher.

Each script declares only the models it needs; missing ones are downloaded into
the models/ bind mount (skipped when already present) before the script runs,
so unused model dirs can be deleted freely and re-fetched on demand.
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
MODELS_DIR = REPO_DIR / "models"
CONTAINER = "comfyui"
IMAGE = "comfyui:2.14.0-cuda132-v0.35.0"


def ensure_models(models):
    """models: iterable of (subdir, repo_id, filename) tuples.
    filename: repo-relative path on HF (may include subdirs, e.g.
    'diffusion_models/x.safetensors') or a plain basename for direct URLs.
    Files land at models/<subdir>/<basename>. Returns {basename: host_path}.
    Downloads only what is missing."""
    out, missing, urls = {}, [], []
    for subdir, repo_id, filename in models:
        path = MODELS_DIR / subdir / Path(filename).name
        out[Path(filename).name] = path
        if path.is_file():
            continue
        if str(repo_id).startswith(("http://", "https://")):
            urls.append((subdir, repo_id, filename))
        else:
            missing.append((subdir, repo_id, filename))

    for subdir, url, filename in urls:
        dest = MODELS_DIR / subdir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"fetching {subdir}/{filename} ...", flush=True)
        r = subprocess.call(["curl", "-fL", "--retry", "3", "-o", str(dest / filename), url])
        if r != 0 or not (dest / filename).is_file():
            sys.exit(f"download failed: {subdir}/{filename}")
        print(f"fetched {subdir}/{filename}", flush=True)

    if not missing:
        return out

    script = (
        "import os, shutil\n"
        "from huggingface_hub import hf_hub_download\n"
        + "\n".join(
            f"p = hf_hub_download({repo!r}, {name!r}, local_dir='/ComfyUI/models/_fetch')\n"
            f"dst = '/ComfyUI/models/{subdir}/{Path(name).name}'\n"
            f"os.makedirs('/ComfyUI/models/{subdir}', exist_ok=True)\n"
            f"shutil.move(p, dst)\n"
            f"os.chown(dst, 1000, 1000)"
            for subdir, repo, name in missing)
    )
    cmds = [
        ["docker", "exec", "-e", "HF_TOKEN", CONTAINER, "python", "-c", script],
        ["docker", "run", "--rm", "-e", "HF_TOKEN",
         "-v", f"{MODELS_DIR}:/ComfyUI/models", "--entrypoint", "python", IMAGE, "-c", script],
    ]
    for cmd in cmds:
        if subprocess.call(cmd) == 0:
            break
    else:
        sys.exit("model download failed")
    subprocess.call(["docker", "exec", CONTAINER, "rm", "-rf", "/ComfyUI/models/_fetch"])

    for subdir, _, filename in missing:
        path = MODELS_DIR / subdir / Path(filename).name
        if not path.is_file():
            sys.exit(f"downloaded file missing: {subdir}/{filename}")
        print(f"fetched {subdir}/{path.name}", flush=True)
    return out
