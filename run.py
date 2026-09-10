#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
IMAGE = "comfyui:2.14.0-cuda132-v0.35.0"
CONTAINER_NAME = "comfyui"
PORT = "8188"
MOUNT_DIRS = ["models", "output", "input", "user", "hf-cache"]


def load_env_file(path: Path) -> dict:
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the ComfyUI docker container")
    parser.add_argument("--build", action="store_true", help="Build the image first")
    args = parser.parse_args()

    if args.build:
        import build
        rc = build.build()
        if rc != 0:
            return rc

    for d in MOUNT_DIRS:
        (REPO_DIR / d).mkdir(exist_ok=True)

    env = {**load_env_file(REPO_DIR / ".env"), **os.environ}

    hf_env = ["-e", "HF_HOME=/ComfyUI/hf-cache"]
    token = env.get("HF_TOKEN")
    if token:
        hf_env += ["-e", f"HF_TOKEN={token}"]

    run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--shm-size", "4g",
        "--restart", "unless-stopped",
        "-p", f"{PORT}:8188",
        "-v", f"{REPO_DIR / 'models'}:/ComfyUI/models",
        "-v", f"{REPO_DIR / 'output'}:/ComfyUI/output",
        "-v", f"{REPO_DIR / 'input'}:/ComfyUI/input",
        "-v", f"{REPO_DIR / 'user'}:/ComfyUI/user",
        "-v", f"{REPO_DIR / 'hf-cache'}:/ComfyUI/hf-cache",
        *hf_env,
        IMAGE,
    ]
    result = run(cmd)
    if result.returncode != 0:
        return result.returncode

    print(f"ComfyUI: http://localhost:{PORT}  (logs: docker logs -f {CONTAINER_NAME})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
