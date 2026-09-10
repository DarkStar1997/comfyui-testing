#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
WHEEL_IMAGE = "comfyui-wheels:2.14.0-cu132"
IMAGE = "comfyui:2.14.0-cuda132-v0.35.0"
DEFAULT_CUDA_ARCH = "8.9"


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def build(no_cache: bool = False, cuda_arch: str = DEFAULT_CUDA_ARCH) -> int:
    extra = ["--no-cache"] if no_cache else []

    result = run([
        "docker", "build", *extra,
        "-f", "Dockerfile.torchaudio",
        "--build-arg", f"TORCH_CUDA_ARCH_LIST={cuda_arch}",
        "-t", WHEEL_IMAGE,
        str(REPO_DIR),
    ])
    if result.returncode != 0:
        return result.returncode

    return run(["docker", "build", *extra, "-t", IMAGE, str(REPO_DIR)]).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the ComfyUI image (torchaudio wheel stage, then runtime stage)")
    parser.add_argument("--no-cache", action="store_true", help="Do not use the docker build cache")
    parser.add_argument("--cuda-arch", default=DEFAULT_CUDA_ARCH,
                        help=f"TORCH_CUDA_ARCH_LIST for the torchaudio build "
                             f"(default: {DEFAULT_CUDA_ARCH} = RTX 4090)")
    args = parser.parse_args()
    return build(no_cache=args.no_cache, cuda_arch=args.cuda_arch)


if __name__ == "__main__":
    sys.exit(main())
