# syntax=docker/dockerfile:1
ARG TORCHAUDIO_WHEEL_IMAGE=comfyui-torchaudio:2.11.0

FROM ${TORCHAUDIO_WHEEL_IMAGE} AS torchaudio-wheels
FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ARG COMFYUI_VERSION=v0.35.0

# ---------- system deps: git, ffmpeg, opencv libs ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------- venv as the exposed python/pip ----------
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# ---------- ComfyUI: pinned clone + core deps ----------
RUN git clone --depth 1 --branch ${COMFYUI_VERSION} https://github.com/Comfy-Org/ComfyUI /ComfyUI \
    && rm -rf /ComfyUI/.git
COPY --from=torchaudio-wheels /dist/ /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install opencv-python==4.14.0.94 /tmp/wheels/torchaudio-*.whl \
    && pip install -r /ComfyUI/requirements.txt \
    && rm -rf /tmp/wheels

# ---------- custom nodes: manager (pip, pinned by ComfyUI) + add more nodes here ----------
WORKDIR /ComfyUI/custom_nodes

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /ComfyUI/manager_requirements.txt

# Crystools resource monitors: dir must be 'comfyui-crystools' (hardcoded asset URL in styles.js);
# seds fix 1.26.0 selectors for frontend >=1.51.10 (menu no longer has .comfyui-menu ancestor)
RUN git clone --depth 1 --branch 1.26.0 \
        https://github.com/crystian/ComfyUI-Crystools comfyui-crystools \
    && rm -rf comfyui-crystools/.git \
    && sed -i 's/\.comfyui-menu #crystools-monitors-root/#crystools-monitors-root/' \
        comfyui-crystools/web/monitor.css comfyui-crystools/web/monitorUI.js
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r comfyui-crystools/requirements.txt

# ---------- runtime ----------
WORKDIR /ComfyUI
ENV PYTHONUNBUFFERED=1 HF_HOME=/ComfyUI/hf-cache
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8188/',timeout=5).status==200 else 1)"
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-manager"]
