# Optional native mjlab runtime. The host NVIDIA runtime supplies the driver.
FROM ghcr.io/astral-sh/uv:0.13.0-python3.13-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libegl1 libgl1 libglib2.0-0 tmux \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/supermimic
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY . .
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV PATH="/workspace/supermimic/.venv/bin:${PATH}" MUJOCO_GL=egl
CMD ["bash"]
