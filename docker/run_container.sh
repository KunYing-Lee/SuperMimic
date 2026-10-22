#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?Usage: docker/run_container.sh GPU_UUID}
docker build -t supermimic-mjlab:locked -f docker/Dockerfile .
docker run --rm -it --gpus "device=$GPU" --shm-size=16g \
  -v "$PWD/data:/workspace/supermimic/data" \
  -v "$PWD/logs:/workspace/supermimic/logs" \
  -e CUDA_VISIBLE_DEVICES=0 supermimic-mjlab:locked
