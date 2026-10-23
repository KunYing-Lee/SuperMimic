#!/usr/bin/env bash
# Foreground native mjlab training. Use tmux for persistent remote execution.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:?Usage: CUDA_VISIBLE_DEVICES=GPU-uuid ./run_train.sh RUN [arguments...]}
shift
: "${CUDA_VISIBLE_DEVICES:?Select exactly one available GPU by UUID before launching}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo 'Training requires one explicitly selected GPU.' >&2
  exit 2
fi
RUN_ROOT=${RUN_ROOT:-"$PWD/logs"}
mkdir -p "$RUN_ROOT/$RUN"
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="$PWD/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
exec .venv/bin/python supermimic/train.py --device cuda:0 --log_dir "$RUN_ROOT/$RUN" "$@"
