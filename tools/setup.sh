#!/usr/bin/env bash
# Native mjlab environment; all dependencies live in this project's .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked --python 3.13
uv pip check
uv run python tools/download_data.py
uv run python tools/urdf_to_mjcf.py
