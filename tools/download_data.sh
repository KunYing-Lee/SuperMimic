#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python tools/download_data.py
uv run python tools/urdf_to_mjcf.py
