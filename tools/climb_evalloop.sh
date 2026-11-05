#!/bin/bash
# climb_evalloop.sh <run dir> <clip[,clip...]> <hbold data> <eval dir> [every 2000]
# Evaluates each ckpt_N.pt (N a multiple of `every`) as it appears, by the shared protocol
# (tools/climb_policy_eval.py's defaults), then refreshes the run's best.pt and its top-3
# list (tools/climb_best.py), until the run's last checkpoint is evaluated.
# The caller sets the environment: PY (python), PYTHONPATH (hbold), CUDA_VISIBLE_DEVICES.
set -u
run=$1; clips=$2; data=$3; out=$4; every=${5:-2000}
tools=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}
mkdir -p "$out"; name=$(basename "$run")
last=$($PY -c "import json; print(json.load(open('$run/args.json'))['iterations'])")
while true; do
  for ck in $(ls "$run"/ckpt_*.pt 2>/dev/null | sort); do
    n=$(basename "$ck" .pt); n=$((10#${n#ckpt_})); [ $((n % every)) = 0 ] || [ $n = $last ] || continue
    j=$out/${name}_$(basename "$ck" .pt).json; [ -f "$j" ] && continue
    $PY "$tools/climb_policy_eval.py" "$clips" "$data" "$ck" "$j.tmp" > "$j.log" 2>&1 && mv "$j.tmp" "$j"
    $PY "$tools/climb_best.py" "$run" "$out/${name}_ckpt_*.json" --data "$data" >> "$out/best_$name.log" 2>&1
  done
  [ -f "$out/${name}_ckpt_$(printf %06d $last).json" ] && break
  sleep 300
done
echo "all done: best $(cat "$run/best.json" 2>/dev/null | head -c 200)"
