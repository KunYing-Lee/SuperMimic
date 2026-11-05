"""Keep a run's best checkpoint by its from-t0 evals (tools/climb_policy_eval.py).

Scans the run's eval json files, ranks the checkpoints and, when the best one
changes, copies it to <run dir>/best.pt (a copy, so pruning the original never
removes it) with best.json beside it: the checkpoint, its metrics and its eval
file. best.pt is listed in the run's KEEP_CITED.txt.

Ranking, from t0:
- one clip: complete share, then the median frame reached;
- several: the mean over clips of median frame reached / clip length, then
  the number of completed rollouts over all clips.

usage: python tools/climb_best.py <run dir> '<eval json glob>' [--data <hbold data>]
  e.g. tools/climb_best.py runs/mv2_seed1 'eval_v2/mv2_seed1_ckpt_*.json' --data /root/leeky/projects/hbold-multi/data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("evals", help="glob of the run's eval json files (named ..._ckpt_<iteration>.json)")
parser.add_argument("--data", default=os.environ.get("HBOLD_DATA"), help="hbold data dir: clip lengths for several clips")
args = parser.parse_args()


def median_reached(t: dict) -> float:
    each = t.get("reached_each")
    return float(np.median(each)) if each else float(t["frames_gained_median"])


lengths: dict[str, int] = {}


def clip_length(clip: str) -> int:
    if clip not in lengths:
        if not args.data:
            raise SystemExit("several clips need --data for their lengths")
        with np.load(os.path.join(args.data, "kilter", clip, "motion.npz")) as z:
            lengths[clip] = int(z["joint_pos"].shape[0])
    return lengths[clip]


def score(d: dict) -> tuple[tuple[float, float], dict]:
    clips = d.get("clips") or {"clip": {"from_t0": d["from_t0"]}}
    if len(clips) == 1:
        t = next(iter(clips.values()))["from_t0"]
        m = {"complete_share": t["complete_share"], "median_reached": median_reached(t)}
        return (m["complete_share"], m["median_reached"]), m
    per = {c: {"median_reached": median_reached(v["from_t0"]), "complete_share": v["from_t0"]["complete_share"],
               "length": clip_length(c)} for c, v in clips.items()}
    frac = float(np.mean([p["median_reached"] / p["length"] for p in per.values()]))
    done = float(sum(p["complete_share"] * d.get("envs", 1) for p in per.values()))
    return (frac, done), {"mean_median_fraction": frac, "completed_rollouts": done, "clips": per}


candidates = []
for f in glob.glob(args.evals):
    hit = re.search(r"ckpt_(\d+)\.json$", f)
    if not hit:
        continue
    it = int(hit.group(1))
    try:
        key, metrics = score(json.load(open(f)))
    except (OSError, ValueError, KeyError) as err:
        print(f"skip {f}: {err}")
        continue
    candidates.append((key, it, f, metrics))
if not candidates:
    raise SystemExit("no evals yet")
candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)  # ties: the later checkpoint

best_json = os.path.join(args.run, "best.json")
current = json.load(open(best_json)) if os.path.exists(best_json) else None
for key, it, f, metrics in candidates:
    ck = os.path.join(args.run, f"ckpt_{it:06d}.pt")
    if current and current["iteration"] == it:
        print(f"best unchanged: {it}")
        break
    if not os.path.exists(ck):
        print(f"best by evals is {it} but {ck} is gone; trying the next")
        continue
    tmp = os.path.join(args.run, "best.pt.tmp")
    shutil.copyfile(ck, tmp)
    os.replace(tmp, os.path.join(args.run, "best.pt"))
    record = {"iteration": it, "checkpoint": ck, "eval": f, "key": list(key), "metrics": metrics}
    with open(best_json + ".tmp", "w") as stream:
        json.dump(record, stream, indent=1)
    os.replace(best_json + ".tmp", best_json)
    print(f"best is now {it}: {key}")
    break
keep = os.path.join(args.run, "KEEP_CITED.txt")
lines = open(keep).read().splitlines() if os.path.exists(keep) else ["Cited checkpoints: never prune."]
if not any(line.startswith("best.pt") for line in lines):
    lines.append("best.pt  the run's best checkpoint by its from-t0 evals (tools/climb_best.py; see best.json)")
    with open(keep, "w") as stream:
        stream.write("\n".join(lines) + "\n")
