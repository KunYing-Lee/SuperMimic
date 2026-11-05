"""Prune a finished SuperMimic run's checkpoints, keeping everything a result can cite.

Kept: the final checkpoint; best.pt; the top checkpoints by the run's from-t0
evals (recomputed here from the eval json, not only read from notes); every
checkpoint named in the run's KEEP_CITED.txt; and any --keep. Deleted:
the other ckpt_*.pt and the replay-carrying last.pt. Refuses a run that a
live process still writes to or whose last checkpoint is under a day old.

usage: python tools/climb_prune.py <run dir> '<eval json glob>' [--data <hbold data>] [--keep ckpt_N.pt ...] [--dry_run]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_best import TOP, ranked, update  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("evals")
parser.add_argument("--data", default=os.environ.get("HBOLD_DATA"))
parser.add_argument("--keep", nargs="*", default=[])
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--min_age_h", type=float, default=24.0, help="refuse if the newest checkpoint is younger")
args = parser.parse_args()
run = os.path.abspath(args.run)

live = subprocess.run(["bash", "-c", f"ps -eo args | grep -F -- '{run} ' | grep -v grep | grep -c train.py"],
                      capture_output=True, text=True).stdout.strip()
if live not in ("", "0"):
    raise SystemExit(f"{run}: a training process still writes here")
cks = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))
if not cks:
    raise SystemExit("no checkpoints")
age_h = (time.time() - max(os.path.getmtime(c) for c in cks)) / 3600
if age_h < args.min_age_h:
    raise SystemExit(f"newest checkpoint is {age_h:.1f} h old (< {args.min_age_h} h): treat the run as live")

update(run, args.evals, args.data)  # best.pt and the top list from the evals as they stand
keep = {os.path.basename(cks[-1])} | set(args.keep)
keep |= {f"ckpt_{it:06d}.pt" for _, it, _, _ in ranked(args.evals, args.data)[:TOP]}
cited = os.path.join(run, "KEEP_CITED.txt")
if os.path.exists(cited):
    keep |= set(re.findall(r"ckpt_\d{6}\.pt", open(cited).read()))
freed = 0
for f in [*cks, os.path.join(run, "last.pt")]:
    if not os.path.exists(f) or os.path.basename(f) in keep:
        continue
    freed += os.path.getsize(f)
    if not args.dry_run:
        os.remove(f)
print(f"{'would prune' if args.dry_run else 'pruned'} {freed / 1e9:.2f} GB from {run}; kept {sorted(keep)} + best.pt")
