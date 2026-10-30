"""Phase 2 gates for the climb simulator on a GPU, written as JSON.

1. model: ClimbSim compiles hbold's motion-tracking model (physics fields equal).
2. capacity: over STEPS steps of noisy random actions from task starts, no contact
   or constraint buffer overflows and the count of non-finite (invalid) steps.
3. batch: the same 64 starts and reference-PD actions stepped alone and inside the full batch
   end within 1 cm of each other (sites) after 32 steps in 99% of them.
4. t0: the limb sites at reset match the reference's to 1 mm.

usage: python tools/climb_gates.py <clip> <hbold data dir> <out.json> [num_envs] [steps]
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
clip, data_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
os.environ["HBOLD_DATA"] = data_dir

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hbold.tasks.motion_tracking.config.g1.env_cfgs import unitree_g1_motion_tracking_env_cfg  # noqa: E402
from mjlab.scene import Scene  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import ClimbSim, climb_physics  # noqa: E402
from supermimic.env import make_sim  # noqa: E402

dev = "cuda:0"
torch.manual_seed(0)
cfg = make_sim(dev)
cfg.physics = climb_physics(clip)
kin, motion, mdp = climb_parts(clip, dev)
env = ClimbSim(cfg, N, motion, mdp, clip_id=clip)
res: dict = {"clip": clip, "num_envs": N, "steps": STEPS}

# 1. model
ref = Scene(unitree_g1_motion_tracking_env_cfg(clip).scene, device="cpu").compile()
mine = env.sim.mj_model
fields = ("geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_priority", "geom_condim", "geom_contype",
          "geom_conaffinity", "geom_solref", "geom_solimp", "body_mass", "body_inertia", "body_pos", "body_quat",
          "jnt_range", "dof_damping", "dof_armature", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
diff = {f: float(np.abs(getattr(mine, f) - getattr(ref, f)).max()) for f in fields
        if getattr(mine, f).shape == getattr(ref, f).shape}
shape = [f for f in fields if getattr(mine, f).shape != getattr(ref, f).shape]
res["model"] = {"max_abs_diff": diff, "shape_mismatch": shape, "pass": not shape and max(diff.values()) == 0.0}

# 4. t0
env.reset(torch.arange(N, device=dev))
env.refresh()
names = ("left_seat", "right_seat", "left_ball", "right_ball")
sim_sites = env.robot.data.site_pos_w[:, [env.robot.site_names.index(n) for n in names]]
t0_mm = 1000 * float((kin.site_poses(*env.state())[0] - sim_sites).abs().max())
ref_mm = 1000 * float((sim_sites - motion.hold_pos[env.f]).abs().max())
res["t0"] = {"fk_vs_sim_mm": t0_mm, "sim_vs_reference_mm": ref_mm, "pass": ref_mm < 1.0}

# 2. capacity
d = env.sim.wp_data
invalid, worst_con, worst_efc = 0, 0, 0
low, high = env.action_bounds()
for _ in range(STEPS):
    a = torch.clamp(torch.randn(N, 29, device=dev), low, high)
    out = env.step(a)
    invalid += int((~out["valid"]).sum())
    worst_con = max(worst_con, int(d.nacon.numpy()[0]))
    worst_efc = max(worst_efc, int(d.nefc.numpy().max()))
cap_con, cap_efc = int(d.naconmax), int(cfg.physics.njmax)
res["capacity"] = {"invalid_steps": invalid, "env_steps": N * STEPS, "contacts_peak": worst_con,
                   "contacts_cap": cap_con, "constraints_peak_per_world": worst_efc, "constraints_cap": cap_efc,
                   "pass": worst_con < cap_con and worst_efc < cap_efc}

# 3. batch
B = 64
small = ClimbSim(cfg, B, motion, mdp, clip_id=clip)
frames = env.sample_frames(B)[0]
ids = torch.arange(B, device=dev)
env.reset(ids, frames)
small.reset(ids, frames)
small.xi.copy_(env.xi[:B])
small._apply_context(ids)
alive = torch.ones(B, dtype=torch.bool, device=dev)  # an ended episode is reset: no longer comparable
for _ in range(32):
    # PD targets on the reference's next joints: the starts stay on the wall longer than under noise
    a = (motion.q[small.f + 1] - small.default_q - small.xi[:, 1:30]) / small.alpha
    big_a = torch.zeros(N, 29, device=dev)
    big_a[:B] = a
    alive &= ~env.step(big_a)["done"][:B] & ~small.step(a)["done"]
gap = (kin.site_poses(*env.state())[0][:B] - kin.site_poses(*small.state())[0]).norm(dim=-1).amax(-1)[alive]
res["batch"] = {"compared": int(alive.sum()), "site_gap_m_p99": float(gap.quantile(0.99)) if len(gap) else None,
                "site_gap_m_max": float(gap.max()) if len(gap) else None,
                "pass": bool(len(gap) and float(gap.quantile(0.99)) < 0.01)}
print(json.dumps(res, indent=1))
with open(out_path, "w") as f:
    json.dump(res, f, indent=1)
