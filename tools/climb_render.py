"""Render a recorded climb episode (tools/climb_policy_eval.py --record) on its
route: the simulated robot solid, the reference motion as a translucent ghost,
the camera following the pelvis.

usage: python tools/climb_render.py <traj.npz> <clip> <hbold data> <out.mp4> [--fps 25] [--width 960 --height 720]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("traj")
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("out")
parser.add_argument("--fps", type=int, default=25, help="video frames per second (the episode runs at 50 Hz)")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--title", default="")
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1  # noqa: E402
from hbold.asset_zoo.routes import kilter  # noqa: E402
from hbold.data import motion_path, route_path  # noqa: E402

z = np.load(args.traj)
root_p, root_quat, q, frames = (z[f"{args.clip}_{k}"] for k in ("root_p", "root_quat", "q", "frame"))
joint_names = [str(n) for n in z["joint_names"]]
ref = np.load(motion_path(args.clip))

spec = kilter.scene_spec(kilter.load(route_path(args.clip)))
ghost = g1.get_spec()
for geom in ghost.geoms:
    geom.rgba = (0.35, 0.65, 1.0, 0.25)
    geom.contype = geom.conaffinity = 0
spec.attach(ghost, prefix="ref_", frame=spec.worldbody.add_frame())
spec.visual.global_.offwidth, spec.visual.global_.offheight = args.width, args.height
light = spec.worldbody.add_light()
light.pos, light.dir, light.diffuse = (-3.0, -1.0, 4.0), (0.6, 0.2, -0.8), (0.8, 0.8, 0.8)
spec.visual.headlight.ambient, spec.visual.headlight.diffuse = (0.55, 0.55, 0.55), (0.75, 0.75, 0.75)
model = spec.compile()
data = mujoco.MjData(model)


def free_qadr(prefix: str) -> int:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.joint(j).name.startswith(prefix):
            return int(model.jnt_qposadr[j])
    raise KeyError(f"no free joint {prefix!r}")


robot_free = next(int(model.jnt_qposadr[j]) for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and not model.joint(j).name.startswith("ref_"))
ghost_free = free_qadr("ref_")
robot_adr = [int(model.joint(n).qposadr[0]) for n in joint_names]
ref_names = [str(n) for n in ref["joint_names"]]
ghost_adr = [int(model.joint("ref_" + n).qposadr[0]) for n in ref_names]

renderer = mujoco.Renderer(model, args.height, args.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 4.0, 60.0, -5.0  # from the side: Kilter boards overhang
stride = max(1, round(50 / args.fps))
ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
                           "-pix_fmt", "yuv420p", "-vcodec", "libx264", args.out], stdin=subprocess.PIPE)
look = root_p[0].copy()
for t in range(0, len(frames), stride):
    f = int(frames[t])
    data.qpos[robot_free:robot_free + 3] = root_p[t]
    data.qpos[robot_free + 3:robot_free + 7] = root_quat[t]
    data.qpos[robot_adr] = q[t]
    data.qpos[ghost_free:ghost_free + 3] = ref["body_pos_w"][f, 0]
    data.qpos[ghost_free + 3:ghost_free + 7] = ref["body_quat_w"][f, 0]
    data.qpos[ghost_adr] = ref["joint_pos"][f]
    mujoco.mj_kinematics(model, data)
    look = 0.9 * look + 0.1 * root_p[t]  # a steadier camera than the pelvis itself
    cam.lookat[:] = look
    renderer.update_scene(data, cam)
    ffmpeg.stdin.write(renderer.render().tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print(f"{args.out}: {len(range(0, len(frames), stride))} frames, reference frames {int(frames[0])}-{int(frames[-1])}",
      file=sys.stderr)
