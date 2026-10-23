"""Analytical geometry checks for the retargeter's independent convex oracle."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("coal")
spec = importlib.util.spec_from_file_location(
    "convex_distance", Path(__file__).parents[1] / "tools/convex_distance.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("x", [0.5, 0.25, 0.01])
def test_signed_sphere_distance_and_translation_gradient(x):
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".2" mass="1"/></body>
      <body mocap="true"><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.mocap_pos[0] = [x, 0.03, 0.04]
    mujoco.mj_forward(m, d)
    oracle = module.ConvexDistance(m)
    oracle.update(d)
    distance, a, b, normal = oracle.distance(0, 1)
    assert distance == pytest.approx(np.linalg.norm(d.mocap_pos[0]) - 0.3, abs=1e-9)
    np.testing.assert_allclose(b - a, distance * normal, atol=1e-9)
    q = d.qpos.copy()
    numeric = []
    for i in range(3):
        d.qpos[:] = q
        d.qpos[i] += 1e-6
        mujoco.mj_forward(m, d)
        oracle.update(d)
        numeric.append((oracle.distance(0, 1)[0] - distance) / 1e-6)
    np.testing.assert_allclose(numeric, -normal, atol=1e-5)


def test_capsule_length_matches_mujoco_half_length():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="capsule" size=".1 .2"/>
      <geom type="sphere" size=".05" pos="0 0 .5"/>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    oracle = module.ConvexDistance(m)
    oracle.update(d)
    assert oracle.distance(0, 1)[0] == pytest.approx(0.15, abs=1e-9)
