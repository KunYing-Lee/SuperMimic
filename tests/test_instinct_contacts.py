"""Contact constraint derivatives must work on both sides of a surface."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "instinct_pairs", Path(__file__).parents[1] / "tools/instinct_pairs.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize("point", [[0.4, 0.1, 0.2], [0.08, 0.03, 0.02]])
def test_signed_surface_distance_and_tangent_gradient(point):
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body mocap="true"><geom name="probe" type="sphere" size="0.00001" contype="0" conaffinity="0"/></body>
      <body name="body"><freejoint/><geom name="surface" type="sphere" size=".2" mass="1"/></body>
      </worldbody></mujoco>""")
    probe, geom = m.geom("probe").id, m.geom("surface").id
    d = mujoco.MjData(m)
    d.mocap_pos[0] = point
    mujoco.mj_forward(m, d)
    distance, analytic = helpers.anchor_surface_distance(m, d, [geom], probe)
    assert distance == pytest.approx(np.linalg.norm(point) - 0.2, abs=1e-8)
    q = d.qpos.copy()
    numeric = []
    for i in range(m.nv):
        v = np.zeros(m.nv)
        v[i] = 1
        d.qpos[:] = q
        mujoco.mj_integratePos(m, d.qpos, v, 1e-6)
        mujoco.mj_forward(m, d)
        perturbed, _ = helpers.anchor_surface_distance(m, d, [geom], probe)
        numeric.append((perturbed - distance) / 1e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)


def test_source_loader_rejects_joint_name_dimension_only_match(tmp_path):
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
      <body><joint name="actual"/><geom type="sphere" size=".1" mass="1"/></body>
      </body></worldbody></mujoco>""")
    path = tmp_path / "wrong.npz"
    np.savez(
        path,
        joint_names=["wrong"],
        joint_pos=[[0.0]],
        base_pos_w=[[0, 0, 1]],
        base_quat_w=[[1, 0, 0, 0]],
        framerate=90,
    )
    with pytest.raises(ValueError, match="joint names differ"):
        helpers.load_qpos(path, m)


def test_so3_chart_jacobian_matches_native_quaternion_difference():
    path = Path(__file__).parents[1] / "tools/strict_omni_sqp.py"
    spec = importlib.util.spec_from_file_location("strict_omni_sqp", path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>""")
    x = np.array([0.3, -0.4, 0.7])
    base = m.qpos0.copy()
    mujoco.mj_integratePos(m, base, np.r_[np.zeros(3), x], 1.0)
    numeric = np.empty((3, 3))
    for i in range(3):
        step = x.copy()
        step[i] += 1e-7
        q = m.qpos0.copy()
        mujoco.mj_integratePos(m, q, np.r_[np.zeros(3), step], 1.0)
        v = np.zeros(6)
        mujoco.mj_differentiatePos(m, v, 1e-7, base, q)
        numeric[:, i] = v[3:]
    np.testing.assert_allclose(solver.right_jacobian(x), numeric, atol=1e-7)


def test_source_resampling_preserves_constant_rate_and_rotation():
    from scipy.spatial.transform import Rotation

    t = np.arange(91) / 90
    q = np.zeros((91, 8))
    q[:, 0] = t
    q[:, 7] = 2 * t
    q[:, 3:7] = Rotation.from_rotvec(t[:, None] * np.array([0.0, 0.0, 1.0])).as_quat()[
        :, [3, 0, 1, 2]
    ]
    out = helpers.resample_source(q, 90, 100)
    assert len(out) == 101
    target = np.arange(101) / 100
    np.testing.assert_allclose(out[:, 0], target, atol=1e-12)
    np.testing.assert_allclose(out[:, 7], 2 * target, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(out[:, [4, 5, 6, 3]]).as_rotvec()[:, 2], target, atol=1e-12
    )
