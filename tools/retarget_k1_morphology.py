"""Constrained, fixed-root K1 morphology retargeting with explicit provenance.

Solve the joint trajectory on the official K1 model. Position/orientation,
posture and source-relative velocity objectives preserve the demonstrations;
joint limits, exact ground support, and enabled self-collision pairs are
constraints. Failed frames reject the whole output: there is no root shift,
velocity clipping, trajectory smoothing, or silent frame dropping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supermimic.robot import K1  # noqa: E402

SOURCE_JOINT_NAMES = (
    "Head_Yaw",
    "Left_Shoulder_Pitch",
    "Right_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Head_Pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
)


class ConstrainedRetargeter:
    def __init__(
        self,
        model_path,
        position_weight=1.0,
        orientation_weight=0.01,
        posture_weight=1e-4,
        velocity_weight=1e-5,
        clearance=0.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joints = [m.joint(name).id for name in SOURCE_JOINT_NAMES]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.bodies = [m.body("Trunk").id] + [int(m.jnt_bodyid[j]) for j in self.joints]
        self.bounds = [tuple(m.jnt_range[j]) for j in self.joints]
        self.velocity_limits = np.array(
            [K1.motor(name).velocity_limit for name in SOURCE_JOINT_NAMES]
        )
        self.geoms = [
            g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
        ]
        self.pairs = []
        exclusions = set(m.exclude_signature)
        for index, a in enumerate(self.geoms):
            for b in self.geoms[index + 1 :]:
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
                if (
                    wa == wb
                    or m.body_weldid[m.body_parentid[wa]] == wb
                    or m.body_weldid[m.body_parentid[wb]] == wa
                ):
                    continue
                if ((min(ba, bb) << 16) + max(ba, bb)) in exclusions:
                    continue
                if (m.geom_contype[a] & m.geom_conaffinity[b]) or (
                    m.geom_contype[b] & m.geom_conaffinity[a]
                ):
                    self.pairs.append((a, b))
        self.weights = dict(
            position=position_weight,
            orientation=orientation_weight,
            posture=posture_weight,
            velocity=velocity_weight,
        )
        self.clearance = clearance
        # Query proximity contacts with a fixed positive search margin. The
        # contact.dist values remain geometric distances; the margin does not
        # inflate shapes and is never used by the training simulator.
        self.proximity_margin = 0.05
        m.geom_margin[self.geoms] = self.proximity_margin
        self.cached_q = None
        self.constraint_cache = None

    def _forward(self, q):
        if self.cached_q is None or not np.array_equal(q, self.cached_q):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self.cached_q = q.copy()
        return self.data

    def _jac_point(self, point, body):
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, point, body)
        return jp[:, self.dadr], jr[:, self.dadr]

    def set_frame(self, source, frame, previous, fps):
        self.data.qpos[:3] = source["body_pos_w"][frame, 0]
        self.data.qpos[3:7] = source["body_quat_w"][frame, 0]
        self.target_p = source["body_pos_w"][frame]
        self.target_R = Rotation.from_quat(
            source["body_quat_w"][frame][:, [1, 2, 3, 0]]
        ).as_matrix()
        self.source_q = source["joint_pos"][frame].astype(float)
        self.previous = previous
        self.delta_source = (
            (self.source_q - source["joint_pos"][frame - 1])
            if previous is not None
            else None
        )
        self.fps = fps
        self.cached_q = None
        self.constraint_cache = None
        self.step_bounds = (
            self.bounds
            if previous is None
            else [
                (max(lo, old - limit / fps), min(hi, old + limit / fps))
                for (lo, hi), old, limit in zip(
                    self.bounds, previous, self.velocity_limits, strict=True
                )
            ]
        )

    def objective(self, q):
        d = self._forward(q)
        loss, grad = 0.0, np.zeros(22)
        for index, body in enumerate(self.bodies):
            jp, jr = self._jac_point(d.xpos[body], body)
            dp = d.xpos[body] - self.target_p[index]
            R = d.xmat[body].reshape(3, 3)
            dR = R - self.target_R[index]
            loss += 0.5 * self.weights["position"] * np.dot(dp, dp)
            grad += self.weights["position"] * (dp @ jp)
            loss += 0.5 * self.weights["orientation"] * np.sum(dR * dR)
            grad += self.weights["orientation"] * (np.cross(R.T, dR.T).sum(axis=0) @ jr)
        dq = q - self.source_q
        loss += 0.5 * self.weights["posture"] * np.dot(dq, dq)
        grad += self.weights["posture"] * dq
        if self.previous is not None:
            residual = (q - self.previous - self.delta_source) * self.fps
            loss += 0.5 * self.weights["velocity"] * np.dot(residual, residual)
            grad += self.weights["velocity"] * residual * self.fps
        return loss, grad

    def _ground_point(self, geom):
        m, d = self.model, self.data
        R = d.geom_xmat[geom].reshape(3, 3)
        down = R[2]
        size = m.geom_size[geom]
        kind = m.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[geom]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start : start + count]
            local = vertices[np.argmin(vertices @ down)]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = -np.sign(down) * size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            local = -down * size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local = -down * size[0]
            local[2] -= np.sign(down[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            local = np.zeros(3)
            norm = np.linalg.norm(down[:2])
            if norm > 0:
                local[:2] = -size[0] * down[:2] / norm
            local[2] = -size[1] * np.sign(down[2])
        else:
            raise ValueError(
                f"Unsupported collision geom type {kind}; no ground-support approximation allowed"
            )
        return d.geom_xpos[geom] + R @ local

    def constraints(self, q, pairs):
        key = tuple(pairs)
        cached = self.constraint_cache
        if cached is not None and cached[1] == key and np.array_equal(cached[0], q):
            return cached[2], cached[3]
        self._forward(q)
        values, rows = [], []
        for geom in self.geoms:
            point = self._ground_point(geom)
            jac, _ = self._jac_point(point, self.model.geom_bodyid[geom])
            values.append(point[2] - self.clearance)
            rows.append(jac[2])
        if pairs:
            indices = [self.pairs.index(pair) for pair in pairs]
            distances, gradients = self._contact_query(q, with_jacobian=True)
            values.extend(distances[indices] - self.clearance)
            rows.extend(gradients[indices])
        values, rows = np.array(values), np.stack(rows)
        self.constraint_cache = (q.copy(), key, values, rows)
        return values, rows

    def _contact_query(self, q, with_jacobian=False):
        self._forward(q)
        mujoco.mj_collision(self.model, self.data)
        lookup = {}
        for contact in self.data.contact[: self.data.ncon]:
            pair = tuple(sorted(int(g) for g in contact.geom))
            if pair in lookup and lookup[pair][0] <= contact.dist:
                continue
            gradient = np.zeros(22)
            if with_jacobian:
                # The envelope theorem gives the derivative of distance at the
                # supporting points. The contact midpoint gives the same normal
                # projection: n dot (omega cross (n * distance/2)) is zero.
                a, b = contact.geom
                ja, _ = self._jac_point(contact.pos, self.model.geom_bodyid[a])
                jb, _ = self._jac_point(contact.pos, self.model.geom_bodyid[b])
                gradient = contact.frame[:3] @ (jb - ja)
            lookup[pair] = (float(contact.dist), gradient)
        # Missing pairs lie outside the configured positive contact margin;
        # this strictly feasible bound has zero derivative while out of range.
        result = [
            lookup.get(tuple(sorted(pair)), (self.proximity_margin, np.zeros(22)))
            for pair in self.pairs
        ]
        return np.array([r[0] for r in result]), np.stack([r[1] for r in result])

    def all_distances(self, q):
        return self._contact_query(q)[0]

    def solve(self, start, max_iterations, tolerance, optimality_tolerance=1e-10):
        # Constraint generation: add every violated self-collision inequality,
        # re-solve, then certify against the complete pair set before accepting.
        q = start.copy()
        active = set()
        total_iterations = 0
        for _ in range(len(self.pairs) + 1):
            distances = self.all_distances(q)
            active.update(
                i
                for i, distance in enumerate(distances)
                if distance < self.clearance + tolerance
            )
            pairs = [self.pairs[i] for i in sorted(active)]
            result = minimize(
                self.objective,
                q,
                method="SLSQP",
                jac=True,
                bounds=self.step_bounds,
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda x: self.constraints(x, pairs)[0],
                        "jac": lambda x: self.constraints(x, pairs)[1],
                    }
                ],
                options={"ftol": optimality_tolerance, "maxiter": max_iterations},
            )
            q = result.x
            total_iterations += result.nit
            ground = self.constraints(q, [])[0]
            distances = self.all_distances(q)
            violated = {
                i
                for i, distance in enumerate(distances)
                if distance < self.clearance - tolerance
            }
            bounds_ok = all(
                lo - tolerance <= value <= hi + tolerance
                for value, (lo, hi) in zip(q, self.step_bounds)
            )
            certified = ground.min() >= -tolerance and not violated and bounds_ok
            if certified and result.success:
                return q, {
                    "success": True,
                    "iterations": total_iterations,
                    "objective": float(result.fun),
                    "min_ground_m": float(ground.min() + self.clearance),
                    "min_self_distance_m": float(distances.min()),
                    "message": str(result.message),
                }
            if violated - active:
                active.update(violated)
                continue
            constraint_values, constraint_jacobian = self.constraints(q, pairs)
            multipliers = np.asarray(result.multipliers)
            lagrangian = self.objective(q)[1] - constraint_jacobian.T @ multipliers
            for i, (lo, hi) in enumerate(self.step_bounds):
                if (q[i] <= lo + 1e-7 and lagrangian[i] > 0) or (
                    q[i] >= hi - 1e-7 and lagrangian[i] < 0
                ):
                    lagrangian[i] = 0.0
            return q, {
                "success": False,
                "iterations": total_iterations,
                "objective": float(result.fun),
                "min_ground_m": float(ground.min() + self.clearance),
                "min_self_distance_m": float(distances.min()),
                "message": str(result.message),
                "solver_success": bool(result.success),
                "projected_lagrangian_gradient_inf": float(np.abs(lagrangian).max()),
                "complementarity_inf": float(
                    np.abs(multipliers * constraint_values).max()
                ),
                "candidate_q": q.tolist(),
                "bounds_ok": bool(bounds_ok),
            }
        raise RuntimeError("Constraint generation did not terminate")


def resample_source(source, start, stop, target_fps):
    source_fps = float(source["fps"][0])
    count = round((stop - start - 1) * target_fps / source_fps) + 1
    source_time = np.arange(stop - start) / source_fps
    target_time = np.arange(count) / target_fps
    if not np.isclose(target_time[-1], source_time[-1], atol=1e-9):
        raise ValueError(
            "Source interval endpoint must lie on the target sampling grid"
        )
    selected = {"fps": np.array([target_fps], dtype=np.float64)}
    index = np.minimum(
        np.searchsorted(source_time, target_time, side="right") - 1,
        len(source_time) - 2,
    )
    blend = (target_time - source_time[index]) * source_fps
    for key, value in source.items():
        if key == "fps":
            continue
        part = value[start:stop]
        if key == "body_quat_w":
            result = np.empty((count, part.shape[1], 4))
            for body in range(part.shape[1]):
                rotations = Rotation.from_quat(part[:, body, [1, 2, 3, 0]])
                result[:, body] = Slerp(source_time, rotations)(target_time).as_quat()[
                    :, [3, 0, 1, 2]
                ]
        else:
            weight = blend.reshape((-1,) + (1,) * (part.ndim - 1))
            result = part[index] * (1 - weight) + part[index + 1] * weight
        selected[key] = result.astype(np.float32)
    return selected, target_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=Path(K1.mjcf_path))
    parser.add_argument("--start", "--source-start", type=int, default=0)
    parser.add_argument("--stop", "--source-stop", type=int)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--target-fps", type=int, default=100)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-5)
    parser.add_argument("--optimality-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Retarget output/report already exists")
    with np.load(args.source, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive}
    total = len(source["joint_pos"])
    stop = total if args.stop is None else args.stop
    if not 0 <= args.start < stop <= total:
        raise ValueError("Invalid source frame range")
    selected, target_time = resample_source(source, args.start, stop, args.target_fps)
    reference = {key: value.copy() for key, value in selected.items()}
    fps = float(args.target_fps)
    solver = ConstrainedRetargeter(args.mjcf)
    qs, records = [], []
    for frame in range(len(target_time)):
        previous = qs[-1] if qs else None
        solver.set_frame(selected, frame, previous, fps)
        initial = (
            selected["joint_pos"][frame].astype(float)
            if previous is None
            else previous + solver.delta_source
        )
        q, record = solver.solve(
            initial,
            args.max_iterations,
            args.feasibility_tolerance,
            args.optimality_tolerance,
        )
        record["source_frame"] = float(
            target_time[frame] * source["fps"][0] + args.start
        )
        record["target_frame"] = frame
        records.append(record)
        print(json.dumps(record), flush=True)
        if not record["success"]:
            break
        qs.append(q)
    report = {
        "schema_version": 1,
        "retarget_script_sha256": script_sha256,
        "self_collision_jacobian": "analytic_contact_normal_projection",
        "source_file": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "model_file": str(args.mjcf.resolve()),
        "model_sha256": hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
        "source_frame_range": [args.start, stop],
        "source_frame_start": args.start,
        "source_frame_stop": stop,
        "fps": fps,
        "source_fps": float(source["fps"][0]),
        "target_fps": args.target_fps,
        "robot_xml_sha256": hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
        "method": "fixed_root_constrained_nonlinear_ik_with_source_relative_velocity_objective",
        "objective_weights": solver.weights,
        "joint_velocity_limits_rad_s": solver.velocity_limits.tolist(),
        "velocity_constraints": "abs(q_t-q_previous) <= velocity_limit / target_fps",
        "clearance_m": solver.clearance,
        "feasibility_tolerance_m": args.feasibility_tolerance,
        "optimality_tolerance": args.optimality_tolerance,
        "max_iterations_per_solve": args.max_iterations,
        "joint_names": list(SOURCE_JOINT_NAMES),
        "body_names": [solver.model.body(i).name for i in solver.bodies],
        "records": records,
        "success": len(qs) == len(target_time),
        "converged": len(qs) == len(target_time),
        "root_preserved": True,
        "postprocessing": [],
        "collision_distance_oracle": "mujoco_native_proximity_contact_distance",
        "proximity_query_margin_m": solver.proximity_margin,
        "query_ccd_tolerance": solver.model.opt.ccd_tolerance,
        "query_ccd_iterations": solver.model.opt.ccd_iterations,
        "independent_100hz_native_contact_audit_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if report["success"]:
        selected["joint_pos"] = np.array(qs, dtype=np.float32)
        selected["joint_vel"] = np.gradient(
            selected["joint_pos"], 1 / fps, axis=0
        ).astype(np.float32)
        for frame, q in enumerate(qs):
            solver.set_frame(selected, frame, None, fps)
            data = solver._forward(q)
            selected["body_pos_w"][frame] = data.xpos[solver.bodies]
            selected["body_quat_w"][frame] = data.xquat[solver.bodies]
            # Preserve the supplied root representation exactly; its normalized
            # MuJoCo rotation is used only for the kinematic calculation.
            selected["body_pos_w"][frame, 0] = reference["body_pos_w"][frame, 0]
            selected["body_quat_w"][frame, 0] = reference["body_quat_w"][frame, 0]
            data.qvel[:3] = reference["body_lin_vel_w"][frame, 0]
            R = data.xmat[solver.bodies[0]].reshape(3, 3)
            data.qvel[3:6] = R.T @ reference["body_ang_vel_w"][frame, 0]
            data.qvel[solver.dadr] = selected["joint_vel"][frame]
            for index, body in enumerate(solver.bodies):
                jp = np.zeros((3, solver.model.nv))
                jr = np.zeros_like(jp)
                mujoco.mj_jacBody(solver.model, data, jp, jr, body)
                selected["body_lin_vel_w"][frame, index] = jp @ data.qvel
                selected["body_ang_vel_w"][frame, index] = jr @ data.qvel
        report["max_body_position_error_m"] = float(
            np.linalg.norm(
                selected["body_pos_w"] - reference["body_pos_w"],
                axis=-1,
            ).max()
        )
        reference_rotations = Rotation.from_quat(
            reference["body_quat_w"].reshape(-1, 4)[:, [1, 2, 3, 0]]
        )
        output_rotations = Rotation.from_quat(
            selected["body_quat_w"].reshape(-1, 4)[:, [1, 2, 3, 0]]
        )
        report["max_body_orientation_error_rad"] = float(
            (reference_rotations.inv() * output_rotations).magnitude().max()
        )
        report["max_joint_delta_rad"] = float(
            np.abs(selected["joint_pos"] - reference["joint_pos"]).max()
        )
        report["max_joint_speed_rad_s"] = float(np.abs(selected["joint_vel"]).max())
        report["max_joint_velocity_limit_ratio"] = float(
            (
                np.abs(np.diff(selected["joint_pos"], axis=0))
                * fps
                / solver.velocity_limits
            ).max()
        )
        np.savez_compressed(args.output, **selected)
        report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["success"]:
        raise SystemExit(
            "Retarget failed certification; report saved, no motion emitted"
        )


if __name__ == "__main__":
    main()
