"""Constrained OmniRetarget objective solved by SLSQP on an SO(3) chart.

All collision rows have fixed identities within a frame. There is no constraint
dropping, resolver switching, root correction or smoothing after optimization.
"""

import time
import mujoco
import numpy as np
from scipy.optimize import minimize


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def right_jacobian(v, inverse=False):
    theta = np.linalg.norm(v)
    K = skew(v)
    if inverse:
        c = (
            1 / 12 + theta**2 / 720
            if theta < 1e-4
            else (1 - 0.5 * theta / np.tan(theta / 2)) / theta**2
        )
        return np.eye(3) + 0.5 * K + c * (K @ K)
    a = 0.5 - theta**2 / 24 if theta < 1e-4 else (1 - np.cos(theta)) / theta**2
    b = 1 / 6 - theta**2 / 120 if theta < 1e-4 else (theta - np.sin(theta)) / theta**3
    return np.eye(3) - a * K + b * (K @ K)


def collision_pairs(model):
    geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    excluded = set(map(int, model.exclude_signature))
    pairs = []
    for k, a in enumerate(geoms):
        for b in geoms[k + 1 :]:
            ba, bb = map(int, model.geom_bodyid[[a, b]])
            wa, wb = map(int, model.body_weldid[[ba, bb]])
            if wa == wb:
                continue
            pa = int(model.body_weldid[model.body_parentid[wa]])
            pb = int(model.body_weldid[model.body_parentid[wb]])
            if wa == pb or wb == pa:
                continue
            if (min(ba, bb) << 16) | max(ba, bb) in excluded:
                continue
            if not (
                (model.geom_contype[a] & model.geom_conaffinity[b])
                or (model.geom_contype[b] & model.geom_conaffinity[a])
            ):
                continue
            pairs.append((a, b))
    return pairs


def solve(
    engine,
    q_init,
    target_laplacian,
    L,
    Kron,
    terrain_points,
    max_iter=200,
    q_last=None,
    target_base_orientation=None,
    object_points=None,
    bone_targets=None,
    root_translation=None,
):
    if bone_targets is not None or root_translation is not None:
        raise ValueError(
            "Unsupported extra objective for this contact-constrained adapter"
        )
    from instinct_pairs import world_samples
    from convex_distance import ConvexDistance

    m, d = engine.robot_model, engine.robot_data
    started = time.perf_counter()
    indices = engine.dof_indices
    if not np.array_equal(indices, np.arange(m.nv)):
        raise ValueError("Full floating-base optimization required")
    samples = engine._terrain_collision_geoms
    bodies = np.concatenate([np.full(len(p), m.geom_bodyid[g]) for g, p in samples])
    groups = {b: np.flatnonzero(bodies == b) for b in np.unique(bodies)}
    if not hasattr(engine, "_exact_self_oracle"):
        engine._exact_self_pairs = collision_pairs(m)
        engine._exact_self_oracle = ConvexDistance(m)
    pairs = engine._exact_self_pairs
    oracle = engine._exact_self_oracle
    env = (
        terrain_points
        if object_points is None
        else np.vstack([terrain_points, object_points])
    )
    bounds = [(None, None)] * m.nv
    for j in range(1, m.njnt):
        adr = int(m.jnt_qposadr[j])
        dof = int(m.jnt_dofadr[j])
        lo, hi = map(float, m.jnt_range[j])
        if q_last is not None:
            from supermimic.robot import K1

            radius = K1.motor(m.joint(j).name).velocity_limit * engine.dt
            lo = max(lo, q_last[adr] - radius)
            hi = min(hi, q_last[adr] + radius)
        bounds[dof] = (lo - q_init[adr], hi - q_init[adr])
    cached_x = None
    cached = None

    def residual(reference, q):
        r = engine._configuration_residual(reference, q)
        J = np.eye(m.nv)
        J[3:6, 3:6] = right_jacobian(r[3:6], inverse=True)
        return r, J

    def evaluate(x):
        nonlocal cached_x, cached
        if cached_x is not None and np.array_equal(cached_x, x):
            return cached
        q = engine._integrate_optimized_step(q_init, x)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        oracle.update(d)
        transform = np.eye(m.nv)
        transform[3:6, 3:6] = right_jacobian(x[3:6])
        Jp, positions, _ = engine._compute_robot_jacobians(q)
        points = np.vstack([[positions[n] for n in engine.source_target_names], env])
        err = (L @ points - target_laplacian).reshape(-1)
        jac = Kron @ np.vstack([Jp, np.zeros((3 * len(env), m.nv))])
        value = 10 * float(err @ err)
        gradient = 20 * jac.T @ err
        r, j = residual(engine.neutral_qpos, q)
        value += float(engine.joint_regularization_diag @ r**2)
        gradient += 2 * j.T @ (engine.joint_regularization_diag * r)
        if q_last is not None:
            r, j = residual(q_last, q)
            value += engine.smooth_weight * float(r @ r)
            gradient += 2 * engine.smooth_weight * j.T @ r
        if target_base_orientation is not None:
            reference = q.copy()
            reference[3:7] = target_base_orientation
            r, j = residual(reference, q)
            value += engine.base_rotation_weight * float(r[3:6] @ r[3:6])
            gradient += 2 * engine.base_rotation_weight * j[3:6].T @ r[3:6]
        diff = q[:3] - q_init[:3]
        value += float(engine.base_translation_weights @ diff**2)
        gradient[:3] += 2 * engine.base_translation_weights * diff
        values = []
        rows = []
        surface_points = world_samples(d, samples)
        closest, _, triangles = engine._closest_points_on_terrain(surface_points)
        normals = engine._terrain_face_normals[triangles]
        signed = np.sum((surface_points - closest) * normals, axis=1)
        terrain_jac = np.empty((len(surface_points), m.nv))
        for b, ids in groups.items():
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, d.xpos[b], int(b))
            terrain_jac[ids] = (
                normals[ids] @ jp
                + np.cross(surface_points[ids] - d.xpos[b], normals[ids]) @ jr
            )
        values.extend(signed + engine.penetration_tolerance)
        rows.extend(terrain_jac)
        for a, b in pairs:
            distance, p1, p2, normal = oracle.distance(a, b)
            Ja = engine._calc_contact_jacobian_from_point(
                m.geom_bodyid[a], p1, input_world=True
            )
            Jb = engine._calc_contact_jacobian_from_point(
                m.geom_bodyid[b], p2, input_world=True
            )
            values.append(distance + engine.penetration_tolerance)
            rows.append(normal @ (Jb - Ja))
        contact_rows, _ = engine.contact_distances()
        for row, rhs in contact_rows:
            values.append(-rhs)
            rows.append(row)
        cached_x = x.copy()
        cached = (
            value,
            gradient @ transform,
            np.asarray(values),
            np.asarray(rows) @ transform,
            q,
        )
        return cached

    history = []

    def callback(x):
        value, _, c, _, _ = evaluate(x)
        history.append(
            {
                "iteration": len(history) + 1,
                "cost": float(value),
                "max_violation": float(max(0, -c.min())),
            }
        )

    # Exact constraint generation: each reduced problem is followed by an
    # evaluation of EVERY inequality. Any violated surface constraint is added
    # permanently. Acceptance requires a converged solve and full feasibility;
    # omitted strictly feasible rows have zero KKT multipliers. This avoids
    # thousands of redundant coplanar rows in SLSQP without relaxing a bound.
    x = np.zeros(m.nv)
    initial = evaluate(x)[2]
    n_surface = len(bodies)
    active = set(range(n_surface, len(initial)))  # all self and contact constraints
    active.update(getattr(engine, "_retained_terrain_rows", set()))
    for ids in groups.values():
        active.add(int(ids[np.argmin(initial[ids])]))
    generation = 0
    for generation in range(32):
        selected = np.array(sorted(active), dtype=int)
        result = minimize(
            lambda x: evaluate(x)[0],
            x,
            jac=lambda x: evaluate(x)[1],
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": lambda x: evaluate(x)[2][selected],
                "jac": lambda x: evaluate(x)[3][selected],
            },
            method="SLSQP",
            callback=callback,
            options={"maxiter": max_iter, "ftol": 1e-6},
        )
        x = result.x
        full = evaluate(x)[2]
        additions = []
        for ids in groups.values():
            k = int(ids[np.argmin(full[ids])])
            if full[k] < -1e-7 and k not in active:
                additions.append(k)
        if not additions:
            break
        active.update(additions)
    value, gradient, c, _, q = evaluate(result.x)
    violation = float(max(0.0, -c.min()))
    engine._retained_terrain_rows = {i for i in active if i < n_surface}
    engine.last_solve_diagnostics = {
        "success": bool(result.success and violation <= 1e-6),
        "converged": bool(result.success),
        "failure_reason": ("full_constraint_violation" if violation > 1e-6 else None)
        if result.success
        else result.message,
        "iterations": len(history),
        "cost": float(value),
        "max_hard_violation": violation,
        "constraint_generations": generation + 1,
        "active_inequalities": len(active),
        "full_inequalities": len(c),
        "optimizer": "scipy SLSQP on SO(3) chart",
        "optimality_criterion": "SLSQP ftol=1e-6; full inequality violation <=1e-6",
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
    }
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return q
