"""Coal GJK/EPA distances for exact MuJoCo primitive and convex-hull geometry.

A single signed-distance oracle serves both separated and intersecting shapes.
No collision constraint is discarded or replaced when a query fails.
"""

import coal
import mujoco
import numpy as np


class ConvexDistance:
    def __init__(self, model, extra_geoms=()):
        self.model = model
        self.objects = {}
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True
        self.request.gjk_tolerance = 1e-10
        self.request.epa_tolerance = 1e-10
        self.request.gjk_max_iterations = 1000
        self.request.epa_max_iterations = 1000
        for g in range(model.ngeom):
            if not (
                model.geom_contype[g] or model.geom_conaffinity[g] or g in extra_geoms
            ):
                continue
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                shape = coal.Sphere(size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                shape = coal.Capsule(size[0], 2 * size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                shape = coal.Cylinder(size[0], 2 * size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                shape = coal.Box(*(2 * size))
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                vertices = coal.StdVec_Vec3s()
                for v in model.mesh_vert[start : start + count]:
                    vertices.append(np.array(v, dtype=float))
                shape = coal.Convex.convexHull(vertices, True, "Qt")
            else:
                raise ValueError(f"Unsupported exact collision geometry {kind}")
            self.objects[g] = coal.CollisionObject(shape)

    def update(self, data):
        for g, obj in self.objects.items():
            obj.setTransform(
                coal.Transform3s(data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g])
            )

    def distance(self, a, b):
        result = coal.DistanceResult()
        value = coal.distance(self.objects[a], self.objects[b], self.request, result)
        p1, p2 = (
            np.array(result.getNearestPoint1()),
            np.array(result.getNearestPoint2()),
        )
        normal = np.array(result.normal)
        if (
            not np.isfinite(np.r_[value, p1, p2, normal]).all()
            or abs(np.linalg.norm(normal) - 1) > 1e-6
        ):
            raise RuntimeError("Invalid convex signed-distance witness")
        if np.linalg.norm((p2 - p1) - value * normal) > 1e-6:
            raise RuntimeError("Inconsistent convex signed-distance witness")
        return float(value), p1, p2, normal
