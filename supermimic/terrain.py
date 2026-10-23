"""Shared terrain geometry, differentiable height scans and exact terrain depth.

The manifest describes an infinite horizontal ground plane and axis-aligned,
ground-supported boxes for each motion's terrain. These are the same primitives
instantiated by the simulator. Height scans use a 5 cm bilinear height map, as in
InstinctMJ; depth observations ray-cast the actual primitives, not that map.
There is no robot self-occlusion in this terrain-only camera.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .rotations import matvec3, yaw_mat
from .terrain_layout import atlas_layout


class TerrainBank:
    def __init__(self, manifest: dict, device="cpu", resolution: float = 0.05):
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported terrain manifest schema")
        if resolution <= 0:
            raise ValueError("height-map resolution must be positive")
        definitions = manifest["terrains"]
        if not definitions or [entry["id"] for entry in definitions] != list(
            range(len(definitions))
        ):
            raise ValueError("terrain IDs must be contiguous and ordered from zero")
        self.manifest, self.definitions = manifest, definitions
        self.device, self.resolution = torch.device(device), resolution
        maps, origins, sizes, ground, all_boxes = [], [], [], [], []
        for terrain in definitions:
            floor = float(terrain["ground_height"])
            bounds = np.asarray(terrain["bounds_xy"], dtype=np.float64)
            if (
                bounds.shape != (2, 2)
                or not np.isfinite(bounds).all()
                or not np.isfinite(floor)
            ):
                raise ValueError("terrain bounds and ground heights must be finite")
            if (bounds[1] <= bounds[0]).any():
                raise ValueError("terrain bounds must have positive width and height")
            origin = np.floor(bounds[0] / resolution) * resolution
            nx, ny = np.ceil((bounds[1] - origin) / resolution).astype(int) + 1
            x = origin[0] + np.arange(nx) * resolution
            y = origin[1] + np.arange(ny) * resolution
            heights = np.full((ny, nx), floor, dtype=np.float32)
            boxes = []
            for box in terrain["boxes"]:
                center, half = np.asarray(box["center"]), np.asarray(box["half_size"])
                if (
                    center.shape != (3,)
                    or half.shape != (3,)
                    or not np.isfinite(center).all()
                    or not np.isfinite(half).all()
                    or (half <= 0).any()
                ):
                    raise ValueError(
                        "boxes need finite centers and positive half sizes"
                    )
                lower, upper = center - half, center + half
                if lower[2] > floor + 1e-7 or upper[2] <= floor:
                    raise ValueError(
                        "height-map terrain boxes must be supported by the ground plane"
                    )
                if (lower[:2] <= bounds[0]).any() or (upper[:2] >= bounds[1]).any():
                    raise ValueError("terrain bounds must extend beyond all boxes")
                footprint = (
                    (y[:, None] >= lower[1])
                    & (y[:, None] <= upper[1])
                    & (x[None, :] >= lower[0])
                    & (x[None, :] <= upper[0])
                )
                heights[footprint] = np.maximum(heights[footprint], upper[2])
                boxes.append(np.concatenate((lower, upper)))
            maps.append(heights)
            origins.append(origin)
            sizes.append([nx, ny])
            ground.append(floor)
            all_boxes.append(boxes)
        ny_max, nx_max = max(x.shape[0] for x in maps), max(x.shape[1] for x in maps)
        packed = np.zeros((len(maps), ny_max, nx_max), dtype=np.float32)
        for index, heights in enumerate(maps):
            packed[index].fill(ground[index])
            packed[index, : heights.shape[0], : heights.shape[1]] = heights
        self.heights = torch.as_tensor(packed, device=self.device)
        self.origins = torch.tensor(
            np.asarray(origins), dtype=torch.float32, device=self.device
        )
        self.sizes = torch.tensor(sizes, dtype=torch.long, device=self.device)
        self.ground = torch.tensor(ground, dtype=torch.float32, device=self.device)
        # At least one inactive slot makes the empty-box (flat) case shape-stable.
        max_boxes = max(1, max(map(len, all_boxes)))
        boxes = np.zeros((len(maps), max_boxes, 6), dtype=np.float32)
        active = np.zeros((len(maps), max_boxes), dtype=bool)
        for index, terrain_boxes in enumerate(all_boxes):
            if terrain_boxes:
                boxes[index, : len(terrain_boxes)] = terrain_boxes
                active[index, : len(terrain_boxes)] = True
        self.boxes = torch.as_tensor(boxes, device=self.device)
        self.box_active = torch.as_tensor(active, device=self.device)
        x, y = (
            torch.linspace(-0.8, 0.8, 17, device=self.device),
            torch.linspace(-0.5, 0.5, 11, device=self.device),
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        self.offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
        self.num_rays = self.scan_dim = self.offsets.shape[0]
        # Distinct physical tiles share one compiled static scene. Subtracting
        # these offsets recovers each motion's original coordinates exactly.
        offsets, spacing, side = atlas_layout(definitions)
        self.world_origins = torch.as_tensor(offsets, device=self.device)
        self.tile_spacing = torch.as_tensor(
            spacing, dtype=torch.float32, device=self.device
        )
        self.tile_side = side
        installed_boxes = []
        for local_boxes, offset in zip(all_boxes, offsets, strict=True):
            for box in local_boxes:
                installed_boxes.append(box + np.tile(offset, 2))
        self.world_boxes = torch.as_tensor(
            np.asarray(installed_boxes, dtype=np.float32).reshape(-1, 6)
            if installed_boxes
            else np.zeros((1, 6), dtype=np.float32),
            device=self.device,
        )
        self.world_box_active = torch.full(
            (len(self.world_boxes),),
            bool(installed_boxes),
            device=self.device,
            dtype=torch.bool,
        )

    def add_to_spec(self, spec):
        """Create the exact manifest primitives on spatially separated tiles."""
        import mujoco

        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.1],
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.25, 0.28, 0.32, 1.0],
        )
        offsets = self.world_origins.cpu().tolist()
        for terrain, offset in zip(self.definitions, offsets, strict=True):
            for index, box in enumerate(terrain["boxes"]):
                spec.worldbody.add_geom(
                    name=f"terrain_{terrain['id']}_box_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[
                        v + delta
                        for v, delta in zip(box["center"], offset, strict=True)
                    ],
                    size=box["half_size"],
                    friction=[1.0, 0.005, 0.0001],
                    rgba=[0.38, 0.42, 0.47, 1.0],
                )

    @classmethod
    def from_manifest(cls, path: str | Path, device="cpu", resolution: float = 0.05):
        return cls(json.loads(Path(path).read_text()), device, resolution)

    def ground_height(
        self, xy: torch.Tensor, terrain_ids: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear height of the complete physical atlas in the caller's frame.

        Tiles occupy nonoverlapping grid cells. Each query selects its actual
        physical cell, including when a robot leaves its assigned motion tile.
        Empty cells and space outside a map are the compiled world's z=0 plane.
        Indexed vertex lookups avoid replicating height maps per environment.
        """
        assigned_origins = self.world_origins[terrain_ids]
        world_xy = xy + assigned_origins[:, None, :2]
        cells = (
            (world_xy / self.tile_spacing + (self.tile_side - 1) / 2 + 0.5)
            .floor()
            .long()
        )
        in_grid = ((cells >= 0) & (cells < self.tile_side)).all(-1)
        safe_cells = cells.clamp(0, self.tile_side - 1)
        raw_ids = safe_cells[..., 1] * self.tile_side + safe_cells[..., 0]
        occupied = in_grid & (raw_ids < len(self.definitions))
        ids = raw_ids.clamp(max=len(self.definitions) - 1)
        coordinates = (
            world_xy - self.world_origins[ids, :2] - self.origins[ids]
        ) / self.resolution
        low = coordinates.floor().long()
        fraction = coordinates - low
        dimensions = self.sizes[ids]
        inside = ((coordinates >= 0) & (coordinates <= dimensions - 1)).all(-1)
        ny, nx = self.heights.shape[-2:]
        flat = self.heights.flatten()

        def vertex(dx, dy):
            ix = torch.minimum(
                torch.maximum(low[..., 0] + dx, torch.zeros_like(low[..., 0])),
                dimensions[..., 0] - 1,
            )
            iy = torch.minimum(
                torch.maximum(low[..., 1] + dy, torch.zeros_like(low[..., 1])),
                dimensions[..., 1] - 1,
            )
            return flat[ids * (ny * nx) + iy * nx + ix]

        fx, fy = fraction.unbind(-1)
        local_height = (
            (1 - fx) * (1 - fy) * vertex(0, 0)
            + fx * (1 - fy) * vertex(1, 0)
            + (1 - fx) * fy * vertex(0, 1)
            + fx * fy * vertex(1, 1)
        )
        world_height = torch.where(
            occupied & inside,
            local_height + self.world_origins[ids, 2],
            torch.zeros_like(local_height),
        )
        height = world_height - assigned_origins[:, None, 2]
        return torch.where(torch.isfinite(xy).all(-1), height, torch.nan)

    def scan(
        self, p: torch.Tensor, R: torch.Tensor, terrain_ids: torch.Tensor
    ) -> torch.Tensor:
        """Root-minus-terrain height at 17x11 yaw-aligned locations (metres)."""
        heading = yaw_mat(R)[..., :2, :2]
        xy = p[:, None, :2] + (heading[:, None] * self.offsets[None, :, None]).sum(-1)
        return p[:, 2:3] - self.ground_height(xy, terrain_ids)

    @staticmethod
    def _ray_boxes(origins, directions, lower, upper, active):
        origin = origins[:, None, None, :]
        direction = directions[:, :, None, :]
        parallel = direction == 0
        divisor = torch.where(parallel, torch.ones_like(direction), direction)
        first, last = (
            (lower[:, None] - origin) / divisor,
            (upper[:, None] - origin) / divisor,
        )
        near, far = torch.minimum(first, last), torch.maximum(first, last)
        near = torch.where(parallel, -torch.inf, near).amax(-1)
        far = torch.where(parallel, torch.inf, far).amin(-1)
        parallel_outside = (
            parallel & ((origin < lower[:, None]) | (origin > upper[:, None]))
        ).any(-1)
        hit = (
            ~parallel_outside
            & (far >= torch.maximum(near, torch.zeros_like(near)))
            & active[:, None]
        )
        distance = torch.where(near >= 0, near, far)
        return torch.where(hit, distance, torch.inf).amin(-1)

    def raycast(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        terrain_ids: torch.Tensor,
        max_pairs: int = 1_048_576,
        box_chunk: int = 8,
    ) -> torch.Tensor:
        """Nearest nonnegative ray parameter, or +inf on miss.

        origins Bx3; directions BxPx3 need not be unit length. Standard slab
        intersections include vertical box sides. Parallel rays are handled
        analytically, without perturbing directions. Chunking computes the exact
        same minimum while bounding intermediate ray/box pair counts.
        """
        if max_pairs < 1 or box_chunk < 1:
            raise ValueError("ray-cast block sizes must be positive")
        batch, pixels = directions.shape[:2]
        if batch < 1 or pixels < 1:
            raise ValueError("ray casting requires nonempty batches and rays")
        # The physics model contains every static tile. Rays must see those same
        # boxes even after the robot crosses its assigned tile's boundary.
        origins = origins + self.world_origins[terrain_ids]
        lower, upper = self.world_boxes[None].split(3, -1)
        active = self.world_box_active[None]
        box_chunk = min(box_chunk, lower.shape[1])
        ray_chunk = max(1, min(pixels, max_pairs // (batch * box_chunk)))
        outputs = []
        for start in range(0, pixels, ray_chunk):
            rays = directions[:, start : start + ray_chunk]
            dz = rays[..., 2]
            plane = -origins[:, 2:3] / torch.where(dz == 0, torch.ones_like(dz), dz)
            # MuJoCo planes have only their outward-facing side: rays travelling
            # upward from below the ground do not intersect its back face.
            nearest = torch.where((dz < 0) & (plane >= 0), plane, torch.inf)
            for box_start in range(0, lower.shape[1], box_chunk):
                end = box_start + box_chunk
                nearest = torch.minimum(
                    nearest,
                    self._ray_boxes(
                        origins,
                        rays,
                        lower[:, box_start:end],
                        upper[:, box_start:end],
                        active[:, box_start:end],
                    ),
                )
            outputs.append(nearest)
        return torch.cat(outputs, 1)

    def render_depth(self, p, R, terrain_ids, camera=None):
        """Exact terrain-only pinhole depth, distance to image plane in metres.

        Camera frame is x-forward/y-left/z-up; rays have camera x=1. Thus the
        intersection parameter is axial depth, not Euclidean ray distance.
        """
        from .depth import CameraCfg, camera_pose

        camera = camera or CameraCfg()
        camera_p, camera_R = camera_pose(p, R, camera)
        half_height = math.tan(math.radians(camera.vertical_fov_deg) / 2)
        v = (
            (torch.arange(camera.height, device=p.device, dtype=p.dtype) + 0.5)
            / camera.height
            * 2
            - 1
        ) * half_height
        u = (
            (
                (torch.arange(camera.width, device=p.device, dtype=p.dtype) + 0.5)
                / camera.width
                * 2
                - 1
            )
            * half_height
            * camera.width
            / camera.height
        )
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        rays = torch.stack((torch.ones_like(vv), -uu, -vv), -1).reshape(-1, 3)
        directions = matvec3(camera_R[:, None], rays[None])
        depth = self.raycast(camera_p, directions, terrain_ids)
        return depth.clamp(camera.near, camera.far).reshape(
            -1, 1, camera.height, camera.width
        )
