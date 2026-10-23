"""Explicit terrain-camera geometry and shared real/imagined depth processing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .rotations import mat_to_6d, matmul3, matvec3, so3_exp


@dataclass(frozen=True)
class CameraCfg:
    """Root-mounted pinhole camera, x-forward/y-left/z-up.

    This is a terrain-only simulated camera. Camera offsets are fixed unless a
    caller explicitly supplies randomized mount context to camera_pose.
    """

    height: int = 18
    width: int = 32
    vertical_fov_deg: float = 58.0
    near: float = 0.1
    far: float = 5.0
    position: tuple[float, float, float] = (0.08, 0.0, 0.25)
    rotation_vector: tuple[float, float, float] = (0.0, math.radians(20.0), 0.0)

    def __post_init__(self):
        if self.height < 1 or self.width < 1 or not 0 < self.vertical_fov_deg < 180:
            raise ValueError("camera image dimensions and field of view must be valid")
        if not 0 < self.near < self.far:
            raise ValueError("camera clipping planes require 0 < near < far")


def camera_pose(p, R, cfg: CameraCfg, context=None):
    """World mount pose; optional context is translation + rotation-vector offset."""
    offset = p.new_tensor(cfg.position).expand_as(p)
    rotation = p.new_tensor(cfg.rotation_vector).expand_as(p)
    if context is not None:
        offset = offset + context[..., :3]
        mount_R = matmul3(so3_exp(rotation), so3_exp(context[..., 3:]))
    else:
        mount_R = so3_exp(rotation)
    return p + matvec3(R, offset), matmul3(R, mount_R)


def relative_pose(anchor_state, state):
    """9D body-frame translation and relative SO(3) first-two-column features."""
    anchor_p, anchor_R = anchor_state[:2]
    p, R = state[:2]
    inverse = anchor_R.transpose(-1, -2)
    return torch.cat(
        (matvec3(inverse, p - anchor_p), mat_to_6d(matmul3(inverse, R))), -1
    )


class DepthPipeline:
    """Normalize clean axial depth and apply the same sensor noise in both paths.

    Range-dependent Gaussian error is specified in metres. No resizing,
    inpainting, image repairs or temporal filtering alter the target images.
    """

    def __init__(
        self,
        camera: CameraCfg | None = None,
        noise_std: float = 0.01,
        range_noise_scale: float = 0.001,
    ):
        if noise_std < 0 or range_noise_scale < 0:
            raise ValueError("depth noise standard deviations must be nonnegative")
        self.camera = camera or CameraCfg()
        self.noise_std, self.range_noise_scale = noise_std, range_noise_scale

    def clean_from_sensor(self, meters):
        cfg = self.camera
        return (meters.clamp(cfg.near, cfg.far) - cfg.near) / (cfg.far - cfg.near)

    clean = clean_from_sensor

    def meters_from_clean(self, clean):
        cfg = self.camera
        return clean * (cfg.far - cfg.near) + cfg.near

    def noise_predicted(self, clean):
        if self.noise_std == 0 and self.range_noise_scale == 0:
            return clean
        meters = self.meters_from_clean(clean)
        noise = torch.randn_like(meters) * (
            self.noise_std + self.range_noise_scale * meters.square()
        )
        # A ray miss is a saturated measurement, not a noisy finite surface.
        noisy = torch.where(clean >= 1.0, meters, meters + noise)
        return self.clean_from_sensor(noisy)

    noise = noise_predicted
