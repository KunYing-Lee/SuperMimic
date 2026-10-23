"""Deterministic static terrain placement shared by simulation and rendering."""

from __future__ import annotations

import math

import numpy as np


def atlas_layout(definitions):
    """Return per-terrain world translations, grid spacing and grid side length.

    Each local ground plane maps to the one physical world plane at z=0.
    Bounds occupy nonoverlapping grid cells with 20 metres between bounds.
    """
    bounds = np.asarray([item["bounds_xy"] for item in definitions])
    spacing = (bounds[:, 1] - bounds[:, 0]).max(0) + 20.0
    side = math.ceil(math.sqrt(len(definitions)))
    grid = np.asarray(
        [(i % side, i // side) for i in range(len(definitions))], dtype=float
    )
    grid -= (side - 1) / 2
    offsets = np.zeros((len(definitions), 3), dtype=np.float32)
    offsets[:, :2] = grid * spacing - bounds.mean(1)
    offsets[:, 2] = -np.asarray([item["ground_height"] for item in definitions])
    return offsets, spacing, side
