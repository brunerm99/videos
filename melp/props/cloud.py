from __future__ import annotations

from pathlib import Path

import numpy as np
from manimlib import RIGHT, get_colormap_from_colors

from .mesh import TriangleMeshMobject


MODEL_PATH = Path(__file__).resolve().parent / "static" / "cloud_mesh.npz"


class Cloud(TriangleMeshMobject):
    default_colors = (
        "#91A6B0",
        "#D8E4E9",
        "#F4F8FA",
    )

    def __init__(
        self,
        width=3.5,
        color=None,
        colors=None,
        opacity=0.95,
        creation_direction=RIGHT,
        **kwargs,
    ):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Run melp/props/export_cloud_mesh.py first."
            )

        direction = np.asarray(creation_direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            raise ValueError("creation_direction must be nonzero")

        with np.load(MODEL_PATH) as mesh:
            points = mesh["points"].copy()
            normals = mesh["normals"].copy()

        if color is not None:
            colors = (color, color)
        cloud_colors = tuple(colors or self.default_colors)
        if len(cloud_colors) < 2:
            raise ValueError("colors must contain at least two colors")

        heights = points[:, 2]
        height_range = max(np.ptp(heights), 1e-9)
        normalized_heights = (heights - heights.min()) / height_range
        rgba = get_colormap_from_colors(cloud_colors)(normalized_heights)
        rgba[:, 3] *= opacity

        self.creation_direction = direction / direction_norm
        super().__init__(points, normals, rgba, **kwargs)

        if width is not None:
            if width <= 0:
                raise ValueError("width must be positive")
            self.set_width(width)

    def pointwise_become_partial(self, cloud, a, b):
        if b >= 1:
            self.deactivate_clip_plane()
            return self

        projections = cloud.get_points() @ self.creation_direction
        lower = projections.min()
        span = projections.max() - lower
        cutoff = lower + b * span
        if b <= 0:
            cutoff = lower - max(1e-6, 1e-5 * span)
        self.set_clip_plane(-self.creation_direction, cutoff)
        return self
