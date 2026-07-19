from __future__ import annotations

import numpy as np
from manimlib import ParametricSurface, get_colormap_from_colors


class Terrain(ParametricSurface):
    reference_size = (40.0, 28.0)
    default_colors = (
        "#245947",
        "#4DAA57",
        "#7CEA9C",
    )
    default_profile = {
        "broad_ridge": 1.25,
        "east_hill": 0.95,
        "valley_depth": 0.52,
        "rolling_ground": 0.25,
        "fine_relief": 0.12,
        "minor_ridges": 0.10,
    }

    def __init__(
        self,
        size=(40.0, 28.0),
        height_scale=1.0,
        profile=None,
        colors=None,
        level_sites=(),
        site_radius=1.15,
        resolution=(84, 58),
        creation_axis=0,
        opacity=1,
        shading=(0.2, 0.45, 0.25),
        **kwargs,
    ):
        self.width, self.depth = self._normalize_size(size)
        self.height_scale = height_scale
        self.profile = {**self.default_profile, **(profile or {})}
        self.colors = tuple(colors or self.default_colors)
        self.level_sites = tuple(level_sites)
        self.site_radius = site_radius

        super().__init__(
            lambda x, y: np.array([x, y, self.height_at(x, y)]),
            u_range=(-self.width / 2, self.width / 2),
            v_range=(-self.depth / 2, self.depth / 2),
            resolution=resolution,
            prefered_creation_axis=creation_axis,
            color=self.colors[len(self.colors) // 2],
            opacity=opacity,
            shading=shading,
            depth_test=True,
            **kwargs,
        )
        self.apply_height_colors()

    @staticmethod
    def _normalize_size(size):
        if np.isscalar(size):
            width = depth = float(size)
        else:
            width, depth = map(float, size)
        if width <= 0 or depth <= 0:
            raise ValueError("Terrain size must be positive")
        return width, depth

    def raw_height_at(self, x, y):
        reference_x = x * self.reference_size[0] / self.width
        reference_y = y * self.reference_size[1] / self.depth
        profile = self.profile

        broad_ridge = profile["broad_ridge"] * np.exp(
            -((reference_x + 3.0) / 6.8) ** 2
            - ((reference_y - 1.8) / 3.2) ** 2
        )
        east_hill = profile["east_hill"] * np.exp(
            -((reference_x - 5.0) / 4.6) ** 2
            - ((reference_y + 2.4) / 3.0) ** 2
        )
        shallow_valley = -profile["valley_depth"] * np.exp(
            -(reference_x / 5.2) ** 2 - ((reference_y + 0.6) / 2.4) ** 2
        )
        rolling_ground = profile["rolling_ground"] * np.sin(
            0.42 * reference_x + 0.55 * reference_y
        )
        fine_relief = profile["fine_relief"] * np.sin(
            0.78 * reference_x - 0.34 * reference_y + 0.8
        )
        minor_ridges = (
            profile["minor_ridges"]
            * np.sin(1.08 * reference_x + 0.22 * reference_y)
            * np.sin(0.24 * reference_x - 0.82 * reference_y)
        )
        return self.height_scale * (
            broad_ridge
            + east_hill
            + shallow_valley
            + rolling_ground
            + fine_relief
            + minor_ridges
        )

    def height_at(self, x, y):
        height = self.raw_height_at(x, y)
        for site_x, site_y in self.level_sites:
            site_height = self.raw_height_at(site_x, site_y)
            distance = np.hypot(x - site_x, y - site_y)
            grade = np.exp(-((distance / self.site_radius) ** 6))
            height += grade * (site_height - height)
        return height

    def apply_height_colors(self):
        heights = self.get_points()[:, 2]
        height_range = max(np.ptp(heights), 1e-9)
        normalized_heights = (heights - heights.min()) / height_range
        self.set_rgba_array(
            get_colormap_from_colors(self.colors)(normalized_heights)
        )
        return self

    def set_profile(self, profile):
        self.profile = {**self.default_profile, **profile}
        self.init_points()
        self.apply_height_colors()
        return self
