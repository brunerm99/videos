from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
from manimlib import RIGHT, get_colormap_from_colors

from ..animation_helpers import VT
from .beam import Beam
from .mesh import TriangleMeshMobject


MODEL_PATH = Path(__file__).resolve().parent / "static" / "cloud_mesh.npz"
SHADER_PATH = Path(__file__).resolve().parent / "shaders" / "cloud"
_SHADER_TOKENS = count(1)


def _update_beam_highlight(cloud):
    beam = cloud._highlight_beam
    if beam is None:
        return cloud.set_uniforms({"beam_highlight_strength": 0.0})

    beam_range = beam.get_range()
    cloud.set_uniforms(
        {
            "beam_origin": beam.get_origin(),
            "beam_direction": beam.get_direction(),
            "beam_range": beam_range,
            "beam_slope": beam.get_base_radius() / beam_range,
            "beam_highlight_strength": float(~cloud.beam_highlight),
        }
    )
    return cloud


class Cloud(TriangleMeshMobject):
    shader_folder = str(SHADER_PATH)
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
        self.beam_highlight = VT(0)
        self._highlight_beam = None
        self._shader_token = next(_SHADER_TOKENS)
        super().__init__(points, normals, rgba, **kwargs)

        if width is not None:
            if width <= 0:
                raise ValueError("width must be positive")
            self.set_width(width)

    def copy(self, deep=False):
        result = super().copy(deep=deep)
        result.beam_highlight = VT(float(~self.beam_highlight))
        result._shader_token = next(_SHADER_TOKENS)
        result.uniforms["cloud_shader_token"] = result._shader_token
        return result

    def init_uniforms(self):
        super().init_uniforms()
        self.uniforms.update(
            {
                "beam_origin": np.zeros(3),
                "beam_direction": np.array([1.0, 0.0, 0.0]),
                "beam_range": 1.0,
                "beam_slope": 0.0,
                "beam_inside_opacity": 1.0,
                "beam_outside_opacity": 1.0,
                "beam_highlight_feather": 0.0,
                "beam_highlight_strength": 0.0,
                "cloud_shader_token": self._shader_token,
            }
        )

    def bind_beam_highlight(
        self,
        beam,
        *,
        inside_opacity=1.0,
        outside_opacity=0.12,
        feather=0.03,
    ):
        """Bind an opacity mask for the part of this cloud inside ``beam``.

        The opacity arguments multiply the cloud's existing per-vertex alpha.
        Animate ``beam_highlight`` from 0 to 1 to reveal the mask.
        """
        if not isinstance(beam, Beam):
            raise TypeError("beam must be a Beam")
        values = {
            "inside_opacity": inside_opacity,
            "outside_opacity": outside_opacity,
            "feather": feather,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if inside_opacity > 1 or outside_opacity > 1:
            raise ValueError("opacity multipliers must be at most 1")

        self._highlight_beam = beam
        self.set_uniforms(
            {
                "beam_inside_opacity": float(inside_opacity),
                "beam_outside_opacity": float(outside_opacity),
                "beam_highlight_feather": float(feather),
            }
        )
        if _update_beam_highlight not in self.updaters:
            self.add_updater(_update_beam_highlight, call=False)
        return _update_beam_highlight(self)

    def highlight_in_beam(self, beam, **kwargs):
        self.bind_beam_highlight(beam, **kwargs)
        self.beam_highlight @= 1
        return _update_beam_highlight(self)

    def clear_beam_highlight(self):
        self.beam_highlight @= 0
        self._highlight_beam = None
        self.remove_updater(_update_beam_highlight)
        return self.set_uniforms({"beam_highlight_strength": 0.0})

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
