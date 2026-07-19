from __future__ import annotations

import numpy as np
from manimlib import BLUE, Cone, ORIGIN, RIGHT


__all__ = ["Beam"]


class Beam(Cone):
    """A conical radar beam with its apex at ``origin``.

    ``beamwidth`` is the full angular width in radians and ``range`` is in
    Manim scene units.
    """

    def __init__(
        self,
        beamwidth,
        range,
        *,
        origin=ORIGIN,
        direction=RIGHT,
        color=BLUE,
        opacity=0.15,
        shading=(0.05, 0.15, 0.1),
        **kwargs,
    ):
        beamwidth = float(beamwidth)
        beam_range = float(range)
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)

        if not np.isfinite(beamwidth) or not 0 < beamwidth < np.pi:
            raise ValueError("beamwidth must be finite and in (0, PI) radians")
        if not np.isfinite(beam_range) or beam_range <= 0:
            raise ValueError("range must be positive and finite")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin must be a finite 3D point")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite 3D vector")

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            raise ValueError("direction must be nonzero")
        direction = direction / direction_norm

        half_angle = beamwidth / 2.0
        base_radius = beam_range * np.tan(half_angle)

        self.beamwidth = beamwidth
        self.range = beam_range
        self.half_angle = half_angle
        self.base_radius = base_radius

        cone_axis = -direction
        kwargs.setdefault("v_range", (1.0, 0.0))
        super().__init__(
            height=beam_range,
            radius=base_radius,
            axis=cone_axis,
            color=color,
            opacity=opacity,
            shading=shading,
            **kwargs,
        )

        points = self.get_points()
        apex = points[np.argmax(points @ cone_axis)]
        self.shift(origin - apex)

    def _get_reference_grid(self):
        return self.data["du_point"].reshape((*self.resolution, 3))

    def _get_reference_ring(self, index):
        ring = self._get_reference_grid()[:, index]
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            return ring[:-1]
        return ring

    def get_origin(self):
        return self._get_reference_ring(0).mean(axis=0)

    def get_start(self):
        return self.get_origin()

    def get_end(self):
        return self._get_reference_ring(-1).mean(axis=0)

    def get_direction(self):
        direction = self.get_end() - self.get_origin()
        return direction / np.linalg.norm(direction)

    def get_range(self):
        return np.linalg.norm(self.get_end() - self.get_origin())

    def get_base_radius(self):
        base_ring = self._get_reference_ring(-1)
        return np.linalg.norm(base_ring - self.get_end(), axis=1).max()

    def contains_points(self, points, *, atol=1e-9):
        """Return whether points lie inside this beam's finite cone."""
        points = np.asarray(points, dtype=float)
        is_single_point = points.shape == (3,)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be nonnegative and finite")

        shape = points.shape[:-1]
        offsets = points.reshape((-1, 3)) - self.get_origin()
        direction = self.get_direction()
        beam_range = self.get_range()
        slope = self.get_base_radius() / beam_range

        axial = offsets @ direction
        radial = offsets - axial[:, None] * direction
        radial_squared = np.einsum("ij,ij->i", radial, radial)
        cone_radius = np.maximum(axial, 0.0) * slope
        inside = (
            (axial >= -atol)
            & (axial <= beam_range + atol)
            & (radial_squared <= (cone_radius + atol) ** 2)
        ).reshape(shape)
        return bool(inside) if is_single_point else inside

    @property
    def origin(self):
        return self.get_origin()

    @property
    def direction(self):
        return self.get_direction()
