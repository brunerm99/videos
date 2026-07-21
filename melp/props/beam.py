from __future__ import annotations

import numpy as np
from manimlib import BLUE, ORIGIN, RIGHT, TAU, Arc, Cone, DotCloud, ParametricSurface

__all__ = ["Beam"]


class Beam(Cone):
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

    def _get_surface_ring(self, index):
        ring = self.get_points().reshape((*self.resolution, 3))[:, index]
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

    def get_beamwidth_arc(
        self,
        range,
        camera_frame,
        *,
        color=None,
        stroke_width=4,
        stroke_opacity=1,
        depth_test=False,
    ):
        arc_range = float(range)
        beam_range = self.get_range()
        if not np.isfinite(arc_range) or arc_range <= 0:
            raise ValueError("range must be positive and finite")
        if arc_range > beam_range:
            raise ValueError(f"range must not exceed the beam range ({beam_range:g})")

        origin = self.get_origin()
        direction = self.get_direction()
        arc_midpoint = origin + arc_range * direction
        camera_direction = camera_frame.get_implied_camera_location() - arc_midpoint
        camera_distance = np.linalg.norm(camera_direction)
        camera_view_direction = (
            camera_direction / camera_distance
            if camera_distance > 1e-9
            else np.zeros(3)
        )
        screen_width_direction = np.cross(camera_view_direction, direction)
        screen_width_norm = np.linalg.norm(screen_width_direction)
        if screen_width_norm < 1e-8:
            camera_right = camera_frame.get_inverse_camera_rotation_matrix()[0]
            screen_width_direction = (
                camera_right - np.dot(camera_right, direction) * direction
            )
            screen_width_norm = np.linalg.norm(screen_width_direction)
        if screen_width_norm < 1e-8:
            fallback_axis = np.eye(3)[np.argmin(np.abs(direction))]
            screen_width_direction = np.cross(fallback_axis, direction)
            screen_width_norm = np.linalg.norm(screen_width_direction)
        screen_width_direction /= screen_width_norm

        arc_normal = np.cross(direction, screen_width_direction)
        half_beamwidth = np.arctan2(self.get_base_radius(), beam_range)
        arc = Arc(
            start_angle=-half_beamwidth,
            angle=2 * half_beamwidth,
            radius=arc_range,
            arc_center=ORIGIN,
            stroke_color=self.get_color() if color is None else color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )
        arc.apply_matrix(
            np.column_stack((direction, screen_width_direction, arc_normal))
        )
        arc.shift(origin)
        return arc.apply_depth_test() if depth_test else arc.deactivate_depth_test()

    def _validate_section_ranges(self, start_range, stop_range):
        start_range = float(start_range)
        stop_range = float(stop_range)
        beam_range = self.get_range()
        if not np.isfinite(start_range) or not np.isfinite(stop_range):
            raise ValueError("section ranges must be finite")
        if start_range < 0:
            raise ValueError("start_range must be nonnegative")
        if stop_range <= start_range:
            raise ValueError("stop_range must be greater than start_range")
        if stop_range > beam_range:
            raise ValueError(
                f"stop_range must not exceed the beam range ({beam_range:g})"
            )
        return start_range, stop_range, beam_range

    def get_section(self, start_range, stop_range, **kwargs):
        start_range, stop_range, beam_range = self._validate_section_ranges(
            start_range,
            stop_range,
        )

        base_ring = self._get_surface_ring(-1)
        origin = self.get_origin()

        def point_at_range(angle, distance):
            ring_index = (angle % TAU) * len(base_ring) / TAU
            lower_index = int(np.floor(ring_index)) % len(base_ring)
            upper_index = (lower_index + 1) % len(base_ring)
            alpha = ring_index - np.floor(ring_index)
            base_point = (1 - alpha) * base_ring[lower_index] + alpha * base_ring[
                upper_index
            ]
            return origin + (distance / beam_range) * (base_point - origin)

        section_config = {
            "color": self.get_color(),
            "opacity": self.get_opacity(),
            "shading": tuple(self.get_shading()),
            "depth_test": self.depth_test,
            "resolution": self.resolution,
        }
        section_config.update(kwargs)
        return ParametricSurface(
            point_at_range,
            u_range=(0, TAU),
            v_range=(start_range, stop_range),
            **section_config,
        )

    def get_section_particles(
        self,
        start_range,
        stop_range,
        count=180,
        *,
        max_radius_fraction=1.0,
        seed=None,
        **kwargs,
    ):
        start_range, stop_range, beam_range = self._validate_section_ranges(
            start_range,
            stop_range,
        )
        try:
            numeric_count = float(count)
        except (TypeError, ValueError) as exc:
            raise ValueError("count must be a positive integer") from exc
        if (
            not np.isfinite(numeric_count)
            or not numeric_count.is_integer()
            or numeric_count < 1
        ):
            raise ValueError("count must be a positive integer")
        count = int(numeric_count)
        max_radius_fraction = float(max_radius_fraction)
        if not np.isfinite(max_radius_fraction) or not 0 < max_radius_fraction <= 1:
            raise ValueError("max_radius_fraction must be in (0, 1]")

        rng = np.random.default_rng(seed)
        origin = self.get_origin()
        base_ring = self._get_surface_ring(-1)
        base_center = base_ring.mean(axis=0)

        distances = np.cbrt(rng.uniform(start_range**3, stop_range**3, count))
        angles = rng.uniform(0, TAU, count)
        radial_fractions = np.sqrt(rng.uniform(0, max_radius_fraction**2, count))
        ring_indices = angles * len(base_ring) / TAU
        lower_indices = np.floor(ring_indices).astype(int) % len(base_ring)
        upper_indices = (lower_indices + 1) % len(base_ring)
        interpolation = ring_indices - np.floor(ring_indices)
        boundary_points = (1 - interpolation)[:, None] * base_ring[
            lower_indices
        ] + interpolation[:, None] * base_ring[upper_indices]
        range_fractions = distances / beam_range
        points = (
            origin
            + range_fractions[:, None] * (base_center - origin)
            + (range_fractions * radial_fractions)[:, None]
            * (boundary_points - base_center)
        )

        kwargs.setdefault("radius", 0.04)
        kwargs.setdefault("color", self.get_color())
        kwargs.setdefault("glow_factor", 0.3)
        return DotCloud(points, **kwargs)

    def contains_points(self, points, *, atol=1e-9):
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
