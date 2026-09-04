from __future__ import annotations

import numpy as np
from manimlib import GREY_A, ORIGIN, OUT, Cone, Cylinder, Disk3D, Group, Point
from manimlib.utils.space_ops import (
    get_norm,
    normalize,
    rotation_between_vectors,
    z_to_vector,
)

__all__ = ["Arrow3D"]


def _point3(value, name):
    try:
        point = np.array(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 2D or 3D point") from exc

    if point.shape == (2,):
        point = np.append(point, 0.0)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite 2D or 3D point")
    return point


def _positive_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


class Arrow3D(Group):
    """A shaded, depth-tested arrow for vectors in three dimensions."""

    def __init__(
        self,
        start=ORIGIN,
        end=OUT,
        buff=0,
        shaft_radius=0.05,
        tip_radius=0.14,
        tip_length=0.32,
        max_shaft_radius_to_length_ratio=0.05,
        max_tip_radius_to_length_ratio=0.18,
        max_tip_length_to_length_ratio=0.35,
        color=GREY_A,
        tip_color=None,
        opacity=1,
        shading=(0.2, 0.4, 0.25),
        shaft_resolution=(33, 3),
        tip_resolution=(33, 5),
        depth_test=True,
        **kwargs,
    ):
        start = _point3(start, "start")
        end = _point3(end, "end")
        shaft_radius = _positive_finite(shaft_radius, "shaft_radius")
        tip_radius = _positive_finite(tip_radius, "tip_radius")
        tip_length = _positive_finite(tip_length, "tip_length")
        max_shaft_radius_to_length_ratio = _positive_finite(
            max_shaft_radius_to_length_ratio,
            "max_shaft_radius_to_length_ratio",
        )
        max_tip_radius_to_length_ratio = _positive_finite(
            max_tip_radius_to_length_ratio,
            "max_tip_radius_to_length_ratio",
        )
        max_tip_length_to_length_ratio = _positive_finite(
            max_tip_length_to_length_ratio,
            "max_tip_length_to_length_ratio",
        )

        try:
            buff = float(buff)
        except (TypeError, ValueError) as exc:
            raise ValueError("buff must be nonnegative and finite") from exc
        if not np.isfinite(buff) or buff < 0:
            raise ValueError("buff must be nonnegative and finite")

        direction = end - start
        full_length = get_norm(direction)
        if full_length == 0:
            raise ValueError("start and end must be different points")
        if 2 * buff >= full_length:
            raise ValueError("buff must be less than half the arrow length")

        unit_direction = normalize(direction)
        start = start + buff * unit_direction
        end = end - buff * unit_direction
        length = get_norm(end - start)

        shaft_radius = min(
            shaft_radius,
            max_shaft_radius_to_length_ratio * length,
        )
        tip_scale = min(
            1.0,
            max_tip_radius_to_length_ratio * length / tip_radius,
            max_tip_length_to_length_ratio * length / tip_length,
        )
        tip_radius *= tip_scale
        tip_length *= tip_scale

        tip_color = color if tip_color is None else tip_color
        tip_base = end - tip_length * unit_direction
        shaft_length = length - tip_length

        self.shaft = Cylinder(
            height=shaft_length,
            radius=shaft_radius,
            axis=unit_direction,
            resolution=shaft_resolution,
            color=color,
            opacity=opacity,
            shading=shading,
            depth_test=depth_test,
        ).shift(0.5 * (start + tip_base))

        self.tail_cap = (
            Disk3D(
                radius=shaft_radius,
                resolution=(2, shaft_resolution[0]),
                color=color,
                opacity=opacity,
                shading=shading,
                depth_test=depth_test,
            )
            .apply_matrix(z_to_vector(unit_direction))
            .shift(start)
        )

        self.tip = Cone(
            height=tip_length,
            radius=tip_radius,
            axis=unit_direction,
            resolution=tip_resolution,
            color=tip_color,
            opacity=opacity,
            shading=shading,
            depth_test=depth_test,
        )
        tip_point = self.tip.get_points()[
            np.argmax(np.dot(self.tip.get_points(), unit_direction))
        ]
        self.tip.shift(end - tip_point)

        self.tip_cap = (
            Disk3D(
                radius=tip_radius,
                resolution=(2, tip_resolution[0]),
                color=tip_color,
                opacity=opacity,
                shading=shading,
                depth_test=depth_test,
            )
            .apply_matrix(z_to_vector(unit_direction))
            .shift(tip_base)
        )

        self.start_anchor = Point(start)
        self.end_anchor = Point(end)
        self.body = Group(self.tail_cap, self.shaft)

        super().__init__(
            self.body,
            self.tip_cap,
            self.tip,
            self.start_anchor,
            self.end_anchor,
            **kwargs,
        )

        if depth_test:
            self.apply_depth_test()
        else:
            self.deactivate_depth_test()

    def get_start(self):
        return self.start_anchor.get_location()

    def get_end(self):
        return self.end_anchor.get_location()

    def get_start_and_end(self):
        return self.get_start(), self.get_end()

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return normalize(self.get_vector())

    def get_length(self):
        return get_norm(self.get_vector())

    def get_tip(self):
        return self.tip

    def put_start_and_end_on(self, start, end):
        start = _point3(start, "start")
        end = _point3(end, "end")
        target_vector = end - start
        target_length = get_norm(target_vector)
        if target_length == 0:
            raise ValueError("start and end must be different points")

        current_start = self.get_start()
        current_vector = self.get_vector()
        current_length = get_norm(current_vector)
        self.scale(target_length / current_length, about_point=current_start)
        self.apply_matrix(
            rotation_between_vectors(current_vector, target_vector),
            about_point=current_start,
        )
        self.shift(start - self.get_start())
        return self
