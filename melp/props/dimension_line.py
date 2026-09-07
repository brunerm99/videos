from __future__ import annotations

from typing import Callable

import numpy as np
from manimlib import (
    MED_SMALL_BUFF,
    OUT,
    RIGHT,
    Group,
    GrowFromCenter,
    LaggedStart,
    Line,
    Mobject,
    ShowCreation,
    Tex,
)
from manimlib.utils.space_ops import get_norm, normalize

__all__ = ["DimensionLine"]


def _vector3(value, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3D vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3D vector")
    if get_norm(vector) == 0:
        raise ValueError(f"{name} must be nonzero")
    return vector


def _nonnegative_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return value


class DimensionLine(Group):
    def __init__(
        self,
        mobject: Mobject,
        direction=RIGHT,
        placement=None,
        buff=MED_SMALL_BUFF,
        tick_length=0.25,
        label=None,
        label_constructor=Tex,
        label_direction=None,
        label_buff=MED_SMALL_BUFF,
        line_config=None,
        camera_frame=None,
        **kwargs,
    ):
        if not isinstance(mobject, Mobject):
            raise TypeError("mobject must be a Mobject")

        direction = normalize(_vector3(direction, "direction"))
        camera_rotation = (
            camera_frame.get_inverse_camera_rotation_matrix()
            if camera_frame is not None
            else None
        )
        if placement is None:
            if camera_rotation is None:
                placement = np.cross(direction, OUT)
                if get_norm(placement) < 1e-8:
                    placement = np.cross(direction, RIGHT)
            else:
                placement = np.cross(camera_rotation[2], direction)
                if get_norm(placement) < 1e-8:
                    placement = camera_rotation[1]
                if np.dot(placement, camera_rotation[1]) < 0:
                    placement = -placement
        placement = normalize(_vector3(placement, "placement"))
        placement = placement - np.dot(placement, direction) * direction
        if get_norm(placement) < 1e-8:
            raise ValueError("placement must not be parallel to direction")
        placement = normalize(placement)
        buff = _nonnegative_finite(buff, "buff")
        tick_length = _nonnegative_finite(tick_length, "tick_length")
        label_buff = _nonnegative_finite(label_buff, "label_buff")

        line_config = dict(line_config or {})
        line_config.setdefault("buff", 0)

        center = mobject.get_center()
        points = mobject.get_all_points() - center
        distances = points @ direction
        if len(distances) == 0 or np.isclose(distances.min(), distances.max()):
            raise ValueError("mobject has no extent along direction")
        offset = ((points @ placement).max() + buff) * placement
        start = center + distances.min() * direction + offset
        end = center + distances.max() * direction + offset
        self.line = Line(start, end, **line_config)
        tick_vector = 0.5 * tick_length * placement

        self.start_tick = Line(
            self.line.get_start() - tick_vector,
            self.line.get_start() + tick_vector,
            **line_config,
        )
        self.end_tick = Line(
            self.line.get_end() - tick_vector,
            self.line.get_end() + tick_vector,
            **line_config,
        )

        self.label = None
        members = [self.start_tick, self.line, self.end_tick]
        if label is not None:
            if isinstance(label, str):
                if not callable(label_constructor):
                    raise TypeError("label_constructor must be callable")
                label = label_constructor(label)
                if not isinstance(label, Mobject):
                    raise TypeError("label_constructor must return a Mobject")
            elif not isinstance(label, Mobject):
                raise TypeError("label must be a string, Mobject, or None")

            if label_direction is None:
                label_direction = placement
            else:
                label_direction = normalize(
                    _vector3(label_direction, "label_direction")
                )
            if camera_rotation is not None:
                label.apply_matrix(camera_rotation.T, about_point=label.get_center())
            line_center = self.line.get_center()
            line_extent = (
                (self.line.get_all_points() - line_center) @ label_direction
            ).max()
            label_extent = (
                (label.get_all_points() - label.get_center()) @ label_direction
            ).min()
            self.label = label.move_to(
                line_center
                + (line_extent - label_extent + label_buff) * label_direction
            )
            members.append(self.label)

        super().__init__(*members, **kwargs)

    def get_creation_animation(
        self,
        lag_ratio=0.2,
        label_anim: Callable | None = None,
        **kwargs,
    ):
        animations = [
            ShowCreation(self.start_tick),
            ShowCreation(self.line),
            ShowCreation(self.end_tick),
        ]
        if self.label is not None:
            label_anim = label_anim or GrowFromCenter
            animations.append(label_anim(self.label))
        creation = LaggedStart(
            *animations,
            lag_ratio=lag_ratio,
            **kwargs,
        )
        creation.group = self
        creation.mobject = self
        return creation
