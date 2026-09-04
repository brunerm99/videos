from __future__ import annotations

import numpy as np
from manimlib import (
    MED_SMALL_BUFF,
    ORIGIN,
    OUT,
    RIGHT,
    GrowFromCenter,
    Group,
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
    """A line with end ticks for showing one dimension of a mobject."""

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
        **kwargs,
    ):
        if not isinstance(mobject, Mobject):
            raise TypeError("mobject must be a Mobject")

        direction = normalize(_vector3(direction, "direction"))
        if placement is None:
            placement = np.cross(direction, OUT)
        placement = normalize(_vector3(placement, "placement"))
        buff = _nonnegative_finite(buff, "buff")
        tick_length = _nonnegative_finite(tick_length, "tick_length")
        label_buff = _nonnegative_finite(label_buff, "label_buff")

        line_config = dict(line_config or {})
        line_config.setdefault("buff", 0)

        start = mobject.get_bounding_box_point(-direction)
        end = mobject.get_bounding_box_point(direction)
        if np.allclose(start, end):
            raise ValueError("mobject has no extent along direction")

        self.line = Line(start, end, **line_config).next_to(
            mobject,
            placement,
            buff=buff,
        )

        line_direction = normalize(self.line.get_end() - self.line.get_start())
        tick_direction = np.cross(OUT, line_direction)
        if get_norm(tick_direction) == 0:
            tick_direction = np.cross(RIGHT, line_direction)
        tick_vector = 0.5 * tick_length * normalize(tick_direction)

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
            self.label = label.next_to(
                self.line,
                label_direction,
                buff=label_buff,
            )
            members.append(self.label)

        super().__init__(*members, **kwargs)

    def get_creation_animation(self, lag_ratio=0.2, **kwargs):
        animations = [
            ShowCreation(self.start_tick),
            ShowCreation(self.line),
            ShowCreation(self.end_tick),
        ]
        if self.label is not None:
            animations.append(GrowFromCenter(self.label))
        creation = LaggedStart(
            *animations,
            lag_ratio=lag_ratio,
            **kwargs,
        )
        creation.group = self
        creation.mobject = self
        return creation
