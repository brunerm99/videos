from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Callable

import numpy as np
from manimlib import Animation, Mobject, ORIGIN, OUT, ValueTracker
from manimlib.utils.bezier import interpolate
from manimlib.utils.paths import path_along_arc, straight_path
from manimlib.utils.rate_functions import linear, smooth

__all__ = ["VT", "fh", "fw", "multi_anim", "rotate_vector", "unit"]


@dataclass
class _MultiAnimOperation:
    start_time: float
    run_time: float
    rate_func: Callable[[float], float]
    apply: Callable[[Mobject, float], None]


class _MultiAnim(Animation):
    def __init__(self, mobject: Mobject, lag_ratio: float = 0):
        lag_ratio = float(lag_ratio)
        if not np.isfinite(lag_ratio) or lag_ratio < 0:
            raise ValueError("lag_ratio must be a finite non-negative number")

        self.operation_lag_ratio = lag_ratio
        self.operations: list[_MultiAnimOperation] = []
        self.timeline_run_time = 0.0
        super().__init__(mobject, run_time=0, rate_func=linear)

    def _add_operation(
        self,
        apply: Callable[[Mobject, float], None],
        run_time: float,
        rate_func: Callable[[float], float],
    ):
        run_time = float(run_time)
        if not np.isfinite(run_time) or run_time <= 0:
            raise ValueError("run_time must be a finite positive number")
        if not callable(rate_func):
            raise TypeError("rate_func must be callable")

        if self.operations:
            previous = self.operations[-1]
            start_time = (
                previous.start_time
                + self.operation_lag_ratio * previous.run_time
            )
        else:
            start_time = 0.0

        operation = _MultiAnimOperation(
            start_time=start_time,
            run_time=run_time,
            rate_func=rate_func,
            apply=apply,
        )
        self.operations.append(operation)
        self.timeline_run_time = max(
            self.timeline_run_time,
            start_time + run_time,
        )
        self.run_time = self.timeline_run_time
        return self

    @staticmethod
    def _get_path_func(path_arc, path_arc_axis, path_func):
        if path_func is not None:
            if not callable(path_func):
                raise TypeError("path_func must be callable")
            return path_func
        if path_arc == 0:
            return straight_path
        return path_along_arc(path_arc, path_arc_axis)

    def shift(
        self,
        vector,
        *,
        path_arc=0,
        path_arc_axis=OUT,
        path_func=None,
        rate_func=smooth,
        run_time=1,
    ):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,):
            raise ValueError("shift vector must have shape (3,)")

        path = self._get_path_func(path_arc, path_arc_axis, path_func)
        start = np.array([ORIGIN])
        target = np.array([vector])

        def apply(mobject, alpha):
            mobject.shift(path(start, target, alpha)[0])

        return self._add_operation(apply, run_time, rate_func)

    def move_to(
        self,
        point_or_mobject,
        aligned_edge=ORIGIN,
        coor_mask=np.array([1, 1, 1]),
        *,
        offset=ORIGIN,
        path_arc=0,
        path_arc_axis=OUT,
        path_func=None,
        rate_func=smooth,
        run_time=1,
    ):
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject

        target = np.asarray(target, dtype=float)
        aligned_edge = np.asarray(aligned_edge, dtype=float)
        coor_mask = np.asarray(coor_mask, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if target.shape != (3,):
            raise ValueError("move_to target must have shape (3,)")
        if aligned_edge.shape != (3,):
            raise ValueError("aligned_edge must have shape (3,)")
        if coor_mask.shape != (3,):
            raise ValueError("coor_mask must have shape (3,)")
        if offset.shape != (3,):
            raise ValueError("offset must have shape (3,)")

        path = self._get_path_func(path_arc, path_arc_axis, path_func)

        def apply(mobject, alpha):
            start = mobject.get_bounding_box_point(aligned_edge)
            end = start + (target - start) * coor_mask + offset
            point = path(
                np.array([start]),
                np.array([end]),
                alpha,
            )[0]
            mobject.shift(point - start)

        return self._add_operation(apply, run_time, rate_func)

    def scale(
        self,
        scale_factor,
        min_scale_factor=1e-8,
        about_point=None,
        about_edge=ORIGIN,
        *,
        rate_func=smooth,
        run_time=1,
    ):
        if isinstance(scale_factor, numbers.Number):
            target_factor = max(scale_factor, min_scale_factor)
            start_factor = 1.0
        else:
            target_factor = np.asarray(scale_factor, dtype=float).clip(
                min=min_scale_factor
            )
            start_factor = np.ones_like(target_factor)

        if about_point is not None:
            about_point = np.asarray(about_point, dtype=float)
        about_edge = np.asarray(about_edge, dtype=float)

        def apply(mobject, alpha):
            mobject.scale(
                interpolate(start_factor, target_factor, alpha),
                min_scale_factor=min_scale_factor,
                about_point=about_point,
                about_edge=about_edge,
            )

        return self._add_operation(apply, run_time, rate_func)

    def begin(self):
        if not self.operations:
            raise ValueError("multi_anim requires at least one operation")
        super().begin()

    def interpolate_mobject(self, alpha):
        self.mobject.become(self.starting_mobject)
        current_time = np.clip(alpha, 0, 1) * self.timeline_run_time

        for operation in self.operations:
            operation_alpha = np.clip(
                (current_time - operation.start_time) / operation.run_time,
                0,
                1,
            )
            operation.apply(
                self.mobject,
                operation.rate_func(operation_alpha),
            )


def multi_anim(mobject: Mobject, lag_ratio: float = 0):
    """Build one animation with independently timed mobject operations."""
    return _MultiAnim(mobject, lag_ratio=lag_ratio)


class VT(ValueTracker):
    def __invert__(self):
        return self.get_value()

    def __matmul__(self, value):
        return self.animate.set_value(value)

    def __imatmul__(self, value):
        self.set_value(value)
        return self

    def __add__(self, difference):
        return self.animate.increment_value(difference)

    def __iadd__(self, difference):
        self.increment_value(difference)
        return self

    def __sub__(self, difference):
        return self.animate.increment_value(-difference)

    def __isub__(self, difference):
        self.increment_value(-difference)
        return self

    def __mul__(self, factor):
        return self.animate.set_value(self.get_value() * factor)

    def __imul__(self, factor):
        self.set_value(self.get_value() * factor)
        return self

    def __truediv__(self, divisor):
        if divisor == 0:
            raise ZeroDivisionError("division by zero")
        return self.animate.set_value(self.get_value() / divisor)

    def __itruediv__(self, divisor):
        if divisor == 0:
            raise ZeroDivisionError("division by zero")
        self.set_value(self.get_value() / divisor)
        return self


def fh(scene, scale=1):
    return scene.frame.get_height() * scale


def fw(scene, scale: float = 1):
    return scene.frame.get_width() * scale


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def rotate_vector(vector, angle, axis=OUT):
    axis = unit(axis)
    vector = np.asarray(vector, dtype=float)
    cross_matrix = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    rotation = (
        np.identity(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross_matrix
    )
    return rotation @ vector
