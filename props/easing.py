# props/easing.py

from __future__ import annotations

import math
from collections.abc import Callable

from manim.utils import rate_functions

_MISSING = object()


def cubic_bezier(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    epsilon: float = 1e-7,
    max_newton_iters: int = 10,
    max_bisection_iters: int = 24,
):
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    cx = 3.0 * x1
    bx = 3.0 * (x2 - x1) - cx
    ax = 1.0 - cx - bx

    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by

    def sample_x(u: float) -> float:
        return ((ax * u + bx) * u + cx) * u

    def sample_dx(u: float) -> float:
        return (3.0 * ax * u + 2.0 * bx) * u + cx

    def sample_y(u: float) -> float:
        return ((ay * u + by) * u + cy) * u

    def solve_u(t: float) -> float:
        # Newton-Raphson pass
        u = t
        for _ in range(max_newton_iters):
            x_err = sample_x(u) - t
            if abs(x_err) < epsilon:
                return u
            dx = sample_dx(u)
            if abs(dx) < 1e-6:
                break
            u -= x_err / dx
            u = min(max(u, 0.0), 1.0)

        # Bisection fallback
        lo = 0.0
        hi = 1.0
        u = t
        for _ in range(max_bisection_iters):
            x = sample_x(u)
            x_err = x - t
            if abs(x_err) < epsilon:
                return u
            if x < t:
                lo = u
            else:
                hi = u
            u = 0.5 * (lo + hi)
        return u

    @rate_functions.unit_interval
    def _ease(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return sample_y(solve_u(t))

    return _ease


_bezier_out_in = cubic_bezier(0.17, 0.581, 0.759, 0.275)


def bezier_out_in(t: float) -> float:
    return _bezier_out_in(t)


def _elastic_settings(amplitude: float, period: float) -> tuple[float, float, float]:
    a = max(1.0, float(amplitude))
    p = max(1e-6, float(period))
    s = p / (2 * math.pi) * math.asin(1 / a)
    return a, p, s


def _elastic_rate_func(
    value_func: Callable[[float, float, float], float],
    amplitude: float,
    period: float,
) -> Callable[[float], float]:
    def _ease(t: float) -> float:
        return value_func(t, amplitude, period)

    return _ease


def _elastic_value_or_factory(
    value_func: Callable[[float, float, float], float],
    t_or_amplitude: float | object,
    period: float | object,
    amplitude: float | object,
    default_period: float,
):
    if amplitude is not _MISSING:
        p = default_period if period is _MISSING else period
        if t_or_amplitude is _MISSING:
            return _elastic_rate_func(value_func, amplitude, p)
        return value_func(t_or_amplitude, amplitude, p)

    if period is not _MISSING:
        a = 1.0 if t_or_amplitude is _MISSING else t_or_amplitude
        return _elastic_rate_func(value_func, a, period)

    if t_or_amplitude is _MISSING:
        return _elastic_rate_func(value_func, 1.0, default_period)

    if float(t_or_amplitude) > 1.0:
        return _elastic_rate_func(value_func, t_or_amplitude, default_period)

    return value_func(t_or_amplitude, 1.0, default_period)


@rate_functions.unit_interval
def _ease_in_elastic_value(
    t: float, amplitude: float = 1.0, period: float = 0.3
) -> float:
    if t == 0 or t == 1:
        return t
    a, p, s = _elastic_settings(amplitude, period)
    x = t - 1
    return -(a * pow(2, 10 * x) * math.sin((x - s) * (2 * math.pi) / p))


@rate_functions.unit_interval
def _ease_out_elastic_value(
    t: float, amplitude: float = 1.0, period: float = 0.3
) -> float:
    if t == 0 or t == 1:
        return t
    a, p, s = _elastic_settings(amplitude, period)
    return a * pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1


@rate_functions.unit_interval
def _ease_in_out_elastic_value(
    t: float, amplitude: float = 1.0, period: float = 0.45
) -> float:
    if t == 0 or t == 1:
        return t
    a, p, s = _elastic_settings(amplitude, period)
    x = 2 * t - 1
    if x < 0:
        return -0.5 * (a * pow(2, 10 * x) * math.sin((x - s) * (2 * math.pi) / p))
    return a * pow(2, -10 * x) * math.sin((x - s) * (2 * math.pi) / p) * 0.5 + 1


def ease_in_elastic(
    t_or_amplitude: float | object = _MISSING,
    period: float | object = _MISSING,
    *,
    amplitude: float | object = _MISSING,
):
    return _elastic_value_or_factory(
        _ease_in_elastic_value, t_or_amplitude, period, amplitude, 0.3
    )


def ease_out_elastic(
    t_or_amplitude: float | object = _MISSING,
    period: float | object = _MISSING,
    *,
    amplitude: float | object = _MISSING,
):
    return _elastic_value_or_factory(
        _ease_out_elastic_value, t_or_amplitude, period, amplitude, 0.3
    )


def ease_in_out_elastic(
    t_or_amplitude: float | object = _MISSING,
    period: float | object = _MISSING,
    *,
    amplitude: float | object = _MISSING,
):
    return _elastic_value_or_factory(
        _ease_in_out_elastic_value, t_or_amplitude, period, amplitude, 0.45
    )
