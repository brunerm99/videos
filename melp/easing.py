"""Inputs are normalized to [0, 1]; back and elastic outputs may overshoot."""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = [
    "bezier_out_in",
    "cubic_bezier",
    "ease_in_back",
    "ease_in_bounce",
    "ease_in_circ",
    "ease_in_cubic",
    "ease_in_elastic",
    "ease_in_expo",
    "ease_in_out_back",
    "ease_in_out_bounce",
    "ease_in_out_circ",
    "ease_in_out_cubic",
    "ease_in_out_elastic",
    "ease_in_out_expo",
    "ease_in_out_quad",
    "ease_in_out_quart",
    "ease_in_out_quint",
    "ease_in_out_sine",
    "ease_in_quad",
    "ease_in_quart",
    "ease_in_quint",
    "ease_in_sine",
    "ease_out_back",
    "ease_out_bounce",
    "ease_out_circ",
    "ease_out_cubic",
    "ease_out_elastic",
    "ease_out_expo",
    "ease_out_quad",
    "ease_out_quart",
    "ease_out_quint",
    "ease_out_sine",
    "get_phase_eased_sine",
    "linear",
    "smootherstep",
]

_BACK_C1 = 1.70158
_BACK_C2 = _BACK_C1 * 1.525
_BACK_C3 = _BACK_C1 + 1
_ELASTIC_C4 = 2 * math.pi / 3
_ELASTIC_C5 = 2 * math.pi / 4.5


def linear(x: float) -> float:
    return x


def ease_in_sine(x: float) -> float:
    return 1 - math.cos(x * math.pi / 2)


def ease_out_sine(x: float) -> float:
    return math.sin(x * math.pi / 2)


def ease_in_out_sine(x: float) -> float:
    return -(math.cos(math.pi * x) - 1) / 2


def ease_in_quad(x: float) -> float:
    return x * x


def ease_out_quad(x: float) -> float:
    return 1 - (1 - x) * (1 - x)


def ease_in_out_quad(x: float) -> float:
    if x < 0.5:
        return 2 * x * x
    return 1 - (-2 * x + 2) ** 2 / 2


def ease_in_cubic(x: float) -> float:
    return x * x * x


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    if x < 0.5:
        return 4 * x * x * x
    return 1 - (-2 * x + 2) ** 3 / 2


def ease_in_quart(x: float) -> float:
    return x * x * x * x


def ease_out_quart(x: float) -> float:
    return 1 - (1 - x) ** 4


def ease_in_out_quart(x: float) -> float:
    if x < 0.5:
        return 8 * x * x * x * x
    return 1 - (-2 * x + 2) ** 4 / 2


def ease_in_quint(x: float) -> float:
    return x * x * x * x * x


def ease_out_quint(x: float) -> float:
    return 1 - (1 - x) ** 5


def ease_in_out_quint(x: float) -> float:
    if x < 0.5:
        return 16 * x * x * x * x * x
    return 1 - (-2 * x + 2) ** 5 / 2


def ease_in_expo(x: float) -> float:
    if x == 0:
        return 0
    return 2 ** (10 * x - 10)


def ease_out_expo(x: float) -> float:
    if x == 1:
        return 1
    return 1 - 2 ** (-10 * x)


def ease_in_out_expo(x: float) -> float:
    if x == 0 or x == 1:
        return x
    if x < 0.5:
        return 2 ** (20 * x - 10) / 2
    return (2 - 2 ** (-20 * x + 10)) / 2


def ease_in_circ(x: float) -> float:
    return 1 - math.sqrt(1 - x * x)


def ease_out_circ(x: float) -> float:
    return math.sqrt(1 - (x - 1) ** 2)


def ease_in_out_circ(x: float) -> float:
    if x < 0.5:
        return (1 - math.sqrt(1 - (2 * x) ** 2)) / 2
    return (math.sqrt(1 - (-2 * x + 2) ** 2) + 1) / 2


def ease_in_back(x: float) -> float:
    return _BACK_C3 * x * x * x - _BACK_C1 * x * x


def ease_out_back(x: float) -> float:
    return 1 + _BACK_C3 * (x - 1) ** 3 + _BACK_C1 * (x - 1) ** 2


def ease_in_out_back(x: float) -> float:
    if x < 0.5:
        return ((2 * x) ** 2 * ((_BACK_C2 + 1) * 2 * x - _BACK_C2)) / 2
    return (
        (2 * x - 2) ** 2 * ((_BACK_C2 + 1) * (2 * x - 2) + _BACK_C2) + 2
    ) / 2


def ease_in_elastic(x: float) -> float:
    if x == 0 or x == 1:
        return x
    return -(2 ** (10 * x - 10)) * math.sin((10 * x - 10.75) * _ELASTIC_C4)


def ease_out_elastic(x: float) -> float:
    if x == 0 or x == 1:
        return x
    return 2 ** (-10 * x) * math.sin((10 * x - 0.75) * _ELASTIC_C4) + 1


def ease_in_out_elastic(x: float) -> float:
    if x == 0 or x == 1:
        return x
    if x < 0.5:
        return -(
            2 ** (20 * x - 10) * math.sin((20 * x - 11.125) * _ELASTIC_C5)
        ) / 2
    return (
        2 ** (-20 * x + 10) * math.sin((20 * x - 11.125) * _ELASTIC_C5) / 2
        + 1
    )


def ease_out_bounce(x: float) -> float:
    n1 = 7.5625
    d1 = 2.75

    if x < 1 / d1:
        return n1 * x * x
    if x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def ease_in_bounce(x: float) -> float:
    return 1 - ease_out_bounce(1 - x)


def ease_in_out_bounce(x: float) -> float:
    if x < 0.5:
        return (1 - ease_out_bounce(1 - 2 * x)) / 2
    return (1 + ease_out_bounce(2 * x - 1)) / 2


def cubic_bezier(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    epsilon: float = 1e-7,
    max_newton_iters: int = 10,
    max_bisection_iters: int = 24,
) -> Callable[[float], float]:
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    cx = 3 * x1
    bx = 3 * (x2 - x1) - cx
    ax = 1 - cx - bx

    cy = 3 * y1
    by = 3 * (y2 - y1) - cy
    ay = 1 - cy - by

    def sample_x(u: float) -> float:
        return ((ax * u + bx) * u + cx) * u

    def sample_dx(u: float) -> float:
        return (3 * ax * u + 2 * bx) * u + cx

    def sample_y(u: float) -> float:
        return ((ay * u + by) * u + cy) * u

    def solve_u(x: float) -> float:
        u = x
        for _ in range(max_newton_iters):
            error = sample_x(u) - x
            if abs(error) < epsilon:
                return u
            derivative = sample_dx(u)
            if abs(derivative) < 1e-6:
                break
            u = min(max(u - error / derivative, 0), 1)

        low = 0.0
        high = 1.0
        u = x
        for _ in range(max_bisection_iters):
            sample = sample_x(u)
            error = sample - x
            if abs(error) < epsilon:
                return u
            if sample < x:
                low = u
            else:
                high = u
            u = (low + high) / 2
        return u

    def easing(x: float) -> float:
        if x == 0 or x == 1:
            return x
        return sample_y(solve_u(x))

    return easing


_BEZIER_OUT_IN = cubic_bezier(0.17, 0.581, 0.759, 0.275)


def bezier_out_in(x: float) -> float:
    return _BEZIER_OUT_IN(x)


def smootherstep(x: float) -> float:
    return x**3 * (x * (x * 6 - 15) + 10)


def get_phase_eased_sine(
    periods: float = 5,
    amplitude: float = 1,
    phase: float = 0,
) -> Callable[[float], float]:
    def phase_eased_sine(x: float) -> float:
        eased_x = smootherstep(x)
        return amplitude * math.sin(2 * math.pi * periods * eased_x + phase)

    return phase_eased_sine
