"""Configurable easing rate function factories for Manim animations."""

from __future__ import annotations

import math

from manim.utils import rate_functions


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
    """Return a CSS/anime.js-style cubic-bezier rate function.

    The solver inverts x(u)=t to preserve timing semantics, then evaluates y(u).
    """
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
    """Preset: cubicBezier(0.17, 0.581, 0.759, 0.275)."""
    return _bezier_out_in(t)


def ease_in_elastic(amplitude: float = 1.0, period: float = 0.3):
    """Return a configurable ease-in elastic rate function.

    Parameters follow the common elastic form used by JS animation libraries:
    larger ``amplitude`` increases overshoot, and ``period`` controls oscillation.
    """
    a = max(1.0, float(amplitude))
    p = max(1e-6, float(period))
    s = p / (2 * math.pi) * math.asin(1 / a)

    @rate_functions.unit_interval
    def _ease(t: float) -> float:
        if t == 0 or t == 1:
            return t
        x = t - 1
        return -(a * pow(2, 10 * x) * math.sin((x - s) * (2 * math.pi) / p))

    return _ease


def ease_out_elastic(amplitude: float = 1.0, period: float = 0.3):
    """Return a configurable ease-out elastic rate function."""
    a = max(1.0, float(amplitude))
    p = max(1e-6, float(period))
    s = p / (2 * math.pi) * math.asin(1 / a)

    @rate_functions.unit_interval
    def _ease(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return a * pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

    return _ease


def ease_in_out_elastic(amplitude: float = 1.0, period: float = 0.45):
    """Return a configurable ease-in-out elastic rate function."""
    a = max(1.0, float(amplitude))
    p = max(1e-6, float(period))
    s = p / (2 * math.pi) * math.asin(1 / a)

    @rate_functions.unit_interval
    def _ease(t: float) -> float:
        if t == 0 or t == 1:
            return t
        x = 2 * t - 1
        if x < 0:
            return -0.5 * (a * pow(2, 10 * x) * math.sin((x - s) * (2 * math.pi) / p))
        return a * pow(2, -10 * x) * math.sin((x - s) * (2 * math.pi) / p) * 0.5 + 1

    return _ease
