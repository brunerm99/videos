"""Configurable easing rate function factories for Manim animations."""

from __future__ import annotations

import math

from manim.utils import rate_functions


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
