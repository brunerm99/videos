from __future__ import annotations

import numpy as np
from manimlib import OUT, ValueTracker

__all__ = ["VT", "fh", "fw", "rotate_vector", "unit"]


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
