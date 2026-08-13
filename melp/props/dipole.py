from __future__ import annotations

import numpy as np
from manimlib import RIGHT, UP, Cylinder, Group


__all__ = ["Dipole"]

METAL_COLOR = "#AEB8BC"


def _positive_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


class Dipole(Group):
    """Simple center-fed dipole made from four cylindrical conductors."""

    def __init__(
        self,
        length=4,
        radius=0.06,
        feed_gap=0.2,
        feed_length=0.7,
        color=METAL_COLOR,
        opacity=1,
        shading=(0.15, 0.35, 0.2),
        resolution=(32, 4),
        **kwargs,
    ):
        length = _positive_finite(length, "length")
        radius = _positive_finite(radius, "radius")
        feed_gap = _positive_finite(feed_gap, "feed_gap")
        feed_length = _positive_finite(feed_length, "feed_length")
        if feed_gap >= length:
            raise ValueError("feed_gap must be smaller than length")

        arm_length = (length - feed_gap) / 2
        arm_offset = (length + feed_gap) / 4
        self.length = length
        self.radius = radius
        self.feed_gap = feed_gap
        self.feed_length = feed_length
        self.arm_length = arm_length

        cylinder_config = {
            "radius": radius,
            "resolution": resolution,
            "color": color,
            "opacity": opacity,
            "shading": shading,
        }

        self.left_arm = Cylinder(
            height=arm_length,
            axis=RIGHT,
            **cylinder_config,
        ).shift(-arm_offset * RIGHT)
        self.right_arm = Cylinder(
            height=arm_length,
            axis=RIGHT,
            **cylinder_config,
        ).shift(arm_offset * RIGHT)
        self.left_feed = Cylinder(
            height=feed_length,
            axis=UP,
            **cylinder_config,
        ).shift(-feed_gap / 2 * RIGHT - feed_length / 2 * UP)
        self.right_feed = Cylinder(
            height=feed_length,
            axis=UP,
            **cylinder_config,
        ).shift(feed_gap / 2 * RIGHT - feed_length / 2 * UP)

        self.arms = Group(self.left_arm, self.right_arm)
        self.feed_legs = Group(self.left_feed, self.right_feed)

        super().__init__(self.arms, self.feed_legs, **kwargs)
