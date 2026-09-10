import numpy as np
from manimlib import BLACK, ORIGIN, WHITE, YELLOW, Dot, Line, Text, VGroup, always_redraw

__all__ = ["ChargePair"]


class ChargePair(VGroup):
    def __init__(
        self,
        center=ORIGIN,
        angle=0,
        radius=0.035,
        separation=0.13,
        font="",
        positive_color=YELLOW,
        negative_color=WHITE,
        symbol_color=BLACK,
        **kwargs,
    ):
        def get_charge(symbol, color):
            return VGroup(
                Dot(radius=radius, fill_color=color),
                Text(symbol, font=font, fill_color=symbol_color).set_width(radius * 1.2),
            )

        offset = separation / 2 * np.array([np.cos(angle), np.sin(angle), 0])
        self.positive = get_charge("+", positive_color).move_to(center + offset)
        self.negative = get_charge("-", negative_color).move_to(center - offset)
        self.connector = always_redraw(
            lambda: (
                Line(
                    self.positive.get_center(),
                    self.negative.get_center(),
                    stroke_color=WHITE,
                    stroke_width=2,
                    stroke_opacity=0.65,
                )
                .deactivate_depth_test()
                .set_z_index(1)
            )
        )
        super().__init__(self.connector, self.positive, self.negative, **kwargs)
        self.deactivate_depth_test().set_z_index(1)
