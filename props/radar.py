# props/radar.py

from manim import *


class WeatherRadarTower:
    def __init__(self, **kwargs):
        width_scale = 2
        LINE_STYLE = dict(
            color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * width_scale * 2
        )

        leg = Line(ORIGIN, UP * 3, **LINE_STYLE)
        self.left_leg = leg.copy().shift(LEFT)
        self.right_leg = leg.copy().shift(RIGHT)
        self.middle_leg = leg.copy().shift(DOWN / 1.5)

        self.conn1_y_shift = DOWN / 2
        self.conn1 = Line(
            self.middle_leg.get_center() + self.conn1_y_shift,
            self.right_leg.get_center() + self.conn1_y_shift,
            **LINE_STYLE,
        )
        self.conn2 = Line(
            self.middle_leg.get_center() + self.conn1_y_shift,
            self.left_leg.get_center() + self.conn1_y_shift,
            **LINE_STYLE,
        )
        self.conn3 = self.conn1.copy().shift(-self.conn1_y_shift * 2)
        self.conn4 = self.conn2.copy().shift(-self.conn1_y_shift * 2)

        self.radome = Circle(radius=1.08, **LINE_STYLE).next_to(
            self.middle_leg,
            direction=UP,
            buff=0,
        )

        self.vgroup = VGroup(
            self.left_leg,
            self.right_leg,
            self.middle_leg,
            self.conn1,
            self.conn2,
            self.conn3,
            self.conn4,
            self.radome,
        ).move_to(ORIGIN)

    def get_animation(self):
        return LaggedStart(
            AnimationGroup(
                Create(self.left_leg),
                Create(self.middle_leg),
                Create(self.right_leg),
            ),
            AnimationGroup(
                Create(self.conn1),
                Create(self.conn2),
                Create(self.conn3),
                Create(self.conn4),
            ),
            Create(self.radome),
            lag_ratio=0.75,
        )


class FMCWRadarCartoon:
    def __init__(self, text="FMCW"):
        self.rect = Rectangle(height=3, width=1.5)
        self.label = Tex(text).rotate(PI / 2)
        self.line_1 = (
            Line(ORIGIN, RIGHT / 2)
            .next_to(self.rect, direction=RIGHT, buff=0)
            .shift(UP)
        )
        self.line_2 = (
            Line(ORIGIN, RIGHT / 2)
            .next_to(self.rect, direction=RIGHT, buff=0)
            .shift(DOWN)
        )
        self.antenna_tx = (
            AnnularSector(inner_radius=1, outer_radius=1.2, angle=PI)
            .rotate(PI / 2)
            .scale(0.5)
            .next_to(self.line_1, direction=RIGHT, buff=0)
        )
        self.antenna_rx = (
            AnnularSector(inner_radius=1, outer_radius=1.2, angle=PI)
            .rotate(PI / 2)
            .scale(0.5)
            .next_to(self.line_2, direction=RIGHT, buff=0)
        )

        self.vgroup = VGroup(
            self.rect,
            self.label,
            self.line_1,
            self.line_2,
            self.antenna_tx,
            self.antenna_rx,
        ).move_to(ORIGIN)

    def get_animation(self):
        return Succession(
            GrowFromCenter(VGroup(self.rect, self.label)),
            LaggedStart(
                AnimationGroup(Create(self.line_1), Create(self.line_2)),
                AnimationGroup(
                    GrowFromCenter(self.antenna_tx), GrowFromCenter(self.antenna_rx)
                ),
                lag_ratio=0.6,
            ),
        )
