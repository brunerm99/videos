# resolution.py


from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
import sys
from MF_Tools import VT

sys.path.insert(0, "..")

from props import WeatherRadarTower
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR


class Intro(Scene):
    def construct(self): ...


class RangeResolution(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4)

        self.play(radar.get_animation())

        self.wait(0.5)

        self.play(radar.vgroup.animate.to_corner(DL, LARGE_BUFF))

        self.wait(0.5)

        target1 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .to_edge(RIGHT, LARGE_BUFF * 2)
            .shift(UP / 2)
            .set_fill(GREEN)
            .set_color(GREEN)
        )
        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .to_edge(RIGHT, LARGE_BUFF * 1)
            .shift(DOWN)
            .set_fill(ORANGE)
            .set_color(ORANGE)
        )

        ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=config.frame_width * 0.8,
                y_length=radar.radome.height,
            )
            .set_opacity(0)
            .next_to(radar.radome, RIGHT, 0)
        )
        target1_line = Line(target1.get_left(), radar.radome.get_right())
        target1_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target1_line.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target1_line.get_angle())
            .set_opacity(0)
        )
        target1_ax.shift(target1.get_left() - target1_ax.c2p(0, 0))

        target2_line = Line(target2.get_left(), radar.radome.get_right())
        target2_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target2_line.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target2_line.get_angle())
            .set_opacity(0)
        )
        target2_ax.shift(target2.get_left() - target2_ax.c2p(0, 0))
        # self.add(target1_ax)
        # self.add(ax, target)
        xmax = VT(0)
        xmax_t1 = VT(0)
        xmax_t2 = VT(0)
        pw = 0.2
        f = 10
        tx = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax - pw), ~xmax, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax_t1 - pw), ~xmax_t1, 1 / 200],
                color=RX_COLOR,
            )
        )
        rx2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax_t2 - pw), ~xmax_t2, 1 / 200],
                color=RX_COLOR,
            )
        )
        self.add(tx, rx1, rx2)

        radar.vgroup.set_z_index(1)

        to_target1 = Arrow(
            radar.radome.get_right(), target1.get_left(), color=TX_COLOR
        ).shift(DOWN / 3)
        from_target1 = Arrow(
            target1.get_left(), radar.radome.get_right(), color=RX_COLOR
        ).shift(UP / 3)

        self.play(
            LaggedStart(
                GrowArrow(to_target1),
                target1.shift(RIGHT * 8).animate.shift(LEFT * 8),
                GrowArrow(from_target1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(FadeOut(to_target1, from_target1))

        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        pw_line = Line(ax.c2p(~xmax - pw, 1.2), ax.c2p(~xmax, 1.2))
        pw_line_l = Line(pw_line.get_start() + DOWN / 8, pw_line.get_start() + UP / 8)
        pw_line_r = Line(pw_line.get_end() + DOWN / 8, pw_line.get_end() + UP / 8)

        pw_label_val = MathTex(r"1 \mu s").next_to(pw_line, UP)
        pw_label = MathTex(r"\tau = 1 \mu s").next_to(pw_line, UP)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in pw_label_val[0]], lag_ratio=0.15),
            LaggedStart(
                Create(pw_line_l),
                Create(pw_line),
                Create(pw_line_r),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(pw_label_val[0], pw_label[0][-3:]),
                *[GrowFromCenter(m) for m in pw_label[0][:-3]],
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        self.play(FadeOut(pw_line, pw_line_l, pw_line_r, pw_label_val))

        self.wait(0.5)

        self.play(
            LaggedStart(
                xmax @ (ax.p2c(target2.get_left())[0]),
                xmax_t1 @ (pw / 2),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(target2.shift(RIGHT * 8).animate.shift(LEFT * 8), xmax_t2 @ (pw / 2))

        # self.add(pw_label_val)

        # self.add(target2)

        self.wait(2)
