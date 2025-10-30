# amp_gain.py

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from manim import *
from MF_Tools import VT
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower, get_blocks
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True

load_dotenv("../.env")
FONT = os.getenv("FONT")

BLOCKS = get_blocks()

GOOD = BLUE
OK = GREY
BAD = RED
TARGET1_COLOR = GREEN
TARGET2_COLOR = ORANGE
TARGET3_COLOR = BLUE
INPUT_COLOR = BLUE
OUTPUT_COLOR = ORANGE


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


class Amp(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        amp_tri = (
            Triangle(stroke_width=DEFAULT_STROKE_WIDTH * 2, color=GREEN)
            .rotate(PI / 6)
            .set_z_index(1)
        )
        amp_box = (
            RoundedRectangle(
                width=amp_tri.width * 2,
                height=amp_tri.width * 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(amp_tri)
            .set_z_index(1)
        )
        amp = Group(amp_box, amp_tri)

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(amp_box, rate_func=rate_functions.ease_out_elastic),
                GrowFromCenter(amp_tri, rate_func=rate_functions.ease_out_elastic),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        A1 = 0.5

        input_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 1.5,
            y_length=amp.height,
        ).next_to(amp, LEFT, 0)
        input_ax.shift(amp.get_left() - input_ax.c2p(1, 0))

        f = 4
        input_plot = input_ax.plot(
            lambda t: A1 * np.sin(2 * PI * f * t),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            x_range=[0, 1, 1 / 200],
            color=INPUT_COLOR,
        )

        output_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 1.5,
            y_length=amp.height,
        )
        output_ax.shift(amp.get_right() - output_ax.c2p(0, 0))

        f = 4
        output_plot = output_ax.plot(
            lambda t: 2 * A1 * np.sin(2 * PI * f * t),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            x_range=[0, 1, 1 / 200],
            color=OUTPUT_COLOR,
        )

        input_arrow = Arrow(
            input_ax.c2p(0, 0),
            input_ax.c2p(1, 0),
        ).next_to(input_ax, UP)
        output_arrow = Arrow(
            output_ax.c2p(0, 0),
            output_ax.c2p(1, 0),
        ).next_to(output_ax, UP)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(input_plot),
                    GrowArrow(input_arrow),
                ),
                AnimationGroup(
                    Create(output_plot),
                    GrowArrow(output_arrow),
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        gain_sym = MathTex(r"G", color=GREEN).scale(2).next_to(amp, DOWN, LARGE_BUFF)
        input_arrow_curve = CurvedArrow(
            input_plot.get_bottom(), gain_sym.get_left() + [-0.1, 0, 0], angle=PI / 4
        )
        output_arrow_curve = CurvedArrow(
            gain_sym.get_right() + [0.1, 0, 0], output_plot.get_bottom(), angle=PI / 4
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(input_arrow, input_arrow_curve),
                GrowFromCenter(gain_sym),
                ReplacementTransform(output_arrow, output_arrow_curve),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        gain_eqn = (
            MathTex(r"G = \frac{V_{\text{out}}}{V_{\text{in}}} = \frac{2}{1}")
            .scale(2)
            .move_to(gain_sym)
            .shift(DOWN / 2)
        )
        gain_eqn[0][0].set_color(GREEN)
        gain_eqn[0][2:6].set_color(OUTPUT_COLOR)
        gain_eqn[0][7:10].set_color(INPUT_COLOR)
        gain_eqn[0][11].set_color(OUTPUT_COLOR)
        gain_eqn[0][13].set_color(INPUT_COLOR)

        self.play(
            LaggedStart(
                FadeOut(input_arrow_curve, output_arrow_curve),
                self.camera.frame.animate.scale(1.2).shift(DOWN),
                ReplacementTransform(gain_sym[0][0], gain_eqn[0][0]),
                FadeIn(gain_eqn[0][1]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ip_ld = Line(
            input_plot.get_corner(DL),
            input_plot.get_corner(DL) + LEFT / 2,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).shift(LEFT * 0.2)
        ip_lu = (
            Line(
                input_plot.get_corner(DL),
                input_plot.get_corner(DL) + LEFT / 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .set_y(input_plot.get_top()[1])
            .shift(LEFT * 0.2)
        )
        ip_line = Line(
            ip_ld.get_midpoint(),
            ip_lu.get_midpoint(),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        vin_label = (
            MathTex(r"V_{\text{in}}")
            .scale(2)
            .set_color(INPUT_COLOR)
            .next_to(ip_ld, DOWN, MED_SMALL_BUFF)
        )

        op_ld = Line(
            output_plot.get_corner(DR),
            output_plot.get_corner(DR) + RIGHT / 2,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).shift(RIGHT * 0.2)
        op_lu = (
            Line(
                output_plot.get_corner(DR),
                output_plot.get_corner(DR) + RIGHT / 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .set_y(output_plot.get_top()[1])
            .shift(RIGHT * 0.2)
        )
        op_line = Line(
            op_ld.get_midpoint(),
            op_lu.get_midpoint(),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        vout_label = (
            MathTex(r"V_{\text{out}}")
            .scale(2)
            .set_color(OUTPUT_COLOR)
            .next_to(op_ld, DOWN, MED_SMALL_BUFF)
        )

        # self.add(gain_eqn[0][1:])
        self.play(
            LaggedStart(
                Create(ip_ld),
                Create(ip_line),
                Create(ip_lu),
                LaggedStart(*[FadeIn(m) for m in vin_label[0]], lag_ratio=0.1),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(op_ld),
                Create(op_line),
                Create(op_lu),
                LaggedStart(*[FadeIn(m) for m in vout_label[0]], lag_ratio=0.1),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(vout_label[0], gain_eqn[0][2:6], path_arc=-PI / 3),
                Create(gain_eqn[0][6]),
                ReplacementTransform(vin_label[0], gain_eqn[0][7:10], path_arc=PI / 3),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in gain_eqn[0][10:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        gain_soln = (
            MathTex(r"G = \frac{V_{\text{out}}}{V_{\text{in}}} = 2 \ [\text{unitless}]")
            .scale(2)
            .move_to(gain_eqn, LEFT)
        )
        gain_soln[0][0].set_color(GREEN)
        gain_soln[0][2:6].set_color(OUTPUT_COLOR)
        gain_soln[0][7:10].set_color(INPUT_COLOR)

        self.play(
            LaggedStart(
                ReplacementTransform(gain_eqn[0][:11], gain_soln[0][:11]),
                ShrinkToCenter(gain_eqn[0][12]),
                ShrinkToCenter(gain_eqn[0][13]),
                ReplacementTransform(gain_eqn[0][11], gain_soln[0][11]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(*[FadeIn(m) for m in gain_soln[0][12:]]))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        vpv = (
            MathTex(r"\left[ \frac{V}{V} \right]")
            .scale(2)
            .move_to(gain_soln[0][12:], LEFT)
        )
        wpw = (
            MathTex(r"\left[ \frac{W}{W} \right]")
            .scale(2)
            .move_to(gain_soln[0][12:], LEFT)
        )

        self.play(
            LaggedStart(
                ReplacementTransform(gain_soln[0][12], vpv[0][0]),
                FadeOut(gain_soln[0][13:-1]),
                GrowFromCenter(vpv[0][1]),
                GrowFromCenter(vpv[0][2]),
                GrowFromCenter(vpv[0][3]),
                ReplacementTransform(gain_soln[0][-1], vpv[0][-1]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ReplacementTransform(a, b) for a, b in zip(vpv[0], wpw[0])],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        xu = Text("x", font=FONT, color=RED).scale(2).move_to(wpw[0][1])
        xd = Text("x", font=FONT, color=RED).scale(2).move_to(wpw[0][3])

        self.play(
            LaggedStart(
                GrowFromCenter(xu),
                GrowFromCenter(xd),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        np.random.seed(1)

        self.play(
            LaggedStart(
                *[
                    FadeOut(m, shift=shift)
                    for m, shift in zip(
                        [*wpw[0], xu, xd],
                        np.random.random((7, 3)) * 2 - 1,
                    )
                ],
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)
        self.camera.frame.save_state()

        desired_page = 3
        npages = desired_page * 2
        theta_step = PI * 0.03
        pages = (
            Group(
                *[
                    ImageMobject(f"./static/adl8154-{npages + 1 - idx:02}.png")
                    .scale_to_fit_height(fh(self, 0.8))
                    .rotate(-theta_step * npages / 2 + idx * theta_step)
                    for idx in range(1, npages + 1)
                ]
            )
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self, 1.5))
        )

        self.add(pages)
        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 1.5)))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        width = pages.width
        self.play(
            pages[-1]
            .animate(rate_func=rate_functions.ease_in_circ, run_time=0.3)
            .shift(LEFT * width),
        )
        pages[-1].set_z_index(-npages)
        self.play(
            pages[-1]
            .animate(rate_func=rate_functions.ease_out_circ, run_time=0.3)
            .shift(RIGHT * width),
        )
        self.play(
            pages[-2]
            .animate(rate_func=rate_functions.ease_in_circ, run_time=0.3)
            .shift(LEFT * width),
        )
        pages[-2].set_z_index(-npages)
        self.play(
            pages[-2]
            .animate(rate_func=rate_functions.ease_out_circ, run_time=0.3)
            .shift(RIGHT * width),
        )
        self.play(
            pages[-3]
            .animate(rate_func=rate_functions.ease_in_circ, run_time=0.3)
            .shift(LEFT * width),
        )
        pages[-3].set_z_index(-npages)
        self.play(
            LaggedStart(
                pages[-3]
                .animate(rate_func=rate_functions.ease_out_circ, run_time=0.3)
                .shift(RIGHT * width),
                self.camera.frame.animate.scale(0.35).shift(UP * 2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        highlight = (
            Rectangle(
                width=fw(self, 0.5),
                height=fh(self, 0.03),
                color=YELLOW,
                stroke_opacity=0,
                fill_color=YELLOW,
                fill_opacity=0.3,
            )
            .move_to(self.camera.frame)
            .shift(LEFT + UP * 0.14)
        )

        self.play(FadeIn(highlight))

        self.wait(0.5)

        self.remove(ip_ld, ip_line, ip_lu, op_ld, op_line, op_lu)
        self.play(self.camera.frame.animate.restore())

        self.wait(2)
