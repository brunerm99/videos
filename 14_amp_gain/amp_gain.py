# amp_gain.py

import os
import sys
from random import shuffle

import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from MF_Tools import VT
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import VideoMobject, WeatherRadarTower, get_blocks
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


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        npages = 23
        theta_step = PI * 0.005
        pages = Group(
            *[
                ImageMobject(
                    f"./static/adl8154-{npages + 1 - idx:02}.png"
                ).scale_to_fit_height(fh(self, 0.8))
                # .rotate(-theta_step * npages / 2 + idx * theta_step)
                for idx in range(1, npages + 1)
            ]
        ).move_to(self.camera.frame)

        pages_new = (
            pages.copy()[::-1]
            .arrange_in_grid(rows=3, cols=8, buff=MED_SMALL_BUFF)
            .scale_to_fit_width(fw(self, 0.9))
        )

        self.play(GrowFromCenter(pages))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[Transform(a, b) for a, b in zip(pages, pages_new[::-1])],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        pages_list = list(pages)

        shuffle(pages_list)

        self.play(
            LaggedStart(
                *[
                    Succession(
                        ShrinkToCenter(m),
                        FadeOut(m, run_time=0.1),
                    )
                    for m in pages_list
                ],
                lag_ratio=0.1,
            ),
            run_time=1.5,
        )
        gain_label = Text("Gain", font=FONT, color=GREEN).scale_to_fit_width(
            fw(self, 0.3)
        )
        self.play(Write(gain_label))

        self.wait(0.5)

        spec1 = Text("1 dB Compression", font=FONT).scale(0.7)
        spec2 = Text("Saturated output power", font=FONT).scale(0.7)
        spec3 = Text("Power added efficiency", font=FONT).scale(0.7)
        spec4 = Text("Return loss", font=FONT).scale(0.7)
        spec5 = Text("...", font=FONT).scale(0.7)
        Group(spec1, spec2, spec3, spec4, spec5).arrange(
            DOWN, MED_SMALL_BUFF, aligned_edge=LEFT
        ).next_to(gain_label, RIGHT, LARGE_BUFF * 2)
        unit1 = Tex(r"| $P_{1dB}$").next_to(spec1, RIGHT, SMALL_BUFF)
        unit2 = Tex(r"| $P_{\text{sat}}$").next_to(spec2, RIGHT, SMALL_BUFF)
        unit3 = Tex(r"| PAE").next_to(spec3, RIGHT, SMALL_BUFF)
        unit4 = Tex(r"| $S_{11}$, $S_{22}$").next_to(spec4, RIGHT, SMALL_BUFF)

        bez1 = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            spec1.get_left() + [-1, 0, 0],
            spec1.get_left() + [-0.1, 0, 0],
        )
        bez2 = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            spec2.get_left() + [-1, 0, 0],
            spec2.get_left() + [-0.1, 0, 0],
        )
        bez3 = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            spec3.get_left() + [-1, 0, 0],
            spec3.get_left() + [-0.1, 0, 0],
        )
        bez4 = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            spec4.get_left() + [-1, 0, 0],
            spec4.get_left() + [-0.1, 0, 0],
        )
        bez5 = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            spec5.get_left() + [-1, 0, 0],
            spec5.get_left() + [-0.1, 0, 0],
        )

        all_group = Group(gain_label, spec1, unit1, spec2, unit2, unit3, unit4, spec5)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.1
                ).move_to(all_group),
                Create(bez1),
                Write(spec1),
                LaggedStart(*[FadeIn(m) for m in unit1[0]], lag_ratio=0.2),
                Create(bez2),
                Write(spec2),
                LaggedStart(*[FadeIn(m) for m in unit2[0]], lag_ratio=0.2),
                Create(bez3),
                Write(spec3),
                LaggedStart(*[FadeIn(m) for m in unit3[0]], lag_ratio=0.2),
                Create(bez4),
                Write(spec4),
                LaggedStart(*[FadeIn(m) for m in unit4[0]], lag_ratio=0.2),
                Create(bez5),
                Write(spec5),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)


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


class FrequencyDependence(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        page = ImageMobject(f"./static/adl8154-07.png").scale_to_fit_height(
            fh(self, 0.8)
        )
        self.add(page)
        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(page.height * 0.3).shift(
                RIGHT * 1.12 + UP * 0.215
            ),
            run_time=2,
        )

        self.wait(0.5)

        amp = rf.Network("../../notebooks/data/ADL8154ACPZN_SParameters_25C.S2p")
        bw_mask = (amp.f > 300e6) & (amp.f < 7e9)
        s21 = interp1d(
            amp.f[bw_mask] / 1e9,
            amp.s_db[bw_mask][:, 1, 0],
            fill_value="extrapolate",
        )

        ax_x = VT(self.camera.frame.get_center()[0])
        ax_y = VT(self.camera.frame.get_center()[1])
        x_length = VT(fw(self, 0.42))
        y_length = VT(fh(self, 0.595))

        x_ticks = np.arange(1, 7 + 1, 1).astype(int)
        y_ticks = np.arange(4, 20 + 4, 4).astype(int)

        def get_s21_ax():
            s21_ax = (
                Axes(
                    x_range=[0, 7, 0.5],
                    y_range=[0, 20, 2],
                    tips=False,
                    x_length=~x_length,
                    y_length=~y_length,
                    x_axis_config=dict(
                        numbers_with_elongated_ticks=x_ticks,
                        include_numbers=True,
                        numbers_to_include=x_ticks,
                        font_size=DEFAULT_FONT_SIZE * 0.15,
                        label_constructor=lambda x: Text(x, font=FONT),
                        line_to_number_buff=SMALL_BUFF,
                        decimal_number_config=dict(num_decimal_places=0),
                        longer_tick_multiple=2,
                    ),
                    y_axis_config=dict(
                        numbers_with_elongated_ticks=y_ticks,
                        include_numbers=True,
                        numbers_to_include=y_ticks,
                        font_size=DEFAULT_FONT_SIZE * 0.15,
                        label_constructor=lambda x: Text(x, font=FONT),
                        line_to_number_buff=SMALL_BUFF,
                        decimal_number_config=dict(num_decimal_places=0),
                        longer_tick_multiple=2,
                    ),
                    axis_config=dict(
                        stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
                        tick_size=0.025,
                        # stroke_color=BLACK,
                    ),
                )
                .move_to([~ax_x, ~ax_y, 0])
                .set_z_index(-2)
            )
            return s21_ax

        def get_s21_plot():
            s21_ax = get_s21_ax()
            s21_plot = s21_ax.plot(
                s21,
                x_range=[0.3, 7, 1 / 200],
                stroke_width=DEFAULT_STROKE_WIDTH * 0.6,
                color=ORANGE,
                use_smoothing=False,
            ).set_z_index(1)
            return s21_plot

        ax = always_redraw(get_s21_ax)
        plot = always_redraw(get_s21_plot)

        f1_line = Line(
            ax.c2p(0.1, 0) + DOWN * 0.05,
            ax.c2p(0.1, 0) + UP * 0.05,
            color=BLACK,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.4,
        ).set_z_index(1)
        f2_line = Line(
            ax.c2p(6, 0) + DOWN * 0.05,
            ax.c2p(6, 0) + UP * 0.05,
            color=BLACK,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.4,
        ).set_z_index(1)
        f_line = Line(
            f1_line.get_midpoint(),
            f2_line.get_midpoint(),
            stroke_width=DEFAULT_STROKE_WIDTH * 0.4,
            color=BLACK,
        ).set_z_index(1)
        f_label = (
            MathTex(r"10 \text{ MHz} \le f \le 6 \text{ GHz}", color=BLACK)
            .scale_to_fit_width(f_line.width * 0.8)
            .next_to(f_line, UP, SMALL_BUFF)
        ).set_z_index(2)
        f_box = SurroundingRectangle(
            f_label,
            color=BLACK,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.3,
            buff=SMALL_BUFF * 0.5,
            corner_radius=0.05,
            fill_color=BLUE,
            fill_opacity=1,
        ).set_z_index(1)

        self.play(
            LaggedStart(
                Create(f1_line),
                Create(f_line),
                Create(f2_line),
                lag_ratio=0.3,
            ),
            LaggedStart(
                FadeIn(f_box),
                FadeIn(f_label),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        temp_box = Polygon(
            ax.c2p(0.1, 8),
            ax.c2p(0.1, 13.7),
            ax.c2p(1.85, 13.7),
            ax.c2p(1.85, 8),
            color=BLACK,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.3,
        ).set_z_index(1)

        self.play(Create(temp_box))

        self.wait(0.5)

        volt_box = temp_box.copy().shift(DOWN * 1.7)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * 1.5),
                Create(volt_box),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        page2 = (
            ImageMobject(f"./static/adl8154-08.png")
            .scale_to_fit_height(page.height)
            .next_to(page, DOWN, SMALL_BUFF * 0.5)
        )
        self.add(page2)

        curr_box = volt_box.copy().stretch(1.2, dim=1).shift(DOWN * 3.12)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * 3.3),
                Create(curr_box),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())
        self.remove(page2, volt_box, curr_box)

        self.wait(0.5)

        self.add(ax)

        self.play(Create(plot), run_time=2)

        self.wait(0.5)

        gain_label = (
            Text("Gain (dB)", font=FONT)
            .scale(0.2)
            .rotate(PI / 2)
            .next_to(ax, LEFT, SMALL_BUFF)
        )
        freq_label = (
            Text("Frequency (GHz)", font=FONT).scale(0.2).next_to(ax, DOWN, SMALL_BUFF)
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(temp_box),
                    Uncreate(f1_line),
                    Uncreate(f_line),
                    Uncreate(f2_line),
                    FadeOut(f_label, f_box),
                ),
                FadeOut(page),
                Create(ax),
                Write(freq_label),
                Write(gain_label),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale(1.2).shift(DOWN * 0.1),
            AnimationGroup(
                freq_label.animate.shift(DOWN * 0.07),
                gain_label.animate.shift(LEFT * 0.5),
            ),
            x_length @ fw(self, 0.7),
            run_time=3,
        )

        self.wait(2)


class BD(MovingCameraScene):
    def construct(self):
        self.wait(2)
