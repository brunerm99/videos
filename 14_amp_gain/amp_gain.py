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
from props import (
    VideoMobject,
    WeatherRadarTower,
    get_blocks,
    get_filt_block,
    get_phase_shifter,
    get_splitter,
)
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


class IntroV2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        you = ImageMobject(
            # "../../../media/rf-channel-assets/VintageRaccoonStanding_NoBackground.png"
            "../../../Downloads/Chill Raccoon Standing.png"
        ).scale_to_fit_height(fh(self, 0.5))
        you_label = Text("You", font=FONT).next_to(you, DOWN, MED_SMALL_BUFF)
        an_rf = (
            Text("(an RF engineer)", font=FONT)
            .scale(0.6)
            .next_to(you_label, DOWN, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                FadeIn(you),
                Write(you_label),
                Write(an_rf),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        datasheet_scrolling = (
            VideoMobject("./static/datasheet-scrolling.mkv", speed=2)
            .next_to(self.camera.frame.get_left(), LEFT)
            .scale_to_fit_width(fw(self, 0.6))
        )
        self.add(datasheet_scrolling)

        self.wait(2)

        self.play(
            Group(datasheet_scrolling, Group(you, you_label, an_rf)).animate.arrange(
                RIGHT
            )
        )

        self.wait(8)

        op1db = Text("OP1dB", font=FONT)
        oip3 = Text("OIP3", font=FONT)
        pn = Text("Phase Noise", font=FONT)
        gain = Text("Gain", font=FONT)
        etc = Text("etc.", font=FONT)
        Group(op1db, oip3, pn, gain, etc).arrange(RIGHT, SMALL_BUFF).move_to(
            self.camera.frame
        ).shift(DOWN * fh(self))
        oip3.shift(DOWN)
        gain.shift(DOWN)

        bez1 = CubicBezier(
            datasheet_scrolling.get_bottom() + [0, -0.1, 0],
            datasheet_scrolling.get_bottom() + [0, -1, 0],
            op1db.get_top() + [0, 2, 0],
            op1db.get_top() + [0, 0.1, 0],
        )
        bez2 = CubicBezier(
            datasheet_scrolling.get_bottom() + [0, -0.1, 0],
            datasheet_scrolling.get_bottom() + [0, -1, 0],
            oip3.get_top() + [0, 2, 0],
            oip3.get_top() + [0, 0.1, 0],
        )
        bez3 = CubicBezier(
            datasheet_scrolling.get_bottom() + [0, -0.1, 0],
            datasheet_scrolling.get_bottom() + [0, -1, 0],
            pn.get_top() + [0, 2, 0],
            pn.get_top() + [0, 0.1, 0],
        )
        bez4 = CubicBezier(
            datasheet_scrolling.get_bottom() + [0, -0.1, 0],
            datasheet_scrolling.get_bottom() + [0, -1, 0],
            gain.get_top() + [0, 2, 0],
            gain.get_top() + [0, 0.1, 0],
        )
        bez5 = CubicBezier(
            datasheet_scrolling.get_bottom() + [0, -0.1, 0],
            datasheet_scrolling.get_bottom() + [0, -1, 0],
            etc.get_top() + [0, 2, 0],
            etc.get_top() + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            self.camera.frame.animate(run_time=2).shift(DOWN * fh(self)),
            LaggedStart(
                Create(bez1),
                Create(bez2),
                Create(bez3),
                Create(bez4),
                Create(bez5),
                Write(op1db),
                Write(oip3),
                Write(pn),
                Write(gain),
                Write(etc),
                lag_ratio=0.3,
            ),
        )

        self.remove(datasheet_scrolling)

        self.wait(0.5)

        lna_amp_tri = (
            Triangle(stroke_width=DEFAULT_STROKE_WIDTH * 2, color=GREEN)
            .rotate(PI / 6)
            .set_z_index(1)
        )
        lna_amp_box = (
            RoundedRectangle(
                width=lna_amp_tri.width * 2,
                height=lna_amp_tri.width * 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(lna_amp_tri)
            .set_z_index(1)
        )
        lna = Group(lna_amp_box, lna_amp_tri)

        bp_filt = get_filt_block(width=lna.height, passband="band")

        driver_amp_tri = (
            Triangle(stroke_width=DEFAULT_STROKE_WIDTH * 2, color=GREEN)
            .rotate(PI / 6)
            .set_z_index(1)
        )
        driver_amp_box = (
            RoundedRectangle(
                width=driver_amp_tri.width * 2,
                height=driver_amp_tri.width * 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(driver_amp_tri)
            .set_z_index(1)
        )
        driver = Group(driver_amp_box, driver_amp_tri)
        to_lna = Line(LEFT, RIGHT)
        lna_to_filt = Line(LEFT, RIGHT)
        bp_filt_to_driver = Line(LEFT, RIGHT)
        from_driver = Line(LEFT, RIGHT)

        bd_group = (
            Group(
                to_lna,
                lna,
                lna_to_filt,
                bp_filt,
                bp_filt_to_driver,
                driver,
                from_driver,
            )
            .arrange(RIGHT, 0)
            .scale_to_fit_width(fw(self))
            .move_to(self.camera.frame.get_bottom())
        )

        self.play(
            LaggedStart(
                Create(bd_group[0]),
                GrowFromCenter(bd_group[1]),
                Create(bd_group[2]),
                GrowFromCenter(bd_group[3]),
                Create(bd_group[4]),
                GrowFromCenter(bd_group[5]),
                Create(bd_group[6]),
                lag_ratio=0.2,
            ),
            self.camera.frame.animate.scale_to_fit_width(bd_group.width * 1.1).move_to(
                Group(bd_group, op1db, oip3, pn, gain)
            ),
            run_time=2.5,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                lna[0]
                .animate(rate_func=rate_functions.there_and_back)
                .set_fill(opacity=0.5, color=YELLOW),
                driver[0]
                .animate(rate_func=rate_functions.there_and_back)
                .set_fill(opacity=0.5, color=YELLOW),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        ps = get_phase_shifter(width=lna.width).move_to(bp_filt)
        filt_to_ps = Line(LEFT, RIGHT).next_to(ps, LEFT, 0)
        bd_group.add(ps, filt_to_ps)

        new_bd_group_copy = Group(
            to_lna.copy(), lna.copy(), lna_to_filt.copy(), bp_filt.copy()
        ).shift(LEFT * (filt_to_ps.width + ps.width))

        self.play(
            LaggedStart(
                Group(to_lna, lna, lna_to_filt, bp_filt).animate.shift(
                    LEFT * (filt_to_ps.width + ps.width)
                ),
                self.camera.frame.animate.scale_to_fit_width(
                    Group(bd_group, new_bd_group_copy).width * 1.1
                ).move_to(Group(bd_group, new_bd_group_copy)),
                GrowFromCenter(ps),
                Create(filt_to_ps),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        splitter = (
            get_splitter(width=lna.width, n=4)
            .next_to(from_driver, RIGHT, 0)
            .shift(DOWN * lna.height * 1.5)
        )
        bd_group.add(splitter)

        p1 = splitter.get_corner(UL) + DOWN * splitter.height / 8
        splitter_p1_bez = CubicBezier(
            from_driver.get_start(),
            from_driver.get_start() + [1, 0, 0],
            p1 + [-1, 0, 0],
            p1,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    bd_group.width * 1.1
                ).move_to(bd_group),
                FadeIn(splitter[0]),
                ReplacementTransform(from_driver, splitter_p1_bez),
                LaggedStart(
                    *[Create(m) for m in splitter[1:]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            bp_filt[0]
            .animate(rate_func=rate_functions.there_and_back)
            .set_fill(opacity=0.5, color=YELLOW),
        )

        self.wait(0.5)

        amp_specs = ["OP1dB", "Gain", "Noise Figure", "OIP3", "..."]
        filt_specs = ["Passband", "Stopband", "Ripple", "Insertion loss", "..."]
        splitter_specs = ["Insertion loss", "Isolation", "Phase unbalance", "..."]
        ps_specs = ["Phase error", "Settle time", "Insertion loss", "..."]

        lna_specs = (
            Text("\n".join(amp_specs), font=FONT)
            .scale(0.7)
            .next_to(lna, DOWN, MED_SMALL_BUFF, LEFT)
        )
        driver_specs = (
            Text("\n".join(amp_specs), font=FONT)
            .scale(0.7)
            .next_to(driver, DOWN, MED_SMALL_BUFF, LEFT)
        )
        filt_specs = (
            Text("\n".join(filt_specs), font=FONT)
            .scale(0.7)
            .next_to(bp_filt, DOWN, MED_SMALL_BUFF, LEFT)
        )
        ps_specs = (
            Text("\n".join(ps_specs), font=FONT)
            .scale(0.7)
            .next_to(ps, DOWN, MED_SMALL_BUFF, LEFT)
        )
        splitter_specs = (
            Text("\n".join(splitter_specs), font=FONT)
            .scale(0.7)
            .next_to(splitter, DOWN, MED_SMALL_BUFF, LEFT)
        )

        self.next_section(skip_animations=skip_animations(True))

        # self.add(lna_specs, driver_specs, splitter_specs, filt_specs, ps_specs)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).shift(DOWN + RIGHT * 0.5),
                Write(lna_specs),
                Write(filt_specs),
                Write(ps_specs),
                Write(driver_specs),
                Write(splitter_specs),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        episode_1 = Text("Episode 1:", font=FONT).scale(0.8)
        amp_gain = Text("RF Amplifier Gain", font=FONT)
        title_group = (
            Group(episode_1, amp_gain)
            .arrange(DOWN, MED_LARGE_BUFF)
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
        )
        amp_gain[-4:].set_color(GREEN)

        self.next_section(skip_animations=skip_animations(False))

        # self.add(title_group)
        self.play(
            LaggedStart(
                LaggedStart(
                    TransformFromCopy(
                        lna_specs[5:9], amp_gain[-4:].copy(), path_arc=PI / 3
                    ),
                    TransformFromCopy(
                        driver_specs[5:9], amp_gain[-4:].copy(), path_arc=PI / 3
                    ),
                    # .copy()
                    # .animate(run_time=3)
                    # .set_color(GREEN)
                    # .move_to(amp_gain[-4:]),
                    self.camera.frame.animate.scale_to_fit_width(
                        title_group.width * 2
                    ).move_to(title_group),
                    lag_ratio=0.05,
                    run_time=3,
                ),
                Write(episode_1),
                Write(amp_gain[:-4]),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


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
        # self.play(self.camera.frame.animate.restore())
        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 3)))

        self.wait(2)


class DB(MovingCameraScene):
    def construct(self):
        db_label = Text("dB", font=FONT).scale(2)
        self.play(Write(db_label))

        self.wait(0.5)

        of_vpv = (
            MathTex(r"\left( \frac{V}{V} \right)")
            .scale_to_fit_height(db_label.height * 2)
            .next_to(self.camera.frame.get_right(), RIGHT)
        )

        self.play(Group(db_label, of_vpv).animate.arrange(RIGHT, MED_SMALL_BUFF))

        self.wait(0.5)

        of_wpw = (
            MathTex(r"\left( \frac{W}{W} \right)")
            .scale_to_fit_height(db_label.height * 2)
            .move_to(of_vpv, LEFT)
        )

        self.play(
            LaggedStart(
                *[ReplacementTransform(a, b) for a, b in zip(of_vpv[0], of_wpw[0])],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        add = (
            Text("+", font=FONT, color=GREEN)
            .scale_to_fit_height(db_label.height)
            .next_to(db_label, DOWN, LARGE_BUFF * 2)
            .shift(LEFT)
        )
        mult = (
            Text("x", font=FONT, color=RED)
            .scale_to_fit_height(db_label.height)
            .next_to(of_wpw, DOWN, LARGE_BUFF * 2)
            .shift(RIGHT)
            .set_y(add.get_y())
        )
        add_bez = CubicBezier(
            db_label.get_bottom() + [0, -0.1, 0],
            db_label.get_bottom() + [0, -1, 0],
            add.get_top() + [0, 1, 0],
            add.get_top() + [0, 0.1, 0],
        )
        mult_bez = CubicBezier(
            of_wpw.get_bottom() + [0, -0.1, 0],
            of_wpw.get_bottom() + [0, -1, 0],
            mult.get_top() + [0, 1, 0],
            mult.get_top() + [0, 0.1, 0],
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                Create(mult_bez),
                GrowFromCenter(mult),
                Create(add_bez),
                GrowFromCenter(add),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        osc = BLOCKS.get("oscillator").copy()
        lp_filter = BLOCKS.get("lp_filter").copy()
        amp = BLOCKS.get("amp").copy()
        bp_filter = BLOCKS.get("filter").copy()

        cascade_bd = (
            Group(
                osc,
                Line(ORIGIN, RIGHT),
                lp_filter,
                Line(ORIGIN, RIGHT),
                amp,
                Line(ORIGIN, RIGHT),
                bp_filter,
            )
            .arrange(RIGHT, 0)
            .scale_to_fit_width(fw(self, 0.8))
            .next_to(self.camera.frame.get_bottom(), DOWN)
        )

        bd_group = Group(db_label, cascade_bd)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    bd_group.height * 1.3
                ).move_to(bd_group),
                *[
                    GrowFromCenter(m) if idx % 2 == 0 else Create(m)
                    for idx, m in enumerate(cascade_bd)
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(*cascade_bd),
                self.camera.frame.animate.restore(),
                AnimationGroup(
                    ShrinkToCenter(mult),
                    ShrinkToCenter(add),
                ),
                AnimationGroup(
                    Uncreate(mult_bez),
                    Uncreate(add_bez),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        vpv_to_log = MathTex(
            r"20 \cdot \log_{10}{\left( \frac{V}{V} \right)}"
        ).scale_to_fit_height(db_label.height * 2)
        wpw_to_log = MathTex(
            r"10 \cdot \log_{10}{\left( \frac{W}{W} \right)}"
        ).scale_to_fit_height(db_label.height * 2)

        xpx_group = Group(vpv_to_log, wpw_to_log).arrange(DOWN, LARGE_BUFF)

        db_group = Group(db_label.copy(), xpx_group).arrange(RIGHT, LARGE_BUFF * 3)

        vpv_bez = CubicBezier(
            db_group[0].get_right() + [0.1, 0, 0],
            db_group[0].get_right() + [1, 0, 0],
            vpv_to_log.get_left() + [-1, 0, 0],
            vpv_to_log.get_left() + [-0.1, 0, 0],
        )
        wpw_bez = CubicBezier(
            db_group[0].get_right() + [0.1, 0, 0],
            db_group[0].get_right() + [1, 0, 0],
            wpw_to_log.get_left() + [-1, 0, 0],
            wpw_to_log.get_left() + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                db_label.animate.move_to(db_group[0]),
                ReplacementTransform(of_wpw[0], wpw_to_log[0][8:]),
                Create(wpw_bez),
                LaggedStart(*[FadeIn(m) for m in wpw_to_log[0][:8]]),
                Create(vpv_bez),
                LaggedStart(*[FadeIn(m) for m in vpv_to_log[0]]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        amp_gain = (
            MathTex(r"G_{\text{dB}} = 14 \text{ dB}")
            .scale(2)
            .next_to(wpw_to_log, DOWN, LARGE_BUFF)
        )
        amp_gain[0][:3].set_color(GREEN)

        all_group = Group(amp_gain, wpw_to_log, db_label, vpv_to_log)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.3
                ).move_to(all_group),
                *[FadeIn(m) for m in amp_gain[0]],
                lag_ratio=0.1,
            ),
            run_time=2,
        )

        self.wait(2)


class DB2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        db_label = Text("dB", font=FONT).scale(2)
        self.play(Write(db_label))

        self.wait(0.5)

        lna_amp_tri = (
            Triangle(stroke_width=DEFAULT_STROKE_WIDTH * 2, color=GREEN)
            .rotate(PI / 6)
            .set_z_index(1)
        )
        lna_amp_box = (
            RoundedRectangle(
                width=lna_amp_tri.width * 2,
                height=lna_amp_tri.width * 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(lna_amp_tri)
            .set_z_index(1)
        )
        lna = Group(lna_amp_box, lna_amp_tri)

        bp_filt = get_filt_block(width=lna.height, passband="band")

        driver_amp_tri = (
            Triangle(stroke_width=DEFAULT_STROKE_WIDTH * 2, color=GREEN)
            .rotate(PI / 6)
            .set_z_index(1)
        )
        driver_amp_box = (
            RoundedRectangle(
                width=driver_amp_tri.width * 2,
                height=driver_amp_tri.width * 2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(driver_amp_tri)
            .set_z_index(1)
        )
        driver = Group(driver_amp_box, driver_amp_tri)
        to_lna = Line(LEFT, RIGHT)
        lna_to_filt = Line(LEFT, RIGHT)
        bp_filt_to_driver = Line(LEFT, RIGHT)
        from_driver = Line(LEFT, RIGHT)

        bd_group = (
            Group(
                to_lna,
                lna,
                lna_to_filt,
                bp_filt,
                bp_filt_to_driver,
                driver,
                from_driver,
            )
            .arrange(RIGHT, 0)
            .scale_to_fit_width(fw(self))
            .next_to(db_label, DOWN, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                FadeOut(db_label),
                Create(bd_group[0]),
                GrowFromCenter(bd_group[1]),
                Create(bd_group[2]),
                GrowFromCenter(bd_group[3]),
                Create(bd_group[4]),
                GrowFromCenter(bd_group[5]),
                Create(bd_group[6]),
                lag_ratio=0.2,
            ),
            self.camera.frame.animate.scale_to_fit_width(bd_group.width * 1.1).move_to(
                bd_group
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                bd_group[2].animate.set_opacity(0.1),
                bd_group[3].animate.set_opacity(0.1),
                bd_group[4].animate.set_opacity(0.1),
                AnimationGroup(
                    bd_group[5][0].animate.set_stroke(opacity=0.1),
                    bd_group[5][1].animate.set_stroke(opacity=0.1),
                ),
                bd_group[6].animate.set_opacity(0.1),
                lag_ratio=0.1,
            ),
            run_time=1,
        )

        self.wait(0.5)

        lna_gain = (
            MathTex(r"G = 17\times").scale(1.8).next_to(lna, DOWN, MED_SMALL_BUFF)
        )
        lna_gain[0][0].set_color(GREEN)
        filt_gain = (
            MathTex(r"G = 0.63\times").scale(1.8).next_to(bp_filt, DOWN, MED_SMALL_BUFF)
        )
        filt_gain[0][0].set_color(RED)
        driver_gain = (
            MathTex(r"G = 8.2\times").scale(1.8).next_to(driver, DOWN, MED_SMALL_BUFF)
        )
        driver_gain[0][0].set_color(GREEN)

        self.play(FadeIn(lna_gain))

        self.wait(0.5)

        self.play(
            LaggedStart(
                bd_group[0].animate.set_opacity(0.1),
                AnimationGroup(
                    bd_group[1][0].animate.set_stroke(opacity=0.1),
                    bd_group[1][1].animate.set_stroke(opacity=0.1),
                    lna_gain.animate.set_opacity(0.1),
                ),
                bd_group[2].animate.set_opacity(1),
                bd_group[3].animate.set_opacity(1),
                FadeIn(filt_gain),
                lag_ratio=0.15,
            )
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                bd_group[2].animate.set_opacity(0.1),
                bd_group[3].animate.set_opacity(0.1),
                filt_gain.animate.set_opacity(0.1),
                bd_group[4].animate.set_opacity(1),
                AnimationGroup(
                    bd_group[5][0].animate.set_stroke(opacity=1),
                    bd_group[5][1].animate.set_stroke(opacity=1),
                ),
                FadeIn(driver_gain),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                bd_group[0].animate.set_opacity(1),
                AnimationGroup(
                    bd_group[1][0].animate.set_stroke(opacity=1),
                    bd_group[1][1].animate.set_stroke(opacity=1),
                ),
                lna_gain.animate.set_opacity(1).scale(0.8),
                bd_group[2].animate.set_opacity(1),
                bd_group[3].animate.set_opacity(1),
                filt_gain.animate.set_opacity(1).scale(0.8),
                driver_gain.animate.scale(0.8),
                bd_group[6].animate.set_opacity(1),
            ),
            run_time=2,
        )

        self.wait(0.5)

        g_tot = (
            MathTex(r"G_{\text{total}} = 17 \times 0.63 \times 8.2")
            .scale(2)
            .next_to(filt_gain, DOWN, LARGE_BUFF)
        )
        g_tot[0][0].set_color(GREEN)
        g_tot_equal = (
            MathTex(r"G_{\text{total}} = 17 \times 0.63 \times 8.2 = 87.7")
            .scale(2)
            .move_to(g_tot)
        )
        g_tot_equal[0][0].set_color(GREEN)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                FadeIn(g_tot[0][0]),
                FadeIn(g_tot[0][1:6]),
                FadeIn(g_tot[0][6]),
                TransformFromCopy(lna_gain[0][2:-1], g_tot[0][7:9]),
                FadeIn(g_tot[0][9]),
                TransformFromCopy(filt_gain[0][2:-1], g_tot[0][10:14]),
                FadeIn(g_tot[0][14]),
                TransformFromCopy(driver_gain[0][2:-1], g_tot[0][15:18]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(g_tot_equal.width * 1.2),
                ReplacementTransform(g_tot[0], g_tot_equal[0][:-5]),
                LaggedStart(*[FadeIn(m) for m in g_tot_equal[0][-5:]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        lna_db = (
            MathTex(r"G_{\text{dB}} = 24.6 \text{ dB}")
            .scale(1.4)
            .next_to(lna_gain, DOWN, LARGE_BUFF)
            .shift(LEFT * 2)
        )
        lna_db[0][:3].set_color(GREEN)
        lna_line = CubicBezier(
            lna_gain.get_bottom() + [0, -0.1, 0],
            lna_gain.get_bottom() + [0, -1, 0],
            lna_db.get_top() + [0, 1, 0],
            lna_db.get_top() + [0, 0.1, 0],
        )

        filt_db = (
            MathTex(r"G_{\text{dB}} = -4.0 \text{ dB}")
            .scale(1.4)
            .next_to(filt_gain, DOWN, LARGE_BUFF)
        )
        filt_db[0][:3].set_color(RED)
        filt_line = CubicBezier(
            filt_gain.get_bottom() + [0, -0.1, 0],
            filt_gain.get_bottom() + [0, -1, 0],
            filt_db.get_top() + [0, 1, 0],
            filt_db.get_top() + [0, 0.1, 0],
        )

        driver_db = (
            MathTex(r"G_{\text{dB}} = 18.3 \text{ dB}")
            .scale(1.4)
            .next_to(driver_gain, DOWN, LARGE_BUFF)
            .shift(RIGHT * 2)
        )
        driver_db[0][:3].set_color(GREEN)
        driver_line = CubicBezier(
            driver_gain.get_bottom() + [0, -0.1, 0],
            driver_gain.get_bottom() + [0, -1, 0],
            driver_db.get_top() + [0, 1, 0],
            driver_db.get_top() + [0, 0.1, 0],
        )

        self.play(
            g_tot_equal[0].animate.set_opacity(0.1).shift(DOWN * 3.5),
            self.camera.frame.animate.scale(1.2).shift(DOWN * 2),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(lna_line),
                LaggedStart(*[FadeIn(m) for m in lna_db[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(filt_line),
                LaggedStart(*[FadeIn(m) for m in filt_db[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(driver_line),
                LaggedStart(
                    *[FadeIn(m) for m in driver_db[0]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

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
        # bp_filt = get_filt_block(width=fh(self, 0.3), passband="band")
        # lp_filt = get_filt_block(width=fh(self, 0.3), passband="low")
        # hp_filt = get_filt_block(width=fh(self, 0.3), passband="high")
        # self.add(Group(lp_filt, bp_filt, hp_filt).arrange(RIGHT))
        # self.add(get_phase_shifter(width=fh(self, 0.3)))
        self.add(get_splitter(width=fh(self, 0.3), n=4))

        self.wait(2)
