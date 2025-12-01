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
    get_amp,
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
GAIN_COLOR = GREEN


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
            MathTex(r"G = \frac{\text{out}}{\text{in}}")
            .scale(2)
            .move_to(gain_sym)
            .shift(DOWN / 2)
        )
        gain_eqn[0][0].set_color(GREEN)
        gain_eqn[0][2:5].set_color(OUTPUT_COLOR)
        gain_eqn[0][6:].set_color(INPUT_COLOR)

        gain_eqn_pv = (
            MathTex(
                r"G = \frac{\text{out}}{\text{in}} \Rightarrow \frac{V_{\text{out}}}{V_{\text{in}}} \text{ or } \frac{P_{\text{out}}}{P_{\text{in}}}"
            )
            .scale(2)
            .move_to(gain_eqn)
        )
        gain_eqn_pv[0][0].set_color(GREEN)
        gain_eqn_pv[0][2:5].set_color(OUTPUT_COLOR)
        gain_eqn_pv[0][6:8].set_color(INPUT_COLOR)
        gain_eqn_pv[0][9:13].set_color(OUTPUT_COLOR)
        gain_eqn_pv[0][19:23].set_color(OUTPUT_COLOR)
        gain_eqn_pv[0][14:17].set_color(INPUT_COLOR)
        gain_eqn_pv[0][24:].set_color(INPUT_COLOR)

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
            MathTex(r"\text{in}")
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
            MathTex(r"\text{out}")
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
            ),
            LaggedStart(
                Create(op_ld),
                Create(op_line),
                Create(op_lu),
                LaggedStart(*[FadeIn(m) for m in vout_label[0]], lag_ratio=0.1),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(vout_label[0], gain_eqn[0][2:5], path_arc=-PI / 3),
                Create(gain_eqn[0][5]),
                ReplacementTransform(vin_label[0], gain_eqn[0][6:], path_arc=PI / 3),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(gain_eqn[0][:8], gain_eqn_pv[0][:8]),
                GrowFromCenter(gain_eqn_pv[0][8]),
                LaggedStart(
                    *[GrowFromCenter(m) for m in gain_eqn_pv[0][9:17]],
                    lag_ratio=0.05,
                ),
                GrowFromCenter(gain_eqn_pv[0][17:19]),
                LaggedStart(
                    *[GrowFromCenter(m) for m in gain_eqn_pv[0][19:]],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        gain_soln = (
            MathTex(r"G = \frac{\text{out}}{\text{in}} = \frac{2}{1}")
            .scale(2)
            .move_to(gain_eqn_pv)
        )
        gain_soln[0][0].set_color(GREEN)
        gain_soln[0][2:5].set_color(OUTPUT_COLOR)
        gain_soln[0][6:8].set_color(INPUT_COLOR)
        gain_soln[0][9].set_color(OUTPUT_COLOR)
        gain_soln[0][11].set_color(INPUT_COLOR)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in gain_eqn_pv[0][8:]], lag_ratio=0.02),
                ReplacementTransform(gain_eqn_pv[0][:8], gain_soln[0][:8]),
                GrowFromCenter(gain_soln[0][8]),
                GrowFromCenter(gain_soln[0][9]),
                GrowFromCenter(gain_soln[0][10]),
                GrowFromCenter(gain_soln[0][11]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        vpv = (
            MathTex(r"\left[ \frac{V}{V} \right]")
            .scale(2)
            .next_to(gain_soln, RIGHT, MED_SMALL_BUFF)
        )
        vpv.shift((gain_soln[0][-2].get_y() - vpv[0][2].get_y()) * UP)
        wpw = MathTex(r"\left[ \frac{W}{W} \right]").scale(2).move_to(vpv, LEFT)
        wpw.shift((vpv[0][2].get_y() - wpw[0][2].get_y()) * UP)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in vpv[0]],
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
                ShrinkToCenter(gain_soln[0][-1]),
                ShrinkToCenter(gain_soln[0][-2]),
                gain_soln[0][-3].animate.set_y(gain_soln[0][-2].get_y()),
                lag_ratio=0.3,
            ),
            LaggedStart(
                *[
                    FadeOut(m, shift=shift)
                    for m, shift in zip(
                        [*wpw[0], xu, xd],
                        np.random.random((7, 3)) * 2 - 1,
                    )
                ],
                lag_ratio=0.05,
            ),
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
        # self.play(self.camera.frame.animate.shift(DOWN * fh(self, 3)))

        db_label = (
            Text("dB", font=FONT)
            .scale_to_fit_width(fw(self, 0.15))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self, 1.5))
            .set_y(gain_soln.get_y())
        )
        self.add(db_label)

        arrow = Arrow(gain_soln.get_right(), db_label.get_left())

        self.play(
            LaggedStart(
                GrowArrow(arrow),
                self.camera.frame.animate.move_to(db_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(arrow))

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
        db_label = Text("dB", font=FONT).scale_to_fit_width(fw(self, 0.15))
        # self.play(Write(db_label))
        self.add(db_label)

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
                self.camera.frame.animate.scale(1.2),
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

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(vpv_bez),
                    Uncreate(wpw_bez),
                ),
                wpw_to_log.animate.set_opacity(0.1).shift(DOWN),
                Unwrite(db_label),
                self.camera.frame.animate.scale_to_fit_width(
                    vpv_to_log.width * 1.5
                ).move_to(vpv_to_log),
                lag_ratio=0.3,
            )
        )

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
            .next_to(vpv_to_log, DOWN, LARGE_BUFF * 4)
        )
        self.remove(wpw_to_log)

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
                bd_group
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                bd_group[2].animate.set_opacity(0.1),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in bd_group[3]]
                ),
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

        lna_gain = MathTex(r"G = 17").scale(1.8).next_to(lna, DOWN, MED_SMALL_BUFF)
        lna_gain[0][0].set_color(GREEN)
        filt_gain = (
            MathTex(r"G = 0.63").scale(1.8).next_to(bp_filt, DOWN, MED_SMALL_BUFF)
        )
        filt_gain[0][0].set_color(RED)
        driver_gain = (
            MathTex(r"G = 8.2").scale(1.8).next_to(driver, DOWN, MED_SMALL_BUFF)
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
                AnimationGroup(*[m.animate.set_stroke(opacity=1) for m in bd_group[3]]),
                FadeIn(filt_gain),
                lag_ratio=0.15,
            )
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                bd_group[2].animate.set_opacity(0.1),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in bd_group[3]]
                ),
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
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                bd_group[0].animate.set_opacity(1),
                AnimationGroup(
                    bd_group[1][0].animate.set_stroke(opacity=1),
                    bd_group[1][1].animate.set_stroke(opacity=1),
                ),
                lna_gain.animate.set_opacity(1).scale(0.8),
                bd_group[2].animate.set_opacity(1),
                AnimationGroup(*[m.animate.set_stroke(opacity=1) for m in bd_group[3]]),
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

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                FadeIn(g_tot[0][0]),
                FadeIn(g_tot[0][1:6]),
                FadeIn(g_tot[0][6]),
                TransformFromCopy(lna_gain[0][2:], g_tot[0][7:9]),
                FadeIn(g_tot[0][9]),
                TransformFromCopy(filt_gain[0][2:], g_tot[0][10:14]),
                FadeIn(g_tot[0][14]),
                TransformFromCopy(driver_gain[0][2:], g_tot[0][15:18]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

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

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        g_tot_db = (
            MathTex(r"G_{\text{dB,tot}} = 24.6 - 4.0 + 18.3 \approx 40 \text{ dB}")
            .scale(1.5)
            .next_to(filt_db, DOWN, LARGE_BUFF)
        )
        g_tot_db[0][:7].set_color(GREEN)

        self.play(
            LaggedStart(
                FadeIn(g_tot_db[0][:7]),
                GrowFromCenter(g_tot_db[0][7]),
                TransformFromCopy(lna_db[0][4:-2], g_tot_db[0][8:12]),
                TransformFromCopy(filt_db[0][4], g_tot_db[0][12]),
                TransformFromCopy(filt_db[0][5:8], g_tot_db[0][13:16]),
                GrowFromCenter(g_tot_db[0][16]),
                TransformFromCopy(driver_db[0][4:8], g_tot_db[0][17:21]),
                *[FadeIn(m) for m in g_tot_db[0][21:]],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        wpw_to_log = MathTex(
            r"10 \cdot \log_{10}{\left( \frac{W}{W} \right)}"
        ).scale_to_fit_height(db_label.height * 2)
        ne = MathTex(r"\neq").scale(3).set_color(RED)

        log_group = (
            Group(vpv_to_log.copy(), ne, wpw_to_log)
            .arrange(DOWN, MED_SMALL_BUFF)
            .move_to(vpv_to_log)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(log_group),
                vpv_to_log.animate.move_to(log_group[0]),
                GrowFromCenter(ne),
                FadeIn(wpw_to_log),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        table = (
            ImageMobject("./static/adl8154-table.png")
            .scale_to_fit_width(fw(self, 0.7))
            .next_to(log_group, UP, MED_LARGE_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                Group(log_group, table).height * 1.1
            ).move_to(Group(log_group, table)),
            GrowFromCenter(table),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        gain_box = (
            Rectangle(
                height=table.height * 0.07,
                width=table.width * 0.03,
                color=GAIN_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .move_to(table)
            .shift(UP * 1.5 + LEFT * 1.5)
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(gain_box.width * 20)
                .move_to(gain_box)
                .shift(LEFT * 1.5),
                Create(gain_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        self.play(
            vpv_to_log[0][:2]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .scale(1.5)
        )

        self.wait(0.5)

        self.play(
            wpw_to_log[0][:2]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .scale(1.5)
        )

        self.wait(0.5)

        vpv_val = (
            MathTex(r"\Rightarrow 7.9 \left[\frac{V}{V}\right]")
            .scale_to_fit_height(vpv_to_log.height)
            .next_to(vpv_to_log.copy().shift(LEFT * 3), RIGHT, MED_SMALL_BUFF)
        )
        wpw_val = (
            MathTex(r"\Rightarrow 63.1 \left[\frac{W}{W}\right]")
            .scale_to_fit_height(vpv_to_log.height)
            .next_to(wpw_to_log.copy().shift(LEFT * 3), RIGHT, MED_SMALL_BUFF)
        )

        vpv_bez = CubicBezier(
            gain_box.get_right() + [0.1, 0, 0],
            gain_box.get_right() + [1, 0, 0],
            vpv_val[0][1:4].get_top() + [0, 1, 0],
            vpv_val[0][1:4].get_top() + [0, 0.1, 0],
            color=GREEN,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        wpw_bez = CubicBezier(
            gain_box.get_right() + [0.1, 0, 0],
            gain_box.get_right() + [10, 0, 0],
            wpw_val[0].get_right() + [5, 1, 0],
            wpw_val[0].get_right() + [0.1, 0.1, 0],
            color=GREEN,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                vpv_to_log.animate.shift(LEFT * 3),
                AnimationGroup(
                    LaggedStart(*[FadeIn(m) for m in vpv_val[0]], lag_ratio=0.1),
                    Create(vpv_bez),
                ),
                wpw_to_log.animate.shift(LEFT * 3),
                AnimationGroup(
                    LaggedStart(*[FadeIn(m) for m in wpw_val[0]], lag_ratio=0.1),
                    Create(wpw_bez),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ex = (
            Text("Exercise", font=FONT)
            .scale(2)
            .next_to(self.camera.frame.get_corner(UL), DR, MED_SMALL_BUFF)
            .shift(RIGHT * fw(self, 1.5))
        )
        exbox = SurroundingRectangle(
            ex,
            color=GREEN,
            corner_radius=0.2,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            buff=MED_LARGE_BUFF,
        )

        problem = (
            Paragraph(
                "Why does a linear voltage gain, x,",
                "correspond to a gain in dB of 20 log(x),",
                "but a linear power gain, y, corresponds",
                "to a gain in dB of 10 log(y)?",
                font=FONT,
                line_spacing=LARGE_BUFF,
                alignment="center",
            )
            .scale_to_fit_width(fw(self, 0.5))
            .next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF * 3)
            .shift(RIGHT * fw(self, 1.5))
        )

        hint_ohm = MathTex(r"V = IR").scale(3)
        hint_power = MathTex(r"P = VI").scale(3)
        hint = Text("Hint:", font=FONT).scale(2)
        hints = (
            Group(hint, hint_ohm, hint_power)
            .arrange(RIGHT, LARGE_BUFF * 3)
            .next_to(problem, DOWN, LARGE_BUFF * 2)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(RIGHT * fw(self, 1.5)),
                Write(ex),
                Create(exbox),
                Write(problem),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        syms = (
            Group(
                Tex(r"$V$ | Voltage").scale(2.5),
                Tex(r"$I$ | Current").scale(2.5),
                Tex(r"$P$ | Power").scale(2.5),
                Tex(r"$R$ | Resistance").scale(2.5),
            )
            .arrange_in_grid(2, 2, LARGE_BUFF * 1)
            .next_to(hints, DOWN, LARGE_BUFF * 2)
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                Write(hint),
                FadeIn(hint_ohm),
                FadeIn(hint_power),
                *[GrowFromCenter(m) for m in syms],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.remove(vpv_bez, wpw_bez, vpv_to_log, wpw_to_log, *wpw_val[0], *vpv_val[0])

        self.play(
            self.camera.frame.animate.scale_to_fit_width(table.width * 1.1).move_to(
                table
            )
        )

        self.wait(0.5)

        min_box = gain_box.copy().stretch(1.2, 0).shift(UP * 0.7 + LEFT * 0.8)
        typ_box = gain_box.copy().stretch(1.2, 0).shift(UP * 0.7)
        max_box = gain_box.copy().stretch(1.2, 0).shift(UP * 0.7 + RIGHT * 0.85)

        self.play(
            LaggedStart(
                TransformFromCopy(gain_box, min_box),
                TransformFromCopy(gain_box, typ_box),
                TransformFromCopy(gain_box, max_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class FrequencyDependence(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        page = ImageMobject(f"./static/adl8154-07.png").scale_to_fit_height(
            fh(self, 0.8)
        )
        self.play(GrowFromCenter(page))
        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(page.height * 0.3).shift(
                RIGHT * 1.12 + UP * 0.215
            ),
            run_time=2,
        )

        self.wait(0.5)

        top = (
            ImageMobject("./static/adl8154-top.png")
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(self.camera.frame.get_corner(UR), UR, MED_SMALL_BUFF)
        )
        top_box = SurroundingRectangle(top, buff=0, color=RED)
        top_arrow = Arrow(
            top.get_corner(DL),
            top.get_corner(UR) + [-top.width * 0.06, -top.height * 0.12, 0],
            color=RED,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.6,
            max_tip_length_to_length_ratio=0.1,
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(top),
                Create(top_box),
                FadeIn(top),
                lag_ratio=0.2,
            )
        )
        self.play(GrowArrow(top_arrow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(top, top_box, top_arrow),
                self.camera.frame.animate.restore(),
                lag_ratio=0.2,
            )
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
                color=GAIN_COLOR,
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

        self.play(self.camera.frame.animate.restore().shift(DOWN * 0.2))
        self.remove(page2, volt_box, curr_box)

        self.wait(0.5)

        volt_box = (
            temp_box.copy()
            .stretch(0.4, dim=1)
            .stretch(0.8, dim=0)
            .shift(DOWN * 1 + RIGHT * 0.38)
        )
        curr_box = volt_box.copy().stretch(1.3, dim=0).shift(RIGHT * 0.32)

        self.play(Create(volt_box))

        self.wait(0.5)

        self.play(ReplacementTransform(volt_box, curr_box))

        self.wait(0.5)

        self.add(ax)

        self.play(
            Uncreate(curr_box),
            Create(plot),
            self.camera.frame.animate.shift(UP * 0.2),
            run_time=2,
        )

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

        self.next_section(skip_animations=skip_animations(True))

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

        self.wait(0.5)

        amp = (
            get_amp(width=fh(self, 0.2), stroke_width_mult=0.5)
            .move_to(self.camera.frame)
            .shift(UP * fh(self, 5))
        )
        self.add(amp)

        input_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 1.5,
            y_length=amp.height,
        ).next_to(amp, LEFT, 0)
        input_ax.shift(amp.get_left() - input_ax.c2p(1, 0))

        A1 = VT(0.5)
        A2 = VT(0)
        A3 = VT(0)
        A4 = VT(0)
        f1 = 4
        f2 = 8.8
        f3 = 16.7
        f4 = 24
        phi0 = VT(0)
        phi1 = VT(0)
        phi2 = VT(0)
        phi3 = VT(0)
        stroke_width_vt = VT(0.5)
        input_offset = VT(0)
        input_plot = always_redraw(
            lambda: input_ax.plot(
                lambda t: ~A1 * np.sin(2 * PI * f1 * t + ~phi0)
                + ~A2 * np.sin(2 * PI * f2 * t + ~phi1)
                + ~A3 * np.sin(2 * PI * f3 * t + ~phi2)
                + ~A4 * np.sin(2 * PI * f4 * t + ~phi3)
                + ~input_offset,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=INPUT_COLOR,
            )
        )

        output_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 1.5,
            y_length=amp.height,
        )
        output_ax.shift(amp.get_right() - output_ax.c2p(0, 0))

        f1 = 4
        output_plot = output_ax.plot(
            lambda t: 2 * ~A1 * np.sin(2 * PI * f1 * t),
            stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
            x_range=[0, 1, 1 / 200],
            color=OUTPUT_COLOR,
        )
        self.add(output_plot, input_plot)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_height(amp.height / 0.3).move_to(amp)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(amp),
                FadeOut(output_plot),
                AnimationGroup(
                    self.camera.frame.animate.scale_to_fit_width(
                        input_plot.width * 2
                    ).move_to(input_plot),
                    stroke_width_vt @ 0.3,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        A4.save_state()
        A3.save_state()
        A2.save_state()
        A1.save_state()
        phi0.save_state()
        phi1.save_state()
        phi2.save_state()
        phi3.save_state()
        self.play(
            LaggedStart(
                A4 @ 0.1,
                phi0 @ (PI * 0.2),
                A1 @ 0.2,
                phi1 @ (PI * 0.4),
                A3 @ 0.2,
                phi2 @ (PI * 0.6),
                phi3 @ (PI * 0.1),
                A2 @ 0.4,
                lag_ratio=0.2,
            ),
            run_time=3,
        )
        # self.play(
        #     LaggedStart(
        #         A2 @ 0,
        #         phi3 @ 0,
        #         phi2 @ 0,
        #         A3 @ 0,
        #         phi1 @ 0,
        #         A1 @ 0.5,
        #         phi0 @ 0,
        #         A4 @ 0,
        #         lag_ratio=0.2,
        #     )
        # )

        nz_bw = MathTex(r"B > 0").scale(0.5).next_to(input_plot, UP, SMALL_BUFF * 0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).shift(UP * SMALL_BUFF * 0.5),
                *[FadeIn(m) for m in nz_bw[0]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        opacity2 = VT(0)
        opacity3 = VT(0)
        opacity4 = VT(0)
        offset2 = VT(0)
        offset3 = VT(0)
        offset4 = VT(0)
        A2_single = VT(0.4)
        A3_single = VT(0.2)
        A4_single = VT(0.1)
        A2_single_out = VT(0.4)
        A3_single_out = VT(0.2)
        A4_single_out = VT(0.1)
        input2 = always_redraw(
            lambda: input_ax.plot(
                lambda t: ~A2_single * np.sin(2 * PI * f2 * t + ~phi1) + ~offset2,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=INPUT_COLOR,
                stroke_opacity=~opacity2,
            )
        )
        input3 = always_redraw(
            lambda: input_ax.plot(
                lambda t: ~A3_single * np.sin(2 * PI * f3 * t + ~phi2) + ~offset3,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=INPUT_COLOR,
                stroke_opacity=~opacity3,
            )
        )
        input4 = always_redraw(
            lambda: input_ax.plot(
                lambda t: ~A4_single * np.sin(2 * PI * f4 * t + ~phi3) + ~offset4,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=INPUT_COLOR,
                stroke_opacity=~opacity4,
            )
        )
        self.add(input2, input3, input4)

        self.play(
            FadeOut(nz_bw),
            self.camera.frame.animate.restore()
            .scale(1.4)
            .move_to(input_ax.c2p(0.5, 1.5)),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    offset4 @ 3,
                    opacity4 @ 3,
                    A4 @ 0,
                ),
                AnimationGroup(
                    offset3 @ 2,
                    opacity3 @ 2,
                    A3 @ 0,
                ),
                AnimationGroup(
                    offset2 @ 1,
                    opacity2 @ 1,
                    A2 @ 0,
                ),
                lag_ratio=0.5,
            ),
            run_time=3,
        )

        self.wait(0.5)

        f1_label = Text("1 GHz", font=FONT).scale(0.2).next_to(input_plot, LEFT)
        f2_label = Text("2.4 GHz", font=FONT).scale(0.2).next_to(input2, LEFT)
        f3_label = Text("4.3 GHz", font=FONT).scale(0.2).next_to(input3, LEFT)
        f4_label = Text("6.8 GHz", font=FONT).scale(0.2).next_to(input4, LEFT)

        self.next_section(skip_animations=skip_animations(True))

        decomp_group = Group(
            f1_label, f2_label, f3_label, f4_label, input_plot, input2, input3, input4
        )
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    decomp_group.height * 1.4
                ).move_to(decomp_group),
                FadeIn(f4_label),
                FadeIn(f3_label),
                FadeIn(f2_label),
                FadeIn(f1_label),
                lag_ratio=0.3,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.play(
            ax_x @ (self.camera.frame.get_center() + RIGHT * fw(self, 0.9))[0],
            ax_y @ (self.camera.frame.get_center() + RIGHT * fw(self, 0.9))[1],
            x_length @ fw(self, 0.7),
            y_length @ fh(self, 0.7),
            self.camera.frame.animate.scale_to_fit_width(fw(self, 1.8)).move_to(
                self.camera.frame.get_center() + RIGHT * fw(self, 0.45)
            ),
        )

        self.wait(0.5)

        f1_dot_x = VT(0)
        f1_dot_y = VT(0)
        f1_dot_scale = VT(0)

        xdot = always_redraw(
            lambda: Dot(radius=DEFAULT_DOT_RADIUS * 0.5, color=BLUE)
            .scale(~f1_dot_scale)
            .move_to(ax.c2p(~f1_dot_x, ~f1_dot_y))
            .set_z_index(3)
        )
        line_to_dot1_x = always_redraw(
            lambda: DashedLine(
                ax.c2p(~f1_dot_x, 0),
                xdot.get_center(),
                color=BLUE,
                dash_length=DEFAULT_DASH_LENGTH * 0.6,
                dashed_ratio=0.6,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.4,
            ).set_z_index(2)
        )
        self.add(xdot)

        self.play(LaggedStart(f1_dot_scale @ 1, f1_dot_x @ 1, lag_ratio=0.3))
        self.add(line_to_dot1_x)

        self.wait(0.5)

        self.play(f1_dot_y @ ax.i2gc(1, plot)[1])
        # self.play(xdot.animate.shift(UP))

        self.wait(0.5)

        f1_bez = CubicBezier(
            input_plot.get_right() + [0.05, 0, 0],
            input_plot.get_right() + [0.4, 0, 0],
            xdot.get_center() + [-0.2, -0.2, 0],
            xdot.get_center() + [-0.05, -0.05, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        line_to_dot1_y = always_redraw(
            lambda: DashedLine(
                ax.c2p(0, ax.i2gc(~f1_dot_x, plot)[1]),
                xdot.get_center(),
                color=BLUE,
                dash_length=DEFAULT_DASH_LENGTH * 0.6,
                dashed_ratio=0.6,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.4,
            ).set_z_index(2)
        )
        self.play(LaggedStart(Create(line_to_dot1_y), Create(f1_bez), lag_ratio=0.3))

        self.wait(0.5)

        f1_gain = Text(
            f"(G={ax.i2gc(1, plot)[1]:.1f} dB)", font=FONT
        ).scale_to_fit_height(f1_label.height)
        f1_gain[1].set_color(GREEN)
        f1_label_group = (
            Group(f1_label.copy(), f1_gain)
            .arrange(RIGHT, SMALL_BUFF * 0.8)
            .move_to(f1_label, RIGHT)
        )

        f2_gain = Text(
            f"(G={ax.i2gc(2.4, plot)[1]:.1f} dB)", font=FONT
        ).scale_to_fit_height(f2_label.height)
        f2_gain[1].set_color(GREEN)
        f2_label_group = (
            Group(f2_label.copy(), f2_gain)
            .arrange(RIGHT, SMALL_BUFF * 0.8)
            .move_to(f2_label, RIGHT)
        )

        f3_gain = Text(
            f"(G={ax.i2gc(4.3, plot)[1]:.1f} dB)", font=FONT
        ).scale_to_fit_height(f3_label.height)
        f3_gain[1].set_color(GREEN)
        f3_label_group = (
            Group(f3_label.copy(), f3_gain)
            .arrange(RIGHT, SMALL_BUFF * 0.8)
            .move_to(f3_label, RIGHT)
        )

        f4_gain = Text(
            f"(G={ax.i2gc(6.8, plot)[1]:.1f} dB)", font=FONT
        ).scale_to_fit_height(f4_label.height)
        f4_gain[1].set_color(GREEN)
        f4_label_group = (
            Group(f4_label.copy(), f4_gain)
            .arrange(RIGHT, SMALL_BUFF * 0.8)
            .move_to(f4_label, RIGHT)
        )

        self.play(
            LaggedStart(
                Uncreate(f1_bez),
                self.camera.frame.animate.scale(1.3).shift(LEFT * 0.28),
                f1_label.animate.move_to(f1_label_group[0]),
                LaggedStart(*[FadeIn(m) for m in f1_gain], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    f1_dot_x @ 2.4,
                    f1_dot_y @ ax.i2gc(2.4, plot)[1],
                ),
                lag_ratio=0.3,
            )
        )
        f2_bez = CubicBezier(
            input2.get_right() + [0.05, 0, 0],
            input2.get_right() + [0.4, 0, 0],
            xdot.get_center() + [-0.2, -0.2, 0],
            xdot.get_center() + [-0.05, -0.05, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(f2_bez),
                f2_label.animate.move_to(f2_label_group[0]),
                LaggedStart(*[FadeIn(m) for m in f2_gain], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(f2_bez),
                AnimationGroup(
                    f1_dot_x @ 4.3,
                    f1_dot_y @ ax.i2gc(4.3, plot)[1],
                ),
                lag_ratio=0.5,
            )
        )
        f3_bez = CubicBezier(
            input3.get_right() + [0.05, 0, 0],
            input3.get_right() + [0.4, 0, 0],
            xdot.get_center() + [-0.2, -0.2, 0],
            xdot.get_center() + [-0.05, -0.05, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(f3_bez),
                f3_label.animate.move_to(f3_label_group[0]),
                LaggedStart(*[FadeIn(m) for m in f3_gain], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(f3_bez),
                AnimationGroup(
                    f1_dot_x @ 6.8,
                    f1_dot_y @ ax.i2gc(6.8, plot)[1],
                ),
                lag_ratio=0.5,
            )
        )
        f4_bez = CubicBezier(
            input4.get_right() + [0.05, 0, 0],
            input4.get_right() + [0.4, 0, 0],
            xdot.get_center() + [-0.2, -0.2, 0],
            xdot.get_center() + [-0.05, -0.05, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(f4_bez),
                f4_label.animate.move_to(f4_label_group[0]),
                LaggedStart(*[FadeIn(m) for m in f4_gain], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        plots = Group(input_plot, input2, input3, input4)
        amp_new = amp.copy().next_to(plots, RIGHT, MED_LARGE_BUFF)
        into_amp_up = CubicBezier(
            input4.get_corner(UR) + [0.05, 0, 0],
            input4.get_corner(UR) + [0.3, 0, 0],
            amp_new.get_left() + [-0.3, 0, 0],
            amp_new.get_left() + [-0.05, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )
        into_amp_dn = CubicBezier(
            input_plot.get_corner(DR) + [0.05, 0, 0],
            input_plot.get_corner(DR) + [0.3, 0, 0],
            amp_new.get_left() + [-0.3, 0, 0],
            amp_new.get_left() + [-0.05, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.play(
            LaggedStart(
                Uncreate(f4_bez),
                ax_x + 3,
                AnimationGroup(
                    Create(into_amp_up),
                    Create(into_amp_dn),
                ),
                GrowFromCenter(amp_new),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        output_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=amp.width * 1.5,
                y_length=amp.height,
            )
            .next_to(amp_new, RIGHT, MED_LARGE_BUFF)
            .set_y(input_ax.get_y())
        )

        output_offset1 = VT(-1 - 1)
        output_offset2 = VT(2 - 1)
        output_offset3 = VT(5 - 1)
        output_offset4 = VT(7 - 1)
        output_offset = VT(1.5)
        A1_single = VT(0.5)
        A1_single_out = VT(0.5)
        g1 = VT(3)
        g2 = VT(2.8)
        g3 = VT(2.6)
        g4 = VT(1)
        output_f1_opacity = VT(0)
        output_f2_opacity = VT(0)
        output_f3_opacity = VT(0)
        output_f4_opacity = VT(0)
        output_opacity = VT(0)
        output_en_1 = VT(1)
        output_en_2 = VT(1)
        output_en_3 = VT(1)
        output_en_4 = VT(1)
        output_en_1_full = VT(0)
        output_en_2_full = VT(0)
        output_en_3_full = VT(0)
        output_en_4_full = VT(0)
        output1 = always_redraw(
            lambda: output_ax.plot(
                lambda t: ~output_en_1
                * ~g1
                * ~A1_single
                * np.sin(2 * PI * f1 * t + ~phi0)
                + ~output_offset1,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~output_f1_opacity,
            )
        )
        output2 = always_redraw(
            lambda: output_ax.plot(
                lambda t: ~output_en_2
                * ~g2
                * ~A2_single
                * np.sin(2 * PI * f2 * t + ~phi1)
                + ~output_offset2,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~output_f2_opacity,
            )
        )
        output3 = always_redraw(
            lambda: output_ax.plot(
                lambda t: ~output_en_3
                * ~g3
                * ~A3_single
                * np.sin(2 * PI * f3 * t + ~phi2)
                + ~output_offset3,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~output_f3_opacity,
            )
        )
        output4 = always_redraw(
            lambda: output_ax.plot(
                lambda t: ~output_en_4
                * ~g4
                * ~A4_single
                * np.sin(2 * PI * f4 * t + ~phi3)
                + ~output_offset4,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~output_f4_opacity,
            )
        )
        output_gain_scale = VT(1)
        output_plot = always_redraw(
            lambda: output_ax.plot(
                lambda t: ~output_gain_scale
                * (
                    ~output_en_1_full
                    * ~g1
                    * ~A1_single_out
                    * np.sin(2 * PI * f1 * t + ~phi0)
                    + ~output_en_2_full
                    * ~g2
                    * ~A2_single_out
                    * np.sin(2 * PI * f2 * t + ~phi1)
                    + ~output_en_3_full
                    * ~g3
                    * ~A3_single_out
                    * np.sin(2 * PI * f3 * t + ~phi2)
                    + ~output_en_4_full
                    * ~g4
                    * ~A4_single_out
                    * np.sin(2 * PI * f4 * t + ~phi3)
                )
                + ~output_offset,
                stroke_width=DEFAULT_STROKE_WIDTH * ~stroke_width_vt,
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~output_opacity,
            )
        )
        self.add(output1, output2, output3, output4, output_plot)

        from_amp_up = CubicBezier(
            amp_new.get_right() + [0.05, 0, 0],
            amp_new.get_right() + [0.3, 0, 0],
            output4.get_corner(UL) + [-0.3, 0, 0],
            output4.get_corner(UL) + [-0.05, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )
        from_amp_dn = CubicBezier(
            amp_new.get_right() + [0.05, 0, 0],
            amp_new.get_right() + [0.3, 0, 0],
            output1.get_corner(DL) + [-0.3, 0, 0],
            output1.get_corner(DL) + [-0.05, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(RIGHT * 0.5),
                AnimationGroup(
                    Create(from_amp_up),
                    Create(from_amp_dn),
                ),
                output_f1_opacity @ 1,
                output_f2_opacity @ 1,
                output_f3_opacity @ 1,
                output_f4_opacity @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        from_amp = Line(
            [(output_plot.get_left() + [-0.05, 0, 0])[0], input_ax.c2p(0, 1.5)[1], 0],
            [(amp_new.get_right() + [0.05, 0, 0])[0], input_ax.c2p(0, 1.5)[1], 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        to_amp = Line(
            [(input_plot.get_right() + [0.05, 0, 0])[0], input_ax.c2p(0, 1.5)[1], 0],
            [(amp_new.get_left() + [-0.05, 0, 0])[0], input_ax.c2p(0, 1.5)[1], 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    output_offset2 @ 1.5,
                    output_en_2 @ 0,
                    output_f2_opacity @ 0,
                    output_opacity @ 1,
                    output_en_2_full @ 1,
                ),
                AnimationGroup(
                    output_offset3 @ 1.5,
                    output_en_3 @ 0,
                    output_f3_opacity @ 0,
                    output_en_3_full @ 1,
                ),
                AnimationGroup(
                    output_offset1 @ 1.5,
                    output_en_1 @ 0,
                    output_f1_opacity @ 0,
                    output_en_1_full @ 1,
                ),
                AnimationGroup(
                    output_offset4 @ 1.5,
                    output_en_4 @ 0,
                    output_f4_opacity @ 0,
                    output_en_4_full @ 1,
                ),
                AnimationGroup(
                    ReplacementTransform(from_amp_dn, from_amp),
                    Transform(from_amp_up, from_amp.copy()),
                ),
                lag_ratio=0.3,
            ),
            LaggedStart(
                input_offset @ 1.5,
                AnimationGroup(
                    A2_single @ 0,
                    offset2 @ 1.5,
                    opacity2 @ 0,
                    A2 @ ~A2_single,
                ),
                AnimationGroup(
                    A3_single @ 0,
                    offset3 @ 1.5,
                    opacity3 @ 0,
                    A3 @ ~A3_single,
                ),
                AnimationGroup(
                    A4_single @ 0,
                    offset4 @ 1.5,
                    opacity4 @ 0,
                    A4 @ ~A4_single,
                ),
                AnimationGroup(
                    ReplacementTransform(into_amp_dn, to_amp),
                    Transform(into_amp_up, to_amp.copy()),
                ),
                lag_ratio=0.3,
            ),
            run_time=5,
        )

        self.wait(0.5)

        self.play(output_gain_scale @ 20)

        self.wait(0.5)

        self.play(output_gain_scale @ 1)

        self.wait(0.5)

        arrow = Line(
            input_plot.get_bottom(),
            input_plot.get_bottom() + DOWN * 2,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )
        f_plot_label = (
            Text("Power / Frequency", font=FONT)
            .scale_to_fit_width(fw(self, 0.5))
            .next_to(arrow, DOWN, SMALL_BUFF)
        )
        self.add(f_plot_label)

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(f_plot_label),
                Create(arrow),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(Uncreate(arrow), FadeOut(f_plot_label))

        self.wait(2)


class FreqSpectrum(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        x_length = fw(self, 0.7)
        y_length = fh(self, 0.6)
        x_ticks = np.arange(0, 10, 2)
        y_ticks = np.arange(-50, 10, 10)
        fax = Axes(
            x_range=[0, 8, 1],
            y_range=[-50, 0, 5],
            tips=False,
            x_length=x_length,
            y_length=y_length,
            x_axis_config=dict(
                numbers_with_elongated_ticks=x_ticks,
                include_numbers=True,
                numbers_to_include=x_ticks,
                font_size=DEFAULT_FONT_SIZE * 0.5,
                label_constructor=lambda x: Text(x, font=FONT),
                line_to_number_buff=MED_LARGE_BUFF,
                decimal_number_config=dict(num_decimal_places=0),
                exclude_origin_tick=False,
                longer_tick_multiple=2,
            ),
            y_axis_config=dict(
                numbers_with_elongated_ticks=y_ticks,
                include_numbers=True,
                exclude_origin_tick=False,
                numbers_to_include=y_ticks,
                numbers_to_exclude=[],
                font_size=DEFAULT_FONT_SIZE * 0.5,
                label_constructor=lambda x: Text(x, font=FONT),
                line_to_number_buff=MED_LARGE_BUFF,
                decimal_number_config=dict(num_decimal_places=0),
                longer_tick_multiple=2,
            ),
            axis_config=dict(
                stroke_width=DEFAULT_STROKE_WIDTH,
                tick_size=0.1,
                # stroke_color=BLACK,
            ),
        ).shift(UP * 0.5 + RIGHT * 0.5)
        fax.x_axis.shift((fax.c2p(0, -50) - fax.x_axis.n2p(0)) * UP)
        xlabel = (
            Text("Frequency (GHz)", font=FONT)
            .scale(0.7)
            .next_to(fax.c2p(4, -50), DOWN, LARGE_BUFF)
        )
        ylabel = (
            Text("Power (dBm)", font=FONT)
            .scale(0.7)
            .rotate(PI / 2)
            .next_to(fax, LEFT, MED_SMALL_BUFF)
        )
        # self.add(fax, xlabel, ylabel)

        max_time = 10
        fs = 2**4

        f1 = 1
        f2 = 2.4
        f3 = 4.3
        f4 = 6.8
        A1 = VT(0)
        A2 = VT(0)
        A3 = VT(0)
        A4 = VT(0)
        # A1 = VT(10 ** (0 / 10))
        # A2 = VT(10 ** (-0.8 / 10))
        # A3 = VT(10 ** (-1.5 / 10))
        # A4 = VT(10 ** (-11 / 10))
        X_k_opacity = VT(0)

        x1 = VT(8)

        def get_fft(
            gain=None, color=BLUE, x0_inp=None, x1_inp=None, width=DEFAULT_STROKE_WIDTH
        ):
            def updater():
                if x0_inp is None:
                    x0_local = VT(0)
                else:
                    x0_local = x0_inp
                if x1_inp is None:
                    x1_local = x1
                else:
                    x1_local = x1_inp
                N = max_time * fs
                t = np.linspace(0, max_time, N)
                sig = np.sum(
                    [
                        ~A * np.sin(2 * PI * f * t)
                        for A, f in zip(
                            [A1, A2, A3, A4],
                            [f1, f2, f3, f4],
                        )
                    ],
                    axis=0,
                ) * signal.windows.blackman(N)

                fft_len = 2**10
                sig_fft = fftshift(np.abs(fft(sig, fft_len) / (N / 2)))
                freq = np.linspace(-fs / 2, fs / 2, fft_len)

                sig_fft_log = 10 * np.log10(sig_fft)
                if gain is not None:
                    f_fft_log = interp1d(
                        freq,
                        np.clip(sig_fft_log + gain(freq), -50, None),
                        fill_value="extrapolate",
                    )
                else:
                    f_fft_log = interp1d(
                        freq, np.clip(sig_fft_log, -50, None), fill_value="extrapolate"
                    )
                return fax.plot(
                    f_fft_log,
                    x_range=[~x0_local, ~x1_local, 1 / 200],
                    color=color,
                    stroke_opacity=~X_k_opacity,
                    stroke_width=width,
                )

            return updater

        X_k = always_redraw(get_fft())
        self.add(X_k)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(fax),
                Write(xlabel),
                Write(ylabel),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        f1_label = Text("1 GHz", font=FONT).scale(0.5).next_to(fax.c2p(1, -5), UP)
        f2_label = Text("2.4 GHz", font=FONT).scale(0.5).next_to(fax.c2p(2.4, -6), UP)
        f3_label = Text("4.3 GHz", font=FONT).scale(0.5).next_to(fax.c2p(4.3, -10), UP)
        f4_label = Text("6.8 GHz", font=FONT).scale(0.5).next_to(fax.c2p(6.8, -11), UP)

        self.play(
            LaggedStart(
                X_k_opacity @ 1,
                AnimationGroup(
                    FadeIn(f1_label),
                    A1 @ 0.5,
                ),
                AnimationGroup(
                    FadeIn(f2_label),
                    A2 @ 0.4,
                ),
                AnimationGroup(
                    FadeIn(f3_label),
                    A3 @ 0.2,
                ),
                AnimationGroup(
                    FadeIn(f4_label),
                    A4 @ 0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        specan = (
            ImageMobject("../props/static/SpecAn_Empty.png")
            .scale(2.1)
            .shift(RIGHT * 4.2 + DOWN * 0.5)
        )
        # self.add(specan)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                FadeIn(specan),
                self.camera.frame.animate.scale_to_fit_width(
                    specan.width * 1.1
                ).move_to(specan),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.restore(),
                FadeOut(specan),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        amp = rf.Network("../../notebooks/data/ADL8154ACPZN_SParameters_25C.S2p")
        bw_mask = (amp.f > 0) & (amp.f < 8e9)
        gain_top = 20
        s21 = interp1d(
            amp.f[bw_mask] / 1e9,
            amp.s_db[bw_mask][:, 1, 0] - gain_top,
            fill_value="extrapolate",
        )
        s21_plot = always_redraw(
            lambda: fax.plot(s21, color=GREEN, x_range=[0, ~x1, 1 / 200])
        )
        gain_ticks = np.arange(0, gain_top + 4, 4)

        gnl = NumberLine(
            x_range=[0, gain_top, 2],
            length=fax.y_axis.length,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tick_size=0.1,
            numbers_with_elongated_ticks=gain_ticks,
            include_numbers=True,
            exclude_origin_tick=False,
            numbers_to_include=gain_ticks,
            numbers_to_exclude=[],
            font_size=DEFAULT_FONT_SIZE * 0.5,
            label_constructor=lambda x: Text(x, font=FONT),
            line_to_number_buff=MED_LARGE_BUFF,
            decimal_number_config=dict(num_decimal_places=0),
            longer_tick_multiple=2,
            label_direction=RIGHT,
            rotation=PI / 2,
        )
        gnl.shift(fax.c2p(8, -50) - gnl.n2p(0))
        gain_label = (
            Text("Gain (dB)", font=FONT, color=GAIN_COLOR)
            .rotate(PI / 2)
            .scale_to_fit_width(ylabel.width)
            .next_to(gnl, RIGHT, MED_SMALL_BUFF)
        )
        all_group = Group(fax, gnl, xlabel, ylabel, gain_label)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.15
                ).move_to(all_group),
                Create(gnl),
                Write(gain_label),
                AnimationGroup(
                    Create(s21_plot),
                    LaggedStart(
                        f1_label.animate.set_opacity(0),
                        f2_label.animate.set_opacity(0),
                        f3_label.animate.set_opacity(0),
                        f4_label.animate.set_opacity(0),
                        lag_ratio=0.2,
                    ),
                ),
                # ylabel.animate.set_color(BLUE),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        s21_full = interp1d(
            amp.f[bw_mask] / 1e9,
            amp.s_db[bw_mask][:, 1, 0],
            fill_value="extrapolate",
        )

        self.play(x1 @ 0)
        output_plot = always_redraw(get_fft(s21_full, OUTPUT_COLOR))
        self.add(output_plot)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            x1.animate(run_time=12).set_value(8),
            self.camera.frame.animate.shift(UP * 0.6),
        )

        self.wait(0.5)

        x0_inp = VT(0)
        x1_inp = VT(0)
        input_plot_highlight = always_redraw(
            get_fft(
                color=YELLOW,
                x0_inp=x0_inp,
                x1_inp=x1_inp,
                width=DEFAULT_STROKE_WIDTH * 2,
            )
        )
        self.add(input_plot_highlight)
        self.next_section(skip_animations=skip_animations(True))

        self.play(LaggedStart(x1_inp @ 8, x0_inp @ 8, lag_ratio=0.2), run_time=2)

        self.remove(input_plot_highlight)
        x0_inp @= 0
        x1_inp @= 0

        self.wait(0.5)

        gain_plot = always_redraw(
            lambda: fax.plot(
                s21,
                color=YELLOW,
                x_range=[~x0_inp, ~x1_inp, 1 / 200],
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
        )
        self.add(gain_plot)
        self.next_section(skip_animations=skip_animations(True))

        self.play(LaggedStart(x1_inp @ 8, x0_inp @ 8, lag_ratio=0.2), run_time=2)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            self.camera.frame.animate.scale(10).shift(RIGHT * 100 + DOWN * 100),
            run_time=2,
        )

        self.wait(2)


class Wrapup(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        def amp_graphic(pn, f0, f1, g):
            amp = (
                get_amp(width=fh(self, 0.2), stroke_width_mult=1.5)
                .move_to(self.camera.frame)
                .shift(UP * fh(self, 5))
            )
            freq_label = Text(f"{g} dB @ {f0} → {f1} GHz", font=FONT)
            freq_label[: len(f"{g}")].set_color(GREEN)
            freq_label[len(f"{g}") + 3 : len(f"{g}") + 3 + len(f"{f0}")].set_color(BLUE)
            freq_label[
                len(f"{g}") + 4 + len(f"{f0}") : len(f"{g}")
                + 4
                + len(f"{f0}")
                + len(f"{f1}")
            ].set_color(BLUE)
            pn_label = Text(f"{pn}", font=FONT).next_to(
                freq_label, DOWN, MED_SMALL_BUFF, LEFT
            )
            return Group(
                amp, Group(freq_label, pn_label).next_to(amp, RIGHT, MED_SMALL_BUFF)
            )

        amps = Group(
            *[
                amp_graphic(pn, f0, f1, g)
                for pn, f0, f1, g in [
                    ("ADL8154", 0.1, 6, 18),
                    ("TRF1208", 0.01, 11, 16),
                    ("PMA3-24323LN+", 24, 32, 16.6),
                    ("AD8353", 0.001, 2.7, 15.6),
                ]
            ]
        ).arrange(DOWN, LARGE_BUFF, aligned_edge=LEFT)
        amps.shift((self.camera.frame.get_center()[1] - amps[0].get_y()) * UP).shift(
            DOWN
        )
        # self.add(amps)

        self.play(
            LaggedStart(
                LaggedStart(
                    *[FadeIn(m, shift=LEFT * 3) for m in amps],
                    lag_ratio=0.35,
                    run_time=5,
                ),
                self.camera.frame.animate(
                    rate_func=rate_functions.linear, run_time=6
                ).set_y(amps[-1].get_y()),
                # lag_ratio=0.2,
            )
        )
        self.play(
            self.camera.frame.animate.scale_to_fit_height(amps.height * 1.1).move_to(
                amps
            )
        )

        self.wait(0.5)

        sat = (
            ImageMobject("../props/static/Satellite Cartoon.png")
            .scale_to_fit_height(fh(self, 0.5))
            .next_to(amps, LEFT, MED_SMALL_BUFF)
        )
        sat_group = Group(sat, amps)

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(sat_group),
                GrowFromCenter(sat),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in amps[0][0]],
                    amps[0][1][0].animate.set_opacity(0.1),
                    amps[0][1][1].animate.set_opacity(0.1),
                ),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in amps[-1][0]],
                    amps[-1][1][0].animate.set_opacity(0.1),
                    amps[-1][1][1].animate.set_opacity(0.1),
                ),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in amps[1][0]],
                    amps[1][1][0].animate.set_opacity(0.1),
                    amps[1][1][1].animate.set_opacity(0.1),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        router = (
            ImageMobject("../props/static/Router Cartoon.png")
            .scale_to_fit_height(fh(self, 0.5))
            .move_to(sat)
        )

        self.play(
            LaggedStart(
                sat.animate.shift(UP * 10),
                router.shift(DOWN * 10).animate.shift(UP * 10),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in amps[2][0]],
                    amps[2][1][0].animate.set_opacity(0.1),
                    amps[2][1][1].animate.set_opacity(0.1),
                ),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=1) for m in amps[0][0]],
                    amps[0][1][0].animate.set_opacity(1),
                    amps[0][1][1].animate.set_opacity(1),
                ),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=1) for m in amps[-1][0]],
                    amps[-1][1][0].animate.set_opacity(1),
                    amps[-1][1][1].animate.set_opacity(1),
                ),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=1) for m in amps[1][0]],
                    amps[1][1][0].animate.set_opacity(1),
                    amps[1][1][1].animate.set_opacity(1),
                ),
            ),
        )

        self.wait(0.5)

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

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self)),
                GrowFromCenter(op1db),
                GrowFromCenter(oip3),
                GrowFromCenter(pn),
                GrowFromCenter(gain),
                GrowFromCenter(etc),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        sub = (
            ImageMobject("../props/static/subscribe.png")
            .scale_to_fit_width(fw(self, 0.6))
            .move_to(self.camera.frame)
            .shift(DOWN * 2.5)
        )

        bez1 = CubicBezier(
            op1db.copy().shift(UP * 2.5).get_bottom() + [0, -0.1, 0],
            op1db.copy().shift(UP * 2.5).get_bottom() + [0, -2, 0],
            sub.get_top() + [0, 1, 0],
            sub.get_top() + [0, 0.1, 0],
        )
        bez2 = CubicBezier(
            oip3.copy().shift(UP * 2.5).get_bottom() + [0, -0.1, 0],
            oip3.copy().shift(UP * 2.5).get_bottom() + [0, -1, 0],
            sub.get_top() + [0, 1, 0],
            sub.get_top() + [0, 0.1, 0],
        )
        bez3 = CubicBezier(
            pn.copy().shift(UP * 2.5).get_bottom() + [0, -0.1, 0],
            pn.copy().shift(UP * 2.5).get_bottom() + [0, -2, 0],
            sub.get_top() + [0, 1, 0],
            sub.get_top() + [0, 0.1, 0],
        )
        bez4 = CubicBezier(
            gain.copy().shift(UP * 2.5).get_bottom() + [0, -0.1, 0],
            gain.copy().shift(UP * 2.5).get_bottom() + [0, -1, 0],
            sub.get_top() + [0, 1, 0],
            sub.get_top() + [0, 0.1, 0],
        )
        bez5 = CubicBezier(
            etc.copy().shift(UP * 2.5).get_bottom() + [0, -0.1, 0],
            etc.copy().shift(UP * 2.5).get_bottom() + [0, -2, 0],
            sub.get_top() + [0, 1, 0],
            sub.get_top() + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                op1db.animate.shift(UP * 2.5),
                oip3.animate.shift(UP * 2.5),
                pn.animate.shift(UP * 2.5),
                gain.animate.shift(UP * 2.5),
                etc.animate.shift(UP * 2.5),
                Create(bez1),
                Create(bez2),
                Create(bez3),
                Create(bez4),
                Create(bez5),
                sub.shift(DOWN * 5).animate.shift(UP * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        nb0 = ImageMobject("./static/nb0.png").scale_to_fit_height(fh(self, 0.8))
        nb1 = ImageMobject("./static/nb1.png").scale_to_fit_height(fh(self, 0.8))
        nb2 = ImageMobject("./static/nb2.png").scale_to_fit_height(fh(self, 0.8))
        nbs = Group(nb0, nb1, nb2).arrange(RIGHT, LARGE_BUFF)
        nbs.shift(self.camera.frame.get_center() - nb0.get_center())

        self.play(
            LaggedStart(
                ShrinkToCenter(sub),
                AnimationGroup(
                    Uncreate(bez1),
                    Uncreate(bez2),
                    Uncreate(bez3),
                    Uncreate(bez4),
                    Uncreate(bez5),
                ),
                AnimationGroup(
                    ShrinkToCenter(op1db),
                    ShrinkToCenter(oip3),
                    ShrinkToCenter(pn),
                    ShrinkToCenter(gain),
                    ShrinkToCenter(etc),
                ),
                GrowFromCenter(nb0),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(nb1),
            GrowFromCenter(nb1),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(nb2),
            GrowFromCenter(nb2),
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
