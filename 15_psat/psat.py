# psat.py

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


class LinearRegion(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        linax = Axes(
            x_range=[0, 10 * 4, 1],
            y_range=[0, 10 * 4, 1],
            x_length=fh(self, 0.7 * 4),
            y_length=fh(self, 0.7 * 4),
            tips=False,
            axis_config=dict(
                stroke_width=DEFAULT_STROKE_WIDTH * 1.3,
            ),
        )
        linax.shift(self.camera.frame.get_center() - linax.c2p(5, 5))
        inp_label = Text("Input", font=FONT).next_to(linax.c2p(5, 0), DOWN)
        outp_label = (
            Text("Output", font=FONT).rotate(PI / 2).next_to(linax.c2p(0, 5), LEFT)
        )

        x1 = VT(0)
        linplot = always_redraw(
            lambda: linax.plot(
                lambda x: x,
                color=BLUE,
                x_range=[0, ~x1, 1 / 200],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
        )

        dot_opacity = VT(0)
        dot = always_redraw(
            lambda: Dot(color=YELLOW)
            .scale(2)
            .set_opacity(~dot_opacity)
            .move_to(linax.c2p(~x1, ~x1))
            .set_z_index(1)
        )
        xline = always_redraw(
            lambda: DashedLine(
                linax.c2p(~x1, 0),
                linax.c2p(~x1, ~x1),
                dashed_ratio=0.6,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )
        yline = always_redraw(
            lambda: DashedLine(
                linax.c2p(0, ~x1),
                linax.c2p(~x1, ~x1),
                dashed_ratio=0.6,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.play(
            LaggedStart(
                Create(linax),
                Write(inp_label),
                Write(outp_label),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.add(yline, xline, dot, linplot)

        self.play(
            LaggedStart(
                dot_opacity @ 1,
                x1.animate(run_time=5).set_value(40),
                lag_ratio=0.7,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        amp = get_amp(width=fh(self, 0.3))
        self.play(
            LaggedStart(
                FadeOut(*self.mobjects),
                LaggedStart(
                    GrowFromCenter(
                        amp[0], rate_func=rate_functions.ease_out_elastic, run_time=1.4
                    ),
                    GrowFromCenter(
                        amp[1], rate_func=rate_functions.ease_out_elastic, run_time=1.4
                    ),
                    lag_ratio=0.15,
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        gain_label = MathTex(r"G = 20 \text{ dB}").scale(1.5).next_to(amp, DOWN)
        lin_gain_label = (
            MathTex(r"= 100 \ \frac{W}{W}")
            .scale(1.5)
            .next_to(gain_label[0][1:], DOWN, aligned_edge=LEFT)
        )
        gain_label[0][0].set_color(GAIN_COLOR)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in gain_label[0]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                *[FadeIn(m) for m in lin_gain_label[0]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        gain_thumbnail_img = (
            ImageMobject("../14_amp_gain/static/Gain Thumbnail.png")
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(lin_gain_label, DR, LARGE_BUFF)
            .shift(RIGHT)
        )
        thumbnail_box = SurroundingRectangle(gain_thumbnail_img, buff=0)
        gain_thumbnail = Group(gain_thumbnail_img, thumbnail_box)

        thumbnail_bez = CubicBezier(
            lin_gain_label.get_right() + [0.1, 0, 0],
            lin_gain_label.get_right() + [1, 0, 0],
            gain_thumbnail.get_left() + [-1, 0, 0],
            gain_thumbnail.get_left() + [-0.1, 0, 0],
        )

        all_group = Group(amp, gain_label, lin_gain_label, gain_thumbnail)
        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.1
                ).move_to(all_group),
                Create(thumbnail_bez),
                GrowFromCenter(gain_thumbnail),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShrinkToCenter(gain_thumbnail),
                Uncreate(thumbnail_bez),
                self.camera.frame.animate.restore(),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        inp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 2,
            y_length=amp.height,
        )
        inp_ax.shift(amp.get_left() - inp_ax.c2p(1, 0))
        outp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp.width * 2,
            y_length=amp.height,
        )
        outp_ax.shift(amp.get_right() - outp_ax.c2p(0, 0))

        A = VT(0.15)
        f = 3
        G = 10
        x1_in = VT(0)
        x1_out = VT(0)
        clip = 3
        inp = always_redraw(
            lambda: inp_ax.plot(
                lambda t: ~A * np.sin(2 * PI * f * t),
                x_range=[0, ~x1_in, 1 / 1000],
                color=INPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
        )
        outp = always_redraw(
            lambda: outp_ax.plot(
                lambda t: np.clip(~A * G * np.sin(2 * PI * f * t), -clip, clip),
                x_range=[0, ~x1_out, 1 / 1000],
                color=OUTPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
        )
        self.add(inp, outp)

        self.play(
            LaggedStart(
                FadeOut(lin_gain_label),
                gain_label.animate.shift(DOWN * 2),
                x1_in @ 1,
                x1_out @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        def get_lines(ref, side=LEFT):
            def line_updater():
                inp_line = Line(
                    ref.get_corner(side + DOWN),
                    ref.get_corner(side + UP),
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                ).shift(side * 0.2)
                return inp_line

            def u_updater():
                inp_line = line_updater()
                inp_line_u = Line(
                    inp_line.get_top() + LEFT * 0.15,
                    inp_line.get_top() + RIGHT * 0.15,
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                )
                return inp_line_u

            def d_updater():
                inp_line = line_updater()
                inp_line_d = Line(
                    inp_line.get_bottom() + LEFT * 0.15,
                    inp_line.get_bottom() + RIGHT * 0.15,
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                )
                return inp_line_d

            return line_updater, u_updater, d_updater

        inp_line_upd, inp_line_u_upd, inp_line_d_upd = get_lines(ref=inp, side=LEFT)
        outp_line_upd, outp_line_u_upd, outp_line_d_upd = get_lines(
            ref=outp, side=RIGHT
        )

        inp_line = inp_line_upd()
        inp_line_d = inp_line_d_upd()
        inp_line_u = inp_line_u_upd()

        outp_line = outp_line_upd()
        outp_line_d = outp_line_d_upd()
        outp_line_u = outp_line_u_upd()

        inp_pow_label = Text("10.0 mW", font=FONT).next_to(inp_line, LEFT)
        outp_pow_label = Text("1.0 W", font=FONT).next_to(outp_line, RIGHT)

        inp_level = VT(10)
        outp_level = VT(1)
        inp_pow_label = always_redraw(
            lambda: Text(f"{~inp_level:.2f} mW", font=FONT).next_to(
                inp_line, LEFT, MED_SMALL_BUFF
            )
        )
        outp_pow_label = always_redraw(
            lambda: Text(f"{~outp_level:.2f} W", font=FONT).next_to(
                outp_line, RIGHT, MED_SMALL_BUFF
            )
        )

        all_group = Group(inp_pow_label, amp, outp_pow_label)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(all_group.width * 1.2)
                .move_to(all_group)
                .set_x(amp.get_x()),
                Create(inp_line_d),
                Create(inp_line),
                Create(inp_line_u),
                FadeIn(inp_pow_label),
                Create(outp_line_d),
                Create(outp_line),
                Create(outp_line_u),
                FadeIn(outp_pow_label),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))

        self.remove(
            inp_line,
            inp_line_u,
            inp_line_d,
            outp_line,
            outp_line_u,
            outp_line_d,
        )
        inp_line_upd = always_redraw(inp_line_upd)
        inp_line_d_upd = always_redraw(inp_line_d_upd)
        inp_line_u_upd = always_redraw(inp_line_u_upd)

        outp_line_upd = always_redraw(outp_line_upd)
        outp_line_d_upd = always_redraw(outp_line_d_upd)
        outp_line_u_upd = always_redraw(outp_line_u_upd)
        self.add(
            inp_line_upd,
            inp_line_u_upd,
            inp_line_d_upd,
            outp_line_upd,
            outp_line_u_upd,
            outp_line_d_upd,
            inp_pow_label,
            outp_pow_label,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        scale_2 = 2

        self.play(
            inp_level @ (~inp_level * scale_2),
            outp_level @ (~outp_level * scale_2),
            A @ (~A * scale_2),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        scale_3 = 1.5
        scale_3_comp = 1.2

        self.play(
            inp_level @ (~inp_level * scale_3),
            outp_level @ (~outp_level * scale_3_comp),
            A @ (~A * scale_3),
            run_time=2,
        )

        self.wait(0.5)

        # TODO: need to include this part
        # And for a while, that's exactly what happens. Every time we increase the input power by some amount, the output power increases by that same amount plus the gain.
        # This is the linear region, and it's where we want to operate most of the time because the amplifier behaves predictably.

        scale_4 = 1.5
        scale_4_comp = 1.1

        self.play(
            inp_level @ (~inp_level * scale_4),
            outp_level @ (~outp_level * scale_4_comp),
            A @ (~A * scale_4),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        scale_5 = 3
        scale_5_comp = 1.05

        self.play(
            inp_level @ (~inp_level * scale_5),
            outp_level @ (~outp_level * scale_5_comp),
            A @ (~A * scale_5),
            run_time=2,
        )

        self.wait(2)


class CompressionStarts(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))

        self.wait(2)


class Counter(Scene):
    def construct(self):
        start = 0
        end = 9

        texts = Group(*[Text(f"{i}", font=FONT) for i in range(start, end + 1, 1)])
        texts.arrange(DOWN, buff=0, aligned_edge=RIGHT)
        self.add(texts)

        for i in range(start, end + 1, 1):
            speed = (i / (end - start)) + 1
            print(i, speed)

        # self.add(text)

        # self.wait(2)
