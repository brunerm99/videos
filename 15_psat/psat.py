# psat.py
import os
import sys
from random import shuffle

import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from manim.utils.color.X11 import BROWN1
from MF_Tools import VT
from networkx import center
from numpy.fft import fft, fftshift
from pyglet.media.drivers.pulse.interface import PA_INVALID_WRITABLE_SIZE
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import (
    Bjt,
    Capacitor,
    Fet,
    Inductor,
    Resistor,
    VideoMobject,
    WeatherRadarTower,
    cubic_bezier,
    ease_in_out_elastic,
    ease_out_elastic,
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
FONT = os.getenv("FONT", "")

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
PAE_COLOR = YELLOW


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


intro_elastic = ease_out_elastic(1.34, 0.61)
in_out_elastic = ease_in_out_elastic(1.34, 0.61)


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        amp = get_amp(width=fh(self, 0.3))

        self.play(
            LaggedStart(
                GrowFromCenter(amp[0], rate_func=intro_elastic, run_time=1),
                GrowFromCenter(amp[1], rate_func=intro_elastic, run_time=1),
                lag_ratio=0.15,
            ),
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
        psat = 3.2
        inp = always_redraw(
            lambda: inp_ax.plot(
                lambda t: ~A * np.sin(2 * PI * f * t),
                x_range=[0, ~x1_in, 1 / 1000],
                color=INPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 3,
                use_smoothing=False,
            )
        )
        outp = always_redraw(
            lambda: outp_ax.plot(
                lambda t: np.clip(~A * G * np.sin(2 * PI * f * t), -psat, psat),
                x_range=[0.001, ~x1_out, 1 / 1000],
                color=OUTPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 3,
                use_smoothing=False,
            )
        )
        self.add(inp, outp)

        self.play(
            LaggedStart(
                x1_in @ 1,
                x1_out @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        # self.remove(inp)
        # self.camera.frame.move_to(Group(amp, outp_ax))
        self.play(A @ 0.45, self.camera.frame.animate.scale(1.2), run_time=4)

        self.wait(0.5)

        datasheet = (
            ImageMobject("../14_amp_gain/static/adl8154-03.png")
            .scale_to_fit_height(fh(self, 0.7))
            .rotate(-PI * 0.05)
            .next_to(self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF)
            .shift(UP * fh(self))
        )
        psat_label = MathTex(r"P_{\mathrm{sat}}", color=BLUE)
        p1db_label = MathTex(r"P_{\mathrm{1dB}}", color=GREEN)
        labels = (
            Group(p1db_label, psat_label)
            .arrange(DOWN, MED_LARGE_BUFF)
            .scale_to_fit_width(fw(self, 0.25))
            .next_to(datasheet, LEFT, LARGE_BUFF * 2)
        )
        psat_bez = CubicBezier(
            datasheet.get_center()
            + [-datasheet.width * 0.27, datasheet.height * 0.12, 0],
            datasheet.get_center() + [-datasheet.width / 4 - 2, -2, 0],
            psat_label.get_corner(UR) + [2, 1, 0],
            psat_label.get_corner(UR) + [-0.5, 0, 0],
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        )
        p1db_bez = CubicBezier(
            datasheet.get_center()
            + [-datasheet.width * 0.29, datasheet.height * 0.18, 0],
            datasheet.get_center() + [-datasheet.width / 4, datasheet.height * 0.5, 0],
            p1db_label.get_right() + [2, 1, 0],
            p1db_label.get_right() + [0.1, 0, 0],
            color=GREEN,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * fh(self)),
                FadeIn(datasheet),
                Create(p1db_bez),
                GrowFromCenter(p1db_label),
                Create(psat_bez),
                GrowFromCenter(psat_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        p1db_rect = SurroundingRectangle(
            p1db_label, stroke_width=DEFAULT_STROKE_WIDTH * 3
        )
        psat_rect = SurroundingRectangle(
            psat_label, stroke_width=DEFAULT_STROKE_WIDTH * 3
        )

        self.play(Create(p1db_rect))

        self.wait(0.5)

        self.play(ReplacementTransform(p1db_rect, psat_rect))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                Uncreate(p1db_bez),
                ShrinkToCenter(datasheet.set_z_index(-10)),
                FadeOut(psat_rect),
                ShrinkToCenter(p1db_label),
                Uncreate(psat_bez),
                ShrinkToCenter(psat_label),
                FadeOut(p1db_rect),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        p1_thumbnail = ImageMobject(
            "../14_amp_gain/static/Gain Thumbnail.png"
        ).scale_to_fit_width(fw(self, 0.25))
        p1_box = SurroundingRectangle(p1_thumbnail, buff=0, stroke_opacity=0)
        p1 = Group(p1_thumbnail, p1_box)
        p2_thumbnail = ImageMobject("./static/Psat Thumbnail.png").scale_to_fit_width(
            fw(self, 0.25)
        )
        p2_box = SurroundingRectangle(p2_thumbnail, buff=0, stroke_opacity=0)
        p2 = Group(p2_thumbnail, p2_box)
        p3_thumbnail = Text("???", font=FONT)
        p3_box = p2_box.copy().move_to(p3_thumbnail)
        p3 = Group(p3_thumbnail, p3_box)
        ps = Group(p1, p2, p3).arrange(RIGHT, LARGE_BUFF).move_to(self.camera.frame)
        p1_label = Text("Gain", font=FONT).next_to(p1, DOWN, MED_SMALL_BUFF)
        p2_label = Text("Compression", font=FONT).next_to(p2, DOWN, MED_SMALL_BUFF)
        p3_label = Text("Next...", font=FONT).next_to(p3, DOWN, MED_SMALL_BUFF)

        a12 = Arrow(p1.get_right(), p2.get_left(), buff=0)
        a23 = Arrow(p2.get_right(), p3.get_left(), buff=0)
        a3x = Arrow(p3.get_right(), p3.get_right() + [4, 0, 0], buff=0)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                p1.shift(DOWN * fh(self, 0.8)).animate.shift(UP * fh(self, 0.8)),
                Write(p1_label),
                GrowArrow(a12),
                p2.shift(DOWN * fh(self, 0.8)).animate.shift(UP * fh(self, 0.8)),
                Write(p2_label),
                GrowArrow(a23),
                p3.shift(DOWN * fh(self, 0.8)).animate.shift(UP * fh(self, 0.8)),
                Write(p3_label),
                lag_ratio=0.5,
            )
        )
        self.play(
            LaggedStart(
                p1_box.animate.set_stroke(opacity=0.2),
                p2_box.animate.set_stroke(opacity=1),
                p3_box.animate.set_stroke(opacity=0.2),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(a3x),
                self.camera.frame.animate.shift(RIGHT * fw(self, 2)),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


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
            lambda: (
                Dot(color=YELLOW)
                .scale(2)
                .set_opacity(~dot_opacity)
                .move_to(linax.c2p(~x1, ~x1))
                .set_z_index(1)
            )
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

        amp = get_amp(width=fh(self, 0.3)).shift(RIGHT * fw(self, 2.8))
        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                # FadeOut(*self.mobjects),
                self.camera.frame.animate.shift(RIGHT * fw(self, 2.8)),
                LaggedStart(
                    GrowFromCenter(amp[0], rate_func=intro_elastic, run_time=1.4),
                    GrowFromCenter(amp[1], rate_func=intro_elastic, run_time=1.4),
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

        # TODO: update thumbnail for notebook
        notebook_thumbnail_img = (
            ImageMobject("../14_amp_gain/static/Gain Thumbnail.png")
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(gain_thumbnail, UP, LARGE_BUFF)
            .shift(RIGHT / 2)
        )
        thumbnail_box = SurroundingRectangle(notebook_thumbnail_img, buff=0)
        notebook_thumbnail = Group(notebook_thumbnail_img, thumbnail_box)

        notebook_bez = CubicBezier(
            lin_gain_label.get_right() + [0.1, 0, 0],
            lin_gain_label.get_right() + [1, 0, 0],
            notebook_thumbnail.get_left() + [-1, 0, 0],
            notebook_thumbnail.get_left() + [-0.1, 0, 0],
        )

        all_group = Group(amp, gain_label, lin_gain_label, gain_thumbnail)
        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.1
                ).move_to(all_group),
                Create(thumbnail_bez),
                Create(notebook_bez),
                GrowFromCenter(gain_thumbnail),
                GrowFromCenter(notebook_thumbnail),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShrinkToCenter(gain_thumbnail),
                ShrinkToCenter(notebook_thumbnail),
                Uncreate(thumbnail_bez),
                Uncreate(notebook_bez),
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
        psat = 3.2
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
                lambda t: np.clip(~A * G * np.sin(2 * PI * f * t), -psat, psat),
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
        self.next_section(skip_animations=skip_animations(True))

        scale_2 = 2

        self.play(
            inp_level @ (~inp_level * scale_2),
            outp_level @ (~outp_level * scale_2),
            A @ (~A * scale_2),
            run_time=2,
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.8).shift(
                    linax.c2p(5, 5) - self.camera.frame.get_center()
                ),
                x1 @ 5,
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        xshift = 4
        xarrow = Arrow(
            dot.get_center(),
            [linax.c2p(~x1 + xshift)[0], dot.get_center()[1], 0],
            buff=0,
        )

        self.play(GrowArrow(xarrow))

        self.wait(0.5)

        yarrow = xarrow.copy()

        self.play(
            yarrow.animate.rotate(PI / 2, about_point=yarrow.get_start()).shift(
                RIGHT * xarrow.width
            )
        )

        self.wait(0.5)

        self.play(x1 + xshift)

        self.wait(0.5)

        lin_region = Polygon(
            linax.c2p(0, 0),
            linax.c2p(0, 10),
            linax.c2p(10, 10),
            linax.c2p(10, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=GREEN,
        )

        self.play(FadeIn(lin_region))

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        scale_3 = 1.5
        scale_3_comp = 1.2

        self.play(
            inp_level @ (~inp_level * scale_3),
            outp_level @ (~outp_level * scale_3_comp),
            A @ (~A * scale_3),
            run_time=2,
        )

        self.wait(0.5)

        scale_4 = 1.5
        scale_4_comp = 1.1

        self.play(
            inp_level @ (~inp_level * scale_4),
            outp_level @ (~outp_level * scale_4_comp),
            A @ (~A * scale_4),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        scale_5 = 3
        scale_5_comp = 1.05

        self.play(
            inp_level @ (~inp_level * scale_5),
            outp_level @ (~outp_level * scale_5_comp),
            A @ (~A * scale_5),
            run_time=2,
        )

        self.wait(0.5)

        whats_happening = Text("What's happening here?", font=FONT).move_to(
            self.camera.frame.get_top()
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.1).shift(UP),
                Write(whats_happening),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        new_cam = self.camera.frame.copy().move_to(amp).scale(0.01)

        self.play(Transform(self.camera.frame, new_cam))

        self.wait(2)


class CompressionStarts(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))
        width = self.camera.frame.width * 0.1
        spacing = width / 2
        stroke_width_mult = 1.5
        bjt = Bjt(width, stroke_width_mult=stroke_width_mult)

        cap1 = Capacitor(width, stroke_width_mult=stroke_width_mult).next_to(
            bjt, LEFT, spacing * 2
        )
        l1 = Line(
            cap1.get_right(),
            bjt.base,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        ind1: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(l1, DOWN, spacing)
        )
        choke_line = Line(
            ind1.get_top(),
            l1.get_midpoint(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        vb_line = Line(
            ind1.get_bottom(),
            ind1.get_bottom() + DOWN * spacing,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        vb_term: Circle = bjt.base.copy().set_color(RED).next_to(vb_line, DOWN, 0)
        vb = (
            MathTex(r"V_b")
            .scale_to_fit_width(ind1.width)
            .next_to(vb_term, DOWN, MED_SMALL_BUFF)
        )

        ind_emitter: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(bjt.emitter.get_end(), DOWN, spacing)
        )
        le = Line(
            bjt.emitter.get_end(),
            ind_emitter.get_top(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        lc = Line(
            bjt.collector.get_end(),
            bjt.collector.get_end() + UP * spacing,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        ind_collector: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(lc, UP, 0)
        )

        load = (
            Resistor(width=width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(lc.get_midpoint(), RIGHT, spacing)
        )
        ll = Line(
            lc.get_midpoint(),
            load.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        dddr = MathTex(r"\cdots").next_to(load, RIGHT, MED_LARGE_BUFF)
        dddl = MathTex(r"\cdots").next_to(cap1, LEFT, MED_LARGE_BUFF)
        dddu = (
            MathTex(r"\cdots").rotate(PI / 2).next_to(ind_emitter, DOWN, MED_LARGE_BUFF)
        )
        dddd = (
            MathTex(r"\cdots").rotate(PI / 2).next_to(ind_collector, UP, MED_LARGE_BUFF)
        )

        g = Group(
            ind1,
            l1,
            cap1,
            choke_line,
            vb_line,
            vb_term,
            bjt,
            vb,
            le,
            ind_emitter,
            lc,
            ind_collector,
            load,
            ll,
            dddr,
            dddl,
            dddu,
            dddd,
        )

        self.camera.frame.scale_to_fit_height(g.height * 1.1).move_to(g)

        self.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.15))

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.9).move_to(bjt),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.1) for m in g[:6]],
                    *[m.animate.set_stroke(opacity=0.1) for m in g[7:]],
                    *[m.animate.set_opacity(0.1) for m in [dddr, dddl, dddu, dddd, vb]],
                ),
                *[m.animate.set_color(GREEN) for m in bjt.main_body],
                lag_ratio=0.2,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        ax = (
            Axes(
                x_range=[0, 1, 1],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=bjt.width * 3,
                y_length=bjt.height * 1.3,
            )
            .set_opacity(1)
            .next_to(bjt, RIGHT, LARGE_BUFF * 2)
        )

        A = VT(0)
        plot_opacity = VT(0)
        plot = always_redraw(
            lambda: ax.plot(
                lambda t: np.clip(~A * np.sin(2 * PI * 5 * t), -1, 1),
                x_range=[0, 1, 1 / 200],
                color=OUTPUT_COLOR,
                stroke_opacity=~plot_opacity,
                use_smoothing=False,
            )
        )
        self.add(plot)

        top_lim = DashedLine(
            ax.c2p(0, 1),
            ax.c2p(1, 1),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=RED,
        )
        bot_lim = DashedLine(
            ax.c2p(0, -1),
            ax.c2p(1, -1),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=RED,
        )
        top_bez = CubicBezier(
            bjt.get_right() + [0.1, 0, 0],
            bjt.get_right() + [1, 0, 0],
            top_lim.get_start() + [-1, 0, 0],
            top_lim.get_start() + [0, 0, 0],
        )
        bot_bez = CubicBezier(
            bjt.get_right() + [0.1, 0, 0],
            bjt.get_right() + [1, 0, 0],
            bot_lim.get_start() + [-1, 0, 0],
            bot_lim.get_start() + [0, 0, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(Group(bjt, ax)),
                AnimationGroup(Create(top_bez), Create(bot_bez)),
                AnimationGroup(Create(top_lim), Create(bot_lim)),
                plot_opacity @ 1,
                A @ 1,
                lag_ratio=0.3,
            )
        )

        # self.wait(0.5)

        # self.play(A @ 2)

        self.wait(0.5)
        self.play(
            LaggedStart(
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0) for m in g[:6]],
                    *[m.animate.set_stroke(opacity=0) for m in g[7:]],
                    *[m.animate.set_opacity(0) for m in [dddr, dddl, dddu, dddd, vb]],
                ),
                self.camera.frame.animate.set_x(bjt.get_x()),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        fet = (
            Fet(
                width,
                stroke_width_mult=stroke_width_mult,
                mode="enhancement",
                channel="n",
            )
            .next_to(bjt, UP)
            .shift(UP * fh(self))
        )
        other = (
            Text("...", font=FONT)
            .rotate(PI / 2)
            .next_to(bjt, DOWN)
            .shift(DOWN * fh(self))
        )
        types = (
            Group(fet.copy(), bjt.copy().set_color(WHITE), other.copy())
            .arrange(DOWN, MED_LARGE_BUFF)
            .move_to(bjt)
            .shift(LEFT / 2)
        )

        lbrace = BraceBetweenPoints(
            types.get_corner(UL), types.get_corner(DL), color=YELLOW
        )
        rbrace = BraceBetweenPoints(
            types.get_corner(UR), types.get_corner(DR), RIGHT, color=YELLOW
        )

        self.play(
            LaggedStart(
                Transform(fet, types[0]),
                Transform(bjt, types[1]),
                Transform(other, types[2]),
                FadeIn(lbrace),
                FadeIn(rbrace),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        bias = Text("bias", font=FONT).next_to(lbrace, LEFT).shift(UP * 2 + LEFT)
        bias_bez = CubicBezier(
            lbrace.get_left() + [0, 0, 0],
            lbrace.get_left() + [-1, 0, 0],
            bias.get_right() + [1, 0, 0],
            bias.get_right() + [0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                Create(bias_bez),
                Write(bias),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.3).move_to(ax),
        )

        self.wait(0.5)

        self.play(A @ 4, run_time=4)

        self.wait(0.5)

        harmonic = Text("harmonics", font=FONT)
        distortion = Text("distortion", font=FONT)
        etc = Text("etc.", font=FONT)

        effects = (
            Group(harmonic, distortion, etc)
            .arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
            .next_to(ax, RIGHT, LARGE_BUFF)
        )
        effects_bez_u = CubicBezier(
            ax.get_right() + [0.1, 0, 0],
            ax.get_right() + [1, 0, 0],
            effects.get_corner(UL) + [-1, 0, 0],
            effects.get_corner(UL) + [-0.1, 0, 0],
        )
        effects_bez_d = CubicBezier(
            ax.get_right() + [0.1, 0, 0],
            ax.get_right() + [1, 0, 0],
            effects.get_corner(DL) + [-1, 0, 0],
            effects.get_corner(DL) + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(effects_bez_u),
                    Create(effects_bez_d),
                ),
                self.camera.frame.animate.scale(2).move_to(Group(ax, effects)),
                Write(effects[0]),
                Write(effects[1]),
                Write(effects[2]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(A @ 20, run_time=5)

        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(1000))

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


class BJT(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        width = self.camera.frame.width * 0.1
        spacing = width / 2
        stroke_width_mult = 1.5
        bjt = Bjt(width, stroke_width_mult=stroke_width_mult)

        cap1 = Capacitor(width, stroke_width_mult=stroke_width_mult).next_to(
            bjt, LEFT, spacing * 2
        )
        l1 = Line(
            cap1.get_right(),
            bjt.base,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        ind1: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(l1, DOWN, spacing)
        )
        choke_line = Line(
            ind1.get_top(),
            l1.get_midpoint(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        vb_line = Line(
            ind1.get_bottom(),
            ind1.get_bottom() + DOWN * spacing,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        vb_term: Circle = bjt.base.copy().set_color(RED).next_to(vb_line, DOWN, 0)
        vb = (
            MathTex(r"V_b")
            .scale_to_fit_width(ind1.width)
            .next_to(vb_term, DOWN, MED_SMALL_BUFF)
        )

        ind_emitter: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(bjt.emitter.get_end(), DOWN, spacing)
        )
        le = Line(
            bjt.emitter.get_end(),
            ind_emitter.get_top(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        lc = Line(
            bjt.collector.get_end(),
            bjt.collector.get_end() + UP * spacing,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        ind_collector: Inductor = (
            Inductor(width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(lc, UP, 0)
        )

        load = (
            Resistor(width=width, stroke_width_mult=stroke_width_mult)
            .rotate(PI / 2)
            .next_to(lc.get_midpoint(), RIGHT, spacing)
        )
        ll = Line(
            lc.get_midpoint(),
            load.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        dddr = MathTex(r"\cdots").next_to(load, RIGHT, MED_LARGE_BUFF)
        dddl = MathTex(r"\cdots").next_to(cap1, LEFT, MED_LARGE_BUFF)
        dddu = (
            MathTex(r"\cdots").rotate(PI / 2).next_to(ind_emitter, DOWN, MED_LARGE_BUFF)
        )
        dddd = (
            MathTex(r"\cdots").rotate(PI / 2).next_to(ind_collector, UP, MED_LARGE_BUFF)
        )

        g = Group(
            ind1,
            l1,
            cap1,
            choke_line,
            vb_line,
            vb_term,
            bjt,
            vb,
            le,
            ind_emitter,
            lc,
            ind_collector,
            load,
            ll,
            dddr,
            dddl,
            dddu,
            dddd,
        )

        self.camera.frame.scale_to_fit_height(g.height * 1.1).move_to(g)

        self.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.15))

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.9).move_to(bjt),
                AnimationGroup(
                    *[m.animate.set_stroke(opacity=0.2) for m in g[:6]],
                    *[m.animate.set_stroke(opacity=0.2) for m in g[7:]],
                    *[m.animate.set_opacity(0.2) for m in [dddr, dddl, dddu, dddd, vb]],
                ),
                *[m.animate.set_color(GREEN) for m in bjt.main_body],
                lag_ratio=0.2,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        active_label = (
            Text("active,\nnon-linear", font=FONT)
            .scale(0.7)
            .move_to(self.camera.frame.get_corner(UR))
            .shift(LEFT + DOWN)
        )
        active_label_bez = CubicBezier(
            bjt.get_right() + [0.1, 0, 0],
            bjt.get_right() + [1, 0, 0],
            active_label.get_left() + [-1, 0, 0],
            active_label.get_left() + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(
                    Group(bjt, active_label, active_label_bez)
                ),
                Create(active_label_bez),
                Write(active_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        bjt_new = Bjt(bjt.width, stroke_width_mult)
        fet = Fet(
            width, stroke_width_mult=stroke_width_mult, mode="enhancement", channel="n"
        )
        other = Text("...", font=FONT).rotate(PI / 2)

        transistor_types = (
            Group(bjt_new, fet, other)
            .arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
            .move_to(active_label, LEFT)
            .set_y(bjt.get_y())
        )

        types_bez_u = CubicBezier(
            bjt.get_right() + [0.1, 0, 0],
            bjt.get_right() + [1, 0, 0],
            transistor_types.get_corner(UL) + [-1, 0, 0],
            transistor_types.get_corner(UL) + [-0.1, 0.2, 0],
        )
        types_bez_d = CubicBezier(
            bjt.get_right() + [0.1, 0, 0],
            bjt.get_right() + [1, 0, 0],
            transistor_types.get_corner(DL) + [-1, 0, 0],
            transistor_types.get_corner(DL) + [-0.1, -0.2, 0],
        )

        bjt_label = Text("BJT", font=FONT).next_to(bjt_new, RIGHT, MED_LARGE_BUFF)
        fet_label = Text("FET", font=FONT).next_to(fet, RIGHT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in active_label], lag_ratio=0.08),
                self.camera.frame.animate.set_y(bjt.get_y()),
                ReplacementTransform(active_label_bez, types_bez_u),
                FadeIn(bjt_new),
                Write(bjt_label),
                FadeIn(fet),
                Write(fet_label),
                FadeIn(other),
                Create(types_bez_d),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                FadeOut(
                    fet_label,
                    bjt_label,
                    g,
                    other,
                    types_bez_u,
                    types_bez_d,
                    bjt_new,
                ),
                self.camera.frame.animate.scale_to_fit_height(fet.height * 2).move_to(
                    fet
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


class FET(MovingCameraScene):
    def construct(self):
        width = self.camera.frame.width * 0.1
        spacing = width / 2
        stroke_width_mult = 1.5
        fet = Fet(
            width, stroke_width_mult=stroke_width_mult, mode="enhancement", channel="n"
        )
        self.add(fet)


class P1dB(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        amp = get_amp(width=fh(self, 0.3)).shift(DOWN * fh(self))
        gain_label = (
            Text(r"20 dB", font=FONT, color=GAIN_COLOR)
            .scale(1.5)
            .next_to(amp, DOWN, MED_LARGE_BUFF)
        )

        self.add(amp)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        self.play(FadeIn(gain_label))

        self.wait(0.5)

        pin = Arrow(
            amp.get_left() + LEFT * 4, amp.get_left(), buff=0, color=INPUT_COLOR
        )
        pout = Arrow(
            amp.get_right(), amp.get_right() + RIGHT * 4, buff=0, color=OUTPUT_COLOR
        )

        low_power_label = (
            Text("low power", font=FONT, color=INPUT_COLOR)
            .scale(0.7)
            .next_to(pin, UP, MED_SMALL_BUFF)
        )

        output_power_label = (
            Text("low power + 20 dB", font=FONT, color=OUTPUT_COLOR)
            .scale(0.7)
            .next_to(pout, UP, MED_SMALL_BUFF)
            .shift(RIGHT * 0.8)
        )

        self.play(
            LaggedStart(
                GrowArrow(pin),
                Write(low_power_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(pout),
                Write(output_power_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        small_signal_label = (
            Text('"small signal gain"', font=FONT)
            .scale(0.8)
            .next_to(gain_label, DOWN, MED_LARGE_BUFF)
        )

        ss_gain_bez_l = CubicBezier(
            gain_label.get_left() + [-0.1, 0, 0],
            gain_label.get_left() + [-1, 0, 0],
            small_signal_label.get_corner(UL) + [-0.1, 1, 0],
            small_signal_label.get_corner(UL) + [-0.1, 0.1, 0],
        )
        ss_gain_bez_r = CubicBezier(
            gain_label.get_right() + [0.1, 0, 0],
            gain_label.get_right() + [1, 0, 0],
            small_signal_label.get_corner(UR) + [0.1, 1, 0],
            small_signal_label.get_corner(UR) + [0.1, 0.1, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                AnimationGroup(
                    Create(ss_gain_bez_l),
                    Create(ss_gain_bez_r),
                ),
                Write(small_signal_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ax = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
            tips=False,
            axis_config=dict(
                stroke_width=DEFAULT_STROKE_WIDTH * 1.3,
            ),
        ).next_to(self.camera.frame.get_bottom(), DOWN, LARGE_BUFF * 2)

        x_ticks = np.arange(2, 12, 2).astype(int)
        y_ticks = np.arange(22, 32, 2).astype(int)

        ax = Axes(
            x_range=[0, 10, 1],
            y_range=[20, 30, 1],
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
            tips=False,
            x_axis_config=dict(
                numbers_with_elongated_ticks=x_ticks,
                include_numbers=True,
                numbers_to_include=x_ticks,
                font_size=DEFAULT_FONT_SIZE * 0.5,
                label_constructor=lambda x: Text(x, font=FONT),
                line_to_number_buff=MED_SMALL_BUFF,
                decimal_number_config=dict(num_decimal_places=0),
                longer_tick_multiple=2,
            ),
            y_axis_config=dict(
                numbers_with_elongated_ticks=y_ticks,
                include_numbers=True,
                numbers_to_include=y_ticks,
                font_size=DEFAULT_FONT_SIZE * 0.5,
                label_constructor=lambda x: Text(x, font=FONT),
                line_to_number_buff=MED_SMALL_BUFF,
                decimal_number_config=dict(num_decimal_places=0),
                longer_tick_multiple=2,
            ),
            axis_config=dict(
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                tick_size=0.1,
                # stroke_color=BLACK,
            ),
        )

        pin_sym_ax_label = Tex("| $P_{in}$")
        pin_ax_label = Text("Input Power", font=FONT).scale_to_fit_height(
            pin_sym_ax_label.height
        )
        pin_label_group = (
            Group(pin_ax_label, pin_sym_ax_label)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(ax, DOWN, MED_SMALL_BUFF)
            .shift(UP * 0.5)
        )

        pout_sym_ax_label = Tex("| $P_{out}$")
        pout_ax_label = Text("Output Power", font=FONT).scale_to_fit_height(
            pout_sym_ax_label.height
        )
        pout_label_group = (
            Group(pout_ax_label, pout_sym_ax_label)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .rotate(PI / 2)
            .next_to(ax, LEFT, MED_SMALL_BUFF)
            .shift(RIGHT * 0.5)
        )

        self.add(ax)

        ax.x_axis.numbers.set_opacity(0)
        ax.y_axis.numbers.set_opacity(0)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(ax),
                LaggedStart(
                    *[FadeIn(m) for m in [*pout_ax_label, *pout_sym_ax_label[0]]]
                ),
                LaggedStart(
                    *[FadeIn(m) for m in [*pin_ax_label, *pin_sym_ax_label[0]]]
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        x_labels = [m.copy().set_opacity(1) for m in ax.x_axis.numbers]
        y_labels = [m.copy().set_opacity(1) for m in ax.y_axis.numbers]

        self.play(
            LaggedStart(
                pin_label_group.animate.shift(DOWN * 0.5),
                LaggedStart(
                    *[FadeIn(m) for m in x_labels],
                    lag_ratio=0.2,
                ),
                pout_label_group.animate.shift(LEFT * 0.5),
                LaggedStart(
                    *[FadeIn(m) for m in y_labels],
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ideal_plot_x1 = VT(10)
        ideal_plot = always_redraw(
            lambda: ax.plot(
                lambda x: x + 20,
                color=GREEN,
                x_range=[0, ~ideal_plot_x1, 1 / 100],
            )
        )

        G = 20
        P_sat = 28.5

        actual_x1 = VT(0)

        actual_plot = always_redraw(
            lambda: ax.plot(
                lambda x: (
                    x + G + P_sat - np.logaddexp(0, 3 * (x + G - P_sat + 1)) / 3 - P_sat
                ),
                color=OUTPUT_COLOR,
                x_range=[0, ~actual_x1, 1 / 500],
            )
        )

        ideal_sym_label = Tex(r"Ideal $P_{out}(P_{in})$")
        ideal_legend = Line(ORIGIN, RIGHT, color=GREEN)
        ideal_label_group = (
            Group(ideal_legend, ideal_sym_label)
            .arrange(RIGHT, SMALL_BUFF)
            .next_to(self.camera.frame.get_corner(UR), DL)
            .shift(RIGHT * 2)
        )

        actual_sym_label = Tex(r"Actual $P_{out}(P_{in})$")
        actual_legend = Line(ORIGIN, RIGHT, color=OUTPUT_COLOR)
        actual_label_group = (
            Group(actual_legend, actual_sym_label)
            .arrange(RIGHT, SMALL_BUFF)
            .next_to(ideal_label_group, DOWN, MED_SMALL_BUFF, LEFT)
        )

        self.play(
            LaggedStart(
                Create(ideal_plot),
                self.camera.frame.animate.scale(1.1).set_x(
                    Group(ax, ideal_label_group).get_x()
                ),
                FadeIn(ideal_label_group),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.add(actual_plot)

        compression_box = DashedVMobject(
            Rectangle(width=ax.width * 0.35, height=ax.height * 0.35)
        ).move_to(ax.c2p(8.5, 28.5))

        self.play(actual_x1 @ 7)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(compression_box),
                actual_x1 @ 10,
                FadeIn(actual_label_group),
                lag_ratio=0.3,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        p1db_sym_label = Tex(r"$P_{1 \text{dB}}$ |")
        p1db_label = Text("1 dB Compression Point", font=FONT).scale_to_fit_height(
            p1db_sym_label.height
        )
        p1db_label_group = (
            Group(p1db_sym_label, p1db_label)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .move_to(self.camera.frame.get_top())
        )

        all_group = Group(
            ax,
            p1db_label_group,
            ideal_label_group,
            actual_label_group,
            pin_ax_label,
            pout_ax_label,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.1
                ).move_to(all_group),
                LaggedStart(
                    *[FadeIn(m) for m in [*p1db_sym_label[0], *p1db_label]],
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    compression_box.height * 1.2
                ).move_to(compression_box),
                FadeOut(compression_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ip1db = 8.2
        p_ideal = Dot(ax.i2gp(ip1db, ideal_plot))
        p_real = Dot(ax.i2gp(ip1db, actual_plot))
        ip1db_line = Line(p_real.get_center(), p_ideal.get_center())

        ip1db_label = (
            Text("1dB?", font=FONT)
            .scale(0.6)
            .next_to(ip1db_line, RIGHT, MED_SMALL_BUFF)
        )
        ip1db_label[-1].set_color(YELLOW)

        self.play(
            LaggedStart(
                Create(p_real),
                Create(ip1db_line),
                Create(p_ideal),
                LaggedStart(*[FadeIn(m) for m in ip1db_label[:-1]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(GrowFromCenter(ip1db_label[-1]))

        self.wait(0.5)

        self.play(
            FadeOut(ip1db_label[-1]),
            self.camera.frame.animate.scale_to_fit_height(
                all_group.height * 1.2
            ).move_to(all_group),
        )

        self.wait(0.5)

        new_amp = get_amp(width=fh(self, 0.15)).next_to(ax, RIGHT, LARGE_BUFF * 3)

        inp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=new_amp.width * 2,
            y_length=new_amp.height,
        )
        inp_ax.shift(new_amp.get_left() - inp_ax.c2p(1, 0))
        outp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=new_amp.width * 2,
            y_length=new_amp.height,
        )
        outp_ax.shift(new_amp.get_right() - outp_ax.c2p(0, 0))

        A = VT(0.15)
        f = 3
        G_new_amp = 10
        x1_in = VT(0)
        x1_out = VT(0)
        psat = 3.2
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
                lambda t: np.clip(~A * G_new_amp * np.sin(2 * PI * f * t), -psat, psat),
                x_range=[0, ~x1_out, 1 / 1000],
                color=OUTPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
        )
        self.add(inp, outp)

        self.next_section(skip_animations=skip_animations(True))

        amp_ax_group = Group(new_amp, ax, inp_ax, outp_ax)

        ip1db_vt = VT(8.2)
        p_ideal_vt = always_redraw(lambda: Dot(ax.i2gp(~ip1db_vt, ideal_plot)))
        p_real_vt = always_redraw(lambda: Dot(ax.i2gp(~ip1db_vt, actual_plot)))

        self.add(p_real_vt, p_ideal_vt)
        self.remove(p_real, p_ideal)

        self.camera.frame.save_state()

        self.play(
            LaggedStart(
                AnimationGroup(
                    *[
                        m.animate.set_opacity(0)
                        for m in [*actual_label_group, *ideal_label_group]
                    ]
                ),
                self.camera.frame.animate.move_to(amp_ax_group),
                LaggedStart(
                    GrowFromCenter(new_amp[0], rate_func=intro_elastic, run_time=1.4),
                    GrowFromCenter(new_amp[1], rate_func=intro_elastic, run_time=1.4),
                    lag_ratio=0.15,
                ),
                AnimationGroup(
                    ip1db_vt @ 4,
                    FadeOut(ip1db_line, ip1db_label[:-1]),
                ),
                x1_in @ 1,
                x1_out @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(A @ (~A * 3), ip1db_vt @ ip1db, run_time=8)

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(*new_amp),
                    x1_in @ 0,
                    x1_out @ 0,
                ),
                self.camera.frame.animate.restore(),
                AnimationGroup(
                    *[
                        m.animate.set_opacity(1)
                        for m in [*actual_label_group, *ideal_label_group]
                    ]
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        op1db_box = SurroundingRectangle(pout_label_group)
        ip1db_box = SurroundingRectangle(pin_label_group)

        op1db_line = DashedLine(
            [ax.c2p(0, 0)[0], ax.i2gp(ip1db, ideal_plot)[1], 0],
            ax.i2gp(ip1db, ideal_plot),
            dashed_ratio=0.6,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )

        ip1db_line = DashedLine(
            ax.c2p(ip1db, 20),
            ax.i2gp(ip1db, ideal_plot),
            dashed_ratio=0.6,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )

        op1db_label = (
            Text("OP1dB", font=FONT).scale(0.7).next_to(op1db_line.get_start(), UR)
        )
        ip1db_label = (
            Text("IP1dB", font=FONT).scale(0.7).next_to(ip1db_line.get_start(), UR)
        )

        self.play(Create(op1db_box))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(op1db_line),
                Write(op1db_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(op1db_box),
                Create(ip1db_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    op1db_line.animate.set_opacity(0.3),
                    op1db_label.animate.set_opacity(0.3),
                ),
                Create(ip1db_line),
                Write(ip1db_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(actual_label_group, ideal_label_group, p1db_label_group),
                Uncreate(op1db_line),
                FadeOut(op1db_label),
                Uncreate(ip1db_line),
                FadeOut(ip1db_label),
                self.camera.frame.animate.scale_to_fit_height(
                    compression_box.height * 1.5
                ).move_to(compression_box),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        psat_sym_label = Tex(r"$P_{\text{sat}}$ |")
        psat_label = Text("Saturated power", font=FONT).scale_to_fit_height(
            p1db_sym_label.height
        )
        psat_label_group = (
            Group(psat_sym_label, psat_label)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .scale(0.6)
            .move_to(self.camera.frame.get_center() + RIGHT * 2 + DOWN)
        )
        # self.add(psat_label_group)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ip1db_vt @ 18,
                    ideal_plot_x1 @ 18,
                    actual_x1 @ 18,
                    run_time=4,
                ),
                self.camera.frame.animate.shift(RIGHT * 2),
                LaggedStart(
                    *[FadeIn(m) for m in [*psat_sym_label[0], *psat_label]],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        # self.play(FadeOut(psat_label_group), actual_x1 @ 0, ideal_plot_x1 @ 0)

        number_plane = NumberPlane(
            ax.x_range, ax.y_range, ax.x_length, ax.y_length
        ).set_z_index(-10)
        number_plane.shift(ax.c2p(0, 0) - number_plane.c2p(0, 0))

        self.play(
            FadeOut(p_real_vt, p_ideal_vt, psat_label_group),
            self.camera.frame.animate.scale_to_fit_height(ax.height * 1.4).move_to(ax),
            ideal_plot_x1 @ 10,
            actual_x1 @ 10,
            Create(number_plane),
        )

        self.wait(0.5)

        # number_plane.x_lines[7].set_color(YELLOW)
        # number_plane.x_lines[8].set_color(YELLOW)

        self.play(
            LaggedStart(
                *[
                    number_plane.y_lines[idx].animate.set_opacity(0.2)
                    for idx in range(len(number_plane.y_lines))
                ],
                lag_ratio=0.08,
            ),
            LaggedStart(
                *[
                    number_plane.x_lines[idx].animate.set_opacity(0.2)
                    for idx in range(len(number_plane.x_lines))
                    if idx not in [7, 8]
                ],
                lag_ratio=0.08,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        # self.play(
        #     Uncreate(number_plane),
        #     actual_x1 @ 0,
        #     ideal_plot_x1 @ 0,
        #     FadeOut(ax),
        #     FadeOut(pin_label_group, shift=DOWN),
        #     FadeOut(pout_label_group, shift=LEFT),
        #     FadeOut(*x_labels, *y_labels, ip1db_box),
        # )

        self.remove(
            small_signal_label,
            pin,
            pout,
            gain_label,
            ss_gain_bez_l,
            ss_gain_bez_r,
            low_power_label,
            output_power_label,
            amp,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 1.5)))

        amp_3 = get_amp(width=fh(self, 0.15)).move_to(self.camera.frame)

        inp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp_3.width * 4,
            y_length=amp_3.height,
        )
        inp_ax.shift(amp_3.get_left() - inp_ax.c2p(1, 0))
        outp_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=amp_3.width * 4,
            y_length=amp_3.height,
        )
        outp_ax.shift(amp_3.get_right() - outp_ax.c2p(0, 0))

        A = VT(1)
        f = 3
        G_new_amp = 10
        x1_in = VT(0)
        x1_out = VT(0)
        psat = 3.2
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
                lambda t: np.clip(~A * G_new_amp * np.sin(2 * PI * f * t), -psat, psat),
                x_range=[0, ~x1_out, 1 / 1000],
                color=OUTPUT_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
        )
        outp_cam = outp_ax.plot(
            lambda t: psat * np.sin(2 * PI * f * t),
            # lambda t: np.clip(~A * G_new_amp * np.sin(2 * PI * f * t), -psat, psat),
            x_range=[0, 1, 1 / 1000],
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            use_smoothing=False,
        )
        self.add(inp, outp)

        self.play(
            LaggedStart(
                GrowFromCenter(amp_3[0], rate_func=intro_elastic, run_time=1),
                GrowFromCenter(amp_3[1], rate_func=intro_elastic, run_time=1),
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        pterm = Line(UP, DOWN)
        nterm = Line(UP, ORIGIN)
        batt = (
            Group(nterm, pterm)
            .scale(0.5)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(amp_3, UP, LARGE_BUFF * 2)
            .shift(LEFT)
        )
        to_nterm = CubicBezier(
            amp_3.get_top() + [-amp_3.width / 4, 0, 0],
            amp_3.get_top() + [-amp_3.width / 4 - 0.5, 1, 0],
            nterm.get_left() + [-1, 0, 0],
            nterm.get_left(),
        ).set_z_index(-1)
        to_pterm = CubicBezier(
            amp_3.get_top() + [amp_3.width / 4, 0, 0],
            amp_3.get_top() + [amp_3.width / 4, 1, 0],
            pterm.get_right() + [0.5, 0, 0],
            pterm.get_right(),
        ).set_z_index(-1)
        minus = Text("-", font=FONT, color=GRAY).next_to(nterm.get_top(), UL)
        plus = (
            Text("+", font=FONT, color=YELLOW)
            .next_to(pterm.get_top(), UR)
            .set_y(minus.get_y())
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP),
                Create(to_nterm),
                Create(to_pterm),
                Create(nterm),
                Create(pterm),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(plus),
                to_pterm.animate.set_color(YELLOW),
                GrowFromCenter(minus),
                to_nterm.animate.set_color(GRAY),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                x1_in @ 1,
                self.camera.frame.animate.scale(0.4).move_to(outp_ax.c2p(0, 0)),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        top_rail = DashedLine(
            outp_ax.c2p(0, psat + 0.08),
            outp_ax.c2p(1, psat + 0.08),
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )
        bot_rail = DashedLine(
            outp_ax.c2p(0, -(psat + 0.08)),
            outp_ax.c2p(1, -(psat + 0.08)),
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(top_rail), Create(bot_rail)),
                AnimationGroup(
                    x1_out @ 1,
                    MoveAlongPath(self.camera.frame, outp_cam),
                ),
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        supply_box = DashedVMobject(
            SurroundingRectangle(
                Group(batt, plus, minus), color=GREEN, buff=SMALL_BUFF
            ),
            dashed_ratio=0.6,
        )

        self.play(Create(supply_box))

        self.wait(0.5)

        width = self.camera.frame.width * 0.1
        stroke_width_mult = 1

        outp_line = Line(outp_ax.c2p(0, 0), outp_ax.c2p(1, 0)).set_z_index(-1)
        load = (
            Resistor(width=width, stroke_width_mult=stroke_width_mult)
            .next_to(outp_line.get_end(), DR, LARGE_BUFF)
            .set_z_index(-1)
        )
        outp_to_load = CubicBezier(
            outp_line.get_end() + [0, 0, 0],
            outp_line.get_end() + [1, 0, 0],
            load.get_top() + [0, 1, 0],
            load.get_top() + [0, 0, 0],
        )

        load_box = DashedVMobject(
            SurroundingRectangle(Group(load), color=GREEN, buff=SMALL_BUFF),
            dashed_ratio=0.6,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            self.camera.frame.animate(run_time=2)
            .shift(RIGHT * 2 + DOWN / 2)
            .scale(0.9),
            Succession(
                Create(outp_line, rate_func=rate_functions.ease_in_sine),
                Create(outp_to_load, rate_func=rate_functions.linear, run_time=0.5),
                GrowFromCenter(
                    load, rate_func=rate_functions.ease_out_sine, run_time=0.5
                ),
                ReplacementTransform(supply_box, load_box),
            ),
        )

        self.wait(0.5)

        ax_group = Group(
            number_plane,
            ax,
            pin_label_group,
            pout_label_group,
            *x_labels,
            *y_labels,
            ip1db_box,
        )
        self.play(
            *[
                number_plane.x_lines[idx].animate.set_opacity(0.2)
                for idx in range(len(number_plane.x_lines))
                if idx in [7, 8]
            ],
            self.camera.frame.animate.scale_to_fit_height(
                ax_group.height * 1.2
            ).move_to(ax_group),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        p1db_x = 7.7
        psat_x = 9
        psat_xline = Line(ax.c2p(psat_x, 20), ax.i2gp(psat_x, actual_plot), color=RED)
        p1db_xline = Line(ax.c2p(p1db_x, 20), ax.i2gp(p1db_x, actual_plot), color=RED)
        p1db_dot = Dot(ax.i2gp(p1db_x, actual_plot), color=RED)
        psat_dot = Dot(ax.i2gp(psat_x, actual_plot), color=RED)

        area_x1 = VT(p1db_x)
        area_opacity = VT(0.3)

        area = always_redraw(
            lambda: ax.get_area(
                actual_plot,
                x_range=[p1db_x, ~area_x1],
                color=RED,
                opacity=~area_opacity,
                stroke_width=0,
                bounded_graph=ax.plot(lambda _: 20, x_range=[0, 10, 1 / 500]),
            ).set_z_index(-1)
        )

        self.add(area)

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                Create(psat_xline, rate_func=in_out_elastic),
                GrowFromCenter(psat_dot, rate_func=in_out_elastic),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(p1db_xline, rate_func=in_out_elastic),
                GrowFromCenter(p1db_dot, rate_func=in_out_elastic),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(area_x1 @ psat_x, run_time=3)

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    p1db_dot.animate.set_opacity(0.2),
                    p1db_xline.animate.set_opacity(0.2),
                ),
                area_opacity @ 0.1,
                self.camera.frame.animate.set_x(psat_dot.get_x()),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).move_to(amp_3),
                LaggedStart(
                    AnimationGroup(Unwrite(plus), FadeOut(load_box)),
                    Unwrite(minus),
                    AnimationGroup(Uncreate(pterm), Uncreate(nterm)),
                    ShrinkToCenter(load),
                    AnimationGroup(
                        Uncreate(outp_to_load),
                        Uncreate(to_nterm),
                        Uncreate(to_pterm),
                    ),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class WhoCares(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        ax = Axes(
            x_range=[0, 10, 1],
            y_range=[20, 30, 1],
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
            tips=False,
        )

        G = 20
        P_sat = 28.5

        actual_x1 = VT(0)

        actual_plot = always_redraw(
            lambda: ax.plot(
                lambda x: (
                    x + G + P_sat - np.logaddexp(0, 3 * (x + G - P_sat + 1)) / 3 - P_sat
                ),
                color=OUTPUT_COLOR,
                x_range=[0, ~actual_x1, 1 / 1000],
            )
        )
        pae_plot = always_redraw(
            lambda: ax.plot(
                lambda x: (
                    20
                    + 0.08 * x
                    + 7.2 / (1 + np.exp(-1.3 * (x - 6.2)))
                    - 0.10 * np.clip(x - 8.4, 0, None) ** 2
                ),
                color=PAE_COLOR,
                x_range=[0, ~actual_x1, 1 / 1000],
            )
        )

        actual_sym_label = Tex(r"| $P_{out}(P_{in})$")
        actual_label = Text("Power", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5)
        actual_legend = Line(ORIGIN, RIGHT / 2, color=OUTPUT_COLOR)
        actual_label_group = (
            Group(actual_legend, actual_label, actual_sym_label)
            .arrange(RIGHT, SMALL_BUFF)
            .scale(0.7)
            .next_to(self.camera.frame.get_corner(UR), DL)
            .shift(LEFT * 0.8 + DOWN * 0.3)
        )

        pae_sym_label = Tex(r"| $\mathrm{PAE}(P_{in})$")
        pae_label = Text("Efficiency", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5)
        pae_legend = Line(ORIGIN, RIGHT / 2, color=PAE_COLOR)
        pae_label_group = (
            Group(pae_legend, pae_label, pae_sym_label)
            .arrange(RIGHT, SMALL_BUFF)
            .scale(0.7)
            .next_to(actual_label_group, DOWN, SMALL_BUFF, LEFT)
        )

        region_boundary = P_sat - G - 1.5
        linear_region = always_redraw(
            lambda: Polygon(
                ax.c2p(0, 0 + 20),
                ax.c2p(0, 10 + 20),
                ax.c2p(min(region_boundary, ~actual_x1), 10 + 20),
                ax.c2p(min(region_boundary, ~actual_x1), 0 + 20),
                fill_color=GREEN,
                fill_opacity=0.2,
                stroke_opacity=0,
            )
        )
        sat_region = always_redraw(
            lambda: Polygon(
                ax.c2p(region_boundary, 0 + 20),
                ax.c2p(region_boundary, 10 + 20),
                ax.c2p(max(region_boundary, ~actual_x1), 10 + 20),
                ax.c2p(max(region_boundary, ~actual_x1), 0 + 20),
                fill_color=BLUE,
                fill_opacity=0.2,
                stroke_opacity=0,
            )
        )

        linear_region_label = (
            Paragraph("Linear", "Region", alignment="center", font=FONT)
            .scale(0.4)
            .next_to(ax.c2p(region_boundary / 2, 30), DOWN, SMALL_BUFF)
        )
        sat_region_label = (
            Paragraph("Saturation", "Region", alignment="center", font=FONT)
            .scale(0.4)
            .next_to(
                ax.c2p((10 - region_boundary) / 2 + region_boundary, 30),
                DOWN,
                SMALL_BUFF,
            )
        )

        self.add(actual_plot, pae_plot, linear_region, sat_region)

        self.play(
            Create(ax),
            FadeIn(actual_label_group),
            FadeIn(pae_label_group),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                actual_x1.animate(run_time=8).set_value(10),
                FadeIn(linear_region_label),
                FadeIn(sat_region_label),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        comms = (
            Paragraph(
                "LTE",
                "WiFi",
                alignment="center",
                font=FONT,
                line_spacing=MED_LARGE_BUFF,
            )
            .next_to(ax, LEFT, LARGE_BUFF * 2.5)
            .shift(UP * ax.height * 0.25)
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).shift(LEFT),
                GrowFromCenter(comms[0]),
                GrowFromCenter(comms[1]),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            Broadcast(
                Circle(comms.width * 2),
                focal_point=comms.get_center(),
                initial_width=comms.width * 1.2,
            )
        )

        self.wait(0.5)

        select_linear = SurroundingRectangle(linear_region, buff=0)
        select_sat = SurroundingRectangle(sat_region, buff=0)

        self.play(Create(select_linear))

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).next_to(comms, DOWN, SMALL_BUFF)

        self.play(
            LaggedStart(
                comms.animate.scale(0.7).shift(UP).set_opacity(0.2),
                radar.get_animation(),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        for x in radar.vgroup:
            x.set_z_index(1)

        radar_background = (
            SurroundingRectangle(
                Group(radar.left_leg, radar.right_leg, radar.middle_leg),
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
                stroke_opacity=0,
            )
            .stretch(2, 1)
            .move_to(
                Group(radar.left_leg, radar.right_leg, radar.middle_leg),
                aligned_edge=UP,
            )
            .set_z_index(0)
        )
        self.add(radar_background)
        self.play(
            LaggedStart(
                Broadcast(
                    Circle(radar.radome.radius * 3.5, color=YELLOW).set_z_index(-1),
                    focal_point=radar.radome.get_center(),
                    initial_width=radar.radome.radius * 1.08,
                ),
                ReplacementTransform(select_linear, select_sat),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(radar.vgroup, comms),
                self.camera.frame.animate.restore(),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rightarrow = MathTex(r"\Rightarrow")
        lnot = MathTex(r"\lnot")
        linearity_headroom = (
            Group(
                Text("Linearity", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6),
                rightarrow.copy(),
                Text("Headroom", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6),
                rightarrow.copy(),
                lnot.copy(),
                Text("Efficiency", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6),
            )
            .arrange(RIGHT, MED_SMALL_BUFF, aligned_edge=UP)
            .next_to(ax, DOWN, LARGE_BUFF)
        )
        linearity_headroom[1].set_y(linearity_headroom.get_y())
        linearity_headroom[3].set_y(linearity_headroom.get_y())
        linearity_headroom[4].set_y(linearity_headroom.get_y())

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.3).shift(DOWN),
                LaggedStart(
                    *[
                        GrowFromCenter(m)
                        for m in [
                            *linearity_headroom[0],
                            *linearity_headroom[1],
                            *linearity_headroom[2],
                        ]
                    ],
                    lag_ratio=0.03,
                ),
                LaggedStart(
                    *[
                        GrowFromCenter(m)
                        for m in [
                            *linearity_headroom[3],
                            *linearity_headroom[4],
                            *linearity_headroom[5],
                        ]
                    ],
                    lag_ratio=0.03,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


# TODO: re-render
class Practical(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        title = (
            Text("Practical\nConsiderations", font=FONT)
            .scale(0.5)
            .next_to(self.camera.frame.get_corner(UL), DR, MED_SMALL_BUFF)
        ).set_z_index(2)
        title_box = (
            SurroundingRectangle(
                title,
                buff=MED_LARGE_BUFF,
                corner_radius=0.2,
                color=GREEN,
            )
            .shift(UL * MED_SMALL_BUFF)
            .set_z_index(2)
        )

        self.play(
            LaggedStart(
                Write(title),
                Create(title_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        npages = 13
        pages = Group(
            *[
                ImageMobject(
                    f"../14_amp_gain/static/adl8154-{idx:02}.png"
                ).scale_to_fit_height(fh(self, 0.6))
                # .rotate(-theta_step * npages / 2 + idx * theta_step)
                for idx in range(1, npages + 1)
            ]
        ).arrange(DOWN, SMALL_BUFF)
        # pages.next_to(self.camera.frame.get_right(), LEFT, LARGE_BUFF * 1.5)
        pages.set_x(self.camera.frame.get_x())
        pages.shift(UP * (self.camera.frame.get_center() - pages[0].get_center()))

        # self.add(pages)

        self.play(
            pages.shift(DOWN * fh(self, 2)).animate.shift(UP * fh(self, 2)),
            rate_func=in_out_elastic,
            run_time=3,
        )

        self.wait(0.5)

        desired_page = 11

        page_height = (pages[0].get_center() - pages[1].get_center())[1]
        self.play(
            pages.animate.shift(UP * page_height * 11),
            rate_func=in_out_elastic,
            run_time=6,
        )

        self.wait(0.5)

        center_dot = Dot(
            pages[desired_page].get_center()
            + RIGHT * pages[desired_page].width * 0.24
            + UP * 0.15,
            # radius=DEFAULT_DOT_RADIUS * 0.25,
            color=RED,
        )
        # self.add(center_dot)
        box_u = Rectangle(
            width=pages[0].width * 2,
            height=pages[0].width,
            color=RED,
            stroke_opacity=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=0.7,
        ).next_to(center_dot, UP, pages[0].height * 0.09)
        box_d = Rectangle(
            width=pages[0].width * 2,
            height=pages[0].width,
            color=RED,
            stroke_opacity=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=0.7,
        ).next_to(center_dot, DOWN, pages[0].height * 0.13)
        box_l = (
            Rectangle(
                width=pages[0].width * 2,
                height=(box_u.get_bottom() - box_d.get_top())[1] * 1.0007,
                color=RED,
                stroke_opacity=0,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=0.7,
            )
            .next_to(center_dot, LEFT, pages[0].width * 0.2)
            .set_y(Group(box_u, box_d).get_y())
        )
        box_r = (
            Rectangle(
                width=pages[0].width * 2,
                height=(box_u.get_bottom() - box_d.get_top())[1] * 1.0007,
                color=RED,
                stroke_opacity=0,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=0.7,
            )
            .next_to(center_dot, RIGHT, pages[0].width * 0.2)
            .set_y(Group(box_u, box_d).get_y())
        )

        background = Cutout(
            BackgroundRectangle(self.camera.frame),
            *[BackgroundRectangle(m) for m in pages],
            color=BLACK,
            stroke_opacity=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        ).set_z_index(1)
        self.add(background)

        self.next_section(skip_animations=skip_animations(True))

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                box_u.shift(UP * 5).animate.shift(DOWN * 5),
                box_l.shift(LEFT * 5).animate.shift(RIGHT * 5),
                box_d.shift(DOWN * 5).animate.shift(UP * 5),
                box_r.shift(RIGHT * 5).animate.shift(LEFT * 5),
                self.camera.frame.animate.scale(0.25).move_to(center_dot),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        room_temp = (
            Rectangle(
                width=fw(self, 0.07),
                height=center_dot.radius * 0.7,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.25,
                color=GREEN,
            )
            .move_to(center_dot)
            .shift(LEFT * fw(self, 0.11) + DOWN * fh(self, 0.12))
        )

        self.play(Create(room_temp))

        self.wait(0.5)

        self.play(
            Transform(
                room_temp,
                Rectangle(
                    width=fw(self, 0.08),
                    height=center_dot.radius * 0.7,
                    stroke_width=DEFAULT_STROKE_WIDTH * 0.25,
                    color=RED,
                )
                .move_to(center_dot)
                .shift(LEFT * fw(self, 0.11) + DOWN * fh(self, 0.17)),
            )
        )

        self.wait(0.5)

        self.play(
            Transform(
                room_temp,
                Rectangle(
                    width=fw(self, 0.07),
                    height=center_dot.radius * 0.7,
                    stroke_width=DEFAULT_STROKE_WIDTH * 0.25,
                    color=BLUE,
                )
                .move_to(center_dot)
                .shift(LEFT * fw(self, 0.11) + DOWN * fh(self, 0.07)),
            )
        )

        self.wait(0.5)

        self.play(
            Transform(
                room_temp,
                Rectangle(
                    width=fw(self, 0.33),
                    height=center_dot.radius * 1.5,
                    stroke_width=DEFAULT_STROKE_WIDTH * 0.25,
                    color=GREEN,
                )
                .move_to(center_dot)
                .shift(RIGHT * fw(self, 0.01) + DOWN * fh(self, 0.22)),
            )
        )

        self.wait(0.5)

        clip = (
            VideoMobject("/home/marchall/Downloads/clip.mkv")
            .scale_to_fit_height(fh(self))
            .next_to(self.camera.frame, RIGHT, LARGE_BUFF)
        ).set_z_index(5)
        clip_box = SurroundingRectangle(clip, buff=0, color=GREEN).set_z_index(6)
        self.add(clip, clip_box)

        clip_bez = CubicBezier(
            room_temp.get_right() + [0.1, 0, 0],
            room_temp.get_right() + [1, 0, 0],
            clip.get_left() + [-1, 0, 0],
            clip.get_left() + [-0.1, 0, 0],
            color=GREEN,
        ).set_z_index(5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(2.2).move_to(Group(clip, room_temp)),
                Create(clip_bez),
                lag_ratio=0.3,
            )
        )

        self.wait(6)

        self.play(
            self.camera.frame.animate.restore(),
            Uncreate(clip_bez),
            Uncreate(room_temp),
            FadeOut(box_l, box_d, box_r, box_u),
            Group(clip, clip_box).animate.shift(RIGHT * fw(self)),
        )
        self.remove(background)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pages.animate.shift(DOWN * page_height * 9),
            rate_func=in_out_elastic,
            run_time=6,
        )

        self.wait(0.5)

        system = (
            ImageMobject("./static/shipping_container.png")
            .scale_to_fit_width(fw(self, 0.3))
            .next_to(pages[2], RIGHT, LARGE_BUFF * 2)
            .shift(DOWN * 1.5)
        )
        antenna_u = (
            ImageMobject("../props/static/dish_antenna.png")
            .scale_to_fit_width(system.width * 0.6)
            .next_to(system, UP, -LARGE_BUFF * 1.1)
        ).set_z_index(-3)
        antenna_d = (
            ImageMobject("../props/static/dish_antenna.png")
            .scale_to_fit_width(system.width * 0.8)
            .flip(UP + RIGHT)
            .next_to(system, DOWN, 0)
        )

        spec1_bez = CubicBezier(
            pages[2].get_center() + [-pages[2].width / 4, pages[2].height * 0.1, 0],
            pages[2].get_center() + [-pages[2].width / 4 + 1, pages[2].height * 0.1, 0],
            system.get_left() + [-1, 0, 0],
            system.get_left() + [-0.1, 0, 0],
            color=GREEN,
        )

        spec2_bez = CubicBezier(
            pages[2].get_center() + [-pages[2].width / 3, pages[2].height * 0.15, 0],
            pages[2].get_center()
            + [-pages[2].width / 3 + 1, pages[2].height * 0.15, 0],
            system.get_left() + [-1, 0, 0],
            system.get_left() + [-0.1, 0, 0],
            color=GREEN,
        )

        spec3_bez = CubicBezier(
            pages[2].get_center() + [-pages[2].width / 4, pages[2].height * 0.2, 0],
            pages[2].get_center() + [-pages[2].width / 4 + 2, pages[2].height * 0.2, 0],
            system.get_left() + [-1, 0, 0],
            system.get_left() + [-0.1, 0, 0],
            color=GREEN,
        )

        self.play(
            LaggedStart(
                FadeOut(title),
                FadeOut(title_box),
                *[FadeOut(m) for m in [*pages[:2], *pages[3:]]],
                self.camera.frame.animate.scale(0.8).move_to(Group(system, pages[2])),
                Create(spec1_bez),
                Create(spec2_bez),
                Create(spec3_bez),
                system.shift(RIGHT * 5).animate.shift(LEFT * 5),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        u_y = antenna_u.get_y()
        self.add(antenna_u)
        self.play(
            LaggedStart(
                antenna_u.set_z_index(-3).move_to(system).animate.set_y(u_y),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        lb = Text("link budget", font=FONT)
        psu = Text("PSU", font=FONT)
        therm = Text("thermal", font=FONT)
        Group(lb, psu, therm).arrange(DOWN, LARGE_BUFF).next_to(
            system, RIGHT, LARGE_BUFF * 2
        ).set_y(self.camera.frame.get_y())
        psu.shift(LEFT)

        bezs = [
            CubicBezier(
                system.get_right() + [0.1, 0, 0],
                system.get_right() + [1, 0, 0],
                m.get_left() + [-1, 0, 0],
                m.get_left() + [-0.1, 0, 0],
                color=GREEN,
            )
            for m in [lb, psu, therm]
        ]

        self.remove(clip, clip_box)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).shift(RIGHT * fw(self, 0.5)),
                *[
                    LaggedStart(Create(m), Write(t), lag_ratio=0.3)
                    for m, t in zip(bezs, [lb, psu, therm])
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        nb0 = ImageMobject("./static/nb0.png").scale_to_fit_height(fh(self, 0.7))
        nb1 = ImageMobject("./static/nb1.png").scale_to_fit_height(fh(self, 0.7))
        nb_group = (
            Group(nb0, nb1)
            .arrange(RIGHT, LARGE_BUFF)
            .scale_to_fit_width(fw(self, 0.8))
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
        )
        nb_title = Text("op1db_psat.ipynb", font=FONT).next_to(nb_group, UP, LARGE_BUFF)
        Group(nb_title, nb_group).move_to(self.camera.frame).shift(DOWN * fh(self))

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self)),
                Write(nb_title),
                GrowFromCenter(nb0),
                GrowFromCenter(nb1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(nb_title),
                FadeOut(nb0),
                FadeOut(nb1),
                lag_ratio=0.2,
            )
        )

        comment = Text(
            "I don't believe that man has ever been to medical school", font=FONT
        )
        send_comment = Text("Send", font=FONT).next_to(comment, RIGHT, LARGE_BUFF * 1.2)
        send_comment_box = SurroundingRectangle(
            send_comment,
            buff=MED_SMALL_BUFF,
            corner_radius=0.3,
            color=DARK_BLUE,
            fill_color=DARK_BLUE,
            fill_opacity=1,
        )

        comments = SurroundingRectangle(
            Group(comment, send_comment_box, send_comment),
            corner_radius=0.7,
            buff=MED_LARGE_BUFF,
            color=DARK_BLUE,
        )
        send_comment_group = Group(send_comment_box, send_comment)
        comment_group = (
            Group(comments, comment, send_comment_group)
            .scale_to_fit_width(fw(self, 0.8))
            .move_to(self.camera.frame.copy().shift(DOWN * fh(self, 1)))
        )
        self.next_section(skip_animations=skip_animations(False))
        self.add(comments, send_comment_box, send_comment)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 1)))

        cursor = Rectangle(
            color=GREY_A,
            fill_color=GREY_A,
            fill_opacity=1.0,
            height=comment.height,
            width=comment[0].width,
        ).move_to(comment[0])

        self.play(
            TypeWithCursor(comment, cursor, leave_cursor_on=False),
            run_time=3,
            rate_func=cubic_bezier(0.181, 0.524, 0.744, 0.203),
        )
        # self.play(Blink(cursor, blinks=2))
        # self.play(FadeOut(cursor, run_time=0.5))

        self.wait(0.5)

        comment_opacity = VT(0)
        comment_bubble = always_redraw(
            lambda: SurroundingRectangle(
                comment,
                color=DARK_BLUE,
                stroke_opacity=~comment_opacity,
                fill_color=DARK_BLUE,
                fill_opacity=~comment_opacity,
                corner_radius=0.2,
            ).move_to(comment)
        )
        self.add(comment_bubble)

        self.play(
            LaggedStart(
                send_comment_group.animate(
                    rate_func=rate_functions.there_and_back, run_time=0.6
                ).scale(0.7),
                comment_opacity @ 1,
                comment.animate(run_time=4)
                .next_to(self.camera.frame.get_top(), UP, LARGE_BUFF)
                .set_x(comment.get_x()),
            )
        )

        self.wait(0.5)

        self.play(
            FadeIn(
                BackgroundRectangle(
                    self.camera.frame,
                    color=BACKGROUND_COLOR,
                    fill_opacity=1,
                    stroke_opacity=0,
                ).set_z_index(10)
            )
        )

        self.wait(2)


class Plug(MovingCameraScene):
    def construct(self):
        mems = (
            Text("Channel memberships", font=FONT)
            .scale(0.7)
            .next_to(self.camera.frame.get_top(), DOWN, MED_LARGE_BUFF)
        )

        self.play(Write(mems))

        self.wait(0.5)

        paview_vid = (
            VideoMobject("/home/marchall/media/obs/paview_overview.mp4", speed=2)
            .scale_to_fit_width(fw(self, 0.7))
            .next_to(mems, DOWN, MED_LARGE_BUFF)
        )
        paview_box = SurroundingRectangle(paview_vid, buff=0, color=GREEN)
        paview = Group(paview_vid, paview_box)

        self.play(paview.shift(DOWN * fh(self)).animate.shift(UP * fh(self)))

        self.wait(10)

        walkthrough_vid = (
            VideoMobject("/home/marchall/media/obs/walkthrough_clip.mp4", speed=1)
            .scale_to_fit_width(fw(self, 0.7))
            .next_to(mems, DOWN, MED_LARGE_BUFF)
        )
        walkthrough_box = SurroundingRectangle(walkthrough_vid, buff=0, color=GREEN)
        walkthrough = Group(walkthrough_vid, walkthrough_box)

        self.play(
            paview.animate.scale_to_fit_width(fw(self, 0.45)).next_to(
                self.camera.frame.get_left(), RIGHT
            )
        )

        self.play(walkthrough.shift(DOWN * fh(self)).animate.shift(UP * fh(self)))
        self.play(
            walkthrough.animate.scale_to_fit_width(fw(self, 0.45)).next_to(
                self.camera.frame.get_right(), LEFT
            )
        )

        self.wait(10)

        in_the_description = (
            Text("In the description", font=FONT)
            .scale(0.7)
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self) + DOWN * 2)
        )
        self.add(in_the_description)

        source_code = (
            Text("// source code", font=FONT, color=GRAY)
            .scale(0.5)
            .next_to(in_the_description, UR, LARGE_BUFF)
            .shift(UP * 2)
        )
        resources = (
            Text("Resources", font=FONT)
            .scale(0.5)
            .next_to(in_the_description, UL, LARGE_BUFF)
            .shift(UP * 3)
        )

        mugs = (
            ImageMobject("./static/mug.png")
            .scale_to_fit_height(fh(self, 0.3))
            .next_to(self.camera.frame.get_corner(DL), UR)
            .shift(RIGHT / 2 + UP + DOWN * fh(self))
        )

        walkthrough_bez = CubicBezier(
            walkthrough.get_bottom() + [0, -0.1, 0],
            walkthrough.get_bottom() + [0, -3, 0],
            in_the_description.get_top() + [0, 3, 0],
            in_the_description.get_top() + [0, 0.1, 0],
        )

        paview_bez = CubicBezier(
            paview.get_bottom() + [0, -0.1, 0],
            paview.get_bottom() + [0, -3, 0],
            in_the_description.get_top() + [0, 3, 0],
            in_the_description.get_top() + [0, 0.1, 0],
        )

        source_code_bez = CubicBezier(
            source_code.get_bottom() + [0, -0.1, 0],
            source_code.get_bottom() + [0, -1, 0],
            in_the_description.get_top() + [in_the_description.width * 0.3, 1, 0],
            in_the_description.get_top() + [in_the_description.width * 0.3, 0.1, 0],
        )

        resources_bez = CubicBezier(
            resources.get_bottom() + [0, -0.1, 0],
            resources.get_bottom() + [0, -2, 0],
            in_the_description.get_top() + [-in_the_description.width / 4, 1.5, 0],
            in_the_description.get_top() + [-in_the_description.width / 4, 0.1, 0],
        )

        mugs_bez = CubicBezier(
            mugs.get_bottom() + [0.5, -0.1, 0],
            mugs.get_bottom() + [0.5, -1.5, 0],
            in_the_description.get_bottom() + [0, -1.5, 0],
            in_the_description.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                Create(paview_bez),
                Create(walkthrough_bez),
                self.camera.frame.animate.shift(DOWN * fh(self)),
                Write(source_code),
                Write(resources),
                Create(resources_bez),
                Create(source_code_bez),
                GrowFromCenter(mugs),
                Create(mugs_bez),
                lag_ratio=0.2,
            )
        )
        self.remove(walkthrough, paview)

        self.wait(0.5)

        stats_title = Text("Stats for Nerds", font=FONT).scale(0.7)
        stats_table = (
            Table(
                [
                    ["Lines of code", "5,822"],
                    ["Script word count", "1,959"],
                    ["Days to make", "22"],
                    ["Git commits", "15"],
                ],
                element_to_mobject=Text,
                element_to_mobject_config=dict(
                    font=FONT, font_size=DEFAULT_FONT_SIZE * 0.7
                ),
            )
            .scale(0.5)
            .next_to(stats_title, direction=DOWN, buff=MED_LARGE_BUFF)
        )
        for row in stats_table.get_rows():
            row[1].set_color(GREEN)

        stats_group = (
            VGroup(stats_title, stats_table)
            .next_to(self.camera.frame.get_corner(UR), DL, LARGE_BUFF * 2)
            .shift(DOWN * fh(self))
        )

        thank_you_sabrina = Text(
            "Thank you, Sabrina, for\nediting the whole video :)",
            font=FONT,
            font_size=DEFAULT_FONT_SIZE * 0.5,
        ).next_to(stats_group, DOWN)

        marshall_bruner = (
            Text("Marshall Bruner", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5)
            .set_y(thank_you_sabrina.get_y())
            .set_x(self.camera.frame.get_x() - fw(self, 0.25))
        )

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))
        self.play(
            LaggedStart(
                FadeIn(marshall_bruner, shift=UP),
                AnimationGroup(FadeIn(stats_title, shift=DOWN), FadeIn(stats_table)),
                Write(thank_you_sabrina),
                lag_ratio=0.9,
                run_time=4,
            )
        )

        self.wait(2)
