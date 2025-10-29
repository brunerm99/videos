# pulse_compression.py

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


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


def chirp_pulse(t_val, pulse_start, pulse_width, f0, f1, amp, phase, ramp="quadratic"):
    t_rel = t_val - pulse_start - 1 / f0 / 4

    if -1 / f0 / 4 <= t_rel <= pulse_width:
        return amp * signal.chirp(
            t_rel,
            f0=f0,
            t1=pulse_width,
            f1=f1,
            method=ramp,
            phi=phase,
        )
    else:
        return 0


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        f1 = VT(1.5)
        f2 = VT(2.7)
        power_norm_1 = VT(-5)
        power_norm_2 = VT(-5)

        stop_time = 8
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        noise_mu = 0
        noise_sigma_db = -10
        noise_sigma = 10 ** (noise_sigma_db / 10)

        np.random.seed(0)
        noise = np.random.normal(loc=noise_mu, scale=noise_sigma, size=t.size)

        ymin = -50
        ax = Axes(
            x_range=[0, 4, 0.5],
            y_range=[0, -5 - ymin, (-ymin - 5) / 8],
            tips=False,
            x_length=fw(self, 0.7),
            y_length=fh(self, 0.6),
        ).to_edge(DOWN, MED_LARGE_BUFF)
        rres_eqn = Tex(r"| $\Delta R$")
        rres_text = Text("Range resolution", font=FONT).scale_to_fit_height(
            rres_eqn.height
        )
        rres_group = (
            Group(rres_text, rres_eqn)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(ax, UP, LARGE_BUFF)
        )
        rres_text.shift(UP * (rres_eqn[0][0].get_center() - rres_text.get_center()))

        snr_eqn = Tex(r"| SNR")
        snr_text = Text("Signal-to-noise Ratio", font=FONT).scale_to_fit_height(
            snr_eqn.height
        )
        snr_group = (
            Group(snr_text, snr_eqn)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(ax, UP, LARGE_BUFF)
        )
        snr_text.shift(UP * (snr_eqn[0][0].get_center() - snr_text.get_center()))

        x1 = VT(0)

        def get_x_n():
            A_1 = 10 ** (~power_norm_1 / 10)
            A_2 = 10 ** (~power_norm_2 / 10)
            x_n = (
                A_1 * np.sin(2 * PI * ~f1 * t) + A_2 * np.sin(2 * PI * ~f2 * t) + noise
            ) / (A_1 + A_2 + noise_sigma)

            blackman_window = signal.windows.blackman(N)
            x_n_windowed = x_n * blackman_window

            fft_len = N * 4

            X_k = fftshift(fft(x_n_windowed, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = 10 * np.log10(X_k)

            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            func = interp1d(freq, X_k - ymin, fill_value="extrapolate")

            plot = ax.plot(
                func,
                x_range=[0, ~x1, 1 / 1000],
                color=BLUE,
                use_smoothing=True,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            return plot

        plot = always_redraw(get_x_n)

        self.add(plot)

        rres_l = always_redraw(
            lambda: Line(
                ax.c2p(~f1, -ymin - 5),
                ax.c2p(~f1, -ymin - 5) + UP / 4,
            )
        )
        rres_mid = always_redraw(
            lambda: Line(
                Line(
                    ax.c2p(~f1, -ymin - 5),
                    ax.c2p(~f1, -ymin - 5) + UP / 4,
                ).get_midpoint(),
                Line(
                    ax.c2p(~f2, -ymin - 5),
                    ax.c2p(~f2, -ymin - 5) + UP / 4,
                ).get_midpoint(),
            ),
        )
        rres_r = always_redraw(
            lambda: Line(
                ax.c2p(~f2, -ymin - 5),
                ax.c2p(~f2, -ymin - 5) + UP / 4,
            )
        )

        self.play(
            LaggedStart(
                Create(ax),
                x1 @ 4,
                Write(rres_text),
                FadeIn(rres_eqn),
                Create(rres_l),
                Create(rres_mid),
                Create(rres_r),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(f1 + 1, run_time=5, rate_func=rate_functions.there_and_back)

        self.wait(0.5)

        print(~f1)
        print(ax.input_to_graph_coords(1, plot))
        snr_u = always_redraw(
            lambda: Line(
                ax.c2p(~f1, ax.i2gc(~f1, get_x_n())[1]),
                ax.c2p(~f1, ax.i2gc(~f1, get_x_n())[1]) + LEFT / 4,
            ).shift(LEFT)
        )
        snr_mid = always_redraw(
            lambda: Line(
                Line(
                    ax.c2p(~f1, ax.i2gc(~f1, get_x_n())[1]),
                    ax.c2p(~f1, ax.i2gc(~f1, get_x_n())[1]) + LEFT / 4,
                ).get_midpoint(),
                Line(
                    ax.c2p(~f1, -ymin - 25),
                    ax.c2p(~f1, -ymin - 25) + LEFT / 4,
                ).get_midpoint(),
            ).shift(LEFT),
        )
        snr_d = always_redraw(
            lambda: Line(
                ax.c2p(~f1, -ymin - 25),
                ax.c2p(~f1, -ymin - 25) + LEFT / 4,
            ).shift(LEFT)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                # Uncreate(rres_l),
                # Uncreate(rres_mid),
                # Uncreate(rres_r),
                ReplacementTransform(rres_l, snr_u),
                ReplacementTransform(rres_mid, snr_mid),
                ReplacementTransform(rres_r, snr_d),
                rres_group.animate.shift(UP * 5),
                snr_group.shift(UP * 5).animate.shift(DOWN * 5),
                # Create(snr_u),
                # Create(snr_mid),
                # Create(snr_d),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(power_norm_1 - 13, run_time=4)

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        copy_group = (
            Group(rres_group.copy(), snr_group.copy())
            .arrange(DOWN, MED_LARGE_BUFF)
            .move_to(self.camera.frame.copy().shift(UP * fh(self)))
            .shift(LEFT * 3)
        )
        classification = (
            Text("Classification", font=FONT)
            .scale_to_fit_height(rres_text.height)
            .next_to(
                self.camera.frame.copy().shift(UP * fh(self)).get_right(),
                LEFT,
                LARGE_BUFF,
            )
        )
        snr_bez = CubicBezier(
            copy_group[0].get_right() + [0.1, 0, 0],
            copy_group[0].get_right() + [1, 0, 0],
            classification.get_left() + [-1, 0, 0],
            classification.get_left() + [-0.1, 0, 0],
        )
        rres_bez = CubicBezier(
            copy_group[1].get_right() + [0.1, 0, 0],
            copy_group[1].get_right() + [1, 0, 0],
            classification.get_left() + [-1, 0, 0],
            classification.get_left() + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * fh(self)),
                Group(rres_group, snr_group)
                .animate.arrange(DOWN, MED_LARGE_BUFF)
                .move_to(self.camera.frame.copy().shift(UP * fh(self)))
                .shift(LEFT * 3),
                AnimationGroup(Create(snr_bez), Create(rres_bez)),
                Write(classification),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .next_to(classification, RIGHT, LARGE_BUFF * 3)
            .shift(UP)
        )
        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(cloud.width)
            .rotate(PI * 0.75)
            .set_fill(WHITE)
            .next_to(classification, RIGHT, LARGE_BUFF * 3)
            .shift(DOWN * 2 + RIGHT)
        )

        cloud_bez = CubicBezier(
            classification.get_right() + [0.1, 0, 0],
            classification.get_right() + [1, 0, 0],
            cloud.get_left() + [-1, 0, 0],
            cloud.get_left() + [-0.1, 0, 0],
        )
        plane_bez = CubicBezier(
            classification.get_right() + [0.1, 0, 0],
            classification.get_right() + [1, 0, 0],
            plane.get_left() + [-1, 0, 0],
            plane.get_left() + [-0.1, 0, 0],
        )

        class_group = Group(classification, plane, cloud)

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(class_group),
                Create(plane_bez),
                Create(cloud_bez),
                GrowFromCenter(plane),
                GrowFromCenter(cloud),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rres_thumbnail = (
            ImageMobject("../09_resolution/static/Resolution Thumbnail.png")
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(rres_group, LEFT, LARGE_BUFF * 3)
            .shift(UP + LEFT)
        )
        snr_thumbnail = (
            ImageMobject(
                "../07_snr_equation/media/images/snr/Thumbnail2_ManimCE_v0.18.1.png"
            )
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(snr_group, LEFT, LARGE_BUFF * 3)
            .shift(DOWN * 2 + RIGHT)
        )
        rres_thumbnail_rect = SurroundingRectangle(rres_thumbnail, buff=0)
        snr_thumbnail_rect = SurroundingRectangle(snr_thumbnail, buff=0)
        thumbnail_group = Group(snr_thumbnail, rres_thumbnail, snr_group, rres_group)
        snr_bez_thumbnail = CubicBezier(
            snr_group.get_left() + [-0.1, 0, 0],
            snr_group.get_left() + [-1, 0, 0],
            snr_thumbnail.get_right() + [1, 0, 0],
            snr_thumbnail.get_right() + [0.1, 0, 0],
        )
        rres_bez_thumbnail = CubicBezier(
            rres_group.get_left() + [-0.1, 0, 0],
            rres_group.get_left() + [-1, 0, 0],
            rres_thumbnail.get_right() + [1, 0, 0],
            rres_thumbnail.get_right() + [0.1, 0, 0],
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).move_to(thumbnail_group),
                Create(snr_bez_thumbnail),
                AnimationGroup(
                    GrowFromCenter(snr_thumbnail),
                    GrowFromCenter(snr_thumbnail_rect),
                ),
                Create(rres_bez_thumbnail),
                AnimationGroup(
                    GrowFromCenter(rres_thumbnail),
                    GrowFromCenter(rres_thumbnail_rect),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class Issue(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
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
            .set_fill(TARGET1_COLOR)
            .set_color(TARGET1_COLOR)
        )
        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .to_edge(RIGHT, LARGE_BUFF * 1.3)
            .shift(DOWN)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
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
        pw = VT(0.2)
        f = 10
        tx = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax - ~pw), ~xmax, 1 / 200],
                color=TX_COLOR,
            )
        )
        f_rx = VT(10)
        rx1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * ~f_rx * t),
                x_range=[max(0, ~xmax_t1 - ~pw), min(~xmax_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * ~f_rx * t),
                x_range=[max(0, ~xmax_t2 - ~pw), min(~xmax_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )
        self.add(tx, rx1, rx2)

        radar.vgroup.set_z_index(1)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        pw_line = Line(ax.c2p(~xmax - ~pw, 1.2), ax.c2p(~xmax, 1.2))
        pw_line_l = Line(pw_line.get_start() + DOWN / 8, pw_line.get_start() + UP / 8)
        pw_line_r = Line(pw_line.get_end() + DOWN / 8, pw_line.get_end() + UP / 8)

        pw_label = MathTex(r"\tau").scale(1.2).next_to(pw_line, UP)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in pw_label[0]], lag_ratio=0.15),
            LaggedStart(
                Create(pw_line_l),
                Create(pw_line),
                Create(pw_line_r),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            FadeOut(pw_line, pw_line_l, pw_line_r),
            LaggedStart(
                target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
                target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            pw_label.animate.shift(UP),
            LaggedStart(
                xmax @ (ax.p2c(target2.get_left())[0]),
                xmax_t1 @ (~pw / 2),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )

        self.play(
            xmax @ 1.5,
            xmax_t1 @ (0.5 + ~pw / 2),
            xmax_t2 @ (0.5 + ~pw / 2 - target_dist),
            run_time=3,
        )

        self.wait(0.5)

        target1_pw_line = Line(
            target1_ax.c2p(~xmax_t1, -1),
            target1_ax.c2p(~xmax_t1 - ~pw, -1),
        )
        target1_pw_line_l = Line(
            target1_pw_line.get_start() + UP / 6, target1_pw_line.get_start() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())
        target1_pw_line_r = Line(
            target1_pw_line.get_end() + UP / 6, target1_pw_line.get_end() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())

        target2_pw_line = Line(
            target2_ax.c2p(~xmax_t2, 1),
            target2_ax.c2p(~xmax_t2 - ~pw, 1),
        )
        target2_pw_line_l = Line(
            target2_pw_line.get_start() + UP / 6, target2_pw_line.get_start() + DOWN / 6
        ).rotate(target2_pw_line.get_angle())
        target2_pw_line_r = Line(
            target2_pw_line.get_end() + UP / 6, target2_pw_line.get_end() + DOWN / 6
        ).rotate(target2_pw_line.get_angle())
        pw_label_target1 = pw_label.copy().next_to(
            target1_pw_line.get_center(), UP, MED_SMALL_BUFF
        )
        pw_label_target2 = pw_label.copy().next_to(
            target2_pw_line.get_center(), DOWN, MED_SMALL_BUFF
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(target1_pw_line_l),
                    Create(target1_pw_line),
                    Create(target1_pw_line_r),
                    lag_ratio=0.3,
                ),
                TransformFromCopy(pw_label[0], pw_label_target1[0], path_arc=-PI / 3),
                LaggedStart(
                    Create(target2_pw_line_l),
                    Create(target2_pw_line),
                    Create(target2_pw_line_r),
                    lag_ratio=0.3,
                ),
                ReplacementTransform(
                    pw_label[0], pw_label_target2[0], path_arc=-PI / 3
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        line = Line(
            target2_ax.c2p(~xmax_t2, 0.8), target1_ax.c2p(~xmax_t1 - ~pw * 1.05, -0.6)
        )

        overlap = Polygon(
            line.get_start(),
            [line.get_start()[0], line.get_end()[1], 0],
            line.get_end(),
            [line.get_end()[0], line.get_start()[1], 0],
            fill_opacity=0.4,
            fill_color=YELLOW,
            stroke_opacity=0,
        )

        self.play(FadeIn(overlap))

        self.wait(0.5)

        pw_scale_factor = 4
        target1_pw_line_new = Line(
            target1_ax.c2p(~xmax_t1, -1),
            target1_ax.c2p(~xmax_t1 - ~pw / pw_scale_factor, -1),
        )
        target2_pw_line_new = Line(
            target2_ax.c2p(~xmax_t2, 1),
            target2_ax.c2p(~xmax_t2 - ~pw / pw_scale_factor, 1),
        )

        self.play(
            LaggedStart(
                FadeOut(overlap),
                AnimationGroup(
                    f_rx @ 30,
                    pw @ (~pw / pw_scale_factor),
                    Transform(target1_pw_line, target1_pw_line_new),
                    Transform(
                        target1_pw_line_l,
                        Line(
                            target1_pw_line_new.get_start() + UP / 6,
                            target1_pw_line_new.get_start() + DOWN / 6,
                        ).rotate(target1_pw_line_new.get_angle()),
                    ),
                    Transform(
                        target1_pw_line_r,
                        Line(
                            target1_pw_line_new.get_end() + UP / 6,
                            target1_pw_line_new.get_end() + DOWN / 6,
                        ).rotate(target1_pw_line_new.get_angle()),
                    ),
                    Transform(target2_pw_line, target2_pw_line_new),
                    Transform(
                        target2_pw_line_l,
                        Line(
                            target2_pw_line_new.get_start() + UP / 6,
                            target2_pw_line_new.get_start() + DOWN / 6,
                        ).rotate(target2_pw_line.get_angle()),
                    ),
                    Transform(
                        target2_pw_line_r,
                        Line(
                            target2_pw_line_new.get_end() + UP / 6,
                            target2_pw_line_new.get_end() + DOWN / 6,
                        ).rotate(target2_pw_line.get_angle()),
                    ),
                    pw_label_target1[0].animate.next_to(
                        target1_pw_line_new, UP, MED_SMALL_BUFF
                    ),
                    pw_label_target2[0].animate.next_to(
                        target2_pw_line_new, DOWN, MED_SMALL_BUFF
                    ),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(
                    target1_pw_line,
                    target2_pw_line,
                    target1_pw_line_r,
                    target1_pw_line_l,
                    target2_pw_line_r,
                    target2_pw_line_l,
                    pw_label_target1[0],
                    pw_label_target2[0],
                ),
                AnimationGroup(
                    xmax_t1 @ (1 + ~pw),
                    xmax_t2 @ (1 + ~pw),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        fs = 100
        f = 25
        noise_std = VT(0.1)

        x_len = config.frame_width * 0.4
        y_len = config.frame_height * 0.4
        y_max = 30

        ax = Axes(
            x_range=[0, fs / 2, fs / 8],
            y_range=[0, y_max, 10],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).next_to(self.camera.frame, UP)

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        amp = VT(0.1)

        def fft_updater():
            np.random.seed(1)
            noise = np.random.normal(loc=0, scale=~noise_std, size=N)
            x_n = (~amp * np.sin(2 * PI * f * t) + noise) * signal.windows.blackman(N)
            X_k = fftshift(fft(x_n, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = np.clip(10 * np.log10(X_k) + y_max, 0, None)
            f_X_k_log = interp1d(freq, X_k, fill_value="extrapolate")

            plot = ax.plot(f_X_k_log, x_range=[0, fs / 2, 1 / 100], color=RX_COLOR)
            return plot

        X_k_plot = always_redraw(fft_updater)

        self.add(ax, X_k_plot)

        new_ax_copy = ax.copy().move_to(self.camera.frame.get_top()).shift(DOWN * 0.9)

        new_scene = Group(radar.vgroup, target1, target2, new_ax_copy)

        snr_line = always_redraw(
            lambda: Line(
                ax.c2p(f + 3, -lin2db(~noise_std)),
                [ax.c2p(f + 3, 0)[0], ax.input_to_graph_point(f, X_k_plot)[1], 0],
            )
        )
        snr_line_u = always_redraw(
            lambda: Line(
                snr_line.get_top() + LEFT / 8,
                snr_line.get_top() + RIGHT / 8,
            )
        )
        snr_line_d = always_redraw(
            lambda: Line(
                snr_line.get_bottom() + LEFT / 8,
                snr_line.get_bottom() + RIGHT / 8,
            )
        )
        snr_label = always_redraw(
            lambda: Text("SNR", font=FONT).next_to(snr_line, RIGHT, MED_SMALL_BUFF)
        )
        self.add(snr_line, snr_line_u, snr_line_d, snr_label)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    ax.animate.move_to(self.camera.frame.get_top()).shift(DOWN * 0.9),
                    self.camera.frame.animate.scale_to_fit_height(
                        new_scene.height * 1.2
                    )
                    .move_to(new_scene)
                    .set_x(0),
                ),
            )
        )

        self.wait(0.5)

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=x_len,
                y_length=y_len,
            )
            .set_z_index(-1)
            .next_to(self.camera.frame, LEFT)
            .set_y(ax.get_y())
        )

        pw_plot = VT(0.1)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: ~pulse_amp * np.sin(2 * PI * pulse_f * t)
                if t < ~pw_plot
                else 0,
                x_range=[0, 1, 1 / 200],
                color=TX_COLOR,
            )
        )
        self.add(pulse_ax, pulse)

        self.play(
            Group(pulse_ax, ax).animate.arrange(RIGHT, MED_LARGE_BUFF).set_y(ax.get_y())
        )

        tx_pulse_label = Text("Transmit Pulse", font=FONT).next_to(pulse_ax, DOWN)

        self.wait(0.5)

        self.play(Write(tx_pulse_label))

        self.wait(0.5)

        energy_eqn = (
            MathTex(r"E = P \cdot t")
            .scale(2.5)
            .next_to(self.camera.frame.get_bottom(), UP, LARGE_BUFF * 2)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[
                    m.set_opacity(0)
                    .shift(DOWN / 2)
                    .animate.shift(UP / 2)
                    .set_opacity(1)
                    for m in energy_eqn[0]
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(energy_eqn[0][4].animate.set_color(GREEN))

        self.wait(0.5)

        self.play(pw_plot @ (~pw_plot * 3), amp @ (~amp * 3))

        self.wait(0.5)

        self.play(pw_plot @ (~pw_plot / 3), amp @ (~amp / 3))

        self.wait(0.5)

        self.play(energy_eqn[0][4].animate.set_color(WHITE))

        self.wait(0.5)

        self.play(energy_eqn[0][2].animate.set_color(GREEN))

        self.wait(0.5)

        self.play(pulse_amp @ (~pulse_amp * 3), amp @ (~amp * 3))

        self.wait(0.5)

        self.play(pulse_amp @ (~pulse_amp / 3), amp @ (~amp / 3))

        self.wait(0.5)

        self.play(energy_eqn[0][2].animate.set_color(WHITE))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self) * 1.5))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        relation_table = (
            MobjectTable(
                [
                    [
                        MathTex(r"\downarrow").scale(2),
                        Text("+", font=FONT, color="GREEN").scale(2),
                        Text("-", font=FONT, color="RED").scale(2),
                    ],
                    [
                        MathTex(r"\uparrow").scale(2),
                        Text("-", font=FONT, color="RED").scale(2),
                        Text("+", font=FONT, color="GREEN").scale(2),
                    ],
                ],
                col_labels=[
                    MathTex(r"\tau").scale(2),
                    MathTex(r"\Delta R"),
                    Tex("SNR"),
                ],
            )
            .scale(1.5)
            .move_to(self.camera.frame)
        )

        self.play(
            LaggedStart(
                *[Create(m) for m in relation_table.get_horizontal_lines()],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[Create(m) for m in relation_table.get_vertical_lines()], lag_ratio=0.2
            ),
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_col_labels()], lag_ratio=0.2
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_rows()[1]], lag_ratio=0.2
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_rows()[2]], lag_ratio=0.2
            )
        )

        self.wait(0.5)

        first_option = SurroundingRectangle(relation_table.get_rows()[1])
        second_option = SurroundingRectangle(relation_table.get_rows()[2])
        good_rres = SurroundingRectangle(
            relation_table.get_rows()[1][1], buff=MED_LARGE_BUFF
        )
        good_snr = SurroundingRectangle(
            relation_table.get_rows()[2][2], buff=MED_LARGE_BUFF
        )

        self.play(Create(first_option))

        self.wait(0.5)

        first_option.save_state()
        self.play(Transform(first_option, second_option))

        self.wait(0.5)

        self.play(first_option.animate.restore())

        self.wait(0.5)

        self.play(
            TransformFromCopy(first_option, good_snr),
            ReplacementTransform(first_option, good_rres),
        )

        self.wait(0.5)

        pulse_compression_label = (
            Text("Pulse Compression &", font=FONT)
            .next_to(relation_table, DOWN, LARGE_BUFF * 2)
            .shift(RIGHT / 2)
        )
        matched_filtering_label = Text("Matched Filtering", font=FONT).next_to(
            pulse_compression_label, DOWN, MED_SMALL_BUFF
        )

        rres_bez = CubicBezier(
            good_rres.get_bottom() + [0, -0.1, 0],
            good_rres.get_bottom() + [0, -1, 0],
            pulse_compression_label.get_top() + [0.2, 2, 0],
            pulse_compression_label.get_top() + [0, 0.1, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        snr_bez = CubicBezier(
            good_snr.get_bottom() + [0, -0.1, 0],
            good_snr.get_bottom() + [0, -1, 0],
            pulse_compression_label.get_top() + [0, 2, 0],
            pulse_compression_label.get_top() + [0, 0.1, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.3).shift(DOWN * 2),
                Create(rres_bez),
                Create(snr_bez),
                Write(pulse_compression_label),
                Write(matched_filtering_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self) * 1.5))

        self.wait(2)


# TODO: Remove the good,ok,bad labels and notebook after usage
class Options(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        pulse_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.5),
        ).set_z_index(-1)

        rres_nl = NumberLine(
            x_range=[0, 10, 1], length=pulse_ax.height * 1.3, rotation=PI / 2
        ).set_z_index(-1)
        snr_nl = (
            NumberLine(
                x_range=[0, 10, 1], length=pulse_ax.height * 1.3, rotation=PI / 2
            )
            .next_to(rres_nl, RIGHT, LARGE_BUFF * 2)
            .set_z_index(-1)
        )

        Group(pulse_ax, Group(rres_nl, snr_nl)).arrange(RIGHT, LARGE_BUFF * 20)

        rres_label = always_redraw(
            lambda: MathTex(r"\Delta R").next_to(rres_nl, UP, MED_SMALL_BUFF)
        )
        snr_label = always_redraw(
            lambda: Tex(r"SNR").next_to(snr_nl, UP, MED_SMALL_BUFF)
        )

        rres = VT(9)
        rres_dot = always_redraw(
            lambda: Dot(
                color=interpolate_color(OK, BAD, (~rres - 5) / 5)
                if ~rres > 5
                else interpolate_color(GOOD, OK, ~rres / 5),
                radius=DEFAULT_DOT_RADIUS * 3,
            ).move_to(rres_nl.n2p(~rres))
        )

        snr = VT(9)
        snr_dot = always_redraw(
            lambda: Dot(
                color=interpolate_color(OK, GOOD, (~snr - 5) / 5)
                if ~snr > 5
                else interpolate_color(BAD, OK, ~snr / 5),
                radius=DEFAULT_DOT_RADIUS * 3,
            ).move_to(snr_nl.n2p(~snr))
        )

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_f1 = VT(pulse_f)
        pulse_x1 = VT(1)
        pulse_x0 = VT(0)

        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: ~pulse_amp
                * signal.chirp(
                    t - 1 / pulse_f / 4,
                    pulse_f,
                    ~pw_plot,
                    ~pulse_f1,
                    method="quadratic",
                )
                if t < ~pw_plot + ~pulse_x0
                else 0,
                x_range=[~pulse_x0, ~pulse_x1, 1 / 1000],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
            )
        )
        self.add(
            pulse_ax,
            pulse,
            rres_nl,
            snr_nl,
            # snr_bad_label,
            # rres_good_label,
            rres_label,
            snr_label,
            snr_dot,
            rres_dot,
        )

        self.play(
            Group(
                pulse_ax,
                Group(
                    rres_nl,
                    # rres_good_label,
                    snr_nl,
                    # snr_bad_label,
                ),
            ).animate.arrange(RIGHT, LARGE_BUFF)
        )

        rres_bad_label = (
            Text("Bad", color=BAD, font=FONT)
            .scale(0.5)
            .next_to(rres_nl.n2p(10), RIGHT, MED_SMALL_BUFF)
        )
        snr_good_label = (
            Text("Good", color=GOOD, font=FONT)
            .scale(0.5)
            .next_to(snr_nl.n2p(10), LEFT, MED_SMALL_BUFF)
        )
        rres_ok_label = (
            Text("OK", color=OK, font=FONT)
            .scale(0.5)
            .next_to(rres_nl.n2p(5), RIGHT, MED_SMALL_BUFF)
        )
        snr_ok_label = (
            Text("OK", color=OK, font=FONT)
            .scale(0.5)
            .next_to(snr_nl.n2p(5), LEFT, MED_SMALL_BUFF)
        )
        rres_good_label = (
            Text("Good", color=GOOD, font=FONT)
            .scale(0.5)
            .next_to(rres_nl.n2p(0), RIGHT, MED_SMALL_BUFF)
        )
        snr_bad_label = (
            Text("Bad", color=BAD, font=FONT)
            .scale(0.5)
            .next_to(snr_nl.n2p(0), LEFT, MED_SMALL_BUFF)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(rres_bad_label),
                    FadeIn(snr_good_label),
                ),
                AnimationGroup(
                    FadeIn(rres_ok_label),
                    FadeIn(snr_ok_label),
                ),
                AnimationGroup(
                    FadeIn(rres_good_label),
                    FadeIn(snr_bad_label),
                ),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(pw_plot @ 0.1, rres @ 1, snr @ 1, run_time=5)

        self.wait(0.5)

        check = (
            Text("✔", font=FONT, color=GOOD)
            .scale(1.5)
            .next_to(rres_dot, LEFT, SMALL_BUFF)
        )
        qmark = Text("?", font=FONT).scale(1.5).next_to(snr_dot, RIGHT, SMALL_BUFF)

        self.play(GrowFromCenter(check))

        self.wait(0.5)

        self.play(GrowFromCenter(qmark))

        self.wait(0.5)

        self.play(pw_plot @ 0.3, rres @ 9, snr @ 9, run_time=5)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Transform(
                    check,
                    check.copy().next_to(snr_dot, RIGHT, SMALL_BUFF),
                    path_arc=PI / 3,
                ),
                Transform(
                    qmark,
                    qmark.copy().next_to(rres_dot, LEFT, SMALL_BUFF),
                    path_arc=-PI / 3,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Group(
                rres_nl,
                snr_nl,
                qmark,
                check,
                snr_good_label,
                rres_good_label,
                rres_ok_label,
                snr_ok_label,
                rres_bad_label,
                snr_bad_label,
            ).animate.shift(RIGHT * 5),
            self.camera.frame.animate.scale_to_fit_width(pulse_ax.width * 1.5).move_to(
                pulse_ax
            ),
            pw_plot @ 0.5,
        )

        self.wait(0.5)

        pulse_line = Line(
            pulse_ax.c2p(0, ~pulse_amp), pulse_ax.c2p(~pw_plot, ~pulse_amp), color=GREEN
        ).shift(UP / 2)
        pulse_line_l = Line(
            pulse_line.get_left() + DOWN / 4,
            pulse_line.get_left() + UP / 4,
            color=GREEN,
        )
        pulse_line_r = Line(
            pulse_line.get_right() + DOWN / 4,
            pulse_line.get_right() + UP / 4,
            color=GREEN,
        )

        self.play(
            LaggedStart(
                Create(pulse_line_l),
                Create(pulse_line),
                Create(pulse_line_r),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(pulse_f1 @ (pulse_f * 4), run_time=3)

        self.wait(0.5)

        pulse_line_new = Line(
            pulse_ax.c2p(0, ~pulse_amp),
            pulse_ax.c2p(0.1, ~pulse_amp),
            color=GREEN,
        ).shift(UP / 2)

        pulse_line.save_state()
        pulse_line_r.save_state()
        self.play(
            pw_plot @ 0.1,
            Transform(pulse_line, pulse_line_new),
            Transform(
                pulse_line_r,
                Line(
                    pulse_line_new.get_right() + DOWN / 4,
                    pulse_line_new.get_right() + UP / 4,
                    color=GREEN,
                ),
            ),
        )

        self.wait(0.5)

        self.play(
            pw_plot @ 0.5,
            pulse_line.animate.restore(),
            pulse_line_r.animate.restore(),
        )

        self.wait(0.5)

        self.play(pulse_f1 @ pulse_f)

        self.wait(0.5)

        theory_paper = ImageMobject(
            "../props/static/Theory and Design of Chirp Radars.png"
        ).scale_to_fit_height(fh(self, 0.7))
        new_chirp = ImageMobject(
            "../props/static/Chirp A New Radar Technique.jpg"
        ).scale_to_fit_width(theory_paper.width * 1.3)
        fundamentals_of_radar_dsp = ImageMobject(
            "../props/static/Fundamentals of Radar DSP Book Cover.jpg"
        ).scale_to_fit_height(fh(self, 0.7))
        resources = (
            Group(fundamentals_of_radar_dsp, new_chirp, theory_paper)
            .arrange(RIGHT, MED_LARGE_BUFF)
            .scale_to_fit_width(fw(self, 0.9))
            .move_to(self.camera.frame)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(LaggedStart(*[GrowFromCenter(m) for m in resources], lag_ratio=0.3))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.shift(UP * fh(self)) for m in resources], lag_ratio=0.3
            )
        )

        self.remove(*resources)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        # It's honestly pretty incredible how people came to these findings,
        # so as always, you can find some great resources, papers, books,
        # and more in the description, as well as a Python notebook to play
        # around with the concepts yourself.

        nb_img_1 = ImageMobject("./static/nb_img_1.png").scale_to_fit_height(
            fh(self, 0.6)
        )
        nb_img_2 = ImageMobject("./static/nb_img_2.png").scale_to_fit_height(
            fh(self, 0.6)
        )
        nb_img_3 = ImageMobject("./static/nb_img_3.png").scale_to_fit_height(
            fh(self, 0.6)
        )
        nb_img_group = (
            Group(nb_img_1, nb_img_2, nb_img_3)
            .arrange(RIGHT, -MED_SMALL_BUFF)
            .scale_to_fit_width(fw(self, 0.95))
            .next_to(self.camera.frame.get_bottom(), UP, LARGE_BUFF * 2)
        )
        nb_img_2.shift(DOWN)

        nb_label = (
            Text("pulse_compression.ipynb", font=FONT)
            .scale_to_fit_width(fw(self, 0.5))
            .next_to(self.camera.frame.get_top(), DOWN)
        )

        nb_group = (
            Group(nb_label, nb_img_1, nb_img_2, nb_img_3)
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
        )
        nb_bez_1 = CubicBezier(
            nb_label.get_bottom() + [0, -0.1, 0],
            nb_label.get_bottom() + [0, -1, 0],
            nb_img_1.get_top() + [0, 1, 0],
            nb_img_1.get_top() + [0, 0.1, 0],
        )

        nb_bez_2 = CubicBezier(
            nb_label.get_bottom() + [0, -0.1, 0],
            nb_label.get_bottom() + [0, -1, 0],
            nb_img_2.get_top() + [0, 1, 0],
            nb_img_2.get_top() + [0, 0.1, 0],
        )

        nb_bez_3 = CubicBezier(
            nb_label.get_bottom() + [0, -0.1, 0],
            nb_label.get_bottom() + [0, -1, 0],
            nb_img_3.get_top() + [0, 1, 0],
            nb_img_3.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self)),
                Write(nb_label),
                Create(nb_bez_1),
                GrowFromCenter(nb_img_1),
                Create(nb_bez_2),
                GrowFromCenter(nb_img_2),
                Create(nb_bez_3),
                GrowFromCenter(nb_img_3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 1).next_to(
            self.camera.frame.get_left(), LEFT, LARGE_BUFF * 3
        )

        self.remove(
            snr_dot,
            snr_nl,
            qmark,
            check,
            rres_dot,
            rres_nl,
            rres_label,
            snr_label,
            snr_good_label,
            rres_good_label,
            snr_bad_label,
            rres_bad_label,
            snr_ok_label,
            rres_ok_label,
        )
        self.next_section(skip_animations=skip_animations(False))

        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .next_to(pulse, RIGHT, LARGE_BUFF * 3)
        )

        radar.vgroup.shift(LEFT * 5)
        self.play(
            LaggedStart(
                Group(
                    nb_label,
                    nb_bez_1,
                    nb_bez_2,
                    nb_bez_3,
                    nb_img_1,
                    nb_img_2,
                    nb_img_3,
                ).animate.shift(DOWN * 10),
                self.camera.frame.animate.scale_to_fit_height(radar.vgroup.height * 1.7)
                .move_to(
                    Group(
                        radar.vgroup.copy().shift(
                            pulse_ax.c2p(0, 0) - radar.radome.get_right()
                        ),
                        pulse,
                    )
                )
                .shift(UP * 2 + RIGHT * 4),
                FadeOut(pulse_ax),
                pulse_x1 @ ~pw_plot,
                Uncreate(pulse_line_r),
                Uncreate(pulse_line),
                Uncreate(pulse_line_l),
                radar.vgroup.animate.shift(
                    pulse_ax.c2p(0, 0) - radar.radome.get_right()
                ),
                plane.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.2,
            )
        )
        self.remove(
            nb_label,
            nb_bez_1,
            nb_bez_2,
            nb_bez_2,
            nb_img_1,
            nb_img_2,
            nb_img_3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        pulse_rtn_ax = pulse_ax.copy().rotate(PI)
        pulse_rtn_ax.shift(plane.get_left() - pulse_rtn_ax.c2p(0, 0))

        pulse_rtn_x0 = VT(-~pw_plot)
        pulse_rtn_x1 = VT(0)

        pulse_amp_2 = VT(0)
        pulse_amp_3 = VT(0)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0)
        pulse_t_start_3 = VT(0)
        pulse_f1_2 = VT(~pulse_f1)
        allow_2 = VT(0)
        allow_3 = VT(0)
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1_2,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                # use_smoothing=False,
            )
        )

        # pulse_rtn = always_redraw(
        #     lambda: pulse_rtn_ax.plot(
        #         lambda t: ~pulse_amp
        #         * signal.chirp(
        #             t - 1 / pulse_f / 4,
        #             pulse_f,
        #             ~pw_plot,
        #             ~pulse_f1,
        #             method="quadratic",
        #         )
        #         if t < ~pw_plot + ~pulse_rtn_x0
        #         else 0,
        #         x_range=[
        #             max(0, ~pulse_rtn_x0),
        #             min(~pulse_rtn_x1, pulse_rtn_ax.p2c(radar.radome.get_right())[0]),
        #             1 / 1000,
        #         ],
        #         stroke_width=DEFAULT_STROKE_WIDTH * 1,
        #         color=RX_COLOR,
        #     )
        # )

        self.next_section(skip_animations=skip_animations(True))
        self.add(pulse_rtn)

        self.play(
            LaggedStart(
                AnimationGroup(
                    pulse_x0 + 3,
                    pulse_x1 + 3,
                ),
                AnimationGroup(pulse_rtn_x0 + 1, pulse_rtn_x1 + 1),
                lag_ratio=0.4,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(pulse_rtn.width * 1.8).move_to(
                pulse_rtn
            )
        )

        self.wait(0.5)

        target_start = VT(~pulse_rtn_x1)
        target_arrow = always_redraw(
            lambda: Arrow(
                pulse_rtn_ax.c2p(~target_start, -1),
                pulse_rtn_ax.c2p(~target_start, -~pulse_amp),
            )
        )
        target_label = always_redraw(
            lambda: Text("Target Start", font=FONT)
            .scale(0.3)
            .next_to(target_arrow, UP, SMALL_BUFF)
        )
        target_start_dot = always_redraw(
            lambda: Dot(pulse_rtn_ax.input_to_graph_point(~target_start, pulse_rtn))
        )

        self.play(
            self.camera.frame.animate.shift(UP),
            Create(target_start_dot),
            FadeIn(target_arrow, target_label),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            target_start.animate(
                rate_func=rate_functions.there_and_back
            ).increment_value(-~pw_plot),
            run_time=4,
        )

        self.wait(0.5)

        self.play(FadeOut(target_arrow, target_start_dot, target_label))

        self.wait(0.5)

        pulse_static = pulse.copy().next_to(pulse_rtn, UP, SMALL_BUFF).shift(LEFT * 5)
        self.add(pulse_static)

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(
                    Group(pulse_rtn, pulse_static.copy().shift(RIGHT * 5))
                ),
                pulse_static.animate(
                    rate_func=rate_functions.ease_out_bounce, run_time=2
                ).shift(RIGHT * 5),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        start = DashedLine(
            pulse_rtn_ax.c2p(~pulse_rtn_x1, -2),
            pulse_rtn_ax.c2p(~pulse_rtn_x1, 1),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )

        self.play(Create(start))

        tx_pulse_label = Text("Tx Pulse", font=FONT, color=TX_COLOR).next_to(
            pulse_static.copy().set_stroke(opacity=0.3).shift(UP * 3),
            UP,
            MED_SMALL_BUFF,
        )
        self.add(tx_pulse_label)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        start_arrow = Arrow(
            radar.radome.get_right() + RIGHT * 2 + DOWN * 2,
            radar.radome.get_right(),
            color=YELLOW,
        )

        self.play(
            LaggedStart(
                Uncreate(start),
                pulse_static.animate.set_stroke(opacity=0.3).shift(UP * 3),
                self.camera.frame.animate.restore(),
                AnimationGroup(
                    pulse_rtn_x1 @ (pulse_rtn_ax.p2c(radar.radome.get_right())[0]),
                    pulse_rtn_x0
                    @ (pulse_rtn_ax.p2c(radar.radome.get_right())[0] - ~pw_plot),
                ),
                GrowArrow(start_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
            .next_to(plane, UP, -LARGE_BUFF)
            .shift(RIGHT / 2)
        )
        target3 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET3_COLOR)
            .set_color(TARGET3_COLOR)
            .next_to(plane, DOWN, -LARGE_BUFF)
            .shift(RIGHT * 1.4)
        )

        pulse_2_opacity = VT(0)
        pulse_3_opacity = VT(0)
        pulse_t2_shift = VT(1.3)
        pulse_t3_shift = VT(1.3)

        pulse_rtn_t2 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(UP * ~pulse_t2_shift)
            .set_stroke(opacity=~pulse_2_opacity)
        )
        pulse_rtn_t3 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(DOWN * ~pulse_t3_shift)
            .set_stroke(opacity=~pulse_3_opacity)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(FadeOut(start_arrow))

        self.wait(0.5)

        self.add(pulse_rtn_t2, pulse_rtn_t3)
        self.play(
            LaggedStart(
                plane.animate.scale(0.5),
                AnimationGroup(
                    pulse_2_opacity @ 1,
                    target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                AnimationGroup(
                    pulse_3_opacity @ 1,
                    target3.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                AnimationGroup(
                    pulse_t_start_2 @ 0.1,
                    pulse_t_start_3 @ 0.22,
                ),
                pulse_amp_2 @ 0.3,
                pulse_amp_3 @ 0.3,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(pulse_t2_shift @ 0, pulse_2_opacity @ 0, lag_ratio=0.2),
                allow_2 @ 1,
                LaggedStart(pulse_t3_shift @ 0, pulse_3_opacity @ 0, lag_ratio=0.2),
                allow_3 @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        comparison_group = Group(
            pulse_rtn, pulse_static.copy().next_to(pulse_rtn, UP, MED_LARGE_BUFF)
        )

        start = DashedLine(
            comparison_group[1].get_corner(UL) + UP / 2,
            comparison_group[1].get_corner(UL) + DOWN * 4.7,
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_static.animate.set_stroke(opacity=1).move_to(comparison_group[1]),
                self.camera.frame.animate.scale_to_fit_height(
                    (pulse_rtn.height + pulse_static.height) * 1.8
                ).move_to(comparison_group),
                Create(start),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            Group(pulse_static, start).animate.shift(
                RIGHT * (pulse_rtn.get_left() - pulse_static.get_left())[0]
            ),
            run_time=0.5,
        )
        self.play(Group(pulse_static, start).animate.shift(RIGHT * 2), run_time=0.5)
        self.play(Group(pulse_static, start).animate.shift(LEFT * 0.4), run_time=0.5)

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_phase_2.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(80),
                pulse_phase_1.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(40),
                pulse_phase_3.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(110),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(0.5)

        lu_bez = CubicBezier(
            pulse_rtn.get_corner(DL) + [0, -0.1, 0],
            pulse_rtn.get_corner(DL) + [0, -1, 0],
            pulse_rtn.get_bottom() + [0, -1.5, 0],
            pulse_rtn.get_bottom() + [0, -3, 0],
        )
        ru_bez = CubicBezier(
            pulse_rtn.get_corner(DR) + [0, -0.1, 0],
            pulse_rtn.get_corner(DR) + [0, -1, 0],
            pulse_rtn.get_bottom() + [0, -1.5, 0],
            pulse_rtn.get_bottom() + [0, -3, 0],
        )

        nums = [str(np.random.randint(0, 2)) for _ in range(16)]
        info = Text(
            "".join(nums), font=FONT, font_size=DEFAULT_FONT_SIZE * 1.2
        ).next_to(ru_bez.get_end(), DOWN, LARGE_BUFF * 3)
        for char, num in zip(info, nums):
            if num == "1":
                char.set_color(GOOD)
            else:
                char.set_color(BAD)

        info_group = Group(pulse_rtn, info)

        ld_bez = CubicBezier(
            pulse_rtn.get_bottom() + [0, -3, 0],
            pulse_rtn.get_bottom() + [0, -4.5, 0],
            info.get_corner(UL) + [0, 1, 0],
            info.get_corner(UL) + [0, 0.1, 0],
        )
        rd_bez = CubicBezier(
            pulse_rtn.get_bottom() + [0, -3, 0],
            pulse_rtn.get_bottom() + [0, -4.5, 0],
            info.get_corner(UR) + [0, 1, 0],
            info.get_corner(UR) + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(radar.vgroup, start, pulse_static, target2, target3, plane),
                self.camera.frame.animate.scale_to_fit_height(
                    info_group.height * 1.2
                ).move_to(info_group),
                pulse_f1.animate(run_time=2).set_value(pulse_f * 3),
                AnimationGroup(Create(lu_bez), Create(ru_bez)),
                AnimationGroup(Create(ld_bez), Create(rd_bez)),
                LaggedStart(*[GrowFromCenter(m) for m in info], lag_ratio=0.1),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self) * 2))

        self.wait(2)


class Encoding(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        f_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.25],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        lfm_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        Group(f_ax, lfm_ax).arrange(DOWN, MED_LARGE_BUFF).to_edge(RIGHT, LARGE_BUFF)
        f_label = (
            Text("Frequency", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(f_ax, LEFT, SMALL_BUFF)
        )
        amp_label = (
            Text("Amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(lfm_ax, LEFT, SMALL_BUFF)
        )

        m = VT(0)
        f = 4
        f1 = VT(f)
        f_plot = always_redraw(
            lambda: f_ax.plot(lambda t: 0.2 + ~m * t, color=TX_COLOR)
        )
        lfm_plot = always_redraw(
            lambda: lfm_ax.plot(
                lambda t: chirp_pulse(t, 0, 1, f, ~f1, 1, 0, ramp="linear"),
                x_range=[0, 1, 1 / 1000],
                color=TX_COLOR,
            )
        )

        lfm_label = (
            Text("Linear\nFrequency\nModulation", font=FONT)
            .scale_to_fit_width(fw(self, 0.25))
            .to_edge(LEFT, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                Write(lfm_label),
                Create(f_ax),
                Create(lfm_ax),
                FadeIn(f_label),
                FadeIn(amp_label),
                lag_ratio=0.3,
            )
        )
        self.play(Create(f_plot), Create(lfm_plot))

        self.wait(0.5)

        self.play(m @ 0.8, f1 @ 30)

        self.wait(0.5)

        thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale_to_fit_width(lfm_label.width * 1.2)
            .next_to(lfm_label, DOWN, LARGE_BUFF * 10)
        )
        tn_box = SurroundingRectangle(thumbnail, buff=0)

        title_top = Text("What is FMCW Radar and", font=FONT).scale(0.5)
        title_bot = Text("why is it useful?", font=FONT).scale(0.5)
        title = Group(title_top, title_bot).arrange(DOWN).next_to(thumbnail, DOWN)
        tn = Group(thumbnail, tn_box, title)

        self.play(
            Group(lfm_label, tn)
            .animate.arrange(DOWN, LARGE_BUFF)
            .set_x(lfm_label.get_x())
        )

        self.wait(0.5)

        phase_ax = Axes(
            x_range=[0, 8, 1],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        phase_amp_ax = Axes(
            x_range=[0, 8, 1],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        Group(phase_ax, phase_amp_ax).arrange(DOWN, MED_LARGE_BUFF).to_edge(
            RIGHT, LARGE_BUFF
        ).shift(DOWN * fh(self))
        phase_label = (
            Text("Phase", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(phase_ax, LEFT, SMALL_BUFF)
        )
        phase_amp_label = (
            Text("Amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(phase_amp_ax, LEFT, SMALL_BUFF)
        )

        np.random.seed(2)
        phase_seq = (np.random.randint(0, 2, 8) - 0.5) * -2
        phase_plot = phase_ax.plot(
            lambda t: 1 if phase_seq[int(np.floor(t))] > 0 else -1,
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_amp_plot = phase_amp_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t)
            * (1 if phase_seq[int(np.floor(t))] > 0 else -1),
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_labels = Group(
            *[
                Text("0", color=BAD, font=FONT).move_to(phase_ax.c2p(idx + 0.5, -0.5))
                if num < 0
                else Text("1", color=GOOD, font=FONT).move_to(
                    phase_ax.c2p(idx + 0.5, 0.5)
                )
                for idx, num in enumerate(phase_seq)
            ]
        )

        phase_mod_label = (
            Text("Phase\nModulation", font=FONT)
            .scale_to_fit_width(fw(self, 0.25))
            .to_edge(LEFT, LARGE_BUFF)
            .shift(DOWN * fh(self))
        )
        self.add(
            phase_labels,
            phase_mod_label,
            phase_amp_plot,
            phase_amp_label,
            phase_label,
            phase_ax,
            phase_amp_ax,
            phase_plot,
        )

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        np.random.seed(3)
        phase_seq_new = (np.random.randint(0, 2, 8) - 0.5) * -2
        phase_plot_new = phase_ax.plot(
            lambda t: 1 if phase_seq_new[int(np.floor(t))] > 0 else -1,
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_amp_plot_new = phase_amp_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t)
            * (1 if phase_seq_new[int(np.floor(t))] > 0 else -1),
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_labels_new = Group(
            *[
                Text("0", color=BAD, font=FONT).move_to(phase_ax.c2p(idx + 0.5, -0.5))
                if num < 0
                else Text("1", color=GOOD, font=FONT).move_to(
                    phase_ax.c2p(idx + 0.5, 0.5)
                )
                for idx, num in enumerate(phase_seq_new)
            ]
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[
                    Transform(old, new)
                    for old, new in zip(phase_labels, phase_labels_new)
                ],
                lag_ratio=0.1,
            ),
            Transform(phase_plot, phase_plot_new),
            Transform(phase_amp_plot, phase_amp_plot_new),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(fh(self, 2.1)).shift(
                UP * fh(self) / 2
            )
        )

        self.wait(0.5)

        new_cam = self.camera.frame.copy().scale(0.5).shift(RIGHT * fw(self, 0.5))
        all_modulations = Text("All Modulation Schemes", font=FONT).next_to(
            new_cam.get_top(), DOWN, LARGE_BUFF
        )

        self.play(
            self.camera.frame.animate.scale(0.5).shift(RIGHT * fw(self, 0.5)),
            Write(all_modulations),
        )

        self.wait(0.5)

        gen_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.4),
            y_length=fh(self, 0.3),
        )
        mod_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.4),
            y_length=fh(self, 0.3),
        )
        arrow = MathTex(r"\Rightarrow").scale(2)
        Group(gen_ax, arrow, mod_ax).arrange(RIGHT, MED_LARGE_BUFF).move_to(new_cam)
        gen_plot = gen_ax.plot(
            lambda t: np.sin(2 * PI * 6 * t),
            x_range=[0, 1, 1 / 400],
            color=TX_COLOR,
        )

        self.play(LaggedStart(Create(gen_ax), Create(gen_plot), lag_ratio=0.3))

        self.wait(0.5)

        gen_bw = (
            MathTex(r"B = 0 \text{ Hz}").scale(2).next_to(gen_ax, DOWN, MED_LARGE_BUFF)
        )
        mod_bw = (
            MathTex(r"B > 0 \text{ Hz}").scale(2).next_to(mod_ax, DOWN, MED_LARGE_BUFF)
        )

        self.play(FadeIn(gen_bw, shift=UP))

        self.wait(0.5)

        qmark = Text("?", font=FONT, color=YELLOW).scale(1.5).move_to(mod_ax).shift(UP)

        self.play(
            LaggedStart(
                GrowFromCenter(arrow),
                Create(mod_ax),
                GrowFromCenter(qmark),
                FadeIn(mod_bw, shift=UP),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(Group(lfm_label, lfm_ax, f_ax)))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        hl_l = VT(0)
        hl_r = VT(0)
        hl = always_redraw(
            lambda: f_ax.plot(
                lambda t: 0.2 + ~m * t, color=YELLOW, x_range=[~hl_l, ~hl_r, 1 / 1000]
            ),
        )

        self.add(hl)

        self.play(LaggedStart(hl_r @ 1, hl_l @ 1, lag_ratio=0.2))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        lines = Group(
            *[
                DashedLine(
                    phase_amp_ax.c2p(2, 1),
                    phase_amp_ax.c2p(2, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
                DashedLine(
                    phase_amp_ax.c2p(4, 1),
                    phase_amp_ax.c2p(4, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
                DashedLine(
                    phase_amp_ax.c2p(7, 1),
                    phase_amp_ax.c2p(7, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
            ]
        )

        self.play(LaggedStart(*[Create(m) for m in lines], lag_ratio=0.3))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(2)


class Overlap(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        pulse_rtn_ax = pulse_ax.copy()
        Group(pulse_ax, pulse_rtn_ax).arrange(
            DOWN,
            MED_LARGE_BUFF,
        )

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_rtn_x0 = VT(-~pw_plot)
        pulse_rtn_x1 = VT(0)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f)
        pulse_amp_2 = VT(0)
        pulse_amp_3 = VT(0)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0)
        pulse_t_start_3 = VT(0)
        pulse_f1_2 = VT(~pulse_f1)
        allow_1 = VT(1)
        allow_2 = VT(0)
        allow_3 = VT(0)
        pulse_start_offset = VT(0)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0 + ~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    0,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                # use_smoothing=False,
            )
        )
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: ~allow_1
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(~pulse_rtn_x0, ~min_x),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                # use_smoothing=False,
            )
        )

        target1 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_height(fh(self, 0.2))
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .to_edge(RIGHT, LARGE_BUFF)
            .set_y(pulse_rtn_ax.get_y())
        )

        self.add(pulse_ax, pulse_rtn_ax, pulse_rtn, pulse)

        self.wait(0.5)

        self.play(
            pulse_rtn_x1 @ ~pw_plot,
            pulse_rtn_x0 @ 0,
            target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        with_lfm = (
            Text("* with linear\n  frequency\n  modulation", font=FONT)
            .scale(0.4)
            .to_corner(UL, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                FadeIn(with_lfm),
                pulse_f1 @ (pulse_f * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(target1.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
            .next_to(target1, UP, 0)
            .shift(LEFT * 2)
            .shift(RIGHT / 2)
        )
        target3 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(target1.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET3_COLOR)
            .set_color(TARGET3_COLOR)
            .next_to(target1, DOWN, 0)
            .shift(LEFT * 2)
            .shift(RIGHT * 1.4)
        )

        self.play(pulse_ax.animate.shift(DOWN * 2))

        self.wait(0.5)

        pulse_2_opacity = VT(0)
        pulse_3_opacity = VT(0)
        pulse_t2_shift = VT(1.3)
        pulse_t3_shift = VT(1.3)

        pulse_rtn_t2 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                ),
                x_range=[
                    max(~pulse_rtn_x0, 0),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(UP * ~pulse_t2_shift)
            .set_stroke(opacity=~pulse_2_opacity)
        )
        pulse_rtn_t3 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(~pulse_rtn_x0, 0),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(DOWN * ~pulse_t3_shift)
            .set_stroke(opacity=~pulse_3_opacity)
        )
        self.add(pulse_rtn_t2, pulse_rtn_t3)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_ax.animate.shift(UP * 2),
                target1.animate.scale(0.5).shift(LEFT * 2),
                target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                pulse_2_opacity @ 1,
                pulse_amp_2 @ ~pulse_amp,
                target3.shift(RIGHT * 10).animate.shift(LEFT * 10),
                pulse_3_opacity @ 1,
                pulse_amp_3 @ ~pulse_amp,
                pulse_t_start_2 @ 0.1,
                pulse_t_start_3 @ 0.28,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        start = always_redraw(
            lambda: DashedLine(
                pulse_ax.c2p(~pulse_start_offset, 1),
                pulse_rtn_ax.c2p(~pulse_start_offset, -2),
                dash_length=DEFAULT_DASH_LENGTH * 3,
                color=YELLOW,
            )
        )

        self.play(Create(start))

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.1)

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.28)

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(start),
                pulse_start_offset @ 0,
                LaggedStart(
                    pulse_2_opacity @ 0,
                    pulse_t2_shift @ 0,
                    allow_2 @ 1,
                    lag_ratio=0.15,
                ),
                LaggedStart(
                    pulse_3_opacity @ 0,
                    pulse_t3_shift @ 0,
                    allow_3 @ 1,
                    lag_ratio=0.15,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.1, run_time=0.5)
        self.play(pulse_start_offset @ 0.2, run_time=0.5)
        self.play(pulse_start_offset @ 0.05, run_time=0.5)
        self.play(pulse_start_offset @ 0, run_time=0.5)

        self.wait(0.5)

        lfm_structure = (
            Text("LFM structured", font=FONT)
            .scale(0.8)
            .next_to(pulse_rtn, DOWN, MED_SMALL_BUFF)
        )
        to_label = CubicBezier(
            with_lfm.get_bottom() + [0, -0.1, 0],
            with_lfm.get_bottom() + [0, -3, 0],
            lfm_structure.get_left() + [-2, 0, 0],
            lfm_structure.get_left() + [-0.1, 0, 0],
        )

        self.play(LaggedStart(Create(to_label), Write(lfm_structure), lag_ratio=0.3))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        axes_group = Group(pulse.copy().shift(DOWN), pulse_rtn)
        self.play(
            LaggedStart(
                Uncreate(to_label),
                FadeOut(lfm_structure),
                target1.animate.shift(RIGHT * 6),
                target3.animate.shift(RIGHT * 6),
                target2.animate.shift(RIGHT * 6),
                self.camera.frame.animate.scale_to_fit_height(
                    axes_group.height * 1.8
                ).move_to(axes_group),
                pulse_ax.animate.shift(DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        xcorr = MathTex(
            r"R_{xy}(\tau) = \left(s_{tx} \star s_{rx}\right)(\tau)"
        ).next_to(pulse, UP)
        xcorr[0][1].set_color(TX_COLOR)
        xcorr[0][2].set_color(RX_COLOR)
        xcorr[0][8:11].set_color(TX_COLOR)
        xcorr[0][12:15].set_color(RX_COLOR)

        self.play(
            Group(pulse_ax, pulse_rtn_ax).animate.shift(DOWN * 0.7),
            xcorr.shift(UP * 5).animate.shift(DOWN * 5),
        )

        self.wait(0.5)

        time_seq = (
            MathTex(r"0,1,2, \ldots t_{max} \text{ s}")
            .scale(0.7)
            .next_to(xcorr[0][4], DOWN, MED_LARGE_BUFF * 1.4)
        )
        time_bez_l = CubicBezier(
            xcorr[0][4].get_bottom() + [0, -0.1, 0],
            xcorr[0][4].get_bottom() + [0, -0.8, 0],
            time_seq.get_corner(UL) + [0, 0.8, 0],
            time_seq.get_corner(UL) + [0, 0.1, 0],
        )
        time_bez_r = CubicBezier(
            xcorr[0][4].get_bottom() + [0, -0.1, 0],
            xcorr[0][4].get_bottom() + [0, -0.8, 0],
            time_seq.get_corner(UR) + [0, 0.8, 0],
            time_seq.get_corner(UR) + [0, 0.1, 0],
        )

        self.play(
            Group(pulse_ax, pulse_rtn_ax).animate.shift(DOWN * 0.3),
            LaggedStart(
                AnimationGroup(Create(time_bez_l), Create(time_bez_r)),
                *[FadeIn(m) for m in time_seq[0]],
                lag_ratio=0.08,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(*time_seq[0]),
                AnimationGroup(Uncreate(time_bez_l), Uncreate(time_bez_r)),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        new_axes = (
            Group(pulse_ax.copy(), pulse_rtn_ax.copy())
            .arrange(DOWN, -SMALL_BUFF)
            .set_y(Group(pulse, pulse_rtn).get_y())
        )
        new_cam = (
            self.camera.frame.copy()
            .scale_to_fit_height(new_axes.height * 1.2)
            .move_to(new_axes)
            .shift(LEFT * 3)
        )

        self.play(
            xcorr.animate.scale_to_fit_width(new_cam.width * 0.4).next_to(
                new_cam.get_top(),
                DOWN,
                MED_SMALL_BUFF,
            ),
            pulse_ax.animate.move_to(new_axes[0]),
            pulse_rtn_ax.animate.move_to(new_axes[1]),
            FadeOut(xcorr),
            self.camera.frame.animate.scale_to_fit_height(new_axes.height * 1.2)
            .move_to(new_axes)
            .shift(LEFT * 3),
        )

        self.wait(2)


class XCorr(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        pulse_rtn_ax = pulse_ax.copy()
        prod_ax = pulse_ax.copy()

        axes = Group(pulse_ax, pulse_rtn_ax).arrange(DOWN, -SMALL_BUFF)

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_rtn_x0 = VT(0)
        pulse_rtn_x1 = VT(~pw_plot)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f * 5)
        pulse_amp_2 = VT(0.3)
        pulse_amp_3 = VT(0.3)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0.1)
        pulse_t_start_3 = VT(0.28)
        allow_1 = VT(1)
        allow_2 = VT(1)
        allow_3 = VT(1)
        pulse_start_offset = VT(0)
        pulse_opacity = VT(1)
        pulse_rtn_opacity = VT(1)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~min_x,
                    min(
                        1,
                        max(
                            ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3),
                            ~pulse_start_offset + ~pw_plot,
                        ),
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                stroke_opacity=~pulse_opacity,
            )
        )
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: ~allow_1
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    ~min_x,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                stroke_opacity=~pulse_rtn_opacity,
            )
        )

        self.add(pulse, pulse_ax, pulse_rtn, pulse_rtn_ax)

        self.camera.frame.scale_to_fit_height(axes.height * 1.2).move_to(axes).shift(
            LEFT * 3
        )

        self.wait(0.5)

        self.play(pulse_start_offset @ -~pw_plot, min_x @ -~pw_plot)

        self.wait(0.5)

        times = MathTex(r"\times").scale(2).next_to(pulse, DOWN, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                pulse_rtn_ax.animate.shift(
                    DOWN
                    * ((pulse_rtn.get_top() - times.get_bottom())[1] + MED_SMALL_BUFF)
                ),
                GrowFromCenter(times),
                lag_ratio=0.3,
            )
        )

        pulse_samples = pulse_ax.get_riemann_rectangles(
            pulse,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            dx=0.005,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )
        pulse_rtn_samples = pulse_rtn_ax.get_riemann_rectangles(
            pulse_rtn,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            dx=0.005,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(m) for m in pulse_samples], lag_ratio=0.05),
                LaggedStart(*[FadeIn(m) for m in pulse_rtn_samples], lag_ratio=0.05),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        equal = MathTex(r"=").scale(2).next_to(pulse_rtn, DOWN, MED_SMALL_BUFF)
        prod_ax.set_x(pulse_ax.get_x())

        prod_ax.shift(
            DOWN * ((prod_ax.c2p(0, 0.5) - equal.get_bottom())[1] + MED_SMALL_BUFF)
        )

        prod = always_redraw(
            lambda: prod_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                * (
                    ~allow_1
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp,
                        phase=~pulse_phase_1,
                    )
                    + ~allow_2
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp_2,
                        phase=~pulse_phase_2,
                    )
                    + ~allow_3
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp_3,
                        phase=~pulse_phase_3,
                    )
                )
                / ~pulse_amp,
                x_range=[
                    ~min_x,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=ORANGE,
            )
        )

        all_axes = Group(pulse_ax, pulse_rtn_ax, prod_ax)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(all_axes.height * 1)
                .move_to(all_axes)
                .shift(LEFT * 3),
                GrowFromCenter(equal),
                lag_ratio=0.3,
            )
        )

        prod_samples = prod_ax.get_riemann_rectangles(
            prod,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            dx=0.005,
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )

        self.add(prod_ax)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        m.animate.set_opacity(0).set_y(prod_ax.c2p(0, 0)[1])
                        for m in pulse_samples
                    ],
                    lag_ratio=0.05,
                    run_time=8,
                ),
                LaggedStart(
                    *[
                        m.animate.set_opacity(0).set_y(prod_ax.c2p(0, 0)[1])
                        for m in pulse_rtn_samples
                    ],
                    lag_ratio=0.05,
                    run_time=8,
                ),
                AnimationGroup(
                    LaggedStart(
                        *[FadeIn(m) for m in prod_samples], lag_ratio=0.05, run_time=8
                    ),
                    Create(
                        prod.set_z_index(-1),
                        run_time=8,
                        rate_func=rate_functions.linear,
                    ),
                ),
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*prod_samples))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        xcorr_ax = pulse_ax.copy().shift(
            UP * (prod_ax.c2p(0, -0.5) - pulse_ax.c2p(0, 2.2))[1]
        )
        xcorr_plot = xcorr_ax.plot(
            lambda t: 0.3 * np.sin(2 * PI * 3 * t),
            color=GREEN,
            x_range=[
                -~pw_plot,
                ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3),
                1 / 200,
            ],
        )

        def get_xcorr():
            fs = 1e3
            t_discreet = np.arange(0, 1, 1 / fs)
            tau = 0.3
            pulse = np.array(
                [
                    chirp_pulse(
                        t_val=t_val,
                        pulse_start=0,
                        pulse_width=tau,
                        f0=20,
                        f1=100,
                        amp=1,
                        phase=0,
                        ramp="linear",
                    )
                    for t_val in t_discreet
                ]
            )
            pulse_starts = [tau, 0.1 + tau, 0.28 + tau]
            rtn = np.array(
                np.sum(
                    [
                        [
                            chirp_pulse(
                                t_val=t_val,
                                pulse_start=ps,
                                pulse_width=tau,
                                f0=20,
                                f1=100,
                                amp=1,
                                phase=0,
                                ramp="linear",
                            )
                            for t_val in t_discreet
                        ]
                        for ps in pulse_starts
                    ],
                    axis=0,
                )
            )

            h_matched = np.conj(pulse[t_discreet < tau][::-1])
            rtn_matched = signal.fftconvolve(rtn, h_matched, mode="full")
            rtn_matched /= rtn_matched.max()
            rtn_matched *= 2
            t_full = np.arange(rtn_matched.size) / fs - tau
            func = interp1d(t_full - tau, rtn_matched, fill_value="extrapolate")
            return func

        xcorr_func = get_xcorr()

        xcorr_amp = VT(1)
        xcorr_plot = always_redraw(
            lambda: xcorr_ax.plot(
                lambda t: xcorr_func(t) * ~xcorr_amp,
                color=GREEN,
                x_range=[
                    -~pw_plot,
                    min(
                        ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3),
                        ~pulse_start_offset,
                    ),
                    1 / 200,
                ],
            )
        )

        all_plots = Group(
            pulse,
            pulse_rtn,
            prod,
            Line(
                xcorr_ax.c2p(-~pw_plot, -0.5),
                xcorr_ax.c2p(
                    ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3), -0.5
                ),
            ),
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_plots.height * 1.2
                ).move_to(all_plots),
                Create(xcorr_plot),
            )
        )

        def get_l_bez():
            return CubicBezier(
                prod_ax.c2p(-~pw_plot, -~pulse_amp) + [0, -0.1, 0],
                prod_ax.c2p(-~pw_plot, -~pulse_amp) + [0, -0.5, 0],
                xcorr_ax.c2p(~pulse_start_offset, 1.4) + [0, 1, 0],
                xcorr_ax.c2p(~pulse_start_offset, 1.4) + [0, 0.3, 0],
            )

        def get_r_bez():
            return CubicBezier(
                prod_ax.c2p(
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    -~pulse_amp,
                )
                + [0, -0.1, 0],
                prod_ax.c2p(
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    -~pulse_amp,
                )
                + [0, -0.5, 0],
                xcorr_ax.c2p(~pulse_start_offset, 1.4) + [0, 1, 0],
                xcorr_ax.c2p(~pulse_start_offset, 1.4) + [0, 0.3, 0],
            )

        l_bez = always_redraw(get_l_bez)
        r_bez = always_redraw(get_r_bez)
        self.play(Create(l_bez), Create(r_bez))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pulse_start_offset @ (-0.05),
            run_time=10,
        )

        self.wait(0.5)

        # target1_box = Polygon(
        #     pulse_rtn_ax.c2p(0, -0.6),
        #     pulse_rtn_ax.c2p(0, 0.6),
        #     pulse_rtn_ax.c2p(~pw_plot, 0.6),
        #     pulse_rtn_ax.c2p(~pw_plot, -0.6),
        #     stroke_opacity=0,
        #     fill_opacity=0.5,
        #     fill_color=PURPLE,
        # )

        target1_l = DashedLine(
            pulse_ax.c2p(0, 0.6),
            pulse_rtn_ax.c2p(0, -0.6),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )
        target1_r = DashedLine(
            pulse_ax.c2p(~pw_plot, 0.6),
            pulse_rtn_ax.c2p(~pw_plot, -0.6),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )

        self.play(
            LaggedStart(
                FadeIn(target1_l),
                FadeIn(target1_r),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            pulse_start_offset @ (0.05),
            run_time=10,
        )
        self.play(
            pulse_start_offset
            @ (~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            run_time=20,
        )

        self.wait(0.5)

        plot_group = Group(
            pulse.copy(),
            pulse_rtn.copy().shift(UP * 0.5),
            xcorr_plot.copy().shift(UP * 2),
        )

        xcorr_label = (
            Text("Cross\nCorrelation", font=FONT)
            .scale(0.7)
            .next_to(plot_group[2], LEFT, LARGE_BUFF)
        )
        tx_label = (
            Text("Transmit", font=FONT)
            .scale(0.7)
            .next_to(xcorr_label, UP, aligned_edge=LEFT)
            .set_y(plot_group[0].get_y())
        )
        rx_label = (
            Text("Receive", font=FONT)
            .scale(0.7)
            .next_to(xcorr_label, UP, aligned_edge=LEFT)
            .set_y(plot_group[1].get_y())
        )
        xcorr = MathTex(
            r"R_{xy}(\tau) = \left(s_{tx} \star s_{rx}\right)(\tau)"
        ).next_to(xcorr_label, DOWN, MED_LARGE_BUFF)
        xcorr[0][0].set_color(GREEN)
        # xcorr[0][3:6].set_color(GREEN)
        xcorr[0][1].set_color(TX_COLOR)
        xcorr[0][2].set_color(RX_COLOR)
        xcorr[0][8:11].set_color(TX_COLOR)
        xcorr[0][12:15].set_color(RX_COLOR)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(l_bez, r_bez, times, equal, prod, target1_l, target1_r),
                LaggedStart(
                    self.camera.frame.animate.scale_to_fit_height(
                        plot_group.height * 1.2
                    )
                    .move_to(plot_group)
                    .shift(LEFT * 2),
                    pulse_rtn_ax.animate.shift(UP * 0.5),
                    xcorr_ax.animate.shift(UP * 2),
                    Write(tx_label),
                    Write(rx_label),
                    Write(xcorr_label),
                    FadeIn(xcorr),
                    lag_ratio=0.3,
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        one = (
            Text("1", font=FONT)
            .scale(0.7)
            .next_to(xcorr_ax.i2gp(0, xcorr_plot), UP, MED_SMALL_BUFF)
        )
        two = (
            Text("2", font=FONT)
            .scale(0.7)
            .next_to(xcorr_ax.i2gp(0.1, xcorr_plot), UP, MED_SMALL_BUFF)
        )
        three = (
            Text("3", font=FONT)
            .scale(0.7)
            .next_to(xcorr_ax.i2gp(0.28, xcorr_plot), UP, MED_SMALL_BUFF)
        )

        tau1 = MathTex(r"\tau_1").next_to(xcorr_ax.c2p(0, -0.8), DOWN, MED_SMALL_BUFF)
        tau2 = MathTex(r"\tau_2").next_to(xcorr_ax.c2p(0.1, -0.8), DOWN, MED_SMALL_BUFF)
        tau3 = MathTex(r"\tau_3").next_to(
            xcorr_ax.c2p(0.28, -0.8), DOWN, MED_SMALL_BUFF
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    GrowFromCenter(one),
                    TransformFromCopy(xcorr[0][4], tau1[0], path_arc=-PI / 3),
                ),
                AnimationGroup(
                    GrowFromCenter(two),
                    TransformFromCopy(xcorr[0][4], tau2[0], path_arc=-PI / 3),
                ),
                AnimationGroup(
                    GrowFromCenter(three),
                    TransformFromCopy(xcorr[0][4], tau3[0], path_arc=-PI / 3),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        larrow1 = Arrow(xcorr_ax.c2p(-0.08, 0.5), xcorr_ax.c2p(-0.01, 0.5), buff=0)
        rarrow1 = Arrow(xcorr_ax.c2p(0.08, 0.5), xcorr_ax.c2p(0.01, 0.5), buff=0)
        larrow2 = Arrow(
            xcorr_ax.c2p(0.1 + -0.08, 0.9), xcorr_ax.c2p(-0.01 + 0.1, 0.9), buff=0
        )
        rarrow2 = Arrow(
            xcorr_ax.c2p(0.1 + 0.08, 0.9), xcorr_ax.c2p(0.01 + 0.1, 0.9), buff=0
        )
        larrow3 = Arrow(
            xcorr_ax.c2p(0.28 + -0.08, 1.3), xcorr_ax.c2p(-0.01 + 0.28, 1.3), buff=0
        )
        rarrow3 = Arrow(
            xcorr_ax.c2p(0.28 + 0.08, 1.3), xcorr_ax.c2p(0.01 + 0.28, 1.3), buff=0
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                AnimationGroup(GrowArrow(larrow1), GrowArrow(rarrow1)),
                AnimationGroup(GrowArrow(larrow2), GrowArrow(rarrow2)),
                AnimationGroup(GrowArrow(larrow3), GrowArrow(rarrow3)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()

        matched_filter_label = Text("Matched Filter", font=FONT).next_to(
            xcorr, DOWN, LARGE_BUFF * 1.5
        )
        mfl_bez_l = CubicBezier(
            xcorr.get_bottom() + [0, -0.1, 0],
            xcorr.get_bottom() + [0, -1, 0],
            matched_filter_label.get_corner(UL) + [-0.1, 1, 0],
            matched_filter_label.get_corner(UL) + [-0.1, 0.1, 0],
        )
        mfl_bez_r = CubicBezier(
            xcorr.get_bottom() + [0, -0.1, 0],
            xcorr.get_bottom() + [0, -1, 0],
            matched_filter_label.get_corner(UR) + [0.1, 1, 0],
            matched_filter_label.get_corner(UR) + [0.1, 0.1, 0],
        )

        mfl_group = Group(
            xcorr, xcorr_label, mfl_bez_l, mfl_bez_r, matched_filter_label, xcorr_plot
        )

        self.play(
            LaggedStart(
                FadeOut(
                    one,
                    two,
                    three,
                    tau1,
                    tau2,
                    tau3,
                    larrow1,
                    rarrow1,
                    larrow2,
                    rarrow2,
                    larrow3,
                    rarrow3,
                ),
                self.camera.frame.animate.scale_to_fit_height(
                    mfl_group.height * 1.3
                ).move_to(mfl_group),
                AnimationGroup(
                    Create(mfl_bez_l),
                    Create(mfl_bez_r),
                ),
                Write(matched_filter_label),
            )
        )

        self.wait(0.5)

        mf_eqn = (
            MathTex(r"\Rightarrow s_{rx}(t) \circledast s^{*}_{tx}(-t)")
            .scale(1.3)
            .next_to(matched_filter_label, RIGHT)
        )

        self.play(GrowFromCenter(mf_eqn[0][0]))

        self.wait(0.5)

        self.play(GrowFromCenter(mf_eqn[0][7]))

        self.wait(0.5)

        self.play(GrowFromCenter(mf_eqn[0][1:7]))

        self.wait(0.5)

        self.play(GrowFromCenter(mf_eqn[0][12:]))

        self.wait(0.5)

        self.play(GrowFromCenter(mf_eqn[0][9]))

        self.wait(0.5)

        self.play(GrowFromCenter(mf_eqn[0][8]), GrowFromCenter(mf_eqn[0][10:12]))

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore().scale(1.2).shift(DOWN))

        self.wait(0.5)

        rx_arrow = CubicBezier(
            mf_eqn[0][1:7].get_top() + [0, 0.1, 0],
            mf_eqn[0][1:7].get_top() + [-0.5, 2, 0],
            pulse_rtn.get_bottom() + [-2, -1, 0],
            pulse_rtn.get_bottom() + [-1.5, 0.3, 0],
            color=RX_COLOR,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        tx_arrow = CubicBezier(
            mf_eqn[0][8:].get_top() + [0, 0.1, 0],
            mf_eqn[0][8:].get_top() + [0, 2, 0],
            pulse.get_bottom() + [3, -3, 0],
            pulse.get_bottom() + [3, -0.1, 0],
            color=TX_COLOR,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(
            Create(rx_arrow),
            mf_eqn[0][1:7].animate.set_color(RX_COLOR),
        )

        self.wait(0.5)

        self.play(
            Create(tx_arrow),
            mf_eqn[0][8:].animate.set_color(TX_COLOR),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale(0.9).shift(UP * 0.2),
            xcorr_label.animate.set_color(GREEN),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(matched_filter_label, tx_arrow, rx_arrow, *mf_eqn[0]),
                    Uncreate(mfl_bez_l),
                    Uncreate(mfl_bez_r),
                ),
                AnimationGroup(
                    pulse_opacity @ 0.3,
                    pulse_rtn_opacity @ 0.3,
                ),
                self.camera.frame.animate.scale(1.5).shift(UP * 1.5),
                xcorr_amp @ 4,
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        pulse_ltail = VT(1)
        pulse_new = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~pulse_start_offset - ~pulse_ltail,
                    ~pulse_start_offset + ~pw_plot,
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                stroke_opacity=1,
            )
        )

        self.play(
            pulse_opacity @ 1,
            pulse_rtn_opacity @ 0,
            self.camera.frame.animate.scale_to_fit_height(pulse.height * 4).move_to(
                pulse_ax.c2p(~pulse_start_offset + ~pw_plot / 2)
            ),
        )

        self.add(pulse_new)
        self.remove(pulse)

        self.wait(0.5)

        self.play(pulse_ltail @ 0)

        self.wait(0.5)

        self.play(pulse_start_offset - 0.2)

        self.wait(0.5)

        self.play(pw_plot @ (~pw_plot * 2), pulse_amp @ (~pulse_amp / 2))

        self.wait(0.5)

        self.play(
            pulse_start_offset + 0.4,
            pw_plot @ (~pw_plot / 6),
            pulse_amp @ (~pulse_amp * 6),
        )

        self.wait(0.5)

        self.remove(tx_label, rx_label)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                xcorr_plot.height * 0.85
            ).move_to(xcorr_plot),
            pulse_ax.animate.shift(
                xcorr_ax.c2p(0, 0) - pulse_ax.c2p(~pulse_start_offset + ~pw_plot / 2, 0)
            ),
        )

        self.wait(0.5)

        self.play(
            pw_plot @ (~pw_plot * 0.2),
            pulse_amp @ 7.65,
            pulse_f1 @ (~pulse_f1 * 4),
            pulse_start_offset + (~pw_plot / 2),
        )

        self.wait(0.5)

        pulse_compression = (
            Text("Pulse Compression", font=FONT)
            .scale_to_fit_width(fw(self, 0.6))
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self) * 2)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self) * 2),
                Write(pulse_compression),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(pulse_compression))

        self.wait(2)


class RRes(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        self.wait(0.5)

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_rtn_x0 = VT(0)
        pulse_rtn_x1 = VT(~pw_plot)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f * 5)
        pulse_amp_2 = VT(0.3)
        pulse_amp_3 = VT(0.3)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0.1)
        pulse_t_start_3 = VT(0.28)
        allow_1 = VT(1)
        allow_2 = VT(1)
        allow_3 = VT(1)
        pulse_start_offset = VT(0)
        pulse_opacity = VT(1)
        pulse_rtn_opacity = VT(1)
        pulse_ltail = VT(0)

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )

        pw_plot = VT(0.3)
        pulse_amp = VT(0.5)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~pulse_start_offset - ~pulse_ltail,
                    ~pulse_start_offset + ~pw_plot,
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                stroke_opacity=1,
            )
        )

        target1 = (
            SVGMobject("../props/static/plane.svg")
            .rotate(PI * 0.75)
            .scale_to_fit_width(fw(self, 0.15))
            .next_to(self.camera.frame.get_center(), RIGHT, 0)
            .set_fill(TARGET1_COLOR)
            .set_color(TARGET1_COLOR)
        )

        # TODO: add animation here
        self.play(Create(pulse))

        next_point = target1.get_left()
        offset = pulse_ax.p2c(next_point)[0] - ~pw_plot

        self.play(
            target1.shift(RIGHT * 8).animate.shift(LEFT * 8),
            pulse_start_offset @ offset,
            self.camera.frame.animate.scale(0.8),
        )

        self.wait(0.5)

        energy = MathTex(r"E = P \cdot t").next_to(pulse, UP, LARGE_BUFF)

        energy_bez_l = CubicBezier(
            pulse.get_corner(UL) + [0, 0.1, 0],
            pulse.get_corner(UL) + [0, 1, 0],
            energy.get_bottom() + [0, -1, 0],
            energy.get_bottom() + [0, -0.1, 0],
        )
        energy_bez_r = CubicBezier(
            pulse.get_corner(UR) + [0, 0.1, 0],
            pulse.get_corner(UR) + [0, 1, 0],
            energy.get_bottom() + [0, -1, 0],
            energy.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(energy_bez_l), Create(energy_bez_r)),
                FadeIn(energy, shift=DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        xcorr_ax = pulse_ax.copy().shift(
            pulse_ax.c2p(~pulse_start_offset + ~pw_plot / 2, -1.5)
            - pulse_ax.c2p(0, 2.2)
        )

        xcorr_amp = VT(1)

        xcorr_x_offset = VT(0)

        def get_xcorr():
            fs = 1e3
            t_discreet = np.arange(0, 1, 1 / fs)
            tau = 0.3
            pulse = np.array(
                [
                    chirp_pulse(
                        t_val=t_val,
                        pulse_start=0,
                        pulse_width=tau,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=1,
                        phase=0,
                        ramp="linear",
                    )
                    for t_val in t_discreet
                ]
            )
            pulse_starts = [tau]
            rtn = np.array(
                np.sum(
                    [
                        [
                            chirp_pulse(
                                t_val=t_val,
                                pulse_start=ps,
                                pulse_width=tau,
                                f0=pulse_f,
                                f1=~pulse_f1,
                                amp=1,
                                phase=0,
                                ramp="linear",
                            )
                            for t_val in t_discreet
                        ]
                        for ps in pulse_starts
                    ],
                    axis=0,
                )
            )

            h_matched = np.conj(pulse[t_discreet < tau][::-1])
            rtn_matched = signal.fftconvolve(rtn, h_matched, mode="full")
            rtn_matched /= rtn_matched.max()
            rtn_matched *= 2 * ~xcorr_amp
            t_full = np.arange(rtn_matched.size) / fs - tau
            func = interp1d(t_full - tau, rtn_matched, fill_value="extrapolate")

            xcorr_plot = always_redraw(
                lambda: xcorr_ax.plot(
                    func,
                    color=GREEN,
                    x_range=[
                        -~pw_plot,
                        ~pw_plot - ~xcorr_x_offset,
                        1 / 200,
                    ],
                )
            )
            return xcorr_plot

        xcorr_plot = always_redraw(get_xcorr)

        all_stuff = Group(pulse, energy, target1, xcorr_plot)

        pulse_to_xcorr = Arrow(
            pulse.get_bottom(), [pulse.get_bottom()[0], xcorr_plot.get_top()[1], 0]
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(all_stuff.height * 1.1)
                .move_to(all_stuff)
                .shift(UP / 2),
                GrowArrow(pulse_to_xcorr),
                Create(xcorr_plot),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target_range_l = Line(
            pulse.get_corner(UL) + [0, 0.1, 0],
            pulse.get_corner(UL) + [0, 0.3, 0],
        )
        target_range_r = Line(
            pulse.get_corner(UR) + [0, 0.1, 0],
            pulse.get_corner(UR) + [0, 0.3, 0],
        )
        target_range = Line(
            target_range_l.get_midpoint(), target_range_r.get_midpoint()
        )
        target_qmark = (
            Text("target?", font=FONT)
            .scale_to_fit_width(pulse.width * 0.8)
            .next_to(target_range, UP, SMALL_BUFF)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(energy, energy_bez_l, energy_bez_r),
                    target1.animate.set_opacity(0.2),
                    pulse_to_xcorr.animate.set_opacity(0.2),
                ),
                self.camera.frame.animate.scale_to_fit_width(pulse.width * 3)
                .move_to(pulse)
                .shift(UP / 2),
                LaggedStart(
                    Create(target_range_l),
                    Create(target_range),
                    Create(target_range_r),
                    Write(target_qmark),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rres_relation = MathTex(r"\Delta R \sim \tau").next_to(
            target_range, UP, MED_SMALL_BUFF
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(target_qmark),
                    Uncreate(target_range_l),
                    Uncreate(target_range),
                    Uncreate(target_range_r),
                ),
                LaggedStart(*[FadeIn(m) for m in rres_relation[0]], lag_ratio=0.08),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        rres_eqn = MathTex(r"\Delta R = \frac{c \tau}{2}").move_to(rres_relation)

        self.play(
            LaggedStart(
                ReplacementTransform(rres_relation[0][:2], rres_eqn[0][:2]),
                ShrinkToCenter(rres_relation[0][2]),
                GrowFromCenter(rres_eqn[0][2]),
                ReplacementTransform(rres_relation[0][3], rres_eqn[0][4]),
                GrowFromCenter(rres_eqn[0][3]),
                GrowFromCenter(rres_eqn[0][5]),
                GrowFromCenter(rres_eqn[0][6]),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        quick_note = (
            Text("Quick Note:", font=FONT)
            .scale_to_fit_width(fw(self, 0.5))
            .next_to(self.camera.frame.get_top(), DOWN)
            .shift(UP * fh(self))
        )

        xcorr = MathTex(
            r"R_{xy}(\tau) = \left(s_{tx} \star s_{rx}\right)(\tau)"
        ).move_to(self.camera.frame.copy().shift(UP * fh(self)))

        self.add(quick_note, xcorr)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * fh(self)),
                rres_eqn.animate.next_to(xcorr, DOWN, MED_LARGE_BUFF),
            )
        )

        self.wait(0.5)

        tau_bez_l = CubicBezier(
            rres_eqn[0][4].get_top() + [0, 0.1, 0],
            rres_eqn[0][4].get_top() + [0, 0.5, 0],
            xcorr[0][4].get_bottom() + [0, -0.5, 0],
            xcorr[0][4].get_bottom() + [0, -0.1, 0],
        )
        tau_bez_r = CubicBezier(
            rres_eqn[0][4].get_top() + [0, 0.1, 0],
            rres_eqn[0][4].get_top() + [0, 0.5, 0],
            xcorr[0][-2].get_bottom() + [0, -0.5, 0],
            xcorr[0][-2].get_bottom() + [0, -0.1, 0],
        )
        xl = (
            Text("x", font=FONT, color=RED).scale(0.7).move_to(tau_bez_l.get_midpoint())
        )
        xr = (
            Text("x", font=FONT, color=RED).scale(0.7).move_to(tau_bez_r.get_midpoint())
        )

        self.play(
            LaggedStart(
                rres_eqn[0][4].animate.set_color(GREEN),
                Create(tau_bez_l),
                GrowFromCenter(xl),
                xcorr[0][4].animate.set_color(YELLOW),
                Create(tau_bez_r),
                GrowFromCenter(xr),
                xcorr[0][-2].animate.set_color(YELLOW),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        pw_arrow = Arrow(
            rres_eqn[0][4].get_right() + [1, -0.5, 0],
            rres_eqn[0][4].get_right(),
            buff=SMALL_BUFF,
        )

        self.play(GrowArrow(pw_arrow))

        self.wait(0.5)

        lag_arrow_l = Arrow(
            xcorr[0][4].get_top() + [-0.5, 1, 0],
            xcorr[0][4].get_top(),
            buff=SMALL_BUFF,
        )
        lag_arrow_r = Arrow(
            xcorr[0][-2].get_top() + [0.5, 1, 0],
            xcorr[0][-2].get_top(),
            buff=SMALL_BUFF,
        )

        self.play(
            LaggedStart(
                FadeOut(pw_arrow),
                AnimationGroup(GrowArrow(lag_arrow_l), GrowArrow(lag_arrow_r)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(xl, xr, lag_arrow_l, lag_arrow_r),
                self.camera.frame.animate.scale_to_fit_height(
                    all_stuff.height * 1.1
                ).move_to(all_stuff),
                AnimationGroup(
                    target1.animate.set_opacity(1),
                    pulse_to_xcorr.animate.set_opacity(1),
                    rres_eqn.animate.next_to(pulse, UP, MED_SMALL_BUFF),
                ),
                lag_ratio=0.3,
            )
        )
        self.play(rres_eqn[0][4].animate.set_color(WHITE))

        self.wait(0.5)

        self.play(pulse_f1 @ pulse_f, run_time=3)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        tri_up = Line(
            xcorr_ax.c2p(-~pw_plot, 0),
            xcorr_ax.c2p(0, 2),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).shift(UP * 0.1)
        tri_down = Line(
            xcorr_ax.c2p(0, 2),
            xcorr_ax.c2p(~pw_plot, 0),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).shift(UP * 0.1)

        self.play(LaggedStart(Create(tri_up), Create(tri_down), lag_ratio=0.7))

        self.wait(0.5)

        pulse_rtn_ax = pulse_ax.copy().next_to(xcorr_ax, LEFT).shift(DOWN + LEFT)
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    offset + ~pulse_ltail,
                    offset + ~pw_plot,
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                stroke_opacity=1,
            )
        )

        new_group = Group(
            pulse_ax.copy().next_to(pulse_rtn_ax, UP, -MED_SMALL_BUFF),
            pulse_rtn_ax,
            xcorr_plot,
        )
        conv = MathTex(r"\circledast").move_to(new_group[:2]).set_x(pulse_rtn.get_x())
        equal = (
            MathTex("=").next_to(pulse_rtn, RIGHT).set_y(conv.get_y()).shift(RIGHT * 2)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                FadeOut(pulse_to_xcorr, target1),
                AnimationGroup(
                    pulse_ax.animate.next_to(pulse_rtn_ax, UP, -MED_SMALL_BUFF),
                    rres_eqn.animate.next_to(
                        pulse_ax.copy()
                        .next_to(pulse_rtn_ax, UP, -MED_SMALL_BUFF)
                        .c2p(0.35, 0.5),
                        UP,
                    ),
                ),
                self.camera.frame.animate.scale_to_fit_width(new_group.width * 1.3)
                .move_to(new_group)
                .shift(RIGHT + UP / 3),
                GrowFromCenter(conv),
                Create(pulse_rtn),
                GrowFromCenter(equal),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        pulse_start_offset_old = ~pulse_start_offset
        xcorr_x_offset_old = ~xcorr_x_offset
        self.play(
            pulse_start_offset - (~pw_plot),
            self.camera.frame.animate.shift(LEFT),
            Uncreate(tri_down),
            Uncreate(tri_up),
            xcorr_x_offset @ (~pw_plot * 2 - 0.01),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pulse_start_offset @ pulse_start_offset_old,
            xcorr_x_offset @ ~pw_plot,
            run_time=6,
        )

        self.wait(0.5)

        self.play(
            pulse_start_offset + ~pw_plot,
            xcorr_x_offset - ~pw_plot,
            run_time=6,
        )

        self.wait(0.5)

        pulse_to_xcorr = Arrow(
            pulse.copy()
            .shift(DOWN * (pulse_ax.c2p(0, 0) - conv.get_center())[1] + LEFT * 2)
            .get_right(),
            [
                xcorr_plot.get_left()[0],
                pulse.copy()
                .shift(DOWN * (pulse_ax.c2p(0, 0) - conv.get_center())[1])
                .get_right()[1],
                0,
            ],
        )

        self.play(
            LaggedStart(
                FadeOut(pulse_rtn, conv, equal),
                pulse_ax.animate.shift(
                    DOWN * (pulse_ax.c2p(0, 0) - conv.get_center())[1]
                ),
                rres_eqn.animate.shift(
                    DOWN * (pulse_ax.c2p(0, 0) - conv.get_center())[1]
                ),
                pulse_start_offset - ~pw_plot,
                GrowArrow(pulse_to_xcorr),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rres_bw_eqn = MathTex(r"\Delta R \approx \frac{c}{2 B}").move_to(rres_eqn, LEFT)

        tri_up = Line(
            xcorr_ax.c2p(-~pw_plot, 0),
            xcorr_ax.c2p(0, 2),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).shift(UP * 0.1)
        tri_down = Line(
            xcorr_ax.c2p(0, 2),
            xcorr_ax.c2p(~pw_plot, 0),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).shift(UP * 0.1)

        qmark = Text("?", font=FONT).move_to(rres_eqn[0][3:], UP).shift(DOWN * 0.1)

        self.play(
            LaggedStart(
                rres_eqn[0][3:].animate.shift(UP * 1.5).set_opacity(0.2),
                GrowFromCenter(qmark),
                pulse_f1.animate(run_time=8).set_value(~pulse_f1 * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(Create(tri_up), Create(tri_down), lag_ratio=0.7))

        self.wait(0.5)

        self.play(FadeOut(tri_up, tri_down, qmark))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(rres_eqn[0][:2], rres_bw_eqn[0][:2]),
                ReplacementTransform(rres_eqn[0][2], rres_bw_eqn[0][2]),
                LaggedStart(
                    *[GrowFromCenter(m) for m in rres_bw_eqn[0][3:]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.remove(pulse_rtn, xcorr_plot)

        self.play(
            FadeOut(pulse_to_xcorr),
            self.camera.frame.animate.scale_to_fit_height(
                Group(pulse, rres_bw_eqn).height * 1.5
            ).move_to(Group(pulse, rres_bw_eqn)),
            AnimationGroup(
                pulse_start_offset - (~pw_plot / 2),
                pw_plot @ (~pw_plot * 2),
            ),
        )

        self.wait(0.5)

        infty_b = MathTex(r"B \rightarrow \infty ?").next_to(pulse, DOWN)

        new_group = Group(rres_bw_eqn, pulse, infty_b)

        self.play(
            pulse_f1.animate(run_time=6).set_value(~pulse_f1 * 2),
            LaggedStart(*[FadeIn(m) for m in infty_b[0]], lag_ratio=0.1),
            self.camera.frame.animate.scale_to_fit_height(
                new_group.height * 1.4
            ).move_to(new_group),
        )

        self.wait(0.5)

        pcb = Text("PCB layout", font=FONT).scale(0.6)
        regulatory = ImageMobject(
            "../props/static/band_allocation_chart.jpg"
        ).scale_to_fit_width(fw(self, 0.35))
        etc = Text("etc.", font=FONT).scale(0.6)
        reasons_group = (
            Group(pcb, regulatory, etc)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(infty_b, DOWN, LARGE_BUFF * 3)
        )
        regulatory.shift(DOWN)
        etc.shift(DOWN / 2)
        bez_l = CubicBezier(
            infty_b.get_bottom() + [0, -0.1, 0],
            infty_b.get_bottom() + [0, -1, 0],
            pcb.get_top() + [0, 1, 0],
            pcb.get_top() + [0, 0.1, 0],
        )
        bez_m = CubicBezier(
            infty_b.get_bottom() + [0, -0.1, 0],
            infty_b.get_bottom() + [0, -1, 0],
            regulatory.get_top() + [0, 1, 0],
            regulatory.get_top() + [0, 0.1, 0],
        )
        bez_r = CubicBezier(
            infty_b.get_bottom() + [0, -0.1, 0],
            infty_b.get_bottom() + [0, -1, 0],
            etc.get_top() + [0, 1, 0],
            etc.get_top() + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(bez_l),
                    Create(bez_m),
                    Create(bez_r),
                    lag_ratio=0.2,
                ),
                self.camera.frame.animate.move_to(reasons_group),
                LaggedStart(
                    Write(pcb),
                    GrowFromCenter(regulatory),
                    Write(etc),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)


class Tradeoffs(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        pulse_f = 20
        pulse_f1 = VT(pulse_f * 5)
        pw_plot = VT(0.3)
        is_abs = VT(0)

        xcorr_ax = (
            Axes(
                x_range=[-~pw_plot, ~pw_plot, 0.5],
                y_range=[-1, 2, 0.5],
                tips=False,
                x_length=fw(self, 0.7),
                y_length=fh(self, 0.7),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        self.add(xcorr_ax)

        xcorr_amp = VT(1)

        xcorr_x_offset = VT(~pw_plot)

        fs = 2**10

        rtn_amp_1 = VT(1)
        rtn_amp_2 = VT(0)

        def get_xcorr():
            t_discreet = np.arange(0, 1, 1 / fs)
            tau = 0.3
            pulse = np.array(
                [
                    chirp_pulse(
                        t_val=t_val,
                        pulse_start=0,
                        pulse_width=tau,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=1,
                        phase=0,
                        ramp="linear",
                    )
                    for t_val in t_discreet
                ]
            )
            pulse_starts = [tau, tau * 1.05]
            pulse_amps = [~rtn_amp_1, ~rtn_amp_2]
            rtn = np.array(
                np.sum(
                    [
                        [
                            pa
                            * chirp_pulse(
                                t_val=t_val,
                                pulse_start=ps,
                                pulse_width=tau,
                                f0=pulse_f,
                                f1=~pulse_f1,
                                amp=1,
                                phase=0,
                                ramp="linear",
                            )
                            for t_val in t_discreet
                        ]
                        for ps, pa in zip(pulse_starts, pulse_amps)
                    ],
                    axis=0,
                )
            )

            h_matched = np.conj(pulse[t_discreet < tau][::-1])
            rtn_matched = signal.fftconvolve(rtn, h_matched, mode="full")
            rtn_matched /= rtn_matched.max()
            rtn_matched *= 2 * ~xcorr_amp
            t_full = np.arange(rtn_matched.size) / fs - tau

            # rtn_matched_db = np.clip(10 * np.log10(np.abs(rtn_matched)), -50, None)
            # rtn_matched_db -= np.nanmin(rtn_matched_db)
            # rtn_matched_db /= np.nanmax(rtn_matched_db)
            # rtn_matched_db *= rtn_matched.max()
            # rtn_matched_db *= 4
            # rtn_matched_db -= 6

            func = interp1d(
                t_full - tau,
                rtn_matched * (1 - ~is_abs) + (np.abs(rtn_matched) * 1.5 - 1) * ~is_abs,
                fill_value="extrapolate",
            )

            xcorr_plot = always_redraw(
                lambda: xcorr_ax.plot(
                    func,
                    color=GREEN,
                    x_range=[
                        -~pw_plot + ~xcorr_x_offset,
                        ~pw_plot - ~xcorr_x_offset,
                        1 / (fs * 1),
                    ],
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                )
            )
            return xcorr_plot

        xcorr_plot = always_redraw(get_xcorr)
        self.add(xcorr_plot)

        self.wait(0.5)

        self.play(xcorr_x_offset @ 0, run_time=2)

        self.wait(0.5)

        non_zero_label = MathTex(r"\Delta R > 0").next_to(
            xcorr_ax.c2p(0, 2), UP, LARGE_BUFF
        )

        l_bez = CubicBezier(
            xcorr_ax.c2p(-0.01, 1) + [0, 0, 0],
            xcorr_ax.c2p(-0.01, 1) + [0, 1, 0],
            non_zero_label.get_corner(DL) + [-0.1, -1, 0],
            non_zero_label.get_corner(DL) + [-0.1, 0, 0],
        )
        r_bez = CubicBezier(
            xcorr_ax.c2p(0.01, 1) + [0, 0, 0],
            xcorr_ax.c2p(0.01, 1) + [0, 1, 0],
            non_zero_label.get_corner(DR) + [0.1, -1, 0],
            non_zero_label.get_corner(DR) + [0.1, 0, 0],
        )
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP),
                AnimationGroup(
                    Create(l_bez),
                    Create(r_bez),
                ),
                LaggedStart(*[FadeIn(m) for m in non_zero_label[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(*non_zero_label[0]),
                AnimationGroup(
                    Uncreate(l_bez),
                    Uncreate(r_bez),
                ),
                self.camera.frame.animate.shift(DOWN),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # self.play(is_db @ 1)

        sinc_opacity = VT(0)
        sinc_amp = VT(0)
        sinc_f = VT(0)
        sinc = always_redraw(
            lambda: xcorr_ax.plot(
                lambda t: ~sinc_amp * np.sinc(2 * PI * ~sinc_f * t),
                x_range=[-~pw_plot, ~pw_plot, 1 / fs],
                color=YELLOW,
                stroke_opacity=~sinc_opacity,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
        )
        self.add(sinc)

        self.next_section(skip_animations=skip_animations(True))

        xcorr_line_legend = Line(
            ORIGIN,
            RIGHT,
            color=GREEN,
            stroke_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        xcorr_legend = MathTex(r"R_{xy}(\tau)").next_to(
            xcorr_line_legend, RIGHT, SMALL_BUFF
        )
        sinc_line_legend = Line(
            ORIGIN,
            RIGHT,
            color=YELLOW,
            stroke_opacity=0.5,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        sinc_legend = MathTex(r"\frac{\sin{(x)}}{x}").next_to(
            sinc_line_legend, RIGHT, SMALL_BUFF
        )
        legend_group = (
            Group(
                Group(xcorr_line_legend, xcorr_legend),
                Group(sinc_line_legend, sinc_legend),
            )
            .arrange(DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)
            .next_to(self.camera.frame.get_corner(UR), DL)
        )

        self.play(
            LaggedStart(
                sinc_opacity @ 0.5,
                sinc_amp @ 2,
                sinc_f @ (pulse_f + 10),
                lag_ratio=0.2,
            ),
            FadeIn(legend_group),
            run_time=3,
        )

        self.wait(0.5)

        vid1 = ImageMobject(
            "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid1_box = SurroundingRectangle(vid1, buff=0, color=ORANGE)
        vid1_group = (
            Group(vid1, vid1_box)
            .next_to(sinc_legend, RIGHT, LARGE_BUFF * 4)
            .shift(UP * 2 + UP + LEFT / 2)
        )
        vid2 = ImageMobject(
            "../08_beamforming/static/Tapering Walkthrough YouTube Thumbnail.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid2_box = SurroundingRectangle(vid2, buff=0, color=ORANGE)
        vid2_group = (
            Group(vid2, vid2_box)
            .next_to(sinc_legend, RIGHT, LARGE_BUFF * 4)
            .shift(UP * 2 + DOWN + RIGHT / 2)
        )

        bez_u = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid1_group.get_left() + [-2, 0, 0],
            vid1_group.get_left() + [-0.1, 0, 0],
            color=RED,
        )
        bez_d = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid2_group.get_left() + [-2, 0, 0],
            vid2_group.get_left() + [-0.1, 0, 0],
            color=RED,
        )
        antenna_pattern_label = (
            Text("Antenna\nPatterns", font=FONT, color=RED)
            .scale(0.6)
            .next_to(bez_u, UP)
        )

        vid3 = ImageMobject(
            "../01_fmcw/media/images/fmcw/thumbnails/comparison.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid3_box = SurroundingRectangle(vid3, buff=0, color=ORANGE)
        vid3_group = (
            Group(vid3, vid3_box)
            .next_to(sinc_legend, DOWN, SMALL_BUFF)
            .shift(UP + LEFT / 2 + DOWN * 3)
        )
        vid4 = ImageMobject(
            "../04_fmcw_doppler/media/images/fmcw_doppler/thumbnails/Thumbnail_2.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid4_box = SurroundingRectangle(vid4, buff=0, color=ORANGE)
        vid4_group = (
            Group(vid4, vid4_box)
            .next_to(sinc_legend, DOWN, SMALL_BUFF)
            .shift(RIGHT * 2 + DOWN * 3)
        )
        vid5 = ImageMobject(
            "../03_cfar/media/images/cfar/thumbnails/Thumbnail_1.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid5_box = SurroundingRectangle(vid5, buff=0, color=ORANGE)
        vid5_group = (
            Group(vid5, vid5_box)
            .next_to(sinc_legend, DOWN, SMALL_BUFF)
            .shift(UP + LEFT / 2 + DOWN * 3 + RIGHT * 6)
        )
        vid6 = ImageMobject(
            "../11_aliasing/static/Aliasing Thumbnail.png"
        ).scale_to_fit_width(fw(self, 0.3))
        vid6_box = SurroundingRectangle(vid6, buff=0, color=ORANGE)
        vid6_group = (
            Group(vid6, vid6_box)
            .next_to(sinc_legend, DOWN, SMALL_BUFF)
            .shift(RIGHT * 2 + DOWN * 3 + RIGHT * 6)
        )

        bez_d1 = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid3_group.get_top() + [0, 1, 0],
            vid3_group.get_top() + [0, 0.1, 0],
            color=BLUE,
        )
        bez_d2 = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid4_group.get_top() + [0, 1, 0],
            vid4_group.get_top() + [0, 0.1, 0],
            color=BLUE,
        )
        bez_d3 = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid5_group.get_top() + [0, 1, 0],
            vid5_group.get_top() + [0, 0.1, 0],
            color=BLUE,
        )
        bez_d4 = CubicBezier(
            sinc_legend.get_right() + [0.1, 0, 0],
            sinc_legend.get_right() + [2, 0, 0],
            vid6_group.get_top() + [0, 3, 0],
            vid6_group.get_top() + [0, 0.1, 0],
            color=BLUE,
        )
        fft_label = (
            Text("Fourier\nTransform", font=FONT, color=BLUE)
            .scale(0.6)
            .next_to(bez_d3.get_midpoint(), LEFT)
            .shift(DOWN + RIGHT)
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.5).shift(
                    RIGHT
                    * (
                        (sinc_line_legend.get_left()[0] - MED_SMALL_BUFF)
                        - self.camera.frame.get_left()
                    )
                    + UP
                ),
                Create(bez_u),
                Create(bez_d),
                Write(antenna_pattern_label),
                GrowFromCenter(vid1_group),
                GrowFromCenter(vid2_group),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        d1 = (
            Dot().scale(2).next_to(Group(vid5_group, vid6_group), RIGHT, MED_LARGE_BUFF)
        )
        d2 = Dot().scale(2).next_to(d1, RIGHT, MED_SMALL_BUFF)
        d3 = Dot().scale(2).next_to(d2, RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                Write(fft_label),
                Create(bez_d1),
                GrowFromCenter(vid3_group),
                Create(bez_d2),
                GrowFromCenter(vid4_group),
                Create(bez_d3),
                GrowFromCenter(vid5_group),
                Create(bez_d4),
                GrowFromCenter(vid6_group),
                GrowFromCenter(d1),
                GrowFromCenter(d2),
                GrowFromCenter(d3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(bez_u),
                    Uncreate(bez_d),
                    Unwrite(antenna_pattern_label),
                    ShrinkToCenter(vid1_group),
                    ShrinkToCenter(vid2_group),
                    Unwrite(fft_label),
                    Uncreate(bez_d1),
                    ShrinkToCenter(vid3_group),
                    Uncreate(bez_d2),
                    ShrinkToCenter(vid4_group),
                    Uncreate(bez_d3),
                    ShrinkToCenter(vid5_group),
                    Uncreate(bez_d4),
                    ShrinkToCenter(vid6_group),
                    ShrinkToCenter(d1),
                    ShrinkToCenter(d2),
                    ShrinkToCenter(d3),
                ),
                self.camera.frame.animate.restore(),
                AnimationGroup(
                    sinc_opacity @ 0,
                    FadeOut(sinc_legend, sinc_line_legend),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        main_lobe_box = Polygon(
            xcorr_ax.c2p(-0.02, 0),
            xcorr_ax.c2p(-0.02, 2.1),
            xcorr_ax.c2p(0.02, 2.1),
            xcorr_ax.c2p(0.02, 0),
        )

        self.play(
            self.camera.frame.animate.scale(0.8),
            Group(xcorr_line_legend, xcorr_legend).animate.next_to(
                self.camera.frame.copy().scale(0.8).get_corner(UR), DL
            ),
            Create(main_lobe_box),
        )

        self.wait(0.5)

        width_r = Line(
            main_lobe_box.get_corner(UR) + [0, 0.1, 0],
            main_lobe_box.get_corner(UR) + [0, 0.3, 0],
        )
        width_l = Line(
            main_lobe_box.get_corner(UL) + [0, 0.1, 0],
            main_lobe_box.get_corner(UL) + [0, 0.3, 0],
        )
        width_line = Line(width_l.get_midpoint(), width_r.get_midpoint())

        sim_rres = MathTex(r"\sim \Delta R")
        sim_rres.shift(width_r.get_midpoint() - sim_rres[0][0].get_left()).shift(
            RIGHT / 4
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP),
                Create(width_l),
                Create(width_line),
                Create(width_r),
                *[GrowFromCenter(m) for m in sim_rres[0]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        side_lobe_box_l = Polygon(
            xcorr_ax.c2p(-~pw_plot, -1),
            xcorr_ax.c2p(-~pw_plot, 0.25),
            xcorr_ax.c2p(-0.005, 0.25),
            xcorr_ax.c2p(-0.005, -1),
        )
        side_lobe_box_r = Polygon(
            xcorr_ax.c2p(0.005, -1),
            xcorr_ax.c2p(0.005, 0.25),
            xcorr_ax.c2p(~pw_plot, 0.25),
            xcorr_ax.c2p(~pw_plot, -1),
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(*sim_rres[0]),
                    Uncreate(width_r),
                    Uncreate(width_line),
                    Uncreate(width_l),
                    lag_ratio=0.1,
                ),
                self.camera.frame.animate.shift(DOWN),
                AnimationGroup(
                    TransformFromCopy(main_lobe_box, side_lobe_box_l),
                    ReplacementTransform(main_lobe_box, side_lobe_box_r),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        xcorr_legend_abs = (
            MathTex(r"\lvert R_{xy}(\tau) \rvert")
            .move_to(xcorr_legend, RIGHT)
            .shift(RIGHT + UP / 2)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(side_lobe_box_l, side_lobe_box_r),
                AnimationGroup(
                    self.camera.frame.animate.scale(1 / 0.8),
                    is_abs @ 1,
                    xcorr_line_legend.animate.next_to(xcorr_legend_abs, LEFT),
                    LaggedStart(
                        ReplacementTransform(
                            xcorr_legend[0], xcorr_legend_abs[0][1:-1]
                        ),
                        GrowFromCenter(xcorr_legend_abs[0][0]),
                        GrowFromCenter(xcorr_legend_abs[0][-1]),
                        lag_ratio=0.3,
                    ),
                    run_time=3,
                ),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(fw(self, 0.2))
            .rotate(PI * 0.75)
            .set_fill(WHITE)
            .set_color(WHITE)
            .move_to(legend_group)
            .shift(RIGHT * 4)
        )
        parachute = (
            ImageMobject("../props/static/Parachute Army Toy.png")
            .scale_to_fit_width(plane.width)
            .next_to(plane, DOWN, LARGE_BUFF)
            .shift(RIGHT)
        )

        all_group = Group(xcorr_plot, plane, parachute, legend_group)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.1
                ).move_to(all_group),
                plane.shift(RIGHT * 8).animate.shift(LEFT * 8),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        plane_bez = CubicBezier(
            plane.get_left() + [-0.1, 0, 0],
            plane.get_left() + [-3, -1, 0],
            xcorr_ax.c2p(0, 1.9) + [1, 0.1, 0],
            xcorr_ax.c2p(0, 1.9) + [0.1, 0.1, 0],
        )
        self.play(Create(plane_bez))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        parachute_bez = CubicBezier(
            parachute.get_left() + [-0.1, 0, 0],
            parachute.get_left() + [-3, -1, 0],
            xcorr_ax.c2p(~pw_plot * 0.05, 0.05) + [1, 0.1, 0],
            xcorr_ax.c2p(~pw_plot * 0.05, -0.2) + [0.1, 0, 0],
        )
        self.play(
            LaggedStart(
                FadeIn(parachute, shift=DOWN + RIGHT),
                rtn_amp_2 @ 0.3,
                Create(parachute_bez),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        sll_line_u = Line(
            xcorr_ax.c2p(0, 2), xcorr_ax.c2p(0, 2) + [-0.5, 0, 0], buff=SMALL_BUFF
        )
        sll_line_d = Line(
            xcorr_ax.c2p(0, 0.35), xcorr_ax.c2p(0, 0.35) + [-0.5, 0, 0], buff=SMALL_BUFF
        )
        sll_line = Line(sll_line_d.get_midpoint(), sll_line_u.get_midpoint())

        sll_label = MathTex(r"\approx -13 \text{ dB}").next_to(sll_line, LEFT)

        self.play(
            LaggedStart(
                AnimationGroup(
                    self.camera.frame.animate.scale_to_fit_width(xcorr_plot.width * 1.3)
                    .move_to(xcorr_plot)
                    .shift(UP / 2),
                    Group(xcorr_legend_abs, xcorr_line_legend).animate.shift(LEFT),
                    plane_bez.animate.set_stroke(opacity=0.2),
                    parachute_bez.animate.set_stroke(opacity=0.2),
                ),
                Create(sll_line_d),
                Create(sll_line),
                Create(sll_line_u),
                FadeIn(sll_label),
                lag_ratio=0.3,
            )
        )

        # self.play(rtn_amp_2 @ 0.5)
        # self.play(pulse_f1 @ (~pulse_f1 * 0.6), run_time=5)

        self.wait(0.5)

        sll_level = DashedLine(
            xcorr_ax.c2p(0, 0.35) + [-0.5, 0, 0],
            xcorr_ax.c2p(0, 0.35) + [2, 0, 0],
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
            dashed_ratio=0.6,
        )
        parachute_level = DashedLine(
            xcorr_ax.c2p(0, -0.22) + [-0.5, 0, 0],
            xcorr_ax.c2p(0, -0.22) + [2, 0, 0],
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
            dashed_ratio=0.6,
        )
        plane_sll_label = (
            Text("Plane\nSide Lobe", font=FONT)
            .scale(0.5)
            .next_to(sll_level, RIGHT, MED_SMALL_BUFF)
        )
        parachute_level_label = (
            Text("Person\nMain Lobe", font=FONT)
            .scale(0.5)
            .next_to(parachute_level, RIGHT, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                Create(sll_level),
                Write(plane_sll_label),
                Create(parachute_level),
                Write(parachute_level_label),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        sll_down = Arrow(
            sll_line.get_bottom(),
            sll_line.get_bottom() + [0, -1.5, 0],
            buff=SMALL_BUFF,
            color=RED,
        )

        self.play(
            LaggedStart(
                FadeOut(
                    parachute_level, sll_level, plane_sll_label, parachute_level_label
                ),
                self.camera.frame.animate.scale(0.7).move_to(sll_line.get_bottom()),
                GrowArrow(sll_down),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        bc = (
            Text("Barker Codes", font=FONT, color=GREEN)
            .scale_to_fit_width(fw(self, 0.6))
            .move_to(self.camera.frame.copy().shift(DOWN * fh(self, 2)))
        )
        soln = (
            Text("Possible solution:", font=FONT)
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(bc, UP, MED_LARGE_BUFF)
        )
        self.add(bc, soln)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 2)))

        self.wait(2)


class BarkerCodes(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.8),
            y_length=fh(self, 0.5),
        )
        barker_len = 13
        barker_codes = [VT(0) for _ in range(barker_len)]

        ts = np.linspace(0, 1, 1000)
        f = 13 / 2

        def get_tx():
            x = np.sum(
                [
                    [
                        np.sin(2 * PI * f * t + np.deg2rad(~phi))
                        if (idx / barker_len) < t < ((idx + 1) / barker_len)
                        else 0
                        for t in ts
                    ]
                    for idx, phi in enumerate(barker_codes)
                ],
                axis=0,
            )
            func = interp1d(ts, x)
            tx = ax.plot(
                func,
                x_range=[0, 1, 1 / 5000],
                color=TX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
            return tx

        tx = always_redraw(get_tx)

        tx_label = Text("Transmit Waveform", font=FONT).next_to(ax, UP, MED_LARGE_BUFF)
        # Group(tx_label, waveform_label, ax).move_to(self.camera.frame)

        self.play(
            LaggedStart(
                Create(ax),
                Create(tx),
                Write(tx_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        label1 = always_redraw(
            lambda: Text(f"{int(~barker_codes[0])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(0.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label2 = always_redraw(
            lambda: Text(f"{int(~barker_codes[1])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(1.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label3 = always_redraw(
            lambda: Text(f"{int(~barker_codes[2])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(2.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label4 = always_redraw(
            lambda: Text(f"{int(~barker_codes[3])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(3.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label5 = always_redraw(
            lambda: Text(f"{int(~barker_codes[4])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(4.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label6 = always_redraw(
            lambda: Text(f"{int(~barker_codes[5])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(5.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label7 = always_redraw(
            lambda: Text(f"{int(~barker_codes[6])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(6.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label8 = always_redraw(
            lambda: Text(f"{int(~barker_codes[7])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(7.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label9 = always_redraw(
            lambda: Text(f"{int(~barker_codes[8])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(8.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label10 = always_redraw(
            lambda: Text(f"{int(~barker_codes[9])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(9.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label11 = always_redraw(
            lambda: Text(f"{int(~barker_codes[10])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(10.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label12 = always_redraw(
            lambda: Text(f"{int(~barker_codes[11])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(11.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )
        label13 = always_redraw(
            lambda: Text(f"{int(~barker_codes[12])}", font=FONT)
            .scale(0.6)
            .next_to(ax.c2p(12.5 / barker_len, -1), DOWN, MED_LARGE_BUFF)
        )

        code_labels = [
            label1,
            label2,
            label3,
            label4,
            label5,
            label6,
            label7,
            label8,
            label9,
            label10,
            label11,
            label12,
            label13,
        ]

        lines = [
            DashedLine(
                ax.c2p(idx / barker_len, 1),
                ax.c2p(idx / barker_len, -1.6),
                color=YELLOW,
                dash_length=DEFAULT_DASH_LENGTH * 2,
                dashed_ratio=0.6,
            )
            for idx in range(barker_len + 1)
        ]

        self.play(
            *[GrowFromCenter(m) for m in code_labels],
            LaggedStart(*[Create(m) for m in lines], lag_ratio=0.3),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        barker_codes_actual = [0, 0, 0, 0, 0, 180, 180, 0, 0, 180, 0, 180, 0]

        self.play(
            LaggedStart(
                *[bc @ barker_codes_actual[idx] for idx, bc in enumerate(barker_codes)]
            ),
            run_time=5,
        )

        self.wait(0.5)

        # self.play(*[Uncreate(m) for m in lines])

        self.wait(0.5)

        rxx_min = -60
        rxx_ax = (
            Axes(
                x_range=[-1, 1, 0.5],
                y_range=[0, -rxx_min, 10],
                tips=False,
                x_length=fw(self, 0.8),
                y_length=fh(self, 0.5),
            )
            .next_to(ax, RIGHT, LARGE_BUFF * 8)
            .shift(DOWN * 6)
        )

        rxx_x1 = VT(-1)

        def get_rxx():
            x = np.sum(
                [
                    [
                        np.sin(2 * PI * f * t + np.deg2rad(~phi))
                        if (idx / barker_len) < t < ((idx + 1) / barker_len)
                        else 0
                        for t in ts
                    ]
                    for idx, phi in enumerate(barker_codes)
                ],
                axis=0,
            )

            Rxx_sig = np.abs(np.correlate(x, x, mode="full"))
            Rxx_sig /= Rxx_sig.max()
            Rxx_sig = np.maximum(Rxx_sig, 1e-12)
            Rxx_sig_db = 20 * np.log10(Rxx_sig)

            ts_rxx = np.linspace(-ts.max(), ts.max(), 2 * ts.size - 1)
            func = interp1d(ts_rxx, np.clip(Rxx_sig_db - rxx_min, 0, None))

            rxx = rxx_ax.plot(
                func,
                x_range=[-1, ~rxx_x1, 1 / 5000],
                color=GREEN,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                use_smoothing=False,
            )
            return rxx

        rxx = always_redraw(get_rxx)
        self.add(rxx_ax, rxx)

        rxx_rect = SurroundingRectangle(
            rxx_ax,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            stroke_opacity=0,
            buff=SMALL_BUFF,
        ).set_z_index(-1)
        ax_rect = SurroundingRectangle(
            ax,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            stroke_opacity=0,
            buff=SMALL_BUFF,
        ).set_z_index(-1)
        self.add(rxx_rect, ax_rect)

        rxx_bez = CubicBezier(
            ax.get_center(),
            ax.get_right() + [5, 2, 0],
            rxx_ax.get_left() + [-5, -2, 0],
            rxx_ax.get_center(),
        ).set_z_index(-2)
        rxx_eqn = Tex(r"| $R_{xx}(\tau)$").next_to(rxx_ax, UP, MED_LARGE_BUFF)
        rxx_text = Text("Barker Code Autocorrelation", font=FONT).scale_to_fit_height(
            rxx_eqn.height
        )
        Group(rxx_text, rxx_eqn).arrange(RIGHT, MED_SMALL_BUFF).scale_to_fit_width(
            rxx_ax.width * 0.9
        ).next_to(rxx_ax, UP, MED_LARGE_BUFF)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                Create(rxx_bez),
                MoveAlongPath(self.camera.frame, rxx_bez),
                Write(rxx_text),
                FadeIn(rxx_eqn),
                rxx_x1 @ 1,
                lag_ratio=0.3,
            ),
            run_time=6,
        )

        self.wait(0.5)

        sll = DashedLine(
            rxx_ax.c2p(-1, -22.3 - rxx_min),
            rxx_ax.c2p(1, -22.3 - rxx_min),
            dashed_ratio=0.6,
            dash_length=DEFAULT_DASH_LENGTH * 4,
            color=YELLOW,
        )
        sll_label = (
            Text("-22.3 dB Side Lobes", font=FONT, color=YELLOW)
            .scale_to_fit_width(rxx_ax.width * 0.3)
            .next_to(sll.get_right(), UL)
        )

        self.play(LaggedStart(Create(sll), FadeIn(sll_label), lag_ratio=0.4))

        self.wait(0.5)

        self.play(MoveAlongPath(self.camera.frame, rxx_bez.reverse_direction()))

        self.wait(0.5)

        barker_codes_poly = [0, 90, 0, 0, 0, 0, 90, 0, 180, 0, 0, 180, 90]

        self.play(
            LaggedStart(
                *[bc @ barker_codes_poly[idx] for idx, bc in enumerate(barker_codes)]
            ),
            run_time=5,
        )

        self.wait(0.5)

        self.remove(sll, sll_label)

        poly_text = Text(
            "Polyphase Code Autocorrelation", font=FONT
        ).scale_to_fit_height(rxx_eqn.height)
        Group(poly_text, rxx_eqn).arrange(RIGHT, MED_SMALL_BUFF).scale_to_fit_width(
            rxx_ax.width * 0.9
        ).next_to(rxx_ax, UP, MED_LARGE_BUFF)
        self.add(poly_text)
        self.remove(rxx_text)

        self.play(MoveAlongPath(self.camera.frame, rxx_bez.reverse_direction()))

        self.wait(0.5)

        barker_codes_poly_2 = [90, 0, 180, 0, 90, 0, 180, 0, 0, 0, 0, 0, 180]

        self.play(
            LaggedStart(
                *[bc @ barker_codes_poly_2[idx] for idx, bc in enumerate(barker_codes)]
            ),
            run_time=5,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)


class WrapUp(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        wrapup_label = Text("Wrap up", font=FONT).to_corner(UL, MED_LARGE_BUFF)

        self.play(wrapup_label.shift(LEFT * 5).animate.shift(RIGHT * 5))

        self.wait(0.5)

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.7),
                y_length=fh(self, 0.6),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )

        pw_plot = VT(0.5)
        pulse_amp = VT(0.5)
        pulse_f = 20
        pulse_start = VT(0)
        pulse_end = VT(1)

        offset = 0.3

        pulse_rtn_ax = pulse_ax.copy().next_to(pulse_ax, DOWN, MED_LARGE_BUFF)
        xcorr_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                # y_range=[-1, 1, 0.5],
                # x_range=[offset - ~pw_plot, ~pw_plot, 0.5],
                y_range=[-1, 2, 0.5],
                tips=False,
                x_length=fw(self, 0.7),
                y_length=fh(self, 0.6),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        xcorr_ax.shift((pulse_rtn_ax.c2p(offset, 0) - xcorr_ax.c2p(0, 0)) + DOWN * 6)
        self.add(xcorr_ax)

        full_xcorr_width = ~pw_plot - xcorr_ax.p2c(pulse_ax.c2p(offset - ~pw_plot))[0]
        xcorr_x_offset = VT(full_xcorr_width)

        pulse_f1 = VT(pulse_f)
        pulse_phase_1 = VT(0)
        pulse_start_offset = VT(0)
        pulse_ltail = VT(~pw_plot)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~pulse_start,
                    (~pulse_end + ~pw_plot - ~pulse_ltail),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                stroke_opacity=1,
            )
        )
        # self.add(pulse)

        self.play(Create(pulse))

        self.wait(0.5)

        bw_label = MathTex(r"B > 0 \text{ Hz}").to_edge(UP, LARGE_BUFF)

        self.play(
            LaggedStart(*[FadeIn(m) for m in bw_label[0]], lag_ratio=0.2),
            pulse_f1 @ (~pulse_f1 * 5),
            run_time=3,
        )

        self.wait(0.5)

        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[~pulse_start, ~pulse_end, 1 / 1000],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                stroke_opacity=1,
            )
        )

        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(fw(self, 0.3))
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .next_to(pulse_ax.c2p(1, 0), RIGHT, LARGE_BUFF * 3)
            .set_y(Group(pulse, pulse_rtn).get_y())
        )
        self.add(plane)

        all_group = Group(pulse, plane)

        tx_arrow = Arrow(pulse_ax.c2p(1, 0), plane.get_left(), color=TX_COLOR)
        rx_arrow = Arrow(plane.get_left(), pulse_rtn_ax.c2p(1, 0), color=RX_COLOR)

        # self.add(pulse_rtn, tx_arrow, rx_arrow)

        self.play(
            LaggedStart(
                FadeOut(bw_label),
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.1
                ).move_to(all_group),
                wrapup_label.animate.next_to(
                    self.camera.frame.copy()
                    .scale_to_fit_width(all_group.width * 1.1)
                    .move_to(all_group)
                    .get_corner(UL),
                    DR,
                    MED_LARGE_BUFF,
                ),
                GrowArrow(tx_arrow),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(rx_arrow),
                Create(pulse_rtn),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        xcorr_amp = VT(1)

        fs = 2**10

        rtn_amp_1 = VT(1)
        rtn_amp_2 = VT(0)

        def get_xcorr():
            t_discreet = np.arange(0, 1, 1 / fs)
            tau = 0.3
            pulse = np.array(
                [
                    chirp_pulse(
                        t_val=t_val,
                        pulse_start=0,
                        pulse_width=tau,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=1,
                        phase=0,
                        ramp="linear",
                    )
                    for t_val in t_discreet
                ]
            )
            pulse_starts = [tau, tau * 1.05]
            pulse_amps = [~rtn_amp_1, ~rtn_amp_2]
            rtn = np.array(
                np.sum(
                    [
                        [
                            pa
                            * chirp_pulse(
                                t_val=t_val,
                                pulse_start=ps,
                                pulse_width=tau,
                                f0=pulse_f,
                                f1=~pulse_f1,
                                amp=1,
                                phase=0,
                                ramp="linear",
                            )
                            for t_val in t_discreet
                        ]
                        for ps, pa in zip(pulse_starts, pulse_amps)
                    ],
                    axis=0,
                )
            )

            h_matched = np.conj(pulse[t_discreet < tau][::-1])
            rtn_matched = signal.fftconvolve(rtn, h_matched, mode="full")
            rtn_matched /= rtn_matched.max()
            rtn_matched *= 2 * ~xcorr_amp
            t_full = np.arange(rtn_matched.size) / fs - tau

            # rtn_matched_db = np.clip(10 * np.log10(np.abs(rtn_matched)), -50, None)
            # rtn_matched_db -= np.nanmin(rtn_matched_db)
            # rtn_matched_db /= np.nanmax(rtn_matched_db)
            # rtn_matched_db *= rtn_matched.max()
            # rtn_matched_db *= 4
            # rtn_matched_db -= 6

            func = interp1d(t_full - tau, rtn_matched, fill_value="extrapolate")

            xcorr_plot = always_redraw(
                lambda: xcorr_ax.plot(
                    func,
                    color=GREEN,
                    x_range=[
                        xcorr_ax.p2c(pulse_ax.c2p(offset - ~pw_plot))[0],
                        ~pw_plot - ~xcorr_x_offset,
                        1 / (fs * 1),
                    ],
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                )
            )
            return xcorr_plot

        xcorr_plot = always_redraw(get_xcorr)
        self.add(xcorr_plot)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        # self.play(xcorr_x_offset @ 0, run_time=2)

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(tx_arrow, rx_arrow),
                    plane.animate.shift(RIGHT * 10),
                ),
                AnimationGroup(
                    self.camera.frame.animate.scale(1.4)
                    .set_x(pulse.get_x())
                    .shift(DOWN * 4),
                    wrapup_label.animate.next_to(
                        self.camera.frame.copy()
                        .scale(1.4)
                        .set_x(pulse.get_x())
                        .shift(DOWN * 4)
                        .get_corner(UL),
                        DR,
                        MED_LARGE_BUFF,
                    ),
                ),
                AnimationGroup(
                    pulse_start @ (offset - ~pw_plot),
                    pulse_start_offset @ (offset - ~pw_plot),
                    pulse_ltail @ 0,
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            xcorr_x_offset @ 0,
            pulse_start_offset @ (~pulse_end),
            run_time=10,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                xcorr_plot.width * 1.2
            ).move_to(xcorr_plot),
        )

        self.wait(0.5)

        nb_label = (
            Text("pulse_compression.ipynb", font=FONT)
            .scale_to_fit_width(fw(self, 0.5))
            .next_to(self.camera.frame.get_top(), DOWN)
        ).shift(DOWN * fh(self))

        nb_img_2 = ImageMobject("./static/nb_img_2.png").scale_to_fit_height(
            fh(self, 0.6)
        )
        nb_img_3 = ImageMobject("./static/nb_img_3.png").scale_to_fit_height(
            fh(self, 0.6)
        )
        Group(nb_img_2, nb_img_3).arrange(RIGHT, MED_LARGE_BUFF).move_to(
            self.camera.frame
        ).shift(DOWN * fh(self))

        self.add(nb_label)

        bez_0 = CubicBezier(
            nb_img_2.get_corner(DL) + [0, -0.1, 0],
            nb_img_2.get_corner(DL) + [-1, -2, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 3) + [-4, 2, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 3) + [-2, 0, 0],
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self)),
                LaggedStart(
                    GrowFromCenter(nb_img_2), GrowFromCenter(nb_img_3), lag_ratio=0.2
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        theory_paper = ImageMobject(
            "../props/static/Theory and Design of Chirp Radars.png"
        ).scale_to_fit_height(fh(self, 0.7))
        new_chirp = ImageMobject(
            "../props/static/Chirp A New Radar Technique.jpg"
        ).scale_to_fit_width(theory_paper.width * 1.3)
        fundamentals_of_radar_dsp = ImageMobject(
            "../props/static/Fundamentals of Radar DSP Book Cover.jpg"
        ).scale_to_fit_height(fh(self, 0.7))
        resources = (
            Group(fundamentals_of_radar_dsp, new_chirp, theory_paper)
            .arrange(RIGHT, MED_LARGE_BUFF)
            .scale_to_fit_width(fw(self, 0.9))
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
        )

        bez_1 = CubicBezier(
            fundamentals_of_radar_dsp.get_bottom() + [0, -0.1, 0],
            fundamentals_of_radar_dsp.get_bottom() + [0, -2, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2) + [0, 3, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2),
        )
        bez_2 = CubicBezier(
            new_chirp.get_bottom() + [0, -0.1, 0],
            new_chirp.get_bottom() + [0, -2, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2) + [0, 3, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2),
        )
        bez_3 = CubicBezier(
            theory_paper.get_bottom() + [0, -0.1, 0],
            theory_paper.get_bottom() + [0, -2, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2) + [0, 3, 0],
            self.camera.frame.get_center() + DOWN * fh(self, 2),
        )

        self.play(
            LaggedStart(
                Create(bez_0),
                self.camera.frame.animate.shift(DOWN * fh(self)),
                LaggedStart(
                    *[GrowFromCenter(m) for m in resources],
                    lag_ratio=0.2,
                ),
                AnimationGroup(
                    Create(bez_1),
                    Create(bez_2),
                    Create(bez_3),
                ),
                lag_ratio=0.5,
            )
        )
        desc = Text("Description", font=FONT).next_to(bez_2, DOWN, SMALL_BUFF)
        self.add(desc)

        self.play(self.camera.frame.animate.move_to(desc))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        osc = ImageMobject("../08_beamforming/static/osc_mug.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        kraken = ImageMobject("../08_beamforming/static/kraken.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        weather = ImageMobject(
            "../08_beamforming/static/weather.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        eqn = ImageMobject("../08_beamforming/static/eqn_mug.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        merch = (
            Group(kraken, osc, eqn, weather)
            .arrange_in_grid(2, 2)
            .scale_to_fit_height(config.frame_height * 0.9)
            .set_y(0)
        )

        shoutout = Text("Huge thanks to:", font=FONT)
        people = [
            "ZacJW",
            "db-isJustARatio",
            "Jea99",
            "Leon",
            "dplynch",
            "Kris",
            "zachdc",
            "misspeled",
            "Florian",
            "Cminor102",
            "w1gx",
            "Ioan",
            "nakribc",
            "alikarb0724",
            "Mark",
        ]
        people_text = (
            Group(
                *[Text(p, font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6) for p in people]
            )
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(shoutout, DOWN)
        )
        people_group = (
            Group(shoutout, people_text)
            .scale_to_fit_height(fh(self, 0.9))
            .next_to(merch, RIGHT)
            .shift(RIGHT * 12)
        )
        people_group = (
            Group(shoutout, people_text)
            .scale_to_fit_height(fh(self, 0.9))
            .next_to(merch, RIGHT)
            .shift(RIGHT * 12)
        )
        website_group = (
            Group(
                merch,
                people_group,
            )
            .scale_to_fit_height(fh(self, 0.9))
            .arrange(RIGHT)
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self) + DOWN * 5)
        )
        website_background = SurroundingRectangle(
            website_group,
            color=BACKGROUND_COLOR,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        )
        self.add(website_background, website_group)
        desc_background = SurroundingRectangle(
            desc, fill_color=BACKGROUND_COLOR, fill_opacity=1, color=BACKGROUND_COLOR
        ).set_z_index(-1)
        self.add(desc_background)

        website_bez = CubicBezier(
            desc.get_center(),
            desc.get_right() + [3, 0, 0],
            website_background.get_left() + [-5, 0, 0],
            website_group.get_center(),
        ).set_z_index(-2)

        self.play(
            LaggedStart(
                Create(website_bez),
                MoveAlongPath(self.camera.frame, website_bez),
                lag_ratio=0.1,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(1000))

        self.wait(2)


class ThumbnailRenders(MovingCameraScene):
    def construct(self):
        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )

        pw_plot = VT(0.6)
        pulse_amp = VT(0.5)
        pulse_f = 20
        pulse_rtn_x0 = VT(0)
        pulse_rtn_x1 = VT(~pw_plot)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f * 5)
        pulse_amp_2 = VT(0.3)
        pulse_amp_3 = VT(0.3)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0.1)
        pulse_t_start_3 = VT(0.28)
        allow_1 = VT(1)
        allow_2 = VT(1)
        allow_3 = VT(1)
        pulse_start_offset = VT(-5)
        pulse_opacity = VT(1)
        pulse_rtn_opacity = VT(1)
        pulse_ltail = VT(0)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~pulse_start_offset - ~pulse_ltail,
                    ~pulse_start_offset + ~pw_plot,
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                stroke_opacity=1,
            )
        )
        self.add(pulse)

        self.camera.frame.scale_to_fit_width(pulse.width * 1.2).move_to(pulse)

        # xcorr_ax = pulse_ax.copy().shift(
        #     pulse_ax.c2p(~pulse_start_offset + ~pw_plot / 2, -1.5)
        #     - pulse_ax.c2p(0, 2.2)
        # )

        # xcorr_amp = VT(1)

        # def get_xcorr():
        #     fs = 1e3
        #     t_discreet = np.arange(0, 1, 1 / fs)
        #     tau = 0.3
        #     pulse = np.array(
        #         [
        #             chirp_pulse(
        #                 t_val=t_val,
        #                 pulse_start=0,
        #                 pulse_width=tau,
        #                 f0=pulse_f,
        #                 f1=~pulse_f1,
        #                 amp=1,
        #                 phase=0,
        #                 ramp="linear",
        #             )
        #             for t_val in t_discreet
        #         ]
        #     )
        #     pulse_starts = [tau]
        #     rtn = np.array(
        #         np.sum(
        #             [
        #                 [
        #                     chirp_pulse(
        #                         t_val=t_val,
        #                         pulse_start=ps,
        #                         pulse_width=tau,
        #                         f0=pulse_f,
        #                         f1=~pulse_f1,
        #                         amp=1,
        #                         phase=0,
        #                         ramp="linear",
        #                     )
        #                     for t_val in t_discreet
        #                 ]
        #                 for ps in pulse_starts
        #             ],
        #             axis=0,
        #         )
        #     )

        #     h_matched = np.conj(pulse[t_discreet < tau][::-1])
        #     rtn_matched = signal.fftconvolve(rtn, h_matched, mode="full")
        #     rtn_matched /= rtn_matched.max()
        #     rtn_matched *= 2 * ~xcorr_amp
        #     t_full = np.arange(rtn_matched.size) / fs - tau
        #     func = interp1d(t_full - tau, rtn_matched, fill_value="extrapolate")

        #     xcorr_plot = always_redraw(
        #         lambda: xcorr_ax.plot(
        #             func,
        #             color=GREEN,
        #             x_range=[
        #                 -~pw_plot,
        #                 ~pw_plot,
        #                 1 / 200,
        #             ],
        #         )
        #     )
        #     return xcorr_plot

        # xcorr_plot = always_redraw(get_xcorr)
        # self.add(xcorr_plot)

        # self.camera.frame.scale_to_fit_width(xcorr_plot.width * 1.5).move_to(xcorr_plot)


class ThumbnailRendering2(MovingCameraScene):
    def construct(self):
        f_scale = 1
        f1 = 1.5 * f_scale
        f2 = 2.7 * f_scale
        f_clutter = 3.7 * f_scale
        power_norm_1 = -18
        power_norm_2 = -100
        power_norm_clutter = -100
        A_1 = 10 ** (power_norm_1 / 10)
        A_2 = 10 ** (power_norm_2 / 10)
        A_clutter = 10 ** (power_norm_clutter / 10)

        stop_time = 8
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        noise_mu = 0
        noise_sigma_db = -10
        noise_sigma = 10 ** (noise_sigma_db / 10)

        np.random.seed(0)
        noise = np.random.normal(loc=noise_mu, scale=noise_sigma, size=t.size)

        x_n = (
            A_1 * np.sin(2 * PI * f1 * t)
            + A_2 * np.sin(2 * PI * f2 * t)
            + A_clutter * np.sin(2 * PI * f_clutter * t)
            + noise
        ) / (A_1 + A_2 + A_clutter + noise_sigma)

        blackman_window = signal.windows.blackman(N)
        x_n_windowed = x_n * blackman_window

        fft_len = N * 4

        X_k = fftshift(fft(x_n_windowed, fft_len))
        X_k /= N / 2
        X_k = np.abs(X_k)
        X_k = 10 * np.log10(X_k)

        freq = np.linspace(-fs / 2, fs / 2, fft_len)

        func = interp1d(freq, X_k, fill_value="extrapolate")

        ax = Axes(x_range=[0, 4 * f_scale, PI / 2], y_range=[-40, -10, 10], tips=False)
        plot = ax.plot(
            func,
            x_range=[0, 4 * f_scale, 1 / 1000],
            color=BLUE,
            use_smoothing=True,
            stroke_width=DEFAULT_STROKE_WIDTH * 5,
        )
        self.add(plot)


class EndScreen(MovingCameraScene):
    def construct(self):
        hours = pd.read_csv("../../../Downloads/Work Hours 2025 (1).csv").dropna(
            subset=["In", "Out", "Category Fill"]
        )
        hours = hours[
            (hours["Category Fill"] == "Videos")
            & (hours["Video Fill"] == "Pulse Compression")
        ]
        hours["In_dt"] = pd.to_datetime(hours["In"], errors="coerce")
        hours["Out_dt"] = pd.to_datetime(hours["Out"], errors="coerce")
        hours["In_mins"] = hours["In_dt"].dt.hour * 60 + hours["In_dt"].dt.minute
        hours["Out_mins"] = hours["Out_dt"].dt.hour * 60 + hours["Out_dt"].dt.minute

        stats_title = Text("Stats for Nerds", font=FONT).scale(0.7)
        stats_table = (
            Table(
                [
                    ["Lines of code", "5,572"],
                    ["Hours", f"{hours['Session Hours'].sum():.1f}"],
                    ["Days", "103"],
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

        thank_you_sabrina = Text(
            "Thank you, Sabrina, for\nediting the whole video :)",
            font=FONT,
            font_size=DEFAULT_FONT_SIZE * 0.4,
        ).to_corner(DL)

        def coverage_array(spans, allow_wrap=True):
            cov = np.zeros(24 * 60, dtype=int)
            for a, b in spans:
                if allow_wrap and b < a:
                    cov[a:] += 1
                    cov[:b] += 1
                else:
                    cov[a:b] += 1
            return cov

        spans = list(hours[["In_mins", "Out_mins"]].itertuples(index=None, name=None))
        cov = coverage_array(spans)
        filter_len = 61
        window = np.ones(filter_len) / filter_len

        t = np.linspace(0, 24, cov.size)

        ax = Axes(
            x_range=[0, 24, 3],
            y_range=[0, 6, 1],
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.3),
            tips=False,
        )
        hours_title = (
            Text("Hours Distribution", font=FONT)
            .scale(0.5)
            .next_to(ax, UP, MED_SMALL_BUFF)
        )

        times = np.arange(0, 27, 3)
        time_labels = Group(
            *[
                Text(f"{h % 12 or 12}{'am' if h < 12 else 'pm'}", font=FONT)
                .scale(0.5)
                .next_to(ax.c2p(h, 0), DOWN, MED_SMALL_BUFF)
                for h in times
            ]
        )
        hour_yaxis = np.arange(1, 7, 1)
        hour_labels = Group(
            *[
                Text(f"{h}", font=FONT)
                .scale(0.5)
                .next_to(ax.c2p(0, h), LEFT, MED_SMALL_BUFF)
                for h in hour_yaxis
            ]
        )

        Group(
            stats_title,
            stats_table,
            Group(ax, time_labels, hour_labels, hours_title),
            # thank_you_sabrina,
        ).arrange(DOWN, MED_LARGE_BUFF).to_edge(RIGHT, MED_LARGE_BUFF)

        scale = VT(0)

        def get_plot():
            plot = always_redraw(
                lambda: ax.plot(
                    interp1d(
                        t,
                        ~scale * np.convolve(cov, window, mode="same"),
                        fill_value="extrapolate",
                    ),
                    color=BLUE,
                    use_smoothing=False,
                )
            )
            area = ax.get_area(plot, color=BLUE, opacity=0.5)
            return plot, area

        plot = always_redraw(lambda: get_plot()[0])
        area = always_redraw(lambda: get_plot()[1])

        self.play(
            LaggedStart(
                AnimationGroup(FadeIn(stats_title, shift=DOWN), FadeIn(stats_table)),
                AnimationGroup(Create(ax), FadeIn(plot, area, hours_title)),
                AnimationGroup(
                    LaggedStart(*[FadeIn(m) for m in time_labels], lag_ratio=0.1),
                    LaggedStart(*[FadeIn(m) for m in hour_labels], lag_ratio=0.1),
                ),
                Write(thank_you_sabrina),
                lag_ratio=0.3,
            )
        )

        self.play(scale @ 1)

        self.wait(2)
