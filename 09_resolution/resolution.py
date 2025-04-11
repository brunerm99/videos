# resolution.py


from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
import sys
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift, fft2

import matplotlib

matplotlib.use("Agg")
from matplotlib.pyplot import get_cmap

sys.path.insert(0, "..")

from props import WeatherRadarTower, VideoMobject
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


def compute_phase_diff(v):
    time_from_vel = 2 * (v * Tc) / c
    return 2 * PI * f * time_from_vel


def compute_f_beat(R):
    return (2 * R * bw) / (c * Tc)


def db_to_lin(x):
    return 10 ** (x / 10)


# Radar setup for doppler stuff
f = 77e9  # Hz
Tc = 40e-6  # chirp time - s
bw = 1.6e9  # bandwidth - Hz
chirp_rate = bw / Tc  # Hz/s

wavelength = c / f
M = 40  # number of chirps in coherent processing interval (CPI)

# Target
R = 20  # m
v = 10  # m/s
f_beat = compute_f_beat(R)
phase_diff = compute_phase_diff(v)
max_time = 15 / f_beat
N = 10000
Ts = max_time / N
fs = 1 / Ts


class Intro(Scene):
    def construct(self): ...


class RangeResolution(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4)

        self.play(radar.get_animation())

        self.wait(0.5)

        self.play(radar.vgroup.animate.to_corner(DL, LARGE_BUFF))

        self.wait(0.5)

        TARGET1_COLOR = GREEN
        TARGET2_COLOR = ORANGE
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
                x_range=[max(0, ~xmax_t1 - pw), min(~xmax_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax_t2 - pw), min(~xmax_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
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

        self.next_section(skip_animations=skip_animations(True))
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
            pw_label.animate.shift(UP),
            LaggedStart(
                xmax @ (ax.p2c(target2.get_left())[0]),
                xmax_t1 @ (pw / 2),
                lag_ratio=0.4,
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(target2.shift(RIGHT * 8).animate.shift(LEFT * 8))

        self.wait(0.5)

        delta_r_line = always_redraw(
            lambda: Line(
                target1.get_center(),
                [target2.get_center()[0], target1.get_center()[1], 0],
            ).shift(UP * 1.5)
        )
        delta_r_line_l = always_redraw(
            lambda: Line(
                target1.get_center() + UP * 1.5 + DOWN / 8,
                target1.get_center() + UP * 1.5 + UP / 8,
            )
        )
        delta_r_line_r = always_redraw(
            lambda: Line(
                delta_r_line.get_right() + DOWN / 8,
                delta_r_line.get_right() + UP / 8,
            )
        )
        delta_r = always_redraw(lambda: MathTex(r"\Delta R").next_to(delta_r_line, UP))
        # self.add(delta_r_line_l, delta_r_line_r, delta_r_line, delta_r)

        self.play(
            LaggedStart(
                Create(delta_r_line_l),
                Create(delta_r_line),
                Create(delta_r_line_r),
                Write(delta_r),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            xmax @ 1.5,
            xmax_t2 @ (pw),
            xmax_t1 + pw,
            run_time=2,
        )

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )

        self.wait(0.5)

        self.play(
            xmax_t1 @ (0.5 + pw / 2),
            xmax_t2 @ (0.5 + pw / 2 - target_dist),
            run_time=3,
        )

        self.wait(0.5)

        line = Line(
            target2_ax.c2p(~xmax_t2, 0.8), target1_ax.c2p(~xmax_t1 - pw * 1.05, -0.6)
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
        # self.add(overlap)

        self.next_section(skip_animations=skip_animations(True))
        tau_gt = MathTex(r"\tau > t_{\Delta R}").next_to(overlap, UP)

        self.play(
            LaggedStart(
                FadeIn(overlap),
                TransformFromCopy(pw_label[0][0], tau_gt[0][0], path_arc=-PI / 3),
                *[GrowFromCenter(m) for m in tau_gt[0][1:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        f1 = VT(2.5)
        f2 = VT(2.7)
        power_norm_1 = VT(-30)
        power_norm_2 = VT(-30)
        stop_time = 16
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        noise_mu = 0
        noise_sigma_db = -10
        noise_sigma = 10 ** (noise_sigma_db / 10)

        np.random.seed(0)
        noise = np.random.normal(loc=noise_mu, scale=noise_sigma, size=t.size)

        return_xmax = 5
        return_ax = Axes(
            x_range=[0, return_xmax, return_xmax / 4],
            y_range=[0, 40, 20],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.3,
        ).next_to(self.camera.frame.get_top(), UP, SMALL_BUFF)
        self.add(return_ax)

        def get_return_plot():
            A_1 = 10 ** (~power_norm_1 / 10)
            A_2 = 10 ** (~power_norm_2 / 10)
            x_n = (
                A_1 * np.sin(2 * PI * ~f1 * t) + A_2 * np.sin(2 * PI * ~f2 * t) + noise
            )

            blackman_window = signal.windows.blackman(N)
            x_n_windowed = x_n * blackman_window

            fft_len = N * 4

            X_k = fftshift(fft(x_n_windowed, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = 10 * np.log10(X_k)
            X_k -= -43

            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            f_X_k = interp1d(freq, X_k, fill_value="extrapolate")

            plot = return_ax.plot(
                f_X_k, x_range=[0, return_xmax, return_xmax / 200], color=RX_COLOR
            )
            return plot

        return_plot = always_redraw(get_return_plot)
        self.add(return_plot)

        all_objs = Group(return_ax.copy().shift(DOWN), radar.vgroup, target1, target2)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                all_objs.height * 1.2
            ).move_to(all_objs),
            return_ax.animate.shift(DOWN),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(overlap, tau_gt),
                AnimationGroup(
                    xmax_t1 @ (1 + pw + target_dist),
                    xmax_t2 @ (1 + pw),
                ),
                power_norm_1 @ 0,
                power_norm_2 @ 0,
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        return_plot_new = get_return_plot()
        target1_label = (
            Text("Target 1?", color=TARGET1_COLOR, font="Maple Mono")
            .scale(0.6)
            .next_to(
                return_ax.input_to_graph_point(~f1, return_plot_new), DOWN, LARGE_BUFF
            )
            .shift(LEFT * 3)
        )
        target1_label_line = CubicBezier(
            target1_label.get_right() + [0.1, 0, 0],
            target1_label.get_right() + [1, 0, 0],
            return_ax.input_to_graph_point(~f1, return_plot_new) + [0, -1, 0],
            return_ax.input_to_graph_point(~f1, return_plot_new) + [0, -0.1, 0],
            color=TARGET1_COLOR,
        )

        target2_label = (
            Text("Target 2?", color=TARGET2_COLOR, font="Maple Mono")
            .scale(0.6)
            .next_to(
                return_ax.input_to_graph_point(~f1, return_plot_new), DOWN, LARGE_BUFF
            )
            .shift(RIGHT * 3)
        )
        target2_label_line = CubicBezier(
            target2_label.get_left() + [-0.1, 0, 0],
            target2_label.get_left() + [-1, 0, 0],
            return_ax.input_to_graph_point(~f2, return_plot_new) + [0, -1, 0],
            return_ax.input_to_graph_point(~f2, return_plot_new) + [0, -0.1, 0],
            color=TARGET2_COLOR,
        )
        # self.add(
        #     target1_label,
        #     target2_label,
        #     target1_label_line,
        #     target2_label_line,
        # )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(target1_label_line),
                    Create(target2_label_line),
                ),
                AnimationGroup(
                    Write(target1_label),
                    Write(target2_label),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target1_label_line.save_state()
        self.play(
            target1.animate.shift(LEFT * 2),
            f1 - 0.5,
            Transform(
                target1_label_line,
                CubicBezier(
                    target1_label.get_right() + [0.1, 0, 0],
                    target1_label.get_right() + [1, 0, 0],
                    return_ax.input_to_graph_point(~f1, return_plot_new)
                    + LEFT * 1.15
                    + [0, -1, 0],
                    return_ax.input_to_graph_point(~f1, return_plot_new)
                    + LEFT * 1.15
                    + [0, -0.1, 0],
                    color=TARGET1_COLOR,
                ),
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            target1.animate.shift(RIGHT * 2.2),
            f1 + 0.6,
            Transform(
                target1_label_line,
                CubicBezier(
                    target1_label.get_right() + [0.1, 0, 0],
                    target1_label.get_right() + [1, 0, 0],
                    return_ax.input_to_graph_point(~f1 + 0.5, return_plot_new)
                    + RIGHT * 0.3
                    + [0, -1, 0],
                    return_ax.input_to_graph_point(~f1 + 0.5, return_plot_new)
                    + RIGHT * 0.3
                    + [0, -0.1, 0],
                    color=TARGET1_COLOR,
                ),
            ),
            run_time=2,
        )

        self.wait(0.5)

        qmark = always_redraw(
            lambda: Text("?", font="Maple Mono")
            .scale(0.8)
            .next_to(delta_r, RIGHT, SMALL_BUFF)
        )

        self.camera.frame.save_state()
        target_group = Group(target1, target2, delta_r)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(target_group.height * 1.3)
                .move_to(target_group)
                .shift(DOWN / 3),
                pw_label.animate.set_opacity(0),
                # pw_label.animate.next_to(target_group, DOWN),
                Write(qmark),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(target1.animate.shift(LEFT), target2.animate.shift(RIGHT * 0.5))

        self.wait(0.5)

        self.play(
            target1.animate.shift(RIGHT * 0.5), target2.animate.shift(RIGHT * 1.5)
        )

        self.wait(0.5)

        xmax_t1 @= 0
        xmax_t2 @= 0
        self.remove(
            target1_label, target2_label, target1_label_line, target2_label_line
        )

        tau_gt = MathTex(r"\tau > t_{\Delta R}").next_to(pw_label, DOWN)

        self.play(
            Write(tau_gt),
            target1.animate.shift(RIGHT * 0.5),
            target2.animate.shift(LEFT * 2),
            self.camera.frame.animate.restore(),
            pw_label.animate.set_opacity(1),
            AnimationGroup(
                xmax_t1 @ (0.5 + pw / 2),
                xmax_t2 @ (0.5 + pw / 2 - target_dist),
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            pw_label[0][0]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW),
            tau_gt[0][0]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW),
        )

        self.wait(0.5)

        target_dist_large = abs(
            (
                ax.p2c(target1.get_left()[0])
                - ax.p2c((target2.get_left() + RIGHT * 1.5)[0])
            )[0]
        )

        self.play(
            target2.animate.shift(RIGHT * 1.5),
            tau_gt[0][1].animate.rotate(PI).set_color(YELLOW),
            f2 + 0.5,
            xmax_t2 - target_dist_large,
            run_time=3,
        )

        self.wait(0.5)

        script = Group(
            *[
                Text(s, font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.5)
                for s in [
                    "...",
                    "than the time it takes to get from target ",
                    "1 to target 2 and back.",
                    'By the way, don\'t forget about this "and back" ',
                    "in my script. I said it because it messed me up a lot.\n",
                    "Dividing delta R by the speed of light will give you ",
                    "the time it takes to travel from one target to the ",
                    "...",
                ]
            ]
        ).arrange(DOWN, aligned_edge=LEFT)
        script_box = SurroundingRectangle(
            script,
            color=GREEN,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
        )
        script_group = Group(script_box, script).set_z_index(3)

        self.play(
            script_group.shift(DOWN * config.frame_height * 1.5).animate.move_to(
                self.camera.frame
            )
        )

        self.wait(0.5)

        box1 = SurroundingRectangle(script[2][-8:-1]).set_z_index(10)
        box2 = SurroundingRectangle(script[3][-8:-1]).set_z_index(10)
        self.play(
            LaggedStart(
                Create(box1),
                Create(box2),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Group(script_group, box1, box2).animate.shift(
                DOWN * 1.5 * config.frame_height
            )
        )

        self.wait(0.5)

        delta_r_over_c = MathTex(
            r"\frac{\Delta R}{c} \left[ \frac{ m}{m/s} \right]",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).next_to(pw_label, UP, MED_LARGE_BUFF)

        self.play(
            return_ax.animate.shift(UP * config.frame_height * 0.8),
            LaggedStart(
                TransformFromCopy(delta_r[0], delta_r_over_c[0][:2], path_arc=-PI / 3),
                LaggedStart(
                    *[GrowFromCenter(m) for m in delta_r_over_c[0][2:]], lag_ratio=0.1
                ),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        x_top = MathTex(
            r"\times", font_size=DEFAULT_FONT_SIZE * 1.5, color=RED
        ).move_to(delta_r_over_c[0][5])
        x_bot = MathTex(
            r"\times", font_size=DEFAULT_FONT_SIZE * 1.5, color=RED
        ).move_to(delta_r_over_c[0][7])
        delta_r_over_c_cunit = MathTex(
            r"\frac{\Delta R}{c} \left[ s \right]",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(delta_r_over_c)

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(Write(x_top), Write(x_bot), lag_ratio=0.2),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ShrinkToCenter(delta_r_over_c[0][5]),
                    ShrinkToCenter(x_top),
                ),
                ShrinkToCenter(delta_r_over_c[0][6]),
                AnimationGroup(
                    ShrinkToCenter(delta_r_over_c[0][7]),
                    ShrinkToCenter(x_bot),
                ),
                ShrinkToCenter(delta_r_over_c[0][8]),
                ReplacementTransform(delta_r_over_c[0][9], delta_r_over_c_cunit[0][5]),
                ReplacementTransform(delta_r_over_c[0][10], delta_r_over_c_cunit[0][6]),
                ReplacementTransform(delta_r_over_c[0][4], delta_r_over_c_cunit[0][4]),
                ReplacementTransform(
                    delta_r_over_c[0][:4], delta_r_over_c_cunit[0][:4]
                ),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        target2_line_new = Line(target2.get_left(), radar.radome.get_right())
        target2_ax_new = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target2_line_new.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target2_line_new.get_angle())
            .set_opacity(0)
        )
        target2_ax_new.shift(target2.get_left() - target2_ax_new.c2p(0, 0))

        xmax_12 = VT(ax.p2c(target1.get_left())[0])
        xmax_21 = VT(0)
        tx_12 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[
                    max(ax.p2c(target1.get_left())[0], ~xmax_12 - pw),
                    min(~xmax_12, ax.p2c(target2.get_left())[0]),
                    1 / 200,
                ],
                color=TX_COLOR,
            )
        )
        self.add(tx_12)

        self.play(xmax_12 @ (ax.p2c(target2.get_left())[0]))

        self.wait(0.5)

        rx_21 = always_redraw(
            lambda: target2_ax_new.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[
                    max(0, ~xmax_21 - pw),
                    min(~xmax_21, target2_ax_new.p2c(target1.get_left())[0]),
                    1 / 200,
                ],
                color=TARGET2_COLOR,
            )
        )
        self.add(rx_21)

        self.play(
            LaggedStart(
                xmax_12 + pw,
                xmax_21 @ (target2_ax_new.p2c(target1.get_left())[0] + pw),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        roundtrip_time = MathTex(
            r"\frac{2 \Delta R}{c} \left[ s \right]",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(delta_r_over_c_cunit)

        self.play(
            LaggedStart(
                AnimationGroup(
                    xmax_t1 @ (1 + pw + target_dist_large),
                    xmax_t2 @ (1 + pw),
                ),
                ReplacementTransform(delta_r_over_c_cunit[0], roundtrip_time[0][1:]),
                GrowFromCenter(roundtrip_time[0][0]),
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # TODO: make the < Transform smoother
        inequality = MathTex(
            r"\tau < \frac{2 \Delta R}{c}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(roundtrip_time)
        inequality[0][1].set_color(YELLOW)
        inequality_rearr = MathTex(
            r"\Delta R > \frac{c \tau}{2}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(roundtrip_time)
        inequality_rearr[0][2].set_color(YELLOW)

        # self.play(tau_gt[0][1].animate.set_color(WHITE))
        self.play(
            LaggedStart(
                AnimationGroup(
                    ShrinkToCenter(roundtrip_time[0][-3:]),
                    ShrinkToCenter(tau_gt[0][2:]),
                ),
                ReplacementTransform(roundtrip_time[0][:-3], inequality[0][2:]),
                ReplacementTransform(tau_gt[0][1], inequality[0][1], path_arc=-PI),
                ReplacementTransform(tau_gt[0][0], inequality[0][0], path_arc=-PI),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(
                    inequality[0][3:5], inequality_rearr[0][:2], path_arc=PI
                ),
                ReplacementTransform(
                    inequality[0][0], inequality_rearr[0][4], path_arc=PI
                ),
                inequality[0][1].animate.rotate(PI).move_to(inequality_rearr[0][2]),
                ReplacementTransform(
                    inequality[0][2], inequality_rearr[0][6], path_arc=PI
                ),
                ReplacementTransform(
                    inequality[0][6], inequality_rearr[0][3], path_arc=PI
                ),
                ReplacementTransform(inequality[0][5], inequality_rearr[0][5]),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        rres_eq = MathTex(
            r"\Delta R = \frac{c \tau}{2}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(roundtrip_time)

        self.play(
            ReplacementTransform(inequality_rearr[0][:2], rres_eq[0][:2]),
            ReplacementTransform(inequality[0][1], rres_eq[0][2]),
            ReplacementTransform(inequality_rearr[0][3:], rres_eq[0][3:]),
        )

        self.wait(0.5)

        rres_box = SurroundingRectangle(
            rres_eq, color=GREEN, buff=MED_SMALL_BUFF, corner_radius=0.2
        )

        self.play(
            Create(rres_box),
            self.camera.frame.animate.scale_to_fit_width(rres_box.width * 2.5).move_to(
                rres_box
            ),
            pw_label.animate.set_x(rres_box.get_x()),
            FadeOut(
                target1,
                target2,
                delta_r_line,
                delta_r_line_l,
                delta_r_line_r,
                delta_r,
                qmark,
            ),
        )

        self.wait(0.5)

        self.play(
            inequality_rearr[0][4]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        self.play(
            pw_label.animate.set_color(YELLOW),
        )

        self.wait(0.5)

        rres_eq_val = MathTex(
            r"\Delta R = \frac{c \tau}{2} = 150 m",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(rres_eq, LEFT)
        rres_box_val = SurroundingRectangle(
            rres_eq_val, color=GREEN, buff=MED_SMALL_BUFF, corner_radius=0.2
        )

        rres_box.save_state()
        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                Transform(rres_box, rres_box_val),
                self.camera.frame.animate.set_x(rres_box_val.get_x()),
                Write(rres_eq_val[0][-5:]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            rres_box.animate.restore(),
            self.camera.frame.animate.restore(),
            Unwrite(rres_eq_val[0][-5:]),
            FadeOut(pw_label),
        )

        self.wait(0.5)

        f_bw = MathTex(r"f(B)", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            rres_eq, DOWN, LARGE_BUFF * 1.5
        )

        bl_bez = CubicBezier(
            rres_eq.get_corner(DL) + [-0.1, -0.1, 0],
            rres_eq.get_corner(DL) + [-0.1, -1, 0],
            f_bw.get_top() + [0, 1, 0],
            f_bw.get_top() + [0, 0.1, 0],
        )
        br_bez = CubicBezier(
            rres_eq.get_corner(DR) + [0.1, -0.1, 0],
            rres_eq.get_corner(DR) + [0.1, -1, 0],
            f_bw.get_top() + [0, 1, 0],
            f_bw.get_top() + [0, 0.1, 0],
        )

        f_bw_group = Group(f_bw, rres_eq)

        self.play(
            LaggedStart(
                Uncreate(rres_box),
                self.camera.frame.animate.scale_to_fit_height(
                    f_bw_group.height * 1.3
                ).move_to(f_bw_group),
                AnimationGroup(Create(br_bez), Create(bl_bez)),
                Write(f_bw),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        time_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.55,
        )
        f_ax = Axes(
            x_range=[0, 20, 5],
            y_range=[0, 30, 20],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.55,
        )
        ax_group = (
            Group(time_ax, f_ax)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(f_bw, DOWN, LARGE_BUFF * 2)
        )
        time_ax_label = time_ax.get_axis_labels(
            MathTex("t", font_size=DEFAULT_FONT_SIZE * 1.4), ""
        )
        f_ax_label = f_ax.get_axis_labels(
            MathTex("f", font_size=DEFAULT_FONT_SIZE * 1.4), ""
        )
        f_ax_label[0].next_to(f_ax.c2p(20, 0), RIGHT)
        self.add(ax_group, time_ax_label, f_ax_label)
        self.remove(radar.vgroup, tx_12, tx)

        fs = 400
        t_new = np.arange(0, 1, 1 / fs)
        # N_new = t_new.size
        fft_len = 2**14

        f = 10

        t_max = VT(0.1)

        def get_sig():
            sig = np.sin(2 * PI * f * t_new)
            sig[(t_new > ~t_max / 2 + 0.5) | (t_new < 0.5 - ~t_max / 2)] = 0
            return sig

        def get_time_plot():
            sig = get_sig()
            f_sig = interp1d(t_new, sig, fill_value="extrapolate")
            return time_ax.plot(
                f_sig,
                x_range=[0, 1, 1 / 400],
                color=BLUE,
                use_smoothing=False,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )

        def get_fft_plot():
            sig = get_sig()
            X_k = 10 * np.log10(np.abs(fft(sig, fft_len)) / (t_new.size / 2)) + 30
            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            f_X_k = interp1d(freq, np.clip(fftshift(X_k), 0, None))
            return f_ax.plot(
                f_X_k,
                x_range=[0, 20, 20 / 400],
                color=ORANGE,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )

        time_plot = always_redraw(get_time_plot)
        f_plot = always_redraw(get_fft_plot)
        self.add(time_plot, f_plot)

        time_label = Tex("Time", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            time_ax, UP, MED_LARGE_BUFF
        )
        bw_label = (
            Tex("Bandwidth", font_size=DEFAULT_FONT_SIZE * 1.5)
            .next_to(f_ax, UP, MED_LARGE_BUFF)
            .set_y(time_label.get_y())
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    Group(ax_group, f_bw_group).width * 1.12
                ).move_to(Group(ax_group, f_bw_group)),
                AnimationGroup(Uncreate(bl_bez), Uncreate(br_bez)),
                FadeOut(f_bw[0][:2], f_bw[0][-1]),
                ReplacementTransform(f_bw[0][2], bw_label[0][0], path_arc=PI / 3),
                AnimationGroup(Write(bw_label[0][1:]), Write(time_label)),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            bw_label.animate.set_color(ORANGE),
            time_label.animate.set_color(BLUE),
        )

        self.wait(0.5)

        self.play(t_max @ 1, run_time=4)

        self.wait(0.5)

        self.play(t_max @ 0.1, run_time=4)

        self.wait(0.5)

        time_bw_prod = MathTex(
            r"\tau \approx \frac{1}{B}", font_size=DEFAULT_FONT_SIZE * 1.5
        ).next_to(rres_eq, DOWN, LARGE_BUFF)

        self.play(TransformFromCopy(rres_eq[0][4], time_bw_prod[0][0], path_arc=PI / 3))
        # self.add(time_bw_prod[0][0])

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in time_bw_prod[0][1:]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                time_ax,
                f_ax,
                time_plot,
                f_plot,
                time_label,
                bw_label,
                time_ax_label,
                f_ax_label,
            ),
            self.camera.frame.animate.scale_to_fit_height(
                Group(rres_eq, time_bw_prod).height * 2
            ).move_to(Group(rres_eq, time_bw_prod)),
        )

        self.wait(0.5)

        rres_eq_bw = MathTex(
            r"\Delta R \approx \frac{c}{2 B}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(rres_eq)

        # self.remove(*rres_eq[0])
        self.remove(*inequality[0], *inequality_rearr[0])

        self.next_section(skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                ReplacementTransform(rres_eq[0][:2], rres_eq_bw[0][:2]),
                AnimationGroup(
                    ReplacementTransform(time_bw_prod[0][1], rres_eq_bw[0][2]),
                    FadeOut(rres_eq[0][2], shift=UP),
                ),
                ReplacementTransform(rres_eq[0][3], rres_eq_bw[0][3]),
                FadeOut(rres_eq[0][4], shift=UP),
                ReplacementTransform(rres_eq[0][5], rres_eq_bw[0][4]),
                FadeOut(time_bw_prod[0][0], time_bw_prod[0][2:4]),
                ReplacementTransform(
                    time_bw_prod[0][-1], rres_eq_bw[0][-1], path_arc=PI / 3
                ),
                # ReplacementTransform(rres_eq[0][-2:], rres_eq_bw[0][-3:-1]),
                ReplacementTransform(rres_eq[0][-1], rres_eq_bw[0][-2]),
                self.camera.frame.animate.move_to(rres_eq_bw),
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4).next_to(
            rres_eq_bw, DOWN, LARGE_BUFF * 2
        ).shift(LEFT * 5)

        self.add(radar.vgroup)

        beam_group = Group(radar.vgroup, rres_eq_bw)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(beam_group.height * 1.5)
            .move_to(beam_group)
            .set_x(rres_eq_bw.get_x())
        )

        self.wait(0.5)

        target1_new = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .set_color(TARGET1_COLOR)
            .next_to(radar.radome, RIGHT, LARGE_BUFF * 3)
            .shift(UP * 1.5)
        )
        target2_new = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
            .next_to(radar.radome, RIGHT, LARGE_BUFF * 1.5)
            # .shift(DOWN)
        )

        r_min = -60

        x_len = config.frame_height * 0.6
        target_line = Line(
            radar.radome.get_right(), Group(target1_new, target2_new).get_center()
        )
        polar_ax = (
            Axes(
                x_range=[r_min, -r_min, r_min / 8],
                y_range=[r_min, -r_min, r_min / 8],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=x_len,
                y_length=x_len,
            )
            .set_opacity(0)
            .rotate(target_line.get_angle())
        )
        polar_ax.shift(radar.radome.get_center() - polar_ax.c2p(0, 0))
        radome_circ = (
            radar.radome.copy()
            .set_fill(color=BACKGROUND_COLOR, opacity=1)
            .set_z_index(-1)
        )
        radar_box = (
            Rectangle(
                width=(
                    radar.left_leg.get_edge_center(RIGHT)
                    - radar.right_leg.get_edge_center(LEFT)
                )[0],
                height=radar.vgroup.height * 0.9,
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
                stroke_opacity=0,
            )
            .set_z_index(-1)
            .move_to(radar.vgroup, DOWN)
        )
        self.add(radome_circ, radar_box)

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        X_weights = np.linspace(-n_elem / 2 + 1 / 2, n_elem / 2 - 1 / 2, n_elem)
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.001)
        theta_max = VT(0.001)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)
        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        theta_min = VT(0.01)
        theta_max = VT(0.01)

        X = np.linspace(-n_elem / 2 - 0.05, n_elem / 2 + 0.05, 2**10)

        def get_f_window():
            window = np.clip(signal.windows.kaiser(2**10, beta=3), 0, None)
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        def get_ap_polar(polar_ax=polar_ax):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                # weights = np.array([~w for w in weight_trackers])
                weights = np.array([get_f_window()(x) for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~theta_min, ~theta_max, 1 / 400],
                    color=TX_COLOR,
                    use_smoothing=False,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)

        self.play(
            LaggedStart(
                target2_new.shift(RIGHT * 18).animate.shift(LEFT * 18),
                target1_new.shift(RIGHT * 18).animate.shift(LEFT * 18),
                AnimationGroup(theta_min @ (-PI), theta_max @ (PI)),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        target1_bez = CubicBezier(
            target1_new.get_top() + [0, 0.1, 0],
            target1_new.get_top() + [0, 1, 0],
            rres_eq_bw.copy().shift(RIGHT * 2.5).get_left() + [-1, 0, 0],
            rres_eq_bw.copy().shift(RIGHT * 2.5).get_left() + [-0.1, 0, 0],
        )
        target2_bez = CubicBezier(
            target2_new.get_top() + [0, 0.1, 0],
            target2_new.get_top() + [0, 3, 0],
            rres_eq_bw.copy().shift(RIGHT * 2.5).get_left() + [-1, 0, 0],
            rres_eq_bw.copy().shift(RIGHT * 2.5).get_left() + [-0.1, 0, 0],
        )
        self.play(
            Create(target1_bez),
            Create(target2_bez),
            rres_eq_bw.animate.shift(RIGHT * 2.5),
        )

        self.wait(0.5)

        self.play(steering_angle - 10)

        self.wait(0.5)

        self.play(steering_angle + 20)

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    radar.vgroup,
                    radar_box,
                    radome_circ,
                    target1_new,
                    target1_bez,
                    target2_new,
                    target2_bez,
                    rres_eq_bw,
                ),
                AnimationGroup(
                    steering_angle @ 0,
                    polar_ax.animate.rotate(-target_line.get_angle() + PI / 2).shift(
                        -polar_ax.c2p(0, 0)
                    ),
                    self.camera.frame.animate.scale_to_fit_width(
                        config.frame_width
                    ).move_to(ORIGIN),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(2)


class AngularResolution(Scene):
    def construct(self):
        r_min = -60

        x_len = config.frame_height * 0.6
        polar_ax = (
            Axes(
                x_range=[r_min, -r_min, r_min / 8],
                y_range=[r_min, -r_min, r_min / 8],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=x_len,
                y_length=x_len,
            )
            .set_opacity(0)
            .rotate(PI / 2)
        )

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        X_weights = np.linspace(-n_elem / 2 + 1 / 2, n_elem / 2 - 1 / 2, n_elem)
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.001)
        theta_max = VT(0.001)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)
        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        theta_min = VT(-PI)
        theta_max = VT(PI)

        X = np.linspace(-n_elem / 2 - 0.05, n_elem / 2 + 0.05, 2**10)

        def get_f_window():
            window = np.clip(signal.windows.kaiser(2**10, beta=3), 0, None)
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        def get_ap_polar(polar_ax=polar_ax):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                # weights = np.array([~w for w in weight_trackers])
                weights = np.array([get_f_window()(x) for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~theta_min, ~theta_max, 1 / 400],
                    color=TX_COLOR,
                    use_smoothing=False,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)


class VelocityResolution(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        title = Text(
            "Velocity Resolution", font_size=DEFAULT_FONT_SIZE * 1, font="Maple Mono"
        )

        self.play(
            title.next_to(self.camera.frame.get_bottom(), DOWN).animate.move_to(ORIGIN)
        )

        self.wait(0.5)

        TARGET1_COLOR = GREEN
        TARGET2_COLOR = BLUE
        car1 = (
            SVGMobject("../props/static/car.svg")
            .to_edge(LEFT, LARGE_BUFF)
            .shift(UP * 1.5)
            .set_fill(TARGET1_COLOR)
            .scale(0.8)
        )
        car2 = (
            SVGMobject("../props/static/car.svg")
            .to_edge(LEFT, LARGE_BUFF * 0.3)
            .shift(DOWN * 1.5)
            .set_fill(TARGET2_COLOR)
            .scale(0.8)
        )

        self.play(
            LaggedStart(
                title.animate.next_to(self.camera.frame.get_top(), UP),
                car1.shift(LEFT * 10).animate.shift(RIGHT * 10),
                car2.shift(LEFT * 10).animate.shift(RIGHT * 10),
                lag_ratio=0.3,
            )
        )
        self.remove(title)

        self.wait(0.5)

        vel1_dot = Dot(color=TARGET1_COLOR).next_to(car1, DOWN, SMALL_BUFF)
        vel1_arrow = Arrow(
            vel1_dot.get_center(),
            vel1_dot.get_center() + RIGHT * 2,
            buff=0,
            color=TARGET1_COLOR,
        )
        vel1_label = MathTex(
            r"v_1 = 100 \text{km} / \text{hr}", color=TARGET1_COLOR
        ).next_to(vel1_arrow, DOWN, SMALL_BUFF)
        vel2_dot = Dot(color=TARGET2_COLOR).next_to(car2, DOWN, SMALL_BUFF)
        vel2_arrow = Arrow(
            vel2_dot.get_center(),
            vel2_dot.get_center() + RIGHT * 2.2,
            buff=0,
            color=TARGET2_COLOR,
        )
        vel2_label = MathTex(
            r"v_2 = 105 \text{km} / \text{hr}", color=TARGET2_COLOR
        ).next_to(vel2_arrow, DOWN, SMALL_BUFF)

        self.play(
            LaggedStart(
                AnimationGroup(Create(vel1_dot), Create(vel2_dot)),
                AnimationGroup(GrowArrow(vel1_arrow), GrowArrow(vel2_arrow)),
                AnimationGroup(Write(vel1_label[0][:2]), Write(vel2_label[0][:2])),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Write(vel1_label[0][2:]),
                Write(vel2_label[0][2:]),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        t = np.arange(0, max_time, 1 / fs)

        window = signal.windows.blackman(N)
        fft_len = N * 8
        max_vel = wavelength / (4 * Tc)
        vel_res = wavelength / (2 * M * Tc)
        rmax = c * Tc * fs / (2 * bw)
        n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        target2_pos = VT(17)

        def plot_rd():
            targets = [(20, 8, 0), (~target2_pos, 10, 0)]
            cpi = np.array(
                [
                    (
                        np.sum(
                            [
                                np.sin(
                                    2 * PI * compute_f_beat(r) * t
                                    + m * compute_phase_diff(v)
                                )
                                * db_to_lin(p)
                                for r, v, p in targets
                            ],
                            axis=0,
                        )
                        + np.random.normal(0, 0.1, N)
                    )
                    * window
                    for m in range(M)
                ]
            )

            ranges_n = np.linspace(-rmax / 2, rmax / 2, N)
            range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)
            range_doppler = range_doppler[(ranges_n >= 0) & (ranges_n <= 40), :]
            range_doppler -= range_doppler.min()
            range_doppler /= range_doppler.max()

            cmap = get_cmap("viridis")
            range_doppler_fmt = np.uint8(cmap(10 * np.log10(range_doppler + 1)) * 255)
            range_doppler_fmt[range_doppler < 0.05] = [0, 0, 0, 0]

            rd_img = (
                ImageMobject(range_doppler_fmt, image_mode="RGBA")
                .stretch_to_fit_width(config.frame_width * 0.4)
                .stretch_to_fit_height(config.frame_width * 0.4)
                .to_edge(RIGHT, LARGE_BUFF)
            )
            rd_img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])
            return rd_img

        rd_img = always_redraw(plot_rd)

        rd_ax = Axes(
            x_range=[-0.5, 10, 2],
            y_range=[-0.5, 10, 2],
            tips=False,
            x_length=rd_img.width,
            y_length=rd_img.height,
            axis_config=dict(stroke_width=DEFAULT_STROKE_WIDTH * 1.2),
        )
        rd_ax.shift(rd_img.get_corner(DL) - rd_ax.c2p(0, 0))
        range_label = (
            Text("Range", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.5)
            .rotate(PI / 2)
            .next_to(rd_ax.c2p(0, 5), LEFT)
        )
        vel_label = Text(
            "Velocity", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.5
        ).next_to(rd_ax.c2p(5, 0), DOWN)

        self.play(
            FadeIn(rd_img),
            Create(rd_ax),
            Write(vel_label),
            Write(range_label),
        )

        self.play(
            Group(car2, vel2_arrow, vel2_dot, vel2_label)
            .animate(run_time=6)
            .shift(RIGHT * (LARGE_BUFF * 1.4)),
            target2_pos + 6,
            run_time=8,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate(
                run_time=0.5, rate_func=rate_functions.ease_in_sine
            ).shift(DOWN * config.frame_height)
        )

        self.wait(2)


class SigProc(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4).to_corner(
            DL, LARGE_BUFF
        ).shift(RIGHT * 4).set_z_index(3)

        self.play(radar.vgroup.shift(LEFT * 10).animate.shift(RIGHT * 10))

        self.wait(0.5)

        sigproc = (
            Rectangle(
                height=radar.vgroup.height * 1.2,
                width=radar.vgroup.width * 2,
                color=BLUE,
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
            )
            .next_to(radar.vgroup, LEFT, LARGE_BUFF, DOWN)
            .set_z_index(-2)
        )
        sigproc_right_box = (
            Rectangle(
                height=radar.vgroup.height * 1.2,
                width=radar.vgroup.width * 2,
                stroke_opacity=0,
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
            )
            .next_to(sigproc, RIGHT, 0.02)
            .set_z_index(1)
        )
        sigproc_conn = CubicBezier(
            radar.radome.get_left(),
            radar.radome.get_left() + [-1, 0, 0],
            sigproc.get_corner(UR) + [1, -0.3, 0],
            sigproc.get_corner(UR) + [0, -0.3, 0],
        ).set_z_index(2)
        proc_label = Text(
            "Processor",
            font_size=DEFAULT_FONT_SIZE * 0.6,
            font="Maple Mono",
        ).next_to(sigproc, UP)
        sig_label = Text(
            "Signal",
            font_size=DEFAULT_FONT_SIZE * 0.6,
            font="Maple Mono",
        ).next_to(proc_label, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                Create(sigproc_conn),
                Create(sigproc),
                AnimationGroup(Write(sig_label), Write(proc_label)),
            )
        )
        self.add(sigproc_right_box)

        self.wait(0.5)

        TARGET1_COLOR = RED
        TARGET2_COLOR = PURPLE
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
        pw = 0.2
        f = 10

        xmax1 = VT(0)
        xmax1_t1 = VT(0)
        xmax1_t2 = VT(0)
        tx1 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax1 - pw), ~xmax1, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx1_1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax1_t1 - pw), min(~xmax1_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx1_2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax1_t2 - pw), min(~xmax1_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )

        xmax2 = VT(0)
        xmax2_t1 = VT(0)
        xmax2_t2 = VT(0)
        tx2 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax2 - pw), ~xmax2, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx2_1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax2_t1 - pw), min(~xmax2_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx2_2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax2_t2 - pw), min(~xmax2_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )

        xmax3 = VT(0)
        xmax3_t1 = VT(0)
        xmax3_t2 = VT(0)
        tx3 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax3 - pw), ~xmax3, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx3_1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax3_t1 - pw), min(~xmax3_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx3_2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax3_t2 - pw), min(~xmax3_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )

        xmax4 = VT(0)
        xmax4_t1 = VT(0)
        xmax4_t2 = VT(0)
        tx4 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax4 - pw), ~xmax4, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx4_1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax4_t1 - pw), min(~xmax4_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx4_2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax4_t2 - pw), min(~xmax4_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )

        xmax5 = VT(0)
        xmax5_t1 = VT(0)
        xmax5_t2 = VT(0)
        tx5 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax5 - pw), ~xmax5, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx5_1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax5_t1 - pw), min(~xmax5_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx5_2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax5_t2 - pw), min(~xmax5_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )

        self.add(
            tx1,
            tx2,
            tx3,
            tx4,
            tx5,
            rx1_1,
            rx2_1,
            rx3_1,
            rx4_1,
            rx5_1,
            rx1_2,
            rx2_2,
            rx3_2,
            rx4_2,
            rx5_2,
        )

        self.play(
            LaggedStart(
                target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
                target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        data1 = Dot(color=ORANGE).move_to(sigproc_conn.get_start()).set_z_index(2)
        data2 = Dot(color=ORANGE).move_to(sigproc_conn.get_start()).set_z_index(2)
        data3 = Dot(color=ORANGE).move_to(sigproc_conn.get_start()).set_z_index(2)
        data4 = Dot(color=ORANGE).move_to(sigproc_conn.get_start()).set_z_index(2)
        data5 = Dot(color=ORANGE).move_to(sigproc_conn.get_start()).set_z_index(2)
        self.add(data1, data2, data3, data4, data5)

        t = np.arange(0, max_time, 1 / fs)

        window = signal.windows.blackman(N)
        fft_len = N * 8
        rmax = c * Tc * fs / (2 * bw)

        targets = [(20, 8, 0), (12, 20, -3)]
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            * db_to_lin(p)
                            for r, v, p in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                # * window
                for m in range(M)
            ]
        )
        cpi -= cpi.min()
        cpi /= cpi.max()

        ax1 = (
            Axes(
                x_range=[0, max_time / 2, max_time],
                y_range=[0, 1, 2],
                tips=False,
                x_length=sigproc.width * 0.9,
                y_length=sigproc.height / 5 * 0.8,
            )
            .set_z_index(2)
            .set_opacity(0)
        )
        ax2 = (
            Axes(
                x_range=[0, max_time / 2, max_time],
                y_range=[0, 1, 2],
                tips=False,
                x_length=sigproc.width * 0.9,
                y_length=sigproc.height / 5 * 0.8,
            )
            .set_z_index(2)
            .set_opacity(0)
        )
        ax3 = (
            Axes(
                x_range=[0, max_time / 2, max_time],
                y_range=[0, 1, 2],
                tips=False,
                x_length=sigproc.width * 0.9,
                y_length=sigproc.height / 5 * 0.8,
            )
            .set_z_index(2)
            .set_opacity(0)
        )
        ax4 = (
            Axes(
                x_range=[0, max_time / 2, max_time],
                y_range=[0, 1, 2],
                tips=False,
                x_length=sigproc.width * 0.9,
                y_length=sigproc.height / 5 * 0.8,
            )
            .set_z_index(2)
            .set_opacity(0)
        )
        ax5 = (
            Axes(
                x_range=[0, max_time / 2, max_time],
                y_range=[0, 1, 2],
                tips=False,
                x_length=sigproc.width * 0.9,
                y_length=sigproc.height / 5 * 0.8,
            )
            .set_z_index(2)
            .set_opacity(0)
        )
        axes = (
            Group(ax1, ax2, ax3, ax4, ax5)
            .arrange(DOWN, SMALL_BUFF * 0.5)
            .move_to(sigproc)
        )
        plot1 = (
            ax1.plot(
                interp1d(t, cpi[0], fill_value="extrapolate"),
                x_range=[0, max_time / 2, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            )
            .shift(RIGHT * sigproc.width)
            .set_z_index(-2)
        )
        plot2 = (
            ax2.plot(
                interp1d(t, cpi[1], fill_value="extrapolate"),
                x_range=[0, max_time / 2, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            )
            .shift(RIGHT * sigproc.width)
            .set_z_index(-2)
        )
        plot3 = (
            ax3.plot(
                interp1d(t, cpi[2], fill_value="extrapolate"),
                x_range=[0, max_time / 2, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            )
            .shift(RIGHT * sigproc.width)
            .set_z_index(-2)
        )
        plot4 = (
            ax4.plot(
                interp1d(t, cpi[3], fill_value="extrapolate"),
                x_range=[0, max_time / 2, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            )
            .shift(RIGHT * sigproc.width)
            .set_z_index(-2)
        )
        plot5 = (
            ax5.plot(
                interp1d(t, cpi[4], fill_value="extrapolate"),
                x_range=[0, max_time / 2, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            )
            .shift(RIGHT * sigproc.width)
            .set_z_index(-2)
        )
        self.add(axes, plot1, plot2, plot3, plot4, plot5)

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )
        self.play(
            LaggedStart(
                LaggedStart(
                    LaggedStart(
                        xmax1 @ (1 + pw),
                        LaggedStart(
                            xmax1_t1 @ (1 + pw + target_dist),
                            xmax1_t2 @ (1 + pw),
                            lag_ratio=0.2,
                        ),
                        lag_ratio=0.35,
                        run_time=2.5,
                    ),
                    MoveAlongPath(data1, sigproc_conn),
                    plot1.set_z_index(-2).animate.shift(LEFT * sigproc.width),
                    lag_ratio=0.6,
                ),
                LaggedStart(
                    LaggedStart(
                        xmax2 @ (1 + pw),
                        LaggedStart(
                            xmax2_t1 @ (1 + pw + target_dist),
                            xmax2_t2 @ (1 + pw),
                            lag_ratio=0.2,
                        ),
                        lag_ratio=0.35,
                        run_time=2.5,
                    ),
                    MoveAlongPath(data2, sigproc_conn),
                    plot2.set_z_index(-2).animate.shift(LEFT * sigproc.width),
                    lag_ratio=0.6,
                ),
                LaggedStart(
                    LaggedStart(
                        xmax3 @ (1 + pw),
                        LaggedStart(
                            xmax3_t1 @ (1 + pw + target_dist),
                            xmax3_t2 @ (1 + pw),
                            lag_ratio=0.2,
                        ),
                        lag_ratio=0.35,
                        run_time=2.5,
                    ),
                    MoveAlongPath(data3, sigproc_conn),
                    plot3.set_z_index(-2).animate.shift(LEFT * sigproc.width),
                    lag_ratio=0.6,
                ),
                LaggedStart(
                    LaggedStart(
                        xmax4 @ (1 + pw),
                        LaggedStart(
                            xmax4_t1 @ (1 + pw + target_dist),
                            xmax4_t2 @ (1 + pw),
                            lag_ratio=0.2,
                        ),
                        lag_ratio=0.35,
                        run_time=2.5,
                    ),
                    MoveAlongPath(data4, sigproc_conn),
                    plot4.set_z_index(-2).animate.shift(LEFT * sigproc.width),
                    lag_ratio=0.6,
                ),
                LaggedStart(
                    LaggedStart(
                        xmax5 @ (1 + pw),
                        LaggedStart(
                            xmax5_t1 @ (1 + pw + target_dist),
                            xmax5_t2 @ (1 + pw),
                            lag_ratio=0.2,
                        ),
                        lag_ratio=0.35,
                        run_time=2.5,
                    ),
                    MoveAlongPath(data5, sigproc_conn),
                    plot5.set_z_index(-2).animate.shift(LEFT * sigproc.width),
                    lag_ratio=0.6,
                ),
                lag_ratio=0.3,
            )
        )
        self.play(FadeOut(data1, data2, data3, data4, data5))

        self.wait(0.5)

        sigproc_new = (
            sigproc.copy()
            .stretch_to_fit_width(config.frame_width * 0.9)
            .stretch_to_fit_height(config.frame_height * 0.85)
            .move_to(sigproc, UR)
        )
        sigproc_label = Text(
            "Signal Processor",
            font_size=DEFAULT_FONT_SIZE * 0.6,
            font="Maple Mono",
        ).next_to(sigproc_new, UP, SMALL_BUFF)
        plots = Group(plot1, plot2, plot3, plot4, plot5)

        cpi_xmax = VT(max_time / 2)

        plot1_ud = always_redraw(
            lambda: ax1.plot(
                interp1d(t, cpi[0], fill_value="extrapolate"),
                x_range=[0, ~cpi_xmax, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            ).set_z_index(1)
        )
        plot2_ud = always_redraw(
            lambda: ax2.plot(
                interp1d(t, cpi[1], fill_value="extrapolate"),
                x_range=[0, ~cpi_xmax, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            ).set_z_index(1)
        )
        plot3_ud = always_redraw(
            lambda: ax3.plot(
                interp1d(t, cpi[2], fill_value="extrapolate"),
                x_range=[0, ~cpi_xmax, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            ).set_z_index(1)
        )
        plot4_ud = always_redraw(
            lambda: ax4.plot(
                interp1d(t, cpi[3], fill_value="extrapolate"),
                x_range=[0, ~cpi_xmax, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            ).set_z_index(1)
        )
        plot5_ud = always_redraw(
            lambda: ax5.plot(
                interp1d(t, cpi[4], fill_value="extrapolate"),
                x_range=[0, ~cpi_xmax, max_time / 200],
                color=ORANGE,
                use_smoothing=True,
            ).set_z_index(1)
        )
        self.add(plot1_ud, plot2_ud, plot3_ud, plot4_ud, plot5_ud)
        self.remove(plot1, plot2, plot3, plot4, plot5)

        self.play(
            Transform(sigproc, sigproc_new),
            self.camera.frame.animate.move_to(sigproc_new, DOWN).shift(DOWN / 3),
            ReplacementTransform(sig_label, sigproc_label[:7]),
            ReplacementTransform(proc_label, sigproc_label[7:]),
            axes.animate.scale_to_fit_height(sigproc_new.height * 0.8).move_to(
                sigproc_new
            ),
        )

        self.wait(0.5)

        num_samples = 10
        sample_rects1 = ax1.get_riemann_rectangles(
            plot1,
            input_sample_type="center",
            x_range=[0, max_time / 2],
            dx=max_time / (2 * num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)
        sample_rects2 = ax2.get_riemann_rectangles(
            plot2,
            input_sample_type="center",
            x_range=[0, max_time / 2],
            dx=max_time / (2 * num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)
        sample_rects3 = ax3.get_riemann_rectangles(
            plot3,
            input_sample_type="center",
            x_range=[0, max_time / 2],
            dx=max_time / (2 * num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)
        sample_rects4 = ax4.get_riemann_rectangles(
            plot4,
            input_sample_type="center",
            x_range=[0, max_time / 2],
            dx=max_time / (2 * num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)
        sample_rects5 = ax5.get_riemann_rectangles(
            plot5,
            input_sample_type="center",
            x_range=[0, max_time / 2],
            dx=max_time / (2 * num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)

        colors = [
            ManimColor.from_hex("#00FFFF"),
            ManimColor.from_hex("#CCFF00"),
            ManimColor.from_hex("#FF69B4"),
            ManimColor.from_hex("#FFA500"),
            ManimColor.from_hex("#FF3333"),
            ManimColor.from_hex("#FFFF00"),
            ManimColor.from_hex("#BF00FF"),
            ManimColor.from_hex("#00BFFF"),
            ManimColor.from_hex("#FFFFFF"),
            ManimColor.from_hex("#FFDAB9"),
        ]
        colors_pastel = [
            ManimColor.from_hex("#A8E6CF"),
            ManimColor.from_hex("#DCE775"),
            ManimColor.from_hex("#FFB3BA"),
            ManimColor.from_hex("#FFD580"),
            ManimColor.from_hex("#FF9AA2"),
            ManimColor.from_hex("#FFFFB3"),
            ManimColor.from_hex("#D5AAFF"),
            ManimColor.from_hex("#B3E5FC"),
            ManimColor.from_hex("#F8F8FF"),
            ManimColor.from_hex("#FFE5B4"),
        ]
        colors_vibrant = [
            ManimColor.from_hex("#4DD0E1"),
            ManimColor.from_hex("#81C784"),
            ManimColor.from_hex("#FFD54F"),
            ManimColor.from_hex("#FF8A65"),
            ManimColor.from_hex("#BA68C8"),
            ManimColor.from_hex("#4FC3F7"),
            ManimColor.from_hex("#AED581"),
            ManimColor.from_hex("#FFF176"),
            ManimColor.from_hex("#64B5F6"),
            ManimColor.from_hex("#FFB74D"),
        ]
        sample_rects_all = VGroup(
            [
                sample_rects1,
                sample_rects2,
                sample_rects3,
                sample_rects4,
                sample_rects5,
            ]
        )
        for sample_rects in sample_rects_all:
            for sample_rect, color in zip(sample_rects, colors_vibrant):
                sample_rect.set_fill(color=color)

        self.next_section(skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                *[
                    LaggedStart(*[FadeIn(sr) for sr in srs], lag_ratio=0.15)
                    for srs in sample_rects_all
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        down_arrow = Arrow(
            sample_rects_all[0][0].get_center(),
            sample_rects_all[-1][0].get_center(),
            buff=0,
        ).set_z_index(3)

        phis = Group(
            *[
                MathTex(r"0").next_to(srs[0], LEFT)
                if n == 0
                else MathTex(r"\phi").next_to(srs[0], LEFT)
                if n == 1
                else MathTex(f"{n}\\phi").next_to(srs[0], LEFT)
                for n, srs in enumerate(sample_rects_all)
            ]
        )

        self.play(
            GrowArrow(down_arrow),
            LaggedStart(*[GrowFromCenter(phi) for phi in phis], lag_ratio=0.2),
        )

        self.wait(0.5)

        fft_arrow = Arrow(
            self.camera.frame.get_center() + LEFT * 0.4,
            self.camera.frame.get_center() + RIGHT,
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
        )
        fft_label = MathTex(r"\mathcal{F}").next_to(fft_arrow, UP)
        fft_line_top = CubicBezier(
            sample_rects_all[0][0].get_corner(UR) + [0.1, 0, 0],
            sample_rects_all[0][0].get_corner(UR) + [1, 0, 0],
            fft_arrow.get_left() + [-1, 0, 0],
            fft_arrow.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH,
        )
        fft_line_bot = CubicBezier(
            sample_rects_all[-1][0].get_corner(DR) + [0.1, 0, 0],
            sample_rects_all[-1][0].get_corner(DR) + [1, 0, 0],
            fft_arrow.get_left() + [-1, 0, 0],
            fft_arrow.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH,
        )

        self.play(
            cpi_xmax @ (max_time / (num_samples * 2)),
            *[
                LaggedStart(
                    *[sr.animate.set_opacity(0) for sr in srs[1:][::-1]], lag_ratio=0.15
                )
                for srs in sample_rects_all
            ],
            FadeOut(down_arrow),
        )

        self.wait(0.5)

        vel_ax = Axes(
            x_range=[0, 20, 5],
            y_range=[0, 30, 10],
            tips=False,
            x_length=config.frame_width * 0.3,
            y_length=config.frame_height * 0.4,
        ).next_to(fft_arrow, RIGHT, MED_LARGE_BUFF)

        fs_new = 400
        t_new = np.arange(0, 1, 1 / fs_new)
        fft_len = 2**14

        f1 = 9
        f2 = 15

        np.random.seed(0)
        window_new = signal.windows.kaiser(t_new.size, beta=3)

        def get_sig():
            sig = (
                np.sin(2 * PI * f1 * t_new)
                + 0.4 * np.sin(2 * PI * f2 * t_new)
                + np.random.normal(0, 0.3, t_new.size)
            ) * window_new
            return sig

        def get_fft_plot():
            sig = get_sig()
            X_k = 10 * np.log10(np.abs(fft(sig, fft_len)) / (t_new.size / 2)) + 30
            freq = np.linspace(-fs_new / 2, fs_new / 2, fft_len)
            f_X_k = interp1d(freq, np.clip(fftshift(X_k), 0, None))
            return vel_ax.plot(
                f_X_k,
                x_range=[0, 20, 20 / 400],
                color=ORANGE,
                # stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )

        vel_plot = get_fft_plot()
        vel_ax_y_label = (
            Text("Magnitude", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.4)
            .rotate(PI / 2)
            .next_to(vel_ax, LEFT, SMALL_BUFF)
        )
        vel_ax_x_label = Text(
            "Velocity", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.4
        ).next_to(vel_ax, DOWN, SMALL_BUFF)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(fft_line_top),
                    Create(fft_line_bot),
                ),
                AnimationGroup(GrowArrow(fft_arrow), Write(fft_label)),
                LaggedStart(
                    Create(vel_ax),
                    AnimationGroup(Write(vel_ax_y_label), Write(vel_ax_x_label)),
                    Create(vel_plot),
                    lag_ratio=0.3,
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        doppler_thumbnail = (
            ImageMobject(
                "../04_fmcw_doppler/media/images/fmcw_doppler/thumbnails/Thumbnail_1.png"
            )
            .scale_to_fit_width(config.frame_width * 0.7)
            .move_to(self.camera.frame)
        ).set_z_index(5)
        thumbnail_box = SurroundingRectangle(
            doppler_thumbnail, color=RED, buff=0
        ).set_z_index(5)
        thumbnail = Group(doppler_thumbnail, thumbnail_box)

        self.play(
            thumbnail.next_to(self.camera.frame, DOWN).animate.move_to(
                self.camera.frame
            )
        )

        self.wait(0.5)

        axes_copy = axes.copy().next_to(
            self.camera.frame.get_left(), RIGHT, LARGE_BUFF * 2
        )

        self.play(
            LaggedStart(
                thumbnail.animate.shift(UP * 10),
                FadeOut(
                    phis,
                    sigproc,
                    sigproc_label,
                    fft_line_top,
                    fft_line_bot,
                    fft_arrow,
                    fft_label,
                    sigproc_conn,
                ),
                axes.animate.move_to(axes_copy),
                sample_rects_all.animate.shift(
                    axes_copy.get_center() - axes.get_center()
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            cpi_xmax @ (max_time / 2),
            *[
                LaggedStart(
                    *[
                        sr.animate.set_stroke(opacity=1).set_fill(opacity=0.7)
                        for sr in srs[1:]
                    ],
                    lag_ratio=0.15,
                )
                for srs in sample_rects_all
            ],
        )

        self.wait(0.5)

        n_ts_disp = 3
        ts_label = MathTex("T_s").next_to(sample_rects_all[0][0], UP)
        n_ts = Group(
            *[
                MathTex(
                    f"{idx if idx > 1 else ''}T_s", font_size=DEFAULT_FONT_SIZE * 0.5
                )
                .next_to(sample_rects[idx - 1], DOWN)
                .set_y(ts_label.get_y())
                # .shift(UP * (0.5 if idx % 2 == 0 else 0))
                for idx in range(1, n_ts_disp + 2)
            ],
            MathTex(r"\cdots")
            .next_to(sample_rects[len(sample_rects) // 2], DOWN)
            .set_y(ts_label.get_y()),
            MathTex(f"NT_s", font_size=DEFAULT_FONT_SIZE * 0.5)
            .next_to(sample_rects[-1], DOWN)
            .set_y(ts_label.get_y()),
        )

        self.play(LaggedStart(FadeIn(*[ts for ts in n_ts], lag_ratio=0.2)))

        self.wait(0.5)

        ts_eqn = MathTex(r"T_s = \frac{1}{f_s}").next_to(n_ts[-1], RIGHT, LARGE_BUFF)

        self.play(
            LaggedStart(
                TransformFromCopy(n_ts[-1][0][1:], ts_eqn[0][:2]),
                *[GrowFromCenter(m) for m in ts_eqn[0][2:]],
                lag_ratio=0.2,
            )
        )

        self.wait(2)


class TradeOff(Scene):
    def construct(self): ...


class ImgTest(Scene):
    def construct(self):
        cmap = get_cmap("viridis")
        image = ImageMobject(
            cmap(
                np.uint8(
                    [
                        [0, 100, 30, 200],
                        [255, 0, 5, 33],
                    ]
                )
            )
        )
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])
        image.height = 7
        self.add(image)
