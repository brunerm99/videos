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

SKIP_ANIMATIONS_OVERRIDE = True


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


def pad2d(x, target_shape):
    pad_rows = target_shape[0] - x.shape[0]
    pad_cols = target_shape[1] - x.shape[1]

    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    return np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")


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


class Intro(MovingCameraScene):
    def construct(self):
        box1 = Square(color=RED).to_edge(LEFT, LARGE_BUFF)
        box2 = Square(color=RED).to_edge(RIGHT, LARGE_BUFF)
        box_line = always_redraw(
            lambda: Line(box1.get_right(), box2.get_left()).shift(DOWN * 2)
        )
        box_line_l = always_redraw(
            lambda: Line(box_line.get_start() + DOWN / 8, box_line.get_start() + UP / 8)
        )
        box_line_r = always_redraw(
            lambda: Line(box_line.get_end() + DOWN / 8, box_line.get_end() + UP / 8)
        )

        self.play(
            GrowFromCenter(box1),
            GrowFromCenter(box2),
            Create(box_line_l),
            Create(box_line),
            Create(box_line_r),
        )

        self.wait(0.5)

        self.play(Group(box1, box2).animate.arrange(RIGHT))

        self.wait(0.5)

        self.play(
            ShrinkToCenter(box1),
            ShrinkToCenter(box2),
            Uncreate(box_line_l),
            Uncreate(box_line),
            Uncreate(box_line_r),
        )

        self.wait(0.5)

        car1 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(BLUE)
            .scale(0.6)
            .to_edge(RIGHT, LARGE_BUFF * 2.5)
            .shift(UP)
        )
        car2 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(YELLOW)
            .scale(0.6)
            .next_to(car1, LEFT, LARGE_BUFF * 1.5)
        )
        car3 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(ORANGE)
            .scale(0.6)
            .next_to(car1, DOWN, MED_LARGE_BUFF)
        )
        car4 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(RED)
            .scale(0.6)
            .next_to(car3, DOWN, MED_LARGE_BUFF)
        )
        you = (
            SVGMobject("../props/static/car.svg")
            .set_fill(WHITE)
            .scale(0.6)
            .next_to(car3, LEFT, LARGE_BUFF * 6)
            .shift(RIGHT * 0.3)
        )
        self.play(
            LaggedStart(
                car1.shift(LEFT * 12).animate.shift(RIGHT * 12),
                car2.shift(LEFT * 12).animate.shift(RIGHT * 12),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                car3.shift(LEFT * 12).animate.shift(RIGHT * 12),
                car4.shift(LEFT * 12).animate.shift(RIGHT * 12),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(you.shift(LEFT * 12).animate.shift(RIGHT * 12))

        self.wait(0.5)

        vlabel = MathTex(r"v_{1} \approx v_{2}").next_to(
            Group(car1, car2), UP, LARGE_BUFF
        )
        vlabel[0][:2].set_color(YELLOW)
        vlabel[0][3:].set_color(BLUE)

        self.play(Write(vlabel))

        self.wait(0.5)

        rlabel = MathTex(r"R_{3} \approx R_{4}").next_to(
            Group(car3, car4), RIGHT, MED_SMALL_BUFF
        )
        rlabel[0][:2].set_color(ORANGE)
        rlabel[0][3:].set_color(RED)

        self.play(Write(rlabel))

        self.wait(0.5)

        radar_beam = Line(you.get_right(), car2.get_left(), color=BLUE)

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
            .rotate(radar_beam.get_angle())
        )
        polar_ax.shift(you.get_right() - polar_ax.c2p(0, 0))

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
                    use_smoothing=True,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot, polar_ax)

        labels1 = (
            MathTex(r"R_1, v_1, \theta_1")
            .set_color(YELLOW)
            .next_to(car2, UP, SMALL_BUFF)
        )
        labels2 = (
            MathTex(r"R_2, v_2, \theta_2").set_color(BLUE).next_to(car1, UP, SMALL_BUFF)
        )
        labels3 = (
            MathTex(r"R_3, v_3, \theta_3")
            .set_color(ORANGE)
            .next_to(car3, RIGHT, SMALL_BUFF)
        )
        labels4 = (
            MathTex(r"R_4, v_4, \theta_4")
            .set_color(RED)
            .next_to(car4, RIGHT, SMALL_BUFF)
        )

        self.play(
            theta_min @ (-PI / 2 - radar_beam.get_angle()),
            theta_max @ (PI / 2 - radar_beam.get_angle()),
            Create(radar_beam),
            Write(labels1),
        )

        self.wait(0.5)

        radar_beam2 = Line(you.get_right(), car1.get_left(), color=BLUE)
        radar_beam3 = Line(you.get_right(), car3.get_left(), color=BLUE)
        radar_beam4 = Line(you.get_right(), car4.get_left(), color=BLUE)

        self.play(
            theta_min @ (-PI / 2 - radar_beam2.get_angle()),
            theta_max @ (PI / 2 - radar_beam2.get_angle()),
            Transform(radar_beam, radar_beam2),
            polar_ax.animate.rotate(radar_beam2.get_angle() - radar_beam.get_angle()),
            Write(labels2),
        )

        self.wait(0.5)

        self.play(
            theta_min @ (-PI / 2 - radar_beam3.get_angle()),
            theta_max @ (PI / 2 - radar_beam3.get_angle()),
            Transform(radar_beam, radar_beam3),
            polar_ax.animate.rotate(radar_beam3.get_angle() - radar_beam2.get_angle()),
            Write(labels3),
        )

        self.wait(0.5)

        self.play(
            theta_min @ (-PI / 2 - radar_beam4.get_angle()),
            theta_max @ (PI / 2 - radar_beam4.get_angle()),
            Transform(radar_beam, radar_beam4),
            polar_ax.animate.rotate(radar_beam4.get_angle() - radar_beam3.get_angle()),
            Write(labels4),
        )

        self.wait(0.5)

        rvt = (
            Group(
                MathTex(r"\Delta R", font_size=DEFAULT_FONT_SIZE * 3),
                MathTex(r"\Delta v", font_size=DEFAULT_FONT_SIZE * 3),
                MathTex(r"\Delta \theta", font_size=DEFAULT_FONT_SIZE * 3),
            )
            .arrange(RIGHT, LARGE_BUFF * 1.5)
            .shift(DOWN * config.frame_height)
        )
        rv_arrow = CurvedArrow(
            rvt[0].get_top() + [0, 0.1, 0],
            rvt[1].get_top() + [0, 0.1, 0],
            angle=-TAU / 4,
        )
        vt_arrow = CurvedArrow(
            rvt[1].get_bottom() + [0, -0.1, 0],
            rvt[2].get_bottom() + [0, -0.1, 0],
            angle=-TAU / 4,
        )
        vt_arrow = CurvedArrow(
            rvt[1].get_bottom() + [0, -0.1, 0],
            rvt[2].get_bottom() + [0, -0.1, 0],
            # angle=-TAU / 4,
        )
        tr_arrow = CurvedArrow(
            rvt[2].get_top() + [0, 0.1, 0],
            rvt[0].get_top() + [-0.2, 0.1, 0],
            angle=TAU / 3,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * config.frame_height),
                ReplacementTransform(labels4[0][0], rvt[0][0][1], path_arc=PI / 2),
                ReplacementTransform(labels4[0][3], rvt[1][0][1], path_arc=PI / 2),
                ReplacementTransform(labels4[0][6], rvt[2][0][1], path_arc=PI / 2),
                GrowFromCenter(rvt[0][0][0]),
                GrowFromCenter(rvt[1][0][0]),
                GrowFromCenter(rvt[2][0][0]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(rv_arrow), Create(vt_arrow), Create(tr_arrow), lag_ratio=0.3
            )
        )

        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"radar\_cheatsheet.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = (
            Group(notebook_box, notebook_reminder)
            .to_edge(DOWN, MED_LARGE_BUFF)
            .shift(DOWN * config.frame_height * 2)
        )
        self.add(notebook)
        nbsc1 = (
            ImageMobject("./static/nb_sc1.png")
            .scale_to_fit_width(config.frame_width * 0.7)
            .next_to(notebook, UP)
        )
        nbsc2 = (
            ImageMobject("./static/nb_sc2.png")
            .scale_to_fit_height(config.frame_height * 0.6)
            .next_to(notebook, UP)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * config.frame_height),
                nbsc1.shift(LEFT * 20).animate.shift(RIGHT * 20),
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                nbsc1.animate.shift(RIGHT * 20),
                nbsc2.shift(LEFT * 20).animate.shift(RIGHT * 20),
            )
        )

        self.wait(0.5)

        self.play(nbsc2.animate.shift(RIGHT * 20), notebook.animate.shift(DOWN * 10))

        self.wait(2)


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

        self.next_section(skip_animations=skip_animations(True))
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


class AngularResolution(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
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
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.001)
        theta_max = VT(0.001)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 2000)
        u = np.sin(theta)
        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        theta_min = VT(-PI)
        theta_max = VT(PI)

        beta = VT(3)
        n_elem_vt = VT(17)

        def get_f_window():
            # window = np.ones(2**10)
            X = np.linspace(-~n_elem_vt / 2 - 0.05, ~n_elem_vt / 2 + 0.05, 2**10)
            window = np.clip(signal.windows.kaiser(2**10, beta=~beta), 0, None)
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        def get_ap_polar(polar_ax=polar_ax):
            def updater():
                X_weights = np.linspace(
                    -~n_elem_vt / 2 + 1 / 2, ~n_elem_vt / 2 - 1 / 2, int(~n_elem_vt)
                )
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
                    use_smoothing=True,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)

        self.wait(0.5)

        theta_label = MathTex(r"\theta").next_to(
            polar_ax.copy().shift(DOWN), UP, LARGE_BUFF
        )
        theta_curve = CurvedDoubleArrow(
            polar_ax.c2p(-r_min / 2, r_min / 2),
            polar_ax.c2p(-r_min / 2, -r_min / 2),
        )

        self.play(
            LaggedStart(
                polar_ax.animate.shift(DOWN),
                GrowFromCenter(theta_label),
                FadeIn(theta_curve),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        title = Text(
            "Angular Resolution", font_size=DEFAULT_FONT_SIZE * 1, font="Maple Mono"
        ).next_to(self.camera.frame, UP)

        self.camera.frame.save_state()
        self.play(Write(title), self.camera.frame.animate.scale(1.2).shift(UP))

        self.wait(0.5)

        self.play(Unwrite(title), self.camera.frame.animate.restore())

        self.wait(0.5)

        theta_vt = VT(0)
        dot_left = always_redraw(
            lambda: Dot(color=ORANGE).move_to(
                polar_ax.input_to_graph_point(~theta_vt, AF_polar_plot)
            )
        )
        dot_right = always_redraw(
            lambda: Dot(color=ORANGE).move_to(
                polar_ax.input_to_graph_point(-~theta_vt, AF_polar_plot)
            )
        )

        # self.add(dot_left)
        self.play(Create(dot_left), Create(dot_right))

        self.wait(0.5)

        self.play(theta_vt @ PI, run_time=3)

        self.wait(0.5)

        fnbw = 4 * PI / (n_elem * ~beta)
        self.play(theta_vt @ (fnbw * 2.06))

        self.wait(0.5)

        theta_3db = 2.1 / n_elem

        self.play(theta_vt @ (theta_3db * 2))

        self.wait(0.5)

        theta_3db_arc = ArcBetweenPoints(
            dot_right.get_center() + [0.1, 0.1, 0],
            dot_right.get_center() + [2, 1, 0],
            angle=-TAU / 8,
            color=ORANGE,
        )

        theta_3db_label = MathTex(
            r"\theta_{3 \text{dB}} \approx \frac{\text{max}}{2}", color=ORANGE
        ).next_to(theta_3db_arc.get_end(), RIGHT, SMALL_BUFF)
        theta_3db_label[0][-5:-2].set_color(GREEN)

        max_dot = Dot(color=GREEN).move_to(
            polar_ax.input_to_graph_point(0, AF_polar_plot)
        )
        max_curve = ArcBetweenPoints(
            max_dot.get_center() + [0.1, 0.1, 0],
            theta_3db_label[0][-5:-2].get_corner(UL) + [-0.1, 0.1, 0],
            angle=-TAU / 8,
            color=GREEN,
        )

        self.play(
            LaggedStart(
                FadeOut(theta_curve),
                Create(theta_3db_arc),
                GrowFromCenter(theta_3db_label[0][:-6]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(theta_3db_label[0][-6]),
                Create(max_dot),
                Create(max_curve),
                LaggedStart(
                    *[GrowFromCenter(m) for m in theta_3db_label[0][-5:]], lag_ratio=0.1
                ),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            Uncreate(max_curve), Uncreate(theta_3db_arc), theta_vt @ (fnbw * 2.06)
        )

        self.wait(0.5)

        fnbw_label = MathTex(r"\theta_{\text{first-null}}", color=ORANGE).next_to(
            theta_3db_label, DOWN, LARGE_BUFF
        )

        fnbw_arc = ArcBetweenPoints(
            dot_right.get_center() + [0.1, 0.1, 0],
            fnbw_label.get_left() + [-0.1, 0, 0],
            angle=-TAU / 8,
            color=ORANGE,
        )

        self.play(
            LaggedStart(
                Create(fnbw_arc),
                *[GrowFromCenter(m) for m in fnbw_label[0]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            polar_ax.animate.rotate(-PI / 2).shift(LEFT * 2),
            FadeOut(fnbw_label, fnbw_arc, theta_label, max_dot),
            theta_vt @ (theta_3db * 2),
        )

        self.wait(0.5)

        TARGET1_COLOR = GREEN
        TARGET2_COLOR = BLUE
        car1 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(TARGET1_COLOR)
            .scale(0.6)
            .flip()
        ).next_to(
            polar_ax.input_to_graph_point(0, AF_polar_plot), RIGHT, LARGE_BUFF * 2
        )
        car2 = (
            SVGMobject("../props/static/car.svg")
            .set_fill(TARGET2_COLOR)
            .scale(0.6)
            .flip()
            .next_to(dot_right, RIGHT, LARGE_BUFF * 2)
            .shift(DOWN * 0.5)
        )

        car1_arrow = Arrow(polar_ax.c2p(0, 0), car1.get_left(), color=TARGET1_COLOR)

        self.play(
            LaggedStart(
                car1.shift(RIGHT * 10).animate.shift(LEFT * 10),
                GrowArrow(car1_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        car2_arrow = Arrow(polar_ax.c2p(0, 0), car2.get_left(), color=TARGET2_COLOR)

        self.play(
            LaggedStart(
                car2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                GrowArrow(car2_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        car_pow = MathTex(r"P_{\text{car}2} \le \frac{P_{\text{car}1}}{2}").next_to(
            polar_ax, UP
        )
        car_pow[0][:5].set_color(TARGET2_COLOR)
        car_pow[0][6:11].set_color(TARGET1_COLOR)

        self.play(Write(car_pow))

        self.wait(0.5)

        self.play(
            FadeOut(car_pow, car1, car2, car1_arrow, car2_arrow, dot_right, dot_left),
            Group(polar_ax, theta_3db_label).animate.arrange(RIGHT, LARGE_BUFF * 3),
        )

        self.wait(0.5)

        theta_3db_label_qmark = Tex(
            r"$\theta_{3 \text{dB}} = $ ?", color=ORANGE
        ).move_to(theta_3db_label, LEFT)

        self.play(Transform(theta_3db_label[0][-6:], theta_3db_label_qmark[0][-2:]))

        self.wait(0.5)

        # self.play(beta @ 0.5)

        self.play(n_elem_vt @ 21, run_time=3)

        self.wait(0.5)

        target1 = (
            (
                SVGMobject("../props/static/plane.svg")
                .scale(0.7)
                .rotate(PI * 0.75)
                .set_fill(TARGET1_COLOR)
                .set_color(TARGET1_COLOR)
            )
            .next_to(polar_ax, RIGHT, LARGE_BUFF * 2)
            .rotate(PI / 6, about_point=polar_ax.c2p(0))
            .rotate(-PI / 6)
        )

        target2 = (
            (
                SVGMobject("../props/static/plane.svg")
                .scale(0.7)
                .rotate(PI * 0.75)
                .set_fill(TARGET2_COLOR)
                .set_color(TARGET2_COLOR)
            )
            .next_to(polar_ax, RIGHT, LARGE_BUFF * 2)
            .rotate(PI / 6, about_point=polar_ax.c2p(0))
            .rotate(-PI / 6)
            .to_edge(DOWN, LARGE_BUFF)
        )

        self.play(
            polar_ax.animate.rotate(PI / 6),
            target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        self.play(
            target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        check = Tex(r"\checkmark", color=GREEN).scale(1.8).next_to(target1, LEFT)
        x = Tex(r"$\times$", color=RED).scale(1.8).next_to(target2, LEFT)

        self.play(LaggedStart(GrowFromCenter(check), GrowFromCenter(x), lag_ratio=0.4))

        self.wait(0.5)

        self.play(
            Group(target1, check).animate.shift(UP * 5),
            Group(target2, x).animate.shift(DOWN * 5),
        )

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/cloud.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .next_to(polar_ax, RIGHT, LARGE_BUFF * 0.5)
            .shift(UP * 0.5)
        )

        self.play(
            polar_ax.animate.shift(DOWN * 2),
            # FadeIn(cloud),
            theta_3db_label.animate.shift(DOWN * 2),
        )

        #   4 * 2 * PI / (~n_elem_vt * ~beta)
        bw_line_dist = -r_min * 2.5
        bw_line_l_color_interp = VT(0)
        bw_line_l = always_redraw(
            lambda: Line(
                polar_ax.c2p(0, 0),
                polar_ax.c2p(
                    bw_line_dist * np.cos(2.1 / ~n_elem_vt * 2),
                    bw_line_dist * np.sin(2.1 / ~n_elem_vt * 2),
                ),
                color=interpolate_color(WHITE, PURPLE, ~bw_line_l_color_interp),
            )
        )
        bw_line_r = always_redraw(
            lambda: Line(
                polar_ax.c2p(0, 0),
                polar_ax.c2p(
                    bw_line_dist * np.cos(-2.1 / ~n_elem_vt * 2),
                    bw_line_dist * np.sin(-2.1 / ~n_elem_vt * 2),
                ),
            )
        )
        bw_line_tri = Polygon(
            bw_line_l.get_start(),
            bw_line_l.get_end(),
            bw_line_r.get_end(),
            color=RED,
            fill_color=RED,
            fill_opacity=1,
        )

        particle_opacity = VT(0.7)

        def get_ellipse(shift, width, height, shifted=True, fade=True):
            def updater():
                coords = polar_ax.p2c(shift + cloud.get_center())
                angle = np.tan(coords[1] / coords[0])
                fnbw = 2.1 / ~n_elem_vt * 2
                color = GREEN if np.abs(angle) <= np.abs(fnbw) else BLUE
                opacity = ~particle_opacity if fade else 0.7
                ell = Ellipse(
                    width=width,
                    height=height,
                    stroke_opacity=0,
                    fill_color=color,
                    fill_opacity=opacity,
                ).shift(shift)
                if shifted:
                    ell.shift(cloud.get_center())
                return ell

            return updater

        n_particles = 100
        np.random.seed(0)
        particles = [
            always_redraw(
                get_ellipse(
                    cloud.width * np.random.normal(0, 0.25) * RIGHT
                    + cloud.height * np.random.normal(0, 0.25) * UP,
                    width=np.random.normal(0.2, 0.03, 1),
                    height=np.random.normal(0.2, 0.03, 1),
                )
            )
            for _ in range(n_particles)
        ]

        def is_overlapping(mob1, mob2):
            left1, right1 = mob1.get_left()[0], mob1.get_right()[0]
            left2, right2 = mob2.get_left()[0], mob2.get_right()[0]
            bottom1, top1 = mob1.get_bottom()[1], mob1.get_top()[1]
            bottom2, top2 = mob2.get_bottom()[1], mob2.get_top()[1]

            if right1 < left2 or right2 < left1:
                return False
            if bottom1 > top2 or bottom2 > top1:
                return False
            return True

        def remove_overlapping_objects(mobjects):
            cleaned = []
            for mob in mobjects:
                if any(is_overlapping(mob, kept) for kept in cleaned):
                    continue
                cleaned.append(mob)
            return cleaned

        particles = Group(*remove_overlapping_objects(particles))
        end_n_elem = ~n_elem_vt + 10
        particles.add(
            always_redraw(
                get_ellipse(
                    polar_ax.c2p(
                        bw_line_dist * np.cos(2.1 / ~n_elem_vt * 2 / np.sqrt(2)),
                        bw_line_dist * np.sin(2.1 / ~n_elem_vt * 2 / np.sqrt(2)),
                    ),
                    width=np.random.normal(0.2, 0.03, 1),
                    height=np.random.normal(0.2, 0.03, 1),
                    shifted=False,
                    fade=False,
                )
            ),
            always_redraw(
                get_ellipse(
                    polar_ax.c2p(
                        bw_line_dist * np.cos(-2.1 / ~n_elem_vt * 2 / np.sqrt(2)),
                        bw_line_dist * np.sin(-2.1 / ~n_elem_vt * 2 / np.sqrt(2)),
                    ),
                    width=np.random.normal(0.2, 0.03, 1),
                    height=np.random.normal(0.2, 0.03, 1),
                    shifted=False,
                    fade=False,
                )
            ),
        )

        # particles[0].set_fill(color=GREEN)
        # coords = polar_ax.p2c(particles[0].get_center())
        # print(np.tan(coords[1] / coords[0]), -4 * 2 * PI / (~n_elem_vt * ~beta))

        # p1 = always_redraw(get_ellipse([0.1, 0.2, 0]))
        # self.add(p1)

        self.next_section(skip_animations=skip_animations(True))
        self.play(Create(bw_line_r), Create(bw_line_l), *[Create(m) for m in particles])

        self.wait(0.5)

        theta_3db_label_1 = Tex(
            r"$\theta_{3 \text{dB}} = 1^\circ$", color=ORANGE
        ).move_to(theta_3db_label, LEFT)

        self.play(
            theta_3db_label.animate.shift(DOWN * 10),
            theta_3db_label_1.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))
        self.play(n_elem_vt @ end_n_elem, run_time=3)

        self.wait(0.5)

        # for idx, p in enumerate(particles):
        #     self.add(Tex(f"{idx}").scale(0.5).move_to(p))

        self.play(particle_opacity @ 0.1)

        self.wait(0.5)

        cam_group = Group(AF_polar_plot, particles, bw_line_l, bw_line_r)
        l = Line(polar_ax.c2p(0, 0), polar_ax.c2p(-r_min, 0))
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                cam_group.height * 1.2
            ).move_to(cam_group),
            theta_3db_label_1.animate.shift(LEFT),
        )

        self.wait(0.5)

        tri_line = Line(bw_line_l.get_end(), bw_line_r.get_end()).set_z_index(-1)

        self.play(Create(tri_line))

        self.wait(0.5)

        theta_arc = ArcBetweenPoints(
            polar_ax.c2p(
                bw_line_dist * 0.25 * np.cos(-2.1 / ~n_elem_vt * 2),
                bw_line_dist * 0.25 * np.sin(-2.1 / ~n_elem_vt * 2),
            ),
            polar_ax.c2p(
                bw_line_dist * 0.25 * np.cos(2.1 / ~n_elem_vt * 2),
                bw_line_dist * 0.25 * np.sin(2.1 / ~n_elem_vt * 2),
            ),
            color=ORANGE,
        )
        line_to_theta = ArcBetweenPoints(
            theta_arc.get_midpoint()
            + 0.1 * np.cos(-2.1 / ~n_elem_vt * 2) * RIGHT
            + 0.1 * np.sin(-2.1 / ~n_elem_vt * 2) * UP,
            theta_3db_label_1.get_corner(UL) + [-0.1, 0, 0],
            color=ORANGE,
            angle=-TAU / 8,
        )

        self.play(LaggedStart(Create(theta_arc), Create(line_to_theta), lag_ratio=0.3))

        self.wait(0.5)

        tri_mid = DashedLine(
            polar_ax.c2p(0, 0),
            tri_line.get_midpoint(),
            dash_length=DEFAULT_DASH_LENGTH,
            dashed_ratio=0.5,
        )

        right_bot = RightAngle(
            tri_line.copy().rotate(PI / 2), tri_line, quadrant=(-1, 1), length=0.2
        )
        right_top = RightAngle(
            tri_line.copy().rotate(PI / 2), tri_line, quadrant=(-1, -1), length=0.2
        )

        self.play(
            LaggedStart(
                Create(tri_mid),
                Create(right_top),
                Create(right_bot),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        top_tri_line = Line(tri_line.get_start(), tri_line.get_midpoint(), color=YELLOW)
        top_tri_label = MathTex("d").next_to(top_tri_line, RIGHT, SMALL_BUFF)

        self.play(Create(top_tri_line), Write(top_tri_label))

        self.wait(0.5)

        half_theta_arc = ArcBetweenPoints(
            polar_ax.c2p(
                bw_line_dist * 0.5 * np.cos(2.1 / ~n_elem_vt * 2),
                bw_line_dist * 0.5 * np.sin(2.1 / ~n_elem_vt * 2),
            ),
            polar_ax.c2p(bw_line_dist * 0.5, 0),
            color=GREEN,
            angle=-TAU / 4,
        )
        sine = (
            MathTex(
                r"\sin{\left(\frac{\theta}{2}\right)} = \frac{d}{2} \cdot \frac{1}{R}",
                font_size=DEFAULT_FONT_SIZE * 0.6,
            )
            .next_to(half_theta_arc, UP, LARGE_BUFF)
            .shift(LEFT * 2)
        )
        sine[0][4:7].set_color(GREEN)

        line_to_theta_half = ArcBetweenPoints(
            polar_ax.c2p(
                bw_line_dist * 0.5 * np.cos(2.1 / ~n_elem_vt * 2),
                bw_line_dist * 0.5 * np.sin(2.1 / ~n_elem_vt * 2),
            )
            + [-0.1, -0.2, 0],
            sine[0][4:7].get_bottom() + [0, -0.1, 0],
            color=GREEN,
            angle=-TAU / 8,
        )

        self.play(Create(half_theta_arc))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Write(sine[0][:8]),
                Create(line_to_theta_half),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Write(sine[0][8]),
                TransformFromCopy(top_tri_label[0], sine[0][9]),
                Write(sine[0][10:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                sine[0][9:12].animate.set_color(YELLOW),
                AnimationGroup(
                    sine[0][-1].animate.set_color(PURPLE),
                    bw_line_l_color_interp @ 1,
                ),
                lag_ratio=0.4,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Uncreate(line_to_theta), Uncreate(line_to_theta_half))

        self.wait(0.5)

        sine_rearr = (
            MathTex(
                r"d = 2 R \sin{\left(\frac{\theta}{2}\right)} \approx R \theta",
                font_size=DEFAULT_FONT_SIZE * 0.6,
            )
            .move_to(sine, LEFT)
            .shift(LEFT / 2)
        )
        sine_rearr[0][0].set_color(YELLOW)
        sine_rearr[0][3].set_color(PURPLE)
        sine_rearr[0][8].set_color(ORANGE)

        self.play(
            # TransformByGlyphMap(sine, sine_rearr),
            LaggedStart(
                ReplacementTransform(sine[0][9], sine_rearr[0][0], path_arc=PI),
                ReplacementTransform(sine[0][8], sine_rearr[0][1], path_arc=PI),
                ShrinkToCenter(sine[0][10]),
                ShrinkToCenter(sine[0][12:15]),
                ReplacementTransform(sine[0][11], sine_rearr[0][2], path_arc=PI),
                ReplacementTransform(sine[0][15], sine_rearr[0][3], path_arc=-PI),
                ReplacementTransform(sine[0][:8], sine_rearr[0][4:-3]),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(Write(sine_rearr[0][-3:]))

        self.wait(0.5)

        theta_qmark = Tex(r"$\theta = $ ?").move_to(sine_rearr)
        theta_qmark[0][0].set_color(ORANGE)

        self.next_section(skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(
                        theta_arc,
                        sine_rearr[0][:8],
                        sine_rearr[0][9:],
                        theta_3db_label_1,
                        half_theta_arc,
                        tri_mid,
                        top_tri_line,
                        tri_line,
                        right_bot,
                        right_top,
                        top_tri_label,
                    ),
                    bw_line_l_color_interp @ 0,
                    particle_opacity @ 1,
                ),
                ReplacementTransform(sine_rearr[0][8], theta_qmark[0][0]),
                Write(theta_qmark[0][1:]),
                lag_ratio=0.4,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)
        theta_f = (
            Tex(r"$\theta = f(\lambda, D) \sim \frac{\lambda}{D}$")
            .move_to(theta_qmark, LEFT)
            .shift(LEFT)
        )
        theta_f[0][0].set_color(ORANGE)

        self.play(
            ReplacementTransform(theta_qmark[0][:2], theta_f[0][:2]),
            ReplacementTransform(theta_qmark[0][2], theta_f[0][2:8]),
        )

        self.next_section(skip_animations=skip_animations(False))
        # self.remove(*theta_f[0], *theta_qmark[0])
        self.wait(0.5)

        self.play(n_elem_vt @ (~n_elem_vt * 2), run_time=3)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(theta_f[0][8]),
                TransformFromCopy(theta_f[0][5], theta_f[0][-1]),
                Create(theta_f[0][-2]),
                TransformFromCopy(theta_f[0][3], theta_f[0][-3]),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                theta_f[0][0]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 3),
                AnimationGroup(
                    theta_f[0][4]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(UP / 3),
                    theta_f[0][9]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(UP / 3),
                ),
                AnimationGroup(
                    theta_f[0][6]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(UP / 3),
                    theta_f[0][11]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(DOWN / 3),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


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

            cmap = get_cmap("coolwarm")
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

        # self.play(
        #     self.camera.frame.animate(
        #         run_time=0.5, rate_func=rate_functions.ease_in_sine
        #     ).shift(DOWN * config.frame_height)
        # )
        self.remove(
            car1,
            car2,
            vel2_arrow,
            vel2_dot,
            vel2_label,
            vel1_arrow,
            vel1_dot,
            *vel1_label[0],
            # rd_img,
            rd_ax,
            range_label,
            vel_label,
        )

        self.wait(0.5)

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
                FadeIn(sigproc),
                AnimationGroup(Write(sig_label), Write(proc_label)),
                lag_ratio=0.3,
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
            for sample_rect in sample_rects:
                sample_rect.set_fill(color=BLUE)

        self.next_section(skip_animations=skip_animations(True))
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
            # GrowArrow(down_arrow),
            LaggedStart(
                *[
                    AnimationGroup(
                        GrowFromCenter(phi), srs[0].animate.set_fill(color=YELLOW)
                    )
                    for phi, srs in zip(phis, sample_rects_all)
                ],
                lag_ratio=0.2,
            ),
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
            # FadeOut(down_arrow),
            *[srs[0].animate.set_fill(color=BLUE) for srs in sample_rects_all],
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

        f1 = VT(8.3)
        f2 = VT(15)

        np.random.seed(0)
        noise = np.random.normal(0, 0.3, t_new.size)
        window_new = signal.windows.kaiser(t_new.size, beta=3)

        def get_sig():
            sig = (
                np.sin(2 * PI * ~f1 * t_new)
                + 0.8 * np.sin(2 * PI * ~f2 * t_new)
                + noise
            ) * window_new
            return sig

        vel_opacity = VT(1)

        def get_fft_plot():
            sig = get_sig()
            X_k = 10 * np.log10(np.abs(fft(sig, fft_len)) / (t_new.size / 2)) + 30
            freq = np.linspace(-fs_new / 2, fs_new / 2, fft_len)
            f_X_k = interp1d(freq, np.clip(fftshift(X_k), 0, None))
            return vel_ax.plot(
                f_X_k,
                x_range=[0, 20, 20 / 400],
                color=ORANGE,
                stroke_opacity=~vel_opacity,
                # stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )

        vel_plot = always_redraw(get_fft_plot)
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

        self.next_section(skip_animations=skip_animations(False))
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
            vel_ax.animate.set_opacity(0.2),
            vel_ax_y_label.animate.set_opacity(0.2),
            vel_ax_x_label.animate.set_opacity(0.2),
            vel_opacity @ 0.2,
        )

        self.next_section(skip_animations=skip_animations(True))
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

        self.wait(0.5)

        prt_label = Tex("PRT").next_to(axes[0], LEFT)
        n_prt = Group(
            *[
                Tex(f"{idx if idx > 1 else ''}PRT", font_size=DEFAULT_FONT_SIZE * 0.5)
                .next_to(ax, LEFT)
                .set_x(prt_label.get_x())
                for idx, ax in enumerate(axes, start=1)
            ],
        )

        self.play(LaggedStart(FadeIn(*[prt for prt in n_prt], lag_ratio=0.2)))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        prt_eqn = MathTex(r"\text{PRT} = \frac{1}{\text{PRF}}").next_to(ts_eqn, RIGHT)

        self.play(
            LaggedStart(
                TransformFromCopy(n_prt[-1][1:], prt_eqn[0][:3], path_arc=PI / 3),
                *[GrowFromCenter(m) for m in prt_eqn[0][3:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.remove(sigproc_right_box)
        fs_ge_prf = (
            MathTex(r"f_s \gg \text{PRF}", r"\rightarrow", r"T_s \ll \text{PRT}")
            .next_to(ts_eqn[0][0], DOWN, LARGE_BUFF, LEFT)
            .set_z_index(6)
        )

        vel_plot_group = VGroup(vel_ax, vel_ax_x_label, vel_ax_y_label)
        self.play(
            LaggedStart(
                vel_plot_group.animate.shift(DOWN),
                TransformFromCopy(ts_eqn[0][-2:], fs_ge_prf[0][:2], path_arc=-PI / 2),
                GrowFromCenter(fs_ge_prf[0][2]),
                TransformFromCopy(prt_eqn[0][-3:], fs_ge_prf[0][-3:], path_arc=PI / 2),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(fs_ge_prf[1]),
                TransformFromCopy(ts_eqn[0][:2], fs_ge_prf[2][:2], path_arc=PI / 2),
                GrowFromCenter(fs_ge_prf[2][2]),
                TransformFromCopy(prt_eqn[0][:3], fs_ge_prf[2][-3:], path_arc=-PI / 2),
                lag_ratio=0.3,
            )
        )
        # self.add(fs_ge_prf[2])

        self.wait(0.5)

        fast_time_label = Text(
            "Fast Time", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.4
        ).next_to(axes, UP)
        slow_time_label = (
            Text("Slow Time", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.4)
            .rotate(PI / 2)
            .next_to(axes, LEFT)
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in n_ts], lag_ratio=0.1),
                Write(fast_time_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in n_prt], lag_ratio=0.1),
                Write(slow_time_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.remove(radar.vgroup)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(vel_plot_group.width * 2)
            .move_to(vel_plot_group)
            .shift(UP / 2),
            vel_opacity @ 1,
            vel_ax.animate.set_opacity(1),
            vel_ax_y_label.animate.set_opacity(1),
            vel_ax_x_label.animate.set_opacity(1),
            Group(
                axes, sample_rects_all, fast_time_label, slow_time_label
            ).animate.shift(LEFT * 2),
            Group(fs_ge_prf, ts_eqn, prt_eqn).animate.shift(UP),
        )

        self.wait(0.5)

        M_disp = 8
        vel_sample_rects = vel_ax.get_riemann_rectangles(
            vel_plot,
            input_sample_type="center",
            x_range=[0, 20],
            dx=20 / M_disp,
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.4,
        ).set_z_index(1)

        self.play(LaggedStart(*[FadeIn(m) for m in vel_sample_rects], lag_ratio=0.1))

        self.wait(0.5)

        target1_label = (
            Text(
                "Target 1",
                color=TARGET1_COLOR,
                font="Maple Mono",
                font_size=DEFAULT_FONT_SIZE * 0.4,
            )
            .next_to(vel_ax.c2p(9), UP, LARGE_BUFF * 4)
            .shift(LEFT * 2)
        )
        target2_label = (
            Text(
                "Target 2",
                color=TARGET2_COLOR,
                font="Maple Mono",
                font_size=DEFAULT_FONT_SIZE * 0.4,
            )
            .next_to(vel_ax.c2p(10), UP, LARGE_BUFF * 4)
            .shift(RIGHT * 2)
        )

        def create_target_bez(label: Text, x, side, color, offset=0):
            def updater():
                vel_plot_temp = get_fft_plot()
                bez = CubicBezier(
                    label.get_edge_center(side) + [side[0] * 0.1, 0, 0],
                    label.get_edge_center(side) + [side[0] * 0.5, 0, 0],
                    vel_ax.input_to_graph_point(~x + offset, vel_plot_temp) + [0, 1, 0],
                    vel_ax.input_to_graph_point(~x + offset, vel_plot_temp)
                    + [0, 0.1, 0],
                    color=color,
                    stroke_width=DEFAULT_STROKE_WIDTH * 0.6,
                )
                return bez

            return updater

        new_vel = 9.3
        target1_bez = always_redraw(
            create_target_bez(target1_label, f1, RIGHT, TARGET1_COLOR)
        )
        target2_bez = always_redraw(
            create_target_bez(target2_label, f2, LEFT, TARGET2_COLOR, 0.2)
        )

        target1_bez_new = CubicBezier(
            target1_label.get_edge_center(RIGHT) + [0.1, 0, 0],
            target1_label.get_edge_center(RIGHT) + [0.5, 0, 0],
            vel_sample_rects[3].get_top() + [0, 1, 0],
            vel_sample_rects[3].get_top() + [0, 0.1, 0],
            color=TARGET1_COLOR,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.6,
        )
        target2_bez_new = CubicBezier(
            target2_label.get_edge_center(LEFT) + [-0.1, 0, 0],
            target2_label.get_edge_center(LEFT) + [-0.5, 0, 0],
            vel_sample_rects[3].get_top() + [0, 1, 0],
            vel_sample_rects[3].get_top() + [0, 0.1, 0],
            color=TARGET2_COLOR,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.6,
        )

        self.play(Write(target1_label), FadeIn(target1_bez))

        self.wait(0.5)

        self.play(Write(target2_label), FadeIn(target2_bez))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(f2 @ (new_vel - 0.2), run_time=2)

        self.wait(0.5)

        self.play(
            LaggedStart(
                vel_opacity @ 0,
                AnimationGroup(
                    ReplacementTransform(target1_bez, target1_bez_new),
                    ReplacementTransform(target2_bez, target2_bez_new),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(target1_label, target2_label, target1_bez_new, target2_bez_new),
            vel_opacity @ 1,
        )

        self.wait(0.5)

        M_disp_init = 3
        ms = [
            *[
                MathTex(f"{n}", font_size=DEFAULT_FONT_SIZE).next_to(
                    vel_sample_rects[n - 1], UP
                )
                for n in range(1, M_disp_init + 2)
            ],
            MathTex(r"\cdots").next_to(vel_sample_rects[5], UP),
            MathTex("M").next_to(vel_sample_rects[-1], UP),
        ]
        ms = Group(
            *[m.set_y(sorted(ms, key=lambda x: x.get_y())[-1].get_y()) for m in ms]
        )

        self.play(LaggedStart(*[FadeIn(m) for m in ms], lag_ratio=0.1))

        self.wait(0.5)

        extent_line = Line(
            [vel_sample_rects[0].get_corner(UL)[0], vel_ax.get_corner(UL)[1], 0],
            [vel_sample_rects[-1].get_corner(UR)[0], vel_ax.get_corner(UL)[1], 0],
        )
        extent_line_l = Line(
            extent_line.get_start() + DOWN / 8,
            extent_line.get_start() + UP / 8,
        )
        extent_line_r = Line(
            extent_line.get_end() + DOWN / 8,
            extent_line.get_end() + UP / 8,
        )

        extent_label = Tex("extent").next_to(extent_line_r, RIGHT)

        self.play(
            LaggedStart(
                ms.animate.shift(UP / 2),
                Create(extent_line_l),
                Create(extent_line),
                Create(extent_line_r),
                Write(extent_label),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        extent_over_m = (
            MathTex(r"\Delta f_d = \frac{\text{extent}}{M}")
            .move_to(extent_label, LEFT)
            .shift(DOWN)
        )

        self.play(
            self.camera.frame.animate.shift(RIGHT * 1.5),
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(extent_label[0], extent_over_m[0][4:10]),
                ),
                Create(extent_over_m[0][-2]),
                AnimationGroup(
                    ReplacementTransform(ms[-1][0], extent_over_m[0][-1], path_arc=PI),
                    FadeOut(ms[:-1]),
                ),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in extent_over_m[0][:4][::-1]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        doppler_eqn = MathTex(r"v = \frac{f_d \lambda}{2}").next_to(
            extent_over_m, DOWN, MED_SMALL_BUFF, LEFT
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    extent_over_m[0][1:3], doppler_eqn[0][2:4], path_arc=PI / 2
                ),
                *[
                    GrowFromCenter(m)
                    for m in [*doppler_eqn[0][4:], *doppler_eqn[0][:2]]
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        extent_l = MathTex(
            r"\frac{-\text{PRF}}{2}", font_size=DEFAULT_FONT_SIZE * 0.6
        ).next_to(extent_line_l, UP, SMALL_BUFF)
        extent_r = MathTex(
            r"\frac{\text{PRF}}{2}", font_size=DEFAULT_FONT_SIZE * 0.6
        ).next_to(extent_line_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                LaggedStart(*[GrowFromCenter(m) for m in extent_l[0]], lag_ratio=0.1),
                LaggedStart(*[GrowFromCenter(m) for m in extent_r[0]], lag_ratio=0.1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        extent_over_m_pli = (
            MathTex(
                r"\Delta f_d = \frac{\frac{\text{PRF}}{2} - \frac{-\text{PRF}}{2}}{M}"
            )
            .move_to(extent_over_m, LEFT)
            .shift(UP / 2 + LEFT / 3)
        )

        extent_over_m_pli_simp = MathTex(r"\Delta f_d = \frac{\text{PRF}}{M}").move_to(
            extent_over_m_pli, LEFT
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(RIGHT * 0.5),
                FadeOut(extent_line, extent_line_l, extent_line_r),
                # TransformByGlyphMap(extent_over_m.shift(UP), extent_over_m_pli),
                ReplacementTransform(extent_over_m[0][:4], extent_over_m_pli[0][:4]),
                ReplacementTransform(extent_over_m[0][10:], extent_over_m_pli[0][16:]),
                ShrinkToCenter(extent_over_m[0][4:10]),
                ReplacementTransform(
                    extent_r[0], extent_over_m_pli[0][4:9], path_arc=-PI / 3
                ),
                GrowFromCenter(extent_over_m_pli[0][9]),
                ReplacementTransform(
                    extent_l[0], extent_over_m_pli[0][10:16], path_arc=-PI / 2
                ),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(
                    extent_over_m_pli[0][:4], extent_over_m_pli_simp[0][:4]
                ),
                ReplacementTransform(
                    extent_over_m_pli[0][4:7], extent_over_m_pli_simp[0][4:7]
                ),
                ShrinkToCenter(extent_over_m_pli[0][7:9]),
                ShrinkToCenter(extent_over_m_pli[0][9]),
                ShrinkToCenter(extent_over_m_pli[0][10:16]),
                ReplacementTransform(
                    extent_over_m_pli[0][16], extent_over_m_pli_simp[0][7]
                ),
                ReplacementTransform(
                    extent_over_m_pli[0][17], extent_over_m_pli_simp[0][8]
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        vel_res_eqn = (
            MathTex(r"\Delta v = \frac{\text{PRF} \lambda}{2 M}")
            .next_to(extent_over_m_pli_simp, DOWN, MED_SMALL_BUFF, LEFT)
            .shift(RIGHT / 2)
        )

        self.play(
            # TransformByGlyphMap(extent_over_m_pli_simp.shift(UP / 2), vel_res_eqn),
            LaggedStart(
                ReplacementTransform(extent_over_m_pli_simp[0][0], vel_res_eqn[0][0]),
                ShrinkToCenter(extent_over_m_pli_simp[0][1:3]),
                ShrinkToCenter(extent_over_m_pli_simp[0][3]),
                ShrinkToCenter(extent_over_m_pli_simp[0][7]),
                ReplacementTransform(doppler_eqn[0][0], vel_res_eqn[0][1]),
                ReplacementTransform(doppler_eqn[0][1], vel_res_eqn[0][2]),
                ReplacementTransform(
                    extent_over_m_pli_simp[0][4:7], vel_res_eqn[0][3:6]
                ),
                ShrinkToCenter(doppler_eqn[0][2:4]),
                ReplacementTransform(doppler_eqn[0][4], vel_res_eqn[0][6]),
                ReplacementTransform(doppler_eqn[0][5], vel_res_eqn[0][7]),
                ReplacementTransform(doppler_eqn[0][6], vel_res_eqn[0][8]),
                ReplacementTransform(
                    extent_over_m_pli_simp[0][8], vel_res_eqn[0][9], path_arc=-PI
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            vel_res_eqn[0][:2]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )
        self.wait(0.5)

        self.play(
            vel_res_eqn[0][6]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )
        self.wait(0.5)

        self.play(
            vel_res_eqn[0][3:6]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )
        self.wait(0.5)

        self.play(
            vel_res_eqn[0][8]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )
        self.wait(0.5)

        self.play(
            vel_res_eqn[0][9]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        rres_eqn = MathTex(
            r"\Delta R = \frac{c \tau}{2} \approx \frac{c}{2 B}",
            color=RED,
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        ares_eqn = MathTex(
            r"\Delta \theta \sim \frac{\lambda}{D}",
            color=BLUE,
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )

        eqn_group = Group(rres_eqn, ares_eqn, vel_res_eqn.copy().scale(1.5)).arrange(
            RIGHT, LARGE_BUFF
        )

        rres_label = Text(
            "Range", font_size=DEFAULT_FONT_SIZE * 1, font="Maple Mono", color=RED
        ).next_to(rres_eqn, UP, LARGE_BUFF)
        rres_group = Group(rres_label, rres_eqn)
        ares_label = (
            Text(
                "Angular",
                font_size=DEFAULT_FONT_SIZE * 1,
                font="Maple Mono",
                color=BLUE,
            ).next_to(ares_eqn, UP, LARGE_BUFF)
            # .set_y(rres_label.get_y())
        )
        ares_group = Group(ares_label, ares_eqn)
        vres_label = (
            Text(
                "Velocity",
                font_size=DEFAULT_FONT_SIZE * 1,
                font="Maple Mono",
                color=ORANGE,
            ).next_to(eqn_group[-1], UP, LARGE_BUFF)
            # .set_y(rres_label.get_y())
        )
        vres_group = Group(vres_label, eqn_group[-1])

        eqn_group_w_label = (
            Group(rres_group, ares_group, vres_group)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(self.camera.frame, DOWN, LARGE_BUFF * 4)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.add(rres_eqn, ares_eqn, rres_label, ares_label)

        self.play(
            LaggedStart(
                vel_res_eqn.animate.scale(1.5).move_to(vres_group[1]).set_color(ORANGE),
                self.camera.frame.animate.scale_to_fit_width(
                    eqn_group_w_label.width * 1.3
                ).move_to(eqn_group_w_label),
                Write(vres_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.wait(2)


class TradeOff(MovingCameraScene):
    def construct(self):
        rres_eqn = MathTex(
            r"\Delta R = \frac{c \tau}{2} \approx \frac{c}{2 B}",
            color=RED,
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        ares_eqn = MathTex(
            r"\Delta \theta \sim \frac{\lambda}{D}",
            color=BLUE,
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        vres_eqn = MathTex(
            r"\Delta v = \frac{\text{PRF} \lambda}{2 M}",
            color=ORANGE,
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )

        eqn_group = Group(rres_eqn, ares_eqn, vres_eqn).arrange(RIGHT, LARGE_BUFF)

        rres_label = Text(
            "Range", font_size=DEFAULT_FONT_SIZE * 1, font="Maple Mono", color=RED
        ).next_to(rres_eqn, UP, LARGE_BUFF)
        rres_group = Group(rres_label, rres_eqn)
        ares_label = (
            Text(
                "Angular",
                font_size=DEFAULT_FONT_SIZE * 1,
                font="Maple Mono",
                color=BLUE,
            ).next_to(ares_eqn, UP, LARGE_BUFF)
            # .set_y(rres_label.get_y())
        )
        ares_group = Group(ares_label, ares_eqn)
        vres_label = (
            Text(
                "Velocity",
                font_size=DEFAULT_FONT_SIZE * 1,
                font="Maple Mono",
                color=ORANGE,
            ).next_to(eqn_group[-1], UP, LARGE_BUFF)
            # .set_y(rres_label.get_y())
        )
        vres_group = Group(vres_label, eqn_group[-1])

        eqn_group_w_label = (
            Group(rres_group, ares_group, vres_group)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(self.camera.frame, DOWN, LARGE_BUFF * 4)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.add(rres_eqn, ares_eqn, rres_label, ares_label, vres_label, vres_eqn)
        self.camera.frame.scale_to_fit_width(eqn_group_w_label.width * 1.3).move_to(
            eqn_group_w_label
        )

        self.wait(0.5)

        self.play(
            ares_label.animate.set_opacity(0.3).shift(UP * 2),
            ares_eqn.animate.set_opacity(0.3).shift(UP * 2),
            Group(rres_group, vres_group)
            .animate.arrange(RIGHT, LARGE_BUFF * 2)
            .move_to(self.camera.frame)
            .shift(DOWN * 2),
        )

        self.wait(0.5)

        M = VT(40)
        N = VT(10_000)

        t = np.arange(0, max_time, 1 / fs)

        rmax = c * Tc * fs / (2 * bw)

        target1_pos = VT(17)
        target1_vel = VT(0)
        target1_pow = VT(0)
        target2_pos = VT(20)
        target2_vel = VT(5)
        target2_pow = VT(0)

        np.random.seed(0)

        M_max = 100
        N_max = 10_000

        rd_height = VT(config.frame_width * 0.4)

        def plot_rd():
            N_curr = int(~N)
            M_curr = int(~M)
            t_nonpad = np.linspace(0, max_time * (N_curr / N_max), N_curr)
            noise = np.random.normal(0, 0.1, N_curr)
            targets = [
                (~target1_pos, ~target1_vel, ~target1_pow),
                (~target2_pos, ~target2_vel, ~target2_pow),
            ]
            window_2d = np.outer(
                signal.windows.blackman(M_curr, 3), signal.windows.blackman(N_curr, 3)
            )
            cpi = (
                np.array(
                    [
                        (
                            np.sum(
                                [
                                    np.sin(
                                        2 * PI * compute_f_beat(r) * t_nonpad
                                        + m * compute_phase_diff(v)
                                    )
                                    * db_to_lin(p)
                                    for r, v, p in targets
                                ],
                                axis=0,
                            )
                            + noise
                        )
                        for m in range(M_curr)
                    ]
                )
                * window_2d
            )

            cpi = pad2d(cpi, (M_max, N_max))

            ranges_n = np.linspace(-rmax / 2, rmax / 2, N_max)
            range_doppler = fftshift(np.abs(fft2(cpi.T, s=[N_max, M_max]))) / (
                N_curr / 2
            )
            range_doppler = range_doppler[(ranges_n >= 0) & (ranges_n <= 40), :]
            range_doppler -= range_doppler.min()
            range_doppler /= range_doppler.max()

            cmap = get_cmap("coolwarm")
            range_doppler_fmt = np.uint8(cmap(10 * np.log10(range_doppler + 1)) * 255)
            range_doppler_fmt[range_doppler < 0.02] = [0, 0, 0, 0]

            rd_img = (
                ImageMobject(range_doppler_fmt, image_mode="RGBA")
                .stretch_to_fit_width(config.frame_width * 0.4)
                .stretch_to_fit_height(config.frame_width * 0.4)
                .next_to(self.camera.frame.get_right(), LEFT, LARGE_BUFF)
            )
            bot = rd_img.get_bottom()
            rd_img.stretch_to_fit_height(~rd_height).next_to(bot, UP, 0)
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

        rd_group = Group(rd_ax, range_label, vel_label, rd_img)
        self.add(
            Group(rd_ax, range_label, vel_label).shift(DOWN * self.camera.frame.height)
        )

        self.play(
            Group(rres_group, vres_group)
            .animate.scale(0.7)
            .arrange(DOWN, LARGE_BUFF)
            .next_to(range_label, LEFT, LARGE_BUFF * 5),
            self.camera.frame.animate.shift(DOWN * self.camera.frame.height),
        )
        self.play(FadeIn(rd_img))

        self.wait(0.5)

        self.play(
            rres_group[0].animate.set_opacity(0.3),
            rres_group[1].animate.set_opacity(0.3),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        M_val = (
            MathTex(f"M = {int(~M)}")
            .move_to(self.camera.frame.get_center(), LEFT)
            .shift(DOWN + LEFT)
        )
        M_val[0][0].set_color(ORANGE)

        self.play(TransformFromCopy(vres_eqn[0][-1], M_val[0][0], path_arc=PI / 3))
        self.play(FadeIn(M_val[0][1:]))

        def update_M_val(m):
            new = (
                MathTex(f"M = {int(~M)}")
                .move_to(self.camera.frame.get_center(), LEFT)
                .shift(DOWN + LEFT)
            )
            new[0][0].set_color(ORANGE)
            m.become(new)

        self.add(M_val)
        M_val.add_updater(update_M_val)

        self.wait(0.5)

        self.play(M @ 10, run_time=3)

        self.wait(0.5)

        self.play(M @ 40, run_time=3)

        self.wait(0.5)

        self.play(M @ 80, run_time=3)

        cpi_eqn = (
            Tex(r"CPI $ = M \cdot $ PRT $\left[s\right]$")
            .next_to(M_val, DOWN, LARGE_BUFF)
            .shift(LEFT / 2)
        )
        cpi_eqn[0][4].set_color(ORANGE)

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in cpi_eqn[0][:4]],
                TransformFromCopy(M_val[0][0], cpi_eqn[0][4], path_arc=PI / 3),
                *[GrowFromCenter(m) for m in cpi_eqn[0][5:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        # self.play(N @ 4000, run_time=3)

        self.wait(0.5)

        theoretical = Text(
            "Theoretical",
            font="Maple Mono",
            font_size=DEFAULT_FONT_SIZE * 0.8,
            color=RED,
        ).next_to(rres_label, UP)

        self.play(
            Write(theoretical),
            rres_label.animate.set_opacity(1),
            rres_eqn.animate.set_opacity(1),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                rres_eqn[0][-1]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(DOWN / 3)
                .set_color(YELLOW),
                rres_eqn[0][4]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP)
                .set_color(YELLOW),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        y_scale = 1.4
        rd_ax_new = Axes(
            x_range=[-0.5, 10, 2],
            y_range=[-0.5, 10 * y_scale, 2],
            tips=False,
            x_length=rd_img.width,
            y_length=rd_img.height * y_scale,
            axis_config=dict(stroke_width=DEFAULT_STROKE_WIDTH * 1.2),
        )
        rd_ax_new.shift(rd_img.get_corner(DL) - rd_ax_new.c2p(0, 0))

        self.next_section(skip_animations=skip_animations(False))
        rd_ax.save_state()
        rd_height_init = ~rd_height
        self.play(
            rd_height @ (~rd_height * y_scale),
            Transform(rd_ax, rd_ax_new),
            run_time=3,
        )

        self.wait(0.5)

        N_label = Tex(r"$N = $ Constant").next_to(M_val, UP, LARGE_BUFF, RIGHT)
        N_label[0][0].set_color(RED)

        self.play(LaggedStart(*[GrowFromCenter(m) for m in N_label[0]], lag_ratio=0.1))

        self.wait(0.5)

        self.play(rd_ax.animate.restore(), rd_height @ rd_height_init)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            rres_eqn.animate(rate_func=rate_functions.there_and_back).set_color(YELLOW),
            run_time=2,
        )

        self.wait(0.5)

        grid = NumberPlane(
            x_range=[-0.5, 10, 1],
            y_range=[-0.5, 10, 1],
            x_length=rd_img.width,
            y_length=rd_img.height,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            },
        )
        grid.shift(rd_ax.c2p(0, 0) - grid.c2p(0, 0))

        rd_ax_new = Axes(
            x_range=[-0.5, 10, 2],
            y_range=[-0.5, 10, 2],
            tips=False,
            x_length=rd_img.width,
            y_length=rd_img.height * y_scale,
            axis_config=dict(stroke_width=DEFAULT_STROKE_WIDTH * 1.2),
        )
        rd_ax_new.shift(rd_img.get_corner(DL) - rd_ax_new.c2p(0, 0))

        self.play(Create(grid))

        self.wait(0.5)

        grid_new = NumberPlane(
            x_range=[-0.5, 10, 1],
            y_range=[-0.5, 10, 1],
            x_length=rd_img.width,
            y_length=rd_img.height,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            },
        ).stretch(y_scale, 1)
        grid_new.shift(rd_ax.c2p(0, 0) - grid_new.c2p(0, 0))

        self.play(Transform(rd_ax, rd_ax_new), Transform(grid, grid_new))

        # self.play(M @ M_max, run_time=3)

        # self.wait(0.5)

        # self.wait(0.5)

        # self.play(N @ N_max, run_time=3)

        self.wait(0.5)

        ares_group.scale(0.7)
        ares_group[0].set_opacity(1)
        ares_group[1].set_opacity(1)

        self.play(
            LaggedStart(
                FadeOut(rd_img, M_val),
                AnimationGroup(
                    eqn_group_w_label.animate.arrange(RIGHT, LARGE_BUFF).move_to(
                        ares_eqn
                    ),
                    self.camera.frame.animate.move_to(ares_eqn),
                ),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back).shift(UP)
                    for m in eqn_group_w_label
                ],
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ShrinkToCenter(m) for m in eqn_group_w_label],
                lag_ratio=0.4,
            )
        )

        self.wait(2)


class WrapUp(MovingCameraScene):
    def construct(self):
        notebook_scrolling = VideoMobject(
            "./static/notebook_scrolling.mp4"
        ).scale_to_fit_width(config.frame_width * 0.6)

        nb_title = Text(
            "The Interactive Radar Cheatsheet",
            font="Maple Mono",
            font_size=DEFAULT_FONT_SIZE,
        ).next_to(notebook_scrolling, UP)

        self.play(
            notebook_scrolling.shift(DOWN * 10).animate.shift(UP * 10),
            nb_title.shift(UP * 5).animate.shift(DOWN * 5),
        )

        self.wait(10)

        self.play(
            notebook_scrolling.animate.shift(DOWN * 10),
            nb_title.animate.shift(UP * 5),
        )

        self.wait(0.5)

        website_url = Text(
            "marshallbruner.com", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 1.5
        )

        self.play(Write(website_url))

        self.wait(0.5)

        phased_array_resource = (
            ImageMobject("./static/phased_array_resource.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .next_to(self.camera.frame, UP)
        )
        rd_resource = (
            ImageMobject("./static/range_doppler_resource.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .next_to(self.camera.frame, DOWN)
        )
        self.play(
            LaggedStart(
                FadeOut(website_url),
                Group(phased_array_resource, rd_resource).animate.arrange(
                    DOWN, MED_LARGE_BUFF
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

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

        self.play(
            LaggedStart(
                AnimationGroup(
                    rd_resource.animate.shift(DOWN * 10),
                    phased_array_resource.animate.shift(DOWN * 10),
                ),
                LaggedStart(
                    GrowFromCenter(osc),
                    GrowFromCenter(kraken),
                    GrowFromCenter(weather),
                    GrowFromCenter(eqn),
                    lag_ratio=0.3,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        tier1 = (
            ImageMobject("../08_beamforming/static/tier1.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(LEFT * 10)
        )
        tier2 = (
            ImageMobject("../08_beamforming/static/tier2.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(RIGHT * 10)
        )
        tier3 = (
            ImageMobject("../08_beamforming/static/tier3.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(RIGHT * 10)
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(0.5)

        self.play(tier1.animate.move_to(ORIGIN))

        self.wait(0.5)

        self.play(Group(tier1, tier2).animate.arrange(RIGHT, LARGE_BUFF))

        self.wait(0.5)

        self.play(Group(tier1, tier2, tier3).animate.arrange(RIGHT, LARGE_BUFF))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ShrinkToCenter(m) for m in Group(tier1, tier2, tier3)],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        shoutout = Text("Huge thanks to:", font="Maple Mono").to_edge(UP, LARGE_BUFF)
        people = [
            "ZacJW",
            "Dag-Vidar Bauer",
            "db-isJustARatio",
            "Jea99",
            "Leon",
            "dplynch",
            "Kris",
        ]
        people_text = (
            Group(
                *[
                    Text(p, font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.6)
                    for p in people
                ]
            )
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(shoutout, DOWN)
        )

        self.play(
            LaggedStart(*[Write(m) for m in [shoutout, *people_text]], lag_ratio=0.2)
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class EndScreen(Scene):
    def construct(self):
        stats_title = Tex("Stats for Nerds")
        stats_table = (
            Table(
                [
                    ["Lines of code", "4,812"],
                    ["Script word count", "2,370"],
                    ["Days to make", "18"],
                    ["Git commits", "14"],
                ]
            )
            .scale(0.5)
            .next_to(stats_title, direction=DOWN, buff=MED_LARGE_BUFF)
        )
        for row in stats_table.get_rows():
            row[1].set_color(GREEN)

        stats_group = (
            VGroup(stats_title, stats_table)
            .move_to(ORIGIN)
            .to_edge(RIGHT, buff=LARGE_BUFF)
        )

        thank_you_sabrina = (
            Text(
                "Thank you, Sabrina, for\nediting the whole video :)",
                font="Maple Mono",
                font_size=DEFAULT_FONT_SIZE * 0.5,
            )
            .next_to(stats_group, DOWN)
            .to_edge(DOWN)
        )

        marshall_bruner = Text(
            "Marshall Bruner", font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.5
        ).next_to([-config["frame_width"] / 4, 0, 0], DOWN, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                FadeIn(marshall_bruner, shift=UP),
                AnimationGroup(FadeIn(stats_title, shift=DOWN), FadeIn(stats_table)),
                Create(thank_you_sabrina),
                lag_ratio=0.9,
                run_time=4,
            )
        )

        self.wait(2)


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
