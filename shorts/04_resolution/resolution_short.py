# resolution_short.py


import sys

import numpy as np
from manim import *
from matplotlib import colormaps
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fft2, fftshift
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d

sys.path.insert(0, "../../")

from props import VideoMobject, WeatherRadarTower
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True
config.pixel_height = 1920
config.pixel_width = 1080
config.frame_height = 14
config.frame_width = 9


FONT = "Maple Mono CN"
TARGET1_COLOR = GREEN
TARGET2_COLOR = ORANGE


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


def db_to_lin(x):
    return 10 ** (x / 10)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(fh(self, 0.3)).next_to(
            self.camera.frame.get_left(), RIGHT, MED_LARGE_BUFF
        ).shift(DOWN * 3)

        cloud = (
            SVGMobject("../../props/static/cloud.svg")
            .scale_to_fit_width(fw(self, 0.4))
            .set_fill(WHITE)
            .set_color(WHITE)
            .set_stroke(width=DEFAULT_STROKE_WIDTH * 3)
            .next_to(self.camera.frame.get_right(), LEFT, LARGE_BUFF)
            .shift(UP * 3)
        )

        plane = (
            SVGMobject("../../props/static/plane.svg")
            .set_fill(WHITE)
            .scale_to_fit_width(fw(self, 0.3))
            .rotate(135 * DEGREES)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(RIGHT)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(radar.left_leg),
                    Create(radar.middle_leg),
                    Create(radar.right_leg),
                ),
                AnimationGroup(
                    Create(radar.conn1),
                    Create(radar.conn2),
                    Create(radar.conn3),
                    Create(radar.conn4),
                ),
                Transform(
                    radar.radome.copy().shift(LEFT * 4),
                    radar.radome,
                    path_arc=-PI,
                    run_time=1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                plane.shift(RIGHT * 5).animate.shift(LEFT * 5),
                cloud.shift(RIGHT * 5).animate.shift(LEFT * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        to_targets = Arrow(
            radar.radome.get_right() + UP / 2,
            cloud.get_bottom() + LEFT + DOWN,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            color=TX_COLOR,
            buff=SMALL_BUFF,
        )

        range_vt = VT(1)
        r = always_redraw(
            lambda: MathTex(f"R = {int(~range_vt)} \\text{{km}}")
            .scale(1.5)
            .next_to(to_targets.get_center(), LEFT, LARGE_BUFF * 1.5, LEFT)
            .shift(UP)
        )

        self.play(Create(to_targets), FadeIn(r))
        self.play(range_vt @ 300, run_time=2)

        self.wait(0.5)

        self.play(FadeOut(r))

        self.wait(0.5)

        r_min = -60

        x_len = to_targets.get_length() * 2.2
        polar_ax = Axes(
            x_range=[r_min, -r_min, r_min / 8],
            y_range=[r_min, -r_min, r_min / 8],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=x_len,
        ).set_opacity(0)
        polar_ax.shift(to_targets.get_start() - polar_ax.c2p(0, 0)).rotate(
            to_targets.get_angle()
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

        theta_min = VT(0)
        theta_max = VT(0)

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

        x = (
            Text("x", color=RED, font=FONT)
            .scale(3)
            .move_to(to_targets.get_center())
            .set_z_index(1)
        )

        self.play(GrowFromCenter(x))

        self.wait(0.5)

        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        self.play(
            LaggedStart(
                AnimationGroup(ShrinkToCenter(x), FadeOut(to_targets)),
                AnimationGroup(theta_min @ (-PI / 2), theta_max @ (PI / 2)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        horn = (
            ImageMobject("../../props/static/horn_antenna.png")
            .scale_to_fit_width(fw(self, 0.3))
            .next_to(radar.vgroup, DOWN, LARGE_BUFF)
        )
        dish = (
            ImageMobject("../../props/static/dish_antenna.png")
            .scale_to_fit_width(fw(self, 0.3))
            .next_to(radar.vgroup, DOWN, LARGE_BUFF)
        )
        patches = (
            ImageMobject("../../props/static/patch_array.png")
            .scale_to_fit_width(fw(self, 0.3))
            .next_to(radar.vgroup, DOWN, LARGE_BUFF)
        )
        antennas = (
            Group(horn, dish, patches)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .move_to(self.camera.frame.get_bottom())
        )

        antennas_bez_l = CubicBezier(
            AF_polar_plot.get_bottom() + [1, 0.5, 0],
            AF_polar_plot.get_bottom() + [1, -3, 0],
            antennas.get_corner(UL) + [0, 1, 0],
            antennas.get_corner(UL) + [0, 0.1, 0],
        )
        antennas_bez_r = CubicBezier(
            AF_polar_plot.get_bottom() + [1, 0.5, 0],
            AF_polar_plot.get_bottom() + [1, -3, 0],
            antennas.get_corner(UR) + [0, 1, 0],
            antennas.get_corner(UR) + [0, 0.1, 0],
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.set_y(
                    Group(antennas, radar.vgroup, cloud, plane).get_y()
                ),
                AnimationGroup(Create(antennas_bez_l), Create(antennas_bez_r)),
                GrowFromCenter(horn),
                GrowFromCenter(dish),
                GrowFromCenter(patches),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShrinkToCenter(patches),
                ShrinkToCenter(dish),
                ShrinkToCenter(horn),
                AnimationGroup(Uncreate(antennas_bez_l), Uncreate(antennas_bez_r)),
                self.camera.frame.animate.restore(),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        bw_line_l = Line(
            polar_ax.c2p(0, 0),
            polar_ax.c2p(-r_min * 1.5, 0),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            color=RED,
        ).rotate(PI / 6, about_point=polar_ax.c2p(0, 0))
        bw_line_r = Line(
            polar_ax.c2p(0, 0),
            polar_ax.c2p(-r_min * 1.5, 0),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            color=RED,
        ).rotate(-PI / 6, about_point=polar_ax.c2p(0, 0))

        self.play(
            LaggedStart(
                AnimationGroup(Create(bw_line_l), Create(bw_line_r)),
                AnimationGroup(
                    cloud.animate.set_color(RED), plane.animate.set_color(RED)
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                cloud.animate.move_to(polar_ax.c2p(-r_min * 1.5, 0)),
                plane.animate.move_to(polar_ax.c2p(-r_min * 1.5, 0)),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        new_cloud = cloud.copy().set_color(WHITE).shift(UP)
        new_plane = plane.copy().set_color(WHITE).shift(LEFT + DOWN * 2)
        to_plane = Arrow(
            radar.radome.get_right() + UP / 2,
            new_plane.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            color=TX_COLOR,
            buff=SMALL_BUFF,
        )
        to_cloud = Arrow(
            radar.radome.get_right() + UP / 2,
            new_cloud.get_corner(DL),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            color=TX_COLOR,
            buff=SMALL_BUFF,
        )

        r1 = MathTex(r"R_1").scale(1.5).next_to(to_cloud.get_midpoint(), UL)
        r2 = MathTex(r"R_2").scale(1.5).next_to(to_plane.get_midpoint(), DR)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    theta_min @ 0,
                    theta_max @ 0,
                    Uncreate(bw_line_l),
                    Uncreate(bw_line_r),
                    Transform(cloud, new_cloud),
                    Transform(plane, new_plane),
                ),
                GrowArrow(to_plane),
                GrowArrow(to_cloud),
                GrowFromCenter(r1),
                GrowFromCenter(r2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        delta_r_qmark = (
            MathTex(r"\Delta R = R_1 - R_2 \ge ?")
            .scale(2)
            .next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF * 2)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeIn(delta_r_qmark[0][:2]),
                FadeIn(delta_r_qmark[0][2]),
                ReplacementTransform(r1[0], delta_r_qmark[0][3:5], path_arc=PI / 3),
                FadeIn(delta_r_qmark[0][5]),
                ReplacementTransform(r2[0], delta_r_qmark[0][6:8], path_arc=PI / 3),
                FadeIn(delta_r_qmark[0][8]),
                FadeIn(delta_r_qmark[0][9]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=to_targets.get_length(),
                y_length=radar.radome.height / 2,
            )
            .set_opacity(0)
            .rotate(to_targets.get_angle())
        )
        ax.shift(to_targets.get_start() - ax.c2p(0, 0))
        target1_line = Arrow(
            radar.radome.get_right() + UP / 2,
            new_cloud.get_corner(DL),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            color=TX_COLOR,
            buff=0,
        )
        target1_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target1_line.get_length(),
                y_length=radar.radome.height / 2,
            )
            .rotate(to_cloud.get_angle() + PI)
            .set_opacity(0)
        )
        target1_ax.shift(target1_line.get_end() - target1_ax.c2p(0, 0))

        target2_line = Arrow(
            radar.radome.get_right() + UP / 2,
            new_plane.get_left(),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            color=TX_COLOR,
            buff=0,
        )
        target2_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target2_line.get_length(),
                y_length=radar.radome.height / 2,
            )
            .rotate(target2_line.get_angle() + PI)
            .set_opacity(0)
        )
        target2_ax.shift(target2_line.get_end() - target2_ax.c2p(0, 0))

        xmax = VT(0)
        xmax_t1 = VT(0)
        xmax_t2 = VT(0)
        pw = 0.3
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

        self.add(ax, target1_ax, target2_ax)

        self.play(FadeOut(to_cloud, to_plane))

        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        tau_line_r = Line(ax.c2p(~xmax, -1), ax.c2p(~xmax, 1))
        tau_line_l = Line(ax.c2p(~xmax - pw, -1), ax.c2p(~xmax - pw, 1))
        tau_line = Line(tau_line_l.get_end(), tau_line_r.get_end()).shift(
            (ax.c2p(~xmax - pw, 0) - tau_line_l.get_start()) * 0.8
        )
        tau = MathTex(r"\tau").scale(2).next_to(tau_line.get_midpoint(), UL, SMALL_BUFF)

        tau_line_r.shift(ax.c2p(~xmax, 0) - tau_line_r.get_start())
        tau_line_l.shift(ax.c2p(~xmax - pw, 0) - tau_line_l.get_start())
        # self.add(tau_line_l, tau_line_r, tau_line, tau)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(tx.width * 3)
                .move_to(tx)
                .shift(UP + LEFT / 2),
                AnimationGroup(
                    Create(tau_line_l), Create(tau_line), Create(tau_line_r)
                ),
                FadeIn(tau),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        delta_r = (
            MathTex(r"\Delta R = R_1 - R_2 \ge \frac{c \tau}{2}")
            .scale(2)
            .move_to(delta_r_qmark)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(tau_line_l),
                    Uncreate(tau_line_r),
                    Uncreate(tau_line),
                ),
                self.camera.frame.animate.restore().scale(1.2),
                ReplacementTransform(delta_r_qmark[0][:-1], delta_r[0][:9]),
                ShrinkToCenter(delta_r_qmark[0][-1]),
                GrowFromCenter(delta_r[0][9]),
                ReplacementTransform(tau[0], delta_r[0][10], path_arc=PI / 3),
                GrowFromCenter(delta_r[0][11]),
                GrowFromCenter(delta_r[0][12]),
                LaggedStart(
                    xmax @ 4,
                    LaggedStart(xmax_t2 @ (1 + pw), xmax_t1 @ (1 + pw), lag_ratio=0.15),
                    lag_ratio=0.15,
                    run_time=3,
                ),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        def compute_phase_diff(v):
            time_from_vel = 2 * (v * Tc) / c
            return 2 * PI * f * time_from_vel

        def compute_f_beat(R):
            return (2 * R * bw) / (c * Tc)

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

        t = np.arange(0, max_time, 1 / fs)

        window = signal.windows.blackman(N)
        fft_len = N * 8
        max_vel = wavelength / (4 * Tc)
        vel_res = wavelength / (2 * M * Tc)
        rmax = c * Tc * fs / (2 * bw)
        n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        target2_pos = VT(17)
        target2_vel = VT(10)

        def plot_rd():
            targets = [(20, 8, 0), (~target2_pos, ~target2_vel, 0)]
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

            cmap = colormaps.get("coolwarm")
            range_doppler_fmt = np.uint8(cmap(10 * np.log10(range_doppler + 1)) * 255)
            range_doppler_fmt[range_doppler < 0.05] = [0, 0, 0, 0]

            rd_img = (
                ImageMobject(range_doppler_fmt, image_mode="RGBA")
                .stretch_to_fit_width(config.frame_width * 0.6)
                .stretch_to_fit_height(config.frame_width * 0.6)
                .next_to(radar.vgroup, DOWN, MED_SMALL_BUFF)
                .set_x(0)
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
            Text("Range", font=FONT, font_size=DEFAULT_FONT_SIZE)
            .rotate(PI / 2)
            .next_to(rd_ax.c2p(0, 5), LEFT)
        )
        vel_label = Text("Velocity", font=FONT, font_size=DEFAULT_FONT_SIZE).next_to(
            rd_ax.c2p(5, 0), DOWN
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(*delta_r[0]),
                self.camera.frame.animate.shift(DOWN * 4),
                Create(rd_ax),
                Write(range_label),
                Write(vel_label),
                FadeIn(rd_img),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            plane.animate.shift(LEFT),
            target2_pos + 10,
            target2_vel.animate(
                rate_func=rate_functions.there_and_back
            ).increment_value(5),
            run_time=4,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        angular_title = Text("Angular Resolution", font=FONT).set_color(RED)
        angular = MathTex(r"\theta \approx \frac{\lambda}{D}").scale(2).set_color(RED)
        range_title = Text("Range Resolution", font=FONT).set_color(BLUE)
        range_eqn = MathTex(r"\Delta R \ge \frac{c \tau}{2}").scale(2).set_color(BLUE)
        velocity_title = Text("Velocity Resolution", font=FONT).set_color(ORANGE)
        velocity = (
            MathTex(r"\Delta v = \frac{\text{PRF} \lambda}{2 M}")
            .scale(2)
            .set_color(ORANGE)
        )
        eqns_group = (
            Group(
                Group(angular_title, angular).arrange(DOWN, MED_SMALL_BUFF),
                Group(range_title, range_eqn).arrange(DOWN, MED_SMALL_BUFF),
                Group(velocity_title, velocity).arrange(DOWN, MED_SMALL_BUFF),
            )
            .arrange(DOWN, LARGE_BUFF)
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
        )

        # self.add(eqns_group)
        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(eqns_group[0]),
                GrowFromCenter(eqns_group[1]),
                GrowFromCenter(eqns_group[2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        thumbnail = ImageMobject(
            "../../09_resolution/static/Resolution Thumbnail.png"
        ).scale_to_fit_width(fw(self, 0.9))
        thumbnail_box = SurroundingRectangle(thumbnail, buff=0)
        thumbnail_group = Group(thumbnail, thumbnail_box).move_to(self.camera.frame)

        self.play(
            angular_title.animate.set_opacity(0.2),
            angular.animate.set_opacity(0.2),
            range_title.animate.set_opacity(0.2),
            range_eqn.animate.set_opacity(0.2),
            velocity_title.animate.set_opacity(0.2),
            velocity.animate.set_opacity(0.2),
            GrowFromCenter(thumbnail_group),
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)
