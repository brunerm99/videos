# resolution_short.py


import sys

import numpy as np
from manim import *
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

        self.wait(2)
