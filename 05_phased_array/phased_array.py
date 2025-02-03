# phased_array.py

import matplotlib.pyplot as plt

import sys
import warnings

import numpy as np
from numpy.fft import fft, fftshift
from manim import *
from scipy.interpolate import interp1d, bisplrep, bisplev
from scipy.constants import c
from scipy import signal
from MF_Tools import VT, TransformByGlyphMap


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR
from props import WeatherRadarTower, get_blocks

config.background_color = BACKGROUND_COLOR

BLOCKS = get_blocks()

SKIP_ANIMATIONS_OVERRIDE = False


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_transform_func(from_var, func=TransformFromCopy):
    def transform_func(m, **kwargs):
        return func(from_var, m, **kwargs)

    return transform_func


def sinc_pattern(theta, phi, L, W, lambda_0):
    k = 2 * np.pi / lambda_0
    E_theta = np.sinc(k * L / 2 * np.sin(theta) * np.cos(phi))
    E_phi = np.sinc(k * W / 2 * np.sin(theta) * np.sin(phi))
    return np.abs(E_theta * E_phi)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


class Intro(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        point_source = Dot(color=BLUE)

        self.play(
            point_source.next_to(
                [0, -config.frame_height / 2, 0], DOWN
            ).animate.move_to(ORIGIN)
        )

        self.wait(0.5)

        x_len = config.frame_width
        y_len = config.frame_height * 0.3
        sine_x_width = 0.2

        def get_plot(func, angle=0):
            sine_ax = (
                Axes(
                    x_range=[0, 1, 0.25],
                    y_range=[-1, 1, 0.5],
                    tips=False,
                    axis_config={
                        "include_numbers": False,
                    },
                    x_length=x_len,
                    y_length=y_len,
                )
                .next_to(point_source, LEFT, 0)
                .rotate(angle, about_point=ORIGIN)
            )

            sine_x1 = VT(-sine_x_width)
            sine_plot = always_redraw(
                lambda: sine_ax.plot(
                    func,
                    color=RX_COLOR,
                    x_range=[
                        min(~sine_x1 - sine_x_width, 1),
                        min(~sine_x1, 1),
                        1 / 1000,
                    ],
                )
            )
            return sine_plot, sine_x1

        plots = [
            get_plot(func, angle)
            for func, angle in [
                (lambda t: np.sin(2 * PI * 10 * t) + np.random.normal(0, 0.1), -PI / 6),
                (
                    lambda t: np.sin(2 * PI * 6 * t) + np.random.normal(0, 0.1),
                    PI * 0.55,
                ),
                (
                    lambda t: np.sin(2 * PI * 12 * t) + np.random.normal(0, 0.3),
                    PI * 0.3,
                ),
                (
                    lambda t: np.sin(2 * PI * 2 * t) + np.random.normal(0, 0.4),
                    -PI * 0.8,
                ),
                (lambda t: np.sin(2 * PI * 6 * t) + np.random.normal(0, 0.2), PI * 0.9),
            ]
        ]

        for plot, _ in plots:
            self.add(plot)
        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(*[x1 @ (1.1 + sine_x_width) for _, x1 in plots], lag_ratio=0.2),
            rate_func=rate_functions.linear,
            run_time=4,
        )
        for plot, _ in plots:
            self.remove(plot)

        self.wait(0.5)

        parabolic = Arc(radius=2, start_angle=PI * 0.75).move_to(point_source)

        self.play(Create(parabolic), FadeOut(point_source))

        self.wait(0.5)

        plots = [
            get_plot(func, angle)
            for func, angle in [
                (
                    lambda t: 0.7 * np.sin(2 * PI * 6 * t) + np.random.normal(0, 0.1),
                    PI * 0.8,
                ),
                (
                    lambda t: 1 * np.sin(2 * PI * 12 * t) + np.random.normal(0, 0.3),
                    PI,
                ),
                (
                    lambda t: 0.6 * np.sin(2 * PI * 2 * t) + np.random.normal(0, 0.4),
                    PI * 1.1,
                ),
                (
                    lambda t: 0.3 * np.sin(2 * PI * 10 * t) + np.random.normal(0, 0.1),
                    PI * 0.9,
                ),
                (
                    lambda t: 0.8 * np.sin(2 * PI * 6 * t) + np.random.normal(0, 0.2),
                    PI * 0.9,
                ),
            ]
        ]

        for plot, _ in plots:
            self.add(plot)
        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[x1 @ (1.1 + sine_x_width) for _, x1 in plots], lag_ratio=0.15
            ),
            rate_func=rate_functions.linear,
            run_time=4,
        )
        for plot, _ in plots:
            self.remove(plot)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        car = (
            SVGMobject("../props/static/car.svg")
            .set_color(WHITE)
            .set_fill(WHITE)
            .to_edge(RIGHT, LARGE_BUFF)
        )

        self.play(
            parabolic.animate.to_edge(LEFT, LARGE_BUFF),
            car.shift(RIGHT * 5).animate.shift(LEFT * 5),
        )

        self.wait(0.5)

        sine_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=(car.get_left()[0] - parabolic.get_right()[0]),
            y_length=y_len,
        ).next_to(parabolic, RIGHT, -SMALL_BUFF)

        sine_x1 = VT(-sine_x_width)
        sine_plot = always_redraw(
            lambda: sine_ax.plot(
                lambda t: np.sin(2 * PI * 6 * t),
                color=TX_COLOR,
                x_range=[
                    min(max(~sine_x1 - sine_x_width, 0), 1),
                    min(max(~sine_x1, 0), 1),
                    1 / 1000,
                ],
            )
        )

        self.add(sine_plot)

        sine_ax_tx = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=(car.get_left()[0] - parabolic.get_right()[0]),
                y_length=y_len,
            )
            .next_to(parabolic, RIGHT, -SMALL_BUFF)
            .flip()
        )

        sine_x1_tx = VT(-sine_x_width)
        sine_plot_tx = always_redraw(
            lambda: sine_ax_tx.plot(
                lambda t: np.sin(2 * PI * 6 * t),
                color=RX_COLOR,
                x_range=[
                    min(max(~sine_x1_tx - sine_x_width, 0), 1),
                    min(max(~sine_x1_tx, 0), 1),
                    1 / 1000,
                ],
            )
        )
        self.add(sine_plot_tx)

        self.play(
            LaggedStart(
                sine_x1 @ (1.1 + sine_x_width),
                sine_x1_tx @ (1.1 + sine_x_width),
                lag_ratio=0.3,
            ),
            rate_func=rate_functions.linear,
            run_time=4,
        )

        self.wait(0.5)

        plane = (
            SVGMobject("../props/static/plane.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .to_corner(UR)
            .shift(LEFT * 3)
        )

        self.play(
            plane.shift(UP * 5).animate.shift(DOWN * 5),
            parabolic.animate.rotate(PI / 6),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phased_array_title = Tex(
            "Phased Array", font_size=DEFAULT_FONT_SIZE * 1.8
        ).to_edge(UP, MED_LARGE_BUFF)

        antennas = Group()
        for _ in range(6):
            antenna_port = Line(DOWN / 2, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)

        antennas.arrange(RIGHT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                AnimationGroup(
                    plane.animate.shift(UP * 5),
                    car.animate.shift(RIGHT * 5),
                    parabolic.animate.shift(LEFT * 5),
                ),
                LaggedStart(*[GrowFromCenter(m) for m in antennas], lag_ratio=0.15),
                phased_array_title.shift(UP * 5).animate.shift(DOWN * 5),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back)
                    .scale(1.4)
                    .set_color(YELLOW)
                    for m in phased_array_title[0]
                ],
                lag_ratio=0.08,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            antennas.animate.to_edge(UP, LARGE_BUFF),
            phased_array_title.animate.shift(UP * 5),
        )

        def get_phase_delay_plot(ant):
            ax = (
                Axes(
                    x_range=[0, 1, 0.25],
                    y_range=[-1, 1, 0.5],
                    tips=False,
                    axis_config={
                        "include_numbers": False,
                    },
                    y_length=LARGE_BUFF,
                    x_length=config.frame_height * 0.4,
                )
                .set_opacity(0)
                .rotate(PI / 2)
                .next_to(ant, DOWN)
                .shift(DOWN * 8)
            )

            phase_shift = VT(0)
            plot = always_redraw(
                lambda: ax.plot(
                    lambda t: np.sin(2 * PI * 2 * t + ~phase_shift),
                    color=TX_COLOR,
                    x_range=[0, 1, 1 / 200],
                )
            )
            return ax, plot, phase_shift

        phase_plots = [get_phase_delay_plot(ant) for ant in antennas]
        for _, plot, _ in phase_plots:
            self.add(plot)

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ax.animate.shift(UP * 8) for ax, _, _ in phase_plots], lag_ratio=0.2
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        np.random.seed(1)
        self.play(
            LaggedStart(
                *[
                    phase_shift @ (2 * np.random.rand() * 2 * PI - PI)
                    for _, _, phase_shift in phase_plots
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        phase_shift_labels = Group(
            *[
                MathTex(f"\\phi_{n}", font_size=DEFAULT_FONT_SIZE * 1.4).next_to(
                    ax, DOWN
                )
                for n, (ax, _, _) in enumerate(phase_plots)
            ]
        )

        self.play(
            LaggedStart(
                *[m.shift(DOWN * 8).animate.shift(UP * 8) for m in phase_shift_labels]
            )
        )

        np.random.seed(2)
        self.play(
            LaggedStart(
                *[
                    phase_shift + (2 * np.random.rand() * 2 * PI - PI)
                    for _, _, phase_shift in phase_plots
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        parabolic = Arc(radius=2, start_angle=PI * 0.75).to_edge(LEFT, MED_LARGE_BUFF)
        arrow = Arrow(
            parabolic.get_right(),
            [
                (phase_plots[0][1].get_left() + RIGHT * 2)[0],
                parabolic.get_right()[1],
                0,
            ],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    antennas.animate.shift(RIGHT * 2),
                    *[ax.animate.shift(RIGHT * 2) for ax, _, _ in phase_plots],
                    phase_shift_labels.animate.shift(RIGHT * 2),
                ),
                parabolic.shift(LEFT * 4).animate.shift(RIGHT * 4),
                GrowArrow(arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(*[FadeOut(m) for m in self.mobjects], lag_ratio=0.08))

        self.wait(2)


class FarField(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=4,
            zoomed_display_width=2,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs,
        )

    def construct(self):
        x_len = config.frame_width
        y_len = config.frame_height * 0.3
        sine_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )

        sine_x_width = 0.2
        sine_x1 = VT(-sine_x_width)
        sine_plot = always_redraw(
            lambda: sine_ax.plot(
                lambda t: np.sin(2 * PI * 10 * t),
                color=TX_COLOR,
                x_range=[~sine_x1 - sine_x_width, ~sine_x1, 1 / 1000],
            )
        )

        self.add(sine_plot)
        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            sine_x1 @ (1.1 + sine_x_width), run_time=4, rate_func=rate_functions.linear
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.remove(sine_ax, sine_plot)

        point_source = Dot()

        self.play(GrowFromCenter(point_source))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # final_wave_radius = VT(point_source.width / 2)
        final_wave_radius = VT(1)
        final_wave = always_redraw(
            lambda: Circle(radius=~final_wave_radius, color=TX_COLOR)
        )

        self.play(
            Broadcast(
                Circle(radius=config.frame_width, color=TX_COLOR).move_to(point_source),
                final_opacity=1,
                lag_ratio=0.3,
                n_mobs=3,
            ),
            run_time=4,
        )
        self.play(GrowFromCenter(final_wave))

        self.wait(0.5)

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to([~final_wave_radius, 0, 0])
        frame.add_updater(lambda m: m.move_to([~final_wave_radius, 0, 0]))
        frame.set_color(PURPLE)
        zoomed_display_frame.set_color(RED)
        zoomed_display.to_corner(UL)

        zd_rect = BackgroundRectangle(
            zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF
        )
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(
            zd_rect, lambda rect: rect.replace(zoomed_display)
        )

        self.play(Create(frame))
        self.activate_zooming()
        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)

        self.wait(0.5)

        range_arrow = always_redraw(
            lambda: Arrow(point_source.get_center(), [~final_wave_radius - 0.2, 0, 0])
        )
        range_label = always_redraw(lambda: MathTex("R").next_to(range_arrow, UP))

        self.play(GrowArrow(range_arrow), FadeIn(range_label))

        self.wait(0.5)

        self.play(final_wave_radius @ 2)

        self.wait(0.5)

        self.play(final_wave_radius @ 3)

        self.wait(0.5)

        self.play(final_wave_radius @ (config.frame_width / 2))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(final_wave_radius @ (config.frame_width * 1.5))

        self.wait(0.5)

        plane_wave_label = (
            Tex(r"$\approx$ Plane wave").set_opacity(0).next_to(zoomed_display_frame)
        )

        self.play(
            FadeOut(range_arrow, range_label),
            LaggedStart(
                *[
                    m.shift(UP).animate.shift(DOWN).set_opacity(1)
                    for m in plane_wave_label[0]
                ],
                lag_ratio=0.08,
            ),
        )

        self.wait(0.5)

        self.play(
            Uncreate(zoomed_display_frame),
            FadeOut(frame, final_wave, point_source, plane_wave_label),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        far_field = Line(UP, DOWN, color=TX_COLOR).to_edge(RIGHT, LARGE_BUFF * 2)
        near_field = Line(UP, DOWN, path_arc=-PI / 5, color=TX_COLOR).to_edge(
            LEFT, LARGE_BUFF * 2
        )
        far_field_label = Tex("Far field").next_to(far_field, UP)
        near_field_label = Tex("Near field").next_to(near_field, UP)

        self.play(
            Group(far_field, far_field_label).shift(RIGHT * 5).animate.shift(LEFT * 5),
            Group(near_field, near_field_label)
            .shift(LEFT * 5)
            .animate.shift(RIGHT * 5),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        far_field_boundary_eqn = MathTex(
            r"d_F = \frac{2 D^2}{\lambda}", font_size=DEFAULT_FONT_SIZE * 1.5
        ).set_opacity(0)

        near_arrow = Arrow(near_field.get_right(), far_field_boundary_eqn.get_left())
        far_arrow = Arrow(far_field_boundary_eqn.get_right(), far_field.get_left())

        far_field_boundary_eqn[0][6].set_opacity(1).save_state()
        far_field_boundary_eqn[0][6].set(width=0)
        self.play(
            LaggedStart(
                far_field_boundary_eqn[0][:2].animate().set_opacity(1),
                far_field_boundary_eqn[0][2].animate().set_opacity(1),
                far_field_boundary_eqn[0][3]
                .shift(UP)
                .animate()
                .shift(DOWN)
                .set_opacity(1),
                far_field_boundary_eqn[0][4:6]
                .shift(UP)
                .animate()
                .shift(DOWN)
                .set_opacity(1),
                far_field_boundary_eqn[0][-1]
                .shift(DOWN)
                .animate()
                .shift(UP)
                .set_opacity(1),
                far_field_boundary_eqn[0][6].animate().restore(),
                lag_ratio=0.2,
            )
        )
        self.play(
            LaggedStart(GrowArrow(near_arrow), GrowArrow(far_arrow), lag_ratio=0.3)
        )

        self.wait(0.5)

        far_field_eqn = MathTex(
            r"d_F \gg \frac{2 D^2}{\lambda}", font_size=DEFAULT_FONT_SIZE * 1.5
        ).move_to([-config.frame_width / 4, 0, 0])

        self.play(
            LaggedStart(
                FadeOut(near_arrow, far_arrow, near_field, near_field_label),
                AnimationGroup(
                    TransformByGlyphMap(
                        far_field_boundary_eqn,
                        far_field_eqn,
                        ([0, 1, 3, 4, 5, 6, 7], [0, 1, 3, 4, 5, 6, 7]),
                        ([2], [2]),
                    ),
                    Group(far_field, far_field_label).animate.move_to(
                        [config.frame_width / 4, 0, 0]
                    ),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(far_field_label, far_field_eqn),
                far_field.animate.rotate(PI / 2)
                .to_edge(UP)
                .stretch_to_fit_width(config.frame_width * 1.1)
                .set_x(0),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


class HeadOn(MovingCameraScene):
    def construct(self):
        np.random.seed(0)

        plane_wave = Line(
            LEFT * (config.frame_width * 1.2 / 2),
            RIGHT * (config.frame_width * 1.2 / 2),
            color=TX_COLOR,
        ).to_edge(UP)

        self.add(plane_wave)
        self.next_section(skip_animations=skip_animations(True))

        antenna_port_left = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_left = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_left, UP)
        )
        antenna_port_right = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_right = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_right, UP)
        )

        antenna_left = Group(antenna_port_left, antenna_tri_left)
        antenna_right = Group(antenna_port_right, antenna_tri_right)
        antennas = (
            Group(antenna_left, antenna_right)
            .arrange(RIGHT, LARGE_BUFF * 3)
            .to_edge(DOWN, -SMALL_BUFF)
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in antennas], lag_ratio=0.2))

        self.wait(0.5)

        x_len = (antenna_tri_left.get_bottom() - antenna_port_left.get_bottom())[1]
        y_len = antenna_tri_left.width
        sine_ax_left = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(antenna_tri_left, DOWN, 0)
        )
        sine_ax_right = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(antenna_tri_right, DOWN, 0)
        )

        noise_std = 0.1
        sine_f = 1

        sine_x0 = VT(0)
        sine_x1 = VT(1)

        sine_left = always_redraw(
            lambda: sine_ax_left.plot(
                lambda t: np.sin(2 * PI * sine_f * t)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~sine_x0, ~sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        sine_right = always_redraw(
            lambda: sine_ax_right.plot(
                lambda t: np.sin(2 * PI * sine_f * t)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~sine_x0, ~sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        self.play(plane_wave.animate.next_to(antennas, UP, 0))
        self.play(FadeOut(plane_wave))
        self.play(Create(sine_left), Create(sine_right))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5, frozen_frame=False)

        lna_left = (
            BLOCKS.get("amp")
            .copy()
            .rotate(-PI / 2)
            .scale_to_fit_width(antenna_tri_left.width)
        ).next_to(antenna_port_left, DOWN, 0)
        lna_to_lp_filter_left = Line(
            lna_left.get_bottom(), lna_left.get_bottom() + DOWN * sine_left.height
        )
        lp_filter_left = (
            BLOCKS.get("lp_filter")
            .copy()
            .scale_to_fit_width(antenna_tri_left.width)
            .next_to(lna_to_lp_filter_left, DOWN, 0)
        )

        lna_right = (
            BLOCKS.get("amp")
            .rotate(-PI / 2)
            .scale_to_fit_width(antenna_tri_right.width)
        ).next_to(antenna_port_right, DOWN, 0)
        lna_to_lp_filter_right = Line(
            lna_right.get_bottom(), lna_right.get_bottom() + DOWN * sine_left.height
        )
        lp_filter_right = (
            BLOCKS.get("lp_filter")
            .scale_to_fit_width(antenna_tri_right.width)
            .next_to(lna_to_lp_filter_right, DOWN, 0)
        )
        lp_filter_to_combiner_left = Line(
            lp_filter_left.get_bottom(),
            lp_filter_left.get_bottom() + DOWN * sine_left.height,
        )
        lp_filter_to_combiner_right = Line(
            lp_filter_right.get_bottom(),
            lp_filter_right.get_bottom() + DOWN * sine_left.height,
        )
        combiner_rect = (
            Rectangle(
                height=lp_filter_left.height,
                width=Group(lp_filter_left, lp_filter_right).width,
            )
            .next_to(lp_filter_to_combiner_left, DOWN, 0)
            .set_x(antennas.get_x())
        )
        combiner_label = MathTex(r"\bold{+}", font_size=DEFAULT_FONT_SIZE * 2).move_to(
            combiner_rect
        )
        combiner = Group(combiner_rect, combiner_label)
        from_combiner = Line(
            combiner.get_bottom(),
            combiner.get_bottom() + DOWN * sine_left.height,
        )

        bd = Group(
            antennas,
            lna_left,
            lna_to_lp_filter_left,
            lp_filter_left,
            lna_right,
            lna_to_lp_filter_right,
            lp_filter_right,
            lp_filter_to_combiner_left,
            lp_filter_to_combiner_right,
        )
        self.add(bd[1:], combiner, from_combiner)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_height(bd.height * 1.3).move_to(bd)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)
        amp_sine_ax_left = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(lna_left, DOWN, 0)
        )
        amp_sine_ax_right = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(lna_right, DOWN, 0)
        )

        amp_sine_x0 = VT(0)
        amp_sine_x1 = VT(0)

        amp_sine_left = always_redraw(
            lambda: amp_sine_ax_left.plot(
                lambda t: 2 * np.sin(2 * PI * sine_f * t)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~amp_sine_x0, ~amp_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        amp_sine_right = always_redraw(
            lambda: amp_sine_ax_right.plot(
                lambda t: 2 * np.sin(2 * PI * sine_f * t)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~amp_sine_x0, ~amp_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        self.add(amp_sine_left, amp_sine_right)

        self.next_section(skip_animations=skip_animations(True))

        filt_sine_ax_left = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(lp_filter_left, DOWN, 0)
        )
        filt_sine_ax_right = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(lp_filter_right, DOWN, 0)
        )

        filt_sine_x0 = VT(0)
        filt_sine_x1 = VT(0)

        amp_scale = 1.5
        filt_sine_left = always_redraw(
            lambda: filt_sine_ax_left.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t),
                x_range=[~filt_sine_x0, ~filt_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        filt_sine_right = always_redraw(
            lambda: filt_sine_ax_right.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t),
                x_range=[~filt_sine_x0, ~filt_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        self.add(filt_sine_left, filt_sine_right)

        self.play(sine_x0 @ 1, amp_sine_x1 @ 1, rate_func=rate_functions.ease_in_sine)
        self.play(
            amp_sine_x0 @ 1, filt_sine_x1 @ 1, rate_func=rate_functions.ease_out_sine
        )

        self.wait(0.5)

        bd.add(combiner, from_combiner)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(bd.height * 1.3).move_to(bd)
        )

        self.wait(0.5)

        combined_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(combiner, DOWN, 0)
        )

        combined_sine_x0 = VT(0)
        combined_sine_x1 = VT(0)

        combined_sine = always_redraw(
            lambda: combined_ax.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t)
                + amp_scale * np.sin(2 * PI * sine_f * t),
                x_range=[~combined_sine_x0, ~combined_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )
        self.add(combined_sine)

        self.play(filt_sine_x0 @ 1, combined_sine_x1 @ 1)

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore(), combined_sine_x1 @ 0)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        angle = -PI / 6
        angled_plane_wave = (
            Line(
                LEFT * 4,
                RIGHT * 4,
                color=TX_COLOR,
            ).rotate(angle)
        ).set_z_index(-2)
        angled_plane_wave.shift(
            self.camera.frame.get_corner(UR) - angled_plane_wave.get_center()
        )

        background_rect = (
            Rectangle(
                width=config.frame_width,
                height=config.frame_height,
                stroke_color=BACKGROUND_COLOR,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            )
            .move_to(antennas, UP)
            .set_z_index(-1)
        )

        self.add(
            angled_plane_wave,
            background_rect,
        )

        phase_left = VT(2 * PI / 3)
        phase_right = VT(0)

        shifted_sine_left_x0 = VT(0)
        shifted_sine_left_x1 = VT(0)
        shifted_sine_right_x0 = VT(0)
        shifted_sine_right_x1 = VT(0)

        shifted_sine_left = always_redraw(
            lambda: sine_ax_left.plot(
                lambda t: np.sin(2 * PI * sine_f * t + ~phase_left)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~shifted_sine_left_x0, ~shifted_sine_left_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        shifted_sine_right = always_redraw(
            lambda: sine_ax_right.plot(
                lambda t: np.sin(2 * PI * sine_f * t + ~phase_right)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~shifted_sine_right_x0, ~shifted_sine_right_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        self.add(shifted_sine_left, shifted_sine_right)

        self.play(
            LaggedStart(
                angled_plane_wave.animate(run_time=3).shift(
                    14 * (LEFT * np.cos(angle) + UP * np.sin(angle))
                ),
                shifted_sine_right_x1 @ 1,
                shifted_sine_left_x1 @ 1,
                lag_ratio=0.4,
            )
        )

        self.wait(0.5, frozen_frame=False)

        shifted_amp_sine_x0 = VT(0)
        shifted_amp_sine_x1 = VT(0)

        shifted_amp_sine_left = always_redraw(
            lambda: amp_sine_ax_left.plot(
                lambda t: 2 * np.sin(2 * PI * sine_f * t + ~phase_left)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~shifted_amp_sine_x0, ~shifted_amp_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        shifted_amp_sine_right = always_redraw(
            lambda: amp_sine_ax_right.plot(
                lambda t: 2 * np.sin(2 * PI * sine_f * t + ~phase_right)
                + np.random.normal(loc=0, scale=noise_std),
                x_range=[~shifted_amp_sine_x0, ~shifted_amp_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        shifted_filt_sine_x0 = VT(0)
        shifted_filt_sine_x1 = VT(0)

        amp_scale = 1.5
        shifted_filt_sine_left = always_redraw(
            lambda: filt_sine_ax_left.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t + ~phase_left),
                x_range=[~shifted_filt_sine_x0, ~shifted_filt_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        shifted_filt_sine_right = always_redraw(
            lambda: filt_sine_ax_right.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t + ~phase_right),
                x_range=[~shifted_filt_sine_x0, ~shifted_filt_sine_x1, 1 / 1000],
                color=RX_COLOR,
            )
        )

        shifted_combined_sine_x0 = VT(0)
        shifted_combined_sine_x1 = VT(0)

        phase_shifter_val_left = VT(0)
        phase_shifter_val_right = VT(0)

        shifted_combined_sine = always_redraw(
            lambda: combined_ax.plot(
                lambda t: amp_scale
                * np.sin(2 * PI * sine_f * t + ~phase_left + ~phase_shifter_val_left)
                + amp_scale
                * np.sin(2 * PI * sine_f * t + ~phase_right + ~phase_shifter_val_right),
                x_range=[
                    ~shifted_combined_sine_x0,
                    ~shifted_combined_sine_x1,
                    1 / 1000,
                ],
                color=RX_COLOR,
            )
        )
        self.add(
            shifted_amp_sine_left,
            shifted_amp_sine_right,
            shifted_filt_sine_left,
            shifted_filt_sine_right,
            shifted_combined_sine,
        )

        self.remove(angled_plane_wave)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(bd.height * 1.3).move_to(bd)
        )

        self.wait(0.5)

        self.play(
            shifted_sine_left_x0 @ 1,
            shifted_sine_right_x0 @ 1,
            shifted_amp_sine_x1 @ 1,
            rate_func=rate_functions.ease_in_sine,
        )
        self.play(
            shifted_amp_sine_x0 @ 1,
            shifted_filt_sine_x1 @ 1,
            rate_func=rate_functions.ease_out_sine,
        )

        self.wait(0.5)

        phase_delta_label = always_redraw(
            lambda: MathTex(
                f"\\Delta \\phi = {~phase_left * 180 / PI:.1f}^\\circ"
            ).next_to(shifted_filt_sine_left, LEFT, MED_LARGE_BUFF)
        )

        self.play(GrowFromCenter(phase_delta_label))

        self.wait(0.5)

        self.play(shifted_combined_sine_x1 @ 1)

        self.wait(0.5)

        self.play(phase_left @ PI)

        self.wait(0.5)

        self.play(phase_left @ (2 * PI / 3))

        self.wait(0.5)

        phase_shifter_left_rect = Square(lna_left.width, color=WHITE).next_to(
            lp_filter_to_combiner_left, DOWN, 0
        )
        phase_shifter_left_label = MathTex(
            r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2
        ).move_to(phase_shifter_left_rect)
        phase_shifter_to_combiner_left = Line(
            phase_shifter_left_rect.get_bottom(),
            phase_shifter_left_rect.get_bottom()
            + DOWN * lp_filter_to_combiner_left.height,
        )
        phase_shifter_right_rect = Square(lna_right.width, color=WHITE).next_to(
            lp_filter_to_combiner_right, DOWN, 0
        )
        phase_shifter_right_label = MathTex(
            r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2
        ).move_to(phase_shifter_right_rect)
        phase_shifter_to_combiner_right = Line(
            phase_shifter_right_rect.get_bottom(),
            phase_shifter_right_rect.get_bottom()
            + DOWN * lp_filter_to_combiner_left.height,
        )

        phase_shifter_left = Group(phase_shifter_left_rect, phase_shifter_left_label)
        phase_shifter_right = Group(phase_shifter_right_rect, phase_shifter_right_label)
        phase_shifters = Group(
            phase_shifter_left,
            phase_shifter_right,
            phase_shifter_to_combiner_left,
            phase_shifter_to_combiner_right,
        )

        bd.add(phase_shifters)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Group(combiner, from_combiner, combined_ax).animate.next_to(
                        phase_shifters, DOWN, 0
                    ),
                    shifted_combined_sine_x1 @ 0,
                ),
                AnimationGroup(
                    GrowFromCenter(phase_shifter_left),
                    GrowFromCenter(phase_shifter_right),
                ),
                AnimationGroup(
                    Create(phase_shifter_to_combiner_left),
                    Create(phase_shifter_to_combiner_right),
                ),
                lag_ratio=0.4,
            )
        )
        self.play(
            self.camera.frame.animate.scale_to_fit_height(bd.height * 1.3).move_to(bd)
        )

        self.wait(0.5)

        shifted_phase_label_left = always_redraw(
            lambda: MathTex(
                f"\\Delta \\phi = {~phase_shifter_val_left * 180 / PI:.1f}^\\circ"
            ).next_to(phase_shifter_left, LEFT, MED_LARGE_BUFF)
        )
        shifted_phase_label_right = always_redraw(
            lambda: MathTex(
                f"\\Delta \\phi = {~phase_shifter_val_right * 180 / PI:.1f}^\\circ"
            ).next_to(phase_shifter_right, RIGHT, MED_LARGE_BUFF)
        )

        self.play(
            FadeIn(shifted_phase_label_left, shift=RIGHT),
            FadeIn(shifted_phase_label_right, shift=LEFT),
        )

        self.wait(0.5)

        phase_shifter_left_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(phase_shifter_left, DOWN, 0)
        )
        phase_shifter_right_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False, "include_ticks": False},
                x_length=x_len,
                y_length=y_len,
            )
            .rotate(-PI / 2)
            .next_to(phase_shifter_right, DOWN, 0)
        )

        after_phase_shift_x0 = VT(0)
        after_phase_shift_x1 = VT(0)
        after_phase_shift_left = always_redraw(
            lambda: phase_shifter_left_ax.plot(
                lambda t: amp_scale
                * np.sin(2 * PI * sine_f * t + ~phase_left + ~phase_shifter_val_left),
                x_range=[
                    ~after_phase_shift_x0,
                    ~after_phase_shift_x1,
                    1 / 1000,
                ],
                color=RX_COLOR,
            )
        )
        after_phase_shift_right = always_redraw(
            lambda: phase_shifter_right_ax.plot(
                lambda t: amp_scale
                * np.sin(2 * PI * sine_f * t + ~phase_right + ~phase_shifter_val_right),
                x_range=[
                    ~after_phase_shift_x0,
                    ~after_phase_shift_x1,
                    1 / 1000,
                ],
                color=RX_COLOR,
            )
        )

        self.add(after_phase_shift_left, after_phase_shift_right)

        self.play(after_phase_shift_x1 @ 1)

        self.wait(0.5)

        self.play(phase_shifter_val_left + PI, phase_shifter_val_right - PI)

        self.wait(0.5)

        self.play(phase_shifter_val_left - PI, phase_shifter_val_right + PI)

        self.wait(0.5)

        self.play(phase_shifter_val_left @ (-~phase_left))

        self.wait(0.5)

        self.play(shifted_combined_sine_x1 @ 1)

        self.wait(0.5)

        plane_wave = Line(
            LEFT * (config.frame_width * 1.2 / 2),
            RIGHT * (config.frame_width * 1.2 / 2),
            color=TX_COLOR,
        ).to_edge(UP)

        self.add(plane_wave)

        self.play(self.camera.frame.animate.restore())

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_left @= 0
        shifted_sine_left_x0 @= 0
        shifted_sine_right_x0 @= 0
        shifted_amp_sine_x1 @= 0
        shifted_amp_sine_x0 @= 0
        shifted_filt_sine_x1 @= 0.01
        shifted_sine_right_x1 @= 0
        shifted_sine_left_x1 @= 0
        combined_sine_x0 @= 0
        combined_sine_x1 @= 0
        shifted_combined_sine_x0 @= 0
        shifted_combined_sine_x1 @= 0
        after_phase_shift_x0 @= 0
        after_phase_shift_x1 @= 0

        self.play(
            LaggedStart(
                plane_wave.animate.set_y(antennas.get_top()[1]),
                AnimationGroup(
                    shifted_sine_right_x1 @ 1,
                    shifted_sine_left_x1 @ 1,
                ),
                lag_ratio=0.4,
            )
        )
        self.play(FadeOut(plane_wave))

        self.wait(0.5, frozen_frame=False)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(bd.height * 1.3).move_to(bd)
        )

        self.wait(0.5)

        self.play(
            shifted_sine_left_x0 @ 1,
            shifted_sine_right_x0 @ 1,
            shifted_amp_sine_x1 @ 1,
            rate_func=rate_functions.ease_in_sine,
        )
        self.play(
            shifted_amp_sine_x0 @ 1,
            shifted_filt_sine_x1 @ 1,
            rate_func=rate_functions.ease_out_sine,
        )

        self.wait(0.5)

        self.play(after_phase_shift_x1 @ 1)

        self.wait(0.5)

        self.play(shifted_combined_sine_x1 @ 1)

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        steering_arrow = Arrow(antennas.get_top(), self.camera.frame.get_top())

        self.play(GrowArrow(steering_arrow))

        self.wait(0.5)

        self.play(
            steering_arrow.animate.rotate(
                angle, about_point=steering_arrow.get_bottom()
            )
        )

        self.wait(0.5)

        phase_shifters_group = Group(
            *phase_shifters, shifted_phase_label_left, shifted_phase_label_right
        )
        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                phase_shifters_group.width * 1.2
            ).move_to(phase_shifters_group)
        )

        self.wait(0.5)

        self.play(phase_shifter_val_left @ 0)
        self.play(phase_shifter_val_left @ (PI / 2), phase_shifter_val_right @ (PI / 6))
        self.play(phase_shifter_val_left @ (PI), phase_shifter_val_right @ (PI / 2))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.remove(steering_arrow)
        delta_phi = MathTex(r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2).move_to(
            self.camera.frame
        )

        self.play(
            LaggedStart(
                FadeOut(*self.mobjects),
                GrowFromCenter(delta_phi),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(delta_phi))

        self.wait(2)


class PhaseCalc(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        self.next_section(skip_animations=skip_animations(True))

        antenna_port_left = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_left = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_left, UP)
        )
        antenna_port_right = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_right = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_right, UP)
        )

        antenna_left = Group(antenna_port_left, antenna_tri_left)
        antenna_right = Group(antenna_port_right, antenna_tri_right)
        antennas = (
            Group(antenna_left, antenna_right)
            .arrange(RIGHT, LARGE_BUFF * 3)
            .to_edge(DOWN, -SMALL_BUFF)
        )

        angle = -PI / 6

        background_rect = (
            Rectangle(
                width=config.frame_width,
                height=config.frame_height,
                stroke_color=BACKGROUND_COLOR,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            )
            .move_to(antennas, UP)
            .set_z_index(-1)
        )

        self.add(background_rect)

        plane_wave = Line(
            antenna_right.get_top() - 8 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            antenna_right.get_top() + 8 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            color=RX_COLOR,
        ).set_z_index(-2)

        self.add(antennas, plane_wave)
        # self.play(
        #     angled_plane_wave.animate(run_time=3).shift(
        #         7 * (LEFT * np.cos(angle) + UP * np.sin(angle))
        #     )
        # )

        self.next_section(skip_animations=skip_animations(False))

        self.camera.frame.save_state()
        self.camera.frame.shift(DOWN * config.frame_height * 2)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        x_len = antenna_right.get_top()[0] - antenna_left.get_top()[0]
        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 0.75, 0.25],
            tips=False,
            x_length=x_len,
            y_length=x_len * 0.75,
        )

        ax.shift(antenna_left.get_top() - ax.c2p(0, 0))

        L_line = Line(
            ax.c2p(0, 0),
            ax.c2p(
                np.cos(PI / 2 + angle) / 2,
                np.sin(PI / 2 + angle) / 2,
            ),
            color=YELLOW,
        )

        L_label = MathTex("L").next_to(L_line.get_midpoint(), RIGHT)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.8).shift(UP / 2),
                Create(L_line),
                GrowFromCenter(L_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(Create(ax), run_time=2)

        self.wait(0.5)

        z_label = ax.get_y_axis_label(r"\hat{z}")

        self.play(GrowFromCenter(z_label))

        self.wait(0.5)

        theta_angle = ArcBetweenPoints(
            ax.c2p(0, 0.25),
            ax.c2p(
                0.5 * np.cos(PI / 2 + angle) / 2,
                0.5 * np.sin(PI / 2 + angle) / 2,
            ),
            angle=-TAU / 4,
        )

        theta = MathTex(r"\theta").next_to(theta_angle, UP).shift(RIGHT / 8)

        self.play(Create(theta_angle), FadeIn(theta))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        qmark_up = Tex("?").move_to(ax.c2p(0.125, 0.6))
        qmark_left = Tex("?").next_to(ax.c2p(0, 0.25), LEFT)

        self.play(
            LaggedStart(
                GrowFromCenter(qmark_left), GrowFromCenter(qmark_up), lag_ratio=0.4
            )
        )

        self.wait(0.5)

        right_tri = Polygon(
            ax.c2p(0, 0),
            L_line.get_end(),
            ax.c2p(1, 0),
            fill_opacity=0.2,
            fill_color=GREEN,
            stroke_opacity=0,
        ).set_z_index(-3)
        right_ang = RightAngle(
            Line(
                ax.c2p(0, 0),
                ax.c2p(
                    np.cos(PI / 2 + angle),
                    np.sin(PI / 2 + angle),
                ),
            ),
            plane_wave,
            quadrant=(-1, -1),
        ).set_z_index(-3)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(right_tri.width * 1.3).move_to(
                right_tri
            ),
            FadeIn(right_tri),
            Create(right_ang),
            FadeOut(qmark_up, qmark_left),
        )

        self.wait(0.5)

        dx = Line(ax.c2p(0, 0), ax.c2p(1, 0), color=YELLOW)
        dx_label = MathTex("d_x").next_to(dx, UP)

        self.play(LaggedStart(Create(dx), FadeIn(dx_label), lag_ratio=0.4))

        self.wait(0.5)

        psi_angle = ArcBetweenPoints(
            ax.c2p(
                0.25 * np.cos(PI / 2 + angle) / 2,
                0.25 * np.sin(PI / 2 + angle) / 2,
            ),
            ax.c2p(0.125, 0),
            angle=-TAU / 4,
        )
        psi = MathTex(r"\psi").next_to(psi_angle, UR, SMALL_BUFF).shift(DOWN / 2)

        self.play(LaggedStart(Create(psi_angle), GrowFromCenter(psi), lag_ratio=0.4))

        self.wait(0.5)

        self.play(
            L_label.animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .shift(UP / 3)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        cos_ang = (
            MathTex(r"\cos{(\text{angle})} = \frac{\text{adjacent}}{\text{hypotenuse}}")
            .to_edge(UP, LARGE_BUFF)
            .shift(RIGHT * 2)
        )

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in [
                        *cos_ang[0][:4],
                        cos_ang[0][4:9],
                        *cos_ang[0][9:11],
                        cos_ang[0][11:19],
                        cos_ang[0][19],
                        cos_ang[0][20:],
                    ]
                ],
                lag_ratio=0.1,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cos_ang_filled = MathTex(r"\cos{(\psi)} = \frac{L}{d_x}").move_to(cos_ang, LEFT)

        self.play(
            TransformByGlyphMap(
                cos_ang,
                cos_ang_filled,
                ([0, 1, 2, 3], [0, 1, 2, 3]),
                ([4, 5, 6, 7, 8], ShrinkToCenter),
                (get_transform_func(psi[0]), [4], {"path_arc": -PI / 2}),
                ([9, 10], [5, 6], {"delay": 0.3}),
                ([11, 12, 13, 14, 15, 16, 17, 18], ShrinkToCenter, {"delay": 0.3}),
                (
                    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                    ShrinkToCenter,
                    {"delay": 0.3},
                ),
                ([19], [8], {"delay": 0.5}),
                (get_transform_func(L_label[0]), [7], {"path_arc": -PI / 3}),
                (get_transform_func(dx_label[0]), [9, 10], {"path_arc": -PI / 3}),
            )
        )

        self.wait(0.5)

        right_ang_phi_theta = RightAngle(
            Line(ax.c2p(0, -1), ax.c2p(0, 1)), Line(ax.c2p(-1, 0), ax.c2p(1, 0))
        )
        right_ang_arrow = Arrow(
            right_ang_phi_theta.get_left() + LEFT * 2 + UP,
            right_ang_phi_theta.get_left(),
        )

        self.play(Create(right_ang_phi_theta), GrowArrow(right_ang_arrow))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cos_ang_90 = MathTex(r"\cos{(90^\circ - \theta)} = \frac{L}{d_x}").move_to(
            cos_ang_filled, LEFT
        )

        self.play(
            TransformByGlyphMap(
                cos_ang_filled,
                cos_ang_90,
                ([0, 1, 2], [0, 1, 2], {"delay": 0.4}),
                ([3], [3], {"delay": 0}),
                ([4], ShrinkToCenter),
                (get_transform_func(theta[0]), [8], {"path_arc": -PI / 3}),
                (GrowFromCenter, [4, 5, 6], {"delay": 0.1}),
                (GrowFromCenter, [7], {"delay": 0.2}),
                ([5], [9], {"delay": 0}),
                ([6], [10], {"delay": 0}),
                ([7, 8, 9, 10], [11, 12, 13, 14], {"delay": 0}),
            ),
            FadeOut(right_ang_phi_theta, right_ang_arrow),
        )

        sin_ang = MathTex(r"\sin{(\theta)} = \frac{L}{d_x}").move_to(
            cos_ang_filled, LEFT
        )

        self.play(
            TransformByGlyphMap(
                cos_ang_90,
                sin_ang,
                ([0, 1, 2], [0, 1, 2]),
                ([3], [3]),
                ([4, 5, 6, 7], ShrinkToCenter),
                ([8], [4]),
                ([9, 10], [5, 6]),
                ([11, 12, 13, 14], [7, 8, 9, 10]),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        L_eqn = MathTex(r"L = d_x \cdot \sin{(\theta)}").move_to(cos_ang_filled, LEFT)

        self.play(
            TransformByGlyphMap(
                sin_ang,
                L_eqn,
                ([7], [0], {"path_arc": PI / 2}),
                ([0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10], {"path_arc": PI / 2}),
                (GrowFromCenter, [4], {"delay": 0.3}),
                ([8], ShrinkToCenter),
                ([9, 10], [2, 3], {"path_arc": -PI}),
                ([6], [1]),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        wl_eqn = MathTex(r"\lambda = \frac{c}{f}").next_to(sin_ang, DOWN)

        self.play(LaggedStart(*[GrowFromCenter(m) for m in wl_eqn[0]], lag_ratio=0.1))

        self.wait(0.5)

        phase_delta_eqn = MathTex(
            r"\Delta \phi = \frac{L}{\lambda} \cdot 2 \pi"
        ).move_to(sin_ang, LEFT)

        fade_eqn_group = (
            Group(L_eqn.copy(), wl_eqn.copy())
            .arrange(DOWN, aligned_edge=RIGHT)
            .next_to(self.camera.frame.get_corner(UR), DL)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    TransformFromCopy(L_eqn[0][0], phase_delta_eqn[0][3]),
                    L_eqn.animate.set_opacity(0.2).move_to(fade_eqn_group[0][0]),
                ),
                *[GrowFromCenter(m) for m in phase_delta_eqn[0][:3]],
                GrowFromCenter(phase_delta_eqn[0][4]),
                AnimationGroup(
                    TransformFromCopy(wl_eqn[0][0], phase_delta_eqn[0][5]),
                    wl_eqn.animate.set_opacity(0.2).move_to(fade_eqn_group[1][0]),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in phase_delta_eqn[0][-3:]],
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_delta_full_eqn = MathTex(
            r"\Delta \phi = \frac{d_x \sin{(\theta)}}{\lambda} \cdot 2 \pi"
        ).move_to(sin_ang, LEFT)

        self.play(
            TransformByGlyphMap(
                phase_delta_eqn,
                phase_delta_full_eqn,
                ([0, 1], [0, 1]),
                ([2], [2]),
                ([3], ShrinkToCenter),
                ([4], [11], {"delay": 0.3}),
                ([5], [12]),
                ([6], [13]),
                ([7, 8], [14, 15]),
                (
                    get_transform_func(L_eqn[0][2:]),
                    [3, 4, 5, 6, 7, 8, 9, 10],
                    {"path_arc": -PI / 5},
                ),
            )
        )

        self.wait(0.5)

        self.play(
            phase_delta_full_eqn[0][3:5]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 3),
            dx_label[0]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 3),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            wl_eqn[0]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_opacity(1),
            phase_delta_full_eqn[0][12]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(DOWN / 3),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            phase_delta_full_eqn[0][9]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 3),
            theta[0]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 3),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                psi,
                psi_angle,
                right_ang,
                wl_eqn,
                L_eqn,
                ax,
                z_label,
                dx,
                right_tri,
                theta,
                theta_angle,
            ),
            self.camera.frame.animate.scale(1 / 0.8),
            phase_delta_full_eqn.animate.to_edge(UP, MED_LARGE_BUFF),
            dx_label.animate.next_to(dx, DOWN),
        )

        antenna_port_left2 = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_left2 = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_left2, UP)
        )
        antenna_left2 = Group(antenna_port_left2, antenna_tri_left2).next_to(
            antenna_left, LEFT, LARGE_BUFF * 3
        )
        antenna_port_left3 = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_left3 = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_left3, UP)
        )
        antenna_left3 = Group(antenna_port_left3, antenna_tri_left3).next_to(
            antenna_left2, LEFT, LARGE_BUFF * 3
        )
        self.next_section(skip_animations=skip_animations(True))

        L_line2 = Line(
            ax.c2p(0, 0),
            ax.c2p(
                np.cos(PI / 2 + angle),
                np.sin(PI / 2 + angle),
            ),
            color=YELLOW,
        )
        L_line2.shift((L_line2.get_start()[0] - antenna_left2.get_top()[0]) * LEFT)
        L_label2 = MathTex("L_2").next_to(L_line2.get_midpoint(), RIGHT)

        L_line3 = Line(
            ax.c2p(0, 0),
            ax.c2p(
                np.cos(PI / 2 + angle) * 1.5,
                np.sin(PI / 2 + angle) * 1.5,
            ),
            color=YELLOW,
        )
        L_line3.shift((L_line3.get_start()[0] - antenna_left3.get_top()[0]) * LEFT)
        L_label3 = MathTex("L_3").next_to(L_line3.get_midpoint(), RIGHT)

        plane_wave_longer = Line(
            antenna_right.get_top()
            - 16 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            antenna_right.get_top()
            + 16 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            color=RX_COLOR,
        ).set_z_index(-2)

        dx_label2 = (
            MathTex("d_x")
            .move_to(dx_label)
            .shift((antenna_right.get_top()[0] - antenna_left.get_top()[0]) * LEFT)
        )
        dx_label3 = (
            MathTex("d_x")
            .move_to(dx_label)
            .shift((antenna_right.get_top()[0] - antenna_left.get_top()[0]) * LEFT * 2)
        )

        delta_phi_1 = MathTex(r"\Delta \phi").next_to(antenna_left, UP).shift(LEFT / 2)
        delta_phi_2 = (
            MathTex(r"2\Delta \phi").next_to(antenna_left2, UP).shift(LEFT / 2)
        )
        delta_phi_3 = (
            MathTex(r"3\Delta \phi_3").next_to(antenna_left3, UP).shift(LEFT / 2)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    Group(antenna_left3, antenna_right).width * 1.2
                ).move_to(Group(antenna_left3, antenna_right), DOWN),
                Transform(plane_wave, plane_wave_longer),
                GrowFromCenter(antenna_left2),
                GrowFromCenter(dx_label2),
                Create(L_line2),
                GrowFromCenter(L_label2),
                GrowFromCenter(antenna_left3),
                GrowFromCenter(dx_label3),
                Create(L_line3),
                GrowFromCenter(L_label3),
                lag_ratio=0.2,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(
                    phase_delta_full_eqn[0][:2], delta_phi_1[0], path_arc=-PI / 3
                ),
                TransformFromCopy(
                    phase_delta_full_eqn[0][:2], delta_phi_2[0][1:], path_arc=-PI / 3
                ),
                GrowFromCenter(delta_phi_2[0][0]),
                TransformFromCopy(
                    phase_delta_full_eqn[0][:2], delta_phi_3[0][1:], path_arc=-PI / 3
                ),
                GrowFromCenter(delta_phi_3[0][0]),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        notebook_img = ImageMobject("./static/2d_taylor_taper.png").scale_to_fit_width(
            config.frame_width * 0.4
        )
        code = (
            Code(
                code="""def compute_af_2d(weights_n, weights_m, d_x, d_y, k_0, u_0, v_0):
    AF_m = np.sum(
        weights_n[:, None, None]
        * np.exp(1j * n[:, None, None] * d_x * k_0 * (U - u_0)),
        axis=0,
    )
    AF_n = np.sum(
        weights_m[:, None, None]
        * np.exp(1j * m[:, None, None] * d_y * k_0 * (V - v_0)),
        axis=0,
    )

    AF = AF_m * AF_n / (M * N)
    return AF""",
                tab_width=4,
                background="window",
                background_stroke_color=WHITE,
                language="Python",
                style="paraiso-dark",
                insert_line_no=True,
            )
            .scale_to_fit_width(config.frame_width * 0.8)
            .next_to(self.camera.frame.get_bottom(), DOWN)
            .shift(DOWN * config.frame_height * 1.5)
        )
        for ln in code.line_numbers:
            ln.set_color(WHITE)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"cfar.ipynb \rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=myTemplate,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).next_to(code, UP, LARGE_BUFF)
        self.add(code, notebook)

        self.play(
            self.camera.frame.animate.scale(0.8)
            .move_to(code, DOWN)
            .shift(DOWN * LARGE_BUFF)
        )

        self.wait(0.5)

        self.play(
            code.animate.scale_to_fit_width(config.frame_width * 0.5)
            .next_to(self.camera.frame.get_left(), RIGHT, MED_LARGE_BUFF)
            .shift(DOWN),
            notebook_img.scale(0.7)
            .next_to(self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF)
            .shift(DOWN + RIGHT * 7)
            .animate.shift(LEFT * 7),
        )

        self.wait(0.5)

        self.play(
            code.animate.shift(LEFT * 10),
            notebook_img.animate.shift(RIGHT * 10),
            FadeOut(notebook),
        )

        self.wait(2)


class Part2Transition(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        antenna_port_left = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_left = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_left, UP)
        )
        antenna_port_right = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri_right = (
            Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port_right, UP)
        )

        antenna_left = Group(antenna_port_left, antenna_tri_left)
        antenna_right = Group(antenna_port_right, antenna_tri_right)
        antennas = (
            Group(antenna_left, antenna_right)
            .arrange(RIGHT, LARGE_BUFF * 20)
            .to_edge(DOWN, -SMALL_BUFF)
        )

        self.play(
            antennas.animate.arrange(RIGHT, LARGE_BUFF * 3).to_edge(DOWN, -SMALL_BUFF)
        )

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .to_edge(UP, MED_LARGE_BUFF)
            .shift(RIGHT * 2)
        )

        tx_up = Line(
            antennas.get_top(),
            [antennas.get_top()[0], cloud.get_bottom()[1], 0],
            color=TX_COLOR,
        )
        tx_to_cloud = Line(antennas.get_top(), cloud.get_bottom(), color=TX_COLOR)

        self.play(Create(tx_up))

        self.wait(0.5)

        self.play(
            ReplacementTransform(tx_up, tx_to_cloud),
            cloud.shift(UP * 5).animate.shift(DOWN * 5),
        )

        self.wait(0.5)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]
        # weight_trackers[n_elem // 2] @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        # patch parameters
        f_patch = 10e9
        lambda_patch_0 = c / f_patch

        epsilon_r = 2.2
        h = 1.6e-3

        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 + 12 * h / lambda_patch_0
        ) ** -0.5
        L = lambda_patch_0 / (2 * np.sqrt(epsilon_eff))
        W = lambda_patch_0 / 2 * np.sqrt(2 / (epsilon_r + 1))
        # /patch parameters

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30
        x_len = config.frame_height * 0.6
        ax = (
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
            .rotate(tx_to_cloud.get_angle())
        )
        ax.shift(antennas.get_top() - ax.c2p(0, 0))

        fnbw = 2 * PI * wavelength_0 / (n_elem * d_x)

        theta_min = VT(0)
        theta_max = VT(0)
        af_opacity = VT(1)

        ep_exp_scale = VT(0)

        def get_ap():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            EP = sinc_pattern(u, 0, L, W, wavelength_0)
            AP = AF * (EP ** (~ep_exp_scale))
            f_AP = interp1d(
                u * PI,
                1.3 * np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None),
                fill_value="extrapolate",
            )
            plot = ax.plot_polar_graph(
                r_func=f_AP,
                theta_range=[~theta_min, ~theta_max, 2 * PI / 200],
                color=TX_COLOR,
                use_smoothing=False,
                stroke_opacity=~af_opacity,
            )
            return plot

        AF_plot = always_redraw(get_ap)

        self.next_section(skip_animations=skip_animations(True))
        self.add(AF_plot)

        self.play(
            FadeOut(tx_to_cloud),
            theta_min @ (-fnbw / 2),
            theta_max @ (fnbw / 2),
            run_time=3,
        )

        self.wait(0.5)

        self.play(theta_min @ (-PI), theta_max @ (PI), run_time=3)

        self.wait(0.5)

        self.play(
            ax.animate.rotate(PI / 2 - tx_to_cloud.get_angle()),
            AnimationGroup(
                *[
                    m.animate.set_stroke(opacity=0.2)
                    for m in [
                        antenna_tri_left,
                        antenna_port_left,
                        antenna_tri_right,
                        antenna_port_right,
                    ]
                ]
            ),
            cloud.animate.shift(UP * 5),
            self.camera.frame.animate.scale(0.9).shift(UP),
        )

        self.wait(0.5)

        ap_label = Tex(
            "Antenna Pattern", r"\ $ = $ Element Pattern $\cdot$ Array Factor"
        ).next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF)

        ap_label[0].save_state()
        self.play(
            LaggedStart(
                *[FadeIn(m) for m in ap_label[0].scale(1.8).set_x(0)], lag_ratio=0.1
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        scan_arrow_ang = VT(0)
        scan_arrow_left = always_redraw(
            lambda: Arrow(
                ax.c2p(0, 0), ax.input_to_graph_point(-~scan_arrow_ang, AF_plot), buff=0
            )
        )
        scan_arrow_right = always_redraw(
            lambda: Arrow(
                ax.c2p(0, 0), ax.input_to_graph_point(~scan_arrow_ang, AF_plot), buff=0
            )
        )

        self.play(GrowArrow(scan_arrow_left), GrowArrow(scan_arrow_right))

        self.wait(0.5)

        self.play(scan_arrow_ang @ PI, run_time=3)
        self.play(FadeOut(scan_arrow_left, scan_arrow_right))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                ap_label[0].animate.restore(),
                GrowFromCenter(ap_label[1][0]),
                GrowFromCenter(ap_label[1][1:8]),
                GrowFromCenter(ap_label[1][8:15]),
                GrowFromCenter(ap_label[1][15]),
                GrowFromCenter(ap_label[1][16:21]),
                GrowFromCenter(ap_label[1][21:]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(ap_label[1][1:15].animate.set_color(YELLOW), af_opacity @ 0.2)

        self.wait(0.5)

        EP_ax_left = (
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
        EP_ax_left.shift(antenna_left.get_top() - EP_ax_left.c2p(0, 0))

        EP_ax_right = (
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
        EP_ax_right.shift(antenna_right.get_top() - EP_ax_right.c2p(0, 0))

        theta = np.linspace(-PI, PI, 500)

        ep_theta_min = VT(0)
        ep_theta_max = VT(0)
        ep_left_opacity = VT(1)
        ep_right_opacity = VT(1)

        def get_plot_ep_func(ax, opacity_vt):
            def plot_ep():
                EP = sinc_pattern(theta, 0, L, W, lambda_patch_0)
                f_EP = interp1d(
                    theta,
                    np.clip(20 * np.log10(np.abs(EP)) - r_min, 0, None),
                    fill_value="extrapolate",
                )
                EP_plot = ax.plot_polar_graph(
                    r_func=f_EP,
                    theta_range=[~ep_theta_min, ~ep_theta_max, 2 * PI / 200],
                    color=TX_COLOR,
                    use_smoothing=False,
                    stroke_opacity=~opacity_vt,
                )

                return EP_plot

            return plot_ep

        EP_plot_left = always_redraw(get_plot_ep_func(EP_ax_left, ep_left_opacity))
        EP_plot_right = always_redraw(get_plot_ep_func(EP_ax_right, ep_right_opacity))

        self.next_section(skip_animations=skip_animations(False))
        self.add(EP_plot_left, EP_plot_right)

        self.play(ep_theta_min @ (-PI / 2), ep_theta_max @ (PI / 2))

        self.wait(0.5)

        self.play(
            ep_left_opacity @ 0.2,
            ep_right_opacity @ 0.2,
            ap_label[1][1:15].animate.set_color(WHITE),
            ap_label[1][16:].animate.set_color(YELLOW),
            af_opacity @ 1,
        )

        self.wait(0.5)

        self.play(ax.animate.rotate(-PI / 3))
        self.play(ax.animate.rotate(2 * PI / 3))
        self.play(ax.animate.rotate(-PI / 3))

        self.wait(0.5)

        taper = signal.windows.taylor(n_elem, nbar=5, sll=23)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n]
                        @ taper[: n_elem // 2][::-1][n],
                        weight_trackers[n_elem // 2 + 1 :][n]
                        @ taper[n_elem // 2 + 1 :][n],
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n] @ 1,
                        weight_trackers[n_elem // 2 + 1 :][n] @ 1,
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            ep_left_opacity @ 1,
            ep_right_opacity @ 1,
            ap_label[1][16:].animate.set_color(WHITE),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    EP_ax_left.animate.set_x(ax.get_x()),
                    ep_left_opacity @ 0,
                    lag_ratio=0.4,
                ),
                LaggedStart(
                    EP_ax_right.animate.set_x(ax.get_x()),
                    ep_right_opacity @ 0,
                    lag_ratio=0.4,
                ),
                ep_exp_scale.animate(run_time=3).set_value(1),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        phased_array_box = (
            Rectangle(
                color=GRAY_BROWN,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                height=config.frame_height * 0.4,
                width=config.frame_width * 0.6,
            )
            .set_z_index(1)
            .next_to(antennas, DOWN, 0)
        )

        gears = SVGMobject("../props/static/Gears.svg").scale(3)
        (red_gear, blue_gear) = gears.scale_to_fit_height(
            phased_array_box.height * 0.8
        ).move_to(phased_array_box)

        gr = 24 / 12

        RA = DecimalNumber(0, 3)
        RV = DecimalNumber(0, 2)
        BV = DecimalNumber(0, 2)

        def Driver(m, dt):
            RV.set_value(RV.get_value() + dt * RA.get_value())
            BV.set_value(-RV.get_value() / gr)
            m.rotate(dt * RV.get_value())

        def Driven(m, dt):
            m.rotate(dt * BV.get_value())

        self.add(gears, phased_array_box)
        red_gear.add_updater(Driver)
        blue_gear.add_updater(Driven)

        RA.set_value(PI / 6)

        # for a, t in AccTime:
        #     # self.add_sound("Click.wav")
        #     self.play(Indicate(RA.set_value(a)), run_time=0.5)
        #     corr = 2 / 60  # missed frame correction
        #     self.wait(t + corr - 0.5)  # -0.5 for run_time=0.5

        scene = Group(phased_array_box, AF_plot, antennas)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(scene.height * 1.1).move_to(
                scene
            ),
            AnimationGroup(
                *[
                    m.animate.set_stroke(opacity=1)
                    for m in [
                        antenna_tri_left,
                        antenna_port_left,
                        antenna_tri_right,
                        antenna_port_right,
                    ]
                ]
            ),
            run_time=2,
        )

        self.wait(6)

        toolbox = (
            ImageMobject("../props/static/toolbox.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .move_to(self.camera.frame.copy().shift(RIGHT * config.frame_width * 2))
        )
        self.add(toolbox)

        self.play(self.camera.frame.animate.shift(RIGHT * config.frame_width * 2))

        self.wait(0.5)

        self.play(toolbox.animate(rate_func=rate_functions.ease_in_back).shift(UP * 8))

        self.wait(0.5)

        mailloux = ImageMobject("../props/static/mailloux.jpg").scale_to_fit_height(
            config.frame_height * 0.7
        )
        adi = ImageMobject("./static/adi_phased_array_article.png").scale_to_fit_height(
            config.frame_height * 0.7
        )
        notebook_sc = ImageMobject("./static/notebook_sc.png").scale_to_fit_height(
            config.frame_height * 0.7
        )
        Group(mailloux, adi, notebook_sc).arrange(RIGHT, MED_LARGE_BUFF).move_to(
            self.camera.frame
        )

        self.play(GrowFromCenter(mailloux))

        self.wait(0.5)

        self.play(GrowFromCenter(adi))

        self.wait(0.5)

        self.play(GrowFromCenter(notebook_sc))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShrinkToCenter(mailloux),
                ShrinkToCenter(adi),
                ShrinkToCenter(notebook_sc),
                lag_ratio=0.2,
            )
        )

        self.wait(2)


# TODO: Re-render
class FourierAnalogy(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))
        max_time = 4
        f_max = 8

        x_len = config.frame_width * 0.8
        y_len = config.frame_height * 0.3
        amp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        f_ax = Axes(
            x_range=[-f_max, f_max, f_max / 4],
            y_range=[0, 2, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        amp_labels = amp_ax.get_x_axis_label(MathTex("t"))
        f_labels = f_ax.get_x_axis_label(MathTex("f"))

        f = VT(5)
        offset = VT(0)
        fs = 200
        amp_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: np.sin(2 * PI * ~f * t) + ~offset,
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )

        amp_plot_group = Group(amp_ax, amp_labels, amp_plot)
        f_plot_group = Group(f_ax, f_labels)
        axes = Group(amp_plot_group, f_plot_group)

        def get_fft():
            N = max_time * fs
            t = np.linspace(0, max_time, N)
            sig = np.sin(2 * PI * ~f * t) + ~offset

            fft_len = N * 8
            sig_fft = fftshift(np.abs(fft(sig, fft_len) / (N / 2)))
            # fft_log = 10 * np.log10(np.abs(fftshift(sig_fft))) + 40
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            f_fft_log = interp1d(freq, sig_fft)
            return f_ax.plot(f_fft_log, x_range=[-f_max, f_max, 1 / fs], color=TX_COLOR)

        f_plot = always_redraw(get_fft)

        # self.add(amp_plot_group, amp_plot, f_plot_group, f_plot)

        ft_method = (
            Tex("Fourier ", "Transform ", "Method", font_size=DEFAULT_FONT_SIZE * 1.8)
            .set_opacity(0)
            .shift(DOWN)
        )

        self.play(
            LaggedStart(
                *[m.animate.shift(UP).set_opacity(1) for m in ft_method], lag_ratio=0.2
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            amp_plot_group.next_to([0, config.frame_height / 2, 0], UP).animate.move_to(
                ORIGIN
            ),
            ft_method.animate.next_to([0, -config.frame_height / 2, 0], DOWN),
        )
        self.remove(ft_method)

        self.wait(0.5)

        f_plot_group.next_to([0, -config.frame_height / 2, 0], DOWN)
        self.add(f_plot)
        self.play(axes.animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        peaks_label = Tex("Peaks").next_to(f_ax.c2p(0, 2.3), RIGHT)
        peak_arrow_right = Arrow(
            peaks_label.get_right(), f_ax.input_to_graph_point(~f, f_plot)
        )
        peak_arrow_left = Arrow(
            peaks_label.get_left(), f_ax.input_to_graph_point(-(~f), f_plot)
        )

        self.play(
            FadeIn(peaks_label),
            GrowArrow(peak_arrow_left),
            GrowArrow(peak_arrow_right),
        )

        self.wait(0.5)

        self.play(FadeOut(peaks_label, peak_arrow_left, peak_arrow_right))

        self.wait(0.5)

        self.play(f @ 0, run_time=2)

        self.wait(0.5)

        self.play(offset @ 1)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        sinc = Tex(r"$\lvert$sinc$\rvert = \lvert \frac{\sin{(x)}}{x} \rvert$").next_to(
            f_ax.c2p(0, 2), RIGHT, MED_LARGE_BUFF
        )

        self.play(GrowFromCenter(sinc[0][1:5]))

        self.wait(0.5)

        self.play(GrowFromCenter(sinc[0][0]), GrowFromCenter(sinc[0][5]))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in sinc[0][6:]],
                lag_ratio=0.15,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        num_samples = 8
        samples = amp_ax.get_vertical_lines_to_graph(
            amp_plot,
            x_range=[0.0625, 1 - 0.0625],
            num_lines=num_samples,
            color=RED,
            line_func=Line,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.8,
        )
        sample_dots = [Dot(sample.get_end(), color=RED) for sample in samples]

        self.play(
            LaggedStart(
                *[
                    LaggedStart(Create(sample), Create(sample_dot), lag_ratio=0.3)
                    for sample, sample_dot in zip(samples, sample_dots)
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.move_to(
                amp_ax.get_bottom() + DOWN * MED_LARGE_BUFF, DOWN
            )
        )

        self.wait(0.5)

        antennas = Group()
        for sample in samples:
            antenna_port = Line(DOWN, UP, color=WHITE).set_x(sample.get_x())
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)

        antennas.next_to(amp_ax, UP, LARGE_BUFF * 1.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(antenna) for antenna in antennas], lag_ratio=0.2
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ts_line = Line(sample_dots[0].get_center(), sample_dots[1].get_center()).shift(
            UP / 3
        )
        ts_line_l = Line(ts_line.get_start() + DOWN / 8, ts_line.get_start() + UP / 8)
        ts_line_r = Line(ts_line.get_end() + DOWN / 8, ts_line.get_end() + UP / 8)
        ts_label = MathTex("T_s").next_to(ts_line, UP)

        self.play(
            LaggedStart(
                Create(ts_line_l),
                Create(ts_line),
                FadeIn(ts_label, shift=DOWN),
                Create(ts_line_r),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        dx_line = Line(antennas[0].get_top(), antennas[1].get_top()).shift(UP / 3)
        dx_line_l = Line(dx_line.get_start() + DOWN / 8, dx_line.get_start() + UP / 8)
        dx_line_r = Line(dx_line.get_end() + DOWN / 8, dx_line.get_end() + UP / 8)
        dx_label = MathTex("d_x").next_to(dx_line, UP)

        self.play(
            LaggedStart(
                Create(dx_line_l),
                Create(dx_line),
                FadeIn(dx_label, shift=DOWN),
                Create(dx_line_r),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        temporal_group = Group(
            amp_ax,
            ts_label,
            amp_labels,
            samples,
            *sample_dots,
            ts_line,
            ts_line_l,
            ts_line_r,
        )
        temporal_box = SurroundingRectangle(temporal_group)
        temporal_label = Tex(r"Temporal\\Sampling").next_to(temporal_box, LEFT)

        spatial_group = Group(antennas, dx_label, dx_line, dx_line_l, dx_line_r)
        spatial_box = SurroundingRectangle(spatial_group)
        spatial_label = Tex(r"Spatial\\Sampling").next_to(spatial_box, LEFT)

        sampling_labels = Group(
            temporal_box, temporal_label, spatial_box, spatial_label
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    sampling_labels.width * 1.1
                )
                .move_to(sampling_labels)
                .shift(UP / 3),
                Create(temporal_box),
                FadeIn(temporal_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(spatial_box),
                FadeIn(spatial_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        antennas_new_y = temporal_group.get_y()
        self.play(
            self.camera.frame.animate.restore(),
            Group(temporal_group, temporal_box, temporal_label).animate.shift(
                LEFT * 20
            ),
            Uncreate(spatial_box),
            FadeOut(spatial_label),
            spatial_group.animate.set_y(antennas_new_y).shift(DOWN / 3),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(*[ShrinkToCenter(m) for m in antennas], lag_ratio=0.1),
            Uncreate(f_plot),
            Uncreate(f_ax),
            Uncreate(amp_plot),
            Uncreate(amp_ax),
            Uncreate(dx_line),
            Uncreate(dx_line_r),
            Uncreate(dx_line_l),
            FadeOut(amp_labels, f_labels, dx_label, sinc),
        )

        # self.next_section(skip_animations=skip_animations(True))
        # self.wait(0.5)

        # dft_eqn = MathTex(
        #     r"X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-j 2 \pi \frac{k}{N} n}",
        #     font_size=DEFAULT_FONT_SIZE * 1.5,
        # )

        # self.play(
        #     LaggedStart(
        #         GrowFromCenter(dft_eqn[0][:2]),
        #         GrowFromCenter(dft_eqn[0][2]),
        #         GrowFromCenter(dft_eqn[0][3:10]),
        #         GrowFromCenter(dft_eqn[0][10:12]),
        #         GrowFromCenter(dft_eqn[0][12]),
        #         GrowFromCenter(dft_eqn[0][13]),
        #         GrowFromCenter(dft_eqn[0][14:16]),
        #         GrowFromCenter(dft_eqn[0][16]),
        #         GrowFromCenter(dft_eqn[0][17]),
        #         GrowFromCenter(dft_eqn[0][18:21]),
        #         GrowFromCenter(dft_eqn[0][21]),
        #         lag_ratio=0.1,
        #     )
        # )

        # self.wait(0.5)

        # what_want = Tex(
        #     "what do we actually want?", font_size=DEFAULT_FONT_SIZE * 1.5
        # ).next_to([0, config.frame_height / 2, 0], UP)

        # self.play(Group(what_want, dft_eqn).animate.arrange(DOWN, LARGE_BUFF))

        # self.wait(0.5)

        # self.play(
        #     LaggedStart(*[ShrinkToCenter(m) for m in dft_eqn[0][::-1]], lag_ratio=0.1),
        #     LaggedStart(
        #         *[ShrinkToCenter(m) for m in what_want[0][::-1]], lag_ratio=0.1
        #     ),
        # )

        self.wait(2)


class CircularCoords(MovingCameraScene):
    def construct(self):
        n_elem = 17  # Must be odd
        weight_trackers = [VT(0) for _ in range(n_elem)]
        weight_trackers[n_elem // 2] @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30
        x_len = config.frame_height * 0.6
        ax = Axes(
            x_range=[r_min, -r_min, r_min / 8],
            y_range=[r_min, -r_min, r_min / 8],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=x_len,
        ).rotate(PI / 2)

        def get_af():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            f_AF = interp1d(
                u * PI,
                np.clip(20 * np.log10(np.abs(AF)) - r_min, 0, None),
                fill_value="extrapolate",
            )
            plot = ax.plot_polar_graph(
                r_func=f_AF, theta_range=[-PI, PI, 2 * PI / 200], color=TX_COLOR
            )
            return plot

        AF_plot = always_redraw(get_af)

        self.next_section(skip_animations=skip_animations(True))
        # self.add(ax, AF_plot)

        self.play(Create(ax), Create(AF_plot))

        self.wait(0.5)

        arrow_theta = VT(0)
        arrow = always_redraw(
            lambda: Arrow(ax.c2p(0, 0), ax.input_to_graph_point(~arrow_theta, AF_plot))
        )

        self.play(GrowArrow(arrow))

        self.wait(0.5)

        self.play(arrow_theta @ (2 * PI), run_time=2)

        self.wait(0.5)

        self.play(FadeOut(arrow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n] @ 1,
                        weight_trackers[n_elem // 2 + 1 :][n] @ 1,
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        ant_pattern_label = Tex(
            r"Antenna\\Pattern", font_size=DEFAULT_FONT_SIZE * 1.5
        ).next_to(ax, LEFT, LARGE_BUFF * 3)

        self.play(
            self.camera.frame.animate.move_to(Group(ant_pattern_label, ax, AF_plot)),
            Write(ant_pattern_label),
        )

        self.wait(0.5)

        ax_rad_labels = Group(
            *[
                MathTex(s).move_to(pos)
                for s, pos in [
                    (r"0", ax.c2p(-r_min, 0) + UP / 2),
                    (r"\frac{\pi}{2}", ax.c2p(0, -r_min) + LEFT / 2),
                    (r"\pi", ax.c2p(r_min, 0) + DOWN / 2),
                    (r"\frac{3 \pi}{2}", ax.c2p(0, r_min) + RIGHT / 2),
                ]
            ]
        )
        ax_deg_labels = Group(
            *[
                MathTex(s).move_to(pos)
                for s, pos in [
                    (r"0^\circ", ax.c2p(-r_min, 0) + UP / 2),
                    (r"90^\circ", ax.c2p(0, -r_min) + LEFT / 2),
                    (r"180^\circ", ax.c2p(r_min, 0) + DOWN / 2),
                    (r"270^\circ", ax.c2p(0, r_min) + RIGHT / 2),
                ]
            ]
        )

        self.play(
            LaggedStart(
                *[GrowFromCenter(label) for label in ax_rad_labels],
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    Transform(rad, deg)
                    for rad, deg in zip(ax_rad_labels, ax_deg_labels)
                ],
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        dft_of = Tex(
            r"$\mathcal{F}(?) \rightarrow$", font_size=DEFAULT_FONT_SIZE * 1.5
        ).move_to(ant_pattern_label)

        dft_y = ant_pattern_label.get_y()

        self.play(
            ant_pattern_label.animate.next_to(
                [ant_pattern_label.get_x(), self.camera.frame.get_top()[1], 0], UP
            ),
            dft_of.next_to(
                [dft_of.get_x(), self.camera.frame.get_bottom()[1], 0], DOWN
            ).animate.set_y(dft_y),
        )

        self.wait(0.5)

        dft_of_ant = Tex(
            r"$\mathcal{F}(\text{antenna}?) \rightarrow$",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(dft_of)

        self.play(
            TransformByGlyphMap(
                dft_of,
                dft_of_ant,
                ([0, 1], [0, 1]),
                (GrowFromCenter, [2, 3, 4, 5, 6, 7, 8], {"delay": 0.3}),
                ([2, 3, 4], [9, 10, 11]),
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.shift(
                UP * (ax_deg_labels.get_bottom() - self.camera.frame.get_top()) + DOWN
            )
        )

        # two_pi_arrow = CurvedArrow(
        #     ax.c2p(0, -r_min) + UP / 2, ax.c2p(0, -r_min) + UP / 2, angle=TAU / 4
        # )

        # self.play(Create(two_pi_arrow))

        # self.next_section(skip_animations=skip_animations(False))
        # self.wait(0.5)

        # self.play(steering_angle @ (10))

        self.wait(2)


class FourierExplanation(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        max_time = 4
        f_max = 8

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.3
        amp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        f_ax = Axes(
            x_range=[-f_max, f_max, f_max / 4],
            y_range=[0, 1, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        amp_labels = amp_ax.get_x_axis_label(MathTex("t"))
        f_labels = f_ax.get_x_axis_label(MathTex("f"))
        x_n_label = MathTex("x[n]").next_to(amp_ax, UP)
        X_k_label = MathTex("X[k]").next_to(f_ax, DOWN)
        amp_plot_group = Group(amp_ax, amp_labels, x_n_label)
        f_plot_group = Group(f_ax, f_labels, X_k_label)
        axes = (
            Group(amp_plot_group, f_plot_group).arrange(DOWN, LARGE_BUFF).to_edge(LEFT)
        )

        f = VT(5)
        offset = VT(0)
        fs = 200
        amp_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: np.sin(2 * PI * ~f * t) + ~offset,
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )

        fft_f = VT(f_max)

        def get_fft(log=False):
            N = max_time * fs
            t = np.linspace(0, max_time, N)
            sig = np.sin(2 * PI * ~f * t) + ~offset

            fft_len = N * 8
            sig_fft = fftshift(np.abs(fft(sig, fft_len) / (N)))
            # sig_fft = (1 - ~log_interp) * sig_fft + ~log_interp * np.log10(
            #     np.abs(sig_fft)
            # )
            if log:
                sig_fft = np.clip(10 * np.log10(np.abs(sig_fft)) + 25, 0, None)
                sig_fft /= sig_fft.max()
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            f_fft = interp1d(freq, sig_fft)
            return f_ax.plot(f_fft, x_range=[-~fft_f, ~fft_f, 1 / fs], color=TX_COLOR)

        f_plot = always_redraw(get_fft)

        self.play(
            Create(amp_ax),
            Create(f_ax),
            GrowFromCenter(amp_labels),
            GrowFromCenter(f_labels),
            GrowFromCenter(x_n_label),
            GrowFromCenter(X_k_label),
        )

        self.wait(0.5)

        dft_arrow = CurvedArrow(
            amp_ax.get_right(), f_ax.get_right(), angle=-PI / 3
        ).shift(RIGHT)
        dft_label = MathTex(r"\mathcal{F}(x[n])").next_to(dft_arrow)

        self.play(
            Create(amp_plot), Create(f_plot), Create(dft_arrow), FadeIn(dft_label)
        )

        self.wait(0.5)

        wave_amp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        wave_amp_labels = wave_amp_ax.get_x_axis_label(MathTex("t"))
        wave_eqn = MathTex(r"e^{-j 2 \pi f t}").next_to(wave_amp_ax, UP)
        wave_amp_plot_group = Group(wave_amp_ax, wave_amp_labels, wave_eqn).next_to(
            self.camera.frame.get_corner(UR), DL, LARGE_BUFF
        )

        wave_f = VT(2)
        wave_offset = VT(0)

        product_amp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        product_amp_labels = product_amp_ax.get_x_axis_label(MathTex("t"))
        product_amp_plot_group = Group(product_amp_ax, product_amp_labels).next_to(
            self.camera.frame.get_corner(UR), DL, LARGE_BUFF
        )

        plot_group = Group(
            amp_plot_group.copy(),
            wave_amp_plot_group,
            product_amp_plot_group,
            f_plot_group.copy(),
        ).arrange_in_grid(2, 2, (LARGE_BUFF, LARGE_BUFF * 2))

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                AnimationGroup(
                    self.camera.frame.animate.scale_to_fit_width(
                        plot_group.width * 1.1
                    ).move_to(plot_group),
                    Uncreate(dft_arrow),
                    ShrinkToCenter(dft_label),
                ),
                AnimationGroup(
                    amp_plot_group.animate.move_to(plot_group[0]),
                    f_plot_group.animate.move_to(plot_group[3]),
                ),
                lag_ratio=0.3,
            )
        )
        # self.play(
        #     Group(amp_ax, amp_labels, x_n_label).animate.next_to(
        #         self.camera.frame.get_corner(UL), DR, LARGE_BUFF
        #     ),
        #     Group(f_ax, f_labels, X_k_label).animate.next_to(
        #         self.camera.frame.get_corner(DR), UL, LARGE_BUFF
        #     ),
        # )

        self.wait(0.5)

        self.play(
            wave_amp_plot_group.shift(UP * 8).animate.shift(DOWN * 8),
            fft_f @ 0,
        )

        self.wait(0.5)

        wave_amp_plot = always_redraw(
            lambda: wave_amp_ax.plot(
                lambda t: np.sin(2 * PI * ~wave_f * t) + ~wave_offset,
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )

        self.play(Create(wave_amp_plot))

        self.wait(0.5)

        self.play(product_amp_plot_group.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        product_amp_plot = always_redraw(
            lambda: product_amp_ax.plot(
                lambda t: (np.sin(2 * PI * ~wave_f * t) + ~wave_offset)
                * (np.sin(2 * PI * ~f * t) + ~offset),
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )

        mult = MathTex(r"x[n] \cdot e^{-j 2 \pi f t}").next_to(product_amp_ax, DOWN)
        product_amp_plot_group.add(mult)

        self.play(
            LaggedStart(
                TransformFromCopy(x_n_label[0], mult[0][:4]),
                GrowFromCenter(mult[0][4]),
                TransformFromCopy(wave_eqn[0], mult[0][5:]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(Create(product_amp_plot))

        self.wait(0.5)

        num_samples = 16
        x_n_samples = always_redraw(
            lambda: amp_ax.get_vertical_lines_to_graph(
                amp_plot,
                x_range=[0.0625, 1 - 0.0625],
                num_lines=num_samples,
                color=RED,
                line_func=Line,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
            )
        )
        x_n_sample_dots = always_redraw(
            lambda: Group(*[Dot(sample.get_end(), color=RED) for sample in x_n_samples])
        )

        product_samples = Group(
            *[
                Line(
                    product_amp_ax.c2p(x, 0),
                    product_amp_ax.input_to_graph_point(x, product_amp_plot),
                    color=RED
                    if product_amp_ax.input_to_graph_coords(x, product_amp_plot)[1] > 0
                    else PURPLE,
                )
                for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
            ]
        )
        product_sample_dots = Group(
            *[
                Dot(
                    product_amp_ax.input_to_graph_point(x, product_amp_plot),
                    color=RED
                    if product_amp_ax.input_to_graph_coords(x, product_amp_plot)[1] > 0
                    else PURPLE,
                )
                for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
            ]
        )

        x_n_samples_pos = Group()
        x_n_samples_neg = Group()
        for sample, dot in zip(x_n_samples, x_n_sample_dots):
            if sample.get_start()[1] > sample.get_end()[1]:
                sample.set_color(PURPLE)
                dot.set_color(PURPLE)
                x_n_samples_neg.add(Group(sample, dot))
            else:
                x_n_samples_pos.add(Group(sample, dot))

        self.play(
            LaggedStart(
                *[
                    LaggedStart(Create(sample), Create(sample_dot), lag_ratio=0.3)
                    for sample, sample_dot in zip(x_n_samples, x_n_sample_dots)
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        # self.add(product_samples, product_sample_dots)

        self.play(
            *[
                AnimationGroup(Create(sample), Create(sample_dot), lag_ratio=0.3)
                for sample, sample_dot in zip(product_samples, product_sample_dots)
            ],
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        product_samples_pos = Group()
        product_samples_neg = Group()
        for sample, dot in zip(product_samples, product_sample_dots):
            if sample.get_start()[1] > sample.get_end()[1]:
                sample.set_color(PURPLE)
                dot.set_color(PURPLE)
                product_samples_neg.add(Group(sample, dot))
            else:
                product_samples_pos.add(Group(sample, dot))

        samples_pos_stacked = (
            Group(*[sample.copy() for sample in product_samples_pos])
            .arrange(UP, 0)
            .next_to(f_ax.c2p(~wave_f), UP, 0)
        )
        samples_neg_stacked = (
            Group(*[sample.copy() for sample in product_samples_neg])
            .arrange(DOWN, 0)
            .next_to(f_ax.c2p(~wave_f), DOWN, 0)
        )

        self.play(
            LaggedStart(
                *[
                    ReplacementTransform(sample, sample_stacked)
                    for sample, sample_stacked in zip(
                        product_samples_pos, samples_pos_stacked
                    )
                ],
                lag_ratio=0.15,
            ),
            LaggedStart(
                *[
                    ReplacementTransform(sample, sample_stacked)
                    for sample, sample_stacked in zip(
                        product_samples_neg, samples_neg_stacked
                    )
                ],
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        samples_stacked = Group(samples_pos_stacked, samples_neg_stacked)

        sample_at_wave_f_line = always_redraw(
            lambda: Line(
                f_ax.c2p(~wave_f, 0),
                f_ax.input_to_graph_point(~wave_f, f_plot),
                color=RED
                if f_ax.input_to_graph_coords(~wave_f, f_plot)[1] > 0
                else PURPLE,
            )
        )
        sample_at_wave_f_dot = always_redraw(
            lambda: Dot(
                f_ax.input_to_graph_point(~wave_f, f_plot),
                color=RED
                if f_ax.input_to_graph_coords(~wave_f, f_plot)[1] > 0
                else PURPLE,
            )
        )
        sample_at_wave_f = Group(sample_at_wave_f_line, sample_at_wave_f_dot)

        self.play(
            ShrinkToCenter(samples_stacked),
            Create(sample_at_wave_f_line),
            Create(sample_at_wave_f_dot),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Indicate(wave_amp_plot))

        self.wait(0.5)

        self.play(Indicate(product_amp_plot))

        f_rad_xlabels = Group(
            MathTex(r"-\pi").next_to(f_ax.c2p(-f_max, 0), DOWN),
            MathTex(r"\frac{-\pi}{2}").next_to(f_ax.c2p(-f_max / 2, 0), DOWN),
            MathTex(r"\frac{\pi}{2}").next_to(f_ax.c2p(f_max / 2, 0), DOWN),
            MathTex(r"\pi").next_to(f_ax.c2p(f_max, 0), DOWN),
        )

        f_fs_xlabels = Group(
            MathTex(r"-\frac{f_s}{2}").next_to(f_ax.c2p(-f_max, 0), DOWN),
            MathTex(r"-\frac{f_s}{4}").next_to(f_ax.c2p(-f_max / 2, 0), DOWN),
            MathTex(r"\frac{f_s}{4}").next_to(f_ax.c2p(f_max / 2, 0), DOWN),
            MathTex(r"\frac{f_s}{2}").next_to(f_ax.c2p(f_max, 0), DOWN),
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in f_fs_xlabels], lag_ratio=0.2)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    ReplacementTransform(fs, rad)
                    for rad, fs in zip(f_rad_xlabels, f_fs_xlabels)
                ],
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        def get_line(x, ax=product_amp_ax, plot=product_amp_plot):
            def updater():
                p0 = ax.c2p(x, 0)
                p1 = ax.input_to_graph_point(x, plot)
                c1 = ax.input_to_graph_coords(x, plot)
                color = RED if c1[1] >= 0 else PURPLE
                line = Line(p0, p1).set_color(color)
                return line

            return updater

        def get_dot(x, ax=product_amp_ax, plot=product_amp_plot):
            def updater():
                p1 = ax.input_to_graph_point(x, plot)
                c1 = ax.input_to_graph_coords(x, plot)
                color = RED if c1[1] >= 0 else PURPLE
                dot = Dot(p1).set_color(color)
                return dot

            return updater

        product_samples_new = [
            always_redraw(get_line(x))
            for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        ]
        product_sample_dots_new = [
            always_redraw(get_dot(x))
            for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        ]
        # for x in np.linspace(0.0625, 1 - 0.0625, num_samples):
        #     line_updater = get_line(x)
        #     line = always_redraw(line_updater)
        #     product_samples_new.append(line)

        # x = 0.0625 * 2
        # product_sample_dots_new = always_redraw(
        #     lambda: Dot(
        #         product_amp_ax.input_to_graph_point(x, product_amp_plot),
        #         color=RED
        #         if product_amp_ax.input_to_graph_coords(x, product_amp_plot)[1] > 0
        #         else PURPLE,
        #     )
        # )

        self.play(
            *[Create(m) for m in product_samples_new],
            *[Create(m) for m in product_sample_dots_new],
        )

        self.wait(0.5)

        self.play(wave_f @ 0, run_time=3)

        self.wait(0.5)

        self.play(wave_f @ f_max, fft_f @ f_max, run_time=10)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        x_n_samples_new = [
            always_redraw(get_line(x, ax=amp_ax, plot=amp_plot))
            for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        ]
        x_n_sample_dots_new = [
            always_redraw(get_dot(x, ax=amp_ax, plot=amp_plot))
            for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        ]

        self.add(*x_n_samples_new, *x_n_sample_dots_new)
        self.remove(*x_n_samples, *x_n_sample_dots)

        sample_at_wave_f_dot_new = always_redraw(
            lambda: Dot(
                f_ax.input_to_graph_point(~wave_f, f_plot),
                color=RED
                if f_ax.input_to_graph_coords(~wave_f, f_plot)[1] > 0
                else PURPLE,
            )
        )
        self.add(sample_at_wave_f_dot_new)
        self.remove(sample_at_wave_f_dot)

        self.play(
            LaggedStart(
                AnimationGroup(wave_f @ 0, fft_f @ 0),
                f @ 0,
                offset @ 1,
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(wave_f @ f_max, fft_f @ f_max, run_time=3)

        self.wait(0.5)

        f_plot.save_state()
        log_f_plot = get_fft(log=True)
        self.play(ReplacementTransform(f_plot, log_f_plot), run_time=2)

        self.wait(0.5)

        fft_group = Group(f_ax, f_rad_xlabels, log_f_plot, X_k_label, f_labels)
        self.play(
            FadeOut(
                amp_plot_group,
                wave_amp_plot_group,
                product_amp_plot_group,
                amp_plot,
                wave_amp_plot,
                product_amp_plot,
                mult,
                wave_eqn,
                *x_n_samples_new,
                *product_samples_new,
                *product_sample_dots_new,
                *x_n_sample_dots_new,
                sample_at_wave_f_dot_new,
            ),
            self.camera.frame.animate.scale_to_fit_width(config.frame_width).move_to(
                ORIGIN
            ),
            fft_group.animate.scale_to_fit_width(config.frame_width * 0.8).move_to(
                ORIGIN
            ),
        )

        self.wait(2)


class AFToPolar(Scene):
    def construct(self):
        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]
        weight_trackers[n_elem // 2] @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30
        x_len = config.frame_height * 0.79  # i have no idea
        y_len = config.frame_height * 0.6
        ax = Axes(
            x_range=[-PI, PI, 0.5],
            y_range=[0, -r_min, 10],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        polar_ax = Axes(
            x_range=[r_min, -r_min, r_min / 8],
            y_range=[r_min, -r_min, r_min / 8],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).rotate(PI / 2)
        polar_ax.shift(ax.c2p(0, 0) - polar_ax.c2p(0, 0))

        def get_af():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = np.clip(
                20 * np.log10(np.abs(compute_af_1d(weights, d_x, k_0, u, u_0))) - r_min,
                0,
                None,
            )
            f_AF = interp1d(u * PI, AF, fill_value="extrapolate")
            return f_AF

        def cartesian_to_polar(point):
            r = point[1]
            theta = point[0]
            return [r * np.cos(-theta + PI / 2), r * np.sin(theta + PI / 2), 0]

        def get_cart_plot():
            f_AF = get_af()
            plot = ax.plot(f_AF, x_range=[-PI, PI, 1 / 200], color=TX_COLOR)
            return plot

        def get_polar_updater(ref):
            def get_polar_plot():
                cart_plot = get_cart_plot()
                plot = (
                    cart_plot.move_to(ORIGIN, DOWN)
                    .apply_function(cartesian_to_polar)
                    .move_to(cart_plot, UP)
                    .move_to(ref, UP)
                )
                return plot

            return get_polar_plot

        AF_plot = get_cart_plot()
        AF_polar_plot = always_redraw(get_polar_updater(AF_plot.copy()))

        self.add(ax, AF_plot)

        self.play(Transform(AF_plot, AF_polar_plot), run_time=2)

        self.wait(0.5)
        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        right_theta_0 = VT(-fnbw * 2)
        right_theta_1 = VT(0)
        left_theta_0 = VT(0)
        left_theta_1 = VT(fnbw * 2)

        r_min_amp = VT(1.55)
        shade_left = always_redraw(
            lambda: polar_ax.plot_polar_graph(
                lambda theta: -r_min * ~r_min_amp,
                theta_range=[~left_theta_0, ~left_theta_1],
                stroke_opacity=0,
            )
        )
        shade_right = always_redraw(
            lambda: polar_ax.plot_polar_graph(
                lambda theta: -r_min * ~r_min_amp,
                theta_range=[~right_theta_0, ~right_theta_1],
                stroke_opacity=0,
            )
        )

        ap_left = always_redraw(
            lambda: ArcPolygon(
                shade_left.get_start(),
                shade_left.get_end(),
                polar_ax.c2p(0, 0),
                fill_opacity=0.3,
                fill_color=BLUE,
                stroke_width=0,
                arc_config=[
                    {
                        "angle": (~right_theta_1 - ~right_theta_0),
                        "stroke_opacity": 0,
                        "stroke_width": 0,
                    },
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                ],
            )
        )
        ap_right = always_redraw(
            lambda: ArcPolygon(
                shade_right.get_start(),
                shade_right.get_end(),
                polar_ax.c2p(0, 0),
                fill_opacity=0.3,
                fill_color=BLUE,
                stroke_width=0,
                arc_config=[
                    {
                        "angle": (~left_theta_1 - ~left_theta_0),
                        "stroke_opacity": 0,
                        "stroke_width": 0,
                    },
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                ],
            )
        )

        main_lobe_label = Tex("Main lobe").next_to(
            ax.input_to_graph_point(0, AF_plot), UP
        )
        side_lobe_left_label = Tex("Side lobes").next_to(
            ax.input_to_graph_point(-PI / 3, AF_polar_plot), UL, LARGE_BUFF
        )
        side_lobe_right_label = Tex("Side lobes").next_to(
            ax.input_to_graph_point(PI / 3, AF_polar_plot), UR, LARGE_BUFF
        )

        self.add(shade_right, shade_left)
        self.play(FadeIn(ap_left, ap_right))

        self.wait(0.5)

        self.play(FadeIn(main_lobe_label))

        self.wait(0.5)

        self.play(
            right_theta_0 @ (-PI),
            right_theta_1 @ (-fnbw * 2),
            left_theta_1 @ (PI),
            left_theta_0 @ (fnbw * 2),
            r_min_amp @ (0.95),
        )

        self.wait(0.5)

        self.play(FadeIn(side_lobe_left_label, side_lobe_right_label))

        self.wait(2)


class FourierTransformPolar(Scene):
    def construct(self):
        x_len = config.frame_height * 0.6
        ax = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=x_len,
        ).to_edge(LEFT)

        X_N_COLOR = GREEN
        EXP_COLOR = YELLOW
        PRODUCT_COLOR = BLUE

        f_x_n = 3
        f_exp = VT(1)
        theta_tracker = VT(0)

        x_n_plot = always_redraw(
            lambda: ax.plot_polar_graph(
                r_func=lambda theta: np.sin(f_x_n * theta),
                theta_range=[0, ~theta_tracker, 2 * PI / 200],
                color=X_N_COLOR,
            )
        )
        exp_plot = always_redraw(
            lambda: ax.plot_polar_graph(
                r_func=lambda theta: np.sin(~f_exp * theta),
                theta_range=[0, ~theta_tracker, 2 * PI / 200],
                color=EXP_COLOR,
            )
        )
        product_plot = always_redraw(
            lambda: ax.plot_polar_graph(
                r_func=lambda theta: np.sin(f_x_n * theta) * np.sin(~f_exp * theta),
                theta_range=[0, ~theta_tracker, 2 * PI / 200],
                color=PRODUCT_COLOR,
            )
        )

        x_n_line = always_redraw(
            lambda: Line(
                ax.c2p(0, 0),
                ax.input_to_graph_point(~theta_tracker, x_n_plot),
                color=X_N_COLOR,
            )
        )
        exp_line = always_redraw(
            lambda: Line(
                ax.c2p(0, 0),
                ax.input_to_graph_point(~theta_tracker, exp_plot),
                color=EXP_COLOR,
            )
        )
        product_line = always_redraw(
            lambda: Line(
                ax.c2p(0, 0),
                ax.input_to_graph_point(~theta_tracker, product_plot),
                color=PRODUCT_COLOR,
            )
        )
        x_n_dot = always_redraw(
            lambda: Dot(
                ax.input_to_graph_point(~theta_tracker, x_n_plot), color=X_N_COLOR
            )
        )
        exp_dot = always_redraw(
            lambda: Dot(
                ax.input_to_graph_point(~theta_tracker, exp_plot), color=EXP_COLOR
            )
        )
        product_dot = always_redraw(
            lambda: Dot(
                ax.input_to_graph_point(~theta_tracker, product_plot),
                color=PRODUCT_COLOR,
            )
        )

        sine_ax = Axes(
            x_range=[0, 4 * PI, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=config.frame_width * 0.7,
            y_length=ax.height,
        ).next_to(ax, RIGHT, 0)
        sine_plot = always_redraw(
            lambda: sine_ax.plot(
                lambda theta: np.sqrt(
                    np.cos(ax.input_to_graph_coords(theta, product_plot)[1]) ** 2
                    + (1j * np.sin(ax.input_to_graph_coords(theta, product_plot)[1]))
                    ** 2
                ),
                x_range=[0, ~theta_tracker, 1 / 100],
                color=YELLOW,
                use_smoothing=False,
            )
        )

        sine_line = always_redraw(
            lambda: DashedLine(
                ax.c2p(np.cos(~theta_tracker), np.sin(~theta_tracker)),
                sine_ax.input_to_graph_point(~theta_tracker, sine_plot),
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.add(
            ax,
            x_n_line,
            exp_line,
            product_line,
            x_n_dot,
            exp_dot,
            product_dot,
            x_n_plot,
            exp_plot,
            product_plot,
            sine_plot,
            # sine_line,
        )

        self.play(theta_tracker @ (2 * PI), run_time=5, rate_func=rate_functions.linear)


class Equation2D(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        fourier_eqn_full = MathTex(
            r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi \frac{k}{N} n}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        fourier_eqn = fourier_eqn_full[0][5:]
        fourier_x = fourier_eqn.get_x()
        fourier_eqn.move_to(ORIGIN)
        self.play(
            LaggedStart(*[GrowFromPoint(m, ORIGIN) for m in fourier_eqn], lag_ratio=0.1)
        )

        self.wait(0.5)

        antennas = Group()
        for _ in range(4):
            antenna_port = Line(DOWN / 4, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)

        antennas.arrange(RIGHT, MED_LARGE_BUFF).next_to(
            [-config.frame_width / 2, 0, 0], LEFT
        )

        self.play(Group(antennas, fourier_eqn).animate.arrange(RIGHT, LARGE_BUFF))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        antennas_x = antennas.get_x()
        antennas_2d = Group(
            *[
                antennas.copy().next_to([antennas_x, config.frame_height / 2, 0], UP)
                for _ in range(3)
            ]
        )
        self.add(*antennas_2d)
        antennas_2d.add(antennas)

        self.play(antennas_2d.animate.arrange(DOWN, MED_LARGE_BUFF).set_x(antennas_x))

        self.wait(0.5)

        self.play(
            antennas_2d.animate.next_to([-config.frame_width / 2, 0, 0], LEFT),
            fourier_eqn.animate.move_to(ORIGIN),
        )
        self.remove(antennas)

        self.wait(0.5)

        ref_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=config.frame_width * 0.5,
                y_length=config.frame_height * 0.3,
            )
            .to_edge(DOWN, LARGE_BUFF)
            .shift(RIGHT * 3)
        )

        ref_f = VT(3)
        ref_sig = ref_ax.plot(lambda t: np.sin(2 * PI * ~ref_f * t), color=ORANGE)

        self.play(
            LaggedStart(
                *[m.animate.set_color(ORANGE) for m in fourier_eqn[11:20]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(BLUE) for m in fourier_eqn[7:11]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            fourier_eqn_full[0][-4]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(*[FadeIn(m) for m in fourier_eqn_full[0][:5]], lag_ratio=0.1),
            fourier_eqn.animate.set_x(fourier_x),
        )

        self.wait(2)


def compute_af_2d(weights_n, weights_m, d_x, d_y, k_0, u_0, v_0, U, V, M, N, m, n):
    AF_m = np.sum(
        weights_n[:, None, None]
        * np.exp(1j * n[:, None, None] * d_x * k_0 * (U - u_0)),
        axis=0,
    )
    AF_n = np.sum(
        weights_m[:, None, None]
        * np.exp(1j * m[:, None, None] * d_y * k_0 * (V - v_0)),
        axis=0,
    )

    AF = AF_m * AF_n / (M * N)
    return AF


class AF3D(ThreeDScene):
    def construct(self):
        plot_vel = (-20, 20)  # m/s
        plot_range = (0, 40)  # m

        # v_ind = np.where((vels > plot_vel[0]) & (vels < plot_vel[1]))[0]
        # vx = vels[v_ind[0] : v_ind[-1]]

        # n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        # r_ind = np.where((n_ranges > plot_range[0]) & (n_ranges < plot_range[1]))[0]
        # ry = n_ranges[r_ind[0] : r_ind[-1]]

        # rdz = range_doppler[r_ind[0] : r_ind[-1], v_ind[0] : v_ind[-1]]
        N = 9
        M = 9
        nbar_n = 5
        nbar_m = 5
        sll_n = 40
        sll_m = 40

        n = np.arange(N)
        m = np.arange(M)
        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = 0
        u_0 = np.sin(steering_angle * PI / 180)

        d_x = wavelength_0 / 2
        d_y = wavelength_0 / 2

        steering_angle_theta = 0
        steering_angle_phi = 0

        u_0 = np.sin(steering_angle_theta * PI / 180)
        v_0 = np.sin(steering_angle_phi * PI / 180)

        window_n = np.ones(N)
        window_m = np.ones(M)

        # window_n = np.zeros(N)
        # window_n[N // 2] = 1
        # window_m = np.zeros(M)
        # window_m[M // 2] = 1
        window_n = signal.windows.taylor(N)
        window_m = signal.windows.taylor(M)
        U_vis = 20
        V_vis = 20
        u2 = np.linspace(-1, 1, U_vis)
        v2 = np.linspace(-1, 1, V_vis)

        U, V = np.meshgrid(u2, v2, indexing="xy")  # mesh grid of sine space
        # X, Y = np.meshgrid(vx, ry, indexing="xy")

        z_min = -40
        AF = np.clip(
            10
            * np.log10(
                np.abs(
                    compute_af_2d(
                        window_n, window_m, d_x, d_y, k_0, u_0, v_0, U, V, M, N, m, n
                    )
                )
            )
            - z_min,
            0,
            None,
        )

        tck = bisplrep(U, V, AF)

        axes = ThreeDAxes(
            x_range=[-1, 1, 1 / 2],
            y_range=[-1, 1, 1 / 2],
            z_range=[0, -z_min],
            x_length=8,
        )
        surface = always_redraw(
            lambda: Surface(
                # lambda u, v: axes.c2p(
                #     spherical_to_cartesian((bisplev(u, v, tck), u, v))
                # ),
                lambda u, v: axes.c2p(u, v, bisplev(u, v, tck)),
                u_range=[-1, 1],
                v_range=[-1, 1],
                resolution=(U_vis, V_vis),
            )
            .set_z(0)
            .set_style(fill_opacity=1)
            .set_fill_by_value(
                axes=axes, colorscale=[(BLUE, 0), (RED, -z_min - 20)], axis=2
            )
        )

        self.set_camera_orientation(theta=45 * DEGREES, phi=50 * DEGREES, zoom=0.7)

        self.add(surface)


class AF3D_Polar(ThreeDScene):
    def construct(self):
        plot_vel = (-20, 20)  # m/s
        plot_range = (0, 40)  # m

        # v_ind = np.where((vels > plot_vel[0]) & (vels < plot_vel[1]))[0]
        # vx = vels[v_ind[0] : v_ind[-1]]

        # n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        # r_ind = np.where((n_ranges > plot_range[0]) & (n_ranges < plot_range[1]))[0]
        # ry = n_ranges[r_ind[0] : r_ind[-1]]

        # rdz = range_doppler[r_ind[0] : r_ind[-1], v_ind[0] : v_ind[-1]]
        N = 9
        M = 9
        nbar_n = 5
        nbar_m = 5
        sll_n = 40
        sll_m = 40

        n = np.arange(N)
        m = np.arange(M)
        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = 0
        u_0 = np.sin(steering_angle * PI / 180)

        d_x = wavelength_0 / 2
        d_y = wavelength_0 / 2

        steering_angle_theta = 0
        steering_angle_phi = 0

        u_0 = np.sin(steering_angle_theta * PI / 180)
        v_0 = np.sin(steering_angle_phi * PI / 180)

        window_n = np.ones(N)
        window_m = np.ones(M)

        # window_n = np.zeros(N)
        # window_n[N // 2] = 1
        # window_m = np.zeros(M)
        # window_m[M // 2] = 1
        window_n = signal.windows.taylor(N)
        window_m = signal.windows.taylor(M)
        U_vis = 40
        V_vis = 40
        u2 = np.linspace(-1, 1, U_vis)
        v2 = np.linspace(-1, 1, V_vis)

        U, V = np.meshgrid(u2, v2, indexing="xy")  # mesh grid of sine space
        # X, Y = np.meshgrid(vx, ry, indexing="xy")

        z_min = -40
        AF = 10 * np.log10(
            np.abs(
                compute_af_2d(
                    window_n, window_m, d_x, d_y, k_0, u_0, v_0, U, V, M, N, m, n
                )
            )
        )
        AF -= AF.min()
        AF /= AF.max()

        tck = bisplrep(U, V, AF)

        axes = ThreeDAxes(
            x_range=[-1, 1, 1 / 2],
            y_range=[-1, 1, 1 / 2],
            z_range=[0, 1],
            x_length=8,
        )

        def compute_surf(u, v):
            Z = bisplev(u, v, tck)
            R = np.sqrt(u**2 + v**2)
            return axes.c2p(
                *spherical_to_cartesian(
                    (
                        R,
                        np.arctan2(R, Z),
                        np.arctan2(v, u),
                    )
                )
            )

        surface = always_redraw(
            lambda: Surface(
                lambda u, v: compute_surf(u, v),
                # lambda u, v: axes.c2p(u, v, bisplev(u, v, tck)),
                u_range=[-1, 1],
                v_range=[-1, 1],
                resolution=(U_vis, V_vis),
            )
            .set_z(0)
            .set_style(fill_opacity=1)
            .set_fill_by_value(
                axes=axes, colorscale=[(BLUE, 0), (RED, -z_min - 20)], axis=2
            )
        )

        self.set_camera_orientation(theta=45 * DEGREES, phi=50 * DEGREES, zoom=0.3)

        self.add(surface)


class FillExample(Scene):
    def construct(self):
        polar_ax = Axes(
            x_range=[-1, 1, 1 / 8],
            y_range=[-1, 1, 1 / 8],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=config.frame_height * 0.7,
            y_length=config.frame_height * 0.7,
        ).rotate(PI / 2)

        shade_left = polar_ax.plot_polar_graph(
            lambda theta: 1, theta_range=[-PI / 3, 0], color=BLUE
        )
        shade_right = polar_ax.plot_polar_graph(
            lambda theta: 1, theta_range=[0, PI / 3], color=RED
        )

        right_fill = VGroup(
            Line(
                polar_ax.c2p(0, 0),
                shade_left.get_start(),
                color=BLUE,
            ),
            Line(
                polar_ax.c2p(0, 0),
                shade_left.get_end(),
                color=BLUE,
            ),
            shade_left,
        )
        left_fill = VGroup(
            Line(
                polar_ax.c2p(0, 0),
                shade_right.get_start(),
                color=RED,
            ),
            Line(
                polar_ax.c2p(0, 0),
                shade_right.get_end(),
                color=RED,
            ),
            shade_right,
        )

        # area = polar_ax.get_area(shade_left, x_range=[-PI / 3, 0])

        ap_left = ArcPolygon(
            shade_right.get_start(),
            shade_right.get_end(),
            polar_ax.c2p(0, 0),
            color=RED,
            # stroke_opacity=0,
            fill_opacity=0.5,
            arc_config=[
                {"angle": PI / 3, "stroke_opacity": 0},
                {"angle": 0, "stroke_opacity": 0},
                {"angle": 0, "stroke_opacity": 0},
            ],
        )
        ap_right = ArcPolygon(
            shade_left.get_start(),
            shade_left.get_end(),
            polar_ax.c2p(0, 0),
            color=BLUE,
            # stroke_opacity=0,
            fill_opacity=0.5,
            arc_config=[
                {"angle": PI / 3, "stroke_opacity": 0},
                {"angle": 0, "stroke_opacity": 0},
                {"angle": 0, "stroke_opacity": 0},
            ],
        )

        self.add(polar_ax, *left_fill, *right_fill, ap_left, ap_right)
