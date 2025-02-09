# phased_array.py

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
from props import WeatherRadarTower, get_blocks, VideoMobject

config.background_color = BACKGROUND_COLOR

BLOCKS = get_blocks()

SKIP_ANIMATIONS_OVERRIDE = True


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

        parabolic = Arc(radius=2, start_angle=PI * 0.75, color=BLUE).move_to(
            point_source
        )

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
            r"phased\_arrays.ipynb \rotatebox[origin=c]{270}{$\looparrowright$}",
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

        self.next_section(skip_animations=skip_animations(False))
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

        self.play(
            *[Create(m) for m in product_samples_new],
            *[Create(m) for m in product_sample_dots_new],
        )

        self.wait(0.5)

        self.play(wave_f @ 0, run_time=3)

        self.wait(0.5)

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

        self.next_section(skip_animations=skip_animations(False))

        self.wait(0.5)

        self.play(wave_f @ f_max, fft_f @ f_max, run_time=10)

        self.next_section(skip_animations=skip_animations(True))
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
        lin2db_label = Tex("Linear ", r"$\rightarrow$ ", "dB").next_to(f_ax, UP)
        self.play(
            ReplacementTransform(f_plot, log_f_plot),
            LaggedStart(*[GrowFromCenter(m) for m in lin2db_label], lag_ratio=0.2),
            run_time=2,
        )

        self.wait(0.5)

        fft_group = Group(f_ax, f_rad_xlabels, log_f_plot, X_k_label, f_labels)
        self.play(
            FadeOut(*self.mobjects)
            # FadeOut(
            #     amp_plot_group,
            #     wave_amp_plot_group,
            #     product_amp_plot_group,
            #     amp_plot,
            #     wave_amp_plot,
            #     product_amp_plot,
            #     mult,
            #     wave_eqn,
            #     *x_n_samples_new,
            #     *product_samples_new,
            #     *product_sample_dots_new,
            #     *x_n_sample_dots_new,
            #     sample_at_wave_f_dot_new,
            #     lin2db_label,
            # ),
            # self.camera.frame.animate.scale_to_fit_width(config.frame_width).move_to(
            #     ORIGIN
            # ),
            # fft_group.animate.scale_to_fit_width(config.frame_width * 0.8).move_to(
            #     ORIGIN
            # ),
        )

        self.wait(2)


class AFToPolar(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]

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

        self.play(FadeIn(ax, AF_plot))

        self.wait(0.5)

        self.play(Transform(AF_plot, AF_polar_plot), run_time=2)

        self.wait(0.5)
        fnbw = 2 * wavelength_0 / (n_elem * d_x)

        right_theta_0 = VT(-0.01)
        right_theta_1 = VT(0)
        left_theta_0 = VT(0)
        left_theta_1 = VT(0.01)

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
        side_lobe_left_label = (
            Tex("Side lobes")
            .next_to(ax.input_to_graph_point(-PI / 3, AF_polar_plot), UL, LARGE_BUFF)
            .shift(UP * 2)
        )
        side_lobe_right_label = (
            Tex("Side lobes")
            .next_to(ax.input_to_graph_point(PI / 3, AF_polar_plot), UR, LARGE_BUFF)
            .shift(UP * 2)
        )

        self.add(shade_left, shade_right, ap_left, ap_right)
        self.play(right_theta_0 @ (-fnbw * 2), left_theta_1 @ (fnbw * 2))

        self.wait(0.5)

        self.play(FadeIn(main_lobe_label, shift=DOWN * 3))

        self.wait(0.5)

        null_label = (
            Tex("Null").next_to(main_lobe_label, RIGHT, LARGE_BUFF * 1.5).shift(DOWN)
        )

        left_null_bez = CubicBezier(
            polar_ax.c2p(0, 0)
            + UP * 0.5 * np.cos(fnbw * 1.6)
            + LEFT * 0.5 * np.sin(fnbw * 1.6),
            polar_ax.c2p(0, 0)
            + UP * 5 * np.cos(fnbw * 1.6)
            + LEFT * 5 * np.sin(fnbw * 1.6),
            null_label.get_left() + LEFT,
            null_label.get_left(),
        )
        right_null_bez = CubicBezier(
            polar_ax.c2p(0, 0)
            + UP * 0.5 * np.cos(-fnbw * 1.6)
            + RIGHT * 0.5 * np.sin(fnbw * 1.6),
            polar_ax.c2p(0, 0)
            + UP * 3 * np.cos(-fnbw * 1.6)
            + RIGHT * 3 * np.sin(fnbw * 1.6),
            null_label.get_left() + LEFT,
            null_label.get_left(),
        )

        self.play(Create(right_null_bez), Create(left_null_bez), FadeIn(null_label))

        self.wait(0.5)

        self.play(FadeOut(right_null_bez, left_null_bez, null_label))

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    right_theta_0 @ (-PI),
                    right_theta_1 @ (-fnbw * 2),
                    left_theta_1 @ (PI),
                    left_theta_0 @ (fnbw * 2),
                ),
                r_min_amp @ (0.95),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        side_lobe_left_bezs = [
            CubicBezier(
                polar_ax.c2p(0, 0) + offset,
                polar_ax.c2p(0, 0) + offset + [-1, 0.5, 0],
                side_lobe_left_label.get_bottom() + [0, -0.5, 0],
                side_lobe_left_label.get_bottom() + [0, -0.1, 0],
            )
            for offset in [[-1.4, 2.4, 0], [-1.7, 1.3, 0], [-1.6, 0.5, 0]]
        ]
        side_lobe_right_bezs = [
            CubicBezier(
                polar_ax.c2p(0, 0) + offset,
                polar_ax.c2p(0, 0) + offset + [1, 0.5, 0],
                side_lobe_right_label.get_bottom() + [0, -0.5, 0],
                side_lobe_right_label.get_bottom() + [0, -0.1, 0],
            )
            for offset in [[1.4, 2.4, 0], [1.7, 1.3, 0], [1.6, 0.5, 0]]
        ]

        self.play(
            LaggedStart(*[Create(bez) for bez in side_lobe_left_bezs], lag_ratio=0.3),
            LaggedStart(*[Create(bez) for bez in side_lobe_right_bezs], lag_ratio=0.3),
            FadeIn(side_lobe_left_label, side_lobe_right_label),
        )

        self.wait(0.5)

        antennas = Group()
        weight_trackers_disp = [VT(1) for _ in range(8)]
        for idx in range(8):
            antenna_port = Line(DOWN / 4, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)

        wt_disp = Group(
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[0]:.2f}").next_to(
                    antennas[0], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[1]:.2f}").next_to(
                    antennas[1], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[2]:.2f}").next_to(
                    antennas[2], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[3]:.2f}").next_to(
                    antennas[3], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[4]:.2f}").next_to(
                    antennas[4], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[5]:.2f}").next_to(
                    antennas[5], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[6]:.2f}").next_to(
                    antennas[6], RIGHT
                )
            ),
            always_redraw(
                lambda: Tex(f"{~weight_trackers_disp[7]:.2f}").next_to(
                    antennas[7], RIGHT
                )
            ),
        )

        antennas.arrange(DOWN, MED_LARGE_BUFF).scale_to_fit_height(
            config.frame_height * 0.8
        ).next_to([-config.frame_width / 2, 0, 0], LEFT, LARGE_BUFF * 1.5)
        self.add(antennas, wt_disp)

        self.next_section(skip_animations=skip_animations(False))
        self.camera.frame.save_state()
        camera_shift = self.camera.frame.get_right()[0] - (
            polar_ax.get_right()[0] + LARGE_BUFF
        )
        self.play(
            *[Uncreate(bez) for bez in [*side_lobe_left_bezs, *side_lobe_right_bezs]],
            FadeOut(
                side_lobe_left_label,
                side_lobe_right_label,
                main_lobe_label,
                ap_left,
                ap_right,
            ),
            self.camera.frame.animate.shift(LEFT * camera_shift),
        )

        self.wait(0.5)

        weights = np.array([~w for w in weight_trackers])
        x = np.arange(weights.size) - weights.size // 2
        taper_ax = (
            Axes(
                x_range=[x.min(), x.max(), 1],
                y_range=[0.3, 1, 0.5],
                tips=False,
                x_length=antennas.height,
                y_length=antennas.width * 3,
            )
            .rotate(-PI / 2)
            .next_to(antennas, RIGHT, LARGE_BUFF * 1.6)
        )

        def plot_taper():
            weights = np.array([~w for w in weight_trackers])
            x = np.arange(weights.size) - weights.size // 2
            f = interp1d(x, weights)
            plot = taper_ax.plot(f, x_range=[x.min(), x.max(), 0.01], color=ORANGE)
            return plot

        taper_plot = always_redraw(plot_taper)

        self.play(FadeIn(taper_ax, taper_plot))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        taper = signal.windows.taylor(n_elem, nbar=5, sll=23)
        taper_disp = signal.windows.taylor(len(antennas), nbar=5, sll=23)

        self.remove(AF_plot)
        AF_polar_plot_new = always_redraw(get_polar_updater(AF_plot.copy()))
        self.add(AF_polar_plot_new)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers_disp[:4][::-1][n] @ taper_disp[:4][::-1][n],
                        weight_trackers_disp[4:][n] @ taper_disp[4:][n],
                    )
                    for n in range(4)
                ],
                lag_ratio=0.3,
            ),
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
            self.camera.frame.animate.shift(RIGHT * camera_shift + UP * 1.3),
            FadeOut(taper_ax, taper_plot, wt_disp, antennas),
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

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .to_edge(UP, -LARGE_BUFF)
        )

        target_arrow = Arrow(polar_ax.c2p(0, 0), cloud.get_bottom())

        self.play(GrowArrow(target_arrow), cloud.shift(UP * 5).animate.shift(DOWN * 5))

        other_cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .to_edge(RIGHT, LARGE_BUFF)
            .set_y(cloud.get_y())
        )
        target_arrow_other = Arrow(polar_ax.c2p(0, 0), other_cloud.get_corner(DL))

        self.wait(0.5)

        target_arrow.save_state()
        self.play(
            Transform(target_arrow, target_arrow_other),
            other_cloud.shift(RIGHT * 5).animate.shift(LEFT * 5),
        )

        self.wait(0.5)

        self.play(target_arrow.animate.restore())

        self.wait(0.5)

        antennas_copy = (
            antennas.copy()
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(AF_polar_plot_new, DOWN)
            .scale(1.2)
        )

        self.play(
            FadeOut(target_arrow, ax, cloud, other_cloud),
            self.camera.frame.animate.shift(DOWN * 2),
            FadeIn(antennas_copy),
        )

        self.wait(0.5)

        scan_left_arrow = Arrow(ORIGIN, LEFT * 2, buff=0)
        scan_right_arrow = Arrow(ORIGIN, RIGHT * 2, buff=0)
        scan_up_arrow = Arrow(ORIGIN, UP * 2, buff=0)
        scan_down_arrow = Arrow(ORIGIN, DOWN * 2, buff=0)

        self.play(
            FadeOut(AF_polar_plot_new, antennas_copy),
            self.camera.frame.animate.restore(),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(scan_left_arrow),
                GrowArrow(scan_right_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(scan_up_arrow),
                GrowArrow(scan_down_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        theta_neg_label = MathTex(r"-\theta").next_to(scan_left_arrow.get_end(), DOWN)
        theta_pos_label = MathTex(r"\theta").next_to(scan_right_arrow.get_end(), DOWN)
        phi_pos_label = MathTex(r"\phi").next_to(scan_up_arrow.get_end(), RIGHT)
        phi_neg_label = MathTex(r"-\phi").next_to(scan_down_arrow.get_end(), RIGHT)

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in [
                        theta_neg_label,
                        theta_pos_label,
                        phi_neg_label,
                        phi_pos_label,
                    ]
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            ShrinkToCenter(
                Group(
                    scan_up_arrow,
                    scan_down_arrow,
                    scan_left_arrow,
                    scan_right_arrow,
                    theta_neg_label,
                    theta_pos_label,
                    phi_neg_label,
                    phi_pos_label,
                )
            )
        )

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


class Equation2D(MovingCameraScene):
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

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                fourier_eqn[4]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 3)
                .set_color(YELLOW),
                fourier_eqn[9]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 3)
                .set_color(YELLOW),
                fourier_eqn[-1]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 3)
                .set_color(YELLOW),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

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

        self.wait(0.5)

        uv_coords = MathTex("u,v", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            fourier_eqn_full[0][-1], UR, LARGE_BUFF
        )

        u_bez = CubicBezier(
            fourier_eqn_full[0][-1].get_right() + [0.1, 0, 0],
            fourier_eqn_full[0][-1].get_right() + [1, 0, 0],
            uv_coords[0][0].get_bottom() + [0, -1, 0],
            uv_coords[0][0].get_bottom() + [0, -0.1, 0],
        )
        v_bez = CubicBezier(
            fourier_eqn_full[0][-1].get_right() + [0.1, 0, 0],
            fourier_eqn_full[0][-1].get_right() + [1, 0, 0],
            uv_coords[0][2].get_bottom() + [0, -1, 0],
            uv_coords[0][2].get_bottom() + [0, -0.1, 0],
        )

        self.remove(*antennas_2d, *antennas)
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.set_x(Group(fourier_eqn_full, uv_coords).get_x()),
            LaggedStart(
                Create(u_bez),
                FadeIn(uv_coords[0][0]),
                FadeIn(uv_coords[0][1]),
                Create(v_bez),
                FadeIn(uv_coords[0][2]),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        antennas_new = Group()
        for _ in range(16):
            antenna_port = Line(DOWN / 4, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas_new.add(antenna)

        antennas_new.arrange_in_grid(4, 4, MED_LARGE_BUFF).next_to(
            self.camera.frame.get_left(), LEFT
        )
        self.add(antennas_new)

        antennas_eqn_group = Group(fourier_eqn_full, antennas_new)
        self.play(
            Uncreate(u_bez),
            FadeOut(uv_coords[0][0]),
            FadeOut(uv_coords[0][1]),
            Uncreate(v_bez),
            FadeOut(uv_coords[0][2]),
            self.camera.frame.animate.scale_to_fit_width(
                antennas_eqn_group.width * 1.1
            ).set_x(antennas_eqn_group.get_x()),
        )

        self.wait(0.5)

        def set_ant_opacity(g, opacity):
            return AnimationGroup(*[m.animate.set_stroke(opacity=opacity) for m in g])

        dx_line = Line(antennas_new[-4].get_top(), antennas_new[-3].get_top()).shift(
            UP / 3
        )
        dx_line_l = Line(dx_line.get_start() + DOWN / 8, dx_line.get_start() + UP / 8)
        dx_line_r = Line(dx_line.get_end() + DOWN / 8, dx_line.get_end() + UP / 8)
        dx_label = MathTex("d_x").next_to(dx_line, UP)

        self.play(
            LaggedStart(
                *[set_ant_opacity(g, 0.1) for g in antennas_new[:-4]],
                Create(dx_line_l),
                Create(dx_line),
                Create(dx_line_r),
                FadeIn(dx_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.wait(2)


class Equation2DV2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        fourier_eqn_full = MathTex(
            r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi \frac{k}{N} n}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        # fourier_eqn = fourier_eqn_full[0][5:]
        self.play(
            LaggedStart(
                *[GrowFromPoint(m, ORIGIN) for m in fourier_eqn_full[0]], lag_ratio=0.1
            )
        )

        self.wait(0.5)

        x_n_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-2, 2, 1],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=config.frame_width * 0.35,
                y_length=config.frame_height * 0.2,
            )
            .to_edge(DOWN, LARGE_BUFF)
            .shift(LEFT * 3 + DOWN * 6)
        )

        x_n_plot = always_redraw(
            lambda: x_n_ax.plot(
                lambda t: np.sin(2 * PI * 3 * t) + np.sin(2 * PI * 10 * t),
                x_range=[0, 1, 1 / 100],
                color=BLUE,
            )
        )

        ref_ax = (
            Axes(
                x_range=[0, 1, 0.25],
                y_range=[-1, 1, 0.5],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=config.frame_width * 0.35,
                y_length=config.frame_height * 0.2,
            )
            .to_edge(DOWN, LARGE_BUFF)
            .shift(RIGHT * 3 + DOWN * 6)
        )

        ref_f = VT(2)

        ref_plot = always_redraw(
            lambda: ref_ax.plot(
                lambda t: np.sin(2 * PI * ~ref_f * t),
                x_range=[0, 1, 1 / 100],
                color=RED,
            )
        )

        self.add(ref_ax, ref_plot, x_n_ax, x_n_plot)

        x_n_bez = CubicBezier(
            fourier_eqn_full.copy().to_edge(UP, LARGE_BUFF)[0][12:16].get_bottom()
            + [0, -0.1, 0],
            fourier_eqn_full.copy().to_edge(UP, LARGE_BUFF)[0][12:16].get_bottom()
            + [0, -1.5, 0],
            x_n_ax.copy().shift(UP * 6).get_top() + [0, 1, 0],
            x_n_ax.copy().shift(UP * 6).get_top() + [0, 0.1, 0],
        )
        ref_bez = CubicBezier(
            fourier_eqn_full.copy().to_edge(UP, LARGE_BUFF)[0][16:].get_bottom()
            + [0, -0.1, 0],
            fourier_eqn_full.copy().to_edge(UP, LARGE_BUFF)[0][16:].get_bottom()
            + [0.5, -1, 0],
            ref_ax.copy().shift(UP * 6).get_top() + [-0.5, 1, 0],
            ref_ax.copy().shift(UP * 6).get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                *[m.animate.set_color(BLUE) for m in fourier_eqn_full[0][12:16]],
                lag_ratio=0.1,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                fourier_eqn_full.animate.to_edge(UP, LARGE_BUFF),
                Create(x_n_bez),
                x_n_ax.animate.shift(UP * 6),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(RED) for m in fourier_eqn_full[0][16:]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                fourier_eqn_full.animate.to_edge(UP, LARGE_BUFF),
                Create(ref_bez),
                ref_ax.animate.shift(UP * 6),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(ref_f @ 10, run_time=2)

        self.wait(0.5)

        self.play(
            Uncreate(ref_bez),
            Uncreate(x_n_bez),
            ref_ax.animate.shift(DOWN * 6),
            x_n_ax.animate.shift(DOWN * 6),
        )
        self.remove(ref_ax, ref_plot, x_n_plot, x_n_ax)

        self.wait(0.5)

        antennas = Group()
        for _ in range(6):
            antenna_port = Line(DOWN / 4, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange(RIGHT, MED_LARGE_BUFF).to_edge(DOWN, MED_LARGE_BUFF).shift(
            DOWN * 6
        )

        ones = Group(
            *[
                Tex("1", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(ant, UP)
                for ant in antennas.copy().shift(UP * 6)
            ]
        )

        w_n_eqn = Tex(
            "$w[n] = [1,1,1,1,1,1]$", font_size=DEFAULT_FONT_SIZE * 1.8
        ).next_to(ones, UP)

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        m.animate.shift(UP * 6), GrowFromCenter(one), lag_ratio=0.6
                    )
                    for m, one in zip(antennas, ones)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        just_weights = w_n_eqn[0][5:]
        just_weights.save_state()
        just_weights.set_x(0)

        self.play(
            LaggedStart(
                GrowFromCenter(w_n_eqn[0][5]),
                ReplacementTransform(ones[0], w_n_eqn[0][6]),
                GrowFromCenter(w_n_eqn[0][7]),
                ReplacementTransform(ones[1], w_n_eqn[0][8]),
                GrowFromCenter(w_n_eqn[0][9]),
                ReplacementTransform(ones[2], w_n_eqn[0][10]),
                GrowFromCenter(w_n_eqn[0][11]),
                ReplacementTransform(ones[3], w_n_eqn[0][12]),
                GrowFromCenter(w_n_eqn[0][13]),
                ReplacementTransform(ones[4], w_n_eqn[0][14]),
                GrowFromCenter(w_n_eqn[0][15]),
                ReplacementTransform(ones[5], w_n_eqn[0][16]),
                GrowFromCenter(w_n_eqn[0][17]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                just_weights.animate.restore(),
                *[GrowFromCenter(m) for m in w_n_eqn[0][:5]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        w_n = w_n_eqn[0][:4]
        w_n_copy = w_n.copy().set_color(BLUE).move_to(fourier_eqn_full[0][12:16], RIGHT)
        self.play(
            TransformFromCopy(w_n, w_n_copy),
            fourier_eqn_full[0][12:16].animate.shift(UP * 6),
            run_time=2,
        )

        self.wait(0.5)

        ref_eqn = fourier_eqn_full[0][-9:]
        self.play(
            w_n_eqn.animate.shift(DOWN * 8),
            antennas.animate.shift(DOWN * 8),
            fourier_eqn_full[0][:-9].animate.set_opacity(0.2),
            w_n_copy.animate.set_opacity(0.2),
        )

        self.wait(0.5)

        new_fourier_eqn = Group(
            fourier_eqn_full[0][:12], fourier_eqn_full[0][16:], w_n_copy
        )
        self.play(
            Group(fourier_eqn_full[0][:12], w_n_copy).animate.scale(0.6).to_corner(UL),
            ref_eqn.animate.move_to(ORIGIN).shift(DOWN),
        )

        self.wait(0.5)

        eulers = MathTex(r"e^{-j \theta t}", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            [config.frame_width / 2, 0, 0], RIGHT
        )
        eulers_impl = MathTex(r"\implies", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            [config.frame_width / 2, 0, 0], RIGHT
        )

        eulers_group = Group(ref_eqn, eulers_impl, eulers)
        self.play(eulers_group.animate.arrange(RIGHT, MED_LARGE_BUFF).shift(DOWN))

        self.wait(0.5)

        sqrt_neg = MathTex(r"\sqrt{-1}", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            eulers_group, UP, LARGE_BUFF
        )
        sqrt_neg_bez_l = CubicBezier(
            ref_eqn[2].get_top() + [0, 0.1, 0],
            ref_eqn[2].get_top() + [0, 1, 0],
            sqrt_neg.get_bottom() + [0, -1, 0],
            sqrt_neg.get_bottom() + [0, -0.1, 0],
        )
        sqrt_neg_bez_r = CubicBezier(
            eulers[0][2].get_top() + [0, 0.1, 0],
            eulers[0][2].get_top() + [0, 1, 0],
            sqrt_neg.get_bottom() + [0, -1, 0],
            sqrt_neg.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ref_eqn[2].animate.set_color(YELLOW),
                    eulers[0][2].animate.set_color(YELLOW),
                ),
                AnimationGroup(Create(sqrt_neg_bez_l), Create(sqrt_neg_bez_r)),
                GrowFromCenter(sqrt_neg),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        time_label = Tex(r"time", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            eulers_group, UP, LARGE_BUFF
        )
        time_label_bez_l = CubicBezier(
            ref_eqn[-1].get_top() + [0, 0.1, 0],
            ref_eqn[-1].get_top() + [0, 1, 0],
            time_label.get_bottom() + [0, -1, 0],
            time_label.get_bottom() + [0, -0.1, 0],
        )
        time_label_bez_r = CubicBezier(
            eulers[0][4].get_top() + [0, 0.1, 0],
            eulers[0][4].get_top() + [0, 1, 0],
            time_label.get_bottom() + [0, -1, 0],
            time_label.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ref_eqn[2].animate.set_color(RED),
                    eulers[0][2].animate.set_color(WHITE),
                ),
                AnimationGroup(
                    ReplacementTransform(sqrt_neg_bez_l, time_label_bez_l),
                    ReplacementTransform(sqrt_neg_bez_r, time_label_bez_r),
                ),
                AnimationGroup(
                    ref_eqn[-1].animate.set_color(YELLOW),
                    eulers[0][4].animate.set_color(YELLOW),
                ),
                ReplacementTransform(sqrt_neg, time_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ref_eqn[-1].animate.set_color(RED),
                    eulers[0][4].animate.set_color(WHITE),
                    Uncreate(time_label_bez_l),
                    Uncreate(time_label_bez_r),
                    ShrinkToCenter(time_label),
                ),
                ref_eqn[3:-1].animate.set_color(YELLOW),
                eulers[0][3].animate.set_color(YELLOW),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        x_len = config.frame_width * 0.25
        y_len = config.frame_width * 0.25
        unit_circle_ax = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        unit_circle_labels = Group(
            *[
                MathTex(s).next_to(unit_circle_ax.c2p(*a), d)
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )
        unit_circle = Circle(unit_circle_ax.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax.c2p(0, 0)
        )
        unit_circle_group = Group(
            unit_circle_ax, unit_circle_labels, unit_circle
        ).to_edge(DOWN, MED_SMALL_BUFF)

        n_counter = VT(0)
        phase_delta = VT(PI / 4)
        phase = VT(~n_counter * ~phase_delta)

        phase_dot = always_redraw(
            lambda: Dot(
                unit_circle_ax.c2p(np.cos(~phase), np.sin(~phase)), color=YELLOW
            )
        )
        phase_line = always_redraw(
            lambda: Line(
                unit_circle_ax.c2p(0, 0),
                unit_circle_ax.c2p(np.cos(~phase), np.sin(~phase)),
                color=YELLOW,
            )
        )

        sine_ax = Axes(
            x_range=[0, 4 * PI, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=config.frame_width * 0.7,
            y_length=unit_circle.height,
        ).next_to(unit_circle, RIGHT, 0)
        sine_plot = always_redraw(
            lambda: sine_ax.plot(
                lambda t: np.sin(t), x_range=[PI / 4, ~phase, 1 / 100], color=YELLOW
            )
        )

        self.add(sine_plot, sine_plot)

        self.play(
            ref_eqn.animate.set_x(0).to_edge(UP),
            FadeOut(eulers_impl),
            eulers.animate.next_to(unit_circle_group, LEFT, LARGE_BUFF),
        )

        self.wait(0.5)

        self.play(
            Create(unit_circle_ax),
            Create(unit_circle),
            LaggedStart(*[FadeIn(m) for m in unit_circle_labels], lag_ratio=0.3),
        )

        self.wait(0.5)

        self.play(Create(phase_dot), Create(phase_line))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            eulers[0][3].animate(rate_func=rate_functions.there_and_back).shift(UP / 2)
        )

        self.wait(0.5)

        self.play(
            ref_eqn[3:-1]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(DOWN / 2)
        )

        self.wait(0.5)

        theta_val = MathTex(
            r"\theta = \frac{\pi}{4}", font_size=DEFAULT_FONT_SIZE * 1.2
        ).next_to(eulers, UP, MED_SMALL_BUFF, LEFT)
        theta_val[0][0].set_color(YELLOW)

        self.play(
            LaggedStart(
                TransformFromCopy(eulers[0][3], theta_val[0][0]),
                Create(theta_val[0][1:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(phase @ (PI / 4))

        self.wait(0.5)

        sine_line = always_redraw(
            lambda: DashedLine(
                unit_circle_ax.c2p(np.cos(~phase), np.sin(~phase)),
                sine_ax.input_to_graph_point(~phase, sine_plot),
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        sine_dot = always_redraw(
            lambda: Dot(sine_ax.input_to_graph_point(~phase, sine_plot), color=RED)
        )

        self.play(Create(sine_line), Create(sine_dot))

        self.wait(0.5)

        t_vt = VT(1)
        t_tracker = always_redraw(
            lambda: MathTex(
                f"t = {~t_vt:.2f}", font_size=DEFAULT_FONT_SIZE * 1.2
            ).next_to(theta_val, UP, MED_SMALL_BUFF, LEFT)
        )

        self.play(FadeIn(t_tracker))

        self.wait(0.5)

        t_final = 14
        self.camera.frame.save_state()
        self.play(
            t_vt @ t_final,
            phase @ (t_final * PI / 4),
            self.camera.frame.animate.shift(RIGHT * 4.5),
            run_time=4,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        self.remove(w_n_eqn)
        antennas.shift(DOWN)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.shift(DOWN * config.frame_height * 1.2))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        background_box = (
            Rectangle(
                height=config.frame_height,
                width=config.frame_width,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
                stroke_opacity=0,
            )
            .set_z_index(-1)
            .move_to(antennas, UP)
        )
        self.add(background_box)

        angle = -PI / 6
        angled_plane_wave = (
            Line(
                LEFT * 4,
                RIGHT * 4,
                color=RX_COLOR,
            ).rotate(angle)
        ).set_z_index(-2)
        angled_plane_wave.shift(
            self.camera.frame.get_corner(UR) - angled_plane_wave.get_center()
        )
        # self.add(angled_plane_wave)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        plane_wave_longer = Line(
            antennas[-1].get_top() - 16 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            antennas[-1].get_top() + 16 * (LEFT * np.cos(angle) + DOWN * np.sin(angle)),
            color=RX_COLOR,
        ).set_z_index(-2)

        self.play(
            # Transform(angled_plane_wave, plane_wave_longer)
            plane_wave_longer.shift(RIGHT * 20).animate.shift(LEFT * 20),
            run_time=2,
            # angled_plane_wave.animate(run_time=3).shift(
            #     14 * (LEFT * np.cos(angle) + UP * np.sin(angle))
            # ),
        )

        # self.add(plane_wave_longer)
        self.next_section(skip_animations=skip_animations(True))

        dx_disp = antennas[1].get_top()[0] - antennas[0].get_top()[0]
        wave_to_ant_2 = Arrow(
            antennas[-2].get_top()
            + (LEFT * np.sin(angle) + UP * np.cos(angle)) / np.sqrt(2),
            antennas[-2].get_top(),
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tip_length=DEFAULT_ARROW_TIP_LENGTH,
        )
        delta_t2 = MathTex(r"\Delta t", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_2.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_phi2 = MathTex(r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_2.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_d2 = MathTex(r"\Delta d", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_2.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        wave_to_ant_3 = Arrow(
            antennas[-3].get_top()
            + dx_disp * (LEFT * np.sin(angle) + UP * np.cos(angle)) / np.sqrt(1),
            antennas[-3].get_top(),
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tip_length=DEFAULT_ARROW_TIP_LENGTH,
        )
        delta_t3 = MathTex(r"2\Delta t", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_3.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_phi3 = MathTex(
            r"2\Delta \phi", font_size=DEFAULT_FONT_SIZE * 0.8
        ).next_to(wave_to_ant_3.get_midpoint(), LEFT, MED_SMALL_BUFF)
        delta_d3 = MathTex(r"2\Delta d", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_3.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        wave_to_ant_4 = Arrow(
            antennas[-4].get_top()
            + 1.5 * dx_disp * (LEFT * np.sin(angle) + UP * np.cos(angle)),
            antennas[-4].get_top(),
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tip_length=DEFAULT_ARROW_TIP_LENGTH,
        )
        delta_t4 = MathTex(r"3\Delta t", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_4.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_phi4 = MathTex(
            r"3\Delta \phi", font_size=DEFAULT_FONT_SIZE * 0.8
        ).next_to(wave_to_ant_4.get_midpoint(), LEFT, MED_SMALL_BUFF)
        delta_d4 = MathTex(r"3\Delta d", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_4.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        wave_to_ant_5 = Arrow(
            antennas[-5].get_top()
            + 2 * dx_disp * (LEFT * np.sin(angle) + UP * np.cos(angle)),
            antennas[-5].get_top(),
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tip_length=DEFAULT_ARROW_TIP_LENGTH,
        )
        delta_t5 = MathTex(r"4\Delta t", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_5.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_phi5 = MathTex(
            r"4\Delta \phi", font_size=DEFAULT_FONT_SIZE * 0.8
        ).next_to(wave_to_ant_5.get_midpoint(), LEFT, MED_SMALL_BUFF)
        delta_d5 = MathTex(r"4\Delta d", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_5.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        wave_to_ant_6 = Arrow(
            antennas[-6].get_top()
            + 2.5 * dx_disp * (LEFT * np.sin(angle) + UP * np.cos(angle)),
            antennas[-6].get_top(),
            buff=0,
            stroke_width=DEFAULT_STROKE_WIDTH,
            tip_length=DEFAULT_ARROW_TIP_LENGTH,
        )
        delta_t6 = MathTex(r"5\Delta t", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_6.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        delta_phi6 = MathTex(
            r"5\Delta \phi", font_size=DEFAULT_FONT_SIZE * 0.8
        ).next_to(wave_to_ant_6.get_midpoint(), LEFT, MED_SMALL_BUFF)
        delta_d6 = MathTex(r"5\Delta d", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            wave_to_ant_6.get_midpoint(), LEFT, MED_SMALL_BUFF
        )
        self.play(
            LaggedStart(
                GrowArrow(wave_to_ant_2),
                GrowFromCenter(delta_t2),
                GrowArrow(wave_to_ant_3),
                GrowFromCenter(delta_t3),
                GrowArrow(wave_to_ant_4),
                GrowFromCenter(delta_t4),
                GrowArrow(wave_to_ant_5),
                GrowFromCenter(delta_t5),
                GrowArrow(wave_to_ant_6),
                GrowFromCenter(delta_t6),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(delta_t2, delta_d2),
                ReplacementTransform(delta_t3, delta_d3),
                ReplacementTransform(delta_t4, delta_d4),
                ReplacementTransform(delta_t5, delta_d5),
                ReplacementTransform(delta_t6, delta_d6),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(delta_d2, delta_phi2),
                ReplacementTransform(delta_d3, delta_phi3),
                ReplacementTransform(delta_d4, delta_phi4),
                ReplacementTransform(delta_d5, delta_phi5),
                ReplacementTransform(delta_d6, delta_phi6),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_delta_full_eqn = MathTex(
            r"\Delta \phi = \frac{d_x \sin{(\theta)}}{\lambda} \cdot 2 \pi"
        ).next_to(self.camera.frame.get_corner(UR), DL, LARGE_BUFF)

        self.play(
            TransformFromCopy(
                delta_phi2[0], phase_delta_full_eqn[0][:2], path_arc=PI / 3
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(phase_delta_full_eqn[0][2]),
                GrowFromCenter(phase_delta_full_eqn[0][3:5]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            GrowFromCenter(phase_delta_full_eqn[0][5:11]),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(phase_delta_full_eqn[0][11]),
                GrowFromCenter(phase_delta_full_eqn[0][12]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(phase_delta_full_eqn[0][13]),
                GrowFromCenter(phase_delta_full_eqn[0][14:]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        phase_delta_full_eqn_theta = MathTex(
            r"\theta = \Delta \phi = \frac{d_x \sin{(\theta)}}{\lambda} \cdot 2 \pi"
        ).next_to(eulers, RIGHT, LARGE_BUFF)
        phase_delta_full_eqn_theta[0][0].set_color(YELLOW)

        self.play(
            self.camera.frame.animate.restore(),
            ReplacementTransform(
                phase_delta_full_eqn[0],
                phase_delta_full_eqn_theta[0][2:],
                path_arc=PI / 3,
            ),
            FadeOut(t_tracker, theta_val),
            unit_circle_group.animate.shift(RIGHT * 10),
            sine_ax.animate.shift(RIGHT * 10),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(
                    eulers[0][3], phase_delta_full_eqn_theta[0][0], path_arc=PI / 2
                ),
                GrowFromCenter(phase_delta_full_eqn_theta[0][1]),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        eulers_w_phase = MathTex(
            r"e^{-j \tfrac{d_x \sin{(\theta)}}{\lambda} \cdot 2 \pi t}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).move_to(eulers, LEFT)

        # self.add(
        #     index_labels(phase_delta_full_eqn_theta[0]),
        #     index_labels(eulers[0]),
        #     index_labels(eulers_w_phase[0]).shift(DOWN),
        #     eulers_w_phase.shift(DOWN),
        # )
        self.play(
            LaggedStart(
                ShrinkToCenter(eulers[0][3]),
                ReplacementTransform(eulers[0][:3], eulers_w_phase[0][:3]),
                ShrinkToCenter(phase_delta_full_eqn_theta[0][:5]),
                ReplacementTransform(eulers[0][-1], eulers_w_phase[0][-1]),
                ReplacementTransform(
                    phase_delta_full_eqn_theta[0][5:],
                    eulers_w_phase[0][3:-1],
                    path_arc=PI / 2,
                ),
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            eulers_w_phase[0][12].animate.set_color(YELLOW),
            eulers_w_phase[0][14:16].animate.set_color(YELLOW),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        eulers_w_phase_2pi = MathTex(
            r"e^{-j \tfrac{2\pi}{\lambda} d_x \sin{(\theta) t}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).move_to(eulers, LEFT)
        eulers_w_phase_2pi[0][3:7].set_color(YELLOW)

        self.play(
            TransformByGlyphMap(
                eulers_w_phase,
                eulers_w_phase_2pi,
                ([0, 1, 2], [0, 1, 2]),
                ([3, 4], [7, 8], {"delay": 0.2}),
                ([5, 6, 7, 8, 9, 10], [9, 10, 11, 12, 13, 14], {"delay": 0.4}),
                ([13], ShrinkToCenter, {"delay": 0}),
                ([14, 15], [3, 4], {"path_arc": PI / 2, "delay": 0.2}),
                ([11], [5], {"delay": 0}),
                ([12], [6], {"delay": 0.4}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.wait(0.5)

        eulers_w_phase_k = MathTex(
            r"e^{-j k d_x \sin{(\theta) t}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).move_to(eulers, LEFT)

        self.play(
            TransformByGlyphMap(
                eulers_w_phase_2pi,
                eulers_w_phase_k,
                ([0, 1, 2], [0, 1, 2]),
                ([3, 4, 5, 6], ShrinkToCenter, {"delay": 0}),
                (GrowFromCenter, [3], {"delay": 0.2}),
                (
                    [7, 8, 9, 10, 11, 12, 13, 14, 15],
                    [4, 5, 6, 7, 8, 9, 10, 11, 12],
                    {"delay": 0.4},
                ),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        not_equal = (
            MathTex(r"\neq", font_size=DEFAULT_FONT_SIZE * 2)
            .next_to(eulers_w_phase_k, UP, LARGE_BUFF)
            .shift(RIGHT)
        )
        freq_k_bez = CubicBezier(
            not_equal.get_top() + [0, 0.1, 0],
            not_equal.get_top() + [0, 1, 0],
            ref_eqn[5].get_bottom() + [0, -1, 0],
            ref_eqn[5].get_bottom() + [0, -0.1, 0],
        )
        wave_k_bez = CubicBezier(
            not_equal.get_bottom() + [0, -0.1, 0],
            not_equal.get_bottom() + [0, -1, 0],
            eulers_w_phase_k[0][3].get_top() + [0, 1, 0],
            eulers_w_phase_k[0][3].get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    eulers_w_phase_k[0][3].animate.set_color(YELLOW),
                    ref_eqn[:5].animate.set_color(WHITE),
                    ref_eqn[6:].animate.set_color(WHITE),
                ),
                GrowFromCenter(not_equal),
                AnimationGroup(Create(freq_k_bez), Create(wave_k_bez)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(freq_k_bez),
            Uncreate(wave_k_bez),
            FadeOut(not_equal),
            eulers_w_phase_k[0][3].animate.set_color(WHITE),
            ref_eqn[5].animate.set_color(WHITE),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(eulers_w_phase_k.animate.move_to(ORIGIN))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            eulers_w_phase_k[0][-1]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .shift(UP / 3)
        )

        self.wait(0.5)

        self.play(
            ref_eqn[-1]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .shift(DOWN / 3)
        )

        self.wait(0.5)

        self.play(
            eulers_w_phase_k.animate.next_to(
                plane_wave_longer.get_midpoint(), RIGHT
            ).shift(UP * 4 + LEFT * 3),
            self.camera.frame.animate.shift(DOWN * config.frame_height * 1.2),
            ref_eqn.animate.set_opacity(0.2),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back)
                    .set_color(YELLOW)
                    .shift(UP / 3)
                    for m in [
                        delta_phi2,
                        delta_phi3,
                        delta_phi4,
                        delta_phi5,
                        delta_phi6,
                    ]
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        eulers_w_phase_m = MathTex(
            r"e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).move_to(eulers_w_phase_k, LEFT)

        self.play(
            ReplacementTransform(eulers_w_phase_k[0][:-1], eulers_w_phase_m[0][:-1]),
            FadeOut(eulers_w_phase_k[0][-1], shift=UP),
            FadeIn(eulers_w_phase_m[0][-1], shift=UP),
        )

        self.wait(0.5)

        ms = Group(
            *[
                MathTex(f"{5 - m}", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
                    ant, DOWN, MED_SMALL_BUFF
                )
                for m, ant in enumerate(antennas)
            ]
        )
        m_eq = MathTex("m = ", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            ms, LEFT, MED_LARGE_BUFF
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in [*m_eq[0], *ms]], lag_ratio=0.2)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    ant.animate(rate_func=rate_functions.there_and_back_with_pause)
                    .set_color(YELLOW)
                    .shift(DOWN / 3)
                    for ant in antennas[::-1]
                ],
                lag_ratio=0.8,
            )
        )

        self.wait(0.5)

        self.play(
            eulers_w_phase_m.animate.move_to(self.camera.frame),
            FadeOut(
                ms,
                m_eq,
                antennas,
                plane_wave_longer,
                wave_to_ant_2,
                wave_to_ant_3,
                wave_to_ant_4,
                wave_to_ant_5,
                wave_to_ant_6,
                delta_phi2,
                delta_phi3,
                delta_phi4,
                delta_phi5,
                delta_phi6,
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        fourier_eqn_new = MathTex(
            r"X[k] = \sum_{n=0}^{N-1} w[n] e^{-j 2 \pi \frac{k}{N} n}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).to_edge(UP, MED_SMALL_BUFF)
        fourier_eqn_new[0][12:16].set_color(BLUE)
        fourier_eqn_new[0][-9:].set_color(RED)
        self.add(fourier_eqn_new)
        self.remove(
            *new_fourier_eqn, *fourier_eqn_full, ref_eqn, fourier_eqn_full[0][:-9]
        )

        self.play(fourier_eqn_new.animate.shift(DOWN * config.frame_height * 1.2))

        self.wait(0.5)

        pa_fourier_eqn_almost = MathTex(
            r"X[k] = \sum_{n=0}^{N-1} w[n] e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).shift(DOWN * config.frame_height * 1.2)
        pa_fourier_eqn_almost[0][12:16].set_color(BLUE)
        pa_fourier_eqn_almost[0][-13:].set_color(RED)
        pa_fourier_eqn = MathTex(
            r"X[\theta] = \sum_{m=0}^{M-1} w[m] e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).shift(DOWN * config.frame_height * 1.2)
        pa_fourier_eqn[0][12:16].set_color(BLUE)
        pa_fourier_eqn[0][-13:].set_color(RED)

        self.play(
            fourier_eqn_new[0][-9:].animate.shift(UP * config.frame_width * 1.2),
            ReplacementTransform(eulers_w_phase_m[0], pa_fourier_eqn_almost[0][-13:]),
        )

        self.wait(0.5)

        self.play(
            ReplacementTransform(
                fourier_eqn_new[0][12:16], pa_fourier_eqn_almost[0][12:16]
            )
        )

        self.wait(0.5)

        self.play(
            ReplacementTransform(fourier_eqn_new[0][:12], pa_fourier_eqn_almost[0][:12])
        )

        self.wait(0.5)

        # fmt:off
        self.play(
            TransformByGlyphMap(
                pa_fourier_eqn_almost,
                pa_fourier_eqn,
                ([0,1,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                 [0,1,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]),
                ([9], [9], {"delay": .2}),
                ([14], [14], {"delay": .2}),
                ([5], [5], {"delay": .8}),
                ([2], [2], {"delay": 1.6}),
            )
        )
        # fmt:on

        self.wait(0.5)

        af_eqn = MathTex(
            r"F[\theta] = \sum_{m=0}^{M-1} w[m] e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).move_to(pa_fourier_eqn)
        af_eqn[0][12:16].set_color(BLUE)
        af_eqn[0][-13:].set_color(RED)

        # fmt:off
        self.play(
            TransformByGlyphMap(
                pa_fourier_eqn,
                af_eqn,
                ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
                 [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]),
                ([0], [0]),
            )
        )
        # fmt:on

        self.wait(2)


class EquationTo2D(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))
        af_eqn = MathTex(
            r"F[\theta] = \sum_{m=0}^{M-1} w[m] e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        af_eqn[0][12:16].set_color(BLUE)
        af_eqn[0][-13:].set_color(RED)
        self.add(af_eqn)

        self.play(af_eqn.animate.scale(0.7).to_corner(UR, MED_LARGE_BUFF))

        self.wait(0.5)

        N = 9
        M = 9

        n = np.arange(N)
        m = np.arange(M)
        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        theta_min = VT(-PI)
        theta_max = VT(0)
        af_opacity = VT(1)

        ep_exp_scale = VT(0)

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]

        r_min = -30

        f_patch = 10e9
        lambda_patch_0 = c / f_patch

        epsilon_r = 2.2
        h = 1.6e-3

        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 + 12 * h / lambda_patch_0
        ) ** -0.5
        L = lambda_patch_0 / (2 * np.sqrt(epsilon_eff))
        W = lambda_patch_0 / 2 * np.sqrt(2 / (epsilon_r + 1))

        x_len = config.frame_height * 0.6
        ax = (
            Axes(
                x_range=[r_min, -r_min, -r_min / 4],
                y_range=[r_min, -r_min, -r_min / 4],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=x_len,
                y_length=x_len,
            )
            .rotate(PI / 2)
            .set_opacity(1)
            .to_edge(LEFT)
        )

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

        theta_dot = always_redraw(
            lambda: Dot().move_to(ax.input_to_graph_point(~theta_max, AF_plot))
        )
        theta_tracker = always_redraw(
            lambda: MathTex(
                f"\\theta = {~theta_max / PI:.2f} \\cdot \\pi",
                font_size=DEFAULT_FONT_SIZE * 1.8,
            )
            .scale(0.7)
            .next_to(af_eqn, DOWN, MED_SMALL_BUFF, LEFT)
        )
        self.play(FadeIn(theta_tracker), Create(ax), run_time=2)

        self.wait(0.5)

        self.play(Create(theta_dot))

        self.wait(0.5)

        self.play(theta_max @ (-PI))

        self.wait(0.5)

        self.add(AF_plot)
        self.play(theta_max @ PI, run_time=8, rate_func=rate_functions.ease_in_out_quad)

        self.wait(0.5)

        self.play(FadeOut(ax, AF_plot, theta_tracker, theta_dot))

        self.wait(0.5)

        one_d = Tex("1D", font_size=DEFAULT_FONT_SIZE * 3)
        two_d = Tex("2D", font_size=DEFAULT_FONT_SIZE * 3)
        Group(one_d, two_d).arrange(RIGHT, LARGE_BUFF * 2)
        one_to_two = Arrow(one_d.get_right(), two_d.get_left())
        Group(one_d, one_to_two, two_d).to_edge(UP, LARGE_BUFF)

        self.play(
            LaggedStart(
                af_eqn.animate.to_edge(DOWN, LARGE_BUFF).set_x(0),
                GrowFromCenter(one_d),
                GrowArrow(one_to_two),
                GrowFromCenter(two_d),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            ShrinkToCenter(one_d),
            FadeOut(one_to_two),
            ShrinkToCenter(two_d),
            af_eqn.animate.move_to(ORIGIN),
        )

        self.wait(2)


class TwoDSpacing(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        af_eqn = MathTex(
            r"F[\theta] = \sum_{m=0}^{M-1} w[m] e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).scale(0.7)
        af_eqn[0][12:16].set_color(BLUE)
        af_eqn[0][-13:].set_color(RED)

        self.add(af_eqn)

        self.wait(0.5)

        antennas = Group()
        for _ in range(16):
            antenna_port = Line(ORIGIN, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.4)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange_in_grid(4, 4, buff=MED_LARGE_BUFF).to_edge(
            LEFT, LARGE_BUFF * 1.8
        )

        dx_line = Line(antennas[-4].get_bottom(), antennas[-3].get_bottom()).shift(
            DOWN / 3
        )
        dx_line_l = Line(dx_line.get_start() + DOWN / 8, dx_line.get_start() + UP / 8)
        dx_line_r = Line(dx_line.get_end() + DOWN / 8, dx_line.get_end() + UP / 8)
        dx_label = MathTex("d_x").next_to(dx_line_r, RIGHT)

        dy_line = Line(antennas[-4].get_corner(UL), antennas[-8].get_corner(UL)).shift(
            LEFT / 3
        )
        dy_line_l = Line(
            dy_line.get_start() + LEFT / 8, dy_line.get_start() + RIGHT / 8
        )
        dy_line_r = Line(dy_line.get_end() + LEFT / 8, dy_line.get_end() + RIGHT / 8)
        dy_label = MathTex("d_y").next_to(dy_line_r, UP)

        theta = MathTex(r"\theta", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            antennas, DOWN, LARGE_BUFF * 0.8
        )
        theta_l = Arrow(
            theta.get_left(), [antennas.get_corner(DL)[0], theta.get_left()[1], 0]
        )
        theta_r = Arrow(
            theta.get_right(), [antennas.get_corner(DR)[0], theta.get_right()[1], 0]
        )
        phi = MathTex(r"\phi", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            antennas, LEFT, LARGE_BUFF
        )
        phi_l = Arrow(
            phi.get_bottom(),
            [phi.get_bottom()[0], antennas.get_corner(DL)[1], 0],
        )
        phi_r = Arrow(
            phi.get_top(),
            [phi.get_top()[0], antennas.get_corner(UL)[1], 0],
        )

        self.play(
            af_eqn.animate.scale(0.8).to_corner(UR, MED_LARGE_BUFF),
            LaggedStart(*[GrowFromCenter(m) for m in antennas[-4:]]),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(dx_line_l),
                Create(dx_line),
                Create(dx_line_r),
                GrowFromCenter(dx_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(theta),
                AnimationGroup(GrowArrow(theta_l), GrowArrow(theta_r)),
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(*[GrowFromCenter(m) for m in antennas[:-4]]))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(dy_line_l),
                Create(dy_line),
                Create(dy_line_r),
                GrowFromCenter(dy_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(phi),
                AnimationGroup(GrowArrow(phi_l), GrowArrow(phi_r)),
            )
        )

        self.wait(0.5)

        m_brace = Brace(antennas, DOWN, LARGE_BUFF * 0.6)
        n_brace = Brace(antennas, LEFT, LARGE_BUFF * 0.6)
        m_label = MathTex("m").next_to(m_brace, DOWN, SMALL_BUFF)
        n_label = MathTex("n").next_to(n_brace, LEFT, SMALL_BUFF)
        m_group = Group(m_brace, m_label)
        n_group = Group(n_brace, n_label)

        self.play(
            LaggedStart(
                Group(theta, theta_l, theta_r).animate.shift(DOWN * 5),
                m_group.shift(DOWN * 6).animate.shift(UP * 6),
                Group(phi, phi_l, phi_r).animate.shift(LEFT * 5),
                n_group.shift(LEFT * 6).animate.shift(RIGHT * 6),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        bm = MathTex("b_m", font_size=DEFAULT_FONT_SIZE * 1.5, color=BLUE)
        cn = MathTex("c_n", font_size=DEFAULT_FONT_SIZE * 1.5, color=BLUE)
        w_n_copy = af_eqn[0][12:16].copy().shift(DOWN * 3 + LEFT)
        amn = MathTex(
            r"a_{m,n}", font_size=DEFAULT_FONT_SIZE * 1.5, color=BLUE
        ).move_to(w_n_copy, RIGHT)
        weights = (
            Group(bm, cn)
            .arrange(DOWN, LARGE_BUFF)
            .next_to(w_n_copy, RIGHT, LARGE_BUFF * 1.5)
        )
        bm_bez = CubicBezier(
            w_n_copy.get_right() + [0.1, 0, 0],
            w_n_copy.get_right() + [1, 0, 0],
            bm.get_left() + [-1, 0, 0],
            bm.get_left() + [-0.1, 0, 0],
        )
        cn_bez = CubicBezier(
            w_n_copy.get_right() + [0.1, 0, 0],
            w_n_copy.get_right() + [1, 0, 0],
            cn.get_left() + [-1, 0, 0],
            cn.get_left() + [-0.1, 0, 0],
        )

        self.play(TransformFromCopy(af_eqn[0][12:16], w_n_copy, path_arc=PI / 2))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(bm_bez),
                GrowFromCenter(bm),
                Create(cn_bez),
                GrowFromCenter(cn),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(ReplacementTransform(w_n_copy, amn))

        self.wait(0.5)

        af_theta_eqn = MathTex(
            r"F[\theta] = \sum_{m=0}^{M-1} b_m e^{-j k d_x \sin{(\theta) m}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).scale(0.7)
        af_phi_eqn = MathTex(
            r"F[\phi] = \sum_{n=0}^{N-1} c_n e^{-j k d_y \sin{(\phi) n}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).scale(0.7)
        Group(af_theta_eqn, af_phi_eqn).arrange(DOWN, LARGE_BUFF)

        af_2d_eqn = MathTex(
            r"F[\theta, \phi] = \sum_{m=0}^{M-1} b_m e^{-j k d_x \sin{(\theta) m}} \sum_{n=0}^{N-1} c_n e^{-j k d_y \sin{(\phi) n}}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).scale(0.7)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Group(
                        antennas,
                        m_group,
                        n_group,
                        dx_label,
                        dx_line,
                        dx_line_l,
                        dx_line_r,
                        dy_label,
                        dy_line,
                        dy_line_l,
                        dy_line_r,
                    ).animate.shift(LEFT * 10),
                    Group(amn, bm, cn, cn_bez, bm_bez).animate.shift(RIGHT * 10),
                ),
                AnimationGroup(
                    ReplacementTransform(af_eqn, af_theta_eqn, path_arc=-PI / 3),
                    TransformFromCopy(af_eqn, af_phi_eqn, path_arc=PI / 3),
                ),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        # fmt:off
        self.play(
            TransformByGlyphMap(
                af_theta_eqn,
                af_2d_eqn,
                ([0, 1, 2], [0, 1, 2]),
                ([3], [5]),
                ([4], [6]),
                ([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
                 [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]),
                (GrowFromCenter, [3]),
                (get_transform_func(af_phi_eqn[0][3], ReplacementTransform),
                 [4]),
                (get_transform_func(af_phi_eqn[0][6:], ReplacementTransform),
                 [29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]),
            ),
            FadeOut(af_phi_eqn[0][:3], af_phi_eqn[0][4:6]),
            run_time=1.5
        )
        # fmt:on

        self.wait(0.5)

        mailloux = (
            ImageMobject("../props/static/mailloux.jpg")
            .scale_to_fit_height(config.frame_height * 0.7)
            .to_edge(DOWN, -MED_SMALL_BUFF)
        )

        self.play(mailloux.shift(DOWN * 10).animate.shift(UP * 10))

        self.wait(0.5)

        self.play(mailloux.animate.shift(DOWN * 10))

        self.wait(0.5)

        dot = (
            MathTex(r"\cdot", font_size=DEFAULT_FONT_SIZE * 1.8)
            .scale(0.7)
            .next_to(
                af_2d_eqn[0][:-22].copy().scale(0.7).to_edge(LEFT), RIGHT, SMALL_BUFF
            )
        )

        self.play(
            LaggedStart(
                af_2d_eqn[0][:-22].animate.scale(0.7).to_edge(LEFT),
                Create(dot),
                af_2d_eqn[0][-22:]
                .animate.scale(0.7)
                .next_to(
                    af_2d_eqn[0][:-22].copy().scale(0.7).to_edge(LEFT)[7],
                    DOWN,
                    LARGE_BUFF * 1.5,
                    LEFT,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        theta_vals = MathTex(
            r"-\pi \le \theta \le \pi", font_size=DEFAULT_FONT_SIZE * 1.5
        )
        phi_vals = MathTex(r"-\pi \le \phi \le \pi", font_size=DEFAULT_FONT_SIZE * 1.5)
        Group(theta_vals, phi_vals).arrange(DOWN, LARGE_BUFF).to_edge(RIGHT, LARGE_BUFF)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in [*theta_vals[0], *phi_vals[0]]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        video = (
            VideoMobject("./static/af_2d_anim_rotating.mp4", speed=1, loop=False)
            .scale_to_fit_width(config.frame_width * 0.7)
            .next_to([config.frame_width / 2, 0, 0], RIGHT)
            .set_z_index(-1)
        )
        self.add(video)

        self.play(
            LaggedStart(
                FadeOut(theta_vals, phi_vals),
                video.animate.move_to([config.frame_width * 0.25, 0, 0]),
                lag_ratio=0.4,
            )
        )

        self.wait(15, frozen_frame=False)

        self.play(
            Group(af_2d_eqn, dot).animate.shift(LEFT * 12),
            video.animate.move_to([-config.frame_width * 0.25, 0, 0]),
        )

        self.wait(0.5)

        sim = MathTex(r"\sim", font_size=DEFAULT_FONT_SIZE * 1.5)

        list = (
            BulletedList(
                "$M, N$",
                "$b_m, c_n$",
                "$d_x, d_y$",
            )
            .scale_to_fit_width(config.frame_width * 0.15)
            .next_to(sim, RIGHT, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                GrowFromCenter(sim),
                *[GrowFromCenter(m) for m in list],
                lag_ratio=0.8,
            )
        )

        self.wait(5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class TwoD(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(
            phi=90 * DEGREES,
            theta=0 * DEGREES,
            zoom=0.7,
        )

        N = 17
        M = 17
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
        d_y = wavelength_0 / 2

        steering_angle_theta = 0
        steering_angle_phi = 0

        u_0 = np.sin(steering_angle_theta * PI / 180)
        v_0 = np.sin(steering_angle_phi * PI / 180)

        theta_min = VT(-PI)
        theta_max = VT(0)
        af_opacity = VT(1)

        ep_exp_scale = VT(0)

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]

        r_min = -50

        f_patch = 10e9
        lambda_patch_0 = c / f_patch

        epsilon_r = 2.2
        h = 1.6e-3

        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 + 12 * h / lambda_patch_0
        ) ** -0.5
        L = lambda_patch_0 / (2 * np.sqrt(epsilon_eff))
        W = lambda_patch_0 / 2 * np.sqrt(2 / (epsilon_r + 1))

        axes = ThreeDAxes(
            tips=False,
            # x_range=(-PI, PI, PI / 2),
            x_range=(-0.5, 0.5, 1 / 4),
            y_range=(-0.5, 0.5, 1 / 4),
            z_range=(-0.5, 0.5, 1 / 4),
            # x_range=[-1, 1, 1 / 2],
            # y_range=[-1, 1, 1 / 2],
            # z_range=[0, -r_min],
            # x_length=8,
        )
        axes_labels = axes.get_axis_labels("x", "y", "z")

        # def get_ap(color=BLUE):
        u_0 = np.sin(~steering_angle * PI / 180)
        weights = np.array([~w for w in weight_trackers])
        AF = compute_af_1d(weights, d_x, k_0, u, u_0)
        EP = sinc_pattern(u, 0, L, W, wavelength_0)
        AP = AF * (EP ** (~ep_exp_scale))
        AP_log = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
        AP_log /= AP_log.max()
        f_AP = interp1d(u * PI, AP_log, fill_value="extrapolate")
        AF_theta_plot = (
            ParametricFunction(
                lambda t: (t, f_AP(t), 0),
                t_range=(-PI, PI),
                color=BLUE,
            )
            .stretch_to_fit_height(axes.height)
            .stretch_to_fit_width(axes.width)
            .move_to(axes)
            .rotate_about_origin(PI / 2, axis=[0, 1, 0])
            .rotate_about_origin(PI / 2, axis=[1, 0, 0])
        )
        AF_phi_plot = (
            ParametricFunction(
                lambda t: (t, f_AP(t), 0),
                t_range=(-PI, PI),
                color=RED,
            )
            .stretch_to_fit_height(axes.height)
            .stretch_to_fit_width(axes.width)
            .move_to(axes)
            .rotate_about_origin(PI / 2, axis=[0, 1, 0])
            .rotate_about_origin(PI / 2, axis=[1, 0, 0])
        )

        self.play(
            Create(axes),
            # FadeIn(axes_labels),
        )

        self.wait(0.5)

        self.play(Create(AF_theta_plot))

        self.wait(0.5)

        self.play(FadeIn(AF_phi_plot))
        # self.move_camera(
        #     phi=0 * DEGREES,
        #     theta=0 * DEGREES,
        #     gamma=10 * DEGREES,
        #     # added_anims=[AF_phi_plot.animate.rotate_about_origin(PI / 2, UP)],
        # )
        self.move_camera(
            phi=75 * DEGREES,
            theta=30 * DEGREES,
            zoom=0.7,
            added_anims=[AF_phi_plot.animate.rotate_about_origin(PI / 2, [0, 0, 1])],
            run_time=2,
        )
        self.begin_ambient_camera_rotation(rate=0.1, about="theta")

        self.wait(5)

        window_n = np.ones(N)
        window_m = np.ones(M)

        U_vis = 40
        V_vis = 40
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
            - r_min,
            0,
            None,
        )
        # AF /= AF.max()
        print(AP_log.max() - AF.max())
        AF -= AP_log.max() - AF.max()

        tck = bisplrep(U, V, AF)

        umax = VT(0.01)
        vmax = VT(0.01)

        axes = ThreeDAxes(
            x_range=[-1, 1, 1 / 2],
            y_range=[-1, 1, 1 / 2],
            z_range=[0, -z_min],
            x_length=8,
        )

        def get_surface():
            surf = (
                Surface(
                    lambda u, v: axes.c2p(u, v, bisplev(u, v, tck)),
                    u_range=[-~umax, ~umax],
                    v_range=[-~vmax, ~vmax],
                    resolution=(U_vis, V_vis),
                )
                .set_z(0)
                .set_style(fill_opacity=1)
                .set_fill_by_value(
                    axes=axes, colorscale=[(BLUE, 0), (RED, -z_min - 20)], axis=2
                )
            )
            surf.shift(
                AF_theta_plot.get_boundary_point([0, 0, 1])
                - surf.get_boundary_point([0, 0, 1])
            )
            print(
                AF_theta_plot.get_boundary_point([0, 0, 1])
                - surf.get_boundary_point([0, 0, 1])
            )
            return surf

        surface = always_redraw(get_surface)

        # self.stop_ambient_camera_rotation()

        # self.move_camera(
        #     phi=0 * DEGREES,
        #     theta=0 * DEGREES,
        #     zoom=0.6,
        #     run_time=2,
        # )

        self.wait(0.5)

        self.add(surface)
        self.play(
            LaggedStart(
                FadeOut(AF_theta_plot, AF_phi_plot),
                AnimationGroup(umax @ 1, vmax @ 1),
                lag_ratio=0.5,
            ),
            run_time=8,
        )

        # self.add(surface)
        self.wait(10)


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


class Thumbnail(Scene):
    def construct(self):
        title = (
            Tex("Phased Arrays")
            .scale_to_fit_width(config.frame_width * 0.5)
            .to_edge(UP, LARGE_BUFF)
        )
        rect_pattern = ImageMobject("./static/rect.png").scale_to_fit_width(
            config.frame_width * 0.6
        )

        antennas = Group()
        for _ in range(16):
            antenna_port = Line(ORIGIN, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.4)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange_in_grid(4, 4, buff=MED_LARGE_BUFF).scale(0.8)
        Group(antennas, rect_pattern).arrange(RIGHT, LARGE_BUFF).to_edge(
            DOWN, LARGE_BUFF
        ).shift(RIGHT + DOWN * 2)
        self.add(title, rect_pattern, antennas)


class Thumbnail2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        antenna_port_left = Line(DOWN * 2, UP, color=GREEN)
        antenna_tri_left = (
            Triangle(color=GREEN).rotate(PI / 3).move_to(antenna_port_left, UP)
        )
        antenna_port_right = Line(DOWN * 2, UP, color=GREEN)
        antenna_tri_right = (
            Triangle(color=GREEN).rotate(PI / 3).move_to(antenna_port_right, UP)
        )

        antenna_left = Group(antenna_port_left, antenna_tri_left)
        antenna_right = Group(antenna_port_right, antenna_tri_right)
        antennas = (
            Group(antenna_left, antenna_right)
            .arrange(RIGHT, LARGE_BUFF * 20)
            .to_edge(DOWN, -SMALL_BUFF)
        )

        self.play(
            antennas.animate.arrange(RIGHT, LARGE_BUFF * 5).to_edge(DOWN, -SMALL_BUFF)
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
                    m.animate.set_stroke(opacity=1)
                    for m in [
                        antenna_tri_left,
                        antenna_port_left,
                        antenna_tri_right,
                        antenna_port_right,
                    ]
                ]
            ),
            cloud.animate.shift(UP * 5),
            self.camera.frame.animate.scale(0.9).shift(UP / 2),
        )

        self.wait(0.5)

        ap_label = (
            Tex("Phased Arrays")
            .next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF)
            .scale(2)
        )
        self.add(ap_label)

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

        self.wait(0.5)

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
        )
