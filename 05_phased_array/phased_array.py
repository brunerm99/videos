# phased_array.py

import matplotlib.pyplot as plt

import sys
import warnings

import numpy as np
from numpy.fft import fft, fftshift
from manim import *
from scipy.interpolate import interp1d
from scipy.constants import c
from MF_Tools import VT, TransformByGlyphMap


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR
from props import WeatherRadarTower, get_blocks

config.background_color = BACKGROUND_COLOR

BLOCKS = get_blocks()

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


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

        self.play(self.camera.frame.animate.restore(), combined_sine_x0 @ 1)

        self.next_section(skip_animations=skip_animations(False))
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

        shifted_combined_sine = always_redraw(
            lambda: combined_ax.plot(
                lambda t: amp_scale * np.sin(2 * PI * sine_f * t + ~phase_left)
                + amp_scale * np.sin(2 * PI * sine_f * t + ~phase_right),
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

        self.wait(2)


class FourierAnalogy(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
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
        amp_plot_group = Group(amp_ax, amp_labels)
        f_plot_group = Group(f_ax, f_labels)
        axes = Group(amp_plot_group, f_plot_group)

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

        self.play(
            amp_plot_group.next_to([0, config.frame_height / 2, 0], UP).animate.move_to(
                ORIGIN
            )
        )

        self.wait(0.5)

        self.play(Create(amp_plot))

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

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        sinc = Tex(r"$\lvert$sinc$\rvert = \frac{\sin{(x)}}{x}$").next_to(
            f_ax.c2p(0, 2), RIGHT, MED_LARGE_BUFF
        )

        self.play(GrowFromCenter(sinc[0][1:5]))

        self.wait(0.5)

        self.play(GrowFromCenter(sinc[0][0]), GrowFromCenter(sinc[0][5]))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in [
                        sinc[0][6],
                        sinc[0][7:13],
                        sinc[0][13],
                        sinc[0][14],
                    ]
                ],
                lag_ratio=0.2,
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

        antennas.next_to(amp_ax, UP, LARGE_BUFF)

        self.play(
            LaggedStart(
                *[GrowFromCenter(antenna) for antenna in antennas], lag_ratio=0.2
            )
        )

        self.wait(2)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


class CircularCoords(Scene):
    def construct(self):
        n_elem = 21  # Must be odd
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
                theta,
                np.clip(20 * np.log10(np.abs(AF)) - r_min, 0, None),
                fill_value="extrapolate",
            )
            plot = ax.plot_polar_graph(r_func=f_AF, theta_range=[-PI, PI, 2 * PI / 200])

            return plot

        AF_plot = always_redraw(get_af)

        self.next_section(skip_animations=skip_animations(True))
        self.add(ax, AF_plot)

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

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(steering_angle @ (10))

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
