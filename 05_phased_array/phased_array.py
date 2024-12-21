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

        self.play(self.camera.frame.animate.restore(), combined_sine_x1 @ 0)

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

        self.wait(0.5)

        self.play(plane_wave.animate.set_y(antennas.get_top()[1]))
        self.play(FadeOut(plane_wave))

        self.wait(2)


# TODO: Re-render
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

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        dft_eqn = MathTex(
            r"X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-j 2 \pi \frac{k}{N} n}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )

        self.play(
            LaggedStart(
                GrowFromCenter(dft_eqn[0][:2]),
                GrowFromCenter(dft_eqn[0][2]),
                GrowFromCenter(dft_eqn[0][3:10]),
                GrowFromCenter(dft_eqn[0][10:12]),
                GrowFromCenter(dft_eqn[0][12]),
                GrowFromCenter(dft_eqn[0][13]),
                GrowFromCenter(dft_eqn[0][14:16]),
                GrowFromCenter(dft_eqn[0][16]),
                GrowFromCenter(dft_eqn[0][17]),
                GrowFromCenter(dft_eqn[0][18:21]),
                GrowFromCenter(dft_eqn[0][21]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        what_want = Tex(
            "what do we actually want?", font_size=DEFAULT_FONT_SIZE * 1.5
        ).next_to([0, config.frame_height / 2, 0], UP)

        self.play(Group(what_want, dft_eqn).animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        self.play(
            LaggedStart(*[ShrinkToCenter(m) for m in dft_eqn[0][::-1]], lag_ratio=0.1),
            LaggedStart(
                *[ShrinkToCenter(m) for m in what_want[0][::-1]], lag_ratio=0.1
            ),
        )

        self.wait(2)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


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

        self.next_section(skip_animations=skip_animations(False))
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

        def get_fft():
            N = max_time * fs
            t = np.linspace(0, max_time, N)
            sig = np.sin(2 * PI * ~f * t) + ~offset

            fft_len = N * 8
            sig_fft = fftshift(np.abs(fft(sig, fft_len) / (N)))
            # fft_log = 10 * np.log10(np.abs(fftshift(sig_fft))) + 40
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            f_fft_log = interp1d(freq, sig_fft)
            return f_ax.plot(
                f_fft_log, x_range=[-~fft_f, ~fft_f, 1 / fs], color=TX_COLOR
            )

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

        wave_amp_plot = always_redraw(
            lambda: wave_amp_ax.plot(
                lambda t: np.sin(2 * PI * ~wave_f * t) + ~wave_offset,
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )
        product_amp_plot = always_redraw(
            lambda: product_amp_ax.plot(
                lambda t: (np.sin(2 * PI * ~wave_f * t) + ~wave_offset)
                * (np.sin(2 * PI * ~f * t) + ~offset),
                x_range=[0, 1, 1 / fs],
                color=TX_COLOR,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
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

        self.play(Create(wave_amp_plot))

        self.wait(0.5)

        self.play(product_amp_plot_group.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

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

        # self.add(
        #     # samples_neg_stacked,
        #     # samples_pos_stacked,
        #     product_samples_pos_static,
        #     product_samples_neg_static,
        # )

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

        # def get_dot_updater(x):
        #     def updater(m):
        #         plot = product_amp_ax.plot(
        #             lambda t: (np.sin(2 * PI * ~wave_f * t) + ~wave_offset)
        #             * (np.sin(2 * PI * ~f * t) + ~offset),
        #             x_range=[0, 1, 1 / fs],
        #         )
        #         m.become(
        #             Dot(
        #                 product_amp_ax.c2p(x, 0),
        #                 product_amp_ax.input_to_graph_point(x, plot),
        #                 color=RED
        #                 if product_amp_ax.input_to_graph_coords(x, plot)[1] > 0
        #                 else PURPLE,
        #             )
        #         )

        #     return updater

        # samples_updaters = []
        # dots_updaters = [
        #     get_dot_updater(x) for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        # ]

        # for dot, updater in zip(product_sample_dots, dots_updaters):
        #     dot.add_updater(updater)

        # product_samples = Group(
        #     *[
        #         Line(
        #             product_amp_ax.c2p(x, 0),
        #             product_amp_ax.input_to_graph_point(x, product_amp_plot),
        #             color=RED
        #             if product_amp_ax.input_to_graph_coords(x, product_amp_plot)[1] > 0
        #             else PURPLE,
        #         )
        #         for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        #     ]
        # )
        # product_sample_dots = Group(
        #     *[
        #         Dot(
        #             product_amp_ax.input_to_graph_point(x, product_amp_plot),
        #             color=RED
        #             if product_amp_ax.input_to_graph_coords(x, product_amp_plot)[1] > 0
        #             else PURPLE,
        #         )
        #         for x in np.linspace(0.0625, 1 - 0.0625, num_samples)
        #     ]
        # )

        # self.wait(.5)

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

        def get_line(x):
            def updater():
                p0 = product_amp_ax.c2p(x, 0)
                p1 = product_amp_ax.input_to_graph_point(x, product_amp_plot)
                c1 = product_amp_ax.input_to_graph_coords(x, product_amp_plot)
                color = RED if c1[1] >= 0 else PURPLE
                line = Line(p0, p1).set_color(color)
                return line

            return updater

        def get_dot(x):
            def updater():
                p1 = product_amp_ax.input_to_graph_point(x, product_amp_plot)
                c1 = product_amp_ax.input_to_graph_coords(x, product_amp_plot)
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
            .rotate(PI / 2)
            .move_to(ax)
        )

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
                lambda u, v: axes.c2p(
                    spherical_to_cartesian((bisplev(u, v, tck), u, v))
                ),
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

        self.set_camera_orientation(theta=45 * DEGREES, phi=50 * DEGREES, zoom=0.7)

        self.add(surface)
