# phased_array.py

import sys
import warnings

import numpy as np
from manim import *
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
            Tex("Plane wave").set_opacity(0).next_to(zoomed_display_frame)
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
        self.wait(0.5)

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

        self.next_section(skip_animations=skip_animations(False))

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

        self.play(
            filt_sine_x0 @ 1,
            combined_sine_x1 @ 1,
            rate_func=rate_functions.ease_out_sine,
        )

        self.wait(2)
