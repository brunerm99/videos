# fmcw_doppler.py

import sys
import warnings

from random import shuffle
import numpy as np
from manim import *
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fftshift, fft2, fft
from scipy import signal, interpolate
from scipy.constants import c
from scipy.interpolate import bisplrep, bisplev


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR, IF_COLOR
from props.helpers import get_plot_values
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


# TODO: Re-render
class TriangularIntro(MovingCameraScene):
    def construct(self):
        carrier_freq = 10
        modulation_freq = 0.5
        modulation_index = 20
        duration = 1
        final_beat_time_shift = 0.2
        fs = 1000
        step = 1 / fs

        x_len = config["frame_width"] * 0.7
        y_len = config["frame_height"] * 0.38

        x0_tri = VT(0.0)
        x1_tri = VT(0.0)
        x0_saw_f = VT(0.0)
        x1_saw_f = VT(0.0)
        x0_saw_amp = VT(0.0)
        x1_saw_amp = VT(0.0)

        amp_ax = Axes(
            x_range=[-0.1, duration + final_beat_time_shift, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        f_ax = Axes(
            x_range=[-0.1, duration + final_beat_time_shift, duration / 4],
            y_range=[-2, 30, 5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        amp_labels = amp_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )
        unknowns = f_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        """ Triangular """
        triangular_modulating_signal = lambda t: modulation_index * np.arcsin(
            np.sin(2 * np.pi * modulation_freq * t)
        )
        triangular_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(triangular_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        triangular_amp = lambda t: np.sin(2 * np.pi * triangular_modulating_cumsum(t))

        tri_f_graph = always_redraw(
            lambda: f_ax.plot(
                triangular_modulating_signal,
                x_range=[~x0_tri, ~x1_tri, 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )
        triangular_amp_graph = always_redraw(
            lambda: amp_ax.plot(
                triangular_amp,
                x_range=[~x0_tri, ~x1_tri, 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )

        """ Sawtooth """
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        sawtooth_amp = lambda t: np.sin(2 * PI * sawtooth_modulating_cumsum(t))

        sawtooth_f_graph = always_redraw(
            lambda: f_ax.plot(
                sawtooth_modulating_signal,
                x_range=[~x0_saw_f, ~x1_saw_f, 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )
        sawtooth_amp_graph = always_redraw(
            lambda: amp_ax.plot(
                sawtooth_amp,
                x_range=[~x0_saw_amp, ~x1_saw_amp, 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )

        amp_ax_group = VGroup(amp_ax, amp_labels, sawtooth_amp_graph)
        f_ax_group = VGroup(f_ax, unknowns, sawtooth_f_graph)

        self.next_section(skip_animations=skip_animations(True))
        self.add(
            sawtooth_amp_graph,
            sawtooth_f_graph,
            triangular_amp_graph,
            tri_f_graph,
        )
        self.play(
            Group(unknowns, f_ax)
            .next_to([0, config["frame_height"] / 2, 0], UP)
            .animate.move_to(ORIGIN)
        )

        self.wait(0.5)

        self.play(x1_saw_f @ (duration - step))

        self.wait(0.5)

        self.play(
            Group(unknowns, f_ax).animate.to_edge(UP, MED_SMALL_BUFF),
            Group(amp_labels, amp_ax)
            .next_to([0, -config["frame_height"] / 2, 0], DOWN)
            .animate.to_edge(DOWN, MED_SMALL_BUFF),
        )

        self.wait(0.5)

        self.play(x1_saw_amp @ (duration - step))

        self.wait(0.5)

        self.play(
            x0_saw_amp @ (duration - step),
            x0_saw_f @ (duration - step),
            x1_tri @ (duration - step),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            Group(amp_labels, amp_ax).animate.next_to(
                [0, -config["frame_height"] / 2, 0], DOWN
            ),
            Group(unknowns, f_ax).animate.move_to(ORIGIN),
        )
        self.remove(amp_labels, amp_ax, triangular_amp_graph, sawtooth_amp_graph)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        down_ramp_hl = Line(
            f_ax.input_to_graph_point(duration / 2, tri_f_graph),
            f_ax.input_to_graph_point(duration, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        down_ramp_label = (
            Tex("Down Ramp")
            .next_to(f_ax.input_to_graph_point(duration * 0.6, tri_f_graph), RIGHT, 0)
            .rotate(down_ramp_hl.get_angle())
            .shift(LEFT / 3 + DOWN / 3)
        )

        self.play(Create(down_ramp_hl), FadeIn(down_ramp_label))

        self.wait(0.5)

        self.play(Uncreate(down_ramp_hl), FadeOut(down_ramp_label))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        beat_time_shift = VT(0)  # s
        doppler_f_shift = VT(0)  # Hz
        # beat_pts_shift = (f_ax.c2p(beat_time_shift, 0) - f_ax.c2p(0, 0)) * RIGHT
        # doppler_pts_shift = (f_ax.c2p(0, doppler_f_shift) - f_ax.c2p(0, 0)) * UP

        tri_f_graph_rx = always_redraw(
            lambda: f_ax.plot(
                lambda t: modulation_index
                * np.arcsin(
                    np.sin(2 * np.pi * modulation_freq * (t - ~beat_time_shift))
                )
                + ~doppler_f_shift,
                x_range=[
                    ~x0_tri + ~beat_time_shift,
                    ~x1_tri + ~beat_time_shift,
                    1 / fs,
                ],
                use_smoothing=False,
                color=RX_COLOR,
            )
        )

        self.play(Create(tri_f_graph_rx))
        self.play(beat_time_shift @ final_beat_time_shift)

        self.wait(0.5)

        time_shift_line = Line(
            f_ax.input_to_graph_point(duration / 2, tri_f_graph) + DOWN * 2,
            f_ax.input_to_graph_point(duration / 2 + ~beat_time_shift, tri_f_graph_rx)
            + DOWN * 2,
        )
        time_shift_line_l = Line(
            time_shift_line.get_start() + DOWN / 8, time_shift_line.get_start() + UP / 8
        )
        time_shift_line_r = Line(
            time_shift_line.get_end() + DOWN / 8, time_shift_line.get_end() + UP / 8
        )

        time_shift_label = MathTex(r"t_{shift}").next_to(
            time_shift_line, UP, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                Create(time_shift_line_l),
                Create(time_shift_line),
                Create(time_shift_line_r),
                FadeIn(time_shift_label, shift=DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        t_dot = VT(~beat_time_shift)

        f_tx_dot = always_redraw(
            lambda: Dot().move_to(f_ax.input_to_graph_point(~t_dot, tri_f_graph))
        )
        f_rx_dot = always_redraw(
            lambda: Dot().move_to(f_ax.input_to_graph_point(~t_dot, tri_f_graph_rx))
        )
        f_tx_label = always_redraw(
            lambda: MathTex(r"f_{TX}", color=TX_COLOR).next_to(f_tx_dot, UL, SMALL_BUFF)
        )
        f_rx_label = always_redraw(
            lambda: MathTex(r"f_{RX}", color=RX_COLOR).next_to(f_rx_dot, UL, SMALL_BUFF)
        )

        self.play(Create(f_tx_dot), Create(f_rx_dot), FadeIn(f_tx_label, f_rx_label))

        self.wait(0.5)

        f_beat_eqn = MathTex(r"f_{TX}", " - ", r"f_{RX}", r"= f_{beat}").to_corner(UL)
        f_beat_eqn[0].set_color(TX_COLOR)
        f_beat_eqn[2].set_color(RX_COLOR)
        f_beat_eqn[3][1:].set_color(IF_COLOR)

        self.play(t_dot @ (duration / 2), run_time=2)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(f_tx_dot),
                ReplacementTransform(f_tx_label, f_beat_eqn[0]),
                Uncreate(f_rx_dot),
                FadeIn(f_beat_eqn[1]),
                ReplacementTransform(f_rx_label, f_beat_eqn[2]),
                FadeIn(f_beat_eqn[3]),
                lag_ratio=0.3,
            )
        )

        # self.wait(0.5)

        # self.play(
        #     Uncreate(time_shift_line),
        #     Uncreate(time_shift_line_l),
        #     Uncreate(time_shift_line_r),
        #     Uncreate(time_shift_label),
        # )

        self.wait(0.5)

        vel_eqn = MathTex(r"v_{\text{target}} ", r"<", r" 0 \text{ m/s}").to_corner(UR)

        self.play(Create(vel_eqn))

        self.wait(0.5)

        self.play(doppler_f_shift @ -5)

        self.wait(0.5)

        self.play(
            Transform(vel_eqn[1], MathTex(">").move_to(vel_eqn[1])),
            doppler_f_shift @ 5,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        doppler_shift_x = f_ax.input_to_graph_point(
            duration / 2 + ~beat_time_shift / 4, tri_f_graph
        )[0]
        doppler_shift_y0 = f_ax.input_to_graph_point(duration / 2, tri_f_graph)[1]
        doppler_shift_y1 = f_ax.input_to_graph_point(
            duration / 2 + ~beat_time_shift, tri_f_graph_rx
        )[1]
        doppler_shift_line = Line(
            [doppler_shift_x, doppler_shift_y0, 0],
            [doppler_shift_x, doppler_shift_y1, 0],
        )
        doppler_shift_line_l = Line(
            doppler_shift_line.get_start() + LEFT / 8,
            doppler_shift_line.get_start() + RIGHT / 8,
        )
        doppler_shift_line_h = Line(
            doppler_shift_line.get_end() + LEFT / 8,
            doppler_shift_line.get_end() + RIGHT / 8,
        )

        doppler_shift_label_long = MathTex(r"f_{\text{Doppler}}").next_to(
            doppler_shift_line, UP, MED_SMALL_BUFF
        )

        self.play(
            LaggedStart(
                Create(doppler_shift_line_l),
                Create(doppler_shift_line),
                Create(doppler_shift_line_h),
                FadeIn(doppler_shift_label_long),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        doppler_shift_label = MathTex(r"f_{d}").next_to(
            doppler_shift_line_h, UP, MED_SMALL_BUFF
        )

        self.play(
            TransformByGlyphMap(
                doppler_shift_label_long,
                doppler_shift_label,
                ([0], [0], {"delay": 0.4}),
                ([1], [1], {"delay": 0.2}),
                ([2, 3, 4, 5, 6, 7], []),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(1.4).shift(DOWN))

        self.wait(0.5)

        mixer = (
            BLOCKS.get("mixer")
            .copy()
            .next_to(self.camera.frame.get_bottom(), UP, MED_LARGE_BUFF)
        )
        lo_arrow = Arrow(
            mixer.get_top() + UP * 2, mixer.get_top(), color=TX_COLOR, buff=0
        )
        rf_arrow = Arrow(
            mixer.get_left() + LEFT * 3, mixer.get_left(), color=RX_COLOR, buff=0
        )
        if_arrow = Arrow(
            mixer.get_right(), mixer.get_right() + RIGHT * 3, color=IF_COLOR, buff=0
        )

        lo_label = MathTex(r"f_{TX}(t)", color=TX_COLOR).next_to(lo_arrow, LEFT)
        rf_label = MathTex(r"f_{RX}(t)", color=RX_COLOR).next_to(rf_arrow, UP)
        if_label = MathTex(r"f_{IF}(t)", color=IF_COLOR).next_to(if_arrow, UP)
        if_equals = MathTex(
            r"f_{IF}(t)", r" = \lvert ", r"f_{TX}(t)", " - ", r"f_{RX}(t)", r" \rvert"
        ).move_to(if_label, LEFT)
        if_equals[0].set_color(IF_COLOR)
        if_equals[2].set_color(TX_COLOR)
        if_equals[4].set_color(RX_COLOR)

        self.play(GrowFromCenter(mixer))

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(GrowArrow(lo_arrow), GrowFromCenter(lo_label)),
                AnimationGroup(GrowArrow(rf_arrow), GrowFromCenter(rf_label)),
                AnimationGroup(GrowArrow(if_arrow), GrowFromCenter(if_label)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(ReplacementTransform(if_label, if_equals[0]), Create(if_equals))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        f_sub_ax = Axes(
            x_range=[-0.1, duration + final_beat_time_shift, duration / 4],
            y_range=[-2, 30, 5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).next_to(f_ax, DOWN, LARGE_BUFF * 1.8)
        f_sub_labels = f_sub_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        x0_sub = VT(~beat_time_shift)
        x1_sub = VT(~beat_time_shift)

        f_sub_graph = always_redraw(
            lambda: f_sub_ax.plot(
                lambda t: np.abs(
                    (
                        modulation_index
                        * np.arcsin(np.sin(2 * np.pi * modulation_freq * t))
                    )  # Tx
                    - (
                        modulation_index
                        * np.arcsin(
                            np.sin(2 * np.pi * modulation_freq * (t - ~beat_time_shift))
                        )
                        + ~doppler_f_shift
                    )  # Rx
                ),
                x_range=[~x0_sub, ~x1_sub, 1 / fs],
                use_smoothing=False,
                color=RX_COLOR,
            )
        )

        f_sub_plot_group = VGroup(
            f_sub_ax,
            # f_sub_labels,
        )
        plots_group = VGroup(f_sub_ax, f_ax)

        self.play(
            LaggedStart(
                ShrinkToCenter(mixer),
                Uncreate(lo_arrow),
                Uncreate(rf_arrow),
                Uncreate(if_arrow),
                FadeOut(lo_label, rf_label),
                vel_eqn.animate.next_to(f_ax, RIGHT).shift(UP),
                f_beat_eqn.animate.next_to(f_ax, LEFT).shift(UP),
                if_equals.animate.next_to(f_sub_ax, DOWN, MED_SMALL_BUFF),
                self.camera.frame.animate.scale_to_fit_height(
                    plots_group.height * 1.5
                ).move_to(plots_group),
                f_sub_plot_group.shift(DOWN * 5).animate.shift(UP * 5),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.add(f_sub_graph)

        x0_disp = VT(~x0_sub)
        x1_disp = VT(~x0_sub)

        x0_line_disp = always_redraw(
            lambda: DashedLine(
                [f_ax.c2p(~x0_disp, 0)[0], f_sub_ax.c2p(0, 0)[1], 0],
                [
                    f_ax.c2p(~x0_disp, 0)[0],
                    f_ax.input_to_graph_point(
                        duration / 2 + ~beat_time_shift, tri_f_graph_rx
                    )[1],
                    0,
                ],
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        x1_line_disp = always_redraw(
            lambda: DashedLine(
                [f_ax.c2p(~x1_disp, 0)[0], f_sub_ax.c2p(0, 0)[1], 0],
                [
                    f_ax.c2p(~x1_disp, 0)[0],
                    f_ax.input_to_graph_point(
                        duration / 2 + ~beat_time_shift, tri_f_graph_rx
                    )[1],
                    0,
                ],
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )

        self.play(Create(x0_line_disp), Create(x1_line_disp))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(x1_disp @ (duration / 2), x1_sub @ (duration / 2))

        self.wait(0.5)

        rx_lower = Line(
            f_ax.input_to_graph_point(~x0_disp, tri_f_graph_rx),
            f_ax.input_to_graph_point(~x1_disp, tri_f_graph_rx),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(Create(rx_lower))

        self.wait(0.5)

        tx_upper = Line(
            f_ax.input_to_graph_point(~x0_disp, tri_f_graph),
            f_ax.input_to_graph_point(~x1_disp, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(LaggedStart(Uncreate(rx_lower), Create(tx_upper), lag_ratio=0.3))

        self.wait(0.5)

        self.play(Uncreate(tx_upper))

        self.wait(0.5)

        beat_time_shift_lower = Line(
            f_ax.c2p(0, 0),
            f_ax.c2p(~beat_time_shift, 0),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        beat_time_shift_right = Line(
            f_ax.c2p(~beat_time_shift, 0),
            f_ax.input_to_graph_point(~beat_time_shift, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        beat_time_shift_hypotenuse = Line(
            f_ax.c2p(0, 0),
            f_ax.input_to_graph_point(~beat_time_shift, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        f_beat_label = MathTex(r"f_{beat}").next_to(beat_time_shift_right)

        self.play(Create(beat_time_shift_lower))

        self.wait(0.5)

        self.play(Create(beat_time_shift_hypotenuse))

        self.wait(0.5)

        self.play(Create(beat_time_shift_right), Create(f_beat_label))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        section_dist = (duration / 2 - ~beat_time_shift) / 2
        f_up = MathTex(r"f_{beat}", " - ", "f_d").next_to(
            f_sub_ax.input_to_graph_point(section_dist + ~beat_time_shift, f_sub_graph),
            UP,
        )
        f_down = MathTex(r"f_{beat}", " + ", "f_d").next_to(
            f_sub_ax.input_to_graph_point(
                section_dist + duration / 2 + ~beat_time_shift, f_sub_graph
            ),
            UP,
        )

        self.play(TransformFromCopy(f_beat_label, f_up[0]))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(f_up[1]),
                TransformFromCopy(doppler_shift_label, f_up[2]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(x0_disp @ (duration / 2))
        self.play(
            x1_sub @ (duration / 2 + ~beat_time_shift),
            x1_disp @ (duration / 2 + ~beat_time_shift),
        )

        # self.next_section(skip_animations=skip_animations(False))
        # self.wait(0.5)

        # zero_cross_dot = Dot(
        #     f_ax.input_to_graph_point(~x0_disp + ~beat_time_shift, tri_f_graph),
        #     color=YELLOW,
        # )

        # self.play(Create(zero_cross_dot))

        # self.wait(0.5)

        # self.play(Uncreate(zero_cross_dot))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(x0_disp @ (duration / 2 + ~beat_time_shift))
        self.play(x1_sub @ duration, x1_disp @ duration)

        self.wait(0.5)

        rx_upper = Line(
            f_ax.input_to_graph_point(~x0_disp, tri_f_graph_rx),
            f_ax.input_to_graph_point(~x1_disp, tri_f_graph_rx),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(Create(rx_upper))

        self.wait(0.5)

        tx_lower = Line(
            f_ax.input_to_graph_point(~x0_disp, tri_f_graph),
            f_ax.input_to_graph_point(~x1_disp, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(LaggedStart(Uncreate(rx_upper), Create(tx_lower), lag_ratio=0.3))

        self.wait(0.5)

        self.play(Uncreate(tx_lower))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(f_beat_label, f_down[0]),
                FadeIn(f_down[1]),
                TransformFromCopy(doppler_shift_label, f_down[2]),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(beat_time_shift_hypotenuse),
            Uncreate(beat_time_shift_right),
            Uncreate(beat_time_shift_lower),
            FadeOut(f_beat_label),
            Uncreate(x0_line_disp),
            Uncreate(x1_line_disp),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        f_up_label = MathTex(r"f_{up}").set_color(YELLOW).next_to(f_up, UP)
        f_down_label = MathTex(r"f_{down}").set_color(YELLOW).next_to(f_down, UP)

        self.play(
            LaggedStart(
                GrowFromCenter(f_up_label),
                GrowFromCenter(f_down_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            f_down_label.animate(rate_func=rate_functions.there_and_back).shift(UP / 2)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        up_ramp = Line(
            f_ax.input_to_graph_point(~beat_time_shift, tri_f_graph),
            f_ax.input_to_graph_point(duration / 2, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        down_ramp = Line(
            f_ax.input_to_graph_point(duration / 2 + ~beat_time_shift, tri_f_graph),
            f_ax.input_to_graph_point(duration, tri_f_graph),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(Create(up_ramp))

        self.wait(0.5)

        self.play(Uncreate(up_ramp))
        self.play(Create(down_ramp))

        self.wait(0.5)

        self.play(Uncreate(down_ramp))

        self.wait(0.5)

        self.play(
            f_up_label.animate.set_color(WHITE),
            f_down_label.animate.set_color(WHITE),
        )

        self.wait(0.5)

        # frame_copy = self.camera.frame.copy().scale_to_fit_width(f_sub_ax.width * 1.2)
        self.play(
            self.camera.frame.animate.scale_to_fit_width(f_sub_ax.width * 1.4).move_to(
                plots_group.get_center(), UP
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        unknowns = Tex(r"Unknown: ", r"$f_{beat}$", ", ", r"$f_d$").next_to(
            self.camera.frame.get_bottom(), UP, LARGE_BUFF
        )
        unknowns[0].set_color(RED)
        knowns = Tex(r"Known: $f_{up}$, $f_{down}$").next_to(
            unknowns, UP, MED_LARGE_BUFF, LEFT
        )
        knowns[0][:6].set_color(GREEN)

        self.play(LaggedStart([GrowFromCenter(m) for m in unknowns], lag_ratio=0.2))

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(knowns[0][:6]),
                TransformFromCopy(f_up_label, knowns[0][6:10]),
                GrowFromCenter(knowns[0][10:11]),
                TransformFromCopy(f_down_label, knowns[0][11:]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        f_beat_eqn = MathTex(r"f_{beat} = \frac{f_{up} + f_{down}}{2}").move_to(knowns)
        f_d_eqn = MathTex(r"f_{d} = \frac{f_{up} - f_{down}}{2}").next_to(
            f_beat_eqn, DOWN, MED_LARGE_BUFF, LEFT
        )

        self.play(
            TransformByGlyphMap(
                knowns,
                f_beat_eqn,
                ([0, 1, 2, 3, 4, 5], []),
                (
                    get_transform_func(unknowns[1], ReplacementTransform),
                    [0, 1, 2, 3, 4],
                ),
                ([], [5], {"delay": 0.2}),
                ([9], [], {"delay": 0.2}),
                ([6, 7, 8], [6, 7, 8], {"delay": 0.3}),
                ([], [9], {"delay": 0.4}),
                ([10, 11, 12, 13, 14], [10, 11, 12, 13, 14], {"delay": 0.5}),
                ([], [15, 16], {"delay": 0.6}),
                run_time=2,
            ),
            unknowns[3].animate.move_to(f_d_eqn[0][:3]),
            FadeOut(unknowns[0], unknowns[2]),
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                f_beat_eqn,
                f_d_eqn,
                (get_transform_func(unknowns[3], ReplacementTransform), [0, 1]),
                ([0, 1, 2, 3, 4], ShrinkToCenter),
                ([5], [2]),
                ([6, 7, 8], [3, 4, 5], {"delay": 0.3}),
                ([9], ShrinkToCenter),
                (GrowFromCenter, [6], {"delay": 0.5}),
                ([10, 11, 12, 13, 14], [7, 8, 9, 10, 11], {"delay": 0.5}),
                ([15, 16], [12, 13], {"delay": 0.7}),
                from_copy=True,
                run_time=2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    f_sub_ax,
                    f_sub_graph,
                    if_equals,
                    f_up,
                    f_down,
                    f_up_label,
                    f_down_label,
                ),
                Group(f_d_eqn, f_beat_eqn)
                .animate.arrange(DOWN, LARGE_BUFF)
                .move_to(self.camera.frame.get_center()),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(Group(f_beat_eqn, f_d_eqn).animate.shift(LEFT * 2))

        range_eqn = MathTex(r"R = \frac{c T_{c} f_{beat}}{2 B}").next_to(
            f_beat_eqn, RIGHT, LARGE_BUFF
        )
        doppler_vel_eqn = (
            MathTex(r"v = \frac{f_d \lambda}{2}")
            .next_to(f_d_eqn, RIGHT, LARGE_BUFF)
            .set_x(range_eqn.get_x())
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                doppler_vel_eqn.shift(RIGHT * 8).animate.shift(LEFT * 8),
                range_eqn.shift(RIGHT * 8).animate.shift(LEFT * 8),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class TriangularNotSufficient(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        radar.vgroup.scale(0.7)
        gas_station = (
            SVGMobject("../props/static/Icon 10.svg")
            .set_color(WHITE)
            .scale(1.7)
            .to_edge(RIGHT)
        )
        car = (
            SVGMobject("../props/static/car.svg")
            .next_to(gas_station, LEFT)
            .shift(DOWN)
            .set_fill(WHITE)
            .scale(0.8)
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(GrowFromCenter(radar.vgroup))

        self.wait(0.5)

        self.play(
            radar.vgroup.animate.to_corner(DL, MED_LARGE_BUFF),
            GrowFromCenter(gas_station),
            GrowFromCenter(car),
        )

        self.wait(0.5)

        beam_pencil = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            car.get_left() + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        beam_u = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            gas_station.get_corner(UL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        beam_l = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            car.get_corner(DL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )

        self.play(Create(beam_pencil))

        self.wait(0.5)

        self.play(
            TransformFromCopy(beam_pencil, beam_l),
            ReplacementTransform(beam_pencil, beam_u),
        )

        self.wait(2)


class RangeDopplerIntro(MovingCameraScene):
    def construct(self):
        stop_time = 16
        fs = 1000

        f1 = 1.5
        f2 = 6

        power_norm_1 = VT(-3)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(3)

        f_max = 8
        y_min = VT(-30)

        x_len = 9.5
        y_len = 4.5

        noise_seed = VT(2)

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, LARGE_BUFF)
        ax_label = ax.get_axis_labels(Tex("$R$"), Tex())

        freq, X_k_log = get_plot_values(
            frequencies=[f1, f2],
            power_norms=[~power_norm_1, ~power_norm_3],
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
        ).values()

        f_X_k = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(f_X_k, color=IF_COLOR, x_range=[0, f_max, 1 / fs])

        peak_1 = ax.input_to_graph_point(f1, return_plot)
        peak_2 = ax.input_to_graph_point(f2, return_plot)

        targets = Tex("Targets").to_edge(UP)

        peak_1_bez = CubicBezier(
            targets.get_bottom() + [0, -0.1, 0],
            targets.get_bottom() + [0, -0.1, 0] + [0, -1, 0],
            peak_1 + [0, 1, 0],
            peak_1 + [0, 0.1, 0],
        )
        peak_2_bez = CubicBezier(
            targets.get_bottom() + [0, -0.1, 0],
            targets.get_bottom() + [0, -0.1, 0] + [0, -1, 0],
            peak_2 + [0, 1, 0],
            peak_2 + [0, 0.1, 0],
        )

        plot_group = VGroup(ax, ax_label, return_plot)

        self.next_section(skip_animations=skip_animations(True))
        # self.add(ax, ax_label, return_plot)

        self.play(plot_group.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(targets, shift=DOWN),
                AnimationGroup(Create(peak_1_bez), Create(peak_2_bez)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(Uncreate(peak_1_bez), Uncreate(peak_2_bez)),
                FadeOut(targets, shift=UP),
                lag_ratio=0.4,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        v1 = MathTex(r"v_1").next_to(peak_1, UP)
        v2 = MathTex(r"v_2").next_to(peak_2, UP)

        self.play(
            LaggedStart(
                FadeIn(v1, shift=DOWN),
                FadeIn(v2, shift=DOWN),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        range_resolution = 0.15
        f2_left_res = DashedLine(
            ax.c2p(f2 - range_resolution, 0),
            [
                ax.c2p(f2 - range_resolution, 0)[0],
                ax.input_to_graph_point(f2, return_plot)[1],
                0,
            ],
            dash_length=DEFAULT_DASH_LENGTH * 3,
        )
        f2_right_res = DashedLine(
            ax.c2p(f2 + range_resolution, 0),
            [
                ax.c2p(f2 + range_resolution, 0)[0],
                ax.input_to_graph_point(f2, return_plot)[1],
                0,
            ],
            dash_length=DEFAULT_DASH_LENGTH * 3,
        )

        multiple_targets = Tex("Multiple targets?").next_to(
            ax.input_to_graph_point(f2, return_plot), UP, LARGE_BUFF * 1.4
        )

        multiple_targets_left_bez = CubicBezier(
            multiple_targets.get_corner(DL) + [0, -0.1, 0],
            multiple_targets.get_corner(DL) + [0, -0.1, 0] + [0, -1, 0],
            f2_left_res.get_top() + [0, 1, 0],
            f2_left_res.get_top() + [0, 0.1, 0],
        )
        multiple_targets_right_bez = CubicBezier(
            multiple_targets.get_corner(DR) + [0, -0.1, 0],
            multiple_targets.get_corner(DR) + [0, -0.1, 0] + [0, -1, 0],
            f2_right_res.get_top() + [0, 1, 0],
            f2_right_res.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(f2_left_res), Create(f2_right_res)),
                AnimationGroup(
                    Create(multiple_targets_left_bez),
                    Create(multiple_targets_right_bez),
                ),
                FadeIn(multiple_targets, shift=DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(f2_left_res),
            Uncreate(f2_right_res),
            Uncreate(multiple_targets_left_bez),
            Uncreate(multiple_targets_right_bez),
            FadeOut(multiple_targets, shift=UP),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        v_spectrum = MathTex(r"v = [v_{(n,1)}, v_{(n,2)}, \ldots , v_{(n,M)}]").to_edge(
            UP, LARGE_BUFF
        )

        self.play(
            FadeOut(v1),
            TransformByGlyphMap(
                v2,
                v_spectrum,
                ([0], [0]),
                ([1], ShrinkToCenter),
                ([], [1], {"delay": 0.2}),
                ([], [2, 27], {"delay": 0.3}),
                ([], [3, 4, 5, 6, 7, 8, 9], {"delay": 0.4}),
                ([], [10, 11, 12, 13, 14, 15, 16], {"delay": 0.5}),
                ([], [17, 18, 19, 20], {"delay": 0.6}),
                ([], [21, 22, 23, 24, 25, 26], {"delay": 0.7}),
            ),
        )

        self.wait(0.5)

        num_samples = 28
        samples = ax.get_vertical_lines_to_graph(
            return_plot, x_range=[0, f_max], num_lines=num_samples, color=BLUE
        )

        sample_rects = ax.get_riemann_rectangles(
            return_plot,
            input_sample_type="right",
            x_range=[0, f_max],
            dx=f_max / num_samples,
            color=BLUE,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)

        self.play(Create(samples), run_time=1)
        self.play(
            *[
                ReplacementTransform(sample, rect)
                for sample, rect in zip(samples, sample_rects)
            ]
        )

        self.wait(0.5)

        self.play(FadeOut(plot_group.set_z_index(-1)))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    Transform(
                        rect,
                        Square(
                            rect.width,
                            color=BLACK,
                            fill_color=BLUE,
                            fill_opacity=0.7,
                            stroke_width=DEFAULT_STROKE_WIDTH / 2,
                        )
                        .move_to(rect)
                        .set_y(0),
                    )
                    for rect in sample_rects
                ],
                lag_ratio=0.02,
            )
        )

        self.wait(0.5)

        v_spectrum_vert = MathTex(
            r"v &=\\ [ &v_{(n,1)},\\ &v_{(n,2)},\\ &\ldots ,\\ &v_{(n,M)}]"
        ).to_edge(LEFT, MED_LARGE_BUFF)

        range_spectrum = MathTex(r"R = \left[ R_1, R_2, \ldots , R_N \right]").to_edge(
            DOWN, MED_LARGE_BUFF
        )

        M = 9
        self.play(
            LaggedStart(
                TransformByGlyphMap(
                    v_spectrum,
                    v_spectrum_vert,
                    ([0], [0]),
                    ([1], [1]),
                    (
                        [2, 3, 4, 5, 6, 7, 8, 9],
                        [2, 3, 4, 5, 6, 7, 8, 9],
                        {"delay": 0.1},
                    ),
                    (
                        [10, 11, 12, 13, 14, 15, 16],
                        [10, 11, 12, 13, 14, 15, 16],
                        {"delay": 0.2},
                    ),
                    ([17, 18, 19, 20], [17, 18, 19, 20], {"delay": 0.3}),
                    (
                        [21, 22, 23, 24, 25, 26, 27],
                        [21, 22, 23, 24, 25, 26, 27],
                        {"delay": 0.4},
                    ),
                ),
                LaggedStart(
                    *[
                        AnimationGroup(
                            TransformFromCopy(
                                sample_rects,
                                sample_rects.copy().shift(
                                    UP * sample_rects.height * idx
                                ),
                            ),
                            TransformFromCopy(
                                sample_rects,
                                sample_rects.copy().shift(
                                    DOWN * sample_rects.height * idx
                                ),
                            ),
                        )
                        for idx in range(1, M)
                    ],
                    lag_ratio=0.02,
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(range_spectrum, shift=UP))

        self.wait(0.5)

        range_doppler_label = Tex(
            "Range-Doppler Spectrum", font_size=DEFAULT_FONT_SIZE * 1.8
        ).to_edge(UP, 0)

        self.play(self.camera.frame.animate.scale(1.2), Create(range_doppler_label))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class HowToLearn(Scene):
    def construct(self):
        phaser = ImageMobject(
            "../props/static/phaser_hardware_2.png"
        ).scale_to_fit_width(config["frame_width"] * 0.3)

        textbook = (
            Tex(r"""
Here we have the coherent processing interval, $T_{CPI}$, which is comprised of $M$ pulse repetition intervals, $T_{PRI}$. This gives us our velocity resolution, 

\begin{equation}
    \Delta f = \frac{1}{M T_{PRI}}
\end{equation}

From this, we can define the pulse train LFM as

\begin{equation}
    s(t) = \sum_{m=0}^{M-1} s_p(t - m T_{PRI})
\end{equation}

where $s_t$ is the single LFM pulse waveform with duration, $T_c$,

\begin{equation}
    s_p = \begin{cases} 
      \exp{j \pi \frac{B}{2 T_c} t^2} & 0 \le t \le T_c \\
      0 & \text{otherwise}
   \end{cases}
\end{equation}

For a target with velocity, $v$, and range, $r$, the equation for the beat signal becomes

\begin{equation}
    b(t) = a \exp{\left[ j 2 \pi \frac{2 B r}{T_c c} (t - m T_{PRI}) \right]} \exp{\left[ j 2 \pi \frac{2 f_c v}{c} t \right]}
\end{equation}

for $(m-1) T_{PRI} \le t \le m T_{PRI}$ and $0 \le m \le M$.

Then if you sample the $M$ pulse repitions with a sampling rate of $T_s$, you get the 2D sampled beat signal,

\begin{equation}
    b[l, m] = a \exp{\left[ j 2 \pi \left( \frac{2 f_c v}{c} + \frac{2 B r}{T_c c} \right) l T_s \right]} \exp{\left[ j 2 \pi \frac{2 f_c v}{c} m T_{PRI} \right]}
\end{equation}

Then taking the FFT on $b[l, m]$, you get

\begin{equation}
    B[p, k] = \frac{1}{\sqrt{N_z M}} \sum_{l=0}^{N_z-1} \sum_{m=0}^{M-1} b[l, m] \exp{\left[ -j 2 \pi \left( \frac{lp}{N_z} + \frac{mk}{M} \right) \right]}
\end{equation}
            """)
            .scale_to_fit_width(config["frame_width"] * 0.8)
            .next_to([0, -config["frame_height"] / 2, 0], DOWN)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            phaser.next_to([0, -config["frame_height"] / 2, 0], DOWN).animate.move_to(
                ORIGIN
            )
        )

        self.wait(0.5)

        self.play(phaser.animate.next_to([0, config["frame_height"] / 2, 0], UP))

        self.wait(0.5)

        self.play(
            textbook.animate.next_to([0, config["frame_height"] / 2, 0], UP),
            run_time=4,
            rate_func=rate_functions.linear,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        speed = VT(1)
        speed_label = always_redraw(
            lambda: Tex(
                f"Speed: {~speed:.2f}x",
                font_size=DEFAULT_FONT_SIZE,
            )
        )

        self.play(Create(speed_label))

        self.wait(0.5)

        example_beat_signal = ImageMobject(
            "../props/static/beat_signal_example.png"
        ).to_corner(UL)
        example_range_doppler = ImageMobject(
            "../props/static/range_doppler_example.png"
        ).to_corner(DR)

        self.play(
            LaggedStart(
                speed @ 0.2,
                FadeIn(example_range_doppler, shift=UP),
                FadeIn(example_beat_signal, shift=DOWN),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            FadeOut(speed_label),
            FadeOut(example_range_doppler, shift=DOWN),
            FadeOut(example_beat_signal, shift=UP),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"fmcw\_range\_doppler.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        notebook_sc = (
            ImageMobject("./static/notebook_sc.png")
            .scale_to_fit_height(config.frame_height * 0.65)
            .to_edge(UP)
        )
        notebook_sc_box = SurroundingRectangle(
            notebook_sc, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook_sc_group = Group(notebook_sc_box, notebook_sc)

        notebook_sc_2 = (
            ImageMobject("./static/notebook_sc_2.png")
            .scale_to_fit_height(config.frame_height * 0.65)
            .to_edge(UP)
        )
        notebook_sc_box_2 = SurroundingRectangle(
            notebook_sc_2, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook_sc_group_2 = Group(notebook_sc_box_2, notebook_sc_2)

        notebook_sc_3 = (
            ImageMobject("./static/notebook_sc_3.png")
            .scale_to_fit_height(config.frame_height * 0.65)
            .to_edge(UP)
        )
        notebook_sc_box_3 = SurroundingRectangle(
            notebook_sc_3, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook_sc_group_3 = Group(notebook_sc_box_3, notebook_sc_3)

        self.play(
            notebook.shift(DOWN * 5).animate.shift(UP * 5),
            notebook_sc_group.shift(UP * 10).animate.shift(DOWN * 10),
        )

        self.wait(0.5)

        self.play(
            notebook_sc_group.animate.shift(LEFT * 18),
            notebook_sc_group_2.shift(RIGHT * 18).animate.shift(LEFT * 18),
        )

        self.wait(0.5)

        self.play(
            notebook_sc_group_2.animate.shift(LEFT * 18),
            notebook_sc_group_3.shift(RIGHT * 18).animate.shift(LEFT * 18),
        )

        self.wait(0.5)

        self.play(
            notebook.animate.shift(DOWN * 5),
            notebook_sc_group_3.animate.shift(UP * 10),
        )
        self.remove(
            notebook, notebook_sc_group, notebook_sc_group_2, notebook_sc_group_3
        )

        self.wait(2)


class Phase(MovingCameraScene):
    def construct(self):
        mixer = BLOCKS.get("mixer").copy()

        fs = 1000
        step = 1 / fs
        x_range = [0, 1, step]
        x_range_lo = [0, 0.7, step]
        x_len = 4
        y_len = 2
        tx_ax = (
            Axes(
                x_range=x_range_lo[:2], y_range=[-2, 2], x_length=x_len, y_length=y_len
            )
            .rotate(-PI / 2)
            .next_to(mixer, direction=UP, buff=0)
        )
        rx_ax = Axes(
            x_range=x_range[:2], y_range=[-2, 2], x_length=x_len, y_length=y_len
        ).next_to(mixer, direction=LEFT, buff=0)
        if_ax = Axes(
            x_range=x_range[:2],
            y_range=[-2, 2],
            x_length=x_len,
            y_length=y_len,
            tips=False,
            axis_config={"include_numbers": False},
        ).next_to(mixer, direction=RIGHT, buff=0)

        A = 1
        f_tx = 12
        f_rx = 10
        tx_signal = tx_ax.plot(
            lambda t: A * np.sin(2 * PI * f_tx * t), x_range=x_range_lo, color=TX_COLOR
        )
        rx_signal = rx_ax.plot(
            lambda t: A * np.sin(2 * PI * f_rx * t), x_range=x_range, color=RX_COLOR
        )
        if_signal = if_ax.plot(
            lambda t: A * np.sin(2 * PI * f_tx * t) * A * np.sin(2 * PI * f_rx * t),
            x_range=x_range,
            color=IF_COLOR,
        )

        lo_port = (
            Tex("LO")
            .scale(0.6)
            .next_to(mixer.get_top(), direction=DOWN, buff=SMALL_BUFF)
        )
        rf_port = (
            Tex("RF")
            .scale(0.6)
            .next_to(mixer.get_left(), direction=RIGHT, buff=SMALL_BUFF)
        )
        if_port = (
            Tex("IF")
            .scale(0.6)
            .next_to(mixer.get_right(), direction=LEFT, buff=SMALL_BUFF)
        )

        tx_eqn = MathTex(r"\sin{\left( 2 \pi f_{TX}(t) t \right)}").next_to(
            tx_signal, LEFT
        )
        tx_eqn[0][6:12].set_color(TX_COLOR)
        rx_eqn = MathTex(r"\sin{\left( 2 \pi f_{TX}(t - t_{shift}) t \right)}").next_to(
            rx_signal, DOWN, LARGE_BUFF
        )
        rx_eqn[0][6:19].set_color(RX_COLOR)

        # self.add(mixer, tx_signal, rx_signal, if_signal, lo_port, rf_port, if_port)

        self.next_section(skip_animations=skip_animations(True))
        self.play(GrowFromCenter(Group(mixer, lo_port, rf_port, if_port)))

        self.wait(0.5)

        self.play(Create(tx_signal), FadeIn(tx_eqn, shift=DOWN))

        self.wait(0.5)

        self.play(Create(rx_signal), FadeIn(rx_eqn, shift=RIGHT))

        self.wait(0.5)

        self.play(Create(if_signal))

        self.wait(0.5)

        lp_filter = BLOCKS.get("lp_filter").copy().next_to(if_signal, RIGHT, 0)

        if_filt_ax = (
            Axes(
                x_range=x_range[:2],
                y_range=[-2, 2],
                x_length=x_len,
                y_length=y_len,
                tips=False,
                axis_config={"include_numbers": False},
            )
            .set_opacity(0)
            .next_to(lp_filter, direction=RIGHT, buff=0)
        )
        phase = VT(0)
        if_signal_filt = always_redraw(
            lambda: if_filt_ax.plot(
                lambda t: A * np.cos(2 * PI * (f_tx - f_rx) * t + ~phase),
                x_range=x_range,
                color=IF_COLOR,
            )
        )
        if_filt_plot_group = VGroup(if_filt_ax, if_signal_filt)

        if_eqn = (
            MathTex(
                r"\sin{\left( 2 \pi (f_{TX}(t) - f_{TX}(t - t_{shift}))  t \right)}",
            )
            .next_to(Group(lp_filter, if_signal_filt), UP, LARGE_BUFF)
            .shift(LEFT)
        )
        if_eqn[0][7:13].set_color(TX_COLOR)
        if_eqn[0][14:27].set_color(RX_COLOR)

        beat_eqn = MathTex(
            r"\sin{\left( 2 \pi f_{beat} t \right)}",
        ).move_to(if_eqn)
        beat_eqn[0][6:11].set_color(IF_COLOR)

        self.play(GrowFromCenter(lp_filter))

        self.wait(0.5)

        sig_group = Group(rx_eqn, if_signal_filt)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(sig_group.width * 1.1)
            .move_to(sig_group)
            .shift(UP * 2),
            Create(if_signal_filt),
        )

        self.wait(0.5)

        self.play(FadeIn(if_eqn, shift=LEFT))

        self.wait(0.5)

        # fmt: off
        self.play(
            TransformByGlyphMap(
                if_eqn,
                beat_eqn,
                ([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]),
                ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [6, 7, 8, 9, 10]),
                ([28, 29], [11, 12]),
            )
        )
        # fmt: on

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    mixer,
                    lo_port,
                    rf_port,
                    if_port,
                    tx_signal,
                    rx_signal,
                    if_signal,
                    tx_eqn,
                    rx_eqn,
                    lp_filter,
                ),
                AnimationGroup(
                    self.camera.frame.animate.restore(),
                    VGroup(beat_eqn, if_filt_plot_group)
                    .animate.arrange(DOWN, LARGE_BUFF)
                    .move_to(ORIGIN),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        phase_label = always_redraw(
            lambda: Tex(f"$\\phi =\\ ${~phase:.2f} radians").to_corner(DL, LARGE_BUFF)
        )
        beat_eqn_w_phase = MathTex(
            r"\sin{\left( 2 \pi f_{beat} t + \phi \right)}",
        ).move_to(beat_eqn)
        beat_eqn_w_phase[0][6:11].set_color(IF_COLOR)

        self.play(
            TransformByGlyphMap(
                beat_eqn,
                beat_eqn_w_phase,
                (
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ),
                ([12], [14]),
                ([], [12], {"delay": 0.2}),
                ([], [13], {"delay": 0.4}),
            ),
        )
        self.play(FadeIn(phase_label, shift=UP * 2))

        self.wait(0.5)

        self.play(phase @ PI)

        self.wait(0.5)

        self.play(phase @ 0)

        self.wait(0.5)

        self.play(
            Group(beat_eqn_w_phase, if_filt_plot_group)
            .animate.arrange(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP / 2),
            FadeOut(phase_label, shift=DOWN * 2),
        )

        self.wait(0.5)

        phase_delta = VT(0.4)  # radians
        phase_delta_label = always_redraw(
            lambda: Tex(f"$\\Delta \\phi =\\ ${~phase_delta:.2f} radians").to_corner(
                DL, MED_SMALL_BUFF
            )
        )

        def create_phase_shifted_beat(n):
            return always_redraw(
                lambda: if_filt_ax.plot(
                    lambda t: A * np.cos(2 * PI * (f_tx - f_rx) * t + n * ~phase_delta),
                    x_range=x_range,
                    color=IF_COLOR,
                ).shift(if_signal_filt.height * 1.2 * n * UP)
            )

        def create_phase_shifted_beat_eqn(n):
            n_fmt = "+" if n == 1 else "-" if n == -1 else f"+ {n}"
            eqn = (
                MathTex(
                    f"\\sin{{\\left( 2 \\pi f_{{beat}} t + \\phi_0 {n_fmt} \\Delta \\phi \\right)}}",
                )
                .move_to(beat_eqn_w_phase)
                .shift(if_signal_filt.height * 1.2 * n * UP)
            )
            eqn[0][6:11].set_color(IF_COLOR)
            return eqn

        n_values = [-2, 2, -1, 1]
        shifted_beats = VGroup(*[create_phase_shifted_beat(n) for n in n_values])
        shifted_beat_eqns = VGroup(
            *[create_phase_shifted_beat_eqn(n) for n in n_values]
        )

        beat_eqn_w_phase_0 = MathTex(
            r"\sin{\left( 2 \pi f_{beat} t + \phi_0 \right)}",
        ).move_to(beat_eqn_w_phase)
        beat_eqn_w_phase_0[0][6:11].set_color(IF_COLOR)

        self.play(
            LaggedStart(
                TransformByGlyphMap(
                    beat_eqn_w_phase,
                    beat_eqn_w_phase_0,
                    (
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    ),
                    ([14], [15]),
                    ([], [14], {"delay": 0.2}),
                ),
                *[
                    AnimationGroup(
                        TransformFromCopy(if_signal_filt, sig),
                        TransformFromCopy(beat_eqn_w_phase, eqn),
                    )
                    for sig, eqn in zip(shifted_beats, shifted_beat_eqns)
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(phase_delta_label, shift=UP * 2))

        self.wait(0.5)

        self.play(phase_delta @ PI, run_time=2)

        self.wait(0.5)

        self.play(phase_delta @ 0, run_time=2)

        self.wait(0.5)

        self.play(phase_delta @ 0.6, run_time=2)

        self.wait(0.5)

        self.play(
            *[Uncreate(sig) for sig in [*shifted_beats, if_signal_filt]],
            FadeOut(*shifted_beat_eqns[:-1], phase_delta_label, beat_eqn_w_phase_0),
            shifted_beat_eqns[-1].animate.move_to(ORIGIN),
        )

        self.wait(0.5)

        self.play(shifted_beat_eqns[-1][0][-3:-1].animate.set_color(YELLOW))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class StationaryTarget(Scene):
    def construct(self): ...


class RealSystem(Scene):
    def construct(self):
        ic = ImageMobject("../props/static/microcontroller.png").scale_to_fit_width(
            config.frame_height * 0.4
        )

        self.play(ic.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        ic_label = Tex("AWR1642").next_to(ic, UL, LARGE_BUFF).shift(LEFT)
        ic_label_l1 = Line(
            ic_label.get_corner(DR) + [1, -0.1, 0],
            ic_label.get_corner(DL) + [-0.1, -0.1, 0],
        )
        ic_label_l2 = Line(
            ic.get_center() + [-0.3, 0.5, 0],
            ic_label.get_corner(DR) + [1, -0.1, 0],
        )
        ic_label_dot = Dot(ic.get_center() + [-0.3, 0.5, 0])

        self.play(
            LaggedStart(
                Create(ic_label_dot),
                Create(ic_label_l2),
                AnimationGroup(Create(ic_label_l1), FadeIn(ic_label)),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        f_label = (
            Tex(r"$76 \le f_c \le 81$ GHz").next_to(ic, DR, LARGE_BUFF).shift(LEFT)
        )
        f_label_eq = Tex(r"$f_c = 77$ GHz").move_to(f_label)
        f_label_l1 = Line(
            f_label.get_corner(DL) + [-0.3, -0.1, 0],
            f_label.get_corner(DR) + [0.1, -0.1, 0],
        )
        f_label_dot = Dot(ic.get_center() + [0.3, -0.5, 0])
        f_label_l2 = Line(
            f_label_dot.get_center(),
            f_label_l1.get_start(),
        )

        self.play(
            LaggedStart(
                Create(f_label_dot),
                Create(f_label_l2),
                AnimationGroup(Create(f_label_l1), FadeIn(f_label)),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                f_label,
                f_label_eq,
                ([0, 1, 2], []),
                ([3, 4], [0, 1], {"delay": 0.2}),
                ([5], [2], {"delay": 0.2}),
                ([6, 7], [3, 4], {"delay": 0.2}),
            )
        )

        self.wait(0.5)

        bw_label = Tex(r"BW = 1.6 GHz").next_to(ic, UR, LARGE_BUFF).shift(LEFT)
        bw_label_l1 = Line(
            bw_label.get_corner(DL) + [-0.3, -0.1, 0],
            bw_label.get_corner(DR) + [0.1, -0.1, 0],
        )
        bw_label_dot = Dot(ic.get_center() + [0.3, 0.5, 0])
        bw_label_l2 = Line(
            bw_label_dot.get_center(),
            bw_label_l1.get_start(),
        )

        chirp_time_label = (
            Tex(r"$T_c = 40 \mu$s").next_to(ic, DL, LARGE_BUFF).shift(LEFT)
        )
        chirp_time_label_l1 = Line(
            chirp_time_label.get_corner(DR) + [0.3, -0.1, 0],
            chirp_time_label.get_corner(DL) + [-0.1, -0.1, 0],
        )
        chirp_time_label_dot = Dot(ic.get_center() + [-0.3, -0.3, 0])
        chirp_time_label_l2 = Line(
            chirp_time_label_dot.get_center(),
            chirp_time_label_l1.get_start(),
        )

        self.play(
            LaggedStart(
                Create(bw_label_dot),
                Create(chirp_time_label_dot),
                Create(bw_label_l2),
                Create(chirp_time_label_l2),
                AnimationGroup(Create(bw_label_l1), FadeIn(bw_label)),
                AnimationGroup(Create(chirp_time_label_l1), FadeIn(chirp_time_label)),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"fmcw\_range\_doppler.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        self.play(notebook.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(notebook.animate.shift(DOWN * 5))
        self.remove(notebook)

        self.wait(0.5)

        self.play(
            FadeOut(bw_label, ic.set_z_index(-5)),
            Uncreate(ic_label_dot),
            Uncreate(ic_label_l1),
            Uncreate(ic_label_l2),
            FadeOut(ic_label),
            Uncreate(f_label_dot),
            Uncreate(f_label_l1),
            Uncreate(f_label_l2),
            FadeOut(f_label_eq),
            Uncreate(chirp_time_label_dot),
            Uncreate(chirp_time_label_l1),
            Uncreate(chirp_time_label_l2),
            FadeOut(chirp_time_label),
            Uncreate(bw_label_dot),
            Uncreate(bw_label_l1),
            Uncreate(bw_label_l2),
        )

        self.wait(2)


class CarVelocity(Scene):
    def construct(self):
        car1 = (
            SVGMobject("../props/static/car.svg", fill_color=WHITE, stroke_color=WHITE)
            .scale_to_fit_width(config.frame_width * 0.2)
            .to_edge(LEFT, LARGE_BUFF)
            .shift(UP * 2)
        )
        car2 = (
            SVGMobject(
                "../props/static/person.svg", fill_color=WHITE, stroke_color=WHITE
            )
            # .flip()
            # .scale_to_fit_width(config.frame_width * 0.2)
            .scale_to_fit_height(car1.height)
            .to_edge(RIGHT, LARGE_BUFF)
            .set_y(car1.get_y())
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(car1.shift(LEFT * 5).animate.shift(RIGHT * 5))

        self.wait(0.5)

        self.play(car2.shift(RIGHT * 5).animate.shift(LEFT * 5))

        self.wait(0.5)

        car_vel_arrow = Arrow(car1.get_left(), car1.get_right()).next_to(car1, DOWN)
        car_vel_label = MathTex(
            r"v &= \text{10 m/s} \\ &\approx \text{22 mph}"
        ).next_to(car_vel_arrow, DOWN)

        self.play(GrowArrow(car_vel_arrow), FadeIn(car_vel_label[0][:7]))

        self.wait(0.5)

        self.play(FadeIn(car_vel_label[0][7:]))

        self.wait(0.5)

        scan_top = Line(
            car1.get_right() + [0.1, 0, 0],
            car2.get_corner(UL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        scan_bot = Line(
            car1.get_right() + [0.1, 0, 0],
            car2.get_corner(DL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        car_return = Arrow(car2.get_left(), car1.get_right(), color=RX_COLOR)

        self.play(
            LaggedStart(
                AnimationGroup(Create(scan_top), Create(scan_bot)),
                GrowArrow(car_return),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        f_beat_1 = MathTex(r"f_{beat}(t_0)")
        f_beat_1_copy = MathTex(r"f_{beat}(t_0)")
        f_beat_group = (
            Group(f_beat_1, f_beat_1_copy)
            .arrange(RIGHT, LARGE_BUFF * 1.5)
            .to_edge(DOWN, LARGE_BUFF)
        )
        f_beat_2 = MathTex(r"f_{beat}(t_0 + 40 \mu \text{s})").move_to(f_beat_1_copy)

        r_1 = MathTex(r"R(t_0)").next_to(f_beat_1, UP, MED_LARGE_BUFF)
        r_1_copy = MathTex(r"R(t_0)").next_to(f_beat_2, UP, MED_LARGE_BUFF)
        r_group = (
            Group(r_1, r_1_copy)
            .arrange(RIGHT, LARGE_BUFF * 1.5)
            .to_edge(DOWN, LARGE_BUFF * 1.5)
        )
        r_2 = MathTex(r"R(t_0 + 40 \mu \text{s})").move_to(r_1_copy)

        f_beat_diff = MathTex(
            r"f_{beat}(t_0) - f_{beat}(t_0 + 40 \mu \text{s}) \approx \text{286 Hz}"
        ).move_to(f_beat_group)
        r_diff = MathTex(
            r"R(t_0) - R(t_0 + 40 \mu \text{s}) \approx \text{0.0004 m}"
        ).move_to(r_group)

        self.play(
            # f_beat_1.shift(DOWN * 5).animate.shift(UP * 5),
            r_1.shift(DOWN * 5).animate.shift(UP * 5),
        )

        self.wait(0.5)

        self.play(
            # f_beat_1_copy.shift(DOWN * 5).animate.shift(UP * 5),
            r_1_copy.shift(DOWN * 5).animate.shift(UP * 5),
        )

        self.wait(0.5)

        self.play(
            # TransformByGlyphMap(
            #     f_beat_1_copy,
            #     f_beat_2,
            #     ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
            #     ([], [8, 9, 10, 11, 12], {"delay": 0.3}),
            #     ([8], [13]),
            # ),
            TransformByGlyphMap(
                r_1_copy,
                r_2,
                ([0, 1, 2, 3], [0, 1, 2, 3]),
                ([], [4, 5, 6, 7, 8], {"delay": 0.3}),
                ([4], [9]),
            ),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                r_1,
                r_diff,
                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                (GrowFromCenter, [5], {"delay": 0.2}),
                (
                    get_transform_func(r_2[0], ReplacementTransform),
                    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    {"delay": 0.4},
                ),
                (GrowFromCenter, [16], {"delay": 0.4}),
                (GrowFromCenter, [17, 18, 19, 20, 21, 22, 23], {"delay": 0.6}),
            )
        )

        # self.wait(0.5)

        # self.play(
        #     TransformByGlyphMap(
        #         f_beat_1,
        #         f_beat_diff,
        #         ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        #         (GrowFromCenter, [9], {"delay": 0.2}),
        #         (
        #             get_transform_func(f_beat_2[0], ReplacementTransform),
        #             [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #             {"delay": 0.4},
        #         ),
        #         (GrowFromCenter, [24], {"delay": 0.4}),
        #         (GrowFromCenter, [25, 26, 27, 28, 29], {"delay": 0.6}),
        #     )
        # )

        self.wait(0.5)

        bw = 1.6e9
        range_resolution_label = Tex(
            f"$\\Delta R = \\frac{{c}}{{2 BW}} =$ {c / (2*bw):.3f} m"
        )
        range_group = (
            Group(r_diff.copy(), range_resolution_label)
            .arrange(RIGHT, LARGE_BUFF)
            .set_y(r_diff.get_y())
        )

        self.play(
            LaggedStart(
                r_diff.animate.move_to(range_group[0]),
                range_resolution_label.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.3,
            )
        )

        # bw_label_hz = Tex(r"BW = $1.6 \cdot 10^{9}$ Hz").next_to(
        #     r_diff, UP, MED_LARGE_BUFF
        # )
        # bw_label = Tex(r"BW = 1.6 GHz").move_to(bw_label_hz)

        # self.play(FadeIn(bw_label_hz))

        # self.wait(0.5)

        # self.play(
        #     TransformByGlyphMap(
        #         bw_label_hz,
        #         bw_label,
        #         ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
        #         ([6, 7, 8, 9], ShrinkToCenter),
        #         ([10, 11], [7, 8]),
        #         (FadeIn, [6]),
        #     )
        # )

        self.wait(0.5)

        beat_eqn_w_phase = MathTex(
            r"\sin{\left( 2 \pi f_{beat} t + \phi \right)}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        beat_eqn_w_phase[0][6:11].set_color(IF_COLOR)

        objs = self.mobjects.copy()
        shuffle(objs)
        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        FadeOut(m) if isinstance(m, Arrow) else ShrinkToCenter(m)
                        for m in objs
                    ],
                    lag_ratio=0.02,
                ),
                GrowFromCenter(beat_eqn_w_phase),
                lag_ratio=0.5,
            )
        )

        self.wait(2)


class PhaseEquation(Scene):
    def construct(self):
        beat_eqn_w_phase = MathTex(
            r"\sin{\left( 2 \pi f_{beat} t + \phi \right)}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        beat_eqn_w_phase[0][6:11].set_color(IF_COLOR)

        self.add(beat_eqn_w_phase)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_eqn = MathTex(
            r"\phi_1 = -(2 \pi \mu \Delta \tau ) t + \pi \mu \Delta \tau^2 + 2 \pi \mu \tau_0 \Delta \tau - 2 \pi f \Delta \tau + \phi_0"
        )
        phase_eqn_minus = MathTex(
            r"\phi_1 - \phi_0 = -(2 \pi \mu \Delta \tau ) t + \pi \mu \Delta \tau^2 + 2 \pi \mu \tau_0 \Delta \tau - 2 \pi f \Delta \tau"
        )
        phase_eqn_delta = MathTex(
            r"\Delta \phi = -(2 \pi \mu \Delta \tau ) t + \pi \mu \Delta \tau^2 + 2 \pi \mu \tau_0 \Delta \tau - 2 \pi f \Delta \tau"
        )

        self.play(
            TransformByGlyphMap(
                beat_eqn_w_phase,
                phase_eqn,
                ([13], [0], {"path_arc": PI / 3}),
                ([0, 1, 2], ShrinkToCenter),
                ([3, 4, 5], ShrinkToCenter),
                ([6, 7, 8, 9, 10], ShrinkToCenter),
                ([11, 12, 14], ShrinkToCenter),
                (GrowFromCenter, [1], {"delay": 0.2}),
                (GrowFromCenter, [2], {"delay": 0.4}),
                (GrowFromCenter, [3, 4, 5, 6, 7, 8, 9, 10, 11], {"delay": 0.6}),
                (GrowFromCenter, [12], {"delay": 0.8}),
                (GrowFromCenter, [13, 14, 15, 16, 17], {"delay": 1}),
                (GrowFromCenter, [18], {"delay": 1.2}),
                (GrowFromCenter, [19, 20, 21, 22, 23, 24, 25], {"delay": 1.4}),
                (GrowFromCenter, [26], {"delay": 1.6}),
                (GrowFromCenter, [27, 28, 29, 30, 31], {"delay": 1.8}),
                (GrowFromCenter, [32], {"delay": 2}),
                (GrowFromCenter, [33, 34], {"delay": 2.2}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                phase_eqn,
                phase_eqn_minus,
                ([0, 1], [0, 1]),
                ([2], [5]),
                ([32], [2], {"path_arc": PI / 3}),
                ([33, 34], [3, 4], {"path_arc": PI / 3, "delay": 0.2}),
                ([slice(3, 32)], [slice(6, 35)]),
            )
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                phase_eqn_minus,
                phase_eqn_delta,
                ([0, 1, 2], ShrinkToCenter),
                ([4], ShrinkToCenter),
                ([3], [1], {"delay": 0.2}),
                (GrowFromCenter, [0], {"delay": 0.2}),
                ([slice(5, 35)], [slice(2, 32)]),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_comp_1 = phase_eqn_delta[0][slice(3, 12)]
        phase_comp_2 = phase_eqn_delta[0][slice(13, 18)]
        phase_comp_3 = phase_eqn_delta[0][slice(19, 26)]
        phase_comp_4 = phase_eqn_delta[0][slice(27, 32)]

        phase_comp_1_label = Tex("1", color=GREEN).next_to(phase_comp_1, UP)
        phase_comp_2_label = Tex("2", color=BLUE).next_to(phase_comp_2, UP)
        phase_comp_3_label = Tex("3", color=RED).next_to(phase_comp_3, UP)
        phase_comp_4_label = Tex("4", color=YELLOW).next_to(phase_comp_4, UP)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(GrowFromCenter(label), comp.animate.set_color(color))
                    for label, comp, color in zip(
                        [
                            phase_comp_1_label,
                            phase_comp_2_label,
                            phase_comp_3_label,
                            phase_comp_4_label,
                        ],
                        [phase_comp_1, phase_comp_2, phase_comp_3, phase_comp_4],
                        (GREEN, BLUE, RED, YELLOW),
                    )
                ],
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # phase_comp_4_copy = MathTex(
        #     r"2 \pi f \Delta \tau",
        #     font_size=DEFAULT_FONT_SIZE,
        #     color=YELLOW,
        # ).shift(DOWN)

        # self.play(
        #     Group(
        #         phase_comp_1_label,
        #         phase_comp_2_label,
        #         phase_comp_3_label,
        #         phase_comp_4_label,
        #         phase_eqn_delta,
        #     ).animate.to_edge(UP, LARGE_BUFF),
        #     TransformFromCopy(phase_comp_4, phase_comp_4_copy[0], path_arc=PI / 2),
        # )
        # self.play(phase_comp_4_copy.animate.scale(1.5))

        # self.wait(0.5)

        # self.play(
        #     phase_comp_4_copy[0][3:5]
        #     .animate(rate_func=rate_functions.there_and_back)
        #     .set_color(WHITE)
        #     .shift(UP / 2)
        # )

        # self.wait(0.5)

        # phase_comp_4_eq = MathTex(
        #     r"2 \pi f \Delta \tau \approx 1.29 \text{ radians}",
        #     font_size=DEFAULT_FONT_SIZE * 1.5,
        # ).move_to(phase_comp_4_copy)
        # phase_comp_4_eq[0][0:5].set_color(YELLOW)

        # self.play(
        #     TransformByGlyphMap(
        #         phase_comp_4_copy,
        #         phase_comp_4_eq,
        #         ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        #         (GrowFromCenter, [5], {"delay": 0.2}),
        #         (
        #             GrowFromCenter,
        #             [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        #             {"delay": 0.4},
        #         ),
        #     )
        # )

        self.wait(0.5)

        self.play(
            Group(
                phase_comp_1_label,
                phase_comp_2_label,
                phase_comp_3_label,
                phase_comp_4_label,
                phase_eqn_delta,
            ).animate.to_edge(UP, MED_LARGE_BUFF)
        )

        self.wait(0.5)

        phase_comp_2_val = Tex(r"$9 \cdot 10^{-10}$", color=BLUE).next_to(
            phase_comp_2, DOWN
        )
        phase_comp_1_val = (
            Tex("0.027", color=GREEN)
            .next_to(phase_comp_1, DOWN)
            .set_y(phase_comp_2_val.get_y())
        )
        phase_comp_3_val = (
            Tex(r"$4.5 \cdot 10^{-5}$", color=RED)
            .next_to(phase_comp_3, DOWN)
            .set_y(phase_comp_2_val.get_y())
        )
        phase_comp_4_val = (
            Tex(r"1.3", color=YELLOW)
            .next_to(phase_comp_4, DOWN)
            .set_y(phase_comp_2_val.get_y())
        )

        # self.play(
        #     LaggedStart(
        #         *[
        #             GrowFromCenter(val)
        #             for val in [phase_comp_1_val, phase_comp_2_val, phase_comp_3_val]
        #         ],
        #         lag_ratio=0.2,
        #     )
        # )

        self.play(GrowFromCenter(phase_comp_1_val))

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
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle = Circle(unit_circle_ax.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax.c2p(0, 0)
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
        pi_label = MathTex(r"\pi \approx 3.14").next_to(unit_circle_ax.c2p(-1, 0), LEFT)

        self.play(
            Create(unit_circle_ax),
            Create(unit_circle),
            LaggedStart(
                *[GrowFromCenter(label) for label in unit_circle_labels],
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                unit_circle_labels[2],
                pi_label,
                ([0], [0]),
                (GrowFromCenter, [1], {"delay": 0.2}),
                (GrowFromCenter, [2, 3, 4, 5], {"delay": 0.4}),
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        angle_1 = 0.027
        angle_2 = 9e-10
        angle_3 = 4.5e-5

        dot_1 = Dot(unit_circle_ax.c2p(np.cos(angle_1), np.sin(angle_1)), color=GREEN)
        dot_2 = Dot(unit_circle_ax.c2p(np.cos(angle_2), np.sin(angle_2)), color=BLUE)
        dot_3 = Dot(unit_circle_ax.c2p(np.cos(angle_3), np.sin(angle_3)), color=RED)

        line_1 = Line(unit_circle_ax.c2p(0, 0), dot_1.get_center(), color=GREEN)
        line_2 = Line(unit_circle_ax.c2p(0, 0), dot_2.get_center(), color=BLUE)
        line_3 = Line(unit_circle_ax.c2p(0, 0), dot_3.get_center(), color=RED)

        self.play(Create(line_1), Create(dot_1))

        self.wait(0.5)

        self.play(Create(line_2), Create(dot_2), GrowFromCenter(phase_comp_2_val))

        self.wait(0.5)

        self.play(Create(line_3), Create(dot_3), GrowFromCenter(phase_comp_3_val))

        self.wait(0.5)

        angle_4 = VT(0)
        dot_4 = always_redraw(
            lambda: Dot(
                unit_circle_ax.c2p(np.cos(~angle_4), np.sin(~angle_4)), color=YELLOW
            )
        )
        line_4 = always_redraw(
            lambda: Line(
                unit_circle_ax.c2p(0, 0),
                unit_circle_ax.c2p(np.cos(~angle_4), np.sin(~angle_4)),
                color=YELLOW,
            )
        )

        self.play(Create(line_4), Create(dot_4), GrowFromCenter(phase_comp_4_val))
        self.play(angle_4 @ 1.29)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        phase_eqn_approx = MathTex(
            r"\Delta \phi \approx -2 \pi f \Delta \tau",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        phase_eqn_approx[0][4:].set_color(YELLOW)

        self.play(
            LaggedStart(
                AnimationGroup(
                    # phase_comp_4_eq.animate.shift(DOWN * 8),
                    Group(
                        unit_circle_ax,
                        unit_circle_labels,
                        line_1,
                        line_2,
                        line_3,
                        dot_1,
                        dot_2,
                        dot_3,
                        pi_label,
                        unit_circle,
                    ).animate.shift(DOWN * 8),
                    Group(
                        phase_comp_1_label,
                        phase_comp_2_label,
                        phase_comp_3_label,
                        phase_comp_4_label,
                    ).animate.shift(UP * 8),
                    FadeOut(
                        Group(
                            phase_comp_1_val,
                            phase_comp_2_val,
                            phase_comp_3_val,
                            phase_comp_4_val,
                        )
                    ),
                ),
                TransformByGlyphMap(
                    phase_eqn_delta,
                    phase_eqn_approx,
                    ([3, 4, 5, 6, 7, 8, 9, 10, 11], ShrinkToCenter, {"delay": 0}),
                    ([12, 13, 14, 15, 16, 17], ShrinkToCenter, {"delay": 0}),
                    ([18, 19, 20, 21, 22, 23, 24, 25], ShrinkToCenter, {"delay": 0}),
                    ([0, 1], [0, 1], {"delay": 0.6}),
                    ([2], [2], {"delay": 0.8}),
                    ([26, 27, 28, 29, 30, 31], [3, 4, 5, 6, 7, 8], {"delay": 1}),
                ),
                lag_ratio=0.4,
            ),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"fmcw\_range\_doppler.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        self.play(notebook.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        self.play(notebook.animate.shift(DOWN * 8))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


# Pulled from notebook
# Radar setup
f = 77e9  # Hz
Tc = 40e-6  # chirp time - s
bw = 1.6e9  # bandwidth - Hz
chirp_rate = bw / Tc  # Hz/s

wavelength = c / f
M = 40  # number of chirps in coherent processing interval (CPI)

range_resolution = c / (2 * bw)


def get_time_from_dist(r):
    return r / c


def get_time_from_vel(v):
    return 2 * (v * Tc) / c


def compute_phase_diff(v):
    time_from_vel = get_time_from_vel(v)
    return 2 * PI * f * time_from_vel


def compute_f_beat(r):
    return (2 * r * bw) / (c * Tc)


max_vel = wavelength / (4 * Tc)
vel_res = wavelength / (2 * M * Tc)


class FastSlowTime(Scene):
    def construct(self):
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
        ]

        target_data = [(compute_f_beat(r), compute_phase_diff(v)) for r, v in targets]

        max_time = 6 / target_data[0][0]
        N = 1000
        Ts = max_time / N
        fs = 1 / Ts

        t = np.arange(0, max_time, 1 / fs)

        phi_0_1 = -PI / 6
        phi_0_2 = phi_0_1 + PI * 0.6
        phi_0_3 = phi_0_1 + PI * 0.3

        f_beat_1_2 = 3
        f_beat_3 = 4.5

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.4

        duration = 1
        amp_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).next_to([0, -config.frame_height / 2, 0], DOWN)

        target_1_color = GREEN
        target_2_color = PURPLE
        target_3_color = GOLD
        targets_color = ORANGE

        A_1 = VT(1)
        A_2 = VT(1)
        A_3 = VT(1)

        target_1_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: ~A_1
                * (
                    np.sin(2 * PI * f_beat_1_2 * t + phi_0_1)
                    + (1 - ~A_2) * np.sin(2 * PI * f_beat_1_2 * t + phi_0_2)
                    + (1 - ~A_3) * np.sin(2 * PI * f_beat_3 * t + phi_0_3)
                ),
                color=interpolate_color(target_1_color, targets_color, (1 - ~A_2)),
                x_range=[0, duration, 1 / 1000],
            )
        )

        target_2_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: ~A_2 * np.sin(2 * PI * f_beat_1_2 * t + phi_0_2),
                color=target_2_color,
                x_range=[0, duration, 1 / 1000],
            )
        )
        target_3_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: ~A_3 * np.sin(2 * PI * f_beat_3 * t + phi_0_3),
                color=target_3_color,
                x_range=[0, duration, 1 / 1000],
            )
        )
        plot_group = Group(amp_ax, target_1_plot, target_2_plot, target_3_plot)

        eqn_1 = MathTex(
            r"\sin{\left(2 \pi f_{beat,1} t + \phi_1\right)}", color=target_1_color
        )
        eqn_2 = MathTex(
            r"\sin{\left(2 \pi f_{beat,1} t + \phi_2\right)}", color=target_2_color
        )
        eqn_3 = MathTex(
            r"\sin{\left(2 \pi f_{beat,2} t + \phi_3\right)}", color=target_3_color
        )
        eqn_group = (
            Group(eqn_1, eqn_2, eqn_3)
            .arrange(RIGHT, LARGE_BUFF)
            .scale_to_fit_width(config.frame_width * 0.9)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(eqn_1),
                GrowFromCenter(eqn_2),
                GrowFromCenter(eqn_3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.add(target_1_plot, target_2_plot, target_3_plot)
        self.play(Group(eqn_group, amp_ax).animate.arrange(DOWN, LARGE_BUFF * 2.5))

        self.wait(0.5)

        plus = MathTex("+").set_y(eqn_1.get_y())
        plus2 = MathTex("+").set_y(eqn_1.get_y())
        plus_group = (
            Group(eqn_1.copy(), plus, eqn_2.copy(), plus2, eqn_3.copy())
            .arrange(RIGHT, SMALL_BUFF)
            .set_y(eqn_1.get_y())
        )

        self.play(
            A_1 @ (1 / 3),
            A_2 @ 0,
            A_3 @ 0,
            GrowFromCenter(plus),
            GrowFromCenter(plus2),
            eqn_1.animate.move_to(plus_group[0]),
            eqn_2.animate.move_to(plus_group[2]),
            eqn_3.animate.move_to(plus_group[4]),
            run_time=2,
        )

        self.wait(0.5)

        num_samples = 20
        sample_rects = amp_ax.get_riemann_rectangles(
            target_1_plot,
            input_sample_type="right",
            x_range=[0, duration],
            dx=duration / num_samples,
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)

        self.play(Create(sample_rects))

        self.wait(0.5)

        self.play(FadeOut(plot_group.set_z_index(-1)))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        ts_line = Line(
            sample_rects[7].get_corner(UL) + UP / 2,
            sample_rects[7].get_corner(UR) + UP / 2,
        )
        ts_line_l = Line(ts_line.get_left() + DOWN / 8, ts_line.get_left() + UP / 8)
        ts_line_r = Line(ts_line.get_right() + DOWN / 8, ts_line.get_right() + UP / 8)
        ts_label = MathTex(r"T_s", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            ts_line, UP
        )

        self.play(
            LaggedStart(
                Create(ts_line_l),
                Create(ts_line),
                Create(ts_line_r),
                FadeIn(ts_label),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))

        self.wait(0.5)

        self.play(
            Uncreate(ts_line),
            Uncreate(ts_line_l),
            Uncreate(ts_line_r),
            Transform(
                ts_label,
                ts_label.copy().next_to(sample_rects[0], DOWN),
                path_arc=PI / 2,
            ),
            LaggedStart(
                *[m.animate.shift(UP * 5) for m in [eqn_1, plus, eqn_2, plus2, eqn_3]],
                lag_ratio=0.3,
            ),
            LaggedStart(
                *[
                    Transform(
                        rect,
                        Square(
                            rect.width,
                            color=BLACK,
                            fill_color=BLUE,
                            fill_opacity=0.7,
                            stroke_width=DEFAULT_STROKE_WIDTH / 2,
                        )
                        .move_to(rect)
                        .set_y(sample_rects[0].get_y()),
                    )
                    for rect in sample_rects
                ],
                lag_ratio=0.05,
            ),
        )
        self.remove(eqn_1, plus, eqn_2, plus2, eqn_3)

        self.wait(0.5)

        n_ts_disp = 3
        n_ts = Group(
            *[
                MathTex(f"{idx}T_s", font_size=DEFAULT_FONT_SIZE * 0.8)
                .next_to(sample_rects[idx - 1], DOWN)
                .set_y(ts_label.get_y())
                .shift(DOWN * (0.5 if idx % 2 == 0 else 0))
                for idx in range(2, n_ts_disp + 2)
            ],
            MathTex(r"\cdots")
            .next_to(sample_rects[len(sample_rects) // 2], DOWN)
            .set_y(ts_label.get_y()),
            MathTex(f"NT_s", font_size=DEFAULT_FONT_SIZE * 0.8)
            .next_to(sample_rects[-1], DOWN)
            .set_y(ts_label.get_y()),
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in n_ts], lag_ratio=0.2))

        self.wait(0.5)

        n_slow_time = 8
        cpi_rects = Group(
            sample_rects,
            *[
                sample_rects.copy().shift(UP * idx * sample_rects.height)
                for idx in range(1, n_slow_time)
            ],
        )

        self.play(
            LaggedStart(
                *[TransformFromCopy(sample_rects, m) for m in cpi_rects[1:][::-1]]
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        n_tc = Group(
            *[
                MathTex(
                    f"{idx+1 if idx+1 > 1 else ''}T_c",
                    font_size=DEFAULT_FONT_SIZE * 0.8,
                ).next_to(cpi_rects[idx], LEFT)
                for idx in range(n_ts_disp)
            ],
            MathTex(r"\cdots")
            .rotate(PI / 2)
            .next_to(cpi_rects[len(cpi_rects) // 2 + 1], LEFT),
            MathTex(f"NT_c", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
                cpi_rects[-1], LEFT
            ),
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in n_tc], lag_ratio=0.2))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        fast_time_label = Tex("Fast Time", font_size=DEFAULT_FONT_SIZE * 1.2).to_edge(
            DOWN
        )
        slow_time_label = (
            Tex("Slow Time", font_size=DEFAULT_FONT_SIZE * 1.2)
            .rotate(PI / 2)
            .to_edge(LEFT)
        )

        self.play(fast_time_label.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(slow_time_label.shift(LEFT * 5).animate.shift(RIGHT * 5))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Indicate(fast_time_label))

        self.wait(0.5)

        self.play(Indicate(slow_time_label))

        self.wait(0.5)

        self.play(Indicate(ts_label))

        self.wait(0.5)

        self.play(Indicate(n_tc[0]))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ShrinkToCenter(m) for m in [ts_label, *n_ts]], lag_ratio=0.08
            ),
            LaggedStart(*[ShrinkToCenter(m) for m in n_tc], lag_ratio=0.08),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        arrows = Group(
            *[
                Arrow(m[0].get_left(), m[-1].get_right(), color=BLACK).set_z_index(2)
                for m in cpi_rects
            ]
        )

        self.play(LaggedStart(*[GrowArrow(m) for m in arrows], lag_ratio=0.1))

        self.wait(0.5)

        self.play(FadeOut(arrows))

        self.wait(0.5)

        f_beat = compute_f_beat(20)
        max_time = 20 / f_beat
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts

        fft_len = N * 8

        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]
        np.random.seed(1)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )

        rmax = c * Tc * fs / (2 * bw)
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        plot_shift_x = VT(0)
        plot_shift_y = VT(0)
        x_max = VT(40)
        x_min = VT(0)

        def plot_fft(n):
            def updater():
                X_k = fftshift(fft(cpi[n], fft_len))
                X_k /= N / 2
                X_k = np.abs(X_k)
                X_k = 10 * np.log10(X_k)
                f_X_k_log = interpolate.interp1d(ranges, X_k, fill_value="extrapolate")

                f_ax = Axes(
                    x_range=[0, 40, 10],
                    y_range=[-40, 0, 10],
                    tips=False,
                    axis_config={"include_numbers": False},
                    x_length=cpi_rects[0].width,
                    y_length=cpi_rects[0].height * 3,
                ).move_to(cpi_rects[n], LEFT)

                cpi_n_fft = (
                    f_ax.plot(
                        f_X_k_log, x_range=[~x_min, ~x_max, 1 / 100], color=IF_COLOR
                    )
                    .set_z_index(5)
                    .shift([~plot_shift_x, ~plot_shift_y, 0])
                )
                return cpi_n_fft

            return updater

        plots = [always_redraw(plot_fft(n)) for n in range(len(cpi_rects))]

        self.next_section(skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        LaggedStart(
                            *[m.animate.set_opacity(0) for m in row], lag_ratio=0.05
                        ),
                        Create(plots[n]),
                    )
                    for n, row in enumerate(cpi_rects)
                ],
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # self.play(
        #     cpi_rects.animate.arrange(UP, MED_SMALL_BUFF, center=True),
        #     y_mult @ 4,
        # )

        target_x_min = targets[0][0] - 3
        target_x_max = targets[0][0] + 3
        bot_ax = Axes(
            x_range=[0, 40, 10],
            y_range=[-40, 0, 10],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=cpi_rects[0].width,
            y_length=cpi_rects[0].height * 3,
        ).move_to(cpi_rects[0], LEFT)
        line_l = DashedLine(
            bot_ax.c2p(target_x_min, -40),
            bot_ax.c2p(target_x_min, 100),
            dash_length=DEFAULT_DASH_LENGTH * 4,
        )
        line_r = DashedLine(
            bot_ax.c2p(target_x_max, -40),
            bot_ax.c2p(target_x_max, 100),
            dash_length=DEFAULT_DASH_LENGTH * 4,
        )
        self.play(FadeIn(line_l, shift=RIGHT), FadeIn(line_r, shift=LEFT))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        mid_ax = Axes(
            x_range=[0, 40, 10],
            y_range=[-40, 0, 10],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=cpi_rects[0].width,
            y_length=cpi_rects[0].height * 3,
        ).move_to(cpi_rects[len(cpi_rects) // 2 + 1], LEFT)

        plots_new = [always_redraw(plot_fft(n)) for n in range(len(cpi_rects))]
        self.add(*plots_new)
        self.remove(*[plots[n] for n in range(len(cpi_rects))])

        plot_shift = -mid_ax.c2p(20, -20) + UP
        self.play(
            Group(line_l, line_r).animate.shift(plot_shift),
            plot_shift_x @ plot_shift[0],
            plot_shift_y @ plot_shift[1],
            x_min @ target_x_min,
            x_max @ target_x_max,
        )

        self.wait(0.5)

        plots_new_2 = Group(*[plot_fft(n)() for n in range(len(cpi_rects))])
        self.add(plots_new_2)
        self.remove(*plots_new)

        self.play(
            FadeOut(line_l, line_r),
            fast_time_label.animate.shift(DOWN * 5),
            slow_time_label.animate.shift(LEFT * 5),
            plots_new_2.animate.arrange(UP, -SMALL_BUFF * 1.5),
        )
        print(cpi_rects[0].width, cpi_rects[0].height * 3)

        self.wait(2)


class PeakPhase(MovingCameraScene):
    def construct(self):
        n_slow_time = 8
        f_beat = compute_f_beat(20)
        max_time = 20 / f_beat
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts

        fft_len = N * 8

        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]
        np.random.seed(1)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )

        rmax = c * Tc * fs / (2 * bw)
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        target_x_min = targets[0][0] - 3
        target_x_max = targets[0][0] + 3

        def plot_fft(n):
            def updater():
                X_k = fftshift(fft(cpi[n], fft_len))
                X_k /= N / 2
                X_k = np.abs(X_k)
                X_k = 10 * np.log10(X_k)
                f_X_k_log = interpolate.interp1d(ranges, X_k, fill_value="extrapolate")

                f_ax = Axes(
                    x_range=[0, 40, 10],
                    y_range=[-40, 0, 10],
                    tips=False,
                    axis_config={"include_numbers": False},
                    x_length=9.050957575757575,
                    y_length=1.3589333333333418,
                )

                cpi_n_fft = f_ax.plot(
                    f_X_k_log,
                    x_range=[target_x_min, target_x_max, 1 / 100],
                    color=IF_COLOR,
                ).set_z_index(5)
                return Group(f_ax, cpi_n_fft)

            return updater

        plots = Group(*[plot_fft(n)()[1] for n in range(n_slow_time)])

        self.add(plots.arrange(UP, -SMALL_BUFF * 1.5).move_to(ORIGIN))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        def create_phase_shifted_beat_eqn(n):
            n_fmt = (
                r"+ \Delta \phi"
                if n == 1
                else r"- \Delta \phi"
                if n == -1
                else ""
                if n == 0
                else f"+ {n} \\Delta \\phi"
            )
            eqn = MathTex(
                f"\\sin{{\\left( 2 \\pi f_{{beat}} t + \\phi_0 {n_fmt} \\right)}}",
            ).next_to(plots[n], LEFT, LARGE_BUFF)
            eqn[0][6:11].set_color(IF_COLOR)
            return eqn

        beat_signals = Group(
            *[create_phase_shifted_beat_eqn(n) for n in range(len(plots))]
        )
        for beat_signal in beat_signals[1:]:
            beat_signal.set_x(beat_signals[0].get_x())

        self.play(
            LaggedStart(
                beat_signals[0].shift(LEFT * 8).animate.shift(RIGHT * 8),
                beat_signals[1].shift(LEFT * 8).animate.shift(RIGHT * 8),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    beat_signal.shift(LEFT * 8).animate.shift(RIGHT * 8)
                    for beat_signal in beat_signals[2:]
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        slow_time_arrow = Arrow(plots[0].get_bottom(), plots[-1].get_top()).shift(
            RIGHT * 2
        )

        self.play(GrowArrow(slow_time_arrow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m[0][-3:-1]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(UP / 3)
                    .set_color(YELLOW)
                    for m in beat_signals[1:]
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ns = [Tex(f"{n}").next_to(plot, RIGHT) for n, plot in enumerate(plots)]
        for n in ns[1:]:
            n.set_x(ns[0].get_x())

        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in ns], lag_ratio=0.2),
            FadeOut(slow_time_arrow),
        )

        self.wait(0.5)

        phase_eqn = MathTex(r"\phi_n = \phi_0 + m \Delta \phi").to_edge(
            RIGHT, LARGE_BUFF
        )

        self.play(phase_eqn.shift(RIGHT * 5).animate.shift(LEFT * 5))

        self.wait(0.5)

        self.play(
            phase_eqn[0][-2:]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        phase_eqn_zero = MathTex(r"\phi_n = m \Delta \phi").move_to(phase_eqn)

        self.play(
            TransformByGlyphMap(
                phase_eqn,
                phase_eqn_zero,
                ([0, 1, 2], [0, 1, 2]),
                ([3, 4, 5], ShrinkToCenter),
                ([6, 7, 8], [3, 4, 5]),
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
        ).to_edge(RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                phase_eqn_zero.animate.next_to(unit_circle_group, UP, MED_SMALL_BUFF),
                unit_circle_group.shift(RIGHT * 5).animate.shift(LEFT * 5),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        n_counter = VT(0)
        phase_delta = VT(1.3)
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

        n_label = MathTex(r"\leftarrow m").next_to(ns[0], RIGHT, SMALL_BUFF)

        self.play(Create(phase_line), Create(phase_dot), FadeIn(n_label, shift=LEFT))

        self.wait(0.5)

        for n in range(1, len(ns)):
            self.play(
                n_counter + 1,
                phase @ (n * ~phase_delta),
                n_label.animate.next_to(ns[n], RIGHT, SMALL_BUFF),
            )
            self.wait(0.5)

        self.wait(0.5)

        n = 0
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.shift(
                RIGHT
                * (
                    unit_circle_group.get_left()
                    - (self.camera.frame.get_left() + LARGE_BUFF)
                )
            ),
            n_label.animate.set_opacity(0),
            n_counter @ 0,
            phase @ (n * ~phase_delta),
            # n_label.animate.next_to(ns[n], RIGHT, SMALL_BUFF),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

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
                lambda t: np.sin(t), x_range=[0, ~phase, 1 / 100], color=YELLOW
            )
        )

        sine_line = always_redraw(
            lambda: DashedLine(
                unit_circle_ax.c2p(np.cos(~phase), np.sin(~phase)),
                sine_ax.input_to_graph_point(~phase, sine_plot),
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.add(sine_plot)
        self.play(Create(sine_line))

        for n in range(1, len(ns)):
            self.play(phase @ (n * ~phase_delta))
            self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        phase_delta_eq = MathTex(f"\\Delta \\phi = {~phase_delta:.1f}").next_to(
            self.camera.frame.get_bottom(), UP, MED_LARGE_BUFF
        )

        phase_delta_delta_tau_eq = MathTex(
            f"\\Delta \\phi \\approx -2 \\pi f \\Delta \\tau = {~phase_delta:.1f}"
        ).move_to(phase_delta_eq)

        phase_delta_vel_eq = MathTex(
            f"\\Delta \\phi \\approx -2 \\pi f \\frac{{2 v T_c}}{{c}} = {~phase_delta:.1f}"
        ).move_to(phase_delta_eq)
        phase_delta_vel_eq_2_6 = MathTex(
            f"\\Delta \\phi \\approx -2 \\pi f \\frac{{2 v T_c}}{{c}} = 2.6"
        ).move_to(phase_delta_vel_eq, LEFT)

        self.play(FadeIn(phase_delta_eq, shift=UP * 3))

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                phase_delta_eq,
                phase_delta_delta_tau_eq,
                ([0, 1], [0, 1]),
                ([2, 3, 4, 5], [9, 10, 11, 12]),
                (GrowFromCenter, [2, 3, 4, 5, 6, 7, 8], {"delay": 0.4}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                phase_delta_delta_tau_eq,
                phase_delta_vel_eq,
                ([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]),
                ([7, 8], ShrinkToCenter),
                (GrowFromCenter, [7, 8, 9, 10, 11, 12], {"delay": 0.4}),
                ([9, 10, 11, 12], [13, 14, 15, 16]),
            )
        )

        self.wait(0.5)

        self.play(
            phase_delta_vel_eq[0][8]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        # self.wait(0.5)

        # self.remove(phase_delta_vel_eq)
        # self.add(phase_delta_vel_eq_updating)
        # # phase_delta_vel_eq.become(phase_delta_vel_eq_updating)

        self.wait(0.5)

        self.play(
            LaggedStart(
                phase_delta_vel_eq[0][-3:].animate.shift(DOWN * 4),
                phase_delta_vel_eq_2_6[0][-3:].shift(DOWN * 4).animate.shift(UP * 4),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(phase @ 0, run_time=3)

        phase_delta @= 2.6

        # self.wait(0.5)

        # self.play(phase_delta @ 2.6)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # Yes, there's probably a much better way of doing this...
        sine_ax_2 = Axes(
            x_range=[0, 8 * PI, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=config.frame_width * 0.7,
            y_length=unit_circle.height,
        ).next_to(unit_circle, RIGHT, 0)
        sine_plot_2 = always_redraw(
            lambda: sine_ax_2.plot(
                lambda t: np.sin(t), x_range=[0, ~phase, 1 / 100], color=YELLOW
            )
        )
        self.add(sine_plot_2)
        self.remove(sine_plot)

        sine_line_2 = always_redraw(
            lambda: DashedLine(
                unit_circle_ax.c2p(np.cos(~phase), np.sin(~phase)),
                sine_ax_2.input_to_graph_point(~phase, sine_plot_2),
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.add(sine_line_2)
        self.remove(sine_line)

        for n in range(1, len(ns)):
            self.play(phase @ (n * ~phase_delta))
            self.wait(0.5)

        self.wait(0.5)

        self.remove(*ns)
        self.play(
            self.camera.frame.animate.restore(),
            Uncreate(sine_plot_2),
            FadeOut(sine_line_2, phase_delta_vel_eq, phase_delta_vel_eq_2_6),
        )

        self.wait(0.5)

        slow_time_arrow = Arrow(plots[0].get_bottom(), plots[-1].get_top()).shift(RIGHT)
        fft_label = (
            Tex("Fourier Transform")
            .rotate(PI / 2)
            .next_to(slow_time_arrow, RIGHT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                GrowArrow(slow_time_arrow),
                FadeIn(fft_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.next_to(self.camera.frame.get_bottom(), DOWN)
        )

        self.wait(2)


class SlowTimeFFT(MovingCameraScene):
    def construct(self):
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]

        sep = "\n\t"
        print(
            "Targets:",
            sep
            + sep.join(
                [
                    f"{idx}: distance = {r:.2f} m, velocity = {v:.2f} m/s"
                    for idx, (r, v) in enumerate(targets, start=1)
                ]
            ),
        )

        f_beat = compute_f_beat(20)
        max_time = 20 / f_beat
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts

        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )
        range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)

        vels = np.linspace(-max_vel, max_vel, M)
        rmax = c * Tc * fs / (2 * bw)
        n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        fft_len = N * 8
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        roi = 20
        roi_idx = np.abs(n_ranges - roi).argmin()

        fft_slow_time_label = Tex("Fourier Transform Along Slow-Time Axis").to_edge(
            UP, LARGE_BUFF
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(FadeIn(fft_slow_time_label))

        self.wait(0.5)

        x_len = config.frame_width * 0.8
        y_len = config.frame_height * 0.6
        ax = Axes(
            x_range=[-max_vel, max_vel, max_vel / 8],
            y_range=[-6, 10, 4],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, MED_LARGE_BUFF)
        ax_labels = Group(
            *[
                MathTex(label, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
                    ax.c2p(v, 0), DOWN
                )
                for label, v in [
                    (r"-v_{max}", -max_vel),
                    (r"-\frac{v_{max}}{2}", -max_vel / 2),
                    (r"\frac{v_{max}}{2}", max_vel / 2),
                    (r"v_{max}", max_vel),
                ]
            ]
        )
        ax_labels_num = Group(
            *[
                MathTex(label, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
                    ax.c2p(v, 0), DOWN
                )
                for label, v in [
                    (f"{int(-max_vel)}", -max_vel),
                    (f"{int(-max_vel/2)}", -max_vel / 2),
                    (f"{int(max_vel/2)}", max_vel / 2),
                    (f"{int(max_vel)}", max_vel),
                ]
            ]
        )

        f_vel_fft = interpolate.interp1d(
            vels, 10 * np.log10(range_doppler[roi_idx]), fill_value="extrapolate"
        )
        vel_plot = ax.plot(f_vel_fft, x_range=[-20, 20, 1 / 100], color=ORANGE)

        self.play(
            LaggedStart(
                Create(ax),
                LaggedStart(
                    *[GrowFromCenter(label) for label in ax_labels],
                    lag_ratio=0.2,
                ),
                Create(vel_plot),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    Transform(label, label_num)
                    for label, label_num in zip(ax_labels, ax_labels_num)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale(1.8).move_to(
                fft_slow_time_label.get_top() + UP, UP
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        x_len = config.frame_width * 0.25
        y_len = config.frame_width * 0.25
        unit_circle_ax_l = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle_l = Circle(unit_circle_ax_l.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax_l.c2p(0, 0)
        )
        unit_circle_labels_l = Group(
            *[
                MathTex(s).next_to(unit_circle_ax_l.c2p(*a), d)
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )

        unit_circle_ax_r = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle_r = Circle(unit_circle_ax_r.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax_r.c2p(0, 0)
        )
        unit_circle_labels_r = Group(
            *[
                MathTex(s).next_to(unit_circle_ax_r.c2p(*a), d)
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )
        unit_circle_l_group = Group(
            unit_circle_ax_l, unit_circle_l, unit_circle_labels_l
        ).next_to(self.camera.frame.get_corner(DL), UR, LARGE_BUFF)
        unit_circle_r_group = Group(
            unit_circle_ax_r, unit_circle_r, unit_circle_labels_r
        ).next_to(self.camera.frame.get_corner(DR), UL, LARGE_BUFF)

        vel_1 = targets[0][1] / np.abs(targets[0][1])
        vel_2 = targets[1][1] / np.abs(targets[0][1])

        angle_1 = VT(0)
        angle_2 = VT(0)

        dot_1 = always_redraw(
            lambda: Dot(
                unit_circle_ax_r.c2p(np.cos(~angle_1), np.sin(~angle_1)), color=YELLOW
            )
        )
        line_1 = always_redraw(
            lambda: Line(
                unit_circle_ax_r.c2p(0, 0),
                unit_circle_ax_r.c2p(np.cos(~angle_1), np.sin(~angle_1)),
                color=YELLOW,
            )
        )
        dot_2 = always_redraw(
            lambda: Dot(
                unit_circle_ax_l.c2p(np.cos(~angle_2), np.sin(~angle_2)), color=YELLOW
            )
        )
        line_2 = always_redraw(
            lambda: Line(
                unit_circle_ax_l.c2p(0, 0),
                unit_circle_ax_l.c2p(np.cos(~angle_2), np.sin(~angle_2)),
                color=YELLOW,
            )
        )

        peak_r = ax.input_to_graph_point(targets[0][1], vel_plot)
        peak_l = ax.input_to_graph_point(targets[1][1], vel_plot)

        peak_r_bez = CubicBezier(
            peak_r + [0.5, 0.1, 0],
            peak_r + [3, 5, 0],
            unit_circle_r_group.get_top() + [0, 5, 0],
            unit_circle_r_group.get_top() + [0, 0.1, 0],
        )
        peak_l_bez = CubicBezier(
            peak_l + [-0.1, 0.1, 0],
            peak_l + [-3, 5, 0],
            unit_circle_l_group.get_top() + [0, 5, 0],
            unit_circle_l_group.get_top() + [0, 0.1, 0],
        )

        self.play(
            unit_circle_l_group.shift(LEFT * 8).animate.shift(RIGHT * 8),
            unit_circle_r_group.shift(RIGHT * 8).animate.shift(LEFT * 8),
            Create(peak_l_bez),
            Create(peak_r_bez),
        )

        self.wait(0.5)

        self.play(Create(line_1), Create(dot_1), Create(line_2), Create(dot_2))

        # angle_1.add_updater(phase_updater(vel_1))
        # angle_2.add_updater(phase_updater(vel_2))

        self.wait(0.5)

        self.play(
            angle_1 @ (vel_1 * 8 * PI),
            angle_2 @ (vel_2 * 8 * PI),
            rate_func=rate_functions.linear,
            run_time=10,
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class Solution(MovingCameraScene):
    def construct(self):
        solution_label = Tex("The Solution", font_size=DEFAULT_FONT_SIZE * 1.8).to_edge(
            UP, LARGE_BUFF
        )

        N = 20
        M = 8
        sample_rects = Group(
            *[
                Square(
                    color=BLACK,
                    fill_color=BLUE,
                    fill_opacity=0.7,
                    stroke_width=DEFAULT_STROKE_WIDTH / 2,
                )
                for _ in range(N)
            ]
        ).arrange(RIGHT, 0)
        cpi_rects = (
            Group(*[sample_rects.copy() for _ in range(M)])
            .arrange(UP, 0)
            .scale_to_fit_height(config.frame_height * 0.5)
            .to_edge(DOWN, LARGE_BUFF)
        )
        fast_time_label = Tex("Fast Time").next_to(cpi_rects, DOWN)
        slow_time_label = Tex("Slow Time").rotate(PI / 2).next_to(cpi_rects, LEFT)

        self.next_section(skip_animations=skip_animations(True))
        self.play(solution_label.shift(UP * 5).animate.shift(DOWN * 5))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        *[GrowFromCenter(rect) for rect in row],
                        lag_ratio=0.03,
                    )
                    for row in cpi_rects
                ],
                fast_time_label.shift(DOWN * 5).animate.shift(UP * 5),
                slow_time_label.shift(LEFT * 5).animate.shift(RIGHT * 5),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        l_brace = Brace(
            Group(cpi_rects, slow_time_label, fast_time_label), direction=LEFT
        )
        fft_label = Tex("2D FFT").rotate(PI / 2).next_to(l_brace, LEFT)
        r_brace = Brace(
            Group(cpi_rects, slow_time_label, fast_time_label), direction=RIGHT
        )
        fft_group = Group(fft_label, l_brace, r_brace)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(fft_group.width * 1.2),
                AnimationGroup(
                    Group(fft_label, l_brace).shift(LEFT * 8).animate.shift(RIGHT * 8),
                    r_brace.shift(RIGHT * 8).animate.shift(LEFT * 8),
                ),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        row_arrows = Group(
            *[
                Arrow(m[0].get_left(), m[-1].get_right(), color=BLACK).set_z_index(2)
                for m in cpi_rects
            ]
        )
        col_arrows = Group(
            *[
                Arrow(m0.get_bottom(), m1.get_top(), color=BLACK).set_z_index(2)
                for m0, m1 in zip(cpi_rects[0], cpi_rects[-1])
            ]
        )

        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in row_arrows], lag_ratio=0.05)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in col_arrows], lag_ratio=0.05)
        )

        self.wait(0.5)

        range_label = Tex("Range").move_to(fast_time_label)
        velocity_label = Tex("Velocity").rotate(PI / 2).move_to(slow_time_label)

        self.play(
            Transform(
                fast_time_label, fast_time_label.copy().shift(DOWN * 5), path_arc=PI
            ),
            ReplacementTransform(
                range_label.copy().shift(DOWN * 5), range_label, path_arc=PI
            ),
        )

        self.wait(0.5)

        self.play(
            Transform(
                slow_time_label, slow_time_label.copy().shift(LEFT * 5), path_arc=PI
            ),
            ReplacementTransform(
                velocity_label.copy().shift(LEFT * 5), velocity_label, path_arc=PI
            ),
        )

        self.wait(0.5)

        range_doppler_code = Code(
            code="from numpy.fft import fftshift, fft2\n\n"
            "range_doppler = 10 * np.log10(fftshift(np.abs(fft2(cpi))))",
            tab_width=4,
            background="window",
            font="FiraCode Nerd Font Mono",
            language="Python",
        )
        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")
        notebook_reminder = Tex(
            r"fmcw\_range\_doppler.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).next_to(
            range_doppler_code, DOWN
        )

        code_group = Group(range_doppler_code, notebook).set_z_index(3)

        self.play(
            code_group.next_to([0, -config.frame_height / 2, 0], DOWN).animate.move_to(
                ORIGIN
            ),
            FadeOut(
                col_arrows,
                row_arrows,
                fft_group,
                cpi_rects,
                velocity_label,
                range_label,
            ),
        )

        self.wait(0.5)

        self.play(
            solution_label.animate.next_to([0, config.frame_height / 2, 0], UP),
            code_group.animate.next_to([0, -config.frame_height / 2, 0], DOWN),
        )

        self.wait(2)


class RangeDoppler(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ic = ImageMobject("../props/static/microcontroller.png").scale_to_fit_width(
            config.frame_height * 0.4
        )
        ic_label = Tex("AWR1642").next_to(ic, UL, LARGE_BUFF).shift(LEFT)
        ic_label_l1 = Line(
            ic_label.get_corner(DR) + [1, -0.1, 0],
            ic_label.get_corner(DL) + [-0.1, -0.1, 0],
        )
        ic_label_l2 = Line(
            ic.get_center() + [-0.3, 0.5, 0],
            ic_label.get_corner(DR) + [1, -0.1, 0],
        )
        ic_label_dot = Dot(ic.get_center() + [-0.3, 0.5, 0])

        f_label = Tex(r"$f_c = 77$ GHz").next_to(ic, DR, LARGE_BUFF).shift(LEFT)
        f_label_l1 = Line(
            f_label.get_corner(DL) + [-0.3, -0.1, 0],
            f_label.get_corner(DR) + [0.1, -0.1, 0],
        )
        f_label_dot = Dot(ic.get_center() + [0.3, -0.5, 0])
        f_label_l2 = Line(
            f_label_dot.get_center(),
            f_label_l1.get_start(),
        )

        bw_label = Tex(r"BW = 1.6 GHz").next_to(ic, UR, LARGE_BUFF).shift(LEFT)
        bw_label_l1 = Line(
            bw_label.get_corner(DL) + [-0.3, -0.1, 0],
            bw_label.get_corner(DR) + [0.1, -0.1, 0],
        )
        bw_label_dot = Dot(ic.get_center() + [0.3, 0.5, 0])
        bw_label_l2 = Line(
            bw_label_dot.get_center(),
            bw_label_l1.get_start(),
        )

        chirp_time_label = (
            Tex(r"$T_c = 40 \mu$s").next_to(ic, DL, LARGE_BUFF).shift(LEFT)
        )
        chirp_time_label_l1 = Line(
            chirp_time_label.get_corner(DR) + [0.3, -0.1, 0],
            chirp_time_label.get_corner(DL) + [-0.1, -0.1, 0],
        )
        chirp_time_label_dot = Dot(ic.get_center() + [-0.3, -0.3, 0])
        chirp_time_label_l2 = Line(
            chirp_time_label_dot.get_center(),
            chirp_time_label_l1.get_start(),
        )

        self.play(ic.shift(DOWN * 8).animate.shift(UP * 8))
        self.play(
            LaggedStart(
                Create(ic_label_dot),
                Create(ic_label_l2),
                AnimationGroup(Create(ic_label_l1), FadeIn(ic_label)),
                lag_ratio=0.5,
            ),
            LaggedStart(
                Create(f_label_dot),
                Create(f_label_l2),
                AnimationGroup(Create(f_label_l1), FadeIn(f_label)),
                lag_ratio=0.5,
            ),
            LaggedStart(
                Create(bw_label_dot),
                Create(bw_label_l2),
                AnimationGroup(Create(bw_label_l1), FadeIn(bw_label)),
                lag_ratio=0.5,
            ),
            LaggedStart(
                Create(chirp_time_label_dot),
                Create(chirp_time_label_l2),
                AnimationGroup(Create(chirp_time_label_l1), FadeIn(chirp_time_label)),
                lag_ratio=0.5,
            ),
        )

        self.wait(0.5)

        duration = 1
        fs = 5000
        step = 1 / fs

        sawtooth_f = VT(1)

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.4
        f_ax = Axes(
            x_range=[0, duration, duration / 4],
            y_range=[0, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        f_ax_labels = f_ax.get_axis_labels(MathTex("t"), MathTex("f"))
        f_ax_labels[1].shift(LEFT / 2)

        sawtooth_f_graph = always_redraw(
            lambda: f_ax.plot(
                lambda t: (signal.sawtooth(2 * PI * ~sawtooth_f * t) + 1) / 2,
                x_range=[0, 1, step],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )
        f_plot_group = VGroup(f_ax, f_ax_labels).to_edge(DOWN, LARGE_BUFF)

        chirp_time_line = Line(f_ax.c2p(0, 1.2), f_ax.c2p(1, 1.2))
        chirp_time_line_r = Line(
            chirp_time_line.get_end() + DOWN / 8, chirp_time_line.get_end() + UP / 8
        )
        chirp_time_line_l = Line(
            chirp_time_line.get_start() + DOWN / 8, chirp_time_line.get_start() + UP / 8
        )

        bw_line = Line(f_ax.c2p(1.05, 0), f_ax.c2p(1.05, 1))
        bw_line_r = Line(bw_line.get_end() + LEFT / 8, bw_line.get_end() + RIGHT / 8)
        bw_line_l = Line(
            bw_line.get_start() + LEFT / 8,
            bw_line.get_start() + RIGHT / 8,
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(ic_label_dot),
                    Uncreate(ic_label_l1),
                    Uncreate(ic_label_l2),
                    FadeOut(ic_label),
                    Uncreate(f_label_dot),
                    Uncreate(f_label_l1),
                    Uncreate(f_label_l2),
                    Uncreate(chirp_time_label_dot),
                    Uncreate(chirp_time_label_l1),
                    Uncreate(chirp_time_label_l2),
                    Uncreate(bw_label_dot),
                    Uncreate(bw_label_l1),
                    Uncreate(bw_label_l2),
                    FadeOut(bw_label[0][2:], f_label[0][2:], chirp_time_label[0][2:]),
                ),
                AnimationGroup(
                    chirp_time_label[0][:2].animate.next_to(chirp_time_line, UP),
                    f_label[0][:2].animate.next_to(f_ax.c2p(0, 0.5), LEFT),
                    bw_label[0][:2].animate.next_to(bw_line, RIGHT),
                    ic.set_z_index(-1).animate.scale(0.5).to_corner(UL),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.add(sawtooth_f_graph)
        f_plot_group.save_state()
        self.play(
            f_plot_group.next_to(
                [0, -config.frame_height / 2, 0], DOWN
            ).animate.restore()
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(chirp_time_line_l),
                Create(chirp_time_line),
                Create(chirp_time_line_r),
                lag_ratio=0.2,
            ),
            LaggedStart(
                Create(bw_line_l),
                Create(bw_line),
                Create(bw_line_r),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        M_disp = 8

        chirp_time_line_new = Line(f_ax.c2p(0, 1.2), f_ax.c2p(1 / M_disp, 1.2))
        chirp_time_line_r_new = Line(
            chirp_time_line_new.get_end() + DOWN / 8,
            chirp_time_line_new.get_end() + UP / 8,
        )

        M_label = always_redraw(
            lambda: Tex(f"$M = $ {int(~sawtooth_f)}").to_edge(UP, LARGE_BUFF)
        )

        self.play(FadeIn(M_label, shift=DOWN * 2))

        self.wait(0.5)

        self.play(
            sawtooth_f @ M_disp,
            Transform(chirp_time_line, chirp_time_line_new),
            Transform(chirp_time_line_r, chirp_time_line_r_new),
            chirp_time_label[0][:2].animate.next_to(chirp_time_line_new, UP),
            run_time=3,
        )

        self.wait(0.5)

        v_res_eqn = MathTex(r"\Delta v = \frac{\lambda}{2 M T_c}").next_to(
            M_label, DOWN
        )

        self.play(v_res_eqn.shift(RIGHT * 8).animate.shift(LEFT * 8))

        self.wait(0.5)

        M_disp = 40

        chirp_time_line_new_2 = Line(f_ax.c2p(0, 1.2), f_ax.c2p(1 / M_disp, 1.2))
        chirp_time_line_r_new_2 = Line(
            chirp_time_line_new_2.get_end() + DOWN / 8,
            chirp_time_line_new_2.get_end() + UP / 8,
        )

        self.play(
            sawtooth_f @ M_disp,
            Transform(chirp_time_line, chirp_time_line_new_2),
            Transform(chirp_time_line_r, chirp_time_line_r_new_2),
            chirp_time_label[0][:2].animate.next_to(chirp_time_line_new_2, UP),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        v_res_eqn_eq = MathTex(
            r"\Delta v = \frac{\lambda}{2 M T_c} \approx 1.2 \text{ m/s}"
        ).move_to(v_res_eqn)

        self.play(
            TransformByGlyphMap(
                v_res_eqn,
                v_res_eqn_eq,
                ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                (GrowFromCenter, [9, 10, 11, 12, 13, 14, 15]),
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        chirp_time_line_new_3 = Line(f_ax.c2p(0, 1.2), f_ax.c2p(1, 1.2))
        chirp_time_line_r_new_3 = Line(
            chirp_time_line_new_3.get_end() + DOWN / 8,
            chirp_time_line_new_3.get_end() + UP / 8,
        )
        # chirp_time_label_no_val = MathTex(r"T_c").move_to(chirp_time_label[0][:2])
        cpi_label = Tex("CPI | $M T_c$").next_to(chirp_time_line_new_3, UP)

        M_label_static = Tex(f"$M = $ {int(~sawtooth_f)}").move_to(M_label)

        self.add(M_label_static)
        self.remove(M_label)

        self.play(
            sawtooth_f @ M_disp,
            Transform(chirp_time_line, chirp_time_line_new_3),
            Transform(chirp_time_line_r, chirp_time_line_r_new_3),
            LaggedStart(
                ReplacementTransform(
                    chirp_time_label[0][:2], cpi_label[0][-2:], path_arc=-PI / 6
                ),
                FadeIn(cpi_label[0][:3], shift=DOWN),
                FadeIn(cpi_label[0][3], shift=DOWN),
                FadeIn(cpi_label[0][4], shift=DOWN),
                lag_ratio=0.3,
            ),
            Group(M_label_static, v_res_eqn_eq)
            .animate.arrange(RIGHT, LARGE_BUFF)
            .to_edge(UP, MED_LARGE_BUFF),
        )

        self.wait(0.5)

        v_max_label = MathTex(
            r"v_{max} = \frac{\lambda}{4 T_c} \approx 24.3 \text{ m/s}"
        ).next_to([config.frame_width / 2, M_label_static.get_y(), 0], RIGHT)

        self.play(
            Group(M_label_static, v_res_eqn_eq, v_max_label)
            .animate.arrange(RIGHT, MED_LARGE_BUFF)
            .set_y(M_label_static.get_y()),
            ic.animate.shift(LEFT * 3),
        )

        self.play(
            LaggedStart(
                *[
                    m.animate.shift(UP * 2)
                    for m in [M_label_static, v_max_label, v_res_eqn_eq]
                ],
                lag_ratio=0.2,
            ),
            Group(
                f_plot_group,
                chirp_time_line,
                chirp_time_line_l,
                chirp_time_line_r,
                bw_line,
                bw_line_l,
                bw_line_r,
                cpi_label,
                f_label[0][:2],
                bw_label[0][:2],
            ).animate.shift(DOWN * 8),
        )

        self.wait(2)


class RangeDopplerReal(Scene):
    def construct(self):
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]
        f_beat = compute_f_beat(20)
        max_time = 20 / f_beat
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts
        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                # * window
                for m in range(M)
            ]
        )

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.2

        self.next_section(skip_animations=skip_animations(True))
        # self.add(ax, cpi_n_plot)

        target_1_color = GREEN
        target_2_color = PURPLE
        target_3_color = GOLD

        target_1_label_group = Group(
            Tex("Target 1:", font_size=DEFAULT_FONT_SIZE * 1.5, color=target_1_color),
            MathTex(r"R &= 20 \text{ m} \\ v &= 10 \text{ m/s}", color=target_1_color),
        ).arrange(DOWN, aligned_edge=LEFT)
        target_2_label_group = Group(
            Tex("Target 2:", font_size=DEFAULT_FONT_SIZE * 1.5, color=target_2_color),
            MathTex(r"R &= 20 \text{ m} \\ v &= -10 \text{ m/s}", color=target_2_color),
        ).arrange(DOWN, aligned_edge=LEFT)
        target_3_label_group = Group(
            Tex("Target 3:", font_size=DEFAULT_FONT_SIZE * 1.5, color=target_3_color),
            MathTex(r"R &= 5 \text{ m} \\ v &= 5 \text{ m/s}", color=target_3_color),
        ).arrange(DOWN, aligned_edge=LEFT)

        target_labels = Group(
            target_1_label_group, target_2_label_group, target_3_label_group
        ).arrange(RIGHT, LARGE_BUFF)

        self.play(GrowFromCenter(target_1_label_group))

        self.wait(0.5)

        self.play(GrowFromCenter(target_2_label_group))

        self.wait(0.5)

        self.play(GrowFromCenter(target_3_label_group))

        self.wait(0.5)

        same_range_targets = SurroundingRectangle(
            Group(target_1_label_group, target_2_label_group)
        )

        self.play(Create(same_range_targets))

        self.wait(0.5)

        self.play(Uncreate(same_range_targets))

        self.wait(0.5)

        self.play(target_labels.animate.shift(UP * 8))
        self.remove(target_labels)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        M_disp = 6

        time_series_x0 = [VT(0) for _ in range(M_disp)]
        time_series_x1 = [VT(max_time) for _ in range(M_disp)]
        fft_x0 = [VT(0) for _ in range(M_disp)]
        fft_x1 = [VT(0) for _ in range(M_disp)]

        def create_cpi_n_plot(n, x0, x1):
            ax = Axes(
                x_range=[0, max_time, max_time / 4],
                y_range=[-3, 3, 0.5],
                tips=False,
                axis_config={"include_numbers": False},
                x_length=x_len,
                y_length=y_len,
            ).set_opacity(0)
            f_cpi_n = interpolate.interp1d(t, cpi[n], fill_value="extrapolate")

            def updater():
                cpi_n_plot = ax.plot(
                    f_cpi_n,
                    x_range=[~x0, ~x1, 10 / fs],
                    color=IF_COLOR,
                )
                return cpi_n_plot

            return ax, updater

        cpi_plot_groups = [
            create_cpi_n_plot(n, x0, x1)
            for n, (x0, x1) in enumerate(zip(time_series_x0, time_series_x1))
        ]

        cpi_axes = Group(*[cpi_n_plot[0] for cpi_n_plot in cpi_plot_groups])
        cpi_plots = Group(
            *[always_redraw(cpi_n_plot[1]) for cpi_n_plot in cpi_plot_groups]
        )

        self.play(*[Create(cpi_plot) for cpi_plot in cpi_plots])
        self.play(cpi_axes.animate.arrange(UP, -MED_LARGE_BUFF))

        self.wait(0.5)

        x_len_uc = config.frame_width * 0.1
        y_len_uc = config.frame_width * 0.1
        unit_circle_ax_top = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len_uc,
            y_length=y_len_uc,
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle_top = Circle(unit_circle_ax_top.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax_top.c2p(0, 0)
        )
        unit_circle_labels_top = Group(
            *[
                MathTex(s, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
                    unit_circle_ax_top.c2p(*a), d, SMALL_BUFF
                )
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )

        unit_circle_ax_mid = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len_uc,
            y_length=y_len_uc,
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle_mid = Circle(unit_circle_ax_mid.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax_mid.c2p(0, 0)
        )
        unit_circle_labels_mid = Group(
            *[
                MathTex(s, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
                    unit_circle_ax_mid.c2p(*a), d, SMALL_BUFF
                )
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )
        unit_circle_ax_bot = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len_uc,
            y_length=y_len_uc,
        ).to_edge(DOWN, LARGE_BUFF)

        unit_circle_bot = Circle(unit_circle_ax_bot.c2p(1, 0)[0], color=WHITE).move_to(
            unit_circle_ax_bot.c2p(0, 0)
        )
        unit_circle_labels_bot = Group(
            *[
                MathTex(s, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
                    unit_circle_ax_bot.c2p(*a), d, SMALL_BUFF
                )
                for s, a, d in [
                    (r"0", (1, 0), RIGHT),
                    (r"\pi / 2", (0, 1), UP),
                    (r"\pi", (-1, 0), LEFT),
                    (r"3 \pi / 2", (0, -1), DOWN),
                ]
            ]
        )
        unit_circle_top_group = Group(
            unit_circle_ax_top, unit_circle_top, unit_circle_labels_top
        )
        unit_circle_mid_group = Group(
            unit_circle_ax_mid, unit_circle_mid, unit_circle_labels_mid
        )
        unit_circle_bot_group = Group(
            unit_circle_ax_bot, unit_circle_bot, unit_circle_labels_bot
        )
        unit_circle_group = (
            Group(unit_circle_top_group, unit_circle_mid_group, unit_circle_bot_group)
            .arrange(DOWN, SMALL_BUFF)
            .to_edge(LEFT)
        )
        unit_circle_mid_group.shift(RIGHT)

        phase = VT(0)

        phase_dot_top = always_redraw(
            lambda: Dot(
                unit_circle_ax_top.c2p(np.cos(~phase * 2), np.sin(~phase * 2)),
                color=target_1_color,
            ).set_z_index(1)
        )
        phase_line_top = always_redraw(
            lambda: Line(
                unit_circle_ax_top.c2p(0, 0),
                unit_circle_ax_top.c2p(np.cos(~phase * 2), np.sin(~phase * 2)),
                color=target_1_color,
            ).set_z_index(1)
        )
        phase_dot_mid = always_redraw(
            lambda: Dot(
                unit_circle_ax_mid.c2p(np.cos(~phase * -1), np.sin(~phase * -1)),
                color=target_2_color,
            ).set_z_index(1)
        )
        phase_line_mid = always_redraw(
            lambda: Line(
                unit_circle_ax_mid.c2p(0, 0),
                unit_circle_ax_mid.c2p(np.cos(~phase * -1), np.sin(~phase * -1)),
                color=target_2_color,
            ).set_z_index(1)
        )
        phase_dot_bot = always_redraw(
            lambda: Dot(
                unit_circle_ax_bot.c2p(np.cos(~phase * -1), np.sin(~phase * -1)),
                color=target_3_color,
            ).set_z_index(1)
        )
        phase_line_bot = always_redraw(
            lambda: Line(
                unit_circle_ax_bot.c2p(0, 0),
                unit_circle_ax_bot.c2p(np.cos(~phase * -1), np.sin(~phase * -1)),
                color=target_3_color,
            ).set_z_index(1)
        )

        self.add(
            phase_line_top,
            phase_dot_top,
            phase_line_mid,
            phase_dot_mid,
            phase_line_bot,
            phase_dot_bot,
            unit_circle_group.shift(LEFT * 5),
        )

        self.play(
            unit_circle_group.animate.shift(RIGHT * 5),
            cpi_axes.animate.shift(RIGHT * 2),
        )
        self.play(phase @ (4 * PI), run_time=4, rate_func=rate_functions.linear)

        self.wait(0.5)

        self.play(
            unit_circle_group.animate.shift(LEFT * 5),
            cpi_axes.animate.shift(LEFT * 2),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        cpi_windowed = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )

        fft_len = N * 8
        cpi_range_ffts = fft(cpi_windowed, n=fft_len, axis=1)
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        rmax = c * Tc * fs / (2 * bw)
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)

        def create_cpi_fft_n_plot(n, x0, x1):
            ax = Axes(
                x_range=[0, 40, 40 / 4],
                y_range=[-40, 0, 10],
                tips=False,
                axis_config={"include_numbers": False},
                x_length=x_len,
                y_length=y_len,
            ).set_opacity(0)
            X_k = fftshift(cpi_range_ffts[n])
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = 10 * np.log10(X_k)
            f_cpi_fft_n = interpolate.interp1d(ranges, X_k, fill_value="extrapolate")

            def updater():
                cpi_n_plot = ax.plot(
                    f_cpi_fft_n, x_range=[~x0, ~x1, 1 / 100], color=IF_COLOR
                )
                return cpi_n_plot

            return ax, updater

        cpi_fft_plot_groups = [
            create_cpi_fft_n_plot(n, x0, x1)
            for n, (x0, x1) in enumerate(zip(fft_x0, fft_x1))
        ]

        cpi_fft_axes = (
            Group(*[cpi_fft_n_plot[0] for cpi_fft_n_plot in cpi_fft_plot_groups])
            .arrange(UP, -MED_LARGE_BUFF * 1.2)
            .move_to(cpi_axes)
        )
        cpi_fft_plots = Group(
            *[
                always_redraw(cpi_fft_n_plot[1])
                for cpi_fft_n_plot in cpi_fft_plot_groups
            ]
        )

        self.add(cpi_fft_plots)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(time_series_x0_temp @ max_time, fft_x1_temp @ 40)
                    for time_series_x0_temp, fft_x1_temp in zip(time_series_x0, fft_x1)
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        range_doppler = 10 * np.log10(fftshift(np.abs(fft2(cpi_windowed.T))) / (N / 2))

        range_doppler_image = (
            ImageMobject(range_doppler)
            .scale_to_fit_height(cpi_fft_axes.height)
            .stretch_to_fit_width(cpi_fft_axes.width)
        )

        # self.play()
        self.add(range_doppler_image)

        self.wait(2)


class ImageTest(Scene):
    def construct(self):
        # import matplotlib.pyplot as plt

        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]
        max_time = 20 / 5337025
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts

        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )
        range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)

        rmax = c * Tc * fs / (2 * bw)
        n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        fft_len = N * 8
        ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)
        extent = [-max_vel, max_vel, ranges.min(), ranges.max()]

        img = (
            ImageMobject(10 * np.log10(range_doppler))
            .scale_to_fit_height(config.frame_height * 0.8)
            .stretch_to_fit_width(config.frame_height * 0.8)
        )
        self.add(img)

        # fig, ax = plt.subplots(figsize=(8, 8))
        # range_doppler_plot = ax.imshow(
        #     10 * np.log10(range_doppler),
        #     aspect="auto",
        #     extent=extent,
        #     origin="lower",
        #     # cmap=get_cmap(cmap_name),
        #     vmax=0,
        #     vmin=-10,
        # )
        # ax.set_ylim([0, 40])
        # ax.set_title("Range Doppler Spectrum", fontsize=24)
        # ax.set_xlabel("Velocity (m/s)", fontsize=22)
        # ax.set_ylabel("Range (m)", fontsize=22)
        # # ax.set_ylabel("Frequency (MHz)", fontsize=22)
        # fig.colorbar(range_doppler_plot)
        # plt.show()


class RangeDoppler3D(ThreeDScene):
    def construct(self):
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]

        sep = "\n\t"
        print(
            "Targets:",
            sep
            + sep.join(
                [
                    f"{idx}: distance = {r:.2f} m, velocity = {v:.2f} m/s"
                    for idx, (r, v) in enumerate(targets, start=1)
                ]
            ),
        )

        f_beat = compute_f_beat(20)
        max_time = 20 / f_beat
        N = 10000
        Ts = max_time / N
        fs = 1 / Ts

        t = np.arange(0, max_time, 1 / fs)
        window = signal.windows.blackman(N)
        cpi = np.array(
            [
                (
                    np.sum(
                        [
                            np.sin(
                                2 * PI * compute_f_beat(r) * t
                                + m * compute_phase_diff(v)
                            )
                            for r, v in targets
                        ],
                        axis=0,
                    )
                    + np.random.normal(0, 0.1, N)
                )
                * window
                for m in range(M)
            ]
        )

        range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)

        vels = np.linspace(-max_vel, max_vel, M)
        rmax = c * Tc * fs / (2 * bw)

        plot_vel = (-20, 20)  # m/s
        plot_range = (0, 40)  # m

        v_ind = np.where((vels > plot_vel[0]) & (vels < plot_vel[1]))[0]
        vx = vels[v_ind[0] : v_ind[-1]]

        n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
        r_ind = np.where((n_ranges > plot_range[0]) & (n_ranges < plot_range[1]))[0]
        ry = n_ranges[r_ind[0] : r_ind[-1]]

        rdz = range_doppler[r_ind[0] : r_ind[-1], v_ind[0] : v_ind[-1]]

        X, Y = np.meshgrid(vx, ry, indexing="xy")
        tck = bisplrep(X, Y, 10 * np.log10(rdz))

        axes = ThreeDAxes(
            x_range=[-20, 20, 20],
            y_range=[0, 40, 20],
            z_range=[-25, 10, 10],
            x_length=8,
        )
        res = 25
        u_min = VT(20)
        surface = always_redraw(
            lambda: Surface(
                lambda u, v: axes.c2p(u, v, bisplev(u, v, tck)),
                u_range=[~u_min, 20],
                v_range=[0, 40],
                resolution=(res, res),
            )
            .set_z(0)
            .set_style(fill_opacity=1)
            .set_fill_by_value(axes=axes, colorscale=[(BLUE, -20), (RED, 10)], axis=2)
        )
        # self.add(surface)

        self.set_camera_orientation(theta=0 * DEGREES, phi=0 * DEGREES, zoom=0.6)

        self.play(FadeIn(surface))

        self.wait(0.5)

        self.play(u_min @ -20, run_time=4)

        self.wait(0.5)

        self.move_camera(theta=-50 * DEGREES, phi=60 * DEGREES, zoom=0.6)

        self.wait(0.5)

        self.begin_ambient_camera_rotation(0.1)

        self.wait(20)
