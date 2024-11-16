# fmcw_doppler.py

import sys
import warnings

from random import shuffle
import numpy as np
from manim import *
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fftshift, fft2
from scipy import signal, interpolate
from scipy.constants import c


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR, IF_COLOR
from props.helpers import get_plot_values
from props import WeatherRadarTower, get_blocks

config.background_color = BACKGROUND_COLOR

BLOCKS = get_blocks()

SKIP_ANIMATIONS_OVERRIDE = True


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

        self.wait(0.5)

        self.play(x1_disp @ (duration / 2), x1_sub @ (duration / 2))

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

        self.next_section(skip_animations=skip_animations(False))
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

        self.wait(0.5)

        self.play(x0_disp @ (duration / 2 + ~beat_time_shift))
        self.play(x1_sub @ duration, x1_disp @ duration)

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

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(tx_signal),
                    FadeIn(tx_eqn, shift=DOWN),
                ),
                AnimationGroup(
                    Create(rx_signal),
                    FadeIn(rx_eqn, shift=RIGHT),
                ),
                Create(if_signal),
                lag_ratio=0.8,
            )
        )

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
            SVGMobject("../props/static/car.svg", fill_color=WHITE, stroke_color=WHITE)
            .flip()
            .scale_to_fit_width(config.frame_width * 0.2)
            .to_edge(RIGHT, LARGE_BUFF)
            .shift(UP * 2)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(car1.shift(LEFT * 5).animate.shift(RIGHT * 5))

        self.wait(0.5)

        self.play(car2.shift(RIGHT * 5).animate.shift(LEFT * 5))

        self.wait(0.5)

        car2_vel_arrow = Arrow(car2.get_right(), car2.get_left()).next_to(car2, DOWN)
        car2_vel_label = MathTex(r"v &= \text{60 mph}\\&= \text{26 m/s}").next_to(
            car2_vel_arrow, DOWN
        )

        self.play(GrowArrow(car2_vel_arrow), FadeIn(car2_vel_label[0][:7]))

        self.wait(0.5)

        self.play(FadeIn(car2_vel_label[0][7:]))

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
        r_2 = MathTex(r"R(t_0 + 40 \mu \text{s})").move_to(r_1_copy)

        f_beat_diff = MathTex(
            r"f_{beat}(t_0) - f_{beat}(t_0 + 40 \mu \text{s}) \approx \text{286 Hz}"
        ).move_to(f_beat_group)
        r_diff = MathTex(
            r"R(t_0) - R(t_0 + 40 \mu \text{s}) \approx \text{0.001 m}"
        ).move_to(Group(r_1, r_1_copy))

        self.play(
            f_beat_1.shift(DOWN * 5).animate.shift(UP * 5),
            r_1.shift(DOWN * 5).animate.shift(UP * 5),
        )

        self.wait(0.5)

        self.play(
            f_beat_1_copy.shift(DOWN * 5).animate.shift(UP * 5),
            r_1_copy.shift(DOWN * 5).animate.shift(UP * 5),
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                f_beat_1_copy,
                f_beat_2,
                ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
                ([], [8, 9, 10, 11, 12], {"delay": 0.3}),
                ([8], [13]),
            ),
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
                (GrowFromCenter, [17, 18, 19, 20, 21, 22], {"delay": 0.6}),
            )
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                f_beat_1,
                f_beat_diff,
                ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                (GrowFromCenter, [9], {"delay": 0.2}),
                (
                    get_transform_func(f_beat_2[0], ReplacementTransform),
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                    {"delay": 0.4},
                ),
                (GrowFromCenter, [24], {"delay": 0.4}),
                (GrowFromCenter, [25, 26, 27, 28, 29], {"delay": 0.6}),
            )
        )

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

        self.next_section(skip_animations=skip_animations(False))
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

        self.wait(0.5)

        phase_comp_4_copy = MathTex(
            r"2 \pi f \Delta \tau",
            font_size=DEFAULT_FONT_SIZE,
            color=YELLOW,
        ).shift(DOWN)

        self.play(
            Group(
                phase_comp_1_label,
                phase_comp_2_label,
                phase_comp_3_label,
                phase_comp_4_label,
                phase_eqn_delta,
            ).animate.to_edge(UP, LARGE_BUFF),
            TransformFromCopy(phase_comp_4, phase_comp_4_copy[0], path_arc=PI / 2),
        )
        self.play(phase_comp_4_copy.animate.scale(1.5))

        self.wait(0.5)

        self.play(
            phase_comp_4_copy[0][3:5]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(WHITE)
            .shift(UP / 2)
        )

        self.wait(0.5)

        phase_comp_4_eq = MathTex(
            r"2 \pi f \Delta \tau \approx 1.29 \text{ radians}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(phase_comp_4_copy)
        phase_comp_4_eq[0][0:5].set_color(YELLOW)

        self.play(
            TransformByGlyphMap(
                phase_comp_4_copy,
                phase_comp_4_eq,
                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                (GrowFromCenter, [5], {"delay": 0.2}),
                (
                    GrowFromCenter,
                    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    {"delay": 0.4},
                ),
            )
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

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(val)
                    for val in [phase_comp_1_val, phase_comp_2_val, phase_comp_3_val]
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        phase_eqn_approx = MathTex(
            r"\Delta \phi \approx -2 \pi f \Delta \tau",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        phase_eqn_approx[0][4:].set_color(YELLOW)

        self.play(
            LaggedStart(
                AnimationGroup(
                    phase_comp_4_eq.animate.shift(DOWN * 8),
                    Group(
                        phase_comp_1_label,
                        phase_comp_2_label,
                        phase_comp_3_label,
                        phase_comp_4_label,
                    ).animate.shift(UP * 8),
                    FadeOut(
                        Group(phase_comp_1_val, phase_comp_2_val, phase_comp_3_val)
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
        window = signal.windows.blackman(N)
        # cpi = np.array(
        #     [
        #         (
        #             np.sum(
        #                 [
        #                     np.sin(
        #                         2 * PI * compute_f_beat(r) * t
        #                         + m * compute_phase_diff(v)
        #                     )
        #                     for r, v in targets
        #                 ],
        #                 axis=0,
        #             )
        #             + np.random.normal(0, 0.1, N)
        #         )
        #         * window
        #         for m in range(M)
        #     ]
        # )

        phi_0_1 = -PI / 6
        phi_0_2 = phi_0_1 + PI * 0.6

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

        target_1_color = PURPLE
        target_2_color = GREEN
        targets_color = ORANGE

        A_1 = VT(1)
        A_2 = VT(1)

        target_1_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: ~A_1
                * (
                    np.sin(2 * PI * 3 * t + phi_0_1)
                    + (1 - ~A_2) * np.sin(2 * PI * 3 * t + phi_0_2)
                ),
                color=interpolate_color(target_1_color, targets_color, (1 - ~A_2)),
                x_range=[0, duration, 1 / 1000],
            )
        )

        target_2_plot = always_redraw(
            lambda: amp_ax.plot(
                lambda t: ~A_2 * np.sin(2 * PI * 3 * t + phi_0_2),
                color=target_2_color,
                x_range=[0, duration, 1 / 1000],
            )
        )
        plot_group = Group(amp_ax, target_1_plot, target_2_plot)

        eqn_1 = MathTex(
            r"\sin{\left(2 \pi f_{beat,1} t + \phi_1\right)}", color=target_1_color
        )
        eqn_2 = MathTex(
            r"\sin{\left(2 \pi f_{beat,2} t + \phi_2\right)}", color=target_2_color
        )
        eqn_group = Group(eqn_1, eqn_2).arrange(RIGHT, LARGE_BUFF)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(GrowFromCenter(eqn_1), GrowFromCenter(eqn_2), lag_ratio=0.3)
        )

        self.wait(0.5)

        self.add(target_1_plot, target_2_plot)
        self.play(Group(eqn_group, amp_ax).animate.arrange(DOWN, LARGE_BUFF * 2.5))

        self.wait(0.5)

        plus = MathTex("+").set_y(eqn_1.get_y())
        plus_group = (
            Group(eqn_1.copy(), plus, eqn_2.copy())
            .arrange(RIGHT, SMALL_BUFF)
            .set_y(eqn_1.get_y())
        )

        self.play(
            A_1 @ 0.5,
            A_2 @ 0,
            GrowFromCenter(plus),
            eqn_1.animate.move_to(plus_group[0]),
            eqn_2.animate.move_to(plus_group[2]),
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

        self.wait(0.5)

        ts_line = Line(
            sample_rects[7].get_top() + UP,
            [sample_rects[8].get_top()[0], (sample_rects[7].get_top() + UP)[1], 0],
        )
        ts_line_l = Line(ts_line.get_left() + DOWN / 8, ts_line.get_left() + UP / 8)
        ts_line_r = Line(ts_line.get_right() + DOWN / 8, ts_line.get_right() + UP / 8)
        ts_label = MathTex(r"T_s").next_to(ts_line, UP)

        self.play(
            LaggedStart(
                Create(ts_line_l),
                Create(ts_line),
                Create(ts_line_r),
                FadeIn(ts_label),
                lag_ratio=0.2,
            )
        )

        self.wait(2)


class RangeDoppler3D(ThreeDScene):
    def construct(self):
        targets = [
            (20, 10),  # Target 1 @ 20 m with a velocity of 10 m/s
            (20, -10),  # Target 2 @ 20 m with a velocity of -10 m/s
            (10, 5),  # Target 3 @ 10 m with a velocity of 5 m/s
        ]

        sep = "\n\t"
        print(
            f"Targets:",
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
