# fmcw_doppler.py

import sys
import warnings

from random import randint, randrange, shuffle
import numpy as np
from manim import *
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, interpolate


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

        def get_transform_func(from_var, func=TransformFromCopy):
            def transform_func(m, **kwargs):
                return func(from_var, m, **kwargs)

            return transform_func

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


# class HowToLearn(Scene):
#     def construct(self):
