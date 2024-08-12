# fmcw_implementation.py

from manim import *
import numpy as np
from scipy import signal, constants
import math
import sys

sys.path.insert(0, "..")
from props import get_blocks, get_bd_animation


BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR


TX_COLOR = BLUE
RX_COLOR = RED
GAIN_COLOR = GREEN


BLOCKS = get_blocks()
BLOCK_BUFF = LARGE_BUFF * 2


BD_SCALE = 0.5
PLL_WIDTH = config["frame_width"] * 0.5


def get_splitter_ports(splitter):
    splitter_p1 = splitter.get_right() + (UP * splitter.height / 4)
    splitter_p2 = splitter.get_right() + (DOWN * splitter.height / 4)
    return splitter_p1, splitter_p2


def get_bd():
    input_circle = Circle(radius=0.2)
    input_label = Tex(r"$V_{tune}$ Input").next_to(
        input_circle, direction=UP, buff=SMALL_BUFF
    )
    inp = VGroup(input_circle, input_label)
    pll_block = (
        BLOCKS.get("oscillator")
        .copy()
        .next_to(input_circle, direction=RIGHT, buff=BLOCK_BUFF)
    )
    input_to_vco = Line(input_circle.get_right(), pll_block.get_left())
    pa = (
        BLOCKS.get("amp")
        .copy()
        .next_to(pll_block, direction=RIGHT, buff=BLOCK_BUFF)
        .set_fill(GAIN_COLOR)
    )
    pll_block_to_pa = Line(pll_block.get_right(), pa.get_left())
    splitter = BLOCKS.get("splitter").copy().next_to(pa, buff=BLOCK_BUFF)
    pa_to_splitter = Line(pa.get_right(), splitter.get_left())
    mixer = (
        BLOCKS.get("mixer")
        .copy()
        .next_to(splitter, direction=RIGHT, buff=BLOCK_BUFF / 2)
        .shift(DOWN * BLOCK_BUFF * 3)
    )

    splitter_p1 = splitter.get_right() + (UP * splitter.height / 4)
    splitter_p2 = splitter.get_right() + (DOWN * splitter.height / 4)
    mixer_lo = mixer.get_top()
    mixer_rf = mixer.get_right()
    mixer_if = mixer.get_left()

    splitter_to_mixer = CubicBezier(
        splitter_p2,
        splitter_p2 + [2, 0, 0],
        mixer_lo + [0, 2, 0],
        mixer_lo,
    )

    tx_antenna = (
        BLOCKS.get("antenna")
        .copy()
        .next_to(splitter, buff=BLOCK_BUFF * 4)
        .set_fill(TX_COLOR)
    )
    tx_antenna.shift(UP * (splitter_p1 - tx_antenna.get_bottom()))
    splitter_to_tx_antenna = Line(splitter_p1, tx_antenna.get_bottom())

    lna = (
        BLOCKS.get("amp")
        .copy()
        .rotate(PI)
        .next_to(mixer, buff=BLOCK_BUFF)
        .set_fill(GAIN_COLOR)
    )
    lna_to_mixer = Line(lna.get_left(), mixer_rf)

    rx_antenna = (
        BLOCKS.get("antenna")
        .copy()
        .set_x(tx_antenna.get_x())
        .set_y((lna.get_right() + (lna.height / 2 * UP))[1])
        .set_fill(RX_COLOR)
    )
    rx_antenna_to_lna = Line(rx_antenna.get_bottom(), lna.get_right())

    lp_filter = (
        BLOCKS.get("lp_filter").copy().next_to(mixer, direction=LEFT, buff=BLOCK_BUFF)
    )
    mixer_to_lp_filter = Line(mixer_if, lp_filter.get_right())

    adc = (
        BLOCKS.get("adc")
        .copy()
        .next_to(lp_filter, direction=LEFT, buff=BLOCK_BUFF)
        .flip()
    )
    lp_filter_to_adc = Line(lp_filter.get_left(), adc.get_right())

    signal_proc_label = Tex(r"Signal\\Processor")
    signal_proc_box = SurroundingRectangle(signal_proc_label, buff=MED_SMALL_BUFF)
    signal_proc = Group(signal_proc_box, signal_proc_label).next_to(
        adc, direction=LEFT, buff=BLOCK_BUFF
    )
    adc_to_signal_proc = Line(adc.get_left(), signal_proc.get_right())

    bd = (
        Group(
            inp,
            input_to_vco,
            pll_block,
            pll_block_to_pa,
            pa,
            pa_to_splitter,
            splitter,
            splitter_to_mixer,
            mixer,
            splitter_to_tx_antenna,
            tx_antenna,
            lna,
            lna_to_mixer,
            rx_antenna,
            rx_antenna_to_lna,
            mixer_to_lp_filter,
            lp_filter,
            lp_filter_to_adc,
            adc,
            adc_to_signal_proc,
            signal_proc,
        )
        .scale(BD_SCALE)
        .move_to(ORIGIN)
    )
    return bd, (
        inp,
        input_to_vco,
        pll_block,
        pll_block_to_pa,
        pa,
        pa_to_splitter,
        splitter,
        splitter_to_mixer,
        mixer,
        splitter_to_tx_antenna,
        tx_antenna,
        lna,
        lna_to_mixer,
        rx_antenna,
        rx_antenna_to_lna,
        mixer_to_lp_filter,
        lp_filter,
        lp_filter_to_adc,
        adc,
        adc_to_signal_proc,
        signal_proc,
    )


class BD(Scene):
    def construct(self):
        (
            bd,
            (
                inp,
                input_to_vco,
                pll_block,
                pll_block_to_pa,
                pa,
                pa_to_splitter,
                splitter,
                splitter_to_mixer,
                mixer,
                splitter_to_tx_antenna,
                tx_antenna,
                lna,
                lna_to_mixer,
                rx_antenna,
                rx_antenna_to_lna,
                mixer_to_lp_filter,
                lp_filter,
                lp_filter_to_adc,
                adc,
                adc_to_signal_proc,
                signal_proc,
            ),
        ) = get_bd()

        tx_section = Group(inp, pll_block, splitter, tx_antenna)
        rx_section = Group(adc, lp_filter, mixer, lna, rx_antenna, signal_proc)

        tx_section_box = SurroundingRectangle(
            tx_section, buff=MED_SMALL_BUFF, color=TX_COLOR
        )
        tx_section_box_label = Tex("Transmit").next_to(
            tx_section_box, direction=UP, buff=SMALL_BUFF
        )
        tx_section_box_label.shift(
            LEFT * (tx_section_box.width / 2 - tx_section_box_label.width / 2)
        )
        rx_section_box = SurroundingRectangle(
            rx_section, buff=MED_SMALL_BUFF, color=RX_COLOR
        )
        rx_section_box_label = Tex("Receive").next_to(
            rx_section_box, direction=UP, buff=SMALL_BUFF
        )
        rx_section_box_label.shift(
            LEFT * (rx_section_box.width / 2 - rx_section_box_label.width / 2)
        )

        self.play(get_bd_animation(bd, lagged=True, lag_ratio=0.5), run_time=5)
        self.play(
            Create(rx_section_box),
            Create(tx_section_box),
            Create(rx_section_box_label),
            Create(tx_section_box_label),
        )

        self.wait(1)

        # You'll see slight variations depending on the designer or use case, but the main blocks roughly stay the same.

        splitter_p1, splitter_p2 = get_splitter_ports(splitter)

        tx_bp_filter = (
            BLOCKS.get("bp_filter_generic")
            .copy()
            .scale(BD_SCALE)
            .next_to(tx_antenna.get_bottom(), direction=LEFT, buff=BLOCK_BUFF)
        )
        splitter_to_tx_bp_filter = Line(splitter_p1, tx_bp_filter.get_left())
        tx_bp_filter_to_tx_antenna = Line(
            tx_bp_filter.get_right(), tx_antenna.get_bottom()
        )

        def splitter_to_mixer_updater(m: Mobject):
            splitter_p1, splitter_p2 = get_splitter_ports(splitter)
            mixer_lo = mixer.get_top()

            splitter_to_mixer = CubicBezier(
                splitter_p2,
                splitter_p2 + [1, 0, 0],
                mixer_lo + [0, 1, 0],
                mixer_lo,
            )
            m.become(splitter_to_mixer)

        # splitter_to_mixer.add_updater(splitter_to_mixer_updater)

        self.play(Uncreate(splitter_to_tx_antenna))
        self.play(
            LaggedStart(
                Create(splitter_to_tx_bp_filter),
                GrowFromCenter(tx_bp_filter),
                Create(tx_bp_filter_to_tx_antenna),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        splitter_to_tx_antenna = Line(splitter_p1, tx_antenna.get_bottom())
        self.play(
            LaggedStart(
                ShrinkToCenter(tx_bp_filter),
                Create(splitter_to_tx_antenna),
                lag_ratio=0.6,
            )
        )
        self.remove(splitter_to_tx_bp_filter, tx_bp_filter_to_tx_antenna)

        self.wait(0.5)

        self.play(
            Uncreate(tx_section_box),
            Uncreate(rx_section_box),
            Uncreate(tx_section_box_label),
            Uncreate(rx_section_box_label),
        )

        self.wait(2)


class BlockSections(Scene):
    def construct(self):
        (
            bd,
            (
                inp,
                input_to_vco,
                pll_block,
                pll_block_to_pa,
                pa,
                pa_to_splitter,
                splitter,
                splitter_to_mixer,
                mixer,
                splitter_to_tx_antenna,
                tx_antenna,
                lna,
                lna_to_mixer,
                rx_antenna,
                rx_antenna_to_lna,
                mixer_to_lp_filter,
                lp_filter,
                lp_filter_to_adc,
                adc,
                adc_to_signal_proc,
                signal_proc,
            ),
        ) = get_bd()

        self.add(bd)

        phase_detector = BLOCKS.get("phase_detector").copy()
        to_phase_detector = Line(
            phase_detector.get_left() + (LEFT * BLOCK_BUFF),
            phase_detector.get_left(),
        )
        loop_filter = BLOCKS.get("amp").copy().next_to(phase_detector, buff=BLOCK_BUFF)
        phase_detector_to_loop_filter = Line(
            phase_detector.get_right(), loop_filter.get_left()
        )
        vco = BLOCKS.get("oscillator").copy().next_to(loop_filter, buff=BLOCK_BUFF)
        loop_filter_to_vco = Line(loop_filter.get_right(), vco.get_left())
        from_vco = Line(
            vco.get_right() + (RIGHT * BLOCK_BUFF),
            vco.get_right(),
        )
        n_div_label = Tex(r"$\frac{1}{N}$")
        n_div_box = SurroundingRectangle(n_div_label, buff=MED_SMALL_BUFF)
        ndiv = (
            VGroup(n_div_label, n_div_box)
            .next_to(loop_filter, direction=DOWN, buff=BLOCK_BUFF)
            .scale(1 / BD_SCALE)
        )
        vco_output_conn = Dot(from_vco.get_midpoint(), radius=DEFAULT_DOT_RADIUS * 2)
        vco_to_ndiv_1 = Line(
            vco_output_conn.get_center(),
            [vco_output_conn.get_center()[0], ndiv.get_right()[1], 0],
        )
        vco_to_ndiv_2 = Line(
            [vco_output_conn.get_center()[0], ndiv.get_right()[1], 0],
            ndiv.get_right(),
        )
        vco_to_ndiv = VGroup(vco_to_ndiv_1, vco_to_ndiv_2)

        ndiv_to_phase_detector_1 = Line(
            ndiv.get_left(), [phase_detector.get_bottom()[0], ndiv.get_left()[1], 0]
        )
        ndiv_to_phase_detector_2 = Line(
            [phase_detector.get_bottom()[0], ndiv.get_left()[1], 0],
            phase_detector.get_bottom(),
        )
        ndiv_to_phase_detector = VGroup(
            ndiv_to_phase_detector_1, ndiv_to_phase_detector_2
        )

        self.play(bd.animate.shift(DOWN * 2))

        label_buff = SMALL_BUFF
        label_scale = 0.6
        pll_label = (
            Tex(r"Phase-locked\\Loop")
            .next_to(pll_block, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        pa_label = (
            Tex(r"Power\\Amplifier")
            .next_to(pa, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        lna_label = (
            Tex(r"Low Noise\\Amplifier")
            .next_to(lna, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        splitter_label = (
            Tex("Splitter")
            .next_to(splitter, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        mixer_label = (
            Tex("Mixer")
            .next_to(mixer, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        adc_label = (
            Tex(r"Analog to\\Digital\\Converter")
            .next_to(adc, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        lp_filter_label = (
            Tex(r"Low pass\\filter")
            .next_to(lp_filter, direction=DOWN, buff=label_buff)
            .scale(label_scale)
            .set_opacity(0)
        )
        bd.add(
            pll_label,
            pa_label,
            lna_label,
            splitter_label,
            mixer_label,
            adc_label,
            lp_filter_label,
        )

        pll = (
            VGroup(
                to_phase_detector,
                phase_detector,
                phase_detector_to_loop_filter,
                loop_filter,
                loop_filter_to_vco,
                vco,
                from_vco,
                vco_output_conn,
                vco_to_ndiv_1,
                vco_to_ndiv_2,
                ndiv,
                ndiv_to_phase_detector_1,
                ndiv_to_phase_detector_2,
            )
            .scale(BD_SCALE)
            .next_to(pll_block, direction=UP, buff=LARGE_BUFF)
            .shift(RIGHT * 2)
        )

        pll_box = DashedVMobject(SurroundingRectangle(pll, color=DARK_GRAY))
        pll_box_bound_l = Line(
            pll_block.get_corner(UL), pll_box.get_corner(DL), color=DARK_GRAY
        )
        pll_box_bound_r = Line(
            pll_block.get_corner(UR), pll_box.get_corner(DR), color=DARK_GRAY
        )

        self.play(
            pll_label.animate.set_opacity(1),
            Create(pll_box),
            Create(pll_box_bound_l),
            Create(pll_box_bound_r),
            get_bd_animation(pll, lagged=True, lag_ratio=0.6, run_time=2),
        )

        self.wait(0.5)

        bd_w_pll = Group(bd, pll, pll_box, pll_box_bound_l, pll_box_bound_r)

        self.play(
            bd_w_pll.animate.scale_to_fit_height(config["frame_height"] - 0.5).move_to(
                ORIGIN
            )
        )

        self.wait(0.5)

        self.play(
            Indicate(pa),
            pa_label.animate.set_opacity(1),
            Indicate(lna),
            lna_label.animate.set_opacity(1),
        )

        self.wait(0.5)

        self.play(Indicate(lp_filter), lp_filter_label.animate.set_opacity(1))

        self.wait(0.5)

        self.play(Indicate(splitter), splitter_label.animate.set_opacity(1))

        self.wait(0.5)

        self.play(Indicate(mixer), mixer_label.animate.set_opacity(1))

        self.wait(0.5)

        self.play(Indicate(adc), adc_label.animate.set_opacity(1))

        self.wait(0.5)

        self.play(
            Indicate(tx_antenna),
            Indicate(rx_antenna),
        )

        self.wait(1)

        self.play(bd_w_pll.animate.shift(-pll.get_center()))
        self.play(FadeOut(bd, pll_box_bound_l, pll_box_bound_r, pll_box))
        self.play(pll.animate.scale_to_fit_width(PLL_WIDTH))

        self.wait(2)


class PLL(MovingCameraScene):
    def construct(self):
        phase_detector = BLOCKS.get("phase_detector").copy()
        to_phase_detector = Line(
            phase_detector.get_left() + (LEFT * BLOCK_BUFF),
            phase_detector.get_left(),
        )
        loop_filter = BLOCKS.get("amp").copy().next_to(phase_detector, buff=BLOCK_BUFF)
        phase_detector_to_loop_filter = Line(
            phase_detector.get_right(), loop_filter.get_left()
        )
        vco = BLOCKS.get("oscillator").copy().next_to(loop_filter, buff=BLOCK_BUFF)
        loop_filter_to_vco = Line(loop_filter.get_right(), vco.get_left())
        from_vco = Line(
            vco.get_right() + (RIGHT * BLOCK_BUFF),
            vco.get_right(),
        )
        n_div_label = Tex(r"$\frac{1}{N}$")
        n_div_box = SurroundingRectangle(n_div_label, buff=MED_SMALL_BUFF)
        ndiv = (
            VGroup(n_div_label, n_div_box)
            .next_to(loop_filter, direction=DOWN, buff=BLOCK_BUFF)
            .scale(1 / BD_SCALE)
        )
        vco_output_conn = Dot(from_vco.get_midpoint(), radius=DEFAULT_DOT_RADIUS * 2)
        vco_to_ndiv_0 = Line(
            vco.get_right(),
            vco_output_conn.get_center(),
        )
        vco_to_ndiv_1 = Line(
            vco_output_conn.get_center(),
            [vco_output_conn.get_center()[0], ndiv.get_right()[1], 0],
        )
        vco_to_ndiv_2 = Line(
            [vco_output_conn.get_center()[0], ndiv.get_right()[1], 0],
            ndiv.get_right(),
        )
        vco_to_ndiv = VGroup(vco_to_ndiv_0, vco_to_ndiv_1, vco_to_ndiv_2)

        ndiv_to_phase_detector_1 = Line(
            ndiv.get_left(), [phase_detector.get_bottom()[0], ndiv.get_left()[1], 0]
        )
        ndiv_to_phase_detector_2 = Line(
            [phase_detector.get_bottom()[0], ndiv.get_left()[1], 0],
            phase_detector.get_bottom(),
        )
        ndiv_to_phase_detector = VGroup(
            ndiv_to_phase_detector_1, ndiv_to_phase_detector_2
        )

        pll = (
            VGroup(
                to_phase_detector,
                phase_detector,
                phase_detector_to_loop_filter,
                loop_filter,
                loop_filter_to_vco,
                vco,
                from_vco,
                vco_output_conn,
                vco_to_ndiv_0,
                vco_to_ndiv_1,
                vco_to_ndiv_2,
                ndiv,
                ndiv_to_phase_detector_1,
                ndiv_to_phase_detector_2,
            )
            .scale(BD_SCALE)
            .move_to(ORIGIN)
            .scale_to_fit_width(PLL_WIDTH)
        )

        phase_detector_to_loop_filter_copy = (
            phase_detector_to_loop_filter.copy().set_color(YELLOW)
        )
        loop_filter_to_vco_copy = loop_filter_to_vco.copy().set_color(YELLOW)
        from_vco_copy = from_vco.copy().set_color(YELLOW)
        vco_to_ndiv_0_copy = vco_to_ndiv_0.copy().set_color(YELLOW)
        vco_to_ndiv_1_copy = vco_to_ndiv_1.copy().set_color(YELLOW)
        vco_to_ndiv_2_copy = vco_to_ndiv_2.copy().set_color(YELLOW)
        ndiv_to_phase_detector_1_copy = ndiv_to_phase_detector_1.copy().set_color(
            YELLOW
        )
        ndiv_to_phase_detector_2_copy = ndiv_to_phase_detector_2.copy().set_color(
            YELLOW
        )

        label_scale = 0.5
        pfd_label = Tex(r"Phase Frequency\\Detector (PFD)").scale(label_scale)
        loop_filter_label = Tex(r"Loop Filter").scale(label_scale)
        vco_label = Tex(r"Voltage-Controlled\\Oscillator (VCO)").scale(label_scale)
        ndiv_label = Tex(r"$1/N$ Frequency\\Divider").scale(label_scale)

        """ Modulated signals"""
        x0_reveal_tracker_1 = ValueTracker(0.0)
        x1_reveal_tracker_1 = ValueTracker(0.0)
        x0_reveal_tracker_2 = ValueTracker(0.0)
        x1_reveal_tracker_2 = ValueTracker(0.0)

        carrier_freq = 10  # Carrier frequency in Hz
        modulation_freq = 0.5  # Modulation frequency in Hz
        modulation_index = 20  # Modulation index
        duration = 1
        fs = 1000

        x_len = 6
        y_len = 2.2

        amp_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        f_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
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
        f_labels = f_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        def get_x_reveal_updater(
            ax, func, x0_updater, x1_updater, clip_sq=False, color=WHITE
        ):
            def updater(m: Mobject):
                x1 = (
                    min(x1_updater.get_value(), duration - 1 / fs)
                    if clip_sq
                    else x1_updater.get_value()
                )
                m.become(
                    ax.plot(
                        func,
                        x_range=[x0_updater.get_value(), x1, 1 / fs],
                        use_smoothing=False,
                        color=color,
                    )
                )

            return updater

        """ Triangular """
        triangular_modulating_signal = lambda t: modulation_index * np.arcsin(
            np.sin(2 * np.pi * modulation_freq * t)
        )
        triangular_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(triangular_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        triangular_amp = lambda t: np.sin(2 * np.pi * triangular_modulating_cumsum(t))

        triangular_f_graph = f_ax.plot(
            triangular_modulating_signal,
            x_range=[
                x0_reveal_tracker_1.get_value(),
                x1_reveal_tracker_1.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        triangular_amp_graph = amp_ax.plot(
            triangular_amp,
            x_range=[
                x0_reveal_tracker_1.get_value(),
                x1_reveal_tracker_1.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        triangular_f_graph_updater = get_x_reveal_updater(
            f_ax,
            triangular_modulating_signal,
            x0_reveal_tracker_1,
            x1_reveal_tracker_1,
            color=TX_COLOR,
        )
        triangular_amp_graph_updater = get_x_reveal_updater(
            amp_ax,
            triangular_amp,
            x0_reveal_tracker_1,
            x1_reveal_tracker_1,
            color=TX_COLOR,
        )
        triangular_f_graph.add_updater(triangular_f_graph_updater)
        triangular_amp_graph.add_updater(triangular_amp_graph_updater)

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

        sawtooth_f_graph = f_ax.plot(
            sawtooth_modulating_signal,
            x_range=[
                x0_reveal_tracker_2.get_value(),
                x1_reveal_tracker_2.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        sawtooth_amp_graph = amp_ax.plot(
            sawtooth_amp,
            x_range=[
                x0_reveal_tracker_2.get_value(),
                x1_reveal_tracker_2.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )

        sawtooth_f_graph_updater = get_x_reveal_updater(
            f_ax,
            sawtooth_modulating_signal,
            x0_reveal_tracker_2,
            x1_reveal_tracker_2,
            clip_sq=True,
            color=TX_COLOR,
        )
        sawtooth_amp_graph_updater = get_x_reveal_updater(
            amp_ax,
            sawtooth_amp,
            x0_reveal_tracker_2,
            x1_reveal_tracker_2,
            color=TX_COLOR,
        )
        sawtooth_f_graph.add_updater(sawtooth_f_graph_updater)
        sawtooth_amp_graph.add_updater(sawtooth_amp_graph_updater)

        amp_ax_group = VGroup(amp_ax, amp_labels, sawtooth_amp_graph)
        f_ax_group = VGroup(f_ax, f_labels, sawtooth_f_graph)

        both_graphs = (
            VGroup(f_ax_group, amp_ax_group)
            .arrange(direction=RIGHT, buff=MED_LARGE_BUFF)
            .to_edge(UP)
        )

        """ Progress bar """
        progress_bar = RoundedRectangle(
            height=0.5,
            width=config["frame_width"] * 0.8,
            corner_radius=PI / 16,
            stroke_color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        ).to_edge(DOWN)
        pb_copy = progress_bar.copy()
        progress_bar_box = Rectangle(width=config["frame_width"], height=3).move_to(
            progress_bar
        )
        pb_mover = (
            Rectangle(
                height=1,
                width=config["frame_width"],
                fill_color=RED,
                fill_opacity=1,
                stroke_opacity=0,
            )
            .shift(LEFT * config["frame_width"] * 0.7 + UP * progress_bar.get_y())
            .set_z_index(0)
        )
        c = Cutout(
            progress_bar_box,
            progress_bar,
            fill_opacity=1,
            color=BACKGROUND_COLOR,
        ).set_z_index(1)
        one_hour = Tex("1 hour").next_to(pb_copy, buff=SMALL_BUFF).set_z_index(2)
        empty_pb = VGroup(c, pb_copy, one_hour).shift(LEFT / 2)
        actual_time = (
            Tex("$<$10 mins")
            .next_to(one_hour, direction=UP, buff=SMALL_BUFF)
            .set_z_index(2)
        )
        one_hour_cross = Line(
            one_hour.get_corner(DL), one_hour.get_corner(UR)
        ).set_z_index(2)
        pb_group = VGroup(pb_mover, empty_pb).shift(DOWN * 3)

        adi_article_popup_shift = 6
        adi_article_popup = (
            ImageMobject("../props/static/adi_pll_tutorial.png")
            .scale(0.8)
            .to_edge(DOWN, buff=0)
            .shift(DOWN * (1 + adi_article_popup_shift))
        )
        in_the_desc_arrow = Arrow(
            adi_article_popup.get_corner(UR),
            adi_article_popup.get_corner(UR) + DOWN * 2,
        ).shift(RIGHT / 2)
        in_the_desc = (
            Tex("in the description")
            .rotate(-PI / 2)
            .next_to(in_the_desc_arrow, buff=SMALL_BUFF)
        )
        adi_article_group = Group(
            adi_article_popup,
            in_the_desc,
            in_the_desc_arrow,
            # pll,
        )

        """ Animations """

        self.add(pll)

        self.wait(0.5)

        self.play(pll.animate.to_edge(DOWN))
        self.play(
            Create(f_ax),
            Create(f_labels),
            Create(amp_ax),
            Create(amp_labels),
        )
        self.add(
            triangular_f_graph,
            triangular_amp_graph,
            sawtooth_f_graph,
            sawtooth_amp_graph,
        )

        self.play(x1_reveal_tracker_2.animate.set_value(duration), run_time=1)

        self.wait(0.5)

        self.play(
            LaggedStart(
                x0_reveal_tracker_2.animate.set_value(duration),
                x1_reveal_tracker_1.animate.set_value(duration),
                lag_ratio=0.5,
            ),
            run_time=1,
        )

        self.wait(0.5)

        self.play(x0_reveal_tracker_1.animate.set_value(duration), run_time=1)

        triangular_f_graph.remove_updater(triangular_f_graph_updater)
        triangular_amp_graph.remove_updater(triangular_amp_graph_updater)
        sawtooth_f_graph.remove_updater(sawtooth_f_graph_updater)
        sawtooth_amp_graph.remove_updater(sawtooth_amp_graph_updater)

        self.wait(1)

        self.play(
            FadeOut(
                both_graphs,
                triangular_f_graph,
                triangular_amp_graph,
                sawtooth_f_graph,
                sawtooth_amp_graph,
                shift=UP * 2,
            ),
            pll.animate.move_to(ORIGIN),
        )

        self.wait(1)

        self.play(pb_group.animate.shift(UP * 3))
        self.play(
            LaggedStart(
                pb_mover.animate(rate_func=rate_functions.linear, run_time=5).shift(
                    RIGHT * config["frame_width"] * 0.16
                ),
                AnimationGroup(Create(one_hour_cross), Create(actual_time)),
                lag_ratio=0.3,
            )
        )
        self.play(FadeOut(pb_group, one_hour_cross, actual_time, shift=DOWN * 4))

        self.wait(0.5)

        self.play(adi_article_group.animate.shift(UP * adi_article_popup_shift))
        self.wait(1.5)
        self.play(adi_article_group.animate.shift(DOWN * adi_article_popup_shift))

        self.wait(1)

        self.camera.frame.save_state()

        self.play(self.camera.frame.animate.scale(0.4).move_to(phase_detector))
        self.play(
            FadeIn(
                pfd_label.next_to(phase_detector, direction=UP, buff=SMALL_BUFF),
                shift=DOWN,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(loop_filter))
        self.play(
            FadeIn(
                loop_filter_label.next_to(loop_filter, direction=UP, buff=SMALL_BUFF),
                shift=DOWN,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(vco))
        self.play(
            FadeIn(vco_label.next_to(vco, direction=UP, buff=SMALL_BUFF), shift=DOWN)
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(ndiv))
        self.play(
            FadeIn(ndiv_label.next_to(ndiv, direction=DOWN, buff=SMALL_BUFF), shift=UP)
        )

        self.wait(0.5)

        self.play(Restore(self.camera.frame))

        # highlight_color = YELLOW

        # self.play(phase_detector.animate.set_color(highlight_color))

        # self.wait(0.5)

        # self.play(Create(phase_detector_to_loop_filter_copy))

        # self.wait(0.5)

        # self.play(loop_filter.animate.set_color(highlight_color))

        # self.wait(0.5)

        # self.play(Create(loop_filter_to_vco_copy))

        # self.wait(0.5)

        # self.play(vco.animate.set_color(highlight_color))

        # self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         Create(vco_to_ndiv_0_copy),
        #         vco_output_conn.animate.set_color(highlight_color),
        #         lag_ratio=0.5,
        #     )
        # )
        # self.play(Create(vco_to_ndiv_1_copy))
        # self.play(Create(vco_to_ndiv_2_copy))

        # self.wait(0.5)

        # self.play(ndiv.animate.set_color(highlight_color))

        # self.wait(0.5)

        # self.play(Create(ndiv_to_phase_detector_1_copy))
        # self.play(Create(ndiv_to_phase_detector_2_copy))

        # self.wait(0.5)

        self.wait(2)


""" Testing """


class ProgressBar(Scene):
    def construct(self):
        progress_bar = RoundedRectangle(
            height=0.5,
            width=config["frame_width"] * 0.8,
            corner_radius=PI / 16,
            stroke_color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        ).to_edge(DOWN)
        pb_copy = progress_bar.copy()
        progress_bar_box = Rectangle(width=config["frame_width"], height=3).move_to(
            progress_bar
        )
        moving_box = (
            Rectangle(
                height=3,
                width=config["frame_width"],
                fill_color=RED,
                fill_opacity=1,
                stroke_opacity=0,
            )
            .shift(LEFT * config["frame_width"] * 0.7 + UP * progress_bar.get_y())
            .set_z_index(0)
        )
        c = Cutout(
            progress_bar_box,
            progress_bar,
            fill_opacity=1,
            color=BACKGROUND_COLOR,
        ).set_z_index(1)
        self.add(c, pb_copy)
        self.add(moving_box)
        self.play(
            moving_box.animate.shift(RIGHT * config["frame_width"] * 0.1),
            rate_func=rate_functions.linear,
            run_time=4,
        )
