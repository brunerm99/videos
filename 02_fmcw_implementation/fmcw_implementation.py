# fmcw_implementation.py

from manim import *
import numpy as np
from scipy import signal, constants
import math
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from props import get_blocks, get_bd_animation, get_resistor, get_diode


BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR


TX_COLOR = BLUE
RX_COLOR = RED
GAIN_COLOR = GREEN
IF_COLOR = ORANGE


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
        loop_filter = (
            BLOCKS.get("lp_filter").copy().next_to(phase_detector, buff=BLOCK_BUFF)
        )
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
        n_div_label_n2 = Tex(r"$\frac{1}{2}$")
        n_div_box = SurroundingRectangle(
            n_div_label, buff=MED_SMALL_BUFF, color=WHITE, fill_opacity=0
        )
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
            Tex(r"Signal\\Generator")
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

        pll_label_specific = Tex("Phase-locked Loop")

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
        self.play(
            LaggedStart(
                FadeOut(bd, pll_box_bound_l, pll_box_bound_r, pll_box),
                pll.animate.scale_to_fit_width(PLL_WIDTH),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(pll_label_specific.to_edge(UP, buff=LARGE_BUFF), shift=DOWN))

        self.wait(2)


class PLL(MovingCameraScene):
    def construct(self):
        phase_detector = BLOCKS.get("phase_detector").copy()
        to_phase_detector = Line(
            phase_detector.get_left() + (LEFT * BLOCK_BUFF),
            phase_detector.get_left(),
        )
        loop_filter = (
            BLOCKS.get("lp_filter").copy().next_to(phase_detector, buff=BLOCK_BUFF)
        )
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
        n_div_label_n2 = Tex(r"$\frac{1}{2}$")
        n_div_box = SurroundingRectangle(
            n_div_label, buff=MED_SMALL_BUFF, color=WHITE, fill_opacity=0
        )
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

        label_scale = 0.5
        pfd_label = Tex(r"Phase Frequency\\Detector (PFD)").scale(label_scale)
        loop_filter_label = Tex(r"Loop Filter").scale(label_scale)
        vco_label = Tex(r"Voltage-Controlled\\Oscillator (VCO)").scale(label_scale)
        ndiv_label_below = Tex(r"$1/N$ Frequency\\Divider").scale(label_scale)

        lo = BLOCKS.get("oscillator").copy()
        lo_label = Tex(r"Local Oscillator\\(LO)").scale(label_scale)

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

        self.play(
            self.camera.frame.animate.scale(1.2).move_to(
                VGroup(ndiv, vco_to_ndiv_2, ndiv_to_phase_detector_1)
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(1 / 1.2).move_to(ndiv))
        self.play(
            FadeIn(
                ndiv_label_below.next_to(ndiv, direction=DOWN, buff=SMALL_BUFF),
                shift=UP,
            )
        )

        self.wait(0.5)

        self.play(Restore(self.camera.frame))

        self.wait(0.5)

        self.play(
            VGroup(
                ndiv_to_phase_detector,
                n_div_label,
                vco_to_ndiv,
                phase_detector_to_loop_filter,
                loop_filter,
                loop_filter_to_vco,
                vco,
                vco_output_conn,
                from_vco,
            ).animate.set_opacity(0.2),
            n_div_box.animate.set_stroke(color=WHITE, opacity=0.2).set_fill(opacity=0),
        )

        lo.scale_to_fit_width(vco.width).next_to(
            to_phase_detector, direction=LEFT, buff=0
        )
        lo_label.next_to(lo, direction=UP, buff=SMALL_BUFF)
        self.play(GrowFromCenter(lo), FadeIn(lo_label, shift=DOWN))

        self.wait(0.5)

        A = 1
        f_lo = 2
        fs = 1000
        step = 1 / fs
        x_len = 5
        y_len = 2

        lo_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1.5, 1.5, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        lo_f_label = (
            Tex(r"$f_{LO}=1$ MHz")
            .next_to(lo_ax, direction=UP, buff=MED_SMALL_BUFF)
            .shift(RIGHT)
        )

        lo_labels = lo_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )

        self.camera.frame.save_state()

        pll.add(lo, lo_label, pfd_label, loop_filter_label, vco_label, ndiv_label_below)

        lo_ax_group = VGroup(lo_ax, lo_labels, lo_f_label).to_corner(UL).shift(UL)

        lo_signal = lo_ax.plot(
            lambda t: A * np.sin(2 * PI * f_lo * t),
            x_range=[0, 1, step],
        )

        self.play(
            self.camera.frame.animate.scale(1.2),
            pll.animate.to_edge(RIGHT).shift(RIGHT),
        )

        lo_f_plot_p1 = to_phase_detector.get_midpoint() + [-0.2, 0.1, 0]
        lo_f_plot_p1_handle = lo_f_plot_p1 + [0.5, 2, 0]
        lo_f_plot_p2 = lo_ax_group.get_right() + [0.5, 0, 0]
        lo_f_plot_p2_handle = lo_f_plot_p2 + [1, 0, 0]

        lo_f_bezier = CubicBezier(
            lo_f_plot_p1,
            lo_f_plot_p1_handle,
            lo_f_plot_p2_handle,
            lo_f_plot_p2,
        )
        lo_f_bezier_arrowhead = (
            Triangle(fill_opacity=1, fill_color=WHITE, stroke_color=WHITE)
            .scale(0.2)
            .rotate(-PI / 6)
        ).move_to(lo_f_bezier.get_end())
        lo_f_bezier_group = VGroup(lo_f_bezier, lo_f_bezier_arrowhead)

        self.play(
            LaggedStart(
                AnimationGroup(
                    lo_label.animate.shift(LEFT / 6),
                    pfd_label.animate.shift(RIGHT / 6),
                ),
                Create(lo_f_bezier_group),
                Create(lo_ax_group),
                Create(lo_signal),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        pfd_output_label_xshift = 0.3
        pfd_output_label_all = Tex("$I$", r"$(f_{diff})$").next_to(
            phase_detector_to_loop_filter.get_midpoint() + [0, 0, 0],
            direction=UP,
            buff=LARGE_BUFF * 2.5,
        )
        pfd_output_label = Tex("$I$").move_to(pfd_output_label_all)
        pfd_output_label_p1 = phase_detector_to_loop_filter.get_midpoint() + [0, 0.1, 0]
        pfd_output_label_p1_handle = pfd_output_label_p1 + [0, 0.5, 0]
        pfd_output_label_p2 = pfd_output_label.get_bottom() + [0, -0.5, 0]
        pfd_output_label_p2_handle = pfd_output_label_p1 + [
            pfd_output_label_xshift,
            0.7,
            0,
        ]

        pfd_output_label_bezier = CubicBezier(
            pfd_output_label_p1,
            pfd_output_label_p1_handle,
            pfd_output_label_p2_handle,
            pfd_output_label_p2,
        )
        pfd_output_label_bezier_arrowhead = (
            Triangle(fill_opacity=1, fill_color=WHITE, stroke_color=WHITE)
            .scale(0.2)
            .move_to(pfd_output_label_bezier.get_end())
            .rotate(10 * DEGREES)
        )
        pfd_output_label_bezier_group = VGroup(
            pfd_output_label_bezier, pfd_output_label_bezier_arrowhead
        )

        self.play(
            LaggedStart(
                phase_detector_to_loop_filter.animate.set_opacity(1),
                Create(pfd_output_label_bezier_group),
                Create(pfd_output_label),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(Transform(pfd_output_label, pfd_output_label_all))

        self.wait(0.5)

        pfd_piecewise_top = Tex(r"$I > 0 \ , \ f_{LO} > f_{\text{feedback}}$")
        pfd_piecewise_bot = Tex(r"$I < 0 \ , \ f_{LO} < f_{\text{feedback}}$").next_to(
            pfd_piecewise_top, direction=DOWN, buff=MED_SMALL_BUFF
        )
        pfd_piecewise_brace = Brace(
            Line(pfd_piecewise_top.get_corner(UL), pfd_piecewise_bot.get_corner(DL)),
            direction=LEFT,
        )
        pfd_piecewise_eq = Tex("$=$").next_to(
            pfd_piecewise_brace, direction=LEFT, buff=MED_SMALL_BUFF
        )
        pfd_piecewise = (
            VGroup(
                pfd_piecewise_top,
                pfd_piecewise_bot,
                pfd_piecewise_brace,
                pfd_piecewise_eq,
            )
            .scale(0.7)
            .next_to(pfd_output_label, direction=RIGHT, buff=MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                FadeIn(pfd_piecewise, shift=LEFT),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(Indicate(pfd_piecewise_top))

        self.wait(0.5)

        self.play(Indicate(pfd_piecewise_bot))

        self.wait(0.5)

        f_vco_tracker = ValueTracker(f_lo / 2)
        ndiv_val_tracker = ValueTracker(2)
        f_fb_tracker = ValueTracker(
            f_vco_tracker.get_value() / ndiv_val_tracker.get_value()
        )

        fb_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1.5, 1.5, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        fb_f_label = (
            Tex(r"No signal")
            .next_to(fb_ax, direction=UP, buff=MED_SMALL_BUFF)
            .shift(RIGHT)
        )
        fb_f_label_w_val = (
            Tex(
                r"$f_{\text{feedback}}=\ $",
                f"{int(1000 * f_fb_tracker.get_value() / f_lo)} kHz",
            )
            .next_to(fb_ax, direction=UP, buff=MED_SMALL_BUFF)
            .shift(RIGHT)
        )

        fb_labels = fb_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )

        fb_ax_group = VGroup(fb_ax, fb_f_label, fb_labels).next_to(
            lo_ax_group, direction=DOWN, aligned_edge=LEFT
        )

        fb_signal = fb_ax.plot(
            lambda t: A
            * np.sin(
                2 * PI * (f_vco_tracker.get_value() / ndiv_val_tracker.get_value()) * t
            ),
            x_range=[0, 1, step],
        )

        fb_f_plot_p1 = ndiv_to_phase_detector_2.get_midpoint() + [-0.1, 0, 0]
        fb_f_plot_p1_handle = fb_f_plot_p1 + [-1, -0.5, 0]
        fb_f_plot_p2 = fb_ax_group.get_right() + [0.5, 0, 0]
        fb_f_plot_p2_handle = fb_f_plot_p2 + [1, 0, 0]

        fb_f_bezier = CubicBezier(
            fb_f_plot_p1,
            fb_f_plot_p1_handle,
            fb_f_plot_p2_handle,
            fb_f_plot_p2,
        )
        fb_f_bezier_arrowhead = (
            Triangle(fill_opacity=1, fill_color=WHITE, stroke_color=WHITE)
            .scale(0.2)
            .rotate(-PI / 6)
        ).move_to(fb_f_bezier.get_end())
        fb_f_bezier_group = VGroup(fb_f_bezier, fb_f_bezier_arrowhead)

        self.play(
            LaggedStart(
                Create(fb_f_bezier_group),
                AnimationGroup(Create(fb_ax), Create(fb_labels), FadeIn(fb_f_label)),
                lag_ratio=0.4,
                run_time=2,
            )
        )

        self.wait(0.5)

        pfd_piecewise_top.set_color(GREEN)
        self.play(Indicate(pfd_piecewise_top, color=GREEN))

        self.wait(0.5)

        self.play(
            LaggedStart(
                loop_filter.animate.set_opacity(1),
                loop_filter_to_vco.animate.set_opacity(1),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        pfd_out_f = 0.5
        pfd_out_length = 1 / pfd_out_f
        _tracker = ValueTracker(0)
        pfd_out = always_redraw(
            lambda: FunctionGraph(
                lambda t: ((signal.square(2 * PI * pfd_out_f * t - PI / 2) + 1) / 2)
                + np.random.normal(0, 0.05, 1)[0],
                x_range=[
                    0,
                    pfd_out_length - step,
                    step,
                ],
                use_smoothing=False,
            ).shift(LEFT * 4 + DOWN * 3.5)
        )
        pfd_out_label = (
            Tex(r"PFD Output\\(Dirty)").scale(0.6).next_to(pfd_out, direction=UP)
        )

        loop_filter_phase_tracker = ValueTracker(0)  # 0 -> PI
        loop_filter_out = FunctionGraph(
            lambda t: (
                (
                    signal.square(
                        2 * PI * pfd_out_f * t
                        - PI / 2
                        + loop_filter_phase_tracker.get_value(),
                        duty=0.5,
                    )
                    + 1
                )
                / 2
            )
            * signal.square(
                2 * PI * pfd_out_f / 2 * t + loop_filter_phase_tracker.get_value() / 2,
                duty=0.5,
            ),
            x_range=[0, pfd_out_length - step, step],
            use_smoothing=False,
        ).shift(DOWN * 3.5)
        loop_filter_out_label = (
            Tex(r"Loop Filter\\Output (Clean)")
            .scale(0.6)
            .next_to(loop_filter_out, direction=UP)
        )
        pfd_out_to_loop_filter_out = Arrow(
            pfd_out.get_right(), loop_filter_out.get_left()
        )

        vco_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1.5, 1.5, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        vco_f_label = (
            Tex(
                r"$f_{VCO}=\ $", f"{int(1000 * (f_vco_tracker.get_value() / f_lo))} kHz"
            )
            .next_to(vco_ax, direction=UP, buff=MED_SMALL_BUFF)
            .shift(RIGHT)
        )

        vco_labels = vco_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )

        vco_ax_group = VGroup(vco_ax, vco_f_label, vco_labels).next_to(
            fb_ax_group, direction=DOWN, aligned_edge=LEFT
        )

        vco_signal = vco_ax.plot(
            lambda t: A * np.sin(2 * PI * f_vco_tracker.get_value() * t),
            x_range=[0, 1, step],
        )

        self.play(Create(pfd_out), FadeIn(pfd_out_label))
        self.play(
            _tracker.animate(
                run_time=3, rate_func=rate_functions.linear
            ).increment_value(1)
        )
        self.play(
            _tracker.animate(
                run_time=6, rate_func=rate_functions.linear
            ).increment_value(1),
            LaggedStart(
                GrowArrow(pfd_out_to_loop_filter_out),
                AnimationGroup(Create(loop_filter_out), FadeIn(loop_filter_out_label)),
                lag_ratio=0.5,
            ),
        )

        self.play(
            FadeOut(
                pfd_out,
                pfd_out_label,
                pfd_out_to_loop_filter_out,
            ),
        )
        loop_filter_shifted_copy = loop_filter.copy().next_to(
            loop_filter_out_label.copy().next_to(
                vco_ax_group, direction=RIGHT, buff=MED_SMALL_BUFF
            ),
            direction=DOWN,
            buff=SMALL_BUFF,
        )
        loop_filter_shift = loop_filter_shifted_copy.get_corner(DL)
        loop_filter_base_shift = UP
        self.play(
            loop_filter_out_label.animate.next_to(
                vco_ax_group, direction=RIGHT, buff=MED_SMALL_BUFF
            ).shift(loop_filter_base_shift),
            loop_filter_out.animate.shift(
                -(
                    loop_filter_out.get_corner(DL)
                    - (loop_filter_shift + loop_filter_base_shift)
                )
            ),
            Create(vco_ax),
            Create(vco_labels),
            FadeIn(vco_f_label),
            Create(vco_signal),
        )

        self.wait(0.5)

        self.play(vco.animate.set_opacity(1), from_vco.animate.set_opacity(1))

        self.wait(0.5)

        def loop_filter_out_updater(m: Mobject):
            m.become(
                FunctionGraph(
                    lambda t: (
                        (
                            signal.square(
                                2 * PI * pfd_out_f * t
                                - PI / 2
                                + loop_filter_phase_tracker.get_value(),
                                duty=0.5,
                            )
                            + 1
                        )
                        / 2
                    )
                    * signal.square(
                        2 * PI * pfd_out_f / 2 * t
                        + loop_filter_phase_tracker.get_value() / 2,
                        duty=0.5,
                    ),
                    x_range=[0, pfd_out_length - step, step],
                    use_smoothing=False,
                ).shift(loop_filter_shift + loop_filter_base_shift)
            )

        def vco_signal_updater(m: Mobject):
            m.become(
                vco_ax.plot(
                    lambda t: A * np.sin(2 * PI * f_vco_tracker.get_value() * t),
                    x_range=[0, 1, step],
                )
            )

        def fb_signal_updater(m: Mobject):
            m.become(
                fb_ax.plot(
                    lambda t: A * np.sin(2 * PI * f_fb_tracker.get_value() * t),
                    x_range=[0, 1, step],
                )
            )

        def vco_f_label_updater(m: Mobject):
            m.become(
                Tex(
                    r"$f_{VCO}=\ $",
                    f"{int(1000 * (f_vco_tracker.get_value() / f_lo))} kHz",
                )
                .next_to(vco_ax, direction=UP, buff=MED_SMALL_BUFF)
                .shift(RIGHT)
            )

        def fb_f_label_updater(m: Mobject):
            m.become(
                Tex(
                    r"$f_{\text{feedback}}=\ $",
                    f"{int(1000 * f_fb_tracker.get_value() / f_lo)} kHz",
                )
                .next_to(fb_ax, direction=UP, buff=MED_SMALL_BUFF)
                .shift(RIGHT)
            )

        loop_filter_out.add_updater(loop_filter_out_updater)
        vco_f_label.add_updater(vco_f_label_updater)
        vco_signal.add_updater(vco_signal_updater)

        self.play(
            loop_filter_phase_tracker.animate.increment_value(2 * PI),
            f_vco_tracker.animate.increment_value(-0.5),
        )

        self.wait(0.5)

        self.play(
            loop_filter_phase_tracker.animate.increment_value(2 * PI),
            f_vco_tracker.animate.increment_value(0.5),
        )

        self.wait(0.5)

        self.play(
            f_vco_tracker.animate(
                run_time=3, rate_func=rate_functions.ease_in_circ
            ).increment_value(6)
        )

        self.wait(0.5)

        self.play(f_vco_tracker.animate(run_time=1).increment_value(-6))

        self.wait(0.5)

        self.play(
            # ndiv.animate.set_opacity(1),
            ndiv_to_phase_detector.animate.set_opacity(1),
            n_div_label.animate.set_opacity(1),
            n_div_box.animate.set_stroke(color=WHITE, opacity=1).set_fill(opacity=0),
            vco_to_ndiv.animate.set_opacity(1),
            vco_output_conn.animate.set_opacity(1),
        )

        self.wait(0.5)

        self.play(Transform(n_div_label, n_div_label_n2.move_to(n_div_label)))

        self.wait(0.5)

        self.play(
            Transform(fb_f_label, fb_f_label_w_val.move_to(fb_f_label)),
            Create(fb_signal),
        )
        fb_signal.add_updater(fb_signal_updater)
        fb_f_label.add_updater(fb_f_label_updater)

        self.wait(0.5)

        self.play(
            f_vco_tracker.animate.increment_value(1),
            f_fb_tracker.animate.increment_value(1 / ndiv_val_tracker.get_value()),
        )
        self.play(
            f_vco_tracker.animate.increment_value(-1),
            f_fb_tracker.animate.increment_value(-1 / ndiv_val_tracker.get_value()),
        )

        self.wait(0.5)

        self.play(Indicate(lo_signal))

        self.wait(0.5)

        fb_signal.remove_updater(fb_signal_updater)
        self.play(Indicate(fb_signal))
        fb_signal.add_updater(fb_signal_updater)

        self.wait(0.5)

        loop_count_start = 1
        n_loops = 10
        loop_f_step = (f_lo - f_fb_tracker.get_value()) / n_loops

        loop_count_tracker = ValueTracker(loop_count_start)
        loop_counter = always_redraw(
            lambda: Tex(
                "Loop count: ", f"{int(loop_count_tracker.get_value())}"
            ).to_corner(DR, buff=LARGE_BUFF)
        )
        loop_counter_box = SurroundingRectangle(
            Tex("Loop count: 10").move_to(loop_counter),
            color=YELLOW,
            buff=SMALL_BUFF * 1.2,
        )

        self.play(FadeIn(loop_counter), Create(loop_counter_box))
        for loop_count in range(loop_count_start, n_loops + 1, 1):
            loop_count_tracker.set_value(loop_count)
            self.play(
                Indicate(loop_counter[1]),
                f_vco_tracker.animate.increment_value(
                    loop_f_step * ndiv_val_tracker.get_value()
                ),
                f_fb_tracker.animate.increment_value(loop_f_step),
                run_time=0.5,
            )

        self.wait(0.5)

        locked_label = Tex("LOCKED", color=GREEN).next_to(
            loop_counter, direction=DOWN, buff=SMALL_BUFF
        )
        self.play(
            LaggedStart(
                Transform(
                    loop_counter_box,
                    SurroundingRectangle(VGroup(loop_counter, locked_label)),
                ),
                FadeIn(locked_label),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        n_loops = 10
        current_loop = int(loop_count_tracker.get_value())
        step_direction = -1

        for loop_count in range(current_loop, current_loop + n_loops + 1, 1):
            loop_count_tracker.set_value(loop_count)
            self.play(
                Indicate(loop_counter[1]),
                f_vco_tracker.animate.increment_value(
                    step_direction * loop_f_step * ndiv_val_tracker.get_value()
                ),
                f_fb_tracker.animate.increment_value(step_direction * loop_f_step),
                loop_filter_phase_tracker.animate.increment_value(2 * PI),
                run_time=0.5,
            )
            step_direction *= -1

        self.wait(1)

        loop_filter_out.remove_updater(loop_filter_out_updater)
        vco_f_label.remove_updater(vco_f_label_updater)
        vco_signal.remove_updater(vco_signal_updater)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(vco_signal),
                    Uncreate(lo_signal),
                    Uncreate(fb_signal),
                    Uncreate(loop_filter_out),
                    Uncreate(loop_counter_box),
                    Uncreate(lo_f_bezier_group),
                    Uncreate(fb_f_bezier_group),
                    Uncreate(lo_ax_group),
                    Uncreate(fb_ax_group),
                    Uncreate(vco_ax_group),
                    FadeOut(
                        loop_counter,
                        loop_filter_out_label,
                        pfd_piecewise,
                        pfd_output_label,
                        pfd_output_label_bezier_group,
                        locked_label,
                    ),
                ),
                pll.animate.move_to(ORIGIN).to_edge(UP, buff=MED_SMALL_BUFF),
                lag_ratio=0.7,
            )
        )

        self.wait(2)


class MixerIntro(MovingCameraScene):
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

        tx_section = Group(
            inp,
            input_to_vco,
            pll_block,
            pll_block_to_pa,
            pa,
            pa_to_splitter,
            splitter,
            splitter_to_tx_antenna,
            tx_antenna,
        )
        tx_section_wo_splitter = Group(
            inp,
            input_to_vco,
            pll_block,
            pll_block_to_pa,
            pa,
            pa_to_splitter,
            tx_antenna,
        )
        tx_section_before_splitter = Group(
            inp,
            input_to_vco,
            pll_block,
            pll_block_to_pa,
            pa,
            pa_to_splitter,
        )
        rx_section = Group(
            splitter_to_mixer,
            mixer,
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

        carrier_freq = 10  # Carrier frequency in Hz
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        duration = 1
        fs = 1000
        A = 0.5
        A_amped = 1

        x_len = 4
        y_len = 1.2

        amp_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        amp_labels = amp_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("", font_size=DEFAULT_FONT_SIZE),
        )

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        sawtooth_amp = lambda t: A * np.sin(2 * PI * sawtooth_modulating_cumsum(t))
        sawtooth_amp_amped = lambda t: A_amped * np.sin(
            2 * PI * sawtooth_modulating_cumsum(t)
        )

        sawtooth_amp_graph = amp_ax.plot(
            sawtooth_amp,
            x_range=[0, 1, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )
        sawtooth_amp_amped_graph = amp_ax.plot(
            sawtooth_amp_amped,
            x_range=[0, 1, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )

        fm_plot_group = VGroup(amp_ax, amp_labels, sawtooth_amp_graph).to_corner(
            UL, buff=SMALL_BUFF
        )

        self.add(bd)

        self.play(bd.animate.to_edge(DOWN))

        fm_plot_pll_p1 = pll_block_to_pa.get_midpoint() + [0, 0.1, 0]
        fm_plot_pll_p2 = fm_plot_group.get_bottom() + [0.1, 0.4, 0]

        fm_plot_bezier_pll = CubicBezier(
            fm_plot_pll_p1,
            fm_plot_pll_p1 + [0, 1, 0],
            fm_plot_pll_p2 + [0.5, -1, 0],
            fm_plot_pll_p2,
        )

        self.play(*[Create(x) for x in fm_plot_group], Create(fm_plot_bezier_pll))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(fm_plot_bezier_pll),
                fm_plot_group.animate.set_x(pa_to_splitter.get_end()[0]),
                lag_ratio=0.3,
            )
        )

        fm_plot_pa_p1 = pa_to_splitter.get_midpoint() + [0, 0.1, 0]
        fm_plot_pa_p2 = fm_plot_group.get_bottom() + [0.1, 0.4, 0]

        fm_plot_bezier_pa = CubicBezier(
            fm_plot_pa_p1,
            fm_plot_pa_p1 + [0, 1, 0],
            fm_plot_pa_p2 + [0.5, -1.5, 0],
            fm_plot_pa_p2,
        )

        sawtooth_amp_graph.save_state()

        self.play(
            LaggedStart(
                Create(fm_plot_bezier_pa),
                Transform(
                    sawtooth_amp_graph,
                    sawtooth_amp_amped_graph.move_to(sawtooth_amp_graph),
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        amp_ax_rx = amp_ax.copy()
        amp_labels_rx = amp_labels.copy()
        sawtooth_amp_graph_rx = sawtooth_amp_graph.copy().set_color(RX_COLOR)
        fm_plot_group_rx = (
            VGroup(amp_ax_rx, amp_labels_rx, sawtooth_amp_graph_rx)
            .move_to(ORIGIN)
            .to_edge(RIGHT, buff=MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                Uncreate(fm_plot_bezier_pa),
                bd.animate.move_to(ORIGIN).shift(LEFT),
                fm_plot_group.animate.next_to(
                    splitter_to_tx_antenna.get_midpoint(), direction=UP
                )
                .to_edge(UP, buff=MED_SMALL_BUFF)
                .shift(LEFT),
                TransformFromCopy(fm_plot_group, fm_plot_group_rx),
                # AnimationGroup(Create(amp_ax_rx), Create(amp_labels_rx)),
                # Create(sawtooth_amp_graph_rx),
                lag_ratio=0.5,
            ),
        )

        fm_plot_tx_p1 = splitter_to_tx_antenna.get_midpoint() + [0, 0.1, 0]
        fm_plot_tx_p2 = fm_plot_group.get_bottom() + [0.1, 0.4, 0]

        fm_plot_tx_bezier = CubicBezier(
            fm_plot_tx_p1,
            fm_plot_tx_p1 + [0, 0.5, 0],
            fm_plot_tx_p2 + [0.5, -0.5, 0],
            fm_plot_tx_p2,
        )

        fm_plot_rx_p1 = splitter_to_mixer.get_midpoint() + [0.1, 0, 0]
        fm_plot_rx_p2 = fm_plot_group_rx.get_left() + [-0.1, 0, 0]

        fm_plot_rx_bezier = CubicBezier(
            fm_plot_rx_p1,
            fm_plot_rx_p1 + [0.5, 0.5, 0],
            fm_plot_rx_p2 + [-0.5, -0.5, 0],
            fm_plot_rx_p2,
        )

        self.play(
            Create(fm_plot_tx_bezier),
            Create(fm_plot_rx_bezier),
            sawtooth_amp_graph.animate.stretch(0.5, dim=1),
            sawtooth_amp_graph_rx.animate.stretch(0.5, dim=1),
        )

        bd.save_state()

        self.play(
            ShrinkToCenter(splitter),
            tx_section_before_splitter.animate.set_y(splitter_to_tx_antenna.get_y()),
            Uncreate(splitter_to_mixer),
        )
        pa_to_tx_antenna = Line(pa.get_right(), splitter_to_tx_antenna.get_left())
        self.play(Transform(pa_to_splitter, pa_to_tx_antenna))

        mixer_lo = mixer.get_top()
        coupler_up = Line(mixer_lo, [mixer_lo[0], pa_to_tx_antenna.get_y() - 0.3, 0])
        coupler_left = Line(coupler_up.get_end(), coupler_up.get_end() + LEFT * 1.5)

        fm_plot_rx_coupler_p1 = coupler_up.get_midpoint() + [0.1, 0, 0]
        fm_plot_rx_coupler_p2 = fm_plot_group_rx.get_left() + [-0.1, 0, 0]

        fm_plot_rx_coupler_bezier = CubicBezier(
            fm_plot_rx_coupler_p1,
            fm_plot_rx_coupler_p1 + [0.5, 0.5, 0],
            fm_plot_rx_coupler_p2 + [-0.5, -0.5, 0],
            fm_plot_rx_coupler_p2,
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(coupler_up),
                    Transform(fm_plot_rx_bezier, fm_plot_rx_coupler_bezier),
                ),
                Create(coupler_left),
                AnimationGroup(
                    sawtooth_amp_graph.animate.stretch((1 / 0.5) * 0.8, dim=1),
                    sawtooth_amp_graph_rx.animate.stretch((1 / 0.5) * 0.2, dim=1),
                ),
                lag_ratio=0.9,
            )
        )

        self.wait(0.5)

        A_propagation = 0.5
        pw = 2

        tx_x_tracker = ValueTracker(0)

        tx_propagation_ax = Axes(
            x_range=[0, duration, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=6,
            y_length=2.2,
        )

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )
        sawtooth_amp = lambda t: A_propagation * np.sin(
            2 * PI * sawtooth_modulating_cumsum(t)
        )

        rotation = PI / 6
        rotation_line = Line(DOWN, UP).rotate(rotation)

        tx_propagation_graph = always_redraw(
            lambda: tx_propagation_ax.plot(
                sawtooth_amp,
                x_range=[0, tx_x_tracker.get_value(), 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            ).flip(rotation_line.get_end() - rotation_line.get_start())
        )

        tx_propagation_group = always_redraw(
            lambda: VGroup(tx_propagation_ax, tx_propagation_graph)
            .rotate(rotation)
            .next_to(tx_antenna.get_corner(UR))
            .shift(UP * tx_antenna.height + LEFT / 2)
        )

        self.add(tx_propagation_graph)

        self.play(
            tx_x_tracker.animate(rate_func=rate_functions.linear, run_time=4).set_value(
                2
            )
        )
        self.play(FadeOut(tx_propagation_graph, run_time=0.5))
        self.remove(tx_propagation_group)

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(sawtooth_amp_graph),
                    Uncreate(sawtooth_amp_graph_rx),
                    FadeOut(amp_labels),
                    FadeOut(amp_labels_rx),
                ),
                AnimationGroup(
                    Uncreate(amp_ax),
                    Uncreate(amp_ax_rx),
                    Uncreate(fm_plot_tx_bezier),
                    Uncreate(fm_plot_rx_bezier),
                ),
                lag_ratio=0.3,
            )
        )
        self.play(
            tx_antenna.animate.set_opacity(0.2),
            pa_to_splitter.animate.set_opacity(0.2),
            pa.animate.set_opacity(0.2),
            inp.animate.set_opacity(0.2),
            input_to_vco.animate.set_opacity(0.2),
            pll_block.animate.set_opacity(0.2),
            pll_block_to_pa.animate.set_opacity(0.2),
            splitter_to_tx_antenna.animate.set_opacity(0.2),
        )
        bd.add(coupler_left, coupler_up)
        self.play(bd.animate.to_edge(UP, buff=LARGE_BUFF))

        self.wait(0.5)

        coupler_up_shifted = coupler_up.copy().shift(UP / 4 + RIGHT / 2)
        reference_tx_arrow = Arrow(
            coupler_up_shifted.get_top() + UP / 2,
            coupler_up_shifted.get_bottom(),
            color=TX_COLOR,
        )
        reference_tx_label = Tex("Tx", color=TX_COLOR).next_to(reference_tx_arrow)

        rra_r = rx_antenna.get_corner(DR) + DOWN
        reference_rx_arrow = Arrow(
            rra_r, [lna_to_mixer.get_end()[0], rra_r[1], 0], color=RX_COLOR
        )
        reference_rx_label = Tex("Rx", color=RX_COLOR).next_to(
            reference_rx_arrow, direction=DOWN
        )

        self.play(GrowArrow(reference_tx_arrow), FadeIn(reference_tx_label))

        self.wait(0.5)

        self.play(GrowArrow(reference_rx_arrow), FadeIn(reference_rx_label))

        self.wait(0.5)

        f_beat = MathTex(r"f_{beat}").next_to(reference_tx_label, buff=MED_LARGE_BUFF)
        qmark = Tex("?", color=YELLOW).next_to(f_beat, buff=SMALL_BUFF)
        f_beat_eqn = MathTex(r"f_{beat}", r"= f_{RX} - f_{TX}").next_to(
            reference_tx_label, buff=MED_LARGE_BUFF
        )

        self.play(FadeIn(f_beat, qmark))

        self.wait(0.5)

        range_eqn = MathTex(r"R = \frac{c T_{c} f_{beat}}{2 B}").to_edge(
            LEFT, buff=LARGE_BUFF
        )
        range_eqn_label = (
            Tex("From last episode:")
            .scale(0.7)
            .next_to(range_eqn, direction=UP, buff=SMALL_BUFF, aligned_edge=LEFT)
        )
        range_eqn_box = SurroundingRectangle(
            VGroup(range_eqn, range_eqn_label), color=ORANGE
        )
        range_eqn_group = VGroup(range_eqn, range_eqn_label, range_eqn_box)
        self.play(FadeIn(range_eqn_group, shift=RIGHT))

        self.wait(0.5)

        self.play(Transform(qmark, f_beat_eqn[1]))

        self.wait(0.5)

        self.play(FadeOut(range_eqn_group, shift=LEFT))

        self.wait(1)

        self.camera.frame.save_state()

        camera_zoom = 0.001
        self.play(self.camera.frame.animate.move_to(mixer))
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(camera_zoom),
                FadeOut(mixer),
                lag_ratio=0.5,
            )
        )

        part_2 = (
            Tex("Part 2: Mixing")
            .move_to(self.camera.frame.get_center())
            .scale(camera_zoom * 2)
        )

        self.play(Create(part_2), run_time=2)

        # self.wait(1)

        # self.remove(f_beat_eqn, f_beat, qmark, range_eqn_group)

        # self.play(
        #     LaggedStart(
        #         ShrinkToCenter(part_2),
        #         AnimationGroup(Restore(self.camera.frame), FadeIn(mixer)),
        #         lag_ratio=0.4,
        #     )
        # )

        self.wait(2)


class Mixer(MovingCameraScene):
    def construct(self):
        part_2 = Tex("Part 2: Mixing").scale(2)

        mixer_circ = Circle(color=WHITE)
        mixer_line_1 = Line(
            mixer_circ.get_bottom(), mixer_circ.get_top(), color=WHITE
        ).rotate(PI / 4)
        mixer_line_2 = mixer_line_1.copy().rotate(PI / 2)
        mixer_x = VGroup(mixer_line_1, mixer_line_2)
        mixer = VGroup(mixer_circ, mixer_x)

        mixer_x_yellow = mixer_x.copy().set_color(YELLOW)

        used_to_subtract = Tex(r"used to\\subtract").next_to(
            mixer, direction=DL, buff=LARGE_BUFF
        )

        uts_p1 = used_to_subtract.get_right() + [0.1, 0, 0]
        uts_p2 = mixer.get_bottom() + [0, -0.1, 0]

        uts_bezier = CubicBezier(
            uts_p1,
            uts_p1 + [1, 0, 0],
            uts_p2 + [0, -1, 0],
            uts_p2,
        )

        multiply_to_subtract = (
            Tex("We multiply to subtract?").scale(1.5).to_edge(UP, buff=LARGE_BUFF)
        )

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
        rx_ax = (
            Axes(x_range=x_range[:2], y_range=[-2, 2], x_length=x_len, y_length=y_len)
            .rotate(PI)
            .next_to(mixer, direction=RIGHT, buff=0)
        )
        if_ax = (
            Axes(
                x_range=x_range[:2],
                y_range=[-2, 2],
                x_length=x_len,
                y_length=y_len,
                tips=False,
                axis_config={"include_numbers": False},
            )
            .rotate(PI)
            .next_to(mixer, direction=LEFT, buff=0)
        )
        if_sub_ax = if_ax.copy()
        if_add_ax = if_ax.copy()

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

        if_sub_signal = if_sub_ax.plot(
            lambda t: A * np.cos(2 * PI * (f_tx - f_rx) * t),
            x_range=x_range,
            color=IF_COLOR,
        )
        if_add_signal = if_add_ax.plot(
            lambda t: A * np.cos(2 * PI * (f_tx + f_rx) * t),
            x_range=x_range,
            color=IF_COLOR,
        )

        if_ax_group = VGroup(if_ax, if_signal)
        if_sub_ax_group = VGroup(if_sub_ax, if_sub_signal)
        if_add_ax_group = VGroup(if_add_ax, if_add_signal)

        f_tx_label = MathTex(r"f_{TX}", r"(t)", color=TX_COLOR).next_to(
            tx_signal, buff=MED_SMALL_BUFF
        )
        f_tx_at_t0_label = Tex(
            r"$f_{TX}$", r"$(t_{0})$", f"$ = {f_tx}$ GHz", color=TX_COLOR
        ).next_to(tx_signal, buff=MED_SMALL_BUFF)
        f_rx_label = (
            MathTex(r"f_{TX}", r"(t)", color=RX_COLOR).next_to(
                rx_signal, direction=UP, buff=MED_SMALL_BUFF, aligned_edge=LEFT
            )
            # .shift(RIGHT * 1.5)
        )
        f_rx_w_shift_label = (
            Tex(
                r"$f_{TX}$",
                r"$(t_{0} - t_{shift})$",
                f"$ = {f_rx}$ GHz",
                color=RX_COLOR,
            ).next_to(rx_signal, direction=UP, buff=MED_SMALL_BUFF, aligned_edge=LEFT)
            # .shift(RIGHT * 1.5)
        )

        lo_arrow = (
            Arrow(tx_signal.get_end(), tx_signal.get_end() + DOWN * 3, color=TX_COLOR)
            .next_to(tx_signal, direction=LEFT)
            .shift(DOWN / 2)
        )
        rf_arrow = Arrow(
            rx_signal.get_start(), rx_signal.get_end(), color=RX_COLOR
        ).next_to(rx_signal, direction=DOWN)
        if_arrow = Arrow(
            if_signal.get_start(), if_signal.get_end(), color=IF_COLOR
        ).next_to(if_signal, direction=DOWN)

        lo_port = (
            Tex("LO")
            .scale(0.6)
            .next_to(mixer.get_top(), direction=DOWN, buff=SMALL_BUFF)
        )
        rf_port = (
            Tex("RF")
            .scale(0.6)
            .next_to(mixer.get_right(), direction=LEFT, buff=SMALL_BUFF)
        )
        if_port = (
            Tex("IF")
            .scale(0.6)
            .next_to(mixer.get_left(), direction=RIGHT, buff=SMALL_BUFF)
        )

        diode = (
            get_diode()
            .next_to(lo_arrow, direction=LEFT, buff=LARGE_BUFF * 2)
            .scale(0.5)
            .set_stroke(color=WHITE, width=DEFAULT_STROKE_WIDTH * 1.5)
            .rotate(PI / 6)
        )
        diode_p = (
            Tex("+", color=YELLOW, stroke_width=DEFAULT_STROKE_WIDTH * 2)
            .next_to(diode, direction=UP, buff=MED_SMALL_BUFF)
            .shift(LEFT / 2)
        )
        diode_n = (
            Tex("-", color=RED, stroke_width=DEFAULT_STROKE_WIDTH * 2)
            .next_to(diode, direction=DOWN, buff=MED_SMALL_BUFF)
            .shift(RIGHT / 2)
        )

        out_of_scope = Tex("Out of scope", color=BLACK)
        out_of_scope_box = SurroundingRectangle(
            out_of_scope, color=YELLOW, fill_color=YELLOW, fill_opacity=1
        )
        out_of_scope_group = (
            VGroup(out_of_scope_box, out_of_scope).move_to(diode).rotate(PI / 6)
        )

        article_rotation = 10 * DEGREES
        marki_mixer_video = (
            ImageMobject("../props/static/marki_mixer_talk.png")
            .shift(RIGHT)
            .rotate(-article_rotation)
            .scale(0.7)
        )
        rfmw_mixer_article = (
            ImageMobject("../props/static/rfmw_mixer_article.png")
            .rotate(article_rotation)
            .shift(LEFT)
        )

        article_shift = 7

        rfmw_mixer_article.to_edge(DOWN, buff=0).scale(0.7).shift(DOWN).shift(
            DOWN * article_shift
        )
        marki_mixer_video.move_to(rfmw_mixer_article).shift(RIGHT * 2)

        in_the_desc_arrow = Arrow(
            rfmw_mixer_article.get_corner(UR),
            rfmw_mixer_article.get_corner(UR) + DOWN * 2,
        ).next_to(marki_mixer_video, direction=RIGHT, buff=MED_LARGE_BUFF)
        in_the_desc = (
            Tex("in the description")
            .rotate(-PI / 2)
            .next_to(in_the_desc_arrow, buff=SMALL_BUFF)
        )

        articles = Group(
            rfmw_mixer_article, marki_mixer_video, in_the_desc, in_the_desc_arrow
        )

        """ Reminder """

        def get_t_shift_dist(t0: float, t1: float, graph, axes: Axes):
            return (
                axes.input_to_graph_point(t1, graph)
                - axes.input_to_graph_point(t0, graph)
            ) * RIGHT

        carrier_freq = 10
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        duration = 1
        fs = 1000
        A = 1

        x_len = 6
        y_len = 2.2

        f_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-2, 30, 5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_corner(DL)

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )
        sawtooth_amp = lambda t: A * np.sin(2 * PI * sawtooth_modulating_cumsum(t))
        sawtooth_f_tx_graph = f_ax.plot(
            sawtooth_modulating_signal,
            x_range=[0, 1 - 1 / fs, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )

        t_shift = 0.1
        t_shift_dist = get_t_shift_dist(0, t_shift, sawtooth_f_tx_graph, f_ax)
        sawtooth_f_rx_graph = (
            sawtooth_f_tx_graph.copy().set_color(RX_COLOR).shift(t_shift_dist)
        )

        t_now = 0.4
        f_tx_dot = Dot(f_ax.input_to_graph_point(t_now, sawtooth_f_tx_graph))
        f_rx_dot = Dot(
            f_ax.input_to_graph_point(t_now - t_shift, sawtooth_f_rx_graph)
        ).shift(t_shift_dist)

        f_tx_graph_label = (
            Tex("12 GHz", color=TX_COLOR).move_to(f_tx_dot).shift(UP * 1.5 + RIGHT)
        )
        f_rx_graph_label = (
            Tex("10 GHz", color=RX_COLOR).move_to(f_tx_dot).shift(UP + LEFT)
        )

        f_tx_dot_bezier = CubicBezier(
            f_tx_dot.get_center() + [0.1, 0.1, 0],
            f_tx_dot.get_center() + [0.5, 0.5, 0],
            f_tx_graph_label.get_bottom() + [0, -0.5, 0],
            f_tx_graph_label.get_bottom() + [0, -0.1, 0],
            color=TX_COLOR,
        )
        f_rx_dot_bezier = CubicBezier(
            f_rx_dot.get_center() + [-0.1, 0.1, 0],
            f_rx_dot.get_center() + [-0.5, 0.5, 0],
            f_rx_graph_label.get_bottom() + [0, -0.5, 0],
            f_rx_graph_label.get_bottom() + [0, -0.1, 0],
            color=RX_COLOR,
        )

        unrealistic = Tex(r"2 GHz difference\\unrealistic").next_to(
            f_tx_dot, direction=UP, buff=LARGE_BUFF * 2
        )
        unrealistic_to_f_tx = Arrow(
            unrealistic.get_bottom(), f_tx_graph_label.get_top()
        )
        unrealistic_to_f_rx = Arrow(
            unrealistic.get_bottom(), f_rx_graph_label.get_top()
        )

        """ /Reminder"""

        if_eqn_scale = 0.7
        sine_rx_label = (
            MathTex(r"\sin [ 2 \pi", r"f_{TX}(t_{0} - t_{shift})", r"t]")
            .scale(if_eqn_scale)
            .next_to(if_signal, direction=UP, buff=MED_SMALL_BUFF)
        )
        if_mult = (
            mixer_x_yellow.copy()
            .scale(0.25)
            .next_to(sine_rx_label, direction=UP, buff=MED_SMALL_BUFF)
        )
        sine_tx_label = (
            MathTex(r"\sin [ 2 \pi", r"f_{TX}(t_{0})", r"t]")
            .next_to(if_mult, direction=UP, buff=MED_SMALL_BUFF)
            .scale(if_eqn_scale)
        )
        sine_tx_label[1].set_color(TX_COLOR)
        sine_rx_label[1].set_color(RX_COLOR)
        if_eqn_right = VGroup(sine_rx_label, if_mult, sine_tx_label)
        if_eqn_brace = Brace(
            if_eqn_right,
            direction=LEFT,
            color=IF_COLOR,
            fill_color=IF_COLOR,
            fill_opacity=1,
        )
        if_eqn_right.remove(if_mult)
        if_eqn_right.add(mixer_x_yellow)
        if_label = (
            Tex(r"IF", r"$\ =\ $", color=IF_COLOR)
            .scale(if_eqn_scale)
            .next_to(if_eqn_brace, direction=LEFT, buff=SMALL_BUFF)
        )
        if_eqn_group = VGroup(if_label, if_eqn_brace, if_eqn_right)

        mult_sines = MathTex(
            r"\sin ( 2 \pi",
            r"f_1",
            r" t ) \times \sin ( 2 \pi ",
            r"f_2",
            r" t )",
            r"\ =\ ",
            r"\frac{1}{2}",
            r"[",
            r"\cos ( ",
            r"f_1",
            r" - ",
            r"f_2",
            r") 2 \pi t",
            r"\ -\ ",
            r"\cos ( ",
            r"f_1",
            r" + ",
            r"f_2",
            r") 2 \pi t",
            r"]",
        )
        mult_sines_sub = MathTex(
            r"\cos ( ",
            r"f_{TX}(t_0)",
            r" - ",
            r"f_{TX}(t_0 - t_{shift})",
            r") 2 \pi t",
        ).scale(0.6)
        mult_sines_sub[1].set_color(TX_COLOR)
        mult_sines_sub[3].set_color(RX_COLOR)

        mult_sines_add = MathTex(
            r"\cos ( ",
            r"f_{TX}(t_0)",
            r" + ",
            r"f_{TX}(t_0 - t_{shift})",
            r") 2 \pi t",
        ).scale(0.6)
        mult_sines_add[1].set_color(TX_COLOR)
        mult_sines_add[3].set_color(RX_COLOR)

        if_sub_f_label = Tex(f"{f_tx - f_rx} GHz").scale(0.7)
        if_add_f_label = Tex(f"{f_tx + f_rx} GHz").scale(0.7)

        if_sub_f_beat_label = MathTex(r"f_{beat}")

        mixer_2 = mixer.copy()
        lo_port_2 = lo_port.copy()
        rf_port_2 = rf_port.copy()
        if_port_2 = if_port.copy()
        if_filter = (
            BLOCKS.get("lp_filter")
            .next_to(mixer_2, direction=LEFT, buff=0)
            .shift(LEFT * if_signal.width)
        )
        mixer_2_to_if_filter = Arrow(
            mixer_2.get_left(), if_filter.get_right(), color=IF_COLOR
        ).next_to(mixer_2, direction=LEFT, buff=0)
        if_filter.next_to(mixer_2_to_if_filter, direction=LEFT, buff=0)

        mixer_2_bd = VGroup(
            mixer_2,
            lo_port_2,
            rf_port_2,
            if_port_2,
            if_filter,
            mixer_2_to_if_filter,
        ).scale(0.7)

        if_filter_signal_label = MathTex(r"\sin (2 \pi f_{beat} t)", color=IF_COLOR)

        f_beat_label = Tex(f"$f_{{beat}} = {f_tx - f_rx}$ GHz", color=IF_COLOR)

        f_conversion_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1.5, 1.5, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        """ Animations """

        self.add(part_2)

        self.wait(0.5)

        self.play(FadeOut(part_2, shift=UP * 5), FadeIn(mixer, shift=UP * 5))

        self.wait(0.5)

        self.play(GrowFromCenter(mixer_x_yellow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(used_to_subtract),
                Create(uts_bezier),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        self.play(Create(multiply_to_subtract))

        self.wait(0.5)

        self.play(
            FadeOut(multiply_to_subtract, shift=UP),
            FadeOut(used_to_subtract),
            Uncreate(uts_bezier),
        )

        self.wait(0.5)

        self.play(Create(tx_signal), GrowArrow(lo_arrow), FadeIn(f_tx_label))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Transform(f_tx_label[1], f_tx_at_t0_label[1]),
                Create(f_tx_at_t0_label[2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(lo_port))

        self.wait(0.5)

        self.play(
            LaggedStart(Create(diode), Create(diode_p), Create(diode_n), lag_ratio=0.9)
        )

        self.wait(0.5)

        self.play(GrowFromCenter(out_of_scope_group))

        self.wait(0.5)

        self.add(rfmw_mixer_article, marki_mixer_video)

        self.play(articles.animate.to_edge(DOWN, buff=0))

        self.wait(1)

        self.play(
            articles.animate.move_to(DOWN * 10),
            FadeOut(diode, diode_p, diode_n, out_of_scope_group, shift=UP * 2),
        )
        self.remove(articles)

        self.wait(0.5)

        self.play(Create(rx_signal), GrowArrow(rf_arrow), FadeIn(f_rx_label))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Transform(f_rx_label[1], f_rx_w_shift_label[1]),
                Create(f_rx_w_shift_label[2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(rf_port))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(f_ax),
                Create(sawtooth_f_tx_graph),
                Create(sawtooth_f_rx_graph),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(f_tx_dot),
                    Create(f_tx_dot_bezier),
                    FadeIn(f_tx_graph_label),
                    lag_ratio=0.4,
                ),
                LaggedStart(
                    Create(f_rx_dot),
                    Create(f_rx_dot_bezier),
                    FadeIn(f_rx_graph_label),
                    lag_ratio=0.4,
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(unrealistic, unrealistic_to_f_tx, unrealistic_to_f_rx))

        self.wait(1)

        self.play(
            FadeOut(
                f_ax,
                sawtooth_f_tx_graph,
                sawtooth_f_rx_graph,
                f_tx_dot,
                f_tx_dot_bezier,
                f_tx_graph_label,
                f_rx_dot,
                f_rx_dot_bezier,
                f_rx_graph_label,
                unrealistic,
                unrealistic_to_f_tx,
                unrealistic_to_f_rx,
                shift=DOWN * 2,
            ),
        )

        self.wait(0.5)

        self.play(FadeIn(if_label))
        self.play(
            LaggedStart(
                FadeIn(sine_tx_label[0]),
                TransformFromCopy(f_tx_label, sine_tx_label[1]),
                FadeIn(sine_tx_label[2]),
                lag_ratio=0.6,
            )
        )
        self.play(Transform(mixer_x_yellow, if_mult))
        self.play(
            LaggedStart(
                FadeIn(sine_rx_label[0]),
                TransformFromCopy(f_rx_label, sine_rx_label[1]),
                FadeIn(sine_rx_label[2]),
                lag_ratio=0.6,
            )
        )
        self.play(FadeIn(if_eqn_brace))

        self.wait(0.5)

        self.play(FadeIn(if_port))
        self.play(Create(if_signal), GrowArrow(if_arrow))

        self.wait(0.5)

        self.camera.frame.save_state()

        self.play(
            LaggedStart(
                FadeOut(
                    mixer,
                    tx_signal,
                    lo_arrow,
                    lo_port,
                    f_tx_label,
                    rx_signal,
                    rf_arrow,
                    rf_port,
                    f_rx_label,
                    if_port,
                    if_arrow,
                    f_tx_at_t0_label[2],
                    f_rx_w_shift_label[2],
                ),
                self.camera.frame.animate.move_to(if_signal),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        if_ax_group_cornered = (
            if_ax_group.copy()
            .rotate(PI)
            .next_to(
                self.camera.frame.get_corner(UL),
                direction=DR,
                buff=MED_LARGE_BUFF,
            )
            .shift(RIGHT)
        )
        self.play(
            LaggedStart(
                if_eqn_group.animate.next_to(
                    self.camera.frame.get_corner(UR), direction=DL, buff=MED_LARGE_BUFF
                ).shift(LEFT),
                Succession(
                    FadeIn(if_ax),
                    Transform(if_ax_group, if_ax_group_cornered, path_arc=-PI),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(Indicate(sine_tx_label))
        self.wait(0.2)
        self.play(Indicate(sine_rx_label))

        self.wait(0.5)

        self.play(Create(mult_sines.move_to(self.camera.frame.get_center()).scale(0.8)))

        self.wait(0.5)

        subtracted_box = SurroundingRectangle(mult_sines[8:13])
        summed_box = SurroundingRectangle(mult_sines[14:19])

        self.play(Create(summed_box))

        self.wait(0.5)

        self.play(Transform(summed_box, subtracted_box))

        self.wait(0.5)

        self.play(
            VGroup(
                mult_sines[1],
                mult_sines[9],
                mult_sines[15],
            ).animate.set_color(TX_COLOR)
        )

        self.wait(0.2)

        self.play(
            VGroup(
                mult_sines[3],
                mult_sines[11],
                mult_sines[17],
            ).animate.set_color(RX_COLOR)
        )

        self.wait(0.5)

        self.play(
            TransformFromCopy(
                if_label[0],
                if_label[0]
                .copy()
                .next_to(if_signal, direction=UP, buff=MED_SMALL_BUFF),
            )
        )

        self.wait(0.2)

        self.play(Uncreate(summed_box))
        self.play(
            FadeOut(mult_sines[:6]),
            mult_sines[6:].animate.next_to(
                self.camera.frame.get_edge_center(RIGHT),
                direction=LEFT,
                buff=SMALL_BUFF,
            ),
        )

        if_sub_ax_group.next_to(if_ax_group, direction=DOWN, buff=MED_LARGE_BUFF)
        if_add_ax_group.next_to(if_sub_ax_group, direction=DOWN, buff=MED_LARGE_BUFF)
        self.play(
            LaggedStart(
                TransformFromCopy(if_ax_group, if_add_ax_group.flip()),
                TransformFromCopy(if_ax_group, if_sub_ax_group.flip()),
                lag_ratio=0.4,
            ),
            LaggedStart(
                TransformFromCopy(
                    mult_sines[8:13],
                    mult_sines_sub.next_to(
                        if_sub_signal, direction=UP, buff=MED_SMALL_BUFF
                    ).shift(RIGHT),
                ),
                TransformFromCopy(
                    mult_sines[14:19],
                    mult_sines_add.next_to(
                        if_add_signal, direction=UP, buff=MED_SMALL_BUFF
                    ).shift(RIGHT),
                ),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        if_sub_f_label.next_to(if_sub_ax_group, direction=LEFT, buff=SMALL_BUFF)
        if_add_f_label.next_to(if_add_ax_group, direction=LEFT, buff=SMALL_BUFF)

        self.play(
            LaggedStart(
                FadeIn(if_sub_f_label),
                FadeIn(if_add_f_label),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        if_sub_box = SurroundingRectangle(
            VGroup(
                if_sub_f_label,
                if_sub_ax_group,
                mult_sines_sub,
            )
        )
        if_add_box = SurroundingRectangle(
            VGroup(
                if_add_f_label,
                if_add_ax_group,
                mult_sines_add,
            )
        )
        if_sub_f_beat_label.next_to(if_sub_box, direction=DR, buff=LARGE_BUFF)
        if_sub_f_beat_to_box_arrow = Arrow(
            if_sub_f_beat_label.get_corner(UL), if_sub_box.get_corner(DR) + UP / 2
        )

        self.play(
            LaggedStart(
                Create(if_sub_box),
                FadeIn(if_sub_f_beat_label),
                GrowArrow(if_sub_f_beat_to_box_arrow),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(if_sub_f_beat_label, if_sub_f_beat_to_box_arrow),
                Transform(if_sub_box, if_add_box),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        mixer_2_bd.next_to(
            self.camera.frame.get_corner(DR), direction=UL, buff=MED_LARGE_BUFF
        )
        self.play(
            LaggedStart(
                GrowFromCenter(VGroup(mixer_2, lo_port_2, rf_port_2, if_port_2)),
                GrowArrow(mixer_2_to_if_filter),
                GrowFromCenter(if_filter),
                lag_ratio=0.8,
            ),
        )

        self.wait(0.5)

        x_out_hf_1 = Line(
            if_add_box.get_corner(DL),
            if_add_box.get_corner(UR),
            color=RED,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        )
        x_out_hf_2 = Line(
            if_add_box.get_corner(UL),
            if_add_box.get_corner(DR),
            color=RED,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        )

        self.play(
            Create(x_out_hf_1),
            Create(x_out_hf_2),
            VGroup(if_add_ax_group, if_add_f_label, mult_sines_add).animate.set_opacity(
                0.2
            ),
        )

        self.wait(0.5)

        all_except_bd = Group(*self.mobjects).remove(
            mixer,
            mixer_2,
            mixer_2_to_if_filter,
            lo_port_2,
            rf_port_2,
            if_port_2,
            if_filter,
            if_signal,
        )
        self.play(FadeOut(all_except_bd))

        if_signal_original = if_signal.copy().next_to(mixer, direction=LEFT, buff=0)
        self.play(
            LaggedStart(
                AnimationGroup(
                    Restore(self.camera.frame),
                    Transform(mixer_2, mixer),
                    Transform(lo_port_2, lo_port),
                    Transform(rf_port_2, rf_port),
                    Transform(if_port_2, if_port),
                    if_filter.animate.next_to(
                        if_signal_original, direction=LEFT, buff=0
                    ),
                    Transform(if_signal, if_signal_original),
                    Transform(mixer_2_to_if_filter, if_arrow),
                ),
                AnimationGroup(
                    Create(rx_signal),
                    Create(rf_arrow),
                    Create(tx_signal),
                    Create(lo_arrow),
                    FadeIn(f_tx_at_t0_label, f_rx_w_shift_label),
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        bd_scale = 0.6
        current_bd = Group(
            if_filter,
            mixer_2,
            lo_port_2,
            rf_port_2,
            if_port_2,
            if_signal,
            mixer_2_to_if_filter,
            rx_signal,
            rf_arrow,
            tx_signal,
            lo_arrow,
            f_tx_at_t0_label,
            f_rx_w_shift_label,
        )

        self.play(current_bd.animate.scale(bd_scale).shift(DOWN))

        if_sub_signal.scale(bd_scale).next_to(if_filter, direction=LEFT, buff=0).flip()
        if_filter_signal_label.scale(bd_scale).next_to(
            if_sub_signal, direction=UP, buff=MED_SMALL_BUFF
        )
        f_beat_label.scale(bd_scale).next_to(
            if_sub_signal, direction=UP, buff=MED_SMALL_BUFF
        )

        self.play(
            Create(if_sub_signal),
            # FadeIn(if_filter_signal_label),
            FadeIn(f_beat_label),
        )

        self.wait(0.5)

        current_bd.add(f_beat_label, if_sub_signal)

        # self.play(current_bd.animate.to_edge(UP, buff=MED_SMALL_BUFF))

        self.play(FadeOut(*self.mobjects, shift=UP * 3))

        self.wait(2)


class MixerProducts(Scene):
    def construct(self):
        f_lo = 12
        f_if = 2
        f_rf_l = f_lo - f_if
        f_rf_h = f_lo + f_if

        lo_conversion_loss = 6  # dB
        rf_h_power_relative_to_lo = ValueTracker(40)  # dB
        rf_l_power_relative_to_lo = ValueTracker(15)  # dB

        stop_time = 4
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)
        mag_offset = 60

        x_len = config["frame_width"] * 0.8
        y_len = config["frame_height"] * 0.5

        f_max = 20
        y_min = -28
        f_ax = Axes(
            x_range=[-0.1, f_max, f_max / 8],
            y_range=[0, 40, 20],
            tips=False,
            axis_config={"include_numbers": True},
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, buff=LARGE_BUFF)

        ax_labels = f_ax.get_axis_labels(
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
            Tex(r"$\lvert$", "$X(f)$", r"$\rvert$", font_size=DEFAULT_FONT_SIZE),
        )
        ax_labels.save_state()
        ax_labels_f_spelled = f_ax.get_axis_labels(
            Tex("frequency", font_size=DEFAULT_FONT_SIZE),
            Tex(
                r"$\lvert$",
                "$X($",
                "$f$",
                "$)$",
                r"$\rvert$",
                font_size=DEFAULT_FONT_SIZE,
            ),
        )

        if_plot = f_ax.plot_line_graph([0], [0], add_vertex_dots=False)

        def get_plot_values(ports=["lo", "rf", "if"], y_min=None):
            lo_signal = np.sin(2 * PI * f_lo * t)
            if_signal = np.sin(2 * PI * f_if * t) / (
                10
                ** (
                    (
                        lo_conversion_loss
                        + min(
                            rf_h_power_relative_to_lo.get_value(),
                            rf_l_power_relative_to_lo.get_value(),
                        )
                    )
                    / 10
                )
            )
            rf_l_signal = np.sin(2 * PI * f_rf_l * t) / (
                10 ** (rf_l_power_relative_to_lo.get_value() / 10)
            )
            rf_h_signal = np.sin(2 * PI * f_rf_h * t) / (
                10 ** (rf_h_power_relative_to_lo.get_value() / 10)
            )
            rf_signals = rf_l_signal + rf_h_signal

            signals = {"lo": lo_signal, "rf": rf_signals, "if": if_signal}
            summed_signals = sum([signals.get(port) for port in ports])
            # summed_signals = lo_signal + if_signal + rf_l_signal + rf_h_signal

            blackman_window = signal.windows.blackman(N)
            summed_signals *= blackman_window

            fft_len = 2**18
            summed_fft = np.fft.fft(summed_signals, fft_len) / (N / 2)
            summed_fft /= summed_fft.max()
            summed_fft_log = 10 * np.log10(np.fft.fftshift(summed_fft))
            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = summed_fft_log[indices]

            # if y_min is not None:
            y_values[y_values < y_min] = y_min
            y_values -= y_min

            return dict(x_values=x_values, y_values=y_values)

        if_plot = f_ax.plot_line_graph(
            **get_plot_values(ports=["if"], y_min=y_min),
            add_vertex_dots=False,
            line_color=IF_COLOR,
        )
        rf_plot = f_ax.plot_line_graph(
            **get_plot_values(ports=["rf"], y_min=y_min),
            add_vertex_dots=False,
            line_color=RX_COLOR,
        )
        lo_plot = f_ax.plot_line_graph(
            **get_plot_values(ports=["lo"], y_min=y_min),
            add_vertex_dots=False,
            line_color=TX_COLOR,
        )

        # plot = f_ax.plot_line_graph(
        #     np.arange(0, f_max, 1), np.linspace(0, 1, f_max), add_vertex_dots=False
        # )

        # plt.plot(freq, 10 * np.log10(np.fft.fftshift(summed_fft)))
        # plt.xlim(0, 20)
        # plt.ylim(-60, 10)

        # self.add(f_ax, f_labels, plot)

        self.play(AnimationGroup(Create(f_ax), FadeIn(ax_labels)))

        self.wait(0.5)

        self.play(Transform(ax_labels[0], ax_labels_f_spelled[0]))

        self.wait(0.5)

        self.play(Indicate(ax_labels[1]))

        self.wait(0.5)

        self.play(Restore(ax_labels))

        self.wait(0.5)

        self.play(Create(lo_plot, run_time=1.5))
        self.play(Create(rf_plot, run_time=1.5))
        self.play(Create(if_plot, run_time=1.5))

        self.wait(0.5)

        # if_plot.add_updater(
        #     lambda m: m.become(
        #         f_ax.plot_line_graph(
        #             **get_plot_values(ports=["if"], y_min=y_min),
        #             add_vertex_dots=False,
        #             line_color=IF_COLOR,
        #         )
        #     )
        # )
        # rf_plot.add_updater(
        #     lambda m: m.become(
        #         f_ax.plot_line_graph(
        #             **get_plot_values(ports=["rf"], y_min=y_min),
        #             add_vertex_dots=False,
        #             line_color=RX_COLOR,
        #         )
        #     )
        # )
        # lo_plot.add_updater(
        #     lambda m: m.become(
        #         f_ax.plot_line_graph(
        #             **get_plot_values(ports=["lo"], y_min=y_min),
        #             add_vertex_dots=False,
        #             line_color=TX_COLOR,
        #         )
        #     )
        # )

        # self.play(rf_h_power_relative_to_lo.animate(run_time=3).increment_value(-50))

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


class LoopFilter(Scene):
    def construct(self):
        step = 1 / 1000
        pfd_out_f = 0.5
        pfd_out_length = 1 / pfd_out_f

        ref_box = Square().to_edge(LEFT, buff=LARGE_BUFF)

        loop_filter_phase_tracker = ValueTracker(0)
        loop_filter_out = always_redraw(
            lambda: FunctionGraph(
                lambda t: (
                    (
                        signal.square(
                            2 * PI * pfd_out_f * t
                            - PI / 2
                            + loop_filter_phase_tracker.get_value(),
                            duty=0.5,
                        )
                        + 1
                    )
                    / 2
                )
                * signal.square(
                    2 * PI * pfd_out_f / 2 * t
                    + loop_filter_phase_tracker.get_value() / 2,
                    duty=0.5,
                ),
                x_range=[0, pfd_out_length - step, step],
                use_smoothing=False,
            )
            # .next_to(ref_box)
            .shift(LEFT + DOWN * 1.5)
        )

        self.add(loop_filter_out)

        self.play(loop_filter_phase_tracker.animate.increment_value(2 * PI), run_time=1)

        self.wait(0.5)

        self.play(loop_filter_phase_tracker.animate.increment_value(2 * PI), run_time=1)

        self.wait(1)


class CurveFunction(Scene):
    def construct(self):
        carrier_freq = 10  # Carrier frequency in Hz
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        duration = 1
        fs = 1000
        A = 1

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

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )
        sawtooth_amp = lambda t: A * np.sin(2 * PI * sawtooth_modulating_cumsum(t))
        sawtooth_amp_graph = amp_ax.plot(
            sawtooth_amp,
            x_range=[0, 1, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )

        p1 = sawtooth_amp_graph.get_start()
        p2 = sawtooth_amp_graph.get_end()

        bez = CubicBezier(
            p1,
            p1 + [1, 1, 0],
            p2 + [-1, 1, 0],
            p2,
        )

        self.add(sawtooth_amp_graph, bez)

        self.play(
            ApplyPointwiseFunction(
                bez.get_curve_functions_with_lengths, sawtooth_amp_graph
            )
        )


class Propagation(Scene):
    def construct(self):
        carrier_freq = 10  # Carrier frequency in Hz
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        duration = 1
        fs = 1000
        A = 1

        x_tracker = ValueTracker(0)

        x_len = 6
        y_len = 2.2

        amp_ax = Axes(
            x_range=[0, duration, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )
        sawtooth_amp = lambda t: A * np.sin(2 * PI * sawtooth_modulating_cumsum(t))

        antenna = BLOCKS.get("antenna").copy().shift(RIGHT + UP)
        rx_antenna = BLOCKS.get("antenna").copy().shift(RIGHT + DOWN)

        rotation = PI / 6
        rotation_line = Line(DOWN, UP).rotate(rotation)

        sawtooth_amp_graph = always_redraw(
            lambda: amp_ax.plot(
                sawtooth_amp,
                x_range=[0, x_tracker.get_value(), 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            ).flip(rotation_line.get_end() - rotation_line.get_start())
        )

        propagation_group = always_redraw(
            lambda: VGroup(amp_ax, sawtooth_amp_graph)
            .rotate(rotation)
            .next_to(antenna.get_corner(UR))
            .shift(UP * antenna.height / 2 + LEFT / 2)
        )

        rx_x_tracker = ValueTracker(0)

        rx_propagation_ax = Axes(
            x_range=[0, duration, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=6,
            y_length=2.2,
        )

        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        rx_propagation_graph = always_redraw(
            lambda: rx_propagation_ax.plot(
                sawtooth_amp,
                x_range=[0, min(rx_x_tracker.get_value(), duration), 1 / fs],
                use_smoothing=False,
                color=TX_COLOR,
            ).flip(UP)
        )

        rx_propagation_group = always_redraw(
            lambda: VGroup(rx_propagation_ax, rx_propagation_graph)
            .flip()
            .next_to(rx_antenna.get_corner(UR))
            # .shift(UP * rx_antenna.height + LEFT / 2)
        )

        # self.add(sawtooth_amp_graph, amp_ax, antenna)
        self.add(rx_propagation_ax, rx_antenna, rx_propagation_graph)

        # self.play(
        #     x_tracker.animate(rate_func=rate_functions.linear, run_time=6).set_value(2)
        # )
        self.play(
            rx_x_tracker.animate(rate_func=rate_functions.linear, run_time=6).set_value(
                duration
            )
        )

        self.wait(2)


class PassiveComponents(Scene):
    def construct(self):
        diode_tri = Triangle(color=WHITE).rotate(PI / 3)
        diode_line = Line(
            LEFT * diode_tri.width / 2, RIGHT * diode_tri.width / 2
        ).next_to(diode_tri, direction=DOWN, buff=0)
        diode_conn_1 = Line(diode_tri.get_top() + UP / 2, diode_tri.get_top())
        diode_conn_2 = Line(diode_tri.get_bottom() + DOWN / 2, diode_tri.get_bottom())
        diode = VGroup(diode_tri, diode_line, diode_conn_1, diode_conn_2)

        rotation = -PI / 4
        res_line_start = Line(LEFT / 2, ORIGIN).rotate(-rotation / 2)
        res_line_start_inv = Line(ORIGIN, RIGHT / 2).rotate(-rotation / 2)
        res_line_1 = VGroup(res_line_start, res_line_start_inv)
        res_line_2 = (
            Line(LEFT / 2, RIGHT / 2)
            .rotate(rotation / 2)
            .next_to(res_line_1, direction=DOWN, buff=0)
        )
        res_line_3 = (
            Line(LEFT / 2, RIGHT / 2)
            .rotate(-rotation / 2)
            .next_to(res_line_2, direction=DOWN, buff=0)
        )
        res_line_4 = (
            Line(LEFT / 2, RIGHT / 2)
            .rotate(rotation / 2)
            .next_to(res_line_3, direction=DOWN, buff=0)
        )
        res_line_end_inv = Line(LEFT / 2, ORIGIN)
        res_line_end = Line(ORIGIN, RIGHT / 2)
        res_line_5 = (
            VGroup(res_line_end, res_line_end_inv)
            .rotate(-rotation / 2)
            .next_to(res_line_4, direction=DOWN, buff=0)
        )
        res_conn_1 = Line(res_line_start.get_end() + UP / 2, res_line_start.get_end())
        res_conn_2 = Line(res_line_end.get_start() + DOWN / 2, res_line_end.get_start())
        resistor = VGroup(
            res_conn_1,
            res_line_start,
            res_line_2,
            res_line_3,
            res_line_4,
            res_line_end,
            res_conn_2,
        )

        self.add(
            VGroup(
                diode,
                resistor,
            ).arrange(RIGHT, buff=LARGE_BUFF)
        )


class Articles(Scene):
    def construct(self):
        article_rotation = 10 * DEGREES
        marki_mixer_video = (
            ImageMobject("../props/static/marki_mixer_talk.png")
            .shift(RIGHT)
            .rotate(-article_rotation)
            .scale(0.7)
        )
        rfmw_mixer_article = (
            ImageMobject("../props/static/rfmw_mixer_article.png")
            .rotate(article_rotation)
            .shift(LEFT)
        )

        article_shift = 7

        rfmw_mixer_article.to_edge(DOWN, buff=0).scale(0.7).shift(DOWN).shift(
            DOWN * article_shift
        )
        marki_mixer_video.move_to(rfmw_mixer_article).shift(RIGHT * 2)

        self.add(rfmw_mixer_article, marki_mixer_video)

        self.play(
            rfmw_mixer_article.animate.to_edge(DOWN, buff=0),
            marki_mixer_video.animate.to_edge(DOWN, buff=MED_LARGE_BUFF),
        )

        self.wait(1)

        self.play(
            rfmw_mixer_article.animate.move_to(DOWN * 10),
            marki_mixer_video.animate.move_to(DOWN * 10),
        )

        self.wait(2)
