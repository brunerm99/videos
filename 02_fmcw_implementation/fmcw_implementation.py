# fmcw_implementation.py


import math
import sys
import warnings
from random import normalvariate, random, seed
from typing import Iterable, Union

import numpy as np
from manim import *
from numpy.fft import fft, fftshift
from scipy import constants, signal
from skrf import Frequency, Network

warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from props import (
    FMCWRadarCartoon,
    VideoMobject,
    get_bd_animation,
    get_blocks,
    get_diode,
    get_resistor,
)
from props.style import AUREOLIN, CITRINE, PIGMENT_GREEN

BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR


TX_COLOR = BLUE
RX_COLOR = RED
GAIN_COLOR = GREEN
IF_COLOR = ORANGE
FILTER_COLOR = GREEN


BLOCKS = get_blocks()
BLOCK_BUFF = LARGE_BUFF * 2


BD_SCALE = 0.5
PLL_WIDTH = config["frame_width"] * 0.5

SKIP_ANIMATIONS_OVERRIDE = False


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_splitter_ports(splitter):
    splitter_p1 = splitter.get_right() + (UP * splitter.height / 4)
    splitter_p2 = splitter.get_right() + (DOWN * splitter.height / 4)
    return splitter_p1, splitter_p2


# I should make this into an object
def get_bd(full_pll: bool = False, rx_section_gap=BLOCK_BUFF * 3):
    input_circle = Circle(radius=0.2)
    input_label = Tex(r"$V_{tune}$ Input").next_to(
        input_circle, direction=UP, buff=SMALL_BUFF
    )
    inp = VGroup(input_circle, input_label)
    if full_pll:
        phase_detector = (
            BLOCKS.get("phase_detector").copy().next_to(input_circle, buff=BLOCK_BUFF)
        )
        input_to_pll = Line(
            input_circle.get_right(),
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
        pll_block_to_pa = Line(
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
        vco_output_conn = Dot(
            pll_block_to_pa.get_midpoint(), radius=DEFAULT_DOT_RADIUS * 2
        )
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

        pll_block = VGroup(
            input_to_pll,
            phase_detector,
            phase_detector_to_loop_filter,
            loop_filter,
            loop_filter_to_vco,
            vco,
            pll_block_to_pa,
            vco_output_conn,
            vco_to_ndiv_1,
            vco_to_ndiv_2,
            ndiv,
            ndiv_to_phase_detector_1,
            ndiv_to_phase_detector_2,
        )
        pa = (
            BLOCKS.get("amp")
            .copy()
            .next_to(pll_block_to_pa, direction=RIGHT, buff=0)
            .set_fill(GAIN_COLOR)
        )
    else:
        pll_block = (
            BLOCKS.get("oscillator")
            .copy()
            .next_to(input_circle, direction=RIGHT, buff=BLOCK_BUFF)
        )
        input_to_pll = Line(input_circle.get_right(), pll_block.get_left())
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
        .shift(DOWN * rx_section_gap)
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
            input_to_pll,
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
    # This is atrocious...
    if full_pll:
        return (
            bd,
            (
                inp,
                input_to_pll,
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
            (
                input_to_pll,
                phase_detector,
                phase_detector_to_loop_filter,
                loop_filter,
                loop_filter_to_vco,
                vco,
                pll_block_to_pa,
                vco_output_conn,
                vco_to_ndiv_1,
                vco_to_ndiv_2,
                ndiv,
                ndiv_to_phase_detector_1,
                ndiv_to_phase_detector_2,
            ),
        )
    else:
        return (
            bd,
            (
                inp,
                input_to_pll,
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
        )


def get_fade_group(group, opacity, **animate_kwargs) -> List[Animation]:
    animations = []
    for m in group:
        if type(m) == VDict:
            animations.append(m.animate(**animate_kwargs).set_stroke(opacity=opacity))
        else:
            animations.append(m.animate(**animate_kwargs).set_opacity(opacity))
    return animations


# Copy/pasted from abul4fia's package b/c not in conda...
def play_timeline(scene: Scene, timeline):
    """
    Plays a timeline of animations on a given scene.
    Args:
        scene (Scene): The scene to play the animations on.
        timeline (dict): A dictionary where the keys are the times at which the animations should start,
            and the values are the animations to play at that time. The values can be a single animation
            or an iterable of animations.

    Notes:
        Each animation in the timeline can have a different duration, so several animations can be
        running in parallel. If the value for a given time is an iterable, all the animations
        in the iterable are started at once (although they can end at different times depending
        on their run_time)
        The method returns when all animations have finished playing.
    Returns:
        None
    """
    previous_t = 0
    ending_time = 0
    for t, anims in sorted(timeline.items()):
        to_wait = t - previous_t
        if to_wait > 0:
            scene.wait(to_wait)
        previous_t = t
        if not isinstance(anims, Iterable):
            anims = [anims]
        for anim in anims:
            turn_animation_into_updater(anim)
            scene.add(anim.mobject)
            ending_time = max(ending_time, t + anim.run_time)
    if ending_time > t:
        scene.wait(ending_time - t)


class Intro(Scene):
    def construct(self):
        self.next_section(skip_animations=True)
        fmcw = FMCWRadarCartoon()

        carrier_freq = 10
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
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
        ).scale(1.1)
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

        sawtooth_amp_graph = amp_ax.plot(
            sawtooth_amp,
            x_range=[0, duration, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )

        plot_group = VGroup(amp_ax, sawtooth_amp_graph)

        range_eqn_color = ORANGE
        range_eqn = Tex(
            r"$R = \frac{c T_{c} f_{beat}}{2 B}$", color=range_eqn_color
        ).scale(1.2)

        useful_color = GREEN
        useful = Tex(
            r"\begin{itemize}"
            r"\item Lower peak power"
            r"\item Small minimum range"
            r"\item Fine range resolution"
            r"\end{itemize}",
            color=useful_color,
        )
        useful_brace = Brace(useful, direction=LEFT, sharpness=1)
        useful_group = VGroup(useful, useful_brace)

        VGroup(plot_group, range_eqn, useful_group).arrange(
            direction=DOWN, buff=MED_LARGE_BUFF
        ).to_edge(RIGHT, buff=LARGE_BUFF)

        kicad = ImageMobject("../props/static/kicad.png")

        gears = SVGMobject("../props/static/Gears.svg").scale(3)
        (red_gear, blue_gear) = gears.shift(DOWN * 0.5 + RIGHT)

        gr = 24 / 12

        red_accel = ValueTracker(0)
        red_vel = ValueTracker(0)
        blue_vel = ValueTracker(0)

        def driver_updater(m, dt):
            red_vel.set_value(red_vel.get_value() + dt * red_accel.get_value())
            blue_vel.set_value(-red_vel.get_value() / gr)
            m.rotate(dt * red_vel.get_value())

        def driven_updater(m, dt):
            m.rotate(dt * blue_vel.get_value())

        red_gear.add_updater(driver_updater)
        blue_gear.add_updater(driven_updater)

        f_beat_eqn = MathTex(r"f_{beat} = f_{TX} - f_{RX}")

        amp, lp_filter, mixer, oscillator, phase_shifter, switch = (
            BLOCKS.get("amp").copy().scale(0.7),
            BLOCKS.get("lp_filter").copy().scale(0.7),
            BLOCKS.get("mixer").copy().scale(0.7),
            BLOCKS.get("oscillator").copy().scale(0.7),
            BLOCKS.get("phase_shifter").copy().scale(0.7),
            BLOCKS.get("spdt_switch").copy().scale(0.7),
        )
        rf_blocks = Group(
            amp, lp_filter, mixer, oscillator, phase_shifter, switch
        ).arrange_in_grid(rows=3, cols=2, buff=(MED_LARGE_BUFF, MED_LARGE_BUFF))

        self.play(fmcw.get_animation())

        self.play(fmcw.vgroup.animate.scale(0.8).to_edge(LEFT, buff=LARGE_BUFF))

        self.wait(0.3)

        tx_wf_p0 = fmcw.vgroup.get_right() + [0.1, 0, 0]
        tx_wf_p1 = amp_ax.get_left() + [-0.1, 0, 0]
        tx_wf_bez = CubicBezier(
            tx_wf_p0,
            tx_wf_p0 + [1, 0, 0],
            tx_wf_p1 + [-1, 0, 0],
            tx_wf_p1,
            color=TX_COLOR,
        )

        range_eqn_p0 = fmcw.vgroup.get_right() + [0.1, 0, 0]
        range_eqn_p1 = range_eqn.get_left() + [-0.1, 0, 0]
        range_eqn_bez = CubicBezier(
            range_eqn_p0,
            range_eqn_p0 + [1, 0, 0],
            range_eqn_p1 + [-1, 0, 0],
            range_eqn_p1,
            color=range_eqn_color,
        )

        useful_p0 = fmcw.vgroup.get_right() + [0.1, 0, 0]
        useful_p1 = useful_group.get_left() + [-0.1, 0, 0]
        useful_bez = CubicBezier(
            useful_p0,
            useful_p0 + [1, 0, 0],
            useful_p1 + [-1, 0, 0],
            useful_p1,
            color=useful_color,
        )

        self.play(Create(tx_wf_bez), Create(amp_ax), Create(sawtooth_amp_graph))

        self.play(Create(range_eqn_bez), Create(range_eqn))

        self.play(Create(useful_bez), FadeIn(useful_brace), Create(useful))

        self.wait(0.5)

        fmcw_copy = fmcw.vgroup.copy()
        Group(fmcw_copy, kicad).arrange(center=True, buff=LARGE_BUFF)
        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(tx_wf_bez),
                    Uncreate(range_eqn_bez),
                    Uncreate(useful_bez),
                    FadeOut(plot_group, range_eqn, useful_group),
                ),
                Transform(fmcw.vgroup, fmcw_copy),
                lag_ratio=0.5,
            )
        )
        self.play(GrowFromCenter(kicad))

        self.wait(0.5)

        fmcw.vgroup.remove(fmcw.label)
        fmcw_centered = (
            fmcw.vgroup.copy()
            .move_to(ORIGIN)
            .scale_to_fit_height(config["frame_height"] * 0.8)
        )
        self.play(
            LaggedStart(
                FadeOut(kicad),
                AnimationGroup(
                    FadeOut(fmcw.label),
                    Transform(fmcw.vgroup, fmcw_centered),
                ),
            )
        )
        gears.rotate(-PI / 2).scale_to_fit_width(fmcw.rect.width * 0.8).move_to(
            fmcw.rect
        )
        self.play(FadeIn(gears))
        red_accel.set_value(PI / 12)

        fmcw_w_gears = Group(fmcw.vgroup, gears)

        self.wait(1)

        self.play(fmcw_w_gears.animate.scale(0.5))

        wanted = (
            VGroup(sawtooth_amp_graph.scale(0.6), f_beat_eqn)
            .arrange(DOWN, buff=LARGE_BUFF)
            .to_edge(LEFT)
        )
        wanted.next_to(fmcw_w_gears, direction=LEFT, buff=LARGE_BUFF)
        rf_blocks.next_to(fmcw_w_gears, direction=RIGHT, buff=LARGE_BUFF)

        sawtooth_p0 = sawtooth_amp_graph.get_right() + [0.1, 0, 0]
        sawtooth_p1 = fmcw_w_gears.get_left() + [-0.1, 0, 0]
        sawtooth_bez = CubicBezier(
            sawtooth_p0,
            sawtooth_p0 + [1, 0, 0],
            sawtooth_p1 + [-1, 0, 0],
            sawtooth_p1,
            # color=sawtooth_color,
        )

        f_beat_p0 = f_beat_eqn.get_right() + [0.1, 0, 0]
        f_beat_p1 = fmcw_w_gears.get_left() + [-0.1, 0, 0]
        f_beat_bez = CubicBezier(
            f_beat_p0,
            f_beat_p0 + [1, 0, 0],
            f_beat_p1 + [-1, 0, 0],
            f_beat_p1,
            # color=f_beat_color,
        )

        rf_block_1_p0 = fmcw_w_gears.get_right() + [0.1, 0, 0]
        rf_block_1_p1 = amp.get_left() + [-0.1, 0, 0]
        rf_block_1_bez = CubicBezier(
            rf_block_1_p0,
            rf_block_1_p0 + [1, 0, 0],
            rf_block_1_p1 + [-1, 0, 0],
            rf_block_1_p1,
            # color=rf_block_1_color,
        )

        rf_block_2_p0 = fmcw_w_gears.get_right() + [0.1, 0, 0]
        rf_block_2_p1 = mixer.get_left() + [-0.1, 0, 0]
        rf_block_2_bez = CubicBezier(
            rf_block_2_p0,
            rf_block_2_p0 + [1, 0, 0],
            rf_block_2_p1 + [-1, 0, 0],
            rf_block_2_p1,
            # color=rf_block_2_color,
        )

        rf_block_3_p0 = fmcw_w_gears.get_right() + [0.1, 0, 0]
        rf_block_3_p1 = phase_shifter.get_left() + [-0.1, 0, 0]
        rf_block_3_bez = CubicBezier(
            rf_block_3_p0,
            rf_block_3_p0 + [1, 0, 0],
            rf_block_3_p1 + [-1, 0, 0],
            rf_block_3_p1,
            # color=rf_block_3_color,
        )

        self.play(
            LaggedStart(
                Create(sawtooth_amp_graph),
                Create(f_beat_eqn),
                lag_ratio=0.6,
            )
        )
        self.play(Create(sawtooth_bez), Create(f_beat_bez))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(rf_block_1_bez),
                GrowFromCenter(amp),
                GrowFromCenter(lp_filter),
                Create(rf_block_2_bez),
                GrowFromCenter(mixer),
                GrowFromCenter(oscillator),
                Create(rf_block_3_bez),
                GrowFromCenter(phase_shifter),
                GrowFromCenter(switch),
                lag_ratio=0.3,
            )
        )

        self.wait(2)

        all_except_title = Group(*self.mobjects)

        title = Tex("FMCW Radar Part 2:").scale(1.5)
        subtitle = Tex("Implementation").scale(1.5)
        hline = Line(LEFT, RIGHT)
        hline.width = config["frame_width"] * 0.8

        hline.next_to(
            all_except_title.copy().scale(0.8).to_edge(DOWN, buff=MED_LARGE_BUFF),
            direction=UP,
            buff=MED_LARGE_BUFF,
        )
        subtitle.next_to(hline, direction=UP, buff=MED_SMALL_BUFF)
        title.next_to(subtitle, direction=UP, buff=MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                all_except_title.animate.scale(0.8).to_edge(DOWN, buff=MED_LARGE_BUFF),
                AnimationGroup(GrowFromCenter(hline), Create(title), Create(subtitle)),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(
            all_except_title.animate(run_time=1.5).shift(DOWN * 5),
            ShrinkToCenter(hline),
        )

        self.wait(2)


class Part2Series(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=False)
        title = Tex("FMCW Radar Part 2:").scale(1.5)
        subtitle = (
            Tex("Implementation")
            .scale(1.5)
            .next_to(title, direction=DOWN, buff=MED_SMALL_BUFF)
        )
        titles = VGroup(title, subtitle).move_to([-3.61670042e-1, 2.13766246, 0])

        nl = (
            NumberLine(
                x_range=[0, 4, 1], length=config["frame_width"] * 1.8, tick_size=0.3
            )
            .move_to(ORIGIN, aligned_edge=LEFT)
            .shift(DOWN * 2)
        )
        part_1_thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale(0.4)
            .next_to(nl.n2p(0), direction=UP, buff=LARGE_BUFF)
        )
        part_1_thumbnail_box = SurroundingRectangle(part_1_thumbnail, buff=0)
        part_1_label = Tex("Intro to FMCW Radar").next_to(
            part_1_thumbnail, direction=UP, buff=SMALL_BUFF
        )
        part_2_thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale(0.4)
            .next_to(nl.n2p(1), direction=UP, buff=LARGE_BUFF)
        )
        part_2_thumbnail_box = SurroundingRectangle(part_2_thumbnail, buff=0)
        part_2_label = Tex("Implementation").next_to(
            part_2_thumbnail, direction=UP, buff=SMALL_BUFF
        )
        this_video = Tex("This video", color=GREEN).next_to(
            nl.n2p(1), direction=DOWN, buff=MED_LARGE_BUFF
        )
        part_3_thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale(0.4)
            .next_to(nl.n2p(2), direction=UP, buff=LARGE_BUFF)
        )
        part_3_thumbnail_box = SurroundingRectangle(part_3_thumbnail, buff=0)
        part_3_label = Tex(r"Velocity Calculation and\\the Radar Cube").next_to(
            part_3_thumbnail, direction=UP, buff=SMALL_BUFF
        )
        part_4_thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale(0.4)
            .next_to(nl.n2p(3), direction=UP, buff=LARGE_BUFF)
        )
        part_4_thumbnail_box = SurroundingRectangle(part_4_thumbnail, buff=0)
        part_4_qmark = Tex("?").scale(3).move_to(part_4_thumbnail)
        part_4_label = Tex(r"To be announced").next_to(
            part_4_thumbnail, direction=UP, buff=SMALL_BUFF
        )

        self.add(titles)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(nl, run_time=2),
                FadeIn(
                    part_1_thumbnail, part_1_thumbnail_box, part_1_label, shift=DOWN
                ),
                AnimationGroup(
                    FadeIn(part_2_thumbnail, part_2_thumbnail_box, shift=DOWN),
                    FadeIn(this_video, shift=UP),
                    Transform(titles, part_2_label),
                ),
                lag_ratio=0.5,
            )
        )

        self.add(
            part_3_thumbnail,
            part_3_thumbnail_box,
            part_3_label,
            part_4_qmark,
            part_4_thumbnail_box,
            part_4_label,
        )

        part_3_arrow = Arrow(
            part_3_thumbnail_box.get_bottom() + [1, -2, 0],
            part_3_thumbnail_box.get_bottom(),
        )
        part_4_arrow = Arrow(
            part_4_thumbnail_box.get_bottom() + [-1, -2, 0],
            part_4_thumbnail_box.get_bottom(),
        )
        part_1_arrow = Arrow(
            part_1_thumbnail_box.get_corner(DL) + [-2, -2, 0],
            part_1_thumbnail_box.get_corner(DL),
        )

        self.wait(0.5)

        self.play(GrowArrow(part_1_arrow))

        self.wait(0.5)

        self.play(
            FadeOut(part_1_arrow),
            self.camera.frame.animate.move_to([nl.n2p(2.5)[0], 0, 0]),
            run_time=2,
        )

        self.wait(0.5)

        self.play(GrowArrow(part_3_arrow), GrowArrow(part_4_arrow))

        self.wait(0.5)

        self.play(
            FadeOut(part_3_arrow, part_4_arrow),
            self.camera.frame.animate.move_to([nl.n2p(1)[0], 0, 0]),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            Uncreate(nl),
            FadeOut(
                part_1_thumbnail,
                part_1_thumbnail_box,
                part_1_label,
                part_3_label,
                part_3_thumbnail,
                part_3_thumbnail_box,
                # part_4_thumbnail,
                part_4_thumbnail_box,
                part_4_qmark,
                part_4_label,
                shift=UP,
            ),
            FadeOut(this_video, shift=DOWN),
        )

        self.play(self.camera.frame.animate.scale(0.1), FadeOut(*self.mobjects))

        # self.play(self.camera.frame.animate(run_time=6).shift(RIGHT * nl.n2p(3)))

        self.wait(2)


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


class MixerProducts(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        f_lo = 12
        f_if = 2
        f_rf_l = f_lo - f_if
        f_rf_h = f_lo + f_if

        lo_conversion_loss = 6  # dB
        lo_loss = ValueTracker(40)  # dB
        rf_h_loss = ValueTracker(40)  # dB
        rf_l_loss = ValueTracker(40)  # dB
        if_loss = ValueTracker(40)

        stop_time = 4
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)
        mag_offset = 60

        x_len = config["frame_width"] * 0.8
        y_len = config["frame_height"] * 0.4

        f_max = 20
        y_min = -28
        ax = Axes(
            x_range=[-0.1, f_max, f_max / 8],
            y_range=[0, -y_min, 20],
            tips=False,
            axis_config={
                "include_numbers": False,
                "include_ticks": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, buff=LARGE_BUFF)

        ax_x_label = ax.get_x_axis_label(Tex("$f$", font_size=DEFAULT_FONT_SIZE))
        ax_y_label = ax.get_y_axis_label(
            Tex(r"$\lvert$", "$X(f)$", r"$\rvert$", font_size=DEFAULT_FONT_SIZE),
            edge=LEFT,
            direction=LEFT,
        ).rotate(PI / 2)

        ax_x_label.save_state()
        ax_x_label_spelled = ax.get_x_axis_label(
            Tex("frequency", font_size=DEFAULT_FONT_SIZE)
        )

        lo_tick_pos = ax.c2p(f_lo, 0, 0)
        lo_tick = Line(lo_tick_pos + DOWN / 4, lo_tick_pos + UP / 4)
        lo_tick_label = Tex(f"{f_lo}").next_to(lo_tick, direction=DOWN, buff=SMALL_BUFF)

        rf_l_tick_pos = ax.c2p(f_rf_l, 0, 0)
        rf_l_tick = Line(rf_l_tick_pos + DOWN / 4, rf_l_tick_pos + UP / 4)
        rf_l_tick_label = Tex(f"{f_rf_l}").next_to(
            rf_l_tick, direction=DOWN, buff=SMALL_BUFF
        )

        rf_h_tick_pos = ax.c2p(f_rf_h, 0, 0)
        rf_h_tick = Line(rf_h_tick_pos + DOWN / 4, rf_h_tick_pos + UP / 4)
        rf_h_tick_label = Tex(f"{f_rf_h}").next_to(
            rf_h_tick, direction=DOWN, buff=SMALL_BUFF
        )

        if_tick_pos = ax.c2p(f_if, 0, 0)
        if_tick = Line(if_tick_pos + DOWN / 4, if_tick_pos + UP / 4)
        if_tick_label = Tex(f"{f_if}").next_to(if_tick, direction=DOWN, buff=SMALL_BUFF)

        unit_label = (
            Tex("GHz").next_to(ax_x_label, direction=DOWN).set_y(lo_tick_label.get_y())
        )

        lo_line_legend = Line(ORIGIN, RIGHT, color=TX_COLOR).to_corner(
            UR, buff=MED_LARGE_BUFF
        )
        lo_legend = Tex("LO", color=TX_COLOR).next_to(
            lo_line_legend, direction=LEFT, buff=SMALL_BUFF
        )
        rf_line_legend = Line(ORIGIN, RIGHT, color=RX_COLOR).next_to(
            lo_line_legend, direction=DOWN, buff=MED_LARGE_BUFF
        )
        rf_legend = Tex("RF", color=RX_COLOR).next_to(
            rf_line_legend, direction=LEFT, buff=SMALL_BUFF
        )
        if_line_legend = Line(ORIGIN, RIGHT, color=IF_COLOR).next_to(
            rf_line_legend, direction=DOWN, buff=MED_LARGE_BUFF
        )
        if_legend = Tex("IF", color=IF_COLOR).next_to(
            if_line_legend, direction=LEFT, buff=SMALL_BUFF
        )
        filter_line_legend = Line(ORIGIN, RIGHT, color=FILTER_COLOR).next_to(
            if_line_legend, direction=DOWN, buff=MED_LARGE_BUFF
        )
        filter_legend = Tex("Filter", color=FILTER_COLOR).next_to(
            filter_line_legend, direction=LEFT, buff=SMALL_BUFF
        )

        f_if_eqn_desired = Tex(r"$f_{LO} - f_{RF}$")
        f_if_eqn_and = Tex("and")
        f_if_eqn_flipped = Tex(r"$f_{RF} - f_{IF}$")
        f_if_eqn = VGroup(f_if_eqn_desired, f_if_eqn_and, f_if_eqn_flipped)

        if_plot = ax.plot_line_graph([0], [0], add_vertex_dots=False)

        def get_plot_values(ports=["lo", "rf_l", "rf_h", "if"], y_min=None):
            lo_signal = np.sin(2 * PI * f_lo * t) / (10 ** (lo_loss.get_value() / 10))
            if_signal = np.sin(2 * PI * f_if * t) / (10 ** (if_loss.get_value() / 10))
            rf_l_signal = np.sin(2 * PI * f_rf_l * t) / (
                10 ** (rf_l_loss.get_value() / 10)
            )
            rf_h_signal = np.sin(2 * PI * f_rf_h * t) / (
                10 ** (rf_h_loss.get_value() / 10)
            )

            signals = {
                "lo": lo_signal,
                "rf_l": rf_l_signal,
                "rf_h": rf_h_signal,
                "if": if_signal,
            }
            summed_signals = sum([signals.get(port) for port in ports])

            blackman_window = signal.windows.blackman(N)
            summed_signals *= blackman_window

            fft_len = 2**18
            summed_fft = np.fft.fft(summed_signals, fft_len) / (N / 2)
            # summed_fft /= summed_fft.max()
            summed_fft_log = 10 * np.log10(np.fft.fftshift(summed_fft))
            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = summed_fft_log[indices]

            if y_min is not None:
                y_values[y_values < y_min] = y_min
                y_values -= y_min

            return dict(x_values=x_values, y_values=y_values)

        if_plot = ax.plot_line_graph(
            **get_plot_values(ports=["if"], y_min=y_min),
            add_vertex_dots=False,
            line_color=IF_COLOR,
        )
        rf_l_plot = ax.plot_line_graph(
            **get_plot_values(ports=["rf_l"], y_min=y_min),
            add_vertex_dots=False,
            line_color=RX_COLOR,
        )
        rf_h_plot = ax.plot_line_graph(
            **get_plot_values(ports=["rf_h"], y_min=y_min),
            add_vertex_dots=False,
            line_color=RX_COLOR,
        )
        lo_plot = ax.plot_line_graph(
            **get_plot_values(ports=["lo"], y_min=y_min),
            add_vertex_dots=False,
            line_color=TX_COLOR,
        )

        time_ax_x_len = 3
        time_ax_y_len = 2
        time_ax_x_max = 0.4
        rf_h_A = 0.3
        rf_A = 0.5
        rf_l_time_ax = Axes(
            x_range=[0, time_ax_x_max, time_ax_x_max / 4],
            y_range=[-rf_A, rf_A, rf_A / 2],
            tips=False,
            axis_config={
                "include_numbers": False,
                "include_ticks": True,
            },
            x_length=time_ax_x_len,
            y_length=time_ax_y_len,
        )
        rf_h_time_ax = Axes(
            x_range=[0, time_ax_x_max, time_ax_x_max / 4],
            y_range=[-rf_A, rf_A, rf_A / 2],
            tips=False,
            axis_config={
                "include_numbers": False,
                "include_ticks": True,
            },
            x_length=time_ax_x_len,
            y_length=time_ax_y_len,
        )

        rf_l_signal_time = rf_l_time_ax.plot(
            lambda t: rf_A * np.sin(2 * PI * f_rf_l * t), color=RX_COLOR
        )
        rf_h_signal_time = rf_h_time_ax.plot(
            lambda t: rf_A * rf_h_A * np.sin(2 * PI * f_rf_h * t), color=RX_COLOR
        )
        rf_summed_signal_time = rf_l_time_ax.plot(
            lambda t: rf_A
            * (np.sin(2 * PI * f_rf_l * t) + rf_h_A * np.sin(2 * PI * f_rf_h * t)),
            color=RX_COLOR,
        )

        rf_l_plot_group = VGroup(rf_l_time_ax, rf_l_signal_time, rf_summed_signal_time)
        rf_h_plot_group = VGroup(rf_h_time_ax, rf_h_signal_time)

        plot_group = VGroup(
            ax,
            ax_x_label,
            ax_y_label,
            lo_plot,
            rf_l_plot,
            rf_h_plot,
            if_plot,
            if_tick,
            if_tick_label,
            lo_tick,
            lo_tick_label,
            rf_l_tick,
            rf_l_tick_label,
            rf_h_tick,
            rf_h_tick_label,
            unit_label,
            f_if_eqn_desired,
        )

        def get_f_rect(f, loss_tracker):
            top = ax.c2p(f, -y_min - loss_tracker.get_value(), 0)
            bot = ax.c2p(f, 0, 0)
            mid = ax.c2p(f, (-y_min - loss_tracker.get_value()) / 2, 0)
            return top, bot, mid

        def get_rect_updater(f, loss_tracker):
            def if_rect_updater(m: Mobject):
                top, bot, mid = get_f_rect(f, loss_tracker)
                m.become(Rectangle(width=1, height=Line(bot, top).height).move_to(mid))

            return if_rect_updater

        self.play(Create(ax), FadeIn(ax_x_label, ax_y_label))

        self.next_section(skip_animations=skip_animations(True))

        self.wait(0.5)

        self.play(Transform(ax_x_label, ax_x_label_spelled))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Indicate(ax_y_label))

        self.wait(0.5)

        self.play(Restore(ax_x_label))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Create(lo_plot), Create(rf_l_plot), Create(rf_h_plot), Create(if_plot)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        def get_plot_updater(ports, color):
            def updater(m: Mobject):
                m.become(
                    ax.plot_line_graph(
                        **get_plot_values(ports=ports, y_min=y_min),
                        add_vertex_dots=False,
                        line_color=color,
                    )
                )

            return updater

        if_plot_updater = get_plot_updater(ports=["if"], color=IF_COLOR)
        rf_l_plot_updater = get_plot_updater(ports=["rf_l"], color=RX_COLOR)
        rf_h_plot_updater = get_plot_updater(ports=["rf_h"], color=RX_COLOR)
        lo_plot_updater = get_plot_updater(ports=["lo"], color=TX_COLOR)

        if_plot.add_updater(if_plot_updater)
        rf_l_plot.add_updater(rf_l_plot_updater)
        rf_h_plot.add_updater(rf_h_plot_updater)
        lo_plot.add_updater(lo_plot_updater)

        self.play(
            lo_loss.animate(run_time=1.5).set_value(0),
            FadeIn(lo_line_legend, lo_legend),
            FadeIn(lo_tick, lo_tick_label),
            FadeIn(unit_label),
        )
        self.play(
            rf_l_loss.animate(run_time=1.5).set_value(8),
            FadeIn(rf_line_legend, rf_legend),
            FadeIn(rf_l_tick, rf_l_tick_label),
        )
        self.play(
            if_loss.animate(run_time=1.5).set_value(15),
            FadeIn(if_line_legend, if_legend),
            FadeIn(if_tick, if_tick_label),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(1)

        f_if_eqn_desired.next_to(
            ax.c2p(f_if, -y_min - if_loss.get_value()), direction=UP, buff=SMALL_BUFF
        ).shift(RIGHT / 4)
        self.play(FadeIn(f_if_eqn_desired))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        f_if_eqn_desired_copy = f_if_eqn_desired.copy()
        f_if_eqn_flipped.move_to(f_if_eqn_desired)
        f_if_eqn_and.next_to(f_if_eqn_flipped, direction=UP, buff=SMALL_BUFF)
        f_if_eqn_desired_copy.next_to(f_if_eqn_and, direction=UP, buff=SMALL_BUFF)

        self.play(
            LaggedStart(
                Transform(f_if_eqn_desired, f_if_eqn_desired_copy),
                FadeIn(f_if_eqn_and),
                FadeIn(f_if_eqn_flipped),
                lag_ratio=0.8,
            )
        )

        plot_group.remove(f_if_eqn_desired)
        plot_group.add(f_if_eqn)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(1)

        self.play(FadeIn(rf_h_tick, rf_h_tick_label))

        self.wait(0.5)

        self.play(plot_group.animate.to_edge(DOWN, buff=MED_SMALL_BUFF))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_l_p1 = ax.c2p(f_rf_l, -y_min - rf_l_loss.get_value(), 0) + DOWN / 4
        rf_l_plot_group.next_to(rf_l_p1, direction=UP).shift(LEFT * 2).to_edge(UP)
        rf_l_p2 = rf_l_plot_group.get_bottom()
        rf_l_bezier = CubicBezier(
            rf_l_p1,
            rf_l_p1 + [0, 0.5, 0],
            rf_l_p2 + [0, -1, 0],
            rf_l_p2,
        )

        self.play(
            LaggedStart(
                Create(rf_l_bezier),
                Create(rf_l_time_ax),
                Create(rf_l_signal_time),
                lag_ratio=0.7,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(rf_h_loss.animate.set_value(rf_l_loss.get_value() + 10))

        rf_h_p1 = ax.c2p(f_rf_h, -y_min - rf_h_loss.get_value(), 0) + DOWN / 4
        rf_h_plot_group.next_to(
            rf_l_time_ax, direction=RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 3
        )
        rf_h_p2 = rf_h_plot_group.get_bottom() + [0, -0.2, 0]
        rf_h_bezier = CubicBezier(
            rf_h_p1,
            rf_h_p1 + [0.5, 0.5, 0],
            rf_h_p2 + [0, -0.5, 0],
            rf_h_p2,
        )

        self.play(
            LaggedStart(
                Create(rf_h_bezier),
                Create(rf_h_time_ax),
                Create(rf_h_signal_time),
                lag_ratio=0.7,
            )
        )

        if_rect = Rectangle()
        if_rect_updater = get_rect_updater(f_if, if_loss)
        if_rect.add_updater(if_rect_updater)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Create(if_rect))
        self.play(
            f_if_eqn.animate.next_to(if_rect, direction=UP, buff=MED_SMALL_BUFF).shift(
                RIGHT / 4
            )
        )

        self.wait(0.5)

        f_if_eqn_updater = lambda m: m.next_to(
            if_rect, direction=UP, buff=MED_SMALL_BUFF
        ).shift(RIGHT / 4)
        f_if_eqn.add_updater(f_if_eqn_updater)

        self.play(
            if_loss.animate.increment_value(-5),
            # f_if_eqn_desired.animate.shift(UP * 0.8),
        )

        f_if_eqn.remove_updater(f_if_eqn_updater)

        self.wait(0.5)

        if_rect.remove_updater(if_rect_updater)
        self.play(Uncreate(if_rect))

        self.play(
            Uncreate(rf_l_bezier),
            Uncreate(rf_h_bezier),
            LaggedStart(
                Uncreate(rf_l_signal_time),
                Uncreate(rf_l_time_ax),
                lag_ratio=0.7,
            ),
            LaggedStart(
                Uncreate(rf_h_signal_time),
                Uncreate(rf_h_time_ax),
                lag_ratio=0.7,
            ),
        )
        self.play(plot_group.animate.move_to(ORIGIN))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_l_to_if = (
            ArcBetweenPoints(
                rf_l_tick_label.get_bottom() + [0, -0.1, 0],
                if_tick_label.get_bottom() + [0, -0.1, 0],
                angle=-TAU / 8,
                color=RX_COLOR,
            )
            .add_tip()
            .set_z_index(1)
        )
        rf_h_to_if = ArcBetweenPoints(
            rf_h_tick_label.get_bottom() + [0, -0.1, 0],
            if_tick_label.get_bottom() + [0.15, -0.25, 0],
            angle=-TAU / 6,
            color=RX_COLOR,
        ).set_z_index(0)
        lo_to_if = ArcBetweenPoints(
            lo_tick_label.get_bottom() + [0, -0.1, 0],
            if_tick_label.get_bottom() + [0.15, -0.25, 0],
            angle=-TAU / 7,
            color=TX_COLOR,
        ).set_z_index(0)

        self.play(
            LaggedStart(
                Create(rf_l_to_if),
                Create(rf_h_to_if),
                Create(lo_to_if),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        block_scale = 0.8
        mixer = (
            BLOCKS.get("mixer")
            .copy()
            .scale(block_scale)
            .to_edge(UP, buff=LARGE_BUFF)
            .shift(LEFT)
        )

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

        mixer_group = VGroup(mixer, lo_port, rf_port, if_port)

        fs = 1000
        step = 1 / fs
        x_range = [0, 0.75, step]
        x_range_lo = [0, 0.7, step]
        x_len = 2.5
        y_len = 2
        lo_ax = (
            Axes(
                x_range=x_range_lo[:2], y_range=[-2, 2], x_length=x_len, y_length=y_len
            )
            .rotate(-PI / 2)
            .next_to(mixer, direction=UP, buff=0)
        )
        rf_filt_ax = Axes(
            x_range=x_range[:2], y_range=[-2, 2], x_length=x_len, y_length=y_len
        ).rotate(PI)
        rf_ax = (
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

        A = 1
        lo_signal = lo_ax.plot(
            lambda t: A * np.sin(2 * PI * f_lo * t), x_range=x_range_lo, color=TX_COLOR
        )
        rf_l_A = 0.8
        rf_h_A = 1 - rf_l_A
        rf_l_A_filt = 0.9
        rf_h_A_filt = 1 - rf_l_A_filt
        rf_filt_signal = rf_ax.plot(
            lambda t: rf_l_A_filt * np.sin(2 * PI * f_rf_l * t)
            + rf_h_A_filt * np.sin(2 * PI * f_rf_h * t),
            x_range=x_range,
            color=RX_COLOR,
        )
        rf_signal = rf_ax.plot(
            lambda t: rf_l_A * np.sin(2 * PI * f_rf_l * t)
            + rf_h_A * np.sin(2 * PI * f_rf_h * t),
            x_range=x_range,
            color=RX_COLOR,
        )
        if_signal = if_ax.plot(
            lambda t: A
            * np.sin(2 * PI * f_lo * t)
            * (
                rf_l_A * np.sin(2 * PI * f_rf_l * t)
                + rf_h_A * np.sin(2 * PI * f_rf_h * t)
            ),
            x_range=x_range,
            color=IF_COLOR,
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(rf_l_to_if),
                    Uncreate(rf_h_to_if),
                    Uncreate(lo_to_if),
                ),
                AnimationGroup(
                    plot_group.animate.to_edge(DOWN, buff=MED_LARGE_BUFF),
                    FadeIn(mixer_group, shift=DOWN * 2),
                ),
                Create(lo_signal),
                Create(rf_signal),
                Create(if_signal),
                lag_ratio=0.7,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_plus = (
            Text("+", color=YELLOW)
            .scale(2)
            .next_to(rf_signal, direction=DOWN, buff=MED_LARGE_BUFF)
        )

        rf_plus_to_rf_signal = Line(
            rf_plus.get_top() + [0, 0.1, 0], rf_signal.get_bottom() + [0, -0.1, 0]
        )

        rf_l_p1 = ax.c2p(f_rf_l, -y_min - rf_l_loss.get_value(), 0) + DOWN / 4
        rf_l_p2 = rf_plus.get_left() + [-0.1, 0, 0]
        rf_l_to_mixer_rf_bezier = CubicBezier(
            rf_l_p1,
            rf_l_p1 + [0.5, 1, 0],
            rf_l_p2 + [-0.5, 0, 0],
            rf_l_p2,
        )

        rf_h_p1 = ax.c2p(f_rf_h, -y_min - rf_h_loss.get_value(), 0) + DOWN / 4
        rf_h_p2 = rf_plus.get_right() + [0.1, 0, 0]
        rf_h_to_mixer_rf_bezier = CubicBezier(
            rf_h_p1,
            rf_h_p1 + [0.5, 1, 0],
            rf_h_p2 + [0.5, 0, 0],
            rf_h_p2,
        )

        self.play(
            LaggedStart(
                Create(rf_h_to_mixer_rf_bezier),
                Create(rf_l_to_mixer_rf_bezier),
                FadeIn(rf_plus),
                GrowFromCenter(rf_plus_to_rf_signal),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Uncreate(rf_h_to_mixer_rf_bezier),
            Uncreate(rf_l_to_mixer_rf_bezier),
            FadeOut(rf_plus),
            ShrinkToCenter(rf_plus_to_rf_signal),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_signal_copy = rf_signal.copy()
        lo_signal_copy = lo_signal.copy()
        if_signal_copy = if_signal.copy()

        rf_filt_ax_group = VGroup(rf_filt_ax, rf_filt_signal)
        rf_ax_group = VGroup(rf_ax, rf_signal_copy)
        if_ax_group = VGroup(if_ax, if_signal_copy)
        lo_ax_group = VGroup(lo_ax, lo_signal_copy)
        mixer_group_copy = mixer_group.copy()

        bd_left = VGroup(if_ax_group, lo_ax_group, mixer_group_copy)
        bd_left.save_state()
        bd_left.to_edge(LEFT, buff=MED_LARGE_BUFF)

        rf_filt_signal.next_to(mixer_group_copy, direction=RIGHT, buff=0)
        rf_filt_signal_copy = rf_filt_signal.copy()
        lp_filter_ntwk = Network("./data/LFCW-1062+_Plus25DegC_Unit1.s2p")
        lp_filter_plot = ax.plot_line_graph(
            lp_filter_ntwk.f / 1e9,
            lp_filter_ntwk.s_db[:, 1, 0] - y_min,
            add_vertex_dots=False,
            line_color=FILTER_COLOR,
        )
        lp_filter = (
            BLOCKS.get("lp_filter")
            .copy()
            .scale(block_scale)
            .next_to(rf_filt_signal, direction=RIGHT, buff=0)
        )
        bp_filter = BLOCKS.get("filter").copy().scale(block_scale).move_to(lp_filter)
        rf_ax_group_copy = rf_ax_group.next_to(lp_filter, direction=RIGHT, buff=0)

        # So let's add this filter
        self.play(
            Transform(if_signal, if_signal_copy),
            Transform(lo_signal, lo_signal_copy),
            Transform(mixer_group, mixer_group_copy),
            Transform(rf_signal, rf_signal_copy),
        )
        self.play(GrowFromCenter(bp_filter))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_l_top, rf_l_bot, rf_l_mid = get_f_rect(f_rf_l, rf_l_loss)
        rf_l_rect = Rectangle(
            width=1, height=Line(rf_l_bot, rf_l_top).height, color=FILTER_COLOR
        ).move_to(rf_l_mid)

        self.play(
            FadeIn(filter_line_legend, filter_legend, shift=LEFT),
            Create(rf_l_rect),
            rf_h_loss.animate.increment_value(3),
        )
        self.play(Create(rf_filt_signal))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Uncreate(rf_l_rect),
            Uncreate(rf_filt_signal),
            rf_h_loss.animate.increment_value(-3),
        )
        self.play(
            FadeOut(bp_filter, shift=DOWN),
            FadeIn(lp_filter, shift=DOWN),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        lp_filter_shift = DOWN * 5
        lp_filter_datasheet = (
            ImageMobject("../props/static/LFCW-1062+_datasheet.png")
            .scale(1.5)
            .to_edge(DOWN, buff=0)
            .shift(DOWN + lp_filter_shift)
        )
        lp_filter_label = (
            Tex("LFCW-1062+")
            .scale(0.8)
            .next_to(lp_filter_datasheet, direction=UP, buff=SMALL_BUFF)
        )
        lp_filter_datasheet_group = Group(lp_filter_datasheet, lp_filter_label)

        self.play(lp_filter_datasheet_group.animate.shift(-lp_filter_shift))

        self.wait(1)

        self.play(
            lp_filter_label.animate.next_to(lp_filter, direction=UP, buff=SMALL_BUFF),
            lp_filter_datasheet.animate.shift(lp_filter_shift),
        )
        self.remove(lp_filter_datasheet)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(FadeOut(f_if_eqn))
        self.play(Create(lp_filter_plot))
        self.play(rf_h_loss.animate.increment_value(3))
        self.play(Create(rf_filt_signal_copy))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        if_plot.remove_updater(if_plot_updater)
        rf_l_plot.remove_updater(rf_l_plot_updater)
        rf_h_plot.remove_updater(rf_h_plot_updater)
        lo_plot.remove_updater(lo_plot_updater)

        filter_section = Group(
            lp_filter, rf_filt_signal_copy, rf_signal_copy, lp_filter_label, rf_signal
        )
        bd_section = Group(
            *filter_section, mixer_group_copy, mixer_group, mixer, lo_signal, if_signal
        )
        all_except_filter_section = Group(*self.mobjects).remove(*bd_section)
        # all_except_filter_section.save_state()

        self.play(
            *get_fade_group(all_except_filter_section, opacity=0.2),
            self.camera.frame.animate.move_to(lp_filter)
            .shift(DOWN / 2)
            .scale_to_fit_width(filter_section.width * 1.2),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ideal_rf_input = rf_ax.plot(
            lambda t: np.sin(2 * PI * f_rf_l * t),
            x_range=x_range,
            color=RX_COLOR,
        ).next_to(
            rf_filt_signal_copy.get_midpoint(), direction=DOWN, buff=LARGE_BUFF * 1.2
        )
        ideal_rf_input_label = Tex("ideal RF input").next_to(
            ideal_rf_input, direction=DOWN, buff=MED_SMALL_BUFF
        )
        ideal_rf_input_arrow = Arrow(
            ideal_rf_input_label.get_left(),
            ideal_rf_input.get_right(),
        )

        ideal_if_signal = (
            if_ax.plot(
                lambda t: A * np.sin(2 * PI * f_lo * t) * (np.sin(2 * PI * f_rf_l * t)),
                x_range=x_range,
                color=IF_COLOR,
            )
            .next_to(if_signal.get_midpoint(), direction=DOWN, buff=LARGE_BUFF * 1.2)
            .set_y(ideal_rf_input.get_y())
        )

        ideal_if_label = Tex("ideal IF signal").next_to(
            ideal_if_signal, direction=DOWN, buff=MED_SMALL_BUFF
        )
        ideal_if_arrow = Arrow(
            ideal_if_label.get_left(),
            ideal_if_signal.get_right(),
        )

        self.play(
            TransformFromCopy(rf_filt_signal, ideal_rf_input),
            FadeIn(ideal_rf_input_label),
            # GrowArrow(ideal_rf_input_arrow),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(
                bd_section.remove(lo_signal)
            ).scale_to_fit_width(bd_section.width * 1.2),
        )

        self.play(
            TransformFromCopy(if_signal, ideal_if_signal),
            # GrowArrow(ideal_if_arrow),
            FadeIn(ideal_if_label),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # self.remove(lp_filter_plot)
        # all_except_filter_section.remove(lp_filter_plot)
        # TODO: Figure out how to fade these back in without filling plots
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(
                        ideal_if_label,
                        # ideal_if_arrow,
                        ideal_rf_input_label,
                        # ideal_rf_input_arrow,
                        # filter_legend,
                        # filter_line_legend,
                    ),
                    Uncreate(ideal_if_signal),
                    Uncreate(ideal_rf_input),
                    self.camera.frame.animate.move_to(ORIGIN).scale_to_fit_width(
                        config["frame_width"]
                    ),
                ),
                AnimationGroup(*get_fade_group(all_except_filter_section, opacity=1)),
                lag_ratio=0.6,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rf_l_plot.add_updater(rf_l_plot_updater)
        rf_h_plot.add_updater(rf_h_plot_updater)
        self.play(
            # FadeOut(filter_legend, filter_line_legend),
            Uncreate(lp_filter_plot, run_time=2),
            rf_h_loss.animate.increment_value(-3),
            # ShrinkToCenter(lp_filter),
            # Uncreate(rf_filt_signal_copy),
            # bd_left.animate.restore(),
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        ssb_label = (
            Tex(r"Single Sideband\\Mixer").scale(0.8).next_to(mixer, direction=DOWN)
        )
        self.play(FadeIn(ssb_label, shift=UP))

        rf_l_top, rf_l_bot, rf_l_mid = get_f_rect(f_rf_l, rf_l_loss)
        rf_l_rect = Rectangle(
            width=1, height=Line(rf_l_bot, rf_l_top).height, color=FILTER_COLOR
        ).move_to(rf_l_mid)
        rf_h_top, rf_h_bot, rf_h_mid = get_f_rect(f_rf_h, rf_h_loss)
        rf_h_rect = Rectangle(
            width=1, height=Line(rf_h_bot, rf_h_top).height, color=FILTER_COLOR
        ).move_to(rf_h_mid)

        rf_h_rect.save_state()

        self.play(Create(rf_h_rect), rf_l_loss.animate.increment_value(5))

        self.wait(0.5)

        self.play(
            Transform(rf_h_rect, rf_l_rect),
            rf_l_loss.animate.increment_value(-5),
            rf_h_loss.animate.increment_value(5),
        )

        high_side_eqn = Tex(r"High side\\$f_{LO} > f_{RF}$").to_edge(RIGHT)
        low_side_eqn = Tex(r"Low side \\ $f_{LO} < f_{RF}$").to_edge(RIGHT)

        self.wait(0.5)

        self.play(FadeIn(high_side_eqn, shift=LEFT))

        self.wait(0.5)

        self.play(
            FadeIn(low_side_eqn, shift=UP),
            FadeOut(high_side_eqn, shift=UP),
            rf_h_rect.animate.restore(),
            rf_l_loss.animate.increment_value(5),
            rf_h_loss.animate.increment_value(-5),
        )

        self.wait(0.5)

        self.play(
            FadeOut(low_side_eqn, shift=DOWN),
            FadeIn(high_side_eqn, shift=DOWN),
            Transform(rf_h_rect, rf_l_rect),
            rf_l_loss.animate.increment_value(-5),
            rf_h_loss.animate.increment_value(5),
        )

        self.wait(0.3)

        self.play(Indicate(lo_tick_label))

        self.wait(0.3)

        self.play(Indicate(rf_l_tick_label))

        self.wait(0.5)

        rf_l_plot.remove_updater(rf_l_plot_updater)
        rf_h_plot.remove_updater(rf_h_plot_updater)
        all_objects = Group(*self.mobjects).remove(
            rf_l_plot,
            rf_h_plot,
            lo_plot,
            if_plot,
            ax,
            if_signal_copy,
            rf_filt_signal_copy,
            rf_signal_copy,
            lo_signal_copy,
        )
        self.play(
            Uncreate(rf_l_plot),
            Uncreate(rf_h_plot),
            Uncreate(lo_plot),
            Uncreate(if_plot),
            Uncreate(ax),
            Uncreate(if_signal_copy),
            Uncreate(rf_filt_signal_copy),
            Uncreate(rf_signal_copy),
            Uncreate(lo_signal_copy),
            FadeOut(*all_objects),
        )

        self.wait(2)


class HardwareWrapUp(Scene):
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
            (
                input_to_pll,
                phase_detector,
                phase_detector_to_loop_filter,
                loop_filter,
                loop_filter_to_vco,
                vco,
                pll_block_to_pa,
                vco_output_conn,
                vco_to_ndiv_1,
                vco_to_ndiv_2,
                ndiv,
                ndiv_to_phase_detector_1,
                ndiv_to_phase_detector_2,
            ),
        ) = get_bd(True, rx_section_gap=BLOCK_BUFF * 5)
        bd.scale_to_fit_width(config["frame_width"] * 0.8).to_corner(
            UL, buff=LARGE_BUFF
        )  # .shift(UP / 2)

        pll_block.save_state()
        pll_block.scale_to_fit_width(config["frame_width"] * 0.5).move_to(ORIGIN)

        self.play(
            Create(input_to_pll),
            Create(phase_detector_to_loop_filter),
            Create(loop_filter_to_vco),
            Create(pll_block_to_pa),
            Create(vco_to_ndiv_1),
            Create(vco_to_ndiv_2),
            Create(ndiv_to_phase_detector_1),
            Create(ndiv_to_phase_detector_2),
            Create(vco_output_conn),
            GrowFromCenter(phase_detector),
            GrowFromCenter(loop_filter),
            GrowFromCenter(vco),
            GrowFromCenter(ndiv),
            run_time=1.5,
        )

        input_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=3,
            y_length=1.5,
        ).next_to(input_to_pll, direction=LEFT)
        sine_plot = input_ax.plot(lambda t: np.sin(2 * PI * 3 * t), color=TX_COLOR)

        fm_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=3,
            y_length=1.5,
        ).next_to(pll_block_to_pa, direction=RIGHT)

        carrier_freq = 10
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        fs = 5000
        A = 1

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

        sine_plot = input_ax.plot(lambda t: np.sin(2 * PI * 3 * t), color=TX_COLOR)
        sine_plot_arrow = Arrow(
            ORIGIN, RIGHT * sine_plot.width, color=TX_COLOR
        ).next_to(sine_plot, direction=DOWN)
        fm_plot = fm_ax.plot(
            sawtooth_amp,
            x_range=[0, 1, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )
        fm_plot_arrow = Arrow(ORIGIN, RIGHT * fm_plot.width, color=TX_COLOR).next_to(
            fm_plot, direction=DOWN
        )

        self.play(Create(sine_plot), GrowArrow(sine_plot_arrow))
        self.play(Create(fm_plot), GrowArrow(fm_plot_arrow))

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(
            Uncreate(sine_plot),
            Uncreate(fm_plot),
            FadeOut(sine_plot_arrow, fm_plot_arrow),
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(pll_block.animate.restore())
        self.play(GrowFromCenter(pa), Create(pa_to_splitter))
        self.play(
            GrowFromCenter(splitter),
            Create(splitter_to_tx_antenna),
        )

        self.wait(0.3)

        tx_arrow = Arrow(
            tx_antenna.get_corner(UR),
            tx_antenna.get_corner(UR) + [3, 2, 0],
            color=TX_COLOR,
        )
        rx_arrow = Arrow(
            rx_antenna.get_corner(UR) + [3, 2, 0],
            rx_antenna.get_corner(UR),
            color=RX_COLOR,
        )

        self.play(
            LaggedStart(
                GrowFromCenter(tx_antenna),
                GrowArrow(tx_arrow),
                GrowArrow(rx_arrow),
                GrowFromCenter(rx_antenna),
                Create(rx_antenna_to_lna),
                GrowFromCenter(lna),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(splitter_to_mixer),
                    Create(lna_to_mixer),
                ),
                GrowFromCenter(mixer),
                Create(mixer_to_lp_filter),
                GrowFromCenter(lp_filter),
                Create(lp_filter_to_adc),
                lag_ratio=0.5,
            )
        )

        self.wait(0.2)

        f_beat_dot = Dot(lp_filter_to_adc.get_end())
        f_beat = MathTex(r"f_{beat}").next_to(
            f_beat_dot, direction=LEFT, buff=LARGE_BUFF
        )
        f_beat_arrow = Arrow(f_beat_dot.get_center(), f_beat.get_right())

        self.play(
            LaggedStart(
                Create(f_beat_dot),
                GrowArrow(f_beat_arrow),
                FadeIn(f_beat),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        duration = 1
        x_len = 3
        y_len = 1
        amp_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        f1 = 3
        f2 = 3.2
        f3 = 2.8
        p1 = 0
        p2 = PI / 3
        p3 = PI / 6
        f_beat_plot_1 = amp_ax.plot(
            lambda t: np.sin(2 * PI * f1 * t + p1), color=IF_COLOR
        ).next_to(f_beat, direction=LEFT, buff=LARGE_BUFF)
        f_beat_plot_2 = amp_ax.plot(
            lambda t: np.sin(2 * PI * f2 * t + p2), color=IF_COLOR
        ).next_to(f_beat_plot_1, direction=UP)
        f_beat_plot_3 = amp_ax.plot(
            lambda t: np.sin(2 * PI * f3 * t + p3), color=IF_COLOR
        ).next_to(f_beat_plot_1, direction=DOWN)

        f_beat_plot_sum = amp_ax.plot(
            lambda t: np.sin(2 * PI * f1 * t + p1)
            + np.sin(2 * PI * f2 * t + p2)
            + np.sin(2 * PI * f3 * t + p3),
            color=IF_COLOR,
        ).move_to(f_beat_plot_1)

        f_beat_1_line = Line(
            f_beat.get_left() + [-0.2, 0, 0], f_beat_plot_1.get_right() + [0.2, 0, 0]
        )

        f_beat_2_line_p1 = f_beat.get_left() + [-0.2, 0, 0]
        f_beat_2_line_p2 = f_beat_plot_2.get_right() + [0.2, 0, 0]
        f_beat_2_bez = CubicBezier(
            f_beat_2_line_p1,
            f_beat_2_line_p1 + [-0.5, 0, 0],
            f_beat_2_line_p2 + [0.5, 0, 0],
            f_beat_2_line_p2,
        )

        f_beat_3_line_p1 = f_beat.get_left() + [-0.2, 0, 0]
        f_beat_3_line_p2 = f_beat_plot_3.get_right() + [0.2, 0, 0]
        f_beat_3_bez = CubicBezier(
            f_beat_3_line_p1,
            f_beat_3_line_p1 + [-0.5, 0, 0],
            f_beat_3_line_p2 + [0.5, 0, 0],
            f_beat_3_line_p2,
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(f_beat_1_line), Create(f_beat_plot_1), lag_ratio=0.3
                ),
                LaggedStart(Create(f_beat_2_bez), Create(f_beat_plot_2), lag_ratio=0.3),
                LaggedStart(Create(f_beat_3_bez), Create(f_beat_plot_3), lag_ratio=0.3),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        f_beat_plot_parts = VGroup(f_beat_plot_1, f_beat_plot_2, f_beat_plot_3)
        self.play(
            Transform(f_beat_plot_parts, f_beat_plot_sum),
            Uncreate(f_beat_2_bez),
            Uncreate(f_beat_3_bez),
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(
            Uncreate(input_to_pll),
            Uncreate(phase_detector_to_loop_filter),
            Uncreate(loop_filter_to_vco),
            Uncreate(pll_block_to_pa),
            Uncreate(vco_to_ndiv_1),
            Uncreate(vco_to_ndiv_2),
            Uncreate(ndiv_to_phase_detector_1),
            Uncreate(ndiv_to_phase_detector_2),
            Uncreate(vco_output_conn),
            Uncreate(pa_to_splitter),
            Uncreate(splitter_to_tx_antenna),
            Uncreate(splitter_to_mixer),
            Uncreate(rx_antenna_to_lna),
            Uncreate(lna_to_mixer),
            Uncreate(mixer_to_lp_filter),
            Uncreate(lp_filter_to_adc),
            Uncreate(f_beat_dot),
            Uncreate(f_beat_1_line),
            ShrinkToCenter(phase_detector),
            ShrinkToCenter(loop_filter),
            ShrinkToCenter(vco),
            ShrinkToCenter(ndiv),
            ShrinkToCenter(splitter),
            ShrinkToCenter(pa),
            ShrinkToCenter(mixer),
            ShrinkToCenter(lna),
            ShrinkToCenter(lp_filter),
            ShrinkToCenter(tx_antenna),
            ShrinkToCenter(rx_antenna),
            FadeOut(tx_arrow, rx_arrow, f_beat, f_beat_arrow),
        )

        radar_plot = PolarPlane(
            azimuth_step=4,
            size=config["frame_width"] / 4,
            radius_config={
                "stroke_color": WHITE,
                "include_tip": False,
            },
            azimuth_direction="CW",
        ).next_to(ORIGIN, direction=RIGHT, buff=LARGE_BUFF)

        rmax = radar_plot.get_y_range().max()

        radar_scan_vel = ValueTracker(0)
        radar_scan_angle = ValueTracker(-PI)

        scan_line = Line(
            radar_plot.pr2pt(0, 0),
            radar_plot.pr2pt(rmax, radar_scan_angle.get_value()),
            color=RED,
        )

        def scan_updater(m: Mobject, dt):
            radar_scan_angle.set_value(
                radar_scan_angle.get_value() + radar_scan_vel.get_value() * dt
            )
            m.become(
                Line(
                    radar_plot.pr2pt(0, 0),
                    radar_plot.pr2pt(rmax, radar_scan_angle.get_value()),
                    color=RED,
                )
            )

        def show_dot_updater(m: Mobject):
            dot_angle = radar_plot.pt2pr([m.get_x(), m.get_y(), 0])[1]
            if radar_scan_angle.get_value() >= dot_angle:
                m.set_opacity(1)
            else:
                m.set_opacity(0)

        targets = VGroup()
        for r, theta in [
            [rmax * random(), 30 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 54 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 92 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 100 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 140 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 200 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 230 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 305 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 290 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 350 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 300 * DEGREES + random() * 6 * DEGREES],
        ]:
            dot = Dot(radar_plot.pr2pt(r, theta))
            dot.add_updater(show_dot_updater)
            targets.add(dot)

        self.add(targets)
        self.play(
            Create(radar_plot),
            Create(scan_line),
            f_beat_plot_parts.animate.next_to(ORIGIN, direction=LEFT, buff=LARGE_BUFF),
        )

        radar_scan_vel.set_value(1)
        scan_line.add_updater(scan_updater)

        self.wait(4)

        block_buff_scale = 0.6

        computer = (
            BLOCKS.get("computer")
            .copy()
            .next_to(ORIGIN, buff=BLOCK_BUFF * block_buff_scale / 2)
        )
        mystery_block = Square().next_to(
            computer, direction=LEFT, buff=BLOCK_BUFF * block_buff_scale
        )
        mystery_qmark = Tex("?", color=WHITE).scale(1.5).move_to(mystery_block)
        mystery_block_group = Group(mystery_block, mystery_qmark)

        radar_plot_group = VGroup(scan_line, radar_plot, targets)
        self.play(
            GrowFromCenter(computer),
            LaggedStart(
                DrawBorderThenFill(mystery_block),
                FadeIn(mystery_qmark),
            ),
            f_beat_plot_parts.animate.next_to(
                mystery_block, direction=LEFT, buff=BLOCK_BUFF * block_buff_scale
            ),
            radar_plot_group.animate.scale_to_fit_width(
                f_beat_plot_parts.width
            ).next_to(computer, direction=RIGHT, buff=BLOCK_BUFF * block_buff_scale),
        )

        f_beat_to_mystery = Arrow(
            f_beat_plot_parts.get_right(), mystery_block.get_left()
        )
        mystery_to_computer = Arrow(mystery_block.get_right(), computer.get_left())
        out_of_computer = Arrow(computer.get_right(), radar_plot_group.get_left())

        self.play(
            LaggedStart(
                GrowArrow(f_beat_to_mystery),
                GrowArrow(mystery_to_computer),
                GrowArrow(out_of_computer),
                lag_ratio=0.6,
            )
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(Group(*self.mobjects).animate.shift(UP * 1.5))

        python_computer = (
            BLOCKS.get("computer")
            .set_fill(GREEN)
            .copy()
            .scale(0.8)
            .next_to(computer, direction=DOWN, buff=LARGE_BUFF * 1.5)
        )
        python_logo = (
            ImageMobject("../props/static/python-logo-only.png")
            .scale_to_fit_height(python_computer.height * 0.65)
            .move_to(python_computer)
            # .shift(UP / 6)
        )
        python_computer_group = Group(python_computer, python_logo)
        mcu = (
            ImageMobject("../props/static/microcontroller.png")
            .scale_to_fit_width(computer.width)
            .next_to(python_computer_group, direction=LEFT, buff=MED_SMALL_BUFF)
        )
        mcu_label = Tex(r"$\mu$ Controller").next_to(
            mcu, direction=DOWN, buff=SMALL_BUFF
        )
        mcu_group = Group(mcu, mcu_label)
        fpga = (
            ImageMobject("../props/static/mcu.png")
            .scale_to_fit_width(computer.width)
            .next_to(python_computer_group, direction=RIGHT, buff=MED_SMALL_BUFF)
        )
        fpga_label = Tex(r"FPGA").next_to(fpga, direction=DOWN, buff=SMALL_BUFF)
        fpga_group = Group(fpga, fpga_label)

        computer_bottom = computer.get_bottom() + [0, -0.1, 0]
        python_computer_p1 = python_computer_group.get_top() + [0, 0.1, 0]
        python_bez = CubicBezier(
            computer_bottom,
            computer_bottom + [0, -0.5, 0],
            python_computer_p1 + [0, 0.5, 0],
            python_computer_p1,
        )

        fpga_p1 = fpga_group.get_top() + [0, 0.1, 0]
        fpga_bez = CubicBezier(
            computer_bottom,
            computer_bottom + [0, -0.5, 0],
            fpga_p1 + [0, 0.5, 0],
            fpga_p1,
        )

        mcu_p1 = mcu_group.get_top() + [0, 0.1, 0]
        mcu_bez = CubicBezier(
            computer_bottom,
            computer_bottom + [0, -0.5, 0],
            mcu_p1 + [0, 0.5, 0],
            mcu_p1,
        )

        self.play(
            LaggedStart(
                LaggedStart(Create(python_bez), GrowFromCenter(python_computer_group)),
                LaggedStart(Create(fpga_bez), GrowFromCenter(fpga_group)),
                LaggedStart(Create(mcu_bez), GrowFromCenter(mcu_group)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        analog_label = (
            Tex("Analog")
            .next_to(f_beat_to_mystery, direction=UP, buff=LARGE_BUFF * 1.5)
            .shift(LEFT / 2)
        )
        digital_label = (
            Tex("Digital")
            .next_to(mystery_to_computer, direction=UP, buff=LARGE_BUFF * 1.5)
            .shift(RIGHT / 2)
        )

        analog_p1 = f_beat_to_mystery.get_top() + [0, 0.1, 0]
        analog_p2 = analog_label.get_bottom() + [0, -0.1, 0]
        analog_bez = CubicBezier(
            analog_p1,
            analog_p1 + [0, 0.5, 0],
            analog_p2 + [0, -0.5, 0],
            analog_p2,
        )

        digital_p1 = mystery_to_computer.get_top() + [0, 0.1, 0]
        digital_p2 = digital_label.get_bottom() + [0, -0.1, 0]
        digital_bez = CubicBezier(
            digital_p1,
            digital_p1 + [0, 0.5, 0],
            digital_p2 + [0, -0.5, 0],
            digital_p2,
        )

        self.play(
            LaggedStart(
                LaggedStart(Create(analog_bez), FadeIn(analog_label), lag_ratio=0.4),
                LaggedStart(Create(digital_bez), FadeIn(digital_label), lag_ratio=0.4),
                lag_ratio=0.8,
            )
        )

        self.wait(0.5)

        adc = (
            BLOCKS.get("adc")
            .copy()
            .scale_to_fit_width(mystery_block.width)
            .move_to(mystery_block_group)
        )
        adc_label = Tex("ADC").move_to(adc)
        adc_group = Group(adc, adc_label)

        self.play(
            FadeOut(mystery_block_group, shift=DOWN),
            FadeIn(adc_group, shift=DOWN),
        )

        self.wait(0.5)

        remove_group = Group(*self.mobjects)

        sequence = VGroup(
            *[
                Text(
                    "".join([f"{n}" for n in list(np.random.randint(0, 2, 40))]),
                    disable_ligatures=True,
                    font="FiraCode Nerd Font Mono",
                    color=GREEN,
                    stroke_opacity=1,
                    # fill_color=BACKGROUND_COLOR,
                    # fill_opacity=1,
                ).set_z_index(2)
                for _ in range(13)
            ]
        ).arrange(direction=DOWN, buff=MED_SMALL_BUFF)
        sequence.next_to(
            LEFT * self.camera.frame_width / 2, direction=LEFT, buff=SMALL_BUFF
        )
        for s in sequence:
            s.shift(random() * 2 * LEFT)
        sequence_backgrounds = VGroup(
            *[
                BackgroundRectangle(
                    m, fill_color=BACKGROUND_COLOR, fill_opacity=1, buff=0
                )
                .stretch((m.height + SMALL_BUFF * 2.6) / m.height, dim=1)
                .set_z_index(1)
                for m in sequence
            ]
        )
        sequence_group = VGroup(sequence_backgrounds, sequence)

        part_3 = Tex("Part 3: Signal Processing").scale(2)

        self.add(sequence_group)

        self.play(
            LaggedStart(
                sequence_group.animate(
                    rate_func=rate_functions.linear, run_time=3.5
                ).shift(config["frame_width"] * RIGHT * 2.3),
                FadeOut(remove_group),
                Create(part_3),
                lag_ratio=0.4,
            )
        )

        self.wait(1)

        self.play(FadeOut(part_3))

        self.wait(2)


class Digitization(MovingCameraScene):
    def construct(self):
        x_len = 4.5
        y_len = 3
        duration = 1
        time_ax = Axes(
            x_range=[0, duration, duration / 4],
            y_range=[-1.2, 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        time_ax_label = time_ax.get_axis_labels(Tex("$t$"), Tex(""))

        stop_time = 4
        fs = 1000
        N = fs * stop_time
        f_if = 4
        t = np.linspace(0, stop_time, N)
        if_loss = 10
        f_max = 20
        y_min = -28

        f_ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -y_min, -y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
                # "include_ticks": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        f_ax_label = f_ax.get_axis_labels(Tex("$f$"), Tex(r"$\lvert X(f) \rvert$"))

        def get_plot_values(y_min=None):
            if_signal = np.sin(2 * PI * f_if * t) / (10 ** (if_loss / 10))

            blackman_window = signal.windows.blackman(N)
            if_signal *= blackman_window

            fft_len = 2**20
            summed_fft = np.fft.fft(if_signal, fft_len) / (N / 2)
            # summed_fft /= summed_fft.max()
            summed_fft_log = 10 * np.log10(np.fft.fftshift(summed_fft))
            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = summed_fft_log[indices]

            if y_min is not None:
                y_values[y_values < y_min] = y_min
                y_values -= y_min

            return dict(x_values=x_values, y_values=y_values)

        f = 1.5
        if_signal = time_ax.plot(lambda t: np.sin(2 * PI * f * t), color=IF_COLOR)

        if_freq = f_ax.plot_line_graph(
            **get_plot_values(y_min=y_min),
            add_vertex_dots=False,
            line_color=IF_COLOR,
        )

        time_ax_group = VGroup(time_ax, time_ax_label, if_signal)
        f_ax_group = VGroup(f_ax, f_ax_label, if_freq)
        VGroup(time_ax_group, f_ax_group).arrange(buff=LARGE_BUFF * 1.5).to_edge(
            DOWN, buff=MED_LARGE_BUFF
        )

        beat_signal_label = (
            Tex("Beat Signal").scale(1.2).to_edge(UP, buff=MED_LARGE_BUFF)
        )
        digitized_beat_signal_label = (
            Tex("Digitized ", "Beat Signal").scale(1.2).move_to(beat_signal_label)
        )

        beat_signal_p1 = beat_signal_label.get_bottom() + [0, -0.1, 0]
        time_ax_p2 = time_ax_group.get_top() + [0, 0.1, 0]
        f_ax_p2 = f_ax_group.get_top() + [0, 0.1, 0]
        time_ax_bez = CubicBezier(
            beat_signal_p1,
            beat_signal_p1 + [0, -1, 0],
            time_ax_p2 + [0, 1, 0],
            time_ax_p2,
        )
        f_ax_bez = CubicBezier(
            beat_signal_p1,
            beat_signal_p1 + [0, -1, 0],
            f_ax_p2 + [0, 1, 0],
            f_ax_p2,
        )

        # self.play(Create(time_ax), Create(signal))

        self.next_section(skip_animations=False)

        self.play(Create(beat_signal_label))
        self.play(
            LaggedStart(
                LaggedStart(
                    Create(time_ax_bez),
                    AnimationGroup(Create(time_ax), FadeIn(time_ax_label)),
                    Create(if_signal),
                    lag_ratio=0.4,
                ),
                LaggedStart(
                    Create(f_ax_bez),
                    AnimationGroup(Create(f_ax), FadeIn(f_ax_label)),
                    Create(if_freq),
                    lag_ratio=0.4,
                ),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(time_ax_bez),
                    Uncreate(f_ax_bez),
                    Uncreate(f_ax),
                    FadeOut(f_ax_label),
                    Uncreate(if_freq),
                ),
                time_ax_group.animate.move_to(ORIGIN),
                lag_ratio=0.5,
            )
        )

        adc = BLOCKS.get("adc").copy()
        adc_label = Tex("ADC").move_to(adc)
        adc_group = Group(adc, adc_label).scale(0.8)

        self.camera.frame.save_state()

        self.play(
            time_ax_group.animate.to_edge(LEFT, buff=SMALL_BUFF),
            GrowFromCenter(adc_group),
        )
        self.play(self.camera.frame.animate.scale(0.8).move_to(time_ax_group))

        self.wait(0.5)

        samples = time_ax.get_vertical_lines_to_graph(
            if_signal, x_range=[0, duration], num_lines=15, color=BLUE
        )

        v_readings = VGroup(
            *[
                Tex(f"$v_{{{idx}}}$")
                .next_to(time_ax_group, direction=UP, buff=MED_SMALL_BUFF)
                .set_x(sample.get_end()[0])
                .shift(UP / 2 if idx % 2 == 0 else 0)
                for idx, sample in enumerate(samples[:5])
            ]
        )
        v_readings.add(
            *[
                Tex(".")
                .next_to(time_ax_group, direction=UP, buff=MED_SMALL_BUFF)
                .set_x(sample.get_end()[0])
                for idx, sample in enumerate(samples[8:11])
            ]
        )
        v_readings.add(
            Tex(f"$v_{{N}}$")
            .next_to(time_ax_group, direction=UP, buff=MED_SMALL_BUFF)
            .set_x(samples[-1].get_end()[0])
        )

        sample_1 = samples[7].get_end()
        sample_2 = samples[8].get_end()

        sample_1_p2 = sample_1 + [-1, -1, 0]
        sample_1_bez = CubicBezier(
            sample_1,
            sample_1 + [0, -0.5, 0],
            sample_1_p2 + [0, 0.5, 0],
            sample_1_p2,
        )

        sample_period = MathTex(r"T_s = \frac{1}{f_s}").next_to(
            sample_1_p2, buff=MED_SMALL_BUFF
        )
        sample_2_p2 = Dot().next_to(sample_period, buff=MED_SMALL_BUFF).get_center()
        sample_2_bez = CubicBezier(
            sample_2,
            sample_2 + [0, -0.5, 0],
            sample_2_p2 + [0, 0.5, 0],
            sample_2_p2,
        )

        n_ax = time_ax.copy()
        n_ax_label = n_ax.get_axis_labels(Tex("$n$"), Tex(""))
        if_signal_n_ax = n_ax.plot(lambda t: np.sin(2 * PI * f * t), color=IF_COLOR)
        n_ax_group = VGroup(n_ax, n_ax_label, if_signal_n_ax).to_edge(
            RIGHT, buff=SMALL_BUFF
        )

        self.play(Create(samples), Create(v_readings), run_time=2)

        self.wait(0.5)

        self.play(Create(sample_1_bez), Create(sample_2_bez), FadeIn(sample_period))

        self.next_section(skip_animations=False)
        self.wait(0.5)

        self.play(
            Uncreate(sample_1_bez),
            Uncreate(sample_2_bez),
            sample_period.animate.next_to(
                adc_group, direction=DOWN, buff=MED_SMALL_BUFF
            ),
            Uncreate(v_readings),
            self.camera.frame.animate.restore(),
        )

        self.wait(0.2)

        to_adc = Arrow(time_ax_group.get_right(), adc_group.get_left())
        from_adc = Arrow(adc_group.get_right(), n_ax_group.get_left())

        samples_n_ax = n_ax.get_vertical_lines_to_graph(
            if_signal_n_ax, x_range=[0, duration], num_lines=15, color=BLUE
        )

        self.play(
            Create(n_ax),
            FadeIn(n_ax_label),
            samples.animate.move_to(samples_n_ax),
            LaggedStart(GrowArrow(to_adc), GrowArrow(from_adc), lag_ratio=0.5),
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        time_signal_label = MathTex("x(", "t", ")").next_to(
            time_ax_group, direction=UP, buff=MED_SMALL_BUFF
        )
        n_signal_label = MathTex("x[", "n", "]").next_to(
            n_ax_group, direction=UP, buff=MED_SMALL_BUFF
        )

        self.play(
            LaggedStart(
                FadeIn(time_signal_label[0]),
                TransformFromCopy(time_ax_label[0], time_signal_label[1]),
                FadeIn(time_signal_label[2]),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(n_signal_label[0]),
                TransformFromCopy(n_ax_label[0], n_signal_label[1]),
                FadeIn(n_signal_label[2]),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(Create(if_signal_n_ax))

        self.wait(0.5)

        n_ax_group_all = Group(
            n_ax_group,
            samples,
            n_signal_label,
        )

        self.play(
            FadeOut(time_ax_group, adc_group, to_adc, from_adc, time_signal_label),
            n_ax_group_all.animate.move_to(ORIGIN),
            sample_period.animate.to_edge(DOWN, buff=MED_LARGE_BUFF),
            FadeIn(digitized_beat_signal_label[0]),
            Transform(beat_signal_label, digitized_beat_signal_label[1]),
        )

        self.wait(0.3)

        self.play(
            FadeOut(sample_period, shift=DOWN),
            FadeOut(beat_signal_label, digitized_beat_signal_label[0], shift=UP),
        )

        self.next_section(skip_animations=False)
        self.wait(0.5)

        computer = BLOCKS.get("computer").copy()
        computer_eyes = VGroup(Dot(), Dot()).arrange().move_to(computer).shift(UP * 0.4)
        computer_smile = Arc(start_angle=-3 * TAU / 8).move_to(computer).scale(0.7)
        computer_group = Group(computer, computer_eyes, computer_smile).to_edge(
            RIGHT, buff=LARGE_BUFF * 1.5
        )

        self.play(
            FadeIn(computer, shift=LEFT * 2),
            n_ax_group_all.animate.to_edge(LEFT, buff=LARGE_BUFF * 1.5),
        )

        to_computer = Arrow(n_ax_group.get_right(), computer.get_left())

        self.play(
            GrowArrow(to_computer),
            Create(computer_eyes),
            Create(computer_smile),
        )

        self.wait(0.5)

        disclaimer = Tex(r"\textit{disclaimer}").scale(2)

        self.play(
            LaggedStart(FadeOut(*self.mobjects), Create(disclaimer), lag_ratio=0.7)
        )

        self.wait(2)


class Disclaimer(Scene):
    def construct(self):
        disclaimer = Tex(r"\textit{disclaimer}").scale(2)
        disclaimer_corner = (
            disclaimer.copy().scale(0.7).to_corner(UL, buff=MED_SMALL_BUFF)
        )
        disclaimer_box = SurroundingRectangle(
            disclaimer_corner, buff=MED_SMALL_BUFF * 1.2
        )

        self.add(disclaimer)

        self.play(
            LaggedStart(
                Transform(disclaimer, disclaimer_corner),
                Create(disclaimer_box),
                lag_ratio=0.8,
            )
        )

        self.wait(0.5)

        in_this_video = (
            Tex("In this video:").to_edge(LEFT, buff=LARGE_BUFF).shift(UP * 2)
        )
        in_this_video_bar = Line(
            in_this_video.get_corner(DL), in_this_video.get_corner(DR)
        ).next_to(in_this_video, direction=DOWN, buff=SMALL_BUFF)
        range_calc = BulletedList(
            r"Range calculation \\ $\left( R = \frac{c T_{c} f_{beat}}{2 B} \right)$"
        ).next_to(
            in_this_video_bar, direction=DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF
        )

        not_in_this_video = (
            Tex("NOT", " in this video:").to_edge(RIGHT, buff=LARGE_BUFF).shift(UP * 2)
        ).set_color_by_tex("NOT", RED)
        not_in_this_video_bar = Line(
            not_in_this_video.get_corner(DL), not_in_this_video.get_corner(DR)
        ).next_to(not_in_this_video, direction=DOWN, buff=SMALL_BUFF)
        not_in_this_video_list = BulletedList(
            "Velocity calculation",
            "Clutter filtering",
            "CFAR",
            "Monopulse tracking",
            "...",
        ).next_to(
            not_in_this_video_bar,
            direction=DOWN,
            aligned_edge=LEFT,
            buff=MED_LARGE_BUFF,
        )

        gloss_over = Tex("I'll gloss over:").shift(UP * 2 + LEFT * 2)
        gloss_over_bar = Line(
            gloss_over.get_corner(DL), gloss_over.get_corner(DR)
        ).next_to(gloss_over, direction=DOWN, buff=SMALL_BUFF)
        gloss_over_list = BulletedList(
            "Sampling theory", "Fourier transform", "Window functions", "..."
        ).next_to(
            gloss_over_bar,
            direction=DOWN,
            aligned_edge=LEFT,
            buff=MED_LARGE_BUFF,
        )

        fft_video = (
            ImageMobject("../props/static/fourier_transform_video_3b1b.jpg")
            .scale_to_fit_width(4)
            .next_to(gloss_over_list, direction=RIGHT, buff=LARGE_BUFF * 0.9)
            .shift(DOWN * 2.5)
        )
        fft_video_label = (
            Tex(
                r"But what is the Fourier Transform?\\A visual introduction.\\- 3blue1brown"
            )
            .scale(0.4)
            .next_to(fft_video, direction=UP, buff=SMALL_BUFF)
        )
        sampling_video = (
            ImageMobject("../props/static/sampling_signals_video.jpg")
            .scale_to_fit_width(4)
            .next_to(gloss_over_list, direction=RIGHT, buff=LARGE_BUFF * 1.5)
            .shift(UP * 1.5)
        )
        sampling_video_label = (
            Tex(
                r"Sampling Signals:\\Introduction Lecture\\Iain Explains Signals, Systems, and Digital Comms"
            )
            .scale(0.4)
            .next_to(sampling_video, direction=UP, buff=SMALL_BUFF)
        )

        fft_video_p1 = gloss_over_list[1].get_right() + [0.1, 0, 0]
        fft_video_p2 = fft_video.get_left() + [-0.1, 0, 0]
        fft_video_bez = CubicBezier(
            fft_video_p1,
            fft_video_p1 + [0.5, 0, 0],
            fft_video_p2 + [-0.5, 0, 0],
            fft_video_p2,
        )

        sampling_video_p1 = gloss_over_list[0].get_right() + [0.1, 0, 0]
        sampling_video_p2 = sampling_video.get_left() + [-0.1, 0, 0]
        sampling_video_bez = CubicBezier(
            sampling_video_p1,
            sampling_video_p1 + [0.5, 0, 0],
            sampling_video_p2 + [-0.5, 0, 0],
            sampling_video_p2,
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(in_this_video), Create(in_this_video_bar)),
                Create(range_calc),
                lag_ratio=0.7,
            )
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(not_in_this_video), Create(not_in_this_video_bar)
                ),
                Create(not_in_this_video_list, run_time=2),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                in_this_video,
                in_this_video_bar,
                range_calc,
                shift=LEFT,
            ),
            FadeOut(
                not_in_this_video,
                not_in_this_video_bar,
                not_in_this_video_list,
                shift=RIGHT,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(Create(gloss_over), Create(gloss_over_bar)),
                Create(gloss_over_list, run_time=2),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    AnimationGroup(Create(fft_video_bez), FadeIn(fft_video_label)),
                    GrowFromCenter(fft_video),
                    lag_ratio=0.5,
                ),
                LaggedStart(
                    AnimationGroup(
                        Create(sampling_video_bez), FadeIn(sampling_video_label)
                    ),
                    GrowFromCenter(sampling_video),
                    lag_ratio=0.5,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(disclaimer),
            Uncreate(disclaimer_box),
            Uncreate(fft_video_bez),
            Uncreate(sampling_video_bez),
            ShrinkToCenter(fft_video),
            ShrinkToCenter(sampling_video),
            FadeOut(
                gloss_over,
                gloss_over_bar,
                gloss_over_list,
                fft_video_label,
                sampling_video_label,
            ),
        )

        self.wait(2)


class IFSignalComponents(Scene):
    def construct(self):
        seed(1)

        range_eqn = Tex(r"$R = \frac{c T_{c} f_{beat}}{2 B}$").scale(1.8)
        f_beat = MathTex(r"f_{beat}").scale(1.8).to_corner(UR)
        # tex_labels = index_labels(range_eqn[0])

        radar = FMCWRadarCartoon()
        target_2 = Square(side_length=0.7).to_corner(UR, buff=MED_SMALL_BUFF)
        target_1 = (
            Square(side_length=0.8)
            .next_to(target_2, direction=DOWN, buff=SMALL_BUFF)
            .shift(LEFT / 3)
        )
        target_1_label = Tex("1").move_to(target_1)
        target_2_label = Tex("2").move_to(target_2)
        radar.vgroup.scale(0.4).next_to(
            Group(target_1, target_2), direction=LEFT, buff=LARGE_BUFF * 2
        )
        radar_beam_l = Line(
            radar.antenna_tx.get_right(),
            target_1.get_corner(DL) + [0, -0.1, 0],
            color=TX_COLOR,
        )
        radar_beam_r = Line(
            radar.antenna_tx.get_right(),
            [target_1.get_corner(DL)[0], target_2.get_corner(UL)[1] + 0.1, 0],
            color=TX_COLOR,
        )
        target_1_reflection = Arrow(
            target_1.get_left(), radar.antenna_rx.get_right(), color=RX_COLOR
        )
        target_2_reflection = Arrow(
            target_2.get_left(), radar.antenna_rx.get_right(), color=RX_COLOR
        )
        ground_clutter = Line(
            [
                radar.antenna_rx.get_center()[0],
                (target_1.get_corner(DR) + [0.2, -0.1, 0])[1],
                0,
            ],
            target_1.get_corner(DR) + [0.2, -0.1, 0],
        )
        ground_clutter_reflection = Arrow(
            ground_clutter.get_midpoint(), radar.antenna_rx.get_right(), color=RX_COLOR
        )

        # self.add(
        #     radar.vgroup,
        #     radar_beam_l,
        #     radar_beam_r,
        #     target_1_label,
        #     target_2_label,
        #     target_1,
        #     target_2,
        #     target_1_reflection,
        #     target_2_reflection,
        #     ground_clutter,
        #     ground_clutter_reflection,
        # )

        x_len = 4.5
        y_len = 2.5
        duration = 1
        time_ax = (
            Axes(
                x_range=[0, duration, duration / 4],
                y_range=[-1.2, 1.2, 0.5],
                tips=False,
                axis_config={"include_numbers": False},
                x_length=x_len,
                y_length=y_len,
            )
            .to_corner(DL, buff=MED_LARGE_BUFF)
            .shift(UP)
        )
        time_ax_label = time_ax.get_axis_labels(Tex("$t$"), Tex(""))

        stop_time = 16
        fs = 1000
        N = fs * stop_time
        f_if = 4
        t = np.linspace(0, stop_time, N)
        if_loss = 10
        f_max = 8
        y_min = -40

        f_ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -y_min, -y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        f_ax_label = f_ax.get_axis_labels(Tex("$f$"), Tex(r"$\lvert X(f) \rvert$"))

        def get_fft_values(x_n, fs, stop_time, fft_len=2**18, f_max=20, y_min=None):
            N = stop_time * fs

            X_k = fft(x_n, fft_len) / (N / 2)
            X_k = 10 * np.log10(fftshift(X_k))
            X_k -= X_k.max()

            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = X_k[indices]

            if y_min is not None:
                y_values += -y_min
                y_values[y_values < 0] = 0

            return x_values, y_values

        f1 = 1.5
        f2 = 2.7
        beat_signal_1 = time_ax.plot(lambda t: np.sin(2 * PI * f1 * t), color=IF_COLOR)
        beat_signal_2 = time_ax.plot(
            lambda t: np.sin(2 * PI * f2 * t), color=IF_COLOR
        ).next_to(beat_signal_1, direction=UP, buff=MED_LARGE_BUFF)
        beat_signal_group = VGroup(beat_signal_1, beat_signal_2)
        beat_signals = time_ax.plot(
            lambda t: 0.5 * np.sin(2 * PI * f1 * t) + 0.5 * np.sin(2 * PI * f2 * t),
            color=IF_COLOR,
        )

        noise_sigma = 0.2
        noise = time_ax.plot(
            lambda t: normalvariate(mu=0, sigma=noise_sigma), color=IF_COLOR
        ).next_to(beat_signal_1, direction=UP, buff=MED_LARGE_BUFF)
        beat_signals_w_noise_group = VGroup(beat_signal_group, noise)
        beat_signals_w_noise = time_ax.plot(
            lambda t: (
                np.sin(2 * PI * f1 * t)
                + np.sin(2 * PI * f2 * t)
                + normalvariate(mu=0, sigma=noise_sigma)
            )
            / 2.2,
            color=IF_COLOR,
        )

        f_clutter = 3.7
        beat_signal_clutter = time_ax.plot(
            lambda t: np.sin(2 * PI * f_clutter * t), color=IF_COLOR
        ).next_to(beat_signal_1, direction=UP, buff=MED_LARGE_BUFF)
        beat_signals_w_noise_clutter_group = VGroup(
            beat_signals_w_noise_group, beat_signal_clutter
        )
        beat_signals_w_noise_clutter = time_ax.plot(
            lambda t: (
                np.sin(2 * PI * f1 * t)
                + np.sin(2 * PI * f2 * t)
                + normalvariate(mu=0, sigma=noise_sigma)
                + np.sin(2 * PI * f_clutter * t)
            )
            / 3.2,
            color=IF_COLOR,
        )

        signal_plus_beat_1 = MathTex("+")
        signal_plus_beat_2 = MathTex("+")
        signal_plus_noise = MathTex("+")
        signal_eqn = Tex(r"Signal = ", r"$\sin{(2 \pi f_{beat,1} t)}$").next_to(
            beat_signal_1, direction=RIGHT, buff=LARGE_BUFF
        )
        beat_signal_2_eqn = MathTex(r"\sin{(2 \pi f_{beat,2} t)}")
        noise_eqn = Tex(r"\textit{noise}")
        beat_signal_clutter_eqn = MathTex(r"\sin{(2 \pi f_{beat,\text{clutter}} t)}")

        self.next_section(skip_animations=skip_animations(True))

        self.play(Create(range_eqn), run_time=2)

        self.wait(0.5)

        self.play(Indicate(range_eqn[0][0]))

        self.wait(0.2)

        self.play(Indicate(range_eqn[0][5:10]))

        self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         FadeOut(range_eqn[0][0:5], range_eqn[0][10:]),
        #         Transform(range_eqn[0][5:10], f_beat),
        #         lag_ratio=0.5,
        #     )
        # )

        self.play(FadeOut(range_eqn))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(time_ax), FadeIn(time_ax_label), radar.get_animation()
                ),
                AnimationGroup(
                    Create(beat_signal_1), Create(target_1), FadeIn(target_1_label)
                ),
                AnimationGroup(
                    FadeIn(signal_eqn), Create(radar_beam_l), Create(radar_beam_r)
                ),
                GrowArrow(target_1_reflection),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(beat_signal_2), Create(target_2), FadeIn(target_2_label)
                ),
                AnimationGroup(
                    FadeIn(
                        beat_signal_2_eqn.next_to(
                            beat_signal_2, direction=RIGHT, buff=LARGE_BUFF
                        ),
                    ),
                    GrowArrow(target_2_reflection),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            Transform(beat_signal_group, beat_signals),
            beat_signal_2_eqn.animate.next_to(
                signal_eqn[1], direction=DOWN, aligned_edge=LEFT, buff=SMALL_BUFF
            ),
            FadeIn(signal_plus_beat_1.next_to(signal_eqn, buff=SMALL_BUFF)),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(noise),
                FadeIn(noise_eqn.next_to(noise, direction=RIGHT, buff=LARGE_BUFF)),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            Transform(beat_signals_w_noise_group, beat_signals_w_noise),
            FadeIn(signal_plus_beat_2.next_to(beat_signal_2_eqn, buff=SMALL_BUFF)),
            noise_eqn.animate.next_to(
                beat_signal_2_eqn, direction=DOWN, aligned_edge=LEFT, buff=SMALL_BUFF
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(Create(beat_signal_clutter), Create(ground_clutter)),
                AnimationGroup(
                    FadeIn(
                        beat_signal_clutter_eqn.next_to(
                            beat_signal_clutter, direction=RIGHT, buff=LARGE_BUFF
                        )
                    ),
                    GrowArrow(ground_clutter_reflection),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            Transform(beat_signals_w_noise_clutter_group, beat_signals_w_noise_clutter),
            FadeIn(signal_plus_noise.next_to(noise_eqn, buff=SMALL_BUFF)),
            beat_signal_clutter_eqn.animate.next_to(
                noise_eqn, direction=DOWN, aligned_edge=LEFT, buff=SMALL_BUFF
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        signal_eqn_group = VGroup(
            signal_eqn,
            signal_plus_beat_1,
            beat_signal_2_eqn,
            signal_plus_beat_2,
            beat_signal_clutter_eqn,
            signal_plus_noise,
            noise_eqn,
        )

        ax_group_scale = 1
        time_ax_group = VGroup(time_ax, beat_signal_1, time_ax_label)

        self.play(
            LaggedStart(
                FadeOut(
                    radar.vgroup,
                    radar_beam_l,
                    radar_beam_r,
                    target_2_reflection,
                    target_1_reflection,
                    ground_clutter_reflection,
                    ground_clutter,
                    target_1,
                    target_1_label,
                    target_2,
                    target_2_label,
                    shift=UP,
                ),
                signal_eqn_group.animate.arrange(buff=SMALL_BUFF)
                .scale(0.8)
                .to_edge(UP, buff=MED_SMALL_BUFF),
                time_ax_group.animate.scale(ax_group_scale)
                .move_to(ORIGIN)
                .to_edge(LEFT, buff=MED_LARGE_BUFF),
                lag_ratio=0.5,
            )
        )

        power_norm_1 = -6
        power_norm_2 = -9
        power_norm_clutter = 0
        A_1 = 10 ** (power_norm_1 / 10)
        A_2 = 10 ** (power_norm_2 / 10)
        A_clutter = 10 ** (power_norm_clutter / 10)

        noise_mu = 0
        noise_sigma_db = -10
        noise_sigma = 10 ** (noise_sigma_db / 10)

        np.random.seed(0)
        noise_npi = np.random.normal(loc=noise_mu, scale=noise_sigma, size=t.size)

        x_n = (
            A_1 * np.sin(2 * PI * f1 * t)
            + A_2 * np.sin(2 * PI * f2 * t)
            + A_clutter * np.sin(2 * PI * f_clutter * t)
            + noise_npi
        ) / (A_1 + A_2 + A_clutter + noise_sigma)

        blackman_window = signal.windows.blackman(N)
        x_n_windowed = x_n * blackman_window

        freq, X_k = get_fft_values(
            x_n_windowed,
            fs=fs,
            stop_time=stop_time,
            fft_len=2**18,
            f_max=f_max,
            y_min=y_min,
        )

        X_k_plot = f_ax.plot_line_graph(
            freq, X_k, line_color=TX_COLOR, add_vertex_dots=False
        )

        f_ax_group = (
            Group(f_ax, f_ax_label, X_k_plot)
            .scale(ax_group_scale)
            .to_edge(RIGHT, buff=MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                Create(f_ax), FadeIn(f_ax_label), Create(X_k_plot), lag_ratio=0.5
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        beat_signal_1_vline = f_ax.get_vertical_line(
            f_ax.c2p(f1, -y_min + power_norm_1),
            color=CITRINE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.4,
        )
        beat_signal_2_vline = f_ax.get_vertical_line(
            f_ax.c2p(f2, -y_min + power_norm_2),
            color=PIGMENT_GREEN,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.4,
        )
        clutter_vline = f_ax.get_vertical_line(
            f_ax.c2p(f_clutter, -y_min + power_norm_clutter),
            color=RED,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.4,
        )

        # beat_signal_1_graph_label = MathTex(r"f_{beat,1}", color=CITRINE).next_to(
        #     beat_signal_1_vline, direction=DOWN, buff=MED_SMALL_BUFF
        # )
        # beat_signal_2_graph_label = MathTex(r"f_{beat,2}", color=PIGMENT_GREEN).next_to(
        #     beat_signal_2_vline, direction=DOWN, buff=MED_SMALL_BUFF
        # )
        # clutter_graph_label = MathTex(r"f_{beat,clutter}", color=RED).next_to(
        #     clutter_vline, direction=DOWN, buff=MED_SMALL_BUFF
        # )

        self.play(
            LaggedStart(
                AnimationGroup(
                    signal_eqn[1].animate.set_color(CITRINE),
                    Create(beat_signal_1_vline),
                    rate_func=rate_functions.ease_in_out_quart,
                ),
                AnimationGroup(
                    beat_signal_2_eqn.animate.set_color(PIGMENT_GREEN),
                    Create(beat_signal_2_vline),
                    rate_func=rate_functions.ease_in_out_quart,
                ),
                AnimationGroup(
                    beat_signal_clutter_eqn.animate.set_color(RED),
                    Create(clutter_vline),
                    rate_func=rate_functions.ease_in_out_quart,
                ),
                lag_ratio=0.8,
            )
        )

        self.wait(0.5)

        samples = time_ax.get_vertical_lines_to_graph(
            beat_signals_w_noise_clutter,
            x_range=[0, duration],
            num_lines=30,
            color=BLUE,
        )

        n_ax_label = time_ax.get_x_axis_label(Tex("$n$"))
        x_n_label = Tex("x[n]").next_to(time_ax, direction=UP, buff=MED_SMALL_BUFF)

        self.play(
            Create(samples),
            FadeOut(time_ax_label[0], shift=DOWN),
            FadeIn(n_ax_label, shift=DOWN),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        fft_label = Tex("Fourier Transform").scale(1.2).to_edge(DOWN, buff=LARGE_BUFF)

        fft_mystery_label = Tex("?").scale(1.6).move_to(fft_label)
        fft_mystery_box = SurroundingRectangle(
            fft_mystery_label, color=WHITE, buff=MED_SMALL_BUFF
        )
        fft_mystery_box.stretch(
            fft_mystery_label.width * 2.5 / fft_mystery_box.width, dim=0
        )
        fft_box = fft_mystery_box.copy().stretch(
            fft_label.width * 1.2 / fft_mystery_box.width, dim=0
        )

        fft_box_width_tracker = ValueTracker(fft_mystery_box.width)

        to_fft_mystery_box_p1 = time_ax.get_bottom() + [0, -0.1, 0]
        to_fft_mystery_box_p2 = fft_mystery_box.get_left() + [-0.1, 0, 0]
        to_fft_mystery_box_bez = CubicBezier(
            to_fft_mystery_box_p1,
            to_fft_mystery_box_p1 + [0, -1.5, 0],
            to_fft_mystery_box_p2 + [-1.5, 0, 0],
            to_fft_mystery_box_p2,
        )

        from_fft_mystery_box_p2 = f_ax.get_bottom() + [0, -0.1, 0]
        from_fft_mystery_box_p1 = fft_mystery_box.get_right() + [0.1, 0, 0]
        from_fft_mystery_box_bez = CubicBezier(
            from_fft_mystery_box_p1,
            from_fft_mystery_box_p1 + [1.5, 0, 0],
            from_fft_mystery_box_p2 + [0, -1.5, 0],
            from_fft_mystery_box_p2,
        )

        to_fft_box_p1 = time_ax.get_bottom() + [0, -0.1, 0]
        to_fft_box_p2 = fft_box.get_left() + [-0.1, 0, 0]
        to_fft_box_bez = CubicBezier(
            to_fft_box_p1,
            to_fft_box_p1 + [0, -0.5, 0],
            to_fft_box_p2 + [-0.5, 0, 0],
            to_fft_box_p2,
        )

        from_fft_box_p2 = f_ax.get_bottom() + [0, -0.1, 0]
        from_fft_box_p1 = fft_box.get_right() + [0.1, 0, 0]
        from_fft_box_bez = CubicBezier(
            from_fft_box_p1,
            from_fft_box_p1 + [0.5, 0, 0],
            from_fft_box_p2 + [0, -0.5, 0],
            from_fft_box_p2,
        )

        self.play(
            LaggedStart(
                Create(to_fft_mystery_box_bez),
                AnimationGroup(
                    Create(fft_mystery_box),
                    FadeIn(fft_mystery_label),
                ),
                Create(from_fft_mystery_box_bez),
                lag_ratio=0.5,
            )
        )

        self.add(to_fft_mystery_box_bez, from_fft_mystery_box_bez)

        self.wait(0.5)

        self.play(
            Transform(fft_mystery_label, fft_label),
            Transform(fft_mystery_box, fft_box),
            Transform(to_fft_mystery_box_bez, to_fft_box_bez),
            Transform(from_fft_mystery_box_bez, from_fft_box_bez),
            FadeIn(x_n_label),
        )

        self.wait(0.5)

        self.wait(2)

        # self.add(
        #     range_eqn,
        #     tex_labels,
        #     range_eqn.copy().next_to(range_eqn, direction=DOWN, buff=LARGE_BUFF),
        # )


class FFTImplementations(Scene):
    def construct(self):
        language_logos = Group(
            *[
                ImageMobject(f"../props/static/{fname}").scale_to_fit_width(1)
                for fname in [
                    "c_logo.webp",
                    "cpp_logo.png",
                    # "fortran_logo.svg",
                    # "matlab_logo.svg",
                    # "r_logo.svg",
                    "rustacean.png",
                    "verilog_logo.png",
                    "python-logo-only.png",
                ]
            ]
        )
        self.add(language_logos.arrange_in_grid(4, 4))


class FFT(Scene):
    def construct(self): ...


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


class Gears(Scene):
    def construct(self):
        gears = SVGMobject("../props/static/Gears.svg").scale(3)
        (red_gear, blue_gear) = gears.shift(DOWN * 0.5 + RIGHT)

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

        self.add(gears)
        red_gear.add_updater(Driver)
        blue_gear.add_updater(Driven)

        RA.set_value(PI / 6)

        # for a, t in AccTime:
        #     # self.add_sound("Click.wav")
        #     self.play(Indicate(RA.set_value(a)), run_time=0.5)
        #     corr = 2 / 60  # missed frame correction
        #     self.wait(t + corr - 0.5)  # -0.5 for run_time=0.5

        self.wait(6 + 1)


class TexTest(Scene):
    def construct(self):
        tex = MathTex(r"\frac{f_{beat}}{a^2}")
        indexs = index_labels(tex[0], color=RED)
        tex[0][3].set_color(YELLOW)
        tex[0][6].set_color(RED)
        # self.add(tex, indexs)

        # ssb_label = (
        #     Tex(r"Single Sideband\\Mixer").scale(0.8).next_to(tex, direction=DOWN)
        # )
        # self.play(FadeIn(ssb_label, shift=UP))
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[at]{easylist}")
        l = Tex(
            r"\begin{easylist}"
            r"\ListProperties(Style1*=\bfseries,Numbers2=l,Mark1={},Mark2={},Indent2=1em)"
            r"@ Something"
            r"@@ apple"
            r"@@ pear"
            r"@@ banana"
            r"@ Something"
            r"@@ Frogs"
            r"\end{easylist}",
            tex_template=myTemplate,
        )

        self.add(l)


class TestBD(Scene):
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
        ) = get_bd(True)
        self.add(bd.scale_to_fit_width(config["frame_width"] * 0.8))


class RadarPlot(Scene):
    def construct(self):
        radar_plot = PolarPlane(
            azimuth_step=4,
            size=config["frame_width"] / 4,
            radius_config={
                "stroke_color": WHITE,
                "include_tip": False,
            },
            azimuth_direction="CW",
        )

        rmax = radar_plot.get_y_range().max()

        radar_scan_vel = ValueTracker(1)
        radar_scan_angle = ValueTracker(-PI)

        scan_line = Line(
            radar_plot.pr2pt(0, 0),
            radar_plot.pr2pt(rmax, radar_scan_angle.get_value()),
            color=RED,
        )

        def scan_updater(m: Mobject, dt):
            radar_scan_angle.set_value(
                radar_scan_angle.get_value() + radar_scan_vel.get_value() * dt
            )
            m.become(
                Line(
                    radar_plot.pr2pt(0, 0),
                    radar_plot.pr2pt(rmax, radar_scan_angle.get_value()),
                    color=RED,
                )
            )

        def show_dot_updater(m: Mobject):
            dot_angle = radar_plot.pt2pr([m.get_x(), m.get_y(), 0])[1]
            if radar_scan_angle.get_value() >= dot_angle:
                m.set_opacity(1)
            else:
                m.set_opacity(0)

        scan_line.add_updater(scan_updater)

        targets = VGroup()
        for r, theta in [
            [rmax * random(), 30 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 54 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 92 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 100 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 140 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 200 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 230 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 305 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 290 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 350 * DEGREES + random() * 6 * DEGREES],
            [rmax * random(), 300 * DEGREES + random() * 6 * DEGREES],
        ]:
            dot = Dot(radar_plot.pr2pt(r, theta))
            dot.add_updater(show_dot_updater)
            targets.add(dot)

        self.add(radar_plot, targets, scan_line)

        self.wait(8)


class ProcessorBlock(Scene):
    def construct(self):
        mcu = ImageMobject("../props/static/microcontroller.png")
        mcu2 = ImageMobject("../props/static/mcu.png")

        self.add(Group(mcu, mcu2).arrange())


class BinaryScreenWipe(Scene):
    def construct(self):
        sequence = VGroup(
            *[
                Text(
                    "".join([f"{n}" for n in list(np.random.randint(0, 2, 40))]),
                    disable_ligatures=True,
                    font="FiraCode Nerd Font Mono",
                    color=GREEN,
                    stroke_opacity=1,
                    # fill_color=BACKGROUND_COLOR,
                    # fill_opacity=1,
                )
                for _ in range(13)
            ]
        ).arrange(direction=DOWN, buff=MED_SMALL_BUFF)
        sequence.next_to(
            LEFT * self.camera.frame_width / 2, direction=LEFT, buff=SMALL_BUFF
        )
        for s in sequence:
            s.shift(random() * 2 * LEFT)
        sequence_backgrounds = VGroup(
            *[
                BackgroundRectangle(
                    m, fill_color=BACKGROUND_COLOR, fill_opacity=1, buff=0
                ).stretch((m.height + SMALL_BUFF * 2.6) / m.height, dim=1)
                for m in sequence
            ]
        )
        # for sb in sequence_backgrounds:
        #     sb.stretch()
        sequence_group = VGroup(sequence_backgrounds, sequence)

        part_3 = Tex("Part 3: Signal Processing").scale(2)

        self.add(part_3)
        self.add(sequence_group)

        self.play(
            sequence_group.animate(rate_func=rate_functions.linear, run_time=3.5).shift(
                config["frame_width"] * RIGHT * 2.3
            ),
        )

        self.wait(2)


class SmilingComputer(Scene):
    def construct(self):
        computer = BLOCKS.get("computer").copy()
        computer_eyes = VGroup(Dot(), Dot()).arrange().move_to(computer).shift(UP * 0.4)
        computer_smile = (
            Arc(start_angle=-TAU / 8 - TAU / 4).move_to(computer).scale(0.7)
        )

        self.add(
            computer,
            computer_eyes,
            computer_smile,
        )
