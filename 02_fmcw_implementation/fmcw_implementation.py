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


class BD(Scene):
    def construct(self):
        # self.add(Group(*list(BLOCKS.values())).arrange_in_grid(4, 6))

        input_circle = Circle(radius=0.2)
        input_label = Tex(r"$V_{tune}$ Input").next_to(
            input_circle, direction=UP, buff=SMALL_BUFF
        )
        inp = VGroup(input_circle, input_label)
        vco = BLOCKS.get("oscillator").next_to(
            input_circle, direction=RIGHT, buff=BLOCK_BUFF
        )
        input_to_vco = Line(input_circle.get_right(), vco.get_left())
        amp1 = (
            BLOCKS.get("amp")
            .next_to(vco, direction=RIGHT, buff=BLOCK_BUFF)
            .set_fill(GAIN_COLOR)
        )
        vco_to_amp1 = Line(vco.get_right(), amp1.get_left())
        splitter = BLOCKS.get("splitter").next_to(amp1, buff=BLOCK_BUFF)
        amp1_to_splitter = Line(amp1.get_right(), splitter.get_left())
        mixer = (
            BLOCKS.get("mixer")
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

        lp_filter = BLOCKS.get("lp_filter").next_to(
            mixer, direction=LEFT, buff=BLOCK_BUFF
        )
        mixer_to_lp_filter = Line(mixer_if, lp_filter.get_right())

        adc = (
            BLOCKS.get("adc").next_to(lp_filter, direction=LEFT, buff=BLOCK_BUFF).flip()
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
                vco,
                vco_to_amp1,
                amp1,
                amp1_to_splitter,
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
            .scale(0.5)
            .move_to(ORIGIN)
        )

        tx_section = Group(inp, vco, splitter, tx_antenna)
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

        # self.add(bd)

        # self.add(
        #     tx_section_box, rx_section_box, tx_section_box_label, rx_section_box_label
        # )

        self.play(get_bd_animation(bd, lagged=True, lag_ratio=0.5), run_time=5)
        self.play(
            Create(rx_section_box),
            Create(tx_section_box),
            Create(rx_section_box_label),
            Create(tx_section_box_label),
        )

        self.wait(2)
