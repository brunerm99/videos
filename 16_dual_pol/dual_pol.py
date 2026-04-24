# dual_pol.py
import math
import os
import pprint
import sys
from copy import deepcopy
from functools import lru_cache
from math import sqrt
from random import shuffle
from turtle import width

import numpy as np
import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from manim.utils.color.X11 import BROWN1
from MF_Tools import VT
from networkx import center
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import PchipInterpolator, interp1d

sys.path.insert(0, "..")
from props import (
    Bjt,
    Capacitor,
    Fet,
    Inductor,
    Resistor,
    VideoMobject,
    WeatherRadarTower,
    cubic_bezier,
    ease_in_out_elastic,
    ease_out_elastic,
    get_amp,
    get_blocks,
    get_filt_block,
    get_phase_shifter,
    get_splitter,
)
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False

load_dotenv("../.env")
FONT = os.getenv("FONT", "")

BLOCKS = get_blocks()

GOOD = BLUE
OK = GREY
BAD = RED
TARGET1_COLOR = GREEN
TARGET2_COLOR = ORANGE
TARGET3_COLOR = BLUE
INPUT_COLOR = BLUE
OUTPUT_COLOR = ORANGE
GAIN_COLOR = GREEN
PAE_COLOR = YELLOW
PRECIP_COLOR = BLUE
HPOL_TX_COLOR = BLUE
VPOL_TX_COLOR = RED
HPOL_RX_COLOR = PURPLE
VPOL_RX_COLOR = ORANGE


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        dual_pol_full = Text("Dual Polarization", font=FONT).scale(1.5)
        radar = Text("Radar", font=FONT).scale(1.5)
        weather = Text("Weather", font="CADILLAC PERSONAL USE").scale(1.5)

        all_group = Group(dual_pol_full, radar).arrange(DOWN, LARGE_BUFF)
        weather_radar_group = (
            Group(weather, radar.copy())
            .arrange(RIGHT, MED_LARGE_BUFF)
            .next_to(dual_pol_full, DOWN, LARGE_BUFF)
        )
        dual_pol = Text("Dual-Pol", font=FONT).scale(1.5).move_to(dual_pol_full)

        self.wait(0.5)

        self.add(dual_pol_full[:4])

        self.wait(0.5)

        self.add(dual_pol_full[4:])

        self.wait(0.5)

        self.add(radar)

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(dual_pol_full[7:]),
                ReplacementTransform(dual_pol_full[4:7], dual_pol[5:8]),
                ReplacementTransform(dual_pol_full[:4], dual_pol[:4]),
                GrowFromCenter(dual_pol[4]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                radar.animate.move_to(weather_radar_group[-1]),
                Write(weather),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        values = [7, 4, 6, 8, 9, 3, 4, 7, 7, 6, 7, 8, 3, 5, 10]
        bar_names = [
            "FMCW Radar",
            "FMCW pt2",
            "CFAR",
            "Doppler/ Range-Velocity",
            "Phased Array",
            "Radar Equation",
            "SNR",
            "Beamforming",
            "Resolution",
            "Aliasing",
            "IQ Sampling",
            "Pulse Compression",
            "RF Gain",
            "RF Compression",
            "Dual-Pol",
        ]

        chart = BarChart(
            values=values,
            # bar_names=bar_names,
            y_range=[0, 10, 1],
            y_length=fh(self, 1.5),
            x_length=fw(self, 2.5),
            x_axis_config={"font_size": 48},
            y_axis_config={"font_size": 80},
        )

        all_group = Group(*all_group, weather, radar)
        chart.shift(all_group.get_center() - chart.c2p(len(values) - 0.25, 11.5))
        cbar_labels = chart.get_bar_labels(font_size=48)
        coolness = (
            Text("Coolness", font=FONT)
            .rotate(PI / 2)
            .next_to(chart, LEFT, MED_LARGE_BUFF)
        )

        labels = Group(
            *[
                Paragraph(*name.split(" "), font=FONT, alignment="center")
                .scale(0.75)
                .move_to(chart.c2p(idx + 0.5, v + 1))
                for idx, (v, name) in enumerate(zip(values[:-1], bar_names[:-1]))
            ]
        )
        labels[7].shift(DOWN * 0.4)
        labels[8].shift(UP * 0.4)

        # self.add(chart, labels)

        self.next_section(skip_animations=skip_animations(False))

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    Group(chart, coolness, all_group).width * 1.1
                ).move_to(Group(chart, coolness, all_group)),
                Create(chart, run_time=3),
                FadeIn(coolness),
                LaggedStart(*[FadeIn(m) for m in labels], lag_ratio=0.1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(chart, coolness, *labels),
            self.camera.frame.animate.restore(),
        )

        self.wait(0.5)

        self.wait(2)


class Background(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(fh(self, 0.5)).set_stroke(
            width=DEFAULT_STROKE_WIDTH * 3
        ).to_corner(DL, MED_LARGE_BUFF).shift(RIGHT).set_z_index(10)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .to_edge(RIGHT, LARGE_BUFF)
            .shift(LEFT * 2 + UP * 2)
        )
        raindrop1 = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cloud.width * 0.15)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(LEFT * cloud.width * 0.3)
            .set_opacity(0.7)
        )
        raindrop2 = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cloud.width * 0.15)
            .next_to(cloud, DOWN, SMALL_BUFF)
            .shift(RIGHT * cloud.width * 0.4)
            .set_opacity(0.7)
        )
        raindrop3 = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cloud.width * 0.15)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(LEFT * cloud.width * 0.1 + DOWN * 0.4)
            .set_opacity(0.7)
        )
        raindrop4 = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cloud.width * 0.15)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(RIGHT * cloud.width * 0.3 + DOWN * 0.7)
            .set_opacity(0.7)
        )
        snowflake1 = (
            ImageMobject("../props/static/snowflake.png")
            .scale_to_fit_width(cloud.width * 0.2)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(RIGHT * cloud.width * 0.1 + DOWN * 0.3)
        )
        snowflake2 = (
            ImageMobject("../props/static/snowflake.png")
            .scale_to_fit_width(cloud.width * 0.2)
            .next_to(cloud, DOWN, MED_SMALL_BUFF)
            .shift(LEFT * cloud.width * 0.2 + DOWN * 1)
        )
        snowflake3 = (
            ImageMobject("../props/static/snowflake.png")
            .scale_to_fit_width(cloud.width * 0.2)
            .next_to(raindrop4, DOWN, SMALL_BUFF)
            .shift(LEFT * 0.5)
        )
        raindrop5 = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cloud.width * 0.15)
            .next_to(snowflake3, LEFT)
            .shift(DOWN * 0.1)
            .set_opacity(0.7)
        )

        self.play(radar.vgroup.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        # self.add(
        #     cloud,
        #     raindrop1,
        #     raindrop2,
        #     raindrop3,
        #     raindrop4,
        #     raindrop5,
        #     snowflake1,
        #     snowflake2,
        #     snowflake3,
        # )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeIn(cloud),
                FadeIn(raindrop1, shift=DOWN),
                FadeIn(raindrop2, shift=DOWN),
                FadeIn(raindrop3, shift=DOWN),
                FadeIn(raindrop4, shift=DOWN),
                FadeIn(raindrop5, shift=DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(snowflake1, shift=DOWN),
                FadeIn(snowflake2, shift=DOWN),
                FadeIn(snowflake3, shift=DOWN),
                lag_ratio=0.3,
            )
        )

        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_height(fh(self, 0.3))
            .rotate(PI * 0.7)
            .set_fill(TARGET1_COLOR)
            .next_to(raindrop2, RIGHT, SMALL_BUFF)
        )

        building = (
            SVGMobject("../props/static/Icon 12.svg")
            .scale_to_fit_height(fh(self, 0.4))
            .to_corner(DR, MED_SMALL_BUFF)
            .shift(LEFT)
            .set_fill(WHITE)
            .set_stroke(color=WHITE, width=DEFAULT_STROKE_WIDTH * 1.5)
        )

        self.wait(0.5)

        self.play(plane.shift(RIGHT * 8).animate.shift(LEFT * 8))

        self.wait(0.5)

        self.play(FadeIn(building))

        self.wait(0.5)

        tx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=(self.camera.frame.get_right() - radar.radome.get_right())[0],
            y_length=fh(self, 0.3),
            tips=False,
        )
        tx_ax.shift(radar.radome.get_right() - tx_ax.c2p(0, 0))

        pw = 0.2
        f = 10
        tx_x1 = VT(0)
        tx_plot = always_redraw(
            lambda: tx_ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                color=HPOL_TX_COLOR,
                x_range=[max(0, ~tx_x1 - pw), ~tx_x1, 1 / 200],
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            ).set_z_index(1)
        )

        raindrop_rx_amp = 0.3
        snowflake_rx_amp = 0.1
        building_rx_amp = 0.5
        plane_rx_amp = 0.4

        plane_rx_x1 = VT(0)

        building_rx_x1 = VT(0)

        snowflake1_rx_x1 = VT(0)
        snowflake2_rx_x1 = VT(0)
        snowflake3_rx_x1 = VT(0)

        raindrop1_rx_x1 = VT(0)
        raindrop2_rx_x1 = VT(0)
        raindrop3_rx_x1 = VT(0)
        raindrop4_rx_x1 = VT(0)
        raindrop5_rx_x1 = VT(0)

        raindrop1_rx_line = Line(raindrop1.get_left(), radar.radome.get_right())
        raindrop1_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=raindrop1_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(raindrop1_rx_line.get_angle())
        raindrop1_rx_ax.shift(raindrop1.get_left() - raindrop1_rx_ax.c2p(0, 0))
        raindrop1_rx_plot = always_redraw(
            lambda: raindrop1_rx_ax.plot(
                lambda t: raindrop_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~raindrop1_rx_x1 - pw),
                    min(1, ~raindrop1_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        raindrop2_rx_line = Line(raindrop2.get_left(), radar.radome.get_right())
        raindrop2_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=raindrop2_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(raindrop2_rx_line.get_angle())
        raindrop2_rx_ax.shift(raindrop2.get_left() - raindrop2_rx_ax.c2p(0, 0))
        raindrop2_rx_plot = always_redraw(
            lambda: raindrop2_rx_ax.plot(
                lambda t: raindrop_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~raindrop2_rx_x1 - pw),
                    min(1, ~raindrop2_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        raindrop3_rx_line = Line(raindrop3.get_left(), radar.radome.get_right())
        raindrop3_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=raindrop3_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(raindrop3_rx_line.get_angle())
        raindrop3_rx_ax.shift(raindrop3.get_left() - raindrop3_rx_ax.c2p(0, 0))
        raindrop3_rx_plot = always_redraw(
            lambda: raindrop3_rx_ax.plot(
                lambda t: raindrop_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~raindrop3_rx_x1 - pw),
                    min(1, ~raindrop3_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        raindrop4_rx_line = Line(raindrop4.get_left(), radar.radome.get_right())
        raindrop4_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=raindrop4_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(raindrop4_rx_line.get_angle())
        raindrop4_rx_ax.shift(raindrop4.get_left() - raindrop4_rx_ax.c2p(0, 0))
        raindrop4_rx_plot = always_redraw(
            lambda: raindrop4_rx_ax.plot(
                lambda t: raindrop_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~raindrop4_rx_x1 - pw),
                    min(1, ~raindrop4_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        raindrop5_rx_line = Line(raindrop5.get_left(), radar.radome.get_right())
        raindrop5_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=raindrop5_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(raindrop5_rx_line.get_angle())
        raindrop5_rx_ax.shift(raindrop5.get_left() - raindrop5_rx_ax.c2p(0, 0))
        raindrop5_rx_plot = always_redraw(
            lambda: raindrop5_rx_ax.plot(
                lambda t: raindrop_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~raindrop5_rx_x1 - pw),
                    min(1, ~raindrop5_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        snowflake1_rx_line = Line(snowflake1.get_left(), radar.radome.get_right())
        snowflake1_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=snowflake1_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(snowflake1_rx_line.get_angle())
        snowflake1_rx_ax.shift(snowflake1.get_left() - snowflake1_rx_ax.c2p(0, 0))
        snowflake1_rx_plot = always_redraw(
            lambda: snowflake1_rx_ax.plot(
                lambda t: snowflake_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~snowflake1_rx_x1 - pw),
                    min(1, ~snowflake1_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        snowflake2_rx_line = Line(snowflake2.get_left(), radar.radome.get_right())
        snowflake2_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=snowflake2_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(snowflake2_rx_line.get_angle())
        snowflake2_rx_ax.shift(snowflake2.get_left() - snowflake2_rx_ax.c2p(0, 0))
        snowflake2_rx_plot = always_redraw(
            lambda: snowflake2_rx_ax.plot(
                lambda t: snowflake_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~snowflake2_rx_x1 - pw),
                    min(1, ~snowflake2_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        snowflake3_rx_line = Line(snowflake3.get_left(), radar.radome.get_right())
        snowflake3_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=snowflake3_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(snowflake3_rx_line.get_angle())
        snowflake3_rx_ax.shift(snowflake3.get_left() - snowflake3_rx_ax.c2p(0, 0))
        snowflake3_rx_plot = always_redraw(
            lambda: snowflake3_rx_ax.plot(
                lambda t: snowflake_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~snowflake3_rx_x1 - pw),
                    min(1, ~snowflake3_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        building_rx_line = Line(building.get_left(), radar.radome.get_right())
        building_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=building_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(building_rx_line.get_angle())
        building_rx_ax.shift(building.get_left() - building_rx_ax.c2p(0, 0))
        building_rx_plot = always_redraw(
            lambda: building_rx_ax.plot(
                lambda t: building_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[
                    max(0, ~building_rx_x1 - pw),
                    min(1, ~building_rx_x1),
                    1 / 200,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        plane_rx_line = Line(plane.get_left(), radar.radome.get_right())
        plane_rx_ax = Axes(
            x_range=[0, 1, 0.125],
            y_range=[-1, 1, 1],
            x_length=plane_rx_line.get_length(),
            y_length=fh(self, 0.3),
            tips=False,
        ).rotate(plane_rx_line.get_angle())
        plane_rx_ax.shift(plane.get_left() - plane_rx_ax.c2p(0, 0))
        plane_rx_plot = always_redraw(
            lambda: plane_rx_ax.plot(
                lambda t: plane_rx_amp * np.sin(2 * PI * f * t),
                color=HPOL_RX_COLOR,
                x_range=[max(0, ~plane_rx_x1 - pw), min(1, ~plane_rx_x1), 1 / 200],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(1)
        )

        self.add(
            tx_plot,
            raindrop1_rx_plot,
            raindrop2_rx_plot,
            raindrop3_rx_plot,
            raindrop4_rx_plot,
            raindrop5_rx_plot,
            snowflake1_rx_plot,
            snowflake2_rx_plot,
            snowflake3_rx_plot,
            plane_rx_plot,
            building_rx_plot,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                tx_x1.animate(run_time=5).set_value(1.1 + pw),
                LaggedStart(
                    raindrop1_rx_x1 @ (1 + pw),
                    snowflake2_rx_x1 @ (1 + pw),
                    raindrop3_rx_x1 @ (1 + pw),
                    raindrop5_rx_x1 @ (1 + pw),
                    snowflake1_rx_x1 @ (1 + pw),
                    snowflake3_rx_x1 @ (1 + pw),
                    raindrop4_rx_x1 @ (1 + pw),
                    raindrop2_rx_x1 @ (1 + pw),
                    building_rx_x1 @ (1 + pw),
                    plane_rx_x1 @ (1 + pw),
                    lag_ratio=0.03,
                    run_time=4,
                ),
                lag_ratio=0.4,
            ),
            # run_time=5,
        )

        self.wait(0.5)

        def echo(t, center, width, amplitude, ripple_freq=0, phase=0):
            envelope = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
            if ripple_freq == 0:
                return envelope

            return envelope * (
                0.55 + 0.45 * np.cos(2 * PI * ripple_freq * (t - center) + phase) ** 2
            )

        def radar_return(t):
            noise_floor = (
                0.02
                + 0.006 * np.sin(2 * PI * 17 * t) ** 2
                + 0.004 * np.sin(2 * PI * 31 * t + 0.8) ** 2
            )

            rain = (
                echo(t, 0.17, 0.035, 0.11, ripple_freq=18, phase=0.2)
                + echo(t, 0.24, 0.045, 0.16, ripple_freq=16, phase=1.0)
                + echo(t, 0.31, 0.03, 0.09, ripple_freq=14, phase=2.1)
            )
            snow = echo(t, 0.48, 0.055, 0.08, ripple_freq=8, phase=0.5) + echo(
                t, 0.55, 0.035, 0.05, ripple_freq=6, phase=1.4
            )
            plane = echo(t, 0.69, 0.018, 0.28, ripple_freq=24, phase=0.4) + echo(
                t, 0.72, 0.01, 0.08, ripple_freq=30, phase=1.1
            )
            building = echo(t, 0.84, 0.015, 0.55) + echo(
                t, 0.88, 0.025, 0.12, ripple_freq=10, phase=0.7
            )

            return np.clip(noise_floor + rain + snow + plane + building, 0, 0.95)

        rtn_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 1, 0.25],
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.5),
            tips=False,
            # axis_config={
            #     "stroke_opacity": 0.55,
            # },
        )
        xlabel = Text("time", font=FONT).scale(0.5).next_to(rtn_ax, DOWN, SMALL_BUFF)
        ylabel = (
            Text("amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(rtn_ax, LEFT, SMALL_BUFF)
        )
        rx_box = SurroundingRectangle(
            Group(rtn_ax, xlabel, ylabel),
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
        ).set_z_index(-1)
        rx_label = (
            Text("Received Signal", font=FONT)
            .scale_to_fit_width(rx_box.width * 0.4)
            .next_to(rx_box.get_corner(UL), UR, SMALL_BUFF)
        )
        rx_group = Group(rtn_ax, rx_box, xlabel, ylabel, rx_label).next_to(
            radar.left_leg, LEFT, LARGE_BUFF * 3, aligned_edge=DOWN
        )
        self.add(rx_group, rx_box)

        rtn_plot = rtn_ax.plot(
            radar_return,
            color=HPOL_RX_COLOR,
            x_range=[0, 1, 1 / 500],
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).set_z_index(2)

        radar.radome.set_fill(opacity=1, color=BACKGROUND_COLOR)
        rtn_bez = CubicBezier(
            radar.radome.get_left() + [0.1, 0, 0],
            radar.radome.get_left() + [-1, 0, 0],
            rx_box.get_right() + [1, -rx_box.height / 4, 0],
            rx_box.get_right() + [-0.1, -rx_box.height / 4, 0],
        ).set_z_index(-10)

        self.play(
            LaggedStart(
                self.camera.frame.animate.set_x(
                    Group(rtn_ax, radar.vgroup).get_x()
                ).shift(LEFT),
                Create(rtn_bez),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(Create(rtn_plot))

        self.wait(0.5)

        reflectivity_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-10, 60, 10],
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.5),
            tips=False,
        ).next_to(rtn_ax, UP, LARGE_BUFF)
        reflectivity_xlabel = (
            Text("range", font=FONT)
            .scale(0.5)
            .next_to(reflectivity_ax, DOWN, SMALL_BUFF)
        )
        reflectivity_ylabel = (
            Text("reflectivity (dBZ)", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(reflectivity_ax, LEFT, SMALL_BUFF)
        )
        reflectivity_box = SurroundingRectangle(
            Group(reflectivity_ax, reflectivity_xlabel, reflectivity_ylabel),
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
        ).set_z_index(-1)
        reflectivity_label = (
            Text("Reflectivity", font=FONT)
            .scale(0.6)
            .next_to(reflectivity_box.get_corner(UL), UR, SMALL_BUFF)
        )
        reflectivity_group = Group(
            reflectivity_ax,
            reflectivity_xlabel,
            reflectivity_ylabel,
            reflectivity_box,
            reflectivity_label,
        )

        doppler_ax = Axes(
            x_range=[-0.5, 0.5, 0.25],
            y_range=[0, 1, 0.25],
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.5),
            tips=False,
        )
        doppler_xlabel = (
            Text("velocity (m/s)", font=FONT)
            .scale(0.5)
            .next_to(doppler_ax, DOWN, SMALL_BUFF)
        )
        doppler_ylabel = (
            Text("amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(doppler_ax, LEFT, SMALL_BUFF)
        )
        doppler_box = SurroundingRectangle(
            Group(doppler_ax, doppler_xlabel, doppler_ylabel),
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
        ).set_z_index(-1)
        doppler_label = (
            Text("Doppler Velocity", font=FONT)
            .scale(0.6)
            .next_to(doppler_box.get_corner(UL), UR, SMALL_BUFF)
        )
        doppler_group = Group(
            doppler_ax,
            doppler_xlabel,
            doppler_ylabel,
            doppler_box,
            doppler_label,
        )

        Group(reflectivity_group, doppler_group).arrange(DOWN, SMALL_BUFF).next_to(
            rtn_ax, LEFT, LARGE_BUFF * 3
        )

        def doppler_peak(t, center, width, amplitude, ripple_freq=0, phase=0):
            envelope = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
            if ripple_freq == 0:
                return envelope

            return envelope * (
                0.6 + 0.4 * np.cos(2 * PI * ripple_freq * (t - center) + phase) ** 2
            )

        def doppler_spectrum(t):
            noise_floor = (
                0.015
                + 0.004 * np.sin(2 * PI * 19 * t + 0.4) ** 2
                + 0.003 * np.sin(2 * PI * 37 * t + 1.2) ** 2
            )

            building = doppler_peak(t, 0.0, 0.015, 0.58) + doppler_peak(
                t, 0, 0.055, 0.08, ripple_freq=5, phase=0.6
            )
            rain = doppler_peak(
                t, -0.06, 0.07, 0.12, ripple_freq=12, phase=0.3
            ) + doppler_peak(t, 0.05, 0.055, 0.08, ripple_freq=10, phase=1.7)
            snow = doppler_peak(t, -0.015, 0.035, 0.05, ripple_freq=7, phase=0.9)
            plane = (
                doppler_peak(t, 0.27, 0.016, 0.32)
                + doppler_peak(t, 0.24, 0.035, 0.10, ripple_freq=18, phase=0.2)
                + doppler_peak(t, 0.31, 0.018, 0.07, ripple_freq=22, phase=1.1)
            )

            return np.clip(noise_floor + building + rain + snow + plane, 0, 0.95)

        doppler_plot = doppler_ax.plot(
            doppler_spectrum,
            color=HPOL_RX_COLOR,
            x_range=[-0.5, 0.5, 1 / 500],
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).set_z_index(2)

        def reflectivity_blob(r, center, width, peak_dbz, ripple_freq=0, phase=0):
            peak_z = 10 ** (peak_dbz / 10)
            envelope = peak_z * np.exp(-0.5 * ((r - center) / width) ** 2)
            if ripple_freq == 0:
                return envelope

            return envelope * (
                0.7 + 0.3 * np.cos(2 * PI * ripple_freq * (r - center) + phase) ** 2
            )

        def reflectivity_profile(r):
            background_z = 10 ** (-8 / 10)

            rain_z = (
                reflectivity_blob(r, 0.17, 0.035, 27, ripple_freq=10, phase=0.3)
                + reflectivity_blob(r, 0.24, 0.045, 34, ripple_freq=8, phase=1.1)
                + reflectivity_blob(r, 0.31, 0.03, 25, ripple_freq=9, phase=2.0)
            )
            snow_z = reflectivity_blob(
                r, 0.48, 0.055, 16, ripple_freq=5, phase=0.6
            ) + reflectivity_blob(r, 0.55, 0.035, 11, ripple_freq=4, phase=1.2)
            plane_z = reflectivity_blob(r, 0.69, 0.018, 43) + reflectivity_blob(
                r, 0.72, 0.012, 32, ripple_freq=12, phase=0.4
            )
            building_z = reflectivity_blob(r, 0.84, 0.015, 55) + reflectivity_blob(
                r, 0.88, 0.025, 36, ripple_freq=6, phase=0.9
            )

            return np.clip(
                lin2db(background_z + rain_z + snow_z + plane_z + building_z), -10, 60
            )

        reflectivity_plot = reflectivity_ax.plot(
            reflectivity_profile,
            color=HPOL_RX_COLOR,
            x_range=[0, 1, 1 / 500],
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).set_z_index(2)

        reflectivity_bez = CubicBezier(
            rx_box.get_left(),
            rx_box.get_left() + [-2, 0, 0],
            reflectivity_box.get_right() + [2, 0, 0],
            reflectivity_box.get_right(),
        )

        doppler_bez = CubicBezier(
            rx_box.get_left(),
            rx_box.get_left() + [-2, 0, 0],
            doppler_box.get_right() + [2, 0, 0],
            doppler_box.get_right(),
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(reflectivity_group, doppler_group, rx_group).height * 1.2
                ).move_to(Group(reflectivity_group, doppler_group, rx_group)),
                LaggedStart(
                    Create(reflectivity_bez),
                    Write(reflectivity_label),
                    Create(reflectivity_box),
                    Create(reflectivity_ax),
                    Write(reflectivity_xlabel),
                    Write(reflectivity_ylabel),
                    Create(reflectivity_plot),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(doppler_bez),
                Write(doppler_label),
                Create(doppler_box),
                Create(doppler_ax),
                Write(doppler_xlabel),
                Write(doppler_ylabel),
                Create(doppler_plot),
                lag_ratio=0.1,
            ),
        )

        # self.play(Create(doppler_ax), Create(doppler_plot))

        # self.add(doppler_group, doppler_plot, reflectivity_group, reflectivity_plot)

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.next_to(reflectivity_group, LEFT, LARGE_BUFF * 8).shift(UP)

        ppi_image = (
            ImageMobject("./media/images/dual_pol/NEXRADPolarPPI_Transparent.png")
            .scale_to_fit_width(radar.radome.width * 4)
            .next_to(radar.radome, RIGHT, LARGE_BUFF)
            .shift(UP * 2)
        )
        ppi_image_bez = CubicBezier(
            radar.radome.get_right() + [0.3, 0, 0],
            radar.radome.get_right() + [1.2, 0, 0],
            ppi_image.get_left() + [0.5, 0, 0],
            ppi_image.get_left() + [1.5, 0, 0],
        )

        self.play(
            LaggedStart(
                doppler_label.animate.set_opacity(0.2),
                doppler_box.animate.set_opacity(0.2),
                doppler_ax.animate.set_opacity(0.2),
                doppler_xlabel.animate.set_opacity(0.2),
                doppler_ylabel.animate.set_opacity(0.2),
                doppler_plot.animate.set_stroke_opacity(0.2),
                self.camera.frame.animate.scale_to_fit_height(
                    reflectivity_group.height * 2.2
                )
                .move_to(reflectivity_group, DR)
                .shift(DR * LARGE_BUFF),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                radar.get_animation(),
                Create(ppi_image_bez),
                GrowFromCenter(ppi_image),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(ppi_image),
                Uncreate(ppi_image_bez),
                radar.vgroup.animate.shift(UP),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        beam_u = Line(
            radar.radome.get_right(),
            radar.radome.get_right() + RIGHT * 6 + UP,
            color=TX_COLOR,
        ).set_z_index(-3)
        beam_d = Line(
            radar.radome.get_right(),
            radar.radome.get_right() + RIGHT * 6 + DOWN,
            color=TX_COLOR,
        ).set_z_index(-3)
        beam = Group(beam_u, beam_d)

        self.play(Create(beam_u), Create(beam_d))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        metadata = _get_nexrad_reflectivity_ppi_data()
        rgba = _reflectivity_dbz_to_rgba(
            metadata["reflectivity_dbz"],
            valid_mask=metadata["valid_mask"],
            vmin=metadata["vmin"],
            vmax=metadata["vmax"],
        )

        theta_135 = 135.0
        max_range = metadata["max_range_km"]
        r = np.linspace(0, max_range, 1200)
        x = r * np.sin(np.deg2rad(theta_135))
        y = r * np.cos(np.deg2rad(theta_135))
        x_idx = np.abs(metadata["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
        y_idx = np.abs(metadata["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
        z = metadata["reflectivity_dbz"][y_idx, x_idx]
        z = np.nan_to_num(z, nan=0)
        order = np.argsort(r)
        r = r[order]
        z = z[order]

        z_func_135 = interp1d(r, z, fill_value="extrapolate")

        nexrad_ax = Axes(
            x_range=[0, 150, 25],
            y_range=[-10, 60, 10],
            x_length=reflectivity_ax.width,
            y_length=reflectivity_ax.height,
            tips=False,
        )
        nexrad_ax.shift(reflectivity_ax.c2p(0, 0) - nexrad_ax.c2p(0, 0))

        nexrad_135_x0 = VT(2.5)
        nexrad_135_x1 = VT(2.5)
        nexrad_135_plot = always_redraw(
            lambda: nexrad_ax.plot(
                z_func_135,
                x_range=[~nexrad_135_x0, ~nexrad_135_x1, 1 / 200],
                color=HPOL_RX_COLOR,
            )
        )

        def dbz_to_manim_color(dbz):
            rgba = (
                _reflectivity_dbz_to_rgba(
                    np.array([[dbz]], dtype=np.float32),
                    valid_mask=np.array([[True]]),
                    vmin=metadata["vmin"],
                    vmax=metadata["vmax"],
                    min_alpha=1.0,
                )[0, 0]
                / 255.0
            )
            return rgb_to_color(rgba[:3])

        reflectivity_plot_x0 = VT(0)
        reflectivity_plot_x1 = VT(1)
        reflectivity_plot_update = always_redraw(
            lambda: reflectivity_ax.plot(
                reflectivity_profile,
                color=HPOL_RX_COLOR,
                x_range=[~reflectivity_plot_x0, ~reflectivity_plot_x1, 1 / 500],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            ).set_z_index(2)
        )
        self.remove(reflectivity_plot)
        self.add(reflectivity_plot_update, nexrad_135_plot)

        self.play(
            LaggedStart(
                beam.animate.rotate(PI / 6, about_point=beam_u.get_start()),
                LaggedStart(
                    reflectivity_plot_x0 @ (~reflectivity_plot_x1),
                    nexrad_135_x1 @ 150,
                    lag_ratio=0.05,
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        theta_90 = 90.0
        max_range = metadata["max_range_km"]
        r = np.linspace(0, max_range, 1200)
        x = r * np.sin(np.deg2rad(theta_90))
        y = r * np.cos(np.deg2rad(theta_90))
        x_idx = np.abs(metadata["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
        y_idx = np.abs(metadata["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
        z = metadata["reflectivity_dbz"][y_idx, x_idx]
        z = np.nan_to_num(z, nan=0)
        order = np.argsort(r)
        r = r[order]
        z = z[order]

        z_func_90 = interp1d(r, z, fill_value="extrapolate")

        nexrad_90_x0 = VT(2.5)
        nexrad_90_x1 = VT(2.5)
        nexrad_90_plot = always_redraw(
            lambda: nexrad_ax.plot(
                z_func_90,
                x_range=[~nexrad_90_x0, ~nexrad_90_x1, 1 / 200],
                color=HPOL_RX_COLOR,
            )
        )

        self.add(nexrad_90_plot)

        self.play(
            LaggedStart(
                beam.animate.rotate(-PI / 3, about_point=beam_u.get_start()),
                AnimationGroup(nexrad_135_x0 @ 150, nexrad_90_x1 @ 150),
                lag_ratio=0.3,
            ),
            run_time=3,
        )
        self.remove(nexrad_135_plot)

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        theta_0 = 0.0
        max_range = metadata["max_range_km"]
        r = np.linspace(0, max_range, 1200)
        x = r * np.sin(np.deg2rad(theta_0))
        y = r * np.cos(np.deg2rad(theta_0))
        x_idx = np.abs(metadata["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
        y_idx = np.abs(metadata["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
        z = metadata["reflectivity_dbz"][y_idx, x_idx]
        z = np.nan_to_num(z, nan=0)
        order = np.argsort(r)
        r = r[order]
        z = z[order]

        z_func_0 = interp1d(r, z, fill_value="extrapolate")

        nexrad_0_x0 = VT(2.5)
        nexrad_0_x1 = VT(2.5)
        nexrad_0_plot = always_redraw(
            lambda: nexrad_ax.plot(
                z_func_0,
                x_range=[~nexrad_0_x0, ~nexrad_0_x1, 1 / 200],
                color=HPOL_RX_COLOR,
            )
        )

        self.add(nexrad_0_plot)

        self.play(
            LaggedStart(
                beam.animate.rotate(PI / 6, about_point=beam_u.get_start()),
                AnimationGroup(nexrad_90_x0 @ 150, nexrad_0_x1 @ 150),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.remove(nexrad_90_plot)

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                reflectivity_group.width * 1.5
            ).move_to(reflectivity_group)
        )

        self.wait(0.5)

        dx = 0.15
        x_min, x_max = 2.5, 150
        rects = nexrad_ax.get_riemann_rectangles(
            nexrad_0_plot,
            x_range=[x_min, x_max],
            dx=dx,
            input_sample_type="center",
            stroke_width=0,
            fill_opacity=1,
            color=HPOL_RX_COLOR,
            show_signed_area=False,
            # bounded_graph=nexrad_ax.plot(lambda _: -10, x_range=[0, 150, 1 / 200]),
        )

        self.play(LaggedStart(*[FadeIn(rect) for rect in rects], lag_ratio=0.01))

        self.wait(0.5)

        x_samples = np.arange(x_min + dx / 2, x_max, dx)
        rects_color = deepcopy(rects)
        for rect, x_sample in zip(rects_color, x_samples):
            if abs(rect.height) < 0.001:
                rect.set_fill(opacity=0).set_stroke_opacity(0)
                continue
            dbz = float(z_func_0(x_sample))
            color = dbz_to_manim_color(dbz)
            rect.set_fill(color, opacity=1.0)
            rect.set_stroke(color, opacity=0.0)

        self.play(
            LaggedStart(
                *[ReplacementTransform(r, rc) for r, rc in zip(rects, rects_color)],
                lag_ratio=0.01,
            )
        )

        self.remove(radar.vgroup, *beam)

        self.wait(0.5)

        theta_deg = 0.0
        beam_width_deg = 2.0
        theta_mask = _nexrad_theta_band_mask(
            metadata["azimuth_grid_deg"], theta_deg=theta_deg, width_deg=beam_width_deg
        )
        theta_mask &= metadata["valid_mask"]

        theta_rgba = rgba.copy()
        theta_rgba[..., 3] = np.where(theta_mask, theta_rgba[..., 3], 0)

        plot_fill = ManimColor.from_hex("#09151D")
        axis_color = ManimColor.from_hex("#D6EEF7")
        slice_color = ManimColor.from_hex("#9EF6FF")

        plot_radius = nexrad_ax.height * 1.5
        plot_center = ORIGIN
        scan_progress = VT(beam_width_deg / 2.0)
        scan_visibility = VT(0.0)

        plot_background = Circle(
            radius=plot_radius,
            fill_color=plot_fill,
            fill_opacity=1,
            stroke_width=0,
        ).move_to(plot_center)

        plot_border = (
            Circle(radius=plot_radius)
            .move_to(plot_center)
            .set_stroke(axis_color, width=2.2, opacity=0.76)
        )
        origin_marker = Dot(plot_center, radius=0.03, color=axis_color)

        ppi_group = Group(plot_background, plot_border, origin_marker).next_to(
            nexrad_ax, LEFT, LARGE_BUFF * 2
        )

        plot_center = origin_marker.get_center()

        def current_scan_rgba():
            progress_mask = _nexrad_relative_sweep_progress_mask(
                metadata["azimuth_grid_deg"],
                start_theta_deg=theta_deg,
                sweep_progress_deg=~scan_progress,
            )
            return _blend_radar_rgba(theta_rgba, rgba, progress_mask)

        def make_data_image():
            image = ImageMobject(current_scan_rgba(), image_mode="RGBA")
            image.scale_to_fit_height(plot_radius * 2)
            image.move_to(plot_center)
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            return image

        def make_scan_trail():
            head_theta_deg = (theta_deg + ~scan_progress) % 360.0
            trail = AnnularSector(
                inner_radius=0,
                outer_radius=plot_radius * 1.002,
                start_angle=PI / 2 - head_theta_deg * DEGREES,
                angle=beam_width_deg * DEGREES,
                fill_color=slice_color,
                fill_opacity=0.075 * ~scan_visibility,
                stroke_width=0,
            )
            trail.move_arc_center_to(plot_center)
            return trail

        def make_scan_line():
            head_theta_deg = (theta_deg + ~scan_progress) % 360.0
            endpoint = _polar_ppi_point(
                plot_center,
                plot_radius,
                metadata["max_range_km"],
                head_theta_deg,
                metadata["max_range_km"],
            )
            glow = Line(plot_center, endpoint).set_stroke(
                slice_color, width=10, opacity=0.08 * ~scan_visibility
            )
            line = Line(plot_center, endpoint).set_stroke(
                slice_color, width=2.4, opacity=0.95 * ~scan_visibility
            )
            tip_halo = Dot(
                endpoint,
                radius=0.085,
                color=slice_color,
                fill_opacity=0.22 * ~scan_visibility,
                stroke_width=0,
            )
            tip = Dot(
                endpoint,
                radius=0.032,
                color=slice_color,
                fill_opacity=0.95 * ~scan_visibility,
                stroke_width=0,
            )
            return VGroup(glow, line, tip_halo, tip).set_z_index(5)

        data_image = always_redraw(make_data_image)
        scan_trail = always_redraw(make_scan_trail)
        scan_line = always_redraw(make_scan_line)

        # self.add(
        #     plot_background,
        #     data_image,
        #     plot_border,
        #     scan_trail,
        #     scan_line,
        #     origin_marker,
        # )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(reflectivity_group, ppi_group).height * 1.3
                ).move_to(Group(reflectivity_group, ppi_group)),
                Create(plot_border),
                FadeIn(plot_background),
                GrowFromCenter(origin_marker),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    rect.animate.stretch_to_fit_height(scan_trail.width * 0.25).set_y(
                        nexrad_ax.c2p(0, 0)[1]
                    )
                    for rect in rects_color
                ],
                lag_ratio=0.01,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.add(scan_line)

        self.play(
            rects_color.animate.rotate(PI / 2)
            .stretch(0.82, dim=1)
            .next_to(origin_marker, UP, MED_SMALL_BUFF)
            # .shift(RIGHT * 0.5)
        )

        self.play(
            FadeIn(data_image),
            FadeOut(rects_color),
            scan_visibility @ 1,
        )

        self.wait(0.5)

        def get_scan_z_over_r():
            max_range = metadata["max_range_km"]
            r = np.linspace(0, max_range, 1200)
            x = r * np.sin(np.deg2rad(~scan_progress - beam_width_deg / 2))
            y = r * np.cos(np.deg2rad(~scan_progress - beam_width_deg / 2))
            x_idx = np.abs(metadata["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
            y_idx = np.abs(metadata["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
            z = metadata["reflectivity_dbz"][y_idx, x_idx]
            z = np.nan_to_num(z, nan=0)
            order = np.argsort(r)
            r = r[order]
            z = z[order]

            z_func_scan = interp1d(r, z, fill_value="extrapolate")

            nexrad_scan_plot = nexrad_ax.plot(
                z_func_scan,
                x_range=[2.5, 150, 1 / 200],
                color=HPOL_RX_COLOR,
            )
            return nexrad_scan_plot

        scan_z_over_r = always_redraw(get_scan_z_over_r)
        self.remove(nexrad_0_plot)
        self.add(scan_z_over_r)

        self.next_section(skip_animations=skip_animations(False))

        self.play(scan_progress @ 360, run_time=30)

        self.wait(0.5)

        # self.play(self.camera.frame.animate.shift(UP * fh(self, 2)))

        # self.wait(0.5)

        self.play(FadeOut(scan_line))

        # ANIMATIONS HERE

        # self.wait(0.5)
        # self.play(
        #     scan_visibility @ 1,
        #     # slice_outline.animate.set_stroke(opacity=0.3),
        #     scan_progress @ 360,
        #     run_time=3.2,
        #     rate_func=rate_functions.linear,
        # )

        # self.wait(0.5)

        # rain_qmark = (
        #     Text("rain?", font=FONT)
        #     .scale(0.3)
        #     .next_to(reflectivity_ax.i2gp(0.3, reflectivity_plot), UP, SMALL_BUFF)
        #     .shift(UP * 0.5 + LEFT)
        # )
        # hail_qmark = (
        #     Text("hail?", font=FONT)
        #     .scale(0.3)
        #     .next_to(rain_qmark, RIGHT, SMALL_BUFF)
        #     .shift(DOWN * 0.2)
        # )
        # bird_qmark = (
        #     Text("bird?", font=FONT)
        #     .scale(0.3)
        #     .next_to(hail_qmark, RIGHT, SMALL_BUFF)
        #     .shift(DOWN * 0.1 + RIGHT * 0.1)
        # )

        # self.play(FadeIn(rain_qmark))

        # self.wait(0.5)

        # self.play(FadeIn(hail_qmark))

        # self.wait(0.5)

        # self.play(FadeIn(bird_qmark))

        # self.wait(0.5)

        # dual_pol = (
        #     Text("Dual-Pol", font=FONT)
        #     .scale_to_fit_width(fw(self, 0.5))
        #     .move_to(self.camera.frame)
        #     .shift(UP * fh(self, 2))
        # )

        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.shift(UP * fh(self, 2)),
        #         Write(dual_pol),
        #         lag_ratio=0.5,
        #     )
        # )

        # self.wait(0.5)

        # self.play(FadeOut(dual_pol))

        self.wait(2)


class Idea3D(ThreeDScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[-2, 2, 0.5],
            x_length=20,
            y_length=10,
            z_length=20,
            tips=False,
            x_axis_config=dict(
                # stroke_color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            y_axis_config=dict(
                # stroke_color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            z_axis_config=dict(
                # stroke_color=ORANGE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
        )
        self.add(axes)

        target_dist = 40
        u0 = VT(-2)
        u1 = VT(2)
        u0_rx = VT(target_dist)
        u1_rx = VT(target_dist + 4)
        hpol_tx_opacity = VT(1)
        hpol_rx_opacity = VT(0)
        vpol_tx_opacity = VT(0)
        vpol_rx_opacity = VT(0)
        hpol_tx_amp = VT(1)
        vpol_tx_amp = VT(1)
        hpol_rx_amp = VT(1)
        vpol_rx_amp = VT(0.6)
        hpol_rotation = VT(0)
        hpol_tx_highlight_opacity = VT(0)
        hpol_tx_highlight_x0 = VT(0)
        hpol_tx_highlight_x1 = VT(0)
        vpol_tx_highlight_x0 = VT(0)
        vpol_tx_highlight_x1 = VT(0)
        vpol_tx_width = VT(0)
        vpol_tx_rotation = VT(-PI / 2)
        hpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=HPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~hpol_rotation, RIGHT)
        ).set_shade_in_3d(True)
        hpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=YELLOW,
                t_range=(~hpol_tx_highlight_x0, ~hpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=YELLOW,
                t_range=(~vpol_tx_highlight_x0, ~vpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=VPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * ~vpol_tx_width,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~vpol_tx_rotation, RIGHT)
        ).set_shade_in_3d(True)
        hpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_rx_amp * np.sin(2 * PI * u), 0),
                color=HPOL_RX_COLOR,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_rx_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_rx_amp * np.sin(2 * PI * u)),
                color=VPOL_RX_COLOR,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~vpol_rx_opacity,
            )
        ).set_shade_in_3d(True)

        self.add(
            hpol_tx,
            hpol_rx,
            vpol_tx,
            vpol_rx,
            hpol_tx_highlight,
            vpol_tx_highlight,
        )

        self.set_camera_orientation(
            zoom=0.4,
            theta=-90 * DEGREES,
            phi=0 * DEGREES,
            gamma=0,
        )

        self.wait(0.5)

        axes.save_state()
        self.move_camera(
            zoom=0.6,
            theta=-160 * DEGREES,
            phi=80 * DEGREES,
            gamma=0,
            run_time=3,
            added_anims=[
                u0 + 2,
                u1 + 2,
                axes.animate.shift(IN * 5 + UP * 5),
            ],
        )

        self.wait(0.5)

        sp1 = Sphere(
            axes.c2p(target_dist, 0, 0),
            radius=9,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp2 = Sphere(
            axes.c2p(target_dist, 2, -1),
            radius=8,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp3 = Sphere(
            axes.c2p(target_dist, -1, -1.2),
            radius=8,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp4 = Sphere(
            axes.c2p(30, 3, -2),
            radius=9,
            fill_opacity=0.5,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        )
        sp1.set_color(WHITE)
        sp2.set_color(WHITE)
        sp3.set_color(WHITE)
        sp4.set_color(WHITE)

        self.play(LaggedStart(FadeIn(sp1), FadeIn(sp2), FadeIn(sp3), lag_ratio=0.2))

        self.wait(0.5)

        self.play(hpol_rotation @ (2 * PI), run_time=3)

        self.wait(0.5)

        self.play(
            LaggedStart(
                hpol_tx_highlight_opacity @ 1,
                hpol_tx_highlight_x1 @ (~u1),
                hpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                vpol_tx_width @ 2,
                vpol_tx_rotation @ 0,
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        hpol_tx_highlight_x0 @= 0
        hpol_tx_highlight_x1 @= 0

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                hpol_tx_highlight_x1 @ (~u1),
                hpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                vpol_tx_highlight_x1 @ (~u1),
                vpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(u0 + target_dist, u1 + target_dist),
                AnimationGroup(hpol_tx_opacity @ 0, vpol_tx_opacity @ 0),
                AnimationGroup(
                    hpol_rx_opacity @ 1,
                    vpol_rx_opacity @ 1,
                    u0_rx - target_dist,
                    u1_rx - target_dist,
                ),
                lag_ratio=0.3,
            ),
            run_time=10,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                sp1.animate.set_fill(opacity=0),
                sp3.animate.set_fill(opacity=0),
                sp2.animate.set_y(axes.c2p(0, 0, 0)[1]).set_z(axes.c2p(0, 0, 0)[2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.move_camera(
            phi=PI / 2,
            theta=PI,
            # zoom=0.25,
            added_anims=[
                Group(sp3, sp2, sp1).animate.shift(-(IN * 5 + UP * 5)),
                axes.animate.set_stroke(opacity=0.3).shift(-(IN * 5 + UP * 5)),
                hpol_rx_opacity @ 0.7,
                vpol_rx_opacity @ 0.7,
            ],
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        hv_relation = MathTex(r"\lvert H \rvert = \lvert V \rvert")
        hv_relation[0][1].set_color(HPOL_RX_COLOR)
        hv_relation[0][5].set_color(VPOL_RX_COLOR)
        for char in hv_relation[0]:
            char.set_opacity(0)
        self.add_fixed_in_frame_mobjects(hv_relation)
        hv_relation.scale(2).to_corner(DL, LARGE_BUFF)

        self.play(
            LaggedStart(
                vpol_rx_amp @ 1,
                LaggedStart(
                    *[m.animate.set_opacity(1) for m in hv_relation[0]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        hv_relation_gt = MathTex(r"\lvert H \rvert > \lvert V \rvert")
        hv_relation_gt[0][1].set_color(HPOL_RX_COLOR)
        hv_relation_gt[0][5].set_color(VPOL_RX_COLOR)
        hv_relation_gt.scale(2).move_to(hv_relation)

        self.play(
            sp2.animate.stretch(1.5, dim=1),
            *[Transform(a, b) for a, b in zip(hv_relation[0], hv_relation_gt[0])],
            vpol_rx_amp @ (~vpol_rx_amp / 1.5),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                axes.animate.shift([0, 0, 20]),
                Group(sp2, sp3, sp1).animate.shift([0, 0, 100]),
                AnimationGroup(
                    *[
                        m.animate.set_opacity(0)
                        for m in [*hv_relation[0], *hv_relation_gt[0]]
                    ]
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


CHILL_BD_SHOW_REFERENCE = os.getenv("CHILL_BD_SHOW_REFERENCE", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CHILL_BD_REFERENCE_OPACITY = 0.22


class ChillBlockDiagram(Group):
    SOURCE_WIDTH = 487
    SOURCE_HEIGHT = 607

    def __init__(
        self,
        show_reference=CHILL_BD_SHOW_REFERENCE,
        reference_opacity=CHILL_BD_REFERENCE_OPACITY,
        **kwargs,
    ):
        super().__init__(**kwargs)

        asset_path = os.path.join(os.path.dirname(__file__), "static", "chill_bd.png")
        self.reference = ImageMobject(asset_path)
        self.reference.height = config.frame_height * 0.96
        self.reference.move_to(ORIGIN)

        self.px_scale = self.reference.width / self.SOURCE_WIDTH
        self.py_scale = self.reference.height / self.SOURCE_HEIGHT
        self.unit = min(self.px_scale, self.py_scale)
        self.stroke = 1.45
        self.font = FONT or "Liberation Serif"
        self.font_scale = 0.52

        if show_reference:
            self.add(self.reference.copy().set_opacity(reference_opacity))
        self.diagram = self.build_diagram()
        self.add(self.diagram)

    def p(self, x, y):
        return (
            self.reference.get_corner(UL)
            + RIGHT * self.px_scale * x
            + DOWN * self.py_scale * y
        )

    def s(self, px):
        return self.unit * px

    def rp(self, x1, y1, x2, y2, fx, fy):
        return self.p(x1 + (x2 - x1) * fx, y1 + (y2 - y1) * fy)

    def wire(self, start, end, arrow=False):
        if arrow:
            return Arrow(
                self.p(*start),
                self.p(*end),
                buff=0,
                color=BLACK,
                stroke_width=self.stroke,
                tip_length=self.s(8),
                max_stroke_width_to_length_ratio=999,
                max_tip_length_to_length_ratio=0.35,
            )
        return Line(
            self.p(*start),
            self.p(*end),
            color=BLACK,
            stroke_width=self.stroke,
        )

    def polywire(self, *points, end_arrow=False):
        segs = VGroup()
        for idx, (start, end) in enumerate(zip(points, points[1:])):
            segs.add(self.wire(start, end, arrow=end_arrow and idx == len(points) - 2))
        return segs

    def box(self, x1, y1, x2, y2):
        rect = Rectangle(
            width=(x2 - x1) * self.px_scale,
            height=(y2 - y1) * self.py_scale,
            color=BLACK,
            stroke_width=self.stroke,
        )
        rect.move_to((self.p(x1, y1) + self.p(x2, y2)) / 2)
        return rect

    def selector_box(self, x1, y1, x2, y2):
        rect = self.box(x1, y1, x2, y2)
        slash = self.wire(
            (x1 + (x2 - x1) * 0.2, y1 + (y2 - y1) * 0.68),
            (x1 + (x2 - x1) * 0.8, y1 + (y2 - y1) * 0.28),
        )
        return VGroup(rect, slash)

    def switch_box(self, x1, y1, x2, y2):
        rect = self.box(x1, y1, x2, y2)
        return VGroup(
            rect,
            self.wire((x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2)),
            self.wire((x1, y1), (x2, y2)),
            self.wire((x1, y2), (x2, y1)),
        )

    def filter_box(self, x1, y1, x2, y2):
        rect = self.box(x1, y1, x2, y2)
        curve = VMobject(color=BLACK, stroke_width=self.stroke)
        rect.is_filter = True
        curve.is_filter = True
        curve.set_points_smoothly(
            [
                self.rp(x1, y1, x2, y2, 0.23, 0.72),
                self.rp(x1, y1, x2, y2, 0.35, 0.42),
                self.rp(x1, y1, x2, y2, 0.5, 0.18),
                self.rp(x1, y1, x2, y2, 0.65, 0.42),
                self.rp(x1, y1, x2, y2, 0.78, 0.72),
            ]
        )
        return VGroup(rect, curve)

    def amp(self, x1, y1, x2, y2):
        return Polygon(
            self.p(x1, y1),
            self.p(x1, y2),
            self.p(x2, (y1 + y2) / 2),
            color=BLACK,
            stroke_width=self.stroke,
            fill_opacity=0,
        )

    def mixer(self, cx, cy, r=14):
        radius = self.s(r)
        circle = Circle(radius=radius, color=BLACK, stroke_width=self.stroke).move_to(
            self.p(cx, cy)
        )
        diag_1 = Line(
            circle.get_corner(UL) + DR * radius * 0.25,
            circle.get_corner(DR) + UL * radius * 0.25,
            color=BLACK,
            stroke_width=self.stroke,
        )
        diag_2 = Line(
            circle.get_corner(UR) + DL * radius * 0.25,
            circle.get_corner(DL) + UR * radius * 0.25,
            color=BLACK,
            stroke_width=self.stroke,
        )
        circle.is_mixer = True
        diag_1.is_mixer = True
        diag_2.is_mixer = True
        return VGroup(circle, diag_1, diag_2)

    def arrow_circle(self, cx, cy, r=14):
        radius = self.s(r)
        circle = Circle(radius=radius, color=BLACK, stroke_width=self.stroke).move_to(
            self.p(cx, cy)
        )
        inner = Arrow(
            circle.get_center() + LEFT * radius * 0.45,
            circle.get_center() + RIGHT * radius * 0.45,
            buff=0,
            color=BLACK,
            stroke_width=self.stroke,
            tip_length=radius * 0.38,
            max_stroke_width_to_length_ratio=999,
            max_tip_length_to_length_ratio=0.9,
        )
        return VGroup(circle, inner)

    def circulator(self, cx, cy, r=14):
        radius = self.s(r)
        center = self.p(cx, cy)
        circle = Circle(radius=radius, color=BLACK, stroke_width=self.stroke).move_to(
            center
        )
        start = center + LEFT * radius * 0.52 + UP * radius * 0.24
        end = center + LEFT * radius * 0.12 + DOWN * radius * 0.36
        inner = CubicBezier(
            start,
            start + DOWN * radius * 0.40,
            end + LEFT * radius * 0.18 + UP * radius * 0.12,
            end,
            color=BLACK,
            stroke_width=self.stroke * 1.15,
        )
        tip = Triangle(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1,
        ).scale(radius * 0.26)
        tip.rotate(-PI * 0.66)
        tip.shift(end - tip.get_center())
        return VGroup(circle, inner, tip)

    def fit_mobject(self, mob, max_width_px=None, max_height_px=None):
        if max_width_px is not None:
            max_width = self.s(max_width_px)
            if mob.width > max_width:
                mob.scale_to_fit_width(max_width)
        if max_height_px is not None:
            max_height = self.s(max_height_px)
            if mob.height > max_height:
                mob.scale_to_fit_height(max_height)
        return mob

    def text(
        self, text, x, y, font_size=22, max_width_px=None, max_height_px=None, **kwargs
    ):
        mob = Text(
            text,
            font=self.font,
            fill_color=BLACK,
            stroke_width=0,
            font_size=font_size * self.font_scale,
            **kwargs,
        )
        self.fit_mobject(mob, max_width_px=max_width_px, max_height_px=max_height_px)
        mob.move_to(self.p(x, y))
        return mob

    def para(
        self,
        *lines,
        x,
        y,
        font_size=20,
        line_spacing=0.62,
        max_width_px=None,
        max_height_px=None,
        **kwargs,
    ):
        mob = Paragraph(
            *lines,
            font=self.font,
            color=BLACK,
            alignment="left",
            line_spacing=line_spacing,
            font_size=font_size * self.font_scale,
            **kwargs,
        )
        self.fit_mobject(mob, max_width_px=max_width_px, max_height_px=max_height_px)
        mob.move_to(self.p(x, y))
        return mob

    def build_diagram(self):
        diagram = VGroup()

        diagram.add(self.text("Frequency Chain:", 62, 18, font_size=24))
        diagram.add(self.box(20, 48, 96, 116))
        diagram.add(
            self.para(
                "DRX Processor",
                "Clocks",
                x=58,
                y=74,
                font_size=16.5,
                max_width_px=58,
                max_height_px=40,
            ).shift(LEFT * self.s(8))
        )
        diagram.add(self.box(20, 130, 112, 154))
        diagram.add(
            self.text(
                "STALO (HP7268)",
                66,
                142,
                font_size=16,
                max_width_px=78,
                max_height_px=13,
            )
        )

        diagram.add(self.polywire((96, 60), (164, 60), (164, 66), end_arrow=True))
        diagram.add(self.text("40 MHZ (LO)", 135, 49, font_size=14))
        diagram.add(self.polywire((96, 104), (164, 104), (164, 92), end_arrow=True))
        diagram.add(self.text("10 MHZ", 122, 96, font_size=14))
        diagram.add(self.mixer(164, 79))

        diagram.add(self.wire((178, 79), (197, 79), arrow=True))
        diagram.add(self.filter_box(198, 68, 229, 100))
        diagram.add(self.polywire((229, 83), (245, 83), (245, 98), end_arrow=True))

        diagram.add(self.polywire((112, 140), (245, 140), (245, 126), end_arrow=True))
        diagram.add(self.mixer(245, 112))
        diagram.add(self.polywire((156, 140), (156, 168), (176, 168), end_arrow=True))
        diagram.add(self.text("To Receivers", 210, 168, font_size=14))

        diagram.add(self.wire((259, 112), (272, 112), arrow=True))
        diagram.add(self.filter_box(272, 98, 303, 128))
        diagram.add(self.wire((303, 113), (319, 113)))
        diagram.add(self.amp(319, 97, 350, 128))

        diagram.add(self.box(296, 48 - 10, 355, 81 - 10))
        diagram.add(
            self.para(
                "I/Q",
                "Modulator",
                x=325,
                y=63 - 10,
                font_size=16,
                max_width_px=40,
                max_height_px=22,
            )
        )
        diagram.add(
            self.polywire(
                (353, 113),
                (353, 82),
                (272, 82),
                (272, 52),
                (296, 52),
                end_arrow=True,
            )
        )

        diagram.add(self.arrow_circle(385, 62 - 10))
        diagram.add(self.arrow_circle(385, 112))
        diagram.add(self.selector_box(411, 49 - 10, 439, 77 - 10))
        diagram.add(self.selector_box(411, 97, 439, 125))
        diagram.add(self.wire((355, 62 - 10), (371, 62 - 10)))
        diagram.add(self.wire((399, 62 - 10), (411, 62 - 10)))
        diagram.add(self.wire((439, 62 - 10), (469, 62 - 10), arrow=True))
        diagram.add(self.wire((350, 112), (371, 112)))
        diagram.add(self.wire((399, 112), (411, 112)))
        # diagram.add(self.wire((439, 112), (469, 112), arrow=True))
        diagram.add(self.text("RF to Transmitters", 430, 90, font_size=14))
        diagram.add(self.text("V", 479, 52, font_size=16))
        diagram.add(self.text("H", 479, 112, font_size=16))

        diagram.add(
            self.text("Transmitter/Receiver: (H only shown)", 108, 192, font_size=23)
        )
        diagram.add(
            self.polywire(
                (440, 111),
                (450, 111),
                (450, 201),
                (48, 201),
                (48, 227),
                (66, 227),
            )
        )
        diagram.add(self.amp(66, 214, 95, 240))
        diagram.add(
            self.text(
                "IPA",
                x=82,
                y=214,
                font_size=16,
                max_width_px=26,
                max_height_px=18,
            )
        )
        diagram.add(self.wire((95, 227), (103, 227)))
        diagram.add(self.arrow_circle(117, 227))
        diagram.add(self.wire((131, 227), (147, 227)))
        diagram.add(self.amp(147, 214, 178, 240))
        diagram.add(
            self.text(
                "Power Amp",
                x=175,
                y=213,
                font_size=16,
                max_width_px=56,
                max_height_px=20,
            )
        )
        diagram.add(self.wire((178, 227), (232, 227), arrow=True))

        diagram.add(
            self.polywire(
                (192, 246 - 6),
                (192, 238 - 6),
                (212, 238 - 6),
                (212, 292),
                (15, 292),
                (15, 469),
            )
        )
        diagram.add(self.text("50 dB", 190, 248, font_size=15))
        diagram.add(self.text("Coupler", 190, 260, font_size=15))

        diagram.add(self.circulator(246, 227))
        diagram.add(
            DoubleArrow(
                self.p(246, 241),
                self.p(246, 269),
                buff=0,
                color=BLACK,
                stroke_width=self.stroke,
                tip_length=self.s(8),
                max_stroke_width_to_length_ratio=999,
                max_tip_length_to_length_ratio=0.28,
            )
        )
        diagram.add(self.text("To Antenna H", 246, 274, font_size=15))
        diagram.add(self.text("Port", 246, 286, font_size=15))

        diagram.add(self.wire((260, 227), (271, 227), arrow=True))
        diagram.add(self.box(271, 210, 320, 251))
        diagram.add(
            self.para(
                "Power",
                "Limit",
                x=295,
                y=229,
                font_size=15,
                max_width_px=32,
                max_height_px=22,
            )
        )
        diagram.add(self.wire((320, 227), (338, 227), arrow=True))
        diagram.add(self.filter_box(340, 212, 370, 243))
        diagram.add(self.wire((370, 227), (389, 227)))
        diagram.add(self.amp(389, 213, 421, 240))
        diagram.add(self.text("LNA", 414, 211, font_size=16))
        diagram.add(
            self.polywire(
                (421, 227),
                (437, 227),
                (437, 323),
                (48, 323),
            )
        )

        diagram.add(self.polywire((48, 323), (48, 358), (84, 358), end_arrow=True))
        diagram.add(self.text("Transfer Sw.", 97, 340, font_size=16))
        diagram.add(self.switch_box(84, 350, 120, 386))
        diagram.add(self.wire((48, 376), (84, 376), arrow=True))
        diagram.add(self.para("From V", "LNA", x=38, y=392, font_size=16))
        diagram.add(self.wire((120, 376), (144, 376), arrow=True))
        diagram.add(self.para("To V", "Receiver", x=147, y=392, font_size=16))

        diagram.add(self.wire((120, 358), (168, 358)))
        diagram.add(self.filter_box(168, 344, 197, 374))
        diagram.add(self.polywire((197, 356), (240, 356), (240, 383), end_arrow=True))
        diagram.add(self.mixer(240, 397))
        diagram.add(self.polywire((160, 440), (240, 440), (240, 412), end_arrow=True))
        diagram.add(self.text("STALO", 168, 434, font_size=18))

        diagram.add(self.wire((254, 397), (268, 397)))
        diagram.add(self.filter_box(268, 385, 297, 415))
        diagram.add(self.polywire((297, 399), (329, 399), (329, 439), end_arrow=True))
        diagram.add(self.text("50 MHZ IF", 330, 390, font_size=16))

        diagram.add(self.mixer(329, 453))
        diagram.add(self.polywire((271, 479), (329, 479), (329, 467), end_arrow=True))
        diagram.add(self.text("40 MHZ LO", 246, 477, font_size=16))
        diagram.add(self.wire((343, 453), (355, 453)))
        diagram.add(self.filter_box(355, 439, 384, 469))
        diagram.add(
            self.polywire(
                (384, 453),
                (398, 453),
                (398, 500),
                (177, 500),
                (177, 548),
                (273, 548),
                end_arrow=True,
            )
        )

        diagram.add(self.wire((15, 469), (63, 469)))
        diagram.add(self.wire((63, 469), (63, 488), arrow=True))
        diagram.add(self.box(30, 488, 98, 548))
        diagram.add(
            self.para(
                "Down",
                "Conversion",
                "Similar to",
                "Receiver",
                x=63,
                y=519,
                font_size=15,
                line_spacing=0.58,
                max_width_px=44,
                max_height_px=48,
            )
        )
        diagram.add(self.polywire((63, 548), (63, 588), (273, 588), end_arrow=True))
        diagram.add(self.text("10 MHz IF", 138, 580, font_size=16))

        diagram.add(
            self.text(
                "10 MHz IF",
                204,
                523,
                font_size=16,
                max_width_px=62,
                max_height_px=18,
            )
        )
        # diagram.add(self.wire((266, 548), (273, 548), arrow=True))

        diagram.add(self.box(273, 516, 370, 602))
        diagram.add(
            self.text(
                "DRX Processor",
                304,
                524,
                font_size=16,
                max_width_px=74,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "Low Chan.",
                304,
                548,
                font_size=16,
                max_width_px=68,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "High Chan.",
                304,
                568,
                font_size=16,
                max_width_px=70,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "Txmit Chan.",
                304,
                588,
                font_size=16,
                max_width_px=72,
                max_height_px=13,
            )
        )

        diagram.add(self.text("25 dB", 194, 558, font_size=15))
        diagram.add(self.text("Coupler", 197, 574, font_size=15))
        diagram.add(
            self.polywire(
                (212, 562),
                (212, 554),
                (228, 554),
                (228, 568),
                (273, 568),
                end_arrow=True,
            )
        )

        # diagram.add(self.mixer(self.SOURCE_WIDTH / 2, self.SOURCE_HEIGHT / 2))

        return diagram


class Idea2D(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        bd = ChillBlockDiagram()

        center = bd.diagram.get_center()
        components = VGroup()
        for c in deepcopy(bd.diagram):
            if type(c) is VGroup:
                components.add(*c)
            else:
                components.add(c)

        diagram_sorted = sorted(
            components,
            key=lambda x: sqrt(np.sum((center - x.get_center()) ** 2)),
            reverse=False,
        )

        chill_aerial = (
            ImageMobject("../props/static/chill_aerial.jpg")
            .scale_to_fit_width(fw(self, 0.4))
            .next_to(bd, LEFT, LARGE_BUFF)
        )
        self.camera.frame.move_to(chill_aerial)

        self.play(chill_aerial.shift(UP * 10).animate.shift(DOWN * 10))

        self.wait(0.5)

        chill_bez = CubicBezier(
            chill_aerial.get_right() + [0.1, 0, 0],
            chill_aerial.get_right() + [1, 0, 0],
            bd.get_left() + [-0.6, bd.height / 4, 0],
            bd.get_left() + [0.3, bd.height / 4, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).move_to(Group(chill_aerial, bd)),
                Create(chill_bez),
                FadeIn(bd.reference),
                lag_ratio=0.3,
            )
        )

        def get_color_anim(m):
            color = (
                GREEN
                if type(m) is Polygon
                else YELLOW
                if hasattr(m, "is_filter")
                else BLUE
                if hasattr(m, "is_mixer")
                else (WHITE)
            )
            return (
                m.animate.set_color(color)
                if type(m) in (Text, Arrow, Paragraph, Triangle, DoubleArrow)
                else m.animate.set_stroke_color(color)
            )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(chill_aerial),
                Uncreate(chill_bez),
                self.camera.frame.animate.move_to(bd),
                FadeOut(bd.reference, run_time=2),
                LaggedStart(
                    *[
                        Succession(
                            GrowArrow(m, rate_func=rate_functions.ease_out_elastic),
                            get_color_anim(m),
                        )
                        if type(m) is Arrow
                        else Succession(
                            GrowFromCenter(
                                m, rate_func=rate_functions.ease_out_elastic
                            ),
                            get_color_anim(m),
                        )
                        for m in diagram_sorted
                    ],
                    lag_ratio=0.01,
                    run_time=3,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        # indices = [
        #     Text(f"{idx}", color=RED, font_size=DEFAULT_FONT_SIZE * 0.3).move_to(
        #         m.get_center()
        #     )
        #     for idx, m in enumerate(bd.diagram)
        # ]

        # self.add(*indices)

        siggen_opacity = VT(0)
        siggen_x1 = VT((bd.diagram[1].get_corner(UL) + LEFT * 0.1)[0])
        siggen_x2 = VT((bd.diagram[1].get_corner(UL) + LEFT * 0.1)[0])
        siggen_y1 = VT((bd.diagram[1].get_corner(DL) + LEFT * 0.1 + DOWN * 0.1)[1])
        siggen_y2 = VT((bd.diagram[1].get_corner(UL) + LEFT * 0.1 + UP * 0.1)[1])
        siggen_box = always_redraw(
            lambda: Polygon(
                (~siggen_x1, ~siggen_y1, 0),
                (~siggen_x1, ~siggen_y2, 0),
                (~siggen_x2, ~siggen_y2, 0),
                (~siggen_x2, ~siggen_y1, 0),
                fill_color=YELLOW,
                fill_opacity=0.2 * ~siggen_opacity,
                stroke_opacity=0 * ~siggen_opacity,
                stroke_color=YELLOW,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        self.add(siggen_box)

        siggen_group = Group(bd.diagram[1], bd.diagram[9])
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.5).move_to(siggen_group),
                siggen_opacity @ 1,
                siggen_x2 @ ((bd.diagram[9].get_right() + RIGHT * 0.1)[0]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        siggen_group.add(
            bd.diagram[14],
            bd.diagram[4],
            bd.diagram[15],
            bd.diagram[16],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).move_to(siggen_group),
                siggen_x2 @ ((siggen_group.get_right() + RIGHT * 0.05)[0]),
                siggen_y1 @ ((siggen_group.get_bottom() + DOWN * 0.1)[1]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        siggen_x3 = VT(~siggen_x2)
        siggen_x4 = VT(~siggen_x2)
        siggen_y3 = VT((bd.diagram[18].get_corner(UL) + UL * 0.1)[1])
        siggen_y4 = VT((bd.diagram[18].get_corner(DL) + DL * 0.1)[1])

        siggen_amp_box = always_redraw(
            lambda: Polygon(
                (~siggen_x1, ~siggen_y1, 0),
                (~siggen_x1, ~siggen_y2, 0),
                (~siggen_x2, ~siggen_y2, 0),
                (~siggen_x3, ~siggen_y3, 0),
                (~siggen_x4, ~siggen_y3, 0),
                (~siggen_x4, ~siggen_y4, 0),
                (~siggen_x3, ~siggen_y4, 0),
                (~siggen_x2, ~siggen_y1, 0),
                fill_color=YELLOW,
                fill_opacity=0.2 * ~siggen_opacity,
                stroke_opacity=0 * ~siggen_opacity,
                stroke_color=YELLOW,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )

        self.add(siggen_amp_box)
        self.remove(siggen_box)

        self.play(siggen_x4 @ ((bd.diagram[20].get_right() + RIGHT * 0.05)[0]))

        self.wait(0.5)

        gap = 0.05
        vpol_x2 = VT(~siggen_x2 + gap)
        vpol_tx_opacity = VT(0)

        vpol_tx_box = always_redraw(
            lambda: Polygon(
                (~siggen_x2 + gap, ~siggen_y2 + gap * 2, 0),
                (~siggen_x2 + gap, ~siggen_y3 + gap, 0),
                (~vpol_x2, ~siggen_y3 + gap, 0),
                (~vpol_x2, ~siggen_y2 + gap * 2, 0),
                # (~siggen_x3, ~siggen_y4, 0),
                # (~siggen_x2, ~siggen_y1, 0),
                fill_color=VPOL_TX_COLOR,
                fill_opacity=0.2 * ~vpol_tx_opacity,
                stroke_opacity=0 * ~vpol_tx_opacity,
                stroke_color=VPOL_TX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )

        self.add(vpol_tx_box)

        self.play(
            LaggedStart(
                vpol_tx_opacity @ 1,
                vpol_x2 @ (bd.diagram[34].get_right()[0] + 0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        dots = (
            Group(
                Dot(radius=DEFAULT_DOT_RADIUS * 0.8),
                Dot(radius=DEFAULT_DOT_RADIUS * 0.8),
                Dot(radius=DEFAULT_DOT_RADIUS * 0.8),
            )
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(bd.diagram[34], RIGHT, MED_SMALL_BUFF)
        )

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.move_to(Group(vpol_tx_box, siggen_box)),
            LaggedStart(*[GrowFromCenter(m) for m in dots], lag_ratio=0.1),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        hpol_group = Group(
            bd.diagram[25],
            bd.diagram[27],
            bd.diagram[38],
            bd.diagram[41],
            bd.diagram[43],
            bd.diagram[45],
        )
        self.play(
            self.camera.frame.animate.move_to(hpol_group),
            LaggedStart(*[FadeOut(m) for m in dots[::-1]], lag_ratio=0.1),
        )

        self.wait(0.5)

        hpol_tx_opacity = VT(0)

        hpol_tx_x2 = VT(~siggen_x4)
        hpol_tx_y2 = VT(~siggen_y4)
        hpol_tx_y2_static = VT(~siggen_y4)

        hpol_tx_box = always_redraw(
            lambda: Polygon(
                (~siggen_x4 + gap, ~siggen_y4, 0),
                (~siggen_x4 + gap, ~siggen_y3, 0),
                (~hpol_tx_x2, ~siggen_y3, 0),
                (~hpol_tx_x2, ~hpol_tx_y2, 0),
                (~hpol_tx_x2 - gap * 3, ~hpol_tx_y2, 0),
                (~hpol_tx_x2 - gap * 3, ~hpol_tx_y2_static, 0),
                fill_color=HPOL_TX_COLOR,
                fill_opacity=0.2 * ~hpol_tx_opacity,
                stroke_opacity=0 * ~hpol_tx_opacity,
                stroke_color=HPOL_TX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        self.add(hpol_tx_box)

        self.play(
            LaggedStart(
                hpol_tx_opacity @ 1,
                hpol_tx_x2 @ (bd.diagram[27].get_right()[0] + 0.2),
                lag_ratio=0.3,
            ),
            rate_func=rate_functions.ease_in_sine,
        )

        self.play(
            hpol_tx_y2 @ (bd.diagram[39].get_top()[1] + 0.05),
            rate_func=rate_functions.linear,
            run_time=0.4,
        )

        hpol_tx_x3 = VT(~hpol_tx_x2)

        hpol_tx_box_2 = always_redraw(
            lambda: Polygon(
                (~hpol_tx_x2 - gap * 3, ~hpol_tx_y2, 0),
                (~hpol_tx_x2 - gap * 3, ~hpol_tx_y2 + gap * 3, 0),
                (~hpol_tx_x3, ~hpol_tx_y2 + gap * 3, 0),
                (~hpol_tx_x3, ~hpol_tx_y2, 0),
                fill_color=HPOL_TX_COLOR,
                fill_opacity=0.2 * ~hpol_tx_opacity,
                stroke_opacity=0 * ~hpol_tx_opacity,
                stroke_color=HPOL_TX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        self.add(hpol_tx_box_2)

        self.play(
            hpol_tx_x3 @ (bd.diagram[37].get_left()[0] - gap * 2),
            rate_func=rate_functions.linear,
            run_time=0.7,
        )

        hpol_tx_y3 = VT(bd.diagram[38].get_corner(DL)[1] - gap)
        hpol_tx_x4 = VT(~hpol_tx_x3)
        hpol_tx_box_3 = always_redraw(
            lambda: Polygon(
                (~hpol_tx_x3, ~hpol_tx_y2, 0),
                (~hpol_tx_x3, ~hpol_tx_y3, 0),
                (~hpol_tx_x4, ~hpol_tx_y3, 0),
                (~hpol_tx_x4, ~hpol_tx_y2, 0),
                fill_color=HPOL_TX_COLOR,
                fill_opacity=0.2 * ~hpol_tx_opacity,
                stroke_opacity=0 * ~hpol_tx_opacity,
                stroke_color=HPOL_TX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        self.add(hpol_tx_box_3)

        self.play(
            hpol_tx_x4 @ (bd.diagram[45].get_right()[0]),
            rate_func=rate_functions.linear,
            run_time=0.5,
        )

        self.wait(0.5)

        tx_indices = [
            107,
            117,
            118,
            126,
            136,
            137,
            143,
            130,
            27,
            76,
            67,
            54,
            57,
            38,
            39,
            48,
            33,
            23,
            26,
            15,
        ]

        def set_component_opacity(m, opacity, color=None):
            # print(type(m))
            return (
                m.animate.set_opacity(opacity=opacity).set_color(color)
                if type(m) in [Triangle, Arrow, DoubleArrow, Text, Paragraph]
                else m.animate.set_stroke(opacity=opacity, color=color)
            )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            hpol_tx_opacity @ 0,
            vpol_tx_opacity @ 0,
            siggen_opacity @ 0,
            LaggedStart(
                *[
                    set_component_opacity(m, 0.05)
                    for idx, m in enumerate(diagram_sorted)
                    if idx not in tx_indices
                ],
                lag_ratio=0.005,
            ),
        )

        sorted_indices = [
            Text(f"{idx}", color=RED, font_size=DEFAULT_FONT_SIZE * 0.3).move_to(
                m.get_center()
            )
            for idx, m in enumerate(diagram_sorted)
        ]
        # self.add(
        #     *[si for idx, si in enumerate(sorted_indices) if idx not in tx_indices]
        # )

        # self.camera.frame.scale_to_fit_height(
        #     Group(*diagram_sorted).height * 1.1
        # ).move_to(Group(*diagram_sorted))
        # self.play(*[set_component_opacity(m, 1) for m in diagram_sorted])

        self.wait(0.5)

        vpol_tx_bd = Group(*[diagram_sorted[idx].copy() for idx in tx_indices[1:]])
        vpol_tx_bd.shift(diagram_sorted[153].get_right() - vpol_tx_bd[0].get_left())
        vpol_tx_bd[7:].shift(DOWN * 1.5)
        line_to_copy = Line(vpol_tx_bd[5].get_end(), vpol_tx_bd[7].get_start())
        vpol_tx_bd = Group(
            *vpol_tx_bd[:6],
            vpol_tx_bd[6]
            .stretch_to_fit_height(line_to_copy.height)
            .move_to(line_to_copy),
            *vpol_tx_bd[7:],
        )

        # print("hi", vpol_tx_bd[0][0])

        self.play(
            LaggedStart(
                *[
                    TransformFromCopy(diagram_sorted[h_index], v)
                    for h_index, v in zip(tx_indices[1:], vpol_tx_bd)
                ]
            )
        )

        self.wait(0.5)

        # print(vpol_tx_bd[0])

        self.play(
            LaggedStart(
                *[
                    set_component_opacity(diagram_sorted[idx], 1, VPOL_TX_COLOR)
                    for idx in [
                        112,
                        108,
                        111,
                        122,
                        133,
                        131,
                        140,
                        145,
                        146,
                        148,
                        150,
                        151,
                        153,
                    ]
                ],
                *[
                    set_component_opacity(vpol_component, 1, VPOL_TX_COLOR)
                    for vpol_component in vpol_tx_bd
                ],
                diagram_sorted[154]
                .animate.set_opacity(1)
                .set_color(VPOL_TX_COLOR)
                .next_to(diagram_sorted[150], UP, MED_SMALL_BUFF),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        # self.add(vpol_tx_bd)

        for idx in tx_indices:
            diagram_sorted[idx].save_state()

        self.play(
            LaggedStart(
                *[
                    set_component_opacity(diagram_sorted[idx], 1, HPOL_TX_COLOR)
                    for idx in tx_indices
                ],
                diagram_sorted[149]
                .animate.set_opacity(1)
                .set_color(HPOL_TX_COLOR)
                .next_to(diagram_sorted[136], DOWN, MED_SMALL_BUFF),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.5).move_to(diagram_sorted[133]),
                AnimationGroup(
                    diagram_sorted[133].animate.set_color(YELLOW),
                    diagram_sorted[131].animate.set_color(YELLOW),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ps_inp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.3),
            y_length=fh(self, 0.3),
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        ).set_z_index(-2)
        ps_outp_ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.3),
            y_length=fh(self, 0.3),
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        ).set_z_index(-2)
        ps_arrow = Arrow(ORIGIN, RIGHT, color=YELLOW)

        ps_group = (
            Group(ps_inp_ax, ps_arrow, ps_outp_ax)
            .arrange(RIGHT, SMALL_BUFF)
            .next_to(diagram_sorted[133], UP, LARGE_BUFF)
        )

        ps_inp_x1 = VT(0)
        ps_outp_x1 = VT(0)
        ps_inp = always_redraw(
            lambda: ps_inp_ax.plot(
                lambda t: np.sin(2 * PI * 2 * t),
                x_range=[0, ~ps_inp_x1, 1 / 200],
                color=BLUE,
            ).set_z_index(2)
        )
        ps = VT(0)
        ps_outp = always_redraw(
            lambda: ps_outp_ax.plot(
                lambda t: np.sin(2 * PI * 2 * t + ~ps),
                x_range=[0, ~ps_outp_x1, 1 / 200],
                color=RED,
            ).set_z_index(2)
        )

        self.add(ps_inp, ps_outp)

        ps_inp_bez = CubicBezier(
            diagram_sorted[122].get_top() + [0, 0.1, 0],
            diagram_sorted[122].get_top() + [0, 0.7, 0],
            ps_inp_ax.get_bottom() + [0, -0.7, 0],
            ps_inp_ax.get_bottom() + [0, -0.1, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )
        ps_outp_bez = CubicBezier(
            diagram_sorted[140].get_top() + [0, 0.1, 0],
            diagram_sorted[140].get_top() + [0, 0.7, 0],
            ps_outp_ax.get_bottom() + [0, -0.7, 0],
            ps_outp_ax.get_bottom() + [0, -0.1, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(Group(ps_group, diagram_sorted[133])),
                Create(ps_inp_bez),
                Create(ps_inp_ax),
                ps_inp_x1 @ 1,
                GrowArrow(ps_arrow),
                Create(ps_outp_bez),
                Create(ps_outp_ax),
                ps_outp_x1 @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            ps @ (5 * PI / 2),
            rate_func=rate_functions.ease_in_out_elastic,
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ps_inp_x1 @ 0,
                    ps_outp_x1 @ 0,
                    FadeOut(ps_inp_ax, ps_outp_ax, ps_arrow),
                ),
                AnimationGroup(Uncreate(ps_inp_bez), Uncreate(ps_outp_bez)),
                self.camera.frame.animate.restore(),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[set_component_opacity(m, 0) for m in vpol_tx_bd[::-1]],
                *[
                    set_component_opacity(diagram_sorted[idx], 0.1, color=WHITE)
                    for idx in [
                        112,
                        108,
                        111,
                        122,
                        133,
                        131,
                        140,
                        145,
                        146,
                        148,
                        150,
                        151,
                        153,
                        154,
                    ][::-1]
                ],
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[diagram_sorted[idx].animate.restore() for idx in tx_indices],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.8).move_to(
                    Group(diagram_sorted[23], diagram_sorted[54])
                ),
                diagram_sorted[54]
                .animate.set_stroke(width=DEFAULT_STROKE_WIDTH * 0.75)
                .set_color(YELLOW),
                diagram_sorted[23]
                .animate.set_stroke(width=DEFAULT_STROKE_WIDTH * 0.75)
                .set_color(YELLOW),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(
                Group(
                    diagram_sorted[23],
                    diagram_sorted[54],
                    diagram_sorted[11],
                    diagram_sorted[14],
                    diagram_sorted[16],
                    diagram_sorted[2],
                    diagram_sorted[1],
                    diagram_sorted[5],
                )
            ),
            AnimationGroup(
                diagram_sorted[11].animate.set_stroke(opacity=1).set_fill(opacity=1),
                diagram_sorted[14].animate.set_stroke(opacity=1),
                diagram_sorted[16].animate.set_stroke(opacity=1),
            ),
            AnimationGroup(
                diagram_sorted[5].animate.set_stroke(opacity=1).set_fill(opacity=1),
            ),
            diagram_sorted[2].animate.set_stroke(opacity=1).set_fill(opacity=1),
            diagram_sorted[1].animate.set_stroke(opacity=1).set_fill(opacity=1),
        )

        self.wait(0.5)

        rx_indices = [
            17,
            24,
            25,
            37,
            50,
            52,
            61,
            74,
            84,
            89,
            85,
            0,
            63,
            55,
            59,
            42,
            43,
            36,
            44,
            45,
            # down-conversion
            # 32,
            # 31,
            # 21,
            # 8,
            # 9,
            # 4,
            # 7,
            # 18,
            # 19,
            # 20,
            # 46,
            # 41,
            # 34,
            # 22,
            # 28,
            # 29,
            # 35,
            # 40,
            # 51,
            # 64,
            # 65,
            # 66,
            # 58,
            # 68,
            # 73,
            # 69,
            # 81,
            # 82,
            # 91,
            # 110,
            # 77,
            # 92,
            # 88,
            # 108,
        ]
        rx_path_h = Group(*[diagram_sorted[idx] for idx in rx_indices])

        self.play(
            self.camera.frame.animate.move_to(rx_path_h),
            LaggedStart(
                *[
                    set_component_opacity(m, 0.05)
                    for idx, m in enumerate(diagram_sorted)
                    if idx in [*tx_indices, 149]
                ],
                *[
                    set_component_opacity(m, 1)
                    for idx, m in enumerate(diagram_sorted)
                    if idx in rx_indices
                ],
                lag_ratio=0.005,
            ),
        )

        self.wait(0.5)

        rx_path_v = [m.copy() for m in rx_path_h[:-6]]

        new_rx_cable = (
            diagram_sorted[85].copy().stretch(2, 1).move_to(diagram_sorted[85], UP)
        )
        new_rx_path_last = deepcopy(rx_path_h[-9:]).shift(
            new_rx_cable.get_end() - rx_path_h[-9].get_start()
        )

        # rx_path_h[-9].rotate(PI / 6).set_color(YELLOW)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                Group(rx_path_h, new_rx_path_last).height * 1.2
            ).move_to(Group(rx_path_h, new_rx_path_last)),
            LaggedStart(
                Transform(diagram_sorted[85], new_rx_cable.set_color(HPOL_RX_COLOR)),
                *[
                    Transform(a, b.set_color(HPOL_RX_COLOR))
                    for a, b in zip(rx_path_h[-9:], new_rx_path_last)
                ],
                lag_ratio=0.05,
            ),
        )

        self.wait(0.5)

        rx_path_v.pop()
        rx_path_v.append(
            Arrow(
                [rx_path_v[-2].get_end()[0], new_rx_path_last[-6].get_end()[1], 0],
                new_rx_path_last[-6].get_end(),
                color=VPOL_RX_COLOR,
            )
        )
        # rx_path_v[-1].set_color(VPOL_RX_COLOR).shift(
        #     new_rx_path_last[-6].get_end() - rx_path_v[-1].get_end()
        # )
        for m in rx_path_v:
            m.shift(DOWN + LEFT * 0.5)

        rx_path_v[-2] = Line(
            rx_path_v[-2].get_start(),
            [rx_path_v[-2].get_end()[0], new_rx_path_last[-6].get_end()[1], 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
        )
        rx_path_v[-1] = Arrow(
            [rx_path_v[-2].get_end()[0], new_rx_path_last[-6].get_end()[1], 0],
            new_rx_path_last[-6].get_end(),
            color=VPOL_RX_COLOR,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            # max_stroke_width_to_length_ratio=0.25,
            max_tip_length_to_length_ratio=0.1,
            buff=0,
        )

        rx_path_v = Group(*rx_path_v)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                *[TransformFromCopy(h, v) for h, v in zip(rx_path_h, rx_path_v)],
                lag_ratio=0.1,
            )
        )

        # self.add(
        #     rx_path_v,
        #     rx_path_v[-2:],
        #     Dot([rx_path_v[-2].get_end()[0], new_rx_path_last[-6].get_end()[1], 0]),
        #     Dot(new_rx_path_last[-6].get_end()),
        # )

        self.wait(2)


def _thurai_drop_c1(deq):
    return (1 / np.pi) * (0.02914 * deq**2 + 0.9263 * deq + 0.07791)


def _thurai_drop_c2(deq):
    return -0.01938 * deq**2 + 0.4698 * deq + 0.09538


def _thurai_drop_c3(deq):
    return -0.06123 * deq**3 + 1.3880 * deq**2 - 10.41 * deq + 28.34


def _thurai_drop_c4(deq):
    if deq > 4:
        return -0.01352 * deq**3 + 0.2014 * deq**2 - 0.8964 * deq + 1.226
    if 1.5 <= deq <= 4:
        return 0.0
    return np.nan


def _thurai_drop_half_width(deq, y):
    c1_val = _thurai_drop_c1(deq)
    c2_val = _thurai_drop_c2(deq)
    c3_val = _thurai_drop_c3(deq)
    c4_val = _thurai_drop_c4(deq)

    y = np.asarray(y, dtype=float)
    y_over_c2 = np.clip(y / c2_val, -1.0, 1.0)
    term1 = np.sqrt(np.maximum(0.0, 1.0 - y_over_c2**2))
    term2 = np.arccos(np.clip(y / (c3_val * c2_val), -1.0, 1.0))
    term3 = c4_val * y_over_c2**2 + 1.0
    return c1_val * term1 * term2 * term3


def _thurai_drop_outline_points(deq, n_samples=721):
    c2_val = _thurai_drop_c2(deq)
    y_values = np.linspace(-c2_val, c2_val, n_samples)
    x_values = _thurai_drop_half_width(deq, y_values)

    right_side = [np.array([x, y, 0.0]) for x, y in zip(x_values, y_values)]
    left_side = [
        np.array([-x, y, 0.0]) for x, y in zip(x_values[-2:0:-1], y_values[-2:0:-1])
    ]
    return right_side + left_side


def _thurai_drop_flow_lines(deq):
    c2_val = _thurai_drop_c2(deq)
    y_values = np.linspace(-c2_val, c2_val, 361)
    max_half_width = float(np.nanmax(_thurai_drop_half_width(deq, y_values)))
    flow_color = interpolate_color(PRECIP_COLOR, WHITE, 0.22)
    line_specs = [
        dict(offset=0.18 * max_half_width, top_y=1.22, bot_y=-1.18, opacity=0.65),
        dict(offset=0.34 * max_half_width, top_y=1.34, bot_y=-1.32, opacity=0.42),
        dict(offset=0.4 * max_half_width, top_y=1.44, bot_y=-1.42, opacity=0.42),
    ]
    side_y_levels = np.array([0.86, 0.46, -0.08, -0.52, -0.84]) * c2_val

    lines = VGroup()
    for side in (-1, 1):
        for idx, spec in enumerate(line_specs):
            x_levels = _thurai_drop_half_width(deq, side_y_levels)
            flow_line = VMobject()
            flow_line.set_points_smoothly(
                [
                    np.array(
                        [
                            side * (x_levels[0] + spec["offset"] * 0.05),
                            spec["top_y"] * c2_val,
                            0.0,
                        ]
                    ),
                    np.array(
                        [
                            side * (x_levels[0] + spec["offset"] * 0.22),
                            side_y_levels[0] + 0.04 * c2_val,
                            0.0,
                        ]
                    ),
                    np.array(
                        [
                            side * (x_levels[1] + spec["offset"]),
                            side_y_levels[1],
                            0.0,
                        ]
                    ),
                    np.array(
                        [
                            side * (x_levels[2] + spec["offset"] * 0.95),
                            side_y_levels[2],
                            0.0,
                        ]
                    ),
                    np.array(
                        [
                            side * (x_levels[3] + spec["offset"] * 0.55),
                            side_y_levels[3],
                            0.0,
                        ]
                    ),
                    np.array(
                        [
                            side * (x_levels[4] + spec["offset"] * (0.18 + 0.05 * idx)),
                            spec["bot_y"] * c2_val,
                            0.0,
                        ]
                    ),
                ]
            )
            flow_line.set_stroke(
                flow_color,
                width=DEFAULT_STROKE_WIDTH * (2.1 - 0.2 * idx),
                opacity=spec["opacity"],
            )
            lines.add(flow_line.reverse_direction())

    return lines


class DropShape(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        deq_min = 1.5
        deq_target = 6.0
        deq = VT(deq_min)
        drop_scale = 0.95

        def make_drop_shape():
            points = _thurai_drop_outline_points(~deq)
            drop = Polygon(
                *points,
                color=PRECIP_COLOR,
                fill_color=PRECIP_COLOR,
                fill_opacity=0.12,
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
            )
            drop.scale(drop_scale)
            drop.set_z_index(2)
            return drop

        def make_flow_lines(deq):
            flow_lines = _thurai_drop_flow_lines(deq)
            flow_lines.scale(drop_scale * 1.5)
            flow_lines.set_z_index(1)
            return flow_lines

        flow_lines = make_flow_lines(~deq)
        drop_shape = always_redraw(make_drop_shape)

        self.add(drop_shape)

        self.wait(0.5)

        self.play(self.camera.frame.shift(DOWN * fh(self)).animate.shift(UP * fh(self)))

        self.wait(0.5)

        drop_arrow = Arrow(
            drop_shape.get_bottom() + [0, -0.1, 0],
            self.camera.frame.get_bottom() + [0, 0.3, 0],
        )

        self.play(GrowArrow(drop_arrow))

        self.wait(0.5)

        self.play(FadeOut(drop_arrow))

        self.wait(0.5)

        flow_lines = [*flow_lines, *deepcopy(flow_lines), *deepcopy(flow_lines)]

        dots = [
            Dot(fill_opacity=0, stroke_opacity=0).move_to(x.get_start())
            for x in flow_lines
        ]

        traced_paths = [
            TracedPath(
                d.get_center,
                dissipating_time=0.5,
                stroke_opacity=[0, 1],
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            for d in dots
        ]

        self.add(*traced_paths)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        MoveAlongPath(d1.move_to(fl1.get_start()), fl1, run_time=1.5),
                        MoveAlongPath(d2.move_to(fl2.get_start()), fl2, run_time=1.5),
                    )
                    for (d1, d2), (fl1, fl2) in zip(
                        zip(dots[: len(dots) // 2], dots[len(dots) // 2 :]),
                        zip(
                            flow_lines[: len(flow_lines) // 2],
                            flow_lines[len(flow_lines) // 2 :],
                        ),
                    )
                ],
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        flow_lines = sorted(
            [
                *make_flow_lines(~deq),
                *make_flow_lines((deq_target - ~deq) / 3 + ~deq),
                *make_flow_lines(2 * (deq_target - ~deq) / 3 + ~deq),
                *make_flow_lines(deq_target),
            ],
            key=lambda x: x.get_start()[0],
        )

        dots = [
            Dot(fill_opacity=0, stroke_opacity=0).move_to(x.get_start())
            for x in flow_lines
        ]

        traced_paths = [
            TracedPath(
                d.get_center,
                dissipating_time=0.5,
                stroke_opacity=[0, 1],
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            for d in dots
        ]

        self.add(*traced_paths)

        pprint.pprint([m.get_start() for m in flow_lines])

        self.play(
            deq @ deq_target,
            LaggedStart(
                *[
                    AnimationGroup(
                        MoveAlongPath(d1.move_to(fl1.get_start()), fl1, run_time=1.5),
                        MoveAlongPath(d2.move_to(fl2.get_start()), fl2, run_time=1.5),
                    )
                    for (d1, d2), (fl1, fl2) in zip(
                        zip(dots[: len(dots) // 2][::-1], dots[len(dots) // 2 :]),
                        zip(
                            flow_lines[: len(flow_lines) // 2][::-1],
                            flow_lines[len(flow_lines) // 2 :],
                        ),
                    )
                ],
                lag_ratio=0.4,
            ),
            run_time=4,
        )

        self.wait(0.5)

        # self.wait(0.5)

        zdr_gt = MathTex(r"Z_{dr} > 0")
        rr = Text("-> Rain rate++", font=FONT)
        rr[-1:].set_color(GREEN)
        zdr_label = (
            Group(zdr_gt.scale_to_fit_height(rr.height), rr)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(drop_shape, DOWN, LARGE_BUFF),
        )
        zdr_gt.set_y(rr[0].get_y())

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in [*zdr_gt[0], *rr]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)


def _get_nexrad_archive_volume(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
):
    import urllib.request
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parent / "static" / "nexrad"
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_path = cache_dir / Path(s3_key).name
    if not raw_path.exists():
        urllib.request.urlretrieve(
            f"https://unidata-nexrad-level2.s3.amazonaws.com/{s3_key}", raw_path
        )
    return raw_path


def _get_nexrad_field_display_limits(radar, sweep, field_name, field_data):
    if field_name == "reflectivity":
        return -20.0, 75.0, "dBZ"

    if field_name == "velocity":
        nyquist_velocity = np.nan
        instrument_parameters = radar.instrument_parameters or {}
        nyquist_parameter = instrument_parameters.get("nyquist_velocity")
        if nyquist_parameter is not None:
            nyquist_data = np.ma.filled(nyquist_parameter["data"], np.nan).astype(
                np.float32
            )
            sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
            sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
            if len(nyquist_data) >= sweep_end:
                nyquist_velocity = float(
                    np.nanmedian(nyquist_data[sweep_start:sweep_end])
                )
            elif len(nyquist_data) > sweep:
                nyquist_velocity = float(nyquist_data[sweep])

        if not np.isfinite(nyquist_velocity) or nyquist_velocity <= 0:
            finite = np.abs(field_data[np.isfinite(field_data)])
            nyquist_velocity = (
                float(np.nanpercentile(finite, 99)) if finite.size else 35.0
            )

        nyquist_velocity = float(np.ceil(max(5.0, nyquist_velocity) / 5.0) * 5.0)
        return -nyquist_velocity, nyquist_velocity, "m/s"

    finite = field_data[np.isfinite(field_data)]
    if finite.size == 0:
        return 0.0, 1.0, str(radar.fields[field_name].get("units", ""))

    return (
        float(np.nanmin(finite)),
        float(np.nanmax(finite)),
        str(radar.fields[field_name].get("units", "")),
    )


@lru_cache(maxsize=16)
def _get_nexrad_ppi_data_cached(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
    field_name="reflectivity",
):
    from datetime import datetime

    import pyart

    raw_path = _get_nexrad_archive_volume(s3_key=s3_key)
    radar = pyart.io.read_nexrad_archive(str(raw_path))
    field_data = np.ma.filled(
        radar.get_field(sweep, field_name).astype(np.float32), np.nan
    )
    sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
    sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
    azimuths = radar.azimuth["data"][sweep_start:sweep_end].astype(np.float32)
    ranges_km = radar.range["data"].astype(np.float32) / 1000.0

    order = np.argsort(azimuths)
    azimuths = azimuths[order]
    field_data = field_data[order]

    gate_mask = ranges_km <= max_range_km
    ranges_km = ranges_km[gate_mask]
    field_data = field_data[:, gate_mask]

    radial_step_km = float(np.median(np.diff(ranges_km)))
    x_coords = np.linspace(-max_range_km, max_range_km, resolution, dtype=np.float32)
    y_coords = np.linspace(max_range_km, -max_range_km, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    range_grid_km = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    azimuth_grid_deg = (np.degrees(np.arctan2(grid_x, grid_y)) + 360.0) % 360.0

    azimuth_ext = np.concatenate([azimuths, [azimuths[0] + 360.0]])
    azimuth_edges = (azimuth_ext[:-1] + azimuth_ext[1:]) / 2.0
    first_edge = azimuth_edges[-1] - 360.0
    wrapped_azimuth = azimuth_grid_deg.copy()
    wrapped_azimuth[wrapped_azimuth < first_edge] += 360.0
    ray_index = np.searchsorted(azimuth_edges, wrapped_azimuth, side="right") % len(
        azimuths
    )

    range_edges = np.concatenate(
        [
            [max(0.0, ranges_km[0] - radial_step_km / 2.0)],
            ranges_km[:-1] + radial_step_km / 2.0,
            [ranges_km[-1] + radial_step_km / 2.0],
        ]
    )
    gate_index = np.searchsorted(range_edges, range_grid_km, side="right") - 1
    valid_mask = (
        (range_grid_km >= range_edges[0])
        & (range_grid_km <= max_range_km)
        & (gate_index >= 0)
        & (gate_index < len(ranges_km))
    )

    cartesian_field = np.full(range_grid_km.shape, np.nan, dtype=np.float32)
    cartesian_field[valid_mask] = field_data[
        ray_index[valid_mask], gate_index[valid_mask]
    ]
    valid_mask &= np.isfinite(cartesian_field)

    vmin, vmax, units = _get_nexrad_field_display_limits(
        radar, sweep, field_name, cartesian_field
    )

    scan_time = datetime.strptime(
        radar.time["units"].split("since ", 1)[1], "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "station": str(
            radar.metadata.get("instrument_name", s3_key.split("/")[-1][:4])
        ),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elevation_deg": round(float(radar.fixed_angle["data"][sweep]), 2),
        "vcp_pattern": int(radar.metadata.get("vcp_pattern", -1)),
        "sweep": int(sweep),
        "max_range_km": float(max_range_km),
        "field_name": field_name,
        "units": units,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "source": "Unidata NEXRAD Level II archive on AWS S3",
        "field_data": cartesian_field,
        "valid_mask": valid_mask,
        "azimuth_grid_deg": azimuth_grid_deg,
        "range_grid_km": range_grid_km,
        "x_coords_km": x_coords,
        "y_coords_km": y_coords,
    }


@lru_cache(maxsize=16)
def _get_nexrad_fixed_angles_cached(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
):
    import pyart

    raw_path = _get_nexrad_archive_volume(s3_key=s3_key)
    radar = pyart.io.read_nexrad_archive(str(raw_path))
    return tuple(float(angle) for angle in radar.fixed_angle["data"])


@lru_cache(maxsize=32)
def _get_nexrad_valid_sweeps_cached(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
    field_name="reflectivity",
):
    import pyart

    raw_path = _get_nexrad_archive_volume(s3_key=s3_key)
    radar = pyart.io.read_nexrad_archive(str(raw_path))

    if field_name not in radar.fields:
        return tuple()

    valid_sweeps = []
    for candidate_sweep in range(radar.nsweeps):
        try:
            field_data = np.ma.filled(
                radar.get_field(candidate_sweep, field_name).astype(np.float32), np.nan
            )
        except Exception:
            continue

        finite_fraction = float(np.isfinite(field_data).mean())
        if finite_fraction > 0:
            valid_sweeps.append(
                (
                    int(candidate_sweep),
                    float(radar.fixed_angle["data"][candidate_sweep]),
                    finite_fraction,
                )
            )

    return tuple(valid_sweeps)


def _get_nexrad_best_sweep_for_field(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
    requested_sweep=6,
    field_name="reflectivity",
):
    valid_sweeps = _get_nexrad_valid_sweeps_cached(
        s3_key=s3_key,
        field_name=field_name,
    )
    if not valid_sweeps:
        raise RuntimeError(f"No valid sweeps found for NEXRAD field '{field_name}'")

    fixed_angles = _get_nexrad_fixed_angles_cached(s3_key=s3_key)
    requested_angle = None
    if 0 <= requested_sweep < len(fixed_angles):
        requested_angle = fixed_angles[requested_sweep]

    best_sweep, _, _ = min(
        valid_sweeps,
        key=lambda item: (
            abs(item[1] - requested_angle) if requested_angle is not None else 0.0,
            abs(item[0] - requested_sweep),
            -item[2],
        ),
    )
    return best_sweep


def _get_nexrad_reflectivity_ppi_data(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
):
    data = _get_nexrad_ppi_data_cached(
        s3_key=s3_key,
        sweep=sweep,
        max_range_km=max_range_km,
        resolution=resolution,
        field_name="reflectivity",
    )
    data_copy = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    data_copy["reflectivity_dbz"] = data_copy.pop("field_data")
    return data_copy


def _get_nexrad_velocity_ppi_data(
    s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
):
    selected_sweep = _get_nexrad_best_sweep_for_field(
        s3_key=s3_key,
        requested_sweep=sweep,
        field_name="velocity",
    )
    data = _get_nexrad_ppi_data_cached(
        s3_key=s3_key,
        sweep=selected_sweep,
        max_range_km=max_range_km,
        resolution=resolution,
        field_name="velocity",
    )

    data_copy = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    data_copy["velocity_ms"] = data_copy.pop("field_data")
    return data_copy


def _get_nexrad_colormap(cmap_name, fallback_name):
    from matplotlib import colormaps

    try:
        return colormaps[cmap_name]
    except KeyError:
        return colormaps[fallback_name]


def _reflectivity_dbz_to_rgba(
    reflectivity_dbz,
    valid_mask=None,
    vmin=-20.0,
    vmax=75.0,
    cmap_name="NWSRef",
    min_alpha=0.45,
):
    if valid_mask is None:
        valid_mask = np.isfinite(reflectivity_dbz)

    normalized = np.clip((reflectivity_dbz - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_nexrad_colormap(cmap_name, "turbo")
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(reflectivity_dbz.shape, dtype=np.uint8)
    if np.any(valid_mask):
        alpha_strength = min_alpha + (1.0 - min_alpha) * np.clip(
            (reflectivity_dbz[valid_mask] - 2.0) / 30.0, 0.0, 1.0
        )
        alpha[valid_mask] = np.uint8(255 * alpha_strength)
    rgba[..., 3] = alpha
    return rgba


def _velocity_ms_to_rgba(
    velocity_ms,
    valid_mask=None,
    vmin=-35.0,
    vmax=35.0,
    cmap_name="NWSVel",
    min_alpha=0.55,
):
    if valid_mask is None:
        valid_mask = np.isfinite(velocity_ms)

    normalized = np.clip((velocity_ms - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_nexrad_colormap(cmap_name, "coolwarm")
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(velocity_ms.shape, dtype=np.uint8)
    if np.any(valid_mask):
        max_velocity = max(abs(vmin), abs(vmax), 1.0)
        alpha_strength = min_alpha + (1.0 - min_alpha) * np.clip(
            np.abs(velocity_ms[valid_mask]) / max_velocity, 0.0, 1.0
        )
        alpha[valid_mask] = np.uint8(255 * alpha_strength)
    rgba[..., 3] = alpha
    return rgba


def _nexrad_scan_reveal_mask(azimuth_grid_deg, sweep_azimuth_deg, trail_deg=45.0):
    azimuth_lag_deg = (sweep_azimuth_deg - azimuth_grid_deg) % 360.0
    return azimuth_lag_deg <= trail_deg


def _nexrad_sweep_progress_mask(azimuth_grid_deg, sweep_azimuth_deg):
    return azimuth_grid_deg <= sweep_azimuth_deg


def _blend_radar_rgba(base_rgba, next_rgba, update_mask):
    return np.where(update_mask[..., None], next_rgba, base_rgba)


def _build_nexrad_colorbar_image(
    height_px=900,
    width_px=56,
    vmin=-20.0,
    vmax=75.0,
    cmap_name="NWSRef",
    fallback_cmap_name="turbo",
):
    gradient = np.linspace(vmax, vmin, height_px, dtype=np.float32)[:, None]
    cmap = _get_nexrad_colormap(cmap_name, fallback_cmap_name)
    rgba = (cmap((gradient - vmin) / (vmax - vmin)) * 255).astype(np.uint8)
    rgba = np.repeat(rgba, width_px, axis=1)
    rgba[..., 3] = 255
    return rgba


def _polar_ppi_point(center, plot_radius, max_range_km, azimuth_deg, range_km):
    scaled_radius = plot_radius * (range_km / max_range_km)
    theta = azimuth_deg * DEGREES
    return (
        center
        + RIGHT * (scaled_radius * np.sin(theta))
        + UP * (scaled_radius * np.cos(theta))
    )


def _nexrad_theta_band_mask(azimuth_grid_deg, theta_deg, width_deg=8.0):
    delta_deg = ((azimuth_grid_deg - theta_deg + 180.0) % 360.0) - 180.0
    return np.abs(delta_deg) <= width_deg / 2.0


def _nexrad_relative_sweep_progress_mask(
    azimuth_grid_deg, start_theta_deg, sweep_progress_deg
):
    relative_deg = (azimuth_grid_deg - start_theta_deg) % 360.0
    return relative_deg <= sweep_progress_deg


class NEXRADPolarPPI(MovingCameraScene):
    def construct(self):
        metadata = _get_nexrad_reflectivity_ppi_data()
        rgba = _reflectivity_dbz_to_rgba(
            metadata["reflectivity_dbz"],
            valid_mask=metadata["valid_mask"],
            vmin=metadata["vmin"],
            vmax=metadata["vmax"],
        )

        plot_fill = ManimColor.from_hex("#09151D")
        axis_color = ManimColor.from_hex("#D6EEF7")
        label_color = ManimColor.from_hex("#B7D4E2")

        plot_radius = min(fh(self, 0.42), fw(self, 0.32))
        plot_center = LEFT * 2.1

        plot_background = Circle(
            radius=plot_radius,
            fill_color=plot_fill,
            fill_opacity=1,
            stroke_width=0,
        ).move_to(plot_center)
        plot_border = (
            Circle(radius=plot_radius)
            .move_to(plot_center)
            .set_stroke(axis_color, width=2.2, opacity=0.76)
        )

        data_image = (
            ImageMobject(rgba, image_mode="RGBA")
            .scale_to_fit_height(plot_radius * 2)
            .move_to(plot_center)
        )
        data_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        grid = VGroup()
        for azimuth in range(0, 360, 15):
            is_primary = azimuth % 90 == 0
            is_secondary = azimuth % 45 == 0
            grid.add(
                Line(
                    plot_center,
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        azimuth,
                        metadata["max_range_km"],
                    ),
                    stroke_color=axis_color,
                    stroke_width=1.6 if is_primary else 1.15 if is_secondary else 0.9,
                    stroke_opacity=0.22
                    if is_primary
                    else 0.13
                    if is_secondary
                    else 0.06,
                )
            )

        range_rings = VGroup()
        range_labels = VGroup()
        for ring_km in (50, 100, 150):
            ring_radius = plot_radius * ring_km / metadata["max_range_km"]
            range_rings.add(
                Circle(radius=ring_radius)
                .move_to(plot_center)
                .set_stroke(
                    axis_color,
                    width=1.7 if ring_km == 150 else 1.2,
                    opacity=0.18 if ring_km == 150 else 0.1,
                )
            )
            range_labels.add(
                Text(f"{ring_km} km", font=FONT, color=label_color)
                .scale(0.23)
                .move_to(
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        230,
                        ring_km,
                    )
                    + LEFT * 0.14
                    + DOWN * 0.03
                )
            )

        direction_labels = VGroup()
        for label, azimuth in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            direction_labels.add(
                Text(label, font=FONT, color=axis_color)
                .scale(0.24)
                .move_to(
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        azimuth,
                        metadata["max_range_km"] * 1.08,
                    )
                )
            )

        origin_marker = Dot(plot_center, radius=0.03, color=axis_color)

        colorbar_units = Text("dBZ", font=FONT, color=label_color).scale(0.22)

        colorbar = ImageMobject(
            _build_nexrad_colorbar_image(vmin=metadata["vmin"], vmax=metadata["vmax"]),
            image_mode="RGBA",
        ).scale_to_fit_height(plot_radius * 1.72)
        colorbar.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        colorbar_frame = Rectangle(
            width=colorbar.width + 0.18,
            height=colorbar.height + 0.18,
            fill_opacity=0,
            stroke_color=axis_color,
            stroke_opacity=0.42,
            stroke_width=1.2,
        ).move_to(colorbar)

        colorbar_group = Group(colorbar_frame, colorbar)
        colorbar_group.next_to(plot_background, RIGHT, buff=0.95)
        colorbar_group.align_to(plot_background, UP).shift(DOWN * 0.02)
        colorbar_units.next_to(colorbar_group, UP, buff=0.16)

        tick_labels = VGroup()
        for tick in (-20, 0, 20, 40, 60):
            y_offset = colorbar.height * (
                (tick - metadata["vmin"]) / (metadata["vmax"] - metadata["vmin"])
            )
            tick_y = colorbar.get_bottom()[1] + y_offset
            tick_anchor = np.array([colorbar.get_right()[0], tick_y, 0])
            tick_line = Line(
                tick_anchor + RIGHT * 0.02,
                tick_anchor + RIGHT * 0.14,
                stroke_color=axis_color,
                stroke_width=1,
                stroke_opacity=0.6,
            )
            tick_text = Text(f"{tick}", font=FONT, color=label_color).scale(0.2)
            tick_text.next_to(tick_line, RIGHT, buff=0.07)
            tick_labels.add(VGroup(tick_line, tick_text))

        chart_group = Group(
            plot_background,
            grid,
            data_image,
            range_rings,
            plot_border,
            range_labels,
            direction_labels,
            origin_marker,
            colorbar_group,
            tick_labels,
            colorbar_units,
        )
        chart_group.move_to(ORIGIN)

        self.add(
            plot_background,
            grid,
            data_image,
            range_rings,
            plot_border,
            # range_labels,
            # direction_labels,
            origin_marker,
            # colorbar_group,
            # tick_labels,
            # colorbar_units,
        )


class NEXRADPolarPPIThetaSlice(MovingCameraScene):
    def construct(self):
        metadata = _get_nexrad_reflectivity_ppi_data()
        rgba = _reflectivity_dbz_to_rgba(
            metadata["reflectivity_dbz"],
            valid_mask=metadata["valid_mask"],
            vmin=metadata["vmin"],
            vmax=metadata["vmax"],
        )

        theta_deg = 0.0
        beam_width_deg = 2.0
        theta_mask = _nexrad_theta_band_mask(
            metadata["azimuth_grid_deg"], theta_deg=theta_deg, width_deg=beam_width_deg
        )
        theta_mask &= metadata["valid_mask"]

        theta_rgba = rgba.copy()
        theta_rgba[..., 3] = np.where(theta_mask, theta_rgba[..., 3], 0)

        plot_fill = ManimColor.from_hex("#09151D")
        axis_color = ManimColor.from_hex("#D6EEF7")
        slice_color = ManimColor.from_hex("#9EF6FF")

        plot_radius = min(fh(self, 0.42), fw(self, 0.42))
        plot_center = ORIGIN
        scan_progress = VT(beam_width_deg / 2.0)
        scan_visibility = VT(0.0)

        plot_background = Circle(
            radius=plot_radius,
            fill_color=plot_fill,
            fill_opacity=1,
            stroke_width=0,
        ).move_to(plot_center)

        grid = VGroup()
        for azimuth in range(0, 360, 15):
            is_primary = azimuth % 90 == 0
            is_secondary = azimuth % 45 == 0
            grid.add(
                Line(
                    plot_center,
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        azimuth,
                        metadata["max_range_km"],
                    ),
                    stroke_color=axis_color,
                    stroke_width=1.6 if is_primary else 1.15 if is_secondary else 0.9,
                    stroke_opacity=0.22
                    if is_primary
                    else 0.13
                    if is_secondary
                    else 0.06,
                )
            )

        range_rings = VGroup()
        for ring_km in (50, 100, 150):
            ring_radius = plot_radius * ring_km / metadata["max_range_km"]
            range_rings.add(
                Circle(radius=ring_radius)
                .move_to(plot_center)
                .set_stroke(
                    axis_color,
                    width=1.7 if ring_km == 150 else 1.2,
                    opacity=0.18 if ring_km == 150 else 0.1,
                )
            )

        plot_border = (
            Circle(radius=plot_radius)
            .move_to(plot_center)
            .set_stroke(axis_color, width=2.2, opacity=0.76)
        )
        origin_marker = Dot(plot_center, radius=0.03, color=axis_color)

        slice_outline = AnnularSector(
            inner_radius=0,
            outer_radius=plot_radius * 1.002,
            start_angle=PI / 2 - (theta_deg + beam_width_deg / 2) * DEGREES,
            angle=beam_width_deg * DEGREES,
            fill_opacity=0,
            stroke_width=2,
            stroke_opacity=0,
            stroke_color=slice_color,
        )
        slice_outline.move_arc_center_to(plot_center)

        def current_scan_rgba():
            progress_mask = _nexrad_relative_sweep_progress_mask(
                metadata["azimuth_grid_deg"],
                start_theta_deg=theta_deg,
                sweep_progress_deg=~scan_progress,
            )
            return _blend_radar_rgba(theta_rgba, rgba, progress_mask)

        def make_data_image():
            image = ImageMobject(current_scan_rgba(), image_mode="RGBA")
            image.scale_to_fit_height(plot_radius * 2)
            image.move_to(plot_center)
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            return image

        def make_scan_trail():
            head_theta_deg = (theta_deg + ~scan_progress) % 360.0
            trail = AnnularSector(
                inner_radius=0,
                outer_radius=plot_radius * 1.002,
                start_angle=PI / 2 - head_theta_deg * DEGREES,
                angle=beam_width_deg * DEGREES,
                fill_color=slice_color,
                fill_opacity=0.075 * ~scan_visibility,
                stroke_width=0,
            )
            trail.move_arc_center_to(plot_center)
            return trail

        def make_scan_line():
            head_theta_deg = (theta_deg + ~scan_progress) % 360.0
            endpoint = _polar_ppi_point(
                plot_center,
                plot_radius,
                metadata["max_range_km"],
                head_theta_deg,
                metadata["max_range_km"],
            )
            glow = Line(plot_center, endpoint).set_stroke(
                slice_color, width=10, opacity=0.08 * ~scan_visibility
            )
            line = Line(plot_center, endpoint).set_stroke(
                slice_color, width=2.4, opacity=0.95 * ~scan_visibility
            )
            tip_halo = Dot(
                endpoint,
                radius=0.085,
                color=slice_color,
                fill_opacity=0.22 * ~scan_visibility,
                stroke_width=0,
            )
            tip = Dot(
                endpoint,
                radius=0.032,
                color=slice_color,
                fill_opacity=0.95 * ~scan_visibility,
                stroke_width=0,
            )
            return VGroup(glow, line, tip_halo, tip)

        data_image = always_redraw(make_data_image)
        scan_trail = always_redraw(make_scan_trail)
        scan_line = always_redraw(make_scan_line)

        self.add(
            plot_background,
            grid,
            data_image,
            range_rings,
            plot_border,
            slice_outline,
            scan_trail,
            scan_line,
            origin_marker,
        )

        self.wait(0.5)
        self.play(
            scan_visibility @ 1,
            # slice_outline.animate.set_stroke(opacity=0.3),
            scan_progress @ 360,
            run_time=3.2,
            rate_func=rate_functions.linear,
        )
        # self.play(
        #     scan_visibility @ 0,
        #     slice_outline.animate.set_stroke(opacity=0),
        #     run_time=0.35,
        # )
        self.wait(0.3)


class NEXRADPolarPPIScan(MovingCameraScene):
    def construct(self):
        volume_keys = (
            "2024/05/07/KTLX/KTLX20240507_023811_V06",
            "2024/05/07/KTLX/KTLX20240507_024501_V06",
            "2024/05/07/KTLX/KTLX20240507_025126_V06",
        )
        volumes = [
            _get_nexrad_reflectivity_ppi_data(
                s3_key=key,
                sweep=6,
                max_range_km=150.0,
                resolution=768,
            )
            for key in volume_keys
        ]
        metadata = volumes[0]
        rgba_frames = [
            _reflectivity_dbz_to_rgba(
                volume["reflectivity_dbz"],
                valid_mask=volume["valid_mask"],
                vmin=volume["vmin"],
                vmax=volume["vmax"],
            )
            for volume in volumes
        ]
        blank_rgba = np.zeros_like(rgba_frames[0])

        plot_fill = ManimColor.from_hex("#09151D")
        axis_color = ManimColor.from_hex("#D6EEF7")
        label_color = ManimColor.from_hex("#B7D4E2")
        scan_color = ManimColor.from_hex("#9EF6FF")

        plot_radius = min(fh(self, 0.42), fw(self, 0.32))
        plot_center = LEFT * 1.35

        plot_background = Circle(
            radius=plot_radius,
            fill_color=plot_fill,
            fill_opacity=1,
            stroke_width=0,
        ).move_to(plot_center)
        plot_border = (
            Circle(radius=plot_radius)
            .move_to(plot_center)
            .set_stroke(axis_color, width=2.2, opacity=0.76)
        )

        grid = VGroup()
        for azimuth in range(0, 360, 15):
            is_primary = azimuth % 90 == 0
            is_secondary = azimuth % 45 == 0
            grid.add(
                Line(
                    plot_center,
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        azimuth,
                        metadata["max_range_km"],
                    ),
                    stroke_color=axis_color,
                    stroke_width=1.6 if is_primary else 1.15 if is_secondary else 0.9,
                    stroke_opacity=0.22
                    if is_primary
                    else 0.13
                    if is_secondary
                    else 0.06,
                )
            )

        range_rings = VGroup()
        range_labels = VGroup()
        for ring_km in (50, 100, 150):
            ring_radius = plot_radius * ring_km / metadata["max_range_km"]
            range_rings.add(
                Circle(radius=ring_radius)
                .move_to(plot_center)
                .set_stroke(
                    axis_color,
                    width=1.7 if ring_km == 150 else 1.2,
                    opacity=0.18 if ring_km == 150 else 0.1,
                )
            )
            range_labels.add(
                Text(f"{ring_km} km", font=FONT, color=label_color)
                .scale(0.23)
                .move_to(
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        230,
                        ring_km,
                    )
                    + LEFT * 0.14
                    + DOWN * 0.03
                )
            )

        direction_labels = VGroup()
        for label, azimuth in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            direction_labels.add(
                Text(label, font=FONT, color=axis_color)
                .scale(0.24)
                .move_to(
                    _polar_ppi_point(
                        plot_center,
                        plot_radius,
                        metadata["max_range_km"],
                        azimuth,
                        metadata["max_range_km"] * 1.08,
                    )
                )
            )

        origin_marker = Dot(plot_center, radius=0.03, color=axis_color)

        colorbar_units = Text("dBZ", font=FONT, color=label_color).scale(0.22)
        colorbar = ImageMobject(
            _build_nexrad_colorbar_image(vmin=metadata["vmin"], vmax=metadata["vmax"]),
            image_mode="RGBA",
        ).scale_to_fit_height(plot_radius * 1.72)
        colorbar.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        colorbar_frame = Rectangle(
            width=colorbar.width + 0.18,
            height=colorbar.height + 0.18,
            fill_opacity=0,
            stroke_color=axis_color,
            stroke_opacity=0.42,
            stroke_width=1.2,
        ).move_to(colorbar)
        colorbar_group = Group(colorbar_frame, colorbar)
        colorbar_group.next_to(plot_background, RIGHT, buff=0.95)
        colorbar_group.align_to(plot_background, UP).shift(DOWN * 0.02)
        colorbar_units.next_to(colorbar_group, UP, buff=0.16)

        tick_labels = VGroup()
        for tick in (-20, 0, 20, 40, 60):
            y_offset = colorbar.height * (
                (tick - metadata["vmin"]) / (metadata["vmax"] - metadata["vmin"])
            )
            tick_y = colorbar.get_bottom()[1] + y_offset
            tick_anchor = np.array([colorbar.get_right()[0], tick_y, 0])
            tick_line = Line(
                tick_anchor + RIGHT * 0.02,
                tick_anchor + RIGHT * 0.14,
                stroke_color=axis_color,
                stroke_width=1,
                stroke_opacity=0.6,
            )
            tick_text = Text(f"{tick}", font=FONT, color=label_color).scale(0.2)
            tick_text.next_to(tick_line, RIGHT, buff=0.07)
            tick_labels.add(VGroup(tick_line, tick_text))

        scan_angle = ValueTracker(0.0)
        scan_visibility = ValueTracker(1.0)
        scan_state = {"base_idx": -1, "target_idx": 0}

        def current_scan_rgba():
            base_rgba = (
                blank_rgba
                if scan_state["base_idx"] < 0
                else rgba_frames[scan_state["base_idx"]]
            )
            target_rgba = rgba_frames[scan_state["target_idx"]]
            progress_mask = _nexrad_sweep_progress_mask(
                metadata["azimuth_grid_deg"], scan_angle.get_value()
            )
            return _blend_radar_rgba(base_rgba, target_rgba, progress_mask)

        def make_data_image():
            image = ImageMobject(current_scan_rgba(), image_mode="RGBA")
            image.scale_to_fit_height(plot_radius * 2)
            image.move_to(plot_center)
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            return image

        def make_scan_trail():
            trail = AnnularSector(
                inner_radius=0,
                outer_radius=plot_radius * 1.002,
                start_angle=PI / 2 - scan_angle.get_value() * DEGREES,
                angle=5 * DEGREES,
                fill_color=scan_color,
                fill_opacity=0.075 * scan_visibility.get_value(),
                stroke_width=0,
            )
            trail.move_arc_center_to(plot_center)
            return trail

        def make_scan_line():
            endpoint = _polar_ppi_point(
                plot_center,
                plot_radius,
                metadata["max_range_km"],
                scan_angle.get_value(),
                metadata["max_range_km"],
            )
            glow = Line(plot_center, endpoint).set_stroke(
                scan_color, width=10, opacity=0.08 * scan_visibility.get_value()
            )
            line = Line(plot_center, endpoint).set_stroke(
                scan_color, width=2.4, opacity=0.95 * scan_visibility.get_value()
            )
            tip_halo = Dot(
                endpoint,
                radius=0.085,
                color=scan_color,
                fill_opacity=0.22 * scan_visibility.get_value(),
                stroke_width=0,
            )
            tip = Dot(
                endpoint,
                radius=0.032,
                color=scan_color,
                fill_opacity=0.95 * scan_visibility.get_value(),
                stroke_width=0,
            )
            return VGroup(glow, line, tip_halo, tip)

        data_image = always_redraw(make_data_image)
        scan_trail = always_redraw(make_scan_trail)
        scan_line = always_redraw(make_scan_line)

        self.add(
            plot_background,
            grid,
            data_image,
            range_rings,
            plot_border,
            range_labels,
            direction_labels,
            scan_trail,
            scan_line,
            origin_marker,
            colorbar_group,
            tick_labels,
            colorbar_units,
        )

        sweep_duration = 3.2
        self.wait(0.2)

        for idx in range(len(rgba_frames)):
            scan_state["target_idx"] = idx
            scan_angle.set_value(0)
            self.play(
                scan_angle.animate.set_value(360),
                run_time=sweep_duration,
                rate_func=linear,
            )
            scan_state["base_idx"] = idx
            if idx < len(rgba_frames) - 1:
                self.wait(0.1)

        self.play(scan_visibility.animate.set_value(0), run_time=0.35)
        self.wait(0.3)


class DimPPIConversion(MovingCameraScene):
    def construct(self):
        metadata = _get_nexrad_reflectivity_ppi_data()
        rgba = _reflectivity_dbz_to_rgba(
            metadata["reflectivity_dbz"],
            valid_mask=metadata["valid_mask"],
            vmin=metadata["vmin"],
            vmax=metadata["vmax"],
        )

        theta = 135.0
        x0 = np.argsort(np.abs(metadata["x_coords_km"]))[:2]
        north = metadata["y_coords_km"] >= 0
        r = metadata["range_grid_km"][north][:, x0].mean(axis=1)
        z = np.nanmean(metadata["reflectivity_dbz"][north][:, x0], axis=1)
        valid = np.any(metadata["valid_mask"][north][:, x0], axis=1)
        order = np.argsort(r)
        r = r[order]
        z = z[order]
        valid = valid[order]
        r_valid = r[valid]
        z_valid = z[valid]

        z_func = interp1d(r_valid, z_valid, fill_value="extrapolate")

        ax = Axes(
            x_range=[0, 120, 25],
            y_range=[-10, 60, 10],
            x_length=fw(self, 0.7),
            y_length=fh(self, 0.6),
            tips=False,
        )

        plot = ax.plot(z_func, x_range=[2.5, 120, 1 / 1000], color=BLUE)

        def dbz_to_manim_color(dbz):
            rgba = (
                _reflectivity_dbz_to_rgba(
                    np.array([[dbz]], dtype=np.float32),
                    valid_mask=np.array([[True]]),
                    vmin=metadata["vmin"],
                    vmax=metadata["vmax"],
                    min_alpha=1.0,
                )[0, 0]
                / 255.0
            )
            return rgb_to_color(rgba[:3])

        dx = 0.15
        x_min, x_max = 2.5, 120
        rects = ax.get_riemann_rectangles(
            plot,
            x_range=[x_min, x_max],
            dx=dx,
            input_sample_type="center",
            stroke_width=0,
            fill_opacity=1,
        )
        x_samples = np.arange(x_min + dx / 2, x_max, dx)
        for rect, x_sample in zip(rects, x_samples):
            dbz = float(z_func(x_sample))
            color = dbz_to_manim_color(dbz)
            rect.set_fill(color, opacity=1.0)
            rect.set_stroke(color, opacity=0.0)

        self.add(ax, plot, rects)

        self.wait(0.5)


class ZDR(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        s3_key = "2024/05/07/KTLX/KTLX20240507_023811_V06"
        sweep = 0
        max_range_km = 150.0
        resolution = 640

        reflectivity_metadata = _get_nexrad_reflectivity_ppi_data(
            s3_key=s3_key,
            sweep=sweep,
            max_range_km=max_range_km,
            resolution=resolution,
        )
        velocity_metadata = _get_nexrad_velocity_ppi_data(
            s3_key=s3_key,
            sweep=sweep,
            max_range_km=max_range_km,
            resolution=resolution,
        )

        reflectivity_rgba = _reflectivity_dbz_to_rgba(
            reflectivity_metadata["reflectivity_dbz"],
            valid_mask=reflectivity_metadata["valid_mask"],
            vmin=reflectivity_metadata["vmin"],
            vmax=reflectivity_metadata["vmax"],
        )
        velocity_rgba = _velocity_ms_to_rgba(
            velocity_metadata["velocity_ms"],
            valid_mask=velocity_metadata["valid_mask"],
            vmin=velocity_metadata["vmin"],
            vmax=velocity_metadata["vmax"],
        )

        plot_fill = ManimColor.from_hex("#09151D")
        axis_color = ManimColor.from_hex("#D6EEF7")
        label_color = ManimColor.from_hex("#B7D4E2")
        scan_color = ManimColor.from_hex("#9EF6FF")

        plot_radius = min(fh(self, 0.23), fw(self, 0.15))
        beam_width_deg = 4.0
        scan_progress = VT(0.0)
        scan_visibility = VT(0.0)

        def format_tick_label(value):
            if np.isclose(value, 0.0):
                value = 0.0
            if np.isclose(value, round(value)):
                return f"{int(round(value))}"
            return f"{value:.1f}"

        def make_ppi_panel(metadata, rgba, title_text, units, tick_values, cmap_name):
            plot_background = Circle(
                radius=plot_radius,
                fill_color=plot_fill,
                fill_opacity=1,
                stroke_width=0,
            )
            plot_border = Circle(radius=plot_radius).set_stroke(
                axis_color, width=2.2, opacity=0.76
            )
            origin_marker = Dot(ORIGIN, radius=0.03, color=axis_color)

            grid = VGroup()
            for azimuth in range(0, 360, 15):
                is_primary = azimuth % 90 == 0
                is_secondary = azimuth % 45 == 0
                grid.add(
                    Line(
                        ORIGIN,
                        _polar_ppi_point(
                            ORIGIN,
                            plot_radius,
                            metadata["max_range_km"],
                            azimuth,
                            metadata["max_range_km"],
                        ),
                        stroke_color=axis_color,
                        stroke_width=1.6
                        if is_primary
                        else 1.15
                        if is_secondary
                        else 0.9,
                        stroke_opacity=0.22
                        if is_primary
                        else 0.13
                        if is_secondary
                        else 0.06,
                    )
                )

            range_rings = VGroup()
            range_labels = VGroup()
            for ring_km in (50, 100, 150):
                ring_radius = plot_radius * ring_km / metadata["max_range_km"]
                range_rings.add(
                    Circle(radius=ring_radius).set_stroke(
                        axis_color,
                        width=1.7 if ring_km == 150 else 1.2,
                        opacity=0.18 if ring_km == 150 else 0.1,
                    )
                )
                range_labels.add(
                    Text(f"{ring_km} km", font=FONT, color=label_color)
                    .scale(0.2)
                    .move_to(
                        _polar_ppi_point(
                            ORIGIN,
                            plot_radius,
                            metadata["max_range_km"],
                            230,
                            ring_km,
                        )
                        + LEFT * 0.1
                        + DOWN * 0.02
                    )
                )

            direction_labels = VGroup()
            for label, azimuth in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
                direction_labels.add(
                    Text(label, font=FONT, color=axis_color)
                    .scale(0.22)
                    .move_to(
                        _polar_ppi_point(
                            ORIGIN,
                            plot_radius,
                            metadata["max_range_km"],
                            azimuth,
                            metadata["max_range_km"] * 1.08,
                        )
                    )
                )

            title = Text(title_text, font=FONT).scale(0.42)
            title.next_to(plot_border, UP, MED_SMALL_BUFF)

            colorbar = ImageMobject(
                _build_nexrad_colorbar_image(
                    height_px=720,
                    vmin=metadata["vmin"],
                    vmax=metadata["vmax"],
                    cmap_name=cmap_name,
                    fallback_cmap_name="coolwarm" if cmap_name == "NWSVel" else "turbo",
                ),
                image_mode="RGBA",
            ).scale_to_fit_height(plot_radius * 1.68)
            colorbar.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            colorbar_frame = Rectangle(
                width=colorbar.width + 0.16,
                height=colorbar.height + 0.16,
                fill_opacity=0,
                stroke_color=axis_color,
                stroke_opacity=0.42,
                stroke_width=1.2,
            ).move_to(colorbar)
            colorbar_group = Group(colorbar_frame, colorbar)
            colorbar_group.next_to(plot_background, RIGHT, buff=0.7)
            colorbar_group.align_to(plot_background, UP).shift(DOWN * 0.02)

            colorbar_units = Text(units, font=FONT, color=label_color).scale(0.2)
            colorbar_units.next_to(colorbar_group, UP, buff=0.12)

            tick_labels = VGroup()
            for tick in tick_values:
                tick_y = colorbar.get_bottom()[1] + colorbar.height * (
                    (tick - metadata["vmin"]) / (metadata["vmax"] - metadata["vmin"])
                )
                tick_anchor = np.array([colorbar.get_right()[0], tick_y, 0])
                tick_line = Line(
                    tick_anchor + RIGHT * 0.02,
                    tick_anchor + RIGHT * 0.12,
                    stroke_color=axis_color,
                    stroke_width=1,
                    stroke_opacity=0.6,
                )
                tick_text = Text(
                    format_tick_label(tick), font=FONT, color=label_color
                ).scale(0.18)
                tick_text.next_to(tick_line, RIGHT, buff=0.06)
                tick_labels.add(VGroup(tick_line, tick_text))

            blank_rgba = np.zeros_like(rgba)

            def current_scan_rgba():
                progress_mask = _nexrad_relative_sweep_progress_mask(
                    metadata["azimuth_grid_deg"],
                    start_theta_deg=0.0,
                    sweep_progress_deg=~scan_progress,
                )
                return _blend_radar_rgba(blank_rgba, rgba, progress_mask)

            def make_data_image():
                image = ImageMobject(current_scan_rgba(), image_mode="RGBA")
                image.scale_to_fit_height(plot_radius * 2)
                image.move_to(origin_marker.get_center())
                image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
                return image

            def make_scan_trail():
                head_theta_deg = ~scan_progress % 360.0
                trail = AnnularSector(
                    inner_radius=0,
                    outer_radius=plot_radius * 1.002,
                    start_angle=PI / 2 - head_theta_deg * DEGREES,
                    angle=beam_width_deg * DEGREES,
                    fill_color=scan_color,
                    fill_opacity=0.075 * ~scan_visibility,
                    stroke_width=0,
                )
                trail.move_arc_center_to(origin_marker.get_center())
                return trail

            def make_scan_line():
                head_theta_deg = ~scan_progress % 360.0
                plot_center = origin_marker.get_center()
                endpoint = _polar_ppi_point(
                    plot_center,
                    plot_radius,
                    metadata["max_range_km"],
                    head_theta_deg,
                    metadata["max_range_km"],
                )
                glow = Line(plot_center, endpoint).set_stroke(
                    scan_color, width=9, opacity=0.08 * ~scan_visibility
                )
                line = Line(plot_center, endpoint).set_stroke(
                    scan_color, width=2.2, opacity=0.95 * ~scan_visibility
                )
                tip_halo = Dot(
                    endpoint,
                    radius=0.08,
                    color=scan_color,
                    fill_opacity=0.22 * ~scan_visibility,
                    stroke_width=0,
                )
                tip = Dot(
                    endpoint,
                    radius=0.03,
                    color=scan_color,
                    fill_opacity=0.95 * ~scan_visibility,
                    stroke_width=0,
                )
                return VGroup(glow, line, tip_halo, tip).set_z_index(5)

            data_image = always_redraw(make_data_image)
            scan_trail = always_redraw(make_scan_trail).set_z_index(4)
            scan_line = always_redraw(make_scan_line)

            return Group(
                plot_background,
                grid,
                data_image,
                range_rings,
                plot_border,
                range_labels,
                direction_labels,
                scan_trail,
                scan_line,
                origin_marker,
                colorbar_group,
                tick_labels,
                colorbar_units,
                title,
            )

        reflectivity_panel = make_ppi_panel(
            reflectivity_metadata,
            reflectivity_rgba,
            "Reflectivity",
            "dBZ",
            (-20, 0, 20, 40, 60),
            "NWSRef",
        )

        velocity_limit = max(
            abs(velocity_metadata["vmin"]), abs(velocity_metadata["vmax"])
        )
        velocity_panel = make_ppi_panel(
            velocity_metadata,
            velocity_rgba,
            "Velocity",
            "m/s",
            (
                -velocity_limit,
                -velocity_limit / 2,
                0.0,
                velocity_limit / 2,
                velocity_limit,
            ),
            "NWSVel",
        )

        panels = Group(reflectivity_panel, velocity_panel).arrange(
            RIGHT, buff=LARGE_BUFF * 1.1, aligned_edge=UP
        )
        panels.move_to(ORIGIN).shift(DOWN * 0.25)

        header = Text("Single-Pol Products", font=FONT).scale(0.72)
        subtitle = Text(
            (
                f"{reflectivity_metadata['station']} | DBZ sw {reflectivity_metadata['sweep']}"
                f" | Vel sw {velocity_metadata['sweep']}"
                f" | elev {reflectivity_metadata['elevation_deg']} deg"
            ),
            font=FONT,
            color=label_color,
        ).scale(0.22)
        header_group = Group(header, subtitle).arrange(DOWN, SMALL_BUFF)
        header_group.next_to(panels, UP, LARGE_BUFF * 0.8)

        self.play(
            LaggedStart(
                FadeIn(header_group, shift=DOWN * 0.2),
                FadeIn(reflectivity_panel, scale=0.97),
                FadeIn(velocity_panel, scale=0.97),
                lag_ratio=0.18,
            ),
            run_time=2.4,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            scan_visibility @ 1,
            scan_progress @ 360,
            run_time=6,
            rate_func=linear,
        )

        self.wait(2)


class Zdr3D(ThreeDScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[-2, 2, 0.5],
            x_length=20,
            y_length=10,
            z_length=20,
            tips=False,
            x_axis_config=dict(
                # stroke_color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            y_axis_config=dict(
                # stroke_color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            z_axis_config=dict(
                # stroke_color=ORANGE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
        )

        target_dist = 40
        u0 = VT(-2)
        u1 = VT(2)
        u0_rx = VT(target_dist)
        u1_rx = VT(target_dist + 4)
        hpol_tx_opacity = VT(1)
        hpol_rx_opacity = VT(0)
        vpol_tx_opacity = VT(0)
        vpol_rx_opacity = VT(0)
        hpol_tx_amp = VT(1)
        vpol_tx_amp = VT(0.6)
        hpol_rx_amp = VT(1)
        vpol_rx_amp = VT(0.6)
        hpol_rotation = VT(0)
        hpol_tx_highlight_opacity = VT(0)
        hpol_tx_highlight_x0 = VT(0)
        hpol_tx_highlight_x1 = VT(0)
        vpol_tx_highlight_x0 = VT(0)
        vpol_tx_highlight_x1 = VT(0)
        vpol_tx_width = VT(2)
        vpol_tx_rotation = VT(0)
        hpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=HPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~hpol_rotation, RIGHT)
        ).set_shade_in_3d(True)
        hpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=YELLOW,
                t_range=(~hpol_tx_highlight_x0, ~hpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=YELLOW,
                t_range=(~vpol_tx_highlight_x0, ~vpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)

        # self.add(
        #     hpol_tx,
        #     vpol_tx,
        #     hpol_tx_highlight,
        #     vpol_tx_highlight,
        # )

        self.set_camera_orientation(
            zoom=0.4,
            theta=-90 * DEGREES,
            phi=0 * DEGREES,
            gamma=0,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(axes),
                Create(hpol_tx),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        vpol_amp_arrow_left = Arrow3D(
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, 0]),
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, ~vpol_tx_amp]),
            thickness=0.1,
            height=0.75,
            base_radius=0.3,
        )
        vpol_amp_arrow_right = Arrow3D(
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, 0]),
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, -~vpol_tx_amp]),
            thickness=0.1,
            height=0.75,
            base_radius=0.3,
        )

        hpol_amp_arrow_left = Arrow3D(
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, 0]),
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, ~hpol_tx_amp, 0]),
            thickness=0.1,
            height=0.75,
            base_radius=0.3,
        )
        hpol_amp_arrow_right = Arrow3D(
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, 0, 0]),
            axes.copy().shift(IN * 5 + UP * 5).c2p([0, -~hpol_tx_amp, 0]),
            thickness=0.1,
            height=0.75,
            base_radius=0.3,
        )

        zh = (
            MathTex(r"\lvert Z_h \rvert")
            .rotate(3 * PI / 2)
            .rotate(PI / 2, DOWN)
            .scale(2)
            .move_to(axes.copy().shift(IN * 5 + UP * 5).c2p(-0.3, -0.8, 0))
        )
        zv = (
            MathTex(r"\lvert Z_v \rvert")
            .rotate(3 * PI / 2)
            .rotate(PI / 2, DOWN)
            .scale(2)
            .move_to(axes.copy().shift(IN * 5 + UP * 5).c2p(0, 0.3, 0.5))
        )

        hpol_amp_arrow_left.save_state()
        hpol_amp_arrow_right.save_state()
        hpol_amp_arrow_left.stretch(0.001, 0).stretch(0.001, 1).move_to(
            axes.copy().shift(IN * 5 + UP * 5).c2p(0, 0, 0)
        )
        hpol_amp_arrow_right.stretch(0.001, 0).stretch(0.001, 1).move_to(
            axes.copy().shift(IN * 5 + UP * 5).c2p(0, 0, 0)
        )
        self.move_camera(
            zoom=0.6,
            theta=-160 * DEGREES,
            phi=80 * DEGREES,
            gamma=0,
            run_time=3,
            added_anims=[
                LaggedStart(
                    AnimationGroup(
                        u0 + 2,
                        u1 + 2,
                        axes.animate.shift(IN * 5 + UP * 5),
                    ),
                    AnimationGroup(
                        Restore(hpol_amp_arrow_left),
                        Restore(hpol_amp_arrow_right),
                        FadeIn(zh),
                    ),
                    lag_ratio=2,
                )
            ],
        )

        self.wait(0.5)

        vpol_amp_arrow_left.save_state()
        vpol_amp_arrow_right.save_state()
        vpol_amp_arrow_left.stretch(0.001, 0).stretch(0.001, 1).move_to(
            axes.c2p(0, 0, 0)
        )
        vpol_amp_arrow_right.stretch(0.001, 0).stretch(0.001, 1).move_to(
            axes.c2p(0, 0, 0)
        )

        vpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=VPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * ~vpol_tx_width,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~vpol_tx_rotation, RIGHT)
        ).set_shade_in_3d(True)

        self.move_camera(
            zoom=0.5,
            # theta=-170 * DEGREES,
            phi=75 * DEGREES,
            # gamma=0,
            run_time=3,
            added_anims=[
                LaggedStart(
                    Create(vpol_tx),
                    AnimationGroup(
                        Restore(vpol_amp_arrow_left),
                        Restore(vpol_amp_arrow_right),
                        FadeIn(zv),
                    ),
                    lag_ratio=2,
                )
            ],
        )

        self.wait(0.5)
        zh_static = MathTex(r"\lvert Z_h \rvert").set_opacity(0)
        zv_static = MathTex(r"\lvert Z_v \rvert").set_opacity(0)
        self.add_fixed_in_frame_mobjects(zh_static, zv_static)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            Group(
                axes,
                hpol_amp_arrow_left,
                hpol_amp_arrow_right,
                vpol_amp_arrow_left,
                vpol_amp_arrow_right,
            ).animate.shift(UP * 80 + IN * 26),
            zh.animate.shift(DOWN * 20 + OUT * 13),
            zv.animate.shift(DOWN * 18 + OUT * 13),
            rate_func=rate_functions.ease_in_sine,
        )

        self.wait(2)


class Zdr3dP2(MovingCameraScene):
    def construct(self):
        zdr = MathTex(
            r"\frac{\lvert Z_h \rvert}{\lvert Z_v \rvert}"
        ).scale_to_fit_height(fh(self, 0.4))

        self.play(
            LaggedStart(
                self.camera.frame.shift(DL * fw(self)).animate.shift(UR * fw(self)),
                ReplacementTransform(
                    zdr[0][:4].copy().shift(DOWN + LEFT * fw(self, 0.5)),
                    zdr[0][:4],
                    rate_func=rate_functions.ease_out_sine,
                ),
                ReplacementTransform(
                    zdr[0][5:9].copy().shift(DOWN * 3 + LEFT * fw(self, 0.5)),
                    zdr[0][5:9],
                    rate_func=rate_functions.ease_out_sine,
                ),
                Create(zdr[0][4]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        zdr_log = MathTex(
            r"10 \log{\left(\frac{\lvert Z_h \rvert}{\lvert Z_v \rvert}\right)}"
        ).scale_to_fit_height(fh(self, 0.4))

        self.play(
            LaggedStart(
                ReplacementTransform(zdr[0], zdr_log[0][6:-1]),
                LaggedStart(
                    *[FadeIn(m) for m in [*zdr_log[0][:6], zdr_log[0][-1]]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        zdr_log_zero = (
            MathTex(
                r"10 \log{\left(\frac{\lvert Z_h \rvert}{\lvert Z_v \rvert}\right)} \approx 0"
            )
            .scale_to_fit_height(fh(self, 0.4))
            .move_to(zdr_log, LEFT)
        )

        circle = Circle(
            radius=fh(self, 0.25), color=BLUE, stroke_width=DEFAULT_STROKE_WIDTH * 2
        ).next_to(zdr_log_zero, RIGHT, LARGE_BUFF * 1.5)
        circle_bez_d = CubicBezier(
            zdr_log_zero.get_right() + [0.1, 0, 0],
            zdr_log_zero.get_right() + [1, 0, 0],
            circle.get_corner(DL) + [-1, 0, 0],
            circle.get_corner(DL),
        )
        circle_bez_u = CubicBezier(
            zdr_log_zero.get_right() + [0.1, 0, 0],
            zdr_log_zero.get_right() + [1, 0, 0],
            circle.get_corner(UL) + [-1, 0, 0],
            circle.get_corner(UL),
        )

        self.play(
            LaggedStart(
                ReplacementTransform(zdr_log[0], zdr_log_zero[0][:-2]),
                self.camera.frame.animate.scale(1.4).move_to(
                    Group(zdr_log_zero, circle)
                ),
                FadeIn(zdr_log_zero[0][-2]),
                FadeIn(zdr_log_zero[0][-1]),
                AnimationGroup(Create(circle_bez_u), Create(circle_bez_d)),
                Create(circle),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        self.play(Indicate(zdr_log_zero[0][-1]))

        self.wait(0.5)

        self.play(Circumscribe(circle))

        self.wait(2)


class ZdrP2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        resolution = (
            VideoMobject("./static/Resolution.mov", loop=False, speed=1)
            .scale_to_fit_width(fw(self, 1))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self))
        )
        self.add(resolution)

        self.play(self.camera.frame.animate.shift(RIGHT * fw(self)))

        self.wait(14)

        resolution.pause()

        self.wait(0.5)

        resolution.play()

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.7).shift(RIGHT * 0.3 + UP * 1.5))

        resolution.pause()

        self.wait(0.5)

        line_l = (
            Line(ORIGIN, UP / 4)
            .rotate(PI / 6)
            .move_to(self.camera.frame)
            .shift(UP * 0.15 + LEFT * 0.7)
        )
        line_r = line_l.copy().shift([2 * np.cos(PI / 6), 2 * np.sin(PI / 6), 0])
        line_m = Line(line_l.get_midpoint(), line_r.get_midpoint())

        rres_label = (
            MathTex(r"\Delta R")
            .rotate(PI / 6)
            .move_to(line_m.get_midpoint())
            .shift([0.25 * -np.cos(PI / 6), 0.25 * np.sin(PI / 6), 0])
        )

        self.play(
            LaggedStart(
                Create(line_l),
                Create(line_m),
                Create(line_r),
                LaggedStart(*[FadeIn(m) for m in rres_label[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.restore(),
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in rres_label[0]], lag_ratio=0.1),
                Uncreate(line_r),
                Uncreate(line_m),
                Uncreate(line_l),
                lag_ratio=0.3,
                run_time=1,
            ),
        )

        self.wait(0.5)

        resolution.play()

        self.wait(8)

        resolution.pause()

        self.wait(0.5)

        vol = (
            Circle(color=RED).move_to(self.camera.frame).shift(LEFT * 1.9 + DOWN * 0.25)
        )

        self.play(Create(vol))

        self.wait(0.5)

        scan_progress = VT(135)
        beam_width_deg = 1

        s3_key = "2024/05/07/KTLX/KTLX20240507_023811_V06"
        sweep = 6
        max_range_km = 150.0
        zdr_res = 1200
        zdr = _get_nexrad_ppi_data_cached(
            s3_key=s3_key,
            sweep=sweep,
            max_range_km=max_range_km,
            resolution=zdr_res,
            field_name="differential_reflectivity",
        )

        xmin = 70
        xmax = 92
        x_des = 75.3

        newcam = self.camera.frame.get_center() + UP * fh(self, 1.5)

        x_ticks = np.arange(xmin, xmax, 1)

        def get_nexrad_ax():
            ax = Axes(
                x_range=[xmin, xmax, 1],
                y_range=[-5, 5, 2.5],
                x_length=fw(self, 0.7) * ((xmax - xmin) / 4),
                y_length=fh(self, 0.7),
                x_axis_config=dict(
                    numbers_with_elongated_ticks=x_ticks,
                    include_numbers=True,
                    numbers_to_include=x_ticks,
                    font_size=DEFAULT_FONT_SIZE * 0.5,
                    label_constructor=lambda x: Text(x, font=FONT),
                    line_to_number_buff=MED_SMALL_BUFF,
                    decimal_number_config=dict(num_decimal_places=0),
                    longer_tick_multiple=2,
                ),
                tips=False,
            )
            ax.shift(newcam - ax.c2p(x_des, 0))
            return ax

        plot_opacity = VT(1)

        def get_scan_z_over_r():
            max_range = zdr["max_range_km"]
            r = np.linspace(0, max_range, 1200)
            x = r * np.sin(np.deg2rad(~scan_progress - beam_width_deg / 2))
            y = r * np.cos(np.deg2rad(~scan_progress - beam_width_deg / 2))
            x_idx = np.abs(zdr["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
            y_idx = np.abs(zdr["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
            z = zdr["field_data"][y_idx, x_idx]
            z = np.nan_to_num(z, nan=0)
            order = np.argsort(r)
            r = r[order]
            z = z[order]

            z_func_scan = interp1d(r, z, fill_value="extrapolate")

            nexrad_scan_plot = nexrad_ax.plot(
                z_func_scan,
                x_range=[xmin, xmax, 1 / 200],
                color=HPOL_RX_COLOR,
                stroke_opacity=~plot_opacity,
            )
            return nexrad_scan_plot

        dx = 0.15

        def get_samples():
            zdr_plot = get_scan_z_over_r()
            rects = nexrad_ax.get_riemann_rectangles(
                zdr_plot,
                x_range=[xmin, xmax],
                dx=dx,
                input_sample_type="center",
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
                fill_opacity=0.3,
                color=HPOL_RX_COLOR,
                show_signed_area=True,
                # bounded_graph=nexrad_ax.plot(lambda _: 0, x_range=[0, 150, 1 / 200]),
            )
            return rects

        nexrad_ax = get_nexrad_ax().set_z_index(0)
        zdr_plot = always_redraw(get_scan_z_over_r)
        samples = get_samples().set_z_index(1)
        lblock = (
            Rectangle(
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
                stroke_opacity=0,
                color=YELLOW,
                width=fw(self, 0.5),
                height=fh(self),
            )
            .next_to(nexrad_ax.c2p(x_des - 2, 0), LEFT, MED_SMALL_BUFF)
            .set_z_index(2)
        )
        rblock = (
            Rectangle(
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
                stroke_opacity=0,
                color=YELLOW,
                width=fw(self, 0.5),
                height=fh(self),
            )
            .next_to(nexrad_ax.c2p(x_des + 2, 0), RIGHT, MED_SMALL_BUFF)
            .set_z_index(2)
        )
        xlabel = (
            Text("Range (km)\n->", font=FONT)
            .next_to(nexrad_ax.c2p(x_des, 0), DOWN, LARGE_BUFF)
            .shift(RIGHT * 2.6)
        )
        ylabel = (
            Text("ZDR", font=FONT)
            .rotate(PI / 2)
            .next_to(nexrad_ax.c2p(x_des - 2, 0), LEFT, MED_LARGE_BUFF)
        )
        self.add(nexrad_ax, samples, zdr_plot, lblock, rblock, xlabel, ylabel)

        vol_bez = CubicBezier(
            vol.get_top() + [0, 0.1, 0],
            vol.get_top() + [0, 3, 0],
            nexrad_ax.c2p(x_des, 0) + [0, -3, 0],
            nexrad_ax.c2p(x_des, 0) + [0, -0.2, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(vol_bez),
                    self.camera.frame.animate.move_to(nexrad_ax.c2p(x_des, 0)),
                ),
                samples[math.floor((x_des - xmin) / dx)].animate.set_fill(
                    color=YELLOW, opacity=0.6
                ),
                lag_ratio=0.5,
            )
        )
        # self.remove(resolution)

        self.wait(0.5)

        print("shift by", nexrad_ax.c2p(88.198, 0) - nexrad_ax.c2p(x_des, 0))

        self.play(
            Group(nexrad_ax, samples).animate.shift(
                LEFT * (nexrad_ax.c2p(88.198, 0) - nexrad_ax.c2p(x_des, 0))
            ),
            run_time=3,
        )

        self.wait(0.5)

        cloud_l = (
            SVGMobject("../props/static/cloud.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_width(fw(self, 0.3))
        )
        cloud_r = (
            SVGMobject("../props/static/cloud.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_width(fw(self, 0.3))
        )
        cloud_arrow = Arrow(LEFT, RIGHT)
        clouds = (
            Group(cloud_l, cloud_arrow, cloud_r)
            .arrange(RIGHT, SMALL_BUFF)
            .move_to(self.camera.frame.get_top())
            .shift(UP * 0.5)
        )

        zero_bez_l = CubicBezier(
            vol_bez.get_end() + [0, 0.6, 0],
            vol_bez.get_end() + [0, 2.5, 0],
            clouds.get_corner(DL) + [0, -1, 0],
            clouds.get_corner(DL) + [0, -0.1, 0],
        )
        zero_bez_r_1 = CubicBezier(
            vol_bez.get_end() + [0, 0.6, 0],
            vol_bez.get_end() + [0, 2.5, 0],
            clouds[0].get_corner(DR) + [0, -1, 0],
            clouds[0].get_corner(DR) + [0, -0.1, 0],
        )
        zero_bez_r = CubicBezier(
            vol_bez.get_end() + [0, 0.6, 0],
            vol_bez.get_end() + [0, 2.5, 0],
            clouds.get_corner(DR) + [0, -1, 0],
            clouds.get_corner(DR) + [0, -0.1, 0],
        )

        # self.add(clouds)
        self.play(
            LaggedStart(
                FadeOut(xlabel, ylabel),
                AnimationGroup(
                    self.camera.frame.animate.scale(1.2).shift(UP * 2),
                    plot_opacity @ 0.2,
                    nexrad_ax.animate.set_stroke(opacity=0.2),
                    Uncreate(vol_bez),
                    *[m.animate.set_opacity(0.2) for m in nexrad_ax.x_axis.numbers],
                    *[
                        m.animate.set_stroke(opacity=0.2).set_fill(opacity=0.2)
                        for m in samples
                    ],
                ),
                AnimationGroup(Create(zero_bez_l), Create(zero_bez_r_1)),
                FadeIn(cloud_l),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        np.random.seed(0)
        cloud_center = cloud_l.get_center() + DOWN * cloud_l.height * 0.03
        drop_radius = cloud_l.width * 0.01
        drop_rows = [
            (-0.20, 4, 0.10),
            (-0.12, 7, 0.20),
            (-0.05, 10, 0.29),
            (0.03, 12, 0.35),
            (0.11, 11, 0.31),
            (0.19, 8, 0.22),
            (0.26, 5, 0.12),
        ]

        sphere_drops = []
        for y_frac, count, half_width_frac in drop_rows:
            row_offset = np.random.normal(0, cloud_l.width * 0.008)
            x_positions = np.linspace(-half_width_frac, half_width_frac, count)
            for x_frac in x_positions:
                x_frac = np.clip(
                    x_frac + np.random.normal(0, 0.012),
                    -half_width_frac,
                    half_width_frac,
                )
                sphere_drops.append(
                    Circle(
                        radius=drop_radius * np.random.uniform(0.88, 1.18),
                        color=BLUE,
                        fill_color=BLUE,
                        fill_opacity=1,
                        stroke_width=0,
                    ).move_to(
                        cloud_center
                        + RIGHT * (x_frac * cloud_l.width + row_offset)
                        + UP
                        * (
                            y_frac * cloud_l.height
                            + np.random.normal(0, cloud_l.height * 0.008)
                        )
                    )
                )

        shuffle(sphere_drops)

        sphere_drops = Group(*sphere_drops)
        sphere_drops.shift(DOWN * cloud_l.height * 0.1)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in sphere_drops],
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowArrow(cloud_arrow),
                ReplacementTransform(zero_bez_r_1, zero_bez_r),
                FadeIn(cloud_r),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        np.random.seed(1)
        cloud_center = cloud_r.get_center() + DOWN * cloud_r.height * 0.03
        drop_radius = cloud_r.width * 0.01
        # drop_rows = [
        #     (-0.20, 4, 0.10),
        #     (-0.12, 7, 0.20),
        #     (-0.05, 10, 0.29),
        #     (0.03, 12, 0.35),
        #     (0.11, 11, 0.31),
        #     (0.19, 8, 0.22),
        #     (0.26, 5, 0.12),
        # ]

        sphere_drops_r = []
        for y_frac, count, half_width_frac in drop_rows:
            row_offset = np.random.normal(0, cloud_r.width * 0.008)
            x_positions = np.linspace(-half_width_frac, half_width_frac, count)
            for x_frac in x_positions:
                x_frac = np.clip(
                    x_frac + np.random.normal(0, 0.012),
                    -half_width_frac,
                    half_width_frac,
                )
                sphere_drops_r.append(
                    Circle(
                        radius=drop_radius * np.random.uniform(0.88, 1.18),
                        color=BLUE,
                        fill_color=BLUE,
                        fill_opacity=1,
                        stroke_width=0,
                    ).move_to(
                        cloud_center
                        + RIGHT * (x_frac * cloud_r.width + row_offset)
                        + UP
                        * (
                            y_frac * cloud_r.height
                            + np.random.normal(0, cloud_r.height * 0.008)
                        )
                    )
                )

        shuffle(sphere_drops_r)

        for sd in sphere_drops_r[: len(sphere_drops_r) // 3]:
            sd.stretch(1.5, 0)

        for sd in sphere_drops_r[-len(sphere_drops_r) // 3 :]:
            sd.stretch(1.5, 1)

        sphere_drops_r = Group(*sphere_drops_r)
        sphere_drops_r.shift(DOWN * cloud_l.height * 0.1)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in sphere_drops_r],
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)

        sphere_l = Circle(
            color=BLUE,
            fill_color=BLUE,
            fill_opacity=1,
            radius=sphere_drops[0].radius * 5,
        ).move_to(cloud_l.get_center())
        sphere_r = Circle(
            color=BLUE,
            fill_color=BLUE,
            fill_opacity=1,
            radius=sphere_drops[0].radius * 5,
        ).move_to(cloud_r.get_center())

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                LaggedStart(
                    *[m.animate.move_to(sphere_l).set_opacity(0) for m in sphere_drops],
                    lag_ratio=0.05,
                ),
                sphere_l.scale(1 / 100).animate(run_time=3).scale(100),
                LaggedStart(
                    *[
                        m.animate.move_to(sphere_r).set_opacity(0)
                        for m in sphere_drops_r
                    ],
                    lag_ratio=0.05,
                ),
                sphere_r.scale(1 / 100).animate(run_time=3).scale(100),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        new_sample = (
            samples[int((88.198 - xmin) / dx) - 1]
            .copy()
            .set_stroke(opacity=1)
            .set_fill(opacity=1)
            .stretch(1.3, 1)
            .move_to(samples[int((88.198 - xmin) / dx)], DOWN)
        )

        zero_bez_l_new = CubicBezier(
            new_sample.get_top() + [0, 0.1, 0],
            new_sample.get_top() + [0, 1, 0],
            clouds.get_corner(DL) + [0, -1, 0],
            clouds.get_corner(DL) + [0, -0.1, 0],
        )
        zero_bez_r_new = CubicBezier(
            new_sample.get_top() + [0, 0.1, 0],
            new_sample.get_top() + [0, 1, 0],
            clouds.get_corner(DR) + [0, -1, 0],
            clouds.get_corner(DR) + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(zero_bez_l, zero_bez_l_new),
                    Transform(zero_bez_r, zero_bez_r_new),
                ),
                AnimationGroup(
                    Transform(samples[int((88.198 - xmin) / dx)], new_sample),
                    sphere_l.animate.stretch(2, 0),
                    sphere_r.animate.stretch(2, 0),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        new_neg_sample = (
            samples[int((88.198 - xmin) / dx) - 1]
            .copy()
            .set_stroke(opacity=1)
            .set_fill(opacity=1, color=RED)
            .stretch(1, 1)
            .next_to(samples[int((88.198 - xmin) / dx)], DOWN, 0)
        )

        zero_bez_l_neg_new = CubicBezier(
            new_neg_sample.get_top() + [0, 0.1, 0],
            new_neg_sample.get_top() + [0, 1, 0],
            clouds.get_corner(DL) + [0, -1, 0],
            clouds.get_corner(DL) + [0, -0.1, 0],
        )
        zero_bez_r_neg_new = CubicBezier(
            new_neg_sample.get_top() + [0, 0.1, 0],
            new_neg_sample.get_top() + [0, 1, 0],
            clouds.get_corner(DR) + [0, -1, 0],
            clouds.get_corner(DR) + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(samples[int((88.198 - xmin) / dx)], new_neg_sample),
                    sphere_l.animate.stretch(1 / 2, 0).stretch(2, 1),
                    sphere_r.animate.stretch(1 / 2, 0).stretch(2, 1),
                ),
                AnimationGroup(
                    Transform(zero_bez_l, zero_bez_l_neg_new),
                    Transform(zero_bez_r, zero_bez_r_neg_new),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        who_cares = Text("who cares?", font=FONT).next_to(clouds, UP)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    Group(clouds, who_cares).width * 1.3
                ).move_to(Group(clouds, who_cares)),
                Write(who_cares),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class RhoHV(MovingCameraScene):
    def construct(self): ...
