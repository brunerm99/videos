# dual_pol.py
import os
import sys
from random import shuffle
from turtle import width

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

SKIP_ANIMATIONS_OVERRIDE = True

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

        self.next_section(skip_animations=skip_animations(False))

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

        self.play(
            LaggedStart(
                doppler_label.animate.set_opacity(0.2),
                doppler_box.animate.set_opacity(0.2),
                doppler_ax.animate.set_opacity(0.2),
                doppler_xlabel.animate.set_opacity(0.2),
                doppler_ylabel.animate.set_opacity(0.2),
                doppler_plot.animate.set_stroke_opacity(0.2),
                self.camera.frame.animate.scale_to_fit_height(
                    reflectivity_group.height * 1.4
                ).move_to(reflectivity_group),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        rain_qmark = (
            Text("rain?", font=FONT)
            .scale(0.3)
            .next_to(reflectivity_ax.i2gp(0.3, reflectivity_plot), UP, SMALL_BUFF)
            .shift(UP * 0.5 + LEFT)
        )
        hail_qmark = (
            Text("hail?", font=FONT)
            .scale(0.3)
            .next_to(rain_qmark, RIGHT, SMALL_BUFF)
            .shift(DOWN * 0.2)
        )
        bird_qmark = (
            Text("bird?", font=FONT)
            .scale(0.3)
            .next_to(hail_qmark, RIGHT, SMALL_BUFF)
            .shift(DOWN * 0.1 + RIGHT * 0.1)
        )

        self.play(FadeIn(rain_qmark))

        self.wait(0.5)

        self.play(FadeIn(hail_qmark))

        self.wait(0.5)

        self.play(FadeIn(bird_qmark))

        self.wait(0.5)

        dual_pol = (
            Text("Dual-Pol", font=FONT)
            .scale_to_fit_width(fw(self, 0.5))
            .move_to(self.camera.frame)
            .shift(UP * fh(self, 2))
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * fh(self, 2)),
                Write(dual_pol),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(dual_pol))

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


class Idea2D(MovingCameraScene):
    def construct(self): ...


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
        self.add(self.build_diagram())

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
            circle.get_corner(UL) + DR * radius * 0.12,
            circle.get_corner(DR) + UL * radius * 0.12,
            color=BLACK,
            stroke_width=self.stroke,
        )
        diag_2 = Line(
            circle.get_corner(UR) + DL * radius * 0.12,
            circle.get_corner(DL) + UR * radius * 0.12,
            color=BLACK,
            stroke_width=self.stroke,
        )
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
        start = center + LEFT * radius * 0.36 + UP * radius * 0.08
        end = center + LEFT * radius * 0.22 + DOWN * radius * 0.24
        inner = CubicBezier(
            start,
            start + DOWN * radius * 0.22,
            end + LEFT * radius * 0.18 + UP * radius * 0.02,
            end,
            color=BLACK,
            stroke_width=self.stroke * 1.15,
        )
        tip = Triangle(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1,
        ).scale(radius * 0.19)
        tip.rotate(-PI * 0.77)
        tip.shift(end - tip.get_top())
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
        diagram.add(self.polywire((229, 83), (248, 83), (248, 98), end_arrow=True))

        diagram.add(self.polywire((112, 140), (245, 140), (245, 126), end_arrow=True))
        diagram.add(self.mixer(245, 112))
        diagram.add(self.polywire((156, 140), (156, 168), (176, 168), end_arrow=True))
        diagram.add(self.text("To Receivers  (? Mixers)", 222, 164, font_size=14))

        diagram.add(self.wire((259, 112), (272, 112), arrow=True))
        diagram.add(self.filter_box(272, 98, 303, 128))
        diagram.add(self.wire((303, 112), (319, 112)))
        diagram.add(self.amp(319, 97, 350, 128))

        diagram.add(self.box(296, 48, 355, 81))
        diagram.add(
            self.para(
                "I/Q",
                "Modulator",
                x=325,
                y=63,
                font_size=16,
                max_width_px=40,
                max_height_px=22,
            )
        )
        diagram.add(
            self.polywire(
                (353, 113), (353, 95), (287, 95), (287, 56), (296, 56), end_arrow=True
            )
        )

        diagram.add(self.arrow_circle(385, 62))
        diagram.add(self.arrow_circle(385, 112))
        diagram.add(self.selector_box(411, 49, 439, 77))
        diagram.add(self.selector_box(411, 97, 439, 125))
        diagram.add(self.wire((355, 62), (371, 62)))
        diagram.add(self.wire((399, 62), (411, 62)))
        diagram.add(self.wire((439, 62), (469, 62), arrow=True))
        diagram.add(self.wire((350, 112), (371, 112)))
        diagram.add(self.wire((399, 112), (411, 112)))
        diagram.add(self.wire((439, 112), (469, 112), arrow=True))
        diagram.add(self.text("RF to Transmitters", 430, 90, font_size=14))
        diagram.add(self.text("V", 479, 64, font_size=16))
        diagram.add(self.text("H", 479, 112, font_size=16))

        diagram.add(
            self.text("Transmitter/Receiver: (H only shown)", 108, 195, font_size=23)
        )
        diagram.add(
            self.polywire((440, 125), (440, 201), (48, 201), (48, 236), (66, 236))
        )
        diagram.add(self.amp(66, 214, 95, 240))
        diagram.add(
            self.para(
                "IPA",
                "(Staclb)",
                x=82,
                y=214,
                font_size=15,
                max_width_px=26,
                max_height_px=18,
            )
        )
        diagram.add(self.wire((95, 227), (103, 227)))
        diagram.add(self.arrow_circle(117, 227))
        diagram.add(self.wire((131, 227), (147, 227)))
        diagram.add(self.amp(147, 214, 178, 240))
        diagram.add(
            self.para(
                "Power Amp",
                "(VA-87B)",
                x=175,
                y=213,
                font_size=15,
                max_width_px=56,
                max_height_px=20,
            )
        )
        diagram.add(self.wire((178, 227), (232, 227), arrow=True))

        diagram.add(self.polywire((206, 227), (206, 245), (187, 245), (187, 293)))
        diagram.add(self.text("50 dB", 189, 257, font_size=15))
        diagram.add(self.text("Coupler", 190, 274, font_size=15))

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
        diagram.add(self.text("To Antenna H", 258, 274, font_size=15))
        diagram.add(self.text("Port", 232, 292, font_size=15))

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
            self.polywire((421, 227), (437, 227), (437, 293), (15, 293), (15, 469))
        )

        diagram.add(self.polywire((48, 323), (48, 356), (63, 356), end_arrow=True))
        diagram.add(self.text("Transfer Sw.", 97, 340, font_size=16))
        diagram.add(self.switch_box(63, 356, 96, 387))
        diagram.add(self.wire((37, 387), (63, 387), arrow=True))
        diagram.add(self.para("From V", "LNA", x=38, y=392, font_size=16))
        diagram.add(self.wire((96, 387), (135, 387), arrow=True))
        diagram.add(self.para("To V", "Receiver", x=147, y=400, font_size=16))

        diagram.add(self.wire((96, 356), (168, 356)))
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
        diagram.add(self.polywire((384, 453), (398, 453), (398, 500), (177, 500)))

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
        diagram.add(self.text("10 MHZ IF", 134, 580, font_size=16))

        diagram.add(self.box(176, 500, 266, 548))
        diagram.add(
            self.text(
                "10 MHZ IF",
                220,
                523,
                font_size=16,
                max_width_px=62,
                max_height_px=18,
            )
        )
        diagram.add(self.wire((266, 548), (273, 548), arrow=True))

        diagram.add(self.box(273, 516, 370, 602))
        diagram.add(
            self.text(
                "DRX Processor",
                321,
                530,
                font_size=16,
                max_width_px=74,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "Low Chan.",
                321,
                558,
                font_size=16,
                max_width_px=68,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "High Chan.",
                323,
                586,
                font_size=16,
                max_width_px=70,
                max_height_px=13,
            )
        )
        diagram.add(
            self.text(
                "Txmit Chan.",
                325,
                594,
                font_size=16,
                max_width_px=72,
                max_height_px=13,
            )
        )

        diagram.add(self.text("25 dB", 194, 558, font_size=15))
        diagram.add(self.text("Coupler", 197, 574, font_size=15))
        diagram.add(self.polywire((222, 561), (222, 574), (273, 574), end_arrow=True))

        return diagram


class ChillBD(MovingCameraScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.add(ChillBlockDiagram())


class DropShape(Scene):
    def construct(self):
        Deq_min = 1.5
        Deq_max = 6
        Deq_target = 5.25
        Deq = VT(Deq_min)

        def c1(Deq):
            return (1 / np.pi) * (0.02914 * Deq**2 + 0.9263 * Deq + 0.07791)

        def c2(Deq):
            return -0.01938 * Deq**2 + 0.4698 * Deq + 0.09538

        def c3(Deq):
            return -0.06123 * Deq**3 + 1.3880 * Deq**2 - 10.41 * Deq + 28.34

        def c4(Deq):
            if Deq > 4:
                return -0.01352 * Deq**3 + 0.2014 * Deq**2 - 0.8964 * Deq + 1.226
            elif 1.5 <= Deq <= 4:
                return 0
            else:
                return np.nan

        def x_function(Deq, y):
            c1_val = c1(Deq)
            c2_val = c2(Deq)
            c3_val = c3(Deq)
            c4_val = c4(Deq)

            if np.abs(y / c2_val) > 1:
                return np.nan

            term1 = np.sqrt(1 - (y / c2_val) ** 2)
            term2 = np.arccos(y / (c3_val * c2_val))
            term3 = c4_val * (y / c2_val) ** 2 + 1

            return c1_val * term1 * term2 * term3

        y_values = np.linspace(-2.5, 2.5, 10_000)

        x_values_positive = [x_function(~Deq, y) for y in y_values]
        x_values_negative = [-x_function(~Deq, y) for y in y_values]

        xvn = np.array(x_values_negative)
        ind = np.where(~np.isnan(xvn))
        xvn = xvn[ind]
        yvn = np.array(y_values)[ind]

        xvp = np.array(x_values_positive)
        ind = np.where(~np.isnan(xvp))
        xvp = xvp[ind]
        yvp = np.array(y_values)[ind]

        f_vn = interp1d(xvn, yvn, fill_value="extrapolate")
        f_vp = interp1d(xvp, yvp, fill_value="extrapolate")

        rain_ax = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={
                "stroke_color": LIGHT_GRAY,
                "stroke_opacity": 0.6,
            },
        )

        hail_ax = rain_ax.copy()

        rain_label = Tex("Rain Drop").next_to(rain_ax, UP)
        hail_label = Tex("Hail").next_to(hail_ax, UP)

        rain_group = VGroup(rain_label, rain_ax)

        lplot = rain_ax.plot_line_graph(
            xvn,
            yvn,
            line_color=PRECIP_COLOR,
            add_vertex_dots=False,
        )
        rplot = rain_ax.plot_line_graph(
            xvp,
            yvp,
            line_color=PRECIP_COLOR,
            add_vertex_dots=False,
        )

        def get_plot_updater(scalar=1):
            def updater(m: Mobject):
                x_values = np.array([scalar * x_function(~Deq, y) for y in y_values])
                ind = np.where(~np.isnan(x_values))
                xv = x_values[ind]
                yv = np.array(y_values)[ind]
                m.become(
                    rain_ax.plot_line_graph(
                        xv, yv, add_vertex_dots=False, line_color=PRECIP_COLOR
                    )
                )

            return updater

        lplot.add_updater(get_plot_updater(scalar=-1))
        rplot.add_updater(get_plot_updater(scalar=1))

        # Deq_nl = NumberLine(
        #     x_range=[Deq_min, Deq_max, 0.5],
        #     include_numbers=True,
        #     include_tip=False,
        #     length=config["frame_width"] * 0.6,
        # ).to_edge(DOWN)

        # def clamp_drop_y(y):
        #     drop_half_height = c2(~Deq)
        #     return np.clip(y, -0.92 * drop_half_height, 0.92 * drop_half_height)

        # def drop_half_width(y):
        #     return x_function(~Deq, clamp_drop_y(y))

        # def build_updraft_streamline(
        #     side,
        #     start_x_factor,
        #     lower_gap,
        #     upper_gap,
        #     top_gap,
        #     start_pad,
        #     end_pad,
        #     lower_y_factor,
        #     upper_y_factor,
        #     stroke_width,
        #     stroke_opacity,
        #     tip_scale,
        # ):
        #     drop_half_height = c2(~Deq)
        #     max_half_width = drop_half_width(0)
        #     lower_y = clamp_drop_y(lower_y_factor * drop_half_height)
        #     mid_lower_y = clamp_drop_y(0.55 * lower_y_factor * drop_half_height)
        #     shoulder_y = clamp_drop_y(-0.08 * drop_half_height)
        #     upper_y = clamp_drop_y(upper_y_factor * drop_half_height)
        #     mid_upper_y = clamp_drop_y(0.5 * (upper_y + 0.82 * drop_half_height))
        #     top_y = clamp_drop_y(0.82 * drop_half_height)
        #     start_x = 0.3 * start_x_factor * max_half_width + 0.7 * (
        #         drop_half_width(lower_y) + 0.35 * lower_gap
        #     )
        #     end_x = drop_half_width(top_y) + 0.75 * top_gap

        #     y_knots = np.array(
        #         [
        #             -drop_half_height - start_pad,
        #             lower_y,
        #             mid_lower_y,
        #             shoulder_y,
        #             upper_y,
        #             mid_upper_y,
        #             top_y,
        #             drop_half_height + end_pad,
        #         ]
        #     )
        #     x_knots = np.array(
        #         [
        #             start_x,
        #             drop_half_width(lower_y) + lower_gap,
        #             drop_half_width(mid_lower_y) + 0.7 * lower_gap + 0.3 * upper_gap,
        #             drop_half_width(shoulder_y) + upper_gap,
        #             drop_half_width(upper_y) + upper_gap,
        #             drop_half_width(mid_upper_y) + 0.45 * top_gap + 0.55 * upper_gap,
        #             drop_half_width(top_y) + top_gap,
        #             end_x,
        #         ]
        #     )
        #     x_interp = PchipInterpolator(y_knots, x_knots)

        #     curve = ParametricFunction(
        #         lambda y: rain_ax.c2p(side * float(x_interp(y)), y),
        #         t_range=[y_knots[0], y_knots[-1], (y_knots[-1] - y_knots[0]) / 120],
        #     ).set_stroke(
        #         WHITE,
        #         width=stroke_width,
        #         opacity=stroke_opacity,
        #     )
        #     if tip_scale <= 0:
        #         return curve

        #     tip_anchor = rain_ax.c2p(side * x_knots[-1], y_knots[-1])
        #     tail_anchor = rain_ax.c2p(side * x_knots[-2], y_knots[-2])
        #     tangent = normalize(tip_anchor - tail_anchor)
        #     tip = Triangle(
        #         fill_color=WHITE,
        #         fill_opacity=stroke_opacity,
        #         stroke_width=0,
        #     ).scale(tip_scale)
        #     tip.rotate(angle_of_vector(tangent) - PI / 2)
        #     tip.shift(tip_anchor - tip.get_boundary_point(tangent))
        #     return VGroup(curve, tip)

        # streamline_specs = [
        #     dict(
        #         start_x_factor=0.24,
        #         lower_gap=0.34,
        #         upper_gap=0.34,
        #         top_gap=0.44,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=2.5,
        #         stroke_opacity=0.58,
        #         tip_scale=0,
        #     ),
        #     dict(
        #         start_x_factor=0.48,
        #         lower_gap=0.56,
        #         upper_gap=0.56,
        #         top_gap=0.72,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=2.2,
        #         stroke_opacity=0.42,
        #         tip_scale=0,
        #     ),
        #     dict(
        #         start_x_factor=0.72,
        #         lower_gap=0.78,
        #         upper_gap=0.78,
        #         top_gap=1,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=1.8,
        #         stroke_opacity=0.24,
        #         tip_scale=0,
        #     ),
        # ]
        # wind_lines = always_redraw(
        #     lambda: VGroup(
        #         *(
        #             build_updraft_streamline(side=side, **spec)
        #             for spec in streamline_specs
        #             for side in (-1, 1)
        #         ),
        #     ).set_z_index(-1)
        # )

        self.add(
            # wind_lines,
            lplot,
            rplot,
        )

        # # self.play(
        # #     Create(rain_ax),
        # #     Create(lplot),
        # #     Create(rplot),
        # # )

        # b1 = always_redraw(
        #     lambda: CubicBezier(
        #         rain_ax.i2gp(0.1, rplot),
        #         rain_ax.i2gp(0.1, rplot),
        #     )
        # )
        self.add(Dot(rain_ax.c2p(0.1, f_vp(0.1))))

        self.play(Deq @ Deq_target, run_time=6)

        self.wait(2)
