# dual_pol.py
import os
import sys
from random import shuffle

import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from manim.utils.color.X11 import BROWN1
from MF_Tools import VT
from networkx import center
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import interp1d

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


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


class DualPol(ThreeDScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))
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
        u1_rx = VT(target_dist + 2)
        hpol_tx_opacity = VT(1)
        hpol_rx_opacity = VT(0)
        vpol_tx_opacity = VT(0)
        vpol_rx_opacity = VT(0)
        hpol_tx_amp = VT(1)
        hpol_rx_amp = VT(1)
        vpol_rx_amp = VT(0.2)
        hpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=BLUE,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_tx_opacity,
            )
        ).set_shade_in_3d(True)
        hpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_rx_amp * np.sin(2 * PI * u), 0),
                color=PURPLE,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_rx_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_rx_amp * np.sin(2 * PI * u)),
                color=ORANGE,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~vpol_rx_opacity,
            )
        ).set_shade_in_3d(True)

        self.add(hpol_tx, hpol_rx, vpol_rx)

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

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                AnimationGroup(u0 + target_dist, u1 + target_dist),
                hpol_tx_opacity @ 0,
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

        self.wait(2)
