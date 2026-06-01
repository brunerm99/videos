# z.py

import math
import os
import pprint
import sys
from copy import deepcopy
from functools import lru_cache
from math import sqrt
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from manim.utils.color.X11 import BROWN1
from matplotlib.pylab import diag
from MF_Tools import VT, SurroundingRectangleUnion
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
    PPIGrid,
    Resistor,
    VideoMobject,
    WeatherRadarTower,
    cubic_bezier,
    ease_in_out_elastic,
    ease_out_elastic,
    get_amp,
    get_blocks,
    get_filt_block,
    get_phase_eased_sine,
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
NEXRAD_CASE_S3_KEY = "2024/05/07/KTLX/KTLX20240507_023811_V06"
NEXRAD_VOLUME_DIR = Path(__file__).resolve().parent / "static" / "nexrad"
NEXRAD_VOLUME_PATH = NEXRAD_VOLUME_DIR / Path(NEXRAD_CASE_S3_KEY).name


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
        z_eqn = MathTex(
            r"Z = \int_{0}^{\infty} N_0 D^\mu e^{-\Lambda D} D^6 \, dD"
        ).scale_to_fit_width(fw(self, 0.9))
        # self.add(z_eqn)


class PowerVsZ(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        z = MathTex(r"Z").scale(2)
        pr = MathTex(r"P_r").scale(2)
        zep = MathTex(r"Z = P_r").scale(2)
        zsp = MathTex(r"Z \sim P_r").scale(2)

        self.play(GrowFromCenter(z))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(z[0], zep[0][0]),
                GrowFromCenter(zep[0][1]),
                zep[0][2:].shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                zep[0][1].animate.shift(UP).set_opacity(0.2),
                AnimationGroup(
                    zsp[0][1]
                    .shift(DOWN)
                    .set_opacity(0)
                    .animate.shift(UP)
                    .set_opacity(1),
                    ReplacementTransform(zep[0][0], zsp[0][0]),
                    ReplacementTransform(zep[0][2:], zsp[0][2:]),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(fh(self, 0.4)).set_stroke(
            width=DEFAULT_STROKE_WIDTH * 2
        ).next_to(self.camera.frame.get_corner(DL), UR, LARGE_BUFF)

        teardrop = (
            ImageMobject("../props/static/raindrop.png")
            .scale_to_fit_width(radar.radome.width * 0.8)
            .move_to(self.camera.frame)
            .shift(UP + RIGHT * 2)
        )

        self.play(
            LaggedStart(
                *[FadeOut(m) for m in zsp[0][::-1]],
                FadeOut(zep[0][1]),
                radar.get_animation(run_time=2),
                teardrop.shift(UP * fh(self)).animate.shift(DOWN * fh(self)),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        bw = PI / 9
        z_vec = Line(radar.radome.get_center(), teardrop.get_left())
        zhat = z_vec.get_angle()
        radius = VT(radar.radome.width / 2)
        arc = always_redraw(
            lambda: Arc(
                radius=~radius,
                start_angle=zhat - bw / 2,
                angle=bw,
                arc_center=radar.radome.get_center(),
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            ).set_z_index(-1)
        )
        self.add(arc)

        rx_radius = VT(teardrop.width)
        rx_scatter = always_redraw(
            lambda: Arc(
                radius=max(0, ~radius - z_vec.get_length()),
                start_angle=PI - zhat + bw / 2,
                angle=bw - TAU,
                arc_center=teardrop.get_center(),
                color=BLUE,
                stroke_opacity=0.2,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            ).set_z_index(-2)
        )
        rx_backscatter = always_redraw(
            lambda: Arc(
                radius=max(0, ~radius - z_vec.get_length()),
                start_angle=PI - zhat + bw / 2,
                angle=bw,
                arc_center=teardrop.get_center(),
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            ).set_z_index(-2)
        )

        radar_mask1 = SurroundingRectangle(
            Group(radar.left_leg, radar.right_leg, radar.middle_leg),
            buff=0,
            stroke_opacity=0,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
        ).set_z_index(-1)
        radar_mask2 = (
            Circle(
                radius=radar.radome.width / 2,
                stroke_opacity=0,
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
            )
            .set_z_index(-1)
            .move_to(radar.radome)
        )

        self.add(rx_backscatter, rx_scatter)
        self.add(radar_mask1, radar_mask2)

        self.next_section(skip_animations=skip_animations(False))

        # self.add(z_vec)
        self.play(
            LaggedStart(
                radius @ (z_vec.get_length() * 3),
            ),
            run_time=10,
        )

        self.wait(0.5)

        pr = (
            MathTex(r"P_r")
            .scale(2)
            .next_to(radar.radome, UP, LARGE_BUFF)
            .set_color(RX_COLOR)
        )
        pr_bez = CubicBezier(
            radar.radome.get_left(),
            radar.radome.get_left() + [-0.5, 0.5, 0],
            pr.get_corner(DL) + [-0.5, -0.5, 0],
            pr.get_corner(DL),
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(LEFT),
                Create(pr_bez),
                FadeIn(pr),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        # self.play()

        self.wait(2)


class WeatherRadarEqn(MovingCameraScene):
    def construct(self):
        pr_eqn = MathTex(
            r"P_r(R) = \frac{P_t G^2 \lambda^2 \theta_{\text{3dB}} \phi_{\text{3dB}} c\tau}{512 (2\ln 2) \pi^2 R^2 L} \eta"
        )
        pr_eqn2 = MathTex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} \frac{c \tau \Omega}{2} \frac{\pi^5 |K|^2 Z}{\lambda^2 R^2}"
        )
        Group(pr_eqn, pr_eqn2).arrange(DOWN, MED_LARGE_BUFF)
        self.add(pr_eqn, pr_eqn2)
