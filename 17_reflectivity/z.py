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
from props.nexrad import get_nexrad_product_ppi_data
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

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_width(radar.radome.width * 2)
            .move_to(self.camera.frame)
            .shift(UP + RIGHT * 2)
        )

        self.play(
            LaggedStart(
                *[FadeOut(m) for m in zsp[0][::-1]],
                FadeOut(zep[0][1]),
                radar.get_animation(run_time=2),
                cloud.shift(UP * fh(self)).animate.shift(DOWN * fh(self)),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        bw = PI / 9
        z_vec = Line(radar.radome.get_center(), cloud.get_left())
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

        rx_radius = VT(cloud.width)
        rx_scatter = always_redraw(
            lambda: Arc(
                radius=max(0, ~radius - z_vec.get_length()),
                start_angle=PI - zhat + bw / 2,
                angle=bw - TAU,
                arc_center=cloud.get_center(),
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
                arc_center=cloud.get_center(),
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

        self.next_section(skip_animations=skip_animations(True))

        # self.add(z_vec)
        self.play(
            LaggedStart(
                radius @ (z_vec.get_length() * 3),
            ),
            run_time=10,
        )

        self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.shift(LEFT),
        #         Create(pr_bez),
        #         FadeIn(pr),
        #         lag_ratio=0.2,
        #     )
        # )

        pr_ax = (
            Axes(
                x_range=[0, 1, 0.1],
                y_range=[0, 1, 0.25],
                tips=False,
                x_length=fw(self, 0.4),
                y_length=fh(self, 0.4),
            )
            .next_to(radar.vgroup, UP, LARGE_BUFF)
            .shift(UP * fh(self))
        )

        cloud_range = VT(0.7)
        cloud_range2 = VT(0.3)
        pr_norm = VT(0)
        norm_val1 = VT(0.3)
        norm_val2 = VT(0.5)
        cloud2_show = VT(0)

        def get_rx():
            range_samples = np.linspace(0, 1, 200)
            cloud_sigma = 0.055
            rng = np.random.default_rng(7)
            received_power = (
                0.08
                + (1 - (~cloud_range * 0.9 * (1 - ~pr_norm)) - ~norm_val1 * ~pr_norm)
                * np.exp(-0.5 * ((range_samples - ~cloud_range) / cloud_sigma) ** 2)
                + rng.normal(0, 0.025, range_samples.size)
                + ~norm_val2
                * ~cloud2_show
                * np.exp(-0.5 * ((range_samples - ~cloud_range2) / cloud_sigma) ** 2)
            )
            received_power = np.clip(received_power, pr_ax.y_range[0], None)
            return lambda r: np.interp(r, range_samples, received_power)

        pr_plot = always_redraw(
            lambda: pr_ax.plot(
                get_rx(),
                x_range=[0, 1, 1 / 800],
                color=RX_COLOR,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                use_smoothing=False,
            )
        )

        pr = MathTex(r"P_r").scale(1.5).rotate(PI / 2).next_to(pr_ax, LEFT, SMALL_BUFF)
        pr_bez = CubicBezier(
            radar.radome.get_left(),
            radar.radome.get_left() + [-0.5, 0.5, 0],
            pr_ax.copy().shift(DOWN * fh(self)).get_corner(DL) + [-0.5, -0.5, 0],
            pr_ax.copy().shift(DOWN * fh(self)).get_corner(DL),
        )

        self.wait(0.5)

        self.add(pr_ax, pr_plot)

        self.play(
            LaggedStart(
                self.camera.frame.animate.set_x(
                    Group(
                        pr_ax.copy().shift(DOWN * fh(self)), radar.vgroup, cloud
                    ).get_x()
                ).shift(UP),
                Create(pr_bez),
                Group(pr_ax, pr).animate.shift(DOWN * fh(self)),
                lag_ratio=0.3,
            )
        )

        # self.play(Create(pr_ax), Create(pr_plot))

        self.wait(0.5)

        self.remove(rx_backscatter, rx_scatter)

        self.camera.frame.save_state()
        pr.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(Group(pr_ax, pr)).scale(0.7),
                pr.animate.scale(2 / 1.5)
                .set_color(RX_COLOR)
                .next_to(pr_ax, LEFT, SMALL_BUFF),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(self.camera.frame.animate.restore(), pr.animate.restore())

        self.wait(0.5)

        self.play(
            cloud.animate.shift(LEFT * fw(self, 0.3) + DOWN),
            cloud_range @ 0.3,
            run_time=3,
        )

        self.wait(0.5)

        strong_arrow = Arrow(
            pr_ax.c2p(~cloud_range),
            pr_ax.i2gp(
                ~cloud_range,
                pr_ax.plot(
                    get_rx(),
                    x_range=[0, 1, 1 / 800],
                    color=RX_COLOR,
                    stroke_width=DEFAULT_STROKE_WIDTH * 1,
                    use_smoothing=False,
                ),
            ),
            buff=SMALL_BUFF,
        )

        self.play(GrowArrow(strong_arrow))

        self.wait(0.5)

        self.play(FadeOut(strong_arrow))

        self.wait(0.5)

        self.play(
            cloud.animate.shift(RIGHT * fw(self, 0.4) + UP),
            cloud_range @ 0.8,
            run_time=4,
        )

        self.wait(0.5)

        weak_arrow = Arrow(
            pr_ax.c2p(~cloud_range),
            pr_ax.i2gp(
                ~cloud_range,
                pr_ax.plot(
                    get_rx(),
                    x_range=[0, 1, 1 / 800],
                    color=RX_COLOR,
                    stroke_width=DEFAULT_STROKE_WIDTH * 1,
                    use_smoothing=False,
                ),
            ),
            buff=SMALL_BUFF,
        )

        self.play(GrowArrow(weak_arrow))

        self.wait(0.5)

        self.play(FadeOut(weak_arrow))

        self.wait(0.5)

        self.play(
            cloud_range @ 0.3,
            cloud.animate.shift(LEFT * fw(self, 0.4) + DOWN),
            pr_norm @ 1,
        )

        self.wait(0.5)

        intensity_value = DashedLine(
            pr_ax.c2p(0.1, 0.79),
            pr_ax.c2p(0.9, 0.79),
            dash_length=DEFAULT_DASH_LENGTH * 2,
            color=YELLOW,
        )

        self.play(Create(intensity_value))

        self.wait(0.5)

        self.play(
            cloud_range @ 0.5,
            cloud.animate.shift(RIGHT * fw(self, 0.2) + UP / 2),
        )

        self.wait(0.5)

        self.play(
            cloud_range @ 0.7,
            cloud.animate.shift(RIGHT * fw(self, 0.2) + UP / 2),
        )

        self.wait(0.5)

        dist_line = DoubleArrow(
            radar.radome.get_right(),
            cloud.get_left(),
            buff=0,
            color=BLUE,
        )
        range_label = MathTex(r"R").next_to(
            dist_line.get_midpoint(),
            DOWN,
            SMALL_BUFF * 1.5,
        )

        self.play(
            LaggedStart(
                GrowArrow(dist_line),
                GrowFromCenter(range_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        other_things = MathTex(r"P_t,\ G,\ \lambda,\, \ldots").next_to(
            radar.right_leg, RIGHT, MED_LARGE_BUFF, DOWN
        )

        self.play(
            LaggedStart(
                GrowFromCenter(other_things[0][0]),
                GrowFromCenter(other_things[0][1]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(other_things[0][2]),
                GrowFromCenter(other_things[0][3]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(other_things[0][4]),
                GrowFromCenter(other_things[0][5]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(other_things[0][6]),
                GrowFromCenter(other_things[0][7]),
                GrowFromCenter(other_things[0][8]),
                GrowFromCenter(other_things[0][9]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        z_box = (
            RoundedRectangle(
                width=fw(self, 0.5),
                height=fh(self, 0.3),
                corner_radius=0.2,
                stroke_color=YELLOW,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            )
            .next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF)
            .set_z_index(1)
        )

        z = (
            MathTex(r"Z \sim")
            .scale_to_fit_height(z_box.height * 0.3)
            .next_to(z_box.get_left(), RIGHT, z_box.width * 0.1)
            .set_z_index(2)
        )
        lbrace = (
            Brace(z_box, LEFT)
            .scale(0.9)
            .next_to(z, RIGHT, MED_SMALL_BUFF)
            .set_y(z_box.get_y())
            .set_z_index(2)
        )
        rbrace = (
            Brace(z_box, RIGHT)
            .scale(0.9)
            .next_to(z_box.get_right(), LEFT, MED_SMALL_BUFF)
            .set_y(z_box.get_y())
            .set_z_index(2)
        )

        # self.add(z_box, z)

        z_group = Group(z_box, z)

        self.play(z_group.shift(UP * fh(self, 0.5)).animate.shift(DOWN * fh(self, 0.5)))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        # self.add(pr.copy().set_z_index(2).rotate(-PI / 2).next_to(z, RIGHT, SMALL_BUFF))

        pr_new = (
            pr.copy()
            .set_color(RED)
            .rotate(-PI / 2)
            .next_to(lbrace, RIGHT, SMALL_BUFF)
            .set_z_index(2)
        )

        self.play(
            LaggedStart(
                ReplacementTransform(pr.set_z_index(2), pr_new, path_arc=PI / 2),
                LaggedStart(FadeIn(lbrace), FadeIn(rbrace), lag_ratio=0.2),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        pt_new = (
            MathTex(r"P_t")
            .set_color(BLUE)
            .scale_to_fit_height(pr_new.height)
            .set_z_index(2)
        )
        g_new = MathTex(r"G").scale_to_fit_height(pr_new.height).set_z_index(2)
        lam_new = MathTex(r"\lambda").scale_to_fit_height(pr_new.height).set_z_index(2)
        dots_new = (
            MathTex(r"\ldots").scale_to_fit_width(pr_new.width * 1.2).set_z_index(2)
        )

        vars = (
            Group(
                pr_new.copy(),
                range_label.copy().scale_to_fit_height(pr_new.height).set_z_index(2),
                pt_new,
                g_new,
                lam_new,
                dots_new,
            )
            .arrange_in_grid(2, 3)
            .scale_to_fit_width(Group(lbrace, rbrace).width * 0.8)
            .move_to(Group(lbrace, rbrace))
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                Transform(pr_new.set_z_index(2), vars[0]),
                ReplacementTransform(range_label[0].set_z_index(2), vars[1]),
                ReplacementTransform(other_things[0][:2].set_z_index(2), vars[2]),
                FadeOut(other_things[0][2]),
                ReplacementTransform(other_things[0][3].set_z_index(2), vars[3]),
                FadeOut(other_things[0][4]),
                ReplacementTransform(other_things[0][5].set_z_index(2), vars[4]),
                FadeOut(other_things[0][6]),
                ReplacementTransform(other_things[0][7:].set_z_index(2), vars[5]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            Group(z_box, z, lbrace, rbrace, vars, pr_new)
            .animate.move_to(self.camera.frame)
            .shift(UP * fh(self)),
            Uncreate(intensity_value),
        )

        self.wait(0.5)

        dist_line_new = DoubleArrow(
            radar.radome.get_right(),
            cloud.copy().shift(UP + LEFT).get_left(),
            buff=0,
            color=BLUE,
        )

        cloud2 = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_width(cloud.width * 0.6)
            .next_to(cloud.copy().shift(UP + LEFT), DOWN, MED_SMALL_BUFF)
            .shift(LEFT * 3)
        )
        dist_line2 = DoubleArrow(
            radar.radome.get_right(),
            cloud2.get_left(),
            buff=0,
            color=BLUE,
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    cloud.animate.shift(UP + LEFT),
                    Transform(dist_line, dist_line_new),
                    cloud_range @ 0.6,
                ),
                cloud2.shift(DOWN * fh(self)).animate.shift(UP * fh(self)),
                GrowArrow(dist_line2),
                cloud2_show @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        dist_line_new_new = DoubleArrow(
            radar.radome.get_right(),
            cloud.copy().set_x(cloud2.get_x()).get_corner(DL),
            buff=0,
            color=BLUE,
        )
        dist_line_new2 = DoubleArrow(
            radar.radome.get_right(),
            cloud2.copy().set_x(cloud.get_x()).get_left(),
            buff=0,
            color=BLUE,
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    cloud.animate.set_x(cloud2.get_x()),
                    Transform(dist_line, dist_line_new_new),
                    cloud_range @ ~cloud_range2,
                ),
                AnimationGroup(
                    cloud2.animate.set_x(cloud.get_x()),
                    Transform(dist_line2, dist_line_new2),
                    cloud_range2 @ ~cloud_range,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(0.5)

        pr_final = pr_new.copy()
        to = Arrow(LEFT, RIGHT)
        z_final = MathTex(r"Z").scale_to_fit_height(pr_final[0][0].height)
        final_group = (
            Group(pr_final, to, z_final)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .move_to(self.camera.frame)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[FadeOut(m) for m in [*vars, lbrace, rbrace, z[0][1], z_box]],
                ReplacementTransform(pr_new[0], pr_final[0], path_arc=PI / 2),
                GrowArrow(to),
                ReplacementTransform(z[0][0], z_final[0], path_arc=PI / 2),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(pr_final, z_final, to),
                self.camera.frame.animate.scale(0.8),
                pr_ax.animate.move_to(self.camera.frame),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                norm_val1.animate(rate_func=get_phase_eased_sine(2)).increment_value(
                    0.3
                ),
                cloud_range.animate(rate_func=get_phase_eased_sine(2)).increment_value(
                    -0.15
                ),
                cloud_range2.animate(rate_func=get_phase_eased_sine(2)).increment_value(
                    -0.15
                ),
                norm_val2.animate(rate_func=get_phase_eased_sine(2)).increment_value(
                    -0.3
                ),
                lag_ratio=0.1,
                run_time=8,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        # self.play(FadeOut(*self.mobjects))

        # self.wait(.5)

        self.wait(2)


class ResolutionVolume(MovingCameraScene):
    def construct(self):
        zdata = get_nexrad_product_ppi_data("reflectivity")
        zppi = PPIGrid(zdata, init_scan_progress=0).next_to(
            self.camera.frame.get_top(), UP
        )
        self.add(zppi)

        self.play(self.camera.frame.animate.move_to(zppi))

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
