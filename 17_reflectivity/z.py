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


class ResolutionVolume(ZoomedScene):
    def __init__(self, **kwargs):
        w = config.frame_height * 0.5
        h = config.frame_height * 0.35
        super().__init__(
            zoomed_display_height=h,
            zoomed_display_width=w,
            zoom_factor=0.03,
            zoomed_camera_config={
                "default_frame_stroke_width": DEFAULT_STROKE_WIDTH * 0.25,
            },
            # image_frame_stroke_width=0.1,
            # zoomed_camera_config={
            #     "default_frame_stroke_width": 3,
            # },
            **kwargs,
        )

    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        zdata = get_nexrad_product_ppi_data("reflectivity")
        zppi = PPIGrid(zdata, init_scan_progress=0).next_to(
            self.camera.frame.get_top(), UP
        )
        self.add(zppi)

        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(zppi),
                zppi.scan_progress.animate(run_time=3).set_value(360),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        zoom_display = self.zoomed_display.next_to(
            self.camera.frame.get_corner(UL), DR
        ).shift(UP / 2)

        zoom_frame = self.zoomed_camera.frame
        dip = zppi.data_image.height
        zoom_frame.move_to(
            zppi.data_image.get_center() + UP * dip * 0.23 + LEFT * dip * 0.22
        )

        self.play(self.camera.frame.animate.shift(UP))

        self.activate_zooming(animate=True)

        zoom_line_dl = always_redraw(
            lambda: Line(
                zoom_frame.get_corner(DL),
                zoom_display.get_corner(DL),
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        zoom_line_ur = always_redraw(
            lambda: Line(
                zoom_frame.get_corner(UR),
                zoom_display.get_corner(UR),
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )

        self.wait(0.5)

        self.play(
            Create(zoom_line_dl),
            Create(zoom_line_ur),
            self.camera.frame.animate.shift(LEFT),
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(fh(self, 0.4)).next_to(
            zppi.data_image, RIGHT, LARGE_BUFF * 1
        ).shift(DOWN).set_stroke(width=DEFAULT_STROKE_WIDTH * 2)

        self.play(
            LaggedStart(
                FadeOut(zoom_display),
                AnimationGroup(Uncreate(zoom_line_dl), Uncreate(zoom_line_ur)),
                FadeOut(zoom_frame),
                FadeOut(zppi.colorbar, zppi.colorbar_ticks),
                self.camera.frame.animate.set_y(zppi.data_image.get_y())
                .scale(0.7)
                .move_to(Group(zppi.data_image, radar.vgroup))
                .shift(RIGHT),
                radar.vgroup.shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        bw = PI / 6
        beam_u = Line(
            radar.radome.get_right(),
            radar.radome.get_right() + RIGHT * fw(self, 0.2),
            color=BLUE,
        ).set_z_index(-1)
        beam_d = Line(
            radar.radome.get_right(),
            radar.radome.get_right() + RIGHT * fw(self, 0.2),
            color=BLUE,
        ).set_z_index(-1)
        ppi_bw = 8
        ppi_range = 150
        ppi_beam_u = always_redraw(
            lambda: Line(
                zppi.ra2gp(0, 0),
                zppi.ra2gp(ppi_range, -ppi_bw / 2),
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )
        ppi_beam_d = always_redraw(
            lambda: Line(
                zppi.ra2gp(0, 0),
                zppi.ra2gp(ppi_range, ppi_bw / 2),
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
            )
        )

        self.play(
            LaggedStart(
                Succession(
                    AnimationGroup(
                        Create(beam_u),
                        Create(beam_d),
                    ),
                    AnimationGroup(
                        beam_u.animate.rotate(
                            bw / 2,
                            about_point=radar.radome.get_right(),
                        ),
                        beam_d.animate.rotate(
                            -bw / 2,
                            about_point=radar.radome.get_right(),
                        ),
                    ),
                ),
                AnimationGroup(
                    Create(ppi_beam_u),
                    Create(ppi_beam_d),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        r_offset = VT(70)
        rr = VT(30)

        def get_ppi_rr():

            l1 = ArcBetweenPoints(
                zppi.ra2gp(~r_offset, -ppi_bw / 2),
                zppi.ra2gp(~r_offset + ~rr, -ppi_bw / 2),
                angle=0,
            )
            l2 = ArcBetweenPoints(
                zppi.ra2gp(~r_offset + ~rr, ppi_bw / 2),
                zppi.ra2gp(~r_offset, ppi_bw / 2),
                angle=0,
            )
            a1 = ArcBetweenPoints(
                zppi.ra2gp(~r_offset, ppi_bw / 2),
                zppi.ra2gp(~r_offset, -ppi_bw / 2),
                angle=PI / 3,
            )
            a2 = ArcBetweenPoints(
                zppi.ra2gp(~r_offset + ~rr, ppi_bw / 2),
                zppi.ra2gp(~r_offset + ~rr, -ppi_bw / 2),
                angle=PI / 3,
            )
            return (
                ArcPolygonFromArcs(l1, a2, l2, a1)
                .set_fill(color=BLUE, opacity=0.9)
                .set_stroke(opacity=0)
            )

        ppi_rr = always_redraw(get_ppi_rr)

        self.play(FadeIn(ppi_rr))

        self.wait(2)


class WeatherRadarEqn(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        # pr_eqn = MathTex(
        #     r"P_r(R) = \frac{P_t G^2 \lambda^2 \theta_{\text{3dB}} \phi_{\text{3dB}} c\tau}{512 (2\ln 2) \pi^2 R^2 L} \eta"
        # )
        pr_eqn1 = MathTex(
            r"P_r = \frac{P_t G^2 c \tau \Omega \pi^5 |K|^2 Z}{2 (4 \pi)^3 L \lambda^2 R^2}"
        ).scale(2)
        pr_eqn1[0][:2].set_color(RX_COLOR)
        pr_eqn1[0][16].set_color(BLUE)

        pr_eqn2 = MathTex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} \frac{c \tau \Omega}{2} \frac{\pi^5 |K|^2 Z}{\lambda^2 R^2}"
        ).scale(2)
        pr_eqn2[0][:2].set_color(RX_COLOR)
        pr_eqn2[0][25].set_color(BLUE)
        # Group(pr_eqn, pr_eqn2).arrange(DOWN, MED_LARGE_BUFF)
        # self.add(pr_eqn, pr_eqn2)

        pr_to_z = MathTex(r"P_r \rightarrow Z").scale(3)
        pr_to_z[0][:2].set_color(RX_COLOR)
        pr_to_z[0][3].set_color(BLUE)

        self.play(
            LaggedStart(
                pr_to_z[0][:2]
                .shift(DOWN)
                .set_opacity(0)
                .animate.shift(UP)
                .set_opacity(1),
                pr_to_z[0][2]
                .shift(DOWN)
                .set_opacity(0)
                .animate.shift(UP)
                .set_opacity(1),
                pr_to_z[0][3]
                .shift(DOWN)
                .set_opacity(0)
                .animate.shift(UP)
                .set_opacity(1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(pr_to_z[0][:2], pr_eqn1[0][:2]),
                FadeOut(pr_to_z[0][2]),
                ReplacementTransform(pr_to_z[0][3], pr_eqn1[0][16]),
                LaggedStart(
                    *[
                        GrowFromPoint(m, pr_eqn1.get_center())
                        for m in [*pr_eqn1[0][2:16], *pr_eqn1[0][17:]]
                    ],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        pr_eqn1p2 = MathTex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} \frac{c \tau \Omega \pi^5 |K|^2 Z}{2 \lambda^2 R^2}"
        ).scale(2)
        pr_eqn1p2[0][:2].set_color(RX_COLOR)
        pr_eqn1p2[0][23].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1[0][:3], pr_eqn1p2[0][:3]),
                ReplacementTransform(pr_eqn1[0][7:17], pr_eqn1p2[0][14:24]),
                ReplacementTransform(pr_eqn1[0][3:7], pr_eqn1p2[0][3:7]),
                ReplacementTransform(pr_eqn1[0][17], pr_eqn1p2[0][24]),
                ReplacementTransform(
                    pr_eqn1[0][19:25], pr_eqn1p2[0][8:14], path_arc=-PI
                ),
                ReplacementTransform(pr_eqn1[0][18], pr_eqn1p2[0][25]),
                GrowFromCenter(pr_eqn1p2[0][7]),
                ReplacementTransform(pr_eqn1[0][25:29], pr_eqn1p2[0][26:31]),
                lag_ratio=0.3,
            ),
            run_time=3.5,
        )

        self.wait(0.5)

        self.play(
            pr_eqn1p2[0][8:14].animate.set_color(ORANGE),
            pr_eqn1p2[0][3:8].animate.set_color(ORANGE),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p2[0][3:5]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(UP)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p2[0][13]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(DOWN)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p2[0][5:7]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(UP)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p2[0][8:13]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(DOWN)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        pr_eqn1p3 = MathTex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} \frac{c \tau \Omega}{2} \frac{\pi^5 |K|^2 Z}{\lambda^2 R^2}"
        ).scale(2)
        pr_eqn1p3[0][:2].set_color(RX_COLOR)
        pr_eqn1p3[0][25].set_color(BLUE)
        pr_eqn1p3[0][8:14].set_color(ORANGE)
        pr_eqn1p3[0][3:8].set_color(ORANGE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1p2[0][:3], pr_eqn1p3[0][:3]),
                ReplacementTransform(pr_eqn1p2[0][3:14], pr_eqn1p3[0][3:14]),
                ReplacementTransform(pr_eqn1p2[0][17:24], pr_eqn1p3[0][19:26]),
                ReplacementTransform(pr_eqn1p2[0][24], pr_eqn1p3[0][26]),
                ReplacementTransform(pr_eqn1p2[0][26:30], pr_eqn1p3[0][27:31]),
                ReplacementTransform(pr_eqn1p2[0][14:17], pr_eqn1p3[0][14:17]),
                GrowFromCenter(pr_eqn1p3[0][17]),
                ReplacementTransform(pr_eqn1p2[0][25], pr_eqn1p3[0][18]),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(pr_eqn1p3[0][14:19].animate.set_color(RED))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p3[0][15]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(UP)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(pr_eqn1p3[0][16].animate.shift(UP * 2).set_color(YELLOW))

        self.wait(0.5)

        omega = MathTex(
            r"\Omega_{\text{eff}} \approx \frac{\pi \theta_{\text{3dB}} \phi_{\text{3dB}} }{8 \ln 2}"
        ).scale(2)
        omega.set_color(YELLOW)
        omega.shift(pr_eqn1p3[0][16].get_center() - omega[0][0].get_center())

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(omega),
                LaggedStart(*[FadeIn(m) for m in omega[0][1:]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(omega[0][1:]),
                self.camera.frame.animate.restore(),
                pr_eqn1p3[0][16].animate.shift(DOWN * 2).set_color(RED),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(pr_eqn1p3[0][19:].animate.set_color(BLUE))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(pr_eqn1p3[0][22].animate.shift(UP).set_color(YELLOW))

        self.wait(0.5)

        kw = (
            MathTex(r"K_w")
            .scale(2)
            .next_to(pr_eqn1p3[0][22], LEFT, LARGE_BUFF)
            .shift(UP / 2)
        )
        kw_bez = CubicBezier(
            pr_eqn1p3[0][22].get_top() + [0, 0.1, 0],
            pr_eqn1p3[0][22].get_top() + [0, 0.5, 0],
            kw.get_right() + [0.5, 0, 0],
            kw.get_right() + [0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                Create(kw_bez),
                FadeIn(kw),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(kw),
                Uncreate(kw_bez),
                pr_eqn1p3[0][22].animate.shift(DOWN).set_color(BLUE),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p3[0][27]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(DOWN)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p3[0][29]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(DOWN)
            .set_color(YELLOW)
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            pr_eqn1p3[0][25]
            .animate(rate_func=rate_functions.there_and_back_with_pause, run_time=2)
            .shift(UP)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        z_eqn = MathTex(
            r"Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} \frac{2}{c \tau \Omega} \frac{\lambda^2 R^2}{\pi^5 |K|^2}"
        ).scale(2)
        z_eqn[0][0].set_color(BLUE)
        z_eqn[0][2:4].set_color(RED)
        z_eqn[0][4:15].set_color(ORANGE)
        z_eqn[0][15:20].set_color(RED)
        z_eqn[0][20:].set_color(BLUE)

        # self.add(
        #     index_labels(pr_eqn1p3[0]),
        #     z_eqn.next_to(pr_eqn1p3, DOWN),
        #     index_labels(z_eqn[0]),
        # )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1p3[0][25], z_eqn[0][0], path_arc=PI / 2),
                ReplacementTransform(pr_eqn1p3[0][:2], z_eqn[0][2:4], path_arc=PI / 2),
                ReplacementTransform(pr_eqn1p3[0][2], z_eqn[0][1]),
                ReplacementTransform(pr_eqn1p3[0][3:7], z_eqn[0][11:15], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][8:14], z_eqn[0][4:10], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][7], z_eqn[0][10]),
                ReplacementTransform(pr_eqn1p3[0][14:17], z_eqn[0][17:20], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][18], z_eqn[0][15], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][17], z_eqn[0][16]),
                ReplacementTransform(pr_eqn1p3[0][19:25], z_eqn[0][25:31], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][27:31], z_eqn[0][20:24], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[0][26], z_eqn[0][24]),
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                z_eqn[0][22:24]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(UP),
                z_eqn[0][15:20]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(UP),
                z_eqn[0][11:13]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(DOWN),
                z_eqn[0][27:31]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(DOWN),
                z_eqn[0][13:15]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(DOWN),
                z_eqn[0][4:9]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(UP),
                z_eqn[0][25:27]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(DOWN),
                z_eqn[0][9]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(UP),
                z_eqn[0][20:22]
                .animate(rate_func=rate_functions.there_and_back, run_time=2)
                .shift(UP),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        units = (
            MathTex(r"\left[ \frac{\text{m}^6}{\text{m}^3} \right]")
            .scale(2)
            .next_to(z_eqn[0][24], RIGHT, MED_SMALL_BUFF)
        )

        all_group = Group(z_eqn, units)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.2
                ).move_to(all_group),
                LaggedStart(*[GrowFromCenter(m) for m in units[0]], lag_ratio=0.1),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        z_eqn_mm = MathTex(
            r"Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} \frac{2}{c \tau \Omega} \frac{\lambda^2 R^2}{\pi^5 |K|^2} 10^{18}"
        ).scale(2)
        z_eqn_mm[0][0].set_color(BLUE)
        z_eqn_mm[0][2:4].set_color(RED)
        z_eqn_mm[0][4:15].set_color(ORANGE)
        z_eqn_mm[0][15:20].set_color(RED)
        z_eqn_mm[0][20:31].set_color(BLUE)

        mm_units = (
            MathTex(r"\left[ \frac{\text{mm}^6}{\text{m}^3} \right]")
            .scale(2)
            .next_to(z_eqn_mm, RIGHT, MED_SMALL_BUFF)
        )
        all_group = Group(z_eqn_mm, mm_units)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    all_group.width * 1.2
                ).move_to(all_group),
                ReplacementTransform(z_eqn[0], z_eqn_mm[0][:31]),
                ReplacementTransform(units[0][0], mm_units[0][0]),
                ReplacementTransform(units[0][1], mm_units[0][1]),
                GrowFromCenter(mm_units[0][2]),
                ReplacementTransform(units[0][2], mm_units[0][3]),
                ReplacementTransform(units[0][3], mm_units[0][4]),
                ReplacementTransform(units[0][4], mm_units[0][5]),
                ReplacementTransform(units[0][5], mm_units[0][6]),
                ReplacementTransform(units[0][6], mm_units[0][7]),
                *[GrowFromCenter(m) for m in z_eqn_mm[0][-4:]],
            ),
        )

        self.wait(0.5)

        drop = (
            Circle(
                radius=mm_units.width * 0.25,
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_color=BLUE,
            )
            .next_to(mm_units, RIGHT, LARGE_BUFF * 3)
            .shift(UP * fh(self, 0.3))
        )
        mm6 = mm_units[0][1:4]
        drop_bez = CubicBezier(
            mm6.get_right() + [0.1, 0, 0],
            mm6.get_right() + [2, 0, 0],
            drop.get_left() + [-2, 0, 0],
            drop.get_left() + [-0.1, 0, 0],
            color=BLUE,
        )

        units_expl = Group(mm_units, drop)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(0.7)
                .set_x(units_expl.get_x())
                .shift(UP),
                mm6.animate.set_color(BLUE),
                mm_units[0][-1].animate.set_opacity(0.2),
                Create(drop_bez),
                GrowFromCenter(drop),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        diam = DoubleArrow(drop.get_left(), drop.get_right(), buff=0)

        self.play(GrowArrow(diam))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(drop, diam),
                Uncreate(drop_bez),
                mm_units[0][-1].animate.set_opacity(1),
                mm6.animate.set_color(WHITE),
                self.camera.frame.animate.restore(),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        pr2z = CurvedArrow(
            z_eqn_mm[0][2:4].get_top() + [0, 0.1, 0],
            z_eqn_mm[0][0].get_top() + [0, 0.1, 0],
            angle=PI,
        )

        self.play(FadeIn(pr2z))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        z_val = (
            MathTex(r"Z = 20 \ \text{dBZ}")
            .scale(3)
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self, 1.5))
        )
        z_val[0][0].set_color(BLUE)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(z_eqn_mm[0][0], z_val[0][0], path_arc=PI / 3),
                    self.camera.frame.animate.shift(DOWN * fh(self, 1.5)),
                ),
                LaggedStart(*[FadeIn(m) for m in z_val[0][1:]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class Rayleigh(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        sq = Square(
            fh(self, 0.1), fill_color=BACKGROUND_COLOR, fill_opacity=1
        ).set_z_index(0)
        sq_back = (
            Square(fh(self, 0.1))
            .set_z_index(-1)
            .shift(UR * sq.height / (2 * math.sqrt(2)))
        )
        lul = Line(sq.get_corner(UL), sq_back.get_corner(UL))
        ldr = Line(sq.get_corner(DR), sq_back.get_corner(DR))
        lur = Line(sq.get_corner(UR), sq_back.get_corner(UR))
        cube = Group(sq, sq_back, lul, ldr, lur)  # .shift(LEFT * fw(self, 0.25))

        self.play(cube.shift(LEFT * fw(self)).animate.shift(RIGHT * fw(self)))

        self.wait(0.5)

        raindrop = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_width(cube.width)
            .set_opacity(0.7)
        )
        snowflake = ImageMobject("../props/static/snowflake.png").scale_to_fit_width(
            cube.width
        )
        dots = Text("...", font=FONT).scale_to_fit_width(cube.width)
        stuff = (
            Group(raindrop, snowflake, dots)
            .arrange(DOWN, LARGE_BUFF)
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self, 0.15))
        )

        bez_u = CubicBezier(
            cube.copy().shift(LEFT * fw(self, 0.15)).get_center()
            + cube.width * 0.3 * RIGHT,
            cube.copy().shift(LEFT * fw(self, 0.15)).get_center()
            + cube.width * 1.3 * RIGHT,
            stuff.get_corner(UL) + [-1, 0, 0],
            stuff.get_corner(UL),
            color=GREEN,
        )
        bez_d = CubicBezier(
            cube.copy().shift(LEFT * fw(self, 0.15)).get_center()
            + cube.width * 0.3 * RIGHT,
            cube.copy().shift(LEFT * fw(self, 0.15)).get_center()
            + cube.width * 1.3 * RIGHT,
            stuff.get_corner(DL) + [-1, -0.1, 0],
            stuff.get_corner(DL) + [0, -0.1, 0],
            color=GREEN,
        )

        self.play(
            LaggedStart(
                cube.animate.shift(LEFT * fw(self, 0.15)),
                AnimationGroup(Create(bez_u), Create(bez_d)),
                FadeIn(raindrop),
                FadeIn(snowflake),
                LaggedStart(*[FadeIn(m) for m in dots], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        bez_r_u = CubicBezier(
            stuff.get_corner(UR),
            stuff.get_corner(UR) + [1, 0, 0],
            self.camera.frame.get_right() + [-2, 0, 0],
            self.camera.frame.get_right(),
            color=GREEN,
        )
        bez_r_d = CubicBezier(
            stuff.get_corner(DR) + [0, -0.1, 0],
            stuff.get_corner(DR) + [1, -0.1, 0],
            self.camera.frame.get_right() + [-2, 0, 0],
            self.camera.frame.get_right(),
            color=GREEN,
        )

        fsw = (
            MathTex(r"f/\lambda")
            .scale(2)
            .next_to(bez_r_d.get_end(), RIGHT, MED_SMALL_BUFF)
        )
        wl_f_relation = MathTex(r"f = \frac{c}{\lambda}").scale(2).move_to(fsw, LEFT)

        self.play(
            LaggedStart(
                AnimationGroup(Create(bez_r_d), Create(bez_r_u)),
                self.camera.frame.animate.scale(0.7).move_to(fsw),
                FadeIn(fsw[0][0]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        bar = wl_f_relation[0][3].copy()
        bar.rotate(70 * DEGREES).scale_to_fit_height(fsw[0][1].height).move_to(
            fsw[0][1]
        )

        self.play(
            LaggedStart(
                FadeIn(bar),
                FadeIn(fsw[0][2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                ReplacementTransform(fsw[0][0], wl_f_relation[0][0]),
                ReplacementTransform(fsw[0][2], wl_f_relation[0][4]),
                ReplacementTransform(bar, wl_f_relation[0][3]),
                GrowFromCenter(wl_f_relation[0][1]),
                GrowFromCenter(wl_f_relation[0][2]),
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        f_val = (
            MathTex(r"f = 2 - 30 \text{ GHz}")
            .scale(2)
            .move_to(wl_f_relation[0][0], LEFT)
        )

        self.play(
            LaggedStart(
                FadeOut(wl_f_relation[0][3]),
                FadeOut(wl_f_relation[0][2]),
                FadeOut(wl_f_relation[0][1]),
                wl_f_relation[0][4]
                .animate.next_to(
                    wl_f_relation[0][0], DOWN, LARGE_BUFF, aligned_edge=LEFT
                )
                .set_opacity(0.5),
                self.camera.frame.animate.scale(1.2).move_to(f_val),
                LaggedStart(*[FadeIn(m) for m in f_val[0][1:]], lag_ratio=0.1),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        wl_val = MathTex(r"\lambda \gg").scale(2).move_to(wl_f_relation[0][-1], LEFT)
        raindrop = (
            SVGMobject("../props/static/raindorp.svg")
            .set_fill(BLUE)
            .set_stroke(color=BLACK, width=DEFAULT_STROKE_WIDTH)
            .scale_to_fit_height(wl_f_relation[0][-1].height * 1.2)
            .set_opacity(0.7)
            .next_to(wl_val, RIGHT, MED_SMALL_BUFF * 1.2)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.set_y(Group(raindrop, f_val).get_y()),
                wl_f_relation[0][-1].animate.set_opacity(1),
                FadeIn(wl_val[0][1]),
                FadeIn(raindrop),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(fh(self, 0.4)).set_stroke(
            width=DEFAULT_STROKE_WIDTH * 2
        ).next_to(self.camera.frame.get_corner(DL), UR, LARGE_BUFF).shift(UP * fh(self))

        self.add(radar.vgroup)

        self.play(
            self.camera.frame.animate.shift(UP * fh(self)),
            raindrop.animate.move_to(self.camera.frame)
            .shift(UP * fh(self))
            .shift(UP + RIGHT * 2),
        )

        self.wait(0.5)

        bw = PI / 9
        z_vec = Line(radar.radome.get_center(), raindrop.get_left())
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

        rx_radius = VT(raindrop.width)
        rx_scatter = always_redraw(
            lambda: Arc(
                radius=max(0, ~radius - z_vec.get_length()),
                start_angle=PI - zhat + bw / 2,
                angle=bw - TAU,
                arc_center=raindrop.get_center(),
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
                arc_center=raindrop.get_center(),
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

        self.play(
            LaggedStart(
                radius @ (z_vec.get_length() * 3),
            ),
            run_time=10,
        )

        self.remove(rx_backscatter, rx_scatter)

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(2)
