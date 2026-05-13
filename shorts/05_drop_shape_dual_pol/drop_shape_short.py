# drop_shape_short.py

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
from moderngl_window.geometry import sphere
from networkx import center
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, "../../")
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
config.pixel_height = 1920
config.pixel_width = 1080
config.frame_height = 14
config.frame_width = 9

SKIP_ANIMATIONS_OVERRIDE = True

load_dotenv("../../.env")
FONT = os.getenv("FONT", "")


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
        self.camera.frame.shift(DOWN * fh(self))

        self.wait(0.5)

        teardrop = (
            ImageMobject("../../props/static/raindrop.png")
            .scale_to_fit_width(fw(self, 0.4))
            .move_to(self.camera.frame)
        )

        self.play(teardrop.shift(UP * fh(self)).animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(0.5)

        # flow_lines = [*flow_lines, *deepcopy(flow_lines), *deepcopy(flow_lines)]

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
        self.next_section(skip_animations=skip_animations(True))

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
            ),
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
            run_time=9,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        nexrad = (
            VideoMobject("../../16_dual_pol/static/nexrad.mov", loop=False, speed=1)
            .scale_to_fit_width(fw(self, 2))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self))
        ).set_z_index(-1)
        self.add(nexrad)

        self.play(self.camera.frame.animate.shift(RIGHT * fw(self)))

        self.wait(9)

        thumbnail = (
            ImageMobject(
                "../../16_dual_pol/static/Thumbnail - How Weather Radars See Shape.png"
            )
            .scale_to_fit_width(fw(self, 0.8))
            .move_to(self.camera.frame)
        )
        thumbnail_box = SurroundingRectangle(thumbnail, buff=0, color=GREEN)
        thumbnail_group = Group(thumbnail, thumbnail_box).set_z_index(1)

        self.play(GrowFromCenter(thumbnail_group))

        self.wait(5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                FadeOut(nexrad, scale=0.9),
                FadeOut(thumbnail_group, scale=0.9),
                lag_ratio=0.3,
            )
        )

        self.wait(0.1)


class Birds(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        nexrad = (
            VideoMobject("../../16_dual_pol/static/nexrad.mov", loop=False, speed=1)
            .scale_to_fit_width(fw(self, 2))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self))
        ).set_z_index(-1)
        self.add(nexrad)

        self.play(self.camera.frame.animate.shift(RIGHT * fw(self)))

        self.wait(4)

        cloud = (
            SVGMobject("../../props/static/cloud.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_width(fw(self, 0.8))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self) + UP * fh(self, 0.25))
        )
        np.random.seed(1)
        cloud_center = cloud.get_center() + DOWN * cloud.height * 0.03
        drop_radius = cloud.width * 0.01
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

        drop_rows = [
            (-0.20, 4, 0.10),
            (-0.12, 7, 0.20),
            (-0.05, 10, 0.29),
            (0.03, 12, 0.35),
            (0.11, 11, 0.31),
            (0.19, 8, 0.22),
            (0.26, 5, 0.12),
        ]
        for y_frac, count, half_width_frac in drop_rows:
            row_offset = np.random.normal(0, cloud.width * 0.008)
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
                        + RIGHT * (x_frac * cloud.width + row_offset)
                        + UP
                        * (
                            y_frac * cloud.height
                            + np.random.normal(0, cloud.height * 0.008)
                        )
                    )
                )

        shuffle(sphere_drops_r)

        for sd in sphere_drops_r[: len(sphere_drops_r) // 3]:
            sd.stretch(1.5, 0)

        for sd in sphere_drops_r[-len(sphere_drops_r) // 3 :]:
            sd.stretch(1.5, 1)

        sphere_drops_r = Group(*sphere_drops_r)
        sphere_drops_r.shift(DOWN * cloud.height * 0.1)

        self.add(cloud, sphere_drops_r)

        radome = (
            self.camera.frame.get_center()
            + RIGHT * fw(self, 0.25)
            + UP * fh(self, 0.15)
        )
        to_cloud = CubicBezier(
            radome,
            radome + [1, 0, 0],
            cloud.get_left() + [-1, 0, 0],
            cloud.get_left() + [-0.1, 0, 0],
        )

        self.play(
            Succession(Create(to_cloud), Uncreate(to_cloud)),
            self.camera.frame.animate.shift(RIGHT * fw(self)),
        )

        self.wait(0.5)

        bird_1 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif", loop=True
            )
            .scale_to_fit_width(cloud.width * 0.7)
            .next_to(cloud, DOWN, fh(self, 0.1))
        )

        self.play(
            bird_1.shift(LEFT * fw(self))
            .animate(suspend_mobject_updating=False)
            .shift(RIGHT * fw(self))
        )

        self.wait(5)

        self.next_section(skip_animations=skip_animations(True))
        deq_min = 1.5
        deq_target = 6
        deq = VT(4)
        drop_scale = 0.95

        ax_opacity = VT(1)
        ax_rotation = VT(0)
        drop_ax = (
            Axes(
                x_range=[-1.5, 1.5, 0.5],
                y_range=[-1.5, 1.5, 0.5],
                tips=False,
                x_length=fh(self, 0.45),
                y_length=fh(self, 0.45),
            )
            .set_opacity(~ax_opacity)
            .rotate(~ax_rotation)
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self))
        )

        def get_drop_shape_updater(shift=ORIGIN, opacity_mult=1):
            def make_drop_shape():
                points = _thurai_drop_outline_points(~deq)
                drop = (
                    Polygon(
                        *points,
                        color=PRECIP_COLOR,
                        fill_color=PRECIP_COLOR,
                        fill_opacity=0.12 * opacity_mult,
                        stroke_opacity=opacity_mult,
                        stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                    )
                    .scale(drop_scale)
                    .set_z_index(2)
                    .move_to(drop_ax.c2p(0, 0))
                    .rotate(Line(drop_ax.c2p(0, 0), drop_ax.c2p(1, 0)).get_angle())
                    .shift(shift)
                )
                return drop

            return make_drop_shape

        drop_shape = always_redraw(get_drop_shape_updater())

        # self.add(drop_shape)

        self.play(
            ReplacementTransform(sphere_drops_r[len(sphere_drops_r) // 2], drop_shape),
            self.camera.frame.animate.shift(RIGHT * fw(self)),
            run_time=1.5,
        )
        self.remove(bird_1)
        # self.play(deq @ 1.5)

        self.wait(0.5)

        shift = 2.8
        angle = 60
        opacity_fade = 0.2
        drop_shape_2 = always_redraw(
            get_drop_shape_updater(
                shift=(
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 1,
                opacity_mult=1 - (opacity_fade * 1),
            )
        )
        drop_shape_3 = always_redraw(
            get_drop_shape_updater(
                shift=(
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 2,
                opacity_mult=1 - (opacity_fade * 2),
            )
        )
        drop_shape_4 = always_redraw(
            get_drop_shape_updater(
                shift=(
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 3,
                opacity_mult=1 - (opacity_fade * 3),
            )
        )
        drop_shape_5 = always_redraw(
            get_drop_shape_updater(
                shift=(
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 4,
                opacity_mult=1 - (opacity_fade * 4),
            )
        )

        # self.add( drop_shape_2,
        #     drop_shape_3,
        #     drop_shape_4,
        #     drop_shape_5,
        # )

        drop_ax_static = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            tips=False,
            x_length=fh(self, 0.45),
            y_length=fh(self, 0.45),
        ).set_z_index(-1)
        drop_ax_static.shift(drop_ax.c2p(0, 0) - drop_ax_static.c2p(0, 0))
        # self.remove(drop_ax)
        # self.add(drop_ax_static)

        drop_shapes = Group(
            drop_shape,
            drop_shape_2,
            drop_shape_3,
            drop_shape_4,
            drop_shape_5,
        )

        time_arrow = Arrow(
            drop_ax_static.c2p(0, 0),
            drop_ax_static.c2p(0, 0)
            + (
                UP * shift * np.cos(angle * PI / 180)
                + RIGHT * shift * np.sin(angle * PI / 180)
            )
            * 4.5,
            buff=0,
        )

        time_label = (
            Text("time -->", font=FONT)
            .rotate((90 - angle) * PI / 180)
            .next_to(time_arrow, DR, SMALL_BUFF)
            .shift(LEFT * fw(self, 0.6))
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        hpol_arrow = Arrow(
            drop_ax.c2p(0, 0),
            drop_ax.c2p(1.5, 0),
            color=HPOL_RX_COLOR,
            buff=0,
        )
        vpol_arrow = Arrow(
            drop_ax.c2p(0, 0),
            drop_ax.c2p(0, 1.5),
            color=VPOL_RX_COLOR,
            buff=0,
        )

        hpol_arrow_2 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(1.5, 0),
                color=HPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 1
            )
            .set_opacity(1 - (opacity_fade * 1))
        )
        vpol_arrow_2 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(0, 1.5),
                color=VPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 1
            )
            .set_opacity(1 - (opacity_fade * 1))
        )

        hpol_arrow_3 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(1.5, 0),
                color=HPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 2
            )
            .set_opacity(1 - (opacity_fade * 2))
        )
        vpol_arrow_3 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(0, 1.5),
                color=VPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 2
            )
            .set_opacity(1 - (opacity_fade * 2))
        )

        hpol_arrow_4 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(1.5, 0),
                color=HPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 3
            )
            .set_opacity(1 - (opacity_fade * 3))
        )
        vpol_arrow_4 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(0, 1.5),
                color=VPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 3
            )
            .set_opacity(1 - (opacity_fade * 3))
        )

        hpol_arrow_5 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(1.5, 0),
                color=HPOL_RX_COLOR,
                buff=0,
            )
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 4
            )
            .set_opacity(1 - (opacity_fade * 4))
        )
        vpol_arrow_5 = (
            Arrow(
                drop_ax.c2p(0, 0),
                drop_ax.c2p(0, 1.5),
                color=VPOL_RX_COLOR,
                buff=0,
            )
            .set_opacity(1 - (opacity_fade * 4))
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 4
            )
        )
        # self.camera.frame.scale_to_fit_width(Group(*self.mobjects).width * 1.3)

        self.play(
            LaggedStart(
                Create(drop_ax_static),
                GrowArrow(hpol_arrow),
                GrowArrow(vpol_arrow),
                lag_ratio=0.3,
            ),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        all_drop_stuff = Group(
            drop_shapes,
            drop_ax,
            hpol_arrow,
            vpol_arrow,
            hpol_arrow_2,
            vpol_arrow_2,
            hpol_arrow_3,
            vpol_arrow_3,
            hpol_arrow_4,
            vpol_arrow_4,
            hpol_arrow_5,
            vpol_arrow_5,
        )
        self.play(
            self.camera.frame.animate(run_time=2)
            .scale_to_fit_width(all_drop_stuff.width * 1.1)
            .move_to(all_drop_stuff),
            GrowArrow(time_arrow, run_time=2),
            FadeIn(time_label),
            LaggedStart(
                FadeIn(drop_shape_2),
                FadeIn(drop_shape_3),
                FadeIn(drop_shape_4),
                FadeIn(drop_shape_5),
                LaggedStart(
                    GrowArrow(hpol_arrow_2),
                    GrowArrow(vpol_arrow_2),
                    lag_ratio=0.1,
                ),
                LaggedStart(
                    GrowArrow(hpol_arrow_3),
                    GrowArrow(vpol_arrow_3),
                    lag_ratio=0.1,
                ),
                LaggedStart(
                    GrowArrow(hpol_arrow_4),
                    GrowArrow(vpol_arrow_4),
                    lag_ratio=0.1,
                ),
                LaggedStart(
                    GrowArrow(hpol_arrow_5),
                    GrowArrow(vpol_arrow_5),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        bird_ax = drop_ax.copy().next_to(drop_ax, DOWN, fh(self, 0.1))

        bird_1 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif",
                loop=True,
                start_time=0,
            )
            .scale_to_fit_width(bird_ax.width * 1)
            .move_to(bird_ax.c2p(0, 0))
        ).set_z_index(10)
        bird_2 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif",
                loop=True,
                start_time=1 * 0.5,
            )
            .scale_to_fit_width(bird_1.width)
            .move_to(bird_1)
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 1
            )
        ).set_z_index(9)
        bird_3 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif",
                loop=True,
                start_time=2 * 0.5,
            )
            .scale_to_fit_width(bird_1.width)
            .move_to(bird_1)
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 2
            )
        ).set_z_index(8)
        bird_4 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif",
                loop=True,
                start_time=3 * 0.5,
            )
            .scale_to_fit_width(bird_1.width)
            .move_to(bird_1)
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 3
            )
        ).set_z_index(7)
        bird_5 = (
            VideoMobject(
                "../../16_dual_pol/static/Flapping Bird transparent.gif",
                loop=True,
                start_time=4 * 0.5,
            )
            .scale_to_fit_width(bird_1.width)
            .move_to(bird_1)
            .shift(
                (
                    UP * shift * np.cos(angle * PI / 180)
                    + RIGHT * shift * np.sin(angle * PI / 180)
                )
                * 4
            )
        ).set_z_index(6)

        hpol_mult_1 = VT(0)
        vpol_mult_1 = VT(0)
        hpol_mult_2 = VT(0)
        vpol_mult_2 = VT(0)
        hpol_mult_3 = VT(0)
        vpol_mult_3 = VT(0)
        hpol_mult_4 = VT(0)
        vpol_mult_4 = VT(0)
        hpol_mult_5 = VT(0)
        vpol_mult_5 = VT(0)

        hpol_arrow_dynamic = always_redraw(
            lambda: Arrow(
                bird_ax.c2p(0, 0),
                bird_ax.c2p(1.5 * ~hpol_mult_1, 0),
                color=HPOL_RX_COLOR,
                buff=0,
            )
        )
        vpol_arrow_dynamic = always_redraw(
            lambda: Arrow(
                bird_ax.c2p(0, 0),
                bird_ax.c2p(0, 1.5 * ~vpol_mult_1),
                color=VPOL_RX_COLOR,
                buff=0,
            )
        )

        hpol_arrow_2_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(1.5 * ~hpol_mult_2, 0),
                    color=HPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 1
                )
                .set_opacity(1 - (opacity_fade * 1))
            )
        )
        vpol_arrow_2_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(0, 1.5 * ~vpol_mult_2),
                    color=VPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 1
                )
                .set_opacity(1 - (opacity_fade * 1))
            )
        )

        hpol_arrow_3_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(1.5 * ~hpol_mult_3, 0),
                    color=HPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 2
                )
                .set_opacity(1 - (opacity_fade * 2))
            )
        )
        vpol_arrow_3_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(0, 1.5 * ~vpol_mult_3),
                    color=VPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 2
                )
                .set_opacity(1 - (opacity_fade * 2))
            )
        )

        hpol_arrow_4_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(1.5 * ~hpol_mult_4, 0),
                    color=HPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 3
                )
                .set_opacity(1 - (opacity_fade * 3))
            )
        )
        vpol_arrow_4_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(0, 1.5 * ~vpol_mult_4),
                    color=VPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 3
                )
                .set_opacity(1 - (opacity_fade * 3))
            )
        )

        hpol_arrow_5_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(1.5 * ~hpol_mult_5, 0),
                    color=HPOL_RX_COLOR,
                    buff=0,
                )
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 4
                )
                .set_opacity(1 - (opacity_fade * 4))
            )
        )
        vpol_arrow_5_dynamic = always_redraw(
            lambda: (
                Arrow(
                    bird_ax.c2p(0, 0),
                    bird_ax.c2p(0, 1.5 * ~vpol_mult_5),
                    color=VPOL_RX_COLOR,
                    buff=0,
                )
                .set_opacity(1 - (opacity_fade * 4))
                .shift(
                    (
                        UP * shift * np.cos(angle * PI / 180)
                        + RIGHT * shift * np.sin(angle * PI / 180)
                    )
                    * 4
                )
            )
        )

        # self.remove(
        #     hpol_arrow,
        #     vpol_arrow,
        #     hpol_arrow_2,
        #     vpol_arrow_2,
        #     hpol_arrow_3,
        #     vpol_arrow_3,
        #     hpol_arrow_4,
        #     vpol_arrow_4,
        #     hpol_arrow_5,
        #     vpol_arrow_5,
        # )

        self.add(
            hpol_arrow_dynamic,
            vpol_arrow_dynamic,
            hpol_arrow_2_dynamic,
            vpol_arrow_2_dynamic,
            hpol_arrow_3_dynamic,
            vpol_arrow_3_dynamic,
            hpol_arrow_4_dynamic,
            vpol_arrow_4_dynamic,
            hpol_arrow_5_dynamic,
            vpol_arrow_5_dynamic,
        )

        time_mult = 2
        f = 2 * time_mult

        def hpol_bird_rate_func(t):
            return (
                rate_functions.there_and_back(t)
                * (
                    np.sin(2 * f * PI * t)
                    + 0.5 * np.sin(4 * f * PI * t)
                    + 0.3 * np.sin(8 * f * PI * t)
                )
                / 1.8
            )

        def vpol_bird_rate_func(t):
            return (
                rate_functions.there_and_back(t)
                * (
                    np.sin(2 * f * PI * t)
                    - 0.5 * np.sin(4 * f * PI * t)
                    + 0.3 * np.sin(8 * f * PI * t)
                )
                / 1.8
            )

        bird_time_arrow = Arrow(
            bird_ax.c2p(0, 0),
            bird_ax.c2p(0, 0)
            + (
                UP * shift * np.cos(angle * PI / 180)
                + RIGHT * shift * np.sin(angle * PI / 180)
            )
            * 4.5,
            buff=0,
        )

        self.next_section(skip_animations=skip_animations(True))

        all_bird_stuff = Group(bird_ax, bird_1, bird_2, bird_3, bird_4, bird_5)
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(
                    Group(all_drop_stuff, all_bird_stuff)
                ),
                Create(bird_ax),
                bird_1.shift(LEFT * fw(self))
                .animate(
                    rate_func=rate_functions.linear,
                    suspend_mobject_updating=False,
                )
                .shift(RIGHT * fw(self)),
                Create(bird_time_arrow),
                bird_2.shift(LEFT * fw(self))
                .animate(
                    rate_func=rate_functions.linear,
                    suspend_mobject_updating=False,
                )
                .shift(RIGHT * fw(self)),
                bird_3.shift(LEFT * fw(self))
                .animate(
                    rate_func=rate_functions.linear,
                    suspend_mobject_updating=False,
                )
                .shift(RIGHT * fw(self)),
                bird_4.shift(LEFT * fw(self))
                .animate(
                    rate_func=rate_functions.linear,
                    suspend_mobject_updating=False,
                )
                .shift(RIGHT * fw(self)),
                bird_5.shift(LEFT * fw(self))
                .animate(
                    rate_func=rate_functions.linear,
                    suspend_mobject_updating=False,
                )
                .shift(RIGHT * fw(self)),
                LaggedStart(
                    Succession(
                        AnimationGroup(hpol_mult_1 @ 1, vpol_mult_1 @ 1, run_time=0.5),
                        AnimationGroup(
                            hpol_mult_1.animate(
                                rate_func=hpol_bird_rate_func
                            ).set_value(1.5),
                            vpol_mult_1.animate(
                                rate_func=vpol_bird_rate_func
                            ).set_value(1.5),
                            run_time=6 * time_mult,
                        ),
                    ),
                    Succession(
                        AnimationGroup(hpol_mult_2 @ 1, vpol_mult_2 @ 1, run_time=0.5),
                        AnimationGroup(
                            hpol_mult_2.animate(
                                rate_func=hpol_bird_rate_func
                            ).set_value(1.5),
                            vpol_mult_2.animate(
                                rate_func=vpol_bird_rate_func
                            ).set_value(1.5),
                            run_time=6 * time_mult,
                        ),
                    ),
                    Succession(
                        AnimationGroup(hpol_mult_3 @ 1, vpol_mult_3 @ 1, run_time=0.5),
                        AnimationGroup(
                            hpol_mult_3.animate(
                                rate_func=hpol_bird_rate_func
                            ).set_value(1.5),
                            vpol_mult_3.animate(
                                rate_func=vpol_bird_rate_func
                            ).set_value(1.5),
                            run_time=6 * time_mult,
                        ),
                    ),
                    Succession(
                        AnimationGroup(hpol_mult_4 @ 1, vpol_mult_4 @ 1, run_time=0.5),
                        AnimationGroup(
                            hpol_mult_4.animate(
                                rate_func=hpol_bird_rate_func
                            ).set_value(1.5),
                            vpol_mult_4.animate(
                                rate_func=vpol_bird_rate_func
                            ).set_value(1.5),
                            run_time=6 * time_mult,
                        ),
                    ),
                    Succession(
                        AnimationGroup(hpol_mult_5 @ 1, vpol_mult_5 @ 1, run_time=0.5),
                        AnimationGroup(
                            hpol_mult_5.animate(
                                rate_func=hpol_bird_rate_func
                            ).set_value(1.5),
                            vpol_mult_5.animate(
                                rate_func=vpol_bird_rate_func
                            ).set_value(1.5),
                            run_time=6 * time_mult,
                        ),
                    ),
                    lag_ratio=0.05,
                ),
                lag_ratio=0.7,
                suspend_mobject_updating=False,
            )
        )

        thumbnail = (
            ImageMobject(
                "../../16_dual_pol/static/Thumbnail - How Weather Radars See Shape.png"
            )
            .scale_to_fit_width(fw(self, 0.8))
            .move_to(self.camera.frame)
            .shift(RIGHT * fw(self))
        )
        thumbnail_box = SurroundingRectangle(thumbnail, buff=0, color=GREEN)
        thumbnail_group = Group(thumbnail, thumbnail_box).set_z_index(1)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(RIGHT * fw(self)),
                GrowFromCenter(thumbnail_group),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(thumbnail_group, scale=0.9))

        self.wait(2)
