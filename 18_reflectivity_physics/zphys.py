# zgl.py

import os
import sys
from random import random, seed, shuffle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from manimlib import *
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, "..")
load_dotenv()

from melp import (
    VT,
    VideoMobject,
    ease_out_cubic,
    fh,
    fw,
    get_nexrad_product_ppi_data,
    unit,
)
from melp.props import (
    Beam,
    Cloud,
    PPIGrid,
    Terrain,
    TriangleMeshMobject,
    WSR88Model,
    get_wsr88_model,
    place_wsr88_model,
    place_wsr88_on_terrain,
)

BACKGROUND_COLOR = "#183340"
RX_COLOR = RED
TX_COLOR = BLUE
TOWER_LIGHT = "#DCE6EB"
TOWER_MID = "#A9BBC4"
BEAM_COLOR = BLUE
GATE_COLOR = TEAL
ACCENT_COLOR = YELLOW
FONT = os.getenv("FONT", "Maple Mono CN")


class BrightBand(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        snow_top = 3.0
        melting_top = 0.6
        melting_bottom = -0.6
        rain_bottom = -3.0

        column_left = -4.5
        column_right = 1.0
        column_width = column_right - column_left
        column_center_x = (column_left + column_right) / 2

        snow_layer = Rectangle(
            width=column_width,
            height=snow_top - melting_top,
            stroke_width=0,
            fill_color=TOWER_LIGHT,
            fill_opacity=0.035,
        ).move_to([column_center_x, (snow_top + melting_top) / 2, 0])
        melting_layer = Rectangle(
            width=column_width,
            height=melting_top - melting_bottom,
            stroke_width=0,
            fill_color=TEAL,
            fill_opacity=0.14,
        ).move_to([column_center_x, 0, 0])
        rain_layer = Rectangle(
            width=column_width,
            height=melting_bottom - rain_bottom,
            stroke_width=0,
            fill_color=BLUE_E,
            fill_opacity=0.06,
        ).move_to([column_center_x, (melting_bottom + rain_bottom) / 2, 0])
        layers = VGroup(snow_layer, melting_layer, rain_layer).set_z_index(-2)

        melting_label = (
            VGroup(
                Text("Melting Layer", font=FONT).scale(0.4),
                Text("(Bright Band)", font=FONT).scale(0.4),
            )
            .arrange(DOWN, SMALL_BUFF)
            .set_color(TOWER_LIGHT)
            .next_to(melting_layer, LEFT, SMALL_BUFF)
        )

        profile_x = 2.4
        profile_width = 2.2
        melting_edges = (
            VGroup(
                Line(
                    [column_left, melting_top, 0],
                    [profile_x + profile_width, melting_top, 0],
                ),
                Line(
                    [column_left, melting_bottom, 0],
                    [profile_x + profile_width, melting_bottom, 0],
                ),
            )
            .set_stroke(TEAL, 1.5, opacity=0.5)
            .set_z_index(-1)
        )

        profile_axes = VGroup(
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x, snow_top, 0],
            ),
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x + profile_width, rain_bottom, 0],
            ),
        ).set_stroke(GREY_B, 2, opacity=0.75)
        profile_label = (
            Text("Reflectivity", font=FONT, font_size=28)
            .set_color(TOWER_LIGHT)
            .next_to(profile_axes[1], DOWN, buff=0.18)
        )

        rng = np.random.default_rng(4)
        particle_count = 54
        particle_x = rng.uniform(
            column_left + 0.35, column_right - 0.35, particle_count
        )
        particle_size = rng.uniform(0.09, 0.14, particle_count)
        drift_phase = rng.uniform(0, TAU, particle_count)
        rotation_rate = rng.uniform(-0.8, 0.8, particle_count)
        cycle_rate = rng.uniform(0.88, 1.12, particle_count)

        snow_duration = 5.0
        melting_duration = 1.5
        rain_duration = 1.5
        cycle_duration = snow_duration + melting_duration + rain_duration
        age_offsets = np.linspace(0, cycle_duration, particle_count, endpoint=False)
        rng.shuffle(age_offsets)

        hydrometeors = Group()
        for radius in particle_size:
            snowflake = VGroup(
                *[
                    Line(-radius * RIGHT, radius * RIGHT).rotate(angle)
                    for angle in (0, PI / 3, 2 * PI / 3)
                ]
            ).set_stroke(TOWER_LIGHT, 1.5)
            water_shell = (
                Circle(radius=radius * 1.15)
                .set_fill(
                    BLUE_A,
                    opacity=0,
                )
                .set_stroke(BLUE_A, 1.5, opacity=0)
            )
            raindrop = Line(
                radius * 1.25 * UP,
                radius * 1.25 * DOWN,
            ).set_stroke(BLUE_A, 3, opacity=0)

            hydrometeors.add(Group(water_shell, snowflake, raindrop))

        elapsed = [0.0]
        current_heights = np.zeros(particle_count)
        current_reflectivity = np.zeros(particle_count)

        def update_hydrometeors(group, dt):
            elapsed[0] += dt

            for index, particle in enumerate(group):
                age = (
                    elapsed[0] * cycle_rate[index] + age_offsets[index]
                ) % cycle_duration

                if age < snow_duration:
                    alpha = age / snow_duration
                    height = snow_top + alpha * (melting_top - snow_top)
                    snow_opacity = 1
                    shell_opacity = 0
                    rain_opacity = 0
                    drift_scale = 1
                    reflectivity = 0.35
                elif age < snow_duration + melting_duration:
                    alpha = (age - snow_duration) / melting_duration
                    height = melting_top + alpha * (melting_bottom - melting_top)
                    snow_opacity = 1 - alpha
                    shell_opacity = np.sin(PI * alpha)
                    rain_opacity = alpha
                    drift_scale = 1 - 0.7 * alpha
                    reflectivity = 0.35 + 2.2 * shell_opacity + 0.65 * alpha
                else:
                    alpha = (age - snow_duration - melting_duration) / rain_duration
                    height = melting_bottom + alpha * (rain_bottom - melting_bottom)
                    snow_opacity = 0
                    shell_opacity = 0
                    rain_opacity = 1
                    drift_scale = 0.3
                    reflectivity = 1

                x_position = particle_x[index] + drift_scale * 0.12 * np.sin(
                    0.8 * elapsed[0] + drift_phase[index]
                )
                particle.move_to([x_position, height, 0])
                current_heights[index] = height
                current_reflectivity[index] = reflectivity

                water_shell, snowflake, raindrop = particle
                snowflake.set_opacity(snow_opacity)
                water_shell.set_opacity(0.7 * shell_opacity)
                raindrop.set_opacity(rain_opacity)
                snowflake.rotate(rotation_rate[index] * dt)

        update_hydrometeors(hydrometeors, 0)
        hydrometeors.add_updater(update_hydrometeors, call=False)
        hydrometeors.set_z_index(1)

        profile_heights = np.linspace(rain_bottom, snow_top, 73)
        profile_values = np.zeros_like(profile_heights)
        melting_bottom_index = np.argmin(np.abs(profile_heights - melting_bottom))
        melting_top_index = np.argmin(np.abs(profile_heights - melting_top))
        profile_indices = (
            np.arange(melting_bottom_index + 1),
            np.arange(melting_bottom_index, melting_top_index + 1),
            np.arange(melting_top_index, len(profile_heights)),
        )
        profile = VGroup(
            VMobject().set_stroke(BLUE_A, 4),
            VMobject().set_stroke(ACCENT_COLOR, 5),
            VMobject().set_stroke(TOWER_LIGHT, 4),
        ).set_z_index(2)

        def update_profile(curves, dt):
            vertical_distance = (
                profile_heights[:, None] - current_heights[None, :]
            ) / 0.14
            kernel = np.exp(-0.5 * vertical_distance**2)
            target_values = np.clip(
                0.02 + kernel @ current_reflectivity / 9,
                0.02,
                0.98,
            )
            response = 1 if dt == 0 else 1 - np.exp(-dt / 0.18)
            profile_values[:] += response * (target_values - profile_values)

            for curve, indices in zip(curves, profile_indices):
                points = np.column_stack(
                    (
                        profile_x + profile_width * profile_values[indices],
                        profile_heights[indices],
                        np.zeros(len(indices)),
                    )
                )
                curve.set_points_smoothly(points)

        update_profile(profile, 0)
        profile.add_updater(update_profile, call=False)

        overview = Group(
            layers,
            melting_edges,
            melting_label,
            profile_axes,
            profile_label,
            profile,
        )
        overview_center = overview.get_center()
        self.frame.reorient(0, 0, 0, center=overview_center, height=7.2)

        self.add(
            layers,
            melting_edges,
            melting_label,
            profile_axes,
            profile_label,
            hydrometeors,
            profile,
        )

        self.wait(20)
