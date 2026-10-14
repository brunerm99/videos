import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manimlib import *

from melp import VT, ease_in_out_sine, fh, fw


BACKGROUND_COLOR = "#183340"
RFTK_WEIGHT_COLORS = ("#1e2c31", "#2d8fbf", "#f07828")
RFTK_AF_COLORS = (
    "#1e2c31",
    "#2d8fbf",
    "#3dbcac",
    "#f09050",
    "#f07828",
    "#fc6255",
)

MIN_ARRAY_SIZE = 4
MAX_ARRAY_SIZE = 32
TRANSITION_SECONDS = 14
FULL_ARRAY_SECONDS = 1.2
START_HOLD_SECONDS = 0.8
AF_FLOOR_DB = -40
AF_ROTATIONS = 2
POP_SIZE_SPAN = 0.5
POP_OVERSHOOT = 2.2

AF_RADIAL_SAMPLES = 129
AF_ANGULAR_SAMPLES = 193


def _rgb(hex_color):
    color = hex_color.lstrip("#")
    return np.array(
        [int(color[index : index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    ) / 255


def _smoothstep(values):
    values = np.clip(values, 0, 1)
    return values * values * (3 - 2 * values)


def _ease_out_back(values):
    values = np.clip(values, 0, 1)
    shifted = values - 1
    return 1 + (POP_OVERSHOOT + 1) * shifted**3 + POP_OVERSHOOT * shifted**2


def _popped_array_size(array_size):
    target_size = 2 * np.ceil(array_size / 2)
    previous_size = target_size - 2
    pop_progress = np.clip(
        (array_size - target_size + POP_SIZE_SPAN) / POP_SIZE_SPAN,
        0,
        1,
    )
    return previous_size + 2 * _ease_out_back(pop_progress)


def _interpolate_palette(values, palette):
    values = np.asarray(values, dtype=float)
    colors = np.array([_rgb(color) for color in palette])
    scaled = np.clip(values, 0, 1) * (len(colors) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(colors) - 1)
    blend = (scaled - lower)[..., None]
    return colors[lower] * (1 - blend) + colors[upper] * blend


def _rotation_matrix_x(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, cosine, -sine],
            [0, sine, cosine],
        ]
    )


def _rotation_matrix_z(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0],
            [sine, cosine, 0],
            [0, 0, 1],
        ]
    )


class ArraySizeShort(InteractiveScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "fps": 60,
    }

    def construct(self):
        cycle = VT(0)
        loop_seconds = TRANSITION_SECONDS * 2 + FULL_ARRAY_SECONDS

        coordinates = (
            np.arange(MAX_ARRAY_SIZE) - (MAX_ARRAY_SIZE - 1) / 2
        )
        element_x, element_y = np.meshgrid(coordinates, coordinates)
        element_x = element_x.ravel()
        element_y = element_y.ravel()
        activation_size = 2 * (
            np.maximum(np.abs(element_x), np.abs(element_y)) + 0.5
        )

        rho = np.linspace(0, 1, AF_RADIAL_SAMPLES)
        phi = np.linspace(0, TAU, AF_ANGULAR_SAMPLES)
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
        u_grid = rho_grid * np.cos(phi_grid)
        v_grid = rho_grid * np.sin(phi_grid)
        z_grid = np.sqrt(np.maximum(0, 1 - rho_grid**2))

        element_center = DOWN * fh(self, 0.144)
        af_center = UP * fh(self, 0.21)
        element_panel_width = fw(self, 0.9)
        af_scale = fw(self, 0.43)
        tilt = _rotation_matrix_x(-68 * DEGREES)

        def state_at(cycle_progress):
            elapsed = cycle_progress * loop_seconds
            if elapsed < TRANSITION_SECONDS:
                transition = elapsed / TRANSITION_SECONDS
                growth = ease_in_out_sine(transition)
            elif elapsed < TRANSITION_SECONDS + FULL_ARRAY_SECONDS:
                growth = 1
            else:
                transition = (
                    elapsed - TRANSITION_SECONDS - FULL_ARRAY_SECONDS
                ) / TRANSITION_SECONDS
                growth = 1 - ease_in_out_sine(transition)

            array_size = MIN_ARRAY_SIZE + growth * (
                MAX_ARRAY_SIZE - MIN_ARRAY_SIZE
            )
            azimuth = TAU * AF_ROTATIONS * cycle_progress
            return array_size, azimuth

        elements = DotCloud(
            points=np.zeros((MAX_ARRAY_SIZE**2, 3)),
            radius=0,
            anti_alias_width=2,
        )

        def update_elements(mobject):
            array_size, _ = state_at(~cycle)
            pop_progress = np.clip(
                (array_size - activation_size + POP_SIZE_SPAN)
                / POP_SIZE_SPAN,
                0,
                1,
            )
            weights = _smoothstep(pop_progress)
            pitch = element_panel_width / array_size
            points = np.column_stack(
                [
                    element_x * pitch,
                    element_y * pitch,
                    np.zeros_like(element_x),
                ]
            )
            points += element_center

            pop_scale = _ease_out_back(pop_progress)
            radii = pitch * 0.34 * pop_scale
            colors = _interpolate_palette(weights, RFTK_WEIGHT_COLORS)
            alpha = (weights > 1e-6).astype(float)[..., None]
            rgba = np.concatenate([colors, alpha], axis=1)

            mobject.set_points(points)
            mobject.set_radii(radii)
            mobject.set_rgba_array(rgba)

        update_elements(elements)
        elements.add_updater(update_elements)

        array_factor = Surface(
            color=WHITE,
            shading=(0.2, 0.15, 0.25),
            depth_test=True,
            resolution=(AF_RADIAL_SAMPLES, AF_ANGULAR_SAMPLES),
        )

        def update_array_factor(mobject):
            array_size, azimuth = state_at(~cycle)
            array_size = _popped_array_size(array_size)

            af_u = np.abs(
                np.sinc(array_size * u_grid / 2) / np.sinc(u_grid / 2)
            )
            af_v = np.abs(
                np.sinc(array_size * v_grid / 2) / np.sinc(v_grid / 2)
            )
            amplitude = np.minimum(af_u * af_v, 1)
            af_db = np.maximum(
                20
                * np.log10(
                    np.maximum(amplitude, 10 ** (AF_FLOOR_DB / 20))
                ),
                AF_FLOOR_DB,
            )
            radial_scale = np.clip(
                (af_db - AF_FLOOR_DB) / -AF_FLOOR_DB,
                0,
                1,
            )

            points = np.stack(
                [
                    u_grid * radial_scale,
                    v_grid * radial_scale,
                    z_grid * radial_scale,
                ],
                axis=-1,
            )
            rotation = tilt @ _rotation_matrix_z(azimuth)
            points = points @ rotation.T
            points = points * af_scale + af_center

            tangent_rho = np.gradient(points, axis=0)
            tangent_phi = np.gradient(points, axis=1)
            flat_points = points.reshape((-1, 3))

            color_values = np.clip(
                (af_db - AF_FLOOR_DB) / -AF_FLOOR_DB,
                0,
                1,
            )
            colors = _interpolate_palette(color_values, RFTK_AF_COLORS)
            rgba = np.concatenate(
                [colors, np.ones((*colors.shape[:-1], 1))],
                axis=-1,
            ).reshape((-1, 4))

            mobject.data["point"][:] = flat_points
            mobject.data["du_point"][:] = (
                points + tangent_rho
            ).reshape((-1, 3))
            mobject.data["dv_point"][:] = (
                points + tangent_phi
            ).reshape((-1, 3))
            mobject.data["rgba"][:] = rgba
            mobject.note_changed_data()

        update_array_factor(array_factor)
        array_factor.add_updater(update_array_factor)

        self.add(array_factor, elements)
        self.wait(START_HOLD_SECONDS)
        self.play(
            cycle.animate(run_time=loop_seconds, rate_func=linear).set_value(1)
        )
