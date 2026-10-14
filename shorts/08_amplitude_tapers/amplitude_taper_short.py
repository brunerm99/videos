import sys
from pathlib import Path

import numpy as np
from scipy import signal

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
TAPER_NAMES = (
    "Rectangular",
    "Taylor",
    "Chebyshev",
    "Hamming",
    "Hann",
    "Blackman",
)

ARRAY_SIZE = 16
LOOP_SECONDS = 24
AF_FLOOR_DB = -40
AF_ROTATIONS = 2
TRANSITION_START = 0.20
TRANSITION_END = 0.80
NAME_TRANSITION_START = 0.35
NAME_TRANSITION_END = 0.55

AF_RADIAL_SAMPLES = 129
AF_ANGULAR_SAMPLES = 193


def _rgb(hex_color):
    color = hex_color.lstrip("#")
    return np.array(
        [int(color[index : index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    ) / 255


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


def _normalized(window):
    window = np.asarray(window, dtype=float)
    return window / np.max(window)


def _taper_weights():
    return np.array(
        [
            np.ones(ARRAY_SIZE),
            _normalized(
                signal.windows.taylor(
                    ARRAY_SIZE,
                    nbar=3,
                    sll=30,
                    norm=True,
                    sym=True,
                )
            ),
            _normalized(
                signal.windows.chebwin(
                    ARRAY_SIZE,
                    at=60,
                    sym=True,
                )
            ),
            _normalized(signal.windows.hamming(ARRAY_SIZE, sym=True)),
            _normalized(signal.windows.hann(ARRAY_SIZE, sym=True)),
            _normalized(signal.windows.blackman(ARRAY_SIZE, sym=True)),
        ]
    )


class AmplitudeTaperShort(InteractiveScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "fps": 60,
    }

    def construct(self):
        cycle = VT(0)
        tapers = _taper_weights()
        taper_count = len(tapers)

        coordinates = np.arange(ARRAY_SIZE) - (ARRAY_SIZE - 1) / 2
        element_x, element_y = np.meshgrid(coordinates, coordinates)
        element_x = element_x.ravel()
        element_y = element_y.ravel()

        rho = np.linspace(0, 1, AF_RADIAL_SAMPLES)
        phi = np.linspace(0, TAU, AF_ANGULAR_SAMPLES)
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
        u_grid = rho_grid * np.cos(phi_grid)
        v_grid = rho_grid * np.sin(phi_grid)
        z_grid = np.sqrt(np.maximum(0, 1 - rho_grid**2))

        phase_positions = coordinates[:, None, None]
        steering_u = np.exp(1j * PI * phase_positions * u_grid)
        steering_v = np.exp(1j * PI * phase_positions * v_grid)
        responses_u = np.einsum("tn,nij->tij", tapers, steering_u)
        responses_v = np.einsum("tn,nij->tij", tapers, steering_v)
        taper_sums = np.sum(tapers, axis=1)

        element_center = DOWN * fh(self, 0.145)
        af_center = UP * fh(self, 0.21)
        element_panel_width = fw(self, 0.86)
        element_pitch = element_panel_width / ARRAY_SIZE
        af_scale = fw(self, 0.43)
        tilt = _rotation_matrix_x(-68 * DEGREES)

        element_points = np.column_stack(
            [
                element_x * element_pitch,
                element_y * element_pitch,
                np.zeros_like(element_x),
            ]
        )
        element_points += element_center

        def state_at(cycle_progress):
            segment_position = (cycle_progress % 1) * taper_count
            taper_index = int(np.floor(segment_position))
            next_taper_index = (taper_index + 1) % taper_count
            local_progress = segment_position - taper_index
            transition_progress = np.clip(
                (local_progress - TRANSITION_START)
                / (TRANSITION_END - TRANSITION_START),
                0,
                1,
            )
            blend = ease_in_out_sine(transition_progress)
            azimuth = TAU * AF_ROTATIONS * cycle_progress
            return (
                taper_index,
                next_taper_index,
                local_progress,
                blend,
                azimuth,
            )

        elements = DotCloud(
            points=element_points,
            radius=element_pitch * 0.34,
            anti_alias_width=2,
        )

        def update_elements(mobject):
            taper_index, next_taper_index, _, blend, _ = state_at(~cycle)
            taper = (
                tapers[taper_index] * (1 - blend)
                + tapers[next_taper_index] * blend
            )
            weights = np.clip(np.outer(taper, taper).ravel(), 0, 1)
            colors = _interpolate_palette(weights, RFTK_WEIGHT_COLORS)
            rgba = np.concatenate(
                [colors, np.ones((len(colors), 1))],
                axis=1,
            )

            mobject.set_points(element_points)
            mobject.set_radii(
                element_pitch * (0.27 + 0.07 * np.sqrt(weights))
            )
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
            taper_index, next_taper_index, _, blend, azimuth = state_at(
                ~cycle
            )
            response_u = (
                responses_u[taper_index] * (1 - blend)
                + responses_u[next_taper_index] * blend
            )
            response_v = (
                responses_v[taper_index] * (1 - blend)
                + responses_v[next_taper_index] * blend
            )
            taper_sum = (
                taper_sums[taper_index] * (1 - blend)
                + taper_sums[next_taper_index] * blend
            )
            amplitude = np.minimum(
                np.abs(response_u * response_v) / taper_sum**2,
                1,
            )
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

        label_y = fh(self, 0.118)
        label_travel = fw(self, 1.10)
        taper_labels = VGroup(
            *[
                Text(
                    name,
                    font="Maple Mono CN",
                    font_size=36,
                )
                .set_color(WHITE)
                .set_opacity(0.92)
                .set_z_index(10)
                for name in TAPER_NAMES
            ]
        )

        def update_taper_labels(mobject):
            taper_index, next_taper_index, local_progress, _, _ = state_at(
                ~cycle
            )
            name_progress = ease_in_out_sine(
                np.clip(
                    (local_progress - NAME_TRANSITION_START)
                    / (NAME_TRANSITION_END - NAME_TRANSITION_START),
                    0,
                    1,
                )
            )
            for index, label in enumerate(mobject):
                if index == taper_index:
                    label.move_to(
                        np.array(
                            [-label_travel * name_progress, label_y, 0]
                        )
                    )
                    label.set_opacity(0.92)
                elif index == next_taper_index:
                    label.move_to(
                        np.array(
                            [
                                label_travel * (1 - name_progress),
                                label_y,
                                0,
                            ]
                        )
                    )
                    label.set_opacity(0.92)
                else:
                    label.set_opacity(0)

        update_taper_labels(taper_labels)
        taper_labels.add_updater(update_taper_labels)

        self.add(array_factor, elements, taper_labels)
        self.play(
            cycle.animate(run_time=LOOP_SECONDS, rate_func=linear).set_value(1)
        )
