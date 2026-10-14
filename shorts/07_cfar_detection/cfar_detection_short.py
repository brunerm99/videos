import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manimlib import *

from melp import VT, fh, fw


BACKGROUND_COLOR = "#183340"
RETURN_COLOR = RED
NOISE_COLOR = PURPLE
TARGET_COLOR = GREEN
THRESHOLD_COLOR = YELLOW

LOOP_SECONDS = 30
START_HOLD_SECONDS = 0.6
SAMPLE_COUNT = 401
NUM_REFERENCE_CELLS = 30
NUM_GUARD_CELLS = 10
NUM_THRESHOLD_DASHES = 28


def _gaussian(x_values, center, width):
    return np.exp(-0.5 * ((x_values - center) / width) ** 2)


def _cfar_threshold(signal):
    reference_sum = np.zeros_like(signal)
    for offset in range(
        NUM_GUARD_CELLS + 1,
        NUM_GUARD_CELLS + NUM_REFERENCE_CELLS + 1,
    ):
        reference_sum += np.roll(signal, offset)
        reference_sum += np.roll(signal, -offset)

    reference_mean = reference_sum / (2 * NUM_REFERENCE_CELLS)
    return reference_mean + 0.105


class CFARDetectionShort(InteractiveScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "fps": 60,
    }

    def construct(self):
        cycle = VT(0)
        x_values = np.linspace(0, 1, SAMPLE_COUNT)

        axes = Axes(
            x_range=[0, 1, 0.125],
            y_range=[0, 1.15, 0.2],
            width=fw(self, 0.88),
            height=fh(self, 0.47),
            axis_config={
                "stroke_width": DEFAULT_STROKE_WIDTH * 0.55,
                "stroke_opacity": 0.58,
                "tick_size": 0.055,
            },
        ).shift(UP * fh(self, 0.10))
        axes_origin = axes.c2p(0, 0)
        axes_x_basis = axes.c2p(1, 0) - axes_origin
        axes_y_basis = axes.c2p(0, 1) - axes_origin

        def scene_state():
            phase = TAU * ~cycle

            clutter_shift = 0.018 * np.sin(phase)
            fine_shift = 0.022 * np.sin(phase - 0.7)
            clutter = (
                0.105
                + 0.030
                * np.sin(TAU * (11 * (x_values + fine_shift) + 0.12))
                + 0.018
                * np.sin(TAU * (23 * (x_values - fine_shift) - 0.20))
                + 0.020
                * np.sin(TAU * (37 * x_values + 0.13 * np.sin(phase)))
                + 0.335
                * _gaussian(
                    x_values,
                    0.29 + clutter_shift,
                    0.105,
                )
                + 0.245
                * _gaussian(
                    x_values,
                    0.73 - 0.8 * clutter_shift,
                    0.145,
                )
                + 0.105
                * _gaussian(
                    x_values,
                    0.53 + 0.015 * np.cos(phase),
                    0.040,
                )
            )
            clutter = np.clip(clutter, 0.025, None)

            travel = 0.5 - 0.5 * np.cos(phase)
            target_positions = np.array(
                [
                    0.10 + 0.78 * travel,
                    0.88 - 0.70 * travel,
                    0.23 + 0.54 * (0.5 + 0.5 * np.sin(phase - 0.35)),
                ]
            )
            target_strengths = np.array([0.62, 0.12, 0.09])
            target_widths = np.array([0.010, 0.012, 0.009])

            targets = np.zeros_like(x_values)
            for position, strength, width in zip(
                target_positions,
                target_strengths,
                target_widths,
            ):
                targets += strength * _gaussian(
                    x_values,
                    position,
                    width,
                )

            received = clutter + targets
            threshold = _cfar_threshold(received)
            return received, threshold, target_positions

        def graph_points(values, indices=None):
            if indices is None:
                indices = np.arange(SAMPLE_COUNT)
            return (
                axes_origin
                + x_values[indices, None] * axes_x_basis
                + values[indices, None] * axes_y_basis
            )

        noise_region = VMobject()
        noise_region.set_fill(NOISE_COLOR, opacity=0.18)
        noise_region.set_stroke(width=0)

        return_curve = VMobject()
        return_curve.set_stroke(
            RETURN_COLOR,
            width=DEFAULT_STROKE_WIDTH,
        )

        threshold_dashes = VGroup(
            *[
                VMobject().set_stroke(
                    THRESHOLD_COLOR,
                    width=DEFAULT_STROKE_WIDTH * 0.75,
                )
                for _ in range(NUM_THRESHOLD_DASHES)
            ]
        )

        detection_regions = VGroup(
            *[
                VMobject()
                .set_fill(TARGET_COLOR, opacity=0.48)
                .set_stroke(width=0)
                for _ in range(3)
            ]
        )
        detection_caps = VGroup(
            *[
                VMobject().set_stroke(
                    TARGET_COLOR,
                    width=DEFAULT_STROKE_WIDTH * 1.25,
                )
                for _ in range(3)
            ]
        )

        def clear_path(mobject):
            mobject.set_points(np.zeros((0, 3)))

        def update_plot(_mobject):
            received, threshold, target_positions = scene_state()
            return_points = graph_points(received)
            threshold_points = graph_points(threshold)

            baseline_right = axes.c2p(1, 0)
            baseline_left = axes.c2p(0, 0)
            noise_region.set_points_as_corners(
                np.vstack(
                    [
                        baseline_left,
                        threshold_points,
                        baseline_right,
                    ]
                )
            )
            return_curve.set_points_smoothly(return_points)

            dash_step = SAMPLE_COUNT / NUM_THRESHOLD_DASHES
            for dash_index, dash in enumerate(threshold_dashes):
                start = int(dash_index * dash_step)
                end = int((dash_index + 0.58) * dash_step)
                end = min(max(end, start + 2), SAMPLE_COUNT)
                dash.set_points_smoothly(threshold_points[start:end])

            for region, cap, position in zip(
                detection_regions,
                detection_caps,
                target_positions,
            ):
                local = np.abs(x_values - position) < 0.045
                detected = local & (received > threshold)
                indices = np.flatnonzero(detected)
                if len(indices) < 2:
                    clear_path(region)
                    clear_path(cap)
                    continue

                split_indices = np.where(np.diff(indices) > 1)[0] + 1
                runs = np.split(indices, split_indices)
                run = max(runs, key=len)
                if len(run) < 2:
                    clear_path(region)
                    clear_path(cap)
                    continue

                upper = graph_points(received, run)
                lower = graph_points(threshold, run)[::-1]
                region.set_points_as_corners(np.vstack([upper, lower]))
                cap.set_points_smoothly(upper)

        update_plot(return_curve)
        return_curve.add_updater(update_plot)

        self.add(
            axes,
            noise_region,
            return_curve,
            threshold_dashes,
            detection_regions,
            detection_caps,
        )
        self.wait(START_HOLD_SECONDS)
        self.play(
            cycle.animate(run_time=LOOP_SECONDS, rate_func=linear).set_value(1)
        )
