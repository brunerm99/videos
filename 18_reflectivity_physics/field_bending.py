import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from manimlib import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from melp import VT, fh, fw

load_dotenv()

BACKGROUND_COLOR = "#183340"
FONT = os.getenv("FONT", "Maple Mono CN")
FIELD_COLOR = BLUE_B
HIGHLIGHT_COLOR = YELLOW
AMPLITUDE = 0.42
PERIOD = 3.0
WAVE_SPEED = 2.4
N_CYCLES = 5
N_FIELD_LINES = 32


def charge_motion(time):
    time = np.asarray(time)
    omega = TAU / PERIOD
    phase = omega * time
    u = np.clip(time / PERIOD, 0, 1)
    ramp = 10 * u**3 - 15 * u**4 + 6 * u**5
    ramp_velocity = 30 * u**2 * (1 - u) ** 2 / PERIOD
    ramp_acceleration = 60 * u * (1 - u) * (1 - 2 * u) / PERIOD**2
    position = AMPLITUDE * (1 + ramp * (np.cos(phase) - 1))
    velocity = AMPLITUDE * (
        ramp_velocity * (np.cos(phase) - 1) - ramp * omega * np.sin(phase)
    )
    acceleration = AMPLITUDE * (
        ramp_acceleration * (np.cos(phase) - 1)
        - 2 * ramp_velocity * omega * np.sin(phase)
        - ramp * omega**2 * np.cos(phase)
    )
    return position, velocity, acceleration


def electric_field(points, time):
    position = charge_motion(time)[0]
    distance = np.hypot(points[:, 0], points[:, 1] - position)
    retarded_time = time - distance / WAVE_SPEED

    for _ in range(5):
        position, velocity, _ = charge_motion(retarded_time)
        dy = points[:, 1] - position
        distance = np.hypot(points[:, 0], dy)
        kappa = 1 - dy * velocity / (distance * WAVE_SPEED)
        retarded_time -= (retarded_time + distance / WAVE_SPEED - time) / kappa

    position, velocity, acceleration = charge_motion(retarded_time)
    dy = points[:, 1] - position
    distance = np.hypot(points[:, 0], dy)
    nx = points[:, 0] / distance
    ny = dy / distance
    beta = velocity / WAVE_SPEED
    kappa = 1 - ny * beta
    radial = (1 - beta**2) / (kappa**3 * distance**2)
    radiation = acceleration / (WAVE_SPEED**2 * kappa**3 * distance)

    # Lienard-Wiechert field, with the common q / (4 pi epsilon_0) omitted.
    return np.column_stack(
        (
            nx * radial + nx * ny * radiation,
            (ny - beta) * radial - nx**2 * radiation,
        )
    )


def trace_field_lines(time, extent, step=0.065):
    angles = np.arange(N_FIELD_LINES) * TAU / N_FIELD_LINES
    points = 0.12 * np.column_stack((np.sin(angles), np.cos(angles)))
    points[:, 1] += charge_motion(time)[0]
    samples = [points.copy()]

    def direction(points):
        field = electric_field(points, time)
        return field / np.linalg.norm(field, axis=1, keepdims=True)

    for _ in range(int(2.1 * extent / step)):
        midpoint = points + step / 2 * direction(points)
        points = points + step * direction(midpoint)
        samples.append(points.copy())
        if np.all(np.linalg.norm(points, axis=1) > extent):
            break

    curves = np.zeros((N_FIELD_LINES, len(samples), 3))
    curves[:, :, :2] = np.transpose(samples, (1, 0, 2))
    return curves


class FieldBending(ThreeDScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 30}
    default_frame_orientation = (0, 0)
    always_depth_test = False

    def construct(self):
        self.frame.set_height(11)
        time = VT(0)
        extent = np.hypot(fw(self), fh(self)) / 2 + 1
        field_cache = {"time": None, "curves": None}

        def get_field_lines():
            if field_cache["time"] != ~time:
                field_cache["curves"] = trace_field_lines(~time, extent)
                field_cache["time"] = ~time
            lines = VGroup()
            for index, points in enumerate(field_cache["curves"]):
                highlighted = index in (N_FIELD_LINES // 4, 3 * N_FIELD_LINES // 4)
                m = VMobject(
                    stroke_color=HIGHLIGHT_COLOR if highlighted else FIELD_COLOR,
                    stroke_width=3 if highlighted else 1.4,
                    stroke_opacity=1 if highlighted else 0.65,
                )
                m.set_points_smoothly(points[::2])
                lines.add(m)
            return lines.deactivate_depth_test()

        field_lines = always_redraw(get_field_lines)
        charge_template = (
            VGroup(
                Dot(radius=fh(self, 0.015), fill_color=YELLOW),
                Text("+", font=FONT, fill_color=BACKGROUND_COLOR).set_height(
                    fh(self, 0.02)
                ),
            )
            .deactivate_depth_test()
            .set_z_index(2)
        )
        charge = always_redraw(
            lambda: charge_template.copy().move_to(UP * charge_motion(~time)[0])
        )

        # def get_acceleration_arrow():
        #     position, _, acceleration = charge_motion(~time)
        #     start = UP * position + RIGHT * 0.4
        #     length = acceleration / (AMPLITUDE * (TAU / PERIOD) ** 2) * 0.8
        #     m = Arrow(
        #         start,
        #         start + UP * (length if abs(length) > 0.01 else 0.01),
        #         buff=0,
        #         fill_color=WHITE,
        #         thickness=2,
        #         tip_width_ratio=4,
        #     )
        #     m.set_opacity(min(1, abs(length) / 0.08))
        #     return m.deactivate_depth_test().set_z_index(2)
        #
        # acceleration_arrow = always_redraw(get_acceleration_arrow)
        # title = Text("An oscillating charge", font=FONT).set_height(fh(self, 0.033))
        # title.to_corner(UL, buff=0.4)
        # title.add_background_rectangle(color=BACKGROUND_COLOR, opacity=0.95, buff=0.15)
        # title.deactivate_depth_test().set_z_index(3)

        self.add(field_lines, charge, perp_stroke=False)
        self.wait(0.7)
        self.play(
            time @ (N_CYCLES * PERIOD), run_time=N_CYCLES * PERIOD, rate_func=linear
        )
        self.wait(0.8)
