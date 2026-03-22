from __future__ import annotations

import sys

import numpy as np
from manim import *

sys.path.insert(0, "..")
from props.style import BACKGROUND_COLOR

config.background_color = BACKGROUND_COLOR


TOWER_LIGHT = ManimColor.from_hex("#DCE6EB")
TOWER_MID = ManimColor.from_hex("#A9BBC4")
RADOME_COLOR = ManimColor.from_hex("#F4F7F8")
RADOME_SHADE = ManimColor.from_hex("#D3DDE1")
BASE_COLOR = ManimColor.from_hex("#4D626D")


def shaded(color: ManimColor, amount: float = 0.3) -> ManimColor:
    return interpolate_color(color, BLACK, amount)


def tube(start, end, radius, color, stroke_amount=0.35, resolution=(20, 24)):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    axis = end - start
    return (
        Cylinder(
            radius=radius,
            height=float(np.linalg.norm(axis)),
            direction=axis,
            resolution=resolution,
            fill_color=color,
            fill_opacity=1,
            checkerboard_colors=False,
            stroke_color=shaded(color, stroke_amount),
            stroke_width=0.45,
        )
        .move_to((start + end) / 2)
        .set_shade_in_3d(True)
    )


def joint(center, radius, color):
    return Sphere(
        center=np.array(center, dtype=float),
        radius=radius,
        resolution=(14, 24),
        fill_color=color,
        fill_opacity=1,
        checkerboard_colors=False,
        stroke_color=shaded(color, 0.28),
        stroke_width=0.35,
    ).set_shade_in_3d(True)


def disk(center, radius, height, color, stroke_amount=0.3):
    return (
        Cylinder(
            radius=radius,
            height=height,
            direction=UP,
            resolution=(20, 24),
            fill_color=color,
            fill_opacity=1,
            checkerboard_colors=False,
            stroke_color=shaded(color, stroke_amount),
            stroke_width=0.35,
        )
        .move_to(center)
        .set_shade_in_3d(True)
    )


class RadarTower3D(Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        front_left_base = np.array([-1.28, 0.0, 0.92])
        front_right_base = np.array([1.28, 0.0, 0.92])
        rear_base = np.array([0.0, -0.62, -1.0])

        front_left_top = front_left_base + UP * 3.45
        front_right_top = front_right_base + UP * 3.45
        rear_top = rear_base + UP * 3.45

        hub = np.array([0.0, 3.22, 0.0])
        mast_top = hub + UP * 0.72
        radome_center = hub + UP * 1.82

        leg_radius = 0.088
        brace_radius = 0.055
        support_radius = 0.07

        feet = Group(
            disk(front_left_base + DOWN * 0.06, 0.16, 0.08, BASE_COLOR),
            disk(front_right_base + DOWN * 0.06, 0.16, 0.08, BASE_COLOR),
            disk(rear_base + DOWN * 0.06, 0.16, 0.08, BASE_COLOR),
        )

        legs = Group(
            tube(front_left_base, front_left_top, leg_radius, TOWER_LIGHT),
            tube(front_right_base, front_right_top, leg_radius, TOWER_LIGHT),
            tube(rear_base, rear_top, leg_radius, TOWER_MID),
        )

        brace_levels = (0.82, 1.82)
        truss = Group()
        nodes = Group()
        for level in brace_levels:
            left_node = front_left_base + UP * level
            right_node = front_right_base + UP * level
            rear_node = rear_base + UP * level

            truss.add(tube(left_node, right_node, brace_radius, TOWER_MID))
            truss.add(tube(rear_node, left_node, brace_radius, TOWER_MID))
            truss.add(tube(rear_node, right_node, brace_radius, TOWER_MID))

            nodes.add(
                joint(left_node, 0.095, TOWER_LIGHT),
                joint(right_node, 0.095, TOWER_LIGHT),
                joint(rear_node, 0.095, TOWER_MID),
            )

        upper_ring = Group(
            tube(front_left_top, front_right_top, brace_radius, TOWER_MID),
            tube(front_left_top, rear_top, brace_radius, TOWER_MID),
            tube(front_right_top, rear_top, brace_radius, TOWER_MID),
        )

        top_supports = Group(
            tube(front_left_top, hub, support_radius, TOWER_LIGHT),
            tube(front_right_top, hub, support_radius, TOWER_LIGHT),
            tube(rear_top, hub, support_radius, TOWER_MID),
        )

        mast = Group(
            tube(hub, mast_top, 0.12, TOWER_LIGHT),
            disk(hub + UP * 0.16, 0.36, 0.07, TOWER_MID),
            joint(hub, 0.12, TOWER_LIGHT),
        )

        radome = Sphere(
            center=radome_center,
            radius=1.16,
            resolution=(24, 40),
            fill_color=RADOME_COLOR,
            fill_opacity=1,
            checkerboard_colors=False,
            stroke_color=shaded(RADOME_SHADE, 0.12),
            stroke_width=0.35,
        ).set_shade_in_3d(True)

        self.add(
            feet,
            legs,
            truss,
            upper_ring,
            top_supports,
            mast,
            nodes,
            radome,
        )
        self.move_to(ORIGIN)


class RadarTowerSceneBase(ThreeDScene):
    camera_phi = 56 * DEGREES
    camera_theta = -128 * DEGREES
    camera_gamma = 0
    camera_zoom = 0.9
    frame_center = (0.0, 2.05, 0.0)
    tower_shift = LEFT * 0.35 + UP * 0.32

    def setup_tower_scene(self):
        self.set_camera_orientation(
            phi=self.camera_phi,
            theta=self.camera_theta,
            zoom=self.camera_zoom,
            frame_center=self.frame_center,
            gamma=self.camera_gamma,
        )
        light_source = getattr(self.renderer.camera, "light_source", None)
        if light_source is not None:
            light_source.move_to(7 * LEFT + 9 * UP + 11 * OUT)

        tower = RadarTower3D().shift(self.tower_shift)
        self.add(tower)
        return tower

    def construct(self):
        self.setup_tower_scene()
        self.wait(0.1)


class WeatherRadar3D(RadarTowerSceneBase):
    camera_theta = -32 * DEGREES
    camera_zoom = 0.84


class WeatherRadar3DFrontCheck(RadarTowerSceneBase):
    camera_phi = 56 * DEGREES
    camera_theta = -90 * DEGREES
    camera_zoom = 0.88


class WeatherRadar3DSideCheck(RadarTowerSceneBase):
    camera_phi = 56 * DEGREES
    camera_theta = -22 * DEGREES
    camera_zoom = 0.92


class WeatherRadar3DTopCheck(RadarTowerSceneBase):
    camera_phi = 24 * DEGREES
    camera_theta = -128 * DEGREES
    camera_zoom = 0.86
    frame_center = (0.0, 2.35, 0.0)


class CenteredRadarTowerSceneBase(RadarTowerSceneBase):
    camera_phi = 0 * DEGREES
    camera_theta = -90 * DEGREES
    camera_zoom = 0.72
    frame_center = (0.0, 0.0, 0.0)
    tower_shift = ORIGIN


class DroneRadarTowerSceneBase(CenteredRadarTowerSceneBase):
    camera_phi = 68 * DEGREES
    camera_theta = -120 * DEGREES
    camera_zoom = 0.74
    frame_center = (0.0, 0.15, 0.0)


class WeatherRadar3DSpin(DroneRadarTowerSceneBase):
    camera_phi = 0 * DEGREES
    spin_duration = 1
    camera_gamma = 0

    def construct(self):
        self.setup_tower_scene()
        self.begin_ambient_camera_rotation(
            rate=-TAU / self.spin_duration, about="theta"
        )
        # self.wait(self.spin_duration)


class WeatherRadar3DSpinDroneCheckStart(DroneRadarTowerSceneBase):
    pass


class WeatherRadar3DSpinDroneCheckQuarter(DroneRadarTowerSceneBase):
    camera_theta = -60 * DEGREES


class WeatherRadar3DSpinDroneCheckHalf(DroneRadarTowerSceneBase):
    camera_theta = 0 * DEGREES


class WeatherRadar3DSpinDroneCheckThreeQuarter(DroneRadarTowerSceneBase):
    camera_theta = 60 * DEGREES


class WeatherRadar3DSpinCheckFront(CenteredRadarTowerSceneBase):
    camera_theta = -90 * DEGREES


class WeatherRadar3DSpinCheckFrontLeft(CenteredRadarTowerSceneBase):
    camera_theta = -120 * DEGREES


class WeatherRadar3DSpinCheckFrontLeftTight(CenteredRadarTowerSceneBase):
    camera_theta = -105 * DEGREES


class WeatherRadar3DSpinCheckQuarter(CenteredRadarTowerSceneBase):
    camera_theta = -15 * DEGREES


class WeatherRadar3DSpinCheckFrontRightTight(CenteredRadarTowerSceneBase):
    camera_theta = -75 * DEGREES


class WeatherRadar3DSpinCheckFrontRight(CenteredRadarTowerSceneBase):
    camera_theta = -60 * DEGREES


class WeatherRadar3DSpinCheckBack(CenteredRadarTowerSceneBase):
    camera_theta = 75 * DEGREES


class WeatherRadar3DSpinCheckOpposite(CenteredRadarTowerSceneBase):
    camera_theta = 165 * DEGREES
