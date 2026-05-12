from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from manim import *

STYLE_PATH = Path(__file__).resolve().parents[1] / "props" / "style.py"
STYLE_SPEC = importlib.util.spec_from_file_location("radar_tower_style", STYLE_PATH)
if STYLE_SPEC is None or STYLE_SPEC.loader is None:
    raise ImportError(f"Unable to load style module from {STYLE_PATH}")
style_module = importlib.util.module_from_spec(STYLE_SPEC)
STYLE_SPEC.loader.exec_module(style_module)
BACKGROUND_COLOR = style_module.BACKGROUND_COLOR

config.background_color = BACKGROUND_COLOR


TOWER_LIGHT = ManimColor.from_hex("#DCE6EB")
TOWER_MID = ManimColor.from_hex("#A9BBC4")
RADOME_COLOR = ManimColor.from_hex("#F4F7F8")
RADOME_SHADE = ManimColor.from_hex("#D3DDE1")
BASE_COLOR = ManimColor.from_hex("#4D626D")


def shaded(color: ManimColor, amount: float = 0.3) -> ManimColor:
    return interpolate_color(color, BLACK, amount)


def tripod_vertices(side_length: float, y: float = 0.0):
    altitude = np.sqrt(3) * side_length / 2
    front_z = altitude / 3
    rear_z = -2 * altitude / 3
    return (
        np.array([-side_length / 2, y, front_z], dtype=float),
        np.array([side_length / 2, y, front_z], dtype=float),
        np.array([0.0, y, rear_z], dtype=float),
    )


def point_between(start, end, alpha: float):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    return start + (end - start) * alpha


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

        base_side = 3.3
        top_side = 1.62
        tower_height = 3.9
        hub = np.array([0.0, 4.18, 0.0])
        mast_top = hub + UP * 0.72
        radome_radius = 1.16
        radome_center = np.array([0.0, 5.38, 0.0])

        front_left_base, front_right_base, rear_base = tripod_vertices(base_side)
        front_left_top, front_right_top, rear_top = tripod_vertices(
            top_side,
            y=tower_height,
        )

        base_points = (front_left_base, front_right_base, rear_base)
        top_points = (front_left_top, front_right_top, rear_top)
        edge_pairs = ((0, 1), (1, 2), (2, 0))

        leg_radius = 0.088
        brace_radius = 0.046
        support_radius = 0.072

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

        brace_levels = (0.26, 0.52, 0.78)
        ring_levels = [
            tuple(
                point_between(base_point, top_point, alpha)
                for base_point, top_point in zip(base_points, top_points)
            )
            for alpha in brace_levels
        ]

        rings = Group()
        diagonals = Group()
        nodes = Group()
        for ring_index, ring_points in enumerate([*ring_levels, top_points]):
            for i, j in edge_pairs:
                rings.add(tube(ring_points[i], ring_points[j], brace_radius, TOWER_MID))

            node_radius = 0.078 if ring_index < len(ring_levels) else 0.088
            nodes.add(
                joint(ring_points[0], node_radius, TOWER_LIGHT),
                joint(ring_points[1], node_radius, TOWER_LIGHT),
                joint(ring_points[2], node_radius, TOWER_MID),
            )

        all_levels = [base_points, *ring_levels, top_points]
        for bay_index, (lower_points, upper_points) in enumerate(
            zip(all_levels[:-1], all_levels[1:])
        ):
            for edge_index, (i, j) in enumerate(edge_pairs):
                if (bay_index + edge_index) % 2 == 0:
                    start_point = lower_points[i]
                    end_point = upper_points[j]
                else:
                    start_point = lower_points[j]
                    end_point = upper_points[i]

                diagonals.add(
                    tube(
                        start_point,
                        end_point,
                        brace_radius * 0.92,
                        TOWER_MID,
                        stroke_amount=0.3,
                    )
                )

        top_supports = Group(
            tube(front_left_top, hub, support_radius, TOWER_LIGHT),
            tube(front_right_top, hub, support_radius, TOWER_LIGHT),
            tube(rear_top, hub, support_radius, TOWER_MID),
        )

        mast = Group(
            tube(hub, mast_top, 0.12, TOWER_LIGHT),
            disk(hub + UP * 0.14, 0.4, 0.08, TOWER_MID),
            joint(hub, 0.12, TOWER_LIGHT),
        )

        radome_mount = disk(
            radome_center + DOWN * (radome_radius * 0.98),
            0.46,
            0.1,
            TOWER_MID,
            stroke_amount=0.24,
        )

        self.radome = Sphere(
            center=radome_center,
            radius=radome_radius,
            resolution=(24, 42),
            fill_color=RADOME_COLOR,
            fill_opacity=1,
            checkerboard_colors=False,
            stroke_color=shaded(RADOME_SHADE, 0.12),
            stroke_width=0.35,
        ).set_shade_in_3d(True)

        self.add(
            feet,
            legs,
            diagonals,
            rings,
            top_supports,
            mast,
            radome_mount,
            nodes,
            self.radome,
        )
        self.rotate(PI / 2, axis=RIGHT, about_point=ORIGIN)


class RadarTowerSceneBase(ThreeDScene):
    camera_phi = 58 * DEGREES
    camera_theta = -128 * DEGREES
    camera_gamma = 0
    camera_zoom = 0.96
    frame_center = (0.0, 0.0, 3.1)
    tower_shift = ORIGIN

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
    camera_zoom = 0.92


class WeatherRadar3DFrontCheck(RadarTowerSceneBase):
    camera_theta = -90 * DEGREES
    camera_zoom = 0.94


class WeatherRadar3DSideCheck(RadarTowerSceneBase):
    camera_theta = -22 * DEGREES
    camera_zoom = 0.94


class WeatherRadar3DTopCheck(RadarTowerSceneBase):
    camera_phi = 24 * DEGREES
    camera_theta = -128 * DEGREES
    camera_zoom = 0.9
    frame_center = (0.0, 0.0, 3.25)


class CenteredRadarTowerSceneBase(RadarTowerSceneBase):
    camera_phi = 60 * DEGREES
    camera_theta = -90 * DEGREES
    camera_zoom = 0.86
    frame_center = (0.0, 0.0, 3.1)
    tower_shift = ORIGIN


class DroneRadarTowerSceneBase(CenteredRadarTowerSceneBase):
    camera_phi = 68 * DEGREES
    camera_theta = -120 * DEGREES
    camera_zoom = 0.88
    frame_center = (0.0, 0.0, 3.2)


class WeatherRadar3DSpin(DroneRadarTowerSceneBase):
    camera_phi = 68 * DEGREES
    spin_duration = 6
    camera_gamma = 0

    def construct(self):
        self.setup_tower_scene()
        self.begin_ambient_camera_rotation(
            rate=-TAU / self.spin_duration, about="theta"
        )
        self.wait(self.spin_duration)


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
