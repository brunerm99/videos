from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np
from manimlib import *
from melp import VT, fh, fw, rotate_vector, unit
from melp.props import (
    Terrain,
    get_wsr88_model,
    place_wsr88_model,
    place_wsr88_on_terrain,
)

BACKGROUND_COLOR = "#183340"
TOWER_LIGHT = "#DCE6EB"
TOWER_MID = "#A9BBC4"
BEAM_COLOR = BLUE
GATE_COLOR = TEAL
ACCENT_COLOR = YELLOW
FONT = os.getenv("FONT", "Maple Mono CN")

PPI_PATH = Path(__file__).resolve().parent / "static" / "resolution_volume_ppi.png"
PIXEL_ZOOM_PATH = (
    Path(__file__).resolve().parent / "static" / "resolution_volume_pixel_zoom.png"
)

STOP_AFTER_STAGE = int(os.getenv("RESOLUTION_VOLUME_STAGE", "0"))
BIN_COLOR = "#F79F00"
RX_COLOR = RED


def get_beam_basis(axis):
    axis = unit(axis)
    horizontal = unit(np.cross(OUT, axis))
    vertical = unit(np.cross(axis, horizontal))
    return horizontal, vertical


def beam_point(origin, axis, distance, angle, half_angle):
    horizontal, vertical = get_beam_basis(axis)
    radius = distance * np.tan(half_angle)
    return (
        origin
        + distance * axis
        + radius * np.cos(angle) * horizontal
        + radius * np.sin(angle) * vertical
    )


def get_beam(origin, axis, max_range, half_angle):
    axis = unit(axis)
    surface = ParametricSurface(
        lambda distance, angle: beam_point(
            origin,
            axis,
            distance,
            angle,
            half_angle,
        ),
        u_range=(0.15, max_range),
        v_range=(0, TAU),
        resolution=(24, 36),
        color=BEAM_COLOR,
        opacity=0.075,
        shading=(0.05, 0.15, 0.1),
        depth_test=True,
    )

    edge_angles = (0, PI / 2, PI, 3 * PI / 2)
    edges = VGroup(
        *[
            Line(
                origin,
                beam_point(origin, axis, max_range, angle, half_angle),
                stroke_color=BEAM_COLOR,
                stroke_width=1.5,
                stroke_opacity=0.55,
            ).apply_depth_test()
            for angle in edge_angles
        ]
    )
    centerline = Line(
        origin,
        origin + max_range * axis,
        stroke_color=BEAM_COLOR,
        stroke_width=1,
        stroke_opacity=0.35,
    ).apply_depth_test()
    return Group(surface, edges, centerline)


def get_resolution_volume(
    origin,
    axis,
    range_start,
    gate_length,
    half_angle,
    color=GATE_COLOR,
    opacity=0.3,
    outline_opacity=0.9,
):
    axis = unit(axis)
    range_end = range_start + gate_length

    side = ParametricSurface(
        lambda distance, angle: beam_point(
            origin,
            axis,
            distance,
            angle,
            half_angle,
        ),
        u_range=(range_start, range_end),
        v_range=(0, TAU),
        resolution=(8, 36),
        color=color,
        opacity=opacity,
        shading=(0.1, 0.2, 0.15),
        depth_test=True,
    )

    def get_cap(distance):
        horizontal, vertical = get_beam_basis(axis)
        radius = distance * np.tan(half_angle)
        center = origin + distance * axis
        return ParametricSurface(
            lambda radial, angle: (
                center
                + radial * radius * np.cos(angle) * horizontal
                + radial * radius * np.sin(angle) * vertical
            ),
            u_range=(0, 1),
            v_range=(0, TAU),
            resolution=(5, 36),
            color=color,
            opacity=opacity * 1.25,
            shading=(0.1, 0.2, 0.15),
            depth_test=True,
        )

    rings = VGroup(
        *[
            ParametricCurve(
                lambda angle, distance=distance: beam_point(
                    origin,
                    axis,
                    distance,
                    angle,
                    half_angle,
                ),
                t_range=(0, TAU, TAU / 48),
                stroke_color=color,
                stroke_width=2.2,
                stroke_opacity=outline_opacity,
            ).apply_depth_test()
            for distance in (range_start, range_end)
        ]
    )

    connectors = VGroup(
        *[
            Line(
                beam_point(origin, axis, range_start, angle, half_angle),
                beam_point(origin, axis, range_end, angle, half_angle),
                stroke_color=color,
                stroke_width=1.5,
                stroke_opacity=outline_opacity,
            ).apply_depth_test()
            for angle in (0, PI / 2, PI, 3 * PI / 2)
        ]
    )
    return Group(side, get_cap(range_start), get_cap(range_end), rings, connectors)


def get_scatterer_points(
    origin,
    axis,
    range_start,
    gate_length,
    half_angle,
    count=160,
    seed=11,
):
    rng = np.random.default_rng(seed)
    axis = unit(axis)
    horizontal, vertical = get_beam_basis(axis)
    distances = rng.uniform(range_start, range_start + gate_length, count)
    radii = np.sqrt(rng.uniform(0, 0.82**2, count))
    angles = rng.uniform(0, TAU, count)
    beam_radii = distances * np.tan(half_angle)
    return (
        origin
        + distances[:, None] * axis
        + (radii * beam_radii * np.cos(angles))[:, None] * horizontal
        + (radii * beam_radii * np.sin(angles))[:, None] * vertical
    )


def get_ppi_composition():
    ppi = ImageMobject(str(PPI_PATH), height=6.35)
    ppi.fix_in_frame().shift(RIGHT * 2.05)

    zoom = ImageMobject(str(PIXEL_ZOOM_PATH), height=2.25)
    zoom.fix_in_frame().to_edge(LEFT, buff=0.45).shift(UP * 1.05)

    sample_box = Square(
        side_length=0.48,
        stroke_color=WHITE,
        stroke_width=2.5,
    ).move_to(ppi.get_center() + LEFT * 1.35 + UP * 1.5)
    sample_box.fix_in_frame()

    connectors = VGroup(
        Line(sample_box.get_corner(UL), zoom.get_corner(UR), stroke_width=1.4),
        Line(sample_box.get_corner(DL), zoom.get_corner(DR), stroke_width=1.4),
    ).fix_in_frame()

    label = Text(
        "one displayed pixel?",
        font=FONT,
        font_size=30,
        color=WHITE,
    ).next_to(zoom, DOWN, buff=0.22)
    label.fix_in_frame()
    return Group(ppi, zoom, sample_box, connectors, label)


class TwoRadarTerrain(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def refresh_terrain(self):
        self.terrain.set_profile(self.terrain_profile)
        for radar, (site_x, site_y) in zip(self.radars, self.terrain_sites):
            base_z = radar.get_bounding_box()[0, 2]
            target_z = self.terrain.height_at(site_x, site_y) + 0.015
            radar.shift((target_z - base_z) * OUT)
        return self

    def construct(self):
        self.frame.reorient(
            -38,
            66,
            0,
            center=(0, 0, 1.6),
            height=13.4,
        )

        self.terrain_sites = (
            (-4.6, -1.55),
            (4.15, 1.45),
        )
        self.terrain_profile = {
            "broad_ridge": 1.25,
            "east_hill": 0.95,
            "valley_depth": 0.52,
            "rolling_ground": 0.25,
            "fine_relief": 0.12,
            "minor_ridges": 0.10,
        }
        self.terrain = Terrain(
            size=(40, 28),
            profile=self.terrain_profile,
            level_sites=self.terrain_sites,
        )
        self.west_radar = place_wsr88_on_terrain(
            self.terrain_sites[0],
            self.terrain,
            rotation=22 * DEG,
        )
        self.east_radar = place_wsr88_on_terrain(
            self.terrain_sites[1],
            self.terrain,
            rotation=-18 * DEG,
        )
        self.radars = (self.west_radar, self.east_radar)

        self.play(ShowCreation(self.terrain), run_time=2.4)
        self.play(
            LaggedStart(
                FadeIn(self.west_radar),
                FadeIn(self.east_radar),
                lag_ratio=0.22,
                run_time=1.8,
            )
        )
        self.play(
            self.frame.animate.reorient(
                -42,
                64,
                0,
                center=(0, 0, 1.75),
                height=12.8,
            ),
            run_time=4.0,
        )
        self.wait(1.0)
        self.embed()


class RadarModelPlayground(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-35, 68, 0, center=ORIGIN, height=13.2)
        self.frame.make_orientation_default()

        radar = get_wsr88_model(height=5.4)
        radar.rotate(25 * DEG, axis=OUT)
        radar.move_to(ORIGIN)

        floor_z = radar.get_points()[:, 2].min() - 0.03
        grid = VGroup()
        for offset in np.arange(-3, 3.01, 0.5):
            is_major = np.isclose(offset % 1, 0)
            opacity = 0.16 if is_major else 0.07
            width = 1.0 if is_major else 0.6
            grid.add(
                Line(
                    [offset, -3, floor_z],
                    [offset, 3, floor_z],
                    stroke_color=TOWER_MID,
                    stroke_width=width,
                    stroke_opacity=opacity,
                ),
                Line(
                    [-3, offset, floor_z],
                    [3, offset, floor_z],
                    stroke_color=TOWER_MID,
                    stroke_width=width,
                    stroke_opacity=opacity,
                ),
            )
        grid.apply_depth_test()

        title = Text("WSR-88 playground", font=FONT, font_size=30)
        hints = VGroup(
            Text("hold d + move mouse   orbit", font=FONT, font_size=17),
            Text("hold f + move mouse   pan", font=FONT, font_size=17),
            Text("mouse drag   pan       scroll   zoom", font=FONT, font_size=17),
            Text("r   reset camera", font=FONT, font_size=17),
            Text(
                "s   select       g / h / v   move (experimental)",
                font=FONT,
                font_size=17,
            ),
        )
        hints.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        hints.set_color(TOWER_MID)

        controls = VGroup(title, hints)
        controls.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        controls.fix_in_frame().to_corner(UL, buff=0.3).set_z_index(100)

        self.radar = radar
        self.add(grid, radar, controls)
        self.disable_interaction(grid, controls)
        self.wait(0.1)


class ResolutionVolumeModelCheck(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-35, 68, 0, center=(0, 0, -0.35), height=14.5)

        radar = get_wsr88_model(height=5.4)
        radar.rotate(25 * DEG, axis=OUT)

        origin_marker = Sphere(
            radius=0.09,
            color=ACCENT_COLOR,
            resolution=(21, 11),
        ).move_to(radar.feed_horn_origin)

        title = Text("WSR-88 model check", font=FONT, font_size=34)
        title.fix_in_frame().to_corner(UL, buff=0.3)

        self.add(radar, origin_marker, title)
        self.wait(0.5)


class ResolutionVolumeDomeLift(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-35, 68, 0, center=(0, 0, 3.6), height=9.2)

        radar = get_wsr88_model(height=5.4)
        radar.rotate(25 * DEG, axis=OUT)

        title = Text("radome lift check", font=FONT, font_size=34)
        title.fix_in_frame().to_edge(UP)

        self.add(radar, title)
        self.wait(0.5)
        self.play(
            radar.dome.animate.shift(radar.dome_lift_vector),
            run_time=2.5,
        )
        self.wait(0.75)


class ResolutionVolumePartsCheck(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-35, 68, 0, center=(0, 0, 0.3), height=15.4)

        radar = get_wsr88_model(height=5.4)
        radar.rotate(25 * DEG, axis=OUT)
        for section in radar.base_sections:
            section.set_opacity(0)

        beam = always_redraw(
            lambda: Line(
                radar.beam_origin,
                radar.beam_origin + 1.8 * radar.beam_direction,
                stroke_color=BEAM_COLOR,
                stroke_width=5,
            ).apply_depth_test()
        )
        title = Text("indexable WSR-88 parts", font=FONT, font_size=34)
        title.fix_in_frame().to_corner(UL, buff=0.3)

        self.add(radar, beam, title)
        self.play(
            LaggedStart(
                *[section.animate.set_opacity(1) for section in radar.base_sections],
                lag_ratio=0.15,
                run_time=3.0,
            )
        )
        self.play(
            radar.dome.animate.shift(radar.dome_lift_vector),
            run_time=2.0,
        )
        self.play(
            radar.antenna.animate.rotate(
                18 * DEG,
                axis=UP,
                about_point=radar.antenna.get_center(),
            ),
            run_time=2.0,
        )
        self.play(
            FadeOut(radar.feed_horn),
            FadeOut(radar.base),
            run_time=1.0,
        )
        self.play(
            FadeIn(radar.feed_horn),
            FadeIn(radar.base),
            run_time=1.0,
        )
        self.wait(0.5)


class ResolutionVolumeRadomePanelsCheck(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-35, 68, 0, center=(0, 0, 3.6), height=7.4)

        radar = get_wsr88_model(height=5.4, indexed_radome_panels=True)
        radar.rotate(25 * DEG, axis=OUT)

        upper_panel = radar.dome.panels[108]
        lower_panel = radar.dome.panels[151]

        title = Text(
            "adjacent dome panels pulled out together",
            font=FONT,
            font_size=32,
        )
        title.fix_in_frame().to_edge(UP)

        self.add(radar, title)
        self.wait(0.5)
        self.play(
            upper_panel.animate.set_color(ACCENT_COLOR).shift(2.0 * RIGHT + 0.25 * OUT),
            lower_panel.animate.set_color(ACCENT_COLOR).shift(2.0 * RIGHT - 0.25 * OUT),
            run_time=2.0,
        )
        self.wait(1.0)


class ResolutionVolume(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}
    stop_after_stage = STOP_AFTER_STAGE

    def at_checkpoint(self, stage):
        return self.stop_after_stage == stage

    def construct(self):
        self.frame.reorient(0, 0, 0, center=ORIGIN, height=8)

        ppi_composition = get_ppi_composition()
        ppi, zoom, sample_box, connectors, pixel_label = ppi_composition

        self.play(FadeIn(ppi))
        self.play(
            LaggedStart(
                ShowCreation(sample_box),
                ShowCreation(connectors),
                FadeIn(zoom),
                FadeIn(pixel_label),
                lag_ratio=0.18,
                run_time=2.2,
            )
        )
        self.wait(0.5)
        if self.at_checkpoint(1):
            return

        radar, beam_origin = place_wsr88_model(
            height=3.3,
            shift=np.array([-4.05, 0.0, 0.0]),
            dome_raised=True,
        )
        beam_axis = unit(np.array([1.0, 0.0, 0.075]))
        half_angle = 6.5 * DEG
        beam = get_beam(
            beam_origin,
            beam_axis,
            max_range=8.8,
            half_angle=half_angle,
        )

        beam_label = VGroup(
            Text("finite beamwidth", font=FONT, font_size=31),
            Text("angular width exaggerated", font=FONT, font_size=19, color=GREY_B),
        ).arrange(DOWN, buff=0.08, aligned_edge=RIGHT)
        beam_label.fix_in_frame().to_corner(UR, buff=0.35)

        self.play(
            FadeOut(ppi_composition),
            self.frame.animate.reorient(
                -35,
                68,
                0,
                center=(0.15, 0.0, 1.8),
                height=7.4,
            ),
            FadeIn(radar),
            run_time=2.2,
        )
        self.remove(ppi, zoom, sample_box, connectors, pixel_label, ppi_composition)
        self.play(
            LaggedStart(
                FadeIn(beam),
                FadeIn(beam_label),
                lag_ratio=0.25,
                run_time=1.8,
            )
        )
        self.wait(0.5)
        if self.at_checkpoint(2):
            return

        gate_length = 1.1
        near_range = 2.15
        active_gate = get_resolution_volume(
            beam_origin,
            beam_axis,
            near_range,
            gate_length,
            half_angle,
        )

        gate_label = VGroup(
            Text("finite range gate", font=FONT, font_size=29),
            Tex(r"\Delta R", font_size=48, color=GATE_COLOR),
        ).arrange(DOWN, buff=0.08)
        gate_label.fix_in_frame().to_corner(DR, buff=0.4)

        self.play(
            LaggedStart(
                FadeIn(active_gate),
                FadeIn(gate_label),
                lag_ratio=0.28,
                run_time=1.6,
            )
        )
        self.wait(0.5)
        if self.at_checkpoint(3):
            return

        near_gate = get_resolution_volume(
            beam_origin,
            beam_axis,
            near_range,
            gate_length,
            half_angle,
            color=TEAL_D,
            opacity=0.075,
            outline_opacity=0.28,
        )
        self.add(near_gate)

        far_range = 6.45
        far_gate = get_resolution_volume(
            beam_origin,
            beam_axis,
            far_range,
            gate_length,
            half_angle,
        )
        comparison = VGroup(
            Text("same beamwidth", font=FONT, font_size=27),
            Tex(r"\text{same }\Delta R", font_size=38),
            Tex(r"V \propto R^2", font_size=52, color=ACCENT_COLOR),
        ).arrange(DOWN, buff=0.14, aligned_edge=RIGHT)
        comparison.fix_in_frame().to_corner(UR, buff=0.35)

        gate_transforms = [
            Transform(active_gate[index], far_gate[index]) for index in range(3)
        ]
        gate_transforms.extend(
            Transform(source, target)
            for source_group, target_group in zip(active_gate[3:], far_gate[3:])
            for source, target in zip(source_group, target_group)
        )

        self.play(
            FadeOut(beam_label),
            FadeIn(comparison),
            AnimationGroup(*gate_transforms),
            self.frame.animate.reorient(
                -27,
                66,
                0,
                center=(0.25, 0.0, 1.85),
                height=7.4,
            ),
            run_time=3.0,
        )
        self.wait(0.5)
        if self.at_checkpoint(4):
            return

        scatterer_points = get_scatterer_points(
            beam_origin,
            beam_axis,
            far_range,
            gate_length,
            half_angle,
        )
        scatterers = DotCloud(
            scatterer_points,
            radius=0.045,
            color=BLUE_B,
            glow_factor=0.35,
        )
        scatterers.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)
        scatterer_radii = np.linspace(0.028, 0.065, len(scatterer_points))
        np.random.default_rng(4).shuffle(scatterer_radii)
        scatterers.set_radii(np.zeros_like(scatterer_radii))
        self.add(scatterers)

        particle_label = (
            Text(
                "many scatterers inside one volume",
                font=FONT,
                font_size=29,
            )
            .fix_in_frame()
            .to_edge(DOWN, buff=0.34)
        )

        self.play(
            scatterers.animate.set_radii(scatterer_radii),
            FadeIn(particle_label),
            run_time=1.8,
        )

        return_lines = VGroup(
            *[
                Line(
                    point,
                    beam_origin,
                    stroke_color=RX_COLOR,
                    stroke_width=1,
                    stroke_opacity=0.38,
                ).apply_depth_test()
                for point in scatterer_points[::12]
            ]
        )
        self.play(
            LaggedStart(
                *[ShowCreation(line) for line in return_lines],
                lag_ratio=0.05,
                run_time=1.7,
            )
        )
        self.play(
            AnimationGroup(*[FadeOut(line) for line in return_lines]),
            run_time=0.5,
        )
        self.remove(*return_lines, return_lines)
        self.wait(0.5)
        if self.at_checkpoint(5):
            return

        volume_center = beam_origin + (far_range + gate_length / 2) * beam_axis
        sample_symbol = GlowDot(
            volume_center,
            radius=0.18,
            color=BIN_COLOR,
            glow_factor=1.4,
        )
        readout = VGroup(
            Text("one radar sample", font=FONT, font_size=31),
            Tex(r"32\,\mathrm{dBZ}", font_size=50, color=BIN_COLOR),
        ).arrange(DOWN, buff=0.1)
        readout.fix_in_frame().to_corner(DR, buff=0.38)

        self.play(
            FadeOut(gate_label),
            FadeOut(particle_label),
            scatterers.animate.set_radii(np.zeros_like(scatterer_radii)),
            FadeIn(readout),
            run_time=1.5,
        )
        self.remove(scatterers)
        self.add(sample_symbol)
        self.wait(0.5)
        if self.at_checkpoint(6):
            return

        final_ppi = ImageMobject(str(PPI_PATH), height=6.45)
        final_ppi.fix_in_frame().shift(LEFT * 2.55)
        display_pixel = Square(
            side_length=0.34,
            fill_color=BIN_COLOR,
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=2,
        ).move_to(final_ppi.get_center() + LEFT * 1.35 + UP * 1.52)
        display_pixel.fix_in_frame()

        summary = VGroup(
            Text("many scatterers", font=FONT, font_size=30),
            Tex(r"\Downarrow", font_size=44, color=GREY_A),
            Text("one radar sample", font=FONT, font_size=30),
            Tex(r"32\,\mathrm{dBZ}", font_size=54, color=BIN_COLOR),
        ).arrange(DOWN, buff=0.13)
        summary.fix_in_frame().to_edge(RIGHT, buff=0.65).set_z_index(102)

        final_ppi.set_z_index(101)
        display_pixel.set_z_index(102)

        connector = (
            Line(
                display_pixel.get_right(),
                summary.get_left(),
                stroke_color=WHITE,
                stroke_width=1.4,
                stroke_opacity=0.7,
            )
            .fix_in_frame()
            .set_z_index(102)
        )

        transition_cover = Rectangle(
            width=FRAME_WIDTH * 1.05,
            height=FRAME_HEIGHT * 1.05,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            stroke_width=0,
        )
        transition_cover.fix_in_frame().set_z_index(100)

        self.play(
            FadeIn(transition_cover),
            FadeOut(comparison),
            FadeOut(readout),
            self.frame.animate.reorient(0, 0, 0, center=ORIGIN, height=8),
            run_time=1.5,
        )
        self.remove(
            radar,
            beam,
            near_gate,
            active_gate,
            sample_symbol,
            transition_cover,
        )
        self.play(
            LaggedStart(
                FadeIn(final_ppi),
                FadeIn(display_pixel),
                ShowCreation(connector),
                FadeIn(summary),
                lag_ratio=0.22,
                run_time=1.8,
            )
        )
        self.wait(2)


class ResolutionVolumeStage1(ResolutionVolume):
    stop_after_stage = 1


class ResolutionVolumeStage2(ResolutionVolume):
    stop_after_stage = 2


class ResolutionVolumeStage3(ResolutionVolume):
    stop_after_stage = 3


class ResolutionVolumeStage4(ResolutionVolume):
    stop_after_stage = 4


class ResolutionVolumeStage5(ResolutionVolume):
    stop_after_stage = 5


class ResolutionVolumeStage6(ResolutionVolume):
    stop_after_stage = 6
