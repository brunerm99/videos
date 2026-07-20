# zgl.py

import os
import sys
from random import seed, shuffle

import numpy as np
from dotenv import load_dotenv
from manimlib import *

sys.path.insert(0, "..")
load_dotenv()

from melp import VT, get_nexrad_product_ppi_data
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
TOWER_LIGHT = "#DCE6EB"
TOWER_MID = "#A9BBC4"
BEAM_COLOR = BLUE
GATE_COLOR = TEAL
ACCENT_COLOR = YELLOW
FONT = os.getenv("FONT", "Maple Mono CN")


class Intro(InteractiveScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "fps": 60,
    }

    def refresh_terrain(self):
        self.terrain.set_profile(self.terrain_profile)
        for radar, (site_x, site_y) in zip(self.radars, self.terrain_sites):
            base_z = radar.get_bounding_box()[0, 2]
            target_z = self.terrain.height_at(site_x, site_y) + 0.015
            radar.shift((target_z - base_z) * OUT)
        return self

    def construct(self):
        self.terrain_profile = {
            "broad_ridge": 1.25,
            "east_hill": 0.95,
            "valley_depth": 0.52,
            "rolling_ground": 0.25,
            "fine_relief": 0.12,
            "minor_ridges": 0.10,
        }
        self.terrain_sites = (
            (-4.6, -1.55),
            (4.15, 1.45),
            (8, -9),
        )
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
        self.third_radar = place_wsr88_on_terrain(
            self.terrain_sites[2],
            self.terrain,
            rotation=0 * DEG,
        )

        self.west_radar.az @= 180 * DEG
        self.west_radar.el @= 0
        self.radars = (self.west_radar, self.east_radar)

        self.frame.reorient(37, 75, 0, center=(0, 0, 1.6), height=17.4)

        self.wait(0.5)

        self.play(
            ShowCreation(self.terrain),
            self.frame.animate(run_time=4).reorient(
                -38,
                66,
                0,
                center=self.west_radar.base_section_6.get_center(),
                height=4.4,
            ),
        )

        self.wait(0.5)

        tower_parts = [
            *self.west_radar.base_sections[::-1],
            self.west_radar.fixed_body,
            self.west_radar.antenna,
        ]
        for part in tower_parts:
            part.set_opacity(0)

        self.west_radar.dome.set_opacity(0)
        self.west_radar.dome.deactivate_depth_test()
        self.add(self.west_radar)

        radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in tower_parts],
            lag_ratio=0.3,
            run_time=5,
        )
        radar_build.group = self.west_radar
        radar_build.mobject = self.west_radar

        self.play(
            radar_build,
            self.frame.animate(run_time=5).reorient(
                -38,
                76,
                0,
                center=self.west_radar.antenna.get_center(),
                height=4.4,
            ),
        )

        self.wait(0.5)

        self.cloud = Cloud(width=2.4)
        self.cloud.move_to(
            self.west_radar.antenna.get_center() + 2.4 * RIGHT + 1.15 * OUT
        )
        az, el = self.west_radar.get_scan_angles_to_point(self.cloud.get_center())
        self.play(
            ShowCreation(self.cloud),
            self.west_radar.az @ az,
            self.west_radar.el @ el,
            self.frame.animate.reorient(
                -38,
                86,
                0,
                center=Group(self.west_radar.antenna, self.cloud).get_center(),
                height=4.4,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.west_radar.add_updater(lambda rad: rad.point_at(self.cloud.get_center()))

        self.play(self.cloud.animate.shift(DOWN).rotate(-PI / 8), run_time=3)

        self.wait(0.5)

        z_data = get_nexrad_product_ppi_data(
            "reflectivity",
            s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
            sweep="lowest",
            max_range_km=150,
            resolution=600,
        )
        ppi = PPIGrid(
            z_data,
            colorbar_label=z_data["units"],
            font=FONT,
            init_scan_progress=0,
        )

        self.frame.save_state()
        self.play(
            Succession(
                FadeIn(ppi),
                ppi.scan_progress.animate(run_time=2).set_value(360),
            ),
            self.frame.animate(run_time=3).reorient(
                0, 0, center=ppi.get_center(), height=5.5
            ),
        )

        self.wait(0.5)

        self.play(FadeOut(ppi), self.frame.animate(run_time=3).restore())

        self.wait(0.5)

        self.west_beam = self.west_radar.get_beam(
            beamwidth=8 * DEG,
            range=11,
            color=BLUE,
            opacity=0.3,
        )
        self.play(ShowCreation(self.west_beam), run_time=3)

        # self.cloud.bind_beam_highlight(
        #     self.west_beam,
        #     outside_opacity=0.12,
        #     feather=0.04,
        # )
        # self.play(self.cloud.beam_highlight @ 1, run_time=1.25)

        self.wait(0.5)

        east_tower_parts = [
            *self.east_radar.base_sections[::-1],
            self.east_radar.fixed_body,
            self.east_radar.antenna,
        ]
        for part in east_tower_parts:
            part.set_opacity(0)
        self.east_radar.dome.set_opacity(0)
        self.east_radar.dome.deactivate_depth_test()

        self.add(self.east_radar)

        east_radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in east_tower_parts],
            lag_ratio=0.3,
            run_time=5,
        )
        east_radar_build.group = self.west_radar
        east_radar_build.mobject = self.west_radar

        east_az, east_el = self.east_radar.get_scan_angles_to_point(
            self.cloud.get_center()
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    east_radar_build,
                    self.frame.animate(run_time=5).reorient(
                        -8,
                        86,
                        0,
                        center=Group(
                            self.west_radar.antenna,
                            self.east_radar.antenna,
                            self.cloud,
                        ).get_center(),
                        height=7.4,
                    ),
                ),
                AnimationGroup(
                    self.east_radar.az @ east_az,
                    self.east_radar.el @ east_el,
                    run_time=2,
                ),
                lag_ratio=0.75,
            )
        )

        self.wait(0.5)

        self.east_beam = self.east_radar.get_beam(
            beamwidth=8 * DEG,
            range=11,
            color=BLUE,
            opacity=0.3,
        )
        self.play(ShowCreation(self.east_beam), run_time=3)

        self.wait(0.5)

        third_tower_parts = [
            *self.third_radar.base_sections[::-1],
            self.third_radar.fixed_body,
            self.third_radar.antenna,
        ]
        for part in third_tower_parts:
            part.set_opacity(0)
        self.third_radar.dome.set_opacity(0)
        self.third_radar.dome.deactivate_depth_test()

        self.add(self.third_radar)

        third_radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in third_tower_parts],
            lag_ratio=0.3,
            run_time=3,
        )
        third_radar_build.group = self.west_radar
        third_radar_build.mobject = self.west_radar

        self.cloud_2 = Cloud(width=2.4)
        self.cloud_2.move_to(
            self.third_radar.antenna.get_center() + UP * 4 + RIGHT * -3 + OUT
        )
        third_az, third_el = self.third_radar.get_scan_angles_to_point(
            self.cloud_2.get_center()
        )

        self.play(
            LaggedStart(
                self.frame.animate(run_time=4).reorient(
                    -20,
                    76,
                    0,
                    center=Group(
                        self.west_radar.antenna,
                        self.east_radar.antenna,
                        self.third_radar.antenna,
                    ).get_center(),
                    height=13.4,
                ),
                third_radar_build,
                ShowCreation(self.cloud_2),
                self.third_radar.az @ third_az,
                self.third_radar.el @ third_el,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.third_beam = self.third_radar.get_beam(
            beamwidth=8 * DEG,
            range=11,
            color=BLUE,
            opacity=0.3,
        )

        self.play(ShowCreation(self.third_beam), run_time=3)

        self.wait(0.5)

        self.play(self.frame.animate(run_time=3).shift(OUT * 30))

        self.wait(2)

        self.embed()


class NexradPPI(InteractiveScene):
    def construct(self):
        data = get_nexrad_product_ppi_data(
            "reflectivity",
            s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
            sweep="lowest",
            max_range_km=150,
            resolution=600,
        )
        ppi = PPIGrid(
            data,
            colorbar_label=data["units"],
            font="Maple Mono CN",
            init_scan_progress=0,
        )
        self.add(ppi)
        self.play(ppi.scan_progress @ 360, run_time=4)

        self.embed()


class BeamVolumeParticles(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        self.frame.reorient(-28, 72, 0, center=ORIGIN, height=7.4)

        beam_range = 10.2
        beam = Beam(
            beamwidth=12 * DEG,
            range=beam_range,
            origin=5.1 * LEFT,
            direction=RIGHT,
            color=BEAM_COLOR,
            opacity=0.09,
        )

        title = Text("radar beam", font=FONT, font_size=34)
        title.fix_in_frame().to_edge(UP, buff=0.32).set_z_index(100)

        self.play(ShowCreation(beam), FadeIn(title), run_time=2.0)
        self.wait(0.5)

        origin = beam.get_origin()
        direction = beam.get_direction()
        slope = beam.get_base_radius() / beam.get_range()

        radial_1 = np.cross(OUT, direction)
        radial_1 /= np.linalg.norm(radial_1)
        radial_2 = np.cross(direction, radial_1)
        radial_2 /= np.linalg.norm(radial_2)

        bin_start = 0.52 * beam.get_range()
        bin_end = 0.70 * beam.get_range()

        def bin_point(distance, angle, radial_fraction=1.0):
            radius = radial_fraction * distance * slope
            return (
                origin
                + distance * direction
                + radius * np.cos(angle) * radial_1
                + radius * np.sin(angle) * radial_2
            )

        bin_shell = beam.get_section(
            bin_start,
            bin_end,
            color=GATE_COLOR,
            opacity=0.18,
            shading=(0.08, 0.18, 0.12),
            depth_test=False,
            resolution=(36, 8),
        )
        bin_rings = VGroup(
            *[
                ParametricCurve(
                    lambda angle, distance=distance: bin_point(distance, angle),
                    t_range=(0, TAU, TAU / 48),
                    stroke_color=GATE_COLOR,
                    stroke_width=2.5,
                    stroke_opacity=0.95,
                ).apply_depth_test()
                for distance in (bin_start, bin_end)
            ]
        )
        bin_connectors = VGroup(
            *[
                Line(
                    bin_point(bin_start, angle),
                    bin_point(bin_end, angle),
                    stroke_color=GATE_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.8,
                ).apply_depth_test()
                for angle in np.arange(0, TAU, TAU / 4)
            ]
        )
        volume_bin = Group(bin_shell, bin_rings, bin_connectors)

        bin_title = VGroup(
            Text("one volume bin", font=FONT, font_size=34),
            Tex(r"\Delta R", font_size=46, color=GATE_COLOR),
        ).arrange(DOWN, buff=0.08)
        bin_title.fix_in_frame().to_edge(UP, buff=0.28).set_z_index(100)

        bin_center = origin + 0.5 * (bin_start + bin_end) * direction
        self.play(
            FadeIn(volume_bin),
            FadeOut(title),
            FadeIn(bin_title),
            self.frame.animate.reorient(
                -32,
                68,
                0,
                center=bin_center,
                height=4.6,
            ),
            run_time=2.2,
        )
        self.wait(0.5)

        rng = np.random.default_rng(8)
        particle_count = 180
        distances = np.cbrt(rng.uniform(bin_start**3, bin_end**3, particle_count))
        angles = rng.uniform(0, TAU, particle_count)
        radial_fractions = np.sqrt(rng.uniform(0, 0.84**2, particle_count))
        particle_points = np.array(
            [
                bin_point(distance, angle, radial_fraction)
                for distance, angle, radial_fraction in zip(
                    distances,
                    angles,
                    radial_fractions,
                )
            ]
        )
        if not np.all(beam.contains_points(particle_points)):
            raise RuntimeError("Generated a particle outside the beam")

        particle_radii = np.concatenate(
            [
                rng.uniform(0.022, 0.038, particle_count // 3),
                rng.uniform(0.042, 0.068, particle_count // 3),
                rng.uniform(0.075, 0.115, particle_count - 2 * (particle_count // 3)),
            ]
        )
        rng.shuffle(particle_radii)

        particle_clouds = Group()
        particle_animations = []
        for indices in np.array_split(rng.permutation(particle_count), 5):
            cloud = DotCloud(
                particle_points[indices],
                radius=0.04,
                color=BLUE_B,
                glow_factor=0.3,
            )
            cloud.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)
            cloud.set_radii(np.zeros(len(indices)))
            particle_clouds.add(cloud)
            particle_animations.append(cloud.animate.set_radii(particle_radii[indices]))

        particle_label = VGroup(
            Text("many particles", font=FONT, font_size=31),
            Text(
                "different sizes (not to scale)",
                font=FONT,
                font_size=22,
                color=BLUE_A,
            ),
        ).arrange(DOWN, buff=0.08)
        particle_label.fix_in_frame().to_edge(DOWN, buff=0.3).set_z_index(100)

        self.add(particle_clouds)
        self.play(
            LaggedStart(*particle_animations, lag_ratio=0.18, run_time=2.6),
            FadeIn(particle_label),
        )
        self.wait(0.5)

        summary = Text(
            "many particles  →  one measurement",
            font=FONT,
            font_size=29,
        )
        summary.fix_in_frame().to_edge(DOWN, buff=0.3).set_z_index(100)
        self.play(
            FadeOut(particle_label),
            FadeIn(summary),
            self.frame.animate.reorient(
                -42,
                64,
                0,
                center=bin_center,
                height=4.6,
            ),
            run_time=2.0,
        )
        self.wait(1.5)


class BeamSectionParticlesTest(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR}

    def construct(self):
        beam = Beam(
            beamwidth=12 * DEG,
            range=10.2,
            origin=5.1 * LEFT,
            direction=RIGHT,
            color=BEAM_COLOR,
            opacity=0.09,
        )
        section_start = 0.52 * beam.get_range()
        section_stop = 0.70 * beam.get_range()
        section = beam.get_section(
            section_start,
            section_stop,
            color=GATE_COLOR,
            opacity=0.18,
            shading=(0.08, 0.18, 0.12),
            depth_test=False,
            resolution=(36, 8),
        )
        particle_count = 180
        particles = beam.get_section_particles(
            section_start,
            section_stop,
            count=particle_count,
            max_radius_fraction=0.84,
            seed=8,
            radius=0.04,
            color=BLUE_B,
            glow_factor=0.3,
        )
        particles.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)

        rng = np.random.default_rng(9)
        particles_per_size = particle_count // 3
        particle_radii = np.concatenate(
            [
                rng.uniform(0.022, 0.038, particles_per_size),
                rng.uniform(0.042, 0.068, particles_per_size),
                rng.uniform(
                    0.075,
                    0.115,
                    particle_count - 2 * particles_per_size,
                ),
            ]
        )
        rng.shuffle(particle_radii)
        particles.set_radii(np.zeros(len(particle_radii)))

        section_center = (
            beam.get_origin()
            + 0.5 * (section_start + section_stop) * beam.get_direction()
        )
        self.frame.reorient(-32, 68, 0, center=section_center, height=4.6)
        self.add(beam)
        self.play(FadeIn(section), run_time=1.2)
        self.add(particles)
        self.play(particles.animate.set_radii(particle_radii), run_time=2)
        self.wait(0.5)


class ResolutionVolumeP2(InteractiveScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "fps": 60,
    }

    def refresh_terrain(self):
        self.terrain.set_profile(self.terrain_profile)
        for radar, (site_x, site_y) in zip(self.radars, self.terrain_sites):
            base_z = radar.get_bounding_box()[0, 2]
            target_z = self.terrain.height_at(site_x, site_y) + 0.015
            radar.shift((target_z - base_z) * OUT)
        return self

    def construct(self):
        self.terrain_profile = {
            "broad_ridge": 1.25,
            "east_hill": 0.95,
            "valley_depth": 0.52,
            "rolling_ground": 0.4,
            "fine_relief": 0.12,
            "minor_ridges": 0.10,
        }
        self.terrain_sites = (
            (-4.6, -1.55),
            (4.15, 1.45),
            (8, -9),
        )
        self.terrain = Terrain(
            size=(40, 28),
            profile=self.terrain_profile,
            level_sites=self.terrain_sites,
        )
        west_radar_rotation = 22 * DEG
        self.west_radar = place_wsr88_on_terrain(
            self.terrain_sites[0],
            self.terrain,
            rotation=west_radar_rotation,
        )

        self.add(self.west_radar, self.terrain)

        self.frame.reorient(
            40, 85, 0, center=self.west_radar.antenna.get_center(), height=5.4
        ).shift(OUT * 10)

        self.play(self.frame.animate(run_time=2).shift(IN * 10))

        self.wait(0.5)

        panel_indices = list(np.arange(len(self.west_radar.upper_dome)))
        seed(0)
        shuffle(panel_indices)

        panel_fade = LaggedStart(
            *[
                self.west_radar.upper_dome[idx].animate.set_opacity(0)
                for idx in panel_indices
            ],
            lag_ratio=0.01,
        )

        self.play(
            LaggedStart(
                panel_fade,
                self.west_radar.el @ (20 * DEG),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.west_radar.upper_dome.deactivate_depth_test()

        bw_deg = VT(8)
        beam_range = VT(0.1)
        beam_opacity = VT(0)
        beam = always_redraw(
            lambda: self.west_radar.get_beam(
                ~bw_deg * DEG, ~beam_range, opacity=~beam_opacity
            )
        )
        self.add(beam)

        self.play(
            AnimationGroup(
                beam_range @ 8,
                beam_opacity.animate(run_time=0.3).set_value(1),
            ),
            self.frame.animate.reorient(
                25,
                80,
                0,
                center=Group(self.west_radar.antenna).get_center(),
                height=2.4,
            ).shift(RIGHT),
        )

        self.wait(0.5)

        arc = beam.get_beamwidth_arc(
            ~beam_range / 4,
            self.frame,
            color=WHITE,
            stroke_width=10,
        )

        def position_beamwidth_arc_label(label):
            return label.next_to(
                self.frame.to_fixed_frame_point(arc.point_from_proportion(0.5)),
                RIGHT * np.cos(~self.west_radar.el) + OUT * np.sin(~self.west_radar.el),
                buff=SMALL_BUFF,
            )

        arc_label = Tex(r"\theta").scale(1).fix_in_frame()
        position_beamwidth_arc_label(arc_label)

        self.play(
            LaggedStart(
                ShowCreation(arc),
                FadeIn(arc_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(arc), FadeOut(arc_label))

        self.wait(0.5)

        gate_opacity = VT(0)
        rres = 0.7
        gate_start = VT(2)
        gate_out_shift = VT(0)
        gate = always_redraw(
            lambda: beam.get_section(
                ~gate_start,
                ~gate_start + rres,
                color=GATE_COLOR,
                opacity=~gate_opacity,
                shading=(0.08, 0.18, 0.12),
                depth_test=False,
                resolution=(36, 8),
            ).shift(OUT * ~gate_out_shift)
        )

        self.add(gate)
        self.play(
            self.frame.animate(run_time=3).reorient(
                60,
                80,
                0,
                center=Group(self.west_radar.antenna, gate).get_center(),
                height=3.4,
            ),
            LaggedStart(
                beam_opacity @ 0.1,
                gate_opacity @ 0.5,
                lag_ratio=0.4,
                run_time=3,
            ),
        )

        self.wait(0.5)

        self.play(gate_start @ 0.3, run_time=3)

        self.wait(0.5)

        self.play(
            gate_start @ 6,
            self.frame.animate.reorient(
                70,
                80,
                0,
                center=Group(self.west_radar.antenna, beam).get_center(),
                height=4,
            ).shift(RIGHT),
            run_time=3,
        )

        self.wait(0.5)

        particle_count = 30
        particles = beam.get_section_particles(
            ~gate_start,
            ~gate_start + rres,
            count=particle_count,
            max_radius_fraction=0.84,
            seed=8,
            radius=0.04,
            color=BLUE_B,
            glow_factor=0.3,
        )

        particles.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)

        rng = np.random.default_rng(10)
        particles_per_size = particle_count // 3
        particle_radii = np.concatenate(
            [
                rng.uniform(0.022, 0.038, particles_per_size),
                rng.uniform(0.042, 0.068, particles_per_size),
                rng.uniform(
                    0.075,
                    0.115,
                    particle_count - 2 * particles_per_size,
                ),
            ]
        )
        rng.shuffle(particle_radii)
        particles.set_radii(np.zeros(len(particle_radii)))

        self.add(beam)
        self.add(particles)
        self.play(
            beam_range @ 12,
            self.frame.animate.reorient(45, 85, center=gate.get_center(), height=2),
            particles.animate.set_radii(particle_radii),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            AnimationGroup(
                gate_out_shift @ 10,
                particles.animate.shift(OUT * 10),
            ),
            self.frame.animate.reorient(
                0, 90, center=gate.copy().shift(OUT * 10).get_center(), height=5
            ),
            run_time=2,
        )

        self.wait(0.5)

        dbz_value = (
            Tex(r"\rightarrow 31.2 \text{ dBZ}")
            .scale(1.2)
            .fix_in_frame()
            .next_to(
                self.frame.to_fixed_frame_point(gate.get_center()),
                RIGHT,
                LARGE_BUFF,
            )
        )
        gate_screen = self.frame.to_fixed_frame_point(gate.get_center())
        composition_center = 0.5 * (gate_screen + dbz_value.get_center())
        composition_center[2] = 0

        camera_shift = self.frame.from_fixed_frame_point(
            composition_center,
            relative=True,
        )

        self.play(self.frame.animate.shift(camera_shift))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    FadeIn(m)
                    for m in dbz_value.next_to(
                        self.frame.to_fixed_frame_point(gate.get_center()),
                        RIGHT,
                        LARGE_BUFF,
                    )
                ]
            )
        )

        self.wait(0.5)

        res_thumbnail = ImageMobject(
            "../09_resolution/static/Resolution Thumbnail.png"
        ).fix_in_frame()
        thumbnail_box = SurroundingRectangle(res_thumbnail, buff=0, color=YELLOW)
        thumbnail_label = (
            Group(
                Text(
                    "How Radars Tell Targets Apart",
                    font=FONT,
                ).fix_in_frame(),
                Text(
                    "(and When They Can’t)",
                    font=FONT,
                ).fix_in_frame(),
            )
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(thumbnail_box, DOWN)
        )

        thumbnail = (
            Group(res_thumbnail, thumbnail_box, thumbnail_label)
            .fix_in_frame()
            .shift(UP / 4)
        )

        self.play(GrowFromCenter(thumbnail))

        self.wait(0.5)

        self.play(FadeOut(thumbnail))

        self.wait(0.5)

        self.play(FadeOut(dbz_value), self.frame.animate.shift(OUT * 20))

        self.embed()
