# zgl.py

import os
import sys
from random import random, seed, shuffle

import numpy as np
from dotenv import load_dotenv
from manimlib import *

sys.path.insert(0, "..")
load_dotenv()

from melp import VT, fh, fw, get_nexrad_product_ppi_data, unit
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
        self.cloud.set_z_index(10)
        self.play(
            Succession(
                FadeIn(ppi),
                ppi.scan_progress.animate(run_time=2).set_value(360),
            ),
            self.frame.animate(run_time=3).reorient(
                0, 0, center=ppi.get_center(), height=5.5
            ),
        )
        self.cloud.set_z_index(0)

        self.wait(0.5)

        self.play(FadeOut(ppi), self.frame.animate(run_time=3).restore())

        self.wait(0.5)

        self.west_beam = self.west_radar.get_beam(
            beamwidth=8 * DEG,
            range=11,
            color=BLUE,
            opacity=0.3,
        )
        beams = [self.west_beam]
        self.play(ShowCreation(self.west_beam), run_time=3)

        # for beam in beams:
        #     beam.deactivate_depth_test()
        # self.bring_to_back(*beams)
        # self.bring_to_back(self.terrain)

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
        beams.append(self.east_beam)
        self.play(ShowCreation(self.east_beam), run_time=3)

        # for beam in beams:
        #     beam.deactivate_depth_test()
        # self.bring_to_back(*beams)
        # self.bring_to_back(self.terrain)

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
        beams.append(self.third_beam)

        self.play(ShowCreation(self.third_beam), run_time=3)

        # for beam in beams:
        #     beam.deactivate_depth_test()
        # self.bring_to_back(*beams)
        # self.bring_to_back(self.terrain)

        self.wait(0.5)

        zl = Text("Reflectivity", font=FONT).next_to(ppi, UP)
        zb = RoundedRectangle(
            width=zl.get_width() * 1.2,
            height=zl.get_height() * 1.2,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            stroke_opacity=0,
            corner_radius=0.2,
        ).move_to(zl)

        self.frame.save_state()
        self.play(
            LaggedStart(
                self.frame.animate(run_time=2.5)
                .reorient(0, 0, 0, center=ppi.get_center(), height=6)
                .shift(UP * 0.5),
                FadeIn(ppi),
                FadeIn(zb),
                Write(zl),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            self.frame.animate(run_time=2).restore(),
            FadeOut(ppi),
            FadeOut(zb),
            FadeOut(zl),
        )

        # self.play(self.frame.animate(run_time=3).shift(OUT * 30))

        self.wait(0.5)

        # self.frame.save_state()
        self.east_radar.feed_horn.save_state()
        self.east_radar["waveguides"].save_state()
        self.play(
            LaggedStart(
                self.frame.animate(run_time=3).reorient(
                    5, 85, 0, center=self.east_radar.antenna.get_center(), height=1
                ),
                self.east_radar.feed_horn.animate.set_color(ORANGE),
                self.east_radar["waveguides"].animate.set_color(ORANGE),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        radars = (self.west_radar, self.east_radar, self.third_radar)

        for beam in beams:
            beam.deactivate_depth_test()

        self.bring_to_back(self.terrain)
        self.bring_to_front(*beams)
        self.bring_to_front(*radars)
        self.bring_to_front(self.cloud, self.cloud_2)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=3).reorient(
                    -20,
                    100,
                    0,
                    center=Group(
                        self.west_radar.antenna,
                        self.east_radar.antenna,
                        self.third_radar.antenna,
                    ).get_center(),
                    height=12.4,
                ),
                self.west_beam.animate.set_opacity(0),
                self.east_beam.animate.set_opacity(0),
                self.third_beam.animate.set_opacity(0),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        west_screen_pos = self.frame.to_fixed_frame_point(
            self.west_radar.antenna.get_center()
        )
        west_radar_coords = (
            Text(
                "41.262778, -112.447770",
                font=FONT,
                fill_color=BLUE,
            )
            .scale(0.4)
            .fix_in_frame()
            .move_to(west_screen_pos + 0.35 * UP)
            .set_z_index(100)
        )
        east_screen_pos = self.frame.to_fixed_frame_point(
            self.east_radar.antenna.get_center()
        )
        east_radar_coords = (
            Text(
                "43.901892, -114.928347",
                font=FONT,
                fill_color=BLUE,
            )
            .scale(0.4)
            .fix_in_frame()
            .move_to(east_screen_pos + 0.35 * UP)
            .set_z_index(100)
        )
        third_screen_pos = self.frame.to_fixed_frame_point(
            self.third_radar.antenna.get_center()
        )
        third_radar_coords = (
            Text(
                "41.239847, -110.293874",
                font=FONT,
                fill_color=BLUE,
            )
            .scale(0.4)
            .fix_in_frame()
            .move_to(third_screen_pos + 0.35 * UP)
            .set_z_index(100)
        )

        self.play(
            LaggedStart(
                GrowFromCenter(
                    west_radar_coords,
                    rate_func=utils.rate_functions.overshoot,
                    run_time=0.5,
                ),
                GrowFromCenter(
                    third_radar_coords,
                    rate_func=utils.rate_functions.overshoot,
                    run_time=0.5,
                ),
                GrowFromCenter(
                    east_radar_coords,
                    rate_func=utils.rate_functions.overshoot,
                    run_time=0.5,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(west_radar_coords),
            FadeOut(east_radar_coords),
            FadeOut(third_radar_coords),
        )

        self.wait(0.5)

        self.play(
            self.frame.animate.reorient(
                10, 102, 0, center=self.cloud.get_center(), height=2.5
            ).shift(IN * 0.3)
        )

        seed(0)

        def get_drop_spec():
            return (
                random.random() * 2 - 1,
                random.random() * 0.8 - 0.4,
                random.random() * 0.5,
                random.random() * 0.2,
            )

        n_drops = 20
        cloud_bottom = self.cloud.get_bounding_box_point(IN)
        drop_specs = [get_drop_spec() for _ in range(n_drops)]
        self.raindrops = []
        for x_offset, y_offset, length, gap in drop_specs:
            drop = (
                Cylinder(
                    height=length,
                    radius=0.014,
                    axis=IN,
                    resolution=(24, 3),
                    color=BLUE,
                    shading=(0.15, 0.35, 0.2),
                )
                .rotate(-20 * DEG, axis=UP)
                .set_opacity(0.82)
            )
            drop.move_to(
                cloud_bottom
                + x_offset * RIGHT
                + y_offset * UP
                + (0.5 * length + gap) * IN
            )
            self.raindrops.append(drop)

        # self.add(*self.raindrops)
        self.play(
            LaggedStart(
                *[ShowCreation(m, run_time=0.4) for m in self.raindrops],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        flake_specs = [get_drop_spec() for _ in range(n_drops)]
        self.snowflakes = []
        for x_offset, y_offset, length, gap in flake_specs:
            drop = Sphere(
                radius=random.random() * 0.05 + 0.02,
                color=WHITE,
                resolution=(16, 8),
                shading=(0.1, 0.25, 0.15),
            ).set_opacity(0.82)
            drop.move_to(
                cloud_bottom
                + x_offset * RIGHT
                + y_offset * UP
                + (0.5 * length + gap) * IN
            )
            self.snowflakes.append(drop)

        self.play(
            LaggedStart(
                *[Uncreate(m, run_time=0.4) for m in self.raindrops],
                *[ShowCreation(m, run_time=0.4) for m in self.snowflakes],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(self.frame.animate.shift(OUT * 10))

        # self.wait(2)

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


class WeatherRadarEqn(InteractiveScene):
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
            # (-4.6, -1.55),
            (4.15, 1.45),
            # (8, -9),
        )
        self.terrain = Terrain(
            size=(40, 28),
            profile=self.terrain_profile,
            level_sites=self.terrain_sites,
        )
        radar_rotation = 22 * DEG
        self.radar = place_wsr88_on_terrain(
            self.terrain_sites[0],
            self.terrain,
            rotation=radar_rotation,
        )

        pr_to_z = Tex(r"P_r \rightarrow Z").rescale_to_fit(fw(self, 0.3), 0)
        pr_to_z[:2].set_color(RED)
        pr_to_z[3].set_color(BLUE)

        self.play(LaggedStart(*[FadeIn(m, shift=UP) for m in pr_to_z], lag_ratio=0.2))

        self.wait(0.5)

        pr_eqn1 = Tex(
            r"P_r = \frac{P_t G^2 c \tau \Omega \pi^5 |K|^2 Z}{2 (4 \pi)^3 L \lambda^2 R^2}"
        ).scale(2)
        pr_eqn1[:2].set_color(RX_COLOR)
        pr_eqn1[16].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_to_z[:2], pr_eqn1[:2]),
                FadeOut(pr_to_z[2]),
                ReplacementTransform(pr_to_z[3], pr_eqn1[16]),
                LaggedStart(
                    *[
                        GrowFromPoint(m, pr_eqn1.get_center())
                        for m in [*pr_eqn1[2:16], *pr_eqn1[17:]]
                    ],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        eqn_label = (
            Text("Weather Radar Equation", font=FONT)
            .rescale_to_fit(fw(self, 0.7), 0)
            .next_to(pr_eqn1, UP, fh(self, 1))
        )

        pr_eqn1.save_state()
        self.play(Group(eqn_label, pr_eqn1).animate.arrange(DOWN, buff=LARGE_BUFF * 1))

        self.wait(0.5)

        mug = ImageMobject("../15_psat/static/mug.png").rescale_to_fit(fh(self, 0.7), 0)

        self.play(mug.shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)))

        self.wait(0.5)

        self.play(mug.animate.shift(RIGHT * fw(self)))

        self.wait(0.5)

        self.play(eqn_label.animate.shift(UP * fh(self)), pr_eqn1.animate.restore())
        self.remove(mug, eqn_label)

        self.wait(0.5)

        tower_parts = [
            *self.radar.base_sections[::-1],
            self.radar.fixed_body,
            self.radar.antenna,
        ]
        for part in tower_parts:
            part.set_opacity(0)
        self.radar.dome.set_opacity(0)
        self.radar.dome.deactivate_depth_test()

        self.add(self.radar)

        radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in tower_parts],
            lag_ratio=0.3,
            run_time=3,
        )
        radar_build.group = self.radar
        radar_build.mobject = self.radar

        pr_eqn1.fix_in_frame()
        self.play(
            LaggedStart(
                LaggedStart(
                    self.frame.animate.reorient(
                        0, 85, 0, center=self.radar.antenna.get_center(), height=6
                    ),
                    pr_eqn1.animate.rescale_to_fit(fw(self, 0.55), 0).next_to(
                        self.frame.get_top(), DOWN, MED_SMALL_BUFF
                    ),
                    LaggedStart(radar_build),
                    ShowCreation(self.terrain),
                    lag_ratio=0.3,
                ),
                AnimationGroup(
                    self.radar.az @ (-60 * DEG),
                    self.radar.el @ (20 * DEG),
                    run_time=2,
                ),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        pr_eqn1p2 = Tex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} "
            r"\frac{c \tau \Omega \pi^5 |K|^2 Z}{2 \lambda^2 R^2}"
        )
        pr_eqn1p2.fix_in_frame().rescale_to_fit(fw(self, 0.7), 0).move_to(pr_eqn1)
        pr_eqn1p2[:2].set_color(RX_COLOR)
        pr_eqn1p2[23].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1[:3], pr_eqn1p2[:3]),
                ReplacementTransform(pr_eqn1[7:17], pr_eqn1p2[14:24]),
                ReplacementTransform(pr_eqn1[3:7], pr_eqn1p2[3:7]),
                ReplacementTransform(pr_eqn1[17], pr_eqn1p2[24]),
                ReplacementTransform(pr_eqn1[19:25], pr_eqn1p2[8:14], path_arc=-PI),
                ReplacementTransform(pr_eqn1[18], pr_eqn1p2[25]),
                GrowFromCenter(pr_eqn1p2[7]),
                ReplacementTransform(pr_eqn1[25:29], pr_eqn1p2[26:31]),
                lag_ratio=0.3,
            ),
            run_time=3.5,
        )

        self.wait(0.5)

        self.play(
            pr_eqn1p2[8:14].animate.set_color(ORANGE),
            pr_eqn1p2[3:8].animate.set_color(ORANGE),
        )

        self.wait(0.5)

        beam_range = VT(0.1)
        beam_opacity = VT(0)
        self.beam = always_redraw(
            lambda: self.radar.get_beam(
                8 * DEG,
                ~beam_range,
                opacity=~beam_opacity,
            ).deactivate_depth_test()
        )

        def get_beam_edges():
            origin = self.beam.get_origin()
            direction = self.beam.get_direction()
            camera_direction = self.frame.get_implied_camera_location() - origin
            screen_side = np.cross(camera_direction, direction)
            if np.linalg.norm(screen_side) < 1e-8:
                camera_right = self.frame.get_inverse_camera_rotation_matrix()[0]
                screen_side = camera_right - np.dot(camera_right, direction) * direction
            screen_side = unit(screen_side)
            base_center = self.beam.get_end()
            base_radius = self.beam.get_base_radius()
            stroke_opacity = np.clip(2.5 * ~beam_opacity, 0, 1)
            return VGroup(
                *[
                    Line(
                        origin,
                        base_center + sign * base_radius * screen_side,
                        stroke_color=BEAM_COLOR,
                        stroke_width=2,
                        stroke_opacity=stroke_opacity,
                    )
                    for sign in (-1, 1)
                ]
            ).deactivate_depth_test()

        beam_edges = always_redraw(get_beam_edges)

        self.add(self.beam)
        self.bring_to_front(self.radar)
        self.add(beam_edges)

        pt = pr_eqn1p2[3:5]
        pt.save_state()
        self.play(
            LaggedStart(
                pt.animate.set_color(YELLOW)
                .scale(1.2)
                .shift(UL * pt.get_height() * 0.2),
                self.frame.animate.reorient(
                    0, 85, 0, center=self.radar.antenna.get_center(), height=4
                ),
                beam_opacity @ 0.4,
                beam_range @ 8,
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(pt.animate.restore())

        self.wait(0.5)

        l = pr_eqn1p2[13]
        l.save_state()
        self.radar.waveguides.save_state()
        self.play(
            LaggedStart(
                l.animate.set_color(YELLOW).scale(1.2).shift(DR * l.get_height() * 0.2),
                self.frame.animate.reorient(
                    140, 85, 0, center=self.radar.waveguides.get_center(), height=1
                ).shift(OUT * 0.2),
                self.radar.waveguides.animate.set_color(RED),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(
            self.radar.waveguides.animate.restore(),
            self.frame.animate.reorient(
                0, 85, 0, center=self.radar.antenna.get_center(), height=4
            ),
            l.animate.restore(),
        )

        self.wait(0.5)

        g2 = pr_eqn1p2[5:7]
        g2.save_state()
        self.radar.reflector.save_state()
        self.play(
            LaggedStart(
                g2.animate.set_color(YELLOW).scale(1.2).shift(UR * 0.2),
                self.frame.animate.reorient(
                    10, 85, 0, center=self.radar.antenna.get_center(), height=2
                ).shift(OUT * 0.2),
                self.radar.reflector.animate.set_color(GREEN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.radar.reflector.animate.restore(),
            g2.animate.restore(),
            self.frame.animate.reorient(
                0, 85, 0, center=self.radar.antenna.get_center(), height=4
            ),
        )

        self.wait(0.5)

        self.play(
            pr_eqn1p2[8:13]
            .animate(
                rate_func=utils.rate_functions.there_and_back_with_pause,
                run_time=2,
            )
            .set_color(YELLOW)
            .shift(DL * 0.2),
        )

        self.wait(0.5)

        vol_start = VT(0)
        vol_end = VT(0.1)
        vol_opacity = VT(0)
        rres = VT(0.1)
        resvol_color_interp = VT(0)
        self.resvol = always_redraw(
            lambda: self.beam.get_section(
                ~vol_start,
                ~vol_start + ~rres,
                color=interpolate_color(GATE_COLOR, RED, ~resvol_color_interp),
                opacity=~vol_opacity,
                shading=(0.08, 0.18, 0.12),
                depth_test=False,
                resolution=(36, 8),
            )
        )
        self.add(self.resvol)
        self.bring_to_front(self.radar)
        self.bring_to_front(beam_edges)

        pr_eqn1p3 = Tex(
            r"P_r = \frac{P_t G^2}{(4 \pi)^3 L} "
            r"\frac{c \tau \Omega}{2} "
            r"\frac{\pi^5 |K|^2 Z}{\lambda^2 R^2}"
        )
        pr_eqn1p3.fix_in_frame()
        pr_eqn1p3.rescale_to_fit(fw(self, 1), 0).move_to(pr_eqn1p2)
        pr_eqn1p3[:2].set_color(RX_COLOR)
        pr_eqn1p3[25].set_color(BLUE)
        pr_eqn1p3[8:14].set_color(ORANGE)
        pr_eqn1p3[3:8].set_color(ORANGE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1p2[:3], pr_eqn1p3[:3]),
                ReplacementTransform(pr_eqn1p2[3:14], pr_eqn1p3[3:14]),
                ReplacementTransform(pr_eqn1p2[17:24], pr_eqn1p3[19:26]),
                ReplacementTransform(pr_eqn1p2[24], pr_eqn1p3[26]),
                ReplacementTransform(pr_eqn1p2[26:30], pr_eqn1p3[27:31]),
                ReplacementTransform(pr_eqn1p2[14:17], pr_eqn1p3[14:17]),
                GrowFromCenter(pr_eqn1p3[17]),
                ReplacementTransform(pr_eqn1p2[25], pr_eqn1p3[18]),
                lag_ratio=0.3,
                run_time=2,
            ),
        )

        self.wait(0.5)

        self.play(pr_eqn1p3[14:19].animate.set_color(RED))

        self.wait(0.5)

        self.play(
            LaggedStart(
                beam_opacity @ 0.1,
                vol_opacity @ 0.5,
                Succession(
                    rres @ 0.5,
                    vol_start.animate(run_time=3).set_value(6),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        tau = pr_eqn1p3[15]
        tau.save_state()
        self.play(tau.animate.set_color(YELLOW).scale(1.2).shift(UP * 0.2))

        self.wait(0.5)

        beam_point = self.radar.beam_location_to_point(
            range=4,
            az=0,
            el=0,
        )

        self.play(vol_start @ 1.5, self.frame.animate.shift(OUT * 0.5))

        self.wait(0.5)

        rres_line_start = self.radar.beam_location_to_point(
            range=1.5,
            az=0,
            el=6 * DEG,
        )
        rres_line_end = self.radar.beam_location_to_point(
            range=1.5 + ~rres,
            az=0,
            el=6 * DEG,
        )
        dot1 = (
            Sphere(
                radius=0.05,
                color=WHITE,
                resolution=(16, 8),
                shading=(0.1, 0.25, 0.15),
            )
            .set_opacity(1)
            .move_to(rres_line_start)
        )
        dot2 = (
            Sphere(
                radius=0.05,
                color=WHITE,
                resolution=(16, 8),
                shading=(0.1, 0.25, 0.15),
            )
            .set_opacity(1)
            .move_to(rres_line_end)
        )
        rres_line = Line(dot1.get_center(), dot2.get_center())
        rres_label = Tex(r"\Delta R").fix_in_frame()

        rres_label_opacity = VT(0)

        def position_rres_label(label):
            return (
                label.next_to(
                    self.frame.to_fixed_frame_point(rres_line.get_center()),
                    UP,
                    SMALL_BUFF * 1.5,
                )
                .shift(LEFT * SMALL_BUFF * 0.7)
                .set_opacity(~rres_label_opacity)
            )

        position_rres_label(rres_label)
        rres_label.add_updater(position_rres_label)
        self.add(rres_label)

        self.play(
            LaggedStart(
                ShowCreation(rres_line),
                rres_label_opacity @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rres_eqn = (
            Tex(r"\Delta R = \frac{c \tau}{2}").fix_in_frame().next_to(pr_eqn1p3, DOWN)
        )
        rres_eqn[4].set_color(RED)

        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(rres_label[:], rres_eqn[:2]),
                GrowFromCenter(rres_eqn[2]),
                GrowFromCenter(rres_eqn[3]),
                TransformFromCopy(tau, rres_eqn[4]),
                GrowFromCenter(rres_eqn[5]),
                GrowFromCenter(rres_eqn[6]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            tau.animate.restore().set_color(YELLOW),
            pr_eqn1p3[14].animate.set_color(YELLOW),
            pr_eqn1p3[17:19].animate.set_color(YELLOW),
        )

        self.wait(0.5)

        self.play(rres_eqn[3:].animate.set_color(YELLOW))

        self.wait(0.5)

        self.play(LaggedStart(*[ShrinkToCenter(m) for m in rres_eqn], lag_ratio=0.1))

        self.wait(0.5)

        self.play(
            self.frame.animate.reorient(
                30, 78, 0, center=self.resvol.get_center(), height=2
            ).shift(OUT * 0.1)
        )

        self.wait(0.5)

        theta_line = Line(
            self.radar.beam_location_to_point(~vol_start + ~rres, el=4 * DEG, az=0),
            self.radar.beam_location_to_point(~vol_start + ~rres, el=-4 * DEG, az=0),
        )
        phi_line = Line(
            self.radar.beam_location_to_point(~vol_start + ~rres, az=-4 * DEG, el=0),
            self.radar.beam_location_to_point(~vol_start + ~rres, az=4 * DEG, el=0),
        )

        theta_label = Tex(r"\angle \theta").fix_in_frame()
        phi_label = Tex(r"\angle \phi").fix_in_frame()

        theta_opacity = VT(0)
        phi_opacity = VT(0)

        def position_theta_label(label):
            return label.set_opacity(~theta_opacity).next_to(
                self.frame.to_fixed_frame_point(theta_line.get_end()),
                DOWN,
            )

        def position_phi_label(label):
            return label.set_opacity(~phi_opacity).next_to(
                self.frame.to_fixed_frame_point(phi_line.get_end()),
                RIGHT,
            )

        position_theta_label(theta_label)
        position_phi_label(phi_label)
        theta_label.add_updater(position_theta_label)
        phi_label.add_updater(position_phi_label)
        self.add(theta_label, phi_label)

        self.play(ShowCreation(theta_line), theta_opacity @ 1)

        self.wait(0.5)

        self.play(ShowCreation(phi_line), phi_opacity @ 1)

        self.wait(0.5)

        self.play(
            self.frame.animate(run_time=2)
            .reorient(0, 85, 0, center=self.radar.antenna.get_center(), height=4)
            .shift(OUT * 0.1),
            theta_opacity @ 0,
            phi_opacity @ 0,
            FadeOut(theta_line),
            FadeOut(phi_line),
            rres_label_opacity @ 0,
            FadeOut(rres_line),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    tau.animate.restore().set_color(RED),
                    pr_eqn1p3[14].animate.set_color(RED),
                    pr_eqn1p3[17:19].animate.set_color(RED),
                ),
                resvol_color_interp @ 1,
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(resvol_color_interp @ 0)

        self.wait(0.5)

        blue_section = pr_eqn1p3[19:]
        blue_section.save_state()
        self.play(blue_section.animate.set_color(BLUE))

        self.wait(0.5)

        self.cloud = Cloud(width=2.4).move_to(
            self.radar.beam_location_to_point(3, az=0, el=0)
        )

        self.play(
            self.frame.animate.reorient(
                -63,
                100,
                0,
                center=Group(self.radar.antenna, self.cloud).get_center(),
                height=4,
            ).shift(UP * 0.5 + OUT * 0.3),
            vol_start @ 3.1,
        )

        self.wait(0.5)

        particle_count = 20
        particles = self.beam.get_section_particles(
            ~vol_start,
            ~vol_start + ~rres,
            count=particle_count,
            max_radius_fraction=0.44,
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

        self.add(particles)
        self.play(particles.animate.set_radii(particle_radii), run_time=4)

        self.embed()

        # self.add(self.radar, self.terrain)


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
