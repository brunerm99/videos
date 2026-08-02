# zgl.py
# manimgl is incredible

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


class Intro(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

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
            vmin=-20,
            vmax=70,
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
            vmin=-20,
            vmax=70,
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

        # yes, I know these are backwards. I'm too far in and too tired to do it the right way
        theta_label = Tex(r"\angle \phi").fix_in_frame()
        phi_label = Tex(r"\angle \theta").fix_in_frame()

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

        self.cloud.bind_beam_highlight(self.beam, outside_opacity=0.1, feather=0)

        cloud_vol_start = 2.6
        particle_count = 60
        particles = self.beam.get_section_particles(
            cloud_vol_start,
            cloud_vol_start + ~rres,
            count=particle_count,
            max_radius_fraction=0.8,
            seed=8,
            color=BLUE_B,
            glow_factor=0.3,
        )

        particles.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)

        rng = np.random.default_rng(10)
        particles_per_size = particle_count // 3
        particle_radii = (
            np.concatenate(
                [
                    rng.uniform(0.012, 0.028, particles_per_size),
                    rng.uniform(0.022, 0.048, particles_per_size),
                    rng.uniform(
                        0.055,
                        0.095,
                        particle_count - 2 * particles_per_size,
                    ),
                ]
            )
            * 0.4
        )
        rng.shuffle(particle_radii)
        particles.set_radii(np.zeros(len(particle_radii)))

        self.add(particles)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=2)
                .reorient(
                    -63,
                    100,
                    0,
                    center=Group(self.radar.antenna, self.cloud).get_center(),
                    height=4,
                )
                .shift(UP * 0.5 + OUT * 0.3),
                FadeIn(self.cloud, run_time=2),
                vol_start @ cloud_vol_start,
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.cloud.beam_highlight.animate(run_time=2).set_value(1),
                particles.animate(run_time=3).set_radii(particle_radii),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        orbit_theta = VT(-60)
        self.play(
            self.frame.animate(run_time=2)
            .reorient(~orbit_theta, 95, 0, center=self.cloud.get_center(), height=2.2)
            .shift(OUT * 0.1)
        )

        self.wait(0.5)

        kw = pr_eqn1p3[-10:-6]
        kw.save_state()
        self.play(kw.animate.set_color(YELLOW).shift(UP * 0.2))

        self.wait(0.5)

        def frame_orbiter(frame):
            frame.reorient(
                ~orbit_theta, 95, 0, center=self.cloud.get_center(), height=2.2
            ).shift(OUT * 0.1)

        self.frame.add_updater(frame_orbiter)

        water_label = (
            VGroup(Text("Water", font=FONT), Text("Droplets", font=FONT))
            .arrange(DOWN, SMALL_BUFF)
            .rescale_to_fit(fw(self, 0.6), 0)
        )
        water_box = RoundedRectangle(
            width=water_label.get_width() * 1.2,
            height=water_label.get_height() * 1.2,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            stroke_color=BLUE,
            stroke_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            corner_radius=0.2,
        ).move_to(water_label)
        water_callout = (
            Group(water_box, water_label)
            .fix_in_frame()
            .to_corner(DR, buff=MED_LARGE_BUFF)
            .set_z_index(100)
        )

        connector_opacity = VT(0)
        connector_top = always_redraw(
            lambda: (
                Line(
                    water_callout.get_left(),
                    self.frame.to_fixed_frame_point(
                        self.radar.beam_location_to_point(
                            ~vol_start + ~rres, az=0, el=-4 * DEG
                        )
                    ),
                    stroke_width=DEFAULT_STROKE_WIDTH * 3,
                    color=BLUE,
                )
                .fix_in_frame()
                .set_opacity(~connector_opacity)
            )
        )
        connector_bot = always_redraw(
            lambda: (
                Line(
                    water_callout.get_left(),
                    self.frame.to_fixed_frame_point(
                        self.radar.beam_location_to_point(~vol_start, az=0, el=-4 * DEG)
                    ),
                    stroke_width=DEFAULT_STROKE_WIDTH * 3,
                    color=BLUE,
                )
                .fix_in_frame()
                .set_opacity(~connector_opacity)
            )
        )
        self.add(connector_top, connector_bot)

        self.play(
            connector_opacity @ 1,
            FadeIn(water_callout),
        )

        self.wait(0.5)

        self.play(orbit_theta + 360, run_time=15)

        self.frame.remove_updater(frame_orbiter)

        self.wait(0.5)

        self.play(FadeOut(water_callout), connector_opacity @ 0, kw.animate.restore())

        self.wait(0.5)

        wl = pr_eqn1p3[-4:-2]
        wl.save_state()
        self.play(wl.animate.set_color(YELLOW).shift(DOWN * 0.2))

        self.wait(0.5)

        r = pr_eqn1p3[-2:]
        r.save_state()
        self.play(
            LaggedStart(
                wl.animate.restore(),
                r.animate.set_color(YELLOW).shift(DOWN * 0.2),
                self.frame.animate.reorient(
                    -30 + 360,
                    80,
                    0,
                    center=Group(self.radar.antenna, self.cloud).get_center(),
                    height=4,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        r_line = Arrow(
            self.frame.to_fixed_frame_point(
                self.radar.beam_location_to_point(0, az=0, el=0)
            ),
            self.frame.to_fixed_frame_point(
                self.radar.beam_location_to_point(~vol_start + ~rres / 2, az=0, el=0)
            ),
            color=YELLOW,
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
            fill_color=YELLOW,
            fill_opacity=1,
            buff=0,
        ).fix_in_frame()

        self.play(GrowArrow(r_line))

        self.wait(0.5)

        self.play(FadeOut(r_line), r.animate.restore())

        self.wait(0.5)

        z = pr_eqn1p3[-6]
        z.save_state()
        self.play(z.animate.set_color(YELLOW).scale(1.2).shift(UR * 0.2))

        self.wait(0.5)

        self.play(self.frame.animate.shift(OUT * 10))
        self.play(
            pr_eqn1p3.animate.move_to(
                self.frame.to_fixed_frame_point(self.frame.get_center())
            )
        )
        self.play(z.animate.set_color(BLUE).shift(DL * 0.2).scale(1 / 1.2))

        self.wait(0.5)

        z_eqn = (
            Tex(
                r"Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} "
                r"\frac{2}{c \tau \Omega} "
                r"\frac{\lambda^2 R^2}{\pi^5 |K|^2}"
            )
            .fix_in_frame()
            .rescale_to_fit(pr_eqn1p3.get_width(), 0)
            .move_to(pr_eqn1p3)
        )
        z_eqn[0].set_color(BLUE)
        z_eqn[2:4].set_color(RED)
        z_eqn[4:15].set_color(ORANGE)
        z_eqn[15:20].set_color(RED)
        z_eqn[20:].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(pr_eqn1p3[25], z_eqn[0], path_arc=PI / 2),
                ReplacementTransform(pr_eqn1p3[:2], z_eqn[2:4], path_arc=PI / 2),
                ReplacementTransform(pr_eqn1p3[2], z_eqn[1]),
                ReplacementTransform(pr_eqn1p3[3:7], z_eqn[11:15], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[8:14], z_eqn[4:10], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[7], z_eqn[10]),
                ReplacementTransform(pr_eqn1p3[14:17], z_eqn[17:20], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[18], z_eqn[15], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[17], z_eqn[16]),
                ReplacementTransform(pr_eqn1p3[19:25], z_eqn[25:31], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[27:31], z_eqn[20:24], path_arc=PI),
                ReplacementTransform(pr_eqn1p3[26], z_eqn[24]),
            )
        )

        self.wait(0.5)

        units = (
            Tex(r"\left[ \frac{\text{m}^6}{\text{m}^3} \right]")
            .fix_in_frame()
            .rescale_to_fit(z_eqn.get_height(), 1)
            .next_to(z_eqn[24], RIGHT, MED_SMALL_BUFF)
        )

        self.wait(0.5)

        self.play(LaggedStart(*[GrowFromCenter(m) for m in units], lag_ratio=0.1))

        self.wait(0.5)

        z_eqn_mm = (
            Tex(
                r"Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} "
                r"\frac{2}{c \tau \Omega} "
                r"\frac{\lambda^2 R^2}{\pi^5 |K|^2} 10^{18}"
                r"\ \left[ \frac{\text{mm}^6}{\text{m}^3} \right]"
            )
            .fix_in_frame()
            .rescale_to_fit(z_eqn.get_height(), 1)
            .move_to(z_eqn)
        )
        z_eqn_mm[0].set_color(BLUE)
        z_eqn_mm[2:4].set_color(RED)
        z_eqn_mm[4:15].set_color(ORANGE)
        z_eqn_mm[15:20].set_color(RED)
        z_eqn_mm[20:31].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(z_eqn, z_eqn_mm[:-12]),
                LaggedStart(*[GrowFromCenter(m) for m in z_eqn_mm[-12:-8]]),
                ReplacementTransform(units[0], z_eqn_mm[-8]),
                ReplacementTransform(units[1], z_eqn_mm[-7]),
                GrowFromCenter(z_eqn_mm[-6]),
                ReplacementTransform(units[2], z_eqn_mm[-5]),
                ReplacementTransform(units[3], z_eqn_mm[-4]),
                ReplacementTransform(units[4], z_eqn_mm[-3]),
                ReplacementTransform(units[5], z_eqn_mm[-2]),
                ReplacementTransform(units[6], z_eqn_mm[-1]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(z_eqn_mm))

        self.wait(2)

        self.embed()


class Dbz(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        video = VideoMobject("./dbz_callback.mp4", speed=1)
        box = SurroundingRectangle(video, buff=0)
        video_label = Text("earlier...", font=FONT).next_to(
            box, UP, SMALL_BUFF, aligned_edge=LEFT
        )
        callback = Group(video, box, video_label)

        self.play(callback.shift(DOWN * fh(self)).animate.shift(UP * fh(self)))

        self.wait(video.get_time_remaining() - 0.5)
        video.pause()

        self.wait(0.5)

        self.play(callback.animate.shift(UP * fh(self)))
        self.remove(callback)

        lin_units = Tex(r"\frac{\text{mm}^6}{\text{m}^3}").rescale_to_fit(
            fh(self, 0.3), 1
        )

        dbz_units = Tex(
            r"\text{dBZ} = 10 \log{\left(\frac{\text{mm}^6}{\text{m}^3} \right)}"
        ).rescale_to_fit(fh(self, 0.3), 1)

        self.play(LaggedStart(*[FadeIn(m) for m in lin_units], lag_ratio=0.1))

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        ReplacementTransform(a, b)
                        for a, b in zip(lin_units, dbz_units[10:16])
                    ],
                    lag_ratio=0.05,
                ),
                LaggedStart(
                    GrowFromCenter(dbz_units[0]),
                    GrowFromCenter(dbz_units[1]),
                    GrowFromCenter(dbz_units[2]),
                    GrowFromCenter(dbz_units[3]),
                    GrowFromCenter(dbz_units[4]),
                    GrowFromCenter(dbz_units[5]),
                    GrowFromCenter(dbz_units[6]),
                    GrowFromCenter(dbz_units[7]),
                    GrowFromCenter(dbz_units[8]),
                    GrowFromCenter(dbz_units[9]),
                    GrowFromCenter(dbz_units[16]),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

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
            vmin=-20,
            vmax=70,
            colorbar_label=z_data["units"],
            font=FONT,
            init_scan_progress=0,
        ).rescale_to_fit(fh(self, 0.8), 1)
        ppi.remove(ppi.colorbar_label)

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in dbz_units[3:]], lag_ratio=0.1),
                dbz_units[:3]
                .animate.rescale_to_fit(ppi.colorbar_label.get_height(), 1)
                .move_to(ppi.colorbar_label),
                FadeIn(ppi),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(ppi.scan_progress @ 360, run_time=3)

        self.wait(0.5)

        dbz_arrow = CurvedArrow(
            ppi.colorbar_ticks[0].get_right() + RIGHT * 0.1,
            ppi.colorbar_ticks[-1].get_right() + RIGHT * 0.1,
            angle=PI / 2,
        )

        self.play(GrowArrow(dbz_arrow))

        self.wait(0.5)

        self.play(FadeOut(dbz_arrow))

        self.wait(0.5)

        lin_min = (
            Text("0.01", font=FONT)
            .rescale_to_fit(ppi.colorbar_ticks[0].get_height(), 1)
            .next_to(ppi.colorbar_ticks[0], RIGHT, MED_SMALL_BUFF)
        )
        lin_max = (
            Text("10,000,000", font=FONT)
            .rescale_to_fit(ppi.colorbar_ticks[0].get_height() * 1.2, 1)
            .next_to(ppi.colorbar_ticks[-1], RIGHT, MED_SMALL_BUFF)
        )

        lin_arrow = CurvedArrow(
            lin_min.get_right() + RIGHT * 0.1,
            lin_max.get_right() + RIGHT * 0.1,
            angle=PI / 2,
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(m) for m in lin_min], lag_ratio=0.1),
                LaggedStart(*[FadeIn(m) for m in lin_max], lag_ratio=0.1),
                lag_ratio=0.4,
            )
        )
        self.play(GrowArrow(lin_arrow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(GREEN).scale(1.5) for m in ppi.colorbar_ticks],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

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
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

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

        self.play(gate_start - 0.5, rate_func=utils.rate_functions.there_and_back)

        self.wait(0.5)

        self.play(
            self.west_radar.az - (20 * DEG),
            run_time=3,
            rate_func=utils.rate_functions.there_and_back,
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

        # self.wait(0.5)
        #
        # dbz_value = (
        #     Tex(r"\rightarrow 31.2 \text{ dBZ}")
        #     .scale(1.2)
        #     .fix_in_frame()
        #     .next_to(
        #         self.frame.to_fixed_frame_point(gate.get_center()),
        #         RIGHT,
        #         LARGE_BUFF,
        #     )
        # )
        # gate_screen = self.frame.to_fixed_frame_point(gate.get_center())
        # composition_center = 0.5 * (gate_screen + dbz_value.get_center())
        # composition_center[2] = 0
        #
        # camera_shift = self.frame.from_fixed_frame_point(
        #     composition_center,
        #     relative=True,
        # )
        #
        # self.play(self.frame.animate.shift(camera_shift))
        #
        # self.wait(0.5)
        #
        # self.play(
        #     LaggedStart(
        #         *[
        #             FadeIn(m)
        #             for m in dbz_value.next_to(
        #                 self.frame.to_fixed_frame_point(gate.get_center()),
        #                 RIGHT,
        #                 LARGE_BUFF,
        #             )
        #         ]
        #     )
        # )
        #
        # self.wait(0.5)
        #
        # res_thumbnail = ImageMobject(
        #     "../09_resolution/static/Resolution Thumbnail.png"
        # ).fix_in_frame()
        # thumbnail_box = SurroundingRectangle(res_thumbnail, buff=0, color=YELLOW)
        # thumbnail_label = (
        #     Group(
        #         Text(
        #             "How Radars Tell Targets Apart",
        #             font=FONT,
        #         ).fix_in_frame(),
        #         Text(
        #             "(and When They Can’t)",
        #             font=FONT,
        #         ).fix_in_frame(),
        #     )
        #     .arrange(DOWN, MED_SMALL_BUFF)
        #     .next_to(thumbnail_box, DOWN)
        # )
        #
        # thumbnail = (
        #     Group(res_thumbnail, thumbnail_box, thumbnail_label)
        #     .fix_in_frame()
        #     .shift(UP / 4)
        # )
        #
        # self.play(GrowFromCenter(thumbnail))
        #
        # self.wait(0.5)
        #
        # self.play(FadeOut(thumbnail))
        #
        # self.wait(0.5)
        #
        # self.play(FadeOut(dbz_value), self.frame.animate.shift(OUT * 20))

        self.embed()


class PowerVsZP1(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        self.cloud = Cloud(width=2.4)

        rng = np.random.default_rng(1)

        def get_drop_spec():
            return (
                rng.uniform(-1, 1),
                rng.uniform(-0.4, 0.4),
                rng.uniform(0, 0.3),
                rng.uniform(0, 1),
            )

        max_drops = 100
        n_drops = VT(20)
        show_rain = VT(0)
        drop_specs = [get_drop_spec() for _ in range(max_drops)]
        shuffle(drop_specs)

        def get_raindrops():
            cloud_bottom = self.cloud.get_bounding_box_point(IN)
            raindrops = []

            for x_offset, y_offset, length, gap in drop_specs[: int(~n_drops)]:
                drop = (
                    Cylinder(
                        height=length * ~show_rain,
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
                raindrops.append(drop)
            return Group(raindrops)

        self.raindrops = always_redraw(get_raindrops)

        hud_opacity = VT(1)

        dbz_slider = (
            NumberLine(
                x_range=[-20, 70, 10],
                include_numbers=True,
                width=fh(self, 0.7),
                decimal_number_config={
                    "font_size": 24,
                    "num_decimal_places": 0,
                    "text_config": {"font": FONT},
                },
            )
            .fix_in_frame()
            .shift(LEFT * fw(self, 0.35))
            .rotate(PI / 2)
        )
        for num in dbz_slider.numbers:
            num.rotate(-PI / 2)
        dbz_label = (
            Text("Z [dBZ]", font=FONT)
            .fix_in_frame()
            .next_to(dbz_slider, UP, MED_SMALL_BUFF)
        )

        self.add(self.raindrops)

        dbz_value = VT(10)
        dbz_value_dot = always_redraw(
            lambda: Dot(
                dbz_slider.n2p(~dbz_value),
                radius=DEFAULT_DOT_RADIUS * 2,
                color=BLUE,
                fill_color=BLUE,
                fill_opacity=~hud_opacity,
                stroke_opacity=~hud_opacity,
            ).fix_in_frame()
        )

        self.wait(0.5)

        self.play(
            ShowCreation(dbz_slider),
            Write(dbz_label),
            ShowCreation(dbz_value_dot),
            self.frame.animate.reorient(
                0, 90, 0, center=self.cloud.get_center(), height=4
            ).shift(IN),
            LaggedStart(ShowCreation(self.cloud), show_rain @ 1, lag_ratio=0.9),
            run_time=3,
        )

        self.wait(0.5)

        frame_rotation = VT(0)

        def rotate_frame(frame):
            frame.reorient(
                ~frame_rotation, 90, 0, center=self.cloud.get_center(), height=4
            ).shift(IN)

        # self.frame.add_updater(rotate_frame)

        total_time = 10
        self.play(
            n_drops.animate(run_time=total_time / 2).set_value(100),
            dbz_value.animate(run_time=total_time / 2).set_value(60),
        )

        self.wait(0.5)

        self.play(
            n_drops.animate(run_time=total_time / 2).set_value(10),
            dbz_value.animate(run_time=total_time / 2).set_value(5),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    n_drops.animate(run_time=total_time / 2).set_value(100),
                    dbz_value.animate(run_time=total_time / 2).set_value(60),
                    run_time=3,
                ),
                AnimationGroup(
                    self.frame.animate.reorient(
                        0, 90, 0, center=self.raindrops.get_center() + UP, height=0.5
                    ),
                    FadeOut(dbz_slider),
                    FadeOut(dbz_label),
                    hud_opacity @ 0,
                    run_time=4,
                ),
                lag_ratio=0.1,
            )
        )

        self.wait(2)

        # self.add(*self.raindrops)
        # self.play(
        #     LaggedStart(
        #         *[ShowCreation(m, run_time=0.4) for m in self.raindrops],
        #         lag_ratio=0.1,
        #     )
        # )

        self.embed()


class PowerVsZP2(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        self.terrain_profile = {
            "broad_ridge": 1.25,
            "east_hill": 0.95,
            "valley_depth": 0.52,
            "rolling_ground": 0.4,
            "fine_relief": 0.12,
            "minor_ridges": 0.10,
        }
        self.terrain_sites = ((4.15, 1.45),)
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

        z = Tex(r"Z").scale(2)
        pr = Tex(r"P_r").scale(2)
        zep = Tex(r"Z = P_r").scale(2)
        zsp = Tex(r"Z \sim P_r").scale(2)

        self.play(GrowFromCenter(z))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(z[0], zep[0]),
                GrowFromCenter(zep[1]),
                zep[2:].shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                zep[1].animate.shift(UP).set_opacity(0.2),
                AnimationGroup(
                    zsp[1].shift(DOWN).set_opacity(0).animate.shift(UP).set_opacity(1),
                    ReplacementTransform(zep[0], zsp[0]),
                    ReplacementTransform(zep[2:], zsp[2:]),
                ),
                lag_ratio=0.2,
            )
        )

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
            run_time=4,
        )
        radar_build.group = self.radar
        radar_build.mobject = self.radar

        self.play(
            LaggedStart(
                AnimationGroup(FadeOut(zsp), FadeOut(zep[1])),
                ShowCreation(self.terrain, run_time=3),
                self.frame.animate(run_time=3).reorient(
                    60, 80, 0, center=self.radar.antenna.get_center(), height=4
                ),
                radar_build,
                self.radar.el.animate(run_time=4).set_value(20 * DEG),
                lag_ratio=0.3,
            )
        )

        beam_range = VT(0.1)
        beam_opacity = VT(0)
        self.beam = always_redraw(
            lambda: self.radar.get_beam(
                8 * DEG, ~beam_range, opacity=~beam_opacity, depth_test=False
            )
        )
        self.add(self.beam)

        self.wait(0.5)

        self.play(
            LaggedStart(
                beam_opacity @ 0.3,
                beam_range.animate(run_time=3).set_value(8),
                lag_ratio=0.3,
            )
        )

        self.cloud = (
            Cloud(width=2.4)
            .move_to(self.radar.beam_location_to_point(4, 0, 0))
            .rotate(-PI / 3, axis=OUT)
            .shift(OUT)
        )

        rng = np.random.default_rng(1)

        def get_drop_spec():
            return (
                rng.uniform(-1, 1),
                rng.uniform(-0.4, 0.4),
                rng.uniform(0, 0.3),
                rng.uniform(0, 1),
            )

        max_drops = 100
        n_drops = VT(100)
        show_rain = VT(0)
        show_snowflakes = VT(0)
        drop_specs = [get_drop_spec() for _ in range(max_drops)]
        shuffle(drop_specs)
        snowflake_specs = [
            (*spec, radius)
            for spec, radius in zip(
                drop_specs,
                rng.uniform(0.02, 0.07, max_drops),
            )
        ]

        def get_raindrops():
            cloud_bottom = self.cloud.get_bounding_box_point(IN)
            raindrops = []
            active_count = max(0, min(int(~n_drops), max_drops))

            for x_offset, y_offset, length, gap in drop_specs[:active_count:2]:
                drop = (
                    Cylinder(
                        height=length * ~show_rain,
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
                raindrops.append(drop)
            return Group(raindrops)

        def get_snowflakes():
            cloud_bottom = self.cloud.get_bounding_box_point(IN)
            snowflakes = []
            active_count = max(0, min(int(~n_drops), max_drops))
            snowflake_scale = np.clip(~show_snowflakes, 0, 1)
            snowflakes_visible = snowflake_scale > 0

            for x_offset, y_offset, length, gap, radius in snowflake_specs[
                1:active_count:2
            ]:
                snowflake = Sphere(
                    radius=max(radius * snowflake_scale, 1e-6),
                    color=WHITE,
                    resolution=(16, 8),
                    shading=(0.1, 0.25, 0.15),
                    depth_test=snowflakes_visible,
                ).set_opacity(0.82 if snowflakes_visible else 0)
                snowflake.move_to(
                    cloud_bottom
                    + x_offset * RIGHT
                    + y_offset * UP
                    + (0.5 * length + gap) * IN
                )
                snowflakes.append(snowflake)
            return Group(snowflakes)

        self.raindrops = always_redraw(get_raindrops)
        self.snowflakes = always_redraw(get_snowflakes)
        self.add(self.raindrops, self.snowflakes)

        self.play(
            self.frame.animate.reorient(
                -20,
                110,
                0,
                center=Group(self.radar.antenna, self.cloud).get_center(),
                height=5,
            ),
            ShowCreation(self.cloud),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                show_rain @ 1,
                show_snowflakes @ 1,
                lag_ratio=0.7,
            ),
            run_time=4,
        )

        precipitation = Group(self.raindrops, self.snowflakes)
        scatter_origin = precipitation.get_center().copy()
        receiver = self.radar.get_feed_horn_origin().copy()
        backscatter_vector = receiver - scatter_origin
        receiver_distance = np.linalg.norm(backscatter_vector)
        backscatter_axis = unit(backscatter_vector)

        reference = OUT if abs(np.dot(backscatter_axis, OUT)) < 0.95 else UP
        tangent1 = unit(np.cross(backscatter_axis, reference))
        tangent2 = unit(np.cross(backscatter_axis, tangent1))

        scatter_radius = VT(0.001)
        scatter_opacity = VT(0)
        backscatter_half_angle = 10 * DEG

        def get_backscatter_visibility():
            return float(~scatter_radius < receiver_distance)

        def get_backscatter_cap(radius):
            return ParametricSurface(
                lambda polar, azimuth: (
                    scatter_origin
                    + radius
                    * (
                        np.cos(polar) * backscatter_axis
                        + np.sin(polar)
                        * (np.cos(azimuth) * tangent1 + np.sin(azimuth) * tangent2)
                    )
                ),
                u_range=(0, backscatter_half_angle),
                v_range=(0, TAU),
                resolution=(14, 48),
                color=BLUE_A,
                opacity=~scatter_opacity * get_backscatter_visibility(),
                shading=(0.1, 0.2, 0.15),
                depth_test=False,
            )

        def get_scatter_wave():
            radius = max(~scatter_radius, 0.001)
            all_directions = Sphere(
                radius=radius,
                resolution=(48, 24),
                color=BLUE,
                opacity=0.07 * ~scatter_opacity,
                shading=(0.05, 0.1, 0.05),
                depth_test=False,
            ).move_to(scatter_origin)
            backscatter = get_backscatter_cap(radius * 1.003)
            return Group(all_directions, backscatter)

        def get_scatter_arrow(direction, opacity, color=BLUE):
            arrow_length = max(~scatter_radius, 0.001)
            return (
                Arrow(
                    scatter_origin,
                    scatter_origin + arrow_length * direction,
                    buff=0,
                    fill_color=color,
                    thickness=2.5,
                )
                .set_perpendicular_to_camera(self.frame)
                .set_opacity(opacity * ~scatter_opacity)
                .deactivate_depth_test()
            )

        def get_scatter_arrows():
            return Group(
                get_scatter_arrow(
                    backscatter_axis,
                    get_backscatter_visibility(),
                    RED,
                ),
                get_scatter_arrow(-backscatter_axis, 0.15),
                get_scatter_arrow(tangent1, 0.15),
                get_scatter_arrow(-tangent1, 0.15),
                get_scatter_arrow(tangent2, 0.15),
                get_scatter_arrow(-tangent2, 0.15),
            )

        scatter_wave = always_redraw(get_scatter_wave)
        scatter_arrows = always_redraw(get_scatter_arrows)
        self.add(scatter_wave, scatter_arrows)

        radar_extent = np.linalg.norm(
            self.radar.antenna.get_all_points() - scatter_origin,
            axis=1,
        ).max()
        target_radius = max(
            1.5
            * max(
                precipitation.get_width(),
                precipitation.get_height(),
            ),
            1.25 * radar_extent,
        )
        self.play(
            LaggedStart(
                self.frame.animate(run_time=12).reorient(
                    -140 + 360,
                    110,
                    0,
                    center=Group(self.radar.antenna, self.cloud).get_center(),
                    height=5,
                ),
                AnimationGroup(
                    scatter_opacity @ 1,
                    scatter_radius.animate(run_time=12).set_value(target_radius),
                ),
                lag_ratio=0.1,
                # rate_func=ease_out_cubic,
            )
        )

        self.wait(0.5)

        self.play(scatter_opacity @ 0)
        self.remove(scatter_wave, scatter_arrows)

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=3)
                .reorient(
                    -160 + 360,
                    85,
                    0,
                    center=self.radar.antenna.get_center(),
                    height=1.5,
                )
                .shift(LEFT * 0.8),
                self.radar.waveguides.animate.set_color(RED),
                lag_ratio=0.7,
            )
        )

        pr_ax = (
            Axes(
                x_range=[0, 1, 0.1],
                y_range=[0, 1, 0.25],
                width=fw(self, 1),
                height=fh(self, 1),
                axis_config={
                    "stroke_width": DEFAULT_STROKE_WIDTH * 0.5,
                    "tick_size": 0.04,
                    "scale_stroke_with_zoom": True,
                },
            )
            .fix_in_frame()
            .next_to(
                self.frame.get_corner(UR),
                DL,
                MED_LARGE_BUFF,
            )
            .shift(LEFT * 1.5 + DOWN * 0.5)
            .shift(RIGHT * 3.5)
        )
        self.add(pr_ax)

        pr_label = (
            Tex(r"P_r", color=RED, fill_color=RED, fill_opacity=1)
            .fix_in_frame()
            .rescale_to_fit(pr_ax.get_height() * 0.2, 1)
            .next_to(pr_ax.copy().shift(LEFT * 3.5), UP, 0)
        )

        cloud_range = VT(0.5)
        cloud_range2 = VT(0.3)
        pr_norm = VT(0)
        norm_val1 = VT(0.3)
        norm_val2 = VT(0.5)
        cloud2_show = VT(0)

        def get_rx():
            range_samples = np.linspace(0, 1, 200)
            cloud_sigma = 0.055
            rng = np.random.default_rng(7)
            received_power = (
                0.08
                + (1 - (~cloud_range * 0.9 * (1 - ~pr_norm)) - ~norm_val1 * ~pr_norm)
                * np.exp(-0.5 * ((range_samples - ~cloud_range) / cloud_sigma) ** 2)
                + rng.normal(0, 0.025, range_samples.size)
                + ~norm_val2
                * ~cloud2_show
                * np.exp(-0.5 * ((range_samples - ~cloud_range2) / cloud_sigma) ** 2)
            )
            received_power = np.clip(received_power, pr_ax.y_range[0], None)
            return lambda r: np.interp(r, range_samples, received_power)

        pr_plot_x1 = VT(0)
        pr_plot_opacity = VT(1)
        pr_plot = always_redraw(
            lambda: (
                pr_ax.get_graph(
                    get_rx(),
                    x_range=[0, ~pr_plot_x1, 1 / 800],
                    color=RX_COLOR,
                    stroke_width=DEFAULT_STROKE_WIDTH * 0.5,
                    scale_stroke_with_zoom=True,
                    use_smoothing=False,
                )
                .set_stroke(opacity=~pr_plot_opacity)
                .fix_in_frame()
            )
        )

        def to_screen_plane(point):
            point = point.copy()
            perspective_scale = (
                1 - point[2] * self.frame.get_scale() / self.frame.get_focal_distance()
            )
            point[:2] /= perspective_scale
            point[2] = 0
            return point

        plot_point = to_screen_plane(pr_ax.c2p(0.5, 0) + LEFT * 3.5)

        def get_bez():
            waveguide_point = to_screen_plane(
                self.frame.to_fixed_frame_point(self.radar.waveguides.get_center())
            )
            return (
                CubicBezier(
                    waveguide_point,
                    waveguide_point + RIGHT * 0.1,
                    plot_point + DOWN * 1.5,
                    plot_point,
                    stroke_width=DEFAULT_STROKE_WIDTH * 1,
                    scale_stroke_with_zoom=True,
                )
                .set_stroke(opacity=~pr_plot_opacity)
                .fix_in_frame()
            )

        to_plot = get_bez()

        self.add(pr_plot)

        self.play(
            ShowCreation(to_plot),
            FadeIn(pr_label),
            pr_ax.animate.shift(LEFT * 3.5),
            pr_plot_x1.animate(run_time=2).set_value(1),
        )

        self.remove(to_plot)
        to_plot = always_redraw(get_bez)
        self.add(to_plot)

        self.wait(0.5)

        self.play(
            self.frame.animate(run_time=2)
            .reorient(
                -140 + 360,
                85,
                0,
                center=Group(self.cloud, self.radar.antenna).get_center(),
                height=4,
            )
            .shift(LEFT)
        )

        self.wait(0.5)

        self.play(
            cloud_range @ 0.2,
            self.cloud.animate.shift(self.radar.get_beam_direction() * -1.5),
        )

        self.wait(0.5)

        self.play(
            cloud_range @ 0.8,
            self.cloud.animate.shift(self.radar.get_beam_direction() * 4.5),
        )

        # i would like to be done now, sir

        self.wait(0.5)

        self.play(pr_norm @ 1)

        self.wait(0.5)

        self.play(
            cloud_range @ 0.2,
            self.cloud.animate.shift(self.radar.get_beam_direction() * -4.5),
        )

        self.wait(0.5)

        self.play(
            # FadeOut(pr_plot),
            pr_plot_opacity @ 0,
            FadeOut(pr_ax),
            # FadeOut(to_plot),
            self.frame.animate(run_time=3).shift(OUT * 10),
            pr_label.animate.move_to(ORIGIN).scale(3),
        )

        self.wait(0.5)

        pr_rel = (
            Tex(
                r"P_r \sim R \text{, } P_t\text{, } G \text{, } \lambda \text{, } \ldots"
            )
            .fix_in_frame()
            .rescale_to_fit(fw(self, 0.8), 0)
            .move_to(pr_label)
        )

        self.play(
            LaggedStart(
                ReplacementTransform(pr_label, pr_rel[:2]),
                *[GrowFromCenter(m) for m in pr_rel[2:]],
                lag_ratio=0.2,
            )
        )
        #
        # self.wait(0.5)
        #
        # z = (
        #     Tex(r"Z")
        #     .fix_in_frame()
        #     .rescale_to_fit(pr_label.get_height(), 1)
        #     .move_to(ORIGIN)
        #     .shift(UP * 10)
        # )
        # self.add(z)

        self.play(pr_rel.animate.shift(DOWN * 10))

        # pr = MathTex(r"P_r").scale(1.5).rotate(PI / 2).next_to(pr_ax, LEFT, SMALL_BUFF)
        # pr_bez = CubicBezier(
        #     radar.radome.get_left(),
        #     radar.radome.get_left() + [-0.5, 0.5, 0],
        #     pr_ax.copy().shift(DOWN * fh(self)).get_corner(DL) + [-0.5, -0.5, 0],
        #     pr_ax.copy().shift(DOWN * fh(self)).get_corner(DL),
        # )

        self.embed()


class PowerVsZP3(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

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

        self.west_radar.az @= 30 * DEG
        self.west_radar.el @= 10 * DEG
        self.east_radar.az @= -70 * DEG
        self.east_radar.el @= 0 * DEG

        self.frame.reorient(
            60,
            75,
            0,
            center=Group(self.west_radar, self.east_radar).get_center(),
            height=11,
        ).shift(OUT * 30)
        self.add(self.terrain, self.west_radar, self.east_radar)

        z = Tex("Z").fix_in_frame().rescale_to_fit(fh(self, 0.1), 1).move_to(ORIGIN)

        self.play(GrowFromCenter(z))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(z),
                self.frame.animate(run_time=3).shift(IN * 30),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.west_cloud = Cloud(3).move_to(
            self.west_radar.beam_location_to_point(3, 0, 0)
        )
        self.east_cloud = (
            Cloud(4)
            .move_to(self.east_radar.beam_location_to_point(7, 0, 0))
            .rotate(PI / 6, axis=OUT)
        )

        self.west_beam = self.west_radar.get_beam(
            8 * DEG, 11, color=BLUE, opacity=1
        ).apply_depth_test()
        self.east_beam = self.east_radar.get_beam(
            8 * DEG, 11, color=BLUE, opacity=1
        ).apply_depth_test()

        self.play(ShowCreation(self.west_cloud), ShowCreation(self.west_beam))

        self.wait(0.5)

        self.play(ShowCreation(self.east_cloud), ShowCreation(self.east_beam))

        self.wait(0.5)

        self.play(self.frame.animate(run_time=3).shift(OUT * 30))

        self.embed()


class Wrapup(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        def pec_sphere_backscatter(size_parameter):
            values = np.atleast_1d(np.asarray(size_parameter, dtype=float))
            result = np.empty_like(values)

            for index, x in enumerate(values.flat):
                orders = np.arange(1, int(np.ceil(x + 4 * x ** (1 / 3) + 2)) + 1)
                jn = spherical_jn(orders, x)
                yn = spherical_yn(orders, x)
                jn_prime = spherical_jn(orders, x, derivative=True)
                yn_prime = spherical_yn(orders, x, derivative=True)

                psi = x * jn
                xi = x * (jn + 1j * yn)
                psi_prime = jn + x * jn_prime
                xi_prime = psi_prime + 1j * (yn + x * yn_prime)
                a_n = -psi_prime / xi_prime
                b_n = -psi / xi

                amplitude = np.sum((2 * orders + 1) * (-1) ** orders * (a_n - b_n))
                result.flat[index] = abs(amplitude) ** 2 / x**2

            return result.item() if np.ndim(size_parameter) == 0 else result

        x_min, x_max = -1, 2
        y_min, y_max = -4, 1
        chart_width = fw(self, 0.72)
        chart_height = fh(self, 0.62)

        x_axis = NumberLine(
            x_range=[x_min, x_max, 1],
            width=chart_width,
            include_tip=False,
            tick_size=0.06,
        )
        y_axis = NumberLine(
            x_range=[y_min, y_max, 1],
            width=chart_height,
            include_tip=False,
            tick_size=0.06,
        ).rotate(PI / 2)
        y_axis.shift(x_axis.get_start() - y_axis.get_start())

        def log_to_point(log_x, log_y):
            return x_axis.n2p(log_x) + y_axis.n2p(log_y) - y_axis.n2p(y_min)

        grid = VGroup(
            *[
                Line(
                    log_to_point(log_x, y_min),
                    log_to_point(log_x, y_max),
                ).set_stroke(GREY_B, 1, opacity=0.2)
                for log_x in range(x_min, x_max + 1)
            ],
            *[
                Line(
                    log_to_point(x_min, log_y),
                    log_to_point(x_max, log_y),
                ).set_stroke(GREY_B, 1, opacity=0.2)
                for log_y in range(y_min, y_max + 1)
            ],
        )

        x_tick_labels = VGroup(
            *[
                Tex(rf"10^{{{exponent}}}")
                .scale(0.55)
                .next_to(x_axis.n2p(exponent), DOWN, SMALL_BUFF)
                for exponent in range(x_min, x_max + 1)
            ]
        )
        y_tick_labels = VGroup(
            *[
                Tex(rf"10^{{{exponent}}}")
                .scale(0.55)
                .next_to(y_axis.n2p(exponent), LEFT, SMALL_BUFF)
                for exponent in range(y_min, y_max + 1)
            ]
        )
        x_label = Tex(r"ka = 2\pi a / \lambda").next_to(x_axis, DOWN, MED_LARGE_BUFF)
        y_label = (
            Tex(r"\sigma / (\pi a^2)")
            .rotate(PI / 2)
            .next_to(y_axis, LEFT, MED_LARGE_BUFF)
        )

        regime_specs = [
            ("Rayleigh", x_min, 0, GREEN),
            ("Mie", 0, 1, PURPLE),
            ("Optical", 1, x_max, YELLOW),
        ]
        regime_bands = VGroup()
        regime_labels = VGroup()
        for name, regime_min, regime_max, color in regime_specs:
            band = Polygon(
                log_to_point(regime_min, y_min),
                log_to_point(regime_max, y_min),
                log_to_point(regime_max, y_max),
                log_to_point(regime_min, y_max),
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.12,
            )
            label = (
                Text(name, font=FONT, color=color)
                .scale(0.6)
                .next_to(band.get_top(), DOWN, SMALL_BUFF)
            )
            regime_bands.add(band)
            regime_labels.add(label)

        ka_values = np.logspace(x_min, x_max, 1600)
        normalized_rcs = pec_sphere_backscatter(ka_values)
        log_ka = np.log10(ka_values)
        log_rcs = np.log10(normalized_rcs)

        regime_curves = VGroup()
        for _, regime_min, regime_max, color in regime_specs:
            regime_mask = (log_ka >= regime_min) & (log_ka <= regime_max)
            points = [
                log_to_point(x_value, y_value)
                for x_value, y_value in zip(
                    log_ka[regime_mask],
                    log_rcs[regime_mask],
                )
            ]
            regime_curves.add(
                VMobject().set_points_as_corners(points).set_stroke(color, 4)
            )

        rcs_chart = Group(
            regime_bands,
            grid,
            x_axis,
            y_axis,
            x_tick_labels,
            y_tick_labels,
            x_label,
            y_label,
            regime_curves,
            regime_labels,
        )
        rcs_chart.move_to(ORIGIN).shift(LEFT * fw(self, 0.25)).rescale_to_fit(
            fw(self, 0.4), 0
        ).fix_in_frame()

        self.play(
            ShowCreation(x_axis),
            ShowCreation(y_axis),
            FadeIn(grid),
            LaggedStart(
                Write(x_tick_labels),
                Write(y_tick_labels),
                Write(x_label),
                Write(y_label),
                lag_ratio=0.15,
            ),
        )

        self.cloud = Cloud(width=3)

        self.frame.reorient(0, 90, 0, center=self.cloud.get_center(), height=5).shift(
            LEFT * 2
        )

        # other_anims = []
        # for idx, (band, label, curve) in enumerate(
        #     zip(
        #         regime_bands,
        #         regime_labels,
        #         regime_curves,
        #     )
        # ):
        #     if idx == 1:
        #         other_anims.append(ShowCreation(self.cloud))
        #
        #     self.play(
        #         FadeIn(band),
        #         Write(label),
        #         ShowCreation(curve),
        #         *other_anims,
        #         run_time=1.5,
        #     )
        #     other_anims.clear()

        regime_sequence = LaggedStart(
            *[
                AnimationGroup(
                    FadeIn(band),
                    Write(label),
                    ShowCreation(curve),
                    run_time=1.5,
                )
                for band, label, curve in zip(
                    regime_bands,
                    regime_labels,
                    regime_curves,
                )
            ],
            lag_ratio=1,
        )

        self.play(
            LaggedStart(
                regime_sequence,
                ShowCreation(self.cloud, run_time=3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(rcs_chart.animate.shift(UP * fh(self, 0.3)))

        self.wait(0.5)

        dsd_eqn = (
            Tex(r"\Large Z = \int_0^\infty N(D)D^6\,dD")
            .fix_in_frame()
            .next_to(rcs_chart, DOWN, fh(self, 0.3))
        )
        regimes_title = (
            Text("Scattering Regimes", font=FONT)
            .fix_in_frame()
            .scale(0.7)
            .next_to(rcs_chart, UP)
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(m) for m in dsd_eqn], lag_ratio=0.1),
                Write(regimes_title),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        notebooks = (
            VideoMobject("~/Movies/2026-07-24_11-41-09.mov", start=20)
            .fix_in_frame()
            .move_to(ORIGIN)
            .rescale_to_fit(fw(self, 0.8), 0)
        )
        website_name = (
            Text("marshallbruner.com", font=FONT)
            .fix_in_frame()
            .next_to(notebooks, UP, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(self.cloud),
                    FadeOut(rcs_chart),
                    FadeOut(dsd_eqn),
                    FadeOut(regimes_title),
                ),
                AnimationGroup(
                    notebooks.shift(DOWN * fh(self, 2)).animate.shift(UP * fh(self, 2)),
                    Write(website_name),
                ),
            )
        )

        self.wait(5)

        merch1 = (
            ImageMobject("../14_amp_gain/static/merch1.png")
            .rescale_to_fit(notebooks.get_height(), 1)
            .fix_in_frame()
            .next_to(notebooks.copy().shift(LEFT * 3), RIGHT)
        )
        merch2 = (
            ImageMobject("../14_amp_gain/static/merch5.png")
            .rescale_to_fit(notebooks.get_height(), 1)
            .fix_in_frame()
            .move_to(merch1)
        )
        merch3 = (
            ImageMobject("../14_amp_gain/static/merch9.png")
            .rescale_to_fit(notebooks.get_height(), 1)
            .fix_in_frame()
            .move_to(merch1)
        )

        self.play(
            Group(notebooks, website_name).animate.shift(LEFT * 3),
            merch1.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        self.play(
            merch1.animate.shift(UP * 10),
            merch2.shift(DOWN * 10).animate.shift(UP * 10),
        )

        self.wait(0.5)

        self.play(
            merch2.animate.shift(UP * 10),
            merch3.shift(DOWN * 10).animate.shift(UP * 10),
        )

        with open(__file__, "r", encoding="utf-8") as source_file:
            lines_str = f"{sum(1 for _ in source_file):,}"

        stats_title = Text("Stats for Nerds", font=FONT).scale(0.7)
        stats_data = [
            ("Lines of code", lines_str),
            ("Script word count", "2,138"),
            ("Days to make", "54"),
            ("Git commits", "39"),
        ]
        row_height = 0.55
        label_column_width = 3.2
        value_column_width = 1.6
        table_width = label_column_width + value_column_width
        table_height = row_height * len(stats_data)

        table_grid = VGroup(
            Rectangle(
                width=table_width,
                height=table_height,
                color=GREY_B,
                stroke_width=2,
            ),
            Line(
                UP * table_height / 2,
                DOWN * table_height / 2,
                color=GREY_B,
                stroke_width=2,
            ).shift(LEFT * table_width / 2 + RIGHT * label_column_width),
            *[
                Line(
                    LEFT * table_width / 2,
                    RIGHT * table_width / 2,
                    color=GREY_B,
                    stroke_width=2,
                ).shift(UP * table_height / 2 + DOWN * row_height * row_index)
                for row_index in range(1, len(stats_data))
            ],
        )
        table_entries = VGroup()
        for row_index, (label_text, value_text) in enumerate(stats_data):
            row_y = table_height / 2 - row_height * (row_index + 0.5)
            label = Text(
                label_text,
                font=FONT,
                font_size=17,
            ).move_to(
                LEFT * table_width / 2 + RIGHT * label_column_width / 2 + UP * row_y
            )
            value = Text(
                value_text,
                font=FONT,
                font_size=17,
                color=GREEN,
            ).move_to(
                LEFT * table_width / 2
                + RIGHT * (label_column_width + value_column_width / 2)
                + UP * row_y
            )
            table_entries.add(label, value)

        stats_table = VGroup(table_grid, table_entries).next_to(
            stats_title,
            DOWN,
            MED_LARGE_BUFF,
        )
        stats_group = VGroup(stats_title, stats_table).to_corner(
            UR,
            buff=LARGE_BUFF * 2,
        )

        thank_you_sabrina = Text(
            "Thank you, Sabrina, for\nediting the whole video :)",
            font=FONT,
            font_size=24,
        ).next_to(stats_group, DOWN)

        marshall_bruner = (
            Text(
                "Marshall Bruner",
                font=FONT,
                font_size=24,
            )
            .set_y(thank_you_sabrina.get_y())
            .set_x(-fw(self, 0.25))
        )
        end_screen = VGroup(
            stats_group,
            thank_you_sabrina,
            marshall_bruner,
        ).fix_in_frame()

        self.play(
            FadeOut(notebooks, shift=UP * fh(self)),
            FadeOut(website_name, shift=UP * fh(self)),
            FadeOut(merch3, shift=UP * fh(self)),
        )

        df = pd.read_csv("~/Downloads/members-2026-07-24_18-24-58.csv")
        df.sort_values(by="Account Signup Date (UTC)", inplace=True)
        members = df[(df["Member Status"] == "Active member")].Nickname.tolist()
        mem_list = (
            Group(*[Text(member, font=FONT) for member in members])
            .arrange_in_grid(
                n_cols=2,
                n_rows=len(members) // 2,
                h_buff=MED_LARGE_BUFF,
                v_buff=SMALL_BUFF,
            )
            .fix_in_frame()
            .move_to(ORIGIN)
            .shift(DOWN / 2)
        )
        members_title = (
            Text("Thanks to the Members!!", font=FONT)
            .fix_in_frame()
            .scale(1.5)
            .next_to(mem_list, UP, MED_LARGE_BUFF)
        )
        members_title[-8:].set_color(ORANGE)

        self.play(
            LaggedStart(
                Write(members_title),
                *[Write(m) for m in mem_list],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(members_title), FadeOut(mem_list))

        self.play(
            LaggedStart(
                FadeIn(marshall_bruner, shift=UP),
                AnimationGroup(
                    FadeIn(stats_title, shift=DOWN),
                    FadeIn(stats_table),
                ),
                Write(thank_you_sabrina),
                lag_ratio=0.9,
                run_time=4,
            )
        )

        self.wait(2)

        self.embed()


class ThumbnailTesting(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

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
            vmin=-20,
            vmax=70,
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

        # self.play(ShowCreation(self.third_beam), run_time=3)
        self.play(
            self.frame.animate(run_time=1)
            .reorient(
                -4,
                86,
                0,
                center=Group(
                    self.west_radar.antenna,
                    self.east_radar.antenna,
                    # self.third_radar.antenna,
                ).get_center(),
                height=8.4,
            )
            .shift(OUT * 0.5),
        )

        self.remove(self.terrain)

        self.embed()
