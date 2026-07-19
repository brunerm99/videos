# zgl.py

import os
import sys

from dotenv import load_dotenv
from manimlib import *

sys.path.insert(0, "..")
load_dotenv()

from melp import get_nexrad_product_ppi_data
from melp.props import (
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
    default_camera_config = {"background_color": BACKGROUND_COLOR}

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
