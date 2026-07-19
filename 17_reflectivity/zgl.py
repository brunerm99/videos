# zgl.py

import os
import sys

from manimlib import *

sys.path.insert(0, "..")

from melp.props import (
    Cloud,
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

        # Keep the dome attached for later, but stop the invisible mesh from
        # hiding the antenna through the depth buffer.
        self.west_radar.dome.set_opacity(0)
        self.west_radar.dome.deactivate_depth_test()
        self.add(self.west_radar)

        radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in tower_parts],
            lag_ratio=0.3,
            run_time=5,
        )
        # ManimGL otherwise promotes the pieces to top-level mobjects and drops
        # the WSR88Model updater.  Keep this animation bound to the full radar.
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

        self.play(self.cloud.animate.shift(DOWN).rotate(-PI / 8), run_time=3)

        # self.play(
        #
        #             self.west_radar.az @ (45 * DEG),
        #             self.west_radar.el @ (60 * DEG),
        #         )

        # self.wait(2)

        self.embed()
