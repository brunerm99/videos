# zphys.py

import os
import sys
from pathlib import Path
from random import random, seed, shuffle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from manimlib import *
from scipy.spatial.transform import Rotation
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, "..")
load_dotenv()


from melp import (
    VT,
    Arrow3D,
    ChargePair,
    DimensionLine,
    GrowArrow,
    VideoMobject,
    ease_in_sine,
    ease_out_back,
    ease_out_cubic,
    ease_out_sine,
    fh,
    fw,
    get_nexrad_product_ppi_data,
    index_labels,
    multi_anim,
    there_and_back_with_final,
    unit,
)
from melp.props import (
    Beam,
    Cloud,
    PPIGrid,
    RainDrop,
    Router,
    Terrain,
    TriangleMeshMobject,
    WSR88Model,
    Yagi,
    get_wsr88_model,
    place_wsr88_model,
    place_wsr88_on_terrain,
)
from melp.props import Dipole as DipoleAntenna
from melp.props.ppi import field_to_rgba

BACKGROUND_COLOR = "#183340"
ICE = "#DBF1FD"
RX_COLOR = RED
TX_COLOR = BLUE
TOWER_LIGHT = "#DCE6EB"
TOWER_MID = "#A9BBC4"
BEAM_COLOR = BLUE
GATE_COLOR = TEAL
ACCENT_COLOR = YELLOW
FONT = os.getenv("FONT", "Maple Mono CN")
DIPOLE_CLOSEUP_HEIGHT = 2.0
DIPOLE_HANDOFF_HEIGHT = 2.8


class IntroP1(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        def get_thumbnail(path):
            thumbnail = ImageMobject(path).rescale_to_fit(fh(self, 0.3), 1)
            box = SurroundingRectangle(thumbnail, buff=0, color=GREEN)
            return Group(thumbnail, box)

        rf_thumbnails = Group(
            Text("RF", font=FONT, fill_color=RED),
            *[
                get_thumbnail(path)
                for path in [
                    "../05_phased_array/media/images/phased_array//thumbnails/Thumbnail1.png",
                    # "../08_beamforming/media/images/beamforming/Thumbnail_ManimCE_v0.18.1.png",
                    # "../11_aliasing/static/Aliasing Thumbnail.png",
                    "../12_iq_sampling/static/IQ Sampling Thumbnail.png",
                    "../09_resolution/static/Resolution Thumbnail.png",
                ]
            ],
            Text("...", font=FONT).rotate(PI / 2),
        ).arrange(DOWN, buff=MED_SMALL_BUFF * 0.9)
        radar_thumbnails = Group(
            Text("Radar", font=FONT, fill_color=YELLOW),
            *[
                get_thumbnail(path)
                for path in [
                    "../01_fmcw/media/images/fmcw/thumbnails/comparison.png",
                    # "../02_fmcw_implementation/media/images/fmcw_implementation/Thumbnail_Option_1.png",
                    "../03_cfar/media/images/cfar/thumbnails/Thumbnail_1.png",
                    # "../06_radar_range_equation/media/images/radar_equation/thumbnails/Thumbnail_1.png",
                    "../04_fmcw_doppler/media/images/fmcw_doppler/thumbnails/Thumbnail_2.png",
                ]
            ],
            Text("...", font=FONT).rotate(PI / 2),
        ).arrange(DOWN, buff=MED_SMALL_BUFF * 0.9)
        wx_radar_thumbnails = Group(
            Text("Weather Radar", font=FONT, fill_color=BLUE),
            *[
                get_thumbnail(path)
                for path in [
                    "../16_dual_pol/static/Thumbnail - How Weather Radars See Shape.png",
                    "../17_reflectivity/static/thumbnail2.png",
                ]
            ],
        ).arrange(DOWN, buff=MED_SMALL_BUFF * 0.9)
        thumbnails = (
            Group(radar_thumbnails, wx_radar_thumbnails, rf_thumbnails).arrange(
                RIGHT, buff=MED_SMALL_BUFF, aligned_edge=UP
            )
            # .rescale_to_fit(fh(self, 0.9), 1)
        )

        self.frame.rescale_to_fit(thumbnails.get_height() * 1.1, 1)

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        *[
                            LaggedStart(FadeIn(t[0]), ShowCreation(t[1]), lag_ratio=0)
                            if type(t) is Group
                            else FadeIn(t)
                            for t in ts
                        ],
                        lag_ratio=0.3,
                    )
                    for ts in thumbnails
                ],
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        this_vid_box = (
            rf_thumbnails[1][1]
            .copy()
            .next_to(wx_radar_thumbnails[-1], DOWN, MED_SMALL_BUFF)
            .set_y(rf_thumbnails[-2].get_y())
        )
        this_vid_text = Text("This video", font=FONT).move_to(this_vid_box)

        self.play(ShowCreation(this_vid_box), Write(this_vid_text))

        self.wait(0.5)

        # self.frame.shift(DOWN * fh(self))

        line = Line(self.frame.get_center(), self.frame.get_center() + UP * 2)
        top = (
            Circle(radius=fh(self, 1 / 4), stroke_color=BLUE)
            .move_to(self.frame)
            .move_to(line.get_end())
        )
        line.rotate(2 * PI / 3, about_point=self.frame.get_center())
        left = (
            Circle(radius=fh(self, 1 / 4), stroke_color=YELLOW)
            .move_to(self.frame)
            .move_to(line.get_end())
        )
        line.rotate(2 * PI / 3, about_point=self.frame.get_center())
        right = (
            Circle(radius=fh(self, 1 / 4), stroke_color=RED)
            .move_to(self.frame)
            .move_to(line.get_end())
        )
        venn = (
            Group(top, left, right)
            .move_to(self.frame)
            .rescale_to_fit(fh(self, 0.95), 1)
        )

        self.add(thumbnails, this_vid_box, this_vid_text)
        # self.add(venn)

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(radar_thumbnails[0]),
                    FadeOut(radar_thumbnails[-1]),
                    FadeOut(wx_radar_thumbnails[0]),
                    FadeOut(rf_thumbnails[0]),
                    FadeOut(rf_thumbnails[-1]),
                    lag_ratio=0,
                ),
                radar_thumbnails[3]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(left.get_bottom(), UP, MED_SMALL_BUFF * 1.2)
                .shift(UP * 0.1),
                radar_thumbnails[2]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(left.get_left(), RIGHT, MED_SMALL_BUFF)
                .shift(RIGHT * 0.7 + DOWN * 0.6),
                radar_thumbnails[1]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(left.get_left(), RIGHT, MED_SMALL_BUFF)
                .shift(RIGHT * 0.5 + UP * 0.8),
                rf_thumbnails[3]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(right.get_bottom(), UP, MED_SMALL_BUFF * 1.2)
                .shift(UP * 0.1),
                rf_thumbnails[2]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(right.get_right(), LEFT, MED_SMALL_BUFF)
                .shift(LEFT * 0.7 + DOWN * 0.6),
                rf_thumbnails[1]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(right.get_right(), LEFT, MED_SMALL_BUFF)
                .shift(LEFT * 0.5 + UP * 0.8),
                wx_radar_thumbnails[1]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(top.get_top(), DOWN, LARGE_BUFF * 1.4)
                .shift(LEFT * 1.2),
                wx_radar_thumbnails[2]
                .animate.rescale_to_fit(top.get_height() * 0.2, 1)
                .next_to(top.get_top(), DOWN, LARGE_BUFF * 1.4)
                .shift(RIGHT * 1.2),
                Group(this_vid_box, this_vid_text)
                .animate.rescale_to_fit(top.get_height() * 0.1, 1)
                .move_to(venn)
                .shift(DOWN * 0.7),
                ShowCreation(left),
                ShowCreation(right),
                ShowCreation(top),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            self.frame.animate.rescale_to_fit(this_vid_text.get_height(), 1).move_to(
                this_vid_text
            ),
            FadeOut(this_vid_text),
            run_time=1.5,
        )

        self.wait(2)

        self.embed()


class IntroP2(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        cloud = (
            Cloud()
            .rotate(-PI / 2, axis=RIGHT)
            .shift(RIGHT * fw(self, 0.3))
            .shift(UP * 10)
        )

        radar, beam_origin = place_wsr88_model(height=fh(self, 1), shift=0)
        radar.rotate(-PI / 2, axis=RIGHT).shift(DOWN * 9 + LEFT * 4 + IN * 4)
        radar.dome.set_opacity(0)
        radar.dome.deactivate_depth_test()
        radar.shift(DOWN * 10)
        self.add(radar)

        self.play(
            LaggedStart(
                self.frame.animate.shift(DOWN * 10),
                radar.el.animate(run_time=2).set_value(20 * DEG),
                cloud.animate.shift(DOWN * 20),
                lag_ratio=0.3,
            )
        )

        dipole = DipoleAntenna(
            length=3,
            radius=0.06,
            feed_gap=0.2,
            feed_length=0.7,
        )
        yagi = Yagi()

        self.embed()


def get_olympex_rhi_cells():
    path = Path(__file__).parent / "static" / "olympex_rhi_20151113_020003_az210.npz"
    with np.load(path) as data:
        ground = data["ground_edges_km"]
        height = data["height_edges_km"]
        edges = np.stack((ground, height), axis=-1)
        corners = np.stack(
            (edges[:-1, :-1], edges[:-1, 1:], edges[1:, 1:], edges[1:, :-1]),
            axis=2,
        )
        values = data["reflectivity_dbz"]
        valid = (
            np.isfinite(values)
            & data["valid_mask"]
            & (corners[..., 0].min(axis=2) < 32.8)
            & (corners[..., 0].max(axis=2) > 0)
            & (corners[..., 1].min(axis=2) < 8)
            & (corners[..., 1].max(axis=2) > 0)
        )
        ray_indices, gate_indices = np.where(valid)
        corners = np.clip(corners[valid], [0, 0], [32.8, 8])
        rgba = (
            field_to_rgba(
                values,
                vmin=10,
                vmax=50,
                cmap_name="NWSRef",
                fallback_cmap_name="NWSRef",
            )[valid]
            / 255
        )
        return {
            "corners": corners,
            "rgba": rgba,
            "reflectivity_dbz": values[valid],
            "elevation_deg": data["elevation_deg"][ray_indices],
            "ranges_km": data["ranges_km"][gate_indices],
        }


class IntroP2v2(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        rabbit_ears = (
            ImageMobject("./static/rabbit_ears.jpg")
            .rescale_to_fit(fw(self, 0.3), 0)
            .move_to(self.frame)
        )

        self.play(rabbit_ears.shift(LEFT * fw(self)).animate.shift(RIGHT * fw(self)))

        self.wait(0.5)

        raindrop = RainDrop().move_to(self.frame).shift(RIGHT * fw(self))

        self.add(raindrop)

        group = Group(rabbit_ears, raindrop)

        self.play(group.animate.arrange(RIGHT, buff=LARGE_BUFF * 3))

        self.wait(0.5)

        dipole = DipoleAntenna(
            length=3,
            radius=0.06,
            feed_gap=0.2,
            feed_length=0.7,
        ).next_to(group, DOWN, buff=LARGE_BUFF * 3)

        def get_rotation_updater(speed, axis):
            def updater(m, dt):
                m.rotate(dt * speed, axis=axis)

            return updater

        rotation_speed = PI / 6

        self.add(dipole)

        dipole.add_updater(get_rotation_updater(rotation_speed, UP), call=False)

        ears_bez = CubicBezier(
            rabbit_ears.get_bottom() + [0, -0.1, 0],
            rabbit_ears.get_bottom() + [0, -1, 0],
            dipole.get_top() + [0, 1, 0],
            dipole.get_top() + [0, 0.1, 0],
        )
        raindrop_bez = CubicBezier(
            raindrop.get_bottom() + [0, -0.1, 0],
            raindrop.get_bottom() + [0, -1, 0],
            dipole.get_top() + [0, 1, 0],
            dipole.get_top() + [0, 0.1, 0],
        )

        dipole_group = Group(group, dipole)

        self.play(
            ShowCreation(ears_bez),
            ShowCreation(raindrop_bez),
            self.frame.animate.rescale_to_fit(dipole_group.get_height() * 1.2, 1).set_y(
                dipole_group.get_y()
            ),
        )

        self.wait(2)

        self.play(
            LaggedStart(
                LaggedStart(
                    Uncreate(raindrop_bez),
                    Uncreate(ears_bez),
                    rabbit_ears.animate.shift(LEFT * fw(self)),
                    raindrop.animate.shift(RIGHT * fw(self)),
                    lag_ratio=0,
                ),
                dipole.animate.set_y(self.frame.get_y()),
                lag_ratio=0.3,
            )
        )

        self.wait(8)

        self.play(dipole.animate.shift(UP * fh(self)))

        self.remove(dipole, rabbit_ears, raindrop)

        self.wait(0.5)

        radar = get_wsr88_model(height=fh(self), indexed_radome_panels=True)
        azimuth = np.arctan2(radar.beam_direction[1], radar.beam_direction[0])
        radar.rotate(-azimuth, axis=OUT, about_point=ORIGIN)
        radar.shift(DOWN * fh(self, 2))
        beam_origin = radar.beam_origin
        radar.base_sections[-1].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-2].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-3].set_opacity(0).deactivate_depth_test()

        self.add(radar)

        panel_indices = list(np.arange(len(radar.upper_dome)))
        panel_rng = np.random.default_rng(0)
        panel_rng.shuffle(panel_indices)

        panel_fade = LaggedStart(
            *[radar.upper_dome[idx].animate.set_opacity(0) for idx in panel_indices],
            lag_ratio=0.01,
        )

        self.play(
            LaggedStart(
                self.frame.animate(run_time=4).reorient(
                    0, 80, 0, center=radar.antenna.get_center(), height=8
                ),
                LaggedStart(
                    panel_fade,
                    radar.el @ (20 * DEG),
                    radar.az @ (-30 * DEG),
                    lag_ratio=0.5,
                    run_time=2,
                ),
                lag_ratio=0.7,
            )
        )

        self.wait(0.5)

        radar.upper_dome.deactivate_depth_test()

        self.wait(0.5)

        beam = radar.get_beam(8 * DEG, 12).deactivate_depth_test()
        arc_range = 6.5
        bw_arc = beam.get_beamwidth_arc(arc_range, self.frame)

        self.play(
            LaggedStart(
                ShowCreation(beam),
                self.frame.animate.shift(RIGHT * 4 + OUT * 2),
                lag_ratio=0.15,
                run_time=2,
            )
        )

        self.wait(0.5)

        bw_label = (
            Tex(r"\theta_{3 \text{dB}}")
            .fix_in_frame()
            .next_to(
                self.frame.to_fixed_frame_point(bw_arc.get_center()),
                RIGHT,
                MED_SMALL_BUFF,
            )
        )

        self.play(LaggedStart(ShowCreation(bw_arc), FadeIn(bw_label), lag_ratio=0.3))

        self.wait(0.5)

        rres = 1
        gate_start = VT(-rres)
        gate_opacity = VT(0.5)
        gate_out_shift = VT(0)
        gate = always_redraw(
            lambda: beam.get_section(
                max(0, ~gate_start),
                max(0.01, ~gate_start + rres),
                color=GATE_COLOR,
                opacity=~gate_opacity,
                shading=(0.08, 0.18, 0.12),
                depth_test=False,
                resolution=(36, 8),
            ).shift(OUT * ~gate_out_shift)
        )

        self.add(gate)

        self.play(
            LaggedStart(
                beam.animate.set_opacity(0.05),
                gate_start @ (arc_range - rres),
                lag_ratio=0.3,
            )
        )

        rres_dim = DimensionLine(
            gate,
            direction=beam.get_direction(),
            camera_frame=self.frame,
            tick_length=0.2,
            label=Tex(r"\Delta R").scale(0.7),
        ).deactivate_depth_test()

        self.play(rres_dim.get_creation_animation(label_anim=FadeIn))

        self.wait(0.5)

        vres_label = Tex(r"\Delta V").fix_in_frame().set_z_index(10)

        def place_vres_label(m):
            view = self.frame.get_view_matrix()
            points = gate.get_points() @ view[:3, :3].T + view[:3, 3]
            perspective = (
                1
                - points[:, 2]
                * self.frame.get_scale()
                / self.frame.get_focal_distance()
            )
            projected = points[:, :2] / perspective[:, None]
            top = np.array(
                [
                    0.5 * (projected[:, 0].min() + projected[:, 0].max()),
                    projected[:, 1].max(),
                    0,
                ]
            )
            m.next_to(top, UP, MED_SMALL_BUFF)

        place_vres_label(vres_label)

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(rres_dim), FadeOut(bw_arc), FadeOut(bw_label), lag_ratio=0.1
                ),
                FadeIn(vres_label, shift=UP),
                lag_ratio=0.5,
            )
        )

        vres_label.add_updater(place_vres_label)

        self.wait(0.5)

        # self.remove(beam_particles)
        beam_particles = beam.get_section_particles(
            arc_range - rres,
            arc_range,
            count=40,
            radius=(0.02, 0.08),
            seed=1,
            max_radius_fraction=0.9,
        )
        # self.add(beam_particles)

        self.play(ShowCreation(beam_particles))

        self.wait(0.5)

        pulse_length = 2 * rres
        n_oscillations = 4
        wavelength = pulse_length / n_oscillations
        wave_speed = 0.5
        pulse_range = VT(0)
        beam_direction = beam.get_direction()
        beam_origin = beam.get_origin()
        particle_points = beam_particles.get_points().copy()
        particle_radii = beam_particles.get_radii().ravel().copy()
        particle_ranges = (particle_points - beam_origin) @ beam_direction
        return_distances = np.linalg.norm(particle_points - beam_origin, axis=1)
        particle_rgba = beam_particles.data["rgba"].copy()
        highlight_rgba = np.array([*color_to_rgb(YELLOW), 1])

        em_wave = beam.get_em_wave(
            wavelength=wavelength,
            n_oscillations=n_oscillations,
            beamwidth=beam.beamwidth,
            e_color=TX_COLOR,
            h_color=BLUE_A,
            stroke_width=3,
        ).set_z_index(2)

        scattered_waves = Group()
        for point, radius, distance in zip(
            particle_points, particle_radii, return_distances
        ):
            return_beam = Beam(
                2 * DEG,
                distance,
                origin=point,
                direction=beam_origin - point,
                resolution=(8, 2),
            )
            scattered_wave = return_beam.get_em_wave(
                wavelength=wavelength,
                n_oscillations=n_oscillations,
                beamwidth=2 * DEG,
                amplitude=0.025 + 0.5 * radius,
                e_color=RX_COLOR,
                h_color=RED_A,
                stroke_width=1.5,
                stroke_opacity=0.65,
            )
            scattered_waves.add(scattered_wave)
        scattered_waves.set_z_index(3)

        def update_energy(m):
            em_wave.range @= ~pulse_range
            for wave, distance in zip(scattered_waves, particle_ranges):
                wave.range @= ~pulse_range - distance
            since_hit = ~pulse_range - particle_ranges
            highlight = np.where(
                since_hit >= 0, np.exp(-np.maximum(since_hit, 0) / 0.18), 0
            )
            beam_particles.set_rgba_array(
                interpolate(particle_rgba, highlight_rgba, highlight[:, None])
            )

        pulse_range.add_updater(update_energy)

        self.add(pulse_range, em_wave, scattered_waves)

        camera_rotation = self.frame.get_inverse_camera_rotation_matrix()
        energy_center = (
            interpolate(radar.antenna.get_center(), gate.get_center(), 0.65)
            + 0.5 * camera_rotation[0]
            + 0.1 * camera_rotation[1]
        )
        pulse_end = (
            max(beam.get_range(), (particle_ranges + return_distances).max())
            + pulse_length
            + 0.05
        )
        pulse_travel = pulse_range @ pulse_end
        pulse_travel.set_anim_args(run_time=pulse_end / wave_speed, rate_func=linear)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=3.5)
                .set_height(fh(self, 0.55))
                .move_to(energy_center),
                pulse_travel,
                lag_ratio=0.65,
            ),
            (gate_opacity @ 0.18).set_anim_args(run_time=3.5),
        )

        pulse_range.clear_updaters()

        self.remove(pulse_range, em_wave, scattered_waves)

        beam_particles.set_rgba_array(particle_rgba)

        self.wait(2)

        rhi_cells = get_olympex_rhi_cells()
        rhi_origin = beam.get_origin().copy()
        rhi_up = radar.get_scan_up_direction()
        rhi_horizontal = beam_direction - np.dot(beam_direction, rhi_up) * rhi_up
        rhi_horizontal /= np.linalg.norm(rhi_horizontal)
        rhi_normal = np.cross(rhi_horizontal, rhi_up)
        rhi_basis = np.column_stack((rhi_horizontal, rhi_up, rhi_normal))
        vertical_exaggeration = 2
        rhi_corners = (
            rhi_origin
            + rhi_cells["corners"][..., 0, None] * rhi_horizontal
            + vertical_exaggeration * rhi_cells["corners"][..., 1, None] * rhi_up
        )
        selected_cell = np.argmin(
            ((rhi_cells["elevation_deg"] - ~radar.el / DEG) / 0.2) ** 2
            + ((rhi_cells["ranges_km"] - (~gate_start + rres / 2)) / 0.125) ** 2
        )
        pixel_color = rgb_to_color(rhi_cells["rgba"][selected_cell, :3])
        pixel = (
            Polygon(
                *rhi_corners[selected_cell],
                fill_color=pixel_color,
                fill_opacity=1,
                stroke_width=0,
            )
            .deactivate_depth_test()
            .set_z_index(2)
        )

        gate.clear_updaters()
        vres_label.clear_updaters()

        self.play(
            gate.animate.set_color(pixel_color).set_opacity(1).set_shading(0, 0, 0),
            FadeOut(beam_particles),
            FadeOut(vres_label),
            run_time=2,
        )

        self.wait(0.75)

        rhi_center = rhi_origin + 16.5 * rhi_horizontal + 7.5 * rhi_up
        rhi_camera_azimuth = np.arctan2(rhi_horizontal[1], rhi_horizontal[0]) / DEG

        self.play(
            self.frame.animate.reorient(
                rhi_camera_azimuth, 90, 0, center=rhi_center, height=23.6
            ),
            radar.animate.scale(0.08, about_point=rhi_origin),
            FadeTransform(gate, pixel),
            FadeOut(beam),
            run_time=6,
        )

        self.wait(0.5)

        def get_rhi_label(text, point, font_size=24, rotation=0):
            position = self.frame.to_fixed_frame_point(
                rhi_origin + point[0] * rhi_horizontal + point[1] * rhi_up
            )
            return (
                Text(text, font=FONT, font_size=font_size, fill_border_width=0)
                .rotate(rotation)
                .fix_in_frame()
                .move_to(position)
            )

        rhi_axes = VGroup()
        rhi_labels = VGroup()
        for x in (0, 5, 10, 15, 20, 25, 30, 32.8):
            rhi_axes.add(
                Line(
                    [x, 0, 0],
                    [x, 16, 0],
                    stroke_color=WHITE,
                    stroke_width=1,
                    stroke_opacity=0.12,
                ),
            )
            rhi_labels.add(get_rhi_label(f"{x:g}", [x, -0.65]))
        for z in (0, 2, 4, 6, 8):
            rhi_axes.add(
                Line(
                    [0, 2 * z, 0],
                    [32.8, 2 * z, 0],
                    stroke_color=WHITE,
                    stroke_width=1,
                    stroke_opacity=0.12,
                ),
            )
            rhi_labels.add(get_rhi_label(str(z), [-0.65, 2 * z]))
        rhi_labels.add(
            get_rhi_label("Range (km)", [16.4, -1.65], 28),
            get_rhi_label("Height (km)", [-1.6, 8], 28, PI / 2),
            get_rhi_label(
                "NPOL / OLYMPEX / 13 Nov 2015 02:00 UTC / az 210 deg",
                [16.4, 17],
                26,
            ),
        )
        colorbar_rgba = (
            field_to_rgba(
                np.linspace(10, 50, 128)[:, None],
                vmin=10,
                vmax=50,
                cmap_name="NWSRef",
                fallback_cmap_name="NWSRef",
            )[:, 0]
            / 255
        )
        for index, rgba in enumerate(colorbar_rgba):
            rhi_axes.add(
                Rectangle(
                    width=0.45,
                    height=16 / 128 + 0.002,
                    fill_color=rgb_to_color(rgba[:3]),
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([34, (index + 0.5) * 16 / 128, 0])
            )
        for value in (10, 20, 30, 40, 50):
            rhi_labels.add(get_rhi_label(str(value), [34.9, (value - 10) * 0.4]))
        rhi_labels.add(get_rhi_label("dBZ", [34, 16.8], 26))
        rhi_axes.apply_matrix(rhi_basis).shift(rhi_origin)
        rhi_axes.deactivate_depth_test().set_z_index(3)
        rhi_labels.set_z_index(5)

        triangle_corners = [0, 1, 2, 0, 2, 3]
        rhi_points = rhi_corners[:, triangle_corners].reshape(-1, 3)
        rhi_rgba = np.repeat(rhi_cells["rgba"], 6, axis=0)
        vertex_elevations = np.repeat(rhi_cells["elevation_deg"], 6)
        rhi_image = (
            TriangleMeshMobject(
                rhi_points,
                np.broadcast_to(rhi_normal, rhi_points.shape),
                rhi_rgba,
            )
            .set_shading(0, 0, 0)
            .deactivate_depth_test()
            .set_z_index(1)
        )
        scan_elevation = VT(0)
        scan_extent = VT(0)

        def update_rhi(m):
            rgba = rhi_rgba.copy()
            rgba[:, 3] *= vertex_elevations <= ~scan_extent
            m.set_rgba_array(rgba)

        def get_scan_points():
            ranges = np.linspace(0, np.hypot(32.8, 8), 256)
            elevation = ~scan_elevation * DEG
            earth_radius = 6371 * 4 / 3
            heights = (
                np.sqrt(
                    earth_radius**2
                    + ranges**2
                    + 2 * earth_radius * ranges * np.sin(elevation)
                )
                - earth_radius
            )
            ground = earth_radius * np.arcsin(
                ranges * np.cos(elevation) / (earth_radius + heights)
            )
            inside = (ground <= 32.8) & (heights <= 8)
            end = np.flatnonzero(~inside)
            stop = int(end[0]) if len(end) else len(ranges) - 1
            ground, heights = ground[: stop + 1], heights[: stop + 1]
            if stop > 0 and not inside[stop]:
                alpha = min(
                    (32.8 - ground[-2]) / max(ground[-1] - ground[-2], 1e-9),
                    (8 - heights[-2]) / max(heights[-1] - heights[-2], 1e-9),
                    1,
                )
                ground[-1] = interpolate(ground[-2], ground[-1], alpha)
                heights[-1] = interpolate(heights[-2], heights[-1], alpha)
            return (
                rhi_origin
                + ground[:, None] * rhi_horizontal
                + vertical_exaggeration * heights[:, None] * rhi_up
            )

        def point_scanner(m):
            radar.point_at(get_scan_points()[-1])

        scan_line = always_redraw(
            lambda: (
                VMobject(fill_opacity=0)
                .set_points_as_corners(get_scan_points())
                .set_stroke(WHITE, 1.6, 0.75)
                .deactivate_depth_test()
                .set_z_index(4)
            )
        )
        rhi_image.add_updater(update_rhi)
        scan_elevation.add_updater(point_scanner)

        self.add(rhi_image, scan_extent, scan_elevation)

        self.play(FadeIn(rhi_axes), FadeIn(rhi_labels), FadeIn(scan_line), run_time=1.5)

        self.play(
            scan_elevation @ 45,
            scan_extent @ 45,
            run_time=12,
            rate_func=linear,
        )

        self.remove(pixel)

        scan_line.clear_updaters()

        self.play(FadeOut(scan_line), run_time=0.75)

        scan_elevation.clear_updaters()
        rhi_image.clear_updaters()

        self.remove(scan_elevation, scan_extent)

        self.wait(0.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        self.embed()


class Rayleigh(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        drop = Sphere(
            radius=1,
            color=BLUE,
            resolution=(24, 16),
            shading=(0.1, 0.25, 0.15),
        ).set_opacity(1)

        self.play(GrowFromCenter(drop))

        self.wait(0.5)

        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            width=drop.get_width(),
            height=drop.get_height(),
            axis_config={
                "stroke_width": DEFAULT_STROKE_WIDTH * 0.5,
                "tick_size": 0.04,
                "scale_stroke_with_zoom": True,
            },
        )

        A = VT(1)
        default_f = 4
        f = VT(4)
        xshift = VT(-15)
        xlen = VT(1)
        ei = always_redraw(
            lambda: ax.get_graph(
                lambda t: ~A * np.sin(2 * PI * ~f * t),
                x_range=[0, default_f * ~xlen / ~f, 1 / 200],
                color=TX_COLOR,
            ).shift(RIGHT * ~xshift)
        )
        self.add(ei)

        def get_drop_overlap():
            wave_x_max = default_f * ~xlen / ~f
            overlap_x_min = max(
                0,
                ax.p2c(drop.get_left() - RIGHT * ~xshift)[0],
            )
            overlap_x_max = min(
                wave_x_max,
                ax.p2c(drop.get_right() - RIGHT * ~xshift)[0],
            )

            if overlap_x_min >= overlap_x_max:
                return VMobject()

            return (
                ax.get_graph(
                    lambda t: ~A * np.sin(2 * PI * ~f * t),
                    x_range=[overlap_x_min, overlap_x_max, 1 / 200],
                    color=ACCENT_COLOR,
                )
                .shift(RIGHT * ~xshift)
                .set_stroke(width=DEFAULT_STROKE_WIDTH * 1.5)
                .set_z_index(1)
            )

        drop_overlap = always_redraw(get_drop_overlap)
        self.add(drop_overlap)

        wlo = VT(0)
        wlol = VT(0)
        wlor = VT(0)
        wl_line = always_redraw(
            lambda: (
                Line(
                    ax.c2p(0, -~A - 0.2), ax.c2p(default_f * ~xlen / ~f / 4, -~A - 0.2)
                )
                .shift(RIGHT * ~xshift)
                .set_opacity(~wlo)
            )
        )
        wl_line_l = always_redraw(
            lambda: (
                Line(ax.c2p(0, -~A - 0.1), ax.c2p(0, -~A - 0.3))
                .shift(RIGHT * ~xshift)
                .set_opacity(~wlol)
            )
        )
        wl_line_r = always_redraw(
            lambda: (
                Line(
                    ax.c2p(default_f * ~xlen / ~f / 4, -~A - 0.1),
                    ax.c2p(default_f * ~xlen / ~f / 4, -~A - 0.3),
                )
                .shift(RIGHT * ~xshift)
                .set_opacity(~wlor)
            )
        )
        wl_label = always_redraw(
            lambda: Tex(r"\lambda").set_opacity(~wlo).next_to(wl_line, DOWN, SMALL_BUFF)
        )
        self.add(wl_line, wl_line_l, wl_line_r, wl_label)

        self.play(
            self.frame.animate.shift(LEFT * 4),
            xshift.animate(run_time=2).set_value(-5),
        )

        self.wait(0.5)

        self.play(LaggedStart(wlol @ 1, wlo @ 1, wlor @ 1, lag_ratio=0.2))

        self.wait(0.5)

        self.play(LaggedStart(f / 2, xshift - 2, lag_ratio=0.1))

        self.wait(0.5)

        coords = [
            [0.9036873, 0.25698839, -0.34249998],
            [-0.42819303, 0.54236555, -0.72283493],
            [0.0, 0.79987285, 0.60016949],
        ]
        orientation = Rotation.from_matrix(coords)

        self.play(
            LaggedStart(
                self.frame.animate.move_to(ORIGIN),
                xshift.animate(run_time=2).set_value(0),
                drop.animate.set_opacity(0.2),
                FadeIn(ax),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(self.frame.animate.set_orientation(orientation).set_height(5))

        self.wait(0.5)

        self.play(
            drop.animate.scale(0.25),
            f / 4,
            self.frame.animate.reorient(
                0, 0, 0, center=drop.get_center(), height=2.5
            ).shift(UP * 0.5),
            run_time=4,
        )

        self.wait(0.5)

        wave_x_max = default_f * ~xlen / ~f
        overlap_x_min = max(
            0,
            ax.p2c(drop.get_left() - RIGHT * ~xshift)[0],
        )
        overlap_x_max = min(
            wave_x_max,
            ax.p2c(drop.get_right() - RIGHT * ~xshift)[0],
        )

        amp_line = Line(
            [ax.i2gp(overlap_x_max, ei)[0], ax.i2gp(overlap_x_min, ei)[1], 0],
            [ax.i2gp(overlap_x_max, ei)[0], ax.c2p(0, 1)[1], 0],
        ).shift(RIGHT * 0.2)
        amp_line_u = DashedLine(
            amp_line.get_end() + RIGHT * 0.05,
            amp_line.get_end() + LEFT * 1,
            dash_length=DEFAULT_DASH_LENGTH / 1.4,
        )
        amp_line_d = DashedLine(
            amp_line.get_start() + RIGHT * 0.05,
            amp_line.get_start() + LEFT * 1,
            dash_length=DEFAULT_DASH_LENGTH / 1.4,
        )
        delA = Tex(r"\Delta A").scale(0.3).next_to(amp_line, RIGHT, SMALL_BUFF)

        phase_line = Line(
            ax.i2gp(overlap_x_min, ei),
            [ax.i2gp(overlap_x_max, ei)[0], ax.i2gp(overlap_x_min, ei)[1], 0],
        ).shift(DOWN * 0.1)
        phase_line_l = DashedLine(
            phase_line.get_start() + DOWN * 0.05,
            phase_line.get_start() + UP * 0.3,
            dash_length=DEFAULT_DASH_LENGTH / 1.4,
        )
        phase_line_r = DashedLine(
            phase_line.get_end() + DOWN * 0.05,
            phase_line.get_end() + UP * 0.3,
            dash_length=DEFAULT_DASH_LENGTH / 1.4,
        )
        phase_label = (
            Tex(r"\Delta \varphi").scale(0.3).next_to(phase_line, DOWN, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                ShowCreation(amp_line),
                LaggedStart(
                    ShowCreation(amp_line_u),
                    ShowCreation(amp_line_d),
                    lag_ratio=0,
                ),
                FadeIn(delA),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShowCreation(phase_line),
                LaggedStart(
                    ShowCreation(phase_line_l),
                    ShowCreation(phase_line_r),
                    lag_ratio=0,
                ),
                FadeIn(phase_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        left_arrow = CurvedArrow(
            self.frame.to_fixed_frame_point(drop.get_top()),
            self.frame.to_fixed_frame_point(drop.get_left()),
            angle=PI / 3,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).fix_in_frame()
        right_arrow = CurvedArrow(
            self.frame.to_fixed_frame_point(drop.get_top()),
            self.frame.to_fixed_frame_point(drop.get_right()),
            angle=-PI / 3,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).fix_in_frame()

        self.play(ShowCreation(left_arrow), ShowCreation(right_arrow), run_time=3)

        self.wait(0.5)

        rayleigh = (
            Text("Rayleigh Regime", font=FONT)
            .scale(0.6)
            .next_to(ax, UP, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(left_arrow),
                    FadeOut(right_arrow),
                    lag_ratio=0,
                ),
                self.frame.animate.shift(UP * 0.4),
                Write(rayleigh),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.next_to(ei, RIGHT, LARGE_BUFF), run_time=3)

        self.embed()


def get_dipole_handoff_frame():
    return CameraFrame().reorient(0, 0, 0, center=ORIGIN, height=DIPOLE_HANDOFF_HEIGHT)


class Dipole(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        antenna_base = Line(DOWN, ORIGIN)
        antenna_tri = (
            Triangle(fill_opacity=1, fill_color=BACKGROUND_COLOR)
            .rotate(PI / 3)
            .rescale_to_fit(antenna_base.get_length() * 1, 1)
            .round_corners(radius=0.05)
            .set_z_index(2)
        )
        antenna_base.next_to(antenna_tri, DOWN, 0).set_z_index(1)

        self.play(
            LaggedStart(
                ShowCreation(antenna_base),
                ShowCreation(antenna_tri),
                lag_ratio=0.3,
            )
        )
        self.add(antenna_tri, antenna_base)

        self.wait(0.5)

        nrads = 5
        circ_prog = VT(0)

        def get_rad_updater(lag=0.3, max_width=5, fade=False):
            def updater(m):
                prog = min(max(~circ_prog - lag, 0.0001), 1)
                opacity = 1
                if fade:
                    opacity = 1 - prog
                m.rescale_to_fit(prog * max_width, 0).set_z_index(-1).set_stroke(
                    opacity=opacity
                )
                if prog >= 1:
                    self.remove(m)

            return updater

        lag = 0.2
        max_prog = 1 + nrads * lag
        rads = VGroup(
            [
                Circle(stroke_color=TX_COLOR)
                .set_z_index(-1)
                .move_to(antenna_tri)
                .add_updater(get_rad_updater(lag=idx * lag, max_width=fw(self, 1.5)))
                for idx in range(nrads)
            ]
        )

        self.add(rads)

        self.play(circ_prog @ max_prog, run_time=8)

        self.wait(0.5)

        self.radar, self.beam_origin = place_wsr88_model(height=fh(self, 1), shift=0)

        self.radar.shift(IN * 30).rotate(-PI / 2, axis=RIGHT, about_point=ORIGIN)

        tower_parts = [
            *self.radar.base_sections[:2][::-1],
            self.radar.fixed_body,
            self.radar.antenna,
        ]
        for part in tower_parts:
            part.set_opacity(0)
        self.radar.dome.set_opacity(0)
        self.radar.dome.deactivate_depth_test()

        radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in tower_parts],
            lag_ratio=0.3,
            run_time=2,
        )
        radar_build.group = self.radar
        radar_build.mobject = self.radar

        radar_destroy = LaggedStart(
            *[part.animate.set_opacity(0) for part in tower_parts],
            lag_ratio=0.3,
            run_time=2,
        )
        radar_destroy.group = self.radar
        radar_destroy.mobject = self.radar

        self.play(
            LaggedStart(
                self.frame.animate.reorient(
                    0, -20, 0, center=self.radar.upper_dome.get_center(), height=8
                ),
                radar_build,
                self.radar.az @ (-45 * DEG),
                self.radar.el @ (20 * DEG),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.radar.reflector.animate.set_color(BLUE))

        self.wait(0.5)

        human = (
            ImageMobject("../props/static/human_outline_white.png")
            .rescale_to_fit(fh(self, 0.6), 1)
            .fix_in_frame()
            .shift(RIGHT * 4)
        )

        router = (
            Router()
            .move_to(self.radar.upper_dome)
            .shift(RIGHT * 12)
            .rotate(-PI / 2, axis=RIGHT)
            .rotate(PI, axis=UP)
            .set_z_index(1)
        )

        self.add(router)

        self.play(
            LaggedStart(
                radar_destroy,
                self.frame.animate(run_time=3)
                .reorient(0, -10, 0, center=router.get_center(), height=8)
                .shift(RIGHT * 3 + UP * 2),
                router.antenna_angle.animate(run_time=2).set_value(100 * DEG),
                human.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        nrads = 5
        circ_prog = VT(0)
        lag = 0.2
        router_rads_r = VGroup(
            [
                Circle(stroke_color=TX_COLOR)
                .set_z_index(-1)
                .move_to(router.antennas[0].get_top())
                .rotate(-100 * DEG, axis=RIGHT)
                .shift(DOWN * np.sin(100 * DEG) + IN * np.cos(100 * DEG))
                .add_updater(
                    get_rad_updater(lag=idx * lag, max_width=fw(self, 1.5), fade=True)
                )
                for idx in range(nrads)
            ]
        )
        router_rads_l = VGroup(
            [
                Circle(stroke_color=TX_COLOR)
                .set_z_index(-1)
                .move_to(router.antennas[1].get_top())
                .rotate(-100 * DEG, axis=RIGHT)
                .shift(DOWN * np.sin(100 * DEG) + IN * np.cos(100 * DEG))
                .add_updater(
                    get_rad_updater(lag=idx * lag, max_width=fw(self, 1.5), fade=True)
                )
                for idx in range(nrads)
            ]
        )

        shoulder = self.frame.from_fixed_frame_point(
            human.get_corner(UL)
            + RIGHT * 0.46 * human.get_width()
            + DOWN * 0.33 * human.get_height()
        )
        rad_dist = min(
            get_norm(shoulder - rad.get_center())
            for rad in [router_rads_r[0], router_rads_l[0]]
        )
        human_lag = 2 * rad_dist / fw(self, 1.5)
        human_rads = VGroup(
            [
                Circle(stroke_color=RX_COLOR)
                .set_z_index(-1)
                .rotate(-100 * DEG, axis=RIGHT)
                .move_to(shoulder)
                .add_updater(
                    get_rad_updater(
                        lag=human_lag + idx * lag, max_width=fw(self, 1.5), fade=True
                    )
                )
                for idx in range(nrads)
            ]
        )

        self.add(router_rads_r, router_rads_l, human_rads)

        max_prog = 1 + nrads * lag + human_lag
        self.play(circ_prog @ max_prog, run_time=4 * max_prog)

        self.wait(0.5)

        self.play(FadeOut(human), FadeOut(router))

        self.wait(0.5)

        raindrop = RainDrop().move_to(router.get_center()).shift(UP * 10)
        self.add(raindrop)

        self.play(
            self.frame.animate(run_time=3).reorient(
                0, 0, 0, center=raindrop.get_center(), height=6
            ),
        )

        self.wait(0.5)

        out_rotation = PI / 6
        rotation_speed = PI / 6
        yagi = (
            Yagi()
            .scale(0.6)
            .move_to(raindrop)
            .shift(RIGHT * 4 + UP * 1.5)
            .rotate(out_rotation, axis=OUT)
            .shift(RIGHT * 10)
        )
        yagi_axis = Line(LEFT, RIGHT).rotate(out_rotation, axis=OUT).get_vector()

        def get_rotation_updater(speed, axis):
            def updater(m, dt):
                m.rotate(dt * speed, axis=axis)

            return updater

        yagi.add_updater(get_rotation_updater(rotation_speed, yagi_axis), call=False)

        dipole = (
            DipoleAntenna(
                length=3,
                radius=0.06,
                feed_gap=0.2,
                feed_length=0.7,
            )
            .move_to(raindrop)
            .shift(RIGHT * 4 + DOWN * 1)
            .shift(RIGHT * 10)
        )

        dipole.add_updater(get_rotation_updater(rotation_speed, UP), call=False)

        self.play(
            self.frame.animate(run_time=3)
            .reorient(0, 0, 0, center=raindrop.get_center(), height=6)
            .shift(RIGHT * 2.5),
            LaggedStart(
                yagi.animate.shift(LEFT * 10),
                dipole.animate.shift(LEFT * 10),
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        bez_u = CubicBezier(
            self.frame.to_fixed_frame_point(raindrop.get_right()) + [0.1, 0, 0],
            self.frame.to_fixed_frame_point(raindrop.get_right()) + [2.5, 0, 0],
            self.frame.to_fixed_frame_point(yagi.get_left()) + [-2.5, 0, 0],
            self.frame.to_fixed_frame_point(yagi.get_left()) + [-0.1, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).fix_in_frame()
        bez_d = CubicBezier(
            self.frame.to_fixed_frame_point(raindrop.get_right()) + [0.1, 0, 0],
            self.frame.to_fixed_frame_point(raindrop.get_right()) + [2.5, 0, 0],
            self.frame.to_fixed_frame_point(dipole.get_left()) + [-2.5, 0, 0],
            self.frame.to_fixed_frame_point(dipole.get_left()) + [-0.1, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).fix_in_frame()

        self.play(
            ShowCreation(bez_u),
            ShowCreation(bez_d),
        )

        self.wait(8)

        self.play(
            LaggedStart(
                LaggedStart(
                    yagi.animate(run_time=1).shift(UP * 5),
                    Uncreate(bez_u),
                    dipole.animate(run_time=1).shift(DOWN * 5),
                    Uncreate(bez_d),
                    lag_ratio=0,
                ),
                LaggedStart(
                    self.frame.animate(run_time=1.5).reorient(
                        0, 0, 0, center=raindrop.get_center(), height=6
                    ),
                    self.camera.light_source.animate(run_time=1.5).move_to(
                        raindrop.get_center() + self.camera.light_source_position
                    ),
                    lag_ratio=0,
                ),
                lag_ratio=0.3,
            )
        )

        self.clear()
        raindrop = RainDrop().move_to(ORIGIN)
        self.frame.move_to(ORIGIN)
        self.camera.light_source.move_to(self.camera.light_source_position)
        self.add(raindrop)

        self.play(
            self.frame.animate.become(get_dipole_handoff_frame()),
            run_time=2.4,
            rate_func=ease_in_sine,
        )

        self.embed()


class DipoleP2(ThreeDScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}
    default_frame_orientation = (0, 0)
    always_depth_test = False
    samples = 0

    def construct(self):
        raindrop = RainDrop().move_to(ORIGIN)
        self.frame.become(get_dipole_handoff_frame())
        self.add(raindrop)
        self.wait(1 / self.camera.fps)

        rng = np.random.default_rng(7)
        pop_time = 0.28
        pair_centers = []
        for row, y in enumerate(np.linspace(-0.72, 0.72, 6)):
            for x in np.linspace(-0.75, 0.75, 6):
                x += 0.15 * (row % 2)
                if x**2 + y**2 > 0.8**2:
                    continue
                offset = np.array([x, y, 0]) + rng.uniform(-0.01, 0.01, 3)
                offset[2] = 0
                pair_centers.append(
                    raindrop.get_center()
                    + offset * [raindrop.get_width() / 2, raindrop.get_height() / 2, 1]
                )
        rng.shuffle(pair_centers)

        def get_charge_pair(center, angle):
            return ChargePair(
                center=center,
                angle=angle,
                radius=DIPOLE_CLOSEUP_HEIGHT * 0.0175,
                separation=DIPOLE_CLOSEUP_HEIGHT * 0.065,
                font=FONT,
                negative_color=ICE,
                symbol_color=BACKGROUND_COLOR,
            )

        charge_pairs = VGroup(
            *[get_charge_pair(center, rng.uniform(0, TAU)) for center in pair_centers]
        )

        self.play(
            self.frame.animate(run_time=0.6, rate_func=ease_out_sine).set_height(
                DIPOLE_CLOSEUP_HEIGHT
            ),
            raindrop.animate.set_opacity(0.3),
            LaggedStart(
                *[
                    GrowFromCenter(m, run_time=pop_time, rate_func=ease_out_back)
                    for m in charge_pairs
                ],
                lag_ratio=0.08,
            ),
        )

        self.add(charge_pairs)

        drop_scale = VT(1)
        base_width = raindrop.get_width()
        base_count = len(charge_pairs)
        drop_anchor = raindrop.get_center().copy()
        drop_size = np.array([base_width, raindrop.get_height(), 1])

        def get_pair_state(pair, offset, progress=1.0):
            center = pair.get_center()
            return {
                "pair": pair,
                "offset": offset,
                "progress": progress,
                "charge_offsets": np.array(
                    [
                        pair.positive.get_center() - center,
                        pair.negative.get_center() - center,
                    ]
                ),
                "points": [
                    (m, m.get_points() - center)
                    for m in pair.get_family()
                    if m.has_points()
                ],
            }

        pair_states = [
            get_pair_state(m, (m.get_center() - raindrop.get_center()) / drop_size)
            for m in charge_pairs
        ]
        pair_states.sort(key=lambda state: np.linalg.norm(state["offset"]))

        def update_drop(m, dt):
            scale = float(~m)
            if not np.isfinite(scale) or scale < 0:
                raise ValueError("drop_scale must be finite and nonnegative")
            raindrop.set_width(base_width * max(scale, 1e-4), about_point=drop_anchor)
            count = int(np.floor(base_count * scale**2 + 0.5))
            center = raindrop.get_center()

            while len(pair_states) < count:
                angles = rng.uniform(0, TAU, 64)
                radius = 0.4 * np.sqrt((len(pair_states) + 0.5) / base_count)
                candidates = np.column_stack(
                    (radius * np.cos(angles), radius * np.sin(angles), np.zeros(64))
                )
                offsets = np.array([state["offset"] for state in pair_states])
                distances = np.linalg.norm(candidates[:, None] - offsets, axis=2)
                offset = candidates[np.argmax(distances.min(axis=1))]
                angles = rng.uniform(0, TAU, 32)
                half_pairs = (
                    DIPOLE_CLOSEUP_HEIGHT
                    * 0.065
                    / 2
                    * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(32)))
                )
                endpoints = (
                    center
                    + offset * drop_size
                    + np.stack((-half_pairs, half_pairs), axis=1)
                )
                occupied = np.concatenate(
                    [
                        center + state["offset"] * drop_size + state["charge_offsets"]
                        for state in pair_states
                    ]
                )
                clearance = np.linalg.norm(endpoints[:, :, None] - occupied, axis=3)
                angle = angles[np.argmax(clearance.min(axis=(1, 2)))]
                pair = get_charge_pair(center + offset * drop_size, angle)
                pair_states.append(get_pair_state(pair, offset, progress=0.0))

            visible_pairs = []
            for index, state in enumerate(pair_states):
                pair = state["pair"]
                direction = 1 if index < count else -1
                progress = np.clip(state["progress"] + direction * dt / pop_time, 0, 1)
                position = center + state["offset"] * drop_size
                if progress != state["progress"]:
                    for part, points in state["points"]:
                        part.set_points(points * ease_out_back(progress) + position)
                else:
                    pair.move_to(position)
                state["progress"] = progress
                if progress > 0:
                    visible_pairs.append(pair)
            if charge_pairs.submobjects != visible_pairs:
                charge_pairs.set_submobjects(visible_pairs)

        drop_scale.add_updater(update_drop, call=False)
        self.add(drop_scale)

        self.wait(0.5)

        sum_charge = Tex(r"\sum_{i=1}^{N}{q_i} = 0").next_to(
            raindrop, LEFT, LARGE_BUFF * 1.5
        )

        new_group = Group(raindrop, sum_charge)

        self.play(
            LaggedStart(
                self.frame.animate.move_to(new_group).rescale_to_fit(
                    new_group.get_width() * 1.2, 0
                ),
                LaggedStart(*[FadeIn(m) for m in sum_charge], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        tabletop = Rectangle(width=raindrop.get_width() * 1.7, height=0.16)
        legs = VGroup(
            *[
                Rectangle(width=0.16, height=1)
                .next_to(tabletop, DOWN, 0)
                .align_to(tabletop, side)
                .shift(-side * 0.12)
                for side in [LEFT, RIGHT]
            ]
        )
        table = (
            VGroup(tabletop, legs)
            .set_fill("#8B5A2B", opacity=1)
            .set_stroke(width=0)
            .next_to(raindrop, DOWN, MED_SMALL_BUFF)
        )
        table_group = Group(sum_charge, raindrop, table)

        self.play(
            LaggedStart(
                self.frame.animate.move_to(table_group).rescale_to_fit(
                    table_group.get_height() * 1.2, 1
                ),
                FadeIn(table, shift=UP),
                lag_ratio=0.2,
            )
        )
        self.wait(0.5)

        self.play(
            FadeOut(sum_charge),
            FadeOut(table, shift=DOWN * 0.3),
            self.frame.animate.move_to(raindrop).set_height(
                raindrop.get_height() * 2.4
            ),
            run_time=1.5,
        )
        self.wait(0.5)

        self.play(drop_scale @ 2, run_time=2)

        self.play(drop_scale @ 0.7, run_time=2)

        self.play(drop_scale @ 1, run_time=1)

        self.wait(0.5)

        radar = get_wsr88_model(height=fh(self), indexed_radome_panels=True)
        radar.rotate(-PI / 2, axis=RIGHT)
        radar.shift(DOWN * radar.get_height() + UP * 3).set_z(raindrop.get_z()).shift(
            LEFT * 6
        )
        radar.base_sections[-1].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-2].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-3].set_opacity(0).deactivate_depth_test()

        self.add(radar)

        radar.upper_dome.set_opacity(0).deactivate_depth_test()
        radar.point_at(raindrop.get_center())

        self.play(
            self.frame.animate.reorient(
                0, 0, 0, center=Group(raindrop, radar.upper_dome).get_center(), height=6
            )
        )

        self.wait(0.5)

        bw = 32 * DEG
        wl = 4
        beam = radar.get_beam(bw, 10)
        em_wave = beam.get_em_wave(
            wavelength=wl,
            n_oscillations=1,
            beamwidth=bw,
            e_color=TX_COLOR,
            h_color=BLUE_A,
        )
        self.add(em_wave)

        self.play(em_wave.range @ (wl + 0.5), run_time=2)

        self.wait(0.5)

        wl_dim = DimensionLine(
            em_wave,
            beam.get_direction(),
            camera_frame=self.frame,
            tick_length=0.2,
            label=Tex(r"\lambda"),
        )

        D_dim = DimensionLine(
            raindrop,
            beam.get_direction(),
            camera_frame=self.frame,
            tick_length=0.2,
            label=Tex(r"D"),
        )

        self.play(
            LaggedStart(
                wl_dim.get_creation_animation(),
                D_dim.get_creation_animation(),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        dllw = (
            Tex(r"D \ll \lambda \rightarrow")
            .fix_in_frame()
            .scale(1.5)
            .next_to(self.frame.get_top(), DOWN, MED_LARGE_BUFF)
            .shift(LEFT + UP)
        )
        rayleigh = (
            Text("Rayleigh", font=FONT)
            .fix_in_frame()
            .next_to(dllw, RIGHT, MED_SMALL_BUFF)
            .set_y(dllw[-1].get_y())
        )

        self.play(
            LaggedStart(
                TransformFromCopy(D_dim.label, dllw[0], path_arc=PI / 3),
                GrowFromCenter(dllw[1]),
                TransformFromCopy(wl_dim.label, dllw[2], path_arc=PI / 3),
                GrowFromCenter(dllw[3]),
                Write(rayleigh),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        dw_ratio = (
            Tex(r"\frac{D}{\lambda} \lessapprox \frac{1}{10} \rightarrow")
            .fix_in_frame()
            .scale(1.5)
        )
        dw_ratio.shift(dllw[-1].get_center() - dw_ratio[-1].get_center())

        self.play(
            LaggedStart(
                FadeOut(dllw[1]),
                ReplacementTransform(dllw[0], dw_ratio[0]),
                ReplacementTransform(dllw[2], dw_ratio[2]),
                GrowFromCenter(dw_ratio[1]),
                GrowFromCenter(dw_ratio[3]),
                GrowFromCenter(dw_ratio[4]),
                GrowFromCenter(dw_ratio[5]),
                GrowFromCenter(dw_ratio[6]),
                GrowFromCenter(dw_ratio[7]),
                ReplacementTransform(dllw[3], dw_ratio[8]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        all_regimes = (
            Tex(
                r"\frac{D}{\lambda} \begin{cases}"
                r"\lessapprox \frac{1}{10} \rightarrow \\ "
                r"\approx 1 \rightarrow \\  "
                r"\gg 1 \rightarrow"
                r"\end{cases}"
            )
            .fix_in_frame()
            .scale(1.5)
            .move_to(ORIGIN)
        )
        rayleigh_label = (
            Text("Rayleigh", font=FONT)
            .fix_in_frame()
            .next_to(all_regimes[13], RIGHT, MED_SMALL_BUFF)
        )
        mie_label = (
            Text("Mie", font=FONT)
            .fix_in_frame()
            .next_to(all_regimes[16], RIGHT, MED_SMALL_BUFF)
        )
        optical_label = (
            Text("Optical", font=FONT)
            .fix_in_frame()
            .next_to(all_regimes[19], RIGHT, MED_SMALL_BUFF)
        )
        regimes_group = (
            Group(all_regimes, rayleigh_label, mie_label, optical_label).move_to(ORIGIN)
            # .shift(UP * fh(self))
        )

        self.play(
            LaggedStart(
                self.frame.animate.shift(UP * fh(self)),
                ReplacementTransform(dw_ratio[:3], all_regimes[:3]),
                ReplacementTransform(dw_ratio[3:9], all_regimes[8:14]),
                ReplacementTransform(rayleigh, rayleigh_label),
                LaggedStart(*[FadeIn(m) for m in all_regimes[3:8]], lag_ratio=0.1),
                LaggedStart(*[FadeIn(m) for m in all_regimes[14:17]], lag_ratio=0.1),
                Write(mie_label),
                LaggedStart(*[FadeIn(m) for m in all_regimes[17:20]], lag_ratio=0.1),
                Write(optical_label),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        nb_obs = (
            VideoMobject("./static/nb_obs.mov", start=5, end=18)
            .fix_in_frame()
            .move_to(ORIGIN)
            .shift(RIGHT * fw(self))
        )

        self.play(
            Group(regimes_group, nb_obs)
            .animate.scale(0.9)
            .arrange(RIGHT, buff=MED_LARGE_BUFF)
            .move_to(ORIGIN)
        )

        self.wait(1)

        itd_arrow = (
            Arrow(ORIGIN, DOWN * 2).fix_in_frame().next_to(nb_obs, DOWN, MED_SMALL_BUFF)
        )
        itd_label = (
            Text("Description", font=FONT)
            .fix_in_frame()
            .next_to(itd_arrow, LEFT, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                GrowArrow(itd_arrow),
                FadeIn(itd_label),
                lag_ratio=0.3,
            )
        )

        self.wait(4)

        self.play(
            FadeOut(all_regimes),
            FadeOut(rayleigh_label),
            FadeOut(mie_label),
            FadeOut(optical_label),
            FadeOut(nb_obs),
            FadeOut(itd_arrow),
            FadeOut(itd_label),
        )
        self.play(self.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        self.remove(D_dim)
        D_dim = always_redraw(
            lambda: DimensionLine(
                raindrop,
                beam.get_direction(),
                camera_frame=self.frame,
                tick_length=0.2,
                label=Tex(r"D"),
            )
        )
        self.add(D_dim)

        self.play(drop_scale @ 0.36, run_time=2)

        self.wait(0.5)

        overlap_opacity = VT(1)
        overlap_thickness = VT(1)

        def get_wave_overlap():
            field = em_wave.e_field
            overlap = VMobject()
            if not field.has_points():
                return overlap

            direction = beam.get_direction()
            origin = beam.get_origin()
            drop_ranges = (raindrop.get_points() - origin) @ direction
            wave_start = np.dot(field.get_start() - origin, direction)
            wave_end = np.dot(field.get_end() - origin, direction)
            if wave_end <= wave_start:
                return overlap

            start, end = np.clip(
                (np.array([drop_ranges.min(), drop_ranges.max()]) - wave_start)
                / (wave_end - wave_start),
                0,
                1,
            )
            if end <= start:
                return overlap

            return (
                field.get_subcurve(start, end)
                .set_stroke(
                    YELLOW,
                    width=~overlap_thickness * 6,
                    opacity=~overlap_opacity * ~em_wave.e_opacity,
                )
                .deactivate_depth_test()
            )

        wave_overlap = always_redraw(get_wave_overlap)
        self.add(wave_overlap)

        _, _, raindrop_range = radar.p2c(raindrop.get_center())

        d = beam.get_direction()
        e_reference = UP if abs(np.dot(d, UP)) < 0.9 else OUT
        e_direction = e_reference - np.dot(e_reference, d) * d
        e_direction /= np.linalg.norm(e_direction)
        normal = np.cross(d, e_direction)
        camera_rotation = Rotation.from_matrix(
            np.column_stack((d, e_direction, normal))
        )

        self.play(FadeOut(D_dim), FadeOut(wl_dim))

        self.play(
            LaggedStart(
                em_wave.range.animate(run_time=1.5).set_value(
                    raindrop_range + wl * 0.15
                ),
                em_wave.h_opacity.animate(run_time=1.5).set_value(0),
                self.frame.animate.set_orientation(camera_rotation)
                .move_to(raindrop.get_center())
                .set_height(6),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(overlap_thickness.animate(run_time=1.5).set_value(2))

        self.wait(0.5)

        self.play(overlap_thickness.animate(run_time=1.5).set_value(1))

        self.wait(0.5)

        self.play(self.frame.animate.scale(0.6))

        self.wait(0.5)

        self.frame.save_state()
        drop_opacity = raindrop.get_opacity()

        center = raindrop.get_center()
        radius = raindrop.get_width() / 2
        prop_range = VT(em_wave.get_start("aer")[2] + 1e-3)
        prop_opacity = VT(1)
        prop_arrow = always_redraw(
            lambda: Arrow3D(
                em_wave.get_start(),
                radar.c2p(0, 0, ~prop_range),
                color=TX_COLOR,
                shaft_radius=0.006,
                tip_radius=0.025,
                tip_length=0.07,
                depth_test=False,
                opacity=~prop_opacity,
            )
        )

        slice_positions = np.linspace(-radius, radius, 8)
        slice_basis = np.column_stack((e_direction, normal, d))
        section_edges = Group()
        for z in slice_positions[1:-1]:
            slice_radius = np.sqrt(radius**2 - z**2)
            edge = Circle(
                radius=slice_radius,
                fill_color=BLUE_B,
                fill_opacity=0.035,
                stroke_color=BLUE_B,
                stroke_width=1.5,
            )
            edge.apply_matrix(slice_basis).shift(center + z * d)
            section_edges.add(edge)

        sections = Group()
        for start, end in zip(slice_positions[:-1], slice_positions[1:]):
            surface = ParametricSurface(
                lambda u, v: (
                    center
                    + v * d
                    + np.sqrt(max(0, radius**2 - v**2))
                    * (np.cos(u) * e_direction + np.sin(u) * normal)
                ),
                u_range=(0, TAU),
                v_range=(start, end),
                resolution=(48, 8),
                color=YELLOW,
                opacity=0.3,
                depth_test=False,
            )
            caps = Group()
            for z in (start, end):
                if abs(z) >= radius:
                    continue
                cap = Circle(
                    radius=np.sqrt(radius**2 - z**2),
                    fill_color=YELLOW,
                    fill_opacity=0.18,
                    stroke_color=YELLOW,
                    stroke_width=2.5,
                )
                cap.apply_matrix(slice_basis).shift(center + z * d)
                caps.add(cap)
            sections.add(Group(surface, caps))

        slice_rotation = Rotation.from_rotvec(-40 * DEG * e_direction) * camera_rotation

        self.play(
            raindrop.animate.set_opacity(0.15),
            FadeOut(charge_pairs),
            em_wave.e_opacity @ 0.1,
            overlap_opacity @ 0,
            self.frame.animate.set_orientation(slice_rotation)
            .move_to(center)
            .set_height(raindrop.get_height() * 2.3),
            run_time=2,
        )

        raindrop.deactivate_depth_test()

        self.play(FadeIn(section_edges))

        self.add(prop_arrow)

        self.play(
            prop_range @ (raindrop_range + slice_positions[1]),
            FadeIn(sections[0]),
            run_time=1.5,
        )

        self.wait(0.5)

        for z, previous_section, section in zip(
            slice_positions[2:], sections[:-1], sections[1:]
        ):
            self.play(
                prop_range @ (raindrop_range + z),
                FadeOut(previous_section),
                FadeIn(section),
                run_time=0.8,
            )

            self.wait(0.4)

        self.play(
            LaggedStart(prop_range + 4, prop_opacity @ 0, lag_ratio=0.5),
            self.frame.animate(run_time=1.5).set_orientation(
                Rotation.from_rotvec(0 * e_direction) * camera_rotation
            ),
        )

        self.wait(0.5)

        slice_centers = (slice_positions[:-1] + slice_positions[1:]) / 2
        slice_volumes = PI * (
            radius**2 * np.diff(slice_positions) - np.diff(slice_positions**3) / 3
        )
        bar_values = slice_volumes / slice_volumes.max()

        chart_base = center + e_direction * radius * 1.3
        bar_width = np.diff(slice_positions)[0] * 0.85
        bars = VGroup()
        for z, value in zip(slice_centers, bar_values):
            bar_height = radius * value
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=TX_COLOR,
                fill_opacity=0.9,
                stroke_width=0,
            )
            bar.apply_matrix(camera_rotation.as_matrix()).move_to(
                chart_base + z * d + e_direction * bar_height / 2
            )
            bars.add(bar)

        chart_line = Line(
            chart_base - radius * d,
            chart_base + radius * d,
            stroke_width=1.5,
            stroke_opacity=0.5,
        )
        chart_label = Text("Relative amplitude", font=FONT).set_height(radius * 0.17)
        chart_label.apply_matrix(camera_rotation.as_matrix()).move_to(
            center + e_direction * radius * 2.95
        )
        chart_guides = VGroup(
            *[
                DashedLine(
                    center + z * d + e_direction * np.sqrt(radius**2 - z**2),
                    chart_base + z * d,
                    dash_length=radius * 0.06,
                    stroke_width=1,
                    stroke_opacity=0.25,
                )
                for z in slice_positions
            ]
        )

        self.play(
            self.frame.animate.shift(e_direction * radius * 0.9).scale(1.1),
            FadeOut(sections[-1]),
            run_time=1.5,
        )

        self.play(
            Write(chart_label),
            ShowCreation(chart_line),
            LaggedStart(*[ShowCreation(m) for m in chart_guides], lag_ratio=0.1),
        )

        self.play(
            LaggedStart(
                *[
                    GrowFromPoint(bar, chart_base + z * d)
                    for z, bar in zip(slice_centers, bars)
                ],
                lag_ratio=0.2,
            ),
            run_time=2.5,
        )

        self.wait(2)

        mean_value = bar_values.mean()

        self.play(
            *[
                bar.animate.apply_matrix(
                    np.eye(3)
                    + (mean_value / value - 1) * np.outer(e_direction, e_direction),
                    about_point=chart_base + z * d,
                )
                for z, value, bar in zip(slice_centers, bar_values, bars)
            ],
            run_time=2,
        )

        self.wait(2)

        for m in charge_pairs:
            m.scale(0.7)

        self.play(
            LaggedStart(
                FadeOut(bars),
                FadeOut(chart_label),
                Uncreate(chart_line),
                FadeOut(chart_guides),
                FadeOut(section_edges),
                lag_ratio=0.15,
            ),
            raindrop.animate.set_opacity(drop_opacity),
            FadeIn(charge_pairs),
            em_wave.e_opacity @ 1,
            overlap_opacity @ 1,
            self.frame.animate.restore(),
            run_time=2,
        )

        self.remove(prop_arrow)

        self.wait(2)

        charge_response = VT(0)
        center = raindrop.get_center()
        drop_radius = (
            0.98 * np.linalg.norm(raindrop.get_points() - center, axis=1).min()
        )
        charge_points = np.vstack(
            [
                charge.get_all_points() - center
                for m in charge_pairs
                for charge in [m.positive, m.negative]
            ]
        )
        field_offsets = charge_points @ e_direction
        transverse_offsets = charge_points - field_offsets[:, None] * e_direction
        charge_travel = np.sqrt(
            np.maximum(0, drop_radius**2 - np.sum(transverse_offsets**2, axis=1))
        ) - np.abs(field_offsets)
        charge_gain = min(
            raindrop.get_height() * 0.16, 0.8 * max(0, charge_travel.min())
        ) / (raindrop_range * np.tan(bw / 2))

        def update_charge_pair(m):
            center = raindrop.get_center()
            sample_range = np.dot(center - beam.get_origin(), beam.get_direction())
            field = em_wave.get_e_field_at_range(sample_range)
            displacement = ~charge_response * charge_gain * field
            m.positive.move_to(center + m.positive_rest_offset + displacement)
            m.negative.move_to(center + m.negative_rest_offset - displacement)
            m.connector.update(0)

        for m in charge_pairs:
            m.positive_rest_offset = m.positive.get_center() - raindrop.get_center()
            m.negative_rest_offset = m.negative.get_center() - raindrop.get_center()
            m.add_updater(update_charge_pair)

        self.play(charge_response @ 1, run_time=1.5)

        self.wait(0.5)

        self.play(em_wave.pulse_length @ (10 * wl), run_time=1.5)

        self.wait(0.5)

        self.play(
            em_wave.range @ (raindrop_range + ~em_wave.pulse_length),
            run_time=8,
        )

        self.wait(2)

        self.embed()


class Charge(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        self.radar, self.beam_origin = place_wsr88_model(height=fh(self, 1), shift=0)

        tower_parts = [
            *self.radar.base_sections[::-1],
            self.radar.fixed_body,
            self.radar.antenna,
        ]
        for part in tower_parts:
            part.set_opacity(0)
        tower_parts = tower_parts[-5:]
        self.radar.dome.set_opacity(0)
        self.radar.dome.deactivate_depth_test()

        radar_build = LaggedStart(
            *[part.animate.set_opacity(1) for part in tower_parts],
            lag_ratio=0.3,
            run_time=3,
        )
        radar_build.group = self.radar
        radar_build.mobject = self.radar

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    self.frame.animate(run_time=3).reorient(
                        30, 75, 0, center=self.radar.dome.get_center(), height=9
                    ),
                    radar_build,
                    lag_ratio=0,
                ),
                self.radar.el.animate(run_time=3).set_value(20 * DEG),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        raindrop = RainDrop().scale(0.2).move_to(self.radar.c2p(0 * DEG, 0 * DEG, 5))

        bw = 8 * DEG
        beam = self.radar.get_beam(bw, 10)
        em_wave = beam.get_em_wave(
            wavelength=2,
            n_oscillations=1,
            beamwidth=bw,
            stroke_width=DEFAULT_STROKE_WIDTH,
            e_highlight_color=BLUE,
            h_highlight_color=RED,
        )
        self.add(em_wave)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=3).reorient(
                    40,
                    75,
                    0,
                    center=Group(self.radar.dome, raindrop).get_center(),
                    height=9,
                ),
                FadeIn(raindrop),
                # ShowCreation(beam, run_time=3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        def face_camera(m, get_center, scale_tracker):
            m.center()
            local_points = [member.get_points().copy() for member in m.get_family()]

            def update(m):
                rotation = self.frame.get_orientation().as_matrix()
                center = get_center()
                for member, points in zip(m.get_family(), local_points):
                    if len(points):
                        scaled_points = points * ~scale_tracker
                        member.set_points(scaled_points @ rotation.T + center)

            return m.add_updater(update)

        rng = np.random.default_rng(0)
        npluses = 8
        directions = rng.normal(size=(npluses, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        offsets = directions * rng.uniform(0.2, 0.3, size=(npluses, 1))
        plus_scales = [VT(1) for _ in range(npluses)]

        back_pluses = VGroup(
            *[
                face_camera(
                    Text("+", font=FONT, fill_color=YELLOW)
                    .scale(0.25)
                    .deactivate_depth_test(),
                    lambda offset=offset: raindrop.get_center() + offset,
                    scale_tracker,
                )
                for offset, scale_tracker in zip(offsets, plus_scales)
            ]
        )
        pluses = VGroup(
            *[
                face_camera(
                    Text("+", font=FONT, fill_color=YELLOW)
                    .scale(0.25)
                    .apply_depth_test(),
                    lambda offset=offset: raindrop.get_center() + offset,
                    scale_tracker,
                )
                for offset, scale_tracker in zip(offsets, plus_scales)
            ]
        )
        self.remove(raindrop)
        self.add(back_pluses, raindrop, pluses)

        self.play(
            LaggedStart(
                em_wave.range.animate(run_time=6).set_value(10),
                self.frame.animate(run_time=4).reorient(
                    20, 90, 0, center=raindrop.get_center(), height=2
                ),
                LaggedStart(
                    *[
                        AnimationGroup(
                            FadeIn(back_plus),
                            FadeIn(plus),
                            lag_ratio=0,
                        )
                        for back_plus, plus in zip(back_pluses, pluses)
                    ],
                    lag_ratio=0.15,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(em_wave.range @ 6)

        self.wait(0.5)

        rotate_rate = VT(0)

        def rotate_cam(m, dt):
            m.rotate(~rotate_rate * dt, axis=beam.direction)

        self.frame.add_updater(rotate_cam)

        self.play(rotate_rate @ (PI / 9), run_time=2)

        self.wait(12)

        wl = (
            Tex(r"\lambda")
            .fix_in_frame()
            .scale(1.5)
            .next_to(
                self.frame.to_fixed_frame_point(self.frame.get_bottom()),
                UP,
                LARGE_BUFF,
            )
        )
        wl_bez_l = CubicBezier(
            wl.get_left() + LEFT * 0.1,
            wl.get_left() + LEFT * 0.5,
            self.frame.to_fixed_frame_point(self.radar.c2p(0, 0, 4)) + DOWN * 1,
            self.frame.to_fixed_frame_point(self.radar.c2p(0, 0, 4)),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        ).fix_in_frame()
        wl_bez_r = CubicBezier(
            wl.get_right() + RIGHT * 0.1,
            wl.get_right() + RIGHT * 1,
            self.frame.to_fixed_frame_point(self.radar.c2p(0, 0, 6)) + DOWN * 2,
            self.frame.to_fixed_frame_point(self.radar.c2p(0, 0, 6)),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        ).fix_in_frame()

        self.play(
            LaggedStart(
                GrowFromCenter(wl),
                LaggedStart(
                    ShowCreation(wl_bez_l), ShowCreation(wl_bez_r), lag_ratio=0
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        D_line = (
            Line(
                self.frame.to_fixed_frame_point(raindrop.get_left()),
                self.frame.to_fixed_frame_point(raindrop.get_right()),
                stroke_width=DEFAULT_STROKE_WIDTH * 3,
            )
            .fix_in_frame()
            .shift(UP * 1.5 + LEFT * 0.2)
        )
        D_line_l = (
            Line(
                D_line.get_start() + UP * 0.1,
                D_line.get_start() + DOWN * 0.1,
                stroke_width=DEFAULT_STROKE_WIDTH * 3,
            )
            .fix_in_frame()
            .rotate(D_line.get_angle(), about_point=D_line.get_start())
        )
        D_line_r = (
            Line(
                D_line.get_end() + UP * 0.1,
                D_line.get_end() + DOWN * 0.1,
                stroke_width=DEFAULT_STROKE_WIDTH * 3,
            )
            .fix_in_frame()
            .rotate(D_line.get_angle(), about_point=D_line.get_end())
        )

        D_label = Tex("D").fix_in_frame().scale(1.5).next_to(D_line, UP, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                ShowCreation(D_line_l),
                ShowCreation(D_line),
                ShowCreation(D_line_r),
                GrowFromCenter(D_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        phase_constant = (
            Tex(r"\Delta \varphi = C")
            .fix_in_frame()
            .scale(1.5)
            .move_to(self.frame.to_fixed_frame_point(self.frame.get_top()))
            .shift(RIGHT * 2.5 + UP * 2.8)
        )

        self.play(LaggedStart(*[FadeIn(m) for m in phase_constant], lag_ratio=0.2))

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    Uncreate(wl_bez_l),
                    Uncreate(wl_bez_r),
                    FadeOut(wl),
                    Uncreate(D_line_l),
                    Uncreate(D_line_r),
                    Uncreate(D_line),
                    FadeOut(phase_constant),
                    FadeOut(D_label),
                ),
                self.frame.animate(run_time=2).scale(0.8),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    plus_scale.animate(
                        rate_func=utils.rate_functions.there_and_back
                    ).increment_value(1)
                    for plus_scale in plus_scales
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        er_eqn = (
            Tex(r"E_{s} = \frac{k_0^2 \mathbf P V}{2 \pi \varepsilon_0 r} \sin{\theta}")
            .fix_in_frame()
            .scale(1.5)
            .move_to(ORIGIN)
            .shift(RIGHT * 3 + DOWN * 2)
        )

        bez_opacity = VT(0)
        plus_bez = always_redraw(
            lambda: CubicBezier(
                self.frame.to_fixed_frame_point(pluses[0].get_center()) + DOWN * 0.2,
                self.frame.to_fixed_frame_point(pluses[0].get_center()) + DOWN * 1,
                er_eqn[6].get_top() + UP * 1,
                er_eqn[6].get_top() + UP * 0.1,
                stroke_width=DEFAULT_STROKE_WIDTH * 4,
                stroke_opacity=~bez_opacity,
            ).fix_in_frame()
        )
        self.add(plus_bez)

        self.play(
            LaggedStart(
                bez_opacity @ 1,
                er_eqn.shift(DOWN * 5).animate.shift(UP * 5),
                lag_ratio=0.3,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        self.play(er_eqn[6].animate.set_color(YELLOW))

        self.wait(0.5)

        er_eqn_ei = (
            Tex(
                r"E_{s} = \frac{k_0^2 \mathbf E_i \varepsilon_0 (\varepsilon_r - 1) V}"
                r"{2 \pi \varepsilon_0 r} \sin{\theta}"
            )
            .fix_in_frame()
            .scale(1.5)
            .move_to(er_eqn)
            .shift(DOWN * 0.5 + LEFT * 0.5)
        )
        er_eqn_ei[6:16].set_color(YELLOW)

        self.play(
            LaggedStart(
                ReplacementTransform(er_eqn[:6], er_eqn_ei[:6]),
                ReplacementTransform(er_eqn[7:], er_eqn_ei[16:]),
                er_eqn[6]
                .animate.set_opacity(0)
                .move_to(er_eqn_ei[10], aligned_edge=UP),
                LaggedStart(*[FadeIn(m) for m in er_eqn_ei[6:16]]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            er_eqn_ei[6:8].animate.set_color(BLUE).shift(UP * 0.4),
            em_wave.e_highlight @ 1,
            run_time=1,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    er_eqn_ei[6:8].animate.shift(DOWN * 0.4).set_color(YELLOW),
                    em_wave.e_highlight @ 0,
                    lag_ratio=0,
                ),
                er_eqn_ei[8:10].animate.shift(UP * 0.4),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                er_eqn_ei[8:10].animate.shift(DOWN * 0.4),
                er_eqn_ei[11:13].animate.shift(UP * 0.4),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                er_eqn_ei[11:13].animate.shift(DOWN * 0.4),
                LaggedStart(
                    er_eqn_ei[13].animate.shift(UP * 0.4),
                    er_eqn_ei[14].animate.shift(UP * 0.4),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                er_eqn_ei[8].animate.shift(UP * 0.4),
                er_eqn_ei[9].animate.shift(UP * 0.4),
                LaggedStart(
                    er_eqn_ei[10].animate.shift(UP * 0.4),
                    er_eqn[6].animate.shift(UP * 0.4),
                    lag_ratio=0,
                ),
                er_eqn_ei[11].animate.shift(UP * 0.4),
                er_eqn_ei[12].animate.shift(UP * 0.4),
                er_eqn_ei[15].animate.shift(UP * 0.4),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        eps_air = (
            Tex(r"\varepsilon_{r \text{,air}}")
            .rescale_to_fit(er_eqn_ei[11:13].get_height(), 1)
            .fix_in_frame()
            .next_to(er_eqn_ei[14], UP, MED_SMALL_BUFF)
            .shift(UP * 0.5 + RIGHT)
        )
        eps_air_bez = CubicBezier(
            er_eqn_ei[14].get_top() + [0, 0.1, 0],
            er_eqn_ei[14].get_top() + [0, 1, 0],
            eps_air.get_bottom() + [0, -1, 0],
            eps_air.get_bottom() + [0, -0.1, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 4,
        ).fix_in_frame()

        self.play(
            LaggedStart(
                ShowCreation(eps_air_bez),
                LaggedStart(*[FadeIn(m) for m in eps_air], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(*[FadeOut(m) for m in eps_air], lag_ratio=0.1),
                Uncreate(eps_air_bez),
                bez_opacity @ 0,
                er_eqn_ei[8:16].animate.shift(DOWN * 0.4),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            raindrop.D.animate(
                rate_func=utils.rate_functions.there_and_back
            ).increment_value(1)
        )

        self.wait(0.5)

        self.play(
            er_eqn_ei[6:16]
            .animate(rate_func=utils.rate_functions.there_and_back)
            .shift(UP * 0.4),
        )

        self.wait(0.5)

        self.play(
            er_eqn_ei[16]
            .animate(rate_func=utils.rate_functions.there_and_back_with_pause)
            .shift(UP * 0.4),
            run_time=1.5,
        )

        self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         er_eqn_ei[13].animate.shift(DOWN * 0.4),
        #         er_eqn_ei[14].animate.shift(DOWN * 0.4),
        #         lag_ratio=0.2,
        #     ),
        # )

        self.embed()


class BrightBandP1(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        bb_label = Text("Bright Band", font=FONT).scale(1.4)

        self.play(Write(bb_label))

        self.wait(0.5)

        ml_label = Text(
            "or Melting Layer", font=FONT, t2s={"Melting Layer": ITALIC}
        ).next_to(bb_label, DOWN, MED_SMALL_BUFF)

        self.play(Write(ml_label))

        self.wait(0.5)

        cloud = Cloud(width=5.4).shift(UP * 7)  # .rotate(PI, axis=UP)

        self.add(cloud)

        self.play(
            self.frame.animate.reorient(0, 70, 0, center=cloud.get_center(), height=8),
            run_time=3,
        )

        self.remove(bb_label, ml_label)

        self.wait(0.5)

        rng = np.random.default_rng(4)
        particle_count = 12
        particle_size = rng.uniform(0.2, 0.5, particle_count)
        rotations = rng.uniform(0, PI, particle_count)

        screen_width = fw(self)
        screen_height = fh(self)
        edge_buffer = 0.2
        gap = 0.25
        shifts = []
        placed_radii = []

        for radius in particle_size:
            for _ in range(1000):
                shift = np.array(
                    [
                        rng.uniform(
                            -screen_width / 2 + radius + edge_buffer,
                            screen_width / 2 - radius - edge_buffer,
                        ),
                        rng.uniform(
                            -screen_height / 2 + radius + edge_buffer,
                            screen_height / 2 - radius - edge_buffer,
                        ),
                    ]
                )

                if all(
                    np.linalg.norm(shift - other_shift) >= radius + other_radius + gap
                    for other_shift, other_radius in zip(shifts, placed_radii)
                ):
                    shifts.append(shift)
                    placed_radii.append(radius)
                    break
            else:
                raise RuntimeError("Could not place snowflakes without overlap")

        snowflakes = Group()

        for radius, rotation, shift in zip(particle_size, rotations, shifts):
            snowflake = (
                VGroup(
                    *[
                        Line(
                            -radius * RIGHT,
                            radius * RIGHT,
                            stroke_width=DEFAULT_STROKE_WIDTH * 2,
                        ).rotate(angle)
                        for angle in (0, PI / 3, 2 * PI / 3)
                    ]
                )
                .rotate(rotation)
                .set_stroke(WHITE, 30)
                .fix_in_frame()
                .move_to(ORIGIN)
                .shift([shift[0], shift[1], 0])
            )
            snowflakes.add(snowflake)

        self.play(
            LaggedStart(
                self.frame.animate(run_time=3).reorient(
                    0, 80, 0, center=cloud.get_center(), height=1
                ),
                cloud.animate(run_time=3).set_opacity(0),
                LaggedStart(
                    *[
                        m.set_stroke(opacity=0)
                        .scale(1 / 100)
                        .animate.scale(100)
                        .set_stroke(opacity=1)
                        for m in snowflakes
                    ],
                    lag_ratio=0.2,
                ),
                lag_ratio=0.2,
            ),
        )
        self.remove(cloud)

        self.wait(0.5)

        main_center = snowflakes[0].get_center()

        def get_sf_vec(center, scale=1):
            angle_rad = Line(main_center, center).fix_in_frame().get_angle()
            return np.array([np.cos(angle_rad), np.sin(angle_rad), 0]) * scale

        self.play(
            LaggedStart(
                *[
                    sf.animate(run_time=1).shift(get_sf_vec(sf.get_center(), 5))
                    for sf in snowflakes[1:]
                ],
                snowflakes[0].animate.move_to(ORIGIN),
                lag_ratio=0.1,
            ),
        )

        self.remove(snowflakes[1:])
        sf = snowflakes[0]

        self.wait(0.5)

        self.play(sf.animate.scale(1.5))

        self.wait(0.5)

        raindrop = (
            RainDrop(opacity=1, unit_size=0.5)
            .fix_in_frame()
            .next_to(sf, RIGHT, LARGE_BUFF)
            .shift(RIGHT * 10)
        )
        self.add(raindrop)

        precip = Group(sf.copy(), raindrop.copy()).arrange(RIGHT, LARGE_BUFF * 3)

        self.play(
            sf.animate.scale(2).move_to(precip[0]).shift(LEFT * 1.5),
            raindrop.animate.move_to(precip[1]).shift(RIGHT * 1.5),
        )

        self.wait(0.5)

        liquid_line = (
            Line(
                raindrop.get_left(),
                raindrop.get_right(),
                stroke_width=DEFAULT_STROKE_WIDTH * 4,
            )
            .fix_in_frame()
            .shift(DOWN * 2)
        )
        liquid_line_l = Line(
            liquid_line.get_left() + DOWN / 8,
            liquid_line.get_left() + UP / 8,
            stroke_width=DEFAULT_STROKE_WIDTH * 4,
        ).fix_in_frame()
        liquid_line_r = Line(
            liquid_line.get_right() + DOWN / 8,
            liquid_line.get_right() + UP / 8,
            stroke_width=DEFAULT_STROKE_WIDTH * 4,
        ).fix_in_frame()
        D_liquid = (
            Tex(r"D_{\text{liquid}}")
            .fix_in_frame()
            .next_to(liquid_line, DOWN, MED_SMALL_BUFF)
        )

        snow_line = (
            Line(
                sf.get_left(),
                sf.get_right(),
                stroke_width=DEFAULT_STROKE_WIDTH * 4,
            )
            .fix_in_frame()
            .set_y(liquid_line.get_y())
        )
        snow_line_l = Line(
            snow_line.get_left() + DOWN / 8,
            snow_line.get_left() + UP / 8,
            stroke_width=DEFAULT_STROKE_WIDTH * 4,
        ).fix_in_frame()
        snow_line_r = Line(
            snow_line.get_right() + DOWN / 8,
            snow_line.get_right() + UP / 8,
            stroke_width=DEFAULT_STROKE_WIDTH * 4,
        ).fix_in_frame()
        D_snow = (
            Tex(r"D_{\text{snow}}")
            .fix_in_frame()
            .next_to(snow_line, DOWN, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                ShowCreation(liquid_line_l),
                ShowCreation(liquid_line),
                ShowCreation(liquid_line_r),
                FadeIn(D_liquid),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShowCreation(snow_line_l),
                ShowCreation(snow_line),
                ShowCreation(snow_line_r),
                FadeIn(D_snow),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(snow_line_l),
                Uncreate(snow_line),
                Uncreate(snow_line_r),
                FadeOut(D_snow),
                Uncreate(liquid_line_l),
                Uncreate(liquid_line),
                Uncreate(liquid_line_r),
                FadeOut(D_liquid),
                raindrop.animate.shift(RIGHT * fw(self) * 3),
                sf.animate.move_to(ORIGIN),
                lag_ratio=0.05,
            )
        )

        tri1 = Polygon(
            sf.get_center(),
            sf[0].get_end(),
            sf[1].get_end(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt1 = tri1.get_center()
        tri1.scale(0.9).move_to(pt1)

        tri2 = Polygon(
            sf.get_center(),
            sf[1].get_end(),
            sf[2].get_end(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt2 = tri2.get_center()
        tri2.scale(0.9).move_to(pt2)

        tri3 = Polygon(
            sf.get_center(),
            sf[2].get_end(),
            sf[0].get_start(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt3 = tri3.get_center()

        tri4 = Polygon(
            sf.get_center(),
            sf[0].get_start(),
            sf[1].get_start(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt4 = tri4.get_center()
        tri4.scale(0.9).move_to(pt4)

        tri5 = Polygon(
            sf.get_center(),
            sf[1].get_start(),
            sf[2].get_start(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt5 = tri5.get_center()
        tri5.scale(0.9).move_to(pt5)

        tri6 = Polygon(
            sf.get_center(),
            sf[2].get_start(),
            sf[0].get_end(),
            fill_opacity=0.5,
            fill_color=WHITE,
            stroke_opacity=0,
        ).fix_in_frame()
        pt6 = tri6.get_center()
        tri6.scale(0.9).move_to(pt6)

        self.play(
            LaggedStart(
                *[
                    FadeIn(
                        m,
                        rate_func=utils.rate_functions.there_and_back_with_pause,
                        run_time=2,
                    )
                    for m in [tri1, tri2, tri3, tri4, tri5, tri6]
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)
        self.remove(raindrop)
        self.play(sf.animate.shift(RIGHT * fw(self) * 10))

        # self.frame.to_fixed_frame_point(self.frame.get_top()),
        # self.frame.to_fixed_frame_point(self.frame.get_bottom()),

        # er_eqn_ei = (
        #     Tex(
        #         r"E_{s} = \frac{k_0^2 \mathbf E_i (\varepsilon_r - 1) V}"
        #         r"{2 \pi r} \sin{\theta}"
        #     )
        #     .fix_in_frame()
        #     .scale(1.5)
        #     .next_to(precip.fix_in_frame(), UP, LARGE_BUFF)
        #     # .shift(UP * 0.6)
        # )
        # er_eqn_ei[9:11].set_color(YELLOW)
        #
        # self.play(
        #     LaggedStart(
        #         Uncreate(liquid_line_r),
        #         Uncreate(snow_line_r),
        #         Uncreate(liquid_line),
        #         Uncreate(snow_line),
        #         Uncreate(liquid_line_l),
        #         Uncreate(snow_line_l),
        #         FadeOut(D_liquid),
        #         FadeOut(D_snow),
        #         LaggedStart(
        #             sf.animate.shift(DOWN),
        #             raindrop.animate.shift(DOWN),
        #             lag_ratio=0,
        #         ),
        #         er_eqn_ei.shift(UP * 5).animate.shift(DOWN * 5),
        #         lag_ratio=0.1,
        #     )
        # )
        # self.play(
        #     er_eqn_ei[9:11]
        #     .animate(rate_func=utils.rate_functions.there_and_back)
        #     .scale(1.5)
        # )
        #
        # self.wait(0.5)
        #
        # eps_r_snow = (
        #     Tex(r"\varepsilon_r \approx 1 - 3")
        #     .fix_in_frame()
        #     .next_to(sf, UP, MED_SMALL_BUFF)
        # )
        # eps_r_snow[:2].set_color(YELLOW)
        # eps_r_liquid = (
        #     Tex(r"\varepsilon_r \approx 80")
        #     .fix_in_frame()
        #     .next_to(raindrop, UP, MED_SMALL_BUFF)
        #     .set_y(eps_r_snow.get_y())
        # )
        # eps_r_liquid[:2].set_color(YELLOW)
        #
        # self.play(
        #     LaggedStart(
        #         TransformFromCopy(er_eqn_ei[9:11], eps_r_snow[:2], path_arc=PI / 3),
        #         LaggedStart(*[FadeIn(m) for m in eps_r_snow[2:]], lag_ratio=0.1),
        #         lag_ratio=0.5,
        #     )
        # )
        #
        # self.wait(0.5)
        #
        # self.play(
        #     LaggedStart(
        #         TransformFromCopy(er_eqn_ei[9:11], eps_r_liquid[:2], path_arc=-PI / 3),
        #         LaggedStart(*[FadeIn(m) for m in eps_r_liquid[2:]], lag_ratio=0.1),
        #         lag_ratio=0.5,
        #     )
        # )
        #
        # es_rel = (
        #     Tex(r"E_{s \text{,snow}} < E_{s \text{,liquid}}")
        #     .fix_in_frame()
        #     .scale(1.5)
        #     .next_to(precip.fix_in_frame(), DOWN, MED_LARGE_BUFF)
        #     .shift(DOWN * 1.4)
        # )
        #
        # self.play(
        #     LaggedStart(
        #         TransformFromCopy(er_eqn_ei[:2], es_rel[:2]),
        #         *[FadeIn(m) for m in es_rel[2:8]],
        #         TransformFromCopy(er_eqn_ei[:2], es_rel[8:10]),
        #         *[FadeIn(m) for m in es_rel[10:]],
        #         lag_ratio=0.2,
        #     )
        # )
        #
        # self.wait(0.5)
        #
        # self.play(Group(*self.mobjects).animate.shift(UP * 10))

        self.embed()


class BrightBandP1p5(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        boundary = DashedLine(
            self.frame.get_top(),
            self.frame.get_bottom(),
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
            dash_length=DEFAULT_DASH_LENGTH * 8,
            positive_space_ratio=0.6,
        )

        self.play(ShowCreation(boundary))

        self.wait(0.5)

        air_label = (
            Text("Air", font=FONT)
            .next_to(self.frame.get_top(), DOWN, MED_SMALL_BUFF)
            .shift(LEFT * fw(self, 0.25))
        )

        self.play(LaggedStart(*[FadeIn(m) for m in air_label], lag_ratio=0.1))

        self.wait(0.5)

        rng = np.random.default_rng(4)
        particle_count = 16
        particle_size = rng.uniform(0.2, 0.5, particle_count)
        rotations = rng.uniform(0, PI, particle_count)

        screen_width = fw(self, 0.5)
        screen_height = fh(self, 0.9)
        edge_buffer = 0.2
        gap = 0.25
        shifts = []
        placed_radii = []

        for radius in particle_size:
            for _ in range(1000):
                shift = np.array(
                    [
                        rng.uniform(
                            -screen_width / 2 + radius + edge_buffer,
                            screen_width / 2 - radius - edge_buffer,
                        ),
                        rng.uniform(
                            -screen_height / 2 + radius + edge_buffer,
                            screen_height / 2 - radius - edge_buffer,
                        ),
                    ]
                )

                if all(
                    np.linalg.norm(shift - other_shift) >= radius + other_radius + gap
                    for other_shift, other_radius in zip(shifts, placed_radii)
                ):
                    shifts.append(shift)
                    placed_radii.append(radius)
                    break
            else:
                raise RuntimeError("Could not place snowflakes without overlap")

        snowflakes = Group()

        for radius, rotation, shift in zip(particle_size, rotations, shifts):
            snowflake = (
                VGroup(
                    *[
                        Line(
                            -radius * RIGHT,
                            radius * RIGHT,
                            stroke_width=DEFAULT_STROKE_WIDTH * 2,
                        ).rotate(angle)
                        for angle in (0, PI / 3, 2 * PI / 3)
                    ]
                )
                .rotate(rotation)
                .set_stroke(WHITE, 10)
                .move_to(ORIGIN)
                .shift([shift[0], shift[1], 0])
            )
            snowflakes.add(snowflake)
        snowflakes.shift(RIGHT * fw(self, 0.25) + DOWN * fh(self, 0.1))

        snow_label = (
            Text("Snow", font=FONT)
            .next_to(self.frame.get_top(), DOWN, MED_SMALL_BUFF)
            .shift(RIGHT * fw(self, 0.25))
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in snowflakes]),
            LaggedStart(*[FadeIn(m) for m in snow_label], lag_ratio=0.1),
        )

        self.wait(0.5)

        zmin = VT(-0.2)
        zmax = VT(~zmin)
        field_rot = PI / 9
        ax = ThreeDAxes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[0, 2, 0.5],
            z_normal=RIGHT,
            height=fh(self, 0.25),
            width=fw(self, 0.25),
            depth=fh(self, 2),
        ).fix_in_frame()
        ax.rotate(PI / 2, axis=UP).rotate(field_rot, axis=RIGHT).shift(
            self.frame.get_left() - ax.c2p(0, 0, 0)
        )

        zhat_opacity = VT(1)
        zhat = always_redraw(
            lambda: Arrow3D(
                self.frame.get_left() + LEFT * 2,
                self.frame.to_fixed_frame_point(ax.c2p(0, 0, ~zmax)),
                buff=SMALL_BUFF,
                color=GREEN,
                opacity=~zhat_opacity,
            )
        )
        self.add(zhat)

        k = 2 * PI * 4

        efield = always_redraw(
            lambda: ax.get_parametric_curve(
                lambda z: [np.sin(k * z), 0, z],
                t_range=(~zmin, ~zmax, 1 / 200),
                color=TX_COLOR,
            ).apply_depth_test()
        )
        hfield = always_redraw(
            lambda: ax.get_parametric_curve(
                lambda z: [0, np.sin(k * z), z],
                t_range=(~zmin, ~zmax, 1 / 200),
                color=RX_COLOR,
            ).apply_depth_test()
        )

        self.add(efield, hfield)

        self.play(
            GrowArrow(zhat),
            air_label.animate.set_color(GREEN),
            zmax @ 0.88,
            run_time=5,
        )

        self.wait(0.5)

        zinc = 1.1
        self.play(
            zmin + zinc,
            zmax + zinc,
            air_label.animate.set_color(WHITE),
            snow_label.animate.set_color(GREEN),
            run_time=5,
        )

        self.wait(0.5)

        air_to_snow = CurvedArrow(
            air_label.get_bottom() + DOWN * 0.1,
            snow_label.get_bottom() + DOWN * 0.1,
            angle=PI / 3,
        )

        self.play(
            LaggedStart(
                air_label.animate(
                    rate_func=utils.rate_functions.there_and_back_with_pause,
                    run_time=1.5,
                )
                .set_color(GREEN)
                .scale(1.5),
                GrowArrow(air_to_snow),
                snow_label.animate(
                    rate_func=utils.rate_functions.there_and_back_with_pause,
                    run_time=1.5,
                )
                .set_color(GREEN)
                .scale(1.5),
                lag_ratio=0.4,
            )
        )
        self.play(
            snow_label.animate(run_time=0.5).set_color(WHITE),
        )

        self.wait(0.5)

        epsr = (
            Tex(r"K = \frac{\varepsilon_{r} - 1}{\varepsilon_{r} + 2}")
            .move_to(ORIGIN)
            .scale(1.5)
            .shift(LEFT * fw(self, 0.25) + UP)
            .set_opacity(0)
        )
        self.add(epsr)

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(air_to_snow),
                    zmin @ 2.5,
                    zmax @ 3.5,
                    zhat_opacity @ 0,
                    lag_ratio=0,
                ),
                epsr.animate.shift(DOWN).set_opacity(1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                epsr[2:4]
                .animate(rate_func=utils.rate_functions.there_and_back)
                .scale(1.5)
                .set_color(GREEN),
                epsr[7:9]
                .animate(rate_func=utils.rate_functions.there_and_back)
                .scale(1.5)
                .set_color(GREEN),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            epsr.animate.next_to(self.frame.get_bottom(), UP, LARGE_BUFF).shift(
                LEFT * fw(self, 0.25)
            )
        )

        epsr_air = (
            Tex(r"\varepsilon_{r \text{,air}} \approx 1")
            .scale(1.5)
            .next_to(air_label, DOWN, MED_SMALL_BUFF)
        )

        epsr_water = Tex(
            r"\varepsilon_{r \text{,water}} \approx 80", fill_color=BLUE
        ).scale(1.5)

        epsr_snow = Tex(
            r"\varepsilon_{r \text{,snow}} \approx 2", fill_color=ICE
        ).scale(1.5)
        epsr_vals = (
            Group(epsr_water, epsr_snow)
            .arrange(DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)
            .next_to(epsr, UP, LARGE_BUFF)
        )

        self.remove(zhat)

        # rd_sf = snowflakes[: len(snowflakes) // 2]
        # sf_only = snowflakes[len(snowflakes) // 2 :]
        #
        # raindrops = Group(
        #     *[
        #         RainDrop(unit_size=rng.uniform(0.4, 0.7)).move_to(sf).set_opacity(0)
        #         for sf in rd_sf
        #     ]
        # )
        # self.add(raindrops)

        # rain_snow_label = Text("Rain, Snow", font=FONT).move_to(snow_label)
        # rain_snow_label[5:].set_opacity(0.1)
        #
        # self.play(
        #     LaggedStart(
        #         *[
        #             LaggedStart(*[m.animate.set_opacity(0.1) for m in sf], lag_ratio=0)
        #             for sf in sf_only
        #         ],
        #         *[
        #             LaggedStart(
        #                 LaggedStart(
        #                     *[m.animate.set_opacity(0) for m in sf],
        #                     lag_ratio=0.1,
        #                 ),
        #                 rd.animate.set_opacity(1),
        #                 lag_ratio=0.4,
        #             )
        #             for sf, rd in zip(rd_sf, raindrops)
        #         ],
        #         ReplacementTransform(snow_label, rain_snow_label[5:]),
        #         Write(rain_snow_label[:5]),
        #         lag_ratio=0.1,
        #     )
        # )
        #
        # self.wait(.5)
        #
        self.play(
            LaggedStart(
                LaggedStart(
                    TransformFromCopy(epsr[2:4], epsr_air[:2]),
                    LaggedStart(*[FadeIn(m) for m in epsr_air[2:6]], lag_ratio=0.15),
                    lag_ratio=0.8,
                ),
                LaggedStart(
                    TransformFromCopy(epsr[2:4], epsr_water[:2]),
                    LaggedStart(*[FadeIn(m) for m in epsr_water[2:8]], lag_ratio=0.15),
                    lag_ratio=0.8,
                ),
                LaggedStart(
                    TransformFromCopy(epsr[2:4], epsr_snow[:2]),
                    LaggedStart(*[FadeIn(m) for m in epsr_snow[2:7]], lag_ratio=0.15),
                    lag_ratio=0.8,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(*[FadeIn(m) for m in epsr_air[6:]], lag_ratio=0.15))

        self.wait(0.5)

        self.play(LaggedStart(*[FadeIn(m) for m in epsr_water[8:]], lag_ratio=0.15))

        self.wait(0.5)

        self.play(LaggedStart(*[FadeIn(m) for m in epsr_snow[7:]], lag_ratio=0.15))

        self.wait(0.5)

        new_snowflake_pos = (
            self.frame.get_center() + DOWN * fh(self) + LEFT * fw(self, 0.2)
        )

        raindrop = (
            RainDrop()
            .rescale_to_fit(snowflakes[2].get_width() * 1.5, 0)
            .move_to(self.frame.copy().shift(DOWN * fh(self)))
            .shift(RIGHT * fw(self, 0.2))
        )

        self.play(
            LaggedStart(
                multi_anim(snowflakes[2])
                .move_to(
                    new_snowflake_pos,
                    path_arc=PI / 3,
                )
                .scale(2, rate_func=there_and_back_with_final(0.5)),
                epsr_water.animate.next_to(new_snowflake_pos, UP, LARGE_BUFF).set_x(
                    raindrop.get_x()
                ),
                epsr_snow.animate.next_to(new_snowflake_pos, UP, LARGE_BUFF),
                Transform(
                    epsr,
                    epsr.copy().next_to(
                        self.frame.copy().shift(DOWN * fh(self)).get_top(),
                        DOWN,
                        MED_SMALL_BUFF,
                    ),
                    path_arc=PI / 3,
                ),
                lag_ratio=0.4,
            ),
            self.frame.animate.shift(DOWN * fh(self)),
            run_time=2,
        )

        self.wait(0.5)

        self.play(GrowFromCenter(raindrop))

        self.wait(0.5)

        d_snow = DimensionLine(
            snowflakes[2],
            direction=RIGHT,
            placement=DOWN,
            buff=MED_LARGE_BUFF,
            label=r"D",
        )
        d_rain = DimensionLine(
            raindrop,
            direction=RIGHT,
            placement=DOWN,
            buff=MED_LARGE_BUFF,
            label=r"D",
        )
        d_rain.shift((d_snow.line.get_y() - d_rain.line.get_y()) * UP)

        self.play(
            LaggedStart(
                d_snow.get_creation_animation(lag_ratio=0.2),
                d_rain.get_creation_animation(lag_ratio=0.2),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        K_diff = Tex(
            r"\frac{K_{\text{water}}}{K_{\text{snow}}} = \frac"
            r"{\left( \frac{\varepsilon_{r \text{,water}} - 1}{\varepsilon_{r \text{,water}} + 2} "
            r"\right)}"
            r"{\left( \frac{\varepsilon_{r \text{,snow}} - 1}{\varepsilon_{r \text{,snow}} + 2} "
            r"\right)}"
        ).next_to(self.frame.get_bottom(), UP)

        self.play(
            LaggedStart(
                AnimationGroup(FadeOut(d_snow), FadeOut(d_rain)),
                AnimationGroup(
                    TransformFromCopy(epsr[0], K_diff[0], path_arc=PI / 6),
                    TransformFromCopy(epsr[0], K_diff[7], path_arc=-PI / 6),
                    TransformFromCopy(epsr[1], K_diff[12]),
                    FadeIn(
                        Group(
                            K_diff[6],
                            K_diff[13],
                            K_diff[35:38],
                            K_diff[57],
                        )
                    ),
                ),
                AnimationGroup(
                    TransformFromCopy(epsr_water[3:8], K_diff[1:6]),
                    TransformFromCopy(epsr_water[:8], K_diff[14:22]),
                    TransformFromCopy(epsr_water[:8], K_diff[25:33]),
                    TransformFromCopy(epsr_snow[3:7], K_diff[8:12]),
                    TransformFromCopy(epsr_snow[:7], K_diff[38:45]),
                    TransformFromCopy(epsr_snow[:7], K_diff[48:55]),
                ),
                AnimationGroup(
                    TransformFromCopy(epsr[4:6], K_diff[22:24]),
                    TransformFromCopy(epsr[4:6], K_diff[45:47]),
                    TransformFromCopy(epsr[6], K_diff[24]),
                    TransformFromCopy(epsr[6], K_diff[47]),
                    TransformFromCopy(epsr[9:11], K_diff[33:35]),
                    TransformFromCopy(epsr[9:11], K_diff[55:57]),
                ),
                AnimationGroup(
                    FadeOut(epsr),
                    FadeOut(epsr_water),
                    FadeOut(epsr_snow),
                ),
                lag_ratio=0.2,
            ),
            run_time=7,
        )

        self.add(K_diff)

        self.wait(0.5)

        K_diff_value = ((80 - 1) / (80 + 2)) / ((2 - 1) / (2 + 2))
        K_diff_val = Tex(
            r"\frac{K_{\text{water}}}{K_{\text{snow}}} = \frac"
            r"{\left( \frac{\varepsilon_{r \text{,water}} - 1}{\varepsilon_{r \text{,water}} + 2} "
            r"\right)}"
            r"{\left( \frac{\varepsilon_{r \text{,snow}} - 1}{\varepsilon_{r \text{,snow}} + 2} "
            rf"\right)}} = {K_diff_value:.2f}"
        ).move_to(K_diff)

        self.play(
            ReplacementTransform(K_diff, K_diff_val[:58]),
            LaggedStart(
                FadeIn(K_diff_val[58]),
                *[GrowFromCenter(m) for m in K_diff_val[59:]],
                lag_ratio=0.15,
            ),
        )

        self.remove(K_diff)
        self.add(K_diff_val)

        self.wait(0.5)

        self.play(self.frame.animate.shift(LEFT * fw(self)))

        self.embed()


class BrightBandP2(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        snow_top = 3.0
        melting_top = 0.6
        melting_bottom = -0.6
        rain_bottom = -3.0

        column_left = -4.5
        column_right = 1.0
        column_width = column_right - column_left
        column_center_x = (column_left + column_right) / 2

        snow_layer = Rectangle(
            width=column_width,
            height=snow_top - melting_top,
            stroke_width=0,
            fill_color=TOWER_LIGHT,
            fill_opacity=0,
        ).move_to([column_center_x, (snow_top + melting_top) / 2, 0])
        melting_layer = Rectangle(
            width=column_width,
            height=melting_top - melting_bottom,
            stroke_width=0,
            fill_color=TEAL,
            fill_opacity=0,
        ).move_to([column_center_x, 0, 0])
        rain_layer = Rectangle(
            width=column_width,
            height=melting_bottom - rain_bottom,
            stroke_width=0,
            fill_color=BLUE_E,
            fill_opacity=0,
        ).move_to([column_center_x, (melting_bottom + rain_bottom) / 2, 0])
        layers = VGroup(snow_layer, melting_layer, rain_layer).set_z_index(-2)

        profile_x = 2.4
        profile_width = 2.2
        melting_edges = (
            VGroup(
                Line(
                    [column_left, melting_top, 0],
                    [profile_x + profile_width, melting_top, 0],
                ),
                Line(
                    [column_left, melting_bottom, 0],
                    [profile_x + profile_width, melting_bottom, 0],
                ),
            )
            .set_stroke(TEAL, 1.5, opacity=0.5)
            .set_z_index(-1)
        )

        profile_axes = VGroup(
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x, snow_top, 0],
            ),
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x + profile_width, rain_bottom, 0],
            ),
        ).set_stroke(GREY_B, 2, opacity=0.75)
        profile_label = Tex(r"E_s").scale(1.5).next_to(profile_axes[1], DOWN, buff=0.18)

        rng = np.random.default_rng(4)
        particle_count = 54
        particle_x = rng.uniform(
            column_left + 0.35, column_right - 0.35, particle_count
        )
        particle_size = rng.uniform(0.09, 0.14, particle_count)
        drift_phase = rng.uniform(0, TAU, particle_count)
        rotation_rate = rng.uniform(-0.8, 0.8, particle_count)
        cycle_rate = rng.uniform(0.88, 1.12, particle_count)

        snow_duration = 5.0
        melting_duration = 1.5
        rain_duration = 1.5
        cycle_duration = snow_duration + melting_duration + rain_duration
        age_offsets = np.linspace(0, cycle_duration, particle_count, endpoint=False)
        rng.shuffle(age_offsets)

        hydrometeors = Group()
        for radius in particle_size:
            snowflake = VGroup(
                *[
                    Line(-radius * RIGHT, radius * RIGHT).rotate(angle)
                    for angle in (0, PI / 3, 2 * PI / 3)
                ]
            ).set_stroke(TOWER_LIGHT, 1.5)
            water_shell = (
                Circle(radius=radius * 1.15)
                .set_fill(BLUE, opacity=0)
                .set_stroke(BLUE, 1.5, opacity=0)
            )
            raindrop = (
                Circle(
                    radius=radius * 0.5,
                )
                .set_stroke(BLUE, 0)
                .set_fill(BLUE, 0.2)
            )

            hydrometeors.add(Group(water_shell, snowflake, raindrop))

        elapsed = [0.0]
        current_heights = np.zeros(particle_count)
        current_reflectivity = np.zeros(particle_count)

        scale_time = VT(1)

        def update_hydrometeors(group, dt):
            dt_scale = dt * ~scale_time
            elapsed[0] += dt_scale

            for index, particle in enumerate(group):
                age = (
                    elapsed[0] * cycle_rate[index] + age_offsets[index]
                ) % cycle_duration

                if age < snow_duration:
                    alpha = age / snow_duration
                    height = snow_top + alpha * (melting_top - snow_top)
                    snow_opacity = 1
                    shell_opacity = 0
                    rain_opacity = 0
                    drift_scale = 1
                    reflectivity = 0.35
                elif age < snow_duration + melting_duration:
                    alpha = (age - snow_duration) / melting_duration
                    height = melting_top + alpha * (melting_bottom - melting_top)
                    snow_opacity = 1 - alpha
                    shell_opacity = np.sin(PI * alpha)
                    rain_opacity = alpha
                    drift_scale = 1 - 0.7 * alpha
                    reflectivity = 0.35 + 2.2 * shell_opacity + 0.65 * alpha
                else:
                    alpha = (age - snow_duration - melting_duration) / rain_duration
                    height = melting_bottom + alpha * (rain_bottom - melting_bottom)
                    snow_opacity = 0
                    shell_opacity = 0
                    rain_opacity = 1
                    drift_scale = 0.3
                    reflectivity = 1

                x_position = particle_x[index] + drift_scale * 0.12 * np.sin(
                    0.8 * elapsed[0] + drift_phase[index]
                )
                particle.move_to([x_position, height, 0])
                current_heights[index] = height
                current_reflectivity[index] = reflectivity

                water_shell, snowflake, raindrop = particle
                snowflake.set_opacity(snow_opacity)
                water_shell.set_opacity(0.7 * shell_opacity)
                raindrop.set_opacity(rain_opacity)
                snowflake.rotate(rotation_rate[index] * dt_scale)

        update_hydrometeors(hydrometeors, 0)
        hydrometeors.add_updater(update_hydrometeors, call=False)
        hydrometeors.set_z_index(1)

        profile_heights = np.linspace(rain_bottom, snow_top, 73)
        profile_values = np.zeros_like(profile_heights)
        melting_bottom_index = np.argmin(np.abs(profile_heights - melting_bottom))
        melting_top_index = np.argmin(np.abs(profile_heights - melting_top))
        profile_indices = (
            np.arange(melting_bottom_index + 1),
            np.arange(melting_bottom_index, melting_top_index + 1),
            np.arange(melting_top_index, len(profile_heights)),
        )
        profile = VGroup(
            VMobject().set_stroke(BLUE_A, 4),
            VMobject().set_stroke(ACCENT_COLOR, 5),
            VMobject().set_stroke(TOWER_LIGHT, 4),
        ).set_z_index(2)
        profile_initialized = [False]

        def update_profile(curves, dt):
            dt_scale = dt * ~scale_time
            vertical_distance = (
                profile_heights[:, None] - current_heights[None, :]
            ) / 0.14
            kernel = np.exp(-0.5 * vertical_distance**2)
            target_values = np.clip(
                0.02 + kernel @ current_reflectivity / 9,
                0.02,
                0.98,
            )
            if not profile_initialized[0]:
                profile_values[:] = target_values
                profile_initialized[0] = True
            elif dt_scale > 0:
                response = 1 - np.exp(-dt_scale / 0.18)
                profile_values[:] += response * (target_values - profile_values)

            for curve, indices in zip(curves, profile_indices):
                points = np.column_stack(
                    (
                        profile_x + profile_width * profile_values[indices],
                        profile_heights[indices],
                        np.zeros(len(indices)),
                    )
                )
                curve.set_points_smoothly(points)

        update_profile(profile, 0)
        profile.add_updater(update_profile, call=False)

        self.add(layers, melting_edges, hydrometeors)

        self.frame.rescale_to_fit(snow_layer.get_height() * 0.9, 1).move_to(
            snow_layer
        ).shift(UP * snow_layer.get_height() * 0.1).shift(
            RIGHT * snow_layer.get_width()
        )

        self.play(
            self.frame.animate.rescale_to_fit(snow_layer.get_height() * 0.9, 1)
            .move_to(snow_layer)
            .shift(UP * snow_layer.get_height() * 0.1),
            run_time=3,
        )

        self.wait(3)

        melting_label = (
            VGroup(
                Text("Melting", font=FONT).scale(0.4),
                Text("Layer", font=FONT).scale(0.4),
            )
            .arrange(DOWN, buff=SMALL_BUFF * 0.5, aligned_edge=LEFT)
            .next_to(melting_layer, LEFT, -SMALL_BUFF)
        )
        self.add(melting_label)

        self.play(
            self.frame.animate.rescale_to_fit(
                Group(snow_layer, melting_layer).get_height() * 1, 1
            ).move_to(Group(snow_layer, melting_layer), RIGHT),
            run_time=3,
        )

        self.wait(5)

        self.play(scale_time @ 0, run_time=3)

        self.wait(0.5)

        idx = 28
        hdm = hydrometeors[idx]
        inner_structure = (
            RoundedRectangle(
                width=fw(self, 0.4),
                height=fh(self, 0.3),
                corner_radius=0.2,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
                stroke_color=GREEN,
                stroke_opacity=1,
            )
            .set_z_index(5)
            .next_to(hdm, RIGHT, MED_LARGE_BUFF)
            .shift(UP)
        )
        hdm_copy = hdm.copy().set_z_index(6)
        hdm_dest = hdm_copy.copy().move_to(inner_structure).shift(LEFT * 0.9)
        hdm_copy.remove(hdm_copy[2])

        hdm_bez = CubicBezier(
            hdm.get_right() + [0.1, 0, 0],
            hdm.get_right() + [1, 0, 0],
            inner_structure.get_bottom() + [0, -1, 0],
            inner_structure.get_bottom() + [0, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )
        hdm_path = CubicBezier(
            hdm.get_center() + [0, 0, 0],
            hdm.get_center() + [1, 0, 0],
            hdm_dest.get_center() + [0, -1, 0],
            hdm_dest.get_center() + [0, 0, 0],
        )

        self.play(ShowCreation(hdm_bez), FadeIn(inner_structure))
        self.play(MoveAlongPath(hdm_copy, hdm_path), run_time=2)

        self.wait(0.5)

        hdm_split = hdm_copy.copy().arrange(DOWN, buff=MED_SMALL_BUFF).move_to(hdm_copy)
        snow_structure = (
            Text("Snow structure", font=FONT)
            .scale(0.4)
            .next_to(hdm_split[1], RIGHT, MED_SMALL_BUFF)
            .set_z_index(6)
        )
        liquid_water = (
            Text("Liquid Water", font=FONT)
            .scale(0.4)
            .next_to(hdm_split[0], RIGHT, MED_SMALL_BUFF)
            .set_z_index(6)
        )

        self.play(
            hdm_copy.animate.arrange(DOWN, buff=MED_SMALL_BUFF)
            .move_to(hdm_copy)
            .set_z_index(6),
        )
        self.play(Write(snow_structure))
        self.play(Write(liquid_water))

        self.wait(0.5)

        large_v = (
            Tex(r"V \uparrow")
            .rescale_to_fit(liquid_water.get_height() * 1.1, 1)
            .move_to(snow_structure, LEFT)
            .set_z_index(6)
        )
        large_v[1].set_color(GREEN)

        large_eps = (
            Tex(r"\varepsilon_r \uparrow")
            .rescale_to_fit(liquid_water.get_height() * 1.1, 1)
            .move_to(liquid_water, LEFT)
            .set_z_index(6)
        )
        large_eps[2].set_color(GREEN)

        self.play(FadeOut(snow_structure))
        self.play(FadeIn(large_v))

        self.play(FadeOut(liquid_water))
        self.play(FadeIn(large_eps))

        self.wait(0.5)

        self.add(profile, profile_label, profile_axes)

        self.play(
            FadeOut(hdm_bez),
            FadeOut(inner_structure),
            FadeOut(hdm_copy),
            FadeOut(large_eps),
            FadeOut(large_v),
        )
        self.play(
            LaggedStart(
                scale_time @ 1,
                self.frame.animate.rescale_to_fit(
                    Group(*self.mobjects).get_width() * 1.3, 0
                ).move_to(Group(*self.mobjects)),
                # .shift(DOWN * 0.5),
                lag_ratio=0.2,
            ),
            run_time=4,
        )

        self.wait(0.5)

        es_propto = Tex(r"E_s \propto")
        reflectivity_label = Text("Reflectivity", font=FONT)
        profile_label_new = (
            Group(es_propto, reflectivity_label)
            .arrange(RIGHT, SMALL_BUFF)
            .rescale_to_fit(profile_axes[1].get_width() * 1.8, 0)
            .move_to(profile_label)
        )

        self.play(
            LaggedStart(
                ReplacementTransform(profile_label, profile_label_new[0][:2]),
                LaggedStart(
                    *[
                        FadeIn(m)
                        for m in [profile_label_new[0][2], *profile_label_new[1]]
                    ],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(2)

        vel_inc_label = Tex("v")
        vel_inc_arrow = Arrow(UP, DOWN)
        vel_inc = (
            Group(vel_inc_label, vel_inc_arrow)
            .arrange(DOWN, buff=MED_SMALL_BUFF)
            .next_to(rain_layer, LEFT, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                FadeIn(vel_inc_label),
                GrowArrow(vel_inc_arrow),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        z_inc = (
            Rectangle(
                width=profile_axes[1].get_width() * 1.1,
                height=melting_layer.get_height() * 1.1,
                stroke_color=GREEN,
            )
            .set_x(profile_axes[1].get_x())
            .set_y(melting_layer.get_y())
        )

        self.play(ShowCreation(z_inc))

        self.wait(5)

        self.play(self.frame.animate.shift(DOWN * fh(self, 2)))

        self.embed()


class BrightBandP3(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        radar = get_wsr88_model(height=fh(self), indexed_radome_panels=True)
        azimuth = np.arctan2(radar.beam_direction[1], radar.beam_direction[0])
        radar.rotate(-azimuth, axis=OUT, about_point=ORIGIN)
        radar.shift(DOWN * fh(self, 2))
        beam_origin = radar.beam_origin
        radar.base_sections[-1].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-2].set_opacity(0).deactivate_depth_test()
        radar.base_sections[-3].set_opacity(0).deactivate_depth_test()

        beam = radar.get_beam(8 * DEG, 12).deactivate_depth_test()
        beam_direction = beam.get_direction()

        rhi_cells = get_olympex_rhi_cells()
        rhi_origin = beam.get_origin().copy()
        rhi_up = radar.get_scan_up_direction()
        rhi_horizontal = beam_direction - np.dot(beam_direction, rhi_up) * rhi_up
        rhi_horizontal /= np.linalg.norm(rhi_horizontal)
        rhi_normal = np.cross(rhi_horizontal, rhi_up)
        rhi_basis = np.column_stack((rhi_horizontal, rhi_up, rhi_normal))
        vertical_exaggeration = 2
        rhi_corners = (
            rhi_origin
            + rhi_cells["corners"][..., 0, None] * rhi_horizontal
            + vertical_exaggeration * rhi_cells["corners"][..., 1, None] * rhi_up
        )

        rhi_center = rhi_origin + 16.5 * rhi_horizontal + 7.5 * rhi_up
        rhi_camera_azimuth = np.arctan2(rhi_horizontal[1], rhi_horizontal[0]) / DEG

        self.play(
            self.frame.animate.reorient(
                rhi_camera_azimuth, 90, 0, center=rhi_center, height=23.6
            ),
            radar.animate.scale(0.08, about_point=rhi_origin),
            run_time=6,
        )

        self.wait(0.5)

        def get_rhi_label(text, point, font_size=24, rotation=0):
            position = self.frame.to_fixed_frame_point(
                rhi_origin + point[0] * rhi_horizontal + point[1] * rhi_up
            )
            return (
                Text(text, font=FONT, font_size=font_size, fill_border_width=0)
                .rotate(rotation)
                .fix_in_frame()
                .move_to(position)
            )

        rhi_axes = VGroup()
        rhi_labels = VGroup()
        for x in (0, 5, 10, 15, 20, 25, 30, 32.8):
            rhi_axes.add(
                Line(
                    [x, 0, 0],
                    [x, 16, 0],
                    stroke_color=WHITE,
                    stroke_width=1,
                    stroke_opacity=0.12,
                ),
            )
            rhi_labels.add(get_rhi_label(f"{x:g}", [x, -0.65]))
        for z in (0, 2, 4, 6, 8):
            rhi_axes.add(
                Line(
                    [0, 2 * z, 0],
                    [32.8, 2 * z, 0],
                    stroke_color=WHITE,
                    stroke_width=1,
                    stroke_opacity=0.12,
                ),
            )
            rhi_labels.add(get_rhi_label(str(z), [-0.65, 2 * z]))
        rhi_labels.add(
            get_rhi_label("Range (km)", [16.4, -1.65], 28),
            get_rhi_label("Height (km)", [-1.6, 8], 28, PI / 2),
            get_rhi_label(
                "NPOL / OLYMPEX / 13 Nov 2015 02:00 UTC / az 210 deg",
                [16.4, 17],
                26,
            ),
        )
        colorbar_rgba = (
            field_to_rgba(
                np.linspace(10, 50, 128)[:, None],
                vmin=10,
                vmax=50,
                cmap_name="NWSRef",
                fallback_cmap_name="NWSRef",
            )[:, 0]
            / 255
        )
        for index, rgba in enumerate(colorbar_rgba):
            rhi_axes.add(
                Rectangle(
                    width=0.45,
                    height=16 / 128 + 0.002,
                    fill_color=rgb_to_color(rgba[:3]),
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([34, (index + 0.5) * 16 / 128, 0])
            )
        for value in (10, 20, 30, 40, 50):
            rhi_labels.add(get_rhi_label(str(value), [34.9, (value - 10) * 0.4]))
        rhi_labels.add(get_rhi_label("dBZ", [34, 16.8], 26))
        rhi_axes.apply_matrix(rhi_basis).shift(rhi_origin)
        rhi_axes.deactivate_depth_test().set_z_index(3)
        rhi_labels.set_z_index(5)

        triangle_corners = [0, 1, 2, 0, 2, 3]
        rhi_points = rhi_corners[:, triangle_corners].reshape(-1, 3)
        rhi_rgba = np.repeat(rhi_cells["rgba"], 6, axis=0)
        vertex_elevations = np.repeat(rhi_cells["elevation_deg"], 6)
        rhi_image = (
            TriangleMeshMobject(
                rhi_points,
                np.broadcast_to(rhi_normal, rhi_points.shape),
                rhi_rgba,
            )
            .set_shading(0, 0, 0)
            .deactivate_depth_test()
            .set_z_index(1)
        )
        scan_elevation = VT(0)
        scan_extent = VT(0)

        def update_rhi(m):
            rgba = rhi_rgba.copy()
            rgba[:, 3] *= vertex_elevations <= ~scan_extent
            m.set_rgba_array(rgba)

        def get_scan_points():
            ranges = np.linspace(0, np.hypot(32.8, 8), 256)
            elevation = ~scan_elevation * DEG
            earth_radius = 6371 * 4 / 3
            heights = (
                np.sqrt(
                    earth_radius**2
                    + ranges**2
                    + 2 * earth_radius * ranges * np.sin(elevation)
                )
                - earth_radius
            )
            ground = earth_radius * np.arcsin(
                ranges * np.cos(elevation) / (earth_radius + heights)
            )
            inside = (ground <= 32.8) & (heights <= 8)
            end = np.flatnonzero(~inside)
            stop = int(end[0]) if len(end) else len(ranges) - 1
            ground, heights = ground[: stop + 1], heights[: stop + 1]
            if stop > 0 and not inside[stop]:
                alpha = min(
                    (32.8 - ground[-2]) / max(ground[-1] - ground[-2], 1e-9),
                    (8 - heights[-2]) / max(heights[-1] - heights[-2], 1e-9),
                    1,
                )
                ground[-1] = interpolate(ground[-2], ground[-1], alpha)
                heights[-1] = interpolate(heights[-2], heights[-1], alpha)
            return (
                rhi_origin
                + ground[:, None] * rhi_horizontal
                + vertical_exaggeration * heights[:, None] * rhi_up
            )

        def point_scanner(m):
            radar.point_at(get_scan_points()[-1])

        scan_line = always_redraw(
            lambda: (
                VMobject(fill_opacity=0)
                .set_points_as_corners(get_scan_points())
                .set_stroke(WHITE, 1.6, 0.75)
                .deactivate_depth_test()
                .set_z_index(4)
            )
        )
        rhi_image.add_updater(update_rhi)
        scan_elevation.add_updater(point_scanner)

        self.add(rhi_image, scan_extent, scan_elevation)

        self.play(FadeIn(rhi_axes), FadeIn(rhi_labels), FadeIn(scan_line), run_time=1.5)

        self.wait(0.5)

        self.play(
            scan_elevation @ 45,
            scan_extent @ 45,
            run_time=12,
            rate_func=linear,
        )

        scan_line.clear_updaters()
        self.embed()


class Dsd(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        snow_top = 3.0
        melting_top = 0.6
        melting_bottom = -0.6
        rain_bottom = -3.0

        column_left = -4.5
        column_right = 1.0
        column_width = column_right - column_left
        column_center_x = (column_left + column_right) / 2

        snow_layer = Rectangle(
            width=column_width,
            height=snow_top - melting_top,
            stroke_width=0,
            fill_color=TOWER_LIGHT,
            fill_opacity=0.035,
        ).move_to([column_center_x, (snow_top + melting_top) / 2, 0])
        melting_layer = Rectangle(
            width=column_width,
            height=melting_top - melting_bottom,
            stroke_width=0,
            fill_color=TEAL,
            fill_opacity=0.14,
        ).move_to([column_center_x, 0, 0])
        rain_layer = Rectangle(
            width=column_width,
            height=melting_bottom - rain_bottom,
            stroke_width=0,
            fill_color=BLUE_E,
            fill_opacity=0.06,
        ).move_to([column_center_x, (melting_bottom + rain_bottom) / 2, 0])
        layers = VGroup(snow_layer, melting_layer, rain_layer).set_z_index(-2)

        melting_label = (
            VGroup(
                Text("Melting Layer", font=FONT).scale(0.4),
                Text("(Bright Band)", font=FONT).scale(0.4),
            )
            .arrange(DOWN, SMALL_BUFF)
            .set_color(TOWER_LIGHT)
            .next_to(melting_layer, LEFT, SMALL_BUFF)
        )

        profile_x = 2.4
        profile_width = 2.2
        melting_edges = (
            VGroup(
                Line(
                    [column_left, melting_top, 0],
                    [profile_x + profile_width, melting_top, 0],
                ),
                Line(
                    [column_left, melting_bottom, 0],
                    [profile_x + profile_width, melting_bottom, 0],
                ),
            )
            .set_stroke(TEAL, 1.5, opacity=0.5)
            .set_z_index(-1)
        )

        profile_axes = VGroup(
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x, snow_top, 0],
            ),
            Line(
                [profile_x, rain_bottom, 0],
                [profile_x + profile_width, rain_bottom, 0],
            ),
        ).set_stroke(GREY_B, 2, opacity=0.75)

        es_propto = Tex(r"E_s \propto")
        reflectivity_label = Text("Reflectivity", font=FONT)
        profile_label = (
            Group(es_propto, reflectivity_label)
            .arrange(RIGHT, SMALL_BUFF)
            .rescale_to_fit(profile_axes[1].get_width() * 1.8, 0)
            .next_to(profile_axes[1], DOWN, buff=0.18)
        )

        rng = np.random.default_rng(4)
        particle_count = 54
        particle_x = rng.uniform(
            column_left + 0.35, column_right - 0.35, particle_count
        )
        snow_diameters = np.clip(
            rng.lognormal(np.log(3.8), 0.28, particle_count),
            1.6,
            7.4,
        )
        rain_diameters = np.clip(
            0.28 * snow_diameters + rng.normal(0.35, 0.15, particle_count),
            0.45,
            2.6,
        )
        drift_phase = rng.uniform(0, TAU, particle_count)
        rotation_rate = rng.uniform(-0.8, 0.8, particle_count)
        cycle_rate = rng.uniform(0.88, 1.12, particle_count)

        snow_duration = 5.0
        melting_duration = 1.5
        rain_duration = 1.5
        cycle_duration = snow_duration + melting_duration + rain_duration
        age_offsets = np.linspace(0, cycle_duration, particle_count, endpoint=False)
        rng.shuffle(age_offsets)

        ice_epsilon_r = 3.2
        water_epsilon_r = 80.0

        def mixed_phase_epsilon_r(melt_fraction):
            return (
                (1 - melt_fraction) * np.cbrt(ice_epsilon_r)
                + melt_fraction * np.cbrt(water_epsilon_r)
            ) ** 3

        hydrometeors = Group()
        for snow_diameter, rain_diameter in zip(snow_diameters, rain_diameters):
            snow_radius = np.interp(snow_diameter, [1.6, 7.4], [0.08, 0.15])
            rain_radius = np.interp(rain_diameter, [0.45, 2.6], [0.04, 0.075])
            snowflake = VGroup(
                *[
                    Line(-snow_radius * RIGHT, snow_radius * RIGHT).rotate(angle)
                    for angle in (0, PI / 3, 2 * PI / 3)
                ]
            ).set_stroke(TOWER_LIGHT, 1.5)
            water_shell = (
                Circle(radius=snow_radius * 1.15)
                .set_fill(
                    BLUE_A,
                    opacity=0,
                )
                .set_stroke(BLUE_A, 1.5, opacity=0)
            )
            raindrop = (
                Circle(radius=rain_radius).set_stroke(BLUE, 0).set_fill(BLUE, 0.2)
            )

            hydrometeors.add(Group(water_shell, snowflake, raindrop))

        elapsed = [0.0]
        current_heights = np.zeros(particle_count)
        current_diameters = np.zeros(particle_count)
        current_epsilon_r = np.zeros(particle_count)
        current_reflectivity = np.zeros(particle_count)

        def update_hydrometeors(group, dt):
            elapsed[0] += dt

            for index, particle in enumerate(group):
                age = (
                    elapsed[0] * cycle_rate[index] + age_offsets[index]
                ) % cycle_duration

                if age < snow_duration:
                    alpha = age / snow_duration
                    height = snow_top + alpha * (melting_top - snow_top)
                    snow_opacity = 1
                    shell_opacity = 0
                    rain_opacity = 0
                    drift_scale = 1
                    diameter = snow_diameters[index]
                    epsilon_r = ice_epsilon_r
                    reflectivity = 0.35
                elif age < snow_duration + melting_duration:
                    alpha = (age - snow_duration) / melting_duration
                    height = melting_top + alpha * (melting_bottom - melting_top)
                    melt_exponent = np.interp(
                        snow_diameters[index],
                        [1.6, 7.4],
                        [0.72, 1.35],
                    )
                    melt_fraction = alpha**melt_exponent
                    snow_opacity = 1 - melt_fraction
                    shell_opacity = np.sin(PI * melt_fraction)
                    rain_opacity = melt_fraction
                    drift_scale = 1 - 0.7 * melt_fraction
                    diameter = (1 - melt_fraction) * snow_diameters[
                        index
                    ] + melt_fraction * rain_diameters[index]
                    epsilon_r = mixed_phase_epsilon_r(melt_fraction)
                    reflectivity = 0.35 + 2.2 * shell_opacity + 0.65 * melt_fraction
                else:
                    alpha = (age - snow_duration - melting_duration) / rain_duration
                    height = melting_bottom + alpha * (rain_bottom - melting_bottom)
                    snow_opacity = 0
                    shell_opacity = 0
                    rain_opacity = 1
                    drift_scale = 0.3
                    diameter = rain_diameters[index]
                    epsilon_r = water_epsilon_r
                    reflectivity = 1

                x_position = particle_x[index] + drift_scale * 0.12 * np.sin(
                    0.8 * elapsed[0] + drift_phase[index]
                )
                particle.move_to([x_position, height, 0])
                current_heights[index] = height
                current_diameters[index] = diameter
                current_epsilon_r[index] = epsilon_r
                current_reflectivity[index] = reflectivity

                water_shell, snowflake, raindrop = particle
                snowflake.set_opacity(snow_opacity)
                water_shell.set_opacity(0.7 * shell_opacity)
                raindrop.set_opacity(rain_opacity)
                snowflake.rotate(rotation_rate[index] * dt)

        update_hydrometeors(hydrometeors, 0)
        hydrometeors.add_updater(update_hydrometeors, call=False)
        hydrometeors.set_z_index(1)

        profile_heights = np.linspace(rain_bottom, snow_top, 73)
        profile_values = np.zeros_like(profile_heights)
        profile_initialized = [False]
        melting_bottom_index = np.argmin(np.abs(profile_heights - melting_bottom))
        melting_top_index = np.argmin(np.abs(profile_heights - melting_top))
        profile_indices = (
            np.arange(melting_bottom_index + 1),
            np.arange(melting_bottom_index, melting_top_index + 1),
            np.arange(melting_top_index, len(profile_heights)),
        )
        profile = VGroup(
            VMobject().set_stroke(BLUE_A, 4),
            VMobject().set_stroke(ACCENT_COLOR, 5),
            VMobject().set_stroke(TOWER_LIGHT, 4),
        ).set_z_index(2)

        profile_x_offset = VT(0)

        def update_profile(curves, dt):
            vertical_distance = (
                profile_heights[:, None] - current_heights[None, :]
            ) / 0.14
            kernel = np.exp(-0.5 * vertical_distance**2)
            target_values = np.clip(
                0.02 + kernel @ current_reflectivity / 9,
                0.02,
                0.98,
            )
            if not profile_initialized[0]:
                profile_values[:] = target_values
                profile_initialized[0] = True
            elif dt > 0:
                response = 1 - np.exp(-dt / 0.18)
                profile_values[:] += response * (target_values - profile_values)

            for curve, indices in zip(curves, profile_indices):
                points = np.column_stack(
                    (
                        profile_x
                        + profile_width * profile_values[indices]
                        + ~profile_x_offset,
                        profile_heights[indices],
                        np.zeros(len(indices)),
                    )
                )
                curve.set_points_smoothly(points)

        update_profile(profile, 0)
        profile.add_updater(update_profile, call=False)

        overview = Group(
            layers,
            melting_edges,
            melting_label,
            profile_axes,
            profile_label,
            profile,
        )
        overview_center = overview.get_center()
        self.frame.reorient(0, 0, 0, center=overview_center, height=7.2)

        self.add(
            layers,
            melting_edges,
            melting_label,
            profile_axes,
            profile_label,
            hydrometeors,
            profile,
        )

        self.play(
            self.frame.shift(DOWN * fh(self, 1.5)).animate.shift(UP * fh(self, 1.5)),
            run_time=1.5,
        )

        self.wait(2)

        snow_exag = (
            snow_layer.copy()
            .set_stroke(color=YELLOW, opacity=0, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
        )
        melting_exag = (
            melting_layer.copy()
            .set_stroke(color=YELLOW, opacity=0, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
        )
        rain_exag = (
            rain_layer.copy()
            .set_stroke(color=YELLOW, opacity=0, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
        )
        self.add(snow_exag, melting_exag, rain_exag)

        self.play(
            LaggedStart(
                snow_exag.animate(
                    run_time=0.8, rate_func=utils.rate_functions.there_and_back
                )
                .scale(1.1)
                .set_stroke(opacity=1),
                melting_exag.animate(
                    run_time=0.8, rate_func=utils.rate_functions.there_and_back
                )
                .scale(1.1)
                .set_stroke(opacity=1),
                rain_exag.animate(
                    run_time=0.8, rate_func=utils.rate_functions.there_and_back
                )
                .scale(1.1)
                .set_stroke(opacity=1),
                lag_ratio=0.2,
            )
        )

        self.remove(snow_exag, melting_exag, rain_exag)

        self.wait(5)

        diameter_min = 0.0
        diameter_max = 8.0
        trace_scale = 2.3

        def diameter_to_x(diameter):
            alpha = (diameter - diameter_min) / (diameter_max - diameter_min)
            return column_left + alpha * column_width

        epsilon_r_min = ice_epsilon_r
        epsilon_r_max = water_epsilon_r
        epsilon_r_colormap = get_color_map("coolwarm")

        def epsilon_r_to_rgb(epsilon_r):
            alpha = np.clip(
                (epsilon_r - epsilon_r_min) / (epsilon_r_max - epsilon_r_min),
                0,
                1,
            )
            return np.array(epsilon_r_colormap(alpha)[:3])

        gate_edges = np.linspace(rain_bottom, snow_top, 18)
        trace_heights = (gate_edges[:-1] + gate_edges[1:]) / 2
        gate_response_width = gate_edges[1] - gate_edges[0]
        diameter_samples = np.linspace(diameter_min + 0.03, diameter_max, 181)
        density_values = np.zeros((len(trace_heights), len(diameter_samples)))
        gate_epsilon_values = np.full(len(trace_heights), epsilon_r_min)
        gate_occupancies = np.zeros(len(trace_heights))
        dsd_reveal = VT(0)
        waterfall_guides = VGroup()
        waterfall_traces = VGroup()
        for height in trace_heights:
            waterfall_guides.add(
                Line(
                    [column_left, height, 0.01],
                    [column_right, height, 0.01],
                )
                .set_stroke(GREY_B, 0.8, opacity=0.22)
                .apply_depth_test()
            )
            waterfall_traces.add(
                VMobject(
                    stroke_color=TOWER_LIGHT,
                    stroke_width=2.4,
                    stroke_opacity=0,
                ).apply_depth_test()
            )

        def update_dsd_traces(traces, dt):
            for gate_index, (trace, height) in enumerate(zip(traces, trace_heights)):
                range_weights = np.clip(
                    1 - np.abs(current_heights - height) / gate_response_width,
                    0,
                    1,
                )
                in_gate = range_weights > 0

                gate_diameters = current_diameters[in_gate]
                if len(gate_diameters):
                    particle_range_weights = range_weights[in_gate]
                    diameter_distance = (
                        diameter_samples[:, None] - gate_diameters[None, :]
                    ) / 0.55
                    target_density = (
                        np.exp(-0.5 * diameter_distance**2)
                        * particle_range_weights[None, :]
                    ).sum(axis=1) / 5
                    volume_weights = particle_range_weights * gate_diameters**3
                    target_epsilon_r = np.average(
                        current_epsilon_r[in_gate],
                        weights=volume_weights,
                    )
                    target_occupancy = min(particle_range_weights.sum() / 3, 1)
                else:
                    target_density = np.zeros_like(diameter_samples)
                    target_epsilon_r = gate_epsilon_values[gate_index]
                    target_occupancy = 0

                density_values[gate_index] = target_density
                gate_epsilon_values[gate_index] = target_epsilon_r
                gate_occupancies[gate_index] = target_occupancy

                points = np.column_stack(
                    (
                        [diameter_to_x(diameter) for diameter in diameter_samples],
                        np.full(len(diameter_samples), height),
                        0.015 + ~dsd_reveal * trace_scale * density_values[gate_index],
                    )
                )
                trace.set_points_smoothly(points)
                trace.set_stroke(
                    rgb_to_color(epsilon_r_to_rgb(gate_epsilon_values[gate_index])),
                    width=2.4,
                    opacity=~dsd_reveal * (0.18 + 0.77 * gate_occupancies[gate_index]),
                )

        update_dsd_traces(waterfall_traces, 0)
        waterfall_traces.add_updater(update_dsd_traces, call=False)

        surface_center = np.array(
            [
                (column_left + profile_x + profile_width) / 2,
                0,
                0.55,
            ]
        )

        melting_top_new = (
            Line(
                [column_left, melting_top, 0],
                [column_right, melting_top, 0],
            )
            .set_stroke(TEAL, 1.5, opacity=0.5)
            .set_z_index(-1)
        )
        melting_bot_new = (
            Line(
                [column_left, melting_bottom, 0],
                [column_right, melting_bottom, 0],
            )
            .set_stroke(TEAL, 1.5, opacity=0.5)
            .set_z_index(-1)
        )
        self.add(waterfall_guides, waterfall_traces)
        self.play(
            LaggedStart(
                Group(profile_axes, profile_label).animate.shift(RIGHT * 10),
                profile_x_offset @ 10,
                Transform(melting_edges[0], melting_top_new),
                Transform(melting_edges[1], melting_bot_new),
                lag_ratio=0,
            ),
            FadeIn(waterfall_guides),
            dsd_reveal @ 1,
            self.frame.animate.reorient(
                0, 52, 0, center=melting_layer.get_center(), height=7.8
            ).shift(DOWN * 0.5),
            run_time=4.5,
        )
        self.remove(profile_axes, profile_label, profile)

        orbit_rate = VT(0)

        def orbit(m, dt):
            m.rotate(~orbit_rate * dt)

        # self.frame.add_updater(orbit)

        self.wait(0.5)

        # self.play(orbit_rate @ (-PI / 12), run_time=2)

        # self.wait(5)

        D_arrow = Arrow(
            rain_layer.get_corner(DL), rain_layer.get_corner(DR), buff=0
        ).shift(DOWN * 0.5)
        D_label = Tex(r"D").next_to(D_arrow, DOWN, SMALL_BUFF)

        self.play(
            LaggedStart(
                GrowArrow(D_arrow),
                GrowFromCenter(D_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        h_arrow = Arrow(
            rain_layer.get_corner(DR), snow_layer.get_corner(UR), buff=0
        ).shift(RIGHT * 0.5)
        h_label = Text("height", font=FONT).next_to(h_arrow, RIGHT, SMALL_BUFF)

        self.play(
            LaggedStart(
                GrowArrow(h_arrow),
                Write(h_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        def get_trace_highlight(trace, time_width=0.2):
            highlight = trace.copy().set_stroke(YELLOW, width=8, opacity=1)

            def update_highlight(mobject, alpha):
                upper = min((1 + time_width) * alpha, 1)
                lower = max((1 + time_width) * alpha - time_width, 0)
                mobject.pointwise_become_partial(trace, lower, upper)

            return UpdateFromAlphaFunc(
                highlight,
                update_highlight,
                run_time=1.2,
                remover=True,
            )

        self.play(
            LaggedStart(
                *[get_trace_highlight(trace) for trace in waterfall_traces],
                lag_ratio=0.08,
            )
        )

        self.wait(0.5)

        colorbar_height = 3.2
        colorbar_width = 0.28
        colorbar_steps = 64
        colorbar_slice_height = colorbar_height / colorbar_steps
        colorbar_gradient = VGroup()
        for index in range(colorbar_steps):
            alpha = (index + 0.5) / colorbar_steps
            epsilon_r = epsilon_r_min + alpha * (epsilon_r_max - epsilon_r_min)
            colorbar_gradient.add(
                Rectangle(
                    width=colorbar_width,
                    height=colorbar_slice_height * 1.02,
                )
                .set_stroke(width=0)
                .set_fill(rgb_to_color(epsilon_r_to_rgb(epsilon_r)), opacity=1)
                .move_to(
                    [
                        0,
                        -colorbar_height / 2 + (index + 0.5) * colorbar_slice_height,
                        0,
                    ]
                )
            )

        colorbar_frame = Rectangle(
            width=colorbar_width,
            height=colorbar_height,
            fill_opacity=0,
            stroke_color=TOWER_LIGHT,
            stroke_width=1,
        )
        colorbar_ticks = VGroup()
        for epsilon_r in (3.2, 20, 40, 60, 80):
            alpha = (epsilon_r - epsilon_r_min) / (epsilon_r_max - epsilon_r_min)
            tick_y = -colorbar_height / 2 + alpha * colorbar_height
            tick = Line(
                [colorbar_width / 2, tick_y, 0],
                [colorbar_width / 2 + 0.1, tick_y, 0],
                stroke_color=WHITE,
                stroke_width=1,
            )
            tick_label = (
                Text(f"{epsilon_r:g}", font=FONT)
                .scale(0.28)
                .next_to(tick, RIGHT, buff=0.06)
            )
            colorbar_ticks.add(VGroup(tick, tick_label))

        colorbar_label = (
            Tex(r"\varepsilon_r")
            .scale(0.7)
            .next_to(
                colorbar_frame,
                UP,
                buff=0.15,
            )
        )
        epsilon_r_colorbar = (
            Group(
                colorbar_gradient,
                colorbar_frame,
                colorbar_ticks,
                colorbar_label,
            )
            .fix_in_frame()
            .to_edge(LEFT, buff=0.35)
        )

        self.play(
            LaggedStart(
                FadeIn(colorbar_gradient),
                ShowCreation(colorbar_frame),
                FadeIn(colorbar_ticks),
                Write(colorbar_label),
                lag_ratio=0.15,
            ),
            run_time=2,
        )

        # self.wait(10)

        self.wait(0.5)

        self.play(
            colorbar_label.animate(rate_func=utils.rate_functions.there_and_back)
            .scale(2)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        snow_line = (
            Line(
                colorbar_frame.get_corner(DR),
                colorbar_frame.get_corner(DR) + colorbar_frame.get_height() * 0.3 * UP,
            )
            .fix_in_frame()
            .shift(RIGHT * 0.6)
        )
        snow_line_d = Line(
            snow_line.get_bottom() + LEFT * 0.06,
            snow_line.get_bottom() + RIGHT * 0.06,
        ).fix_in_frame()
        snow_line_u = Line(
            snow_line.get_top() + LEFT * 0.06,
            snow_line.get_top() + RIGHT * 0.06,
        ).fix_in_frame()
        snow_label = (
            Text("snow", font=FONT)
            .fix_in_frame()
            .scale(0.5)
            .next_to(snow_line, RIGHT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                ShowCreation(snow_line_d),
                ShowCreation(snow_line),
                ShowCreation(snow_line_u),
                FadeIn(snow_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        rain_line = (
            Line(
                colorbar_frame.get_corner(UR),
                colorbar_frame.get_corner(UR)
                + colorbar_frame.get_height() * 0.3 * DOWN,
            )
            .fix_in_frame()
            .shift(RIGHT * 0.6)
        )
        rain_line_d = Line(
            rain_line.get_bottom() + LEFT * 0.06,
            rain_line.get_bottom() + RIGHT * 0.06,
        ).fix_in_frame()
        rain_line_u = Line(
            rain_line.get_top() + LEFT * 0.06,
            rain_line.get_top() + RIGHT * 0.06,
        ).fix_in_frame()
        rain_label = (
            Text("rain", font=FONT)
            .fix_in_frame()
            .scale(0.5)
            .next_to(rain_line, RIGHT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                ShowCreation(rain_line_u),
                ShowCreation(rain_line),
                ShowCreation(rain_line_d),
                FadeIn(rain_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        melting_line = (
            Line(
                colorbar_frame.get_edge_center(RIGHT)
                + colorbar_frame.get_height() * 0.15 * UP,
                colorbar_frame.get_edge_center(RIGHT)
                + colorbar_frame.get_height() * 0.15 * DOWN,
            )
            .fix_in_frame()
            .shift(RIGHT * 0.6)
        )
        melting_line_d = Line(
            melting_line.get_bottom() + LEFT * 0.06,
            melting_line.get_bottom() + RIGHT * 0.06,
        ).fix_in_frame()
        melting_line_u = Line(
            melting_line.get_top() + LEFT * 0.06,
            melting_line.get_top() + RIGHT * 0.06,
        ).fix_in_frame()
        melting_label = (
            Text("melting", font=FONT)
            .fix_in_frame()
            .scale(0.5)
            .next_to(melting_line, RIGHT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                ShowCreation(melting_line_u),
                ShowCreation(melting_line),
                ShowCreation(melting_line_d),
                FadeIn(melting_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        snow_exag = (
            snow_layer.copy()
            .set_stroke(color=YELLOW, opacity=1, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
            .apply_depth_test()
        )
        melting_exag = (
            melting_layer.copy()
            .set_stroke(color=YELLOW, opacity=1, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
            .apply_depth_test()
        )
        rain_exag = (
            rain_layer.copy()
            .set_stroke(color=YELLOW, opacity=1, width=DEFAULT_STROKE_WIDTH)
            .set_fill(opacity=0)
            .apply_depth_test()
        )

        self.play(ShowCreation(snow_exag, run_time=1.5))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(snow_exag, run_time=1.5),
                ShowCreation(melting_exag, run_time=1.5),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(melting_exag, run_time=1.5),
                ShowCreation(rain_exag, run_time=1.5),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        self.wait(0.5)

        rain_mean_diameter = np.mean(rain_diameters)
        rain_mean_width = 1.5
        rain_gate_indices = trace_heights < melting_bottom
        rain_box_depth = max(
            1.2,
            1.15 * trace_scale * np.max(density_values[rain_gate_indices]),
        )
        rain_box_x_min = diameter_to_x(rain_mean_diameter - rain_mean_width / 2)
        rain_box_x_max = diameter_to_x(rain_mean_diameter + rain_mean_width / 2)
        rain_box_z_min = 0.015
        rain_box_z_max = rain_box_z_min + rain_box_depth
        rain_mean_box = (
            Prism(
                width=rain_box_x_max - rain_box_x_min,
                height=melting_bottom - rain_bottom,
                depth=rain_box_depth,
                color=GREEN,
                opacity=0,
            )
            .move_to(
                [
                    diameter_to_x(rain_mean_diameter),
                    (rain_bottom + melting_bottom) / 2,
                    rain_box_z_min + rain_box_depth / 2,
                ]
            )
            .shift(RIGHT * 1.5)
            .apply_depth_test()
        )
        self.add(rain_mean_box)

        self.play(
            multi_anim(rain_mean_box, lag_ratio=0.3).set_opacity(0.2).shift(LEFT * 1.5),
            run_time=1.5,
        )

        self.wait(0.5)

        mu = (
            Tex(r"\mu")
            .fix_in_frame()
            .scale(2)
            .next_to(
                self.frame.to_fixed_frame_point(rain_mean_box.get_corner(UL)),
                UP,
                LARGE_BUFF,
            )
            .shift(LEFT * 1.5)
        )
        mu_bez = CubicBezier(
            self.frame.to_fixed_frame_point(rain_mean_box.get_corner(UL))
            + UP * LARGE_BUFF * 0.8,
            self.frame.to_fixed_frame_point(rain_mean_box.get_corner(UL))
            + UP * LARGE_BUFF * 1.4,
            mu.get_right() + [0.5, 0, 0],
            mu.get_right() + [0.1, 0, 0],
        ).fix_in_frame()

        self.play(
            LaggedStart(
                ShowCreation(mu_bez),
                GrowFromCenter(mu),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        hydrometeors.suspend_updating()
        waterfall_traces.suspend_updating()

        explicitly_faded_mobjects = [
            colorbar_gradient,
            colorbar_frame,
            colorbar_ticks,
            colorbar_label,
            rain_line_u,
            rain_line,
            rain_line_d,
            rain_label,
            melting_line_u,
            melting_line,
            melting_line_d,
            melting_label,
            snow_line_u,
            snow_line,
            snow_line_d,
            snow_label,
            mu,
            rain_mean_box,
            mu_bez,
        ]
        explicitly_faded_ids = {id(mobject) for mobject in explicitly_faded_mobjects}
        remaining_mobjects = [
            mobject
            for mobject in self.mobjects
            if id(mobject) not in explicitly_faded_ids
        ]

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(colorbar_gradient),
                    Uncreate(colorbar_frame),
                    FadeOut(colorbar_ticks),
                    FadeOut(colorbar_label),
                    Uncreate(rain_line_u),
                    Uncreate(rain_line),
                    Uncreate(rain_line_d),
                    FadeOut(rain_label),
                    Uncreate(melting_line_u),
                    Uncreate(melting_line),
                    Uncreate(melting_line_d),
                    FadeOut(melting_label),
                    Uncreate(snow_line_u),
                    Uncreate(snow_line),
                    Uncreate(snow_line_d),
                    FadeOut(snow_label),
                    FadeOut(mu),
                    FadeOut(rain_mean_box),
                    Uncreate(mu_bez),
                    *[FadeOut(mobject) for mobject in remaining_mobjects],
                ),
                self.frame.animate(run_time=4).reorient(
                    0, 0, 0, center=melting_layer.get_center(), height=40
                ),
                lag_ratio=0.4,
            )
        )

        self.embed()
