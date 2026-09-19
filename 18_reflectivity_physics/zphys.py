# zphys.py

import os
import sys
from pathlib import Path
from random import random, seed, shuffle

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, "..")
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from field_bending import charge_motion, trace_field_lines
from manimlib import *
from scipy.spatial.transform import Rotation

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
        raindrop.add_updater(
            lambda m: m.set_width(
                base_width * max(~drop_scale, 1e-4), about_point=drop_anchor
            )
        )

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
        for m in charge_pairs:
            midpoint = (m.positive.get_center() + m.negative.get_center()) / 2
            m.rest_offset = midpoint - center
            m.rest_separation = (m.positive.get_center() - m.negative.get_center()) / 2
            m.charge_radius = m.positive[0].get_width() / 2

        charge_points = np.vstack(
            [
                charge.get_all_points() - charge.get_center() + m.rest_offset
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
            separation = (
                1 - ~charge_response
            ) * m.rest_separation + ~charge_response * charge_gain * field
            midpoint = center + m.rest_offset
            m.positive.move_to(midpoint + separation)
            m.negative.move_to(midpoint - separation)
            symbol_opacity = np.clip(
                2 * np.linalg.norm(separation) / m.charge_radius - 1, 0, 1
            )
            m.positive[1].set_opacity(symbol_opacity)
            m.negative[1].set_opacity(symbol_opacity)
            m.connector.update(0)

        for m in charge_pairs:
            m.add_updater(update_charge_pair)

        self.play(charge_response @ 1, run_time=1.5)

        self.wait(0.5)

        self.play(em_wave.pulse_length @ (10 * wl), run_time=1.5)

        self.wait(0.5)

        self.play(
            em_wave.range @ (raindrop_range + ~em_wave.pulse_length * 0.99),
            run_time=8,
        )

        self.wait(0.5)

        self.frame.save_state()
        self.camera.light_source.save_state()

        dipole = (
            DipoleAntenna()
            .scale(0.6)
            .apply_matrix(self.frame.get_orientation().as_matrix())
            .move_to(self.frame)
            .shift(DOWN * fh(self, 3))
        )
        self.add(dipole)

        self.play(self.frame.animate.shift(DOWN * fh(self, 3)), run_time=1.5)

        self.wait(0.5)

        frame_rotation = self.frame.get_orientation().as_matrix()
        dipole_tilt = Rotation.from_euler(
            "xyz", [20, -25, -12], degrees=True
        ).as_matrix()
        dipole_rotation = frame_rotation @ dipole_tilt

        self.play(
            dipole.animate.apply_matrix(
                frame_rotation @ dipole_tilt @ frame_rotation.T,
                about_point=dipole.get_center(),
            ),
            self.frame.animate.set_focal_distance(fh(self, 8)),
            self.camera.light_source.animate.move_to(
                dipole.get_center() + self.camera.light_source_position
            ),
            run_time=1.5,
        )

        self.wait(0.5)

        dipole_origin = (
            dipole.left_arm.get_center() + dipole.right_arm.get_center()
        ) / 2
        dipole_axis, feed_axis, radiation_axis = dipole_rotation.T
        dipole_length = np.ptp(dipole.arms.get_all_points() @ dipole_axis)
        dipole_scale = dipole_length / dipole.length
        feed_gap = dipole.feed_gap * dipole_scale
        feed_length = dipole.feed_length * dipole_scale
        wire_radius = dipole.radius * dipole_scale
        antenna_time = VT(0)
        antenna_frequency = 1
        radiation_wavelength = 2 * dipole_length
        radiation_speed = antenna_frequency * radiation_wavelength
        radiation_start = dipole_length * 0.35
        radiation_range = dipole_length * 2.7

        def dipole_point(x=0, y=0, z=0):
            return dipole_origin + x * dipole_axis + y * feed_axis + z * radiation_axis

        def get_antenna_motion(time):
            time = np.asarray(time, dtype=float)
            ramp = np.clip(time, 0, 1)
            envelope = 10 * ramp**3 - 15 * ramp**4 + 6 * ramp**5
            envelope_speed = 30 * ramp**2 * (1 - ramp) ** 2
            envelope_acceleration = 60 * ramp * (1 - ramp) * (1 - 2 * ramp)
            omega = TAU * antenna_frequency
            phase = omega * time
            displacement = envelope * np.sin(phase)
            velocity = envelope_speed * np.sin(phase) + omega * envelope * np.cos(phase)
            acceleration = (
                envelope_acceleration * np.sin(phase)
                + 2 * omega * envelope_speed * np.cos(phase)
                - omega**2 * envelope * np.sin(phase)
            )
            return displacement, velocity, acceleration

        def get_radiated_field(distances):
            _, _, acceleration = get_antenna_motion(
                ~antenna_time - distances / radiation_speed
            )
            return (
                -acceleration
                / (TAU * antenna_frequency) ** 2
                * dipole_length
                * 0.55
                * radiation_start
                / distances
            )

        def get_voltage_profile():
            displacement, _, _ = get_antenna_motion(~antenna_time)
            profiles = VGroup()
            if abs(displacement) < 1e-8:
                return profiles
            for side in [-1, 1]:
                positions = side * np.linspace(feed_gap / 2, dipole_length / 2, 41)
                voltages = (
                    dipole_length
                    * 0.25
                    * displacement
                    * np.sin(PI * positions / dipole_length)
                )
                points = [dipole_point(x, y) for x, y in zip(positions, voltages)]
                profiles.add(
                    Polygon(
                        dipole_point(positions[0]),
                        *points,
                        dipole_point(positions[-1]),
                        fill_color=RX_COLOR,
                        fill_opacity=0.35,
                        stroke_width=0,
                    ).deactivate_depth_test()
                )
            return profiles

        arrow_distances = np.arange(
            radiation_start, radiation_range, radiation_wavelength / 12
        )
        field_arrow = Arrow(
            ORIGIN,
            RIGHT,
            buff=0,
            fill_color=TX_COLOR,
            thickness=3,
            tip_width_ratio=5,
        )
        arrow_points = field_arrow.get_points().copy()
        radiation_curves = VGroup(
            *[VMobject().set_stroke(TX_COLOR, 3.5) for _ in [-1, 1]]
        )
        radiation_arrows = VGroup(
            *[VGroup(*[field_arrow.copy() for _ in arrow_distances]) for _ in [-1, 1]]
        )
        antenna_field = VGroup(
            radiation_curves, radiation_arrows
        ).deactivate_depth_test()

        def update_antenna_field(m):
            front = min(radiation_range, radiation_speed * ~m)
            arrow_values = get_radiated_field(arrow_distances)
            for side, curve, arrows in zip([-1, 1], radiation_curves, radiation_arrows):
                if front > radiation_start + 1e-4:
                    distances = np.linspace(radiation_start, front, 161)
                    values = get_radiated_field(distances)
                    curve.set_points_smoothly(
                        dipole_origin
                        + side * distances[:, None] * radiation_axis
                        + values[:, None] * dipole_axis
                    )
                else:
                    curve.clear_points()
                for distance, value, m in zip(arrow_distances, arrow_values, arrows):
                    m.set_points(
                        dipole_point(z=side * distance)
                        + arrow_points[:, 0, None] * value * dipole_axis
                        + arrow_points[:, 1, None]
                        * min(1, abs(value) / (dipole_length * 0.3))
                        * radiation_axis
                    )
                    m.set_opacity(1 if abs(value) > 1e-5 else 0)

        electron_positions = np.concatenate(
            [
                side
                * np.linspace(
                    feed_gap / 2 + dipole_length * 0.09,
                    dipole_length * 0.44,
                    5,
                )
                for side in [-1, 1]
            ]
        )
        electron_surface = (
            frame_rotation[:, 2]
            - np.dot(frame_rotation[:, 2], dipole_axis) * dipole_axis
        )
        electron_surface *= wire_radius / np.linalg.norm(electron_surface)
        electrons = VGroup(
            *[
                VGroup(
                    Dot(radius=dipole_length * 0.01, fill_color=ICE),
                    Text("-", font=FONT, fill_color=BACKGROUND_COLOR).set_width(
                        dipole_length * 0.012
                    ),
                ).apply_matrix(frame_rotation)
                for _ in electron_positions
            ]
        ).deactivate_depth_test()
        charge_positions = np.concatenate(
            [
                side * np.linspace(dipole_length * 0.2, dipole_length * 0.47, 4)
                for side in [-1, 1]
            ]
        )
        antenna_charges = VGroup(
            *[
                VGroup(
                    Text("+", font=FONT, fill_color=YELLOW),
                    Text("-", font=FONT, fill_color=ICE),
                )
                .set_height(dipole_length * 0.045)
                .apply_matrix(frame_rotation)
                .move_to(dipole_point(x, wire_radius * 3))
                for x in charge_positions
            ]
        ).deactivate_depth_test()

        def update_antenna_charges(m):
            displacement, _, _ = get_antenna_motion(~m)
            for x, m in zip(electron_positions, electrons):
                travel = dipole_length * 0.05 * np.cos(PI * x / dipole_length)
                m.move_to(dipole_point(x - travel * displacement) + electron_surface)
            for x, m in zip(charge_positions, antenna_charges):
                charge = displacement * np.sin(PI * x / dipole_length)
                m[0].set_opacity(max(0, charge))
                m[1].set_opacity(max(0, -charge))

        antenna_time.add_updater(update_antenna_charges)
        antenna_time.add_updater(update_antenna_field)
        voltage_profile = always_redraw(get_voltage_profile)
        source_radius = dipole_length * 0.055
        source_y = -feed_length - source_radius * 1.5
        ac_source = VGroup(
            Circle(radius=source_radius, stroke_color=ICE, stroke_width=2)
            .apply_matrix(dipole_rotation)
            .move_to(dipole_point(y=source_y)),
            ParametricCurve(
                lambda t: dipole_point(
                    source_radius * 0.6 * t,
                    source_y + source_radius * 0.3 * np.sin(PI * t),
                ),
                t_range=(-1, 1, 0.05),
                stroke_color=ICE,
                stroke_width=2,
            ),
            *[
                VMobject()
                .set_points_as_corners(
                    [
                        dipole_point(side * feed_gap / 2, -feed_length),
                        dipole_point(side * source_radius, -feed_length),
                        dipole_point(side * source_radius, source_y),
                    ]
                )
                .set_stroke(ICE, 2)
                for side in [-1, 1]
            ],
            Text("AC", font=FONT, fill_color=ICE)
            .set_height(source_radius * 0.8)
            .apply_matrix(frame_rotation)
            .move_to(dipole_point(y=source_y - source_radius * 1.8)),
        ).deactivate_depth_test()
        radiation_guides = VGroup(
            *[
                DashedLine(
                    dipole_point(z=side * dipole_length * 0.15),
                    dipole_point(z=side * radiation_range),
                    dash_length=0.12,
                    stroke_color=ICE,
                    stroke_width=1,
                    stroke_opacity=0.25,
                )
                for side in [-1, 1]
            ]
        ).deactivate_depth_test()
        voltage_label = Tex("V", fill_color=RX_COLOR).set_height(fh(self, 0.06))
        voltage_label.apply_matrix(frame_rotation).move_to(
            dipole_point(-dipole_length * 0.5, dipole_length * 0.34)
        )
        field_label = Tex("E", fill_color=TX_COLOR).set_height(fh(self, 0.06))
        field_label.apply_matrix(frame_rotation).move_to(
            self.frame.get_center() + frame_rotation @ np.array([1.75, -1.25, 0])
        )
        self.remove(dipole)
        self.add(radiation_guides, antenna_field, voltage_profile, dipole)
        self.add(antenna_charges, antenna_time)

        self.play(
            LaggedStart(
                FadeIn(ac_source),
                FadeIn(electrons),
                FadeIn(voltage_label),
                FadeIn(field_label),
                lag_ratio=0.25,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(antenna_time @ 8, run_time=9)

        self.wait(2)

        self.remove(em_wave, wave_overlap)

        self.play(
            self.frame.animate.restore(),
            self.camera.light_source.animate.restore(),
            run_time=1.5,
        )

        self.wait(0.5)

        field_rotation = self.frame.get_orientation().as_matrix()
        field_up = field_rotation[:, 1]
        pair = max(
            charge_pairs,
            key=lambda m: np.dot(m.positive.get_center(), field_up),
        )
        drop_scale.clear_updaters()
        pair.clear_updaters()
        charge = pair.positive
        charge.save_state()
        pair.remove(charge)
        charge.set_z_index(2)
        charge_target = charge.get_center() + field_up * fh(self)
        self.add(charge)

        self.play(
            FadeOut(pair.connector),
            charge.animate.move_to(charge_target).set_height(fh(self, 0.03)),
            self.frame.animate.move_to(charge_target),
            run_time=2,
        )

        self.wait(0.5)

        field_time = VT(0)
        field_opacity = VT(1)
        field_anchor = charge
        field_scale = fh(self) / 11
        field_origin = charge_target - field_up * field_scale * charge_motion(0)[0]
        field_extent = np.hypot(fw(self), fh(self)) / (2 * field_scale) + 1
        field_radius = VT(3 * field_extent * field_scale)
        backscatter_range = VT(0)
        field_cache = {"time": None, "curves": None}

        def get_field_curve(index):
            if field_cache["time"] != ~field_time:
                field_cache["curves"] = trace_field_lines(~field_time, field_extent)
                field_cache["time"] = ~field_time
            points = field_cache["curves"][index] - UP * charge_motion(~field_time)[0]
            if index == 3 * len(field_cache["curves"]) // 4 and ~backscatter_range > 0:
                distances = -field_scale * points[:, 0]
                outside = np.flatnonzero(distances >= ~backscatter_range)
                end = outside[0] if len(outside) else len(distances) - 1
                points = points[: end + 1].copy()
                if len(outside):
                    alpha = (~backscatter_range - distances[end - 1]) / (
                        distances[end] - distances[end - 1]
                    )
                    points[-1] = interpolate(points[-2], points[-1], alpha)
                return_falloff = np.minimum(
                    1, (~field_radius / (-field_scale * points[:, 0])) ** 2
                )
                points[:, 1] *= return_falloff
                return points
            distances = field_scale * np.linalg.norm(points, axis=1)
            outside = np.flatnonzero(distances >= ~field_radius)
            if len(outside):
                end = outside[0]
                points = points[: end + 1].copy()
                points[-1] *= ~field_radius / distances[end]
            return points

        def get_charge_field_lines():
            get_field_curve(0)
            lines = VGroup()
            count = len(field_cache["curves"])
            for index in range(count):
                highlighted = index in (count // 4, 3 * count // 4)
                m = VMobject(
                    stroke_color=YELLOW if highlighted else BLUE_B,
                    stroke_width=3 if highlighted else 1.4,
                    stroke_opacity=~field_opacity * (1 if highlighted else 0.65),
                )
                m.set_points_smoothly(
                    field_anchor.get_center()
                    + field_scale * get_field_curve(index)[::2] @ field_rotation.T
                )
                lines.add(m)
            return lines.deactivate_depth_test()

        charge.add_updater(
            lambda m: m.move_to(
                field_origin + field_up * field_scale * charge_motion(~field_time)[0]
            )
        )
        charge_field_lines = always_redraw(get_charge_field_lines)
        self.add(field_time, field_opacity, charge_field_lines, perp_stroke=False)

        self.play(
            ShowCreation(
                charge_field_lines,
                lag_ratio=0,
                suspend_mobject_updating=True,
            ),
            run_time=1.5,
        )

        self.wait(0.5)

        fwd_arrow = (
            Arrow(ORIGIN, RIGHT * 3, fill_color=YELLOW).fix_in_frame().shift(RIGHT + UP)
        )
        fwd_label = (
            Text("Forward", fill_color=YELLOW, font=FONT)
            .fix_in_frame()
            .next_to(fwd_arrow, UP, MED_SMALL_BUFF)
        )

        bwd_arrow = (
            Arrow(ORIGIN, LEFT * 3, fill_color=YELLOW).fix_in_frame().shift(LEFT + UP)
        )
        bwd_label = (
            Text("Backward", fill_color=YELLOW, font=FONT)
            .fix_in_frame()
            .next_to(bwd_arrow, UP, MED_SMALL_BUFF)
        )

        self.play(
            field_time.animate(run_time=2.45, rate_func=linear).increment_value(2.45)
        )

        self.play(
            LaggedStart(
                field_time.animate(run_time=15, rate_func=linear).increment_value(15),
                LaggedStart(
                    LaggedStart(GrowArrow(fwd_arrow), FadeIn(fwd_label), lag_ratio=0),
                    LaggedStart(GrowArrow(bwd_arrow), FadeIn(bwd_label), lag_ratio=0),
                    lag_ratio=1,
                ),
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        pattern_radius = fw(self, 0.38)
        sine_pattern = (
            ParametricCurve(
                lambda theta: (
                    field_origin
                    + pattern_radius
                    * abs(np.sin(theta))
                    * (np.sin(theta) * field_rotation[:, 0] + np.cos(theta) * field_up)
                ),
                t_range=(0, TAU, 0.01),
                stroke_color=ICE,
                stroke_width=4,
                fill_opacity=0,
            )
            .deactivate_depth_test()
            .set_z_index(1)
        )
        self.add(sine_pattern, perp_stroke=False)

        self.play(
            ShowCreation(sine_pattern, run_time=2),
            field_time.animate(run_time=4.5, rate_func=linear).increment_value(4.5),
        )

        self.play(
            Uncreate(sine_pattern, run_time=2),
            field_opacity.animate(run_time=2).set_value(0),
            FadeOut(fwd_arrow),
            FadeOut(fwd_label),
            FadeOut(bwd_arrow),
            FadeOut(bwd_label),
            field_time.animate(run_time=2, rate_func=linear).increment_value(2),
        )

        self.remove(charge_field_lines, field_opacity)

        charge.clear_updaters()

        self.play(
            charge.animate.restore(),
            self.frame.animate.restore(),
            run_time=2,
        )

        self.remove(charge)
        pair.add(charge)
        self.add(charge_pairs)

        self.play(FadeIn(pair.connector), run_time=0.5)

        self.wait(0.5)

        field_anchor = raindrop
        scatter_direction = unit(raindrop.get_center() - beam.get_origin())
        field_up = unit(
            e_direction - np.dot(e_direction, scatter_direction) * scatter_direction
        )
        field_rotation = np.column_stack(
            (scatter_direction, field_up, np.cross(scatter_direction, field_up))
        )
        field_time @= 0
        field_opacity @= 1
        particle_separation = (
            np.linalg.norm(pair.positive.get_center() - pair.negative.get_center())
            / 2
            * field_up
        )
        field_amplitude = charge_motion(0)[0]

        def update_particle_pair(m):
            midpoint = raindrop.get_center() + m.rest_offset
            separation = (
                particle_separation * charge_motion(~field_time)[0] / field_amplitude
            )
            m.positive.move_to(midpoint + separation)
            m.negative.move_to(midpoint - separation)
            symbol_opacity = np.clip(
                2 * np.linalg.norm(separation) / m.charge_radius - 1, 0, 1
            )
            m.positive[1].set_opacity(symbol_opacity)
            m.negative[1].set_opacity(symbol_opacity)
            m.connector.set_points_as_corners(
                [m.positive.get_center(), m.negative.get_center()]
            )

        for m in charge_pairs:
            m.clear_updaters()
            m.add_updater(update_particle_pair)

        charge_field_lines.update(0)
        self.add(
            field_opacity, charge_field_lines, raindrop, charge_pairs, perp_stroke=False
        )

        self.play(
            ShowCreation(
                charge_field_lines,
                lag_ratio=0,
                suspend_mobject_updating=True,
            ),
            run_time=1.5,
        )

        self.wait(0.5)

        self.play(
            field_time.animate(run_time=3.75, rate_func=linear).increment_value(3.75)
        )

        plane_reveal = VT(0)
        field_radius @= np.hypot(fw(self), fh(self)) / 2
        plane_specs = [
            (index, side)
            for index in range(2, len(field_cache["curves"]) // 2, 2)
            for side in (0, PI)
        ]
        field_planes = Group(
            *[
                Surface(
                    resolution=(61, 25),
                    color=YELLOW
                    if index == len(field_cache["curves"]) // 4
                    else BLUE_B,
                    opacity=0,
                    shading=(0.1, 0.1, 0.2),
                    depth_test=False,
                )
                for index, _ in plane_specs
            ]
        )
        plane_edges = VGroup(
            *[
                VGroup(
                    *[
                        VMobject(
                            stroke_color=YELLOW
                            if index == len(field_cache["curves"]) // 4
                            else BLUE_B,
                            stroke_width=1,
                            stroke_opacity=0,
                        )
                        for _ in range(3)
                    ]
                )
                for index, _ in plane_specs
            ]
        ).deactivate_depth_test()

        def update_field_planes(m):
            half_angle = max(~m, 1e-4) * PI / 2
            for (index, side), plane, edge in zip(
                plane_specs, field_planes, plane_edges
            ):
                curve = get_field_curve(index)
                samples = np.linspace(0, len(curve) - 1, plane.resolution[0])
                radial = np.interp(samples, np.arange(len(curve)), curve[:, 0])
                height = np.interp(samples, np.arange(len(curve)), curve[:, 1])
                angles = np.linspace(
                    side - half_angle, side + half_angle, plane.resolution[1]
                )
                directions = (
                    np.cos(angles)[:, None] * scatter_direction
                    + np.sin(angles)[:, None] * field_rotation[:, 2]
                )
                points = raindrop.get_center() + field_scale * (
                    radial[:, None, None] * directions[None, :, :]
                    + height[:, None, None] * field_up
                )
                plane.set_points(points.reshape(-1, 3))
                plane.data["du_point"][:] = (
                    points + np.gradient(points, axis=0)
                ).reshape(-1, 3)
                plane.data["dv_point"][:] = (
                    points + np.gradient(points, axis=1)
                ).reshape(-1, 3)
                plane.set_opacity(
                    0.16 if index == len(field_cache["curves"]) // 4 else 0.08
                )
                plane.sort_faces_back_to_front(
                    self.camera.get_location() - raindrop.get_center()
                )
                edge[0].set_points_smoothly(points[-1]).set_stroke(opacity=0.35 * ~m)
                for boundary, column in zip(edge[1:], (0, -1)):
                    boundary.set_points_smoothly(points[:, column]).set_stroke(
                        opacity=0.65 * np.sin(PI * ~m)
                    )

        plane_reveal.add_updater(update_field_planes)
        self.add(
            plane_reveal,
            field_radius,
            field_planes,
            plane_edges,
            charge_field_lines,
            raindrop,
            charge_pairs,
            perp_stroke=False,
        )

        self.play(
            plane_reveal @ 1,
            field_radius @ fh(self, 0.65),
            self.frame.animate.set_height(fh(self, 1.7)).set_orientation(
                Rotation.from_matrix(field_rotation)
                * Rotation.from_euler("xy", [15, -30], degrees=True)
            ),
            field_time.animate(rate_func=linear).increment_value(3),
            run_time=3,
        )

        self.play(
            field_time.animate(run_time=3.65, rate_func=linear).increment_value(3.65)
        )

        self.wait(0.5)

        backscatter_index = 3 * len(field_cache["curves"]) // 4
        backscatter_range @= -field_scale * get_field_curve(backscatter_index)[-1, 0]
        field_extent = max(field_extent, 2 * raindrop_range / field_scale)
        field_cache["time"] = None
        backscatter_view = Group(radar.antenna, field_planes)
        backscatter_height = 1.2 * max(
            backscatter_view.get_height(),
            backscatter_view.get_width() * fh(self) / fw(self),
        )
        self.add(backscatter_range)

        self.play(
            backscatter_range @ raindrop_range,
            self.frame.animate.reorient(
                0,
                0,
                0,
                center=backscatter_view.get_center(),
                height=backscatter_height,
            ),
            field_time.animate(rate_func=linear).increment_value(4),
            run_time=4,
        )

        self.play(field_time.animate(run_time=12, rate_func=linear).increment_value(12))

        self.play(field_opacity @ 0, plane_reveal @ 0, run_time=1.5)

        self.remove(
            plane_reveal,
            field_radius,
            field_planes,
            plane_edges,
            charge_field_lines,
        )

        self.wait(0.5)

        vol_eqn = (
            Tex(r"E_s \propto V = \frac{4}{3} \pi r^3 = \frac{1}{6} \pi D^3")
            .fix_in_frame()
            .scale(1.5)
            .move_to(ORIGIN)
            .shift(LEFT * 3 + UP * 2.5)
        )

        self.play(
            LaggedStart(
                self.frame.animate(run_time=2)
                .reorient(0, 0, 0, center=raindrop.get_center(), height=3)
                .shift(UP * 0.3),
                LaggedStart(*[GrowFromCenter(m) for m in vol_eqn], lag_ratio=0.1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        D_dim = always_redraw(
            lambda: DimensionLine(
                raindrop,
                DOWN,
                camera_frame=self.frame,
                tick_length=0.1,
            )
        )
        D_bez = always_redraw(
            lambda: CubicBezier(
                vol_eqn[-2].get_corner(DR) + [0.1, -0.1, 0],
                vol_eqn[-2].get_corner(DR) + [1, -1, 0],
                self.frame.to_fixed_frame_point(D_dim.get_top()) + [0, 1, 0],
                self.frame.to_fixed_frame_point(D_dim.get_top()) + [0, 0.1, 0],
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            ).fix_in_frame()
        )

        self.play(
            LaggedStart(
                FadeIn(D_bez, suspend_mobject_updating=True),
                FadeIn(D_dim, suspend_mobject_updating=True),
                lag_ratio=0.3,
            )
        )

        self.add(D_dim, D_bez)

        self.wait(0.5)

        volume_scale = ~drop_scale
        volume_count = len(charge_pairs)
        volume_radius = (
            0.95
            * np.linalg.norm(
                raindrop.get_points() - raindrop.get_center(), axis=1
            ).min()
        )
        for m in charge_pairs:
            m.clear_updaters()
        charge_template = charge_pairs[0].copy()
        charge_extent = np.linalg.norm(
            charge_template.get_all_points() - charge_template.get_center(), axis=1
        ).max()
        volume_states = [
            get_pair_state(m, m.get_center() - raindrop.get_center())
            for m in charge_pairs
        ]

        def update_volume_charges(m):
            scale = ~m / volume_scale
            count = volume_count * scale**3
            center = raindrop.get_center()
            while len(volume_states) < int(np.ceil(count - 1e-6)):
                pair_scale = ((len(volume_states) + 1) / volume_count) ** (1 / 3)
                candidates = rng.normal(size=(64, 3))
                candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
                candidates *= rng.random((64, 1)) ** (1 / 3)
                candidates *= volume_radius - 1.15 * charge_extent / pair_scale
                offsets = np.array([state["offset"] for state in volume_states])
                distances = np.linalg.norm(
                    candidates[:, None, :2] - offsets[None, :, :2], axis=2
                )
                offset = candidates[np.argmax(distances.min(axis=1))]
                pair = charge_template.copy()
                volume_states.append(get_pair_state(pair, offset))

            visible_pairs = []
            for index, state in enumerate(volume_states):
                progress = np.clip(count - index, 0, 1)
                position = center + state["offset"] * scale
                for part, points in state["points"]:
                    part.set_points(points * ease_out_back(progress) + position)
                if progress > 1e-6:
                    visible_pairs.append(state["pair"])
            if charge_pairs.submobjects != visible_pairs:
                charge_pairs.set_submobjects(visible_pairs)

        drop_scale.add_updater(update_volume_charges)

        self.play(drop_scale * 2, run_time=3)

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=utils.rate_functions.there_and_back)
                    .set_color(YELLOW)
                    .scale(1.3)
                    for m in [*vol_eqn[5:11], *vol_eqn[12:]]
                ],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        pow_vol_eqn = (
            Tex(
                r"P \propto V^2 = \left(\frac{4}{3} \pi r^3\right)^2 = "
                r"\left( \frac{1}{6} \pi D^3 \right)^2"
            )
            .fix_in_frame()
            .scale(1.5)
            .move_to(vol_eqn, LEFT)
        )

        self.play(
            LaggedStart(
                FadeOut(D_bez, suspend_mobject_updating=True),
                vol_eqn[:2].animate.shift(DOWN).set_opacity(0.2),
                GrowFromCenter(pow_vol_eqn[0]),
                ReplacementTransform(vol_eqn[2], pow_vol_eqn[1]),
                LaggedStart(
                    ReplacementTransform(vol_eqn[3], pow_vol_eqn[2]),
                    ReplacementTransform(vol_eqn[4], pow_vol_eqn[4]),
                    ReplacementTransform(vol_eqn[5:11], pow_vol_eqn[6:12]),
                    ReplacementTransform(vol_eqn[11], pow_vol_eqn[14]),
                    ReplacementTransform(vol_eqn[12:], pow_vol_eqn[16:22]),
                    lag_ratio=0,
                ),
                lag_ratio=0.3,
            )
        )

        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in [
                        pow_vol_eqn[3],
                        pow_vol_eqn[5],
                        pow_vol_eqn[12],
                        pow_vol_eqn[13],
                        pow_vol_eqn[15],
                        pow_vol_eqn[22],
                        pow_vol_eqn[23],
                    ]
                ],
                lag_ratio=0.15,
            )
        )

        self.remove(vol_eqn)
        self.add(vol_eqn[:2], pow_vol_eqn)

        self.wait(0.5)

        pow_vol_eqn_r = (
            Tex(r"P \propto V^2 = \left(\frac{4}{3} \pi r^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )
        pow_vol_eqn_2r = (
            Tex(r"P \propto V^2 = \left(\frac{4}{3} \pi (2r)^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )
        pow_vol_eqn_cubed = (
            Tex(r"P \propto V^2 = \left(\frac{4}{3} \pi 2^3 r^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )
        pow_vol_eqn_squared = (
            Tex(r"P \propto V^2 = 2^{3 \cdot 2}\left(\frac{4}{3} \pi r^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )
        pow_vol_eqn_six = (
            Tex(r"P \propto V^2 = 2^6\left(\frac{4}{3} \pi r^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )
        pow_vol_eqn_64 = (
            Tex(r"P \propto V^2 = 64\left(\frac{4}{3} \pi r^3\right)^2")
            .fix_in_frame()
            .scale(1.5)
            .move_to(pow_vol_eqn, LEFT)
        )

        self.play(
            ReplacementTransform(pow_vol_eqn[:14], pow_vol_eqn_r[:14]),
            LaggedStart(*[FadeOut(m) for m in pow_vol_eqn[14:]], lag_ratio=0.08),
            FadeOut(D_dim, suspend_mobject_updating=True),
        )

        self.remove(pow_vol_eqn)
        self.add(pow_vol_eqn_r)

        self.wait(0.5)

        self.play(
            ReplacementTransform(pow_vol_eqn_r[:10], pow_vol_eqn_2r[:10]),
            ReplacementTransform(pow_vol_eqn_r[10], pow_vol_eqn_2r[12]),
            ReplacementTransform(pow_vol_eqn_r[11:], pow_vol_eqn_2r[14:]),
        )

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in pow_vol_eqn_2r[10:12]],
                GrowFromCenter(pow_vol_eqn_2r[13]),
                lag_ratio=0.15,
            )
        )

        self.remove(pow_vol_eqn_r)
        self.add(pow_vol_eqn_2r)

        self.wait(0.5)

        self.play(
            ReplacementTransform(pow_vol_eqn_2r[:10], pow_vol_eqn_cubed[:10]),
            FadeOut(pow_vol_eqn_2r[10]),
            ReplacementTransform(pow_vol_eqn_2r[11], pow_vol_eqn_cubed[10]),
            ReplacementTransform(pow_vol_eqn_2r[12], pow_vol_eqn_cubed[12]),
            FadeOut(pow_vol_eqn_2r[13]),
            TransformFromCopy(
                pow_vol_eqn_2r[14], pow_vol_eqn_cubed[11], path_arc=PI / 3
            ),
            ReplacementTransform(pow_vol_eqn_2r[14:], pow_vol_eqn_cubed[13:]),
            run_time=1.5,
        )

        self.remove(pow_vol_eqn_2r)
        self.add(pow_vol_eqn_cubed)

        self.wait(0.5)

        self.play(
            ReplacementTransform(pow_vol_eqn_cubed[:5], pow_vol_eqn_squared[:5]),
            ReplacementTransform(pow_vol_eqn_cubed[5:10], pow_vol_eqn_squared[9:14]),
            ReplacementTransform(
                pow_vol_eqn_cubed[10:12], pow_vol_eqn_squared[5:7], path_arc=PI
            ),
            GrowFromCenter(pow_vol_eqn_squared[7]),
            TransformFromCopy(
                pow_vol_eqn_cubed[15], pow_vol_eqn_squared[8], path_arc=PI / 3
            ),
            ReplacementTransform(pow_vol_eqn_cubed[12:], pow_vol_eqn_squared[14:]),
            run_time=1.5,
        )

        self.remove(pow_vol_eqn_cubed)
        self.add(pow_vol_eqn_squared)

        self.wait(0.5)

        self.play(
            ReplacementTransform(pow_vol_eqn_squared[:6], pow_vol_eqn_six[:6]),
            ReplacementTransform(pow_vol_eqn_squared[6:9], pow_vol_eqn_six[6]),
            ReplacementTransform(pow_vol_eqn_squared[9:], pow_vol_eqn_six[7:]),
        )

        self.remove(pow_vol_eqn_squared)
        self.add(pow_vol_eqn_six)

        self.wait(0.5)

        self.play(
            ReplacementTransform(pow_vol_eqn_six[:5], pow_vol_eqn_64[:5]),
            ReplacementTransform(pow_vol_eqn_six[5:7], pow_vol_eqn_64[5:7]),
            ReplacementTransform(pow_vol_eqn_six[7:], pow_vol_eqn_64[7:]),
        )

        self.remove(pow_vol_eqn_six)
        self.add(pow_vol_eqn_64)

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(vol_eqn[:2]),
                self.frame.animate.shift(UP * fh(self)),
                pow_vol_eqn_64.animate.move_to(ORIGIN),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        self.embed()


class Plug(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):

        description_arrow = Arrow(
            self.frame.get_center() + DOWN * fh(self, 0.25),
            self.frame.get_center() + DOWN * fh(self, 0.45),
        )
        description = Text("In the description", font=FONT).next_to(
            description_arrow, UP, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                Write(description, run_time=0.8),
                GrowArrow(description_arrow),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        nb_slider = (
            VideoMobject("./static/nb_slider.mov", start=8, end=18)
            .rescale_to_fit(fw(self, 0.5), 0)
            .next_to(description, UP, MED_LARGE_BUFF)
        )

        self.play(nb_slider.shift(UP * fh(self)).animate.shift(DOWN * fh(self)))

        self.wait(2)

        figures = Group()
        for filename, title in [
            (
                "bohren_radiating_dipole.png",
                "Absorption and Scattering\nof Light by Small Particles",
            ),
            (
                "rauber_scattering_regimes.png",
                "Radar Meteorology:\nA First Course",
            ),
            (
                "chandra_scattering_patterns.png",
                "Introduction to Dual Polarization\nWeather Radar",
            ),
            (
                "chandra_melting_layer.png",
                "Introduction to Dual Polarization\nWeather Radar",
            ),
        ]:
            figure = ImageMobject(f"./static/plug_figures/{filename}")
            figure.set_width(fw(self, 0.21))
            if figure.get_height() > fh(self, 0.2):
                figure.set_height(fh(self, 0.2))
            source_title = Text(title, font=FONT, font_size=14)
            if source_title.get_width() > fw(self, 0.21):
                source_title.set_width(fw(self, 0.21))
            source_title.next_to(figure, DOWN, SMALL_BUFF)
            figures.add(Group(figure, source_title))

        figures.arrange_in_grid(n_cols=2, buff=MED_SMALL_BUFF)
        figures.set_x(fw(self, 0.245)).align_to(nb_slider, UP)

        self.play(
            nb_slider.animate(suspend_mobject_updating=False)
            .set_width(fw(self, 0.46))
            .set_x(-fw(self, 0.24)),
            run_time=1.2,
        )

        self.play(
            LaggedStart(
                *[GrowFromCenter(m, rate_func=ease_out_back) for m in figures],
                lag_ratio=0.3,
                run_time=2,
            )
        )

        self.wait(4)

        merch = Group(
            *[
                ImageMobject(f"../props/static/{filename}").set_height(fh(self, 0.48))
                for filename in [
                    "engineering_notebook_cover.png",
                    "radar_equation_mug.webp",
                    "phased_array_shirt_back.webp",
                    "dual_pol_mug.webp",
                ]
            ]
        ).arrange(RIGHT, buff=MED_SMALL_BUFF)
        if merch.get_width() > fw(self, 0.94):
            merch.set_width(fw(self, 0.94))
        merch.next_to(description, UP, MED_LARGE_BUFF)

        self.play(FadeOut(nb_slider), *[FadeOut(m) for m in figures])

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m, rate_func=ease_out_back) for m in merch],
                lag_ratio=0.3,
                run_time=2,
            )
        )

        self.wait(0.5)

        self.play(
            merch[0].animate.set_height(fh(self, 0.58)).move_to(UP * fh(self, 0.12)),
            *[
                m.animate.set_x(fw(self, x))
                for m, x in zip(merch[1:], [-0.31, 0.395, 0.225])
            ],
            run_time=1.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ShrinkToCenter(m) for m in merch],
                FadeOut(description),
                FadeOut(description_arrow),
                lag_ratio=0.15,
            )
        )

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

        clock_time = 5.5
        clock_start = self.time
        clock_speed = 1.0
        clock_target = 1.0
        clock_ramp_duration = 1.0

        def particle_time():
            time = max(self.time - clock_start, 0)
            alpha = min(time / clock_ramp_duration, 1)
            ramp_time = clock_ramp_duration * (
                2.5 * alpha**4 - 3 * alpha**5 + alpha**6
            ) + max(time - clock_ramp_duration, 0)
            return (
                clock_time
                + clock_speed * time
                + (clock_target - clock_speed) * ramp_time
            )

        def set_particle_speed(speed, duration):
            nonlocal \
                clock_time, \
                clock_start, \
                clock_speed, \
                clock_target, \
                clock_ramp_duration
            clock_time = particle_time()
            clock_start = self.time
            clock_speed = clock_target
            clock_target = speed
            clock_ramp_duration = duration

        previous_particle_time = [0.0]
        current_heights = np.zeros(particle_count)
        current_reflectivity = np.zeros(particle_count)

        scale_time = VT(1)

        def update_hydrometeors(group, dt):
            time = particle_time()
            dt_scale = time - previous_particle_time[0]
            previous_particle_time[0] = time

            for index, particle in enumerate(group):
                age = (time * cycle_rate[index] + age_offsets[index]) % cycle_duration

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
                    0.8 * time + drift_phase[index]
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
        previous_profile_time = [particle_time()]

        def update_profile(curves, dt):
            time = particle_time()
            dt_scale = time - previous_profile_time[0]
            previous_profile_time[0] = time
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

        self.wait(1)

        set_particle_speed(0, 1)
        self.play(scale_time @ 0, run_time=1)

        self.wait(0.5)

        idx = 46
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
        set_particle_speed(1, 4 / 1.2)
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

        self.wait(15)

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

        self.wait(14)

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

        bright_band_box = (
            Rectangle(
                width=30.8,
                height=0.7 * vertical_exaggeration,
                stroke_color=BLACK,
                stroke_width=3,
                fill_opacity=0,
            )
            .move_to([17.4, 2.4 * vertical_exaggeration, 0])
            .apply_matrix(rhi_basis)
            .shift(rhi_origin)
            .deactivate_depth_test()
            .set_z_index(5)
        )
        bright_band_label = (
            get_rhi_label(
                "Melting layer / Bright band",
                [17.4, 3.2 * vertical_exaggeration],
                32,
            )
            .set_color(YELLOW)
            .set_backstroke(BACKGROUND_COLOR, 6)
            .set_z_index(6)
        )

        self.wait(0.5)

        self.play(ShowCreation(bright_band_box), run_time=1.5)

        self.wait(1)

        self.play(FadeIn(bright_band_label, shift=UP * 0.15))

        self.wait(0.5)

        rhi_image.clear_updaters()
        scan_elevation.clear_updaters()

        for m in (rhi_labels, bright_band_label):
            m.apply_function(self.frame.from_fixed_frame_point).unfix_from_frame()

        self.play(self.frame.animate.shift(rhi_up * self.frame.get_height()))

        self.wait(0.5)

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

        self.wait(5.55)

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

        self.wait(4.15)

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

        self.wait(1.2)

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

        self.wait(2.0)

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

        self.wait(9.5)

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

        self.wait(5.5)

        self.play(
            LaggedStart(
                Uncreate(snow_exag, run_time=1.5),
                ShowCreation(melting_exag, run_time=1.5),
                lag_ratio=0.5,
            )
        )

        self.wait(7.5)

        self.play(
            LaggedStart(
                Uncreate(melting_exag, run_time=1.5),
                ShowCreation(rain_exag, run_time=1.5),
                lag_ratio=0.5,
            )
        )

        self.wait(1.5)

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

        self.wait(1)

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


class Lots(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        rhi_cells = get_olympex_rhi_cells()
        rhi_origin = np.array([-6.2, -1.7, 0])
        rhi_scale = np.array([0.2, 0.4, 1])

        def rhi_point(range_km, height_km):
            return rhi_origin + rhi_scale * [range_km, height_km, 0]

        corners = np.pad(rhi_cells["corners"], ((0, 0), (0, 0), (0, 1)))
        corners = rhi_origin + corners * rhi_scale
        points = corners[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
        rhi_image = (
            TriangleMeshMobject(
                points,
                np.broadcast_to(OUT, points.shape),
                np.repeat(rhi_cells["rgba"], 6, axis=0),
            )
            .set_shading(0, 0, 0)
            .deactivate_depth_test()
        )
        rhi_axes = VGroup()
        rhi_labels = VGroup()
        for x in (0, 10, 20, 30):
            rhi_axes.add(Line(rhi_point(x, 0), rhi_point(x, 8)))
            rhi_labels.add(
                Text(str(x), font=FONT, font_size=16).next_to(
                    rhi_point(x, 0), DOWN, SMALL_BUFF
                )
            )
        for z in (0, 2, 4, 6, 8):
            rhi_axes.add(Line(rhi_point(0, z), rhi_point(32.8, z)))
            rhi_labels.add(
                Text(str(z), font=FONT, font_size=16).next_to(
                    rhi_point(0, z), LEFT, SMALL_BUFF
                )
            )
        rhi_axes.set_stroke(WHITE, 1, 0.15)
        rhi_labels.add(
            Text("Range (km)", font=FONT, font_size=20).move_to([-2.92, -2.25, 0]),
            Text("Height (km)", font=FONT, font_size=20)
            .rotate(PI / 2)
            .move_to([-6.85, -0.1, 0]),
            Text("NPOL / OLYMPEX / 13 Nov 2015", font=FONT, font_size=16).move_to(
                [-2.92, 1.92, 0]
            ),
        )
        colorbar = VGroup()
        colorbar_rgba = (
            field_to_rgba(
                np.linspace(10, 50, 128)[None, :],
                vmin=10,
                vmax=50,
                cmap_name="NWSRef",
                fallback_cmap_name="NWSRef",
            )[0]
            / 255
        )
        for index, rgba in enumerate(colorbar_rgba):
            colorbar.add(
                Rectangle(
                    width=6.56 / 128 + 0.002,
                    height=0.15,
                    fill_color=rgb_to_color(rgba[:3]),
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([-6.2 + (index + 0.5) * 6.56 / 128, -2.72, 0])
            )
        for value in (10, 20, 30, 40, 50):
            colorbar.add(
                Text(str(value), font=FONT, font_size=16).move_to(
                    [-6.2 + (value - 10) * 6.56 / 40, -3.02, 0]
                )
            )
        colorbar.add(Text("dBZ", font=FONT, font_size=18).move_to([-2.92, -3.4, 0]))
        rhi_plot = Group(rhi_image, rhi_axes, rhi_labels, colorbar)
        rhi_plot.shift(DOWN * fh(self))

        self.play(rhi_plot.animate.shift(UP * fh(self)), run_time=2)

        self.wait(0.5)

        rng = np.random.default_rng(7)
        diameters = np.linspace(0.05, 8, 400)
        snow_density = diameters * np.exp(-diameters / 1.2)
        rain_density = diameters**2 * np.exp(-diameters / 0.25)
        wet_snow_density = diameters * np.exp(-diameters / 0.9)
        snow_density /= np.trapezoid(snow_density, diameters)
        rain_density /= np.trapezoid(rain_density, diameters)
        wet_snow_density /= np.trapezoid(wet_snow_density, diameters)
        melting_density = 0.7 * wet_snow_density + 0.3 * rain_density
        permittivity_colors = get_color_map("coolwarm")
        gate_centers = rhi_cells["corners"].mean(axis=1)
        gate_callouts = []
        snow_particles = VGroup()
        melting_particles = VGroup()
        rain_particles = VGroup()

        for title, height, row_y, density, melt_fraction, particles in (
            ("Snow / ice + air", 4.5, 2.3, snow_density, 0, snow_particles),
            ("Melting / ice + water", 2.35, 0, melting_density, 0.5, melting_particles),
            ("Rain / liquid water", 0.9, -2.3, rain_density, 1, rain_particles),
        ):
            gate_index = np.argmin(np.sum((gate_centers - [26, height]) ** 2, axis=1))
            gate = Polygon(
                *corners[gate_index],
                stroke_color=YELLOW,
                stroke_width=3,
                fill_opacity=0,
            ).set_z_index(4)
            gate_ring = Circle(radius=0.12, color=YELLOW, stroke_width=2).move_to(gate)
            population_box = Rectangle(
                width=2.2,
                height=1.25,
                stroke_color=GREY_B,
                stroke_width=1,
                stroke_opacity=0.5,
            ).move_to([2.65, row_y - 0.08, 0])
            population_title = Text(title, font=FONT, font_size=22).move_to(
                [4.05, row_y + 0.88, 0]
            )
            particle_diameters = np.sort(
                rng.choice(diameters, 24, p=density / density.sum())
            )[::-1]
            particle_positions = []
            for diameter in particle_diameters:
                radius = 0.036 * diameter
                for _ in range(300):
                    position = np.array(
                        [
                            rng.uniform(-1.05 + radius, 1.05 - radius),
                            rng.uniform(-0.575 + radius, 0.575 - radius),
                            0,
                        ]
                    )
                    if all(
                        np.linalg.norm(position - point) > radius + other_radius + 0.035
                        for point, other_radius in particle_positions
                    ):
                        break
                particle_positions.append((position, radius))
                water_fraction = melt_fraction
                if melt_fraction == 0.5:
                    rain_probability = np.interp(
                        diameter, diameters, 0.3 * rain_density / melting_density
                    )
                    water_fraction = (
                        1
                        if rng.random() < rain_probability
                        else np.clip(1 - diameter / 6 + rng.normal(0, 0.1), 0.15, 0.9)
                    )
                color = rgb_to_color(
                    permittivity_colors(0.15 + 0.7 * water_fraction)[:3]
                )
                snowflake = VGroup(
                    *[
                        Line(-radius * RIGHT, radius * RIGHT).rotate(angle)
                        for angle in (0, PI / 3, 2 * PI / 3)
                    ]
                ).set_stroke(color, 1.5)
                water = Circle(radius=radius).set_fill(color, 0.65).set_stroke(color, 1)
                if water_fraction == 0:
                    particle = snowflake
                elif water_fraction == 1:
                    particle = water.stretch(max(0.7, 1 - 0.06 * diameter), 1)
                else:
                    particle = VGroup(
                        water.set_fill(color, 0.15 + 0.4 * water_fraction),
                        snowflake.scale(1 - water_fraction),
                    )
                particles.add(particle.move_to(population_box.get_center() + position))

            axes = VGroup(
                Line([4.35, row_y - 0.65, 0], [6.55, row_y - 0.65, 0]),
                Line([4.35, row_y - 0.65, 0], [4.35, row_y + 0.55, 0]),
            ).set_stroke(GREY_B, 1.3)
            size_labels = VGroup(
                Text("Relative number", font=FONT, font_size=14).move_to(
                    [5.45, row_y + 0.48, 0]
                ),
                Text("Diameter (mm)", font=FONT, font_size=16).move_to(
                    [5.45, row_y - 1.06, 0]
                ),
                *[
                    Text(str(diameter), font=FONT, font_size=14).move_to(
                        [4.35 + diameter * 2.2 / 8, row_y - 0.82, 0]
                    )
                    for diameter in (0, 4, 8)
                ],
            )
            curve = (
                VMobject()
                .set_points_as_corners(
                    np.column_stack(
                        (
                            4.35 + diameters * 2.2 / 8,
                            row_y - 0.65 + 0.92 * density / density.max(),
                            np.zeros_like(diameters),
                        )
                    )
                )
                .set_stroke(
                    rgb_to_color(permittivity_colors(0.15 + 0.7 * melt_fraction)[:3]), 3
                )
            )
            bez = CubicBezier(
                gate_ring.get_right(),
                gate_ring.get_right() + RIGHT * 1.25,
                population_box.get_left() + LEFT * 0.7,
                population_box.get_left(),
                stroke_color=YELLOW,
                stroke_width=2,
            ).set_z_index(3)
            gate_callouts.append(
                LaggedStart(
                    AnimationGroup(ShowCreation(gate), ShowCreation(gate_ring)),
                    ShowCreation(bez),
                    AnimationGroup(FadeIn(population_box), FadeIn(population_title)),
                    LaggedStart(
                        *[GrowFromCenter(m) for m in particles], lag_ratio=0.05
                    ),
                    AnimationGroup(
                        FadeIn(axes), FadeIn(size_labels), ShowCreation(curve)
                    ),
                    lag_ratio=0.25,
                    run_time=3,
                )
            )

        self.play(LaggedStart(*gate_callouts, lag_ratio=0.55))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=utils.rate_functions.there_and_back).shift(
                        UP * 0.3
                    )
                    for m in [*snow_particles, *melting_particles, *rain_particles]
                ],
                lag_ratio=0.02,
            )
        )

        self.wait(0.5)

        selected_pixel_index = np.argmin(
            np.sum((gate_centers - [26, 2.35]) ** 2, axis=1)
        )
        selected_pixel = Polygon(
            *corners[selected_pixel_index],
            fill_color=rgb_to_color(rhi_cells["rgba"][selected_pixel_index, :3]),
            fill_opacity=1,
            stroke_width=0,
        ).set_z_index(5)
        self.add(selected_pixel)

        dim_animations = []
        for m in self.get_mobject_family_members():
            if m is self.frame or m is selected_pixel or not m.has_points():
                continue
            if isinstance(m, VMobject):
                dim_animations.append(
                    m.animate.set_fill(
                        opacity=0.1 if m.get_fill_opacity() else 0, recurse=False
                    ).set_stroke(
                        opacity=0.1 if m.get_stroke_opacity() else 0, recurse=False
                    )
                )
            else:
                dim_animations.append(m.animate.set_opacity(0.1, recurse=False))

        self.play(
            self.frame.animate.move_to(selected_pixel).set_height(
                selected_pixel.get_height() * 5
            ),
            *dim_animations,
            run_time=3,
        )

        self.wait(0.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

        self.embed()


class Wrapup(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        scroll_speed = 0.9

        python_logo = ImageMobject(
            "../props/static/python-logo-only.png"
        ).rescale_to_fit(fh(self, 0.4), 1)
        nb_title = Text("dsd.py", font=FONT)
        nb_group = Group(python_logo, nb_title).arrange(DOWN, buff=MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                GrowFromCenter(python_logo),
                Write(nb_title),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(python_logo), FadeOut(nb_title))

        self.wait(0.5)

        i_saved = Text(
            "I saved these comments in my notes\nto keep me motivated",
            font=FONT,
            font_size=26,
            alignment="CENTER",
        ).next_to(self.frame.get_bottom(), UP, MED_LARGE_BUFF)

        comment_excerpts = [
            (
                "@timwildauer5063",
                'This is close to the fastest I\'ve ever hit "subscribe" '
                "after watching a video.",
            ),
            (
                "@JonBrami",
                "Btw I got my dream RF job last month and just wanted "
                "to thank you for your content.",
            ),
            (
                "@theluftwaffel",
                "Your videos and their great visuals really helped and I'm doing "
                "great now. Thanks again.",
            ),
            (
                "@AlphaSteak58",
                "Super clear explanations and amazing production value. "
                "Thanks for what you do.",
            ),
            # (
            #     "@Deveyus",
            #     "SOMEONE finally said the quiet part out loud. Thank you.",
            # ),
            # ("@reyariass", "The hero we needed"),
            (
                "@OverlordOfNerds",
                "That moment after midnight when it all finally clicks and I feel "
                "enlightened and fascinated at the same time...",
            ),
            (
                "@arthurlondero7036",
                "...your channel has been one of the most valuable finds of my year. "
                "Thank you so much!!!",
            ),
            # (
            #     "@Yitz2",
            #     "Please keep making these, these are honestly amazing.",
            # ),
            (
                "@KrisPvideos",
                "Again a video which is a treasure for humanity.",
            ),
            # (
            #     "@bstanis1237",
            #     "Please, keep going in this content, you don't know how much you aid us",
            # ),
            # (
            #     "@mabo-jq7ug",
            #     "...this is by far the best combination of teaching I have seen...",
            # ),
            # (
            #     "@cubbyhoo",
            #     "...these videos are an amazing resource for trying to learn "
            #     "about this incredibly complex world.",
            # ),
            # (
            #     "@Coder.tahsin",
            #     "...luckily people like you exist who can make complex things "
            #     "easy to understand for people like me...",
            # ),
            # (
            #     "@aposdfzxc23",
            #     "This is an incredibly well produced educational series on the topic "
            #     "and you nailed the presentation.",
            # ),
            (
                "@neilrodricks8630",
                "If not for your videos I would have given up studying radar...",
            ),
            # (
            #     "@yetanotherPC",
            #     "Very intelligent and an incredible storyteller.",
            # ),
            (
                "@kingjamez80",
                "Great video on why chirps are perfect for radar. "
                "Keep it up, I'll watch anything you make.",
            ),
            # (
            #     "@anthonyecheverry9218",
            #     "Out of the loop on next vid but would love a segment "
            #     "about spurs and harmonics!",
            # ),
            (
                "@rfdspguru2",
                "No matter what you choose I'm sure it will be great!",
            ),
        ]

        comments = VGroup(
            *[
                VGroup(
                    Text(author, font=FONT, font_size=28, fill_color=ORANGE),
                    Text(
                        excerpt,
                        font=FONT,
                        font_size=36,
                        alignment="LEFT",
                        line_width=fw(self, 0.44),
                    ),
                ).arrange(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF)
                for author, excerpt in comment_excerpts
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        comments.next_to(self.frame.get_bottom(), DOWN, buff=0)

        caption_background = Rectangle(
            width=fw(self, 1),
            height=i_saved.get_height() + 2 * MED_LARGE_BUFF,
            stroke_width=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        ).next_to(self.frame.get_bottom(), UP, buff=0)

        scroll_distance = comments.get_height() + fh(self, 1) + MED_SMALL_BUFF
        scroll_duration = scroll_distance / scroll_speed

        merch = Group(
            *[
                ImageMobject(f"../props/static/{filename}").rescale_to_fit(
                    fh(self, 0.4), 1
                )
                for filename in [
                    "engineering_notebook_cover.png",
                    "phased_array_shirt_back.webp",
                    "radar_equation_mug.webp",
                    "dual_pol_mug.webp",
                ]
            ]
        ).arrange_in_grid(n_cols=2, buff=MED_SMALL_BUFF)
        merch.move_to(RIGHT * fw(self, 0.25) + UP * fh(self, 0.06))

        member_names = [
            "ZacJW",
            "db-isJustARatio",
            "misspeled",
            "Florian Euchner",
            "w1gx",
            "alikarb0724",
            "Mark Humes",
            "Bob Van Valzah",
            "Randy Gumabong",
            "noleteman",
            # "ebkimaru",
            "Caboose",
            "Michael Shear",
            "Rich",
            "Russell Newman",
            "DelAntenna",
            "Soren Welling",
        ]
        members_header = (
            VGroup(Text("Huge thanks", font=FONT), Text("to the Members!", font=FONT))
            .arrange(DOWN, buff=MED_SMALL_BUFF)
            .rescale_to_fit(fw(self, 0.35), 0)
        )
        members_header[1][-8:-1].set_fill(color=ORANGE)
        members = VGroup(*[Text(name, font=FONT) for name in member_names]).arrange(
            DOWN, buff=MED_SMALL_BUFF
        )
        members.rescale_to_fit(fh(self, 0.58), 1)
        member_thanks = VGroup(members_header, members).arrange(
            DOWN, buff=MED_LARGE_BUFF
        )
        member_thanks.move_to(RIGHT * fw(self, 0.25) + UP * fh(self, 0.06))

        self.add(comments, caption_background)

        scroll_start_time = self.time
        scroll_start_y = comments.get_y()
        comments.add_updater(
            lambda m: m.set_y(
                scroll_start_y + scroll_speed * (self.time - scroll_start_time)
            )
        )

        self.play(Write(i_saved))

        self.wait(6)

        self.play(
            comments.animate(suspend_mobject_updating=False).set_x(-fw(self, 0.22)),
            i_saved.animate.set_x(-fw(self, 0.22)),
            run_time=1.5,
        )

        website_title = (
            Text("marshallbruner.com", font=FONT, fill_color=ORANGE)
            .next_to(merch, DOWN)
            .set_y(i_saved.get_y())
        )

        self.play(
            LaggedStart(
                *[GrowFromCenter(m, rate_func=ease_out_back) for m in merch],
                Write(website_title),
                lag_ratio=0.3,
                run_time=2,
            )
        )

        self.wait(4)

        self.play(
            merch.animate.shift(
                UP * (self.frame.get_top()[1] - merch.get_bottom()[1] + MED_LARGE_BUFF)
            ),
            FadeOut(website_title),
            run_time=1.5,
        )
        self.remove(merch)

        self.play(Write(members_header))

        self.play(Write(members), run_time=3)

        self.wait(max(0, scroll_start_time + scroll_duration - self.time))
        comments.clear_updaters()

        self.embed()


def get_thumbnail_rhi_pixels(
    scene, zoom_pixel_heights, dim_opacity, pixel_anchor, highlight="pixel"
):
    """Zoomed OLYMPEX RHI cells with the melting-layer gate at full opacity.

    Matches the end of Lots. Returns (rhi_image, highlight_mobject); cells past
    the ragged right boundary are dropped whole rather than clipped.
    highlight="pixel" lights the one gate and returns its Polygon;
    highlight="column" lights every gate at that range and returns a VGroup
    of outlines, one per contiguous run of valid gates.
    """
    rhi_cells = get_olympex_rhi_cells()
    rhi_scale = np.array([0.2, 0.4, 1])
    corners = np.pad(rhi_cells["corners"], ((0, 0), (0, 0), (0, 1))) * rhi_scale
    gate_centers = rhi_cells["corners"].mean(axis=1)
    selected = np.argmin(np.sum((gate_centers - [26, 2.35]) ** 2, axis=1))
    pixel_height = np.ptp(corners[selected, :, 1])
    zoom = fh(scene) / (zoom_pixel_heights * pixel_height)
    corners = (corners - corners[selected].mean(axis=0)) * zoom + pixel_anchor
    # Ragged right edge: whole cells only, dropped with rising probability
    # across a band so the boundary reads as scattered pixels, not a cut.
    edge_rng = np.random.default_rng(3)
    cell_centers = corners.mean(axis=1)
    edge_x = -0.8 - 0.6 * (2 * cell_centers[:, 1] / fh(scene)) ** 2
    keep_probability = np.clip((edge_x - cell_centers[:, 0]) / 2.2 + 0.5, 0, 1)
    keep_probability[corners[..., 0].max(axis=1) > 0.1] = 0
    in_window = (
        (edge_rng.random(len(corners)) < keep_probability)
        & (corners[..., 0].max(axis=1) > -fw(scene, 0.5))
        & (corners[..., 1].max(axis=1) > -fh(scene, 0.5))
        & (corners[..., 1].min(axis=1) < fh(scene, 0.5))
    )
    in_window[selected] = True
    if highlight == "column":
        lit = in_window & np.isclose(
            rhi_cells["ranges_km"], rhi_cells["ranges_km"][selected]
        )
    else:
        lit = np.zeros(len(corners), dtype=bool)
        lit[selected] = True
    rgba = rhi_cells["rgba"].copy()
    rgba[:, 3] = dim_opacity
    rgba[lit, 3] = 1
    points = corners[in_window][:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
    rhi_image = (
        TriangleMeshMobject(
            points,
            np.broadcast_to(OUT, points.shape),
            np.repeat(rgba[in_window], 6, axis=0),
        )
        .set_shading(0, 0, 0)
        .deactivate_depth_test()
    )
    if highlight != "column":
        selected_pixel = Polygon(
            *corners[selected],
            fill_color=rgb_to_color(rhi_cells["rgba"][selected, :3]),
            fill_opacity=1,
            stroke_color=YELLOW,
            stroke_width=6,
        ).set_z_index(3)
        return rhi_image, selected_pixel

    # Outline each contiguous run of lit gates; adjacent cells share edges, so
    # walk up the far-range side and back down the near-range side.
    ray_order = np.unique(rhi_cells["elevation_deg"])
    lit_indices = np.flatnonzero(lit)
    lit_indices = lit_indices[np.argsort(rhi_cells["elevation_deg"][lit_indices])]
    ray_indices = np.searchsorted(ray_order, rhi_cells["elevation_deg"][lit_indices])
    outlines = VGroup()
    run_start = 0
    for k in range(1, len(lit_indices) + 1):
        if k < len(lit_indices) and ray_indices[k] == ray_indices[k - 1] + 1:
            continue
        run = corners[lit_indices[run_start:k]]
        outlines.add(
            Polygon(
                run[0, 0],
                run[0, 1],
                *run[:, 2],
                *run[::-1][:, 3],
                stroke_color=YELLOW,
                stroke_width=6,
                fill_opacity=0,
            )
        )
        run_start = k
    return rhi_image, outlines.set_z_index(3)


class Thumbnail(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}
    samples = 4

    def construct(self):
        zoom_pixel_heights = 6
        dim_opacity = 0.15
        pixel_anchor = np.array([-fw(self, 0.25), 0, 0])
        row_scale = 1.75
        box_center = np.array([0.2 + 1.1 * row_scale, fh(self, 0.26), 0])
        dipole_center = np.array([fw(self, 0.25) - 0.45, -fh(self, 0.25), 0])
        dipole_tilt_deg = [25, -40, -15]
        dipole_time = 3.35

        self.frame.set_focal_distance(fh(self, 6))

        rhi_image, selected_pixel = get_thumbnail_rhi_pixels(
            self, zoom_pixel_heights, dim_opacity, pixel_anchor
        )

        # Right top: melting-layer particle population + size distribution (from Lots)
        rng = np.random.default_rng(7)
        diameters = np.linspace(0.05, 8, 400)
        rain_density = diameters**2 * np.exp(-diameters / 0.25)
        wet_snow_density = diameters * np.exp(-diameters / 0.9)
        rain_density /= np.trapezoid(rain_density, diameters)
        wet_snow_density /= np.trapezoid(wet_snow_density, diameters)
        melting_density = 0.7 * wet_snow_density + 0.3 * rain_density
        permittivity_colors = get_color_map("coolwarm")
        box_width = 2.2 * row_scale
        box_height = 1.25 * row_scale
        population_box = Rectangle(
            width=box_width,
            height=box_height,
            stroke_color=GREY_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ).move_to(box_center)
        particle_diameters = np.sort(
            rng.choice(diameters, 24, p=melting_density / melting_density.sum())
        )[::-1]
        particle_positions = []
        melting_particles = VGroup()
        for diameter in particle_diameters:
            radius = 0.036 * row_scale * diameter
            for _ in range(300):
                position = np.array(
                    [
                        rng.uniform(
                            -box_width / 2 + radius + 0.05, box_width / 2 - radius - 0.05
                        ),
                        rng.uniform(
                            -box_height / 2 + radius + 0.05,
                            box_height / 2 - radius - 0.05,
                        ),
                        0,
                    ]
                )
                if all(
                    np.linalg.norm(position - point)
                    > radius + other_radius + 0.035 * row_scale
                    for point, other_radius in particle_positions
                ):
                    break
            particle_positions.append((position, radius))
            rain_probability = np.interp(
                diameter, diameters, 0.3 * rain_density / melting_density
            )
            water_fraction = (
                1
                if rng.random() < rain_probability
                else np.clip(1 - diameter / 6 + rng.normal(0, 0.1), 0.15, 0.9)
            )
            color = rgb_to_color(permittivity_colors(0.15 + 0.7 * water_fraction)[:3])
            snowflake = VGroup(
                *[
                    Line(-radius * RIGHT, radius * RIGHT).rotate(angle)
                    for angle in (0, PI / 3, 2 * PI / 3)
                ]
            ).set_stroke(color, 2.5)
            water = Circle(radius=radius).set_fill(color, 0.65).set_stroke(color, 2)
            if water_fraction == 1:
                particle = water.stretch(max(0.7, 1 - 0.06 * diameter), 1)
            else:
                particle = VGroup(
                    water.set_fill(color, 0.15 + 0.4 * water_fraction),
                    snowflake.scale(1 - water_fraction),
                )
            melting_particles.add(particle.move_to(box_center + position))

        axes_origin = box_center + [box_width / 2 + 0.15 * row_scale, -0.65 * row_scale, 0]
        axes_width = 1.55 * row_scale
        axes_height = 1.2 * row_scale
        dsd_axes = VGroup(
            Line(axes_origin, axes_origin + RIGHT * axes_width),
            Line(axes_origin, axes_origin + UP * axes_height),
        ).set_stroke(GREY_B, 3)
        dsd_curve = (
            VMobject()
            .set_points_as_corners(
                axes_origin
                + np.column_stack(
                    (
                        diameters * axes_width / 8,
                        0.92 * row_scale * melting_density / melting_density.max(),
                        np.zeros_like(diameters),
                    )
                )
            )
            .set_stroke(rgb_to_color(permittivity_colors(0.5)[:3]), 6)
        )
        callout = CubicBezier(
            selected_pixel.get_right() + RIGHT * 0.1,
            selected_pixel.get_right() + RIGHT * 1.6,
            population_box.get_left() + LEFT * 1.2,
            population_box.get_left() + LEFT * 0.1,
            stroke_color=YELLOW,
            stroke_width=4,
        ).set_z_index(3)

        # Right bottom: dipole with radiated field and voltage profile (from DipoleP2)
        dipole = DipoleAntenna(radius=0.09).scale(0.92).move_to(dipole_center)
        dipole_rotation = Rotation.from_euler(
            "xyz", dipole_tilt_deg, degrees=True
        ).as_matrix()
        dipole.apply_matrix(dipole_rotation, about_point=dipole.get_center())
        dipole.apply_depth_test()
        self.camera.light_source.move_to(
            dipole.get_center() + self.camera.light_source_position
        )

        dipole_origin = (
            dipole.left_arm.get_center() + dipole.right_arm.get_center()
        ) / 2
        dipole_axis, feed_axis, radiation_axis = dipole_rotation.T
        dipole_length = np.ptp(dipole.arms.get_all_points() @ dipole_axis)
        feed_gap = dipole.feed_gap * dipole_length / dipole.length
        antenna_time = VT(dipole_time)
        antenna_frequency = 1
        # Shorter than the physical 2 * dipole_length so two humps fit per side
        radiation_wavelength = dipole_length
        radiation_speed = antenna_frequency * radiation_wavelength
        radiation_start = dipole_length * 0.35
        radiation_range = dipole_length * 1.35

        def dipole_point(x=0, y=0, z=0):
            return dipole_origin + x * dipole_axis + y * feed_axis + z * radiation_axis

        def get_antenna_motion(time):
            time = np.asarray(time, dtype=float)
            ramp = np.clip(time, 0, 1)
            envelope = 10 * ramp**3 - 15 * ramp**4 + 6 * ramp**5
            envelope_speed = 30 * ramp**2 * (1 - ramp) ** 2
            envelope_acceleration = 60 * ramp * (1 - ramp) * (1 - 2 * ramp)
            omega = TAU * antenna_frequency
            phase = omega * time
            displacement = envelope * np.sin(phase)
            acceleration = (
                envelope_acceleration * np.sin(phase)
                + 2 * omega * envelope_speed * np.cos(phase)
                - omega**2 * envelope * np.sin(phase)
            )
            return displacement, acceleration

        def get_radiated_field(distances):
            _, acceleration = get_antenna_motion(
                ~antenna_time - distances / radiation_speed
            )
            return (
                -acceleration
                / (TAU * antenna_frequency) ** 2
                * dipole_length
                * 0.7
                * (radiation_start / distances) ** 0.35
            )

        def get_voltage_profile():
            displacement, _ = get_antenna_motion(~antenna_time)
            profiles = VGroup()
            if abs(displacement) < 1e-8:
                return profiles
            for side in [-1, 1]:
                positions = side * np.linspace(feed_gap / 2, dipole_length / 2, 41)
                voltages = (
                    dipole_length
                    * 0.25
                    * displacement
                    * np.sin(PI * positions / dipole_length)
                )
                points = [dipole_point(x, y) for x, y in zip(positions, voltages)]
                profiles.add(
                    Polygon(
                        dipole_point(positions[0]),
                        *points,
                        dipole_point(positions[-1]),
                        fill_color=RX_COLOR,
                        fill_opacity=0.5,
                        stroke_width=0,
                    ).deactivate_depth_test()
                )
            return profiles

        # Offset by half a step so no line lands on a zero crossing
        line_distances = np.arange(
            radiation_start + radiation_wavelength / 16,
            radiation_range,
            radiation_wavelength / 8,
        )
        radiation_curves = VGroup(
            *[VMobject().set_stroke(TX_COLOR, 7) for _ in [-1, 1]]
        )
        radiation_lines = VGroup(
            *[
                VGroup(
                    *[
                        Line(ORIGIN, RIGHT).set_stroke(TX_COLOR, 4, 0.9)
                        for _ in line_distances
                    ]
                )
                for _ in [-1, 1]
            ]
        )
        antenna_field = VGroup(radiation_lines, radiation_curves).deactivate_depth_test()

        def update_antenna_field(m):
            front = min(radiation_range, radiation_speed * ~m)
            line_values = get_radiated_field(line_distances)
            for side, curve, lines in zip([-1, 1], radiation_curves, radiation_lines):
                if front > radiation_start + 1e-4:
                    distances = np.linspace(radiation_start, front, 161)
                    values = get_radiated_field(distances)
                    curve.set_points_smoothly(
                        dipole_origin
                        + side * distances[:, None] * radiation_axis
                        + values[:, None] * dipole_axis
                    )
                else:
                    curve.clear_points()
                for distance, value, line in zip(line_distances, line_values, lines):
                    if abs(value) < 1e-5:
                        line.set_stroke(opacity=0)
                        continue
                    base = dipole_point(z=side * distance)
                    line.put_start_and_end_on(base, base + value * dipole_axis)
                    line.set_stroke(opacity=0.9)

        antenna_time.add_updater(update_antenna_field)
        voltage_profile = always_redraw(get_voltage_profile)
        radiation_guides = VGroup(
            *[
                DashedLine(
                    dipole_point(z=side * dipole_length * 0.15),
                    dipole_point(z=side * radiation_range),
                    dash_length=0.12,
                    stroke_color=ICE,
                    stroke_width=1,
                    stroke_opacity=0.25,
                )
                for side in [-1, 1]
            ]
        ).deactivate_depth_test()

        self.add(rhi_image, selected_pixel, callout)
        self.add(population_box, melting_particles, dsd_axes, dsd_curve)
        self.add(radiation_guides, antenna_field, voltage_profile, dipole, antenna_time)

        self.wait(0.5)

        self.embed()


def segment_distance(a1, b1, a2, b2):
    """Shortest distance between 2D segments a1-b1 and a2-b2."""

    def point_to_segment(point, a, b):
        ab = b - a
        t = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return np.linalg.norm(point - (a + t * ab))

    def orientation(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    if orientation(a1, b1, a2) != orientation(a1, b1, b2) and orientation(
        a2, b2, a1
    ) != orientation(a2, b2, b1):
        return 0.0
    return min(
        point_to_segment(a1, a2, b2),
        point_to_segment(b1, a2, b2),
        point_to_segment(a2, a1, b1),
        point_to_segment(b2, a1, b1),
    )


class Thumbnail2(ThreeDScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}
    default_frame_orientation = (0, 0)
    always_depth_test = False
    samples = 4

    def construct(self):
        field_time = 9.0
        drop_height = fh(self, 0.42)
        drop_field_radius = 2.0
        line_width = 12
        highlight_width = 16
        charge_radius = fh(self, 0.039)
        charge_separation = fh(self, 0.1)
        n_pairs = 5

        # Field lines from the oscillating charge, thick enough to survive downscaling
        field_scale = drop_height / 2 / drop_field_radius
        field_extent = np.hypot(fw(self), fh(self)) / (2 * field_scale) + 1
        curves = trace_field_lines(field_time, field_extent)[::2]
        curves = curves - UP * charge_motion(field_time)[0]
        count = len(curves)
        field_lines = VGroup()
        for index, points in enumerate(curves):
            highlighted = index in (count // 4, 3 * count // 4)
            line = VMobject(
                stroke_color=YELLOW if highlighted else BLUE_B,
                stroke_width=highlight_width if highlighted else line_width,
                stroke_opacity=1 if highlighted else 0.85,
            )
            line.set_points_smoothly(field_scale * points[::2])
            field_lines.add(line)
        field_lines.deactivate_depth_test()

        raindrop = RainDrop(opacity=0.92).set_height(drop_height).move_to(ORIGIN)

        # Polarized charge pairs: same orientation and size, random non-overlapping spots
        rng = np.random.default_rng(18)
        drop_center = raindrop.get_center()
        drop_radii = np.array([raindrop.get_width(), raindrop.get_height(), 1]) / 2
        margin = 1 - (charge_radius / drop_radii[1]) - 0.04
        clearance = 2 * charge_radius + 0.2 * charge_radius
        placed_ends = []
        for _ in range(n_pairs):
            for _ in range(3000):
                fraction = rng.uniform(-0.85, 0.85, 3)
                fraction[2] = 0
                center = drop_center + fraction * drop_radii
                ends = [center + sign * charge_separation / 2 * UP for sign in (1, -1)]
                if any(
                    np.hypot(*((end - drop_center) / drop_radii)[:2]) > margin
                    for end in ends
                ):
                    continue
                if all(
                    segment_distance(ends[0][:2], ends[1][:2], a[:2], b[:2]) > clearance
                    for a, b in placed_ends
                ):
                    placed_ends.append(ends)
                    break

        charge_pairs = VGroup()
        connectors = VGroup()
        for top, bottom in placed_ends:
            pair = ChargePair(
                center=(top + bottom) / 2,
                angle=PI / 2,
                radius=charge_radius,
                separation=charge_separation,
                font=FONT,
                negative_color=ICE,
                symbol_color=BACKGROUND_COLOR,
            )
            # Strokes render above fills within one group, so the connector lives
            # in its own layer under the dots and stops short of their centers
            pair.remove(pair.connector)
            connectors.add(
                Line(
                    top - UP * 0.8 * charge_radius,
                    bottom + UP * 0.8 * charge_radius,
                    stroke_color=WHITE,
                    stroke_width=7,
                    stroke_opacity=0.8,
                ).deactivate_depth_test()
            )
            charge_pairs.add(pair)

        self.add(field_lines, raindrop, connectors, charge_pairs, perp_stroke=False)

        self.wait(0.5)

        self.embed()


class Thumbnail3(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}
    samples = 4

    def construct(self):
        zoom_pixel_heights = 6
        dim_opacity = 0.15
        pixel_anchor = np.array([-fw(self, 0.25), 0, 0])
        drop_height = fh(self, 0.38)
        drop_center = np.array([fw(self, 0.25) - 0.25, 0.9, 0])
        drop_field_radius = 2.0
        field_time = 9.0
        # Field lines fade out before they reach the pixel mosaic
        fade_start_x = 0.6
        fade_end_x = -0.8
        line_width = 7
        highlight_width = 10
        side_line_width = 4
        # Azimuthal planes of the field about the dipole axis (first one is the
        # screen plane) and the tilt of the whole structure. [0] with no tilt is
        # the flat 2D look; e.g. [0, 45, 135] with [22, 10, 0] gives 3D planes.
        plane_azimuths_deg = [0]
        structure_tilt_deg = [0, 0, 0]
        # n_pairs = 1 puts one big +/- pair on the drop's poles (net dipole)
        n_pairs = 5
        charge_radius = drop_height * (0.13 if n_pairs == 1 else 0.085)
        charge_separation = drop_height * (0.7 if n_pairs == 1 else 0.23)

        self.frame.set_focal_distance(fh(self, 3))

        rhi_image, selected_pixel = get_thumbnail_rhi_pixels(
            self, zoom_pixel_heights, dim_opacity, pixel_anchor
        )

        # Right: one drop as a radiating antenna, field lines in 3D
        field_scale = drop_height / 2 / drop_field_radius
        field_extent = np.hypot(fw(self), fh(self)) / (2 * field_scale) + 1
        curves = trace_field_lines(field_time, field_extent)
        curves = curves - UP * charge_motion(field_time)[0]
        count = len(curves)
        structure_tilt = Rotation.from_euler(
            "xyz", structure_tilt_deg, degrees=True
        ).as_matrix()
        field_lines = VGroup()
        for plane_index, azimuth_deg in enumerate(plane_azimuths_deg):
            plane_rotation = Rotation.from_euler("y", azimuth_deg, degrees=True).as_matrix()
            for index in range(0, count, 2 if len(plane_azimuths_deg) == 1 else 4):
                on_axis = index % (count // 2) == 0
                if on_axis and plane_index > 0:
                    continue
                highlighted = plane_index == 0 and index in (count // 4, 3 * count // 4)
                points = curves[index][::2] @ plane_rotation.T @ structure_tilt.T
                points = drop_center + field_scale * points
                # Start each line outside the drop's silhouette so nothing is drawn
                # across its face
                outside = np.flatnonzero(
                    np.hypot(*(points[:, :2] - drop_center[:2]).T) > 0.53 * drop_height
                )
                if len(outside) < 4:
                    continue
                points = points[outside[0] :]
                if highlighted:
                    width = highlight_width
                elif plane_index == 0:
                    width = line_width
                else:
                    width = side_line_width
                line = VMobject(
                    stroke_color=ICE if highlighted else BLUE_B, stroke_width=width
                )
                line.set_points_smoothly(points)
                line_points = line.get_points()
                fade = np.clip(
                    (line_points[:, 0] - fade_end_x) / (fade_start_x - fade_end_x), 0, 1
                )
                # Nearer lines brighter than the ones going into the screen
                depth = np.tanh(line_points[:, 2] / fh(self, 0.25))
                if highlighted:
                    base = 1
                elif plane_index == 0:
                    base = 0.85
                else:
                    base = 0.4 + 0.25 * depth
                line.set_stroke(opacity=list(base * fade))
                field_lines.add(line)
        field_lines.apply_depth_test()

        raindrop = RainDrop(opacity=0.92).set_height(drop_height).move_to(drop_center)

        drop_radii = np.array([raindrop.get_width(), raindrop.get_height(), 1]) / 2
        # Fixed layouts (fractions of the drop radii) so every pair fits and reads
        if n_pairs == 1:
            pair_fractions = [(0, 0)]
        else:
            pair_fractions = [
                (-0.28, 0.36),
                (0.28, 0.36),
                (-0.55, -0.36),
                (0, -0.36),
                (0.55, -0.36),
            ]
        placed_ends = []
        for fx, fy in pair_fractions:
            center = drop_center + np.array([fx, fy, 0]) * drop_radii
            placed_ends.append(
                [center + sign * charge_separation / 2 * UP for sign in (1, -1)]
            )

        charge_pairs = VGroup()
        connectors = VGroup()
        for top, bottom in placed_ends:
            pair = ChargePair(
                center=(top + bottom) / 2,
                angle=PI / 2,
                radius=charge_radius,
                separation=charge_separation,
                font=FONT,
                negative_color=ICE,
                symbol_color=BACKGROUND_COLOR,
            )
            pair.remove(pair.connector)
            connectors.add(
                Line(
                    top - UP * 0.8 * charge_radius,
                    bottom + UP * 0.8 * charge_radius,
                    stroke_color=WHITE,
                    stroke_width=4,
                    stroke_opacity=0.8,
                ).deactivate_depth_test()
            )
            charge_pairs.add(pair)

        # Enter the drop below the highlighted field line so the two do not merge
        callout_end = drop_center + (drop_radii[0] + 0.15) * np.array(
            [-np.cos(35 * DEG), -np.sin(35 * DEG), 0]
        )
        callout = CubicBezier(
            selected_pixel.get_right() + RIGHT * 0.1,
            selected_pixel.get_right() + RIGHT * 2.0,
            callout_end + LEFT * 1.5 + DOWN * 0.5,
            callout_end,
            stroke_color=YELLOW,
            stroke_width=4,
        ).set_z_index(3)

        self.add(rhi_image, selected_pixel)
        self.add(field_lines, raindrop, connectors, charge_pairs, callout)

        self.wait(0.5)

        self.embed()


class Thumbnail4(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}
    samples = 4

    def construct(self):
        zoom_pixel_heights = 8
        dim_opacity = 0.15
        pixel_anchor = np.array([-fw(self, 0.25), 0, 0])
        plot_center = np.array([fw(self, 0.25) - 0.3, 0.2, 0])
        plot_tilt_deg = 45
        # Perspective enlarges the near edge, so the on-screen height ends up ~0.92
        plot_height = fh(self, 0.91)
        trace_width = 5
        n_gates = 17
        particles_per_gate = 3

        self.frame.set_focal_distance(fh(self, 3))

        rhi_image, column_outline = get_thumbnail_rhi_pixels(
            self, zoom_pixel_heights, dim_opacity, pixel_anchor, highlight="column"
        )

        # Right: size distribution per gate stacked by height (Dsd waterfall look)
        column_left, column_right = -1.95, 1.95
        rain_bottom, melting_bottom, melting_top, snow_top = -3.0, -0.6, 0.6, 3.0
        diameter_max = 8.0
        trace_scale = 2.3
        gate_edges = np.linspace(rain_bottom, snow_top, n_gates + 1)
        gate_heights = (gate_edges[:-1] + gate_edges[1:]) / 2
        diameters = np.linspace(0.03, diameter_max, 181)
        permittivity_colors = get_color_map("coolwarm")
        snow_shape = diameters**2 * np.exp(-diameters / 1.3)
        rain_shape = diameters**2 * np.exp(-diameters / 0.45)
        snow_shape /= snow_shape.max()
        rain_shape /= rain_shape.max()
        rng = np.random.default_rng(11)

        def diameter_to_x(diameter):
            return column_left + diameter / diameter_max * (column_right - column_left)

        def melt_fraction(height):
            return float(
                np.clip((melting_top - height) / (melting_top - melting_bottom), 0, 1)
            )

        def gate_color(fraction):
            return rgb_to_color(permittivity_colors(0.1 + 0.8 * fraction)[:3])

        column_width = column_right - column_left
        layers = VGroup(
            Rectangle(
                width=column_width,
                height=snow_top - melting_top,
                stroke_width=0,
                fill_color=TOWER_LIGHT,
                fill_opacity=0.05,
            ).move_to([0, (snow_top + melting_top) / 2, 0]),
            Rectangle(
                width=column_width,
                height=melting_top - melting_bottom,
                stroke_width=0,
                fill_color=TEAL,
                fill_opacity=0.18,
            ).move_to([0, 0, 0]),
            Rectangle(
                width=column_width,
                height=melting_bottom - rain_bottom,
                stroke_width=0,
                fill_color=BLUE_E,
                fill_opacity=0.08,
            ).move_to([0, (melting_bottom + rain_bottom) / 2, 0]),
        )
        melting_edges = VGroup(
            *[
                Line([column_left, height, 0], [column_right, height, 0])
                for height in (melting_top, melting_bottom)
            ]
        ).set_stroke(TEAL, 3, opacity=0.7)
        gate_items = []
        for height in gate_heights:
            fraction = melt_fraction(height)
            density = (1 - fraction) * snow_shape + fraction * rain_shape
            amplitude = 0.65 + 0.35 * (1 - fraction) + 0.8 * fraction * (1 - fraction)
            amplitude *= rng.uniform(0.85, 1)
            guide = Line(
                [column_left, height, 0.01], [column_right, height, 0.01]
            ).set_stroke(GREY_B, 1.5, opacity=0.35)
            points = np.column_stack(
                (
                    diameter_to_x(diameters),
                    np.full_like(diameters, height),
                    0.015 + trace_scale * amplitude * density / density.max(),
                )
            )
            trace = VMobject(
                stroke_color=gate_color(fraction), stroke_width=trace_width
            ).set_points_smoothly(points)
            gate_particles = []
            for diameter in rng.choice(
                diameters, particles_per_gate, p=density / density.sum()
            ):
                radius = 0.05 + 0.025 * min(diameter, 5)
                position = np.array(
                    [
                        diameter_to_x(diameter) + rng.uniform(-0.2, 0.2),
                        height + rng.uniform(-0.1, 0.1),
                        0.02,
                    ]
                )
                snowflake = VGroup(
                    *[
                        Line(-radius * RIGHT, radius * RIGHT).rotate(angle)
                        for angle in (0, PI / 3, 2 * PI / 3)
                    ]
                ).set_stroke(ICE, 3)
                if fraction == 0:
                    particle = snowflake
                elif fraction == 1:
                    particle = (
                        Circle(radius=0.8 * radius)
                        .set_fill(BLUE_B, 0.9)
                        .set_stroke(BLUE_B, 2)
                    )
                else:
                    particle = VGroup(
                        Circle(radius=radius)
                        .set_fill(TOWER_LIGHT, 0.3)
                        .set_stroke(TOWER_LIGHT, 2),
                        snowflake.scale(0.6),
                    )
                gate_particles.append((position[1], particle.move_to(position)))
            gate_items.append((height, guide, trace, gate_particles))

        # Painter's order for the tilted view: far gates (top) first, near gates
        # (bottom) last; within a gate, particles behind the trace go before it.
        plot = VGroup(layers, melting_edges)
        for height, guide, trace, gate_particles in reversed(gate_items):
            gate_particles.sort(key=lambda item: -item[0])
            behind = [m for y, m in gate_particles if y > height]
            in_front = [m for y, m in gate_particles if y <= height]
            plot.add(guide, *behind, trace, *in_front)
        plot.apply_matrix(
            Rotation.from_euler("x", -plot_tilt_deg, degrees=True).as_matrix()
        )
        plot.set_height(plot_height).move_to(plot_center)

        callout_start = np.array([column_outline.get_right()[0] + 0.1, pixel_anchor[1], 0])
        callout_end = (
            melting_edges[0].get_start() + melting_edges[1].get_start()
        ) / 2 + LEFT * 0.15
        callout = CubicBezier(
            callout_start,
            callout_start + RIGHT * 2.0,
            callout_end + LEFT * 1.5,
            callout_end,
            stroke_color=YELLOW,
            stroke_width=4,
        ).set_z_index(3)

        self.add(rhi_image, column_outline, plot, callout)

        self.wait(0.5)

        self.embed()
