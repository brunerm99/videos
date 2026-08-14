# zgl.py

import os
import sys
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
    VideoMobject,
    ease_in_sine,
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
    def construct(self): ...


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
        max_prog = 1 + nrads * lag
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

        self.add(router_rads_r, router_rads_l)

        self.play(circ_prog @ max_prog, run_time=8)

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

        rayleigh_label = (
            Text("Rayleigh: ", font=FONT)
            .fix_in_frame()
            .rescale_to_fit(fw(self, 0.2), 0)
            .next_to(
                self.frame.to_fixed_frame_point(raindrop.get_bottom()),
                DOWN,
                LARGE_BUFF,
            )
        )
        rayleigh_condition = (
            Tex(r"D \ll \lambda")
            .fix_in_frame()
            .rescale_to_fit(rayleigh_label.get_height(), 1)
            .next_to(rayleigh_label, RIGHT, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in [*rayleigh_label, *rayleigh_condition]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        bez_d_center = Line(
            self.frame.to_fixed_frame_point(raindrop.get_right()) + [0.1, 0, 0],
            self.frame.to_fixed_frame_point(
                dipole.copy().set_y(raindrop.get_y()).get_left()
            )
            + [-0.1, 0, 0],
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).fix_in_frame()

        self.play(
            LaggedStart(
                LaggedStart(
                    Uncreate(bez_u),
                    yagi.animate(run_time=2).shift(UP * 5),
                    lag_ratio=0,
                ),
                LaggedStart(
                    ReplacementTransform(bez_d, bez_d_center),
                    dipole.animate.set_y(raindrop.get_y()),
                    lag_ratio=0,
                    run_time=2,
                ),
                lag_ratio=0.2,
            )
        )

        dipole_label = (
            Text("Dipole", font=FONT)
            .fix_in_frame()
            .rescale_to_fit(fw(self, 0.3), 0)
            .next_to(
                self.frame.to_fixed_frame_point(dipole.get_top()), UP, MED_LARGE_BUFF
            )
        )

        self.play(Write(dipole_label))

        self.wait(6)

        self.play(
            FadeOut(rayleigh_condition),
            FadeOut(dipole_label),
            FadeOut(rayleigh_label),
            FadeOut(bez_d_center),
        )
        self.play(
            LaggedStart(
                LaggedStart(
                    dipole.animate.shift(RIGHT * 2),
                    self.frame.animate.shift(RIGHT * 5.5),
                    lag_ratio=0,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        er_eqn = (
            Tex(
                r"E_{\theta} = \frac{k_0^2 \mathbf P V}{2 \pi \epsilon_0 r} \sin{\theta}"
            )
            .fix_in_frame()
            .rescale_to_fit(fw(self, 0.4), 0)
            .next_to(
                self.frame.to_fixed_frame_point(dipole.get_right()),
                RIGHT,
                LARGE_BUFF * 2,
            )
        )

        self.play(LaggedStart(*[FadeIn(m) for m in er_eqn], lag_ratio=0.1))

        self.wait(0.5)

        self.play(self.frame.animate.shift(RIGHT * 10), er_eqn.animate.shift(UP * 6))

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

        raindrop = (
            RainDrop()
            .scale(0.2)
            .move_to(self.radar.beam_location_to_point(5, 0 * DEG, 0 * DEG))
        )

        bw = 8 * DEG
        beam = self.radar.get_beam(bw, 10)
        em_wave = beam.get_em_wave(
            wavelength=2,
            n_oscillations=1,
            beamwidth=bw,
            stroke_width=DEFAULT_STROKE_WIDTH,
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

        self.play(
            LaggedStart(
                em_wave.range.animate(run_time=6).set_value(10),
                self.frame.animate(run_time=4).reorient(
                    20, 90, 0, center=raindrop.get_center(), height=2
                ),
                LaggedStart(
                    *[FadeIn(plus) for plus in pluses],
                    lag_ratio=0.15,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(em_wave.range @ 6)

        self.wait(0.5)

        rotate_rate = PI / 9

        def rotate_cam(m, dt):
            m.rotate(rotate_rate * dt, axis=beam.direction)

        self.play(
            self.frame.animate.rotate(rotate_rate, axis=beam.direction),
            rate_func=ease_in_sine,
        )

        self.frame.add_updater(rotate_cam)

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
            self.frame.to_fixed_frame_point(self.radar.beam_location_to_point(4, 0, 0))
            + DOWN * 1,
            self.frame.to_fixed_frame_point(self.radar.beam_location_to_point(4, 0, 0)),
            stroke_width=DEFAULT_STROKE_WIDTH * 3,
        ).fix_in_frame()
        wl_bez_r = CubicBezier(
            wl.get_right() + RIGHT * 0.1,
            wl.get_right() + RIGHT * 1,
            self.frame.to_fixed_frame_point(self.radar.beam_location_to_point(6, 0, 0))
            + DOWN * 2,
            self.frame.to_fixed_frame_point(self.radar.beam_location_to_point(6, 0, 0)),
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

        self.embed()


class BrightBand(InteractiveScene):
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
        profile_label = (
            Text("Reflectivity", font=FONT, font_size=28)
            .set_color(TOWER_LIGHT)
            .next_to(profile_axes[1], DOWN, buff=0.18)
        )

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
                .set_fill(
                    BLUE_A,
                    opacity=0,
                )
                .set_stroke(BLUE_A, 1.5, opacity=0)
            )
            raindrop = Line(
                radius * 1.25 * UP,
                radius * 1.25 * DOWN,
            ).set_stroke(BLUE_A, 3, opacity=0)

            hydrometeors.add(Group(water_shell, snowflake, raindrop))

        elapsed = [0.0]
        current_heights = np.zeros(particle_count)
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
                snowflake.rotate(rotation_rate[index] * dt)

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
            response = 1 if dt == 0 else 1 - np.exp(-dt / 0.18)
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

        self.wait(20)
