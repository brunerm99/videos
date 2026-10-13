import sys
import wave
from pathlib import Path

import numpy as np
from manimlib import *

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from melp import VT, ease_out_back, fh, fw

BACKGROUND_COLOR = "#183340"
PHASE_RING_COLOR = "#8fa3ab"
PHASE_COLOR = "#f07828"
PHASE_TIP_COLOR = "#f09050"
PHASE_ZERO_COLOR = "#2d8fbf"
AF_COLORS = ("#1e2c31", "#2d8fbf", "#3dbcac", "#f09050", "#f07828", "#fc6255")


def frame_projection(camera):
    refresh_uniforms = camera.refresh_uniforms
    uniforms = camera.gpu.frame_uniforms if hasattr(camera, "gpu") else camera.uniforms

    def update_projection():
        refresh_uniforms()
        frame = camera.frame
        scale = frame.get_scale()
        uniforms.update(
            frame_rescale_factors=(
                2 * scale / frame.get_width(),
                2 * scale / frame.get_height(),
                scale / frame.get_focal_distance(),
            )
        )

    return update_projection


def array_factor_magnitude(u, v, u_0, v_0, array_size):
    delta_u = (u - u_0) / 2
    delta_v = (v - v_0) / 2
    return np.abs(
        np.sinc(array_size * delta_u)
        / np.sinc(delta_u)
        * np.sinc(array_size * delta_v)
        / np.sinc(delta_v)
    )


def phase_rotator(center, radius):
    ring = Circle(radius=radius, stroke_color=PHASE_RING_COLOR, stroke_width=2)
    ring.move_to(center)
    zero_tick = Line(
        center + RIGHT * radius * 0.91,
        center + RIGHT * radius * 1.08,
        stroke_color=PHASE_ZERO_COLOR,
        stroke_width=3,
    )
    hand = Line(
        center,
        center + RIGHT * radius * 0.82,
        stroke_color=PHASE_COLOR,
        stroke_width=4,
    )
    tip = Dot(hand.get_end(), radius=radius * 0.13, fill_color=PHASE_TIP_COLOR)
    hub = Dot(center, radius=radius * 0.07, fill_color=PHASE_COLOR)
    return VGroup(ring, zero_tick, VGroup(hand, tip), hub)


def array_factor_surface(
    array_size,
    af_floor_db,
    surface_resolution,
    view_rotation,
    pattern_scale,
    pattern_origin,
):
    pattern = ParametricSurface(
        lambda theta, phi: np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        ),
        u_range=(0, PI / 2),
        v_range=(0, TAU),
        resolution=surface_resolution,
        shading=(0.18, 0.06, 0.25),
    )
    directions = pattern.get_points().copy()
    color_stops = np.array([color_to_rgb(color) for color in AF_COLORS])

    def update_pattern(mobject, u, v):
        amplitude = array_factor_magnitude(
            directions[:, 0], directions[:, 1], u, v, array_size
        )
        af_db = 20 * np.log10(np.maximum(amplitude, 10 ** (af_floor_db / 20)))
        radial_scale = np.clip((af_db - af_floor_db) / -af_floor_db, 0, 1)
        points = directions * radial_scale[:, None]
        mobject.set_points(pattern_scale * (points @ view_rotation.T) + pattern_origin)

        if "du_point" in mobject.data_dtype.names:
            point_grid = mobject.get_points().reshape((*surface_resolution, 3))
            tangent_u, tangent_v = np.gradient(point_grid, axis=(0, 1))
            tangent_v[0] = tangent_v[1]
            mobject.data["du_point"][:] = (point_grid + 1e-4 * tangent_u).reshape(-1, 3)
            mobject.data["dv_point"][:] = (point_grid + 1e-4 * tangent_v).reshape(-1, 3)

        scaled_colors = radial_scale * (len(color_stops) - 1)
        indices = np.minimum(scaled_colors.astype(int), len(color_stops) - 2)
        blend = (scaled_colors - indices)[:, None]
        rgb = color_stops[indices] * (1 - blend) + color_stops[indices + 1] * blend
        mobject.set_rgba_array(np.column_stack([rgb, np.ones(len(rgb))]))

    update_pattern(pattern, 0, 0)
    return pattern, update_pattern


class ArrayPhaseSteering(ThreeDScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "resolution": (1080, 1920),
        "fps": 60,
    }

    def construct(self):
        self.frame.reorient(0, 0)
        self.frame.set_width(4.5).set_height(8, stretch=True)
        self.camera.refresh_uniforms = frame_projection(self.camera)

        array_size = 8
        af_floor_db = -40
        scan_angle = 35 * DEGREES
        sweep_seconds = 3
        circular_scan_seconds = 8
        hold_seconds = 0.7

        u_0 = VT(0)
        v_0 = VT(0)
        scan_progress = VT(0)
        circular_scan = False
        scan_radius = np.sin(scan_angle)

        def steering_at():
            if circular_scan:
                angle = TAU * ~scan_progress
                return scan_radius * np.cos(angle), scan_radius * np.sin(angle)
            return ~u_0, ~v_0

        radius_ratio = 0.36
        pitch = fw(self, 0.94) / (array_size - 1 + 2.08 * radius_ratio)
        radius = pitch * radius_ratio
        phase_center = DOWN * fh(self, 0.08)
        phases = VGroup()
        for row in range(array_size):
            for column in range(array_size):
                center = phase_center + pitch * np.array(
                    [
                        column - (array_size - 1) / 2,
                        row - (array_size - 1) / 2,
                        0,
                    ]
                )
                rotator = phase_rotator(center, radius)

                def update_phase(mobject, *, row=row, column=column, center=center):
                    u, v = steering_at()
                    phase = -PI * (column * u + row * v)
                    end = center + radius * 0.82 * np.array(
                        [np.cos(phase), np.sin(phase), 0]
                    )
                    mobject[0].set_points_as_corners([center, end])
                    mobject[1].move_to(end)

                rotator[2].add_updater(update_phase)
                phases.add(rotator)

        surface_resolution = (129, 257)
        view_rotation = rotation_matrix(-68 * DEGREES, RIGHT) @ rotation_matrix(
            -45 * DEGREES, OUT
        )
        pattern_scale = fw(self, 0.43)
        pattern_origin = UP * fh(self, 0.225)
        pattern, update_pattern = array_factor_surface(
            array_size,
            af_floor_db,
            surface_resolution,
            view_rotation,
            pattern_scale,
            pattern_origin,
        )

        phases.fix_in_frame()
        pattern.add_updater(lambda mobject: update_pattern(mobject, *steering_at()))
        self.add(pattern, phases)

        self.wait(hold_seconds)

        self.play(u_0 @ scan_radius, run_time=sweep_seconds)

        self.wait(hold_seconds)

        self.play(u_0 @ -scan_radius, run_time=sweep_seconds * 2)

        self.wait(hold_seconds)

        self.play(u_0 @ 0, run_time=sweep_seconds)

        self.play(v_0 @ scan_radius, run_time=sweep_seconds)

        self.wait(hold_seconds)

        self.play(v_0 @ -scan_radius, run_time=sweep_seconds * 2)

        self.wait(hold_seconds)

        self.play(v_0 @ 0, run_time=sweep_seconds)

        self.play(u_0 @ scan_radius, run_time=sweep_seconds)

        circular_scan = True
        self.play(scan_progress @ 1, run_time=circular_scan_seconds, rate_func=linear)

        circular_scan = False

        self.play(u_0 @ 0, v_0 @ 0, run_time=sweep_seconds)

        self.wait(hold_seconds)

        self.embed()


class Steering(ThreeDScene):
    default_camera_config = {
        "background_color": BACKGROUND_COLOR,
        "resolution": (1080, 1920),
        "fps": 60,
    }

    def construct(self):
        self.frame.reorient(0, 0)
        self.frame.set_width(4.5).set_height(8, stretch=True)
        self.camera.refresh_uniforms = frame_projection(self.camera)

        narration_path = (
            Path(__file__).resolve().parents[3]
            / "audio/06_phased_array_steering/phased_array_steering_aligned.wav"
        )
        with wave.open(str(narration_path)) as narration:
            narration_seconds = narration.getnframes() / narration.getframerate()
        narration_end = np.ceil(narration_seconds * self.camera.fps) / self.camera.fps
        mechanical_scan_start = 1.2
        array_reveal_start = 3.35
        phase_reveal_start = 6.45
        phase_reveal_end = 7.7
        steering_start = 494 / 60
        steering_end = 619 / 60
        circular_scan_start = 10.65
        thumbnail_start = 16.25
        thumbnail_seconds = 0.6
        end_pause_seconds = 1.8

        antenna_origin = DOWN * fh(self, 0.1)
        antenna_radius = fw(self, 0.22)
        array_size = 8
        af_floor_db = -40
        surface_resolution = (129, 257)
        view_rotation = rotation_matrix(-68 * DEGREES, RIGHT) @ rotation_matrix(
            -45 * DEGREES, OUT
        )
        pattern_scale = fw(self, 0.43)
        pattern_origin = antenna_origin.copy()
        array_pattern_origin = UP * fh(self, 0.225)
        scan_angle = 35 * DEGREES
        antenna_angle = VT(0)
        steering_radius = VT(0)
        scan_progress = VT(0)
        scan_radius = np.sin(scan_angle)
        circular_scan_seconds = 8

        def steering_at():
            angle = TAU * ~scan_progress
            return (
                ~steering_radius * np.cos(angle),
                ~steering_radius * np.sin(angle),
            )

        antenna = Arc(
            start_angle=4 * PI / 3,
            angle=PI / 3,
            radius=antenna_radius,
            arc_center=antenna_origin + UP * antenna_radius * 0.5,
            stroke_color=WHITE,
            stroke_width=7,
        )

        base_l = Line(
            antenna.get_bottom(),
            antenna.get_bottom() + LEFT * 0.3 + DOWN * 1.3,
            stroke_color=WHITE,
            stroke_width=5,
        )
        base_r = Line(
            antenna.get_bottom(),
            antenna.get_bottom() + RIGHT * 0.3 + DOWN * 1.3,
            stroke_color=WHITE,
            stroke_width=5,
        )

        base_top = Group(base_l, base_r).get_top()

        af, update_af = array_factor_surface(
            array_size,
            af_floor_db,
            surface_resolution,
            view_rotation,
            pattern_scale,
            pattern_origin,
        )
        mechanical_antenna = Group(antenna, af)
        pedestal = VGroup(antenna, base_l, base_r)

        radius_ratio = 0.36
        pitch = fw(self, 0.94) / (array_size - 1 + 2.08 * radius_ratio)
        radius = pitch * radius_ratio
        array_center = DOWN * fh(self, 0.08)
        elements = VGroup()
        for row in range(array_size):
            for column in range(array_size):
                center = array_center + pitch * np.array(
                    [
                        column - (array_size - 1) / 2,
                        row - (array_size - 1) / 2,
                        0,
                    ]
                )
                element = Circle(
                    radius=radius,
                    stroke_color=ORANGE,
                    stroke_width=2,
                    fill_color=ORANGE,
                    fill_opacity=1,
                )
                element.move_to(center)
                elements.add(element)
        elements.fix_in_frame()

        phases = VGroup(
            *[phase_rotator(element.get_center(), radius) for element in elements]
        )
        phases.fix_in_frame()
        rng = np.random.default_rng(0)
        phase_order = rng.permutation(len(elements))

        self.add_sound(str(narration_path))

        self.play(
            LaggedStart(
                ShowCreation(antenna),
                LaggedStart(
                    ShowCreation(base_l),
                    ShowCreation(base_r),
                    lag_ratio=0,
                ),
                ShowCreation(af),
                lag_ratio=0.2,
            ),
            run_time=0.75,
        )

        self.wait(mechanical_scan_start - self.time)

        mechanical_antenna.save_state()
        mechanical_antenna.add_updater(
            lambda mobject: mobject.restore().rotate(
                ~antenna_angle, about_point=base_top
            )
        )
        self.add(mechanical_antenna)

        for target_angle, scan_seconds in (
            (-scan_angle, 0.6),
            (scan_angle, 0.95),
            (0, 0.6),
        ):
            self.play(antenna_angle @ target_angle, run_time=scan_seconds)

        mechanical_antenna.clear_updaters()
        self.remove(mechanical_antenna, base_l, base_r)
        self.add(af, pedestal)

        self.play(
            LaggedStart(
                AnimationGroup(
                    af.animate.shift(array_pattern_origin - pattern_origin),
                    FadeOut(pedestal, shift=DOWN * fh(self, 0.6)),
                    run_time=0.7,
                ),
                LaggedStart(
                    *[
                        GrowFromCenter(element, rate_func=ease_out_back)
                        for element in elements
                    ],
                    lag_ratio=0.025,
                    run_time=0.65,
                ),
                lag_ratio=0.6,
            ),
            run_time=array_reveal_start + 1 - self.time,
        )
        pattern_origin[:] = array_pattern_origin

        self.wait(phase_reveal_start - self.time)

        self.play(
            LaggedStart(
                *[
                    Succession(
                        ReplacementTransform(
                            elements[index],
                            phases[index],
                            run_time=0.15,
                        ),
                        ApplyFunction(
                            lambda mobject, *, center=elements[index].get_center(): (
                                mobject.scale(1.2, about_point=center)
                            ),
                            phases[index],
                            rate_func=there_and_back,
                            run_time=0.2,
                        ),
                    )
                    for index in phase_order
                ],
                lag_ratio=0.04,
                run_time=phase_reveal_end - phase_reveal_start,
            )
        )
        self.remove(elements)
        self.add(phases)

        self.wait(steering_start - self.time)

        for index, rotator in enumerate(phases):
            row, column = divmod(index, array_size)
            center = rotator[0].get_center()

            def update_phase(mobject, *, row=row, column=column, center=center):
                u, v = steering_at()
                phase = -PI * (column * u + row * v)
                end = center + radius * 0.82 * np.array(
                    [np.cos(phase), np.sin(phase), 0]
                )
                mobject[0].set_points_as_corners([center, end])
                mobject[1].move_to(end)

            rotator[2].add_updater(update_phase)
        af.add_updater(lambda mobject: update_af(mobject, *steering_at()))

        self.play(steering_radius @ scan_radius, run_time=steering_end - self.time)

        self.wait(circular_scan_start - self.time)

        scan_seconds = thumbnail_start - self.time
        self.play(
            scan_progress @ (scan_seconds / circular_scan_seconds),
            run_time=scan_seconds,
            rate_func=linear,
        )

        thumbnail_img = ImageMobject(
            str(
                Path(__file__).resolve().parents[2]
                / "05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
        ).set_width(fw(self, 0.9))
        thumbnail_border = SurroundingRectangle(
            thumbnail_img,
            buff=0,
            stroke_color=WHITE,
            stroke_width=4,
        )
        thumbnail = Group(thumbnail_img, thumbnail_border).move_to(self.frame)
        thumbnail.set_z_index(1).fix_in_frame()
        thumbnail_scan = scan_progress @ (
            ~scan_progress + thumbnail_seconds / circular_scan_seconds
        )
        thumbnail_scan.set_anim_args(rate_func=linear)

        self.play(
            GrowFromCenter(thumbnail),
            thumbnail_scan,
            run_time=thumbnail_seconds,
        )

        scan_seconds = narration_end - self.time
        self.play(
            scan_progress @ (~scan_progress + scan_seconds / circular_scan_seconds),
            run_time=scan_seconds,
            rate_func=linear,
        )

        self.play(
            scan_progress @ (~scan_progress + end_pause_seconds / circular_scan_seconds),
            run_time=end_pause_seconds,
            rate_func=linear,
        )

        self.embed()
