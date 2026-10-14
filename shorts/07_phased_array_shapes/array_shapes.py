import numpy as np
from manimgx import *

BACKGROUND_COLOR = "#183340"
ELEMENT_COLOR = "#f07828"
AF_COLORS = ("#1e2c31", "#2d8fbf", "#3dbcac", "#f09050", "#f07828", "#fc6255")
SKIP_ANIMATIONS_OVERRIDE = True

config.background_color = BACKGROUND_COLOR
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_height = 8
config.frame_rate = 60


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def array_factor_fields(directions, element_positions, aperture_weights):
    phase = TAU * (directions[:, :2] @ element_positions.T)
    return aperture_weights @ np.exp(1j * phase).T


def array_factor_surface(
    element_positions,
    aperture_weights,
    af_floor_db,
    surface_resolution,
    view_rotation,
    pattern_scale,
    pattern_origin,
):
    theta, phi = np.meshgrid(
        np.linspace(0, PI / 2, surface_resolution[0]),
        np.linspace(0, TAU, surface_resolution[1]),
        indexing="ij",
    )
    directions = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    ).reshape(-1, 3)
    indices = np.arange(len(directions)).reshape(surface_resolution)
    a = indices[:-1, :-1].ravel()
    b = indices[1:, :-1].ravel()
    c = indices[1:, 1:].ravel()
    d = indices[:-1, 1:].ravel()
    triangles = np.stack([a, b, c, a, c, d], axis=-1).reshape(-1, 3)

    # Sample each ring's field once; redraws only mix these cached fields.
    fields = array_factor_fields(directions, element_positions, aperture_weights)
    weight_sums = aperture_weights.sum(axis=1)
    color_stops = np.array([ManimColor(color).to_rgb() for color in AF_COLORS])

    def pattern_at(aperture_mix, spin_angle=0):
        field = np.einsum("s,sk->k", aperture_mix, fields)
        amplitude = np.abs(field) / max(aperture_mix @ weight_sums, 1e-12)
        af_db = 20 * np.log10(np.maximum(amplitude, 10 ** (af_floor_db / 20)))
        radial_scale = np.clip((af_db - af_floor_db) / -af_floor_db, 0, 1)
        points = directions * radial_scale[:, None]
        rotation = view_rotation @ rotation_matrix(spin_angle, OUT)
        points = pattern_scale * (points @ rotation.T) + pattern_origin

        scaled_colors = radial_scale * (len(color_stops) - 1)
        indices = np.minimum(scaled_colors.astype(int), len(color_stops) - 2)
        blend = (scaled_colors - indices)[:, None]
        rgb = color_stops[indices] * (1 - blend) + color_stops[indices + 1] * blend
        return MeshMobject(
            points,
            triangles,
            vertex_colors=np.column_stack([rgb, np.ones(len(rgb))]),
            shade_in_3d=True,
        )

    return pattern_at


class ArrayShapes(ThreeDScene):
    def construct(self):
        self.next_section("Square aperture", skip_animations=skip_animations(False))
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        array_size = 17
        spacing_wavelengths = 0.5
        aperture_radius = (array_size - 1) / 2
        axis = np.arange(array_size) - aperture_radius
        x, y = np.meshgrid(axis, axis)
        coordinates = np.column_stack([x.ravel(), y.ravel()])
        element_positions = coordinates * spacing_wavelengths
        hex_angles = np.arange(6) * PI / 3
        hex_normals = np.column_stack(
            [np.cos(hex_angles + PI / 6), np.sin(hex_angles + PI / 6)]
        )
        aperture_weights = np.stack(
            [
                np.ones(len(coordinates)),
                np.linalg.norm(coordinates, axis=1) <= aperture_radius + 1e-9,
                np.all(
                    coordinates @ hex_normals.T
                    <= aperture_radius * np.cos(PI / 6) + 1e-9,
                    axis=1,
                ),
            ]
        ).astype(float)
        ring_indices = np.ceil(np.linalg.norm(coordinates, axis=1) - 1e-9).astype(int)
        ring_trackers = []
        transition_groups = []
        field_weights = [aperture_weights[-1]]
        for source, target in zip(aperture_weights[:-1], aperture_weights[1:]):
            changed = source != target
            group = []
            for ring in np.unique(ring_indices[changed])[::-1]:
                ring_mask = changed & (ring_indices == ring)
                tracker = VT(1)
                ring_trackers.append(tracker)
                group.append(tracker)
                field_weights.append(ring_mask.astype(float))
            transition_groups.append(group)
        field_weights = np.array(field_weights)

        def aperture_at():
            return np.array([1, *[~tracker for tracker in ring_trackers]])

        radius_ratio = 0.36
        pitch = fw(self, 0.94) / (array_size - 1 + 2 * radius_ratio)
        radius = pitch * radius_ratio
        array_center = DOWN * fh(self, 0.085)
        elements = VGroup(
            *[
                Circle(
                    radius=radius,
                    stroke_color=ELEMENT_COLOR,
                    stroke_width=2,
                    fill_color=ELEMENT_COLOR,
                    fill_opacity=1,
                ).move_to(array_center + pitch * np.array([x, y, 0]))
                for x, y in coordinates
            ]
        )

        self.camera.add_fixed_in_frame_mobjects(elements)

        af_floor_db = -40
        surface_resolution = (129, 257)
        view_rotation = rotation_matrix(-68 * DEGREES, RIGHT) @ rotation_matrix(
            -45 * DEGREES, OUT
        )
        pattern_scale = fw(self, 0.43)
        pattern_origin = UP * fh(self, 0.225)
        pattern_at = array_factor_surface(
            element_positions,
            field_weights,
            af_floor_db,
            surface_resolution,
            view_rotation,
            pattern_scale,
            pattern_origin,
        )
        af = pattern_at(aperture_at())

        self.play(
            Create(af),
            LaggedStart(
                *[
                    GrowFromCenter(element, rate_func=ease_out_back)
                    for element in elements
                ],
                lag_ratio=0.005,
            ),
            run_time=1,
        )

        element_templates = elements.copy()

        def elements_at():
            weights = aperture_at() @ field_weights
            current_elements = element_templates.copy()
            for element, weight in zip(current_elements, weights):
                scale = max(weight * (1 + 0.85 * np.sin(PI * weight)), 0.001)
                element.scale(scale)
                element.set_opacity(float(weight))
            return current_elements

        self.remove(*elements, af)
        elements = always_redraw(elements_at)
        self.add_fixed_in_frame_mobjects(elements)
        spin_start = self.time
        spin_seconds = 36
        af = always_redraw(
            lambda: pattern_at(
                aperture_at(),
                TAU * (self.time - spin_start) / spin_seconds,
            )
        )
        self.add(af)

        self.wait(3)

        for name, group in zip(
            ("Circular aperture", "Hexagonal aperture"), transition_groups
        ):
            self.next_section(name, skip_animations=skip_animations(False))
            self.play(
                LaggedStart(*[tracker @ 0 for tracker in group], lag_ratio=0.65),
                run_time=2.2,
            )

            self.wait(3)
