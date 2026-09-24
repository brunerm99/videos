import moderngl
import numpy as np
import sys
from pathlib import Path

from manim import *
from matplotlib.colors import LinearSegmentedColormap
from MF_Tools import VT
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from props.style import BACKGROUND_COLOR

RFTK_BACKGROUND = "#1e2c31"
RFTK_MUTED = "#8fa3ab"
RFTK_WEIGHT_COLORS = ("#1e2c31", "#2d8fbf", "#f07828")
RFTK_AF_COLORS = (
    "#1e2c31",
    "#2d8fbf",
    "#3dbcac",
    "#f09050",
    "#f07828",
    "#fc6255",
)

ARRAY_SIZE = 16
INITIAL_ACTIVE_ROW = 1
INITIAL_ACTIVE_COLUMN = 1
RECOVERY_SEED = 20260923
RECOVERY_SECONDS = 14
FULL_ARRAY_SECONDS = 1.2
WEIGHT_RAMP_FRACTION = 0.04
AF_FLOOR_DB = -40
AF_SPIN_DEGREES = 360

ELEMENT_IMAGE_SIZE = 768
AF_IMAGE_SIZE = 768
AF_GRID_SIZE = 193

config.background_color = BACKGROUND_COLOR
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_width = 4.5
config.frame_height = 8


def _rgb(hex_color):
    color = hex_color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _smoothstep(values):
    values = np.clip(values, 0, 1)
    return values * values * (3 - 2 * values)


def _unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-12)


def _look_at(eye, target):
    forward = _unit(target - eye)
    right = _unit(np.cross(forward, np.array([0, 0, 1], dtype=float)))
    up = np.cross(right, forward)
    return np.array(
        [
            [*right, -np.dot(right, eye)],
            [*up, -np.dot(up, eye)],
            [*-forward, np.dot(forward, eye)],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _perspective(field_of_view, aspect, near=0.05, far=100):
    focal_length = 1 / np.tan(np.deg2rad(field_of_view) / 2)
    return np.array(
        [
            [focal_length / aspect, 0, 0, 0],
            [0, focal_length, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def _interpolate_color(weight):
    low, middle, high = map(_rgb, RFTK_WEIGHT_COLORS)
    if weight <= 0.5:
        color = low + (middle - low) * (weight * 2)
    else:
        color = middle + (high - middle) * ((weight - 0.5) * 2)
    return tuple(np.rint(color).astype(np.uint8))


class RecoveryRenderer:
    def __init__(self):
        rng = np.random.default_rng(RECOVERY_SEED)
        initial_index = INITIAL_ACTIVE_ROW * ARRAY_SIZE + INITIAL_ACTIVE_COLUMN
        recovery_order = np.delete(np.arange(ARRAY_SIZE**2), initial_index)
        rng.shuffle(recovery_order)

        self.initial_index = initial_index
        self.recovery_order = recovery_order
        self.activation_progress = np.empty(ARRAY_SIZE**2)
        self.activation_progress[initial_index] = -WEIGHT_RAMP_FRACTION
        self.activation_progress[recovery_order] = np.linspace(
            0,
            1 - WEIGHT_RAMP_FRACTION,
            recovery_order.size,
        )

        self.af_colormap = LinearSegmentedColormap.from_list(
            "rftk_paview",
            RFTK_AF_COLORS,
        )
        self.frequency_axis = np.fft.fftshift(
            2 * np.fft.fftfreq(AF_GRID_SIZE)
        )
        self.u_grid, self.v_grid = np.meshgrid(
            self.frequency_axis,
            self.frequency_axis,
        )
        self.visible_disk = self.u_grid**2 + self.v_grid**2 <= 1
        self._initialize_af_renderer()

        self.last_progress = None
        self.last_azimuth = None
        self.last_weights = None
        self.last_element_image = None
        self.last_af_image = None

    def _initialize_af_renderer(self):
        rows, columns = np.meshgrid(
            np.arange(AF_GRID_SIZE - 1),
            np.arange(AF_GRID_SIZE - 1),
            indexing="ij",
        )
        top_left = rows * AF_GRID_SIZE + columns
        top_right = top_left + 1
        bottom_left = top_left + AF_GRID_SIZE
        bottom_right = bottom_left + 1

        triangle_a = (
            self.visible_disk[:-1, :-1]
            & self.visible_disk[:-1, 1:]
            & self.visible_disk[1:, :-1]
        )
        triangle_b = (
            self.visible_disk[:-1, 1:]
            & self.visible_disk[1:, 1:]
            & self.visible_disk[1:, :-1]
        )
        indices = np.concatenate(
            [
                np.stack(
                    [
                        top_left[triangle_a],
                        top_right[triangle_a],
                        bottom_left[triangle_a],
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        top_right[triangle_b],
                        bottom_right[triangle_b],
                        bottom_left[triangle_b],
                    ],
                    axis=1,
                ),
            ]
        ).astype(np.uint32)

        self.gl_context = moderngl.create_standalone_context(backend="egl")
        self.gl_program = self.gl_context.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec3 in_color;
                out vec3 surface_position;
                out vec3 surface_normal;
                out vec3 surface_color;
                void main() {
                    surface_position = in_position;
                    surface_normal = in_normal;
                    surface_color = in_color;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 eye;
                in vec3 surface_position;
                in vec3 surface_normal;
                in vec3 surface_color;
                out vec4 fragment_color;
                void main() {
                    vec3 normal = normalize(surface_normal);
                    if (!gl_FrontFacing) normal = -normal;
                    vec3 light = normalize(vec3(-0.45, -0.25, 1.0));
                    vec3 view = normalize(eye - surface_position);
                    float diffuse = max(dot(normal, light), 0.0);
                    float rim = pow(1.0 - max(dot(normal, view), 0.0), 3.0);
                    vec3 color = surface_color * (0.62 + 0.36 * diffuse)
                        + vec3(0.045) * rim;
                    fragment_color = vec4(color, 1.0);
                }
            """,
        )
        self.gl_vertices = self.gl_context.buffer(
            reserve=AF_GRID_SIZE**2 * 9 * np.dtype(np.float32).itemsize,
            dynamic=True,
        )
        self.gl_indices = self.gl_context.buffer(indices.tobytes())
        self.gl_vertex_array = self.gl_context.vertex_array(
            self.gl_program,
            [
                (
                    self.gl_vertices,
                    "3f 3f 3f",
                    "in_position",
                    "in_normal",
                    "in_color",
                )
            ],
            self.gl_indices,
            index_element_size=4,
        )

        samples = min(4, self.gl_context.max_samples)
        color = self.gl_context.renderbuffer(
            (AF_IMAGE_SIZE, AF_IMAGE_SIZE),
            components=4,
            samples=samples,
        )
        depth = self.gl_context.depth_renderbuffer(
            (AF_IMAGE_SIZE, AF_IMAGE_SIZE),
            samples=samples,
        )
        self.gl_msaa_framebuffer = self.gl_context.framebuffer(
            color_attachments=[color],
            depth_attachment=depth,
        )
        self.gl_image_texture = self.gl_context.texture(
            (AF_IMAGE_SIZE, AF_IMAGE_SIZE),
            components=4,
        )
        self.gl_image_framebuffer = self.gl_context.framebuffer(
            color_attachments=[self.gl_image_texture]
        )
        self.gl_context.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.gl_context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

    def weights_at(self, progress):
        scaled = (progress - self.activation_progress) / WEIGHT_RAMP_FRACTION
        weights = _smoothstep(scaled)
        weights[self.initial_index] = 1
        if progress >= 1:
            weights.fill(1)
        return weights.reshape((ARRAY_SIZE, ARRAY_SIZE))

    def frames_at(self, progress, azimuth):
        progress = float(np.clip(progress, 0, 1))
        azimuth = float(azimuth)
        if (
            self.last_progress is not None
            and abs(progress - self.last_progress) < 1e-9
            and abs(azimuth - self.last_azimuth) < 1e-9
        ):
            return self.last_element_image, self.last_af_image

        weights = self.weights_at(progress)
        self.last_progress = progress
        self.last_azimuth = azimuth
        self.last_weights = weights
        self.last_element_image = self._render_elements(weights)
        self.last_af_image = self._render_array_factor(weights, azimuth)
        return self.last_element_image, self.last_af_image

    @staticmethod
    def _render_elements(weights):
        supersample = 2
        image_size = ELEMENT_IMAGE_SIZE * supersample
        margin = 23 * supersample
        pitch = (image_size - 2 * margin) / (ARRAY_SIZE - 1)
        radius = pitch * 0.34
        outline_width = max(2, round(pitch * 0.11))

        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        for row in range(ARRAY_SIZE):
            for column in range(ARRAY_SIZE):
                weight = weights[row, column]
                x = margin + column * pitch
                y = image_size - margin - row * pitch
                bounds = (x - radius, y - radius, x + radius, y + radius)

                if weight <= 1e-6:
                    draw.ellipse(
                        bounds,
                        outline=RFTK_MUTED,
                        width=outline_width,
                    )
                else:
                    draw.ellipse(bounds, fill=_interpolate_color(weight))

        image = image.resize(
            (ELEMENT_IMAGE_SIZE, ELEMENT_IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
        return np.asarray(image, dtype=np.uint8)

    def _render_array_factor(self, weights, azimuth):
        field = np.fft.fftshift(
            np.fft.fft2(weights, s=(AF_GRID_SIZE, AF_GRID_SIZE))
        )
        amplitude = np.abs(field) / np.sum(weights)
        amplitude = np.minimum(amplitude, 1)
        af_db = np.maximum(
            20 * np.log10(np.maximum(amplitude, 10 ** (AF_FLOOR_DB / 20))),
            AF_FLOOR_DB,
        )

        radial_scale = np.clip(
            (af_db - AF_FLOOR_DB) / -AF_FLOOR_DB,
            0,
            1,
        )
        radial_scale = np.where(self.visible_disk, radial_scale, np.nan)
        x = np.nan_to_num(self.u_grid * radial_scale)
        y = np.nan_to_num(self.v_grid * radial_scale)
        z = (
            np.sqrt(
                np.maximum(0, 1 - self.u_grid**2 - self.v_grid**2)
            )
            * radial_scale
        )
        z = np.nan_to_num(z)

        positions = np.stack([x, y, z], axis=-1)
        tangent_u = np.gradient(positions, axis=1)
        tangent_v = np.gradient(positions, axis=0)
        normals = np.cross(tangent_u, tangent_v)
        normal_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= np.maximum(normal_lengths, 1e-12)
        colors = self.af_colormap(
            np.clip((af_db - AF_FLOOR_DB) / -AF_FLOOR_DB, 0, 1)
        )[..., :3]
        vertices = np.concatenate([positions, normals, colors], axis=-1).astype(
            np.float32
        )
        self.gl_vertices.write(vertices.tobytes())

        camera_azimuth = np.deg2rad(-45 + azimuth)
        camera_elevation = np.deg2rad(22)
        camera_distance = 3.15
        camera_target = np.array([0, 0, 0.18])
        eye = camera_target + camera_distance * np.array(
            [
                np.cos(camera_elevation) * np.cos(camera_azimuth),
                np.cos(camera_elevation) * np.sin(camera_azimuth),
                np.sin(camera_elevation),
            ]
        )
        matrix = _perspective(38, 1) @ _look_at(eye, camera_target)
        self.gl_program["mvp"].write(matrix.T.astype(np.float32).tobytes())
        self.gl_program["eye"].value = tuple(eye.astype(float))

        self.gl_msaa_framebuffer.use()
        self.gl_msaa_framebuffer.clear(0, 0, 0, 0, depth=1)
        self.gl_vertex_array.render(mode=moderngl.TRIANGLES)
        self.gl_context.copy_framebuffer(
            self.gl_image_framebuffer,
            self.gl_msaa_framebuffer,
        )
        image = np.frombuffer(
            self.gl_image_framebuffer.read(components=4, alignment=1),
            dtype=np.uint8,
        ).reshape((AF_IMAGE_SIZE, AF_IMAGE_SIZE, 4))
        return image[::-1].copy()


class ArrayFailureRecovery(Scene):
    def construct(self):
        self.next_section("failed_array", skip_animations=False)

        renderer = RecoveryRenderer()
        cycle = VT(0)
        element_frame, af_frame = renderer.frames_at(0, 0)

        loop_seconds = RECOVERY_SECONDS * 2 + FULL_ARRAY_SECONDS

        def state_at(cycle_progress):
            elapsed = cycle_progress * loop_seconds
            if elapsed < RECOVERY_SECONDS:
                transition = elapsed / RECOVERY_SECONDS
                progress = rate_functions.ease_in_out_sine(transition)
            elif elapsed < RECOVERY_SECONDS + FULL_ARRAY_SECONDS:
                progress = 1
            else:
                transition = (
                    elapsed - RECOVERY_SECONDS - FULL_ARRAY_SECONDS
                ) / RECOVERY_SECONDS
                progress = 1 - rate_functions.ease_in_out_sine(transition)

            azimuth = AF_SPIN_DEGREES * 2 * cycle_progress
            return progress, azimuth

        elements = (
            ImageMobject(element_frame)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            .scale_to_fit_width(config.frame_width * 0.9)
            .move_to(DOWN * config.frame_height * 0.144)
        )
        array_factor = (
            ImageMobject(af_frame)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            .scale_to_fit_width(config.frame_width * 0.9)
            .move_to(UP * config.frame_height * 0.24)
        )

        def update_elements(mobject):
            progress, azimuth = state_at(~cycle)
            mobject.pixel_array = renderer.frames_at(progress, azimuth)[0]

        def update_array_factor(mobject):
            progress, azimuth = state_at(~cycle)
            mobject.pixel_array = renderer.frames_at(progress, azimuth)[1]

        elements.add_updater(update_elements)
        array_factor.add_updater(update_array_factor)
        self.add(elements, array_factor)

        self.wait(0.8)

        self.next_section("loop", skip_animations=False)
        self.play(
            cycle @ 1,
            run_time=loop_seconds,
            rate_func=rate_functions.linear,
        )
