# snr.py

import sys
import warnings
import random

from numpy.fft import fft, fftshift
from manim import *
import numpy as np
from MF_Tools import TransformByGlyphMap, VT
from scipy.interpolate import interp1d
from scipy import signal

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class EqnIntro(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        # snr_eqn = MathTex(r"\frac{P_t G_t }")

        snr_eqn = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4 k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in snr_eqn[0]], lag_ratio=0.07))

        self.wait(0.5)

        fs = 100
        f = 25
        noise_std = VT(0.001)

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.5
        y_max = 30

        ax = Axes(
            x_range=[0, fs / 2, fs / 8],
            y_range=[0, y_max, 10],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        amp = VT(0)

        def fft_updater():
            np.random.seed(1)
            noise = np.random.normal(loc=0, scale=~noise_std, size=N)
            x_n = (~amp * np.sin(2 * PI * f * t) + noise) * signal.windows.blackman(N)
            X_k = fftshift(fft(x_n, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = np.clip(10 * np.log10(X_k) + y_max, 0, None)
            f_X_k_log = interp1d(freq, X_k, fill_value="extrapolate")

            plot = ax.plot(f_X_k_log, x_range=[0, fs / 2, 1 / 100], color=RX_COLOR)
            return plot

        X_k_plot = always_redraw(fft_updater)

        self.add(ax.next_to([0, -config.frame_height / 2, 0], DOWN))
        self.add(X_k_plot)

        self.play(
            ax.animate.to_edge(DOWN, MED_LARGE_BUFF),
            snr_eqn.animate.scale(0.7).to_edge(UP, MED_LARGE_BUFF),
        )

        self.wait(0.5)

        self.play(amp @ 1)

        self.wait(0.5)

        self.play(noise_std @ 0.2)

        self.wait(0.5)

        snr_line = always_redraw(
            lambda: Line(
                ax.c2p(30, 16.16),
                [
                    ax.c2p(30, 16.16)[0],
                    ax.input_to_graph_point(f, fft_updater())[1],
                    0,
                ],
            )
        )
        snr_line_u = always_redraw(
            lambda: Line(snr_line.get_top() + LEFT / 8, snr_line.get_top() + RIGHT / 8)
        )
        snr_line_d = always_redraw(
            lambda: Line(
                snr_line.get_bottom() + LEFT / 8, snr_line.get_bottom() + RIGHT / 8
            )
        )
        snr_label = always_redraw(lambda: Tex("SNR").next_to(snr_line, RIGHT))
        snr_db_label = always_redraw(
            lambda: MathTex(
                f" = {ax.input_to_graph_coords(f, fft_updater())[1] -16.16:.2f} \\text{{ dB}}"
            ).next_to(snr_label)
        )

        self.play(
            Create(snr_line),
            Create(snr_line_u),
            Create(snr_line_d),
            TransformFromCopy(snr_eqn[0][:3], snr_label[0]),
        )
        self.next_section(skip_animations=skip_animations(False))
        self.add(snr_label)
        self.play(FadeIn(snr_db_label))

        self.wait(0.5, frozen_frame=False)

        self.play(amp @ 0.15, run_time=3)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5, frozen_frame=False)

        self.play(
            ax.animate.next_to([0, -config.frame_height / 2, 0], DOWN),
            snr_eqn.animate.scale(1 / 0.7).move_to(ORIGIN),
        )
        self.remove(
            ax, X_k_plot, snr_line, snr_line_u, snr_line_d, snr_label, snr_db_label
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        snr_eqn_split = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        self.play(
            TransformByGlyphMap(
                snr_eqn,
                snr_eqn_split,
                ([0, 1, 2, 3], [0, 1, 2, 3]),
                ([4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10]),
                ([11], [11]),
                ([12, 13, 14, 15, 16, 17, 18], [12, 13, 14, 15, 16, 17, 18]),
                (GrowFromCenter, [19], {"delay": 0.2}),
                ([12, 13, 14, 15, 16, 17, 18], [12, 13, 14, 15, 16, 17, 18]),
                (GrowFromCenter, [20], {"delay": 0.4}),
                (GrowFromCenter, [21], {"delay": 0.6}),
                ([19, 20, 21, 22, 23, 24], [22, 23, 24, 25, 26, 27]),
            )
        )

        self.wait(0.5)

        self.play(
            snr_eqn_split[0][0].animate.set_color(BLUE),
            snr_eqn_split[0][4:19].animate.set_color(BLUE),
        )

        self.wait(0.5)

        self.play(
            snr_eqn_split[0][1].animate.set_color(RED),
            snr_eqn_split[0][22:].animate.set_color(RED),
        )

        self.wait(2)


class Signal(Scene):
    def construct(self):
        snr_eqn = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        snr_eqn[0][0].set_color(BLUE)
        snr_eqn[0][4:19].set_color(BLUE)
        snr_eqn[0][1].set_color(RED)
        snr_eqn[0][22:].set_color(RED)

        thumbnail_1_img = ImageMobject(
            "../06_radar_range_equation/media/images/radar_equation/thumbnails/Thumbnail_1.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        thumbnail_1_box = SurroundingRectangle(thumbnail_1_img, buff=0)
        thumbnail_1 = Group(thumbnail_1_img, thumbnail_1_box)
        title_1 = Tex(r"The Radar\\Range Equation")
        thumbnail_1_group = (
            Group(thumbnail_1, title_1)
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to([config.frame_width / 2, 0, 0], RIGHT)
        )

        self.add(snr_eqn)

        self.play(FadeOut(snr_eqn[0][:4], snr_eqn[0][-9:]))

        self.wait(0.5)

        signal_eqn = snr_eqn[0][4:-9]

        self.play(
            Group(signal_eqn, thumbnail_1_group).animate.arrange(
                RIGHT, LARGE_BUFF * 1.5
            )
        )

        self.wait(0.5)

        self.play(
            signal_eqn.animate.move_to(ORIGIN),
            thumbnail_1_group.animate.next_to([config.frame_width / 2, 0, 0], RIGHT),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                signal_eqn[:2]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[2:4]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[4:6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[-2:]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(DOWN / 2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(signal_eqn[:2].animate.set_color(YELLOW))

        x_len = config.frame_width * 0.5
        y_len = config.frame_height * 0.5
        ax = Axes(
            x_range=[0, 1, 1 / 4],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(RIGHT, LARGE_BUFF)

        amp = VT(1)

        plot = always_redraw(
            lambda: ax.plot(lambda t: ~amp * np.sin(2 * PI * 3 * t), color=TX_COLOR)
        )
        self.add(ax)
        self.add(plot)

        self.play(
            Create(ax),
            Create(plot),
            signal_eqn.animate.to_edge(LEFT, LARGE_BUFF),
        )

        self.wait(0.5)

        self.play(amp @ 2)

        self.wait(2)


class GasCollision(MovingCameraScene):
    def construct(self):
        up_shift = VT(0)
        right_shift = VT(0)
        box = always_redraw(
            lambda: Square(side_length=3, color=WHITE).shift(
                UP * ~up_shift + RIGHT * ~right_shift
            )
        )
        self.add(box)

        num_particles = 10
        particles = VGroup()
        velocities = []
        colliding = [False] * num_particles

        max_vel = 1.5

        collision_margin = 0.1

        for _ in range(num_particles):
            particle = Dot(radius=0.1, color=BLUE)
            particle.move_to(
                box.get_center()
                + [
                    np.random.uniform(
                        -box.width / 2 + collision_margin,
                        box.width / 2 - collision_margin,
                    ),
                    np.random.uniform(
                        -box.height / 2 + collision_margin,
                        box.height / 2 - collision_margin,
                    ),
                    0,
                ]
            )
            particles.add(particle)
            velocities.append(np.random.uniform(-max_vel, max_vel, size=2))

        self.add(particles)

        def update_particles(group, dt):
            for i, particle in enumerate(group):
                particle.shift(
                    dt * velocities[i][0] * RIGHT + dt * velocities[i][1] * UP
                )

                if (
                    particle.get_center()[0] >= box.get_right()[0] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[0] <= box.get_left()[0] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] >= box.get_top()[1] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] <= box.get_bottom()[1] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                else:
                    colliding[i] = False

                for j, other_particle in enumerate(group):
                    if (
                        i != j
                        and np.linalg.norm(
                            particle.get_center() - other_particle.get_center()
                        )
                        < collision_margin * 2
                    ):
                        velocities[i], velocities[j] = (velocities[j], velocities[i])

        particles.add_updater(update_particles)

        def get_camera_updater(m):
            def camera_updater(cam, dt):
                velocity = m.get_center() - cam.get_center()

                cam.shift(velocity * dt)

            return camera_updater

        self.camera.frame.save_state()
        self.camera.frame.scale(50)

        noise_eqn = MathTex(r"N = k T B_n").scale_to_fit_width(
            self.camera.frame.width * 0.6
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in noise_eqn[0]], lag_ratio=0.06)
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore(), FadeOut(noise_eqn))

        self.wait(2)

        particle_to_follow = particles[np.argmin(np.sum(np.abs(velocities), axis=1))]
        cam_updater = get_camera_updater(particle_to_follow)

        particle_trace = TracedPath(
            particle_to_follow.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).set_z_index(-1)

        self.camera.frame.add_updater(cam_updater)
        self.add(particle_trace)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.3))

        self.wait(3)

        n_potentials = 16
        for idx in range(n_potentials):
            self.play(
                MathTex(f"v_{{{idx}}}", color=YELLOW)
                .scale(0.8)
                .move_to(particle_to_follow)
                .set_opacity(0)
                .animate(rate_func=rate_functions.there_and_back, run_time=0.8)
                .set_opacity(1)
            )

        self.wait(3)

        self.camera.frame.remove_updater(cam_updater)
        self.play(self.camera.frame.animate.restore())

        traced_paths = Group(
            *[
                TracedPath(
                    particle.get_center,
                    dissipating_time=2,
                    stroke_opacity=[0, 1],
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                ).set_z_index(-1)
                for particle in particles
            ]
        )

        self.add(traced_paths)
        # self.remove(particle_trace)

        for idx in range(n_potentials):
            self.play(
                *[
                    MathTex(f"v_{{{pidx},{idx}}}", color=YELLOW)
                    .scale(0.8)
                    .move_to(particle)
                    .set_opacity(0)
                    .animate(rate_func=rate_functions.there_and_back, run_time=0.8)
                    .set_opacity(1)
                    for pidx, particle in enumerate(particles)
                ]
            )

        particles.remove_updater(update_particles)
        for traced_path in traced_paths:
            traced_path.remove_updater(traced_path.update_path)

        potential_labels = Group(
            *[
                MathTex(f"v_{{{pidx},{n_potentials}}}", color=YELLOW)
                .scale(0.8)
                .move_to(particle)
                .set_opacity(0)
                for pidx, particle in enumerate(particles)
            ]
        )
        self.play(potential_labels.animate(run_time=0.8).set_opacity(1))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.shift(
                LEFT
                * (
                    (self.camera.frame.get_right()[0])
                    - (box.get_right()[0] + LARGE_BUFF * 2)
                )
            )
        )

        total_pot = MathTex(
            r"\text{Total potential} = \  &"
            + r" \\ &+ ".join(
                [
                    f"v_{{{pidx},{n_potentials}}}"
                    for pidx, particle in enumerate(particles)
                ]
            )
        ).next_to(self.camera.frame.get_left(), RIGHT, LARGE_BUFF)

        self.add(total_pot)

        self.wait(10)
