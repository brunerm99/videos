# short.py

import sys
import warnings

import numpy as np
from numpy.fft import fft, fftshift
from manim import *
from scipy.interpolate import interp1d, bisplrep, bisplev
from scipy.constants import c
from scipy import signal
from MF_Tools import VT, TransformByGlyphMap


warnings.filterwarnings("ignore")
sys.path.insert(0, "../../")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR
config.pixel_height = 1920
config.pixel_width = 1080
config.frame_height = 14
config.frame_width = 9

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def sinc_pattern(theta, phi, L, W, lambda_0):
    k = 2 * np.pi / lambda_0
    E_theta = np.sinc(k * L / 2 * np.sin(theta) * np.cos(phi))
    E_phi = np.sinc(k * W / 2 * np.sin(theta) * np.sin(phi))
    return np.abs(E_theta * E_phi)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


class Short(MovingCameraScene):
    def construct(self):
        title = (
            Tex("Phased Arrays", color=ORANGE)
            .scale_to_fit_width(config.frame_width * 0.7)
            .next_to([0, config.frame_height / 2, 0], DOWN, LARGE_BUFF * 2)
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in title[0]], lag_ratio=0.1))

        self.wait(0.5)

        antennas = Group()
        for _ in range(6):
            antenna_port = Line(DOWN / 2, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)

        antennas.arrange(RIGHT, MED_LARGE_BUFF).scale_to_fit_width(
            config.frame_width * 0.8
        ).next_to([0, -config.frame_height / 2, 0], UP, LARGE_BUFF * 2)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        GrowFromCenter(antennas[len(antennas) // 2 - 1 - idx]),
                        GrowFromCenter(antennas[len(antennas) // 2 + idx]),
                    )
                    for idx in range(len(antennas) // 2)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(1) for _ in range(n_elem)]
        # weight_trackers[n_elem // 2] @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        # patch parameters
        f_patch = 10e9
        lambda_patch_0 = c / f_patch

        epsilon_r = 2.2
        h = 1.6e-3

        epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 + 12 * h / lambda_patch_0
        ) ** -0.5
        L = lambda_patch_0 / (2 * np.sqrt(epsilon_eff))
        W = lambda_patch_0 / 2 * np.sqrt(2 / (epsilon_r + 1))
        # /patch parameters

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30
        x_len = config.frame_height * 0.6
        ax = (
            Axes(
                x_range=[r_min, -r_min, r_min / 8],
                y_range=[r_min, -r_min, r_min / 8],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=x_len,
                y_length=x_len,
            )
            .set_opacity(0)
            .rotate(PI / 2)
        )
        ax.shift(antennas.get_top() - ax.c2p(0, 0))

        theta_min = VT(0)
        theta_max = VT(0)
        af_opacity = VT(1)

        ep_exp_scale = VT(0)

        def get_ap():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            EP = sinc_pattern(u, 0, L, W, wavelength_0)
            AP = AF * (EP ** (~ep_exp_scale))
            f_AP = interp1d(
                u * PI,
                1.3 * np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None),
                fill_value="extrapolate",
            )
            plot = ax.plot_polar_graph(
                r_func=f_AP,
                theta_range=[~theta_min, ~theta_max, 2 * PI / 200],
                color=TX_COLOR,
                use_smoothing=False,
                stroke_opacity=~af_opacity,
            )
            return plot

        AF_plot = always_redraw(get_ap)

        self.add(AF_plot)

        self.play(theta_min @ (-PI / 2), theta_max @ (PI / 2), run_time=3)

        self.wait(0.5)

        def get_phase_delay_plot(ant):
            ax = (
                Axes(
                    x_range=[0, 1, 0.25],
                    y_range=[-1, 1, 0.5],
                    tips=False,
                    axis_config={
                        "include_numbers": False,
                    },
                    y_length=LARGE_BUFF,
                    x_length=config.frame_height * 0.2,
                )
                .set_opacity(0)
                .rotate(PI / 2)
                .next_to(ant, DOWN)
                .shift(DOWN * 8)
            )

            phase_shift = VT(0)
            plot = always_redraw(
                lambda: ax.plot(
                    lambda t: np.sin(2 * PI * 2 * t + ~phase_shift),
                    color=ORANGE,
                    x_range=[0, 1, 1 / 200],
                )
            )
            return ax, plot, phase_shift

        phase_plots = [get_phase_delay_plot(ant) for ant in antennas]
        phis = Group()
        for idx, (_, plot, _) in enumerate(phase_plots):
            self.add(plot)
            phi = MathTex(
                f"\\phi_{{{idx}}}", font_size=DEFAULT_FONT_SIZE * 1.6
            ).next_to(plot, DOWN)
            phis.add(phi)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.9).shift(DOWN * 4),
            LaggedStart(
                *[ax.animate.shift(UP * 8) for ax, _, _ in phase_plots], lag_ratio=0.2
            ),
            LaggedStart(*[phi.animate.shift(UP * 8) for phi in phis], lag_ratio=0.2),
        )

        self.wait(0.5)

        phase_shift = PI * 0.6
        self.play(
            LaggedStart(
                *[
                    ps @ (phase_shift * idx)
                    for idx, (_, _, ps) in enumerate(phase_plots)
                ],
                lag_ratio=0.2,
            ),
            steering_angle @ (10),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ps @ 0 for idx, (_, _, ps) in enumerate(phase_plots)],
                lag_ratio=0.2,
            ),
            steering_angle @ (0),
            run_time=4,
        )

        self.wait(0.5)

        self.play(*[FadeOut(plot) for _, plot, _ in phase_plots], FadeOut(phis))

        self.wait(0.5)

        weights = np.array([~w for w in weight_trackers])
        x = np.arange(weights.size) - weights.size // 2
        taper_ax = Axes(
            x_range=[x.min(), x.max(), 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=antennas.width,
            y_length=antennas.height,
        ).next_to(antennas, DOWN, LARGE_BUFF * 1.6)

        def plot_taper():
            weights = np.array([~w for w in weight_trackers])
            x = np.arange(weights.size) - weights.size // 2
            f = interp1d(x, weights)
            plot = taper_ax.plot(f, x_range=[x.min(), x.max(), 0.01], color=ORANGE)
            return plot

        taper_plot = always_redraw(plot_taper)
        weight_trackers_disp = [VT(1) for _ in range(6)]

        w_x_step = (x.max() - x.min()) / 6
        w_x = np.arange(x.min() + w_x_step / 2, x.max() - w_x_step / 2, w_x_step)

        test = always_redraw(
            lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[0], taper_plot))
        )
        w_x_dots = VGroup(
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[0], taper_plot))
            ),
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[1], taper_plot))
            ),
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[2], taper_plot))
            ),
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[3], taper_plot))
            ),
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[4], taper_plot))
            ),
            always_redraw(
                lambda: Dot().move_to(taper_ax.input_to_graph_point(w_x[5], taper_plot))
            ),
        )

        wt_disp = Group(
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[0]):.2f}").next_to(
                    antennas[0], DOWN
                )
            ),
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[1]):.2f}").next_to(
                    antennas[1], DOWN
                )
            ),
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[2]):.2f}").next_to(
                    antennas[2], DOWN
                )
            ),
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[3]):.2f}").next_to(
                    antennas[3], DOWN
                )
            ),
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[4]):.2f}").next_to(
                    antennas[4], DOWN
                )
            ),
            always_redraw(
                lambda: Tex(f"{np.abs(~weight_trackers_disp[5]):.2f}").next_to(
                    antennas[5], DOWN
                )
            ),
        )

        self.play(
            FadeIn(*wt_disp),
            Create(taper_ax),
            Create(taper_plot),
            # *[Create(dot) for dot in w_x_dots],
            # Create(test),
            run_time=2,
        )

        self.wait(0.5)

        taper = signal.windows.kaiser(n_elem, beta=2)
        taper_disp = signal.windows.kaiser(len(antennas), beta=2)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers_disp[:3][::-1][n] @ taper_disp[:3][::-1][n],
                        weight_trackers_disp[3:][n] @ taper_disp[3:][n],
                    )
                    for n in range(3)
                ],
                lag_ratio=0.3,
            ),
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n]
                        @ taper[: n_elem // 2][::-1][n],
                        weight_trackers[n_elem // 2 + 1 :][n]
                        @ taper[n_elem // 2 + 1 :][n],
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        taper = signal.windows.taylor(n_elem, nbar=5, sll=23)
        taper_disp = signal.windows.taylor(len(antennas), nbar=5, sll=23)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers_disp[:3][::-1][n] @ taper_disp[:3][::-1][n],
                        weight_trackers_disp[3:][n] @ taper_disp[3:][n],
                    )
                    for n in range(3)
                ],
                lag_ratio=0.3,
            ),
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n]
                        @ taper[: n_elem // 2][::-1][n],
                        weight_trackers[n_elem // 2 + 1 :][n]
                        @ taper[n_elem // 2 + 1 :][n],
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        thumbnail_img = (
            ImageMobject(
                "../../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
            .scale_to_fit_width(config.frame_width * 0.8)
            .move_to(title)
        )
        thumbnail_box = SurroundingRectangle(thumbnail_img, buff=0)
        thumbnail = Group(thumbnail_box, thumbnail_img)
        self.remove(*title[0])

        self.play(
            self.camera.frame.animate.restore(),
            thumbnail.shift(UP * 5).animate.shift(DOWN * 5),
            FadeOut(taper_ax, taper_plot, *wt_disp),
        )

        self.wait(0.5)

        profile_pic = (
            ImageMobject(
                "../../../../media/rf_channel_assets/profile_pictures/Raccoon_Coding_Retro_Channel_Colors.jpg"
            )
            .scale_to_fit_width(config.frame_width * 0.4)
            .next_to(thumbnail, UP, LARGE_BUFF)
        )
        mb = Tex("Marshall Bruner", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            profile_pic, DOWN
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    self.camera.frame.animate.shift(UP * 10),
                    thumbnail.animate.shift(UP * 3),
                ),
                GrowFromCenter(profile_pic.shift(UP * 5)),
                Write(mb.shift(UP * 5)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)
