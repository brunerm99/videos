# beamforming.py

from copy import deepcopy
import warnings
import sys
from manim import *
from MF_Tools import VT, DN
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from scipy.io import wavfile

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_transform_func(from_var, func=TransformFromCopy):
    def transform_func(m, **kwargs):
        return func(from_var, m, **kwargs)

    return transform_func


def square_wave_fourier(x, N, f=1, phi=0):
    sum_terms = np.zeros_like(x)
    for n in range(1, N + 1, 2):
        sum_terms += np.sin(f * n * x + phi) / n
    return (4 / np.pi) * sum_terms


class Intro(Scene):
    def construct(self): ...


class Discontinuity(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.7,
            y_length=config.frame_height * 0.35,
        )

        ripple_offset = VT(-2)
        plot = always_redraw(
            lambda: ax.plot(
                lambda t: 1
                + 0.5
                * np.sin(2 * PI * 10 * t - ~ripple_offset)
                * np.exp(-10 * np.abs(t - ~ripple_offset)),
                x_range=[-1, 1, 1 / 1000],
                color=TX_COLOR,
                use_smoothing=False,
            )
        )

        num_samples = 8
        samples = ax.get_vertical_lines_to_graph(
            plot,
            x_range=[-1 + 0.0625 * 2, 1 - 0.0625 * 2],
            num_lines=num_samples,
            color=RED,
            line_func=Line,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.8,
        )
        sample_dots = Group(*[Dot(sample.get_end(), color=RED) for sample in samples])

        self.play(Create(ax), Create(plot))

        self.play(
            LaggedStart(*[Create(sample) for sample in samples], lag_ratio=0.2),
            LaggedStart(*[Create(dot) for dot in sample_dots], lag_ratio=0.2),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(ripple_offset @ 2, run_time=5)

        self.wait(0.5)

        f_ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[0, 1, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=config.frame_width * 0.7,
            y_length=config.frame_height * 0.35,
        ).next_to([0, -config.frame_height / 2, 0], DOWN)

        samples_y = samples.get_y() - ax.get_y()
        samples.add_updater(
            lambda m: m.shift([0, (m.get_y() - ax.get_y()) - samples_y, 0])
        )

        new_ax_group = Group(ax.copy(), f_ax.copy()).arrange(DOWN, MED_SMALL_BUFF)
        self.play(
            ax.animate.move_to(new_ax_group[0]),
            f_ax.animate.move_to(new_ax_group[1]),
            Group(samples, sample_dots).animate.shift(
                UP * (new_ax_group[0].get_y() - ax.get_y())
            ),
        )

        self.wait(0.5)

        f_labels = f_ax.get_x_axis_label(MathTex("f"))
        ax_labels = ax.get_x_axis_label(MathTex("t"))

        self.play(LaggedStart(Create(ax_labels), Create(f_labels), lag_ratio=0.3))

        self.wait(0.5)

        spike = Line(
            f_ax.c2p(0, 0),
            f_ax.c2p(0, 1),
            color=ORANGE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.8,
        )
        spike_dot = Dot(spike.get_end(), color=ORANGE)

        self.play(LaggedStart(Create(spike), Create(spike_dot), lag_ratio=0.4))

        self.wait(0.5)

        sinc_x0 = VT(0)
        sinc_x1 = VT(0)
        sinc = always_redraw(
            lambda: f_ax.plot(
                lambda t: np.sinc(2 * PI * t),
                x_range=[~sinc_x0, ~sinc_x1, 1 / 100],
                color=ORANGE,
            )
        )
        self.add(sinc)

        self.play(
            LaggedStart(
                FadeOut(spike, spike_dot),
                AnimationGroup(
                    sinc_x0 @ (-PI),
                    sinc_x1 @ (PI),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        window_x0 = VT(-1)
        window_x1 = VT(1)
        plot_windowed = always_redraw(
            lambda: ax.plot(
                lambda t: 1 if np.abs(t) <= 1 else 0,
                x_range=[~window_x0, ~window_x1, 1 / 1000],
                color=BLUE,
                use_smoothing=False,
            )
        ).set_z_index(-1)
        self.add(plot_windowed)
        self.remove(plot)

        sampled_region = Polygon(
            ax.c2p(-1, 0),
            ax.c2p(-1, 1),
            ax.c2p(1, 1),
            ax.c2p(1, 0),
            fill_color=GREEN,
            fill_opacity=0.3,
            stroke_opacity=0,
        ).set_z_index(-2)

        self.play(FadeIn(sampled_region))

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(config.frame_width * 1.5),
            window_x0 @ (-3),
            window_x1 @ (3),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                config.frame_width / 2
            ).move_to(ax.c2p(-1, 0.5))
        )

        self.wait(0.5)

        zero_to = MathTex("0").next_to(ax.c2p(-1.2, 0), UP)
        to_one = MathTex("1").next_to(ax.c2p(-1, 1), UP)
        to_zero = MathTex("0").next_to(ax.c2p(1.2, 0), UP)
        one_to = MathTex("1").next_to(ax.c2p(1, 1), UP)
        zero_to_one = Arrow(zero_to, to_one)
        one_to_zero = Arrow(one_to, to_zero)

        self.play(
            LaggedStart(
                GrowFromCenter(zero_to),
                GrowArrow(zero_to_one),
                GrowFromCenter(to_one),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(ax.c2p(1, 0.5)))

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(one_to),
                GrowArrow(one_to_zero),
                GrowFromCenter(to_zero),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(config.frame_width * 1.6)
            .move_to(ax)
            .shift(UP * 4.5),
            FadeOut(
                one_to,
                to_one,
                to_zero,
                zero_to,
                one_to_zero,
                zero_to_one,
                sampled_region,
            ),
        )

        self.wait(0.5)

        comp_shift_lr = ax.c2p(-1, 0) - ax.c2p(0, 0)
        comp_shift_ud = ax.c2p(0, 1) - ax.c2p(0, 0)
        t_inp = np.linspace(-1, 3, 1000)
        Ns = [1, 5, 9, 13, 17]

        square_comps = Group()
        for N in Ns:
            square_comp = square_wave_fourier(t_inp, N, f=PI / 2)
            f_square_comp = interp1d(t_inp, square_comp)
            square_comp_plot = ax.plot(
                f_square_comp,
                x_range=[t_inp.min(), t_inp.max(), 1 / 1000],
                color=GREEN,
            ).shift(comp_shift_lr)
            square_comps.add(square_comp_plot)

        square_comps_copy = (
            deepcopy(square_comps).arrange(UP).next_to(plot_windowed, UP)
        )

        self.play(*[Create(sc) for sc in square_comps])

        self.wait(0.5)

        self.play(
            square_comps.animate.arrange(UP, buff=-LARGE_BUFF * 1.7).next_to(
                self.camera.frame.get_top(), DOWN, MED_LARGE_BUFF
            )
        )
        # for idx, sc in enumerate(square_comps):
        #     self.play(Create(sc))
        #     self.play(sc.animate.shift(UP * (idx + 1) * comp_shift_ud))

        self.wait(0.5)

        # self.play()

        self.wait(2)


class SquareApprox(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-PI / 2, 3 * PI / 2, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.7,
            y_length=config.frame_height * 0.5,
        )

        x = np.linspace(-PI / 2, 3 * PI / 2, 10_000)

        sq = ax.plot_line_graph(
            x,
            (signal.square(x, 0.5) + 1) / 2,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        sinc = ax.plot_line_graph(
            x,
            square_wave_fourier(x, 1),
            add_vertex_dots=False,
            line_color=ORANGE,
        )
        self.add(ax, sq, sinc)

        N_disp = MathTex("N = 1").to_corner(UL, LARGE_BUFF)
        N_num_disp = N_disp[0][-1]
        N_disp_old = N_num_disp

        self.play(Write(N_disp))

        self.wait(0.5)

        audio_rate = 44100
        sig_f = 100

        data = np.array([])
        total_time = 0
        amplitude = np.iinfo(np.int16).max / 50

        N_max = 101
        for N in range(3, N_max + 1, 2):
            N_disp_old = N_num_disp
            N_num_disp = MathTex(f"{N}").move_to(N_disp_old, LEFT)
            anim_time = max(np.exp(-(4.5 * (N - 3) / 101)), 0.02)
            anim_time = 1

            self.play(
                FadeOut(N_disp_old, shift=UP),
                FadeIn(N_num_disp, shift=UP),
                Transform(
                    sinc,
                    ax.plot_line_graph(
                        x,
                        square_wave_fourier(x, N),
                        add_vertex_dots=False,
                        line_color=ORANGE,
                    ),
                ),
                run_time=anim_time,
            )
            print(N)
            self.wait(anim_time)

            total_time += anim_time * 2

            # audio
            new_x = np.arange(0, 2, 1 / audio_rate)
            new_data = amplitude * square_wave_fourier(new_x, N, f=PI * sig_f)

            data = np.concatenate([data, new_data])

        fname = f"audio/data_Nmax_{N_max}.wav"
        print(f"Writing audio to: {fname}")
        wavfile.write(fname, audio_rate, data.astype(np.int16))

        self.wait(2)


class SincTest(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
        )

        t = np.linspace(0.001, 1, 1000)

        def plot_func():
            sinc = np.sin(2 * PI * 10 * t) / ((2 * PI * 10 * t) ** ~exp)
            sinc /= sinc.max()
            f = interp1d(t, sinc)
            return ax.plot(f, x_range=[0.001, 1, 0.001])

        exp = VT(0)
        plot = always_redraw(plot_func)
        self.add(ax, plot)

        self.wait(0.5)

        self.play(exp @ 1, run_time=4)

        self.wait(2)
