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
from scipy.constants import c
from numpy.fft import fft, fftshift

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props import VideoMobject, get_blocks, get_resistor
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR
# config.frame_width = config.frame_height
# # config.aspect_ratio = 1
# config.pixel_height = 1080
# config.pixel_width = 1080

SKIP_ANIMATIONS_OVERRIDE = True

BLOCKS = get_blocks()


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


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


def lin2db(x):
    return 10 * np.log10(x)


def db2lin(x):
    return 10 ** (x / 10)


class Intro(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.4,
            y_length=config.frame_height * 0.35,
        ).next_to([-config.frame_width / 2, 0, 0], LEFT)
        time_label = Tex("Time").next_to(ax, UP)
        time_group = Group(ax, time_label)
        window_pow = VT(0)

        def get_f_window(mult=1):
            window = (signal.windows.hann(1000) ** ~window_pow) * mult
            return interp1d(np.linspace(0, 1, 1000), window)

        sine_plot = always_redraw(
            lambda: ax.plot(
                lambda t: (
                    (
                        np.sin(2 * PI * 3 * t)
                        + 0.3 * np.sin(2 * PI * 8 * t)
                        + 0.5 * np.sin(2 * PI * 12 * t)
                        # + np.random.normal(0, 0.1)
                    )
                    / 1.9
                )
                * get_f_window()(t),
                x_range=[0, 1, 1 / 1000],
                color=BLUE,
                use_smoothing=False,
            )
        )
        f_ax = Axes(
            x_range=[-15, 15, 5],
            y_range=[-0.5, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.4,
            y_length=config.frame_height * 0.35,
        ).next_to([config.frame_width / 2, 0, 0], RIGHT)
        f_label = Tex("Frequency").next_to(f_ax, UP)
        f_group = Group(f_ax, f_label)
        self.add(sine_plot)

        fs = 200
        max_time = 2
        N = max_time * fs
        fft_len = N * 4
        t = np.linspace(0, max_time, N)
        freq = np.linspace(-fs / 2, fs / 2, fft_len)

        def plot_fft():
            window = signal.windows.hann(N) ** ~window_pow
            sig = (
                (
                    np.sin(2 * PI * 3 * t)
                    + 0.3 * np.sin(2 * PI * 8 * t)
                    + 0.5 * np.sin(2 * PI * 12 * t)
                    # + np.random.normal(0, 0.5, N)
                )
                / 1.9
            ) * window
            X_k = 10 * np.log10(np.abs(fft(sig, fft_len)))
            X_k /= X_k.max()
            # print(X_k)
            f_X_k = interp1d(freq, np.clip(fftshift(X_k), -0.5, None))
            plot = f_ax.plot(f_X_k, x_range=[-15, 15, 1 / 200], color=RED)
            return plot

        fft_plot = always_redraw(plot_fft)
        self.add(fft_plot)

        plot_group = Group(time_group, f_group)
        self.play(plot_group.animate.arrange(RIGHT, LARGE_BUFF))

        self.wait(0.5)

        arrow = Arrow(time_label.get_right(), f_label.get_left())
        self.play(GrowArrow(arrow))

        self.wait(0.5)

        sigproc = Tex("Signal Processing")
        phased_array = Tex("Phased Array")
        sound = Tex("Sound Engineering")
        etc = Tex("etc.")
        Group(sigproc, phased_array, sound, etc).arrange(RIGHT, MED_LARGE_BUFF).shift(
            DOWN * 5
        )
        sigproc_bez = CubicBezier(
            plot_group.get_bottom() + [0, -0.1, 0],
            plot_group.get_bottom() + [0, -1, 0],
            sigproc.get_top() + [0, 1, 0],
            sigproc.get_top() + [0, 0.1, 0],
        )
        phased_array_bez = CubicBezier(
            plot_group.get_bottom() + [0, -0.1, 0],
            plot_group.get_bottom() + [0, -1, 0],
            phased_array.get_top() + [0, 1, 0],
            phased_array.get_top() + [0, 0.1, 0],
        )
        sound_bez = CubicBezier(
            plot_group.get_bottom() + [0, -0.1, 0],
            plot_group.get_bottom() + [0, -1, 0],
            sound.get_top() + [0, 1, 0],
            sound.get_top() + [0, 0.1, 0],
        )
        etc_bez = CubicBezier(
            plot_group.get_bottom() + [0, -0.1, 0],
            plot_group.get_bottom() + [0, -1, 0],
            etc.get_top() + [0, 1, 0],
            etc.get_top() + [0, 0.1, 0],
        )

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.4).shift(
                    (
                        self.camera.frame.get_top()
                        - time_label.get_top()
                        - MED_SMALL_BUFF
                    )
                    * DOWN
                ),
                Create(sigproc_bez),
                GrowFromCenter(sigproc),
                Create(phased_array_bez),
                GrowFromCenter(phased_array),
                Create(sound_bez),
                GrowFromCenter(sound),
                Create(etc_bez),
                GrowFromCenter(etc),
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        self.play(
            ShrinkToCenter(etc),
            Uncreate(etc_bez),
            ShrinkToCenter(sound),
            Uncreate(sound_bez),
            ShrinkToCenter(phased_array),
            Uncreate(phased_array_bez),
            ShrinkToCenter(sigproc),
            Uncreate(sigproc_bez),
        )

        self.wait(0.5)

        fmcw_thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .shift(DOWN * 5)
            .shift(RIGHT * 12)
        )
        fmcw_box = SurroundingRectangle(fmcw_thumbnail)
        fmcw = Group(fmcw_thumbnail, fmcw_box)
        pulsed_thumbnail = (
            ImageMobject(
                "../06_radar_range_equation/media/images/radar_equation/thumbnails/Thumbnail_1.png"
            )
            .scale_to_fit_width(config.frame_width * 0.5)
            .shift(DOWN * 5)
            .shift(RIGHT * 12)
        )
        pulsed_box = SurroundingRectangle(pulsed_thumbnail)
        pulsed = Group(pulsed_thumbnail, pulsed_box)
        cfar_thumbnail = (
            ImageMobject("../03_cfar/media/images/cfar/thumbnails/Thumbnail_1.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .shift(DOWN * 5)
            .shift(RIGHT * 12)
        )
        cfar_box = SurroundingRectangle(cfar_thumbnail)
        cfar = Group(cfar_thumbnail, cfar_box)
        phased_thumbnail = (
            ImageMobject(
                "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
            .scale_to_fit_width(config.frame_width * 0.5)
            .shift(DOWN * 5)
            .shift(RIGHT * 12)
        )
        phased_box = SurroundingRectangle(phased_thumbnail)
        phased = Group(phased_thumbnail, phased_box)

        self.play(fmcw.animate.shift(LEFT * 12))

        self.wait(0.5)

        self.play(
            Group(fmcw, pulsed).animate.arrange(RIGHT, LARGE_BUFF).set_y(fmcw.get_y()),
        )
        self.wait(0.5)

        self.play(
            Group(pulsed, cfar).animate.arrange(RIGHT, LARGE_BUFF).set_y(fmcw.get_y()),
            fmcw.animate.shift(LEFT * 12),
        )
        self.wait(0.5)

        self.play(
            Group(cfar, phased).animate.arrange(RIGHT, LARGE_BUFF).set_y(fmcw.get_y()),
            pulsed.animate.shift(LEFT * 12),
        )

        self.wait(0.5)

        self.play(
            Group(cfar, phased).animate.shift(DOWN * 8),
            self.camera.frame.animate.restore(),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        window_plot = always_redraw(
            lambda: ax.plot(get_f_window(), x_range=[0, 1, 1 / 200], color=YELLOW)
        )
        window_plot_dn = always_redraw(
            lambda: ax.plot(get_f_window(-1), x_range=[0, 1, 1 / 200], color=YELLOW)
        )

        self.play(Create(window_plot), Create(window_plot_dn))

        self.wait(0.5)

        self.play(window_pow @ 1, run_time=3)

        self.wait(0.5)

        phased_thumbnail = (
            ImageMobject(
                "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
            .scale_to_fit_width(config.frame_width * 0.9)
            .shift(LEFT * 18)
        )
        phased_box = SurroundingRectangle(phased_thumbnail)
        phased = Group(phased_thumbnail, phased_box)

        self.play(phased.animate.move_to(ORIGIN))
        self.remove(sine_plot, fft_plot, window_plot, window_plot_dn)

        self.wait(0.5)

        self.play(phased.animate.shift(RIGHT * 18))

        self.wait(0.5)

        line_x0 = VT(0)
        line_x1 = VT(1)
        line_plot = always_redraw(
            lambda: ax.plot(
                lambda x: 1, color=BLUE, x_range=[~line_x0, ~line_x1, 1 / 50]
            )
        )
        sinc_plot = always_redraw(
            lambda: f_ax.plot(
                lambda t: np.clip(
                    0.7 * np.log10(np.abs(np.sinc(2 * PI * t / 15))) + 1, -0.5, None
                ),
                x_range=[-15, 15, 1 / 200],
                color=RED,
            )
        )

        self.play(Create(line_plot), Create(sinc_plot))

        self.wait(0.5)

        comment = (
            ImageMobject("./static/extremely_hateful_comment.png")
            .scale_to_fit_width(config.frame_width * 0.7)
            .to_edge(DOWN, SMALL_BUFF)
        )

        self.play(
            comment.shift(DOWN * 5).animate.shift(UP * 5),
            self.camera.frame.animate.shift(DOWN),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP),
                comment.animate.shift(DOWN * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                FadeOut(arrow),
                AnimationGroup(
                    MoveAlongPath(
                        time_group,
                        Line(
                            time_group.get_center(),
                            time_group.copy()
                            .to_edge(UP, MED_LARGE_BUFF)
                            .set_x(0)
                            .get_center(),
                            path_arc=-PI / 2,
                        ),
                    ),
                    MoveAlongPath(
                        f_group,
                        Line(
                            f_group.get_center(),
                            f_group.copy()
                            .to_edge(DOWN, MED_LARGE_BUFF)
                            .set_x(0)
                            .get_center(),
                            path_arc=-PI / 2,
                        ),
                    ),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        impulse = Line(f_ax.c2p(0, 0), f_ax.c2p(0, 1), color=RED)
        impulse_dot = Dot(impulse.get_top(), color=RED)

        self.play(
            line_x0 @ -4,
            line_x1 @ 5,
            LaggedStart(
                FadeOut(sinc_plot),
                Create(impulse),
                Create(impulse_dot),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        box_x0 = VT(-4)
        box_x1 = VT(5)

        time_box = always_redraw(
            lambda: Polygon(
                ax.c2p(~box_x0, 0),
                ax.c2p(~box_x0, 1.1),
                ax.c2p(~box_x1, 1.1),
                ax.c2p(~box_x1, 0),
                fill_color=GREEN,
                fill_opacity=0.3,
                stroke_opacity=0,
            )
        )

        self.play(FadeIn(time_box))

        self.wait(0.5)

        self.play(box_x0 @ 0, box_x1 @ 1, run_time=2)

        self.wait(0.5)

        sinc_plot = f_ax.plot(
            lambda t: np.clip(
                0.7 * np.log10(np.abs(np.sinc(2 * PI * t / 15))) + 1, -0.5, None
            ),
            x_range=[-15, 15, 1 / 200],
            color=RED,
        )

        self.play(FadeOut(impulse_dot, impulse), Create(sinc_plot))

        self.wait(0.5)

        self.remove(phased, cfar)
        self.play(self.camera.frame.animate.shift(UP * config.frame_height * 2))

        self.wait(2)


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
            # LaggedStart(*[Create(sample) for sample in samples], lag_ratio=0.2),
            LaggedStart(*[Create(dot) for dot in sample_dots], lag_ratio=0.2),
        )

        self.next_section(skip_animations=skip_animations(True))
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
        # samples.add_updater(
        #     lambda m: m.shift([0, (m.get_y() - ax.get_y()) - samples_y, 0])
        # )

        new_ax_group = Group(ax.copy(), f_ax.copy()).arrange(DOWN, MED_SMALL_BUFF)
        self.play(
            ax.animate.move_to(new_ax_group[0]),
            f_ax.animate.move_to(new_ax_group[1]),
            sample_dots.animate.shift(UP * (new_ax_group[0].get_y() - ax.get_y())),
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

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        sinc_x0 = VT(0)
        sinc_x1 = VT(0.03)
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

        self.next_section(skip_animations=skip_animations(True))
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

        # self.camera.frame.save_state()
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

        comp_nums = Group(
            *[
                MathTex(f"{n * 2 + 1}")
                .scale(2)
                .next_to(sc.get_top(), DOWN, MED_SMALL_BUFF)
                for n, sc in enumerate(square_comps)
            ]
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in comp_nums], lag_ratio=0.2))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate(rate_func=rate_functions.ease_in_sine).shift(
                UP * 14
            )
        )
        self.remove(*[c for c in comp_nums])

        self.wait(0.5)

        self.play(
            self.camera.frame.animate(rate_func=rate_functions.ease_out_sine).shift(
                DOWN * 14
            )
        )

        self.wait(0.5)

        N_slash = 2

        crosses = Group()
        for n in range(N_slash):
            cross_ur = Line(
                square_comps[-n].get_corner(DL),
                square_comps[-n].get_corner(UR),
                color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            cross_ul = Line(
                square_comps[-n].get_corner(DR),
                square_comps[-n].get_corner(UL),
                color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            crosses.add(Group(cross_ul, cross_ur))

        self.play(
            LaggedStart(
                *[sc.animate.set_stroke(opacity=0.2) for sc in square_comps[-N_slash:]],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeOut(sc) for sc in square_comps],
                self.camera.frame.animate.scale_to_fit_height(
                    Group(ax, f_ax).height * 1.4
                ).move_to(Group(ax, f_ax)),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class SquareApprox(MovingCameraScene):
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

        N_disp = MathTex("N = 1").to_corner(UL, LARGE_BUFF)
        N_num_disp = N_disp[0][-1]
        N_disp_old = N_num_disp

        # self.play(Write(N_disp))

        self.add(ax, sq, sinc, N_disp)
        self.camera.frame.shift(DOWN * config.frame_height)

        self.play(
            self.camera.frame.animate(rate_func=rate_functions.ease_out_sine).shift(
                UP * config.frame_height
            )
        )

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
            # anim_time = 1

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
            # new_x = np.arange(0, 2, 1 / audio_rate)
            # new_data = amplitude * square_wave_fourier(new_x, N, f=PI * sig_f)

            # data = np.concatenate([data, new_data])

        # fname = f"audio/data_Nmax_{N_max}.wav"
        # print(f"Writing audio to: {fname}")
        # wavfile.write(fname, audio_rate, data.astype(np.int16))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate(rate_func=rate_functions.ease_in_sine).shift(
                DOWN * config.frame_height
            )
        )

        self.wait(2)


class RealWorld(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        N_elem = 8
        N_elem_full = 40
        antennas = Group()
        for n in range(N_elem_full):
            antenna_port = Line(DOWN * 0.3, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange(RIGHT, MED_SMALL_BUFF)

        # self.add(antennas)
        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in antennas[N_elem_full // 2 : N_elem_full // 2 + N_elem // 2]
                ],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in antennas[
                        N_elem_full // 2 - N_elem // 2 : N_elem_full // 2
                    ][::-1]
                ],
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in antennas[N_elem_full // 2 + N_elem // 2 :]
                ],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[
                    GrowFromCenter(m)
                    for m in antennas[: N_elem_full // 2 - N_elem // 2][::-1]
                ],
                lag_ratio=0.2,
            ),
            self.camera.frame.animate(run_time=4).scale_to_fit_width(
                antennas.width * 1.2
            ),
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                *[m for m in antennas[N_elem_full // 2 + N_elem // 2 :]],
                *[m for m in antennas[: N_elem_full // 2 - N_elem // 2][::-1]],
            ),
            self.camera.frame.animate.restore(),
        )

        self.wait(0.5)

        no_elem = DashedVMobject(
            SurroundingRectangle(
                antennas[N_elem_full // 2 - N_elem // 2 - 1], color=RED
            ),
        )
        no_elem_Np1 = DashedVMobject(
            SurroundingRectangle(antennas[N_elem_full // 2 + N_elem // 2], color=RED),
        )
        elems_1 = Group(
            *[
                MathTex("1").next_to(ant, UP)
                for ant in antennas[
                    N_elem_full // 2 - N_elem // 2 : N_elem_full // 2 + N_elem // 2
                ]
            ]
        )
        no_elem_0 = MathTex("0").next_to(no_elem, UP).set_y(elems_1.get_y())
        no_elem_Np1_0 = MathTex("0").next_to(no_elem_Np1, UP).set_y(elems_1.get_y())

        self.play(Create(no_elem), FadeIn(no_elem_0))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(samp) for samp in elems_1],
                AnimationGroup(Create(no_elem_Np1), FadeIn(no_elem_Np1_0)),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[0, 1, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=config.frame_width * 0.6,
            y_length=antennas.height * 2,
        )
        ax.shift(antennas.get_top() - ax.c2p(0, 0))

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.01)
        theta_max = VT(0.01)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -50

        x_len = config.frame_height * 0.6
        polar_ax = (
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
        polar_ax.shift(antennas.get_top() - polar_ax.c2p(0, 0))

        ap_rect_amp = VT(1)
        ap_polar_amp = VT(0.01)

        def get_ap_polar(
            theta_min_inp=theta_min,
            theta_max_inp=theta_max,
            fill_color=GREEN,
            fill_opacity=0,
        ):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                weights = np.array([~w for w in weight_trackers])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None) * ~ap_polar_amp
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~theta_min_inp, ~theta_max_inp, 1 / 400],
                    # theta_range=[~theta_min, ~theta_max, 2 * PI / 200],
                    color=TX_COLOR,
                    use_smoothing=False,
                    # stroke_opacity=~af_opacity,
                    fill_opacity=fill_opacity,
                    fill_color=fill_color,
                ).set_z_index(-2)
                return plot

            return updater

        def get_ap():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            # EP = sinc_pattern(u, 0, L, W, wavelength_0)
            # AP = AF * (EP ** (~ep_exp_scale))
            AP = AF
            AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None) * ~ap_rect_amp
            AP /= AP.max()
            f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
            plot = ax.plot(
                f_AP,
                x_range=[~theta_min, ~theta_max, 1 / 200],
                # theta_range=[~theta_min, ~theta_max, 2 * PI / 200],
                color=TX_COLOR,
                use_smoothing=False,
                # stroke_opacity=~af_opacity,
            )
            return plot

        AF_plot = always_redraw(get_ap)
        AF_polar_plot = always_redraw(get_ap_polar())

        self.add(AF_plot)

        AF_label = Tex("Array Factor").scale(2).to_edge(UP).shift(UP * 2)
        AF_label_bez_l = CubicBezier(
            ax.c2p(-PI, 0) + [0, 1.5, 0],
            ax.c2p(-PI, 0) + [0, 3, 0],
            AF_label.get_bottom() + [0, -1, 0],
            AF_label.get_bottom() + [0, -0.1, 0],
        )
        AF_label_bez_r = CubicBezier(
            ax.c2p(PI, 0) + [0, 1.5, 0],
            ax.c2p(PI, 0) + [0, 3, 0],
            AF_label.get_bottom() + [0, -1, 0],
            AF_label.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                FadeOut(no_elem_0, no_elem_Np1_0, elems_1),
                AnimationGroup(
                    self.camera.frame.animate.shift(UP * 2),
                    theta_min @ (-PI),
                    theta_max @ PI,
                ),
                AnimationGroup(
                    Create(AF_label_bez_l),
                    Create(AF_label_bez_r),
                ),
                FadeIn(AF_label),
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            FadeOut(AF_label),
            Uncreate(AF_label_bez_l),
            Uncreate(AF_label_bez_r),
            self.camera.frame.animate.shift(DOWN),
        )

        self.wait(0.5)

        background_rotate = VT(-PI / 2)
        background = always_redraw(
            lambda: SurroundingRectangle(
                antennas,
                color=BACKGROUND_COLOR,
                buff=0,
                fill_opacity=1,
                fill_color=BACKGROUND_COLOR,
            )
            .stretch(2, 1)
            .move_to(polar_ax.c2p(0, 0), UP)
            .rotate(
                Line(-polar_ax.c2p(0, 0), polar_ax.c2p(0, 1)).get_angle()
                + ~background_rotate,
                about_point=polar_ax.c2p(0, 0),
            )
            .set_z_index(-1)
        )
        self.add(background)
        self.next_section(skip_animations=skip_animations(False))

        # TODO: This animation is bad
        self.add(AF_polar_plot)
        self.play(
            LaggedStart(FadeOut(AF_plot), ap_polar_amp @ 1, lag_ratio=0.5),
            run_time=3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        right_theta_0 = VT(-0.01)
        right_theta_1 = VT(0)
        left_theta_0 = VT(0)
        left_theta_1 = VT(0.01)

        r_min_amp = VT(1.1)
        shade_left = always_redraw(
            lambda: polar_ax.plot_polar_graph(
                lambda theta: -r_min * ~r_min_amp,
                theta_range=[~left_theta_0, ~left_theta_1],
                stroke_opacity=0,
            )
        )
        shade_right = always_redraw(
            lambda: polar_ax.plot_polar_graph(
                lambda theta: -r_min * ~r_min_amp,
                theta_range=[~right_theta_0, ~right_theta_1],
                stroke_opacity=0,
            )
        )

        ap_left = always_redraw(
            lambda: ArcPolygon(
                shade_left.get_start(),
                shade_left.get_end(),
                polar_ax.c2p(0, 0),
                fill_opacity=0.3,
                fill_color=BLUE,
                stroke_width=0,
                arc_config=[
                    {
                        "angle": (~right_theta_1 - ~right_theta_0),
                        "stroke_opacity": 0,
                        "stroke_width": 0,
                    },
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                ],
            )
        )
        ap_right = always_redraw(
            lambda: ArcPolygon(
                shade_right.get_start(),
                shade_right.get_end(),
                polar_ax.c2p(0, 0),
                fill_opacity=0.3,
                fill_color=BLUE,
                stroke_width=0,
                arc_config=[
                    {
                        "angle": (~left_theta_1 - ~left_theta_0),
                        "stroke_opacity": 0,
                        "stroke_width": 0,
                    },
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                    {"angle": 0, "stroke_opacity": 0, "stroke_width": 0},
                ],
            )
        )

        main_lobe_label = Tex("Main lobe").next_to(
            ax.input_to_graph_point(0, AF_plot), UP
        )
        side_lobes_label = Tex("Side lobes").move_to(main_lobe_label).shift(UP / 2)

        fnbw = 2 * wavelength_0 / (n_elem * d_x)
        fnbw = 2 * np.arcsin(2 / n_elem)
        self.add(shade_left, shade_right, ap_left, ap_right)
        self.play(
            LaggedStart(
                AnimationGroup(
                    left_theta_0 @ (-fnbw * np.sqrt(2)),
                    left_theta_1 @ (0),
                    right_theta_0 @ (0),
                    right_theta_1 @ (fnbw * np.sqrt(2)),
                ),
                FadeIn(main_lobe_label, shift=DOWN * 3),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        taylor_taper = signal.windows.taylor(n_elem_full, nbar=5, sll=40)
        rect_small = np.zeros(n_elem_full)
        rect_small[n_elem_full // 2 - n_elem // 2 : n_elem_full // 2] = 1
        rect_small[n_elem_full // 2 : n_elem_full // 2 + n_elem // 2] = 1

        self.play(
            LaggedStart(
                AnimationGroup(
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[: n_elem_full // 2][::-1],
                                taylor_taper[: n_elem_full // 2][::-1],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[n_elem_full // 2 :],
                                taylor_taper[n_elem_full // 2 :],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                ),
                AnimationGroup(
                    left_theta_0 @ (-fnbw * np.sqrt(2) * 0.55),
                    left_theta_1 @ (0),
                    right_theta_0 @ (0),
                    right_theta_1 @ (fnbw * np.sqrt(2) * 0.55),
                ),
                lag_ratio=0.3,
            ),
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[: n_elem_full // 2][::-1],
                                rect_small[: n_elem_full // 2][::-1],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[n_elem_full // 2 :],
                                rect_small[n_elem_full // 2 :],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                ),
                AnimationGroup(
                    left_theta_0 @ (-fnbw * np.sqrt(2)),
                    left_theta_1 @ (0),
                    right_theta_0 @ (0),
                    right_theta_1 @ (fnbw * np.sqrt(2)),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        side_lobe_bez_l = CubicBezier(
            side_lobes_label.get_bottom() + [0, -0.1, 0],
            side_lobes_label.get_bottom() + [0, -1, 0],
            polar_ax.input_to_graph_point(PI * 0.2, AF_polar_plot) + [-0.2, 1, 0],
            polar_ax.input_to_graph_point(PI * 0.2, AF_polar_plot) + [-0.05, 0.2, 0],
        )
        side_lobe_bez_r = CubicBezier(
            side_lobes_label.get_bottom() + [0, -0.1, 0],
            side_lobes_label.get_bottom() + [0, -1, 0],
            polar_ax.input_to_graph_point(-PI * 0.2, AF_polar_plot) + [0.2, 1, 0],
            polar_ax.input_to_graph_point(-PI * 0.2, AF_polar_plot) + [0.05, 0.2, 0],
        )

        self.play(
            LaggedStart(
                ShrinkToCenter(main_lobe_label, shift=DOWN * 3),
                AnimationGroup(
                    left_theta_0 @ (-fnbw * np.sqrt(2) - PI * 0.15),
                    left_theta_1 @ (-fnbw * np.sqrt(2) - PI * 0.02),
                    right_theta_0 @ (fnbw * np.sqrt(2) + PI * 0.02),
                    right_theta_1 @ (fnbw * np.sqrt(2) + PI * 0.15),
                    r_min_amp @ 0.75,
                ),
                GrowFromCenter(side_lobes_label),
                AnimationGroup(
                    Create(side_lobe_bez_l),
                    Create(side_lobe_bez_r),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            left_theta_0 @ ((-fnbw * np.sqrt(2) - PI * 0.15) * 1.5),
            left_theta_1 @ ((-fnbw * np.sqrt(2) - PI * 0.02) * 2),
            right_theta_0 @ ((fnbw * np.sqrt(2) + PI * 0.02) * 2),
            right_theta_1 @ ((fnbw * np.sqrt(2) + PI * 0.15) * 1.5),
            r_min_amp @ 0.68,
            Transform(
                side_lobe_bez_l,
                CubicBezier(
                    side_lobes_label.get_bottom() + [0, -0.1, 0],
                    side_lobes_label.get_bottom() + [0, -1, 0],
                    polar_ax.input_to_graph_point(PI * 0.2 * 1.5, AF_polar_plot)
                    + [-0.4, 1.5, 0],
                    polar_ax.input_to_graph_point(PI * 0.2 * 1.5, AF_polar_plot)
                    + [-0.15, 0.2, 0],
                ),
            ),
            Transform(
                side_lobe_bez_r,
                CubicBezier(
                    side_lobes_label.get_bottom() + [0, -0.1, 0],
                    side_lobes_label.get_bottom() + [0, -1, 0],
                    polar_ax.input_to_graph_point(-PI * 0.2 * 1.5, AF_polar_plot)
                    + [0.4, 1.5, 0],
                    polar_ax.input_to_graph_point(-PI * 0.2 * 1.5, AF_polar_plot)
                    + [0.15, 0.2, 0],
                ),
            ),
        )

        self.wait(0.5)

        self.play(
            left_theta_0 @ (-PI / 2),
            left_theta_1 @ (-PI * 0.38),
            right_theta_1 @ (PI / 2),
            right_theta_0 @ (PI * 0.38),
            # left_theta_1 @ (~left_theta_1),
            Transform(
                side_lobe_bez_l,
                CubicBezier(
                    side_lobes_label.get_bottom() + [0, -0.1, 0],
                    side_lobes_label.get_bottom() + [0, -1, 0],
                    polar_ax.input_to_graph_point(PI * 0.3 * 1.5, AF_polar_plot)
                    + [-1, 1.5, 0],
                    polar_ax.input_to_graph_point(PI * 0.3 * 1.5, AF_polar_plot)
                    + [-0.3, 0.2, 0],
                ),
            ),
            Transform(
                side_lobe_bez_r,
                CubicBezier(
                    side_lobes_label.get_bottom() + [0, -0.1, 0],
                    side_lobes_label.get_bottom() + [0, -1, 0],
                    polar_ax.input_to_graph_point(-PI * 0.3 * 1.5, AF_polar_plot)
                    + [1, 1.5, 0],
                    polar_ax.input_to_graph_point(-PI * 0.3 * 1.5, AF_polar_plot)
                    + [0.3, 0.2, 0],
                ),
            ),
        )

        self.wait(0.5)

        rect_small = np.zeros(n_elem_full)
        rect_small[n_elem_full // 2 - n_elem // 2 - 4 : n_elem_full // 2] = 1
        rect_small[n_elem_full // 2 : n_elem_full // 2 + n_elem // 2 + 4] = 1

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(side_lobe_bez_l),
                    Uncreate(side_lobe_bez_r),
                    FadeOut(side_lobes_label),
                    FadeOut(ap_left, ap_right),
                ),
                AnimationGroup(
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[: n_elem_full // 2][::-1],
                                rect_small[: n_elem_full // 2][::-1],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[n_elem_full // 2 :],
                                rect_small[n_elem_full // 2 :],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        line = Line(
            polar_ax.input_to_graph_point(PI * 0.1, AF_polar_plot),
            polar_ax.input_to_graph_point(PI * 0.1, AF_polar_plot) + [-0.3, 0.5, 0],
        )
        line_mid = Line(ORIGIN, UP * 0.65)
        line_u = Line(
            line_mid.get_top() + LEFT / 8,
            line_mid.get_top() + RIGHT / 8,
        )
        line_d = Line(
            line_mid.get_bottom() + LEFT / 8,
            line_mid.get_bottom() + RIGHT / 8,
        )
        line = (
            Group(line_mid, line_u, line_d)
            .rotate(PI * 0.13)
            .next_to(polar_ax.input_to_graph_point(PI * 0.1, AF_polar_plot), UP, buff=0)
            .shift(LEFT * 0.3)
        )
        self.play(
            LaggedStart(
                Create(line_d),
                Create(line_mid),
                Create(line_u),
            )
        )

        self.wait(0.5)

        rect_small = np.zeros(n_elem_full)
        rect_small[n_elem_full // 2 - n_elem // 2 - 6 : n_elem_full // 2] = 1
        rect_small[n_elem_full // 2 : n_elem_full // 2 + n_elem // 2 + 6] = 1

        self.play(
            LaggedStart(
                AnimationGroup(
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[: n_elem_full // 2][::-1],
                                rect_small[: n_elem_full // 2][::-1],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            wt @ taper_wt
                            for wt, taper_wt in zip(
                                weight_trackers[n_elem_full // 2 :],
                                rect_small[n_elem_full // 2 :],
                            )
                        ],
                        lag_ratio=0.2,
                    ),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        rect_small = np.zeros(n_elem_full)
        rect_small[n_elem_full // 2 - n_elem // 2 : n_elem_full // 2] = 1
        rect_small[n_elem_full // 2 : n_elem_full // 2 + n_elem // 2] = 1

        self.play(
            LaggedStart(
                Uncreate(line_d),
                Uncreate(line_mid),
                Uncreate(line_u),
                lag_ratio=0.15,
            ),
            LaggedStart(
                *[
                    wt @ taper_wt
                    for wt, taper_wt in zip(
                        weight_trackers[: n_elem_full // 2][::-1],
                        rect_small[: n_elem_full // 2][::-1],
                    )
                ],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[
                    wt @ taper_wt
                    for wt, taper_wt in zip(
                        weight_trackers[n_elem_full // 2 :],
                        rect_small[n_elem_full // 2 :],
                    )
                ],
                lag_ratio=0.2,
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        car = (
            SVGMobject("../props/static/car.svg")
            .set_color(WHITE)
            .set_fill(WHITE)
            .to_edge(LEFT, LARGE_BUFF)
        )
        car_right = car.get_right()

        self.play(
            LaggedStart(
                AnimationGroup(
                    LaggedStart(
                        *[
                            FadeOut(m)
                            for m in antennas[
                                N_elem_full // 2 : N_elem_full // 2 + N_elem // 2
                            ]
                        ],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            FadeOut(m)
                            for m in antennas[
                                N_elem_full // 2 - N_elem // 2 : N_elem_full // 2
                            ][::-1]
                        ],
                        lag_ratio=0.2,
                    ),
                    FadeOut(no_elem, no_elem_Np1),
                ),
                self.camera.frame.animate.scale_to_fit_width(
                    config.frame_width
                ).move_to(ORIGIN),
                AnimationGroup(
                    car.shift(LEFT * 8).animate.shift(RIGHT * 8),
                    polar_ax.animate.rotate(-PI / 2).shift(
                        car_right - polar_ax.c2p(0, 0)
                    ),
                    background_rotate @ (-3 * PI / 2),
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        blue_car = (
            SVGMobject("../props/static/car.svg")
            .set_color(BLUE)
            .set_fill(BLUE)
            .to_edge(RIGHT, LARGE_BUFF)
            .shift(DOWN * 2.5 + LEFT * 2.5)
        )
        red_car = (
            SVGMobject("../props/static/car.svg")
            .set_color(RED)
            .set_fill(RED)
            .to_edge(RIGHT, LARGE_BUFF)
        )

        p1 = VT(0)
        p2 = VT(-30)
        f1 = 2.7
        f2 = 1.5

        fs = 1000
        end_time = 8
        N = fs * end_time
        t = np.linspace(0, end_time, N)
        fft_len = N * 4
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        window = signal.windows.blackman(N)
        # window = np.ones(N)

        xmin, xmax = 0, 8
        f_ax = (
            Axes(
                x_range=[xmin, xmax, fs / 4],
                y_range=[0, 43, 10],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=config.frame_width * 0.6,
                y_length=antennas.height * 1.5,
            )
            .next_to(AF_polar_plot, UP)
            .set_x(0)
        )
        # self.add(f_ax)
        # self.camera.frame.shift(UP)
        np.random.seed(0)
        noise = np.random.normal(loc=0, scale=0.1, size=N)

        # noise = np.zeros(N)

        def plot_fft(xmin=xmin, xmax=xmax, color=TX_COLOR):
            def updater():
                x_n = (
                    db2lin(~p1) * np.sin(2 * PI * f1 * t)
                    + db2lin(~p2) * np.sin(2 * PI * f2 * t)
                    + noise
                )
                x_n_windowed = x_n * window
                X_k = fftshift(fft(x_n_windowed, fft_len))
                X_k /= N / 2
                X_k = np.abs(X_k)
                X_k = 10 * np.log10(X_k) + 43

                f_X_k = interp1d(freq, X_k, fill_value="extrapolate")
                fft_plot = f_ax.plot(
                    f_X_k,
                    color=color,
                    x_range=[xmin, xmax, (xmax - xmin) / 1000],
                    use_smoothing=False,
                )
                return fft_plot

            return updater

        fft_plot = always_redraw(plot_fft())

        self.play(red_car.shift(RIGHT * 8).animate.shift(LEFT * 8), run_time=2)

        self.wait(0.5)

        self.play(Create(f_ax), Create(fft_plot), run_time=2)

        self.wait(0.5)

        blue_arrow = Arrow(polar_ax.c2p(0, 0), blue_car.get_corner(UL) + [0.5, -0.5, 0])
        self.play(
            LaggedStart(
                blue_car.shift(LEFT * 12).animate.shift(RIGHT * 12),
                GrowArrow(blue_arrow),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(p2 @ (-13), run_time=2)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        f2bw = (0.3, 0.35)
        fft_plot_close_target = plot_fft(
            xmin=f2 - f2bw[0], xmax=f2 + f2bw[1], color=YELLOW
        )()
        self.play(Create(fft_plot_close_target), run_time=2)

        self.wait(0.5)

        antennas_new = Group()
        for n in range(2):
            antenna_port = Line(DOWN * 0.3, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas_new.add(antenna)
        antennas_new.arrange(RIGHT, MED_SMALL_BUFF).to_edge(LEFT, LARGE_BUFF * 2).shift(
            DOWN
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    car.animate.shift(LEFT * 8),
                    blue_car.animate.shift(RIGHT * 8),
                    red_car.animate.shift(RIGHT * 8),
                    f_ax.animate.shift(UP * 8),
                    Uncreate(fft_plot_close_target),
                    FadeOut(blue_arrow),
                ),
                AnimationGroup(
                    polar_ax.animate.rotate(PI / 2).shift(
                        antennas_new.get_top() - polar_ax.c2p(0, 0)
                    ),
                    background_rotate @ (-PI),
                    antennas_new.shift(DOWN * 8).animate.shift(UP * 8),
                ),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        radar = Tex("RADAR").scale(1.5)
        comms = Tex("Communications").scale(1.5)
        etc = Tex("etc.").scale(1.5)
        Group(radar, comms, etc).arrange(DOWN, LARGE_BUFF, aligned_edge=LEFT).to_edge(
            RIGHT, LARGE_BUFF
        )
        radar_bez = CubicBezier(
            antennas_new.get_corner(UR) + [0.1, 0, 0],
            antennas_new.get_corner(UR) + [2, 0, 0],
            radar.get_left() + [-1, 0, 0],
            radar.get_left() + [-0.1, 0, 0],
        )
        comms_bez = CubicBezier(
            antennas_new.get_corner(UR) + [0.1, 0, 0],
            antennas_new.get_corner(UR) + [2, 0, 0],
            comms.get_left() + [-1, 0, 0],
            comms.get_left() + [-0.1, 0, 0],
        )
        etc_bez = CubicBezier(
            antennas_new.get_corner(UR) + [0.1, 0, 0],
            antennas_new.get_corner(UR) + [2, 0, 0],
            etc.get_left() + [-1, 0, 0],
            etc.get_left() + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(radar_bez),
                    radar.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                LaggedStart(
                    Create(comms_bez),
                    comms.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                LaggedStart(
                    Create(etc_bez),
                    etc.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class TaperIntro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        N_elem = 4
        antennas = Group()
        for n in range(N_elem):
            antenna_port = Line(DOWN * 0.3, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange(RIGHT, MED_SMALL_BUFF)

        ax = Axes(
            x_range=[-N_elem / 2, N_elem / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=antennas.width,
            y_length=antennas.height * 1.05,
        ).set_opacity(0)
        ax.shift(antennas.get_bottom() - ax.c2p(0, 0))

        X = np.linspace(-N_elem / 2 - 0.05, N_elem / 2 + 0.05, 2**10)

        def get_f_window():
            window = np.clip(
                signal.windows.hann(2**10) ** ~hann_scalar
                + signal.windows.taylor(2**10, 5, 30) ** ~taylor_scalar
                + signal.windows.hamming(2**10) ** ~hamming_scalar
                + signal.windows.tukey(2**10, 0.5) ** ~tukey_scalar
                - 3,
                0,
                None,
            )
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        hann_scalar = VT(0)
        taylor_scalar = VT(0)
        hamming_scalar = VT(0)
        tukey_scalar = VT(0)

        disc = always_redraw(
            lambda: ax.plot(
                lambda x: get_f_window()(x)
                if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
                else 0,
                x_range=[-N_elem // 2 - 1, N_elem // 2 + 1, 1 / 100],
                color=BLUE,
                use_smoothing=False,
            ).set_z_index(1)
        )

        cartoon_group = Group(antennas, ax)

        # self.add(antennas, disc)

        self.play(LaggedStart(*[FadeIn(m) for m in antennas], lag_ratio=0.2))

        self.wait(0.5)

        self.play(Create(disc))

        self.wait(0.5)

        r_min = -60

        x_len = config.frame_height * 0.6
        polar_ax = (
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
        polar_ax.shift(antennas.get_top() - polar_ax.c2p(0, 0))

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        X_step = N_elem / n_elem
        X_weights = np.linspace(
            -N_elem / 2 + X_step / 2, N_elem / 2 - X_step / 2, n_elem
        )
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.001)
        theta_max = VT(0.001)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        def get_ap_polar(
            theta_min_inp=theta_min,
            theta_max_inp=theta_max,
            polar_ax=polar_ax,
        ):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                # weights = np.array([~w for w in weight_trackers])
                weights = np.array([get_f_window()(x) for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~theta_min_inp, ~theta_max_inp, 1 / 400],
                    color=TX_COLOR,
                    use_smoothing=False,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)

        self.play(
            LaggedStart(
                Group(cartoon_group, polar_ax).animate.arrange(RIGHT, buff=LARGE_BUFF),
                AnimationGroup(
                    theta_min.animate(run_time=3).set_value(-PI),
                    theta_max.animate(run_time=3).set_value(PI),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                cartoon_group.width * 1.3
            ).move_to(cartoon_group)
        )

        self.wait(0.5)

        disc_left = ax.plot(
            lambda x: 1
            if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
            else 0,
            x_range=[-N_elem // 2 - 0.3, -N_elem // 2 + 0.3, 1 / 100],
            color=YELLOW,
            use_smoothing=False,
        ).set_z_index(2)
        disc_right = ax.plot(
            lambda x: 1
            if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
            else 0,
            x_range=[N_elem // 2 - 0.3, N_elem // 2 + 0.3, 1 / 100],
            color=YELLOW,
            use_smoothing=False,
        ).set_z_index(2)

        self.play(LaggedStart(Create(disc_left), Create(disc_right), lag_ratio=0.6))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(disc_left),
                Uncreate(disc_right),
                self.camera.frame.animate.restore(),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        shade_color_alpha = VT(0)
        shade = always_redraw(
            lambda: ax.get_area(
                disc,
                x_range=[-N_elem / 2 - 1, N_elem / 2 + 1],
                color=interpolate_color(BACKGROUND_COLOR, YELLOW, ~shade_color_alpha),
                stroke_opacity=0.7,
                stroke_width=0,
                opacity=0.7,
                bounded_graph=ax.plot(
                    lambda x: 2, x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100]
                ),
            )
        )
        self.next_section(skip_animations=skip_animations(True))
        self.add(shade)

        self.play(hann_scalar @ 1, run_time=5)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(hann_scalar @ 0, run_time=1)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        side_lobe_ang = PI * 0.3
        side_lobe_arrow_l = Arrow(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(side_lobe_ang),
                1.2 * -r_min * np.sin(side_lobe_ang),
            ),
            polar_ax.c2p(
                -r_min * 0.4 * np.cos(side_lobe_ang),
                -r_min * 0.4 * np.sin(side_lobe_ang),
            ),
        )
        side_lobe_arrow_r = Arrow(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(-side_lobe_ang),
                1.2 * -r_min * np.sin(-side_lobe_ang),
            ),
            polar_ax.c2p(
                -r_min * 0.4 * np.cos(-side_lobe_ang),
                -r_min * 0.4 * np.sin(-side_lobe_ang),
            ),
        )

        main_lobe_ang = PI / 6
        main_lobe_arc = ArcBetweenPoints(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(-main_lobe_ang),
                1.2 * -r_min * np.sin(-main_lobe_ang),
            ),
            polar_ax.c2p(
                1.2 * -r_min * np.cos(main_lobe_ang),
                1.2 * -r_min * np.sin(main_lobe_ang),
            ),
            angle=PI / 2,
        )
        main_lobe_arrow_l = CurvedArrow(
            main_lobe_arc.get_midpoint(), main_lobe_arc.get_end(), angle=PI / 6
        )
        main_lobe_arrow_r = CurvedArrow(
            main_lobe_arc.get_midpoint(), main_lobe_arc.get_start(), angle=-PI / 6
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    hann_scalar @ 1,
                    GrowArrow(side_lobe_arrow_l),
                    GrowArrow(side_lobe_arrow_r),
                ),
                AnimationGroup(
                    Create(main_lobe_arrow_l),
                    Create(main_lobe_arrow_r),
                ),
                lag_ratio=0.5,
            ),
            run_time=10,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(side_lobe_arrow_l),
                AnimationGroup(
                    FadeOut(main_lobe_arrow_l),
                    FadeOut(main_lobe_arrow_r),
                ),
                FadeOut(side_lobe_arrow_r),
            )
        )

        self.next_section(skip_animations=skip_animations(True))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.4).move_to(ax)
        )

        self.wait(0.5)

        sharp_transition = Arrow(ax.c2p(-N_elem / 2, 0), ax.c2p(-N_elem / 2, 1))
        smooth_transition = CurvedArrow(
            ax.c2p(-N_elem / 2, 0) + LEFT / 2 + UP / 4,
            ax.c2p(-N_elem / 2, 1) + RIGHT,
            angle=PI / 4,
        )

        self.play(GrowArrow(sharp_transition))

        self.wait(0.5)

        self.play(ReplacementTransform(sharp_transition, smooth_transition))

        self.wait(0.5)

        f1_plot = ax.plot(
            lambda t: (np.cos(2 * PI * 0.25 * t) + 1) / 2,
            x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100],
            color=ORANGE,
        )
        f2_plot = ax.plot(
            lambda t: (np.cos(2 * PI * 0.5 * t) + 1) / 2,
            x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100],
            color=ORANGE,
        )
        f3_plot = ax.plot(
            lambda t: (np.cos(2 * PI * 1 * t) + 1) / 2,
            x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100],
            color=ORANGE,
        )
        f4_plot = ax.plot(
            lambda t: (np.cos(2 * PI * 2 * t) + 1) / 2,
            x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100],
            color=ORANGE,
        )

        self.play(
            Create(f1_plot),
            Create(f2_plot),
            Create(f3_plot),
            Create(f4_plot),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    f1_plot.height * 6.5
                ).shift(f1_plot.height * 2.2 * UP),
                Group(f1_plot, f2_plot, f3_plot, f4_plot)
                .animate.arrange(UP, center=False)
                .next_to(ax, UP),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                f4_plot.animate.set_stroke(opacity=0.3),
                f3_plot.animate.set_stroke(opacity=0.3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(f4_plot),
                FadeOut(f3_plot),
                FadeOut(f2_plot),
                FadeOut(f1_plot),
                FadeOut(smooth_transition),
                self.camera.frame.animate.scale_to_fit_width(
                    Group(cartoon_group, AF_polar_plot).width * 1.4
                ).move_to(Group(cartoon_group, AF_polar_plot)),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # self.play(shade_color_alpha @ 1)

        self.play(
            LaggedStart(
                antennas[0]
                .set_z_index(-1)
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 2)
                .set_color(YELLOW),
                antennas[-1]
                .set_z_index(-1)
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 2)
                .set_color(YELLOW),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        shade_bot = always_redraw(
            lambda: ax.get_area(
                disc,
                x_range=[-N_elem / 2 - 1, N_elem / 2 + 1],
                color=YELLOW,
                stroke_opacity=0.7,
                stroke_width=0,
                opacity=0.7,
            )
        )

        self.play(FadeIn(shade_bot))

        self.wait(0.5)

        self.play(FadeOut(shade_bot))

        self.wait(0.5)

        N_elem_full = 12
        N_elem_scale = 3
        antennas_full = Group()
        for n in range(N_elem_full):
            antenna_port = Line(DOWN * 0.3, UP, color=WHITE)
            antenna_tri = (
                Triangle(color=WHITE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas_full.add(antenna)
        antennas_full.arrange(RIGHT, MED_SMALL_BUFF).next_to(
            self.camera.frame.get_bottom(), UP, LARGE_BUFF
        ).shift(DOWN * config.frame_height)
        polar_full_ax = (
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
        polar_full_ax.shift(antennas_full.get_top() - polar_full_ax.c2p(0, 0))

        weight_trackers = [VT(1) for _ in range(N_elem_full * N_elem_scale)]

        def get_ap_polar_full(polar_ax=polar_ax):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                weights = np.array([~w for w in weight_trackers])
                # weights = np.array([get_f_window()(x) ** ~scalar for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[-PI / 2, PI / 2, 1 / 400],
                    color=TX_COLOR,
                    use_smoothing=False,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot_full = always_redraw(get_ap_polar_full(polar_ax=polar_full_ax))
        self.add(polar_full_ax, antennas_full, AF_polar_plot_full)

        self.play(self.camera.frame.animate.shift(DOWN * config.frame_height))

        self.wait(0.5)

        N_elem_remaining = 2
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        ant[0].animate.set_stroke(opacity=0.3),
                        ant[1].animate.set_stroke(opacity=0.3),
                    )
                    for ant in antennas_full[: N_elem_full // 2 - N_elem_remaining]
                ],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[
                    AnimationGroup(
                        ant[0].animate.set_stroke(opacity=0.3),
                        ant[1].animate.set_stroke(opacity=0.3),
                    )
                    for ant in antennas_full[N_elem_full // 2 + N_elem_remaining :]
                ],
                lag_ratio=0.2,
            ),
            *[
                w @ 0
                for w in [
                    *weight_trackers[
                        N_elem_full * N_elem_scale // 2
                        + N_elem_remaining * N_elem_scale :
                    ],
                    *weight_trackers[
                        : N_elem_full * N_elem_scale // 2
                        - N_elem_remaining * N_elem_scale
                    ],
                ]
            ],
            run_time=4,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * config.frame_height))
        self.remove(antennas_full, polar_full_ax, AF_polar_plot_full)

        self.wait(0.5)

        usable_to_widen = Arrow(cartoon_group.get_right(), AF_polar_plot.get_left())
        shade_bot = always_redraw(
            lambda: ax.get_area(
                disc,
                x_range=[-N_elem / 2 - 1, N_elem / 2 + 1],
                color=YELLOW,
                stroke_opacity=0.7,
                stroke_width=0,
                opacity=0.7,
            )
        )

        self.play(
            LaggedStart(FadeIn(shade_bot), GrowArrow(usable_to_widen), lag_ratio=0.5)
        )

        self.wait(0.5)

        self.play(FadeOut(shade_bot, usable_to_widen))

        self.wait(0.5)

        hann_label = Tex("Hann Taper").next_to(cartoon_group, UP, LARGE_BUFF)
        taylor_label = Tex(r"Taylor Taper ($\bar{n}=5, sll=40$)").next_to(
            cartoon_group, UP, LARGE_BUFF
        )
        hamming_label = Tex("Hamming Taper").next_to(cartoon_group, UP, LARGE_BUFF)
        tukey_label = Tex(r"Tukey Taper ($\alpha = 0.5$)").next_to(
            cartoon_group, UP, LARGE_BUFF
        )

        self.play(hann_label.shift(UP * 5).animate.shift(DOWN * 5))

        self.wait(0.5)

        self.play(
            LaggedStart(
                hann_label[0][:4].animate.shift(UP * 5),
                GrowFromCenter(taylor_label[0][:6]),
                ReplacementTransform(hann_label[0][-5:], taylor_label[0][6:11]),
                LaggedStart(
                    AnimationGroup(
                        GrowFromCenter(taylor_label[0][11]),
                        GrowFromCenter(taylor_label[0][-1]),
                    ),
                    *[FadeIn(m, shift=UP / 2) for m in taylor_label[0][12:-1]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.2,
            ),
            hann_scalar @ 0,
            taylor_scalar @ 1,
            run_time=3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(
                    AnimationGroup(
                        ShrinkToCenter(taylor_label[0][11]),
                        ShrinkToCenter(taylor_label[0][-1]),
                    ),
                    *[FadeOut(m, shift=UP / 2) for m in taylor_label[0][12:-1]],
                    lag_ratio=0.1,
                ),
                taylor_label[0][:6].animate.shift(UP * 5),
                GrowFromCenter(hamming_label[0][:7]),
                ReplacementTransform(taylor_label[0][6:11], hamming_label[0][7:12]),
                lag_ratio=0.2,
            ),
            taylor_scalar @ 0,
            hamming_scalar @ 1,
            run_time=3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                hamming_label[0][:7].animate.shift(UP * 5),
                GrowFromCenter(tukey_label[0][:5]),
                ReplacementTransform(hamming_label[0][7:12], tukey_label[0][5:-7]),
                LaggedStart(*[GrowFromCenter(m) for m in tukey_label[0][-7:]]),
                lag_ratio=0.2,
            ),
            hamming_scalar @ 0,
            tukey_scalar @ 1,
            run_time=3,
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class TradeSpace(Scene):
    def construct(self):
        table_data = MathTable(
            [
                [
                    r"\textbf{Type}",
                    r"\textbf{Peak Side Lobe Amplitude}",
                    r"\textbf{Approx. Main Lobe Width}",
                ],
                ["Rectangular", "$-13$", r"$4\pi / (M+1)$"],
                ["Bartlett", "$-25$", r"$8\pi /M$"],
                ["Hann", "$-31$", r"$8\pi / M$"],
                ["Hamming", "$-41$", r"$8\pi /M$"],
                ["Blackman", "$-57$", r"$12\pi / M$"],
            ],
            Tex,
        ).scale_to_fit_width(config.frame_width * 0.8)
        table_ref = (
            Tex(
                r"\raggedright Alan V. Oppenheim and Ronald W. Schafer. 2009.\\ \textit{Discrete-Time Signal Processing} (3rd. ed.)"
            )
            .scale(0.4)
            .next_to(table_data, DOWN, aligned_edge=LEFT)
        )
        table = Group(table_data, table_ref)
        self.play(
            table.next_to([0, -config.frame_height / 2, 0], DOWN).animate.move_to(
                ORIGIN
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back)
                    .set_color(YELLOW)
                    .scale(1.5)
                    for m in table_data.get_columns()[1][1:]
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back)
                    .set_color(YELLOW)
                    .scale(1.5)
                    for m in table_data.get_columns()[2][1:]
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        notebook_video = VideoMobject(
            "./static/notebook_sneakpeek_edited.mp4", speed=1, loop=False
        ).scale_to_fit_width(config.frame_width * 0.7)
        notebook_border = SurroundingRectangle(notebook_video, color=BLUE)
        notebook = Group(notebook_video, notebook_border).next_to(
            [0, -config.frame_height / 2, 0], DOWN
        )

        self.play(table.animate.shift(UP * 10), notebook.animate.move_to(ORIGIN))

        self.wait(18)

        resources_video = VideoMobject(
            "./static/resources_sneakpeek.mp4", speed=1, loop=False
        ).scale_to_fit_width(config.frame_width * 0.7)
        resources_border = SurroundingRectangle(resources_video, color=BLUE)
        resources = Group(resources_video, resources_border).next_to(
            [0, -config.frame_height / 2, 0], DOWN
        )

        self.play(notebook.animate.shift(UP * 10), resources.animate.move_to(ORIGIN))

        self.remove(notebook)

        self.wait(6)

        self.play(resources.animate.shift(UP * 10))

        self.wait(2)


# if you look through this animation class, before you judge me, I just want
# you to know that it's late, I'm tired, and there's still a ton to do before it posts in the morning
class Hardware(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        self.camera.frame.scale(2)

        N_elem = 6

        channels = Group()
        for _ in range(N_elem):
            lna = BLOCKS.get("amp").copy().rotate(PI)
            filt = BLOCKS.get("bp_filter_generic").copy()
            phase_shifter_rect = Square(lna.width, color=WHITE)
            phase_shifter_label = MathTex(
                r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2
            ).move_to(phase_shifter_rect)
            phase_shifter = Group(phase_shifter_rect, phase_shifter_label)
            attn_rect = Square(lna.width, color=WHITE)
            resistor = (
                SVGMobject("../props/static/resistor.svg")
                .set_fill(WHITE)
                .set_color(WHITE)
                .scale_to_fit_width(lna.width * 0.7)
                .move_to(attn_rect)
            )
            attn_arrow = Arrow(attn_rect.get_corner(DL), attn_rect.get_corner(UR))
            attn = Group(attn_rect, resistor, attn_arrow)

            antenna_tri = (
                Triangle(color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * 1.5)
                .rotate(PI / 3)
                .scale_to_fit_width(lna.width)
            )
            antenna_port = Line(
                antenna_tri.get_bottom(),
                antenna_tri.get_top(),
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
            antenna = Group(antenna_port, antenna_tri).rotate(-PI / 2)
            attn_to_ps = Line(ORIGIN, RIGHT * 3).flip()
            ps_to_filt = attn_to_ps.copy()
            filt_to_lna = attn_to_ps.copy()
            lna_to_antenna = attn_to_ps.copy()
            from_attn = attn_to_ps.copy()

            blocks = Group(
                from_attn,
                attn,
                attn_to_ps,
                phase_shifter,
                ps_to_filt,
                filt,
                filt_to_lna,
                lna,
                lna_to_antenna,
                antenna,
            ).arrange(RIGHT, 0)
            channels.add(blocks)

        channels.arrange(DOWN, MED_LARGE_BUFF)
        # self.add(channels)

        pa_rx = (
            Tex("Phased Array Receiver").scale(2.5).next_to(channels, UP, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        FadeIn(channel[-1]),
                        Create(channel[-2]),
                        FadeIn(channel[-3]),
                        Create(channel[-4]),
                        FadeIn(channel[-5]),
                        Create(channel[-6]),
                        FadeIn(channel[-7]),
                        Create(channel[-8]),
                        FadeIn(channel[-9]),
                        Create(channel[-10]),
                        lag_ratio=0.1,
                    )
                    for channel in channels
                ],
                self.camera.frame.animate.scale(1.3).shift(UP),
                Write(pa_rx),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.shift(RIGHT * 12 + DOWN).scale(0.9),
            FadeOut(pa_rx),
        )

        self.wait(0.5)

        background = SurroundingRectangle(
            channels,
            stroke_color=BACKGROUND_COLOR,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        ).set_z_index(-1)
        self.next_section(skip_animations=skip_animations(True))
        self.add(background)

        ax1s = Group()
        ax2s = Group()
        ax3s = Group()
        ax4s = Group()
        ax5s = Group()
        ax6s = Group()
        for channel in channels:
            ax1 = (
                Axes(
                    x_range=[0, 1, 0.5],
                    y_range=[-1, 1, 1],
                    tips=False,
                    x_length=channel[-2].width,
                    y_length=channel[-1].height,
                )
                .set_opacity(0)
                .flip()
            )
            ax1.shift(channel[-2].get_start() - ax1.c2p(0, 0))
            self.add(ax1)
            ax1s.add(ax1)

            ax2 = (
                Axes(
                    x_range=[0, 1, 0.5],
                    y_range=[-1, 1, 1],
                    tips=False,
                    x_length=channel[-2].width,
                    y_length=channel[-1].height,
                )
                .set_opacity(0)
                .flip()
            )
            ax2.shift(channel[-4].get_start() - ax2.c2p(0, 0))
            self.add(ax2)
            ax2s.add(ax2)

            ax3 = (
                Axes(
                    x_range=[0, 1, 0.5],
                    y_range=[-1, 1, 1],
                    tips=False,
                    x_length=channel[-2].width,
                    y_length=channel[-1].height,
                )
                .set_opacity(0)
                .flip()
            )
            ax3.shift(channel[-6].get_start() - ax3.c2p(0, 0))
            self.add(ax3)
            ax3s.add(ax3)

            ax4 = (
                Axes(
                    x_range=[0, 1, 0.5],
                    y_range=[-1, 1, 1],
                    tips=False,
                    x_length=channel[-2].width,
                    y_length=channel[-1].height,
                )
                .set_opacity(0)
                .flip()
            )
            ax4.shift(channel[-8].get_start() - ax4.c2p(0, 0))
            self.add(ax4)
            ax4s.add(ax4)

            ax5 = (
                Axes(
                    x_range=[0, 1, 0.5],
                    y_range=[-1, 1, 1],
                    tips=False,
                    x_length=channel[-2].width,
                    y_length=channel[-1].height,
                )
                .set_opacity(0)
                .flip()
            )
            ax5.shift(channel[-10].get_start() - ax5.c2p(0, 0))
            self.add(ax5)
            ax5s.add(ax5)

        ax1_x_mins = [VT(0) for _ in channels]
        ax1_x_maxs = [VT(0) for _ in channels]
        ax1_amps = [VT(0.5) for _ in channels]
        ax1_noise_amps = [VT(1) for _ in channels]
        ax1_phases = [VT(idx * PI / 6) for idx, _ in enumerate(channels)]
        ax1_f_noises = [
            interp1d(np.linspace(0, 1, 1000), np.random.normal(0, 0.2, 1000))
            for _ in channels
        ]
        ax2_x_mins = [VT(0) for _ in channels]
        ax2_x_maxs = [VT(0) for _ in channels]
        ax2_amps = [VT(1) for _ in channels]
        ax2_noise_amps = [VT(1) for _ in channels]
        ax2_phases = [VT(idx * PI / 6) for idx, _ in enumerate(channels)]
        ax2_f_noises = [
            interp1d(np.linspace(0, 1, 1000), np.random.normal(0, 0.2, 1000))
            for _ in channels
        ]
        ax3_x_mins = [VT(0) for _ in channels]
        ax3_x_maxs = [VT(0) for _ in channels]
        ax3_amps = [VT(1) for _ in channels]
        ax3_noise_amps = [VT(1) for _ in channels]
        ax3_phases = [VT(idx * PI / 6) for idx, _ in enumerate(channels)]
        ax3_f_noises = [
            interp1d(np.linspace(0, 1, 1000), np.random.normal(0, 0.2, 1000))
            for _ in channels
        ]
        ax4_x_mins = [VT(0) for _ in channels]
        ax4_x_maxs = [VT(0) for _ in channels]
        ax4_amps = [VT(1) for _ in channels]
        ax4_noise_amps = [VT(0) for _ in channels]
        ax4_phases = [VT(idx * PI / 6) for idx, _ in enumerate(channels)]
        ax4_f_noises = [
            interp1d(np.linspace(0, 1, 1000), np.random.normal(0, 0.2, 1000))
            for _ in channels
        ]
        ax5_x_mins = [VT(0) for _ in channels]
        ax5_x_maxs = [VT(0) for _ in channels]
        ax5_amps = [VT(1) for _ in channels]
        ax5_noise_amps = [VT(0) for _ in channels]
        ax5_phases = [VT(idx * PI) for idx, _ in enumerate(channels)]
        ax5_f_noises = [
            interp1d(np.linspace(0, 1, 1000), np.random.normal(0, 0.2, 1000))
            for _ in channels
        ]

        ax1_plots = Group(
            always_redraw(
                lambda: ax1s[0].plot(
                    lambda t: ~ax1_amps[0] * np.sin(2 * PI * 3 * t + ~ax1_phases[0])
                    + ax1_f_noises[0](t) * ~ax1_noise_amps[0],
                    x_range=[~ax1_x_mins[0], ~ax1_x_maxs[0], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax1s[1].plot(
                    lambda t: ~ax1_amps[1] * np.sin(2 * PI * 3 * t + ~ax1_phases[1])
                    + ax1_f_noises[1](t) * ~ax1_noise_amps[1],
                    x_range=[~ax1_x_mins[1], ~ax1_x_maxs[1], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax1s[2].plot(
                    lambda t: ~ax1_amps[2] * np.sin(2 * PI * 3 * t + ~ax1_phases[2])
                    + ax1_f_noises[2](t) * ~ax1_noise_amps[2],
                    x_range=[~ax1_x_mins[2], ~ax1_x_maxs[2], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax1s[3].plot(
                    lambda t: ~ax1_amps[3] * np.sin(2 * PI * 3 * t + ~ax1_phases[3])
                    + ax1_f_noises[3](t) * ~ax1_noise_amps[3],
                    x_range=[~ax1_x_mins[3], ~ax1_x_maxs[3], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax1s[4].plot(
                    lambda t: ~ax1_amps[4] * np.sin(2 * PI * 3 * t + ~ax1_phases[4])
                    + ax1_f_noises[4](t) * ~ax1_noise_amps[4],
                    x_range=[~ax1_x_mins[4], ~ax1_x_maxs[4], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax1s[5].plot(
                    lambda t: ~ax1_amps[5] * np.sin(2 * PI * 3 * t + ~ax1_phases[5])
                    + ax1_f_noises[5](t) * ~ax1_noise_amps[5],
                    x_range=[~ax1_x_mins[5], ~ax1_x_maxs[5], 1 / 100],
                    color=RED,
                )
            ),
        )

        ax2_plots = Group(
            always_redraw(
                lambda: ax2s[0].plot(
                    lambda t: ~ax2_amps[0] * np.sin(2 * PI * 3 * t + ~ax2_phases[0])
                    + ax2_f_noises[0](t) * ~ax2_noise_amps[0],
                    x_range=[~ax2_x_mins[0], ~ax2_x_maxs[0], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax2s[1].plot(
                    lambda t: ~ax2_amps[1] * np.sin(2 * PI * 3 * t + ~ax2_phases[1])
                    + ax2_f_noises[1](t) * ~ax2_noise_amps[1],
                    x_range=[~ax2_x_mins[1], ~ax2_x_maxs[1], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax2s[2].plot(
                    lambda t: ~ax2_amps[2] * np.sin(2 * PI * 3 * t + ~ax2_phases[2])
                    + ax2_f_noises[2](t) * ~ax2_noise_amps[2],
                    x_range=[~ax2_x_mins[2], ~ax2_x_maxs[2], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax2s[3].plot(
                    lambda t: ~ax2_amps[3] * np.sin(2 * PI * 3 * t + ~ax2_phases[3])
                    + ax2_f_noises[3](t) * ~ax2_noise_amps[3],
                    x_range=[~ax2_x_mins[3], ~ax2_x_maxs[3], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax2s[4].plot(
                    lambda t: ~ax2_amps[4] * np.sin(2 * PI * 3 * t + ~ax2_phases[4])
                    + ax2_f_noises[4](t) * ~ax2_noise_amps[4],
                    x_range=[~ax2_x_mins[4], ~ax2_x_maxs[4], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax2s[5].plot(
                    lambda t: ~ax2_amps[5] * np.sin(2 * PI * 3 * t + ~ax2_phases[5])
                    + ax2_f_noises[5](t) * ~ax2_noise_amps[5],
                    x_range=[~ax2_x_mins[5], ~ax2_x_maxs[5], 1 / 100],
                    color=RED,
                )
            ),
        )

        ax3_plots = Group(
            always_redraw(
                lambda: ax3s[0].plot(
                    lambda t: ~ax3_amps[0] * np.sin(2 * PI * 3 * t + ~ax3_phases[0])
                    + ax3_f_noises[0](t) * ~ax3_noise_amps[0],
                    x_range=[~ax3_x_mins[0], ~ax3_x_maxs[0], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax3s[1].plot(
                    lambda t: ~ax3_amps[1] * np.sin(2 * PI * 3 * t + ~ax3_phases[1])
                    + ax3_f_noises[1](t) * ~ax3_noise_amps[1],
                    x_range=[~ax3_x_mins[1], ~ax3_x_maxs[1], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax3s[2].plot(
                    lambda t: ~ax3_amps[2] * np.sin(2 * PI * 3 * t + ~ax3_phases[2])
                    + ax3_f_noises[2](t) * ~ax3_noise_amps[2],
                    x_range=[~ax3_x_mins[2], ~ax3_x_maxs[2], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax3s[3].plot(
                    lambda t: ~ax3_amps[3] * np.sin(2 * PI * 3 * t + ~ax3_phases[3])
                    + ax3_f_noises[3](t) * ~ax3_noise_amps[3],
                    x_range=[~ax3_x_mins[3], ~ax3_x_maxs[3], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax3s[4].plot(
                    lambda t: ~ax3_amps[4] * np.sin(2 * PI * 3 * t + ~ax3_phases[4])
                    + ax3_f_noises[4](t) * ~ax3_noise_amps[4],
                    x_range=[~ax3_x_mins[4], ~ax3_x_maxs[4], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax3s[5].plot(
                    lambda t: ~ax3_amps[5] * np.sin(2 * PI * 3 * t + ~ax3_phases[5])
                    + ax3_f_noises[5](t) * ~ax3_noise_amps[5],
                    x_range=[~ax3_x_mins[5], ~ax3_x_maxs[5], 1 / 100],
                    color=RED,
                )
            ),
        )

        ax4_plots = Group(
            always_redraw(
                lambda: ax4s[0].plot(
                    lambda t: ~ax4_amps[0] * np.sin(2 * PI * 3 * t + ~ax4_phases[0])
                    + ax4_f_noises[0](t) * ~ax4_noise_amps[0],
                    x_range=[~ax4_x_mins[0], ~ax4_x_maxs[0], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax4s[1].plot(
                    lambda t: ~ax4_amps[1] * np.sin(2 * PI * 3 * t + ~ax4_phases[1])
                    + ax4_f_noises[1](t) * ~ax4_noise_amps[1],
                    x_range=[~ax4_x_mins[1], ~ax4_x_maxs[1], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax4s[2].plot(
                    lambda t: ~ax4_amps[2] * np.sin(2 * PI * 3 * t + ~ax4_phases[2])
                    + ax4_f_noises[2](t) * ~ax4_noise_amps[2],
                    x_range=[~ax4_x_mins[2], ~ax4_x_maxs[2], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax4s[3].plot(
                    lambda t: ~ax4_amps[3] * np.sin(2 * PI * 3 * t + ~ax4_phases[3])
                    + ax4_f_noises[3](t) * ~ax4_noise_amps[3],
                    x_range=[~ax4_x_mins[3], ~ax4_x_maxs[3], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax4s[4].plot(
                    lambda t: ~ax4_amps[4] * np.sin(2 * PI * 3 * t + ~ax4_phases[4])
                    + ax4_f_noises[4](t) * ~ax4_noise_amps[4],
                    x_range=[~ax4_x_mins[4], ~ax4_x_maxs[4], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax4s[5].plot(
                    lambda t: ~ax4_amps[5] * np.sin(2 * PI * 3 * t + ~ax4_phases[5])
                    + ax4_f_noises[5](t) * ~ax4_noise_amps[5],
                    x_range=[~ax4_x_mins[5], ~ax4_x_maxs[5], 1 / 100],
                    color=RED,
                )
            ),
        )

        ax5_plots = Group(
            always_redraw(
                lambda: ax5s[0].plot(
                    lambda t: ~ax5_amps[0] * np.sin(2 * PI * 3 * t + ~ax5_phases[0])
                    + ax5_f_noises[0](t) * ~ax5_noise_amps[0],
                    x_range=[~ax5_x_mins[0], ~ax5_x_maxs[0], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax5s[1].plot(
                    lambda t: ~ax5_amps[1] * np.sin(2 * PI * 3 * t + ~ax5_phases[1])
                    + ax5_f_noises[1](t) * ~ax5_noise_amps[1],
                    x_range=[~ax5_x_mins[1], ~ax5_x_maxs[1], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax5s[2].plot(
                    lambda t: ~ax5_amps[2] * np.sin(2 * PI * 3 * t + ~ax5_phases[2])
                    + ax5_f_noises[2](t) * ~ax5_noise_amps[2],
                    x_range=[~ax5_x_mins[2], ~ax5_x_maxs[2], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax5s[3].plot(
                    lambda t: ~ax5_amps[3] * np.sin(2 * PI * 3 * t + ~ax5_phases[3])
                    + ax5_f_noises[3](t) * ~ax5_noise_amps[3],
                    x_range=[~ax5_x_mins[3], ~ax5_x_maxs[3], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax5s[4].plot(
                    lambda t: ~ax5_amps[4] * np.sin(2 * PI * 3 * t + ~ax5_phases[4])
                    + ax5_f_noises[4](t) * ~ax5_noise_amps[4],
                    x_range=[~ax5_x_mins[4], ~ax5_x_maxs[4], 1 / 100],
                    color=RED,
                )
            ),
            always_redraw(
                lambda: ax5s[5].plot(
                    lambda t: ~ax5_amps[5] * np.sin(2 * PI * 3 * t + ~ax5_phases[5])
                    + ax5_f_noises[5](t) * ~ax5_noise_amps[5],
                    x_range=[~ax5_x_mins[5], ~ax5_x_maxs[5], 1 / 100],
                    color=RED,
                )
            ),
        )

        self.add(*ax1_plots)
        self.add(*ax2_plots)
        self.add(*ax3_plots)
        self.add(*ax4_plots)
        self.add(*ax5_plots)

        amp_label = Tex("LNA").next_to(channels[0][-3], UP, MED_SMALL_BUFF)
        filt_label = Tex("Filter").next_to(channels[0][-5], UP, MED_SMALL_BUFF)
        ps_label = Tex(r"Phase\\Shifter").next_to(channels[0][-7], UP, MED_SMALL_BUFF)
        attn_label = Tex(r"Attenuator").next_to(channels[0][-9], UP, MED_SMALL_BUFF)
        # self.add(amp_label, filt_label, ps_label, attn_label)

        wavefront = (
            Circle(1, stroke_width=DEFAULT_STROKE_WIDTH * 2)
            .move_to(self.camera.frame.get_corner(UR) + UP * 5 * RIGHT * 5)
            .set_z_index(-2)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                wavefront.animate.scale(40),
                LaggedStart(*[x_max @ 1 for x_max in ax1_x_maxs], lag_ratio=0.2),
                lag_ratio=0.4,
            ),
            run_time=3,
        )
        self.remove(wavefront)

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.set_x(channels[0][-3].get_x()),
            # *[x_min @ 1 for x_min in ax1_x_mins],
            *[x_max @ 1 for x_max in ax2_x_maxs],
            *[amp @ 1 for amp in ax2_amps],
            FadeIn(amp_label),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.set_x(channels[0][-5].get_x()),
            # *[x_min @ 1 for x_min in ax2_x_mins],
            *[x_max @ 1 for x_max in ax3_x_maxs],
            *[namp @ 0 for namp in ax3_noise_amps],
            FadeIn(filt_label),
            amp_label.animate.set_opacity(0.3),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.set_x(channels[0][-5].get_x()),
            # *[x_min @ 1 for x_min in ax2_x_mins],
            *[x_max @ 1 for x_max in ax4_x_maxs],
            FadeIn(ps_label),
            filt_label.animate.set_opacity(0.3),
        )

        self.wait(0.5)

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)
        r_min = -60
        x_len = config.frame_height * 1.5
        polar_ax = (
            Axes(
                x_range=[r_min, -r_min, r_min / 8],
                y_range=[r_min, -r_min, r_min / 8],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=x_len,
                y_length=x_len,
            ).set_opacity(0)
            # .rotate(PI / 2)
        )
        polar_ax.shift(channels.get_right() - polar_ax.c2p(0, 0))
        self.add(polar_ax)

        ap_polar_amp = VT(1)
        polar_theta_min = VT(-0.01)
        polar_theta_max = VT(0.01)

        X = np.linspace(-N_elem / 2 - 0.05, N_elem / 2 + 0.05, 2**10)
        hann_scalar = VT(0)
        taylor_scalar = VT(0)
        hamming_scalar = VT(0)
        tukey_scalar = VT(0)

        def get_f_window():
            window = np.clip(
                signal.windows.hann(2**10) ** ~hann_scalar
                + signal.windows.taylor(2**10, 5, 30) ** ~taylor_scalar
                + signal.windows.hamming(2**10) ** ~hamming_scalar
                + signal.windows.tukey(2**10, 0.5) ** ~tukey_scalar
                - 3,
                0,
                None,
            )
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        X_step = N_elem / n_elem
        X_weights = np.linspace(
            -N_elem / 2 + X_step / 2, N_elem / 2 - X_step / 2, n_elem
        )

        def get_ap_polar(
            fill_color=GREEN,
            fill_opacity=0,
        ):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                # weights = np.array([~w for w in weight_trackers])
                weights = np.array([get_f_window()(x) for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None) * ~ap_polar_amp
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~polar_theta_min, ~polar_theta_max, 2 * PI / 200],
                    color=TX_COLOR,
                    use_smoothing=False,
                    # stroke_opacity=~af_opacity,
                    fill_opacity=fill_opacity,
                    fill_color=fill_color,
                )
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)

        self.wait(0.5)

        self.play(
            polar_theta_max @ (PI / 2),
            polar_theta_min @ (-PI / 2),
            self.camera.frame.animate.shift(RIGHT * 3),
        )

        self.next_section(skip_animations=skip_animations(False))

        self.wait(0.5)

        self.play(
            steering_angle @ (-8),
            *[phase @ (-idx * PI) for idx, phase in enumerate(ax4_phases)],
            run_time=2,
        )
        self.wait(0.5)

        self.play(
            steering_angle @ (8),
            *[phase @ (idx * PI) for idx, phase in enumerate(ax4_phases)],
            run_time=2,
        )

        self.wait(0.5)

        phased_thumbnail = (
            ImageMobject(
                "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
            .scale_to_fit_width(config.frame_width * 1)
            .shift(LEFT * 30)
        )
        phased_box = SurroundingRectangle(phased_thumbnail)
        phased = Group(phased_thumbnail, phased_box)

        self.play(phased.animate.move_to(self.camera.frame.get_center()))

        self.wait(0.5)

        self.play(phased.animate.shift(RIGHT * 30))
        self.remove(phased)

        self.wait(0.5)

        self.play(
            FadeIn(attn_label),
            ps_label.animate.set_opacity(0.3),
            self.camera.frame.animate.shift(LEFT * 2),
        )

        self.wait(0.5)

        self.play(
            *[x_max @ 1 for x_max in ax5_x_maxs],
            ax5_amps[0] @ 0.3,
            ax5_amps[1] @ 0.6,
            ax5_amps[2] @ 1,
            ax5_amps[3] @ 1,
            ax5_amps[4] @ 0.6,
            ax5_amps[5] @ 0.3,
            hann_scalar @ 1,
            run_time=2,
        )

        self.wait(0.5)

        top_box = DashedVMobject(
            SurroundingRectangle(channels[0][:2], stroke_width=DEFAULT_STROKE_WIDTH * 2)
        )
        bot_box = DashedVMobject(
            SurroundingRectangle(channels[5][:2], stroke_width=DEFAULT_STROKE_WIDTH * 2)
        )

        self.play(Create(top_box), Create(bot_box))

        self.wait(0.5)

        self.play(
            top_box.animate.move_to(channels[1][:2]),
            bot_box.animate.move_to(channels[4][:2]),
        )

        self.wait(0.5)

        self.play(
            top_box.animate.move_to(channels[2][:2]),
            bot_box.animate.move_to(channels[3][:2]),
        )

        self.wait(0.5)

        self.play(FadeOut(top_box, bot_box))

        self.wait(0.5)

        self.play(
            ax5_amps[0] @ 0.5,
            ax5_amps[1] @ 0.8,
            ax5_amps[2] @ 1,
            ax5_amps[3] @ 1,
            ax5_amps[4] @ 0.8,
            ax5_amps[5] @ 0.5,
            hann_scalar @ 0,
            taylor_scalar @ 1,
        )

        self.wait(0.5)

        self.play(
            ax5_amps[0] @ 0.1,
            ax5_amps[1] @ 0.5,
            ax5_amps[2] @ 0.8,
            ax5_amps[3] @ 0.8,
            ax5_amps[4] @ 0.5,
            ax5_amps[5] @ 0.1,
            taylor_scalar @ 0,
            hamming_scalar @ 1,
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * 30))

        self.wait(2)


class Wrapup(Scene):
    def construct(self):
        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")
        notebook_reminder = Tex(
            r"tapering.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        notebook_sc1 = (
            ImageMobject("./static/notebook_sc_1.png")
            .scale_to_fit_width(config.frame_width * 0.6)
            .to_edge(UP, LARGE_BUFF)
        )
        notebook_sc1_box = SurroundingRectangle(notebook_sc1, color=GREEN)
        notebook_sc2 = (
            ImageMobject("./static/notebook_sc_2.png")
            .scale_to_fit_width(config.frame_width * 0.5)
            .to_edge(UP, LARGE_BUFF)
        )
        notebook_sc2_box = SurroundingRectangle(notebook_sc2, color=GREEN)
        notebook_sc3 = (
            ImageMobject("./static/notebook_sc_3.png")
            .scale_to_fit_width(config.frame_width * 0.7)
            .to_edge(UP, LARGE_BUFF)
        )
        notebook_sc3_box = SurroundingRectangle(notebook_sc3, color=GREEN)

        self.play(
            notebook.shift(DOWN * 8).animate.shift(UP * 8),
            Group(notebook_sc1, notebook_sc1_box)
            .shift(RIGHT * 15)
            .animate.shift(LEFT * 15),
        )

        self.wait(0.5)

        self.play(
            Group(notebook_sc1, notebook_sc1_box).animate.shift(LEFT * 15),
            Group(notebook_sc2, notebook_sc2_box)
            .shift(RIGHT * 15)
            .animate.shift(LEFT * 15),
        )

        self.wait(0.5)

        self.play(
            Group(notebook_sc2, notebook_sc2_box).animate.shift(LEFT * 15),
            Group(notebook_sc3, notebook_sc3_box)
            .shift(RIGHT * 15)
            .animate.shift(LEFT * 15),
        )

        self.wait(0.5)

        self.play(
            Group(notebook_sc3, notebook_sc3_box).animate.shift(LEFT * 15),
            notebook.animate.shift(DOWN * 8),
        )

        self.wait(2)


class Announcement(Scene):
    def construct(self):
        announcement = Tex("Announcement!", color=ORANGE).scale(2)

        fmcw_thumbnail = ImageMobject(
            "../01_fmcw/media/images/fmcw/thumbnails/comparison.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        fmcw_box = SurroundingRectangle(fmcw_thumbnail)
        fmcw = Group(fmcw_thumbnail, fmcw_box)
        impl_thumbnail = ImageMobject(
            "../02_fmcw_implementation/media/images/fmcw_implementation/Thumbnail_Option_1.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        impl_box = SurroundingRectangle(impl_thumbnail)
        impl = Group(impl_thumbnail, impl_box)
        snr_thumbnail = ImageMobject(
            "../07_snr_equation/media/images/snr/Thumbnail2_ManimCE_v0.18.1.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        snr_box = SurroundingRectangle(snr_thumbnail)
        snr = Group(snr_thumbnail, snr_box)
        pulsed_thumbnail = ImageMobject(
            "../06_radar_range_equation/media/images/radar_equation/thumbnails/Thumbnail_1.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        pulsed_box = SurroundingRectangle(pulsed_thumbnail)
        pulsed = Group(pulsed_thumbnail, pulsed_box)
        cfar_thumbnail = ImageMobject(
            "../03_cfar/media/images/cfar/thumbnails/Thumbnail_1.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        cfar_box = SurroundingRectangle(cfar_thumbnail)
        cfar = Group(cfar_thumbnail, cfar_box)
        phased_thumbnail = ImageMobject(
            "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        phased_box = SurroundingRectangle(phased_thumbnail)
        phased = Group(phased_thumbnail, phased_box)
        doppler_thumbnail = ImageMobject(
            "../04_fmcw_doppler/media/images/fmcw_doppler/thumbnails/Thumbnail_2.png"
        ).scale_to_fit_width(config.frame_width * 0.2)
        doppler_box = SurroundingRectangle(doppler_thumbnail)
        doppler = Group(doppler_thumbnail, doppler_box)

        self.play(Write(announcement))

        self.wait(0.5)

        self.play(FadeOut(announcement))

        self.wait(0.5)

        thumbnail_group = Group(
            fmcw, doppler, phased, cfar, impl, snr, pulsed
        ).arrange_in_grid(2, 4)
        thumbnail_group[-3:].shift(config.frame_width * 0.1 * RIGHT)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in thumbnail_group], lag_ratio=0.3)
        )

        self.wait(0.5)

        array_nb = ImageMobject("./static/phased_array_nb.png").scale_to_fit_width(
            config.frame_width * 0.4
        )
        doppler_nb = ImageMobject("./static/doppler_nb.png").scale_to_fit_width(
            config.frame_width * 0.4
        )
        nb_group = (
            Group(array_nb, doppler_nb).arrange(RIGHT, MED_LARGE_BUFF).shift(DOWN * 20)
        )

        self.play(Group(thumbnail_group, nb_group).animate.arrange(DOWN))

        self.wait(0.5)

        memberships = Tex("Channel Memberships!", color=ORANGE).scale(2)

        self.wait(0.5)

        tier1 = (
            ImageMobject("./static/tier1.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(LEFT * 10)
        )
        tier2 = (
            ImageMobject("./static/tier2.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(RIGHT * 10)
        )
        tier3 = (
            ImageMobject("./static/tier3.png")
            .scale_to_fit_width(config.frame_width * 0.25)
            .shift(RIGHT * 10)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    thumbnail_group.animate.shift(UP * 10),
                    nb_group.animate.shift(DOWN * 10),
                ),
                Write(memberships),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            memberships.animate.to_edge(UP, LARGE_BUFF),
            tier1.animate.move_to(ORIGIN),
        )

        self.wait(0.5)

        self.play(Group(tier1, tier2).animate.arrange(RIGHT, LARGE_BUFF))

        self.wait(0.5)

        self.play(Group(tier1, tier2, tier3).animate.arrange(RIGHT, LARGE_BUFF))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(0.5)

        osc = ImageMobject("./static/osc_mug.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        kraken = ImageMobject("./static/kraken.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        weather = ImageMobject("./static/weather.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        eqn = ImageMobject("./static/eqn_mug.png").scale_to_fit_width(
            config.frame_width * 0.3
        )
        merch = (
            Group(kraken, osc, eqn, weather)
            .arrange_in_grid(2, 2)
            .scale_to_fit_height(config.frame_height * 0.9)
            .set_y(0)
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in merch], lag_ratio=0.3))

        self.wait(0.5)

        self.play(
            Group(kraken, osc).animate.shift(UP * 8),
            Group(eqn, weather).animate.shift(DOWN * 8),
        )

        self.wait(0.5)

        website = ImageMobject("./static/website.png").scale_to_fit_width(
            config.frame_width * 0.8
        )

        self.play(GrowFromCenter(website))

        self.wait(0.5)

        qr_mem = ImageMobject(
            "../../../media/rf_channel_assets/qr_codes/marshall_bruner_support.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        qr_merch = ImageMobject(
            "../../../media/rf_channel_assets/qr_codes/merchmarshallbruner-1200.png"
        ).scale_to_fit_width(config.frame_width * 0.3)

        Group(qr_mem, qr_merch).arrange(RIGHT, LARGE_BUFF)

        mem = Tex("Memberships").next_to(qr_mem, DOWN, MED_LARGE_BUFF)
        merch = Tex("Merch").next_to(qr_merch, DOWN, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                FadeOut(website),
                AnimationGroup(
                    GrowFromCenter(qr_mem),
                    mem.shift(DOWN * 8).animate.shift(UP * 8),
                ),
                AnimationGroup(
                    GrowFromCenter(qr_merch),
                    merch.shift(DOWN * 8).animate.shift(UP * 8),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

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


class EndScreen(Scene):
    def construct(self):
        stats_title = Tex("Stats for Nerds")
        stats_table = (
            Table(
                [
                    ["Lines of code", "3,620"],
                    ["Script word count", "1,838"],
                    ["Days to make", "31"],
                    ["Git commits", "36"],
                ]
            )
            .scale(0.5)
            .next_to(stats_title, direction=DOWN, buff=MED_LARGE_BUFF)
        )
        for row in stats_table.get_rows():
            row[1].set_color(GREEN)

        stats_group = (
            VGroup(stats_title, stats_table)
            .move_to(ORIGIN)
            .to_edge(RIGHT, buff=LARGE_BUFF)
        )

        thank_you_sabrina = (
            Tex(r"Thank you, Sabrina, for\\editing the whole video :)")
            .next_to(stats_group, DOWN)
            .to_edge(DOWN)
        )

        marshall_bruner = Tex("Marshall Bruner").next_to(
            [-config["frame_width"] / 4, 0, 0], DOWN, MED_LARGE_BUFF
        )

        self.play(
            LaggedStart(
                FadeIn(marshall_bruner, shift=UP),
                AnimationGroup(FadeIn(stats_title, shift=DOWN), FadeIn(stats_table)),
                Create(thank_you_sabrina),
                lag_ratio=0.9,
                run_time=4,
            )
        )

        self.wait(2)


class Thumbnail(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        N_elem = 4
        antennas = Group()
        for n in range(N_elem):
            antenna_port = Line(DOWN * 0.3, UP, color=ORANGE)
            antenna_tri = (
                Triangle(color=ORANGE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange(RIGHT, MED_SMALL_BUFF)

        ax = Axes(
            x_range=[-N_elem / 2, N_elem / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=antennas.width,
            y_length=antennas.height * 1.05,
        ).set_opacity(0)
        ax.shift(antennas.get_bottom() - ax.c2p(0, 0))

        X = np.linspace(-N_elem / 2 - 0.05, N_elem / 2 + 0.05, 2**10)

        def get_f_window():
            window = np.clip(
                signal.windows.hann(2**10) ** ~hann_scalar
                + signal.windows.taylor(2**10, 5, 30) ** ~taylor_scalar
                + signal.windows.hamming(2**10) ** ~hamming_scalar
                + signal.windows.tukey(2**10, 0.5) ** ~tukey_scalar
                - 3,
                0,
                None,
            )
            f_window = interp1d(X, window, fill_value="extrapolate", kind="nearest")
            return f_window

        hann_scalar = VT(0)
        taylor_scalar = VT(0)
        hamming_scalar = VT(0)
        tukey_scalar = VT(0)

        disc = always_redraw(
            lambda: ax.plot(
                lambda x: get_f_window()(x)
                if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
                else 0,
                x_range=[-N_elem // 2 - 1, N_elem // 2 + 1, 1 / 100],
                color=BLUE,
                use_smoothing=False,
            ).set_z_index(1)
        )

        cartoon_group = Group(antennas, ax)

        # self.add(antennas, disc)

        self.play(LaggedStart(*[FadeIn(m) for m in antennas], lag_ratio=0.2))

        self.wait(0.5)

        self.play(Create(disc))

        self.wait(0.5)

        r_min = -60

        x_len = config.frame_height * 0.6
        polar_ax = (
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
        polar_ax.shift(antennas.get_top() - polar_ax.c2p(0, 0))

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2
        n_elem = 17  # Must be odd
        n_elem_full = 51
        weight_trackers = [VT(0) for _ in range(n_elem_full)]
        X_step = N_elem / n_elem
        X_weights = np.linspace(
            -N_elem / 2 + X_step / 2, N_elem / 2 - X_step / 2, n_elem
        )
        for wt in weight_trackers[
            n_elem_full // 2 - n_elem // 2 : n_elem_full // 2 + n_elem // 2
        ]:
            wt @= 1
        theta_min = VT(-0.001)
        theta_max = VT(0.001)
        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        def get_ap_polar(
            theta_min_inp=theta_min,
            theta_max_inp=theta_max,
            polar_ax=polar_ax,
        ):
            def updater():
                u_0 = np.sin(~steering_angle * PI / 180)
                # weights = np.array([~w for w in weight_trackers])
                weights = np.array([get_f_window()(x) for x in X_weights])
                AF = compute_af_1d(weights, d_x, k_0, u, u_0)
                AP = AF
                AP = np.clip(20 * np.log10(np.abs(AP)) - r_min, 0, None)
                # AP /= AP.max()
                f_AP = interp1d(u * PI, AP, fill_value="extrapolate")
                plot = polar_ax.plot_polar_graph(
                    r_func=f_AP,
                    theta_range=[~theta_min_inp, ~theta_max_inp, 1 / 400],
                    color=TX_COLOR,
                    use_smoothing=False,
                ).set_z_index(-2)
                return plot

            return updater

        AF_polar_plot = always_redraw(get_ap_polar())
        self.add(AF_polar_plot)

        self.play(
            LaggedStart(
                Group(cartoon_group, polar_ax).animate.arrange(RIGHT, buff=LARGE_BUFF),
                AnimationGroup(
                    theta_min.animate(run_time=3).set_value(-PI),
                    theta_max.animate(run_time=3).set_value(PI),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                cartoon_group.width * 1.3
            ).move_to(cartoon_group)
        )

        self.wait(0.5)

        disc_left = ax.plot(
            lambda x: 1
            if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
            else 0,
            x_range=[-N_elem // 2 - 0.3, -N_elem // 2 + 0.3, 1 / 100],
            color=YELLOW,
            use_smoothing=False,
        ).set_z_index(2)
        disc_right = ax.plot(
            lambda x: 1
            if x >= (-N_elem / 2 - 0.05) and x <= (N_elem / 2 + 0.05)
            else 0,
            x_range=[N_elem // 2 - 0.3, N_elem // 2 + 0.3, 1 / 100],
            color=YELLOW,
            use_smoothing=False,
        ).set_z_index(2)

        self.play(LaggedStart(Create(disc_left), Create(disc_right), lag_ratio=0.6))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(disc_left),
                Uncreate(disc_right),
                self.camera.frame.animate.restore(),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        shade_color_alpha = VT(0)
        shade = always_redraw(
            lambda: ax.get_area(
                disc,
                x_range=[-N_elem / 2 - 1, N_elem / 2 + 1],
                color=interpolate_color(BACKGROUND_COLOR, YELLOW, ~shade_color_alpha),
                stroke_opacity=0.7,
                stroke_width=0,
                opacity=0.7,
                bounded_graph=ax.plot(
                    lambda x: 2, x_range=[-N_elem / 2 - 1, N_elem / 2 + 1, 1 / 100]
                ),
            )
        )
        self.next_section(skip_animations=skip_animations(True))
        self.add(shade)

        self.play(hann_scalar @ 1, run_time=5)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(hann_scalar @ 0, run_time=1)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        side_lobe_ang = PI * 0.3
        side_lobe_arrow_l = Arrow(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(side_lobe_ang),
                1.2 * -r_min * np.sin(side_lobe_ang),
            ),
            polar_ax.c2p(
                -r_min * 0.4 * np.cos(side_lobe_ang),
                -r_min * 0.4 * np.sin(side_lobe_ang),
            ),
        )
        side_lobe_arrow_r = Arrow(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(-side_lobe_ang),
                1.2 * -r_min * np.sin(-side_lobe_ang),
            ),
            polar_ax.c2p(
                -r_min * 0.4 * np.cos(-side_lobe_ang),
                -r_min * 0.4 * np.sin(-side_lobe_ang),
            ),
        )

        main_lobe_ang = PI / 6
        main_lobe_arc = ArcBetweenPoints(
            polar_ax.c2p(
                1.2 * -r_min * np.cos(-main_lobe_ang),
                1.2 * -r_min * np.sin(-main_lobe_ang),
            ),
            polar_ax.c2p(
                1.2 * -r_min * np.cos(main_lobe_ang),
                1.2 * -r_min * np.sin(main_lobe_ang),
            ),
            angle=PI / 2,
        )
        main_lobe_arrow_l = CurvedArrow(
            main_lobe_arc.get_midpoint(), main_lobe_arc.get_end(), angle=PI / 6
        )
        main_lobe_arrow_r = CurvedArrow(
            main_lobe_arc.get_midpoint(), main_lobe_arc.get_start(), angle=-PI / 6
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    hann_scalar @ 1,
                    GrowArrow(side_lobe_arrow_l),
                    GrowArrow(side_lobe_arrow_r),
                ),
                AnimationGroup(
                    Create(main_lobe_arrow_l),
                    Create(main_lobe_arrow_r),
                ),
                lag_ratio=0.5,
            ),
            run_time=10,
        )

        self.camera.frame.shift(UP * 2 + LEFT / 2).scale(0.85)
        title = (
            Tex("Intro to", " Beamforming")
            .scale(1.5)
            .to_edge(UP, LARGE_BUFF)
            .shift(UP * 1.5)
            .set_x(self.camera.frame.get_x())
        )
        title[1].set_color(ORANGE)
        Group(ax, antennas, disc, shade).shift(UP)
        self.add(title)
        self.camera.frame.scale_to_fit_width(disc.width).move_to(disc)


class AnalogBeamformer(MovingCameraScene):
    def construct(self):
        self.camera.frame.scale(2)

        N_elem = 6

        channels = Group()
        for _ in range(N_elem):
            lna = BLOCKS.get("amp").copy().set_color(WHITE).rotate(PI)
            filt = BLOCKS.get("bp_filter_generic").copy().rotate(-PI / 2)
            phase_shifter_rect = Square(lna.width, color=WHITE)
            phase_shifter_label = (
                MathTex(r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2)
                .move_to(phase_shifter_rect)
                .rotate(-PI / 2)
            )
            phase_shifter = Group(phase_shifter_rect, phase_shifter_label)
            attn_rect = Square(lna.width, color=WHITE)
            resistor = (
                SVGMobject("../props/static/resistor.svg")
                .set_fill(WHITE)
                .set_color(WHITE)
                .scale_to_fit_width(lna.width * 0.7)
                .move_to(attn_rect)
            )
            attn_arrow = Arrow(attn_rect.get_corner(UL), attn_rect.get_corner(DR))
            attn = Group(attn_rect, resistor, attn_arrow)

            antenna_tri = (
                Triangle(color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * 1.5)
                .rotate(PI / 3)
                .scale_to_fit_width(lna.width)
            )
            antenna_port = Line(
                antenna_tri.get_bottom(),
                antenna_tri.get_top(),
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
            antenna = Group(antenna_port, antenna_tri).rotate(-PI / 2)
            attn_to_ps = Line(ORIGIN, RIGHT * 1).flip()
            ps_to_filt = attn_to_ps.copy()
            filt_to_lna = attn_to_ps.copy()
            lna_to_antenna = attn_to_ps.copy()
            from_attn = attn_to_ps.copy()

            blocks = Group(
                from_attn,
                attn,
                attn_to_ps,
                phase_shifter,
                ps_to_filt,
                filt,
                filt_to_lna,
                lna,
                lna_to_antenna,
                antenna,
            ).arrange(RIGHT, 0)
            channels.add(blocks)

        channels.arrange(DOWN, MED_LARGE_BUFF)
        channels.rotate(PI / 2)
        combiner = Rectangle(width=channels.width, height=lna.height).next_to(
            channels, DOWN, 0
        )
        combiner_plus = MathTex("+").scale(3).move_to(combiner)
        to_adc = attn_to_ps.copy().next_to(combiner, DOWN, 0)
        adc = BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(to_adc, DOWN, 0)
        channels.add(combiner, combiner_plus, adc, to_adc)
        self.camera.frame.scale_to_fit_height(channels.height * 1.1).move_to(channels)
        self.add(channels)


class HybridBeamformer(MovingCameraScene):
    def construct(self):
        self.camera.frame.scale(2)

        N_elem = 6

        channels = Group()
        for _ in range(N_elem):
            lna = BLOCKS.get("amp").copy().set_color(WHITE).rotate(PI)
            filt = BLOCKS.get("bp_filter_generic").copy().rotate(-PI / 2)
            phase_shifter_rect = Square(lna.width, color=WHITE)
            phase_shifter_label = (
                MathTex(r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2)
                .move_to(phase_shifter_rect)
                .rotate(-PI / 2)
            )
            phase_shifter = Group(phase_shifter_rect, phase_shifter_label)
            attn_rect = Square(lna.width, color=WHITE)
            resistor = (
                SVGMobject("../props/static/resistor.svg")
                .set_fill(WHITE)
                .set_color(WHITE)
                .scale_to_fit_width(lna.width * 0.7)
                .move_to(attn_rect)
            )
            attn_arrow = Arrow(attn_rect.get_corner(UL), attn_rect.get_corner(DR))
            attn = Group(attn_rect, resistor, attn_arrow)

            antenna_tri = (
                Triangle(color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * 1.5)
                .rotate(PI / 3)
                .scale_to_fit_width(lna.width)
            )
            antenna_port = Line(
                antenna_tri.get_bottom(),
                antenna_tri.get_top(),
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
            antenna = Group(antenna_port, antenna_tri).rotate(-PI / 2)
            attn_to_ps = Line(ORIGIN, RIGHT * 1).flip()
            ps_to_filt = attn_to_ps.copy()
            filt_to_lna = attn_to_ps.copy()
            lna_to_antenna = attn_to_ps.copy()
            from_attn = attn_to_ps.copy()

            blocks = Group(
                from_attn,
                attn,
                attn_to_ps,
                phase_shifter,
                ps_to_filt,
                filt,
                filt_to_lna,
                lna,
                lna_to_antenna,
                antenna,
            ).arrange(RIGHT, 0)
            channels.add(blocks)

        channels.arrange(DOWN, MED_LARGE_BUFF)
        channels.rotate(PI / 2)
        combiner_l = Rectangle(width=channels[:3].width, height=lna.height).next_to(
            channels[:3], DOWN, 0
        )
        combiner_plus_l = MathTex("+").scale(3).move_to(combiner_l)
        combiner_r = Rectangle(width=channels[:3].width, height=lna.height).next_to(
            channels[-3:], DOWN, 0
        )
        combiner_plus_r = MathTex("+").scale(3).move_to(combiner_r)

        from_comb_l = attn_to_ps.copy().next_to(combiner_l, DOWN, 0)
        from_comb_r = attn_to_ps.copy().next_to(combiner_r, DOWN, 0)

        to_adc_l = attn_to_ps.copy().next_to(combiner_l, DOWN, 0)
        adc_l = BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(to_adc_l, DOWN, 0)
        to_adc_r = attn_to_ps.copy().next_to(combiner_r, DOWN, 0)
        adc_r = BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(to_adc_r, DOWN, 0)
        channels.add(
            adc_r,
            to_adc_r,
            adc_l,
            to_adc_l,
            combiner_plus_l,
            combiner_l,
            combiner_r,
            combiner_plus_r,
            from_comb_l,
            from_comb_r,
        )
        self.camera.frame.scale_to_fit_height(channels.height * 1.1).move_to(channels)
        self.add(channels)


class DigitalBeamformer(MovingCameraScene):
    def construct(self):
        self.camera.frame.scale(2)

        N_elem = 6

        channels = Group()
        for _ in range(N_elem):
            lna = BLOCKS.get("amp").copy().set_color(WHITE).rotate(PI)
            filt = BLOCKS.get("bp_filter_generic").copy().rotate(-PI / 2)
            phase_shifter_rect = Square(lna.width, color=WHITE)
            phase_shifter_label = (
                MathTex(r"\Delta \phi", font_size=DEFAULT_FONT_SIZE * 2)
                .move_to(phase_shifter_rect)
                .rotate(-PI / 2)
            )
            phase_shifter = Group(phase_shifter_rect, phase_shifter_label)
            attn_rect = Square(lna.width, color=WHITE)
            resistor = (
                SVGMobject("../props/static/resistor.svg")
                .set_fill(WHITE)
                .set_color(WHITE)
                .scale_to_fit_width(lna.width * 0.7)
                .move_to(attn_rect)
            )
            attn_arrow = Arrow(attn_rect.get_corner(UL), attn_rect.get_corner(DR))
            attn = Group(attn_rect, resistor, attn_arrow)

            antenna_tri = (
                Triangle(color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * 1.5)
                .rotate(PI / 3)
                .scale_to_fit_width(lna.width)
            )
            antenna_port = Line(
                antenna_tri.get_bottom(),
                antenna_tri.get_top(),
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
            antenna = Group(antenna_port, antenna_tri).rotate(-PI / 2)
            attn_to_ps = Line(ORIGIN, RIGHT * 1).flip()
            ps_to_filt = attn_to_ps.copy()
            filt_to_lna = attn_to_ps.copy()
            lna_to_antenna = attn_to_ps.copy()
            from_attn = attn_to_ps.copy()

            blocks = Group(
                from_attn,
                attn,
                attn_to_ps,
                phase_shifter,
                ps_to_filt,
                filt,
                filt_to_lna,
                lna,
                lna_to_antenna,
                antenna,
            ).arrange(RIGHT, 0)
            channels.add(blocks)

        channels.arrange(DOWN, MED_LARGE_BUFF)
        channels.rotate(PI / 2)

        adc_1 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[0][0], DOWN, 0)
        )
        adc_2 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[1][0], DOWN, 0)
        )
        adc_3 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[2][0], DOWN, 0)
        )
        adc_4 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[3][0], DOWN, 0)
        )
        adc_5 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[4][0], DOWN, 0)
        )
        adc_6 = (
            BLOCKS.get("adc").copy().rotate(-PI / 2).next_to(channels[5][0], DOWN, 0)
        )
        channels.add(
            adc_1,
            adc_2,
            adc_3,
            adc_4,
            adc_5,
            adc_6,
        )
        self.camera.frame.scale_to_fit_height(channels.height * 1.1).move_to(channels)
        self.add(channels)
