# aliasing.py

from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
import sys
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift, fft2


sys.path.insert(0, "..")

from props import WeatherRadarTower, VideoMobject
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class ChangingFS(Scene):
    def construct(self):
        fft_len = 1024
        freq = np.linspace(-PI, PI, fft_len)
        stop_time = 3

        fs = VT(10)
        f1 = VT(2)
        f2 = VT(4)
        abs_vt = VT(0)

        ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.7,
        )

        def plot_X_k():
            t = np.arange(0, stop_time, 1 / ~fs)
            freqs = [~f1, ~f2]
            x_n = np.sum([np.sin(2 * PI * f * t) for f in freqs], axis=0)

            X_k = ((1 - ~abs_vt) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                (~abs_vt) * fft(x_n, fft_len) / (t.size / 2)
            )
            f_X_k = interp1d(freq, np.real(fftshift(X_k)), fill_value="extrapolate")
            return ax.plot(f_X_k, x_range=[-PI, PI, PI / 200], color=ORANGE)

        plot = always_redraw(plot_X_k)

        self.add(ax, plot)

        self.play(f1 @ 8, run_time=20)

        self.wait(0.5)

        self.play(abs_vt @ 1, f1 @ 2, run_time=2)

        self.wait(0.5)

        self.play(f1 @ 8, run_time=20)

        self.wait(2)


class SimpleSignal(Scene):
    def construct(self):
        fft_len = 1024
        freq = np.linspace(-PI, PI, fft_len)
        stop_time = 3
        fs = 10

        f1 = VT(4)
        f2 = VT(8)
        abs_vt = VT(0)

        ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.7,
        )

        def plot_X_k():
            t = np.arange(0, stop_time, 1 / fs)
            x_n = np.sin(2 * PI * 3 * t)
            freqs = [~f1, ~f2]
            x_n = np.sum([np.sin(2 * PI * f * t) for f in freqs], axis=0)

            X_k = ((1 - ~abs_vt) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                (~abs_vt) * fft(x_n, fft_len) / (t.size / 2)
            )
            f_X_k = interp1d(freq, np.real(X_k), fill_value="extrapolate")
            return ax.plot(f_X_k, x_range=[-PI, PI, PI / 200], color=ORANGE)

        plot = always_redraw(plot_X_k)

        self.add(ax, plot)

        self.play(abs_vt @ 1, run_time=2)

        self.wait(2)


class SamplingRecap(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        title = Text(
            "Sampling: a recap",
            font="Maple Mono",
            font_size=DEFAULT_FONT_SIZE * 1.5,
            t2c={"Sampling": ORANGE},
            t2s={"a recap": ITALIC},
        )

        self.play(Write(title))

        self.wait(0.5)

        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        )
        sine = ax.plot(
            lambda t: np.sin(2 * PI * 2 * t),
            x_range=[0, 1, 1 / 200],
            color=ORANGE,
        )

        self.play(
            LaggedStart(
                title.animate.shift(UP * 10),
                Create(ax),
                Create(sine),
                lag_ratio=0.3,
            )
        )
        self.remove(title)

        self.wait(0.5)

        num_samples = 10
        samples = ax.get_vertical_lines_to_graph(
            sine,
            x_range=[1 / num_samples / 2, 1 - 1 / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        dots = Group(*[Dot(s.get_end(), color=BLUE) for s in samples])
        funcs = Group(
            *[
                MathTex(f"f(t_{{{idx}}})").next_to(
                    d, UP if ax.point_to_coords(d.get_center())[1] > 0 else DOWN
                )
                for idx, d in enumerate(dots)
            ]
        )
        func_boxes = Group(
            *[
                SurroundingRectangle(
                    func,
                    stroke_opacity=0,
                    fill_opacity=0.5,
                    fill_color=BACKGROUND_COLOR,
                )
                for func in funcs
            ]
        )

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        Create(l), Create(d), FadeIn(func_box, func), lag_ratio=0.3
                    )
                    for l, d, func, func_box in zip(samples, dots, funcs, func_boxes)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        fs_eqn = MathTex(r"f_s = \frac{1}{T_s}").next_to(ax, UP, LARGE_BUFF * 1.5)

        all_group = Group(fs_eqn, ax, funcs)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.2
                ).move_to(all_group),
                FadeIn(fs_eqn),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            fs_eqn[0][:2]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        self.play(fs_eqn[0][-2:].animate.set_color(YELLOW))

        self.wait(0.5)

        Ts_bez_l = CubicBezier(
            fs_eqn[0][-2:].get_bottom() + [0, -0.1, 0],
            fs_eqn[0][-2:].get_bottom() + [0, -1, 0],
            dots[5].get_top() + [0, 1, 0],
            dots[5].get_top() + [0, 0.1, 0],
            color=YELLOW,
        )
        Ts_bez_r = CubicBezier(
            fs_eqn[0][-2:].get_bottom() + [0, -0.1, 0],
            fs_eqn[0][-2:].get_bottom() + [0, -1, 0],
            dots[6].get_top() + [0, 1, 0],
            dots[6].get_top() + [0, 0.1, 0],
            color=YELLOW,
        )

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        box.animate.set_opacity(0), func.animate.set_opacity(0)
                    )
                    for func, box in zip(funcs[5:7], func_boxes[5:7])
                ],
                Create(Ts_bez_l),
                Create(Ts_bez_r),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        fft_len = 1024
        stop_time = 3
        fs = 10
        freq = np.linspace(-fs / 2, fs / 2, fft_len)

        f_ax = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).next_to(ax, DOWN, LARGE_BUFF)

        interp = VT(0)

        freqs = [3]

        def create_X_k():
            t = np.arange(0, stop_time, 1 / fs)
            x_n = np.sin(2 * PI * 3 * t)
            x_n = np.sum([np.sin(2 * PI * f * t) for f in freqs], axis=0)

            X_k = ((1 - ~interp) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                (~interp) * fft(x_n, fft_len) / (t.size / 2)
            )
            f_X_k = interp1d(freq, np.real(X_k), fill_value="extrapolate")
            return f_ax.plot(f_X_k, x_range=[-fs / 2, fs / 2, fs / 200], color=BLUE)

        f_plot = create_X_k()

        to_freq_arrow = CurvedArrow(
            ax.get_right() + [0.5, 0, 0], f_ax.get_right() + [0.5, 0, 0], angle=-TAU / 4
        )
        fs_label = MathTex(r"f_s", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            to_freq_arrow, RIGHT
        )

        freq_label = Text("Frequency", font="Maple Mono").next_to(f_ax, DOWN)

        all_group = Group(ax, f_ax, freq_label)

        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        AnimationGroup(
                            box.animate.set_opacity(0), func.animate.set_opacity(0)
                        )
                        for func, box in zip(funcs, func_boxes)
                    ],
                    lag_ratio=0.1,
                ),
                AnimationGroup(
                    ShrinkToCenter(fs_eqn[0][2:]),
                    Uncreate(Ts_bez_l),
                    Uncreate(Ts_bez_r),
                ),
                ReplacementTransform(fs_eqn[0][:2], fs_label[0], path_arc=PI / 2),
                self.camera.frame.animate.scale_to_fit_height(all_group.height * 1.2)
                .move_to(all_group)
                .shift(RIGHT),
                Create(to_freq_arrow),
                Create(f_ax),
                Create(f_plot),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(Write(freq_label))

        neg_fs_label = MathTex(
            r"-\frac{f_s}{2}", font_size=DEFAULT_FONT_SIZE * 2
        ).next_to(f_ax.c2p(-fs / 2, 0), LEFT)
        pos_fs_label = MathTex(
            r"\frac{f_s}{2}", font_size=DEFAULT_FONT_SIZE * 2
        ).next_to(f_ax.c2p(fs / 2, 0), RIGHT)

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    TransformFromCopy(
                        fs_label[0], neg_fs_label[0][1:3], path_arc=PI / 3
                    ),
                    TransformFromCopy(
                        fs_label[0], pos_fs_label[0][0:2], path_arc=PI / 3
                    ),
                ),
                AnimationGroup(
                    LaggedStart(
                        *[
                            GrowFromCenter(m)
                            for m in [neg_fs_label[0][0], *neg_fs_label[0][3:]]
                        ],
                        lag_ratio=0.1,
                    ),
                    LaggedStart(
                        *[GrowFromCenter(m) for m in pos_fs_label[0][2:]],
                        lag_ratio=0.1,
                    ),
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        f_ax_group = Group(f_ax, f_plot, neg_fs_label, pos_fs_label, freq_label)

        self.play(
            self.camera.frame.animate.restore(),
            FadeOut(fs_label, to_freq_arrow),
            f_ax_group.animate.shift(DOWN * 8),
        )

        self.wait(0.5)

        ax2 = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).next_to(ax, RIGHT, LARGE_BUFF * 2)

        self.play(
            *[m.animate.set_opacity(0.1) for m in dots],
            *[m.animate.set_opacity(0.1) for m in samples],
        )

        self.wait(0.5)

        one_plot = ax2.plot(lambda t: 1)
        one_samples = ax2.get_vertical_lines_to_graph(
            one_plot,
            x_range=[1 / num_samples / 2, 1 - 1 / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        one_dots = Group(*[Dot(s.get_end(), color=BLUE) for s in one_samples])

        times = MathTex(r"\LARGE\times", color=YELLOW).scale(3).move_to(Group(ax, ax2))
        axes_group = Group(ax, ax2)

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    axes_group.width * 1.2
                ).move_to(axes_group),
                TransformFromCopy(ax, ax2),
                GrowFromCenter(times),
                AnimationGroup(
                    LaggedStart(
                        *[TransformFromCopy(d, od) for d, od in zip(dots, one_dots)],
                        lag_ratio=0.2,
                    ),
                    LaggedStart(
                        *[
                            TransformFromCopy(s, os)
                            for s, os in zip(samples, one_samples)
                        ],
                        lag_ratio=0.2,
                    ),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        delta_label = MathTex(r"\delta (t)", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            ax2, UP, LARGE_BUFF
        )

        self.play(self.camera.frame.animate.shift(UP), Write(delta_label))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                Group(ax2, delta_label).width * 2
            ).move_to(Group(ax2, delta_label))
        )

        self.wait(0.5)

        f_ax2 = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).next_to(ax2, DOWN, LARGE_BUFF)

        one_f_plot = f_ax2.plot(lambda t: 1)
        one_f_samples = f_ax2.get_vertical_lines_to_graph(
            one_f_plot,
            x_range=[-fs / 2 + fs / num_samples / 2, fs / 2 - fs / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        one_f_dots = Group(*[Dot(s.get_end(), color=BLUE) for s in one_f_samples])
        freq2_label = Text("Frequency", font="Maple Mono").next_to(f_ax2, DOWN)

        ax2_group = Group(ax2, f_ax2, delta_label, freq2_label)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    ax2_group.height * 1.1
                ).move_to(ax2_group),
                Write(freq2_label),
                Create(f_ax2),
                AnimationGroup(
                    *[Create(s) for s in one_f_samples],
                    *[Create(d) for d in one_f_dots],
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        # TODO: Add notebook reference

        self.wait(0.5)

        all_group = Group(ax, ax2, f_ax2, freq2_label)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                all_group.height * 1.2
            ).move_to(all_group)
        )

        self.wait(0.5)

        time_box = SurroundingRectangle(
            ax, ax2, delta_label, corner_radius=0.2, buff=MED_SMALL_BUFF
        )

        self.play(Create(time_box))

        self.wait(0.5)

        f_box = SurroundingRectangle(
            f_ax_group.copy().shift(UP * 8),
            f_ax2,
            freq2_label,
            corner_radius=0.2,
            buff=MED_SMALL_BUFF,
        )

        conv = (
            MathTex(r"\circledast", color=YELLOW)
            .scale(3)
            .move_to(times)
            .set_y(f_ax2.get_y())
        )

        self.play(
            LaggedStart(
                f_ax_group.animate.shift(UP * 8),
                GrowFromCenter(conv),
                ReplacementTransform(time_box, f_box),
                lag_ratio=0.4,
            )
        )

        self.wait(2)
