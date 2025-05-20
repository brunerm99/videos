# aliasing.py

from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
import sys
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift, fft2, ifft


sys.path.insert(0, "..")

from props import WeatherRadarTower, VideoMobject
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True

# TODO: Install maple mono CN
FONT = "Maple Mono CN"


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
            font=FONT,
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

        interp = VT(0)
        bw_vt = VT(PI / 6)

        fft_len = 1024
        stop_time = 3
        fs = 10
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        interp_rect_bw = VT(0)

        def get_x_n():
            t = np.linspace(0, 1, freq.size)

            x_n = np.sin(2 * PI * 2 * t)

            freq_for_rect = np.linspace(-PI, PI, fft_len)
            X_k_rect = np.zeros(freq_for_rect.shape)
            X_k_rect[
                (freq_for_rect > -~bw_vt / 2 / 4) & (freq_for_rect < ~bw_vt / 2 / 4)
            ] = 1
            x_n_rect_bw = np.real(fftshift(ifft(fftshift(X_k_rect))))
            x_n_rect_bw /= x_n_rect_bw.max()

            x_n = (1 - ~interp_rect_bw) * x_n + ~interp_rect_bw * x_n_rect_bw

            f_x_n = interp1d(t, x_n, fill_value="extrapolate")
            return ax.plot(f_x_n, x_range=[0, 1, 1 / 200], color=ORANGE)

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

        f_ax = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).next_to(ax, DOWN, LARGE_BUFF)

        f = VT(3)

        def create_X_k(plot_ax, smoothing=True, n_nyquist=1, scalar=None):
            def updater():
                t = np.arange(0, stop_time, 1 / fs)
                # x_n = np.sum([np.sin(2 * PI * ~f * t) for f in freqs], axis=0)
                x_n = np.sin(2 * PI * ~f * t)

                X_k = ((1 - ~interp) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                    (~interp) * fft(x_n, fft_len) / (t.size / 2)
                )
                X_k_rect = np.zeros(fft_len)
                X_k_rect = np.sum(
                    [
                        np.where(np.abs(freq + n * PI) < ~bw_vt / 2, 1, 0)
                        for n in np.arange(-n_nyquist // 2 + 1, n_nyquist // 2 + 1)
                    ],
                    axis=0,
                )
                X_k = (X_k * (1 - ~interp_rect_bw) + X_k_rect * ~interp_rect_bw) * (
                    1 if scalar is None else ~scalar
                )

                f_X_k = interp1d(
                    freq * n_nyquist,
                    np.real(X_k),
                    fill_value="extrapolate",
                )
                return plot_ax.plot(
                    f_X_k,
                    x_range=[-n_nyquist * fs / 2, n_nyquist * fs / 2, fs / 200],
                    color=BLUE,
                    use_smoothing=smoothing,
                )

            return updater

        f_plot = create_X_k(f_ax)()

        to_freq_arrow = CurvedArrow(
            ax.get_right() + [0.5, 0, 0], f_ax.get_right() + [0.5, 0, 0], angle=-TAU / 4
        )
        fs_label = MathTex(r"f_s", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            to_freq_arrow, RIGHT
        )

        freq_label = Text("Frequency", font=FONT).next_to(f_ax, DOWN)

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

        self.next_section(skip_animations=skip_animations(True))

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

        n_nyquist = 3
        f_ax2 = Axes(
            x_range=[-n_nyquist * fs / 2, n_nyquist * fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6 * n_nyquist,
            y_length=config.frame_height * 0.6,
        ).next_to(ax2, DOWN, LARGE_BUFF)
        f_ax2_rdc = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).next_to(ax2, DOWN, LARGE_BUFF)

        one_f_plot = f_ax2.plot(lambda t: 1)
        one_f_samples = f_ax2.get_vertical_lines_to_graph(
            one_f_plot,
            x_range=[
                -n_nyquist * fs / 2 + fs / 2,
                n_nyquist * fs / 2 - fs / 2,
            ],
            num_lines=n_nyquist,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        one_f_dots = Group(*[Dot(s.get_end(), color=BLUE) for s in one_f_samples])
        freq2_label = Text("Frequency", font=FONT).next_to(f_ax2, DOWN)

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

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        all_group = Group(delta_label, ax, ax2, f_ax2_rdc, freq2_label)

        dots_l = (
            MathTex(r"\cdots").scale(3).next_to(f_ax2.c2p(0, 0.5), LEFT, LARGE_BUFF)
        )
        dots_r = (
            MathTex(r"\cdots").scale(3).next_to(f_ax2.c2p(0, 0.5), RIGHT, LARGE_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                all_group.height * 1.2
            ).move_to(all_group),
            Transform(f_ax2, f_ax2_rdc),
            Uncreate(one_f_dots[0]),
            Uncreate(one_f_dots[-1]),
            Uncreate(one_f_samples[0]),
            Uncreate(one_f_samples[-1]),
            FadeIn(dots_l, dots_r),
        )

        self.wait(0.5)

        time_box = SurroundingRectangle(
            delta_label, ax, ax2, delta_label, corner_radius=0.2, buff=MED_SMALL_BUFF
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

        self.wait(0.5)

        f_ax_soln = Axes(
            x_range=[-n_nyquist * fs / 2, n_nyquist * fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6 * n_nyquist,
            y_length=config.frame_height * 0.6,
        ).next_to(Group(f_ax, f_ax2), DOWN, LARGE_BUFF * 3)

        conv_bez_l = CubicBezier(
            f_ax.get_corner(DL) + [0, -0.1, 0],
            f_ax.get_corner(DL) + [0, -3, 0],
            f_ax_soln.get_top() + [0, 3, 0],
            f_ax_soln.get_top() + [0, 0.1, 0],
        )
        conv_bez_r = CubicBezier(
            f_ax2.get_corner(DR) + [0, -0.1, 0],
            f_ax2.get_corner(DR) + [0, -3, 0],
            f_ax_soln.get_top() + [0, 3, 0],
            f_ax_soln.get_top() + [0, 0.1, 0],
        )

        soln_group = Group(f_ax, f_ax2, f_ax_soln)

        self.play(
            LaggedStart(
                Uncreate(f_box),
                self.camera.frame.animate.scale_to_fit_width(soln_group.width * 1.1)
                .move_to(soln_group)
                .shift(DOWN * 0.7),
                AnimationGroup(Create(conv_bez_l), Create(conv_bez_r)),
                Create(f_ax_soln),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        f_ax_nq2_l = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).set_opacity(0)
        f_ax_nq2_l.shift(f_ax_soln.c2p(-fs, 0) - f_ax_nq2_l.c2p(0, 0))
        f_ax_nq2_r = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).set_opacity(0)
        f_ax_nq2_r.shift(f_ax_soln.c2p(fs, 0) - f_ax_nq2_r.c2p(0, 0))
        f_ax_nq1 = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        ).set_opacity(0)
        self.next_section(skip_animations=skip_animations(True))
        f_ax_nq1.shift(f_ax_soln.c2p(0, 0) - f_ax_nq1.c2p(0, 0))

        f_plot_scalar = VT(1)
        f_plot_nq1 = always_redraw(create_X_k(f_ax_soln, False, 3, f_plot_scalar))
        f_plot_nq2_l = always_redraw(create_X_k(f_ax_nq2_l, False))
        f_plot_nq2_r = always_redraw(create_X_k(f_ax_nq2_r, False))

        # self.play(
        #     Create(f_plot_nq2_l), rate_func=rate_functions.ease_in_sine, run_time=0.5
        # )
        self.play(Create(f_plot_nq1), rate_func=rate_functions.linear, run_time=0.5)
        # self.play(
        #     Create(f_plot_nq2_r), rate_func=rate_functions.ease_out_sine, run_time=0.5
        # )

        self.wait(0.5)

        f_labels = Group(
            *[
                fl.next_to(f_ax_soln.c2p(x, 0), DOWN)
                for fl, x in zip(
                    [
                        MathTex(r"-\frac{3 f_s}{2}"),
                        MathTex(r"-f_s"),
                        MathTex(r"-\frac{f_s}{2}"),
                        MathTex(r"0"),
                        MathTex(r"\frac{f_s}{2}"),
                        MathTex(r"f_s"),
                        MathTex(r"\frac{3 f_s}{2}"),
                    ],
                    [-1.5 * fs, -fs, -0.5 * fs, 0, 0.5 * fs, fs, 1.5 * fs],
                )
            ]
        )

        self.play(
            LaggedStart(
                GrowFromCenter(f_labels[0]),
                GrowFromCenter(f_labels[1]),
                ReplacementTransform(neg_fs_label[0], f_labels[2][0], path_arc=PI / 3),
                GrowFromCenter(f_labels[3]),
                ReplacementTransform(pos_fs_label[0], f_labels[4][0], path_arc=PI / 3),
                GrowFromCenter(f_labels[5]),
                GrowFromCenter(f_labels[6]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        f_ax_nq3_l = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        )
        f_ax_nq3_l.shift(f_ax_soln.c2p(-2 * fs, 0) - f_ax_nq3_l.c2p(0, 0))
        f_ax_nq3_r = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.6,
        )
        f_ax_nq3_r.shift(f_ax_soln.c2p(2 * fs, 0) - f_ax_nq3_r.c2p(0, 0))

        # FIXME: Change these zones to every fs/2
        # FIXME: Change the 0-pi zone to green
        zone3_l = Polygon(
            f_ax_nq3_l.c2p(-fs / 2, 0),
            f_ax_nq3_l.c2p(-fs / 2, 1),
            f_ax_nq3_l.c2p(fs / 2, 1),
            f_ax_nq3_l.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=GREEN,
        )
        zone3_r = Polygon(
            f_ax_nq3_r.c2p(-fs / 2, 0),
            f_ax_nq3_r.c2p(-fs / 2, 1),
            f_ax_nq3_r.c2p(fs / 2, 1),
            f_ax_nq3_r.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=GREEN,
        )
        zone2_l = Polygon(
            f_ax_nq2_l.c2p(-fs / 2, 0),
            f_ax_nq2_l.c2p(-fs / 2, 1),
            f_ax_nq2_l.c2p(fs / 2, 1),
            f_ax_nq2_l.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=PURPLE,
        )
        zone2_r = Polygon(
            f_ax_nq2_r.c2p(-fs / 2, 0),
            f_ax_nq2_r.c2p(-fs / 2, 1),
            f_ax_nq2_r.c2p(fs / 2, 1),
            f_ax_nq2_r.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=PURPLE,
        )
        zone1 = Polygon(
            f_ax_nq1.c2p(-fs / 2, 0),
            f_ax_nq1.c2p(-fs / 2, 1),
            f_ax_nq1.c2p(fs / 2, 1),
            f_ax_nq1.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=ORANGE,
        )
        zone2_l_label = Text("Zone 2", font=FONT).next_to(
            zone2_l.get_top(), DOWN, SMALL_BUFF
        )
        zone2_r_label = Text("Zone 2", font=FONT).next_to(
            zone2_r.get_top(), DOWN, SMALL_BUFF
        )
        zone1_label = Text("Zone 1", font=FONT).next_to(
            zone1.get_top(), DOWN, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                FadeIn(zone3_l, zone3_r),
                FadeIn(zone2_l, zone2_r, zone2_l_label, zone2_r_label),
                FadeIn(zone1, zone1_label),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(Ts_bez_l),
                    Uncreate(Ts_bez_r),
                ),
                FadeOut(
                    zone3_l,
                    zone3_r,
                    zone1_label,
                    zone2_l_label,
                    zone2_r_label,
                ),
                AnimationGroup(
                    Group(
                        f_ax_soln, f_ax_nq1, f_ax_nq2_l, f_ax_nq2_r, f_labels
                    ).animate.shift(DOWN * config.frame_height * 2),
                    Group(zone2_l, zone2_r, zone1)
                    .animate.shift(DOWN * config.frame_height * 2)
                    .set_opacity(0.15),
                    self.camera.frame.animate.shift(DOWN * config.frame_height * 2),
                ),
                lag_ratio=0.4,
            ),
            run_time=4,
        )

        sine_upt = always_redraw(get_x_n)
        self.remove(sine)
        self.add(sine_upt)

        self.wait(0.5)

        ax.save_state()
        self.play(ax.animate.next_to(f_ax_soln, UP, LARGE_BUFF))

        self.wait(0.5)

        self.play(interp_rect_bw @ 1, run_time=3)

        self.wait(0.5)

        self.play(bw_vt @ (~bw_vt * 4), run_time=3)

        self.wait(0.5)

        self.play(bw_vt @ (~bw_vt / 2), run_time=3)

        self.wait(0.5)

        self.play(ax.animate.restore())

        self.wait(0.5)

        bw_label = MathTex("B").next_to(
            f_ax_soln.c2p(~bw_vt * fs / PI / 4, 1), UP, LARGE_BUFF * 2
        )
        fs_label = MathTex("f_s").next_to(
            f_ax_soln.c2p(fs / PI / 3, 1), UP, LARGE_BUFF * 4
        )
        fs_bez_l = CubicBezier(
            fs_label.get_bottom() + [0, -0.1, 0],
            fs_label.get_bottom() + [0, -1, 0],
            f_ax_soln.c2p(0, 1) + [0, 2, 0],
            f_ax_soln.c2p(0, 1) + [0, 0.1, 0],
        )
        fs_bez_r = CubicBezier(
            fs_label.get_bottom() + [0, -0.1, 0],
            fs_label.get_bottom() + [0, -1, 0],
            f_ax_soln.c2p(fs / 2, 1) + [0, 2, 0],
            f_ax_soln.c2p(fs / 2, 1) + [0, 0.1, 0],
        )
        bw_bez_l = CubicBezier(
            bw_label.get_bottom() + [0, -0.1, 0],
            bw_label.get_bottom() + [0, -1.5, 0],
            f_ax_soln.c2p(0, 1) + [0, 1.5, 0],
            f_ax_soln.c2p(0, 1) + [0, 0.1, 0],
        )
        bw_bez_r = CubicBezier(
            bw_label.get_bottom() + [0, -0.1, 0],
            bw_label.get_bottom() + [0, -1.5, 0],
            f_ax_soln.c2p(~bw_vt * fs / PI / 2, 1) + [0, 1.5, 0],
            f_ax_soln.c2p(~bw_vt * fs / PI / 2, 1) + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                GrowFromCenter(bw_label),
                AnimationGroup(Create(bw_bez_l), Create(bw_bez_r)),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(fs_label),
                AnimationGroup(Create(fs_bez_l), Create(fs_bez_r)),
                lag_ratio=0.4,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(f_plot_scalar @ 0.5)

        self.wait(0.5)

        self.play(
            ShrinkToCenter(fs_label),
            ShrinkToCenter(bw_label),
            Uncreate(fs_bez_l),
            Uncreate(fs_bez_r),
            Uncreate(bw_bez_l),
            Uncreate(bw_bez_r),
        )

        self.wait(0.5)

        self.play(bw_vt @ (~bw_vt * 5), run_time=5)

        self.wait(0.5)

        self.play(
            Group(ax, f_ax_soln)
            .animate.arrange(DOWN, LARGE_BUFF * 2)
            .move_to(self.camera.frame)
        )

        self.wait(0.5)

        self.play(interp_rect_bw @ 0, f_plot_scalar @ 1, run_time=3)

        self.wait(2)


class SimpleSignal(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        fft_len = 2**10
        stop_time = 3
        fs = 10
        freq = np.linspace(-fs / 2, fs / 2, fft_len)

        f1 = VT(3)
        f2 = VT(2)
        p1 = VT(1)
        p2 = VT(0)

        plot_width = config.frame_width * 0.9
        plot_height = config.frame_height * 0.4

        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=plot_width,
            y_length=plot_height,
        )

        interp = VT(0)

        def create_X_k(
            plot_ax, smoothing=True, shift=0, xmin=None, xmax=None, color=BLUE
        ):
            def updater():
                freq = np.linspace(-fs / 2 + shift, fs / 2 + shift, fft_len)
                t = np.arange(0, stop_time, 1 / fs)
                x_n = (
                    ~p1 * np.sin(2 * PI * ~f1 * t) + ~p2 * np.sin(2 * PI * ~f2 * t)
                ) / (~p1 + ~p2)

                X_k = fftshift(
                    ((1 - ~interp) * np.abs(fft(x_n, fft_len)) / (t.size / 2))
                    + ((~interp) * fft(x_n, fft_len) / (t.size / 2))
                )

                f_X_k = interp1d(freq, np.real(X_k), fill_value="extrapolate")
                if xmin is None or xmax is None:
                    return plot_ax.plot(
                        f_X_k,
                        x_range=[-fs / 2 + shift, fs / 2 + shift, fs / 200],
                        color=color,
                        use_smoothing=smoothing,
                    )
                return plot_ax.plot(
                    f_X_k,
                    x_range=[~xmin + shift, ~xmax + shift, fs / 200],
                    color=color,
                    use_smoothing=smoothing,
                )

            return updater

        n_nyquist = 3
        f_ax = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.3,
            y_length=plot_height,
            x_axis_config=dict(
                numbers_with_elongated_ticks=np.arange(
                    -n_nyquist * fs / 2, (n_nyquist + 1) * fs / 2, fs / 2
                ),
                longer_tick_multiple=3,
            ),
        ).set_opacity(1)

        def get_f_ax(shift=0):
            def updater():
                newax = Axes(
                    x_range=[-fs / 2, fs / 2, 1],
                    y_range=[0, 1, 0.5],
                    tips=False,
                    x_length=config.frame_width * 0.3,
                    y_length=plot_height,
                ).set_opacity(0)
                newax.shift(f_ax.c2p(shift, 0) - newax.c2p(0, 0))
                return newax

            return updater

        Group(ax, f_ax).arrange(DOWN, MED_LARGE_BUFF)
        ax.shift(UP * 7)
        f_ax.shift(DOWN * 7)

        f_ax_nq1 = always_redraw(get_f_ax(0))
        # f_ax_nq2_l = always_redraw(get_f_ax(-fs))
        # f_ax_nq2_r = always_redraw(get_f_ax(fs))

        sine_plot = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * ~f1 * t),
                x_range=[0, 1, 1 / 200],
                color=ORANGE,
            )
        )

        plot_nq1 = always_redraw(create_X_k(f_ax_nq1, smoothing=True))

        self.add(
            f_ax,
            f_ax_nq1,
            plot_nq1,
            # plot_nq2_l,
            # plot_nq2_r,
            ax,
            sine_plot,
        )

        # self.play(Create(sine_plot))

        self.wait(0.5)

        self.play(f_ax.animate.shift(UP * 7), ax.animate.shift(DOWN * 7))

        self.wait(0.5)

        # fc_label = MathTex(r"f = 3 \text{ Hz}").next_to(
        #     f_ax.c2p(~f1, 0), DOWN, MED_LARGE_BUFF
        # )
        # fc_line = f_ax.get_vertical_lines_to_graph(
        #     plot_nq1, x_range=[~f1, ~f1], num_lines=1
        # )

        # self.camera.frame.save_state()
        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.scale(1.2),
        #         Create(fc_line[0]),
        #         FadeIn(fc_label),
        #         lag_ratio=0.3,
        #     ),
        #     run_time=2,
        # )

        # self.wait(0.5)

        # self.play(
        #     Uncreate(fc_line[0]), FadeOut(fc_label), self.camera.frame.animate.restore()
        # )

        # self.wait(0.5)

        time_label = MathTex(r"f_s = 10 \text{ Hz},\ T = 1 \text{ s}").next_to(ax, UP)
        # time_label = Text("hi")
        time_bez_l = CubicBezier(
            ax.get_corner(UL) + [0, 0.1, 0],
            ax.get_corner(UL) + [0, 1, 0],
            time_label.get_left() + [-2, 0, 0],
            time_label.get_left() + [-0.1, 0, 0],
        )
        time_bez_r = CubicBezier(
            ax.get_corner(UR) + [0, 0.1, 0],
            ax.get_corner(UR) + [0, 1, 0],
            time_label.get_right() + [2, 0, 0],
            time_label.get_right() + [0.1, 0, 0],
        )

        f_label = MathTex(r"f = 3 \text{ Hz}", color=ORANGE).next_to(
            ax.input_to_graph_point(9.5 / 12, sine_plot),
            RIGHT,
            LARGE_BUFF,
        )
        # f_label = Text("hi")
        f_line = CubicBezier(
            ax.input_to_graph_point(9.5 / 12, sine_plot) + [0.1, 0.1, 0],
            ax.input_to_graph_point(9.5 / 12, sine_plot) + [0.5, 0.25, 0],
            f_label.get_left() + [-0.5, 0, 0],
            f_label.get_left() + [-0.1, 0, 0],
            color=ORANGE,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.3).shift(UP),
                Create(f_line),
                FadeIn(f_label),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        num_samples = 10
        samples = ax.get_vertical_lines_to_graph(
            sine_plot,
            x_range=[1 / num_samples / 2, 1 - 1 / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        dots = Group(*[Dot(s.get_end(), color=BLUE) for s in samples])

        self.play(
            AnimationGroup(Create(time_bez_l), Create(time_bez_r)),
            FadeIn(time_label),
            LaggedStart(
                *[LaggedStart(Create(s), Create(dot)) for s, dot in zip(samples, dots)],
                lag_ratio=0.1,
            ),
        )

        self.wait(0.5)

        xmin_pos = VT(3)
        xmax_pos = VT(3)
        xmin_neg = VT(-3)
        xmax_neg = VT(-3)

        highlight_nq2_l_pos = always_redraw(
            create_X_k(
                f_ax_nq1, True, shift=-fs, xmin=xmin_pos, xmax=xmax_pos, color=YELLOW
            )
        )
        highlight_nq2_r_pos = always_redraw(
            create_X_k(
                f_ax_nq1, True, shift=fs, xmin=xmin_pos, xmax=xmax_pos, color=YELLOW
            )
        )
        highlight_nq1_pos = always_redraw(
            create_X_k(
                f_ax_nq1, True, shift=0, xmin=xmin_pos, xmax=xmax_pos, color=YELLOW
            )
        )
        highlight_nq2_l_neg = always_redraw(
            create_X_k(
                f_ax_nq1, True, shift=-fs, xmin=xmin_neg, xmax=xmax_neg, color=RED
            )
        )
        highlight_nq2_r_neg = always_redraw(
            create_X_k(
                f_ax_nq1, True, shift=fs, xmin=xmin_neg, xmax=xmax_neg, color=RED
            )
        )
        highlight_nq1_neg = always_redraw(
            create_X_k(f_ax_nq1, True, shift=0, xmin=xmin_neg, xmax=xmax_neg, color=RED)
        )
        self.add(
            # highlight_nq2_l_pos,
            highlight_nq1_pos,
            # highlight_nq2_r_pos,
            # highlight_nq2_l_neg,
            highlight_nq1_neg,
            # highlight_nq2_r_neg,
        )

        self.play(xmin_pos @ (~xmin_pos - 0.3), xmax_pos @ (~xmin_pos + 0.3))

        self.wait(0.5)

        self.play(xmin_neg @ (~xmin_neg - 0.3), xmax_neg @ (~xmin_neg + 0.3))

        self.wait(0.5)

        self.play(xmin_pos @ 3, xmax_pos @ 3)

        self.wait(0.5)

        no_imag = MathTex(
            r"x = \sin{(2 \pi f t)} \leftarrow \text{no imaginary ($j$ or $i$) component}"
        )
        no_imag_box = SurroundingRectangle(
            no_imag, corner_radius=0.2, fill_opacity=1, fill_color=BACKGROUND_COLOR
        )
        no_imag_group = Group(no_imag_box, no_imag).next_to(
            time_label, UP, MED_LARGE_BUFF
        )

        self.play(no_imag_group.shift(UP * 5).animate.shift(DOWN * 5))

        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"aliasing.ipynb \rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
        )
        notebook_reminder_box = SurroundingRectangle(
            notebook_reminder,
            corner_radius=0.2,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
        )
        notebook = (
            Group(notebook_reminder_box, notebook_reminder)
            .set_y(no_imag_group.get_y())
            .shift(LEFT * 12)
        )

        self.play(
            Group(notebook, no_imag_group)
            .animate.arrange(RIGHT)
            .set_y(no_imag_group.get_y())
        )

        self.wait(0.5)

        self.play(Group(notebook, no_imag_group).animate.shift(UP * 5))

        self.wait(0.5)
        self.remove(notebook, no_imag_group)

        self.play(xmin_pos @ (~xmin_pos - 0.3), xmax_pos @ (~xmin_pos + 0.3))

        self.wait(0.5)

        zone1 = Polygon(
            f_ax.c2p(-fs / 2, 0),
            f_ax.c2p(-fs / 2, 1),
            f_ax.c2p(fs / 2, 1),
            f_ax.c2p(fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=ORANGE,
        )

        zone1_label = Text(
            "1st Nyquist Zone", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5
        ).next_to(zone1, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                Group(
                    ax,
                    time_bez_l,
                    time_bez_r,
                    f_line,
                    f_label,
                    time_label,
                    samples,
                    dots,
                ).animate.shift(UP),
                FadeIn(zone1),
                Write(zone1_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        f_labels = Group(
            *[
                fl.next_to(f_ax.c2p(x, 0), DOWN)
                for fl, x in zip(
                    [
                        MathTex(r"-\frac{3 f_s}{2}"),
                        MathTex(r"-f_s"),
                        MathTex(r"-\frac{f_s}{2}"),
                        MathTex(r"0"),
                        MathTex(r"\frac{f_s}{2}"),
                        MathTex(r"f_s"),
                        MathTex(r"\frac{3 f_s}{2}"),
                    ],
                    [-1.5 * fs, -fs, -0.5 * fs, 0, 0.5 * fs, fs, 1.5 * fs],
                )
            ]
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2),
                FadeIn(f_labels[2]),
                FadeIn(f_labels[4]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        nfs_val = MathTex(r"-5 \text{Hz}").move_to(f_labels[2])
        pfs_val = MathTex(r"5 \text{Hz}").move_to(f_labels[4])

        f_labels[2].save_state()
        f_labels[4].save_state()
        self.play(
            LaggedStart(
                Transform(f_labels[2], nfs_val),
                Transform(f_labels[4], pfs_val),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        f_ax_full = Axes(
            x_range=[-n_nyquist * fs / 2, n_nyquist * fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.3 * n_nyquist,
            y_length=plot_height,
            x_axis_config=dict(
                numbers_with_elongated_ticks=np.arange(
                    -n_nyquist * fs / 2, (n_nyquist + 1) * fs / 2, fs / 2
                ),
                longer_tick_multiple=3,
            ),
        ).set_opacity(1)
        f_ax_full.shift(f_ax.c2p(0, 0) - f_ax_full.c2p(0, 0))

        plot_nq2_l = always_redraw(create_X_k(f_ax_nq1, smoothing=True, shift=-fs))
        plot_nq2_r = always_redraw(create_X_k(f_ax_nq1, smoothing=True, shift=fs))

        zone2_l = Polygon(
            f_ax.c2p(-fs + -fs / 2, 0),
            f_ax.c2p(-fs + -fs / 2, 1),
            f_ax.c2p(-fs + fs / 2, 1),
            f_ax.c2p(-fs + fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=PURPLE,
        )
        zone2_r = Polygon(
            f_ax.c2p(fs + -fs / 2, 0),
            f_ax.c2p(fs + -fs / 2, 1),
            f_ax.c2p(fs + fs / 2, 1),
            f_ax.c2p(fs + fs / 2, 0),
            stroke_opacity=0,
            fill_opacity=0.3,
            fill_color=PURPLE,
        )

        zone2_l_label = Text(
            "2nd Nyquist Zone", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5
        ).next_to(zone2_l, UP, SMALL_BUFF)
        zone2_r_label = Text(
            "2nd Nyquist Zone", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5
        ).next_to(zone2_r, UP, SMALL_BUFF)

        f_ax.save_state()
        self.play(
            Transform(f_ax, f_ax_full),
            Create(plot_nq2_l),
            Create(plot_nq2_r),
            LaggedStart(
                FadeIn(zone2_l),
                FadeIn(zone2_r),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            FadeIn(
                highlight_nq2_l_pos,
                highlight_nq2_r_pos,
                highlight_nq2_l_neg,
                highlight_nq2_r_neg,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(Write(zone2_l_label), Write(zone2_r_label), lag_ratio=0.3)
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(zone1.height * 1.7)
            .move_to(zone1)
            .set_x(f_ax_nq1.c2p(fs / 2, 0)[0]),
            zone1.animate.set_opacity(0.2),
            zone2_l.animate.set_opacity(0.2),
            zone2_r.animate.set_opacity(0.2),
            xmin_neg @ -3,
            xmax_neg @ -3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            FadeOut(
                highlight_nq1_pos,
                highlight_nq2_l_pos,
                highlight_nq2_r_pos,
                highlight_nq2_l_neg,
                highlight_nq2_r_neg,
            )
        )

        self.wait(0.5)

        fnew = fs * 0.75
        self.play(f1 @ (fnew), run_time=12)

        self.wait(0.5)

        nq_boundary = DashedLine(
            f_ax_nq1.c2p(fs / 2, 0),
            f_ax_nq1.c2p(fs / 2, 1),
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )

        self.next_section(skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).shift(DOWN * 0.75),
                f_labels[2].animate.restore().shift(DOWN * 0.5),
                f_labels[4].animate.restore().shift(DOWN * 0.5),
                Create(nq_boundary),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.remove(ax, *dots, *samples, sine_plot)

        axes_group = Group(f_ax_nq1)
        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                axes_group.width * 1.2 * 3
            ).move_to(axes_group)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    plot_nq2_r,
                    plot_nq2_l,
                    nq_boundary,
                    zone2_l,
                    zone2_l_label,
                    zone2_r,
                    zone2_r_label,
                ),
                f_ax.animate.restore(),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(zone1, zone1_label))

        self.wait(2)


class ZoomIn(MovingCameraScene):
    def construct(self):
        fs = 10
        plot_width = config.frame_width * 0.9
        plot_height = config.frame_height * 0.4
        n_nyquist = 3
        f_ax = Axes(
            x_range=[-fs / 2, fs / 2, 1],
            y_range=[0, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.3,
            y_length=plot_height,
            x_axis_config=dict(
                numbers_with_elongated_ticks=np.arange(
                    -n_nyquist * fs / 2, (n_nyquist + 1) * fs / 2, fs / 2
                ),
                longer_tick_multiple=3,
            ),
        ).set_opacity(1)

        self.add(f_ax)
        self.camera.frame.scale_to_fit_width(f_ax.width * 1.2 * 3)


class TexTest(Scene):
    def construct(self):
        time_label = MathTex(r"\frac{3 f_s}{2} a")

        self.add(time_label)
