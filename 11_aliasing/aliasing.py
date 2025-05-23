# aliasing.py

from itertools import pairwise
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

FONT = "Maple Mono CN"


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class SamplingRecap2(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.6,
        )

        f = 3
        sine_opacity = VT(1)
        sine = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[0, 1, 1 / 200],
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                stroke_opacity=~sine_opacity,
            )
        )
        sine_static = ax.plot(
            lambda t: np.sin(2 * PI * f * t),
            x_range=[0, 1, 1 / 200],
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            stroke_opacity=~sine_opacity,
        )
        amplitude_label = (
            Text("amplitude", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6)
            .rotate(PI / 2)
            .next_to(ax.c2p(0, 0), LEFT)
        )
        time_label = Text("time", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.6).next_to(
            ax.c2p(1, 0)
        )
        ax_group = Group(ax, sine_static, amplitude_label, time_label)
        self.camera.frame.scale_to_fit_width(ax_group.width * 1.2).move_to(ax_group)

        self.play(
            LaggedStart(
                AnimationGroup(FadeIn(ax), Create(sine_static)),
                Write(amplitude_label),
                Write(time_label),
                lag_ratio=0.4,
            )
        )

        self.remove(sine_static)
        self.add(sine)

        self.wait(0.5)

        samples_opacity = VT(1)

        def plot_sine():
            t = np.arange(0, 1 + 1 / ~fs, 1 / ~fs)
            y = np.sin(2 * PI * f * t)
            return ax.plot_line_graph(
                t,
                y,
                line_color=ORANGE,
                add_vertex_dots=False,
                stroke_opacity=~samples_opacity,
            )

        def plot_samples():
            ts = np.arange(0, 1 + 1 / ~fs, 1 / ~fs)
            ys = np.sin(2 * PI * f * ts)
            return VGroup(
                *[
                    DashedLine(
                        ax.c2p(t, 0),
                        ax.c2p(t, y),
                        color=ORANGE,
                        dash_length=DEFAULT_DASH_LENGTH * 3,
                        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                        stroke_opacity=~samples_opacity,
                    )
                    for t, y in zip(ts, ys)
                ]
            )

        def plot_dots():
            ts = np.arange(0, 1 + 1 / ~fs, 1 / ~fs)
            ys = np.sin(2 * PI * f * ts)
            return VGroup(
                *[
                    Dot(color=ORANGE)
                    .set_opacity(~samples_opacity)
                    .scale(1.3)
                    .move_to(ax.c2p(t, y))
                    for t, y in zip(ts, ys)
                ]
            )

        fs = VT(10)
        samples_static = plot_samples()
        dots_static = plot_dots()

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(Create(s), Create(d))
                    for s, d in zip(samples_static, dots_static)
                ],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        antennas = Group()
        for _ in samples_static:
            antenna_port = Line(ORIGIN, UP * 1.3, color=ORANGE)
            antenna_tri = (
                Triangle(color=ORANGE)
                .scale(0.5)
                .rotate(PI / 3)
                .move_to(antenna_port, UP)
            )
            antenna = Group(antenna_port, antenna_tri)
            antennas.add(antenna)
        antennas.arrange(RIGHT, MED_LARGE_BUFF).move_to(self.camera.frame).shift(
            UP * self.camera.frame.height
        )

        ant_periods = Group()
        for idx, ants in enumerate(pairwise(antennas)):
            l = Line(ants[0].get_top(), ants[1].get_top()).shift(
                UP / 2 if idx % 2 == 0 else UP
            )
            ll = Line(l.get_left() + DOWN / 8, l.get_left() + UP / 8)
            lr = Line(l.get_right() + DOWN / 8, l.get_right() + UP / 8)
            ant_periods.add(Group(ll, l, MathTex("d_x").next_to(l, UP, SMALL_BUFF), lr))

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * self.camera.frame.height),
                LaggedStart(
                    *[m.shift(UP).animate.shift(DOWN) for m in antennas], lag_ratio=0.1
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    LaggedStart(*[Create(m) for m in ls], lag_ratio=0.1)
                    for ls in ant_periods
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        pa_thumbnail = (
            ImageMobject(
                "../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
            )
            .scale_to_fit_width(self.camera.frame.width * 0.3)
            .next_to(self.camera.frame.get_corner(UR), DL, MED_LARGE_BUFF)
        )
        pa_thumbnail_box = SurroundingRectangle(pa_thumbnail, buff=0)
        pa_vid = Group(pa_thumbnail, pa_thumbnail_box)

        self.play(
            self.camera.frame.animate.shift(UP * 2),
            pa_vid.shift(UR * 2 + UP * 2).animate.shift(DL * 2),
        )

        self.wait(0.5)

        part1 = (
            Text("Part 1: Time Domain", font=FONT)
            .scale(1.2)
            .move_to(
                self.camera.frame.copy().shift(UP * self.camera.frame.height * 1.5)
            )
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * self.camera.frame.height * 1.5),
                Write(part1),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        self.remove(part1)

        ts_label_long = MathTex(r"T_s [\text{seconds}]").next_to(
            ax.c2p(0.5, 1), UP, LARGE_BUFF
        )
        ts_label = MathTex(r"T_s [\text{s}]").move_to(ts_label_long)

        ts_bez_l = CubicBezier(
            dots_static[4].get_center() + [0, 0.2, 0],
            dots_static[4].get_center() + [0.2, 0.7, 0],
            ts_label.get_bottom() + [-0.2, -0.7, 0],
            ts_label.get_bottom() + [0, -0.1, 0],
        )
        ts_bez_r = CubicBezier(
            dots_static[5].get_center() + [0, 0.2, 0],
            dots_static[5].get_center() + [0, 5, 0],
            ts_label.get_bottom() + [0, -1, 0],
            ts_label.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            self.camera.frame.animate.shift(UP),
            Create(ts_bez_l),
            Create(ts_bez_r),
            LaggedStart(*[GrowFromCenter(m) for m in ts_label_long[0]], lag_ratio=0.1),
        )

        self.wait(0.5)

        self.play(
            ReplacementTransform(ts_label_long[0][:3], ts_label[0][:3]),
            ReplacementTransform(ts_label_long[0][3:-1], ts_label[0][3:-1]),
            ReplacementTransform(ts_label_long[0][-1], ts_label[0][-1]),
        )

        self.wait(0.5)

        self.remove(pa_vid)
        line_to_dx = ArcBetweenPoints(
            ts_label.get_corner(UR) + [0, 0.1, 0],
            ant_periods[5][2].copy().shift(DOWN * 3).get_bottom() + [0.3, -0.3, 0],
        )
        line_to_dx_head = (
            Triangle(fill_color=WHITE, fill_opacity=1, stroke_color=WHITE)
            .rotate(15 * DEGREES)
            .scale(0.15)
            .move_to(line_to_dx.get_end())
        )

        meters = Tex("$[$m$]$").next_to(
            ant_periods[5][2].copy().shift(DOWN * 3), RIGHT, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(
                    UP * self.camera.frame.height + DOWN * 3
                ),
                Create(line_to_dx),
                FadeIn(line_to_dx_head),
                ant_periods[5][2].animate.shift(DOWN * 3),
                FadeIn(meters),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            FadeOut(line_to_dx_head),
            Uncreate(line_to_dx),
            self.camera.frame.animate.restore(),
        )

        self.wait(0.5)

        self.play(
            ts_label.animate(
                rate_func=rate_functions.there_and_back_with_pause
            ).set_color(YELLOW)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Uncreate(ts_bez_l),
            Uncreate(ts_bez_r),
            ts_label.animate.set_opacity(0.3).next_to(
                self.camera.frame.get_corner(UL), DR, MED_LARGE_BUFF
            ),
        )

        self.wait(0.5)

        sine_label = MathTex(r"\sin{(2 \pi (3 \text{ Hz}) t)}").next_to(
            ax, DOWN, MED_LARGE_BUFF
        )

        sine_label_bez_l = CubicBezier(
            ax.get_corner(DL) + [0, -0.1, 0],
            ax.get_corner(DL) + [0, -1, 0],
            sine_label.get_left() + [-1, 0, 0],
            sine_label.get_left() + [-0.1, 0, 0],
        )
        sine_label_bez_r = CubicBezier(
            ax.get_corner(DR) + [0, -0.1, 0],
            ax.get_corner(DR) + [0, -1, 0],
            sine_label.get_right() + [1, 0, 0],
            sine_label.get_right() + [0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(sine_label_bez_l), Create(sine_label_bez_r)),
                LaggedStart(*[GrowFromCenter(m) for m in sine_label[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        second_line = Line(ax.get_corner(UL), ax.get_corner(UR)).shift(UP / 2)
        second_line_l = Line(
            second_line.get_start() + DOWN / 8, second_line.get_start() + UP / 8
        )
        second_line_r = Line(
            second_line.get_end() + DOWN / 8, second_line.get_end() + UP / 8
        )
        one_second = Tex("1 second").next_to(second_line, UP)

        self.play(
            LaggedStart(
                Create(second_line_l),
                Create(second_line),
                FadeIn(one_second),
                Create(second_line_r),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        highlight_xmax = VT(0)
        highlight = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[
                    max(~highlight_xmax - 1 / 3, 0),
                    min(~highlight_xmax, 1),
                    1 / 200,
                ],
                color=YELLOW,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.7,
            )
        )
        self.add(highlight)
        self.next_section(skip_animations=skip_animations(True))

        self.play(highlight_xmax + (1 / 3), run_time=2)

        self.wait(0.5)

        self.play(highlight_xmax + (1 / 3), run_time=2)

        self.wait(0.5)

        self.play(highlight_xmax + (1 / 3), run_time=2)

        self.wait(0.5)

        self.play(highlight_xmax + (1 / 3), run_time=2)

        self.wait(0.5)

        self.play(
            one_second.animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    d.animate(rate_func=rate_functions.there_and_back)
                    .scale(2)
                    .set_color(YELLOW)
                    for d in dots_static
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        ts_label_val = MathTex(
            r"T_s = 100 \text{ ms} \rightarrow f_s = 10 \text{ Hz}"
        ).next_to(ts_label, RIGHT)

        self.play(
            LaggedStart(
                ShrinkToCenter(ts_label[0][2]),
                ShrinkToCenter(ts_label[0][4]),
                ReplacementTransform(ts_label[0][:2], ts_label_val[0][:2]),
                ReplacementTransform(ts_label[0][3], ts_label_val[0][7]),
                *[GrowFromCenter(m) for m in ts_label_val[0][2:7]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    FadeOut(m)
                    for m in [second_line_l, second_line, one_second, second_line_r]
                ],
                *[GrowFromCenter(m) for m in ts_label_val[0][7:]],
                lag_ratio=0.15,
            )
        )
        self.play(ts_label_val.animate.set_x(self.camera.frame.get_x()))

        self.wait(0.5)

        samples = always_redraw(plot_samples)
        dots = always_redraw(plot_dots)

        sampled_sine = always_redraw(plot_sine)

        self.play(Create(sampled_sine), sine_opacity @ 0.2)

        self.wait(0.5)

        # samples = always_redraw(
        #     lambda: ax.get_vertical_lines_to_graph(
        #         sine_static,
        #         x_range=[0, 1],
        #         num_lines=int(~fs),
        #         color=ORANGE,
        #         stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        #         line_config=dict(dash_length=DEFAULT_DASH_LENGTH * 3),
        #     )
        # )
        # dots = always_redraw(
        #     lambda: VGroup(
        #         *[
        #             Dot(color=ORANGE)
        #             .scale(1.3)
        #             .move_to(ax.input_to_graph_point(x, sine_static))
        #             for x in np.linspace(0, 1, int(~fs))
        #         ]
        #     )
        # )
        # self.next_section(skip_animations=skip_animations(False))
        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)
        self.remove(*dots_static, *samples_static)
        self.add(samples, dots)

        ts_label_lshift = VT(0)
        ts_label_updater = always_redraw(
            lambda: MathTex(
                f"T_s = {int(1000 / ~fs)} \\text{{ ms}} \\rightarrow f_s = {int(~fs)} \\text{{ Hz}}"
            )
            .move_to(ts_label_val, LEFT)
            .shift(LEFT * ~ts_label_lshift)
        )
        self.remove(ts_label_val)
        self.add(ts_label_updater)

        self.play(fs @ 25, run_time=12)

        self.wait(0.5)

        self.play(fs @ 5, run_time=12)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        f_wrong = 2
        wrong_xmax = VT(0)
        wrong_sine = always_redraw(
            lambda: ax.plot(
                lambda t: -np.sin(2 * PI * f_wrong * t),
                x_range=[0, min(~wrong_xmax, 1), 1 / 200],
                color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
        )
        self.add(wrong_sine)

        self.play(wrong_xmax @ (1 / f_wrong))

        self.wait(0.5)

        self.play(wrong_xmax @ 1)

        self.wait(0.5)

        legend = (
            Group(
                Group(
                    Line(
                        ORIGIN,
                        RIGHT,
                        color=BLUE,
                        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                    ),
                    Text("Original", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5),
                ).arrange(RIGHT),
                Group(
                    Line(
                        ORIGIN,
                        RIGHT,
                        color=RED,
                        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                    ),
                    Text("Reconstructed", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5),
                ).arrange(RIGHT),
            )
            .arrange(DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)
            .next_to(self.camera.frame.get_corner(UR), DL, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                samples_opacity @ 0.2,
                sine_opacity @ 1,
                ts_label_lshift @ 3,
                legend[0].shift(RIGHT * 5).animate.shift(LEFT * 5),
                legend[1].shift(RIGHT * 5).animate.shift(LEFT * 5),
                lag_ratio=0.3,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        nyquist_label = Text("Shannon-Nyquist Criterion:", font=FONT)
        nyquist_cri = MathTex(r"f_s \ge 2 f_{\text{max}}")
        nyquist = (
            Group(nyquist_label, nyquist_cri)
            .arrange(RIGHT)
            .next_to(ts_label_updater, UP, LARGE_BUFF)
            .shift(RIGHT * ~ts_label_lshift)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ts_label_lshift @ 0,
                    legend[0].animate.shift(RIGHT * 10),
                    legend[1].animate.shift(RIGHT * 10),
                    self.camera.frame.animate.scale(1.3).shift(UP),
                    samples_opacity @ 1,
                    FadeOut(wrong_sine),
                ),
                Write(nyquist_label),
                LaggedStart(*[GrowFromCenter(m) for m in nyquist_cri], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(fs @ (f * 2 + 1), run_time=3)

        self.wait(0.5)

        f_right = 3
        right_xmax = VT(0)
        right_sine = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f_right * t),
                x_range=[0, min(~right_xmax, 1), 1 / 200],
                color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
        )
        self.add(right_sine)

        self.play(
            legend[0].animate.shift(LEFT * 10),
            legend[1].animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        self.play(right_xmax + (1 / f_right))

        self.wait(0.5)

        self.play(right_xmax + (1 / f_right))

        self.wait(0.5)

        self.play(right_xmax + (1 / f_right))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        ant_periods[5][2].shift(UP * 3)
        self.remove(meters)

        self.play(
            self.camera.frame.animate.scale(1 / 1.3).shift(
                UP * self.camera.frame.height / 1.3 * 1.1
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m[2]
                    .animate(rate_func=rate_functions.there_and_back)
                    .shift(UP / 3)
                    .set_color(YELLOW)
                    for m in ant_periods
                ],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        ant_periods_2 = Group()
        for idx, ants in enumerate(
            pairwise(antennas.copy().arrange(RIGHT, LARGE_BUFF).move_to(antennas))
        ):
            l = Line(ants[0].get_top(), ants[1].get_top()).shift(
                UP / 2 if idx % 2 == 0 else UP
            )
            ll = Line(l.get_left() + DOWN / 8, l.get_left() + UP / 8)
            lr = Line(l.get_right() + DOWN / 8, l.get_right() + UP / 8)
            ant_periods_2.add(
                Group(ll, l, MathTex("d_x").next_to(l, UP, SMALL_BUFF), lr)
            )

        dx_gt = (
            MathTex(r"d_x > \frac{\lambda}{2}")
            .scale(1.6)
            .next_to(antennas, UP, LARGE_BUFF * 2)
        )
        arrow = MathTex(r"\rightarrow").scale(1.6).move_to(dx_gt).shift(RIGHT * 12)
        grating_lobes = (
            Text("Grating Lobes", font=FONT).move_to(dx_gt).shift(RIGHT * 12)
        )

        self.play(
            antennas.animate.arrange(RIGHT, LARGE_BUFF).move_to(antennas),
            *[
                AnimationGroup(*[Transform(a, b) for a, b in zip(ap, ap2)])
                for ap, ap2 in zip(ant_periods, ant_periods_2)
            ],
            LaggedStart(*[GrowFromCenter(m) for m in dx_gt[0]], lag_ratio=0.1),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            Group(dx_gt, arrow, grating_lobes).animate.arrange(RIGHT).move_to(dx_gt)
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        part2 = (
            Text("Part 2: Frequency Spectrum", font=FONT)
            .scale(1.2)
            .move_to(
                self.camera.frame.copy().shift(DOWN * self.camera.frame.height * 2)
            )
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * self.camera.frame.height * 2),
                Write(part2),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(part2))

        self.wait(2)


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
                f1 @ (fs * 0.25),
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
        self.next_section(skip_animations=skip_animations(True))
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

        f_labels = Group(
            *[
                fl.next_to(f_ax.c2p(x, 0), DOWN).shift(DOWN / 2)
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

        f1 = VT(3)
        f2 = VT(2)
        p1 = VT(1)
        p2 = VT(0)
        fft_len = 2**10
        stop_time = 3
        interp = VT(0)

        def create_X_k(
            plot_ax,
            smoothing=True,
            shift=0,
            xmin=None,
            xmax=None,
            colors=[BLUE, BLUE, BLUE],
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

                color = colors[0]
                for i in range(len(colors)):
                    if ~f1 > i * fs / 2:
                        color = colors[i]

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
                    x_range=[xmin + shift, xmax + shift, fs / 200],
                    color=color,
                    use_smoothing=smoothing,
                )

            return updater

        def create_X_k_xlim(
            plot_ax, smoothing=True, shift=0, colors=[BLUE, BLUE, BLUE], neg=False
        ):
            def updater():
                if ~f1 % fs > fs / 2:
                    xmin_new = (fs / 2) - (~f1 % (fs / 2)) - 0.3
                    xmax_new = (fs / 2) - (~f1 % (fs / 2)) + 0.3
                else:
                    xmin_new = (~f1 % (fs / 2)) - 0.3
                    xmax_new = (~f1 % (fs / 2)) + 0.3
                if neg:
                    xmin_new, xmax_new = xmax_new, xmin_new
                    xmin_new *= -1
                    xmax_new *= -1
                return create_X_k(
                    plot_ax,
                    smoothing=smoothing,
                    shift=shift,
                    colors=colors,
                    xmin=xmin_new,
                    xmax=xmax_new,
                )()

            return updater

        fnew = fs * 0.25
        f1 @= fnew
        plot_nq1 = always_redraw(create_X_k(f_ax, smoothing=True, shift=0))
        xmin_pos = VT(~f1 - 0.3)
        xmax_pos = VT(~f1 + 0.3)
        highlight_nq1_neg = always_redraw(
            create_X_k_xlim(
                f_ax, smoothing=True, shift=0, colors=[RED, ORANGE, GREEN], neg=True
            )
        )
        highlight_nq1_pos = always_redraw(
            create_X_k_xlim(
                f_ax, smoothing=True, shift=0, colors=[YELLOW, GREEN, ORANGE], neg=False
            )
        )

        self.add(f_ax, plot_nq1, f_labels[2], f_labels[4])
        self.camera.frame.scale_to_fit_width(f_ax.width * 1.2 * 3)

        f_tracker = always_redraw(
            lambda: MathTex(f"f_1 = {~f1:.2f} \\text{{ Hz}}")
            .next_to(f_ax, UP, LARGE_BUFF, LEFT)
            .shift(LEFT)
        )

        def create_f_line():
            cb = CubicBezier(
                f_tracker.get_right() + [0.1, 0, 0],
                f_tracker.get_right() + [1, 0, 0],
                f_ax.c2p(~f1, 1) + [0, 1, 0],
                f_ax.c2p(~f1, 1) + [0, 0.3, 0],
            )
            tri = (
                Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
                .scale(0.3)
                .rotate(PI)
                .move_to(cb.get_end())
            )
            return cb

        def create_f_tri():
            cb = CubicBezier(
                f_tracker.get_right() + [0.1, 0, 0],
                f_tracker.get_right() + [1, 0, 0],
                f_ax.c2p(~f1, 1) + [0, 1, 0],
                f_ax.c2p(~f1, 1) + [0, 0.3, 0],
            )
            tri = (
                Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
                .scale(0.1)
                .rotate(PI)
                .next_to(cb.get_end(), DOWN, 0)
            )
            return tri

        f_line = always_redraw(create_f_line)
        f_tri = always_redraw(create_f_tri)

        self.play(
            FadeIn(f_tracker),
            Create(f_line),
            FadeIn(f_tri),
            Create(highlight_nq1_pos),
            Create(highlight_nq1_neg),
        )

        self.wait(0.5)

        self.play(f1 @ (fs * 0.75), run_time=12)

        self.wait(0.5)

        self.play(f1 @ (fs / 2))

        self.wait(0.5)

        f_zone2_relationship = MathTex(
            r"f_{\text{real}} \uparrow \hspace{6pt} \Rightarrow \hspace{4pt} f_{\text{apparent}} \downarrow"
        ).next_to(f_ax, DOWN, LARGE_BUFF * 2)
        f_zone2_relationship[0][5].set_color(GREEN)
        f_zone2_relationship[0][-1].set_color(RED)

        for_zone2 = (
            Tex(r"While $f$ in 2$^\text{nd}$ Nyquist Zone:")
            .next_to(self.camera.frame.get_left(), LEFT)
            .set_y(f_zone2_relationship.get_y())
        )

        all_group = Group(f_ax, f_line, f_tri, f_tracker, f_zone2_relationship)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(all_group.height * 1.2)
                .move_to(all_group)
                .set_x(0),
                Write(f_zone2_relationship),
                lag_ratio=0.3,
            ),
            f1.animate(run_time=12).set_value(fs * 0.75),
        )

        self.wait(0.5)

        self.play(
            Group(for_zone2, f_zone2_relationship)
            .animate.arrange(RIGHT, MED_SMALL_BUFF)
            .set_y(f_zone2_relationship.get_y())
        )

        self.wait(0.5)

        zone2_f = MathTex(r"f_{\text{apparent}} = f_s - f_{\text{real}}").next_to(
            Group(for_zone2, f_zone2_relationship), DOWN, LARGE_BUFF
        )

        all_group.add(zone2_f)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(all_group.height * 1.2)
            .move_to(all_group)
            .set_x(0),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(
                    f_zone2_relationship[0][-10:-1], zone2_f[0][:9], path_arc=PI / 2
                ),
                GrowFromCenter(zone2_f[0][9]),
                TransformFromCopy(
                    f_labels[4][0][:2], zone2_f[0][10:12], path_arc=PI / 2
                ),
                GrowFromCenter(zone2_f[0][-6]),
                TransformFromCopy(
                    f_zone2_relationship[0][:5], zone2_f[0][-5:], path_arc=-PI / 2
                ),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        for_zone2_box = SurroundingRectangle(for_zone2, corner_radius=0.2)
        zone2_f_box = SurroundingRectangle(zone2_f, corner_radius=0.2)
        zone2_neg_box = SurroundingRectangle(
            Polygon(
                f_ax.c2p(-fs / 2, 0),
                f_ax.c2p(-fs / 2, 1),
                f_ax.c2p(-fs, 1),
                f_ax.c2p(-fs, 0),
            ),
            corner_radius=0.2,
            buff=0,
        )
        zone2_pos_box = SurroundingRectangle(
            Polygon(
                f_ax.c2p(fs / 2, 0),
                f_ax.c2p(fs / 2, 1),
                f_ax.c2p(fs, 1),
                f_ax.c2p(fs, 0),
            ),
            corner_radius=0.2,
            buff=0,
        )

        zone2_neg_label = (
            VGroup(
                Text("Nyquist", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("Zone 2", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("(-)", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
            )
            .arrange(DOWN)
            .move_to(zone2_neg_box)
        )
        zone2_pos_label = (
            VGroup(
                Text("Nyquist", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("Zone 2", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("(+)", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
            )
            .arrange(DOWN)
            .move_to(zone2_pos_box)
        )

        self.play(
            LaggedStart(
                Create(zone2_f_box),
                AnimationGroup(
                    Create(for_zone2_box),
                    Create(zone2_pos_box),
                    Create(zone2_neg_box),
                    *[Write(m) for m in zone2_neg_label],
                    *[Write(m) for m in zone2_pos_label],
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        zone3_neg_box = SurroundingRectangle(
            Polygon(
                f_ax.c2p(-fs, 0),
                f_ax.c2p(-fs, 1),
                f_ax.c2p(-3 * fs / 2, 1),
                f_ax.c2p(-3 * fs / 2, 0),
            ),
            corner_radius=0.2,
            buff=0,
        )
        zone3_pos_box = SurroundingRectangle(
            Polygon(
                f_ax.c2p(fs, 0),
                f_ax.c2p(fs, 1),
                f_ax.c2p(3 * fs / 2, 1),
                f_ax.c2p(3 * fs / 2, 0),
            ),
            corner_radius=0.2,
            buff=0,
        )
        zone3_neg_label = (
            VGroup(
                Text("Nyquist", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("Zone 3", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("(-)", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
            )
            .arrange(DOWN)
            .move_to(zone3_neg_box)
        )
        zone3_pos_label = (
            VGroup(
                Text("Nyquist", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("Zone 3", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
                Text("(+)", font_size=DEFAULT_FONT_SIZE * 0.5, font=FONT),
            )
            .arrange(DOWN)
            .move_to(zone3_pos_box)
        )

        self.play(
            Transform(zone2_neg_box, zone3_neg_box),
            Transform(zone2_pos_box, zone3_pos_box),
            Transform(zone2_neg_label[0], zone3_neg_label[0]),
            Transform(zone2_neg_label[1], zone3_neg_label[1]),
            Transform(zone2_neg_label[2], zone3_neg_label[2]),
            Transform(zone2_pos_label[0], zone3_pos_label[0]),
            Transform(zone2_pos_label[1], zone3_pos_label[1]),
            Transform(zone2_pos_label[2], zone3_pos_label[2]),
        )

        self.play(f1 @ (fs * 1.25), run_time=12)

        self.wait(0.5)

        f_zone3_relationship = MathTex(
            r"f_{\text{real}} \uparrow \hspace{6pt} \Rightarrow \hspace{4pt} f_{\text{apparent}} \uparrow"
        ).next_to(f_ax, DOWN, LARGE_BUFF * 2)
        f_zone3_relationship[0][5].set_color(GREEN)
        f_zone3_relationship[0][-1].set_color(GREEN)
        for_zone3 = (
            Tex(r"While $f$ in 3$^\text{rd}$ Nyquist Zone:")
            .next_to(self.camera.frame.get_left(), LEFT)
            .set_y(f_zone3_relationship.get_y())
        )
        Group(for_zone3, f_zone3_relationship).arrange(RIGHT, MED_SMALL_BUFF)
        zone3_f = MathTex(r"f_{\text{apparent}} = f_s + f_{\text{real}}").next_to(
            Group(for_zone3, f_zone3_relationship), DOWN, LARGE_BUFF
        )
        for_zone3_box = SurroundingRectangle(for_zone3, corner_radius=0.2)
        zone3_f_box = SurroundingRectangle(zone3_f, corner_radius=0.2)

        zone2_group = Group(
            f_zone2_relationship, for_zone2, zone2_f, for_zone2_box, zone2_f_box
        )
        zone3_group = (
            Group(f_zone3_relationship, for_zone3, zone3_f, for_zone3_box, zone3_f_box)
            .move_to(zone2_group)
            .shift(DOWN * config.frame_height)
        )
        zone2_loc = zone2_group.get_center()
        zone3_loc = zone3_group.get_center()

        self.play(
            LaggedStart(
                zone2_group.animate.move_to(zone3_loc),
                zone3_group.animate.move_to(zone2_loc),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            zone2_group.animate.scale_to_fit_width(self.camera.frame.width * 0.45)
            .next_to(self.camera.frame.get_corner(DL), UR)
            .shift(UP),
            zone3_group.animate.scale_to_fit_width(self.camera.frame.width * 0.45)
            .next_to(self.camera.frame.get_corner(DR), UL)
            .shift(UP),
        )
        self.play(
            FadeOut(
                zone2_f_box,
                zone3_f_box,
                for_zone2_box,
                for_zone3_box,
                zone2_neg_label,
                zone2_pos_label,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        zone2_pos_box.add_updater(lambda m: m.set_x(f_ax.c2p(~f1, 0)[0]))
        zone2_neg_box.add_updater(lambda m: m.set_x(f_ax.c2p(-~f1, 0)[0]))

        for_even = (
            Tex(r"While $f$ in \textit{Even} Nyquist Zones:")
            .scale_to_fit_width(for_zone2.width)
            .move_to(for_zone2, LEFT)
        )
        for_odd = (
            Tex(r"While $f$ in \textit{Odd} Nyquist Zones:")
            .scale_to_fit_width(for_zone2.width)
            .move_to(for_zone3, LEFT)
        )

        even_zone_f = (
            MathTex(r"f_{\text{apparent}} = N f_s - f_{\text{real}}")
            .scale_to_fit_width(zone2_f.width * 1.08)
            .move_to(zone2_f, LEFT)
        )
        odd_zone_f = (
            MathTex(r"f_{\text{apparent}} = N f_s + f_{\text{real}}")
            .scale_to_fit_width(zone2_f.width * 1.08)
            .move_to(zone3_f, LEFT)
        )

        self.play(
            f1.animate(run_time=20).set_value(fs * 4.25),
            ReplacementTransform(for_zone2[0][:8], for_even[0][:8]),
            ReplacementTransform(for_zone3[0][:8], for_odd[0][:8]),
            ReplacementTransform(for_zone2[0][8:11], for_even[0][8:12]),
            ReplacementTransform(for_zone3[0][8:11], for_odd[0][8:11]),
            ReplacementTransform(for_zone2[0][11:], for_even[0][12:]),
            ReplacementTransform(for_zone3[0][11:], for_odd[0][11:]),
            ReplacementTransform(zone3_f[0][:10], odd_zone_f[0][:10]),
            GrowFromCenter(odd_zone_f[0][10]),
            ReplacementTransform(zone3_f[0][10:], odd_zone_f[0][11:]),
            ReplacementTransform(zone2_f[0][:10], even_zone_f[0][:10]),
            GrowFromCenter(even_zone_f[0][10]),
            ReplacementTransform(zone2_f[0][10:], even_zone_f[0][11:]),
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                zone2_group,
                zone3_group,
                f_line,
                f_tri,
                f_tracker,
                even_zone_f,
                odd_zone_f,
                for_even,
                for_odd,
            ),
            self.camera.frame.animate.scale_to_fit_height(f_ax.height * 2).move_to(
                f_ax.c2p(fs / 4, 1)
            ),
        )

        self.wait(2)


class Peak(MovingCameraScene):
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

        f1 = VT(3)
        f2 = VT(2)
        p1 = VT(1)
        p2 = VT(0)
        fft_len = 2**10
        stop_time = 3
        interp = VT(0)

        def create_X_k(
            plot_ax,
            smoothing=True,
            shift=0,
            xmin=None,
            xmax=None,
            colors=[BLUE, BLUE, BLUE],
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

                color = colors[0]
                for i in range(len(colors)):
                    if ~f1 > i * fs / 2:
                        color = colors[i]

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
                    x_range=[xmin + shift, xmax + shift, fs / 200],
                    color=color,
                    use_smoothing=smoothing,
                )

            return updater

        f1 @= fs * 0.25
        plot_nq1 = always_redraw(create_X_k(f_ax, smoothing=True, shift=0))

        self.camera.frame.scale_to_fit_height(f_ax.height * 2).move_to(
            f_ax.c2p(fs / 4, 1)
        )
        self.add(f_ax, plot_nq1)

        self.wait(0.5)

        f1_label = (
            MathTex(r"f_1 = 3 \text{ Hz}")
            .next_to(f_ax.c2p(0, 1), UP, MED_LARGE_BUFF)
            .shift(LEFT)
        )

        def create_f_line():
            cb = CubicBezier(
                f1_label.get_right() + [0.1, 0, 0],
                f1_label.get_right() + [1, 0, 0],
                f_ax.c2p(~f1, 1) + [0, 1, 0],
                f_ax.c2p(~f1, 1) + [0, 0.3, 0],
            )
            tri = (
                Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
                .scale(0.3)
                .rotate(PI)
                .move_to(cb.get_end())
            )
            return cb

        def create_f_tri():
            cb = CubicBezier(
                f1_label.get_right() + [0.1, 0, 0],
                f1_label.get_right() + [1, 0, 0],
                f_ax.c2p(~f1, 1) + [0, 1, 0],
                f_ax.c2p(~f1, 1) + [0, 0.3, 0],
            )
            tri = (
                Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
                .scale(0.1)
                .rotate(PI)
                .next_to(cb.get_end(), DOWN, 0)
            )
            return tri

        f_line = create_f_line()
        f_tri = create_f_tri()

        self.play(
            LaggedStart(FadeIn(f1_label), Create(f_line), FadeIn(f_tri), lag_ratio=0.3)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back).set_color(YELLOW)
                    for m in f1_label[0][-3:]
                ],
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        zone_table = MobjectTable(
            [
                [],
            ],
            row_labels=[
                Tex(r"1\\$(f_1)$"),
                Tex(r"2\\$(f_s - f_1)$"),
                Tex(r"2\\$(f_s + f_1)$"),
                Tex(r"2\\$(2f_s - f_1)$"),
                Tex(r"2\\$(2f_s + f_1)$"),
            ],
        )

        # self.play()

        self.wait(2)


class TexTest(Scene):
    def construct(self):
        time_label = Text(r"Hello", font=FONT, font_size=DEFAULT_FONT_SIZE * 0.5)

        self.add(time_label)
