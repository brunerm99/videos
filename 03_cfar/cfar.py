# cfar.py

import sys
import warnings

from random import randint
import numpy as np
from manim import *
from MF_Tools import VT
from numpy.fft import fft, fftshift
from scipy import signal, interpolate


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

NOISE_COLOR = PURPLE
TARGETS_COLOR = GREEN


SKIP_ANIMATIONS_OVERRIDE = False


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_fft_values(
    x_n,
    fs,
    stop_time,
    fft_len=2**18,
    f_min=0,
    f_max=20,
    y_min=None,
    stage=4,
):
    N = stop_time * fs

    X_k = fftshift(fft(x_n, fft_len))
    if stage > 1:
        X_k /= N / 2
    if stage > 2:
        X_k = np.abs(X_k)
    if stage > 3:
        X_k = 10 * np.log10(X_k)

    freq = np.linspace(-fs / 2, fs / 2, fft_len)

    indices = np.where((freq > f_min) & (freq < f_max))
    x_values = freq[indices]
    y_values = X_k[indices]

    if y_min is not None:
        y_values += -y_min
        y_values[y_values < 0] = 0

    return x_values, y_values


class Intro(Scene):
    def construct(self):
        stop_time = 16
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = VT(-3)
        power_norm_2 = VT(-9)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(-25)

        f_max = 8
        y_min = VT(-30)

        x_len = 9.5
        y_len = 4.5

        noise_seed = VT(2)

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        ax_label = ax.get_axis_labels(Tex("$R$"), Tex())

        def get_plot_values(
            ports=["1", "2", "3", "noise"],
            y_min=None,
            noise_power_db=-20,
        ):
            fft_len = N * 4
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            np.random.seed(int(~noise_seed))
            noise = np.random.normal(
                loc=0, scale=10 ** (noise_power_db / 10), size=t.size
            )

            freq_add = np.zeros(fft_len)
            freq_add[(freq > 6.5) & (freq < f_max)] = 1
            b, a = signal.butter(4, 0.01, btype="low", analog=False)

            bias = 10
            freq_add_smoothed = signal.filtfilt(b, a, freq_add) * bias

            sig1 = np.sin(2 * PI * f1 * t) * (10 ** (~power_norm_1 / 10))
            sig2 = np.sin(2 * PI * f2 * t) * (10 ** (~power_norm_2 / 10))
            sig3 = np.sin(2 * PI * f3 * t) * (10 ** (~power_norm_3 / 10))

            signals = {
                "1": sig1,
                "2": sig2,
                "3": sig3,
                "noise": noise,
            }
            summed_signals = sum([signals.get(port) for port in ports])

            blackman_window = signal.windows.blackman(N)
            summed_signals *= blackman_window

            X_k = fftshift(fft(summed_signals, fft_len))
            X_k /= N / 2
            X_k *= 10 ** (freq_add_smoothed / 10)
            X_k = np.abs(X_k)
            X_k_log = 10 * np.log10(X_k)

            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = X_k_log[indices]

            if y_min is not None:
                y_values[y_values < y_min] = y_min
                y_values -= y_min

            return dict(x_values=x_values, y_values=y_values)

        return_plot = always_redraw(
            lambda: ax.plot_line_graph(
                **get_plot_values(
                    ports=["1", "2", "3", "noise"],
                    noise_power_db=~noise_sigma_db,
                    y_min=~y_min,
                ),
                line_color=RX_COLOR,
                add_vertex_dots=False,
            )
        )

        freqs = np.array([f1, f2, f3])
        powers = np.array([~power_norm_1, ~power_norm_2, ~power_norm_3]) - ~y_min

        ideal_return_plot = always_redraw(
            lambda: VGroup(
                *[
                    VGroup(
                        Line(ax.c2p(f, 0), ax.c2p(f, p), color=RX_COLOR),
                        Dot().move_to(ax.c2p(f, p)),
                    )
                    for f, p in zip(freqs, powers)
                ]
            )
        )

        target_labels = VGroup(
            *[
                Tex(f"{idx + 1}").next_to(ax.c2p(f, p), UP)
                for idx, (f, p) in enumerate(zip(freqs, powers))
            ]
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(Create(ax), FadeIn(ax_label))

        self.wait(0.5)

        self.play(LaggedStart(*[Create(m) for m in ideal_return_plot], lag_ratio=0.3))
        self.add(ideal_return_plot)

        self.wait(0.5)

        self.play(
            LaggedStart(*[FadeIn(m, shift=DOWN) for m in target_labels], lag_ratio=0.3)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Create(return_plot, run_time=3),
            LaggedStart(*[Uncreate(m) for m in ideal_return_plot], lag_ratio=0.3),
            LaggedStart(
                *[m.animate.shift(DOWN / 2) for m in target_labels], lag_ratio=0.3
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(noise_sigma_db @ 3, run_time=2)

        self.wait(0.5)

        # for _ in range(100):
        #     noise_seed @= randint(0, 100)
        #     self.wait(0.02, frozen_frame=False)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.shift(UP / 2).set_color(GREEN) for m in target_labels],
                lag_ratio=0.1,
            ),
        )
        self.play(
            LaggedStart(
                *[m.animate.shift(DOWN / 2).set_color(WHITE) for m in target_labels],
                lag_ratio=0.1,
            ),
        )

        self.wait(0.5)

        peaks_q = (
            Tex(r"noise", r" or\\", "targets", "?")
            .to_edge(UP, LARGE_BUFF)
            .shift(RIGHT * 1.5 + UP * 0.5)
        )
        peaks_q[0].set_color(GREEN)
        peaks_q[2].set_color(RED)

        freq, X_k = get_plot_values(
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            y_min=~y_min,
        ).values()

        poss_peak_bezs = VGroup()
        for r in [(4, 5), (6, f_max - 0.5), (0, f1 - 0.3)]:
            idx_local_max = X_k[(freq > r[0]) & (freq < r[1])].argmax()
            f_local_max = freq[(freq > r[0]) & (freq < r[1])][idx_local_max]
            mag_local_max = X_k[(freq > r[0]) & (freq < r[1])][idx_local_max]
            p2 = ax.c2p(f_local_max, mag_local_max)
            poss_peak_bezs.add(
                CubicBezier(
                    peaks_q.get_bottom() + [0, -0.1, 0],
                    peaks_q.get_bottom() + [0, -1, 0],
                    p2 + [0, 1, 0],
                    p2 + [0, 0.1, 0],
                )
            )

        self.play(
            LaggedStart(
                FadeIn(peaks_q, shift=DOWN),
                *[Create(bez) for bez in poss_peak_bezs],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(peaks_q, target_labels),
            *[Uncreate(m) for m in poss_peak_bezs],
        )

        self.wait(2)


class StaticThreshold(Scene):
    def construct(self):
        stop_time = 16
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = VT(-3)
        power_norm_2 = VT(-9)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(3)

        f_max = 8
        y_min = VT(-30)

        x_len = 9.5
        y_len = 4.5

        noise_seed = VT(2)

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        ax_label = ax.get_axis_labels(Tex("$R$"), Tex())

        def get_plot_values(
            ports=["1", "2", "3", "noise"],
            y_min=None,
            noise_power_db=-20,
            f1l=None,
            f2l=None,
            f3l=None,
        ):
            fft_len = N * 4
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            np.random.seed(int(~noise_seed))
            noise = np.random.normal(
                loc=0, scale=10 ** (noise_power_db / 10), size=t.size
            )

            freq_add = np.zeros(fft_len)
            freq_add[(freq > 6.5) & (freq < f_max)] = 1
            b, a = signal.butter(4, 0.01, btype="low", analog=False)

            bias = 10
            freq_add_smoothed = signal.filtfilt(b, a, freq_add) * bias

            f1l = f1 if f1l is None else f1l
            f2l = f2 if f2l is None else f2l
            f3l = f3 if f3l is None else f3l

            sig1 = np.sin(2 * PI * f1l * t) * (10 ** (~power_norm_1 / 10))
            sig2 = np.sin(2 * PI * f2l * t) * (10 ** (~power_norm_2 / 10))
            sig3 = np.sin(2 * PI * f3l * t) * (10 ** (~power_norm_3 / 10))

            signals = {
                "1": sig1,
                "2": sig2,
                "3": sig3,
                "noise": noise,
            }
            summed_signals = sum([signals.get(port) for port in ports])

            blackman_window = signal.windows.blackman(N)
            summed_signals *= blackman_window

            X_k = fftshift(fft(summed_signals, fft_len))
            X_k /= N / 2
            X_k *= 10 ** (freq_add_smoothed / 10)
            X_k = np.abs(X_k)
            X_k_log = 10 * np.log10(X_k)

            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = X_k_log[indices]

            if y_min is not None:
                y_values[y_values < y_min] = y_min
                y_values -= y_min

            return dict(x_values=x_values, y_values=y_values)

        freq, X_k_log = get_plot_values(
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            y_min=~y_min,
        ).values()
        X_k = 10 ** (X_k_log / 10)

        return_plot = ax.plot_line_graph(
            freq,
            X_k_log,
            line_color=RX_COLOR,
            add_vertex_dots=False,
        )

        static_threshold = np.ones(X_k_log.shape) * 10 ** (-15 / 10)  # linear
        targets_only = np.copy(X_k)
        targets_only[np.where(X_k < static_threshold)] = np.ma.masked

        static_threshold = VT(-10)
        static_threshold_plot = always_redraw(
            lambda: DashedVMobject(
                ax.plot(lambda t: ~static_threshold - ~y_min, color=YELLOW)
            )
        )

        def get_region_polygon_updater(up):
            y_offset = -~y_min if up else 0
            color = GREEN if up else PURPLE

            def get_region_polygon():
                dl = ax.c2p(0, ~static_threshold - ~y_min)
                dr = ax.c2p(f_max, ~static_threshold - ~y_min)
                ul = ax.c2p(0, 0 + y_offset)
                ur = ax.c2p(f_max, 0 + y_offset)
                polygon = Polygon(
                    ur,
                    ul,
                    dl,
                    dr,
                )
                polygon.stroke_width = 1
                polygon.set_fill(color, opacity=0.3)
                polygon.set_stroke(YELLOW_B, opacity=0)
                return polygon

            return get_region_polygon

        targets_region = always_redraw(get_region_polygon_updater(up=True))
        noise_region = always_redraw(get_region_polygon_updater(up=False))

        targets_label = Tex("Targets", color=TARGETS_COLOR).next_to(
            targets_region, RIGHT, SMALL_BUFF
        )
        noise_label = Tex("Noise", color=NOISE_COLOR).next_to(
            noise_region, RIGHT, SMALL_BUFF
        )

        should_noise_label = (
            Tex(r"actually\\", "noise").to_edge(UP, MED_SMALL_BUFF).shift(RIGHT)
        )
        should_targets_label = (
            Tex(r"actually\\", "target").to_edge(UP, MED_SMALL_BUFF).shift(LEFT * 2)
        )
        should_noise_label[1].set_color(NOISE_COLOR)
        should_targets_label[1].set_color(TARGETS_COLOR)

        p2 = ax.c2p(7.2, -7.2 - ~y_min)
        should_noise_bez = CubicBezier(
            should_noise_label.get_bottom() + [0, -0.1, 0],
            should_noise_label.get_bottom() + [0, -1, 0],
            p2 + [0, 1, 0],
            p2 + [0, 0.1, 0],
        )

        p2 = ax.c2p(f2, ~power_norm_2 - ~y_min - 3)
        should_targets_bez = CubicBezier(
            should_targets_label.get_bottom() + [0, -0.1, 0],
            should_targets_label.get_bottom() + [0, -1, 0],
            p2 + [0, 1, 0],
            p2 + [0, 0.1, 0],
        )

        self.add(
            ax,
            ax_label,
            return_plot,
        )

        self.wait(0.5)

        self.play(Create(static_threshold_plot))

        self.wait(0.5)

        self.play(Create(targets_region), FadeIn(targets_label, shift=LEFT))

        self.wait(0.5)

        self.play(Create(noise_region), FadeIn(noise_label, shift=LEFT))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(should_noise_label),
                Create(should_noise_bez),
                FadeIn(should_targets_label),
                Create(should_targets_bez),
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(should_noise_label, should_targets_label),
            Uncreate(should_noise_bez),
            Uncreate(should_targets_bez),
        )

        self.wait(0.5)

        self.play(static_threshold - 10, run_time=0.5)

        self.wait(0.5)

        self.play(static_threshold + 5, run_time=0.5)

        self.wait(0.5)

        self.play(static_threshold - 10, run_time=0.5)

        self.wait(0.5)

        self.play(static_threshold + 12, run_time=0.5)

        self.wait(0.5)

        power_norm_1 @= -6
        power_norm_2 @= -9
        power_norm_3 @= -6

        _, X_k_log_2 = get_plot_values(
            ports=["1", "2", "3", "noise"],
            noise_power_db=2,
            y_min=~y_min,
            f1l=2.3,
            f2l=6.8,
            f3l=7.2,
        ).values()
        X_k_2 = 10 ** (X_k_log / 10)

        return_plot_2 = ax.plot_line_graph(
            freq,
            X_k_log_2,
            line_color=RX_COLOR,
            add_vertex_dots=False,
        )

        return_plot.save_state()
        self.play(
            LaggedStart(Uncreate(return_plot), Create(return_plot_2), lag_ratio=0.3)
        )

        # power_norm_1.restore()
        # power_norm_2.restore()
        # power_norm_3.restore()

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(return_plot_2), return_plot.animate.restore(), lag_ratio=0.3
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(targets_region),
            Uncreate(noise_region),
            Uncreate(static_threshold_plot),
        )

        self.wait(0.5)

        b, a = signal.butter(N=2, Wn=0.02, btype="low")

        X_k_smoothed = interpolate.interp1d(
            freq, signal.filtfilt(b, a, X_k_log) + 4, fill_value="extrapolate"
        )

        dynamic_threshold_plot = ax.plot(X_k_smoothed, color=YELLOW)
        dynamic_threshold_plot_dashed = DashedVMobject(dynamic_threshold_plot)

        dynamic_noise_region = ax.get_area(
            dynamic_threshold_plot,
            x_range=[0, f_max],
            color=NOISE_COLOR,
            opacity=0.3,
            stroke_opacity=0,
        )
        dynamic_targets_region = ax.get_area(
            ax.plot(lambda t: -~y_min),
            bounded_graph=dynamic_threshold_plot,
            x_range=[0, f_max],
            color=TARGETS_COLOR,
            opacity=0.3,
            stroke_opacity=0,
        )

        self.play(
            Create(dynamic_threshold_plot_dashed),
            Create(dynamic_noise_region),
            Create(dynamic_targets_region),
        )

        # TODO: Maybe add updaters here and play with the
        # filter inputs to show how it can be configured?

        self.wait(0.5)

        filter_dynamic_threshold_code = Code(
            code="from scipy import signal\n"
            'b, a = signal.butter(N=2, Wn=0.02, btype="low")\n'
            "offset = 4\n"
            "threshold = signal.filtfilt(b, a, range_spectrum) + offset",
            font="FiraCode Nerd Font Mono",
            background="window",
            language="Python",
        ).next_to([0, -config["frame_height"] / 2, 0], DOWN)

        self.play(filter_dynamic_threshold_code.animate.to_edge(DOWN, MED_SMALL_BUFF))

        self.wait(0.5)

        self.play(
            FadeOut(filter_dynamic_threshold_code, shift=DOWN * 3),
            Uncreate(ax),
            Uncreate(dynamic_threshold_plot_dashed),
            Uncreate(dynamic_noise_region),
            Uncreate(dynamic_targets_region),
            Uncreate(return_plot),
            FadeOut(ax_label),
            FadeOut(targets_label, noise_label, shift=RIGHT),
        )

        self.wait(2)


class CFARIntro(Scene):
    def construct(self): ...
