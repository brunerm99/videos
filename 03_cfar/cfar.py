# cfar.py

import sys
import warnings

from random import randint
import numpy as np
from manim import *
from MF_Tools import VT
from numpy.fft import fft, fftshift
from scipy import signal


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR


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

        power_norm_1 = VT(-6)
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
            np.random.seed(int(~noise_seed))
            noise = np.random.normal(
                loc=0, scale=10 ** (noise_power_db / 10), size=t.size
            )

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

            fft_len = N * 4
            summed_fft = fft(summed_signals, fft_len) / (N / 2)
            summed_fft_log = 10 * np.log10(fftshift(summed_fft))
            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = summed_fft_log[indices]

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

        power_norm_1 = VT(-6)
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
        ):
            fft_len = N * 4
            freq = np.linspace(-fs / 2, fs / 2, fft_len)

            np.random.seed(int(~noise_seed))
            noise = np.random.normal(
                loc=0, scale=10 ** (noise_power_db / 10), size=t.size
            )

            freq_add = np.zeros(fft_len)
            freq_add[(freq > 6.5) & (freq < f_max)] = 1
            b, a = signal.butter(4, 0.001, btype="low", analog=False)

            bias = 10
            freq_add_smoothed = (
                freq_add * bias
            )  # signal.filtfilt(b, a, freq_add) * bias

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

            summed_fft = fft(summed_signals, fft_len) / (N / 2)
            summed_fft_log = 10 * np.log10(fftshift(summed_fft))
            summed_fft_log += freq_add_smoothed
            indices = np.where((freq > 0) & (freq < f_max))
            x_values = freq[indices]
            y_values = summed_fft_log[indices]

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

        self.add(ax, ax_label, return_plot)
