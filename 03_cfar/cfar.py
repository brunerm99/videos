# cfar.py

import sys
import warnings

from random import randint, randrange
import numpy as np
from manim import *
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, interpolate


warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR
from props import WeatherRadarTower

config.background_color = BACKGROUND_COLOR

NOISE_COLOR = PURPLE
TARGETS_COLOR = GREEN

CUT_COLOR = YELLOW
GAP_COLOR = RED
REF_COLOR = BLUE


SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_plot_values(
    power_norm_1,
    power_norm_2,
    power_norm_3,
    fs,
    stop_time,
    ports=["1", "2", "3", "noise"],
    y_min=None,
    noise_power_db=-20,
    noise_seed=0,
    f_max=None,
    f1l=None,
    f2l=None,
    f3l=None,
):
    N = int(fs * stop_time)
    fft_len = N * 4
    freq = np.linspace(-fs / 2, fs / 2, fft_len)

    t = np.linspace(0, stop_time, N)

    f_max = fs / 2 if f_max is None else f_max

    np.random.seed(int(noise_seed))
    noise = np.random.normal(loc=0, scale=10 ** (noise_power_db / 10), size=t.size)

    freq_add = np.zeros(fft_len)
    freq_add[(freq > 6.5) & (freq < f_max)] = 1
    b, a = signal.butter(4, 0.01, btype="low", analog=False)

    bias = 10
    freq_add_smoothed = signal.filtfilt(b, a, freq_add) * bias

    sig1 = np.sin(2 * PI * f1l * t) * (10 ** (power_norm_1 / 10))
    sig2 = np.sin(2 * PI * f2l * t) * (10 ** (power_norm_2 / 10))
    sig3 = np.sin(2 * PI * f3l * t) * (10 ** (power_norm_3 / 10))

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


def cfar_fast(
    x: np.ndarray,
    num_ref_cells: int,
    num_guard_cells: int,
    bias: float = 1,
    method=np.mean,
):
    pad = int((num_ref_cells + num_guard_cells))
    # fmt: off
    window_mean = np.pad(                                                                   # Pad front/back since n_windows < n_points
        method(                                                                             # Apply input method to remaining compute cells
            np.delete(                                                                      # Remove guard cells, CUT from computation
                sliding_window_view(x, (num_ref_cells * 2) + (num_guard_cells * 2)),        # Windows of x including CUT, guard cells, and compute cells
                np.arange(int(num_ref_cells), num_ref_cells + (num_guard_cells * 2) + 1),   # Get indices of guard cells, CUT
                axis=1), 
            axis=1
        ), (pad - 1, pad),                                                               
        "edge"                                                                              # Fill with edge values
    ) * bias                                                                                # Multiply output by bias over which cell is not noise
    # fmt: on
    return window_mean


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

        freq, X_k_log = get_plot_values(
            power_norm_1=~power_norm_1,
            power_norm_2=~power_norm_2,
            power_norm_3=~power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
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
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(should_targets_label),
                Create(should_targets_bez),
                lag_ratio=0.3,
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

        _, X_k_log_2 = get_plot_values(
            power_norm_1=-6,
            power_norm_2=-9,
            power_norm_3=-6,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
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


class CFARIntro(MovingCameraScene):
    def construct(self):
        label = Tex("CFAR").scale(2.5)

        self.next_section(skip_animations=skip_animations(True))
        self.play(FadeIn(label, shift=UP))

        self.wait(0.5)

        label_spelled = Tex(
            r"\raggedright Constant\\ \vspace{2mm} False\\ \vspace{2mm} Alarm\\ \vspace{2mm} Rate"
        ).scale(2)

        self.play(
            TransformByGlyphMap(
                label,
                label_spelled,
                ([0], [0]),
                ([1], [8], {"delay": 0.1}),
                ([2], [13], {"delay": 0.2}),
                ([3], [18], {"delay": 0.3}),
                ([], [1, 2, 3, 4, 5, 6, 7], {"shift": LEFT, "delay": 0.4}),
                ([], [9, 10, 11, 12], {"shift": LEFT, "delay": 0.5}),
                ([], [14, 15, 16, 17, 18], {"shift": LEFT, "delay": 0.6}),
                ([], [19, 20, 21], {"shift": LEFT, "delay": 0.7}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        stop_time = VT(16)
        fs = 1000

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
        ax_label = ax.get_axis_labels(Tex("$R$"), Tex("$A$"))

        freq, X_k_log = get_plot_values(
            power_norm_1=~power_norm_1,
            power_norm_2=~power_norm_2,
            power_norm_3=~power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=~stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
        ).values()
        X_k = 10 ** (X_k_log / 10)

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)

        plot_group = VGroup(ax, ax_label, return_plot).next_to(
            [0, -config["frame_height"] / 2, 0], DOWN
        )

        self.add(plot_group)

        self.play(
            FadeOut(label_spelled), plot_group.animate.to_edge(DOWN, MED_LARGE_BUFF)
        )

        self.wait(0.5)

        range_tracker = VT(0)  # 0 -> f_max

        range_arrow = always_redraw(
            lambda: Arrow(UP, DOWN).next_to(ax, UP).set_x(ax.c2p(~range_tracker, 0)[0])
        )

        range_label = always_redraw(
            lambda: Tex(f"R = {~range_tracker*100:.2f}m").next_to(range_arrow, UP)
        )

        self.play(GrowArrow(range_arrow), FadeIn(range_label))

        self.wait(0.5)

        self.play(range_tracker @ f_max, run_time=2)

        self.wait(0.5)

        self.play(FadeOut(range_arrow, range_label))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        pfa_eqn = Tex(
            # r"P_{FA} = \int_{T}^{+\infty} \frac{z}{\sigma^{2}_{n}} \exp{\left( - \frac{z^2}{2 \sigma^{2}_{n}} \right) dz}"
            r"$P_{FA}$ - constant"
        ).to_edge(UP, MED_SMALL_BUFF)

        p = pfa_eqn[0][0]
        p.save_state()
        fa = pfa_eqn[0][1:3]
        fa.save_state()
        Group(p, fa).set_x(0)

        self.play(FadeIn(p))

        self.wait(0.5)

        self.play(FadeIn(fa))

        self.wait(0.5)

        whole_range = Line()
        whole_range.width = return_plot.width
        whole_range.next_to(return_plot, UP, LARGE_BUFF)
        whole_range_l = Line(whole_range.get_midpoint(), whole_range.get_left())
        whole_range_r = Line(whole_range.get_midpoint(), whole_range.get_right())
        whole_range_l_vert = Line(
            whole_range.get_left() + DOWN / 4, whole_range.get_left() + UP / 4
        )
        whole_range_r_vert = Line(
            whole_range.get_right() + DOWN / 4, whole_range.get_right() + UP / 4
        )
        whole_range_group = VGroup(
            whole_range_l, whole_range_r, whole_range_l_vert, whole_range_r_vert
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    p.animate.restore(),
                    fa.animate.restore(),
                ),
                Write(pfa_eqn[0][3:]),
                AnimationGroup(Create(whole_range_l), Create(whole_range_r)),
                AnimationGroup(Create(whole_range_l_vert), Create(whole_range_r_vert)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        cfar_spelled = Tex("Constant False Alarm Rate").move_to(pfa_eqn)

        self.play(
            TransformByGlyphMap(
                pfa_eqn,
                cfar_spelled,
                ([0, 3], []),
                ([4], [0]),
                ([5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7], {"delay": 0.3}),
                ([1], [8], {"delay": 0.3}),
                ([], [9, 10, 11, 12], {"delay": 0.5}),
                ([2], [13], {"delay": 0.3}),
                ([], [14, 15, 16, 17, 18, 19, 20, 21], {"delay": 0.5}),
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(whole_range_l),
                    Uncreate(whole_range_r),
                    Uncreate(whole_range_l_vert),
                    Uncreate(whole_range_r_vert),
                ),
                plot_group.animate.set_y(0),
                cfar_spelled.animate.shift(UP * 3),
                lag_ratio=0.4,
            ),
        )
        self.remove(cfar_spelled)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        num_samples = 30
        samples = ax.get_vertical_lines_to_graph(
            return_plot, x_range=[0, f_max], num_lines=num_samples, color=BLUE
        )

        self.play(Create(samples), run_time=2)

        self.wait(0.5)

        sample_rects = ax.get_riemann_rectangles(
            return_plot,
            input_sample_type="right",
            x_range=[0, f_max],
            dx=f_max / num_samples,
            color=BLUE,
            stroke_color=BLACK,
            fill_opacity=0.7,
        ).set_z_index(1)

        self.wait(0.5)

        self.play(
            *[
                ReplacementTransform(sample, rect)
                for sample, rect in zip(samples, sample_rects)
            ]
        )

        self.wait(0.5)

        self.play(FadeOut(plot_group.set_z_index(-1)))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    Transform(
                        rect,
                        Square(
                            rect.width,
                            color=BLACK,
                            fill_color=BLUE,
                            fill_opacity=0,
                            stroke_width=DEFAULT_STROKE_WIDTH / 2,
                        )
                        .move_to(rect)
                        .set_y(0),
                    )
                    for rect in sample_rects
                ],
                lag_ratio=0.05,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                sample_rects[:7].width
            ).move_to(sample_rects[2])
        )

        self.wait(0.5)

        sample_labels = VGroup(
            *[
                MathTex(f"A_{{{idx}}}")
                .scale_to_fit_width(rect.width * 0.7)
                .move_to(rect)
                for idx, rect in enumerate(sample_rects)
            ]
        )

        self.play(LaggedStart(*[FadeIn(m) for m in sample_labels[:7]], lag_ratio=0.1))
        self.add(sample_labels[7:])

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        cut_index = num_samples // 2 + num_samples // 6

        n_gap_cells = 3
        gap_index_l = cut_index - n_gap_cells
        gap_index_r = cut_index + n_gap_cells

        n_ref_cells = 4
        ref_index_l = gap_index_l - n_ref_cells
        ref_index_r = gap_index_r + n_ref_cells

        cut = sample_rects[cut_index]

        gap_cells_l = sample_rects[gap_index_l:cut_index]
        gap_cells_r = sample_rects[cut_index + 1 : gap_index_r + 1]

        ref_cells_l = sample_rects[ref_index_l:gap_index_l]
        ref_cells_r = sample_rects[gap_index_r + 1 : ref_index_r + 1]

        cut_label_spelled = (
            Tex(r"Cell\\Under\\Test", color=CUT_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.7)
            .next_to(cut, UP, SMALL_BUFF)
        )
        cut_label = (
            Tex("CUT", color=CUT_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(cut, UP, SMALL_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                sample_rects[:9].width
            ).move_to(Group(cut, cut_label_spelled))
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # Real quick

        rq = Tex("Real quick!")

        params_title = Tex("Parameters:")
        params = BulletedList(
            r"Gap cells", "Reference cells", "Bias", buff=MED_SMALL_BUFF
        ).scale(0.8)

        rq_group = VGroup(rq, params_title, params).arrange(
            DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF
        )
        rq.shift(LEFT)
        rq_box = SurroundingRectangle(
            rq_group,
            color=RED,
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
            stroke_width=DEFAULT_STROKE_WIDTH / 4,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        )
        rq_group = (
            VGroup(rq_box, *rq_group)
            .scale_to_fit_width(sample_rects[:6].width)
            .move_to(self.camera.frame)
        ).set_z_index(3)

        rq.save_state()
        self.play(GrowFromCenter(Group(rq_box, rq.move_to(self.camera.frame))))

        self.wait(0.5)

        self.play(
            LaggedStart(
                rq.animate.restore(),
                Create(params_title),
                Create(params),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(ShrinkToCenter(rq_group))

        self.wait(0.5)

        self.play(
            FadeIn(cut_label_spelled),
            cut.animate.set_fill(color=CUT_COLOR, opacity=0.6),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                cut_label_spelled,
                cut_label,
                ([1, 2, 3, 5, 6, 7, 8, 10, 11, 12], []),
                ([0], [0], {"delay": 0.3}),
                ([4], [1], {"delay": 0.3}),
                ([9], [2], {"delay": 0.3}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        gap_label_l = (
            Tex("Gap", color=GAP_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(gap_cells_l, UP, SMALL_BUFF)
        )
        gap_label_r = gap_label_l.copy().next_to(gap_cells_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(GAP_COLOR, opacity=0.5),
                        cell_r.animate.set_fill(GAP_COLOR, opacity=0.5),
                    )
                    for cell_l, cell_r in zip(gap_cells_l[::-1], gap_cells_r)
                ],
                lag_ratio=0.15,
            ),
            FadeIn(gap_label_l, gap_label_r),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ax = Axes(
            x_range=[f3 - 1, f3 + 1, 0.5],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )
        ax_label = ax.get_axis_labels(Tex("$R$"), Tex("$A$"))

        def plot_return():
            freq, X_k_log = get_plot_values(
                power_norm_1=~power_norm_1,
                power_norm_2=~power_norm_2,
                power_norm_3=~power_norm_3,
                ports=["1", "2", "3", "noise"],
                noise_power_db=~noise_sigma_db,
                noise_seed=~noise_seed,
                y_min=~y_min,
                f_max=f_max,
                fs=fs,
                stop_time=~stop_time,
                f1l=f1,
                f2l=f2,
                f3l=f3,
            ).values()
            f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")
            return ax.plot(f_X_k_log, x_range=[f3 - 1, f3 + 1, 1 / fs], color=RX_COLOR)

        return_plot = always_redraw(plot_return)
        plot_group = VGroup(ax, ax_label, return_plot).next_to(
            [0, -config["frame_height"] / 2, 0], DOWN
        )

        self.add(plot_group)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.2).move_to(ax)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cell_size = VT(0.05)

        def get_sample_poly(idx, color, top_coord=None, bias: VT = None):
            def updater():
                mid = f3 + idx * ~cell_size * 2
                if top_coord is None:
                    top_coord_lc = ax.input_to_graph_coords(mid, return_plot)[1]
                else:
                    top_coord_lc = top_coord
                if bias is None:
                    bias_lc = 1
                else:
                    bias_lc = bias

                c = ~bias_lc if type(bias_lc) is VT else bias_lc
                top = ax.c2p(mid, top_coord_lc * c)[1]
                left_x = ax.input_to_graph_point(mid - ~cell_size, return_plot)[0]
                right_x = ax.input_to_graph_point(mid + ~cell_size, return_plot)[0]
                bot = ax.c2p(0, 0)[1]
                box = Polygon(
                    (left_x, top, 0),
                    (left_x, bot, 0),
                    (right_x, bot, 0),
                    (right_x, top, 0),
                    fill_color=color,
                    fill_opacity=0.5,
                    stroke_width=DEFAULT_STROKE_WIDTH / 2,
                )
                return box

            return updater

        cut_vert_box = always_redraw(get_sample_poly(0, CUT_COLOR))

        gap_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )
        gap_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )

        self.play(FadeIn(cut_vert_box))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cut_marker_midpoint = ax.input_to_graph_point(f3, return_plot) + UP / 4
        cell_size_pts = (
            ax.input_to_graph_point(f3 + ~cell_size, return_plot)[0]
            - cut_marker_midpoint[0]
        ) * 2.2

        cut_width_r = Line(
            cut_marker_midpoint, cut_marker_midpoint + [cell_size_pts, 0, 0]
        )
        cut_width_l = Line(
            cut_marker_midpoint, cut_marker_midpoint + [-cell_size_pts, 0, 0]
        )
        cut_width_end_l = Line(
            cut_width_l.get_end() + UP / 8, cut_width_l.get_end() + DOWN / 8
        )
        cut_width_end_r = Line(
            cut_width_r.get_end() + UP / 8, cut_width_r.get_end() + DOWN / 8
        )
        cut_width_start_l = Line(
            cut_width_l.get_midpoint() + UP / 8, cut_width_l.get_midpoint() + DOWN / 8
        )
        cut_width_start_r = Line(
            cut_width_r.get_midpoint() + UP / 8, cut_width_r.get_midpoint() + DOWN / 8
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(cut_width_l), Create(cut_width_r)),
                AnimationGroup(Create(cut_width_end_l), Create(cut_width_end_r)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Create(cut_width_start_l),
            Create(cut_width_start_r),
            Transform(
                cut_width_l,
                Line(
                    cut_width_l.get_midpoint(),
                    cut_marker_midpoint + [-cell_size_pts, 0, 0],
                ),
            ),
            Transform(
                cut_width_r,
                Line(
                    cut_width_r.get_midpoint(),
                    cut_marker_midpoint + [cell_size_pts, 0, 0],
                ),
            ),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(FadeIn(gap_vert_boxes_l, gap_vert_boxes_r))

        self.wait(0.5)

        cut_width_l.save_state()
        cut_width_r.save_state()
        cut_width_end_l.save_state()
        cut_width_end_r.save_state()
        self.play(
            stop_time @ 8,
            Transform(
                cut_width_l,
                Line(
                    cut_width_l.get_start(),
                    cut_marker_midpoint + [-cell_size_pts * 2, 0, 0],
                ),
            ),
            Transform(
                cut_width_r,
                Line(
                    cut_width_r.get_start(),
                    cut_marker_midpoint + [cell_size_pts * 2, 0, 0],
                ),
            ),
            Transform(
                cut_width_end_l,
                Line(
                    cut_width_l.get_end() + [-cell_size_pts, 0, 0] + UP / 8,
                    cut_width_l.get_end() + [-cell_size_pts, 0, 0] + DOWN / 8,
                ),
            ),
            Transform(
                cut_width_end_r,
                Line(
                    cut_width_r.get_end() + [cell_size_pts, 0, 0] + UP / 8,
                    cut_width_r.get_end() + [cell_size_pts, 0, 0] + DOWN / 8,
                ),
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            stop_time @ 16,
            cut_width_l.animate.restore(),
            cut_width_r.animate.restore(),
            cut_width_end_l.animate.restore(),
            cut_width_end_r.animate.restore(),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            Uncreate(cut_width_l),
            Uncreate(cut_width_r),
            Uncreate(cut_width_start_l),
            Uncreate(cut_width_start_r),
            Uncreate(cut_width_end_l),
            Uncreate(cut_width_end_r),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(cell_size @ 0.1, run_time=1.5)

        self.wait(0.5)

        self.play(cell_size @ 0.05, run_time=1.5)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                Group(cut_label, ref_cells_l, ref_cells_r).width * 1.2
            )
        )

        self.wait(0.5)

        ref_label_l = (
            Tex("Ref", color=REF_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(ref_cells_l, UP, SMALL_BUFF)
        )
        ref_label_r = ref_label_l.copy().next_to(ref_cells_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(REF_COLOR, opacity=0.5),
                        cell_r.animate.set_fill(REF_COLOR, opacity=0.5),
                    )
                    for cell_l, cell_r in zip(ref_cells_l[::-1], ref_cells_r)
                ],
                lag_ratio=0.15,
            ),
            FadeIn(ref_label_l, ref_label_r),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.shift(
                DOWN
                * (self.camera.frame.get_top()[1] - cut_label.get_y() - MED_LARGE_BUFF)
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        Transform(
                            ml,
                            ml.copy().scale(1.2),
                            rate_func=rate_functions.there_and_back,
                        ),
                        Transform(
                            mr,
                            mr.copy().scale(1.2),
                            rate_func=rate_functions.there_and_back,
                        ),
                    )
                    for ml, mr in zip(ref_cells_l[::-1], ref_cells_r)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        techniques = Tex("Techniques")
        averaging = Tex("Averaging")
        smallest = Tex("Smallest Averaging")
        largest = Tex("Largest Averaging")
        techniques_group = (
            VGroup(averaging, smallest, largest)
            .arrange(DOWN)
            .scale_to_fit_width(sample_rects[:6].width)
        )
        techniques_brace = Brace(techniques_group, LEFT, sharpness=0.7)
        techniques.scale_to_fit_width(sample_rects[:4].width).next_to(
            techniques_brace, LEFT, SMALL_BUFF
        )
        Group(techniques, techniques_group, techniques_brace).move_to(self.camera.frame)

        self.play(
            LaggedStart(
                FadeIn(techniques, techniques_brace, shift=RIGHT),
                *[FadeIn(t) for t in techniques_group],
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        self.play(averaging.animate.set_color(GREEN))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(techniques, techniques_brace, smallest, largest),
                averaging.animate.scale(1.5).move_to(self.camera.frame),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            averaging.animate.next_to(
                self.camera.frame.get_bottom(), UP, MED_SMALL_BUFF
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ref_cell_labels_l = sample_labels[ref_index_l:gap_index_l]
        ref_cell_labels_r = sample_labels[gap_index_r + 1 : ref_index_r + 1]

        ref_cell_label_summed = " + ".join(
            [
                label.get_tex_string()
                for label in [*ref_cell_labels_l, *ref_cell_labels_r]
            ]
        )

        mean_eqn_full = (
            MathTex(
                f"\\frac{{ {ref_cell_label_summed} }}{{N}}",
                font_size=DEFAULT_FONT_SIZE * 0.4,
            )
            .move_to(self.camera.frame)
            .shift(DOWN / 2)
        )

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        TransformFromCopy(m, mean_eqn_full[0][idx * 4 : idx * 4 + 3]),
                        FadeIn(mean_eqn_full[0][(idx + 1) * 4 - 1]),
                        lag_ratio=0.3,
                    )
                    for idx, m in enumerate([*ref_cell_labels_l, *ref_cell_labels_r])
                ],
                lag_ratio=0.3,
            )
        )

        self.play(FadeIn(mean_eqn_full[0][-1:]))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        mean_eqn = MathTex(
            f"\\frac{{\\sum_{{i= {gap_index_l-n_ref_cells} }}^{{{gap_index_l-1}}} A_i + \\sum_{{i= {gap_index_r+1} }}^{{{gap_index_r+n_ref_cells}}} A_i}}{{N}}",
            font_size=DEFAULT_FONT_SIZE * 0.6,
        ).move_to(mean_eqn_full)

        self.play(
            TransformByGlyphMap(
                mean_eqn_full,
                mean_eqn,
                (
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                ),
                ([15], [9], {"delay": 0.3}),
                (
                    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18],
                    {"delay": 0.6},
                ),
                ([31], [19], {"delay": 0.9}),
                ([32], [20], {"delay": 0.9}),
            )
        )

        self.wait(0.5)

        mean_eqn_w_bias = MathTex(
            f"\\text{{Bias}} \\cdot \\frac{{\\sum_{{i= {gap_index_l-n_ref_cells} }}^{{{gap_index_l-1}}} A_i + \\sum_{{i= {gap_index_r+1} }}^{{{gap_index_r+n_ref_cells}}} A_i}}{{N}}",
            font_size=DEFAULT_FONT_SIZE * 0.6,
        ).move_to(mean_eqn)

        # fmt: off
        self.play(
            TransformByGlyphMap(
                mean_eqn, 
                mean_eqn_w_bias,
                ([], [0,1,2,3,4]),
                ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
            )
        )
        # fmt: on

        self.wait(0.5)

        cut_threshold = MathTex(r"T_{CUT} = ", font_size=DEFAULT_FONT_SIZE * 0.6)
        eqn_loc = mean_eqn_w_bias.get_center()
        mean_eqn_w_bias_copy = mean_eqn_w_bias.copy()
        Group(cut_threshold, mean_eqn_w_bias_copy).arrange(RIGHT, SMALL_BUFF).move_to(
            eqn_loc
        )

        self.play(
            LaggedStart(
                Transform(mean_eqn_w_bias, mean_eqn_w_bias_copy),
                FadeIn(cut_threshold),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.2).move_to(ax)
        )

        self.wait(0.5)

        ref_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )
        ref_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )

        self.play(FadeIn(ref_vert_boxes_l, ref_vert_boxes_r))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cut_plot_label = Tex(
            "CUT", color=CUT_COLOR, font_size=DEFAULT_FONT_SIZE
        ).next_to(ax.input_to_graph_point(f3, return_plot), UP)
        gap_plot_label = Tex(
            "Gap", color=GAP_COLOR, font_size=DEFAULT_FONT_SIZE
        ).next_to(
            ax.input_to_graph_point(f3 + ~cell_size * 2 * (n_gap_cells), return_plot),
            UP,
            MED_LARGE_BUFF,
        )
        ref_plot_label = Tex(
            "Ref", color=REF_COLOR, font_size=DEFAULT_FONT_SIZE
        ).next_to(
            ax.input_to_graph_point(
                f3 + ~cell_size * 2 * (n_gap_cells + n_ref_cells), return_plot
            ),
            UP,
            MED_LARGE_BUFF,
        )
        self.play(FadeIn(cut_plot_label, shift=DOWN))

        self.wait(0.5)

        self.play(FadeIn(gap_plot_label, shift=DOWN))

        self.wait(0.5)

        self.play(FadeIn(ref_plot_label, shift=DOWN))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        bias = VT(1)
        threshold_top_coord = np.mean(
            [
                ax.input_to_graph_coords(f3 + idx * ~cell_size * 2, return_plot)[1]
                for idx in [
                    *range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells),
                    *range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells),
                ]
            ]
        )
        threshold_line = DashedVMobject(
            ax.plot(lambda t: threshold_top_coord, color=REF_COLOR)
        )

        self.play(FadeIn(threshold_line, shift=UP))

        self.wait(0.5)

        threshold_vert_box = always_redraw(
            get_sample_poly(0, REF_COLOR, top_coord=threshold_top_coord, bias=bias)
        )

        self.play(Create(threshold_vert_box))

        self.wait(0.5)

        self.play(FadeOut(threshold_line))

        self.wait(0.5)

        bias_label = always_redraw(
            lambda: Tex(f"Bias = {~bias:.2f}")
            .next_to(self.camera.frame.get_top(), DOWN, MED_SMALL_BUFF)
            .shift(LEFT * 3)
        )

        self.play(FadeIn(bias_label, shift=DOWN))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(bias @ 2, run_time=2)

        self.wait(0.5)

        cut_above_line = Line(
            threshold_vert_box.get_top(), cut_vert_box.get_top()
        ).shift(LEFT)
        cut_above_line_s = Line(
            cut_above_line.get_start() + LEFT / 6,
            cut_above_line.get_start() + RIGHT / 6,
        )
        cut_above_line_e = Line(
            cut_above_line.get_end() + LEFT / 6,
            cut_above_line.get_end() + RIGHT / 6,
        )

        self.play(
            LaggedStart(
                Create(cut_above_line_e),
                Create(cut_above_line.reverse_direction()),
                Create(cut_above_line_s),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        cut_gt_thresh = MathTex(r" > ", "T_{CUT}").next_to(
            cut_plot_label, RIGHT, SMALL_BUFF
        )
        cut_gt_thresh[1].set_color(REF_COLOR)

        self.play(FadeIn(cut_gt_thresh))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cut_label_new = Tex("CUT", color=CUT_COLOR, font_size=DEFAULT_FONT_SIZE)
        is_a_target = Tex("is a target!", font_size=DEFAULT_FONT_SIZE)
        VGroup(cut_label_new, is_a_target).arrange(RIGHT).move_to(self.camera.frame)

        self.play(
            LaggedStart(
                FadeOut(
                    cut_above_line_s,
                    cut_above_line_e,
                    cut_above_line,
                    plot_group,
                    cut_gt_thresh,
                    bias_label,
                    cut_vert_box,
                    gap_vert_boxes_l,
                    gap_vert_boxes_r,
                    ref_vert_boxes_l,
                    ref_vert_boxes_r,
                    # cut_plot_label,
                    gap_plot_label,
                    ref_plot_label,
                    threshold_vert_box,
                    mean_eqn_w_bias,
                    averaging,
                    cut_threshold,
                ),
                Transform(cut_plot_label, cut_label_new),
                FadeIn(is_a_target),
                # self.camera.frame.animate.scale_to_fit_width(
                #     sample_rects.width * 1.2
                # ).move_to(ORIGIN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class SweepPlot(MovingCameraScene):
    def construct(self):
        stop_time = 16
        fs = 1000

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = VT(-3)
        power_norm_2 = VT(-9)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(3)

        f_max = 8
        y_min = VT(-30)

        x_len = VT(11)
        y_len = 5.5

        noise_seed = VT(2)

        ax = always_redraw(
            lambda: Axes(
                x_range=[0, f_max, f_max / 4],
                y_range=[0, -~y_min, -~y_min / 4],
                tips=False,
                axis_config={
                    "include_numbers": False,
                },
                x_length=~x_len,
                y_length=y_len,
            ).to_edge(LEFT, LARGE_BUFF)
        )
        # ax_label = always_redraw(lambda: ax.get_axis_labels(Tex("$R$"), Tex()))

        freq, X_k_log = get_plot_values(
            power_norm_1=~power_norm_1,
            power_norm_2=~power_norm_2,
            power_norm_3=~power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
        ).values()

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = always_redraw(
            lambda: ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)
        )

        n_gap_cells = 3
        n_ref_cells = 4

        cell_size = VT(0.05)
        f_0 = ~cell_size * 2 * (n_gap_cells + n_ref_cells)
        f = VT(f_0)
        bias = VT(1)

        cfar_plot = always_redraw(
            lambda: ax.plot(
                interpolate.interp1d(
                    freq,
                    cfar_fast(X_k_log, num_guard_cells=8, num_ref_cells=12, bias=~bias),
                    fill_value="extrapolate",
                ),
                x_range=[f_0, ~f, 1 / fs],
                color=REF_COLOR,
            )
        )

        cfar_curr_dot = always_redraw(
            lambda: Dot().move_to(ax.input_to_graph_point(~f, cfar_plot))
        )

        plot_group = VGroup(
            ax,
            # ax_label,
            return_plot,
        )

        def get_sample_poly(idx, color, top_coord=None, bias: VT = None):
            def updater():
                mid = ~f + idx * ~cell_size * 2
                if top_coord is None:
                    top_coord_lc = ax.input_to_graph_coords(mid, return_plot)[1]
                else:
                    top_coord_lc = top_coord
                if bias is None:
                    bias_lc = 1
                else:
                    bias_lc = bias

                c = ~bias_lc if type(bias_lc) is VT else bias_lc
                top = ax.c2p(mid, top_coord_lc * c)[1]
                left_x = ax.input_to_graph_point(mid - ~cell_size, return_plot)[0]
                right_x = ax.input_to_graph_point(mid + ~cell_size, return_plot)[0]
                bot = ax.c2p(0, 0)[1]
                box = Polygon(
                    (left_x, top, 0),
                    (left_x, bot, 0),
                    (right_x, bot, 0),
                    (right_x, top, 0),
                    fill_color=color,
                    fill_opacity=0.5,
                    stroke_width=DEFAULT_STROKE_WIDTH / 2,
                )
                return box

            return updater

        cut_vert_box = always_redraw(get_sample_poly(0, CUT_COLOR))

        gap_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )
        gap_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )

        ref_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )
        ref_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(FadeIn(ax), Create(return_plot))

        self.wait(0.5)

        self.play(
            FadeIn(
                cut_vert_box,
                gap_vert_boxes_l,
                gap_vert_boxes_r,
                ref_vert_boxes_l,
                ref_vert_boxes_r,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        def camera_updater(m: Mobject):
            m.set_x(cut_vert_box.get_x())

        self.play(x_len @ (~x_len * 3))

        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(0.8).set_x(cut_vert_box.get_x()))
        self.camera.frame.add_updater(camera_updater)

        self.add(cfar_plot)

        self.wait(0.5, frozen_frame=False)

        self.play(Create(cfar_curr_dot))

        self.wait(0.5, frozen_frame=False)

        self.play(f @ (8 - f_0), run_time=14, rate_func=rate_functions.ease_in_sine)

        self.wait(0.5)

        self.play(Uncreate(cfar_curr_dot))

        self.wait(0.5)

        self.camera.frame.remove_updater(camera_updater)
        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.1).move_to(ax)
        )

        self.wait(0.5)

        self.play(bias @ 1.5)

        self.wait(0.5)

        self.play(
            FadeOut(
                cut_vert_box,
                gap_vert_boxes_l,
                gap_vert_boxes_r,
                ref_vert_boxes_l,
                ref_vert_boxes_r,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        cfar_plot_2 = ax.plot(
            interpolate.interp1d(
                freq,
                cfar_fast(X_k_log, num_guard_cells=8, num_ref_cells=12, bias=~bias),
                fill_value="extrapolate",
            ),
            x_range=[f_0, ~f, 1 / fs],
            color=REF_COLOR,
        )

        dynamic_noise_region = ax.get_area(
            cfar_plot_2,
            x_range=[f_0, ~f],
            color=NOISE_COLOR,
            opacity=0.3,
            stroke_opacity=0,
        )
        dynamic_targets_region = ax.get_area(
            ax.plot(lambda t: -~y_min + 5, x_range=[f_0, ~f, 1 / fs]),
            bounded_graph=cfar_plot_2,
            x_range=[f_0, ~f],
            color=TARGETS_COLOR,
            opacity=0.3,
            stroke_opacity=0,
        )

        targets_label = Tex("Targets", color=TARGETS_COLOR).next_to(
            dynamic_targets_region, RIGHT, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                FadeIn(dynamic_targets_region),
                # FadeIn(targets_label),
            )
        )

        self.wait(0.5)

        noise_label = Tex("Noise", color=NOISE_COLOR).next_to(
            dynamic_noise_region, RIGHT, SMALL_BUFF
        )

        self.play(
            LaggedStart(
                FadeIn(dynamic_noise_region),
                # FadeIn(noise_label),
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class DesignIntro(Scene):
    def construct(self):
        the_design_process = Tex("The Design Process").scale(1.5)

        notebook_ss = ImageMobject("./static/notebook_screenshot.png")
        notebook_label = Text("cfar.ipynb", font="FiraCode Nerd Font Mono").next_to(
            notebook_ss, UP
        )
        notebook_box = SurroundingRectangle(
            Group(notebook_ss, notebook_label),
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        )
        notebook = (
            Group(notebook_box, notebook_label, notebook_ss)
            .scale_to_fit_width(config["frame_width"] * 0.8)
            .move_to(ORIGIN)
            .set_z_index(2)
        )

        params = (
            BulletedList("Gap Cells", "Reference Cells", "Bias")
            .scale(1.5)
            .to_edge(DOWN, LARGE_BUFF)
        )

        self.play(FadeIn(the_design_process))

        self.wait(0.5)

        self.play(the_design_process.animate.to_edge(UP, LARGE_BUFF))

        self.wait(0.5)

        self.play(FadeIn(params[0], shift=RIGHT))

        self.wait(0.5)

        self.play(FadeIn(params[1], shift=LEFT))

        self.wait(0.5)

        self.play(FadeIn(params[2], shift=RIGHT))

        self.wait(0.5)

        self.play(notebook.shift(DOWN * 8).animate.shift(UP * 8))

        self.wait(0.5)

        self.play(Group(*self.mobjects).animate.shift(UP * 8))

        self.wait(2)


class Designer(Scene):
    def construct(self):
        stop_time = 16
        fs = 1000

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = VT(-3)
        power_norm_2 = VT(-9)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(3)

        f_max = 8
        y_min = VT(-30)

        x_len = VT(11)
        y_len = 5.5

        noise_seed = VT(2)

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=~x_len,
            y_length=y_len,
        ).to_edge(DOWN, MED_SMALL_BUFF)

        ax_label = ax.get_axis_labels(Tex("$R$"), Tex("$A$"))

        freq, X_k_log = get_plot_values(
            power_norm_1=~power_norm_1,
            power_norm_2=~power_norm_2,
            power_norm_3=~power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=~noise_seed,
            y_min=~y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
        ).values()

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)

        n_gap_cells = VT(3)
        n_ref_cells = VT(4)

        cell_size = VT(0.05)
        f_0 = ~cell_size * 2 * (~n_gap_cells + ~n_ref_cells)
        f = VT(f3)
        bias = VT(1)

        cfar_plot = always_redraw(
            lambda: ax.plot(
                interpolate.interp1d(
                    freq,
                    cfar_fast(
                        X_k_log,
                        num_guard_cells=int(8 * 6 / ~n_gap_cells),
                        num_ref_cells=int(12 * 8 / ~n_ref_cells),
                        bias=~bias,
                    ),
                    fill_value="extrapolate",
                ),
                x_range=[f_0, f_max - f_0, 1 / fs],
                color=REF_COLOR,
            )
        )

        def get_sample_poly(idx, color, top_coord=None, bias: VT = None):
            def updater():
                mid = ~f + idx * ~cell_size * 2
                if top_coord is None:
                    top_coord_lc = ax.input_to_graph_coords(mid, return_plot)[1]
                else:
                    top_coord_lc = top_coord
                if bias is None:
                    bias_lc = 1
                else:
                    bias_lc = bias

                c = ~bias_lc if type(bias_lc) is VT else bias_lc
                top = ax.c2p(mid, top_coord_lc * c)[1]
                left_x = ax.input_to_graph_point(mid - ~cell_size, return_plot)[0]
                right_x = ax.input_to_graph_point(mid + ~cell_size, return_plot)[0]
                bot = ax.c2p(0, 0)[1]
                box = Polygon(
                    (left_x, top, 0),
                    (left_x, bot, 0),
                    (right_x, bot, 0),
                    (right_x, top, 0),
                    fill_color=color,
                    fill_opacity=0.3,
                    stroke_width=DEFAULT_STROKE_WIDTH / 8,
                )
                return box

            return updater

        cut_vert_box = always_redraw(get_sample_poly(0, CUT_COLOR))

        gap_vert_boxes_l = always_redraw(
            lambda: VGroup(
                *[
                    get_sample_poly(-idx, GAP_COLOR)()
                    for idx in range(1, int(~n_gap_cells + 1))
                ]
            )
        )
        gap_vert_boxes_r = always_redraw(
            lambda: VGroup(
                *[
                    get_sample_poly(idx, GAP_COLOR)()
                    for idx in range(1, int(~n_gap_cells + 1))
                ]
            )
        )

        ref_vert_boxes_l = always_redraw(
            lambda: VGroup(
                *[
                    get_sample_poly(-idx, REF_COLOR)()
                    for idx in range(
                        int(1 + ~n_gap_cells), int(~n_ref_cells + 1 + ~n_gap_cells)
                    )
                ]
            )
        )
        ref_vert_boxes_r = always_redraw(
            lambda: VGroup(
                *[
                    get_sample_poly(idx, REF_COLOR)()
                    for idx in range(
                        int(1 + ~n_gap_cells), int(~n_ref_cells + 1 + ~n_gap_cells)
                    )
                ]
            )
        )

        plot_group = VGroup(ax, ax_label, return_plot, cfar_plot)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(plot_group.shift(DOWN * 10).animate.shift(UP * 10))

        self.wait(0.5)

        self.play(FadeIn(cut_vert_box))

        self.wait(0.5)

        self.play(FadeIn(gap_vert_boxes_l, gap_vert_boxes_r))

        self.wait(0.5)

        self.play(FadeIn(ref_vert_boxes_l, ref_vert_boxes_r))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        gap_cells_label = Tex("Gap Cells:")
        ref_cells_label = Tex("Ref Cells:")
        bias_label = Tex("Bias:")

        param_labels = (
            VGroup(gap_cells_label, ref_cells_label, bias_label)
            .arrange(DOWN, SMALL_BUFF, aligned_edge=LEFT)
            .to_corner(UR)
            .shift(LEFT)
        )

        gap_cells_num_label = always_redraw(
            lambda: Tex(f"{int(~n_gap_cells)}").next_to(
                gap_cells_label, RIGHT, SMALL_BUFF
            )
        )
        ref_cells_num_label = always_redraw(
            lambda: Tex(f"{int(~n_ref_cells)}").next_to(
                ref_cells_label, RIGHT, SMALL_BUFF
            )
        )
        bias_num_label = always_redraw(
            lambda: Tex(f"{~bias:.2f}").next_to(bias_label, RIGHT, SMALL_BUFF)
        )

        self.play(
            FadeIn(
                param_labels,
                gap_cells_num_label,
                ref_cells_num_label,
                bias_num_label,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(gap_cells_label.animate.set_color(YELLOW))

        # self.next_section(skip_animations=skip_animations(True))
        # self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         *[
        #             AnimationGroup(
        #                 l.animate(rate_func=rate_functions.there_and_back).scale(1.5),
        #                 r.animate(rate_func=rate_functions.there_and_back).scale(1.5),
        #             )
        #             for l, r in zip(gap_vert_boxes_l, gap_vert_boxes_r)
        #         ],
        #         lag_ratio=0.2,
        #     )
        # )

        # Plane vs human
        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        person = ImageMobject("../props/static/person.png").scale_to_fit_height(
            config["frame_height"] * 0.2
        )
        plane = SVGMobject("../props/static/plane.svg").scale(1.2).next_to(person, UR)
        targets = Group(person, plane)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.5).next_to(person, LEFT, LARGE_BUFF * 3)

        beam_bot = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            person.get_corner(DL),
            color=TX_COLOR,
        )
        beam_top = Line(
            radar.radome.get_right() + [0.1, 0, 0], plane.get_corner(UL), color=TX_COLOR
        )

        example = Group(radar.vgroup, beam_top, beam_bot, plane, person).move_to(ORIGIN)

        example_box = SurroundingRectangle(
            example,
            color=RED,
            corner_radius=0.2,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
        )
        example = Group(example_box, *example).set_z_index(3)

        self.play(GrowFromCenter(example_box))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(radar.vgroup),
                GrowFromCenter(person),
                GrowFromCenter(plane),
                AnimationGroup(Create(beam_top), Create(beam_bot)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(ShrinkToCenter(example))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        target_gap_cells = 14
        target_ref_cells = 18

        while ~n_gap_cells < target_gap_cells:
            n_gap_cells += 1
            self.wait(0.2, frozen_frame=False)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        target_gap_cells = 6

        while ~n_gap_cells > target_gap_cells:
            n_gap_cells -= 1
            self.wait(0.1, frozen_frame=False)

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        charvat_book = ImageMobject(
            "../props/static/charvat_radar_book_cover.jpg"
        ).scale_to_fit_height(config["frame_height"] * 0.7)
        purdue_article = ImageMobject(
            "./static/purdue_cfar_screenshot.png"
        ).scale_to_fit_width(config["frame_width"] * 0.3)
        arrow = Arrow(UP * 2, DOWN * 2)
        in_the_description = Tex("in the description").rotate(-PI / 2)
        resources = Group(
            charvat_book, purdue_article, arrow, in_the_description
        ).arrange(RIGHT, MED_SMALL_BUFF)
        resources_box = SurroundingRectangle(
            resources,
            color=RED,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
        )
        resources = Group(resources_box, *resources).set_z_index(3)

        self.play(GrowFromCenter(resources_box))
        self.play(
            LaggedStart(
                GrowFromCenter(charvat_book),
                GrowFromCenter(purdue_article),
                AnimationGroup(GrowArrow(arrow), FadeIn(in_the_description)),
                lag_ratio=0.35,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(resources))

        self.wait(0.5)

        self.play(
            gap_cells_label.animate.set_color(WHITE),
            bias_label.animate.set_color(YELLOW),
        )

        self.play(bias + 2)

        self.wait(0.5)

        self.play(bias - 1)

        self.wait(0.5)

        self.play(
            bias_label.animate.set_color(WHITE),
            ref_cells_label.animate.set_color(YELLOW),
        )

        # self.next_section(skip_animations=skip_animations(True))
        # self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         *[
        #             AnimationGroup(
        #                 l.animate(rate_func=rate_functions.there_and_back).scale(1.5),
        #                 r.animate(rate_func=rate_functions.there_and_back).scale(1.5),
        #             )
        #             for l, r in zip(ref_vert_boxes_l, ref_vert_boxes_r)
        #         ],
        #         lag_ratio=0.2,
        #     )
        # )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        while ~n_ref_cells < target_ref_cells:
            n_ref_cells += 1
            self.wait(0.2, frozen_frame=False)

        self.wait(0.5)

        target_ref_cells = 8

        while ~n_ref_cells > target_ref_cells:
            n_ref_cells -= 1
            self.wait(0.1, frozen_frame=False)

        self.wait(0.5)

        self.play(ref_cells_label.animate.set_color(WHITE))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(2)


class DesignerV2(Scene):
    def construct(self):
        stop_time = VT(16)
        fs = 1000

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = VT(-3)
        power_norm_2 = VT(-9)
        power_norm_3 = VT(0)

        noise_sigma_db = VT(3)

        f_max = 8
        y_min = VT(-30)

        x_len = VT(11)
        y_len = 5.5

        noise_seed = VT(2)

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -~y_min, -~y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=~x_len,
            y_length=y_len,
        ).to_edge(DOWN, MED_SMALL_BUFF)

        ax_label = ax.get_axis_labels(Tex("$R$"), Tex("$A$"))

        def plot_return():
            freq, X_k_log = get_plot_values(
                power_norm_1=~power_norm_1,
                power_norm_2=~power_norm_2,
                power_norm_3=~power_norm_3,
                ports=["1", "2", "3", "noise"],
                noise_power_db=~noise_sigma_db,
                noise_seed=~noise_seed,
                y_min=~y_min,
                f_max=f_max,
                fs=fs,
                stop_time=~stop_time,
                f1l=f1,
                f2l=f2,
                f3l=f3,
            ).values()
            f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")
            return ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)

        return_plot = always_redraw(plot_return)

        n_gap_cells = VT(8)
        n_ref_cells = VT(12)
        bias = VT(1)

        cell_size = VT(0.05)
        f_0 = ~cell_size * 2 * (~n_gap_cells + ~n_ref_cells)
        f = VT(f3)

        def plot_cfar():
            freq, X_k_log = get_plot_values(
                power_norm_1=~power_norm_1,
                power_norm_2=~power_norm_2,
                power_norm_3=~power_norm_3,
                ports=["1", "2", "3", "noise"],
                noise_power_db=~noise_sigma_db,
                noise_seed=~noise_seed,
                y_min=~y_min,
                f_max=f_max,
                fs=fs,
                stop_time=~stop_time,
                f1l=f1,
                f2l=f2,
                f3l=f3,
            ).values()
            return ax.plot(
                interpolate.interp1d(
                    freq,
                    cfar_fast(
                        X_k_log,
                        num_guard_cells=int(~n_gap_cells),
                        num_ref_cells=int(~n_ref_cells),
                        bias=~bias,
                    ),
                    fill_value="extrapolate",
                ),
                # x_range=[f_0, f_max - f_0, 1 / fs],
                x_range=[0, f_max, 1 / fs],
                color=REF_COLOR,
            )

        cfar_plot = always_redraw(plot_cfar)

        gap_label = Tex("Gap cells:")
        gap_slider = NumberLine(
            x_range=[1, 100, 10], length=config["frame_width"] * 0.5
        ).next_to(gap_label)
        ref_label = Tex("Ref cells:")
        ref_slider = NumberLine(
            x_range=[1, 100, 10], length=config["frame_width"] * 0.5
        ).next_to(ref_label)
        bias_label = Tex("Bias:")
        bias_slider = NumberLine(
            x_range=[1, 3, 1], length=config["frame_width"] * 0.5
        ).next_to(bias_label)

        sliders = (
            VGroup(
                VGroup(gap_label, ref_label, bias_label).arrange(
                    DOWN, SMALL_BUFF, aligned_edge=RIGHT
                ),
                VGroup(
                    VGroup(Tex("1"), gap_slider, Tex("100")).arrange(RIGHT, SMALL_BUFF),
                    VGroup(Tex("1"), ref_slider, Tex("100")).arrange(RIGHT, SMALL_BUFF),
                    VGroup(Tex("1"), bias_slider, Tex("3")).arrange(RIGHT, SMALL_BUFF),
                ).arrange(DOWN, SMALL_BUFF, aligned_edge=LEFT),
            )
            .arrange(RIGHT, SMALL_BUFF)
            .to_edge(UP)
        )

        gap_marker = always_redraw(lambda: Dot().move_to(gap_slider.n2p(~n_gap_cells)))
        ref_marker = always_redraw(lambda: Dot().move_to(ref_slider.n2p(~n_ref_cells)))
        bias_marker = always_redraw(lambda: Dot().move_to(bias_slider.n2p(~bias)))

        plot_group = VGroup(ax, ax_label, return_plot, cfar_plot)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(plot_group.shift(DOWN * 10).animate.set_y(0))

        self.wait(0.5)

        self.play(
            plot_group.animate.to_edge(DOWN, MED_SMALL_BUFF),
            FadeIn(sliders, shift=DOWN * 2),
        )

        self.wait(0.5)

        self.play(
            Create(gap_marker),
            Create(ref_marker),
            Create(bias_marker),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"cfar.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        self.play(notebook.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(notebook.animate.shift(DOWN * 5))
        self.remove(notebook)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        person = ImageMobject("../props/static/person.png").scale_to_fit_height(
            config["frame_height"] * 0.2
        )
        plane = SVGMobject("../props/static/plane.svg").scale(1.2).next_to(person, UR)
        targets = Group(person, plane)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.5).next_to(person, LEFT, LARGE_BUFF * 3)

        beam_bot = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            person.get_corner(DL),
            color=TX_COLOR,
        )
        beam_top = Line(
            radar.radome.get_right() + [0.1, 0, 0], plane.get_corner(UL), color=TX_COLOR
        )

        example = Group(radar.vgroup, beam_top, beam_bot, plane, person).move_to(ORIGIN)

        example_box = SurroundingRectangle(
            example,
            color=RED,
            corner_radius=0.2,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
        )
        example = Group(example_box, *example).set_z_index(3)

        self.play(GrowFromCenter(example_box))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(radar.vgroup),
                GrowFromCenter(person),
                GrowFromCenter(plane),
                AnimationGroup(Create(beam_top), Create(beam_bot)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(ShrinkToCenter(example))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(stop_time @ 8, run_time=2)

        self.wait(0.5)

        self.play(n_gap_cells @ 50, run_time=4)

        self.wait(0.5)

        self.play(stop_time @ 16, run_time=2)

        self.wait(0.5)

        self.play(n_gap_cells @ 8, run_time=2)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        charvat_book = ImageMobject(
            "../props/static/charvat_radar_book_cover.jpg"
        ).scale_to_fit_height(config["frame_height"] * 0.7)
        purdue_article = ImageMobject(
            "./static/purdue_cfar_screenshot.png"
        ).scale_to_fit_width(config["frame_width"] * 0.3)
        arrow = Arrow(UP * 2, DOWN * 2)
        in_the_description = Tex("in the description").rotate(-PI / 2)
        resources = Group(
            charvat_book, purdue_article, arrow, in_the_description
        ).arrange(RIGHT, MED_SMALL_BUFF)
        resources_box = SurroundingRectangle(
            resources,
            color=RED,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
        )
        resources = Group(resources_box, *resources).set_z_index(3)

        self.play(GrowFromCenter(resources_box))
        self.play(
            LaggedStart(
                GrowFromCenter(charvat_book),
                GrowFromCenter(purdue_article),
                AnimationGroup(GrowArrow(arrow), FadeIn(in_the_description)),
                lag_ratio=0.35,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(resources))

        self.wait(0.5)

        self.play(bias @ 2, run_time=2)

        self.wait(0.5)

        self.play(bias @ 1.5, run_time=2)

        self.wait(0.5)

        self.play(n_ref_cells @ 90, run_time=4)

        self.wait(0.5)

        self.play(n_ref_cells @ 12, run_time=2)

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class CFARMethods(MovingCameraScene):
    def construct(self):
        n_samples = 30
        sample_rects = (
            VGroup(
                *[
                    Square(
                        color=BLACK,
                        fill_color=BLUE,
                        fill_opacity=0,
                        stroke_width=DEFAULT_STROKE_WIDTH / 2,
                    )
                    for _ in range(n_samples)
                ]
            )
            .arrange(RIGHT, 0)
            .scale_to_fit_width(config["frame_width"] / 1.2)
        )

        sample_labels = VGroup(
            *[
                MathTex(f"A_{{{idx}}}")
                .scale_to_fit_width(rect.width * 0.7)
                .move_to(rect)
                for idx, rect in enumerate(sample_rects)
            ]
        )

        # self.add(sample_rects, sample_labels)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        GrowFromCenter(lb),
                        GrowFromCenter(rb),
                        GrowFromCenter(ll),
                        GrowFromCenter(rl),
                    )
                    for lb, rb, ll, rl in zip(
                        sample_rects[: n_samples // 2][::-1],
                        sample_rects[n_samples // 2 :],
                        sample_labels[: n_samples // 2][::-1],
                        sample_labels[n_samples // 2 :],
                    )
                ],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        cut_index = n_samples // 2 + n_samples // 6

        n_gap_cells = 3
        gap_index_l = cut_index - n_gap_cells
        gap_index_r = cut_index + n_gap_cells

        n_ref_cells = 4
        ref_index_l = gap_index_l - n_ref_cells
        ref_index_r = gap_index_r + n_ref_cells

        cut = sample_rects[cut_index]

        gap_cells_l = sample_rects[gap_index_l:cut_index]
        gap_cells_r = sample_rects[cut_index + 1 : gap_index_r + 1]

        ref_cells_l = sample_rects[ref_index_l:gap_index_l]
        ref_cells_r = sample_rects[gap_index_r + 1 : ref_index_r + 1]

        cut_label = (
            Tex("CUT", color=CUT_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(cut, UP, SMALL_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                sample_rects[: 1 + n_gap_cells * 2 + n_ref_cells * 2 + 2].width
            ).move_to(cut)
        )

        self.wait(0.5)

        gap_label_l = (
            Tex("Gap", color=GAP_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(gap_cells_l, UP, SMALL_BUFF)
        )
        gap_label_r = gap_label_l.copy().next_to(gap_cells_r, UP, SMALL_BUFF)

        ref_label_l = (
            Tex("Ref", color=REF_COLOR)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(ref_cells_l, UP, SMALL_BUFF)
        )
        ref_label_r = ref_label_l.copy().next_to(ref_cells_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                cut.animate.set_fill(CUT_COLOR, opacity=0.5),
                FadeIn(cut_label, shift=DOWN),
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(GAP_COLOR, opacity=0.5),
                        cell_r.animate.set_fill(GAP_COLOR, opacity=0.5),
                    )
                    for cell_l, cell_r in zip(gap_cells_l[::-1], gap_cells_r)
                ],
                FadeIn(gap_label_l, gap_label_r, shift=DOWN),
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(REF_COLOR, opacity=0.5),
                        cell_r.animate.set_fill(REF_COLOR, opacity=0.5),
                    )
                    for cell_l, cell_r in zip(ref_cells_l[::-1], ref_cells_r)
                ],
                FadeIn(ref_label_l, ref_label_r),
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        frame_shift = DOWN * (
            self.camera.frame.get_top()[1] - cut_label.get_y() - MED_LARGE_BUFF
        )

        font_size_zoomed = DEFAULT_FONT_SIZE * 0.5

        greatest = Tex("Greatest", font_size=font_size_zoomed * 1.4).next_to(
            self.camera.frame.copy().shift(frame_shift).get_bottom(), UP
        )
        smallest = Tex("Smallest", font_size=font_size_zoomed * 1.4).next_to(
            self.camera.frame.copy().shift(frame_shift).get_bottom(), UP
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(frame_shift),
                FadeIn(greatest, shift=UP),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ref_cell_labels_l = sample_labels[ref_index_l:gap_index_l]
        ref_cell_labels_r = sample_labels[gap_index_r + 1 : ref_index_r + 1]

        rcl_l_str = "[%s]" % ",".join([s.get_tex_string() for s in ref_cell_labels_l])
        rcl_l_fmt = MathTex(rcl_l_str, font_size=font_size_zoomed)
        rcl_r_str = "[%s]" % ",".join([s.get_tex_string() for s in ref_cell_labels_r])
        rcl_r_fmt = MathTex(rcl_r_str, font_size=font_size_zoomed)
        rcl_fmt = MathTex(
            f"{rcl_l_str}\\ {rcl_r_str}", font_size=font_size_zoomed
        ).move_to(self.camera.frame)

        Group(rcl_l_fmt, rcl_r_fmt).arrange(RIGHT, MED_LARGE_BUFF).move_to(
            self.camera.frame
        )

        # self.add(rcl_fmt, index_labels(rcl_fmt[0]).shift(DOWN))

        # self.play(
        #     LaggedStart(
        #         *[
        #             TransformFromCopy(m1, m2)
        #             for m1, m2 in zip(
        #                 [*ref_cell_labels_l, *ref_cell_labels_r],
        #                 [
        #                     rcl_fmt[0][1:4],
        #                     rcl_fmt[0][5:8],
        #                     rcl_fmt[0][9:12],
        #                     rcl_fmt[0][13:16],
        #                     rcl_fmt[0][18:21],
        #                     rcl_fmt[0][22:25],
        #                     rcl_fmt[0][26:29],
        #                     rcl_fmt[0][30:33],
        #                 ],
        #             )
        #         ]
        #     )
        # )

        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        FadeIn(rcl_l_fmt[0][idx * 4]),
                        TransformFromCopy(m, rcl_l_fmt[0][idx * 4 + 1 : idx * 4 + 4]),
                        lag_ratio=0.3,
                    )
                    for idx, m in enumerate(ref_cell_labels_l.set_z_index(-1))
                ],
                FadeIn(rcl_l_fmt[0][-1:]),
                *[
                    LaggedStart(
                        FadeIn(rcl_r_fmt[0][idx * 4]),
                        TransformFromCopy(m, rcl_r_fmt[0][idx * 4 + 1 : idx * 4 + 4]),
                        lag_ratio=0.3,
                    )
                    for idx, m in enumerate(ref_cell_labels_r.set_z_index(-1))
                ],
                FadeIn(rcl_r_fmt[0][-1:]),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rcl_l_sum_mean = MathTex(
            f"\\text{{mean}} \\left( {rcl_l_str} \\right),\\ ",
            font_size=font_size_zoomed,
        )
        rcl_r_sum_mean = MathTex(
            f"\\text{{mean}} \\left( {rcl_r_str} \\right)",
            font_size=font_size_zoomed,
        )
        Group(rcl_l_sum_mean, rcl_r_sum_mean).arrange(RIGHT, SMALL_BUFF).move_to(
            self.camera.frame
        )

        self.play(
            TransformByGlyphMap(
                rcl_l_fmt,
                rcl_l_sum_mean,
                ([], [0, 1, 2, 3], {"delay": 0.6, "shift": DOWN}),
                ([], [4], {"delay": 0.4, "shift": RIGHT}),
                (
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                ),
                ([], [22], {"delay": 0.4}),
                ([], [23], {"delay": 0.8}),
            ),
            TransformByGlyphMap(
                rcl_r_fmt,
                rcl_r_sum_mean,
                ([], [0, 1, 2, 3], {"delay": 0.6, "shift": DOWN}),
                ([], [4], {"delay": 0.4, "shift": RIGHT}),
                (
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                ),
                ([], [22], {"delay": 0.4}),
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        greatest_max = Tex("max", "$($", font_size=font_size_zoomed).next_to(
            rcl_l_sum_mean, LEFT, SMALL_BUFF
        )
        end_parenthesis = Tex("$)$", font_size=font_size_zoomed).next_to(
            rcl_r_sum_mean, RIGHT, SMALL_BUFF
        )

        greatest_eqn = VGroup(
            greatest_max,
            rcl_l_sum_mean.copy(),
            rcl_r_sum_mean.copy(),
            end_parenthesis,
        ).scale_to_fit_width(self.camera.frame.width * 0.9)

        greatest_min = (
            Tex("min", font_size=font_size_zoomed)
            .move_to(greatest_max[0])
            .match_width(greatest_max[0])
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(rcl_l_sum_mean, greatest_eqn[1]),
                    Transform(rcl_r_sum_mean, greatest_eqn[2]),
                ),
                AnimationGroup(
                    FadeIn(greatest_max, shift=RIGHT),
                    FadeIn(end_parenthesis, shift=LEFT),
                ),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Transform(greatest, smallest, path_arc=PI))

        self.wait(0.5)

        self.play(
            FadeOut(greatest_max[0], shift=DOWN),
            FadeIn(greatest_min.set_color(GREEN), shift=DOWN),
        )

        self.wait(0.5)

        self.play(greatest_min.animate.set_color(WHITE))

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        and_even_more = Tex(
            "Smallest", "... and even more", font_size=font_size_zoomed * 1.4
        ).move_to(smallest, aligned_edge=LEFT)

        self.play(Create(and_even_more[1]))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class Knobs(Scene):
    def construct(self):
        gap_label = Tex("Gap cells:")
        ref_label = Tex("Ref cells:")
        bias_label = Tex("Bias:")
        method_label = Tex("Method:")

        labels = VGroup(
            gap_label,
            ref_label,
            bias_label,
            method_label,
        ).arrange(DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)

        gap_slider = NumberLine(
            x_range=[1, 100, 10], length=config["frame_width"] * 0.5
        ).next_to(gap_label)
        ref_slider = NumberLine(
            x_range=[1, 100, 10], length=config["frame_width"] * 0.5
        ).next_to(ref_label)
        bias_slider = NumberLine(
            x_range=[1, 3, 1], length=config["frame_width"] * 0.5
        ).next_to(bias_label)
        curr_method = Tex("Cell averaging").next_to(method_label)

        values = VGroup(
            gap_slider,
            ref_slider,
            bias_slider,
            curr_method,
        )

        knobs = VGroup(labels, values).move_to(ORIGIN)

        gap = VT(50)
        ref = VT(20)
        bias = VT(1.5)
        gap_dot = always_redraw(lambda: Dot().move_to(gap_slider.n2p(~gap)))
        ref_dot = always_redraw(lambda: Dot().move_to(ref_slider.n2p(~ref)))
        bias_dot = always_redraw(lambda: Dot().move_to(bias_slider.n2p(~bias)))

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(FadeIn(label), Create(slider))
                    for label, slider in zip(labels, values)
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(Create(gap_dot), Create(ref_dot), Create(bias_dot))

        self.wait(0.5)

        not_perfect_label = Tex("not perfect", color=BLACK)
        not_perfect_box = SurroundingRectangle(
            not_perfect_label, stroke_opacity=0, fill_color=YELLOW, fill_opacity=1
        )
        not_perfect = Group(not_perfect_box, not_perfect_label).scale(2).rotate(PI / 6)

        pfa = MathTex(r"P_{FA} > 0", font_size=DEFAULT_FONT_SIZE * 1.7).to_edge(
            DOWN, MED_LARGE_BUFF
        )

        code = Code(
            "./cfar_example_code.py",
            background="window",
            background_stroke_color=WHITE,
            language="Python",
            tab_width=4,
            insert_line_no=True,
            style="paraiso-dark",
        ).scale_to_fit_width(config["frame_width"] * 0.95)
        for ln in code.line_numbers:
            ln.set_color(WHITE)

        self.next_section(skip_animations=skip_animations(True))
        for idx in range(10):
            if idx == 3:
                self.play(
                    gap @ randint(0, 100),
                    ref @ randint(0, 100),
                    bias @ (float(randrange(10, 30)) / 10),
                    Transform(
                        curr_method,
                        Tex(
                            [
                                "Greatest",
                                "Smallest",
                                "Cell averaging",
                                "Order statistic",
                            ][randrange(0, 4)]
                        ).move_to(curr_method, LEFT),
                    ),
                    GrowFromCenter(not_perfect),
                )
                continue
            if idx == 6:
                self.play(
                    gap @ randint(0, 100),
                    ref @ randint(0, 100),
                    bias @ (float(randrange(10, 30)) / 10),
                    Transform(
                        curr_method,
                        Tex(
                            [
                                "Greatest",
                                "Smallest",
                                "Cell averaging",
                                "Order statistic",
                            ][randrange(0, 4)]
                        ).move_to(curr_method, LEFT),
                    ),
                    FadeIn(pfa, shift=UP),
                )
                continue
            self.play(
                gap @ randint(0, 100),
                ref @ randint(0, 100),
                bias @ (float(randrange(10, 30)) / 10),
                Transform(
                    curr_method,
                    Tex(
                        [
                            "Greatest",
                            "Smallest",
                            "Cell averaging",
                            "Order statistic",
                        ][randrange(0, 4)]
                    ).move_to(curr_method, LEFT),
                ),
            )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(0.5)

        self.play(
            code.next_to([0, -config["frame_height"] / 2, 0], DOWN).animate.move_to(
                ORIGIN
            )
        )

        self.wait(0.5)

        stop_time = 16
        fs = 1000

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = -3
        power_norm_2 = -9
        power_norm_3 = 0

        noise_sigma_db = 3

        f_max = 8
        y_min = -30

        x_len = 11

        noise_seed = 2

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -y_min, -y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=(config["frame_height"] - code.height) * 0.8,
        )

        ax_label = ax.get_axis_labels(Tex("$R$"), Tex("$A$"))

        freq, X_k_log = get_plot_values(
            power_norm_1=power_norm_1,
            power_norm_2=power_norm_2,
            power_norm_3=power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=~noise_sigma_db,
            noise_seed=noise_seed,
            y_min=y_min,
            f_max=f_max,
            fs=fs,
            stop_time=stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
        ).values()

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)

        n_gap_cells = 8
        n_ref_cells = 12
        bias = 2

        cfar_plot = ax.plot(
            interpolate.interp1d(
                freq,
                cfar_fast(
                    X_k_log,
                    num_guard_cells=int(n_gap_cells),
                    num_ref_cells=int(n_ref_cells),
                    bias=bias,
                ),
                fill_value="extrapolate",
            ),
            x_range=[0, f_max, 1 / fs],
            color=REF_COLOR,
        )

        plot_group = Group(ax, return_plot, cfar_plot)

        self.play(
            code.animate.to_edge(UP, MED_SMALL_BUFF),
            plot_group.next_to([0, -config["frame_height"] / 2, 0], DOWN)
            .animate.to_edge(DOWN, MED_SMALL_BUFF)
            .set_x(0),
        )

        self.wait(0.5)

        self.play(
            code.animate.next_to([0, config["frame_height"] / 2, 0], UP),
            plot_group.animate.next_to([0, -config["frame_height"] / 2, 0], DOWN),
        )

        self.wait(2)


class InTheDescription(Scene):
    def construct(self):
        text_scale = 0.8

        thanks = Text("Thanks for watching!")

        resources = Text("resources").scale(text_scale)
        caveats = Text("caveats").scale(text_scale)
        source_code = Text(
            "// source code", font="FiraCode Nerd Font Mono", color=GRAY_C
        ).scale(text_scale)
        VGroup(resources, caveats, source_code).arrange(RIGHT, MED_LARGE_BUFF)

        resources_p2 = resources.get_bottom() + [0, -0.1, 0]
        caveats_p2 = caveats.get_bottom() + [0, -0.1, 0]
        source_code_p2 = source_code.get_bottom() + [0, -0.1, 0]

        tip = (
            Triangle(color=WHITE, fill_color=WHITE, fill_opacity=1)
            .scale(0.4)
            .to_edge(DOWN)
            .rotate(PI / 3)
        )
        tip_p1 = tip.get_top()

        resources_bez = CubicBezier(
            tip_p1,
            tip_p1 + [0, 1, 0],
            resources_p2 + [0, -1, 0],
            resources_p2,
        )
        caveats_bez = CubicBezier(
            tip_p1,
            tip_p1 + [0, 1, 0],
            caveats_p2 + [0, -1, 0],
            caveats_p2,
        )
        source_code_bez = CubicBezier(
            tip_p1,
            tip_p1 + [0, 1, 0],
            source_code_p2 + [0, -1, 0],
            source_code_p2,
        )

        self.play(GrowFromCenter(thanks))

        self.wait(0.5)

        self.play(thanks.animate.to_edge(UP))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(tip),
                Create(resources_bez),
                GrowFromCenter(resources),
                lag_ratio=0.4,
            )
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                Create(caveats_bez),
                GrowFromCenter(caveats),
                lag_ratio=0.4,
            )
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                Create(source_code_bez),
                GrowFromCenter(source_code),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class EndScreen(Scene):
    def construct(self):
        stats_title = Tex("Stats for Nerds")
        stats_table = (
            Table(
                [
                    ["Lines of code", "3,682"],
                    ["Script word count", "1,301"],
                    ["Days to make", "20"],
                    ["Git commits", "17"],
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


""" Testing """


class TargetSize(Scene):
    def construct(self):
        person = ImageMobject("../props/static/person.png").scale_to_fit_height(
            config["frame_height"] * 0.2
        )
        plane = SVGMobject("../props/static/plane.svg").scale(1.2).next_to(person, UR)
        targets = Group(person, plane)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.5).next_to(person, LEFT, LARGE_BUFF * 3)

        beam_bot = Line(
            radar.radome.get_right() + [0.1, 0, 0],
            person.get_corner(DL),
            color=TX_COLOR,
        )
        beam_top = Line(
            radar.radome.get_right() + [0.1, 0, 0], plane.get_corner(UL), color=TX_COLOR
        )

        example = Group(radar.vgroup, beam_top, beam_bot, plane, person).move_to(ORIGIN)

        example_box = SurroundingRectangle(
            example,
            color=RED,
            corner_radius=0.2,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
        )
        example = Group(example_box, *example).set_z_index(3)

        self.play(GrowFromCenter(example_box))

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(radar.vgroup),
                GrowFromCenter(person),
                GrowFromCenter(plane),
                AnimationGroup(Create(beam_top), Create(beam_bot)),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


class References(Scene):
    def construct(self):
        charvat_book = ImageMobject(
            "../props/static/charvat_radar_book_cover.jpg"
        ).scale_to_fit_height(config["frame_height"] * 0.7)
        purdue_article = ImageMobject(
            "./static/purdue_cfar_screenshot.png"
        ).scale_to_fit_width(config["frame_width"] * 0.3)
        arrow = Arrow(UP * 2, DOWN * 2)
        in_the_description = Tex("in the description").rotate(-PI / 2)
        resources = Group(
            charvat_book, purdue_article, arrow, in_the_description
        ).arrange(RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                GrowFromCenter(charvat_book),
                GrowFromCenter(purdue_article),
                AnimationGroup(GrowArrow(arrow), FadeIn(in_the_description)),
                lag_ratio=0.35,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(charvat_book, purdue_article, arrow, in_the_description))

        self.wait(2)


class NotebookReminder(Scene):
    def construct(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"cfar.ipynb \rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=myTemplate,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).to_edge(DOWN, MED_LARGE_BUFF)

        self.play(notebook.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(notebook.animate.shift(DOWN * 5))


""" Thumbnail """


class Thumbnail(MovingCameraScene):
    def construct(self):
        stop_time = 16
        fs = 1000

        f1 = 1.5
        f2 = 2.7
        f3 = 3.4

        power_norm_1 = -3
        power_norm_2 = -9
        power_norm_3 = 0

        noise_sigma_db = 3

        f_max = 8
        y_min = -30

        x_len = 30
        y_len = 5.5

        noise_seed = 2

        n_gap_cells = 3
        n_ref_cells = 4

        cell_size = 0.05
        f_0 = cell_size * 2 * (n_gap_cells + n_ref_cells)
        f = f1
        bias = 1

        ax = Axes(
            x_range=[0, f_max, f_max / 4],
            y_range=[0, -y_min, -y_min / 4],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(LEFT, LARGE_BUFF)

        freq, X_k_log = get_plot_values(
            power_norm_1=power_norm_1,
            power_norm_2=power_norm_2,
            power_norm_3=power_norm_3,
            ports=["1", "2", "3", "noise"],
            noise_power_db=noise_sigma_db,
            noise_seed=noise_seed,
            y_min=y_min,
            f_max=2 * f,
            fs=fs,
            stop_time=stop_time,
            f1l=f1,
            f2l=f2,
            f3l=f3,
        ).values()

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(f_X_k_log, x_range=[0, f_max, 1 / fs], color=RX_COLOR)

        cfar_plot = ax.plot(
            interpolate.interp1d(
                freq,
                cfar_fast(X_k_log, num_guard_cells=8, num_ref_cells=12, bias=bias),
                fill_value="extrapolate",
            ),
            x_range=[f_0, f, 1 / fs],
            color=REF_COLOR,
        )

        cfar_curr_dot = Dot().move_to(ax.input_to_graph_point(f, cfar_plot))

        plot_group = VGroup(
            ax,
            # ax_label,
            return_plot,
        )

        def get_sample_poly(idx, color, top_coord=None, bias: VT = None):
            def updater():
                mid = f + idx * cell_size * 2
                if top_coord is None:
                    top_coord_lc = ax.input_to_graph_coords(mid, return_plot)[1]
                else:
                    top_coord_lc = top_coord
                if bias is None:
                    bias_lc = 1
                else:
                    bias_lc = bias

                c = bias_lc if type(bias_lc) is VT else bias_lc
                top = ax.c2p(mid, top_coord_lc * c)[1]
                left_x = ax.input_to_graph_point(mid - cell_size, return_plot)[0]
                right_x = ax.input_to_graph_point(mid + cell_size, return_plot)[0]
                bot = ax.c2p(0, 0)[1]
                box = Polygon(
                    (left_x, top, 0),
                    (left_x, bot, 0),
                    (right_x, bot, 0),
                    (right_x, top, 0),
                    fill_color=color,
                    fill_opacity=0.5,
                    stroke_width=DEFAULT_STROKE_WIDTH / 2,
                )
                return box

            return updater

        cut_vert_box = always_redraw(get_sample_poly(0, CUT_COLOR))

        gap_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )
        gap_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, GAP_COLOR))
                for idx in range(1, n_gap_cells + 1)
            ]
        )

        ref_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )
        ref_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, REF_COLOR))
                for idx in range(1 + n_gap_cells, n_ref_cells + 1 + n_gap_cells)
            ]
        )

        self.add(
            # ax,
            return_plot,
            cut_vert_box,
            gap_vert_boxes_l,
            gap_vert_boxes_r,
            ref_vert_boxes_l,
            ref_vert_boxes_r,
            cfar_plot,
            cfar_curr_dot,
        )

        def camera_updater(m: Mobject):
            m.set_x(cut_vert_box.get_x())

        self.camera.frame.scale(0.9).set_x(cut_vert_box.get_x()).shift(UP / 2)
        # self.camera.frame.add_updater(camera_updater)

        rtd = Tex("Radar Target Detection", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            self.camera.frame.get_top(), DOWN, LARGE_BUFF
        )
        # self.add(rtd)

        target_or_noise = Tex(
            "Target", " or ", "Noise", "?", font_size=DEFAULT_FONT_SIZE * 2
        ).next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF)
        target_or_noise[0].set_color(GREEN)
        target_or_noise[2].set_color(RED)

        p1 = target_or_noise.get_corner(DR) + [-0.5, -0.1, 0]
        p2 = ax.input_to_graph_point(f + cell_size, return_plot) + [0.3, 0, 0]

        bez = CubicBezier(
            p1,
            p1 + [0, -1, 0],
            p2 + [1, 0, 0],
            p2,
        )
        bez_tip = (
            Triangle(stroke_color=WHITE, fill_color=WHITE, fill_opacity=1)
            .scale(0.1)
            .rotate(-PI / 6)
            .move_to(bez.get_end())
        )

        hidden_section = Rectangle(
            height=config["frame_height"],
            width=(config["frame_width"] / 2)
            - ax.input_to_graph_point(2 * f, return_plot)[0],
            stroke_opacity=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        ).next_to(self.camera.frame.get_right(), LEFT, 0)

        # arrow = Arrow(target_or_noise.get_bottom(), gpt)
        self.add(target_or_noise, bez, bez_tip, hidden_section)

        # self.play(f @ (8 - f_0), run_time=14, rate_func=rate_functions.ease_in_sine)
