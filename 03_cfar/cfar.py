# cfar.py

import sys
import warnings

from random import randint
import numpy as np
from manim import *
from MF_Tools import VT, TransformByGlyphMap
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
    N = fs * stop_time
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

        cut_color = YELLOW
        gap_color = RED
        ref_color = BLUE

        cut_index = num_samples // 2 + num_samples // 6

        n_gap_cells = 3
        gap_index_l = cut_index - n_gap_cells
        gap_index_r = cut_index + n_gap_cells

        n_ref_cells = 4
        ref_index_l = gap_index_l - n_ref_cells
        ref_index_r = gap_index_r + n_ref_cells

        cut = sample_rects[cut_index]

        gap_cells_l = sample_rects[gap_index_l:cut_index]
        gap_cells_r = sample_rects[cut_index + 1 : gap_index_r + 2]

        ref_cells_l = sample_rects[ref_index_l:gap_index_l]
        ref_cells_r = sample_rects[gap_index_r + 1 : ref_index_r + 2]

        cut_label_spelled = (
            Tex(r"Cell\\Under\\Test", color=cut_color)
            .scale_to_fit_width(sample_labels[0].width * 1.7)
            .next_to(cut, UP, SMALL_BUFF)
        )
        cut_label = (
            Tex("CUT", color=cut_color)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(cut, UP, SMALL_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                sample_rects[:9].width
            ).move_to(Group(cut, cut_label_spelled))
        )

        self.wait(0.5)

        self.play(
            FadeIn(cut_label_spelled),
            cut.animate.set_fill(color=cut_color, opacity=0.6),
        )

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
            Tex("Gap", color=gap_color)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(gap_cells_l, UP, SMALL_BUFF)
        )
        gap_label_r = gap_label_l.copy().next_to(gap_cells_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(gap_color, opacity=0.5),
                        cell_r.animate.set_fill(gap_color, opacity=0.5),
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
        ).next_to(self.camera.frame.get_bottom(), DOWN, LARGE_BUFF * 2)
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

        f_X_k_log = interpolate.interp1d(freq, X_k_log, fill_value="extrapolate")

        return_plot = ax.plot(
            f_X_k_log, x_range=[f3 - 1, f3 + 1, 1 / fs], color=RX_COLOR
        )

        plot_group = VGroup(ax, ax_label, return_plot).next_to(
            [0, -config["frame_height"] / 2, 0], DOWN
        )

        self.add(plot_group)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(ax.width * 1.2).move_to(ax)
        )

        self.wait(0.5)

        cell_size = VT(0.05)

        def get_sample_poly(idx, color):
            def updater():
                mid = f3 + idx * ~cell_size * 2
                top = ax.input_to_graph_point(mid, return_plot)[1]
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

        cut_vert_box = always_redraw(get_sample_poly(0, cut_color))

        gap_vert_boxes_l = VGroup(
            *[
                always_redraw(get_sample_poly(-idx, gap_color))
                for idx in range(1, n_gap_cells + 1)
            ]
        )
        gap_vert_boxes_r = VGroup(
            *[
                always_redraw(get_sample_poly(idx, gap_color))
                for idx in range(1, n_gap_cells + 1)
            ]
        )

        self.play(FadeIn(cut_vert_box))
        self.play(FadeIn(gap_vert_boxes_l, gap_vert_boxes_r))

        self.wait(0.5)

        self.play(cell_size @ 0.1, run_time=1.5)

        self.wait(0.5)

        self.play(cell_size @ 0.05, run_time=1.5)

        self.next_section(skip_animations=skip_animations(False))
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
            Tex("Ref", color=ref_color)
            .scale_to_fit_width(sample_labels[0].width * 1.5)
            .next_to(ref_cells_l, UP, SMALL_BUFF)
        )
        ref_label_r = ref_label_l.copy().next_to(ref_cells_r, UP, SMALL_BUFF)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        cell_l.animate.set_fill(ref_color, opacity=0.5),
                        cell_r.animate.set_fill(ref_color, opacity=0.5),
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

        self.wait(2)
