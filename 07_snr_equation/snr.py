# snr.py

import sys
import warnings
import random

from numpy.fft import fft, fftshift
from manim import *
import numpy as np
from MF_Tools import TransformByGlyphMap, VT
from scipy.interpolate import interp1d
from scipy import signal
from scipy.constants import c

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR
from props.block_diagram import get_blocks, get_diode
from props import VideoMobject

BLOCKS = get_blocks()

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


def get_transform_func(from_var, func=TransformFromCopy, path_arc=PI):
    def transform_func(m, **kwargs):
        return func(from_var, m, path_arc=path_arc, **kwargs)

    return transform_func


class EqnIntro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        # snr_eqn = MathTex(r"\frac{P_t G_t }")

        snr_eqn = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4 k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in snr_eqn[0]], lag_ratio=0.07))

        self.wait(0.5)

        fs = 100
        f = 25
        noise_std = VT(0.001)

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.5
        y_max = 30

        ax = Axes(
            x_range=[0, fs / 2, fs / 8],
            y_range=[0, y_max, 10],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        amp = VT(0)

        def fft_updater():
            np.random.seed(1)
            noise = np.random.normal(loc=0, scale=~noise_std, size=N)
            x_n = (~amp * np.sin(2 * PI * f * t) + noise) * signal.windows.blackman(N)
            X_k = fftshift(fft(x_n, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = np.clip(10 * np.log10(X_k) + y_max, 0, None)
            f_X_k_log = interp1d(freq, X_k, fill_value="extrapolate")

            plot = ax.plot(f_X_k_log, x_range=[0, fs / 2, 1 / 100], color=RX_COLOR)
            return plot

        X_k_plot = always_redraw(fft_updater)

        self.add(ax.next_to([0, -config.frame_height / 2, 0], DOWN))
        self.add(X_k_plot)

        self.play(
            ax.animate.to_edge(DOWN, MED_LARGE_BUFF),
            snr_eqn.animate.scale(0.7).to_edge(UP, MED_LARGE_BUFF),
        )

        self.wait(0.5)

        self.play(amp @ 1)

        self.wait(0.5)

        self.play(noise_std @ 0.2)

        self.wait(0.5)

        snr_line = always_redraw(
            lambda: Line(
                ax.c2p(30, 16.16),
                [
                    ax.c2p(30, 16.16)[0],
                    ax.input_to_graph_point(f, fft_updater())[1],
                    0,
                ],
            )
        )
        snr_line_u = always_redraw(
            lambda: Line(snr_line.get_top() + LEFT / 8, snr_line.get_top() + RIGHT / 8)
        )
        snr_line_d = always_redraw(
            lambda: Line(
                snr_line.get_bottom() + LEFT / 8, snr_line.get_bottom() + RIGHT / 8
            )
        )
        snr_label = always_redraw(lambda: Tex("SNR").next_to(snr_line, RIGHT))
        snr_db_label = MathTex(
            f" = {ax.input_to_graph_coords(f, fft_updater())[1] -16.16:.2f} \\text{{ dB}}"
        ).next_to(snr_label)

        def snr_updater(m):
            m.become(
                MathTex(
                    f" = {ax.input_to_graph_coords(f, fft_updater())[1] -16.16:.2f} \\text{{ dB}}"
                ).next_to(snr_label)
            )

        self.play(
            Create(snr_line),
            Create(snr_line_u),
            Create(snr_line_d),
            TransformFromCopy(snr_eqn[0][:3], snr_label[0]),
        )
        self.next_section(skip_animations=skip_animations(True))
        self.add(snr_label)
        self.play(FadeIn(snr_db_label))
        snr_db_label.add_updater(snr_updater)

        self.wait(0.5, frozen_frame=False)

        self.play(amp @ 0.15, run_time=3)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5, frozen_frame=False)

        self.play(
            ax.animate.next_to([0, -config.frame_height / 2, 0], DOWN),
            snr_eqn.animate.scale(1 / 0.7).move_to(ORIGIN),
        )
        self.remove(
            ax, X_k_plot, snr_line, snr_line_u, snr_line_d, snr_label, snr_db_label
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        snr_eqn_split = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )

        self.play(
            TransformByGlyphMap(
                snr_eqn,
                snr_eqn_split,
                ([0, 1, 2, 3], [0, 1, 2, 3]),
                ([4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10]),
                ([11], [11]),
                ([12, 13, 14, 15, 16, 17, 18], [12, 13, 14, 15, 16, 17, 18]),
                (GrowFromCenter, [19], {"delay": 0.2}),
                ([12, 13, 14, 15, 16, 17, 18], [12, 13, 14, 15, 16, 17, 18]),
                (GrowFromCenter, [20], {"delay": 0.4}),
                (GrowFromCenter, [21], {"delay": 0.6}),
                ([19, 20, 21, 22, 23, 24], [22, 23, 24, 25, 26, 27]),
            )
        )

        self.wait(0.5)

        self.play(
            snr_eqn_split[0][0].animate.set_color(BLUE),
            snr_eqn_split[0][4:19].animate.set_color(BLUE),
        )

        self.wait(0.5)

        self.play(
            snr_eqn_split[0][1].animate.set_color(RED),
            snr_eqn_split[0][22:].animate.set_color(RED),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        notebook_1_img = ImageMobject("./static/notebook_top.png")
        notebook_2_img = ImageMobject("./static/notebook_2.png")
        notebook_3_img = ImageMobject("./static/notebook_3.png")
        notebook_1_box = SurroundingRectangle(notebook_1_img, buff=0, fill_opacity=0)
        notebook_2_box = SurroundingRectangle(notebook_2_img, buff=0, fill_opacity=0)
        notebook_3_box = SurroundingRectangle(notebook_3_img, buff=0, fill_opacity=0)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"radar\_cheatsheet.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=YELLOW, fill_color=BACKGROUND_COLOR, fill_opacity=0
        )
        notebook_label = (
            Group(notebook_reminder, notebook_box)
            .to_edge(DOWN, LARGE_BUFF)
            .shift(RIGHT * config.frame_width)
        )

        notebook_1_img.scale_to_fit_height(config.frame_height * 0.5).shift(
            RIGHT * config.frame_width
        ).next_to(notebook_reminder, UP)
        notebook_2_img.scale_to_fit_height(config.frame_height * 0.5).shift(
            RIGHT * config.frame_width
        ).next_to(notebook_reminder, UP)
        notebook_3_img.scale_to_fit_height(config.frame_height * 0.5).shift(
            RIGHT * config.frame_width
        ).next_to(notebook_reminder, UP)

        self.add(notebook_1_img)

        self.play(self.camera.frame.animate.shift(RIGHT * config.frame_width))

        self.wait(0.5)

        self.play(notebook_label.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(
            notebook_1_img.animate.shift(LEFT * config.frame_width),
            notebook_2_img.shift(RIGHT * config.frame_width).animate.shift(
                LEFT * config.frame_width
            ),
        )

        self.wait(0.5)

        self.play(
            notebook_2_img.animate.shift(LEFT * config.frame_width),
            notebook_3_img.shift(RIGHT * config.frame_width).animate.shift(
                LEFT * config.frame_width
            ),
        )
        self.remove(notebook_1_img, notebook_2_img)

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(LEFT * config.frame_width))

        self.wait(2)


class Signal(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        snr_eqn = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        snr_eqn[0][0].set_color(BLUE)
        snr_eqn[0][4:19].set_color(BLUE)
        snr_eqn[0][1].set_color(RED)
        snr_eqn[0][22:].set_color(RED)

        radar_eqn = MathTex(
            r"P_r = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).scale(1.5)
        radar_eqn[0][3:12].set_color(GREEN)
        radar_eqn[0][13:19].set_color(BLUE)
        radar_eqn[0][20:22].set_color(YELLOW)

        radar_eqn_title = Tex(
            "The Radar Range Equation", font_size=DEFAULT_FONT_SIZE * 1.8
        ).to_edge(UP, MED_LARGE_BUFF)

        radar_eqn_group = (
            Group(radar_eqn_title, radar_eqn)
            .arrange(DOWN, LARGE_BUFF * 1.5)
            .scale_to_fit_width(config.frame_width * 0.3)
        )

        title_1 = Tex(r"Animated Radar Cheatsheet\\Episode 1")

        thumbnail_1_box = SurroundingRectangle(radar_eqn_group, buff=MED_LARGE_BUFF)
        radar_eqn_group.add(
            thumbnail_1_box, title_1.next_to(thumbnail_1_box, DOWN)
        ).next_to([config.frame_width / 2, 0, 0], RIGHT)

        self.add(snr_eqn)

        self.play(FadeOut(snr_eqn[0][:4], snr_eqn[0][-9:]))

        self.wait(0.5)

        signal_eqn = snr_eqn[0][4:-9]

        self.play(
            Group(signal_eqn, radar_eqn_group).animate.arrange(RIGHT, LARGE_BUFF * 1.5)
        )

        self.wait(0.5)

        both_eqn = MathTex(
            r"P_r = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        both_eqn[0][-19:-10].set_color(GREEN)
        both_eqn[0][-9:-3].set_color(BLUE)
        both_eqn[0][-2:].set_color(YELLOW)

        self.play(
            LaggedStart(
                FadeOut(thumbnail_1_box, title_1, radar_eqn_title),
                ShrinkToCenter(radar_eqn[0][2]),
                Transform(radar_eqn[0][:2], both_eqn[0][:2], path_arc=PI / 2),
                GrowFromCenter(both_eqn[0][2]),
                signal_eqn.animate.move_to(both_eqn[0][3:18]),
                GrowFromCenter(both_eqn[0][18]),
                Transform(radar_eqn[0][3:], both_eqn[0][-19:]),
                lag_ratio=0.3,
            ),
            run_time=2.5,
        )

        self.wait(0.5)

        ae_eqn = MathTex(
            r"A_e = \frac{G \lambda^2}{4 \pi}",
            color=YELLOW,
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).to_edge(DOWN, LARGE_BUFF)

        self.play(
            TransformFromCopy(both_eqn[0][-2:], ae_eqn[0][:2], path_arc=PI / 3),
            Group(radar_eqn, signal_eqn, both_eqn[0][2], both_eqn[0][18]).animate.shift(
                UP
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in ae_eqn[0][2:]],
                lag_ratio=0.1,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        both_eqn_simplified = MathTex(
            r"P_r = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4}",
            color=BLUE,
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        both_eqn_simplified[0][-15:].move_to(radar_eqn[0][3:], LEFT)

        self.play(
            TransformByGlyphMap(
                # both_eqn[0][-19:],
                radar_eqn[0][3:],
                both_eqn_simplified[0][-15:],
                ([0, 1], [0, 1], {"delay": 0}),
                ([2], [2], {"delay": 0.1}),
                ([3], ShrinkToCenter, {"delay": 0.2}),
                ([4], [7], {"delay": 0.3}),
                ([9], ShrinkToCenter, {"delay": 0.4}),
                ([16], ShrinkToCenter, {"delay": 0.5}),
                ([17, 18], ShrinkToCenter, {"delay": 0.6}),
                ([10], [6], {"delay": 0.7}),
                ([7, 14], [13], {"path_arc": PI / 3, "delay": 0.8}),
                ([8, 15], [14], {"path_arc": PI / 3, "delay": 0.9}),
                ([11], ShrinkToCenter, {"delay": 0.2}),
                ([5, 6], [9, 10], {"delay": 1.2}),
                ([12, 13], [9, 10], {"path_arc": -PI / 2, "delay": 1.8}),
                (
                    get_transform_func(ae_eqn[0][-2:], ReplacementTransform, PI / 3),
                    [9, 10],
                    {"delay": 1.6},
                ),
                (
                    get_transform_func(ae_eqn[0][3], ReplacementTransform, PI / 3),
                    [2],
                    {"delay": 1.1},
                ),
                (GrowFromCenter, [3], {"delay": 1.3}),
                (
                    get_transform_func(ae_eqn[0][4:6], ReplacementTransform, PI),
                    [4, 5],
                    {"delay": 1.2},
                ),
                (GrowFromCenter, [8], {"delay": 1.3}),
                (GrowFromCenter, [11], {"delay": 1.4}),
                (GrowFromCenter, [12], {"delay": 1.5}),
                mobA_submobject_index=[],
                mobB_submobject_index=[],
            ),
            FadeOut(ae_eqn[0][:3], ae_eqn[0][6]),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            signal_eqn.animate.move_to(ORIGIN),
            FadeOut(
                both_eqn_simplified[0][-15:],
                radar_eqn[0][:2],
                both_eqn[0][2],
                both_eqn[0][18],
            ),
            # radar_eqn_group.animate.next_to([config.frame_width / 2, 0, 0], RIGHT),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                signal_eqn[:2]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[2:4]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[4:6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[-2:]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(DOWN / 2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(signal_eqn[:2].animate.set_color(YELLOW))

        x_len = config.frame_width * 0.5
        y_len = config.frame_height * 0.5
        ax = Axes(
            x_range=[0, 1, 1 / 4],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(RIGHT, LARGE_BUFF)

        amp = VT(1)

        plot = always_redraw(
            lambda: ax.plot(lambda t: ~amp * np.sin(2 * PI * 3 * t), color=TX_COLOR)
        )
        self.add(ax)
        self.add(plot)

        self.play(
            Create(ax),
            Create(plot),
            signal_eqn.animate.to_edge(LEFT, LARGE_BUFF),
        )

        self.wait(0.5)

        self.play(amp @ 2)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(Group(signal_eqn, ax).animate.to_edge(UP, LARGE_BUFF))

        self.wait(0.5)

        pt = (
            MathTex("P_t", font_size=DEFAULT_FONT_SIZE * 2, color=YELLOW)
            .to_edge(DOWN, LARGE_BUFF)
            .set_x(signal_eqn.get_x())
        )
        pt_up = Arrow(pt.get_bottom(), pt.get_top(), buff=0).next_to(pt)
        pt_group = Group(pt, pt_up)

        cost = Tex(r"\$", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            pt_up, RIGHT, LARGE_BUFF * 2
        )
        cost_up = Arrow(cost.get_bottom(), cost.get_top(), buff=0).next_to(cost)
        cost_group = Group(cost, cost_up)

        power_icon = (
            SVGMobject("../props/static/lightning.svg")
            .set_fill(YELLOW)
            .set_color(YELLOW)
        )
        power_circle = Circle(power_icon.get_height() * 1, color=WHITE).move_to(
            power_icon
        )
        power = (
            Group(power_icon, power_circle)
            .scale_to_fit_height(cost.height * 1.1)
            .next_to(cost_up, RIGHT, LARGE_BUFF * 2)
        )
        power_up = Arrow(power.get_bottom(), power.get_top(), buff=0).next_to(power)
        power_group = Group(power, power_up)

        heat = (
            ImageMobject("../props/static/fire.png")
            .scale_to_fit_height(power.get_height())
            .next_to(power_up, RIGHT, LARGE_BUFF * 2)
        )
        heat_up = Arrow(heat.get_bottom(), heat.get_top(), buff=0).next_to(heat)
        heat_group = Group(heat, heat_up)

        pt_tradeoffs = (
            Group(pt_group, cost_group, power_group, heat_group)
            .arrange(RIGHT, LARGE_BUFF * 2)
            .to_edge(DOWN, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                TransformFromCopy(signal_eqn[:2], pt[0], path_arc=-PI / 3),
                GrowArrow(pt_up),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                cost.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(cost_up),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                power.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(power_up),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                heat.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(heat_up),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            pt_tradeoffs.animate.shift(DOWN * 6),
            ax.animate.shift(RIGHT * 10),
            signal_eqn.animate.set_y(0),
        )
        self.play(signal_eqn[:2].animate.set_color(BLUE))

        self.remove(pt_tradeoffs, ax, plot)

        self.wait(0.5)

        self.play(signal_eqn[2:4].animate.set_color(YELLOW))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(0) for _ in range(n_elem)]
        weight_trackers[n_elem // 2] @= 1
        for wt in [
            *weight_trackers[n_elem // 2 : n_elem // 2 + n_elem // 4],
            *weight_trackers[n_elem // 2 - n_elem // 4 : n_elem // 2],
        ]:
            wt @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30

        AF_scale = VT(1)

        x_len = config.frame_width * 0.6
        af_ax = (
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
            .next_to([config.frame_width / 2, 0, 0], RIGHT)
        )

        theta_min = VT(-PI / 2)
        theta_max = VT(PI / 2)

        def get_af():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            AF_log = np.clip(20 * np.log10(np.abs(AF)) - r_min, 0, None) * ~AF_scale
            f_AF = interp1d(u * PI, AF_log, fill_value="extrapolate")
            plot = af_ax.plot_polar_graph(
                r_func=f_AF,
                theta_range=[~theta_min, ~theta_max, 2 * PI / 200],
                color=TX_COLOR,
            )
            return plot

        AF_plot = always_redraw(get_af)
        self.add(AF_plot)

        self.play(af_ax.animate.to_edge(RIGHT, LARGE_BUFF), run_time=2)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        car = (
            SVGMobject("../props/static/car.svg", fill_color=WHITE, stroke_color=WHITE)
            .scale_to_fit_width(config.frame_width * 0.2)
            .to_edge(RIGHT)
        )

        self.play(
            af_ax.animate.shift(LEFT * 3),
            car.shift(RIGHT * 5).animate.shift(LEFT * 5),
        )

        self.wait(0.5)

        up_ang = VT(0)
        dn_ang = VT(0)

        directivity_arrow = always_redraw(
            lambda: Arrow(
                af_ax.c2p(0, 0), af_ax.input_to_graph_point(~up_ang, AF_plot), buff=0
            )
        )

        self.play(GrowArrow(directivity_arrow))

        self.wait(0.5)

        directivity_arrow_dn = always_redraw(
            lambda: Arrow(
                af_ax.c2p(0, 0), af_ax.input_to_graph_point(~dn_ang, AF_plot), buff=0
            )
        )
        self.add(directivity_arrow_dn)

        self.play(up_ang @ (-PI / 2), dn_ang @ (PI / 2), run_time=5)

        self.wait(0.5)

        self.play(
            ShrinkToCenter(directivity_arrow_dn),
            ShrinkToCenter(directivity_arrow),
            af_ax.animate.shift(RIGHT * 3),
            car.animate.shift(RIGHT * 5),
        )
        self.remove(car)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        n_elem_disp = 8
        phased_array = (
            Group(*[Square(side_length=0.6) for _ in range(n_elem_disp)])
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(af_ax.c2p(0, 0), LEFT)
        )

        self.play(
            # theta_min @ (-PI / 2),
            # theta_max @ (PI / 2),
            LaggedStart(
                *[
                    AnimationGroup(GrowFromCenter(m_up), GrowFromCenter(m_down))
                    for m_up, m_down in zip(
                        phased_array[2 : len(phased_array) // 2][::-1],
                        phased_array[len(phased_array) // 2 : -2],
                    )
                ],
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(GrowFromCenter(m_up), GrowFromCenter(m_down))
                    for m_up, m_down in zip(
                        phased_array[:2][::-1],
                        phased_array[-2:],
                    )
                ],
                lag_ratio=0.3,
            ),
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 4][::-1][n] @ 1,
                        weight_trackers[-n_elem // 4 :][n] @ 1,
                    )
                    for n in range(n_elem // 4)
                ],
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        size = MathTex(r"m^2", font_size=DEFAULT_FONT_SIZE * 2)
        size_up = Arrow(size.get_bottom(), size.get_top(), buff=0).next_to(size)
        size_group = Group(size, size_up)

        cost = Tex(r"\$", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            size_up, RIGHT, LARGE_BUFF * 2
        )
        cost_up = Arrow(cost.get_bottom(), cost.get_top(), buff=0).next_to(cost)
        cost_group = Group(cost, cost_up)

        power_icon = (
            SVGMobject("../props/static/lightning.svg")
            .set_fill(YELLOW)
            .set_color(YELLOW)
        )
        power_circle = Circle(power_icon.get_height() * 1, color=WHITE).move_to(
            power_icon
        )
        power = (
            Group(power_icon, power_circle)
            .scale_to_fit_height(cost.height * 1.1)
            .next_to(cost_up, RIGHT, LARGE_BUFF * 2)
        )
        power_up = Arrow(power.get_bottom(), power.get_top(), buff=0).next_to(power)
        power_group = Group(power, power_up)

        gt_tradeoffs = (
            Group(size_group, cost_group, power_group)
            .arrange(RIGHT, LARGE_BUFF)
            .to_corner(DL, MED_LARGE_BUFF)
            .shift(UP / 2)
        )

        self.play(
            LaggedStart(
                size.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(size_up),
                cost.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(cost_up),
                power.shift(DOWN * 5).animate.shift(UP * 5),
                GrowArrow(power_up),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        ShrinkToCenter(phased_array[: n_elem_disp // 2][::-1][n]),
                        ShrinkToCenter(phased_array[n_elem_disp // 2 :][n]),
                    )
                    for n in range(n_elem_disp // 2)
                ],
                lag_ratio=0.3,
            ),
            AF_scale @ 0,
            gt_tradeoffs.animate.shift(DOWN * 6),
            signal_eqn[2:4].animate.set_color(BLUE),
        )
        self.remove(af_ax, AF_plot, gt_tradeoffs)

        self.wait(0.5)

        self.play(signal_eqn[4:6].animate.set_color(YELLOW))

        self.wait(0.5)

        wavelength_eqn = MathTex(
            r"\lambda = \frac{c}{f}", font_size=DEFAULT_FONT_SIZE * 2
        ).to_edge(UP, LARGE_BUFF)
        wavelength_eqn[0][0].set_color(YELLOW)

        f_relation = MathTex(r"f \sim", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            wavelength_eqn, DOWN, LARGE_BUFF, LEFT
        )
        antenna = BLOCKS.get("antenna").copy().next_to(f_relation, RIGHT)

        self.play(
            LaggedStart(
                TransformFromCopy(signal_eqn[4], wavelength_eqn[0][0], path_arc=PI / 3),
                *[GrowFromCenter(m) for m in wavelength_eqn[0][1:]],
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        range_ax = (
            Axes(
                x_range=[1, 4, 1],
                y_range=[0, 1, 0.5],
                tips=False,
                axis_config={"include_numbers": False},
                x_length=config.frame_width * 0.6,
                y_length=config.frame_height * 0.4,
            )
            .to_corner(DR, LARGE_BUFF)
            .shift(DOWN * 7)
        )

        range_vt = VT(1)
        f = 4

        sine_graph = always_redraw(
            lambda: range_ax.plot(
                lambda r: (np.sin(2 * PI * f * r) + 1) / 2 / (r**2),
                x_range=[1, ~range_vt, 1 / 100],
                use_smoothing=False,
                color=TX_COLOR,
            )
        )
        exp_graph = always_redraw(
            lambda: range_ax.plot(
                lambda r: 1 / (r**2),
                x_range=[1, ~range_vt, 1 / 100],
                use_smoothing=False,
                color=YELLOW,
            )
        )

        range_arrow = always_redraw(
            lambda: Arrow(range_ax.c2p(1, 0) + DOWN, range_ax.c2p(~range_vt, 0) + DOWN)
        )
        range_label = always_redraw(
            lambda: MathTex("R").move_to(range_ax.c2p(~range_vt, 0) + DOWN)
        )
        power_dot = always_redraw(
            lambda: Dot().move_to(range_ax.input_to_graph_point(~range_vt, exp_graph))
        )
        power_label = always_redraw(
            lambda: MathTex("P").move_to(
                range_ax.input_to_graph_point(~range_vt, exp_graph) + UP / 2
            )
        )

        self.add(
            range_ax,
            sine_graph,
            exp_graph,
            range_arrow,
            range_label,
            power_dot,
            power_label,
        )

        self.play(range_ax.animate.to_corner(DR, MED_LARGE_BUFF).shift(UP))

        self.wait(0.5)

        self.play(range_vt @ 4, run_time=2)

        self.wait(0.5)

        self.play(range_ax.animate.shift(DOWN * 7))

        self.remove(
            range_ax,
            sine_graph,
            exp_graph,
            range_arrow,
            range_label,
            power_dot,
            power_label,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                TransformFromCopy(wavelength_eqn[0][-1], f_relation[0][0]),
                GrowFromCenter(f_relation[0][1]),
                GrowFromCenter(antenna),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    ShrinkToCenter(m)
                    for m in [*wavelength_eqn[0], *f_relation[0], antenna]
                ],
                lag_ratio=0.1,
            ),
            signal_eqn[4:6].animate.set_color(BLUE),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(signal_eqn[8:13].animate.set_color(YELLOW))

        self.wait(0.5)

        circle = Circle(color=BLUE, radius=config.frame_width * 0.15)
        line_bot = Line(circle.get_left(), circle.get_right(), path_arc=PI / 3)
        line_top = DashedLine(circle.get_left(), circle.get_right(), path_arc=-PI / 3)
        sphere = (
            Group(circle, line_bot, line_top).to_edge(RIGHT, LARGE_BUFF).shift(DOWN)
        )

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{fontawesome5}")

        sa_sphere = Tex(
            r"$4 \pi r^2$",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        ).next_to(sphere, UP)

        sa_lock = Tex(
            r"\faLock ", tex_template=tex_template, font_size=DEFAULT_FONT_SIZE * 1.8
        ).next_to(sa_sphere, LEFT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                Create(circle),
                Create(line_bot),
                Create(line_top),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in sa_sphere[0]], lag_ratio=0.1)
        )

        self.wait(0.5)

        self.play(GrowFromCenter(sa_lock))

        self.wait(0.5)

        radar_eqn = MathTex(
            r"P_r = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).scale(1.5)
        radar_eqn[0][3:12].set_color(GREEN)
        radar_eqn[0][13:19].set_color(BLUE)
        radar_eqn[0][20:22].set_color(YELLOW)

        radar_eqn_title = Tex(
            "The Radar Range Equation", font_size=DEFAULT_FONT_SIZE * 1.8
        ).to_edge(UP, MED_LARGE_BUFF)

        radar_eqn_group = (
            Group(radar_eqn_title, radar_eqn)
            .arrange(DOWN, LARGE_BUFF * 1.5)
            .scale_to_fit_width(config.frame_width * 0.3)
        )

        title_1 = Tex(r"Animated Radar Cheatsheet\\Episode 1")

        thumbnail_1_box = SurroundingRectangle(radar_eqn_group, buff=MED_LARGE_BUFF)
        radar_eqn_group.add(
            thumbnail_1_box, title_1.next_to(thumbnail_1_box, DOWN)
        ).to_edge(RIGHT, LARGE_BUFF).shift(DOWN * 10)

        self.play(
            Group(sphere, sa_sphere, sa_lock).animate.shift(UP * 10),
            radar_eqn_group.animate.set_y(0),
        )

        self.wait(0.5)

        self.play(
            radar_eqn_group.animate.shift(UP * 10),
            signal_eqn[8:13].animate.set_color(BLUE),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(signal_eqn[6].animate.set_color(YELLOW))

        self.wait(0.5)

        self.play(signal_eqn[-2:].animate.set_color(YELLOW))

        self.wait(0.5)

        related_to_target = MathTex(
            r"\sigma,\ R \propto ", font_size=DEFAULT_FONT_SIZE * 1.8
        )
        related_to_target[0][0].set_color(YELLOW)
        related_to_target[0][2].set_color(YELLOW)

        car = (
            SVGMobject("../props/static/car.svg", fill_color=WHITE, stroke_color=WHITE)
            .scale_to_fit_width(config.frame_width * 0.2)
            .to_edge(RIGHT, MED_LARGE_BUFF)
            .set_y(related_to_target.get_y())
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    signal_eqn[6], related_to_target[0][0], path_arc=PI / 2
                ),
                GrowFromCenter(related_to_target[0][1]),
                TransformFromCopy(
                    signal_eqn[-2], related_to_target[0][2], path_arc=-PI / 2
                ),
                GrowFromCenter(related_to_target[0][3]),
                car.shift(RIGHT * 5).animate.shift(LEFT * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        car_line_up = Line(car.get_bottom(), car.get_top()).next_to(car, LEFT)
        car_line_top = Line(
            car_line_up.get_top() + LEFT / 8, car_line_up.get_top() + RIGHT / 8
        )
        car_line_bot = Line(
            car_line_up.get_bottom() + LEFT / 8, car_line_up.get_bottom() + RIGHT / 8
        )

        self.play(
            LaggedStart(
                Create(car_line_bot),
                Create(car_line_up),
                Create(car_line_top),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        car_line_side = Line(
            car.get_left(), related_to_target.get_right() + RIGHT / 2
        ).next_to(car, LEFT)
        car_line_left = Line(
            car_line_side.get_left() + DOWN / 8, car_line_side.get_left() + UP / 8
        )
        car_line_right = Line(
            car_line_side.get_right() + DOWN / 8, car_line_side.get_right() + UP / 8
        )

        self.play(
            ReplacementTransform(car_line_top, car_line_left),
            ReplacementTransform(car_line_up, car_line_side),
            ReplacementTransform(car_line_bot, car_line_right),
        )

        self.wait(0.5)

        cloud = SVGMobject(
            "../props/static/clouds.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).next_to(car, UP, LARGE_BUFF * 5)

        car_x = car.get_x()
        targets_group_copy = (
            Group(cloud.copy(), car.copy()).arrange(DOWN, LARGE_BUFF).set_x(car_x)
        )

        target_bez_top = CubicBezier(
            related_to_target.get_right() + [0.1, 0, 0],
            related_to_target.get_right() + [1, 0, 0],
            targets_group_copy.get_corner(UL) + [-1, 0, 0],
            targets_group_copy.get_corner(UL) + [-0.1, 0, 0],
        )
        target_bez_bot = CubicBezier(
            related_to_target.get_right() + [0.1, 0, 0],
            related_to_target.get_right() + [1, 0, 0],
            targets_group_copy.get_corner(DL) + [-1, 0, 0],
            targets_group_copy.get_corner(DL) + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                FadeOut(car_line_side, car_line_left, car_line_right),
                AnimationGroup(
                    Group(cloud, car).animate.arrange(DOWN, LARGE_BUFF).set_x(car_x),
                    Create(target_bez_bot),
                    Create(target_bez_top),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            signal_eqn[6].animate.set_color(BLUE),
            signal_eqn[-2:].animate.set_color(BLUE),
            LaggedStart(
                *[
                    ShrinkToCenter(m)
                    for m in [
                        *related_to_target[0],
                        target_bez_top,
                        target_bez_bot,
                        cloud,
                        car,
                    ]
                ],
                lag_ratio=0.05,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        snr_eqn_new = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        signal_eqn_new = snr_eqn_new[0][4:-9]

        self.play(signal_eqn.animate.move_to(signal_eqn_new))

        self.wait(0.5)

        self.play(
            LaggedStart(
                signal_eqn[:2]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[2:4]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[4:6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[6]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 2),
                signal_eqn[-2:]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(DOWN / 2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            snr_eqn[0][:4].shift(LEFT * 10).animate.shift(RIGHT * 10),
            snr_eqn[0][-9:].shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        noise = MathTex("k T_s B_n L", color=RED, font_size=DEFAULT_FONT_SIZE * 1.8)

        self.play(
            LaggedStart(
                snr_eqn[0][:4].animate.shift(LEFT * 10),
                signal_eqn.animate.shift(LEFT * 12),
                ShrinkToCenter(snr_eqn[0][-9]),
                ShrinkToCenter(snr_eqn[0][-8]),
                ShrinkToCenter(snr_eqn[0][-7]),
                snr_eqn[0][-6:].animate.move_to(noise[0]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(snr_eqn[0][-6:].animate.scale(1.5))

        self.wait(2)


class Noise(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))
        noise = MathTex(
            "k T_s B_n L", color=RED, font_size=DEFAULT_FONT_SIZE * 1.8
        ).scale(1.5)

        noise_def = MathTex(
            r"&k\  \text{| Boltzmann's constant} \\ &T_s\  \text{| System temperature} \\ &B_n\  \text{| Receiver bandwidth} \\ &L\  \text{| Loss}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).to_edge(DOWN, LARGE_BUFF)
        noise_def[0][0].set_color(RED)
        noise_def[0][21:23].set_color(RED)
        noise_def[0][41:43].set_color(RED)
        noise_def[0][61].set_color(RED)

        self.add(noise)

        # fmt:off
        self.play(
            TransformByGlyphMap(
                noise,
                noise_def,
                ([0], [0]),
                (FadeIn, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], {"delay": 0.2, "shift": LEFT}),
                ([1, 2], [21, 22],{"delay":.1}),
                (FadeIn, [23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], {"delay": 0.6, "shift": LEFT}),
                ([3, 4], [41, 42],{"delay":.2}),
                (FadeIn, [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60], {"delay": 1, "shift": LEFT}),
                ([5], [61],{"delay":.3}),
                (FadeIn, [62,63,64,65,66], {"delay": 1.4, "shift": LEFT}),
                from_copy=True,
            ),
            noise.animate.to_edge(UP,LARGE_BUFF)
        )
        # fmt:on

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        noise_sep = (
            MathTex(r"k T_s B_n \cdot L", color=RED, font_size=DEFAULT_FONT_SIZE * 1.8)
            .scale(1.5)
            .set_y(noise.get_y())
        )

        self.play(
            TransformByGlyphMap(
                noise,
                noise_sep,
                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                (GrowFromCenter, [5], {"delay": 0.4}),
                ([5], [6]),
            )
        )

        self.wait(0.5)

        self.play(noise_def.animate.shift(DOWN * 10))

        self.wait(0.5)

        thermal = (
            Tex("Thermal noise", font_size=DEFAULT_FONT_SIZE * 1.5)
            .to_edge(LEFT, LARGE_BUFF)
            .shift(DOWN)
        )
        thermal_line = CubicBezier(
            noise_sep[0][:5].get_bottom() + [0, -0.1, 0],
            noise_sep[0][:5].get_bottom() + [0, -1, 0],
            thermal.get_top() + [0, 1, 0],
            thermal.get_top() + [0, 0.1, 0],
        )

        self.play(LaggedStart(Create(thermal_line), FadeIn(thermal), lag_ratio=0.3))

        self.wait(0.5)

        l_term_label = (
            Tex("Extra losses", font_size=DEFAULT_FONT_SIZE * 1.5)
            .to_edge(RIGHT, LARGE_BUFF)
            .shift(DOWN)
        )
        l_line = CubicBezier(
            noise_sep[0][-1].get_bottom() + [0, -0.1, 0],
            noise_sep[0][-1].get_bottom() + [0, -1, 0],
            l_term_label.get_top() + [0, 1, 0],
            l_term_label.get_top() + [0, 0.1, 0],
        )

        self.play(LaggedStart(Create(l_line), FadeIn(l_term_label), lag_ratio=0.3))

        self.wait(0.5)

        title_group = (
            Group(thermal.copy(), noise_sep[0][:5].copy().scale(1 / 1.5))
            .arrange(RIGHT, LARGE_BUFF)
            .to_edge(UP, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(thermal_line), Uncreate(l_line), FadeOut(l_term_label)
                ),
                noise_sep[0][-2].animate.shift(UP * 4),
                noise_sep[0][-1].animate.shift(UP * 4),
                AnimationGroup(
                    thermal.animate.move_to(title_group[0]),
                    noise_sep[0][:5].animate.scale(1 / 1.5).move_to(title_group[1]),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(2)


class Thermal(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        np.random.seed(0)

        thermal = Tex("Thermal noise", font_size=DEFAULT_FONT_SIZE * 1.5)
        noise_eqn = MathTex(r"k T_s B_n", color=RED, font_size=DEFAULT_FONT_SIZE * 1.8)
        title_group = (
            Group(thermal, noise_eqn)
            .arrange(RIGHT, LARGE_BUFF)
            .to_edge(UP, MED_LARGE_BUFF)
        )

        up_shift = VT(0)
        right_shift = VT(0)
        box = always_redraw(
            lambda: Square(side_length=3, color=WHITE).shift(
                UP * ~up_shift + RIGHT * ~right_shift
            )
        )
        self.add(box)

        num_particles = 10
        particles = VGroup()
        velocities = []
        colliding = [False] * num_particles

        max_vel = 1.5

        collision_margin = 0.1

        for _ in range(num_particles):
            particle = Dot(radius=0.1, color=BLUE)
            particle.move_to(
                box.get_center()
                + [
                    np.random.uniform(
                        -1.5 + collision_margin,
                        1.5 - collision_margin,
                    ),
                    np.random.uniform(
                        -1.5 + collision_margin,
                        1.5 - collision_margin,
                    ),
                    0,
                ]
            )
            particles.add(particle)
            velocities.append(np.random.uniform(-max_vel, max_vel, size=2))

        self.add(particles)

        vel_mult = VT(1)

        def update_particles(group, dt):
            for i, particle in enumerate(group):
                particle.shift(
                    dt * velocities[i][0] * ~vel_mult * RIGHT
                    + dt * velocities[i][1] * ~vel_mult * UP
                )

                if (
                    particle.get_center()[0] >= box.get_right()[0] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[0] <= box.get_left()[0] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] >= box.get_top()[1] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] <= box.get_bottom()[1] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                else:
                    colliding[i] = False

                for j, other_particle in enumerate(group):
                    if (
                        i != j
                        and np.linalg.norm(
                            particle.get_center() - other_particle.get_center()
                        )
                        < collision_margin * 2
                    ):
                        velocities[i], velocities[j] = (velocities[j], velocities[i])

        particles.add_updater(update_particles)

        self.add(title_group)

        temp_label = Tex(r"Temp $\sim$").to_edge(LEFT)
        fire = (
            Group(*[ImageMobject("../props/static/fire.png") for _ in range(4)])
            .arrange(RIGHT, MED_SMALL_BUFF)
            .scale_to_fit_height(temp_label.height * 1.2)
            .next_to(temp_label)
        )
        temp_group = Group(temp_label, fire)  # .next_to(box, LEFT, LARGE_BUFF)

        self.add(temp_label, fire[0])

        self.camera.frame.shift(
            LEFT * (self.camera.frame.get_right()[0] - temp_label.get_left()[0]) + LEFT
        )
        title_group.next_to(self.camera.frame.get_top(), DOWN, MED_LARGE_BUFF)
        title_group_x = title_group.get_x()
        self.camera.frame.save_state()

        right_screen_group = Group(temp_group, box)
        self.play(
            title_group.animate.set_x(right_screen_group.get_x()),
            self.camera.frame.animate.set_x(right_screen_group.get_x()),
            run_time=2,
        )

        self.wait(0.5)

        # self.play(
        #     noise_eqn[0][0]
        #     .animate(rate_func=rate_functions.there_and_back)
        #     .shift(DOWN / 3)
        #     .set_color(YELLOW)
        # )

        # self.wait(0.5)

        self.play(
            vel_mult @ 4,
            LaggedStart(*[GrowFromCenter(m) for m in fire[1:]], lag_ratio=0.3),
            run_time=5,
        )

        self.wait(2)

        self.play(
            self.camera.frame.animate.restore(),
            title_group.animate.set_x(title_group_x),
        )
        self.remove(box, particles, temp_group)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        k_terms = Tex(r"$k$ | Energy / temperature").next_to(
            self.camera.frame.get_left(), RIGHT, LARGE_BUFF
        )
        k_val = Tex(r"$k = 1.38 \cdot 10^{-23}$ J/K").move_to(k_terms, LEFT)
        k_val[0][0].set_color(RED)
        k_terms[0][0].set_color(RED)
        k_units = Tex(r"Joules / Kelvin").next_to(
            k_terms[0][2], DOWN, MED_SMALL_BUFF, LEFT
        )
        k_units_abbv = Tex(r"J/K").move_to(k_units, LEFT)
        k_terms_bez = CubicBezier(
            noise_eqn[0][0].get_bottom() + [0, -0.1, 0],
            noise_eqn[0][0].get_bottom() + [0, -1, 0],
            k_terms.get_top() + [0, 1, 0],
            k_terms.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                TransformFromCopy(noise_eqn[0][0], k_terms[0][0], path_arc=PI / 3),
                Create(k_terms_bez),
                FadeIn(k_terms[0][1:]),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(FadeIn(k_units))

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                k_units,
                k_units_abbv,
                ([0], [0]),
                ([6], [1], {"delay": 0.3}),
                ([7], [2], {"delay": 0.5}),
                ([1, 2, 3, 4, 5], ShrinkToCenter),
                ([8, 9, 10, 11, 12], ShrinkToCenter),
                shift_fades=False,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                LaggedStart(
                    *[ShrinkToCenter(m) for m in k_terms[0][2:]], lag_ratio=0.1
                ),
                ReplacementTransform(k_terms[0][0], k_val[0][0]),
                ReplacementTransform(k_terms[0][1], k_val[0][1]),
                LaggedStart(
                    *[GrowFromCenter(m) for m in k_val[0][2:-3]], lag_ratio=0.1
                ),
                ReplacementTransform(k_units_abbv[0], k_val[0][-3:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        # TODO: move to center and add bez
        temp_value_label = Tex(r"$T_s = 290$ K").next_to(k_terms, RIGHT, LARGE_BUFF)
        temp_value_label[0][:2].set_color(RED)
        temp_value_bez = CubicBezier(
            noise_eqn[0][1:3].get_bottom() + [0, -0.1, 0],
            noise_eqn[0][1:3].get_bottom() + [0, -1, 0],
            temp_value_label.get_top() + [0, 1, 0],
            temp_value_label.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    noise_eqn[0][1:3], temp_value_label[0][:2], path_arc=PI / 4
                ),
                Create(temp_value_bez),
                LaggedStart(
                    *[GrowFromCenter(m) for m in temp_value_label[0][2:]],
                    lag_ratio=0.1,
                ),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        kT = Tex(
            r"$k \cdot T_s = 1.38 \cdot 10^{-23} \cdot 290 \  \frac{\text{J} \cdot \text{K}}{\text{K}}$ "
        ).move_to(k_terms, LEFT)
        kT[0][0].set_color(RED)
        kT[0][2:4].set_color(RED)

        # self.add(neweqn)

        self.play(Uncreate(temp_value_bez), Uncreate(k_terms_bez))

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                k_val,
                kT,
                ([0], [0]),
                (GrowFromCenter, [1], {"delay": 2}),
                (
                    get_transform_func(
                        temp_value_label[0][:2],
                        func=ReplacementTransform,
                        path_arc=-PI / 2,
                    ),
                    [2, 3],
                    {"delay": 2.2},
                ),
                ([1], [4], {"delay": 1.8}),
                (
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    {"delay": 1.6},
                ),
                (GrowFromCenter, [15], {"delay": 1.2}),
                (
                    get_transform_func(
                        temp_value_label[0][3:6],
                        func=ReplacementTransform,
                        path_arc=-PI / 2,
                    ),
                    [16, 17, 18],
                    {"delay": 1.4},
                ),
                ([12], [19], {"delay": 0.2}),
                ([14], [23], {"delay": 0.4}),
                ([13], [22], {"delay": 0.6}),
                (GrowFromCenter, [20], {"delay": 0.8}),
                (
                    get_transform_func(
                        temp_value_label[0][-1],
                        func=ReplacementTransform,
                        path_arc=PI / 2,
                    ),
                    [21],
                    {"delay": 1},
                ),
                # ([0, 1], [2, 3], {"path_arc": PI / 3}),
            ),
            ShrinkToCenter(temp_value_label[0][2]),
            run_time=4,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                kT[0][21]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(UP / 3),
                kT[0][23]
                .animate(rate_func=rate_functions.there_and_back)
                .set_color(YELLOW)
                .shift(DOWN / 3),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        kT_val = Tex(r"$k \cdot T_s = 4 \cdot 10^{-21} \  \text{J}$ ").move_to(kT, LEFT)
        kT_val[0][0].set_color(RED)
        kT_val[0][2:4].set_color(RED)

        self.play(
            TransformByGlyphMap(
                kT,
                kT_val,
                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                (
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    [5, 6, 7, 8, 9, 10, 11],
                ),
                ([20], ShrinkToCenter, {"delay": 0.3}),
                ([21], ShrinkToCenter, {"delay": 0.4}),
                ([22], ShrinkToCenter, {"delay": 0.5}),
                ([23], ShrinkToCenter, {"delay": 0.6}),
                ([19], [12], {"delay": 0.8}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        noise_std = VT(0.1)
        fs = 200

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.35
        y_max = 50

        ax = Axes(
            x_range=[0, fs / 2, fs / 8],
            y_range=[y_max * 0.3, y_max, y_max / 2],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        )

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)

        def fft_updater():
            np.random.seed(1)
            noise = np.random.normal(loc=0, scale=~noise_std, size=N)
            X_k = fftshift(fft(noise, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = np.clip(10 * np.log10(X_k) + y_max, 0, None)
            f_X_k_log = interp1d(freq, X_k, fill_value="extrapolate")

            plot = ax.plot(f_X_k_log, x_range=[0, fs / 2, 1 / 100], color=RX_COLOR)
            return plot

        X_k_plot = fft_updater()
        X_k_updater = lambda m: m.become(fft_updater())
        X_k_plot.add_updater(X_k_updater)

        self.add(ax.next_to([0, -config.frame_height / 2, 0], DOWN))
        self.add(X_k_plot)

        self.play(
            kT_val.animate.next_to(title_group, DOWN, LARGE_BUFF),
            ax.animate.next_to(self.camera.frame.get_bottom(), UP, LARGE_BUFF * 1.5),
        )

        self.wait(0.5)

        f_label = MathTex("f = 0").next_to(
            ax.c2p(0, y_max * 0.3), DOWN, MED_LARGE_BUFF, RIGHT
        )

        inf = (
            MathTex(r"\infty")
            .next_to(ax.c2p(fs / 2, y_max * 0.3), DOWN, MED_LARGE_BUFF)
            .set_y(f_label.get_y())
        )
        f_arrow = Arrow(f_label.get_right(), inf.get_left())

        self.play(
            LaggedStart(
                LaggedStart(*[GrowFromCenter(m) for m in f_label[0]], lag_ratio=0.1),
                GrowArrow(f_arrow),
                GrowFromCenter(inf),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        bw_disp = VT(25)
        bw_center_disp = VT(fs / 4)
        bw_box = Rectangle(
            height=ax.height * 0.6,
            width=(ax.c2p(~bw_disp / 2, 0)[0] - ax.c2p(-~bw_disp / 2, 0)[0]),
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.3,
        ).next_to(ax.c2p(~bw_center_disp, y_max * 0.3), UP, 0)

        def update_bw_box(m):
            m.become(
                Rectangle(
                    height=ax.height * 0.6,
                    width=(ax.c2p(~bw_disp / 2, 0)[0] - ax.c2p(-~bw_disp / 2, 0)[0]),
                    color=YELLOW,
                    fill_color=YELLOW,
                    fill_opacity=0.3,
                ).next_to(ax.c2p(~bw_center_disp, y_max * 0.3), UP, 0)
            )

        bw_label = MathTex("B_n", color=RED).next_to(bw_box, UP)
        bw_label_updater = lambda m: m.next_to(bw_box, UP)

        self.play(
            TransformFromCopy(noise_eqn[0][-2:], bw_label[0], path_arc=-PI / 3),
            GrowFromCenter(bw_box),
        )
        self.add(bw_label)
        bw_label.add_updater(bw_label_updater)
        bw_box.add_updater(update_bw_box)

        self.wait(0.5)

        self.play(bw_disp @ 40, bw_center_disp - 20)

        self.wait(0.5)

        self.play(bw_disp @ 10, bw_center_disp + 60)

        self.wait(0.5)

        bw_label.remove_updater(bw_label_updater)
        bw_box.remove_updater(update_bw_box)
        X_k_plot.remove_updater(X_k_updater)
        bw_label_val = MathTex(r"B_n = 1.6 \cdot 10^6 \text{Hz}").move_to(bw_label)
        bw_label_val[0][:2].set_color(RED)

        self.play(
            LaggedStart(
                ReplacementTransform(bw_label[0], bw_label_val[0][:2]),
                LaggedStart(
                    *[GrowFromCenter(m) for m in bw_label_val[0][2:]], lag_ratio=0.1
                ),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        kTB_eqn = MathTex(
            r"k \cdot T_s \cdot B_n = 4 \cdot 10^{-21} \cdot 1.6 \cdot 10^6 \  \text{J} \cdot \text{Hz}"
        ).move_to(kT_val)
        kTB_eqn[0][0].set_color(RED)
        kTB_eqn[0][2:4].set_color(RED)
        kTB_eqn[0][5:7].set_color(RED)
        kTB_val = MathTex(r"k T_s B_n = 6.4 \cdot 10^{-15} \  \text{W}").move_to(
            kTB_eqn
        )
        kTB_val[0][0].set_color(RED)
        kTB_val[0][1:3].set_color(RED)
        kTB_val[0][3:5].set_color(RED)

        self.play(
            TransformByGlyphMap(
                kT_val,
                kTB_eqn,
                ([0, 1, 2, 3], [0, 1, 2, 3]),
                ([4], [7], {"delay": 0.6}),
                (GrowFromCenter, [4], {"delay": 0.2}),
                (
                    get_transform_func(
                        bw_label_val[0][:2],
                        func=ReplacementTransform,
                        path_arc=-PI / 3,
                    ),
                    [5, 6],
                    {"delay": 0.4},
                ),
                (
                    [5, 6, 7, 8, 9, 10, 11],
                    [8, 9, 10, 11, 12, 13, 14],
                    {"delay": 0.6},
                ),
                (GrowFromCenter, [15], {"delay": 0.8}),
                (
                    get_transform_func(
                        bw_label_val[0][3:-2],
                        func=ReplacementTransform,
                        path_arc=-PI / 3,
                    ),
                    [16, 17, 18, 19, 20, 21, 22],
                    {"delay": 0.8},
                ),
                ([12], [23], {"delay": 0.6}),
                (GrowFromCenter, [24], {"delay": 0.6}),
                (
                    get_transform_func(
                        bw_label_val[0][-2:],
                        func=ReplacementTransform,
                        path_arc=-PI / 3,
                    ),
                    [25, 26],
                    {"delay": 1},
                ),
                run_time=3,
            ),
            ShrinkToCenter(bw_label_val[0][2], run_time=1),
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                kTB_eqn,
                kTB_val,
                ([0], [0], {"delay": 0.3}),
                ([1], ShrinkToCenter, {"delay": 0.3}),
                ([2, 3], [1, 2], {"delay": 0.3}),
                ([4], ShrinkToCenter, {"delay": 0.3}),
                ([5, 6], [3, 4], {"delay": 0.3}),
                ([7], [5]),
                (
                    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                    [6, 7, 8, 9, 10, 11, 12, 13, 14],
                    {"delay": 0},
                ),
                ([23, 24, 25, 26], [15], {"delay": 0.3}),
            )
        )

        self.wait(0.5)

        kTB_val_db = MathTex(r"k T_s B_n = -141.9 \  \text{dBW}").move_to(kTB_eqn)
        kTB_val_db[0][0].set_color(RED)
        kTB_val_db[0][1:3].set_color(RED)
        kTB_val_db[0][3:5].set_color(RED)

        self.play(
            TransformByGlyphMap(
                kTB_val,
                kTB_val_db,
                ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
                (
                    [6, 7, 8, 9, 10, 11, 12, 13, 14],
                    [6, 7, 8, 9, 10, 11],
                    {"delay": 0.2},
                ),
                ([15], [14], {"delay": 0.2}),
                (GrowFromCenter, [12, 13], {"delay": 0.4}),
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{graphicx}")

        notebook_reminder = Tex(
            r"radar\_cheatsheet.ipynb\rotatebox[origin=c]{270}{$\looparrowright$}",
            tex_template=tex_template,
            font_size=DEFAULT_FONT_SIZE * 2.5,
        )
        notebook_box = SurroundingRectangle(
            notebook_reminder, color=RED, fill_color=BACKGROUND_COLOR, fill_opacity=1
        )
        notebook = Group(notebook_box, notebook_reminder).next_to(
            self.camera.frame.get_bottom(), UP, MED_LARGE_BUFF
        )

        self.play(notebook.shift(DOWN * 5).animate.shift(UP * 5))

        self.wait(0.5)

        self.play(notebook.animate.shift(DOWN * 5))
        self.remove(notebook)

        self.wait(0.5)

        final_group = Group(kTB_val_db, title_group)
        scale_factor = final_group.width * 1.3 / self.camera.frame.width
        self.play(
            FadeOut(ax, X_k_plot, f_arrow, f_label, inf, bw_box),
            self.camera.frame.animate.scale(scale_factor).move_to(final_group),
            kTB_val_db.animate.set_x(title_group.get_x()),
        )

        self.wait(0.5)

        new_top = (
            self.camera.frame.copy().scale(1 / scale_factor).shift(DOWN * 10).get_top()
        )

        nf = Tex("Loss | L", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
            new_top, DOWN, LARGE_BUFF
        )
        nf[0][-1].set_color(RED)
        self.add(nf)

        self.play(self.camera.frame.animate.scale(1 / scale_factor).shift(DOWN * 10))

        self.wait(2)


class ReceiverNoise(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        rx_noise = Tex("Loss | L", font_size=DEFAULT_FONT_SIZE * 1.5).to_edge(
            UP, LARGE_BUFF
        )
        rx_noise[0][-1].set_color(RED)
        self.add(rx_noise)

        antenna_port = Line(DOWN * 2, UP, color=WHITE)
        antenna_tri = Triangle(color=WHITE).rotate(PI / 3).move_to(antenna_port, UP)

        antenna = (
            Group(antenna_port, antenna_tri)
            .scale(0.8)
            .to_edge(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP)
        )

        path_buff = LARGE_BUFF * 2

        switch = (
            BLOCKS.get("spdt_switch")
            .copy()
            .flip()
            .scale_to_fit_height(antenna_tri.get_height())
            .next_to(antenna_port.get_bottom(), LEFT, path_buff)
        )
        antenna_to_switch = Line(antenna_port.get_bottom(), switch.get_right())

        switch_to_lna = Line(
            switch.get_left() + DOWN * switch.get_height() / 4,
            switch.get_left() + DOWN * switch.get_height() / 4 + LEFT * path_buff * 2,
        )

        limiter_diode = (
            get_diode()
            .scale(0.5)
            .set_stroke(color=WHITE, width=DEFAULT_STROKE_WIDTH * 1.5)
        )
        limiter_gnd = (
            Group(
                *[
                    Line(stroke_width=DEFAULT_STROKE_WIDTH * 1.5).scale(scale)
                    for scale in [1, 0.7, 0.7**2]
                ]
            )
            .scale_to_fit_width(limiter_diode.width * 0.8)
            .arrange(DOWN, SMALL_BUFF)
            .next_to(limiter_diode, DOWN, 0)
        )
        limiter = Group(limiter_diode, limiter_gnd).next_to(
            switch_to_lna.get_midpoint(), DOWN, 0
        )

        lna = (
            Triangle(color=WHITE)
            .scale_to_fit_height(switch.height)
            .rotate(-PI / 6)
            .next_to(switch_to_lna, LEFT, 0)
        )

        self.wait(0.5)

        rx_arrow = Arrow(
            antenna.get_top() + RIGHT * 5 + UP * 3, antenna.get_top(), color=RX_COLOR
        )

        self.play(
            LaggedStart(GrowFromCenter(antenna), GrowArrow(rx_arrow), lag_ratio=0.3)
        )

        self.wait(0.5)

        self.play(FadeOut(rx_arrow))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(antenna_to_switch),
                GrowFromCenter(switch),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        tx_label = (
            Tex("Transmitter", font_size=DEFAULT_FONT_SIZE * 1.2)
            .next_to(switch, LEFT, path_buff)
            .shift(UP * 2)
        )
        rx_label = (
            Tex("Receiver", font_size=DEFAULT_FONT_SIZE * 1.2)
            .next_to(switch, LEFT, path_buff)
            .shift(DOWN * 2)
        )
        tx_box = SurroundingRectangle(tx_label, color=TX_COLOR)
        rx_box = SurroundingRectangle(rx_label, color=TX_COLOR)
        tx = Group(tx_box, tx_label)
        rx = Group(rx_box, rx_label)
        switch_to_tx = CubicBezier(
            switch.get_left() + UP * switch.get_height() / 4,
            switch.get_left() + UP * switch.get_height() / 4 + [-1, 0, 0],
            tx.get_right() + [1, 0, 0],
            tx.get_right(),
        )
        switch_to_rx = CubicBezier(
            switch.get_left() + DOWN * switch.get_height() / 4,
            switch.get_left() + DOWN * switch.get_height() / 4 + [-1, 0, 0],
            rx.get_right() + [1, 0, 0],
            rx.get_right(),
        )

        self.play(
            LaggedStart(
                Create(switch_to_tx),
                GrowFromCenter(tx),
                Create(switch_to_rx),
                GrowFromCenter(rx),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(rx),
                AnimationGroup(
                    tx_box.animate.set_stroke(opacity=0.2),
                    tx_label.animate.set_opacity(0.2),
                    switch_to_tx.animate.set_stroke(opacity=0.2),
                ),
                Create(switch_to_lna),
                FadeOut(switch_to_rx),
                GrowFromCenter(limiter),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        limiter_loss = (
            Tex(r"$L = 0.4$ dB", color=RED, font_size=DEFAULT_FONT_SIZE * 0.7)
            .next_to(limiter, DOWN)
            .set_opacity(0)
            .shift(DOWN / 3)
        )
        switch_loss = (
            Tex(r"$L = 1$ dB", color=RED, font_size=DEFAULT_FONT_SIZE * 0.7)
            .next_to(switch, DOWN)
            .set_opacity(0)
            .shift(DOWN / 3)
        )

        self.play(
            LaggedStart(
                switch_loss.animate.set_opacity(1).shift(UP / 3),
                limiter_loss.animate.set_opacity(1).shift(UP / 3),
                lag_ratio=0.5,
            ),
        )

        self.wait(0.5)

        self.play(GrowFromCenter(lna))

        self.wait(0.5)

        lna_gain = (
            Tex("$G = 20$ dB", font_size=DEFAULT_FONT_SIZE * 0.7)
            .set_color(GREEN)
            .next_to(lna, DOWN)
            .set_opacity(0)
            .shift(DOWN / 3)
        )
        lna_nf = (
            Tex("$NF = 2$ dB", font_size=DEFAULT_FONT_SIZE * 0.7)
            .set_color(RED)
            .next_to(lna_gain, DOWN, SMALL_BUFF)
            .set_opacity(0)
            .shift(DOWN / 3)
        )

        self.play(
            LaggedStart(
                lna_gain.animate.set_opacity(1).shift(UP / 3),
                lna_nf.animate.set_opacity(1).shift(UP / 3),
                lag_ratio=0.5,
            ),
        )

        self.wait(0.5)

        lna_spelled = Tex("Low Noise Amplifier").next_to(lna, UP)
        lna_label = Tex("LNA").next_to(lna, UP)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in lna_spelled[0]], lag_ratio=0.1)
        )

        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                lna_spelled,
                lna_label,
                ([0], [0], {"delay": 0.3}),
                ([3], [1], {"delay": 0.3}),
                ([8], [2], {"delay": 0.3}),
                ([1, 2], ShrinkToCenter),
                ([4, 5, 6, 7], ShrinkToCenter),
                ([9, 10, 11, 12, 13, 14, 15], ShrinkToCenter),
                shift_fades=False,
            )
        )

        self.wait(0.5)

        lna_to_filter = Line(
            lna.get_left(),
            lna.get_left() + LEFT * path_buff,
        )
        lp_filter = (
            BLOCKS.get("bp_filter_generic")
            .copy()
            .scale_to_fit_height(switch.height)
            .next_to(lna_to_filter, LEFT, 0)
        )

        l_switch = (
            BLOCKS.get("spdt_switch")
            .copy()
            .scale_to_fit_height(antenna_tri.height)
            .next_to(lp_filter, LEFT, path_buff)
            .shift(UP * antenna_tri.height / 4)
        )
        lp_filter_to_switch = Line(
            l_switch.get_right() + DOWN * l_switch.get_height() / 4,
            l_switch.get_right() + DOWN * l_switch.get_height() / 4 + RIGHT * path_buff,
        )

        filter_loss = (
            Tex(r"$L = 0.5$ dB", color=RED, font_size=DEFAULT_FONT_SIZE * 0.7)
            .next_to(lp_filter, DOWN)
            .set_opacity(0)
            .shift(DOWN / 3)
        )
        l_switch_loss = (
            Tex(r"$L = 1$ dB", color=RED, font_size=DEFAULT_FONT_SIZE * 0.7)
            .next_to(l_switch, DOWN)
            .set_opacity(0)
            .shift(DOWN / 3)
        )

        bd_group = Group(l_switch, lna_to_filter, lp_filter, lp_filter_to_switch).add(
            *self.mobjects
        )
        self.add(l_switch, lna_to_filter, lp_filter, lp_filter_to_switch)
        new_cam = (
            self.camera.frame.copy()
            .scale_to_fit_width(bd_group.width * 1.2)
            .move_to(bd_group)
            .shift(UP)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(bd_group.width * 1.2)
                .move_to(bd_group)
                .shift(UP),
                rx_noise.animate.next_to(new_cam.get_top(), DOWN, LARGE_BUFF),
                Create(lna_to_filter),
                GrowFromCenter(lp_filter),
                filter_loss.animate.set_opacity(1).shift(UP / 3),
                Create(lp_filter_to_switch),
                GrowFromCenter(l_switch),
                l_switch_loss.animate.set_opacity(1).shift(UP / 3),
                ShrinkToCenter(lna_label),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=rate_functions.there_and_back)
                    .shift(DOWN / 3)
                    .set_color(YELLOW)
                    for m in [
                        l_switch_loss,
                        filter_loss,
                        limiter_loss,
                        switch_loss,
                        rx_noise[0][-1],
                    ]
                ],
                lag_ratio=0.45,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        nf_eqn = MathTex(
            r"F_{\text{total}} = F_1 + \frac{F_2 - 1}{G_1} + \frac{F_3 - 1}{G_1 G_2} + \cdots + \frac{F_N - 1}{G_1 G_2 \dots G_{N-1}}",
            font_size=DEFAULT_FONT_SIZE * 1,
        ).next_to(rx_noise, DOWN, MED_LARGE_BUFF)

        indices = Group(
            *[
                MathTex(
                    f"{idx}", color=YELLOW, font_size=DEFAULT_FONT_SIZE * 1.8
                ).next_to(component, UP)
                for idx, component in enumerate(
                    [antenna, switch, limiter, lna, lp_filter, l_switch], start=1
                )
            ]
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in nf_eqn[0]], lag_ratio=0.1),
            LaggedStart(*[GrowFromCenter(m) for m in indices], lag_ratio=0.35),
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(nf_eqn.width * 1.2).move_to(
                nf_eqn
            ),
            FadeOut(bd_group, indices),
        )

        self.wait(0.5)

        snr_eqn_split = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n \cdot L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        snr_eqn_split[0][0].set_color(BLUE)
        snr_eqn_split[0][1].set_color(RED)
        snr_eqn_split[0][4:19].set_color(BLUE)
        snr_eqn_split[0][22:].set_color(RED)

        self.add(snr_eqn_split.next_to(self.camera.frame.get_top(), UP, LARGE_BUFF * 3))

        self.play(self.camera.frame.animate.move_to(snr_eqn_split))

        self.wait(0.5)

        self.play(
            LaggedStart(
                snr_eqn_split[0][4:19]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(UP / 2)
                .set_color(YELLOW),
                snr_eqn_split[0][22:-2]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(DOWN / 2)
                .set_color(YELLOW),
                snr_eqn_split[0][-1]
                .animate(rate_func=rate_functions.there_and_back)
                .shift(DOWN / 2)
                .set_color(YELLOW),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * config.frame_height))

        self.wait(2)


class WrapUp(Scene):
    def construct(self):
        video = VideoMobject(
            "./static/cheatsheet_notebook.mp4", speed=2
        ).scale_to_fit_width(config.frame_width * 0.7)

        self.play(
            video.next_to([config.frame_width / 2, 0, 0], RIGHT).animate.move_to(ORIGIN)
        )

        self.wait(10, frozen_frame=False)

        self.play(video.animate.next_to([-config.frame_width / 2, 0, 0], LEFT))
        self.remove(video)


class EndScreen(Scene):
    def construct(self):
        stats_title = Tex("Stats for Nerds")
        stats_table = (
            Table(
                [
                    ["Lines of code", "2,692"],
                    ["Script word count", "1,372"],
                    ["Days to make", "18"],
                    ["Git commits", "6"],
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
                LaggedStart(
                    *[GrowFromCenter(m) for m in thank_you_sabrina[0]], lag_ratio=0.06
                ),
                lag_ratio=0.9,
                run_time=4,
            )
        )

        self.wait(2)


class GasCollision(MovingCameraScene):
    def construct(self):
        np.random.seed(0)

        up_shift = VT(0)
        right_shift = VT(0)
        box = always_redraw(
            lambda: Square(side_length=3, color=WHITE).shift(
                UP * ~up_shift + RIGHT * ~right_shift
            )
        )
        self.add(box)

        num_particles = 10
        particles = VGroup()
        velocities = []
        colliding = [False] * num_particles

        max_vel = 1.5

        collision_margin = 0.1

        for _ in range(num_particles):
            particle = Dot(radius=0.1, color=BLUE)
            particle.move_to(
                box.get_center()
                + [
                    np.random.uniform(
                        -box.width / 2 + collision_margin,
                        box.width / 2 - collision_margin,
                    ),
                    np.random.uniform(
                        -box.height / 2 + collision_margin,
                        box.height / 2 - collision_margin,
                    ),
                    0,
                ]
            )
            particles.add(particle)
            velocities.append(np.random.uniform(-max_vel, max_vel, size=2))

        self.add(particles)

        vel_mult = VT(1)

        def update_particles(group, dt):
            for i, particle in enumerate(group):
                particle.shift(
                    dt * velocities[i][0] * ~vel_mult * RIGHT
                    + dt * velocities[i][1] * ~vel_mult * UP
                )

                if (
                    particle.get_center()[0] >= box.get_right()[0] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[0] <= box.get_left()[0] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][0] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] >= box.get_top()[1] - collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                elif (
                    particle.get_center()[1] <= box.get_bottom()[1] + collision_margin
                    and not colliding[i]
                ):
                    velocities[i][1] *= -1
                    colliding[i] = not colliding[i]
                else:
                    colliding[i] = False

                for j, other_particle in enumerate(group):
                    if (
                        i != j
                        and np.linalg.norm(
                            particle.get_center() - other_particle.get_center()
                        )
                        < collision_margin * 2
                    ):
                        velocities[i], velocities[j] = (velocities[j], velocities[i])

        particles.add_updater(update_particles)

        def get_camera_updater(m):
            def camera_updater(cam, dt):
                velocity = m.get_center() - cam.get_center()

                cam.shift(velocity * dt)

            return camera_updater

        self.camera.frame.save_state()
        self.camera.frame.scale(50)

        noise_eqn = MathTex(r"N = k T B_n").scale_to_fit_width(
            self.camera.frame.width * 0.6
        )

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in noise_eqn[0]], lag_ratio=0.06)
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore(), FadeOut(noise_eqn))

        self.wait(2)

        particle_to_follow = particles[np.argmin(np.sum(np.abs(velocities), axis=1))]
        cam_updater = get_camera_updater(particle_to_follow)

        particle_trace = TracedPath(
            particle_to_follow.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).set_z_index(-1)

        self.camera.frame.add_updater(cam_updater)
        self.add(particle_trace)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.3))

        self.wait(3)

        n_potentials = 16
        for idx in range(n_potentials):
            self.play(
                MathTex(f"v_{{{idx}}}", color=YELLOW)
                .scale(0.8)
                .move_to(particle_to_follow)
                .set_opacity(0)
                .animate(rate_func=rate_functions.there_and_back, run_time=0.8)
                .set_opacity(1)
            )

        self.wait(3)

        self.camera.frame.remove_updater(cam_updater)
        self.play(self.camera.frame.animate.restore())

        traced_paths = Group(
            *[
                TracedPath(
                    particle.get_center,
                    dissipating_time=2,
                    stroke_opacity=[0, 1],
                    stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                ).set_z_index(-1)
                for particle in particles
            ]
        )

        self.add(traced_paths)
        # self.remove(particle_trace)

        for idx in range(n_potentials):
            self.play(
                *[
                    MathTex(f"v_{{{pidx},{idx}}}", color=YELLOW)
                    .scale(0.8)
                    .move_to(particle)
                    .set_opacity(0)
                    .animate(rate_func=rate_functions.there_and_back, run_time=0.8)
                    .set_opacity(1)
                    for pidx, particle in enumerate(particles)
                ]
            )

        particles.remove_updater(update_particles)
        for traced_path in traced_paths:
            traced_path.remove_updater(traced_path.update_path)

        potential_labels = Group(
            *[
                MathTex(f"v_{{{pidx},{n_potentials}}}", color=YELLOW)
                .scale(0.8)
                .move_to(particle)
                .set_opacity(0)
                for pidx, particle in enumerate(particles)
            ]
        )
        self.play(potential_labels.animate(run_time=0.8).set_opacity(1))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.shift(
                LEFT
                * (
                    (self.camera.frame.get_right()[0])
                    - (box.get_right()[0] + LARGE_BUFF * 2)
                )
            )
        )

        total_pot = MathTex(
            r"\text{Total potential} = \  &"
            + r" \\ &+ ".join(
                [
                    f"v_{{{pidx},{n_potentials}}}"
                    for pidx, particle in enumerate(particles)
                ]
            )
        ).next_to(self.camera.frame.get_left(), RIGHT, LARGE_BUFF)

        self.add(total_pot)

        self.wait(10)


class SignalTest(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        snr_eqn = MathTex(
            r"\text{SNR} = \frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 1.8,
        )
        snr_eqn[0][0].set_color(BLUE)
        snr_eqn[0][4:19].set_color(BLUE)
        snr_eqn[0][1].set_color(RED)
        snr_eqn[0][22:].set_color(RED)

        radar_eqn = MathTex(
            r"P_r = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).scale(1.5)
        radar_eqn[0][3:12].set_color(GREEN)
        radar_eqn[0][13:19].set_color(BLUE)
        radar_eqn[0][20:22].set_color(YELLOW)

        radar_eqn_title = Tex(
            "The Radar Range Equation", font_size=DEFAULT_FONT_SIZE * 1.8
        ).to_edge(UP, MED_LARGE_BUFF)

        radar_eqn_group = (
            Group(radar_eqn_title, radar_eqn)
            .arrange(DOWN, LARGE_BUFF * 1.5)
            .scale_to_fit_width(config.frame_width * 0.3)
        )

        title_1 = Tex(r"Animated Radar Cheatsheet\\Episode 1")

        thumbnail_1_box = SurroundingRectangle(radar_eqn_group, buff=MED_LARGE_BUFF)
        radar_eqn_group.add(
            thumbnail_1_box, title_1.next_to(thumbnail_1_box, DOWN)
        ).next_to([config.frame_width / 2, 0, 0], RIGHT)

        self.add(snr_eqn)

        self.play(FadeOut(snr_eqn[0][:4], snr_eqn[0][-9:]))

        self.wait(0.5)

        signal_eqn = snr_eqn[0][4:-9]

        self.play(
            Group(signal_eqn, radar_eqn_group).animate.arrange(RIGHT, LARGE_BUFF * 1.5)
        )


class Thumbnail(Scene):
    def construct(self):
        snr_eqn = MathTex(
            r"\frac{P_t G^2 \lambda^2 \sigma}{(4 \pi)^3 R^4} \cdot \frac{1}{k T_s B_n L}",
            font_size=DEFAULT_FONT_SIZE * 2.2,
        )
        snr_eqn[0][0:15].set_color(BLUE)
        snr_eqn[0][18:].set_color(RED)

        title = Tex(
            "Radar",
            " Signal",
            "-to-",
            "Noise",
            " Ratio",
            font_size=DEFAULT_FONT_SIZE * 2.1,
        ).to_edge(UP, MED_LARGE_BUFF)
        title[1].set_color(BLUE)
        title[3].set_color(RED)

        Group(title, snr_eqn).arrange(DOWN, LARGE_BUFF * 1.5)

        self.add(snr_eqn, title)


class Thumbnail2(Scene):
    def construct(self):
        fs = 100
        f = 25
        noise_std = VT(0.2)

        x_len = config.frame_width * 0.7
        y_len = config.frame_height * 0.5
        y_max = 30

        ax = Axes(
            x_range=[0, fs / 2, fs / 8],
            y_range=[0, y_max, 10],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=y_len,
        ).to_edge(DOWN, LARGE_BUFF)

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        amp = VT(1)

        def fft_updater():
            np.random.seed(1)
            noise = np.random.normal(loc=0, scale=~noise_std, size=N)
            x_n = (~amp * np.sin(2 * PI * f * t) + noise) * signal.windows.blackman(N)
            X_k = fftshift(fft(x_n, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = np.clip(10 * np.log10(X_k) + y_max, 0, None)
            f_X_k_log = interp1d(freq, X_k, fill_value="extrapolate")

            plot = ax.plot(f_X_k_log, x_range=[0, fs / 2, 1 / 100], color=ORANGE)
            return plot

        X_k_plot = fft_updater()

        self.add(ax)
        self.add(X_k_plot)

        snr_line = Line(
            ax.c2p(30, 16.16),
            [
                ax.c2p(30, 16.16)[0],
                ax.input_to_graph_point(f, fft_updater())[1],
                0,
            ],
        )
        snr_line_u = Line(snr_line.get_top() + LEFT / 8, snr_line.get_top() + RIGHT / 8)
        snr_line_d = Line(
            snr_line.get_bottom() + LEFT / 8, snr_line.get_bottom() + RIGHT / 8
        )
        snr_label = Tex("SNR = ?", font_size=DEFAULT_FONT_SIZE * 1.8).next_to(
            snr_line, RIGHT
        )
        snr_label[0][0].set_color(BLUE)
        snr_label[0][1].set_color(RED)
        title = Tex(
            "Radar",
            " Signal",
            "-to-",
            "Noise",
            " Ratio",
            font_size=DEFAULT_FONT_SIZE * 2.1,
        ).to_edge(UP, MED_LARGE_BUFF)
        title[1].set_color(BLUE)
        title[3].set_color(RED)

        self.add(snr_line, snr_line_u, snr_line_d, snr_label, title)
