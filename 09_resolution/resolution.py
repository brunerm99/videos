# resolution.py


from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
import sys
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift

sys.path.insert(0, "..")

from props import WeatherRadarTower
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class Intro(Scene):
    def construct(self): ...


class RangeResolution(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4)

        self.play(radar.get_animation())

        self.wait(0.5)

        self.play(radar.vgroup.animate.to_corner(DL, LARGE_BUFF))

        self.wait(0.5)

        TARGET1_COLOR = GREEN
        TARGET2_COLOR = ORANGE
        target1 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .to_edge(RIGHT, LARGE_BUFF * 2)
            .shift(UP / 2)
            .set_fill(TARGET1_COLOR)
            .set_color(TARGET1_COLOR)
        )
        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .to_edge(RIGHT, LARGE_BUFF * 1.3)
            .shift(DOWN)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
        )

        ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=config.frame_width * 0.8,
                y_length=radar.radome.height,
            )
            .set_opacity(0)
            .next_to(radar.radome, RIGHT, 0)
        )
        target1_line = Line(target1.get_left(), radar.radome.get_right())
        target1_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target1_line.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target1_line.get_angle())
            .set_opacity(0)
        )
        target1_ax.shift(target1.get_left() - target1_ax.c2p(0, 0))

        target2_line = Line(target2.get_left(), radar.radome.get_right())
        target2_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target2_line.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target2_line.get_angle())
            .set_opacity(0)
        )
        target2_ax.shift(target2.get_left() - target2_ax.c2p(0, 0))
        # self.add(target1_ax)
        # self.add(ax, target)
        xmax = VT(0)
        xmax_t1 = VT(0)
        xmax_t2 = VT(0)
        pw = 0.2
        f = 10
        tx = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax - pw), ~xmax, 1 / 200],
                color=TX_COLOR,
            )
        )
        rx1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax_t1 - pw), min(~xmax_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax_t2 - pw), min(~xmax_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )
        self.add(tx, rx1, rx2)

        radar.vgroup.set_z_index(1)

        to_target1 = Arrow(
            radar.radome.get_right(), target1.get_left(), color=TX_COLOR
        ).shift(DOWN / 3)
        from_target1 = Arrow(
            target1.get_left(), radar.radome.get_right(), color=RX_COLOR
        ).shift(UP / 3)

        self.play(
            LaggedStart(
                GrowArrow(to_target1),
                target1.shift(RIGHT * 8).animate.shift(LEFT * 8),
                GrowArrow(from_target1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(FadeOut(to_target1, from_target1))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        pw_line = Line(ax.c2p(~xmax - pw, 1.2), ax.c2p(~xmax, 1.2))
        pw_line_l = Line(pw_line.get_start() + DOWN / 8, pw_line.get_start() + UP / 8)
        pw_line_r = Line(pw_line.get_end() + DOWN / 8, pw_line.get_end() + UP / 8)

        pw_label_val = MathTex(r"1 \mu s").next_to(pw_line, UP)
        pw_label = MathTex(r"\tau = 1 \mu s").next_to(pw_line, UP)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in pw_label_val[0]], lag_ratio=0.15),
            LaggedStart(
                Create(pw_line_l),
                Create(pw_line),
                Create(pw_line_r),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(pw_label_val[0], pw_label[0][-3:]),
                *[GrowFromCenter(m) for m in pw_label[0][:-3]],
                lag_ratio=0.15,
            ),
        )

        self.wait(0.5)

        self.play(FadeOut(pw_line, pw_line_l, pw_line_r, pw_label_val))

        self.wait(0.5)

        self.play(
            pw_label.animate.shift(UP),
            LaggedStart(
                xmax @ (ax.p2c(target2.get_left())[0]),
                xmax_t1 @ (pw / 2),
                lag_ratio=0.4,
            ),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(target2.shift(RIGHT * 8).animate.shift(LEFT * 8))

        self.wait(0.5)

        delta_r_line = always_redraw(
            lambda: Line(
                target1.get_center(),
                [target2.get_center()[0], target1.get_center()[1], 0],
            ).shift(UP * 1.5)
        )
        delta_r_line_l = always_redraw(
            lambda: Line(
                target1.get_center() + UP * 1.5 + DOWN / 8,
                target1.get_center() + UP * 1.5 + UP / 8,
            )
        )
        delta_r_line_r = always_redraw(
            lambda: Line(
                delta_r_line.get_right() + DOWN / 8,
                delta_r_line.get_right() + UP / 8,
            )
        )
        delta_r = always_redraw(lambda: MathTex(r"\Delta R").next_to(delta_r_line, UP))
        # self.add(delta_r_line_l, delta_r_line_r, delta_r_line, delta_r)

        self.play(
            LaggedStart(
                Create(delta_r_line_l),
                Create(delta_r_line),
                Create(delta_r_line_r),
                Write(delta_r),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            xmax @ 1.5,
            xmax_t2 @ (pw),
            xmax_t1 + pw,
            run_time=2,
        )

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )

        self.wait(0.5)

        self.play(
            xmax_t1 @ (0.5 + pw / 2),
            xmax_t2 @ (0.5 + pw / 2 - target_dist),
            run_time=3,
        )

        self.wait(0.5)

        line = Line(
            target2_ax.c2p(~xmax_t2, 0.8), target1_ax.c2p(~xmax_t1 - pw * 1.05, -0.6)
        )

        overlap = Polygon(
            line.get_start(),
            [line.get_start()[0], line.get_end()[1], 0],
            line.get_end(),
            [line.get_end()[0], line.get_start()[1], 0],
            fill_opacity=0.4,
            fill_color=YELLOW,
            stroke_opacity=0,
        )
        # self.add(overlap)

        self.next_section(skip_animations=skip_animations(True))
        tau_gt = MathTex(r"\tau > t_{\Delta R}").next_to(overlap, UP)

        self.play(
            LaggedStart(
                FadeIn(overlap),
                TransformFromCopy(pw_label[0][0], tau_gt[0][0], path_arc=-PI / 3),
                *[GrowFromCenter(m) for m in tau_gt[0][1:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        f1 = VT(2.5)
        f2 = VT(2.7)
        power_norm_1 = VT(-30)
        power_norm_2 = VT(-30)
        stop_time = 16
        fs = 1000
        N = fs * stop_time
        t = np.linspace(0, stop_time, N)

        noise_mu = 0
        noise_sigma_db = -10
        noise_sigma = 10 ** (noise_sigma_db / 10)

        np.random.seed(0)
        noise = np.random.normal(loc=noise_mu, scale=noise_sigma, size=t.size)

        return_xmax = 5
        return_ax = Axes(
            x_range=[0, return_xmax, return_xmax / 4],
            y_range=[0, 40, 20],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.3,
        ).next_to(self.camera.frame.get_top(), UP, SMALL_BUFF)
        self.add(return_ax)

        def get_return_plot():
            A_1 = 10 ** (~power_norm_1 / 10)
            A_2 = 10 ** (~power_norm_2 / 10)
            x_n = (
                A_1 * np.sin(2 * PI * ~f1 * t) + A_2 * np.sin(2 * PI * ~f2 * t) + noise
            )

            blackman_window = signal.windows.blackman(N)
            x_n_windowed = x_n * blackman_window

            fft_len = N * 4

            X_k = fftshift(fft(x_n_windowed, fft_len))
            X_k /= N / 2
            X_k = np.abs(X_k)
            X_k = 10 * np.log10(X_k)
            X_k -= -43

            freq = np.linspace(-fs / 2, fs / 2, fft_len)
            f_X_k = interp1d(freq, X_k, fill_value="extrapolate")

            plot = return_ax.plot(
                f_X_k, x_range=[0, return_xmax, return_xmax / 200], color=RX_COLOR
            )
            return plot

        return_plot = always_redraw(get_return_plot)
        self.add(return_plot)

        all_objs = Group(return_ax.copy().shift(DOWN), radar.vgroup, target1, target2)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                all_objs.height * 1.2
            ).move_to(all_objs),
            return_ax.animate.shift(DOWN),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(overlap, tau_gt),
                AnimationGroup(
                    xmax_t1 @ (1 + pw + target_dist),
                    xmax_t2 @ (1 + pw),
                ),
                power_norm_1 @ 0,
                power_norm_2 @ 0,
                lag_ratio=0.3,
            ),
            run_time=4,
        )

        self.wait(0.5)

        return_plot_new = get_return_plot()
        target1_label = (
            Text("Target 1?", color=TARGET1_COLOR, font="Maple Mono")
            .scale(0.6)
            .next_to(
                return_ax.input_to_graph_point(~f1, return_plot_new), DOWN, LARGE_BUFF
            )
            .shift(LEFT * 3)
        )
        target1_label_line = CubicBezier(
            target1_label.get_right() + [0.1, 0, 0],
            target1_label.get_right() + [1, 0, 0],
            return_ax.input_to_graph_point(~f1, return_plot_new) + [0, -1, 0],
            return_ax.input_to_graph_point(~f1, return_plot_new) + [0, -0.1, 0],
            color=TARGET1_COLOR,
        )

        target2_label = (
            Text("Target 2?", color=TARGET2_COLOR, font="Maple Mono")
            .scale(0.6)
            .next_to(
                return_ax.input_to_graph_point(~f1, return_plot_new), DOWN, LARGE_BUFF
            )
            .shift(RIGHT * 3)
        )
        target2_label_line = CubicBezier(
            target2_label.get_left() + [-0.1, 0, 0],
            target2_label.get_left() + [-1, 0, 0],
            return_ax.input_to_graph_point(~f2, return_plot_new) + [0, -1, 0],
            return_ax.input_to_graph_point(~f2, return_plot_new) + [0, -0.1, 0],
            color=TARGET2_COLOR,
        )
        # self.add(
        #     target1_label,
        #     target2_label,
        #     target1_label_line,
        #     target2_label_line,
        # )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(target1_label_line),
                    Create(target2_label_line),
                ),
                AnimationGroup(
                    Write(target1_label),
                    Write(target2_label),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target1_label_line.save_state()
        self.play(
            target1.animate.shift(LEFT * 2),
            f1 - 0.5,
            Transform(
                target1_label_line,
                CubicBezier(
                    target1_label.get_right() + [0.1, 0, 0],
                    target1_label.get_right() + [1, 0, 0],
                    return_ax.input_to_graph_point(~f1, return_plot_new)
                    + LEFT * 1.15
                    + [0, -1, 0],
                    return_ax.input_to_graph_point(~f1, return_plot_new)
                    + LEFT * 1.15
                    + [0, -0.1, 0],
                    color=TARGET1_COLOR,
                ),
            ),
        )

        self.wait(0.5)

        self.play(
            target1.animate.shift(RIGHT * 2.2),
            f1 + 0.6,
            Transform(
                target1_label_line,
                CubicBezier(
                    target1_label.get_right() + [0.1, 0, 0],
                    target1_label.get_right() + [1, 0, 0],
                    return_ax.input_to_graph_point(~f1 + 0.5, return_plot_new)
                    + RIGHT * 0.3
                    + [0, -1, 0],
                    return_ax.input_to_graph_point(~f1 + 0.5, return_plot_new)
                    + RIGHT * 0.3
                    + [0, -0.1, 0],
                    color=TARGET1_COLOR,
                ),
            ),
        )

        self.wait(0.5)

        qmark = always_redraw(
            lambda: Text("?", font="Maple Mono")
            .scale(0.8)
            .next_to(delta_r, RIGHT, SMALL_BUFF)
        )

        self.camera.frame.save_state()
        target_group = Group(target1, target2, delta_r)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(target_group.height * 1.3)
                .move_to(target_group)
                .shift(DOWN / 3),
                pw_label.animate.set_opacity(0),
                # pw_label.animate.next_to(target_group, DOWN),
                Write(qmark),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(target1.animate.shift(LEFT), target2.animate.shift(RIGHT * 0.5))

        self.wait(0.5)

        self.play(
            target1.animate.shift(RIGHT * 0.5), target2.animate.shift(RIGHT * 1.5)
        )

        self.wait(0.5)

        xmax_t1 @= 0
        xmax_t2 @= 0
        self.remove(
            target1_label, target2_label, target1_label_line, target2_label_line
        )

        tau_gt = MathTex(r"\tau > t_{\Delta R}").next_to(pw_label, DOWN)

        self.play(
            Write(tau_gt),
            target1.animate.shift(RIGHT * 0.5),
            target2.animate.shift(LEFT * 2),
            self.camera.frame.animate.restore(),
            pw_label.animate.set_opacity(1),
            AnimationGroup(
                xmax_t1 @ (0.5 + pw / 2),
                xmax_t2 @ (0.5 + pw / 2 - target_dist),
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            pw_label[0][0]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW),
            tau_gt[0][0]
            .animate(rate_func=rate_functions.there_and_back)
            .shift(UP / 3)
            .set_color(YELLOW),
        )

        self.wait(0.5)

        target_dist_large = abs(
            (
                ax.p2c(target1.get_left()[0])
                - ax.p2c((target2.get_left() + RIGHT * 1.5)[0])
            )[0]
        )

        self.play(
            target2.animate.shift(RIGHT * 1.5),
            tau_gt[0][1].animate.rotate(PI).set_color(YELLOW),
            f2 + 0.5,
            xmax_t2 - target_dist_large,
            run_time=3,
        )

        self.wait(0.5)

        script = Group(
            *[
                Text(s, font="Maple Mono", font_size=DEFAULT_FONT_SIZE * 0.5)
                for s in [
                    "...",
                    "than the time it takes to get from target ",
                    "1 to target 2 and back.",
                    'By the way, don\'t forget about this "and back" ',
                    "in my script. I said it because it messed me up a lot.\n",
                    "Dividing delta R by the speed of light will give you ",
                    "the time it takes to travel from one target to the ",
                    "...",
                ]
            ]
        ).arrange(DOWN, aligned_edge=LEFT)
        script_box = SurroundingRectangle(
            script,
            color=GREEN,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            buff=MED_SMALL_BUFF,
            corner_radius=0.2,
        )
        script_group = Group(script_box, script).set_z_index(3)

        self.play(
            script_group.shift(DOWN * config.frame_height * 1.5).animate.move_to(
                self.camera.frame
            )
        )

        self.wait(0.5)

        box1 = SurroundingRectangle(script[2][-8:-1]).set_z_index(10)
        box2 = SurroundingRectangle(script[3][-8:-1]).set_z_index(10)
        self.play(
            LaggedStart(
                Create(box1),
                Create(box2),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Group(script_group, box1, box2).animate.shift(
                DOWN * 1.5 * config.frame_height
            )
        )

        self.wait(0.5)

        delta_r_over_c = MathTex(
            r"\frac{\Delta R}{c} \left[ \frac{ m}{m/s} \right]",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).next_to(pw_label, UP, MED_LARGE_BUFF)

        self.play(
            return_ax.animate.shift(UP * config.frame_height * 0.8),
            LaggedStart(
                TransformFromCopy(delta_r[0], delta_r_over_c[0][:2], path_arc=-PI / 3),
                LaggedStart(
                    *[GrowFromCenter(m) for m in delta_r_over_c[0][2:]], lag_ratio=0.1
                ),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        x_top = MathTex(
            r"\times", font_size=DEFAULT_FONT_SIZE * 1.5, color=RED
        ).move_to(delta_r_over_c[0][5])
        x_bot = MathTex(
            r"\times", font_size=DEFAULT_FONT_SIZE * 1.5, color=RED
        ).move_to(delta_r_over_c[0][7])
        delta_r_over_c_cunit = MathTex(
            r"\frac{\Delta R}{c} \left[ s \right]",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        ).move_to(delta_r_over_c)

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(Write(x_top), Write(x_bot), lag_ratio=0.2),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ShrinkToCenter(delta_r_over_c[0][5]),
                    ShrinkToCenter(x_top),
                ),
                ShrinkToCenter(delta_r_over_c[0][6]),
                AnimationGroup(
                    ShrinkToCenter(delta_r_over_c[0][7]),
                    ShrinkToCenter(x_bot),
                ),
                ShrinkToCenter(delta_r_over_c[0][8]),
                ReplacementTransform(delta_r_over_c[0][9], delta_r_over_c_cunit[0][5]),
                ReplacementTransform(delta_r_over_c[0][10], delta_r_over_c_cunit[0][6]),
                ReplacementTransform(delta_r_over_c[0][4], delta_r_over_c_cunit[0][4]),
                ReplacementTransform(
                    delta_r_over_c[0][:4], delta_r_over_c_cunit[0][:4]
                ),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        target2_line_new = Line(target2.get_left(), radar.radome.get_right())
        target2_ax_new = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=target2_line_new.get_length(),
                y_length=radar.radome.height,
            )
            .rotate(target2_line_new.get_angle())
            .set_opacity(0)
        )
        target2_ax_new.shift(target2.get_left() - target2_ax_new.c2p(0, 0))

        xmax_12 = VT(ax.p2c(target1.get_left())[0])
        xmax_21 = VT(0)
        tx_12 = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[
                    max(ax.p2c(target1.get_left())[0], ~xmax_12 - pw),
                    min(~xmax_12, ax.p2c(target2.get_left())[0]),
                    1 / 200,
                ],
                color=TX_COLOR,
            )
        )
        self.add(tx_12)

        self.play(xmax_12 @ (ax.p2c(target2.get_left())[0]))

        self.wait(0.5)

        rx_21 = always_redraw(
            lambda: target2_ax_new.plot(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[
                    max(0, ~xmax_21 - pw),
                    min(~xmax_21, target2_ax_new.p2c(target1.get_left())[0]),
                    1 / 200,
                ],
                color=TARGET2_COLOR,
            )
        )
        self.add(rx_21)

        self.play(
            LaggedStart(
                xmax_12 + pw,
                xmax_21 @ (target2_ax_new.p2c(target1.get_left())[0] + pw),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(2)


class FontTest(Scene):
    def construct(self):
        tex = Text("Hello", font="Maple Mono")
        self.add(tex)
