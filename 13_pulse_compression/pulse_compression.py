# pulse_compression.py

import sys

from manim import *
from MF_Tools import VT
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower, get_blocks
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True

FONT = "Maple Mono CN"

BLOCKS = get_blocks()

GOOD = BLUE
OK = GREY
BAD = RED


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


class Issue(MovingCameraScene):
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
        pw = VT(0.2)
        f = 10
        tx = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t),
                x_range=[max(0, ~xmax - ~pw), ~xmax, 1 / 200],
                color=TX_COLOR,
            )
        )
        f_rx = VT(10)
        rx1 = always_redraw(
            lambda: target1_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * ~f_rx * t),
                x_range=[max(0, ~xmax_t1 - ~pw), min(~xmax_t1, 1), 1 / 200],
                color=TARGET1_COLOR,
            )
        )
        rx2 = always_redraw(
            lambda: target2_ax.plot(
                lambda t: 0.5 * np.sin(2 * PI * ~f_rx * t),
                x_range=[max(0, ~xmax_t2 - ~pw), min(~xmax_t2, 1), 1 / 200],
                color=TARGET2_COLOR,
            )
        )
        self.add(tx, rx1, rx2)

        radar.vgroup.set_z_index(1)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        pw_line = Line(ax.c2p(~xmax - ~pw, 1.2), ax.c2p(~xmax, 1.2))
        pw_line_l = Line(pw_line.get_start() + DOWN / 8, pw_line.get_start() + UP / 8)
        pw_line_r = Line(pw_line.get_end() + DOWN / 8, pw_line.get_end() + UP / 8)

        pw_label = MathTex(r"\tau").scale(1.2).next_to(pw_line, UP)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in pw_label[0]], lag_ratio=0.15),
            LaggedStart(
                Create(pw_line_l),
                Create(pw_line),
                Create(pw_line_r),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            FadeOut(pw_line, pw_line_l, pw_line_r),
            LaggedStart(
                target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
                target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            pw_label.animate.shift(UP),
            LaggedStart(
                xmax @ (ax.p2c(target2.get_left())[0]),
                xmax_t1 @ (~pw / 2),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )

        self.play(
            xmax @ 1.5,
            xmax_t1 @ (0.5 + ~pw / 2),
            xmax_t2 @ (0.5 + ~pw / 2 - target_dist),
            run_time=3,
        )

        self.wait(0.5)

        target1_pw_line = Line(
            target1_ax.c2p(~xmax_t1, -1),
            target1_ax.c2p(~xmax_t1 - ~pw, -1),
        )
        target1_pw_line_l = Line(
            target1_pw_line.get_start() + UP / 6, target1_pw_line.get_start() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())
        target1_pw_line_r = Line(
            target1_pw_line.get_end() + UP / 6, target1_pw_line.get_end() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())

        target2_pw_line = Line(
            target2_ax.c2p(~xmax_t2, 1),
            target2_ax.c2p(~xmax_t2 - ~pw, 1),
        )
        target2_pw_line_l = Line(
            target2_pw_line.get_start() + UP / 6, target2_pw_line.get_start() + DOWN / 6
        ).rotate(target2_pw_line.get_angle())
        target2_pw_line_r = Line(
            target2_pw_line.get_end() + UP / 6, target2_pw_line.get_end() + DOWN / 6
        ).rotate(target2_pw_line.get_angle())
        pw_label_target1 = pw_label.copy().next_to(
            target1_pw_line.get_center(), UP, MED_SMALL_BUFF
        )
        pw_label_target2 = pw_label.copy().next_to(
            target2_pw_line.get_center(), DOWN, MED_SMALL_BUFF
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(
                    Create(target1_pw_line_l),
                    Create(target1_pw_line),
                    Create(target1_pw_line_r),
                    lag_ratio=0.3,
                ),
                TransformFromCopy(pw_label[0], pw_label_target1[0], path_arc=-PI / 3),
                LaggedStart(
                    Create(target2_pw_line_l),
                    Create(target2_pw_line),
                    Create(target2_pw_line_r),
                    lag_ratio=0.3,
                ),
                ReplacementTransform(
                    pw_label[0], pw_label_target2[0], path_arc=-PI / 3
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        line = Line(
            target2_ax.c2p(~xmax_t2, 0.8), target1_ax.c2p(~xmax_t1 - ~pw * 1.05, -0.6)
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

        self.play(FadeIn(overlap))

        self.wait(0.5)

        pw_scale_factor = 4
        target1_pw_line_new = Line(
            target1_ax.c2p(~xmax_t1, -1),
            target1_ax.c2p(~xmax_t1 - ~pw / pw_scale_factor, -1),
        )
        target2_pw_line_new = Line(
            target2_ax.c2p(~xmax_t2, 1),
            target2_ax.c2p(~xmax_t2 - ~pw / pw_scale_factor, 1),
        )

        self.play(
            LaggedStart(
                FadeOut(overlap),
                AnimationGroup(
                    f_rx @ 30,
                    pw @ (~pw / pw_scale_factor),
                    Transform(target1_pw_line, target1_pw_line_new),
                    Transform(
                        target1_pw_line_l,
                        Line(
                            target1_pw_line_new.get_start() + UP / 6,
                            target1_pw_line_new.get_start() + DOWN / 6,
                        ).rotate(target1_pw_line_new.get_angle()),
                    ),
                    Transform(
                        target1_pw_line_r,
                        Line(
                            target1_pw_line_new.get_end() + UP / 6,
                            target1_pw_line_new.get_end() + DOWN / 6,
                        ).rotate(target1_pw_line_new.get_angle()),
                    ),
                    Transform(target2_pw_line, target2_pw_line_new),
                    Transform(
                        target2_pw_line_l,
                        Line(
                            target2_pw_line_new.get_start() + UP / 6,
                            target2_pw_line_new.get_start() + DOWN / 6,
                        ).rotate(target2_pw_line.get_angle()),
                    ),
                    Transform(
                        target2_pw_line_r,
                        Line(
                            target2_pw_line_new.get_end() + UP / 6,
                            target2_pw_line_new.get_end() + DOWN / 6,
                        ).rotate(target2_pw_line.get_angle()),
                    ),
                    pw_label_target1[0].animate.next_to(
                        target1_pw_line_new, UP, MED_SMALL_BUFF
                    ),
                    pw_label_target2[0].animate.next_to(
                        target2_pw_line_new, DOWN, MED_SMALL_BUFF
                    ),
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(
                    target1_pw_line,
                    target2_pw_line,
                    target1_pw_line_r,
                    target1_pw_line_l,
                    target2_pw_line_r,
                    target2_pw_line_l,
                    pw_label_target1[0],
                    pw_label_target2[0],
                ),
                AnimationGroup(
                    xmax_t1 @ (1 + ~pw),
                    xmax_t2 @ (1 + ~pw),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        fs = 100
        f = 25
        noise_std = VT(0.1)

        x_len = config.frame_width * 0.4
        y_len = config.frame_height * 0.4
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
        ).next_to(self.camera.frame, UP)

        stop_time = 4
        N = stop_time * fs
        t = np.linspace(0, stop_time, N)
        fft_len = N * 8
        freq = np.linspace(-fs / 2, fs / 2, fft_len)
        amp = VT(0.1)

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

        self.add(ax, X_k_plot)

        new_ax_copy = ax.copy().move_to(self.camera.frame.get_top()).shift(DOWN * 0.9)

        new_scene = Group(radar.vgroup, target1, target2, new_ax_copy)

        snr_line = always_redraw(
            lambda: Line(
                ax.c2p(f + 3, -lin2db(~noise_std)),
                [ax.c2p(f + 3, 0)[0], ax.input_to_graph_point(f, X_k_plot)[1], 0],
            )
        )
        snr_line_u = always_redraw(
            lambda: Line(
                snr_line.get_top() + LEFT / 8,
                snr_line.get_top() + RIGHT / 8,
            )
        )
        snr_line_d = always_redraw(
            lambda: Line(
                snr_line.get_bottom() + LEFT / 8,
                snr_line.get_bottom() + RIGHT / 8,
            )
        )
        snr_label = always_redraw(
            lambda: Text("SNR", font=FONT).next_to(snr_line, RIGHT, MED_SMALL_BUFF)
        )
        self.add(snr_line, snr_line_u, snr_line_d, snr_label)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(
                    ax.animate.move_to(self.camera.frame.get_top()).shift(DOWN * 0.9),
                    self.camera.frame.animate.scale_to_fit_height(
                        new_scene.height * 1.2
                    )
                    .move_to(new_scene)
                    .set_x(0),
                ),
            )
        )

        self.wait(0.5)

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=x_len,
                y_length=y_len,
            )
            .set_z_index(-1)
            .next_to(self.camera.frame, LEFT)
            .set_y(ax.get_y())
        )

        pw_plot = VT(0.1)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: ~pulse_amp * np.sin(2 * PI * pulse_f * t)
                if t < ~pw_plot
                else 0,
                x_range=[0, 1, 1 / 200],
                color=TX_COLOR,
            )
        )
        self.add(pulse_ax, pulse)

        self.play(
            Group(pulse_ax, ax).animate.arrange(RIGHT, MED_LARGE_BUFF).set_y(ax.get_y())
        )

        tx_pulse_label = Text("Transmit Pulse", font=FONT).next_to(pulse_ax, DOWN)

        self.wait(0.5)

        self.play(Write(tx_pulse_label))

        self.wait(0.5)

        energy_eqn = (
            MathTex(r"E = P \cdot t")
            .scale(2.5)
            .next_to(self.camera.frame.get_bottom(), UP, LARGE_BUFF * 2)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[
                    m.set_opacity(0)
                    .shift(DOWN / 2)
                    .animate.shift(UP / 2)
                    .set_opacity(1)
                    for m in energy_eqn[0]
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(energy_eqn[0][4].animate.set_color(GREEN))

        self.wait(0.5)

        self.play(pw_plot @ (~pw_plot * 3), amp @ (~amp * 3))

        self.wait(0.5)

        self.play(pw_plot @ (~pw_plot / 3), amp @ (~amp / 3))

        self.wait(0.5)

        self.play(energy_eqn[0][4].animate.set_color(WHITE))

        self.wait(0.5)

        self.play(energy_eqn[0][2].animate.set_color(GREEN))

        self.wait(0.5)

        self.play(pulse_amp @ (~pulse_amp * 3), amp @ (~amp * 3))

        self.wait(0.5)

        self.play(pulse_amp @ (~pulse_amp / 3), amp @ (~amp / 3))

        self.wait(0.5)

        self.play(energy_eqn[0][2].animate.set_color(WHITE))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self) * 1.5))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        relation_table = (
            MobjectTable(
                [
                    [
                        MathTex(r"\downarrow").scale(2),
                        Text("+", font=FONT, color="GREEN").scale(2),
                        Text("-", font=FONT, color="RED").scale(2),
                    ],
                    [
                        MathTex(r"\uparrow").scale(2),
                        Text("-", font=FONT, color="RED").scale(2),
                        Text("+", font=FONT, color="GREEN").scale(2),
                    ],
                ],
                col_labels=[
                    MathTex(r"\tau").scale(2),
                    MathTex(r"\Delta R"),
                    Tex("SNR"),
                ],
            )
            .scale(1.5)
            .move_to(self.camera.frame)
        )

        self.play(
            LaggedStart(
                *[Create(m) for m in relation_table.get_horizontal_lines()],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[Create(m) for m in relation_table.get_vertical_lines()], lag_ratio=0.2
            ),
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_col_labels()], lag_ratio=0.2
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_rows()[1]], lag_ratio=0.2
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(m) for m in relation_table.get_rows()[2]], lag_ratio=0.2
            )
        )

        self.wait(0.5)

        first_option = SurroundingRectangle(relation_table.get_rows()[1])
        second_option = SurroundingRectangle(relation_table.get_rows()[2])
        good_rres = SurroundingRectangle(
            relation_table.get_rows()[1][1], buff=MED_LARGE_BUFF
        )
        good_snr = SurroundingRectangle(
            relation_table.get_rows()[2][2], buff=MED_LARGE_BUFF
        )

        self.play(Create(first_option))

        self.wait(0.5)

        first_option.save_state()
        self.play(Transform(first_option, second_option))

        self.wait(0.5)

        self.play(first_option.animate.restore())

        self.wait(0.5)

        self.play(
            TransformFromCopy(first_option, good_snr),
            ReplacementTransform(first_option, good_rres),
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self) * 1.5))

        self.wait(2)


class Options(MovingCameraScene):
    def construct(self):
        pulse_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.5),
        ).set_z_index(-1)

        rres_nl = NumberLine(
            x_range=[0, 10, 1], length=pulse_ax.height * 1.3, rotation=PI / 2
        ).set_z_index(-1)
        snr_nl = (
            NumberLine(
                x_range=[0, 10, 1], length=pulse_ax.height * 1.3, rotation=PI / 2
            )
            .next_to(rres_nl, RIGHT, LARGE_BUFF * 2)
            .set_z_index(-1)
        )

        Group(pulse_ax, Group(rres_nl, snr_nl)).arrange(RIGHT, LARGE_BUFF * 20)

        rres_label = always_redraw(
            lambda: MathTex(r"\Delta R").next_to(rres_nl, UP, MED_SMALL_BUFF)
        )
        snr_label = always_redraw(
            lambda: Tex(r"SNR").next_to(snr_nl, UP, MED_SMALL_BUFF)
        )

        rres = VT(9)
        rres_dot = always_redraw(
            lambda: Dot(
                color=interpolate_color(OK, BAD, (~rres - 5) / 5)
                if ~rres > 5
                else interpolate_color(GOOD, OK, ~rres / 5),
                radius=DEFAULT_DOT_RADIUS * 3,
            ).move_to(rres_nl.n2p(~rres))
        )

        snr = VT(9)
        snr_dot = always_redraw(
            lambda: Dot(
                color=interpolate_color(OK, GOOD, (~snr - 5) / 5)
                if ~snr > 5
                else interpolate_color(BAD, OK, ~snr / 5),
                radius=DEFAULT_DOT_RADIUS * 3,
            ).move_to(snr_nl.n2p(~snr))
        )

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: ~pulse_amp * np.sin(2 * PI * pulse_f * t)
                if t < ~pw_plot
                else 0,
                x_range=[0, 1, 1 / 1000],
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                color=TX_COLOR,
            )
        )
        self.add(
            pulse_ax,
            pulse,
            rres_nl,
            snr_nl,
            rres_label,
            snr_label,
            snr_dot,
            rres_dot,
        )

        self.play(
            Group(pulse_ax, Group(rres_nl, snr_nl)).animate.arrange(RIGHT, LARGE_BUFF)
        )

        self.wait(0.5)

        self.play(pw_plot @ 0.1, rres @ 1, snr @ 1, run_time=5)

        self.wait(0.5)

        check = (
            Text("✔", font=FONT, color=GOOD)
            .scale(1.5)
            .next_to(rres_dot, LEFT, SMALL_BUFF)
        )
        qmark = Text("?", font=FONT).scale(1.5).next_to(snr_dot, RIGHT, SMALL_BUFF)

        self.play(GrowFromCenter(check))

        self.wait(0.5)

        self.play(GrowFromCenter(qmark))

        self.wait(0.5)

        self.play(pw_plot @ 0.3, rres @ 9, snr @ 9, run_time=5)

        self.wait(0.5)

        self.play(
            LaggedStart(
                Transform(
                    check,
                    check.copy().next_to(snr_dot, RIGHT, SMALL_BUFF),
                    path_arc=PI / 3,
                ),
                Transform(
                    qmark,
                    qmark.copy().next_to(rres_dot, LEFT, SMALL_BUFF),
                    path_arc=PI / 3,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(2)
