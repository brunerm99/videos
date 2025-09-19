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
TARGET1_COLOR = GREEN
TARGET2_COLOR = ORANGE
TARGET3_COLOR = BLUE


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


def chirp_pulse(t_val, pulse_start, pulse_width, f0, f1, amp, phase, ramp="quadratic"):
    t_rel = t_val - pulse_start - 1 / f0 / 4

    if -1 / f0 / 4 <= t_rel <= pulse_width:
        return amp * signal.chirp(
            t_rel,
            f0=f0,
            t1=pulse_width,
            f1=f1,
            method=ramp,
            phi=phase,
        )
    else:
        return 0


class Issue(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 0.4)

        self.play(radar.get_animation())

        self.wait(0.5)

        self.play(radar.vgroup.animate.to_corner(DL, LARGE_BUFF))

        self.wait(0.5)

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
        self.next_section(skip_animations=skip_animations(True))

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
        self.next_section(skip_animations=skip_animations(True))
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
        pulse_f1 = VT(pulse_f)
        pulse_x1 = VT(1)
        pulse_x0 = VT(0)

        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: ~pulse_amp
                * signal.chirp(
                    t - 1 / pulse_f / 4,
                    pulse_f,
                    ~pw_plot,
                    ~pulse_f1,
                    method="quadratic",
                )
                if t < ~pw_plot + ~pulse_x0
                else 0,
                x_range=[~pulse_x0, ~pulse_x1, 1 / 1000],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
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
                    path_arc=-PI / 3,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Group(rres_nl, snr_nl, qmark, check).animate.shift(RIGHT * 5),
            self.camera.frame.animate.scale_to_fit_width(pulse_ax.width * 1.5).move_to(
                pulse_ax
            ),
            pw_plot @ 0.5,
        )

        self.wait(0.5)

        pulse_line = Line(
            pulse_ax.c2p(0, ~pulse_amp), pulse_ax.c2p(~pw_plot, ~pulse_amp), color=GREEN
        ).shift(UP / 2)
        pulse_line_l = Line(
            pulse_line.get_left() + DOWN / 4,
            pulse_line.get_left() + UP / 4,
            color=GREEN,
        )
        pulse_line_r = Line(
            pulse_line.get_right() + DOWN / 4,
            pulse_line.get_right() + UP / 4,
            color=GREEN,
        )

        self.play(
            LaggedStart(
                Create(pulse_line_l),
                Create(pulse_line),
                Create(pulse_line_r),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(pulse_f1 @ (pulse_f * 4), run_time=3)

        self.wait(0.5)

        pulse_line_new = Line(
            pulse_ax.c2p(0, ~pulse_amp),
            pulse_ax.c2p(0.1, ~pulse_amp),
            color=GREEN,
        ).shift(UP / 2)

        pulse_line.save_state()
        pulse_line_r.save_state()
        self.play(
            pw_plot @ 0.1,
            Transform(pulse_line, pulse_line_new),
            Transform(
                pulse_line_r,
                Line(
                    pulse_line_new.get_right() + DOWN / 4,
                    pulse_line_new.get_right() + UP / 4,
                    color=GREEN,
                ),
            ),
        )

        self.wait(0.5)

        self.play(
            pw_plot @ 0.5,
            pulse_line.animate.restore(),
            pulse_line_r.animate.restore(),
        )

        self.wait(0.5)

        self.play(pulse_f1 @ pulse_f)

        self.wait(0.5)

        theory_paper = ImageMobject(
            "../props/static/Theory and Design of Chirp Radars.png"
        ).scale_to_fit_height(fh(self, 0.7))
        new_chirp = ImageMobject(
            "../props/static/Chirp A New Radar Technique.jpg"
        ).scale_to_fit_width(theory_paper.width * 1.3)
        fundamentals_of_radar_dsp = ImageMobject(
            "../props/static/Fundamentals of Radar DSP Book Cover.jpg"
        ).scale_to_fit_height(fh(self, 0.7))
        resources = (
            Group(fundamentals_of_radar_dsp, new_chirp, theory_paper)
            .arrange(RIGHT, MED_LARGE_BUFF)
            .scale_to_fit_width(fw(self, 0.9))
            .move_to(self.camera.frame)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.play(LaggedStart(*[GrowFromCenter(m) for m in resources], lag_ratio=0.3))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.shift(UP * fh(self)) for m in resources], lag_ratio=0.3
            )
        )

        self.remove(*resources)

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale_to_fit_height(config.frame_height * 1).next_to(
            self.camera.frame.get_left(), LEFT, LARGE_BUFF * 3
        )

        self.remove(
            snr_dot,
            snr_nl,
            qmark,
            check,
            rres_dot,
            rres_nl,
            rres_label,
            snr_label,
        )
        self.next_section(skip_animations=skip_animations(True))

        plane = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .next_to(pulse, RIGHT, LARGE_BUFF * 3)
        )

        self.play(
            LaggedStart(
                FadeOut(pulse_ax),
                pulse_x1 @ ~pw_plot,
                Uncreate(pulse_line_r),
                Uncreate(pulse_line),
                Uncreate(pulse_line_l),
                radar.vgroup.animate.shift(
                    pulse_ax.c2p(0, 0) - radar.radome.get_right()
                ),
                self.camera.frame.animate.scale_to_fit_height(radar.vgroup.height * 1.7)
                .move_to(
                    Group(
                        radar.vgroup.copy().shift(
                            pulse_ax.c2p(0, 0) - radar.radome.get_right()
                        ),
                        pulse,
                    )
                )
                .shift(UP * 2 + RIGHT * 4),
                plane.shift(RIGHT * 10).animate.shift(LEFT * 10),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        pulse_rtn_ax = pulse_ax.copy().rotate(PI)
        pulse_rtn_ax.shift(plane.get_left() - pulse_rtn_ax.c2p(0, 0))

        pulse_rtn_x0 = VT(-~pw_plot)
        pulse_rtn_x1 = VT(0)

        pulse_amp_2 = VT(0)
        pulse_amp_3 = VT(0)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0)
        pulse_t_start_3 = VT(0)
        pulse_f1_2 = VT(~pulse_f1)
        allow_2 = VT(0)
        allow_3 = VT(0)
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1_2,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                # use_smoothing=False,
            )
        )

        # pulse_rtn = always_redraw(
        #     lambda: pulse_rtn_ax.plot(
        #         lambda t: ~pulse_amp
        #         * signal.chirp(
        #             t - 1 / pulse_f / 4,
        #             pulse_f,
        #             ~pw_plot,
        #             ~pulse_f1,
        #             method="quadratic",
        #         )
        #         if t < ~pw_plot + ~pulse_rtn_x0
        #         else 0,
        #         x_range=[
        #             max(0, ~pulse_rtn_x0),
        #             min(~pulse_rtn_x1, pulse_rtn_ax.p2c(radar.radome.get_right())[0]),
        #             1 / 1000,
        #         ],
        #         stroke_width=DEFAULT_STROKE_WIDTH * 1,
        #         color=RX_COLOR,
        #     )
        # )

        self.next_section(skip_animations=skip_animations(True))
        self.add(pulse_rtn)

        self.play(
            LaggedStart(
                AnimationGroup(
                    pulse_x0 + 3,
                    pulse_x1 + 3,
                ),
                AnimationGroup(pulse_rtn_x0 + 1, pulse_rtn_x1 + 1),
                lag_ratio=0.4,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale_to_fit_width(pulse_rtn.width * 1.8).move_to(
                pulse_rtn
            )
        )

        self.wait(0.5)

        target_start = VT(~pulse_rtn_x1)
        target_arrow = always_redraw(
            lambda: Arrow(
                pulse_rtn_ax.c2p(~target_start, -1),
                pulse_rtn_ax.c2p(~target_start, -~pulse_amp),
            )
        )
        target_label = always_redraw(
            lambda: Text("Target Start", font=FONT)
            .scale(0.3)
            .next_to(target_arrow, UP, SMALL_BUFF)
        )
        target_start_dot = always_redraw(
            lambda: Dot(pulse_rtn_ax.input_to_graph_point(~target_start, pulse_rtn))
        )

        self.play(
            self.camera.frame.animate.shift(UP),
            Create(target_start_dot),
            FadeIn(target_arrow, target_label),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            target_start.animate(
                rate_func=rate_functions.there_and_back
            ).increment_value(-~pw_plot),
            run_time=4,
        )

        self.wait(0.5)

        self.play(FadeOut(target_arrow, target_start_dot, target_label))

        self.wait(0.5)

        pulse_static = pulse.copy().next_to(pulse_rtn, UP, SMALL_BUFF).shift(LEFT * 5)
        self.add(pulse_static)

        self.next_section(skip_animations=skip_animations(True))
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(
                    Group(pulse_rtn, pulse_static.copy().shift(RIGHT * 5))
                ),
                pulse_static.animate(
                    rate_func=rate_functions.ease_out_bounce, run_time=2
                ).shift(RIGHT * 5),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        start = DashedLine(
            pulse_rtn_ax.c2p(~pulse_rtn_x1, -2),
            pulse_rtn_ax.c2p(~pulse_rtn_x1, 1),
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )

        self.play(Create(start))

        tx_pulse_label = Text("Tx Pulse", font=FONT, color=TX_COLOR).next_to(
            pulse_static.copy().set_stroke(opacity=0.3).shift(UP * 3),
            UP,
            MED_SMALL_BUFF,
        )
        self.add(tx_pulse_label)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        start_arrow = Arrow(
            radar.radome.get_right() + RIGHT * 2 + DOWN * 2,
            radar.radome.get_right(),
            color=YELLOW,
        )

        self.play(
            LaggedStart(
                Uncreate(start),
                pulse_static.animate.set_stroke(opacity=0.3).shift(UP * 3),
                self.camera.frame.animate.restore(),
                AnimationGroup(
                    pulse_rtn_x1 @ (pulse_rtn_ax.p2c(radar.radome.get_right())[0]),
                    pulse_rtn_x0
                    @ (pulse_rtn_ax.p2c(radar.radome.get_right())[0] - ~pw_plot),
                ),
                GrowArrow(start_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
            .next_to(plane, UP, -LARGE_BUFF)
            .shift(RIGHT / 2)
        )
        target3 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(radar.vgroup.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET3_COLOR)
            .set_color(TARGET3_COLOR)
            .next_to(plane, DOWN, -LARGE_BUFF)
            .shift(RIGHT * 1.4)
        )

        pulse_2_opacity = VT(0)
        pulse_3_opacity = VT(0)
        pulse_t2_shift = VT(1.3)
        pulse_t3_shift = VT(1.3)

        pulse_rtn_t2 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(UP * ~pulse_t2_shift)
            .set_stroke(opacity=~pulse_2_opacity)
        )
        pulse_rtn_t3 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=-~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(0, ~pulse_rtn_x0 - max(~pulse_t_start_2, ~pulse_t_start_3)),
                    min(
                        ~pulse_rtn_x1,
                        pulse_rtn_ax.p2c(radar.radome.get_right())[0],
                    ),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(DOWN * ~pulse_t3_shift)
            .set_stroke(opacity=~pulse_3_opacity)
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(FadeOut(start_arrow))

        self.wait(0.5)

        self.add(pulse_rtn_t2, pulse_rtn_t3)
        self.play(
            LaggedStart(
                plane.animate.scale(0.5),
                AnimationGroup(
                    pulse_2_opacity @ 1,
                    target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                AnimationGroup(
                    pulse_3_opacity @ 1,
                    target3.shift(RIGHT * 10).animate.shift(LEFT * 10),
                ),
                AnimationGroup(
                    pulse_t_start_2 @ 0.1,
                    pulse_t_start_3 @ 0.22,
                ),
                pulse_amp_2 @ 0.3,
                pulse_amp_3 @ 0.3,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                LaggedStart(pulse_t2_shift @ 0, pulse_2_opacity @ 0, lag_ratio=0.2),
                allow_2 @ 1,
                LaggedStart(pulse_t3_shift @ 0, pulse_3_opacity @ 0, lag_ratio=0.2),
                allow_3 @ 1,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        comparison_group = Group(
            pulse_rtn, pulse_static.copy().next_to(pulse_rtn, UP, MED_LARGE_BUFF)
        )

        start = DashedLine(
            comparison_group[1].get_corner(UL) + UP / 2,
            comparison_group[1].get_corner(UL) + DOWN * 4.7,
            dash_length=DEFAULT_DASH_LENGTH * 3,
            color=YELLOW,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_static.animate.set_stroke(opacity=1).move_to(comparison_group[1]),
                self.camera.frame.animate.scale_to_fit_height(
                    (pulse_rtn.height + pulse_static.height) * 1.8
                ).move_to(comparison_group),
                Create(start),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            Group(pulse_static, start).animate.shift(
                RIGHT * (pulse_rtn.get_left() - pulse_static.get_left())[0]
            ),
            run_time=0.5,
        )
        self.play(Group(pulse_static, start).animate.shift(RIGHT * 2), run_time=0.5)
        self.play(Group(pulse_static, start).animate.shift(LEFT * 0.4), run_time=0.5)

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_phase_2.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(80),
                pulse_phase_1.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(40),
                pulse_phase_3.animate(
                    rate_func=rate_functions.there_and_back
                ).increment_value(110),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(0.5)

        lu_bez = CubicBezier(
            pulse_rtn.get_corner(DL) + [0, -0.1, 0],
            pulse_rtn.get_corner(DL) + [0, -1, 0],
            pulse_rtn.get_bottom() + [0, -1.5, 0],
            pulse_rtn.get_bottom() + [0, -3, 0],
        )
        ru_bez = CubicBezier(
            pulse_rtn.get_corner(DR) + [0, -0.1, 0],
            pulse_rtn.get_corner(DR) + [0, -1, 0],
            pulse_rtn.get_bottom() + [0, -1.5, 0],
            pulse_rtn.get_bottom() + [0, -3, 0],
        )

        nums = [str(np.random.randint(0, 2)) for _ in range(16)]
        info = Text(
            "".join(nums), font=FONT, font_size=DEFAULT_FONT_SIZE * 1.2
        ).next_to(ru_bez.get_end(), DOWN, LARGE_BUFF * 3)
        for char, num in zip(info, nums):
            if num == "1":
                char.set_color(GOOD)
            else:
                char.set_color(BAD)

        info_group = Group(pulse_rtn, info)

        ld_bez = CubicBezier(
            pulse_rtn.get_bottom() + [0, -3, 0],
            pulse_rtn.get_bottom() + [0, -4.5, 0],
            info.get_corner(UL) + [0, 1, 0],
            info.get_corner(UL) + [0, 0.1, 0],
        )
        rd_bez = CubicBezier(
            pulse_rtn.get_bottom() + [0, -3, 0],
            pulse_rtn.get_bottom() + [0, -4.5, 0],
            info.get_corner(UR) + [0, 1, 0],
            info.get_corner(UR) + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                FadeOut(radar.vgroup, start, pulse_static, target2, target3, plane),
                self.camera.frame.animate.scale_to_fit_height(
                    info_group.height * 1.2
                ).move_to(info_group),
                pulse_f1.animate(run_time=2).set_value(pulse_f * 3),
                AnimationGroup(Create(lu_bez), Create(ru_bez)),
                AnimationGroup(Create(ld_bez), Create(rd_bez)),
                LaggedStart(*[GrowFromCenter(m) for m in info], lag_ratio=0.1),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self) * 2))

        self.wait(2)


class Encoding(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        f_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.25],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        lfm_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        Group(f_ax, lfm_ax).arrange(DOWN, MED_LARGE_BUFF).to_edge(RIGHT, LARGE_BUFF)
        f_label = (
            Text("Frequency", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(f_ax, LEFT, SMALL_BUFF)
        )
        amp_label = (
            Text("Amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(lfm_ax, LEFT, SMALL_BUFF)
        )

        m = VT(0)
        f = 4
        f1 = VT(f)
        f_plot = always_redraw(
            lambda: f_ax.plot(lambda t: 0.2 + ~m * t, color=TX_COLOR)
        )
        lfm_plot = always_redraw(
            lambda: lfm_ax.plot(
                lambda t: chirp_pulse(t, 0, 1, f, ~f1, 1, 0, ramp="linear"),
                x_range=[0, 1, 1 / 1000],
                color=TX_COLOR,
            )
        )

        lfm_label = (
            Text("Linear\nFrequency\nModulation", font=FONT)
            .scale_to_fit_width(fw(self, 0.25))
            .to_edge(LEFT, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                Write(lfm_label),
                Create(f_ax),
                Create(lfm_ax),
                FadeIn(f_label),
                FadeIn(amp_label),
                lag_ratio=0.3,
            )
        )
        self.play(Create(f_plot), Create(lfm_plot))

        self.wait(0.5)

        self.play(m @ 0.8, f1 @ 30)

        self.wait(0.5)

        thumbnail = (
            ImageMobject("../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale_to_fit_width(lfm_label.width * 1.2)
            .next_to(lfm_label, DOWN, LARGE_BUFF * 10)
        )
        tn_box = SurroundingRectangle(thumbnail, buff=0)

        title_top = Text("What is FMCW Radar and", font=FONT).scale(0.5)
        title_bot = Text("why is it useful?", font=FONT).scale(0.5)
        title = Group(title_top, title_bot).arrange(DOWN).next_to(thumbnail, DOWN)
        tn = Group(thumbnail, tn_box, title)

        self.play(
            Group(lfm_label, tn)
            .animate.arrange(DOWN, LARGE_BUFF)
            .set_x(lfm_label.get_x())
        )

        self.wait(0.5)

        phase_ax = Axes(
            x_range=[0, 8, 1],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        phase_amp_ax = Axes(
            x_range=[0, 8, 1],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.5),
            y_length=fh(self, 0.4),
        )
        Group(phase_ax, phase_amp_ax).arrange(DOWN, MED_LARGE_BUFF).to_edge(
            RIGHT, LARGE_BUFF
        ).shift(DOWN * fh(self))
        phase_label = (
            Text("Phase", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(phase_ax, LEFT, SMALL_BUFF)
        )
        phase_amp_label = (
            Text("Amplitude", font=FONT)
            .scale(0.5)
            .rotate(PI / 2)
            .next_to(phase_amp_ax, LEFT, SMALL_BUFF)
        )

        np.random.seed(2)
        phase_seq = (np.random.randint(0, 2, 8) - 0.5) * -2
        phase_plot = phase_ax.plot(
            lambda t: 1 if phase_seq[int(np.floor(t))] > 0 else -1,
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_amp_plot = phase_amp_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t)
            * (1 if phase_seq[int(np.floor(t))] > 0 else -1),
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_labels = Group(
            *[
                Text("0", color=BAD, font=FONT).move_to(phase_ax.c2p(idx + 0.5, -0.5))
                if num < 0
                else Text("1", color=GOOD, font=FONT).move_to(
                    phase_ax.c2p(idx + 0.5, 0.5)
                )
                for idx, num in enumerate(phase_seq)
            ]
        )

        phase_mod_label = (
            Text("Phase\nModulation", font=FONT)
            .scale_to_fit_width(fw(self, 0.25))
            .to_edge(LEFT, LARGE_BUFF)
            .shift(DOWN * fh(self))
        )
        self.add(
            phase_labels,
            phase_mod_label,
            phase_amp_plot,
            phase_amp_label,
            phase_label,
            phase_ax,
            phase_amp_ax,
            phase_plot,
        )

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        np.random.seed(3)
        phase_seq_new = (np.random.randint(0, 2, 8) - 0.5) * -2
        phase_plot_new = phase_ax.plot(
            lambda t: 1 if phase_seq_new[int(np.floor(t))] > 0 else -1,
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_amp_plot_new = phase_amp_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t)
            * (1 if phase_seq_new[int(np.floor(t))] > 0 else -1),
            x_range=[0, 8 - 1 / 1000, 1 / 1000],
            use_smoothing=False,
            color=TX_COLOR,
        )
        phase_labels_new = Group(
            *[
                Text("0", color=BAD, font=FONT).move_to(phase_ax.c2p(idx + 0.5, -0.5))
                if num < 0
                else Text("1", color=GOOD, font=FONT).move_to(
                    phase_ax.c2p(idx + 0.5, 0.5)
                )
                for idx, num in enumerate(phase_seq_new)
            ]
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[
                    Transform(old, new)
                    for old, new in zip(phase_labels, phase_labels_new)
                ],
                lag_ratio=0.1,
            ),
            Transform(phase_plot, phase_plot_new),
            Transform(phase_amp_plot, phase_amp_plot_new),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(fh(self, 2.1)).shift(
                UP * fh(self) / 2
            )
        )

        self.wait(0.5)

        new_cam = self.camera.frame.copy().scale(0.5).shift(RIGHT * fw(self, 0.5))
        all_modulations = Text("All Modulation Schemes", font=FONT).next_to(
            new_cam.get_top(), DOWN, LARGE_BUFF
        )

        self.play(
            self.camera.frame.animate.scale(0.5).shift(RIGHT * fw(self, 0.5)),
            Write(all_modulations),
        )

        self.wait(0.5)

        gen_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.4),
            y_length=fh(self, 0.3),
        )
        mod_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.4),
            y_length=fh(self, 0.3),
        )
        arrow = MathTex(r"\Rightarrow").scale(2)
        Group(gen_ax, arrow, mod_ax).arrange(RIGHT, MED_LARGE_BUFF).move_to(new_cam)
        gen_plot = gen_ax.plot(
            lambda t: np.sin(2 * PI * 6 * t),
            x_range=[0, 1, 1 / 400],
            color=TX_COLOR,
        )

        self.play(LaggedStart(Create(gen_ax), Create(gen_plot), lag_ratio=0.3))

        self.wait(0.5)

        gen_bw = (
            MathTex(r"B = 0 \text{ Hz}").scale(2).next_to(gen_ax, DOWN, MED_LARGE_BUFF)
        )
        mod_bw = (
            MathTex(r"B > 0 \text{ Hz}").scale(2).next_to(mod_ax, DOWN, MED_LARGE_BUFF)
        )

        self.play(FadeIn(gen_bw, shift=UP))

        self.wait(0.5)

        qmark = Text("?", font=FONT, color=YELLOW).scale(1.5).move_to(mod_ax).shift(UP)

        self.play(
            LaggedStart(
                GrowFromCenter(arrow),
                Create(mod_ax),
                GrowFromCenter(qmark),
                FadeIn(mod_bw, shift=UP),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(Group(lfm_label, lfm_ax, f_ax)))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(False))

        hl_l = VT(0)
        hl_r = VT(0)
        hl = always_redraw(
            lambda: f_ax.plot(
                lambda t: 0.2 + ~m * t, color=YELLOW, x_range=[~hl_l, ~hl_r, 1 / 1000]
            ),
        )

        self.add(hl)

        self.play(LaggedStart(hl_r @ 1, hl_l @ 1, lag_ratio=0.2))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(0.5)

        lines = Group(
            *[
                DashedLine(
                    phase_amp_ax.c2p(2, 1),
                    phase_amp_ax.c2p(2, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
                DashedLine(
                    phase_amp_ax.c2p(4, 1),
                    phase_amp_ax.c2p(4, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
                DashedLine(
                    phase_amp_ax.c2p(7, 1),
                    phase_amp_ax.c2p(7, -1),
                    color=YELLOW,
                    dash_length=DEFAULT_DASH_LENGTH * 2,
                ),
            ]
        )

        self.play(LaggedStart(*[Create(m) for m in lines], lag_ratio=0.3))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))

        self.wait(2)


class Overlap(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        pulse_rtn_ax = pulse_ax.copy()
        Group(pulse_ax, pulse_rtn_ax).arrange(
            DOWN,
            MED_LARGE_BUFF,
        )

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_rtn_x0 = VT(-~pw_plot)
        pulse_rtn_x1 = VT(0)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f)
        pulse_amp_2 = VT(0)
        pulse_amp_3 = VT(0)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0)
        pulse_t_start_3 = VT(0)
        pulse_f1_2 = VT(~pulse_f1)
        allow_1 = VT(1)
        allow_2 = VT(0)
        allow_3 = VT(0)
        pulse_start_offset = VT(0)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0 + ~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    0,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
                # use_smoothing=False,
            )
        )
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: ~allow_1
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(~pulse_rtn_x0, ~min_x),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
                # use_smoothing=False,
            )
        )

        target1 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_height(fh(self, 0.2))
            .rotate(PI * 0.75)
            .set_fill(TARGET1_COLOR)
            .to_edge(RIGHT, LARGE_BUFF)
            .set_y(pulse_rtn_ax.get_y())
        )

        self.add(pulse_ax, pulse_rtn_ax, pulse_rtn, pulse)

        self.wait(0.5)

        self.play(
            pulse_rtn_x1 @ ~pw_plot,
            pulse_rtn_x0 @ 0,
            target1.shift(RIGHT * 10).animate.shift(LEFT * 10),
        )

        self.wait(0.5)

        with_lfm = (
            Text("* with linear\n  frequency\n  modulation", font=FONT)
            .scale(0.4)
            .to_corner(UL, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                FadeIn(with_lfm),
                pulse_f1 @ (pulse_f * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        target2 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(target1.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET2_COLOR)
            .set_color(TARGET2_COLOR)
            .next_to(target1, UP, 0)
            .shift(LEFT * 2)
            .shift(RIGHT / 2)
        )
        target3 = (
            SVGMobject("../props/static/plane.svg")
            .scale_to_fit_width(target1.width)
            .rotate(PI * 0.75)
            .scale(0.5)
            .set_fill(TARGET3_COLOR)
            .set_color(TARGET3_COLOR)
            .next_to(target1, DOWN, 0)
            .shift(LEFT * 2)
            .shift(RIGHT * 1.4)
        )

        self.play(pulse_ax.animate.shift(DOWN * 2))

        self.wait(0.5)

        pulse_2_opacity = VT(0)
        pulse_3_opacity = VT(0)
        pulse_t2_shift = VT(1.3)
        pulse_t3_shift = VT(1.3)

        pulse_rtn_t2 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                ),
                x_range=[
                    max(~pulse_rtn_x0, 0),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(UP * ~pulse_t2_shift)
            .set_stroke(opacity=~pulse_2_opacity)
        )
        pulse_rtn_t3 = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    max(~pulse_rtn_x0, 0),
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
            .shift(DOWN * ~pulse_t3_shift)
            .set_stroke(opacity=~pulse_3_opacity)
        )
        self.add(pulse_rtn_t2, pulse_rtn_t3)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                pulse_ax.animate.shift(UP * 2),
                target1.animate.scale(0.5).shift(LEFT * 2),
                target2.shift(RIGHT * 10).animate.shift(LEFT * 10),
                pulse_2_opacity @ 1,
                pulse_amp_2 @ ~pulse_amp,
                target3.shift(RIGHT * 10).animate.shift(LEFT * 10),
                pulse_3_opacity @ 1,
                pulse_amp_3 @ ~pulse_amp,
                pulse_t_start_2 @ 0.1,
                pulse_t_start_3 @ 0.28,
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        start = always_redraw(
            lambda: DashedLine(
                pulse_ax.c2p(~pulse_start_offset, 1),
                pulse_rtn_ax.c2p(~pulse_start_offset, -2),
                dash_length=DEFAULT_DASH_LENGTH * 3,
                color=YELLOW,
            )
        )

        self.play(Create(start))

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.1)

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.28)

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(start),
                pulse_start_offset @ 0,
                LaggedStart(
                    pulse_2_opacity @ 0,
                    pulse_t2_shift @ 0,
                    allow_2 @ 1,
                    lag_ratio=0.15,
                ),
                LaggedStart(
                    pulse_3_opacity @ 0,
                    pulse_t3_shift @ 0,
                    allow_3 @ 1,
                    lag_ratio=0.15,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(pulse_start_offset @ 0.1, run_time=0.5)
        self.play(pulse_start_offset @ 0.2, run_time=0.5)
        self.play(pulse_start_offset @ 0.05, run_time=0.5)
        self.play(pulse_start_offset @ 0, run_time=0.5)

        self.wait(0.5)

        lfm_structure = (
            Text("LFM structured", font=FONT)
            .scale(0.8)
            .next_to(pulse_rtn, DOWN, MED_SMALL_BUFF)
        )
        to_label = CubicBezier(
            with_lfm.get_bottom() + [0, -0.1, 0],
            with_lfm.get_bottom() + [0, -3, 0],
            lfm_structure.get_left() + [-2, 0, 0],
            lfm_structure.get_left() + [-0.1, 0, 0],
        )

        self.play(LaggedStart(Create(to_label), Write(lfm_structure), lag_ratio=0.3))

        self.wait(0.5)

        self.next_section(skip_animations=skip_animations(True))

        axes_group = Group(pulse.copy().shift(DOWN), pulse_rtn)
        self.play(
            LaggedStart(
                Uncreate(to_label),
                FadeOut(lfm_structure),
                target1.animate.shift(RIGHT * 6),
                target3.animate.shift(RIGHT * 6),
                target2.animate.shift(RIGHT * 6),
                self.camera.frame.animate.scale_to_fit_height(
                    axes_group.height * 1.8
                ).move_to(axes_group),
                pulse_ax.animate.shift(DOWN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        xcorr = MathTex(
            r"R_{xy}(\tau) = \left(s_{tx} \star s_{rx}\right)(\tau)"
        ).next_to(pulse, UP)
        xcorr[0][1].set_color(TX_COLOR)
        xcorr[0][2].set_color(RX_COLOR)
        xcorr[0][8:11].set_color(TX_COLOR)
        xcorr[0][12:15].set_color(RX_COLOR)

        self.play(
            Group(pulse_ax, pulse_rtn_ax).animate.shift(DOWN * 0.7),
            xcorr.shift(UP * 5).animate.shift(DOWN * 5),
        )

        self.wait(0.5)

        time_seq = (
            MathTex(r"0,1,2, \ldots t_{max} \text{ s}")
            .scale(0.7)
            .next_to(xcorr[0][4], DOWN, MED_LARGE_BUFF * 1.4)
        )
        time_bez_l = CubicBezier(
            xcorr[0][4].get_bottom() + [0, -0.1, 0],
            xcorr[0][4].get_bottom() + [0, -0.8, 0],
            time_seq.get_corner(UL) + [0, 0.8, 0],
            time_seq.get_corner(UL) + [0, 0.1, 0],
        )
        time_bez_r = CubicBezier(
            xcorr[0][4].get_bottom() + [0, -0.1, 0],
            xcorr[0][4].get_bottom() + [0, -0.8, 0],
            time_seq.get_corner(UR) + [0, 0.8, 0],
            time_seq.get_corner(UR) + [0, 0.1, 0],
        )

        self.play(
            Group(pulse_ax, pulse_rtn_ax).animate.shift(DOWN * 0.3),
            LaggedStart(
                AnimationGroup(Create(time_bez_l), Create(time_bez_r)),
                *[FadeIn(m) for m in time_seq[0]],
                lag_ratio=0.08,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(*time_seq[0]),
                AnimationGroup(Uncreate(time_bez_l), Uncreate(time_bez_r)),
                lag_ratio=0.2,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        new_axes = (
            Group(pulse_ax.copy(), pulse_rtn_ax.copy())
            .arrange(DOWN, -SMALL_BUFF)
            .set_y(Group(pulse, pulse_rtn).get_y())
        )
        new_cam = (
            self.camera.frame.copy()
            .scale_to_fit_height(new_axes.height * 1.2)
            .move_to(new_axes)
            .shift(LEFT * 3)
        )

        self.play(
            xcorr.animate.scale_to_fit_width(new_cam.width * 0.4).next_to(
                new_cam.get_top(),
                DOWN,
                MED_SMALL_BUFF,
            ),
            pulse_ax.animate.move_to(new_axes[0]),
            pulse_rtn_ax.animate.move_to(new_axes[1]),
            self.camera.frame.animate.scale_to_fit_height(new_axes.height * 1.2)
            .move_to(new_axes)
            .shift(LEFT * 3),
        )

        self.wait(2)


class XCorr(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))

        pulse_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=fw(self, 0.5),
                y_length=fh(self, 0.3),
            )
            .set_z_index(-1)
            .set_opacity(0)
        )
        pulse_rtn_ax = pulse_ax.copy()
        prod_ax = pulse_ax.copy()

        axes = Group(pulse_ax, pulse_rtn_ax).arrange(DOWN, -SMALL_BUFF)

        pw_plot = VT(0.3)
        pulse_amp = VT(0.3)
        pulse_f = 20
        pulse_rtn_x0 = VT(0)
        pulse_rtn_x1 = VT(~pw_plot)

        min_x = VT(0)

        pulse_f1 = VT(pulse_f * 5)
        pulse_amp_2 = VT(0.3)
        pulse_amp_3 = VT(0.3)
        pulse_phase_1 = VT(0)
        pulse_phase_2 = VT(0)
        pulse_phase_3 = VT(0)
        pulse_t_start_2 = VT(0.1)
        pulse_t_start_3 = VT(0.28)
        allow_1 = VT(1)
        allow_2 = VT(1)
        allow_3 = VT(1)
        pulse_start_offset = VT(0)
        pulse = always_redraw(
            lambda: pulse_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                ),
                x_range=[
                    ~min_x,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=TX_COLOR,
            )
        )
        pulse_rtn = always_redraw(
            lambda: pulse_rtn_ax.plot(
                lambda t: ~allow_1
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                + ~allow_2
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_2,
                    phase=~pulse_phase_2,
                )
                + ~allow_3
                * chirp_pulse(
                    t,
                    pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp_3,
                    phase=~pulse_phase_3,
                ),
                x_range=[
                    ~min_x,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=RX_COLOR,
            )
        )

        self.add(pulse, pulse_ax, pulse_rtn, pulse_rtn_ax)

        self.camera.frame.scale_to_fit_height(axes.height * 1.2).move_to(axes).shift(
            LEFT * 3
        )

        xcorr = MathTex(
            r"R_{xy}(\tau) = \left(s_{tx} \star s_{rx}\right)(\tau)"
        ).next_to(pulse, UP)
        xcorr[0][1].set_color(TX_COLOR)
        xcorr[0][2].set_color(RX_COLOR)
        xcorr[0][8:11].set_color(TX_COLOR)
        xcorr[0][12:15].set_color(RX_COLOR)

        xcorr.scale_to_fit_width(self.camera.frame.width * 0.4).next_to(
            self.camera.frame.get_top(),
            DOWN,
            MED_SMALL_BUFF,
        )

        self.add(xcorr)

        self.wait(0.5)

        self.play(pulse_start_offset @ -~pw_plot, min_x @ -~pw_plot)

        self.wait(0.5)

        times = MathTex(r"\times").scale(2).next_to(pulse, DOWN, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                pulse_rtn_ax.animate.shift(
                    DOWN
                    * ((pulse_rtn.get_top() - times.get_bottom())[1] + MED_SMALL_BUFF)
                ),
                GrowFromCenter(times),
                lag_ratio=0.3,
            )
        )

        pulse_samples = pulse_ax.get_riemann_rectangles(
            pulse,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            dx=0.005,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )
        pulse_rtn_samples = pulse_rtn_ax.get_riemann_rectangles(
            pulse_rtn,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            dx=0.005,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )

        self.play(
            LaggedStart(
                LaggedStart(*[FadeIn(m) for m in pulse_samples], lag_ratio=0.05),
                LaggedStart(*[FadeIn(m) for m in pulse_rtn_samples], lag_ratio=0.05),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        equal = MathTex(r"=").scale(2).next_to(pulse_rtn, DOWN, MED_SMALL_BUFF)
        prod_ax.set_x(pulse_ax.get_x())

        prod_ax.shift(
            DOWN * ((prod_ax.c2p(0, 0.5) - equal.get_bottom())[1] + MED_SMALL_BUFF)
        )

        prod = always_redraw(
            lambda: prod_ax.plot(
                lambda t: chirp_pulse(
                    t,
                    pulse_start=~pulse_start_offset,
                    pulse_width=~pw_plot,
                    f0=pulse_f,
                    f1=~pulse_f1,
                    amp=~pulse_amp,
                    phase=~pulse_phase_1,
                )
                * (
                    ~allow_1
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp,
                        phase=~pulse_phase_1,
                    )
                    + ~allow_2
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_t_start_2 + ~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp_2,
                        phase=~pulse_phase_2,
                    )
                    + ~allow_3
                    * chirp_pulse(
                        t,
                        pulse_start=~pulse_t_start_3 + ~pulse_rtn_x0,
                        pulse_width=~pw_plot,
                        f0=pulse_f,
                        f1=~pulse_f1,
                        amp=~pulse_amp_3,
                        phase=~pulse_phase_3,
                    )
                ),
                x_range=[
                    ~min_x,
                    min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
                    1 / 1000,
                ],
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                color=ORANGE,
            )
        )

        all_axes = Group(pulse_ax, pulse_rtn_ax, prod_ax)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(all_axes.height * 1)
                .move_to(all_axes)
                .shift(LEFT * 3),
                GrowFromCenter(equal),
                lag_ratio=0.3,
            )
        )

        prod_samples = prod_ax.get_riemann_rectangles(
            prod,
            x_range=[
                ~min_x,
                min(1, ~pulse_rtn_x1 + max(~pulse_t_start_2, ~pulse_t_start_3)),
            ],
            dx=0.005,
            stroke_color=YELLOW,
            color=YELLOW,
            show_signed_area=False,
            input_sample_type="center",
            fill_opacity=1,
            stroke_width=DEFAULT_STROKE_WIDTH * 0.2,
        )

        self.add(prod_ax)
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        m.animate.set_opacity(0).set_y(prod_ax.c2p(0, 0)[1])
                        for m in pulse_samples
                    ],
                    lag_ratio=0.05,
                    run_time=8,
                ),
                LaggedStart(
                    *[
                        m.animate.set_opacity(0).set_y(prod_ax.c2p(0, 0)[1])
                        for m in pulse_rtn_samples
                    ],
                    lag_ratio=0.05,
                    run_time=8,
                ),
                AnimationGroup(
                    LaggedStart(
                        *[FadeIn(m) for m in prod_samples], lag_ratio=0.05, run_time=8
                    ),
                    Create(
                        prod.set_z_index(-1),
                        run_time=8,
                        rate_func=rate_functions.linear,
                    ),
                ),
                lag_ratio=0.05,
            )
        )

        self.wait(0.5)

        # self.play(FadeOut(*prod_samples))

        # self.wait(0.5)

        # self.play()

        self.wait(2)
