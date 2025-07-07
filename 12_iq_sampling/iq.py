# iq.py

import sys

from manim import *
from MF_Tools import VT
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower, get_blocks
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True

FONT = "Maple Mono CN"

BLOCKS = get_blocks()


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


class Hook(MovingCameraScene):
    def construct(self):
        ax = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.4),
            y_length=fh(self, 0.4),
        )  # .to_corner(UL, MED_SMALL_BUFF)
        unit_circle = Circle(
            radius=(ax.c2p(1, 0) - ax.c2p(0, 0))[0],
            color=WHITE,
        ).move_to(ax.c2p(0, 0))
        self.add(
            ax,
            # unit_circle,
        )

        f = 2
        phi = VT(0)

        x_func = lambda t: (np.cos(f * t) + 0.5 * np.cos(f * 4 * t)) / 1.5
        y_func = lambda t: (np.sin(f * t) + 0.5 * np.sin(f * 4 * t)) / 1.5

        self.camera.frame.shift(LEFT * fw(self))
        boring_ax = Axes(
            x_range=[0, 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.8),
            y_length=fh(self, 0.5),
        ).move_to(self.camera.frame)
        boring_cos = boring_ax.plot(x_func, x_range=[0, 2 * PI, 1 / 200], color=ORANGE)

        num_samples = 11
        samples = boring_ax.get_vertical_lines_to_graph(
            boring_cos,
            x_range=[2 * PI / num_samples / 2, 2 * PI - 2 * PI / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        dots = Group(*[Dot(s.get_end(), color=BLUE) for s in samples])

        self.play(
            LaggedStart(
                Create(boring_ax),
                Create(boring_cos),
                LaggedStart(
                    *[
                        LaggedStart(Create(l), Create(d), lag_ratio=0.3)
                        for l, d in zip(samples, dots)
                    ],
                    lag_ratio=0.15,
                    run_time=1,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.wait(0.5)

        l = always_redraw(
            lambda: Line(ax.c2p(0, 0), ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE)
        )
        d = always_redraw(lambda: Dot(ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE))
        d_trace = TracedPath(
            d.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_color=ORANGE,
            stroke_width=DEFAULT_STROKE_WIDTH,
        )
        self.add(l, d, d_trace)

        cos_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        ).rotate(-PI / 2)
        cos_ax.shift(ax.c2p(0, -1) - cos_ax.c2p(0, 0))
        sin_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        )
        sin_ax.shift(ax.c2p(1, 0) - sin_ax.c2p(0, 0))
        self.add(cos_ax, sin_ax)

        sin = always_redraw(
            lambda: sin_ax.plot(y_func, x_range=[0, ~phi, 2 * PI / 200], color=BLUE)
        )
        cos = always_redraw(
            lambda: cos_ax.plot(x_func, x_range=[0, ~phi, 2 * PI / 200], color=GREEN)
        )

        line_to_sin = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                sin_ax.input_to_graph_point(~phi, sin),
                color=BLUE,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )
        line_to_cos = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                cos_ax.input_to_graph_point(~phi, cos),
                color=GREEN,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.add(sin, cos, line_to_cos, line_to_sin)

        arc = CurvedArrow(boring_ax.get_corner(UR), ax.get_corner(UL), angle=-PI / 4)
        self.play(
            LaggedStart(
                self.camera.frame.animate.move_to(ax), Create(arc), lag_ratio=0.2
            )
        )

        self.wait(0.5)

        iq_sampling_label = (
            Group(
                Text("In-Phase &", font=FONT),
                Text("Quadrature", font=FONT),
                Text("Sampling", font=FONT),
            )
            .arrange(DOWN, MED_LARGE_BUFF)
            .scale(1.5)
            .next_to(ax, DR, LARGE_BUFF)
        )
        iq_sampling_label[0][:-1].set_color(GREEN)
        iq_sampling_label[1].set_color(BLUE)

        all_group = Group(ax, cos_ax, sin_ax)
        self.play(
            LaggedStart(
                AnimationGroup(
                    phi @ (4 * PI),
                    self.camera.frame.animate.scale_to_fit_height(
                        all_group.height * 1.2
                    ).move_to(all_group),
                    run_time=14,
                ),
                LaggedStart(
                    *[Write(m, run_time=2.5) for m in iq_sampling_label],
                    lag_ratio=0.4,
                ),
                lag_ratio=0.4,
            ),
        )

        self.wait(2)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)


class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).to_corner(DL, LARGE_BUFF * 1.5)

        eqn = MathTex(r"s(t) = A(t) \cos{(2 \pi f_c t + \phi(t))}")

        self.play(LaggedStart(*[FadeIn(m) for m in eqn[0]], lag_ratio=0.03))
        self.add(eqn)

        self.wait(0.5)

        self.play(
            radar.get_animation(), eqn.animate.to_edge(UP, LARGE_BUFF), run_time=1
        )

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .to_edge(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP)
        )

        # eqn[0][-5:-1].set_color(YELLOW)

        to_cloud = Line(radar.radome.get_right(), cloud.get_left())
        ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=to_cloud.get_length(),
                y_length=fh(self, 0.2),
            )
            .set_opacity(0)
            .rotate(to_cloud.get_angle())
        )
        ax.shift(radar.radome.get_right() - ax.c2p(0, 0))
        rtn_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=to_cloud.get_length(),
                y_length=fh(self, 0.2),
            )
            .set_opacity(0)
            .rotate(to_cloud.get_angle() + PI)
        )
        rtn_ax.shift(cloud.get_left() - rtn_ax.c2p(0, 0))

        phase_vt = VT(0)
        sig_x1 = VT(0)
        A = VT(1)
        pw = 0.4
        sig = always_redraw(
            lambda: ax.plot(
                lambda t: ~A * np.sin(2 * PI * 3 * t),
                x_range=[max(0, ~sig_x1 - pw), min(1, ~sig_x1), 1 / 200],
                use_smoothing=False,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                color=TX_COLOR,
            )
        )
        rtn = always_redraw(
            lambda: rtn_ax.plot(
                lambda t: -~A * np.sin(2 * PI * 3 * t + ~phase_vt * PI),
                x_range=[max(0, (~sig_x1 - 1) - pw), min(1, (~sig_x1 - 1)), 1 / 200],
                use_smoothing=False,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                color=RX_COLOR,
            )
        )
        self.add(sig, rtn)

        self.play(
            LaggedStart(
                eqn[0][15:17].animate.set_color(YELLOW),
                cloud.shift(RIGHT * 10).animate.shift(LEFT * 10),
                AnimationGroup(sig_x1 @ (1.5 + pw / 2), A @ 0.5),
                lag_ratio=0.5,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(eqn[0][15:17].animate.set_color(WHITE))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rtn_ax.save_state()
        self.camera.frame.save_state()
        A.save_state()
        self.play(
            # eqn.animate.shift(DOWN*.7),
            rtn_ax.animate.rotate(-(to_cloud.get_angle())).move_to(self.camera.frame),
            radar.vgroup.animate.shift(LEFT * 4),
            cloud.animate.shift(RIGHT * 4),
            self.camera.frame.animate.scale(0.8),
            A @ 1,
            run_time=2,
        )

        self.wait(0.5)

        mag_line = Line(rtn.get_corner(DL), rtn.get_corner(UL)).shift(LEFT / 4)
        mag_line_u = Line(mag_line.get_top() + LEFT / 8, mag_line.get_top() + RIGHT / 8)
        mag_line_d = Line(
            mag_line.get_bottom() + LEFT / 8, mag_line.get_bottom() + RIGHT / 8
        )
        mag = MathTex(r"A(t)").scale(1.5).next_to(mag_line, LEFT, MED_LARGE_BUFF)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                Create(mag_line_d),
                Create(mag_line),
                Create(mag_line_u),
                TransformFromCopy(eqn[0][5:9], mag[0]),
                lag_ratio=0.2,
            ),
            run_time=2,
        )
        self.add(mag)

        self.wait(0.5)

        range_label = MathTex("R").scale(1.5)
        rcs_label = MathTex(r"\sigma").scale(1.5)
        amp_labels = (
            Group(range_label, rcs_label)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(mag, DOWN, LARGE_BUFF)
            .shift(LEFT / 2)
        )
        range_bez = CubicBezier(
            mag.get_bottom() + [0, -0.1, 0],
            mag.get_bottom() + [0, -1, 0],
            range_label.get_top() + [0, 1, 0],
            range_label.get_top() + [0, 0.1, 0],
        )
        rcs_bez = CubicBezier(
            mag.get_bottom() + [0, -0.1, 0],
            mag.get_bottom() + [0, -0.5, 0],
            rcs_label.get_top() + [0, 0.5, 0],
            rcs_label.get_top() + [0, 0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                Create(range_bez),
                GrowFromCenter(range_label),
                Create(rcs_bez),
                GrowFromCenter(rcs_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        phase_line = Line(
            [rtn_ax.c2p(~sig_x1 - 1 - pw / 2, 0)[0], rtn.get_top()[1], 0],
            [rtn_ax.c2p(~sig_x1 - 1 - pw / 2, 0)[0], rtn.get_top()[1], 0],
        ).shift(UP / 2)
        phase_line_l = Line(
            phase_line.get_left() + DOWN / 8, phase_line.get_left() + UP / 8
        )
        phase_line_r = Line(
            phase_line.get_right() + DOWN / 8, phase_line.get_right() + UP / 8
        )
        phase_label = always_redraw(
            lambda: MathTex(
                f"\\phi = {~phase_vt:.2f} \\pi",
                # color=YELLOW,
            )
            .scale(1.5)
            .next_to(phase_line, UP, MED_LARGE_BUFF)
        )

        self.next_section(skip_animations=skip_animations(True))

        cloud.shift(RIGHT * 3)

        self.play(
            eqn.animate.shift(UP * 0.8),
            Create(phase_line),
            Create(phase_line_l),
            Create(phase_line_r),
            TransformFromCopy(eqn[0][-5], phase_label[0][0], path_arc=PI / 3),
            FadeIn(phase_label[0][1:]),
            self.camera.frame.animate.scale(1 / 0.8),
            run_time=2,
        )
        self.add(phase_label)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        new_phase = 0.3
        phase_line_2 = Line(
            [rtn_ax.c2p(~sig_x1 - 1 - pw / 2, 0)[0], rtn.get_top()[1], 0],
            [
                rtn_ax.c2p(~sig_x1 - 1 - pw / 2 - new_phase / 3, 0)[0],
                rtn.get_top()[1],
                0,
            ],
        ).shift(UP / 2)
        phase_line_l_2 = Line(
            phase_line_2.get_left() + DOWN / 8, phase_line_2.get_left() + UP / 8
        )
        phase_line_r_2 = Line(
            phase_line_2.get_right() + DOWN / 8, phase_line_2.get_right() + UP / 8
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            phase_vt @ 1.2,
            Transform(phase_line, phase_line_2),
            Transform(phase_line_l, phase_line_l_2),
            Transform(phase_line_r, phase_line_r_2),
            run_time=3,
        )

        self.wait(0.5)

        doppler_vel = Text("Doppler\nvelocity", font=FONT).scale(0.6)
        direction = Text("direction", font=FONT).scale(0.6)
        tiny = Text("tiny\nmovements", font=FONT).scale(0.6)
        l1_group = (
            Group(doppler_vel, direction, tiny)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(eqn[0][-5:-1], UP, LARGE_BUFF)
        )

        doppler_vel_bez = CubicBezier(
            eqn[0][-5:-1].get_top() + [0, 0.2, 0],
            eqn[0][-5:-1].get_top() + [0, 1, 0],
            doppler_vel.get_bottom() + [0, -1, 0],
            doppler_vel.get_bottom() + [0, -0.1, 0],
        )
        direction_bez = CubicBezier(
            eqn[0][-5:-1].get_top() + [0, 0.2, 0],
            eqn[0][-5:-1].get_top() + [0, 0.5, 0],
            direction.get_bottom() + [0, -0.5, 0],
            direction.get_bottom() + [0, -0.1, 0],
        )
        tiny_bez = CubicBezier(
            eqn[0][-5:-1].get_top() + [0, 0.2, 0],
            eqn[0][-5:-1].get_top() + [0, 1, 0],
            tiny.get_bottom() + [0, -1, 0],
            tiny.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * 2),
                eqn[0][-5:-1].animate.set_color(YELLOW),
                Create(doppler_vel_bez),
                Write(doppler_vel),
                Create(direction_bez),
                Write(direction),
                Create(tiny_bez),
                Write(tiny),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        # self.wait(0.5)

        # self.play(
        #     LaggedStart(
        #         Create(v_target_bez),
        #         GrowFromCenter(v_target),
        #         Create(pm_v_bez),
        #         GrowFromCenter(pm_v),
        #         Create(r_ll_1_bez),
        #         GrowFromCenter(r_ll_1),
        #         lag_ratio=0.3,
        #     ),
        #     run_time=3,
        # )

        # self.wait(0.5)

        self.play(
            LaggedStart(
                eqn[0][-5:-1].animate.set_color(WHITE),
                FadeOut(
                    direction,
                    direction_bez,
                    range_label,
                    rcs_label,
                    phase_label,
                    doppler_vel_bez,
                    doppler_vel,
                    tiny,
                    tiny_bez,
                    mag_line,
                    mag_line_d,
                    mag_line_u,
                    phase_line,
                    phase_line_l,
                    phase_line_r,
                    mag,
                    range_bez,
                    rcs_bez,
                ),
                AnimationGroup(
                    rtn_ax.animate.restore(),
                    radar.vgroup.animate.shift(RIGHT * 4),
                    cloud.animate.shift(LEFT * 6),
                    self.camera.frame.animate.restore(),
                    A @ 0.5,
                ),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        rx_ax = Axes(
            x_range=[0, 2 * PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=to_cloud.get_length(),
            y_length=radar.vgroup.height,
        ).next_to(radar.vgroup, LEFT, LARGE_BUFF)
        rx_box = SurroundingRectangle(rx_ax, buff=MED_SMALL_BUFF, corner_radius=0.2)
        rx = Text("Receiver", font=FONT).next_to(rx_box, UP, SMALL_BUFF)
        rx_plot = rx_ax.plot(lambda t: np.cos(t), color=RX_COLOR)

        rx_bez_1 = CubicBezier(
            radar.radome.get_left(),
            radar.radome.get_left() + [-0.5, 0, 0],
            rx.get_top() + [4, 0.5, 0],
            rx.get_top() + [0, 0.5, 0],
        ).set_z_index(-2)
        rx_bez_2 = CubicBezier(
            rx.get_top() + [0, 0.5, 0],
            rx.get_top() + [-3, 0.5, 0],
            rx_box.get_left() + [-2, 1, 0],
            rx_box.get_left() + [0, 0.5, 0],
        ).set_z_index(-2)

        self.next_section(skip_animations=skip_animations(True))
        # self.add(rx_ax, rx_box, rx, rx_bez_1, rx_bez_2)

        data = Dot(color=RX_COLOR).move_to(rx_bez_1.get_start()).set_z_index(-1)

        g = Group(rx_ax, radar.vgroup, rx_bez_1, rx_bez_2, rx_box, rx)
        self.play(
            LaggedStart(
                AnimationGroup(
                    Succession(
                        Create(
                            rx_bez_1,
                            rate_func=rate_functions.ease_in_sine,
                            run_time=0.5,
                        ),
                        Create(
                            rx_bez_2,
                            rate_func=rate_functions.ease_out_sine,
                            run_time=0.5,
                        ),
                        FadeIn(rx_ax, rx, rx_box),
                    ),
                    sig_x1 @ (2 + pw),
                ),
                self.camera.frame.animate(run_time=2)
                .scale_to_fit_width(g.width * 1.2)
                .move_to(g),
                Succession(
                    MoveAlongPath(
                        data, rx_bez_1, rate_func=rate_functions.ease_in_sine
                    ),
                    MoveAlongPath(
                        data, rx_bez_2, rate_func=rate_functions.ease_out_sine
                    ),
                    Create(rx_plot),
                    FadeOut(data),
                ),
                lag_ratio=0.3,
            ),
            run_time=5,
        )

        self.wait(0.5)

        num_samples = 11
        samples = rx_ax.get_vertical_lines_to_graph(
            rx_plot,
            x_range=[2 * PI / num_samples / 2, 2 * PI - 2 * PI / num_samples / 2],
            num_lines=num_samples,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            line_func=Line,
        )
        dots = Group(*[Dot(s.get_end(), color=BLUE) for s in samples])
        v_labels = Group(
            *[
                MathTex(f"v_{{{idx + 1}}}").next_to(
                    d, UP if ax.point_to_coords(d.get_center())[1] > 0 else DOWN
                )
                for idx, d in enumerate(dots)
            ]
        )
        v_backgrounds = [
            SurroundingRectangle(
                v,
                buff=SMALL_BUFF,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=0.7,
                stroke_opacity=0,
                corner_radius=0.2,
            )
            for v in v_labels
        ]

        rx_box2 = SurroundingRectangle(rx_ax, buff=LARGE_BUFF, corner_radius=0.2)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(rx_box, rx_box2),
                    rx.animate.next_to(rx_box2, UP, SMALL_BUFF),
                    *[m.animate.set_opacity(0) for m in radar.vgroup],
                    FadeOut(rx_bez_1, rx_bez_2),
                    self.camera.frame.animate.scale(0.9),
                ),
                LaggedStart(
                    *[
                        LaggedStart(Create(l), Create(d), FadeIn(vb, v), lag_ratio=0.3)
                        for l, d, v, vb in zip(samples, dots, v_labels, v_backgrounds)
                    ],
                    lag_ratio=0.2,
                ),
            ),
        )

        self.wait(0.5)

        sin_ax = Axes(
            x_range=[0, 2 * PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=to_cloud.get_length(),
            y_length=radar.vgroup.height,
        ).move_to(rx_ax)
        sin_plot = sin_ax.plot(lambda t: np.sin(t), color=RX_COLOR)

        new_cam_group = Group(rx_ax.copy().next_to(sin_ax, LEFT, LARGE_BUFF), sin_ax)
        cos_label = (
            MathTex(r"\cos{(?)}")
            .scale(1.5)
            .next_to(rx_ax.copy().next_to(sin_ax, LEFT, LARGE_BUFF), UP, MED_LARGE_BUFF)
        )
        sin_label = MathTex(r"\sin{(?)}").scale(1.5).next_to(sin_ax, UP, MED_LARGE_BUFF)
        cos_label[0][-2].set_color(YELLOW)
        sin_label[0][-2].set_color(YELLOW)

        self.play(
            LaggedStart(
                FadeOut(rx_box, rx, *samples, *dots, *v_labels, *v_backgrounds),
                self.camera.frame.animate.scale_to_fit_width(
                    new_cam_group.width * 1.2
                ).move_to(new_cam_group),
                Group(rx_ax, rx_plot).animate.next_to(sin_ax, LEFT, LARGE_BUFF),
                Create(sin_ax),
                Create(sin_plot),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        unknown_cos_sample = Dot(
            rx_ax.input_to_graph_point(PI / 3, rx_plot), color=BLUE
        ).scale(1.5)
        unknown_cos_sample_arc = ArcBetweenPoints(
            unknown_cos_sample.get_center() + [0.2, 0, 0],
            cos_label.get_bottom() + [0, -0.1, 0],
            angle=TAU / 6,
            color=BLUE,
        )
        unknown_sin_sample = Dot(
            sin_ax.input_to_graph_point(PI / 3, sin_plot), color=BLUE
        ).scale(1.5)
        unknown_sin_sample_arc = ArcBetweenPoints(
            unknown_sin_sample.get_center() + [0, 0.2, 0],
            sin_label.get_left() + [-0.1, 0, 0],
            angle=-TAU / 4,
            color=BLUE,
        )

        self.play(
            LaggedStart(
                Create(unknown_cos_sample),
                Create(unknown_cos_sample_arc),
                FadeIn(cos_label),
                lag_ratio=0.3,
            ),
            LaggedStart(
                Create(unknown_sin_sample),
                Create(unknown_sin_sample_arc),
                FadeIn(sin_label),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                unknown_cos_sample,
                unknown_cos_sample_arc,
                unknown_sin_sample,
                unknown_sin_sample_arc,
            ),
            LaggedStart(
                cos_label.animate.shift(UP * 6),
                sin_label.animate.shift(UP * 6),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        y_val = VT(1)
        cos_y_line = always_redraw(
            lambda: DashedLine(
                rx_ax.c2p(0, ~y_val),
                rx_ax.c2p(2 * PI, ~y_val),
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        sin_y_line = always_redraw(
            lambda: DashedLine(
                sin_ax.c2p(0, ~y_val),
                sin_ax.c2p(2 * PI, ~y_val),
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        sin_x_dot_l = always_redraw(
            lambda: Dot(
                sin_ax.input_to_graph_point(
                    (np.arcsin(~y_val)),
                    sin_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
                color=BLUE,
            )
            .set_opacity(0 if ~y_val < 0 else 1)
            .set_z_index(1)
        )
        sin_x_dot_r = always_redraw(
            lambda: Dot(
                sin_ax.input_to_graph_point(
                    (PI - np.arcsin(~y_val)),
                    sin_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
                color=BLUE,
            ).set_z_index(1)
        )
        sin_x_dot_r_2 = always_redraw(
            lambda: Dot(
                sin_ax.input_to_graph_point(
                    (2 * PI + np.arcsin(~y_val)),
                    sin_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
                color=BLUE,
            )
            .set_opacity(0 if ~y_val > 0 else 1)
            .set_z_index(1)
        )
        cos_x_dot_l = always_redraw(
            lambda: Dot(
                rx_ax.input_to_graph_point(
                    (np.arccos(~y_val)),
                    rx_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
                color=BLUE,
            ).set_z_index(1)
        )
        cos_x_dot_r = always_redraw(
            lambda: Dot(
                rx_ax.input_to_graph_point(
                    (2 * PI - np.arccos(~y_val)),
                    rx_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
                color=BLUE,
            ).set_z_index(1)
        )

        self.play(
            Create(cos_y_line),
            Create(sin_y_line),
            Create(sin_x_dot_l),
            Create(sin_x_dot_r),
            Create(cos_x_dot_l),
            Create(cos_x_dot_r),
        )

        self.wait(0.5)

        self.play(y_val @ 0.5, run_time=2)

        self.wait(0.5)

        cos_opt_1_x = Line(
            rx_ax.c2p(PI / 3, -0.2),
            rx_ax.c2p(PI / 3, 0.2),
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        cos_opt_2_x = Line(
            rx_ax.c2p(5 * PI / 3, -0.2),
            rx_ax.c2p(5 * PI / 3, 0.2),
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        cos_opt_1 = MathTex(r"\frac{\pi}{3}").next_to(cos_opt_1_x, DOWN)
        cos_opt_2 = MathTex(r"\frac{5 \pi}{3}").next_to(cos_opt_2_x, DOWN)

        sin_opt_1_x = Line(
            sin_ax.c2p(PI / 6, -0.2),
            sin_ax.c2p(PI / 6, 0.2),
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        sin_opt_2_x = Line(
            sin_ax.c2p(5 * PI / 6, -0.2),
            sin_ax.c2p(5 * PI / 6, 0.2),
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        sin_opt_1 = MathTex(r"\frac{\pi}{6}").next_to(sin_opt_1_x, DOWN)
        sin_opt_2 = MathTex(r"\frac{5 \pi}{6}").next_to(sin_opt_2_x, DOWN)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(cos_opt_1_x),
                    GrowFromCenter(cos_opt_1),
                ),
                AnimationGroup(
                    Create(cos_opt_2_x),
                    GrowFromCenter(cos_opt_2),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(sin_opt_1_x),
                    GrowFromCenter(sin_opt_1),
                ),
                AnimationGroup(
                    Create(sin_opt_2_x),
                    GrowFromCenter(sin_opt_2),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(
                sin_opt_2,
                sin_opt_2_x,
                cos_opt_2_x,
                cos_opt_2,
                sin_opt_1,
                sin_opt_1_x,
                cos_opt_1_x,
                cos_opt_1,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.add(sin_x_dot_r_2)

        self.play(
            y_val.animate(rate_func=rate_functions.there_and_back).set_value(-1),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                cos_label.animate.shift(DOWN * 6),
                sin_label.animate.shift(DOWN * 6),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        doppler_vel = Text("Doppler\nvelocity", font=FONT).scale(0.6)
        direction = Text("direction", font=FONT).scale(0.6)
        tiny = Text("tiny\nmovements", font=FONT).scale(0.6)
        l1_group = (
            Group(doppler_vel, direction, tiny)
            .arrange(RIGHT, MED_LARGE_BUFF)
            .next_to(Group(cos_label, sin_label), UP, LARGE_BUFF)
        )
        cos_theta_bez = CubicBezier(
            cos_label[0][4].get_top() + [0, 0.1, 0],
            cos_label[0][4].get_top() + [0, 0.5, 0],
            l1_group.get_left() + [-1, -0.5, 0],
            l1_group.get_left() + [-0.1, 0, 0],
        )
        sin_theta_bez = CubicBezier(
            sin_label[0][4].get_top() + [0, 0.1, 0],
            sin_label[0][4].get_top() + [0, 0.5, 0],
            l1_group.get_right() + [1, 0, 0],
            l1_group.get_right() + [0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(cos_theta_bez), Create(sin_theta_bez)),
                GrowFromCenter(doppler_vel),
                GrowFromCenter(direction),
                GrowFromCenter(tiny),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ax_group = Group(Group(rx_ax, rx_plot), Group(sin_ax, sin_plot))
        self.play(
            Group(
                cos_label, sin_label, l1_group, cos_theta_bez, sin_theta_bez
            ).animate.shift(UP * 6),
            ax_group.animate.arrange(DOWN, MED_LARGE_BUFF).move_to(ax_group),
        )

        self.wait(0.5)

        vline_1 = DashedLine(
            rx_ax.c2p(PI / 6, 1),
            sin_ax.c2p(PI / 6, -1),
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )
        vline_2 = DashedLine(
            rx_ax.c2p(PI / 3, 1),
            sin_ax.c2p(PI / 3, -1),
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )
        vline_3 = DashedLine(
            rx_ax.c2p(5 * PI / 6, 1),
            sin_ax.c2p(5 * PI / 6, -1),
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )
        vline_4 = DashedLine(
            rx_ax.c2p(5 * PI / 3, 1),
            sin_ax.c2p(5 * PI / 3, -1),
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 2,
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(LaggedStart(Create(vline_2), Create(vline_4), lag_ratio=0.3))

        self.wait(0.5)

        self.play(LaggedStart(Create(vline_1), Create(vline_3), lag_ratio=0.3))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        iq_ax = (
            Axes(
                x_range=[-1, 1, 0.5],
                y_range=[-1, 1, 0.5],
                tips=False,
                x_length=rx_ax.height,
                y_length=rx_ax.height,
            )
            .next_to(ax_group, LEFT)
            .shift(LEFT * 10)
        )
        self.add(ax)

        self.play(
            LaggedStart(
                LaggedStart(
                    *[
                        FadeOut(m)
                        for m in [
                            vline_1,
                            vline_2,
                            vline_3,
                            vline_4,
                            cos_x_dot_l,
                            cos_x_dot_r,
                            sin_x_dot_l,
                            sin_x_dot_r,
                            cos_y_line,
                            sin_y_line,
                        ]
                    ],
                    lag_ratio=0.1,
                ),
                Group(iq_ax, ax_group)
                .animate.arrange(RIGHT, LARGE_BUFF)
                .move_to(self.camera.frame)
                .shift(DOWN + LEFT),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        theta_pi = VT(0)
        theta_label_opacity = VT(0)
        theta_label_vt = always_redraw(
            lambda: MathTex(f"\\theta = 0")
            .next_to(rx_ax, UP, LARGE_BUFF)
            .set_opacity(~theta_label_opacity)
        )

        vline_cos = always_redraw(
            lambda: Line(
                rx_ax.input_to_graph_point(~theta_pi, rx_plot),
                rx_ax.c2p(~theta_pi, 0),
                color=YELLOW,
            )
        )
        vline_sin = always_redraw(
            lambda: Line(
                sin_ax.input_to_graph_point(~theta_pi, sin_plot),
                sin_ax.c2p(~theta_pi, 0),
                color=YELLOW,
            )
        )
        cos_dot = always_redraw(
            lambda: Dot(radius=DEFAULT_DOT_RADIUS * 1.8, color=BLUE).move_to(
                rx_ax.input_to_graph_point(~theta_pi, rx_plot)
            )
        )
        sin_dot = always_redraw(
            lambda: Dot(radius=DEFAULT_DOT_RADIUS * 1.8, color=BLUE).move_to(
                sin_ax.input_to_graph_point(~theta_pi, sin_plot)
            )
        )
        cos_label = always_redraw(
            lambda: MathTex(r"\cos{(\theta)}")
            .next_to(rx_ax, RIGHT)
            .set_opacity(~theta_label_opacity)
        )
        sin_label = always_redraw(
            lambda: MathTex(r"\sin{(\theta)}")
            .next_to(sin_ax, RIGHT)
            .set_opacity(~theta_label_opacity)
        )
        self.add(theta_label_vt, cos_label, sin_label)

        self.play(
            theta_label_opacity @ 1,
            Create(vline_cos),
            Create(vline_sin),
            Create(cos_dot),
            Create(sin_dot),
        )

        self.wait(0.5)

        theta_label = MathTex(f"\\theta = 0").move_to(theta_label_vt)
        self.remove(theta_label_vt)
        self.add(theta_label)
        theta_pi_over_3 = MathTex(r"\theta = \frac{\pi}{3}")
        theta_pi_over_3.shift(
            theta_label[0][0].get_center() - theta_pi_over_3[0][0].get_center()
        )

        self.play(
            FadeOut(theta_label[0][2], shift=UP),
            FadeIn(theta_pi_over_3[0][2:], shift=UP),
            theta_pi @ (PI / 3),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        i_line = Line(iq_ax.c2p(0, 0), iq_ax.c2p(np.cos(PI / 3), 0), color=YELLOW)
        q_line = Line(iq_ax.c2p(0, 0), iq_ax.c2p(0, np.sin(PI / 3)), color=YELLOW)

        self.play(TransformFromCopy(vline_cos, i_line))

        self.wait(0.5)

        self.play(TransformFromCopy(vline_sin, q_line))

        self.wait(0.5)

        i_line_to_dot = DashedLine(
            iq_ax.c2p(np.cos(~theta_pi), 0),
            iq_ax.c2p(np.cos(~theta_pi), np.sin(~theta_pi)),
            dash_length=DEFAULT_DASH_LENGTH * 2,
            color=YELLOW,
        )
        q_line_to_dot = DashedLine(
            iq_ax.c2p(0, np.sin(~theta_pi)),
            iq_ax.c2p(np.cos(~theta_pi), np.sin(~theta_pi)),
            dash_length=DEFAULT_DASH_LENGTH * 2,
            color=YELLOW,
        )

        iq_dot = Dot(color=YELLOW).move_to(
            iq_ax.c2p(np.cos(~theta_pi), np.sin(~theta_pi))
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(i_line_to_dot), Create(q_line_to_dot)),
                Create(iq_dot),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        zero = MathTex("0").next_to(rx_ax.c2p(0, 1), UP, MED_LARGE_BUFF)
        two_pi = MathTex(r"2 \pi").next_to(rx_ax.c2p(2 * PI, 1), UP, MED_LARGE_BUFF)
        zero_to_2pi = Arrow(zero.get_right(), two_pi.get_left())

        self.play(
            Group(
                iq_ax,
                i_line,
                q_line,
                i_line_to_dot,
                q_line_to_dot,
                iq_dot,
            ).animate.shift(DOWN * 8),
            self.camera.frame.animate.set_x(rx_ax.get_x()),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                cos_label.animate(rate_func=rate_functions.there_and_back)
                .scale(1.3)
                .set_color(YELLOW),
                sin_label.animate(rate_func=rate_functions.there_and_back)
                .scale(1.3)
                .set_color(YELLOW),
                Group(theta_pi_over_3[0][2:], theta_label[0][:2]).animate.shift(UP / 2),
                GrowFromCenter(zero),
                GrowArrow(zero_to_2pi),
                GrowFromCenter(two_pi),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Group(
                zero,
                zero_to_2pi,
                two_pi,
                ax_group,
                theta_label[0][:2],
                theta_pi_over_3[0][2:],
            ).animate.shift(RIGHT * 14),
            # Group(
            #     iq_ax,
            #     i_line,
            #     q_line,
            #     i_line_to_dot,
            #     q_line_to_dot,
            #     iq_dot,
            # ).animate.shift(UP * 8),
            # Group(
            #     theta_label[0][:2],
            #     theta_pi_over_3[0][2:],
            # ).animate.next_to(iq_ax, UP, LARGE_BUFF),
            # self.camera.frame.animate.scale_to_fit_height(iq_ax.height * 2)
            # .move_to(iq_ax)
            # .shift(UP),
        )

        # self.wait(0.5)
        # self.next_section(skip_animations=skip_animations(True))

        # sample_rects1 = rx_ax.get_riemann_rectangles(
        #     rx_plot,
        #     input_sample_type="center",
        #     x_range=[0, 1],
        #     dx=1 / (num_samples),
        #     color=BLUE,
        #     show_signed_area=False,
        #     stroke_color=BLACK,
        #     fill_opacity=0.7,
        # )

        # self.play(
        #     rx_plot.animate.set_stroke(opacity=0.2),
        #     LaggedStart(
        #         *[
        #             AnimationGroup(FadeIn(sr), FadeOut(s, dot))
        #             for s, sr, dot in zip(samples, sample_rects1, dots)
        #         ],
        #         lag_ratio=0.1,
        #     ),
        #     run_time=2,
        # )

        # self.wait(0.5)

        # phase_qmark = (
        #     Tex("$\\phi = $ ?").scale(1.5).next_to(rx_box, DOWN, MED_LARGE_BUFF)
        # )

        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.scale_to_fit_height(
        #             Group(rx, rx_box, phase_qmark).height * 1.2
        #         ).move_to(Group(rx, rx_box, phase_qmark)),
        #         LaggedStart(*[FadeIn(m) for m in phase_qmark[0]], lag_ratio=0.15),
        #         lag_ratio=0.25,
        #     )
        # )

        # self.wait(0.5)

        # self.play(
        #     FadeOut(rx_box, rx, sample_rects1, v_labels, *v_backgrounds),
        #     rx_plot.animate.set_stroke(opacity=1),
        #     phase_qmark.animate.shift(DOWN * 6),
        #     self.camera.frame.animate.scale(0.9).shift(UP),
        # )

        # self.wait(0.5)

        # cos = MathTex(r"\cos{(\theta)}").next_to(rx_ax, UP, LARGE_BUFF)
        # cos_bez_l = CubicBezier(
        #     cos.get_bottom() + [0, -0.1, 0],
        #     cos.get_bottom() + [0, -1, 0],
        #     rx_ax.get_corner(UL) + [0, 1, 0],
        #     rx_ax.get_corner(UL) + [0, 0.1, 0],
        # )
        # cos_bez_r = CubicBezier(
        #     cos.get_bottom() + [0, -0.1, 0],
        #     cos.get_bottom() + [0, -1, 0],
        #     rx_ax.get_corner(UR) + [0, 1, 0],
        #     rx_ax.get_corner(UR) + [0, 0.1, 0],
        # )

        # self.remove(rx_plot)
        # rx_f = VT(3)
        # rx_x0 = VT(0)
        # rx_x1 = VT(2 * PI)
        # rx_plot = always_redraw(
        #     lambda: rx_ax.plot(
        #         lambda t: np.cos(~rx_f * t),
        #         x_range=[~rx_x0, ~rx_x1, 1 / 200],
        #         color=RX_COLOR,
        #     )
        # )
        # self.add(rx_plot)

        # self.play(
        #     LaggedStart(*[FadeIn(m) for m in cos[0]], lag_ratio=0.1),
        #     rx_f @ 1,
        #     Create(cos_bez_l),
        #     Create(cos_bez_r),
        # )

        # self.wait(0.5)

        # sample = Line(
        #     rx_ax.c2p(30 * DEGREES, 0),
        #     rx_ax.input_to_graph_point(30 * DEGREES, rx_plot),
        #     color=BLUE,
        # )
        # dot = Dot(sample.get_end(), color=BLUE)

        # self.play(
        #     LaggedStart(
        #         AnimationGroup(
        #             Uncreate(cos_bez_l),
        #             Uncreate(cos_bez_r),
        #         ),
        #         Create(sample),
        #         Create(dot),
        #         lag_ratio=0.3,
        #     ),
        # )

        # self.wait(0.5)

        # cos_w_inp = MathTex(r"\cos{\left(30^{\circ}\right)} \approx 0.87").next_to(
        #     rx_ax, UP, LARGE_BUFF
        # )
        # cos_w_inp.shift(cos[0][4].get_center() - cos_w_inp[0][4:7].get_center())

        # val_bez = CubicBezier(
        #     dot.get_center() + [0, 0.1, 0],
        #     dot.get_center() + [0, 1, 0],
        #     cos_w_inp[0][4:7].get_bottom() + [0, -1, 0],
        #     cos_w_inp[0][4:7].get_bottom() + [0, -0.1, 0],
        # )

        # self.next_section(skip_animations=skip_animations(False))

        # self.play(
        #     LaggedStart(
        #         AnimationGroup(
        #             ReplacementTransform(cos[0][:4], cos_w_inp[0][:4]),
        #             ReplacementTransform(cos[0][5], cos_w_inp[0][7]),
        #         ),
        #         Create(val_bez),
        #         ReplacementTransform(cos[0][4], cos_w_inp[0][4:7]),
        #         LaggedStart(*[FadeIn(m) for m in cos_w_inp[0][8:]], lag_ratio=0.1),
        #         lag_ratio=0.3,
        #     ),
        #     run_time=3,
        # )

        # self.wait(0.5)

        # sample_390 = Line(
        #     rx_ax.c2p(390 * DEGREES, 0),
        #     rx_ax.input_to_graph_point(390 * DEGREES, rx_plot),
        #     color=BLUE,
        # )
        # dot_390 = Dot(sample_390.get_end(), color=BLUE)

        # cos_w_inp_390 = MathTex(r"\cos{\left(390^{\circ}\right)} \approx 0.87").next_to(
        #     cos_w_inp, RIGHT, MED_LARGE_BUFF
        # )

        # val_bez_390 = CubicBezier(
        #     dot_390.get_center() + [0, 0.1, 0],
        #     dot_390.get_center() + [0, 1, 0],
        #     cos_w_inp_390[0][4:8].get_bottom() + [0, -1, 0],
        #     cos_w_inp_390[0][4:8].get_bottom() + [0, -0.1, 0],
        # )

        # rx_ax_2 = Axes(
        #     x_range=[0, 3 * PI, PI / 4],
        #     y_range=[-1, 1, 0.5],
        #     tips=False,
        #     x_length=to_cloud.get_length() * 1.5,
        #     y_length=radar.vgroup.height,
        # )
        # rx_ax_2.shift(rx_ax.c2p(0, 0) - rx_ax_2.c2p(0, 0))

        # self.camera.frame.save_state()
        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.scale(1.1).shift(RIGHT),
        #         TransformFromCopy(rx_ax, rx_ax_2),
        #         rx_x1 @ (3 * PI),
        #         Create(sample_390),
        #         Create(dot_390),
        #         Create(val_bez_390),
        #         LaggedStart(*[FadeIn(m) for m in cos_w_inp_390[0]], lag_ratio=0.1),
        #         lag_ratio=0.3,
        #     )
        # )

        # self.wait(0.5)

        # rx_ax_3 = Axes(
        #     x_range=[-2 * PI, 3 * PI, PI / 4],
        #     y_range=[-1, 1, 0.5],
        #     tips=False,
        #     x_length=to_cloud.get_length() * 2.5,
        #     y_length=radar.vgroup.height,
        # )
        # rx_ax_3.shift(rx_ax.c2p(0, 0) - rx_ax_3.c2p(0, 0))

        # sample_n330 = Line(
        #     rx_ax.c2p(-330 * DEGREES, 0),
        #     rx_ax.input_to_graph_point(-330 * DEGREES, rx_plot),
        #     color=BLUE,
        # )
        # dot_n330 = Dot(sample_n330.get_end(), color=BLUE)

        # cos_w_inp_n330 = MathTex(
        #     r"\cos{\left(-330^{\circ}\right)} \approx 0.87"
        # ).next_to(cos_w_inp, LEFT, 2 * LARGE_BUFF)

        # val_bez_n330 = CubicBezier(
        #     dot_n330.get_center() + [0, 0.1, 0],
        #     dot_n330.get_center() + [0, 1, 0],
        #     cos_w_inp_n330[0][4:8].get_bottom() + [0, -1, 0],
        #     cos_w_inp_n330[0][4:8].get_bottom() + [0, -0.1, 0],
        # )

        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.scale(1.2).move_to(rx_ax_3).shift(UP),
        #         TransformFromCopy(rx_ax, rx_ax_3),
        #         rx_x0 @ (-2 * PI),
        #         Create(sample_n330),
        #         Create(dot_n330),
        #         Create(val_bez_n330),
        #         LaggedStart(*[FadeIn(m) for m in cos_w_inp_n330[0]], lag_ratio=0.1),
        #         lag_ratio=0.3,
        #     )
        # )

        # self.wait(0.5)

        # self.play(
        #     self.camera.frame.animate.restore(),
        #     FadeOut(
        #         rx_ax_2,
        #         rx_ax_3,
        #         val_bez_390,
        #         val_bez_n330,
        #         cos_w_inp_390,
        #         cos_w_inp_n330,
        #         dot_390,
        #         dot_n330,
        #         sample_n330,
        #         sample_390,
        #     ),
        #     rx_x0 @ 0,
        #     rx_x1 @ (2 * PI),
        #     Uncreate(sample),
        #     Uncreate(dot),
        #     Uncreate(val_bez),
        #     cos_w_inp.animate.set_x(self.camera.frame.copy().restore().get_x()),
        # )

        # self.wait(0.5)

        # qmark = Text("?", font=FONT, color=YELLOW).next_to(
        #     cos_w_inp[0][4:7], UP, SMALL_BUFF
        # )
        # self.play(Write(qmark))
        # # self.play(self.camera.frame.animate.shift(UP * 10))

        # self.wait(0.5)

        # cos = (
        #     MathTex(r"\cos{(\theta)}")
        #     .scale(1.5)
        #     .move_to(self.camera.frame)
        #     .shift(UP * fh(self, 1))
        # )

        # self.play(
        #     Transform(cos_w_inp, cos), self.camera.frame.animate.shift(UP * fh(self))
        # )

        self.wait(2)


class PolarPlot(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
        )

        self.play(Create(ax))

        self.wait(0.5)

        theta = VT(30)

        line_to_cos = DashedLine(
            ax.c2p(0, 0),
            ax.c2p(np.cos(~theta * PI / 180), 0),
            dash_length=DEFAULT_DASH_LENGTH * 2,
            color=ORANGE,
        )
        dot = Dot(
            ax.c2p(np.cos(~theta * PI / 180), 0),
            color=ORANGE,
            radius=DEFAULT_DOT_RADIUS * 1.8,
        )
        cos_val_label = Tex(f"{np.cos(~theta * PI / 180):.2f}").next_to(dot, DOWN)
        cos_val_label = MathTex(r"\cos{\left(30^\circ\right)}").next_to(dot, DOWN)

        line_to_sin = DashedLine(
            ax.c2p(np.cos(~theta * PI / 180), 0),
            ax.c2p(np.cos(~theta * PI / 180), np.sin(~theta * PI / 180)),
            dash_length=DEFAULT_DASH_LENGTH * 2,
            color=ORANGE,
        )
        sin_val_label = MathTex(r"\sin{\left(30^\circ\right)}").next_to(
            ax.c2p(0, np.sin(~theta * PI / 180)), LEFT
        )

        self.play(
            LaggedStart(
                Create(line_to_cos), Create(dot), FadeIn(cos_val_label), lag_ratio=0.3
            )
        )

        self.play(
            LaggedStart(
                Create(line_to_sin),
                AnimationGroup(
                    dot.animate.shift(
                        (line_to_sin.get_end() - line_to_cos.get_end()) * UP
                    ),
                    line_to_cos.animate.shift(
                        (line_to_sin.get_end() - line_to_cos.get_end()) * UP
                    ),
                ),
                FadeIn(sin_val_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        pair_label = (
            MathTex(r"\left[ \cos{(30^\circ)}, \sin{(30^\circ)} \right]")
            # .scale(0.7)
            .next_to(dot, UP, SMALL_BUFF)
            .shift(RIGHT)
        )

        line_to_cos_vt = always_redraw(
            lambda: DashedLine(
                ax.c2p(0, np.sin(~theta * PI / 180)),
                ax.c2p(np.cos(~theta * PI / 180), np.sin(~theta * PI / 180)),
                dash_length=DEFAULT_DASH_LENGTH * 2,
                color=ORANGE,
            )
        )
        dot_vt = always_redraw(
            lambda: Dot(
                ax.c2p(np.cos(~theta * PI / 180), np.sin(~theta * PI / 180)),
                color=ORANGE,
                radius=DEFAULT_DOT_RADIUS * 1.8,
            )
        )
        line_to_sin_vt = always_redraw(
            lambda: DashedLine(
                ax.c2p(np.cos(~theta * PI / 180), 0),
                ax.c2p(np.cos(~theta * PI / 180), np.sin(~theta * PI / 180)),
                dash_length=DEFAULT_DASH_LENGTH * 2,
                color=ORANGE,
            )
        )

        def get_cos():
            val = np.cos(~theta * PI / 180)
            pm = "+" if val > 0 else "-"
            return pm, abs(val)

        def get_sin():
            val = np.sin(~theta * PI / 180)
            pm = "+" if val > 0 else "-"
            return pm, abs(val)

        pair_label_shift = VT(1)
        pair_label_vt = always_redraw(
            lambda: (
                MathTex(
                    f"\\left[ {get_cos()[0]}{get_cos()[1]:.2f},{get_sin()[0]}{get_sin()[1]:.2f} \\right]"
                )
                # .scale(0.7)
                .next_to(dot_vt, UP, SMALL_BUFF)
                .shift(RIGHT * ~pair_label_shift)
            )
        )

        self.play(
            LaggedStart(
                FadeIn(pair_label[0][0]),
                TransformFromCopy(cos_val_label[0], pair_label[0][1:9]),
                FadeIn(pair_label[0][9]),
                TransformFromCopy(sin_val_label[0], pair_label[0][10:18]),
                FadeIn(pair_label[0][18]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            ReplacementTransform(pair_label, pair_label_vt),
            FadeOut(cos_val_label, sin_val_label),
        )

        self.wait(0.5)

        self.add(line_to_sin_vt, line_to_cos_vt, dot_vt, pair_label_vt)
        self.remove(line_to_sin, line_to_cos, dot)

        self.next_section(skip_animations=skip_animations(True))

        self.play(theta @ (390), run_time=10)

        self.wait(0.5)

        cos_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
        ).next_to(ax, RIGHT, LARGE_BUFF)
        sin_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.7),
        ).next_to(cos_ax, RIGHT, LARGE_BUFF)

        cos_plot = cos_ax.plot(
            lambda t: np.cos(2 * PI * t), x_range=[0, 1, 1 / 200], color=ORANGE
        )
        sin_plot = sin_ax.plot(
            lambda t: np.sin(2 * PI * t), x_range=[0, 1, 1 / 200], color=ORANGE
        )

        axes = Group(ax, cos_ax, sin_ax)

        self.play(
            LaggedStart(
                pair_label_shift @ -0.4,
                self.camera.frame.animate.scale_to_fit_width(axes.width * 1.1).move_to(
                    axes
                ),
                Create(cos_ax),
                Create(sin_ax),
                Create(cos_plot),
                Create(sin_plot),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        pair_plot_label = (
            MathTex(r"\left[ \cos{(\theta)}, \sin{(\theta)} \right]")
            .scale(1.5)
            .next_to(ax, UP)
        )
        cos_plot_label = MathTex(r"\cos{(\theta)}").scale(1.5).next_to(cos_ax, UP)
        sin_plot_label = MathTex(r"\sin{(\theta)}").scale(1.5).next_to(sin_ax, UP)

        self.play(
            LaggedStart(
                FadeIn(pair_plot_label),
                FadeIn(cos_plot_label),
                FadeIn(sin_plot_label),
                lag_ratio=0.3,
            )
        )
        self.next_section(skip_animations=skip_animations(True))

        self.wait(0.5)

        y_val = VT(1)
        cos_y_line = always_redraw(
            lambda: DashedLine(
                cos_ax.c2p(0, ~y_val),
                cos_ax.c2p(1, ~y_val),
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        sin_y_line = always_redraw(
            lambda: DashedLine(
                sin_ax.c2p(0, ~y_val),
                sin_ax.c2p(1, ~y_val),
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        sin_x_dot_l = always_redraw(
            lambda: Dot(
                sin_ax.input_to_graph_point(
                    (np.arcsin(~y_val)) / (2 * PI),
                    sin_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
            )
        )
        sin_x_dot_r = always_redraw(
            lambda: Dot(
                sin_ax.input_to_graph_point(
                    (PI - np.arcsin(~y_val)) / (2 * PI),
                    sin_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
            )
        )
        cos_x_dot_l = always_redraw(
            lambda: Dot(
                cos_ax.input_to_graph_point(
                    (np.arccos(~y_val)) / (2 * PI),
                    cos_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
            )
        )
        cos_x_dot_r = always_redraw(
            lambda: Dot(
                cos_ax.input_to_graph_point(
                    (2 * PI - np.arccos(~y_val)) / (2 * PI),
                    cos_plot,
                ),
                radius=DEFAULT_DOT_RADIUS * 1.8,
            )
        )

        self.play(
            Create(cos_y_line),
            Create(sin_y_line),
            Create(sin_x_dot_l),
            Create(sin_x_dot_r),
            Create(cos_x_dot_l),
            Create(cos_x_dot_r),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        # TODO: fix the sine val going down negative or change this to 0
        self.play(y_val @ -1, rate_func=rate_functions.there_and_back, run_time=3)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(y_val @ 0.5, run_time=3)

        self.wait(0.5)

        self.play(theta @ (45))

        self.wait(2)


class ComplexSine3D(ThreeDScene):
    def construct(self):
        # Set up the 3D axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 4 * PI, PI],
            x_length=6,
            y_length=6,
            z_length=6,
        )

        # Parametric function: f(t) = sin(t) * e^{i t}
        def param_func(t):
            r = np.sin(t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = t
            return np.array([x, y, z])

        curve = ParametricFunction(param_func, t_range=[0, 4 * PI], color=YELLOW)

        # Labels
        labels = axes.get_axis_labels(x_label="Re", y_label="Im", z_label="t")

        # Add all to scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, labels, curve)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(6)


class ComplexOscillation(MovingCameraScene):
    def construct(self):
        ax = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.4),
            y_length=fh(self, 0.4),
        )  # .to_corner(UL, MED_SMALL_BUFF)
        unit_circle = Circle(
            radius=(ax.c2p(1, 0) - ax.c2p(0, 0))[0],
            color=WHITE,
        ).move_to(ax.c2p(0, 0))
        self.add(
            ax,
            # unit_circle,
        )

        f = 2
        phi = VT(0)

        x_func = lambda t: (np.cos(f * t) + 0.5 * np.cos(f * 4 * t)) / 1.5
        y_func = lambda t: (np.sin(f * t) + 0.5 * np.sin(f * 4 * t)) / 1.5

        l = always_redraw(
            lambda: Line(ax.c2p(0, 0), ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE)
        )
        d = always_redraw(lambda: Dot(ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE))
        d_trace = TracedPath(
            d.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_color=ORANGE,
            stroke_width=DEFAULT_STROKE_WIDTH,
        )
        self.add(l, d, d_trace)

        cos_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        ).rotate(-PI / 2)
        cos_ax.shift(ax.c2p(0, -1) - cos_ax.c2p(0, 0))
        sin_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        )
        sin_ax.shift(ax.c2p(1, 0) - sin_ax.c2p(0, 0))
        self.add(cos_ax, sin_ax)

        sin = always_redraw(
            lambda: sin_ax.plot(y_func, x_range=[0, ~phi, 2 * PI / 200], color=BLUE)
        )
        cos = always_redraw(
            lambda: cos_ax.plot(x_func, x_range=[0, ~phi, 2 * PI / 200], color=GREEN)
        )

        line_to_sin = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                sin_ax.input_to_graph_point(~phi, sin),
                color=BLUE,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )
        line_to_cos = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                cos_ax.input_to_graph_point(~phi, cos),
                color=GREEN,
                dash_length=DEFAULT_DASH_LENGTH * 3,
            )
        )

        self.add(sin, cos, line_to_cos, line_to_sin)

        all_group = Group(ax, cos_ax, sin_ax)
        self.play(
            phi @ (4 * PI),
            self.camera.frame.animate.scale_to_fit_height(
                all_group.height * 1.2
            ).move_to(all_group),
            run_time=16,
        )

        self.wait(2)


class ComplexOscillationPhase(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        ax = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.4),
            y_length=fh(self, 0.4),
        )  # .to_corner(UL, MED_SMALL_BUFF)
        unit_circle = Circle(
            radius=(ax.c2p(1, 0) - ax.c2p(0, 0))[0],
            color=WHITE,
        ).move_to(ax.c2p(0, 0))
        self.add(
            ax,
            # unit_circle,
        )

        f = 2
        phi = VT(0)

        x_func = lambda t: (np.cos(f * t) + 0.5 * np.cos(f * 4 * t)) / 1.5
        y_func = lambda t: (np.sin(f * t) + 0.5 * np.sin(f * 4 * t)) / 1.5
        t = np.linspace(0, f * 2 * PI, 1000)
        z = ((np.cos(f * t) + 0.5 * np.cos(f * 4 * t)) / 1.5) + 1j * (
            (np.sin(f * t) + 0.5 * np.sin(f * 4 * t)) / 1.5
        )
        phi_func = interp1d(
            t, np.unwrap(np.arctan2(np.imag(z), np.real(z))), fill_value="extrapolate"
        )

        l = always_redraw(
            lambda: Line(ax.c2p(0, 0), ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE)
        )
        d = always_redraw(lambda: Dot(ax.c2p(x_func(~phi), y_func(~phi)), color=ORANGE))
        d_trace = TracedPath(
            d.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_color=ORANGE,
            stroke_width=DEFAULT_STROKE_WIDTH,
        )
        self.add(l, d, d_trace)

        phi_radius = 0.3
        phi_angle = always_redraw(
            lambda: Angle(
                Line(ax.c2p(0, 0), ax.c2p(x_func(~phi), y_func(~phi))),
                Line(ax.c2p(0, 0), ax.c2p(1, 0)),
                radius=phi_radius,
                quadrant=(1, 1),
                color=BLUE,
                other_angle=True,
            )
            if ~phi > 0
            else Line(
                ax.c2p(x_func(~phi) * phi_radius, y_func(~phi) * phi_radius),
                ax.c2p(x_func(~phi) * phi_radius, y_func(~phi) * phi_radius),
            )
        )

        cos_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        ).rotate(-PI / 2)
        cos_ax.shift(ax.c2p(0, -1) - cos_ax.c2p(0, 0))
        sin_ax = Axes(
            x_range=[0, f * 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        )
        sin_ax.shift(ax.c2p(1, 0) - sin_ax.c2p(0, 0))

        phi_ax = Axes(
            x_range=[0, PI, 1],
            y_range=[0, 8, 10],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.4),
        ).next_to(sin_ax, DOWN, LARGE_BUFF)

        self.add(cos_ax, sin_ax)

        plot_opacity = VT(1)
        phi_opacity = VT(1)
        phi_plot = always_redraw(
            lambda: phi_ax.plot(
                phi_func,
                x_range=[0, ~phi, 2 * PI / 200],
                color=BLUE,
                stroke_opacity=~phi_opacity,
                use_smoothing=True,
            )
        )
        sin = always_redraw(
            lambda: sin_ax.plot(
                y_func,
                x_range=[0, ~phi, 2 * PI / 200],
                color=BLUE,
                stroke_opacity=~plot_opacity,
            )
        )
        cos = always_redraw(
            lambda: cos_ax.plot(
                x_func,
                x_range=[0, ~phi, 2 * PI / 200],
                color=GREEN,
                stroke_opacity=~plot_opacity,
            )
        )

        line_to_sin = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                sin_ax.input_to_graph_point(~phi, sin),
                color=BLUE,
                dash_length=DEFAULT_DASH_LENGTH * 3,
                stroke_opacity=~plot_opacity,
            )
        )
        line_to_cos = always_redraw(
            lambda: DashedLine(
                ax.c2p(x_func(~phi), y_func(~phi)),
                cos_ax.input_to_graph_point(~phi, cos),
                color=GREEN,
                dash_length=DEFAULT_DASH_LENGTH * 3,
                stroke_opacity=~plot_opacity,
            )
        )

        self.add(sin, cos, line_to_cos, line_to_sin)

        self.wait(0.5)

        all_group = Group(ax, cos_ax, sin_ax)
        self.play(
            phi @ (4 * PI),
            self.camera.frame.animate.scale_to_fit_height(
                all_group.height * 1.2
            ).move_to(all_group),
            run_time=16,
        )

        self.wait(0.5)

        self.remove(d_trace)
        d_trace = TracedPath(
            d.get_center,
            dissipating_time=2,
            stroke_opacity=[0, 1],
            stroke_color=ORANGE,
            stroke_width=DEFAULT_STROKE_WIDTH,
        )

        self.play(phi @ 0)

        self.wait(0.5)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        new_phi = 20 * DEGREES
        phi_label = MathTex(r"\phi", color=BLUE).next_to(
            ax.c2p(
                phi_radius * np.cos(new_phi / 2),
                phi_radius * np.sin(new_phi / 2),
            ),
            UR,
            SMALL_BUFF,
        )

        self.add(phi_angle)
        new_plot_opacity = 0.1
        self.play(
            LaggedStart(
                AnimationGroup(
                    plot_opacity @ new_plot_opacity,
                    cos_ax.animate.set_opacity(new_plot_opacity),
                    sin_ax.animate.set_opacity(new_plot_opacity),
                ),
                self.camera.frame.animate.scale_to_fit_height(ax.height * 1.5).move_to(
                    ax
                ),
                phi @ (new_phi),
                FadeIn(phi_label, shift=DL),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)

        theta_0_line = Line(ax.c2p(0, 0), ax.c2p(1, 0), color=YELLOW)
        phi_line = Line(ax.c2p(0, 0), ax.c2p(x_func(~phi), y_func(~phi)), color=YELLOW)

        self.play(Create(theta_0_line))

        self.wait(0.5)

        self.play(Create(phi_line))

        self.wait(0.5)

        self.play(Uncreate(theta_0_line), Uncreate(phi_line))

        q_line = Line(
            ax.c2p(x_func(~phi), 0), ax.c2p(x_func(~phi), y_func(~phi)), color=YELLOW
        )
        q_label = MathTex("Q").scale(0.6).next_to(q_line, DR, SMALL_BUFF)
        i_line = Line(
            ax.c2p(0, y_func(~phi)), ax.c2p(x_func(~phi), y_func(~phi)), color=YELLOW
        )
        i_label = MathTex("I").scale(0.6).next_to(i_line, UP, SMALL_BUFF)

        self.play(Create(q_line), FadeIn(q_label))

        self.wait(0.5)

        self.play(Create(i_line), FadeIn(i_label))

        self.wait(0.5)

        phi_eqn = MathTex(r"\phi = \tan^{-1}{\left( \frac{Q}{I} \right)}").next_to(
            ax, UP, SMALL_BUFF
        )
        phi_eqn[0][0].set_color(BLUE)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(Uncreate(q_line), Uncreate(i_line)),
                self.camera.frame.animate.scale(1.3).shift(UP),
                LaggedStart(
                    ReplacementTransform(phi_label[0], phi_eqn[0][0]),
                    *[GrowFromCenter(m) for m in phi_eqn[0][1:8]],
                    ReplacementTransform(q_label[0], phi_eqn[0][8]),
                    GrowFromCenter(phi_eqn[0][9]),
                    ReplacementTransform(i_label[0], phi_eqn[0][10]),
                    GrowFromCenter(phi_eqn[0][11]),
                    lag_ratio=0.15,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        all_group = Group(ax, cos_ax, sin_ax, phi_ax)

        self.add(phi_plot)
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    all_group.height * 1.1
                ).move_to(all_group),
                phi_eqn.animate.next_to(phi_ax, UP, SMALL_BUFF),
                FadeIn(phi_ax),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        line_to_phi_val = DashedLine(
            phi_val_line.get_end(),
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=YELLOW,
        )

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        line_to_phi_val = DashedLine(
            phi_val_line.get_end(),
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=YELLOW,
        )

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        self.play(phi + new_phi)

        self.wait(0.5)

        phi_val_line = Line(
            [phi_ax.input_to_graph_point(~phi, phi_plot)[0], phi_ax.c2p(0, 0)[1], 0],
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=BLUE,
        )

        self.play(TransformFromCopy(phi_angle, phi_val_line), run_time=2)

        self.wait(0.5)

        line_to_phi_val = DashedLine(
            phi_val_line.get_end(),
            phi_ax.input_to_graph_point(~phi, phi_plot),
            color=YELLOW,
        )

        # self.play(Create(line_to_phi_val))

        # self.play(phi @ (PI), run_time=16)

        self.wait(2)


class Trig(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(False))

        lo_ax = Axes(
            x_range=[0, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fw(self, 0.25),
            y_length=fh(self, 0.4),
        )
        rf_ax = Axes(
            x_range=[0, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fw(self, 0.25),
            y_length=fh(self, 0.4),
        )
        if_ax = Axes(
            x_range=[0, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fw(self, 0.25),
            y_length=fh(self, 0.4),
        )
        times = MathTex(r"\times").scale(1.5)
        equal = MathTex(r"=").scale(1.5)
        mult_group = (
            Group(rf_ax, times, lo_ax, equal, if_ax)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .scale_to_fit_width(fw(self, 0.9))
        )

        lo_plot = lo_ax.plot(
            lambda t: np.sin(2 * PI * 12 * t),
            color=TX_COLOR,
            x_range=[0, 1, 1 / 200],
        )
        rf_plot = rf_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t),
            color=RX_COLOR,
            x_range=[0, 1, 1 / 200],
        )

        if_plot = if_ax.plot(
            lambda t: np.sin(2 * PI * 12 * t) * np.sin(2 * PI * 1 * t),
            color=IF_COLOR,
            x_range=[0, 1, 1 / 200],
        )

        self.play(
            LaggedStart(
                LaggedStart(Create(rf_ax), Create(rf_plot), lag_ratio=0.3),
                GrowFromCenter(times),
                LaggedStart(Create(lo_ax), Create(lo_plot), lag_ratio=0.3),
                GrowFromCenter(equal),
                LaggedStart(Create(if_ax), Create(if_plot), lag_ratio=0.3),
                lag_ratio=0.3,
            )
        )

        common_trig = Text("Trig Identities", font=FONT).to_edge(UP, MED_LARGE_BUFF)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        self.play(FadeOut(*mult_group, rf_plot, if_plot, lo_plot))

        self.wait(0.5)

        cos_iden = MathTex(
            r"\cos{(a)} \cdot \cos{(b)} = \frac{1}{2} \left[\cos{(a - b)} + \cos{(a + b)}\right]"
        )
        cos_iden[0][4].set_color(RED)
        cos_iden[0][22].set_color(RED)
        cos_iden[0][31].set_color(RED)
        cos_iden[0][11].set_color(BLUE)
        cos_iden[0][24].set_color(BLUE)
        cos_iden[0][33].set_color(BLUE)

        sin_iden = MathTex(
            r"\cos{(a)} \cdot \sin{(b)} = \frac{1}{2} \left[\sin{(a - b)} + \sin{(a + b)}\right]"
        )
        sin_iden[0][4].set_color(RED)
        sin_iden[0][22].set_color(RED)
        sin_iden[0][31].set_color(RED)
        sin_iden[0][11].set_color(BLUE)
        sin_iden[0][24].set_color(BLUE)
        sin_iden[0][33].set_color(BLUE)
        Group(cos_iden, sin_iden).arrange(DOWN, LARGE_BUFF)

        self.play(
            common_trig.shift(UP * 5).animate(run_time=1).shift(DOWN * 5),
            LaggedStart(
                GrowFromCenter(cos_iden[0][:6]),
                GrowFromCenter(cos_iden[0][6]),
                GrowFromCenter(cos_iden[0][7:13]),
                lag_ratio=0.4,
                run_time=2,
            ),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                [GrowFromCenter(m) for m in cos_iden[0][13:22]],
                TransformFromCopy(cos_iden[0][4], cos_iden[0][22], path_arc=PI * 0.7),
                GrowFromCenter(cos_iden[0][23]),
                TransformFromCopy(cos_iden[0][11], cos_iden[0][24], path_arc=-PI * 0.7),
                [GrowFromCenter(m) for m in cos_iden[0][25:31]],
                TransformFromCopy(cos_iden[0][4], cos_iden[0][31], path_arc=-PI * 0.7),
                GrowFromCenter(cos_iden[0][32]),
                TransformFromCopy(cos_iden[0][11], cos_iden[0][33], path_arc=PI * 0.7),
                [GrowFromCenter(m) for m in cos_iden[0][34:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(sin_iden[0][:6]),
                GrowFromCenter(sin_iden[0][6]),
                GrowFromCenter(sin_iden[0][7:13]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                [GrowFromCenter(m) for m in sin_iden[0][13:22]],
                TransformFromCopy(sin_iden[0][4], sin_iden[0][22], path_arc=PI * 0.7),
                GrowFromCenter(sin_iden[0][23]),
                TransformFromCopy(sin_iden[0][11], sin_iden[0][24], path_arc=-PI * 0.7),
                [GrowFromCenter(m) for m in sin_iden[0][25:31]],
                TransformFromCopy(sin_iden[0][4], sin_iden[0][31], path_arc=-PI * 0.7),
                GrowFromCenter(sin_iden[0][32]),
                TransformFromCopy(sin_iden[0][11], sin_iden[0][33], path_arc=PI * 0.7),
                [GrowFromCenter(m) for m in sin_iden[0][34:]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        eqn = (
            MathTex(r"s(t) = A(t) \cos{(2 \pi f_c t + \phi(t))}")
            .to_edge(UP, LARGE_BUFF)
            .shift(UP * 5)
        )
        arrow = MathTex(r"\mathbf{\rightarrow}").scale(1.5)

        cos_ax = Axes(
            x_range=[0, 2 * PI, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fh(self, 0.7),
            y_length=fh(self, 0.2),
        ).set_opacity(0)
        rx_group = (
            Group(eqn, arrow, cos_ax)
            .arrange(RIGHT)
            .to_edge(UP, LARGE_BUFF)
            .shift(UP * 5)
        )
        plot_opacity = VT(1)
        cos_plot = always_redraw(
            lambda: cos_ax.plot(
                lambda t: np.cos(t),
                x_range=[0, 2 * PI, 2 * PI / 200],
                color=RX_COLOR,
                stroke_opacity=~plot_opacity,
            )
        )
        self.add(cos_ax, cos_plot)

        self.play(common_trig.animate.shift(UP * 5))

        self.remove(common_trig)

        self.wait(0.5)

        self.add(cos_iden, sin_iden)
        self.play(
            Group(eqn, arrow, cos_ax).animate.shift(DOWN * 5),
            Group(sin_iden, cos_iden).animate.shift(DOWN * 2),
        )

        self.wait(0.5)

        hw_box = RoundedRectangle(
            width=fw(self, 0.8),
            height=fh(self, 0.8),
            corner_radius=0.2,
            color=ORANGE,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        )
        hw_label = Text("Hardware Implementation", font=FONT).next_to(
            hw_box.get_top(), DOWN, MED_SMALL_BUFF
        )
        hw_group = Group(hw_box, hw_label)

        self.play(
            LaggedStart(
                AnimationGroup(
                    eqn.animate.scale(0.7)
                    .to_corner(UL, MED_LARGE_BUFF)
                    .set_opacity(0.2),
                    FadeOut(arrow),
                    cos_ax.animate.scale(0.7).next_to(
                        eqn.copy().scale(0.7).to_corner(UL, MED_LARGE_BUFF)
                    ),
                    Group(sin_iden, cos_iden).animate.shift(UP),
                    plot_opacity @ 0.2,
                ),
                hw_group.shift(DOWN * fh(self) * 1.2).animate.shift(
                    UP * fh(self) * 1.2
                ),
            )
        )

        self.wait(0.5)

        mixer_out = Circle(color=BLUE)
        mixer_dr = Line(mixer_out.get_top(), mixer_out.get_bottom()).rotate(PI / 4)
        mixer_dl = Line(mixer_out.get_top(), mixer_out.get_bottom()).rotate(-PI / 4)
        lo_port = (
            Text("LO", font=FONT)
            .scale(0.5)
            .next_to(mixer_out.get_top(), DOWN, SMALL_BUFF)
        )
        rf_port = (
            Text("RF", font=FONT)
            .scale(0.5)
            .next_to(mixer_out.get_right(), LEFT, SMALL_BUFF)
        )
        if_port = (
            Text("IF", font=FONT)
            .scale(0.5)
            .next_to(mixer_out.get_left(), RIGHT, SMALL_BUFF)
        )
        mixer = Group(mixer_out, mixer_dr, mixer_dl, lo_port, if_port, rf_port).next_to(
            hw_box.get_bottom(), UP, MED_LARGE_BUFF
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(mixer_out), Create(mixer_dl), Create(mixer_dr)),
                GrowFromCenter(rf_port),
                GrowFromCenter(lo_port),
                GrowFromCenter(if_port),
            )
        )

        self.wait(0.5)

        lo_ax = (
            Axes(
                x_range=[0, 1, 1],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=hw_box.height * 0.4,
                y_length=mixer_out.width * 0.8,
            )
            .rotate(-PI / 2)
            .set_opacity(0)
        )
        lo_ax.shift(mixer_out.get_top() - lo_ax.c2p(1, 0))
        rf_ax = (
            Axes(
                x_range=[0, 1, 1],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=hw_box.height * 0.4,
                y_length=mixer_out.width * 0.8,
            )
            .rotate(PI)
            .set_opacity(0)
        )
        rf_ax.shift(mixer_out.get_right() - rf_ax.c2p(1, 0))
        if_ax = (
            Axes(
                x_range=[0, 1, 1],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=hw_box.height * 0.4,
                y_length=mixer_out.width * 0.8,
            )
            .rotate(PI)
            .set_opacity(0)
        )
        if_ax.shift(mixer_out.get_left() - if_ax.c2p(0, 0))

        lo_plot = lo_ax.plot(
            lambda t: np.sin(2 * PI * 12 * t),
            color=TX_COLOR,
            x_range=[0, 1, 1 / 200],
        )
        rf_plot = rf_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t),
            color=RX_COLOR,
            x_range=[0, 1, 1 / 200],
        )

        if_plot = if_ax.plot(
            lambda t: np.sin(2 * PI * 12 * t) * np.sin(2 * PI * 1 * t),
            color=IF_COLOR,
            x_range=[0, 1, 1 / 200],
        )

        # self.add(lo_ax, rf_ax)

        self.play(LaggedStart(Create(rf_plot), Create(lo_plot), lag_ratio=0.4))

        self.wait(0.5)

        self.play(Create(if_plot))

        self.wait(0.5)

        thumbnail_02 = ImageMobject(
            "../02_fmcw_implementation/media/images/fmcw_implementation/Thumbnail_Option_1.png"
        ).scale_to_fit_width(fw(self, 0.3))
        thumbnail_box = SurroundingRectangle(thumbnail_02, buff=0)
        thumbnail = Group(thumbnail_02, thumbnail_box).next_to(hw_box.get_right(), LEFT)

        self.play(
            LaggedStart(
                Group(mixer, rf_plot, if_plot, lo_plot).animate.shift(LEFT * 2),
                GrowFromCenter(thumbnail),
            )
        )

        self.wait(0.5)

        hw_bez_r = CubicBezier(
            hw_box.get_corner(DR) + [0, 0.1, 0],
            hw_box.get_corner(DR) + [0, -0.5, 0],
            self.camera.frame.get_bottom() + [0, 0.5, 0],
            self.camera.frame.get_bottom() + [0, 0, 0],
            color=ORANGE,
        )
        hw_bez_l = CubicBezier(
            hw_box.get_corner(DL) + [0, 0.1, 0],
            hw_box.get_corner(DL) + [0, -0.5, 0],
            self.camera.frame.get_bottom() + [0, 0.5, 0],
            self.camera.frame.get_bottom() + [0, 0, 0],
            color=ORANGE,
        )

        self.play(Create(hw_bez_l), Create(hw_bez_r))

        self.wait(0.5)

        self.play(
            Group(
                hw_group,
                mixer,
                lo_plot,
                rf_plot,
                if_plot,
                thumbnail,
                hw_bez_l,
                hw_bez_r,
            ).animate.shift(DOWN * fh(self, 1.5))
        )

        self.wait(0.5)

        cos_arrow = CurvedArrow(
            cos_iden.get_top() + [-1, 0.2, 0],
            eqn.get_bottom() + [2, -0.2, 0],
            angle=TAU / 8,
        )
        sin_arrow = CurvedArrow(
            sin_iden.get_corner(UL) + [-0.1, 0.1, 0],
            eqn.get_bottom() + [-0.5, -0.2, 0],
            angle=-TAU / 8,
        )

        self.play(
            LaggedStart(
                eqn.animate.set_opacity(1),
                Create(cos_arrow),
                Create(sin_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            eqn.animate.set_opacity(0.2),
            FadeOut(cos_arrow, sin_arrow),
        )

        self.wait(0.5)

        cos_box_l = SurroundingRectangle(cos_iden[0][18:21])
        cos_box_r = SurroundingRectangle(cos_iden[0][27:30])
        sin_box_l = SurroundingRectangle(sin_iden[0][18:21])
        sin_box_r = SurroundingRectangle(sin_iden[0][27:30])

        self.play(LaggedStart(Create(cos_box_l), Create(cos_box_r), lag_ratio=0.2))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(cos_box_l, sin_box_l),
                ReplacementTransform(cos_box_r, sin_box_r),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(Uncreate(sin_box_l), Uncreate(sin_box_r)),
                sin_iden.animate.shift(DOWN * 8),
                cos_iden.animate.scale(1.1).move_to(self.camera.frame),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        sub = SurroundingRectangle(cos_iden[0][23])
        a_sub = SurroundingRectangle(cos_iden[0][22:24])
        a_sub_b = SurroundingRectangle(cos_iden[0][22:25])

        self.play(FadeIn(sub))

        self.wait(0.5)

        self.play(ReplacementTransform(sub, a_sub))

        self.wait(0.5)

        self.play(ReplacementTransform(a_sub, a_sub_b))

        self.wait(0.5)

        rx_bez_l = CubicBezier(
            cos_iden[0][0].get_corner(UL) + [0, 0.1, 0],
            cos_iden[0][0].get_corner(UL) + [0, 1, 0],
            eqn[0][9].get_corner(DL) + [0, -1, 0],
            eqn[0][9].get_corner(DL) + [0, -0.1, 0],
        )
        rx_bez_r = CubicBezier(
            cos_iden[0][5].get_corner(UR) + [0, 0.1, 0],
            cos_iden[0][5].get_corner(UR) + [0, 1, 0],
            eqn[0][-1].get_corner(DR) + [0, -1, 0],
            eqn[0][-1].get_corner(DR) + [0, -0.1, 0],
        )

        self.play(
            Create(rx_bez_l),
            Create(rx_bez_r),
            eqn[0][9:].animate.set_opacity(1),
        )

        self.wait(0.5)

        a_bez_l = CubicBezier(
            cos_iden[0][3].get_top() + [0, 0.1, 0],
            cos_iden[0][3].get_top() + [0, 1, 0],
            eqn[0][12].get_bottom() + [0, -1, 0],
            eqn[0][12].get_bottom() + [0, -0.1, 0],
        )
        a_bez_r = CubicBezier(
            cos_iden[0][5].get_top() + [0, 0.1, 0],
            cos_iden[0][5].get_top() + [0, 1, 0],
            eqn[0][-1].get_bottom() + [0, -1, 0],
            eqn[0][-1].get_bottom() + [0, -0.1, 0],
        )

        self.play(
            ReplacementTransform(rx_bez_l, a_bez_l),
            ReplacementTransform(rx_bez_r, a_bez_r),
            eqn[0][13:-1].animate.set_color(RED),
        )

        self.wait(0.5)

        rx_inp_sub_b = MathTex(r"2 \pi f_c t + \phi(t) - b").next_to(cos_iden, UP)
        rx_inp_sub_b[0][:-2].set_color(RED)
        rx_inp_sub_b[0][-1].set_color(BLUE)
        a_sub_b_broken_out = MathTex(r"a - b").next_to(rx_inp_sub_b, UP, MED_LARGE_BUFF)
        a_sub_b_broken_out.shift(
            RIGHT
            * (rx_inp_sub_b[0][-2].get_center() - a_sub_b_broken_out[0][1].get_center())
        )
        a_sub_b_broken_out[0][0].set_color(RED)
        a_sub_b_broken_out[0][2].set_color(BLUE)

        # self.add(rx_inp_sub_b, a_sub_b_broken_out)

        self.play(
            LaggedStart(
                AnimationGroup(Uncreate(a_bez_l), Uncreate(a_bez_r), Uncreate(a_sub_b)),
                TransformFromCopy(cos_iden[0][22], a_sub_b_broken_out[0][0]),
                TransformFromCopy(eqn[0][13:-1], rx_inp_sub_b[0][:-2], path_arc=PI / 3),
                AnimationGroup(
                    GrowFromCenter(rx_inp_sub_b[0][-2]),
                    GrowFromCenter(a_sub_b_broken_out[0][-2]),
                ),
                TransformFromCopy(
                    cos_iden[0][24], a_sub_b_broken_out[0][2], path_arc=-PI / 3
                ),
                TransformFromCopy(
                    cos_iden[0][24], rx_inp_sub_b[0][-1], path_arc=-PI / 3
                ),
                lag_ratio=0.4,
            ),
            run_time=5,
        )

        self.wait(0.5)

        qmark = Text("?", font=FONT, color=YELLOW).next_to(
            Group(a_sub_b_broken_out, rx_inp_sub_b), RIGHT, LARGE_BUFF * 2
        )
        b_top_circle = Circle(
            radius=a_sub_b_broken_out[0][-1].height * 0.8, color=YELLOW
        ).move_to(a_sub_b_broken_out[0][-1])
        b_bot_circle = Circle(
            radius=rx_inp_sub_b[0][-1].height * 0.8, color=YELLOW
        ).move_to(rx_inp_sub_b[0][-1])
        b_top_bez = CubicBezier(
            b_top_circle.get_right() + [0.1, 0, 0],
            b_top_circle.get_right() + [1, 0, 0],
            qmark.get_left() + [-1, 0, 0],
            qmark.get_left() + [-0.1, 0, 0],
        )
        b_bot_bez = CubicBezier(
            b_bot_circle.get_right() + [0.1, 0, 0],
            b_bot_circle.get_right() + [1, 0, 0],
            qmark.get_left() + [-1, 0, 0],
            qmark.get_left() + [-0.1, 0, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(b_top_circle),
                    Create(b_bot_circle),
                ),
                AnimationGroup(
                    Create(b_top_bez),
                    Create(b_bot_bez),
                ),
                FadeIn(qmark),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            *[Uncreate(m) for m in [b_top_circle, b_bot_circle, b_top_bez, b_bot_bez]],
            FadeOut(qmark),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            rx_inp_sub_b[0][:5]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 4),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            rx_inp_sub_b[0][5]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 4),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            rx_inp_sub_b[0][6:10]
            .animate(rate_func=rate_functions.there_and_back_with_pause)
            .set_color(YELLOW)
            .shift(UP / 4),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        rx_inp_sub_fc = MathTex(
            r"2 \pi f_c t + \phi(t) - 2 \pi f_c t = \phi(t)"
        ).next_to(cos_iden, UP)
        rx_inp_sub_fc.shift(
            RIGHT
            * (rx_inp_sub_b[0][-2].get_center() - rx_inp_sub_fc[0][10].get_center())
        )
        rx_inp_sub_fc[0][:10].set_color(RED)
        rx_inp_sub_fc[0][11:16].set_color(BLUE)

        self.play(
            LaggedStart(
                cos_iden.animate.shift(DOWN * 2),
                TransformFromCopy(
                    rx_inp_sub_b[0][:5], rx_inp_sub_fc[0][11:16], path_arc=-PI
                ),
                ShrinkToCenter(rx_inp_sub_b[0][-1]),
                ReplacementTransform(rx_inp_sub_b[0][:11], rx_inp_sub_fc[0][:11]),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[GrowFromCenter(m) for m in rx_inp_sub_fc[0][16:]], lag_ratio=0.1
            )
        )

        self.wait(0.5)

        phase_box = SurroundingRectangle(rx_inp_sub_fc[0][17:])

        self.play(Create(phase_box))

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.scale_to_fit_width(
                rx_inp_sub_fc.width * 1.3
            ).move_to(rx_inp_sub_fc),
            Uncreate(phase_box),
        )

        self.wait(0.5)

        pi_val = (
            MathTex(r"6.28\ldots").next_to(rx_inp_sub_fc, DOWN, LARGE_BUFF).shift(LEFT)
        )
        pi_val_bez_l = CubicBezier(
            rx_inp_sub_fc[0][:2].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][:2].get_bottom() + [0, -1, 0],
            pi_val.get_top() + [0, 1, 0],
            pi_val.get_top() + [0, 0.1, 0],
        )
        pi_val_bez_r = CubicBezier(
            rx_inp_sub_fc[0][11:13].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][11:13].get_bottom() + [0, -1, 0],
            pi_val.get_top() + [0, 1, 0],
            pi_val.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    rx_inp_sub_fc[0][:2].animate.set_color(YELLOW),
                    rx_inp_sub_fc[0][11:13].animate.set_color(YELLOW),
                ),
                AnimationGroup(Create(pi_val_bez_l), Create(pi_val_bez_r)),
                pi_val.shift(DOWN * 5).animate.shift(UP * 5),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            pi_val.animate.shift(DOWN * 5),
            rx_inp_sub_fc[0][:2].animate.set_color(RED),
            rx_inp_sub_fc[0][11:13].animate.set_color(BLUE),
            Uncreate(pi_val_bez_l),
            Uncreate(pi_val_bez_r),
        )
        self.remove(pi_val)

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        cos_iden.shift(DOWN * 10)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).next_to(rx_inp_sub_fc, DOWN, LARGE_BUFF * 2)
        self.add(radar.vgroup)

        f_bez_l = CubicBezier(
            rx_inp_sub_fc[0][2:4].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][2:4].get_bottom() + [0, -1, 0],
            radar.radome.get_top() + [0, 1, 0],
            radar.radome.get_top() + [0, 0.1, 0],
        )
        f_bez_r = CubicBezier(
            rx_inp_sub_fc[0][13:15].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][13:15].get_bottom() + [0, -1, 0],
            radar.radome.get_top() + [0, 1, 0],
            radar.radome.get_top() + [0, 0.1, 0],
        )

        tx_ax = Axes(
            x_range=[0, 1, 1],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=fw(self),
            y_length=radar.radome.height * 1.2,
        )
        tx_ax.shift(radar.radome.get_right() - tx_ax.c2p(0, 0))
        pw = 0.2
        x1 = VT(0)
        tx_plot = always_redraw(
            lambda: tx_ax.plot(
                lambda t: np.sin(2 * PI * 5 * t),
                x_range=[max(0, ~x1 - pw), min(1, ~x1), 1 / 200],
                color=TX_COLOR,
            )
        )
        self.add(tx_plot)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(rx_inp_sub_fc, a_sub_b_broken_out, radar.vgroup).height * 1.2
                ).move_to(Group(rx_inp_sub_fc, radar.vgroup)),
                AnimationGroup(
                    rx_inp_sub_fc[0][2:4].animate.set_color(YELLOW),
                    rx_inp_sub_fc[0][13:15].animate.set_color(YELLOW),
                ),
                AnimationGroup(
                    Create(f_bez_l),
                    Create(f_bez_r),
                ),
                x1.animate(run_time=3).set_value(1 + pw),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.restore(),
            Uncreate(f_bez_l),
            Uncreate(f_bez_r),
            rx_inp_sub_fc[0][2:4].animate.set_color(RED),
            rx_inp_sub_fc[0][13:15].animate.set_color(BLUE),
        )
        self.remove(radar.vgroup, tx_plot)
        cos_iden.shift(UP * 10)

        self.wait(0.5)

        time_label = Text("time", font=FONT).next_to(rx_inp_sub_fc, DOWN, LARGE_BUFF)

        time_bez_l = CubicBezier(
            rx_inp_sub_fc[0][4].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][4].get_bottom() + [0, -1, 0],
            time_label.get_top() + [0, 1, 0],
            time_label.get_top() + [0, 0.1, 0],
        )
        time_bez_r = CubicBezier(
            rx_inp_sub_fc[0][15].get_bottom() + [0, -0.1, 0],
            rx_inp_sub_fc[0][15].get_bottom() + [0, -1, 0],
            time_label.get_top() + [0, 1, 0],
            time_label.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    rx_inp_sub_fc[0][4].animate.set_color(YELLOW),
                    rx_inp_sub_fc[0][15].animate.set_color(YELLOW),
                ),
                AnimationGroup(Create(time_bez_l), Create(time_bez_r)),
                Write(time_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            rx_inp_sub_fc[0][4].animate.set_color(RED),
            rx_inp_sub_fc[0][15].animate.set_color(BLUE),
            Uncreate(time_bez_l),
            Uncreate(time_bez_r),
            FadeOut(time_label),
        )

        self.wait(0.5)

        eqn.set_opacity(0.2)

        all_group = Group(cos_iden, rx_inp_sub_fc, a_sub_b_broken_out)
        self.play(
            self.camera.frame.animate.scale_to_fit_width(all_group.width * 1.3)
            .move_to(all_group)
            .set_x(0)
            .shift(DOWN)
        )

        self.wait(0.5)

        cos_iden_plugged = MathTex(
            r"\frac{1}{2} \left[\cos{(2 \pi f_c t + \phi(t) - 2 \pi f_c t)} + \cos{(2 \pi f_c t + \phi(t) + 2 \pi f_c t)}\right]"
        ).move_to(cos_iden)
        cos_iden_plugged[0][8:18].set_color(RED)
        cos_iden_plugged[0][19:24].set_color(BLUE)
        cos_iden_plugged[0][30:40].set_color(RED)
        cos_iden_plugged[0][41:-2].set_color(BLUE)
        cos_iden_simplified = MathTex(
            r"\frac{1}{2} \left[\cos{(\phi(t))} + \cos{(2 \cdot 2 \pi f_c t + \phi(t))}\right]"
        ).move_to(cos_iden_plugged, LEFT)

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                cos_iden[0][:14].animate.shift(UP * 1.5 + LEFT),
                ReplacementTransform(cos_iden[0][14:22], cos_iden_plugged[0][:8]),
                ShrinkToCenter(cos_iden[0][22]),
                TransformFromCopy(
                    rx_inp_sub_fc[0][:10], cos_iden_plugged[0][8:18], path_arc=PI / 3
                ),
                ReplacementTransform(cos_iden[0][23], cos_iden_plugged[0][18]),
                ShrinkToCenter(cos_iden[0][24]),
                TransformFromCopy(
                    rx_inp_sub_fc[0][11:16], cos_iden_plugged[0][19:24], path_arc=PI / 3
                ),
                ReplacementTransform(cos_iden[0][25:31], cos_iden_plugged[0][24:30]),
                ShrinkToCenter(cos_iden[0][31]),
                TransformFromCopy(
                    rx_inp_sub_fc[0][:10], cos_iden_plugged[0][30:40], path_arc=PI / 3
                ),
                ReplacementTransform(cos_iden[0][32], cos_iden_plugged[0][40]),
                ShrinkToCenter(cos_iden[0][33]),
                TransformFromCopy(
                    rx_inp_sub_fc[0][11:16], cos_iden_plugged[0][41:-2], path_arc=PI / 3
                ),
                ReplacementTransform(cos_iden[0][-2:], cos_iden_plugged[0][-2:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                ReplacementTransform(
                    cos_iden_plugged[0][:8],
                    cos_iden_simplified[0][:8],
                ),
                FadeOut(
                    cos_iden_plugged[0][8:14],
                    cos_iden_plugged[0][18:24],
                    shift=DOWN,
                ),
                ReplacementTransform(
                    cos_iden_plugged[0][14:18],
                    cos_iden_simplified[0][8:12],
                ),
                ReplacementTransform(
                    cos_iden_plugged[0][24:30],
                    cos_iden_simplified[0][12:18],
                ),
                ReplacementTransform(
                    cos_iden_plugged[0][41],
                    cos_iden_simplified[0][18],
                    path_arc=PI / 2,
                ),
                ShrinkToCenter(cos_iden_plugged[0][42:-2]),
                GrowFromCenter(cos_iden_simplified[0][19]),
                ReplacementTransform(
                    cos_iden_plugged[0][30:40],
                    cos_iden_simplified[0][20:-2],
                ),
                ShrinkToCenter(cos_iden_plugged[0][40]),
                ReplacementTransform(
                    cos_iden_plugged[0][-2:],
                    cos_iden_simplified[0][-2:],
                ),
                lag_ratio=0.3,
            ),
            run_time=5,
        )

        self.wait(0.5)

        phi_only = SurroundingRectangle(cos_iden_simplified[0][4:13])
        both_terms = SurroundingRectangle(cos_iden_simplified[0][14:-1])

        self.play(Create(phi_only))

        self.wait(0.5)

        self.play(ReplacementTransform(phi_only, both_terms))

        self.wait(0.5)

        self.play(Uncreate(both_terms))

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(GREEN) for m in cos_iden_simplified[0][4:13]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        cos_iden_final = MathTex(
            r"\cos{(a)} \cdot \cos{(b)} = \frac{1}{2} \left[\cos{(\phi(t))} + \cos{(2 \cdot 2 \pi f_c t + \phi(t))}\right]"
        ).next_to(self.camera.frame.get_bottom(), UP)
        cos_iden_final[0][4].set_color(RED)
        cos_iden_final[0][11].set_color(BLUE)
        cos_iden_final[0][4 + 14 : 13 + 14].set_color(GREEN)

        self.remove(*hw_group, lo_plot, rf_plot, if_plot, sin_iden)
        self.play(
            ReplacementTransform(cos_iden[0][:14], cos_iden_final[0][:14]),
            ReplacementTransform(cos_iden_simplified[0], cos_iden_final[0][13:]),
            self.camera.frame.animate.scale_to_fit_width(
                cos_iden_final.width * 1.5
            ).shift(
                DOWN
                * (
                    self.camera.frame.get_top()
                    - (cos_iden_final.get_top() + LARGE_BUFF)
                )
            ),
        )

        self.wait(2)


class RealNumbers(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        cos_iden_final = MathTex(
            r"\cos{(a)} \cdot \cos{(b)} = \frac{1}{2} \left[\cos{(\phi(t))} + \cos{(2 \cdot 2 \pi f_c t + \phi(t))}\right]"
        ).to_edge(UP, LARGE_BUFF)
        cos_iden_final[0][4].set_color(RED)
        cos_iden_final[0][11].set_color(BLUE)
        cos_iden_final[0][4 + 14 : 13 + 14].set_color(GREEN)

        self.camera.frame.scale_to_fit_width(cos_iden_final.width * 1.5).shift(
            DOWN
            * (self.camera.frame.get_top() - (cos_iden_final.get_top() + LARGE_BUFF))
        )

        self.add(cos_iden_final)

        self.play(
            LaggedStart(
                cos_iden_final[0][4 + 14 : 13 + 14].animate.set_color(WHITE),
                cos_iden_final[0][-10:-8].animate.set_color(GREEN),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        fc_units = MathTex(r"f_c = \text{MHz} \rightarrow \text{GHz}")
        fc_units[0][:2].set_color(GREEN)
        fc_path = CubicBezier(
            cos_iden_final[0][-10:-8].get_center(),
            cos_iden_final[0][-10:-8].get_center() + [0, -2, 0],
            fc_units[0][:2].get_center() + [0, 2, 0],
            fc_units[0][:2].get_center(),
        )
        fc_bez = CubicBezier(
            cos_iden_final[0][-10:-8].get_bottom() + [0, -0.1, 0],
            cos_iden_final[0][-10:-8].get_bottom() + [0, -1, 0],
            fc_units[0][:2].get_top() + [0, 1, 0],
            fc_units[0][:2].get_top() + [0, 0.1, 0],
        )
        fc = MathTex(r"f_c = 3 \text{ GHz}")
        fc[0][:2].set_color(GREEN)
        fc_copy = cos_iden_final[0][-10:-8].copy()

        self.play(
            LaggedStart(
                LaggedStart(
                    MoveAlongPath(fc_copy, fc_path),
                    Create(fc_bez),
                    lag_ratio=0.05,
                ),
                LaggedStart(
                    *[GrowFromCenter(m) for m in fc_units[0][2:]], lag_ratio=0.1
                ),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                Uncreate(fc_bez),
                FadeOut(fc_units[0][3:6], shift=DOWN),
                FadeOut(fc_units[0][6], shift=DOWN),
                ReplacementTransform(fc_units[0][2], fc[0][2]),
                ReplacementTransform(fc_copy, fc[0][:2]),
                ReplacementTransform(fc_units[0][7:], fc[0][4:]),
                LaggedStart(*[GrowFromCenter(m) for m in fc[0][3]]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            fc.animate.to_corner(DL, LARGE_BUFF).set_opacity(0.2),
            LaggedStart(
                cos_iden_final[0][-10:-8].animate.set_color(WHITE),
                cos_iden_final[0][8 + 14 : 12 + 14].animate.set_color(GREEN),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        v = MathTex(r"v")
        pm_v = MathTex(r"\pm v")
        small_movements = MathTex(r"\Delta R \ll 1")
        based_on_group = (
            Group(v, pm_v, small_movements)
            .arrange(RIGHT, MED_LARGE_BUFF)
            .next_to(cos_iden_final[0][8 + 14 : 12 + 14], DOWN, LARGE_BUFF)
        )
        based_on_group.shift(
            RIGHT
            * (cos_iden_final[0][8 + 14 : 12 + 14].get_center() - pm_v.get_center())
        )

        v_bez = CubicBezier(
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -0.1, 0],
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -1, 0],
            v.get_top() + [0, 1, 0],
            v.get_top() + [0, 0.1, 0],
        )
        pm_v_bez = CubicBezier(
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -0.1, 0],
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -1, 0],
            pm_v.get_top() + [0, 1, 0],
            pm_v.get_top() + [0, 0.1, 0],
        )
        small_movements_bez = CubicBezier(
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -0.1, 0],
            cos_iden_final[0][8 + 14 : 12 + 14].get_bottom() + [0, -1, 0],
            small_movements.get_top() + [0, 1, 0],
            small_movements.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                Create(v_bez),
                FadeIn(v),
                Create(pm_v_bez),
                FadeIn(pm_v),
                Create(small_movements_bez),
                FadeIn(small_movements),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Uncreate(v_bez),
            FadeOut(v),
            Uncreate(pm_v_bez),
            FadeOut(pm_v),
            Uncreate(small_movements_bez),
            FadeOut(small_movements),
        )

        self.wait(0.5)

        phi_units = MathTex(r"\phi(t) = 2 \pi f_d t + \phi_0")
        phi_units[0][:4].set_color(GREEN)
        phi_path = CubicBezier(
            cos_iden_final[0][22:26].get_center(),
            cos_iden_final[0][22:26].get_center() + [0, -2, 0],
            phi_units[0][:4].get_center() + [0, 2, 0],
            phi_units[0][:4].get_center(),
        )
        phi_bez = CubicBezier(
            cos_iden_final[0][22:26].get_bottom() + [0, -0.1, 0],
            cos_iden_final[0][22:26].get_bottom() + [0, -1, 0],
            phi_units[0][:4].get_top() + [0, 1, 0],
            phi_units[0][:4].get_top() + [0, 0.1, 0],
        )
        phi_copy = cos_iden_final[0][22:26].copy()

        self.play(
            LaggedStart(
                LaggedStart(
                    MoveAlongPath(phi_copy, phi_path),
                    Create(phi_bez),
                    lag_ratio=0.05,
                ),
                LaggedStart(
                    *[GrowFromCenter(m) for m in phi_units[0][4:]], lag_ratio=0.1
                ),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                *[m.animate.set_color(GREEN) for m in phi_units[0][5:10]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(WHITE) for m in phi_units[0][5:10]],
                *[m.animate.set_color(GREEN) for m in phi_units[0][11:]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        phi_0_range = MathTex(r"0 \rightarrow 2 \pi").next_to(
            phi_units[0][11:], DOWN, LARGE_BUFF
        )
        phi_0_bez_r = CubicBezier(
            phi_units[0][11:].get_bottom() + [0, -0.1, 0],
            phi_units[0][11:].get_bottom() + [0, -0.7, 0],
            phi_0_range.get_corner(UR) + [0, 0.7, 0],
            phi_0_range.get_corner(UR) + [0, 0.1, 0],
        )
        phi_0_bez_l = CubicBezier(
            phi_units[0][11:].get_bottom() + [0, -0.1, 0],
            phi_units[0][11:].get_bottom() + [0, -0.7, 0],
            phi_0_range.get_corner(UL) + [0, 0.7, 0],
            phi_0_range.get_corner(UL) + [0, 0.1, 0],
        )

        self.play(
            Create(phi_0_bez_r),
            Create(phi_0_bez_l),
            FadeIn(phi_0_range, shift=DOWN / 2),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(phi_0_bez_r),
                    Uncreate(phi_0_bez_l),
                    FadeOut(phi_0_range, shift=UP / 2),
                ),
                LaggedStart(
                    *[m.animate.set_color(WHITE) for m in phi_units[0][11:][::-1]],
                    *[m.animate.set_color(GREEN) for m in phi_units[0][5:10][::-1]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        fd_eqn = MathTex(r"f_d = \frac{2 v f_c}{c}").next_to(
            phi_units[0][5:10], DOWN, LARGE_BUFF
        )
        fd_eqn[0][:2].set_color(GREEN)
        fd_path = CubicBezier(
            phi_units[0][7:9].get_center(),
            phi_units[0][7:9].get_center() + [0, -0.7, 0],
            fd_eqn[0][:2].get_center() + [0, 0.7, 0],
            fd_eqn[0][:2].get_center(),
        )
        fd_bez = CubicBezier(
            phi_units[0][7:9].get_bottom() + [0, -0.1, 0],
            phi_units[0][7:9].get_bottom() + [0, -0.7, 0],
            fd_eqn[0][:2].get_top() + [0, 0.7, 0],
            fd_eqn[0][:2].get_top() + [0, 0.1, 0],
        )
        fd_copy = phi_units[0][7:9].copy()

        self.play(
            LaggedStart(
                LaggedStart(
                    MoveAlongPath(fd_copy, fd_path),
                    Create(fd_bez),
                    lag_ratio=0.05,
                ),
                LaggedStart(*[GrowFromCenter(m) for m in fd_eqn[0][2:]], lag_ratio=0.1),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.wait(0.5)

        self.play(
            fd_eqn[0][4]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .shift(UP / 3)
        )

        self.wait(0.5)

        self.play(
            fd_eqn[0][8]
            .animate(rate_func=rate_functions.there_and_back)
            .set_color(YELLOW)
            .shift(DOWN / 3)
        )

        self.wait(0.5)

        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).to_corner(DL, LARGE_BUFF * 1.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .to_edge(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP)
        )
        radar_group = Group(radar.vgroup, cloud).shift(DOWN * fh(self, 0.65))

        v_val = MathTex(r"\leftarrow v = 10 \text{ m/s}").next_to(cloud, UP)

        self.next_section(skip_animations=skip_animations(True))

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self, 0.4)),
                radar_group.shift(DOWN * 5).animate.shift(UP * 5),
                LaggedStart(
                    GrowFromCenter(v_val[0][0]),
                    TransformFromCopy(fd_eqn[0][4], v_val[0][1], path_arc=PI / 2),
                    *[GrowFromCenter(m) for m in v_val[0][2:]],
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        fd_eqn_val = MathTex(
            r"f_d = \frac{2 (10 \text{ m/s}) f_c}{c} \approx 200 \text{ Hz}"
        ).move_to(fd_eqn)
        fd_eqn_val[0][:2].set_color(GREEN)

        self.play(
            LaggedStart(
                Uncreate(fd_bez),
                AnimationGroup(
                    ReplacementTransform(fd_copy, fd_eqn_val[0][:2]),
                    ReplacementTransform(fd_eqn[0][2:4], fd_eqn_val[0][2:4]),
                    ReplacementTransform(fd_eqn[0][5:7], fd_eqn_val[0][11:13]),
                    ReplacementTransform(fd_eqn[0][7], fd_eqn_val[0][13]),
                    ReplacementTransform(fd_eqn[0][8], fd_eqn_val[0][14]),
                    ShrinkToCenter(fd_eqn[0][4]),
                ),
                AnimationGroup(
                    GrowFromCenter(fd_eqn_val[0][4]),
                    GrowFromCenter(fd_eqn_val[0][10]),
                ),
                TransformFromCopy(v_val[0][3:], fd_eqn_val[0][5:10], path_arc=PI / 2),
                LaggedStart(*[GrowFromCenter(m) for m in fd_eqn_val[0][14:]]),
                lag_ratio=0.4,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.restore(),
            radar_group.add(v_val).animate.shift(DOWN * 5),
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        phi_units_fd_val = MathTex(r"\phi(t) = 2 \pi (200 \text{ Hz}) t + \phi_0")
        phi_units_fd_val[0][:4].set_color(GREEN)
        phi_units_fd_val[0][5:15].set_color(GREEN)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(phi_copy, phi_units_fd_val[0][:4]),
                    ReplacementTransform(phi_units[0][4:7], phi_units_fd_val[0][4:7]),
                    ReplacementTransform(phi_units[0][9:], phi_units_fd_val[0][14:]),
                    ShrinkToCenter(phi_units[0][7:9]),
                ),
                AnimationGroup(
                    GrowFromCenter(phi_units_fd_val[0][7]),
                    GrowFromCenter(phi_units_fd_val[0][13]),
                ),
                TransformFromCopy(fd_eqn_val[0][16:], phi_units_fd_val[0][8:13]),
                FadeOut(fd_eqn_val),
                lag_ratio=0.3,
            )
        )

        phi_units_vals = MathTex(r"\phi(t) = 2 \pi (200 \text{ Hz}) t + \phi_0")
        phi_units_vals[0][:4].set_color(GREEN)
        phi_units_vals[0][5:15].set_color(GREEN)

        self.wait(0.5)

        self.play(phi_units_fd_val[0][5:15].animate.set_color(WHITE))

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        cos_iden_left = (
            MathTex(r"\cos{(a)} \cdot \cos{(b)}").move_to(cos_iden_final).shift(LEFT)
        )
        cos_iden_left[0][4].set_color(RED)
        cos_iden_left[0][11].set_color(BLUE)

        cos_iden_phi_sub = MathTex(
            r"\frac{1}{2} \left[\cos{(2 \pi (200 \text{ Hz}) t + \phi_0)} + \cos{(2 \cdot 2 \pi f_c t + (2 \pi (200 \text{ Hz}) t + \phi_0))}\right]"
        ).next_to(cos_iden_final, DOWN, SMALL_BUFF)
        # cos_iden_phi_sub[0][8:18].set_color(GREEN)
        cos_iden_vals_1 = (
            MathTex(r"\frac{1}{2} \left[\cos{(2 \pi (200 \text{ Hz}) t + \phi_0)} +")
            .move_to(cos_iden_phi_sub, LEFT)
            .shift(RIGHT)
        )
        cos_iden_vals_2 = (
            MathTex(
                r"\cos{(2 \cdot 2 \pi (3 \text{ GHz}) t + (2 \pi (200 \text{ Hz}) t + \phi_0))}\right]"
            )
            .next_to(cos_iden_vals_1, DOWN, MED_SMALL_BUFF, LEFT)
            .shift(RIGHT / 2)
        )
        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                Uncreate(phi_bez),
                ReplacementTransform(cos_iden_final[0][14:22], cos_iden_phi_sub[0][:8]),
                AnimationGroup(
                    ShrinkToCenter(cos_iden_final[0][22:26]),
                ),
                TransformFromCopy(
                    phi_units_fd_val[0][5:], cos_iden_phi_sub[0][8:21], path_arc=-PI / 3
                ),
                ReplacementTransform(
                    cos_iden_final[0][26:40],
                    cos_iden_phi_sub[0][21:35],
                    path_arc=PI / 4,
                ),
                GrowFromCenter(cos_iden_phi_sub[0][35]),
                ReplacementTransform(
                    phi_units_fd_val[0][5:], cos_iden_phi_sub[0][36:49], path_arc=PI / 3
                ),
                AnimationGroup(
                    ShrinkToCenter(phi_units_fd_val[0][:5]),
                    ShrinkToCenter(cos_iden_final[0][40:44]),
                ),
                GrowFromCenter(cos_iden_phi_sub[0][49]),
                *[
                    ReplacementTransform(a, b, path_arc=-PI / 4)
                    for a, b in zip(cos_iden_final[0][44:], cos_iden_phi_sub[0][50:])
                ],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(fc.animate.set_opacity(1))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(cos_iden_phi_sub[0][:23], cos_iden_vals_1[0]),
                ReplacementTransform(
                    cos_iden_phi_sub[0][23:31],
                    cos_iden_vals_2[0][:8],
                    path_arc=-PI / 3,
                ),
                GrowFromCenter(cos_iden_vals_2[0][8]),
                ReplacementTransform(
                    fc[0][3:],
                    cos_iden_vals_2[0][9:13],
                    path_arc=PI / 3,
                ),
                GrowFromCenter(cos_iden_vals_2[0][13]),
                AnimationGroup(
                    ShrinkToCenter(fc[0][:3]),
                    ShrinkToCenter(cos_iden_phi_sub[0][31:33]),
                ),
                ReplacementTransform(
                    cos_iden_phi_sub[0][33:],
                    cos_iden_vals_2[0][14:],
                    path_arc=-PI / 3,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        cos_iden_vals_2_fs = MathTex(
            r"\cos{(2 \cdot 2 \pi (3 \text{ GHz} + 200 \text{ Hz}) t + \phi_0)}\right]"
        ).move_to(cos_iden_vals_2, LEFT)

        self.play(
            LaggedStart(
                ReplacementTransform(
                    cos_iden_vals_2[0][:13], cos_iden_vals_2_fs[0][:13]
                ),
                FadeOut(cos_iden_vals_2[0][13:15]),
                ReplacementTransform(cos_iden_vals_2[0][15], cos_iden_vals_2_fs[0][13]),
                FadeOut(cos_iden_vals_2[0][16:20]),
                FadeOut(cos_iden_vals_2[0][31]),
                ReplacementTransform(
                    cos_iden_vals_2[0][20:31], cos_iden_vals_2_fs[0][14:25]
                ),
                ReplacementTransform(
                    cos_iden_vals_2[0][32:], cos_iden_vals_2_fs[0][25:]
                ),
                lag_ratio=0.3,
            )
        )
        self.add(cos_iden_vals_2_fs)

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        lf_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        )
        hf_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        )
        Group(lf_ax, hf_ax).arrange(DOWN, MED_SMALL_BUFF).next_to(
            self.camera.frame.get_bottom(), UP, 0
        )
        lf_plot = lf_ax.plot(
            lambda t: np.cos(2 * PI * 1 * t), color=GREEN, x_range=[0, 1, 1 / 200]
        )
        hf_plot = hf_ax.plot(
            lambda t: np.cos(2 * PI * 30 * t), color=PURPLE, x_range=[0, 1, 1 / 1000]
        )
        sum_plot = lf_ax.plot(
            lambda t: np.cos(2 * PI * 30 * t) + np.cos(2 * PI * 1 * t),
            color=ORANGE,
            x_range=[0, 1, 1 / 1000],
        ).shift((hf_ax.c2p(0, 0) - lf_ax.c2p(0, 0)) / 2)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN),
                Create(hf_ax),
                cos_iden_vals_2_fs[0][9:19].animate.set_color(PURPLE),
                Create(lf_ax),
                cos_iden_vals_1[0][11:16].animate.set_color(GREEN),
                Create(hf_plot),
                Create(lf_plot),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            Group(hf_ax, hf_plot).animate.shift(
                (lf_ax.c2p(0, 0) - hf_ax.c2p(0, 0)) / 2
            ),
            Group(lf_ax, lf_plot).animate.shift(
                (hf_ax.c2p(0, 0) - lf_ax.c2p(0, 0)) / 2
            ),
        )

        self.wait(0.5)

        self.play(
            ReplacementTransform(hf_plot, sum_plot),
            FadeOut(lf_plot),
            # cos_iden_vals_2_fs[0][9:19].animate.set_color(ORANGE),
        )

        self.wait(0.5)

        phase_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        )
        self.remove(hf_ax)
        self.next_section(skip_animations=skip_animations(True))

        plus = Text("+", font=FONT).scale(1.5).next_to(sum_plot, UP, MED_LARGE_BUFF)

        lf_ax_2 = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        )
        hf_ax_2 = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        )
        plot_group = (
            Group(lf_ax_2, hf_ax_2)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .next_to(plus, UP, LARGE_BUFF)
        )
        lf_plot_2 = lf_ax_2.plot(
            lambda t: np.cos(2 * PI * 1 * t), color=GREEN, x_range=[0, 1, 1 / 200]
        )
        hf_plot_2 = hf_ax_2.plot(
            lambda t: np.cos(2 * PI * 30 * t), color=PURPLE, x_range=[0, 1, 1 / 1000]
        )

        phase_term = (
            MathTex(r"200 \text{ Hz}")
            .set_color(GREEN)
            .next_to(plot_group[0], UP, MED_LARGE_BUFF)
        )
        phase_and_fc_term = (
            MathTex(r"3 \text{ GHz} + 200 \text{ Hz}")
            .set_color(PURPLE)
            .next_to(plot_group[1], UP, MED_LARGE_BUFF)
        )

        lf_bez = CubicBezier(
            lf_ax_2.get_bottom() + [0, -0.1, 0],
            lf_ax_2.get_bottom() + [0, -1, 0],
            plus.get_left() + [-1, 0, 0],
            plus.get_left() + [-0.1, 0, 0],
        )
        hf_bez = CubicBezier(
            hf_ax_2.get_bottom() + [0, -0.1, 0],
            hf_ax_2.get_bottom() + [0, -1, 0],
            plus.get_right() + [1, 0, 0],
            plus.get_right() + [0.1, 0, 0],
        )
        equal_line = Arrow(
            plus.get_bottom(), [plus.get_bottom()[0], lf_ax.get_top()[1], 0]
        )
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                FadeOut(
                    cos_iden_vals_1[0][:11],
                    cos_iden_vals_1[0][16:],
                    cos_iden_vals_2_fs[0][:9],
                    cos_iden_vals_2_fs[0][19:],
                    cos_iden_final[0][:14],
                ),
                self.camera.frame.animate.scale_to_fit_width(
                    Group(phase_term, phase_and_fc_term, plot_group, sum_plot).width
                    * 1.1
                ).move_to(Group(phase_term, phase_and_fc_term, sum_plot, plot_group)),
                Group(sum_plot, lf_ax).animate.shift(DOWN),
                AnimationGroup(
                    ReplacementTransform(
                        cos_iden_vals_2_fs[0][9:19], phase_and_fc_term[0]
                    ),
                    ReplacementTransform(cos_iden_vals_1[0][11:16], phase_term[0]),
                ),
                AnimationGroup(
                    LaggedStart(Create(lf_ax_2), Create(lf_plot_2)),
                    LaggedStart(Create(hf_ax_2), Create(hf_plot_2)),
                ),
                AnimationGroup(Create(lf_bez), Create(hf_bez)),
                GrowFromCenter(plus),
                GrowArrow(equal_line),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(Indicate(Group(phase_term[0], lf_ax_2, lf_plot_2)))

        self.wait(0.5)

        self.play(Indicate(Group(phase_and_fc_term[0], hf_ax_2, hf_plot_2)))

        self.wait(0.5)

        lpf_block = (
            BLOCKS.get("lp_filter").copy().next_to(lf_ax, RIGHT, LARGE_BUFF * 1.5)
        )
        lpf_label = (
            Group(
                Text("Low-Pass", font=FONT),
                Text("Filter", font=FONT),
            )
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(lpf_block, UP)
        )
        lpf_arrow_l = Arrow(lf_ax.get_right(), lpf_block.get_left(), buff=SMALL_BUFF)

        lpf_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=lf_ax.width,
            y_length=lf_ax.height,
        ).next_to(lpf_block, RIGHT, LARGE_BUFF * 1.5)
        lpf_plot = lpf_ax.plot(
            lambda t: np.cos(2 * PI * 1 * t), color=ORANGE, x_range=[0, 1, 1 / 200]
        )
        lpf_arrow_r = Arrow(lpf_block.get_right(), lpf_ax.get_left(), buff=SMALL_BUFF)

        lpf_group = Group(lpf_arrow_l, lpf_block, lpf_label, lf_ax, lpf_ax)

        self.next_section(skip_animations=skip_animations(True))

        self.remove(*v_val[0], cloud, *radar.vgroup)

        self.play(
            LaggedStart(
                AnimationGroup(
                    *[
                        m.animate.set_stroke(opacity=0.2)
                        for m in [
                            lf_plot_2,
                            lf_ax_2,
                            hf_ax_2,
                            hf_plot_2,
                            hf_bez,
                            lf_bez,
                        ]
                    ],
                    *[
                        m.animate.set_opacity(0.2)
                        for m in [
                            plus,
                            equal_line,
                            phase_term[0],
                            phase_and_fc_term[0],
                        ]
                    ],
                ),
                self.camera.frame.animate.scale_to_fit_width(
                    lpf_group.width * 1.2
                ).move_to(lpf_group),
                GrowArrow(lpf_arrow_l),
                GrowFromCenter(lpf_block),
                FadeIn(lpf_label, shift=DOWN),
                GrowArrow(lpf_arrow_r),
                Create(lpf_ax),
                lag_ratio=0.4,
            ),
            run_time=4,
        )

        self.wait(0.5)

        self.play(Create(lpf_plot))

        self.wait(0.5)

        self.play(
            Group(
                lf_ax,
                sum_plot,
                lpf_block,
                lpf_arrow_l,
                lpf_arrow_r,
                lpf_ax,
                lpf_plot,
                lpf_label,
            ).animate.shift(
                RIGHT * (plus.get_center() - lpf_block.get_center()) + DOWN
            ),
            self.camera.frame.animate.scale(0.9).move_to(plus).shift(DOWN),
            *[
                m.animate.set_stroke(opacity=1)
                for m in [
                    lf_plot_2,
                    lf_ax_2,
                    hf_ax_2,
                    hf_plot_2,
                ]
            ],
            *[
                m.animate.set_opacity(1)
                for m in [
                    phase_term[0],
                    phase_and_fc_term[0],
                ]
            ],
            FadeOut(
                plus,
                equal_line,
                hf_bez,
                lf_bez,
            ),
        )

        self.wait(0.5)

        hf_bez = CubicBezier(
            hf_ax_2.get_bottom() + [0, -0.1, 0],
            hf_ax_2.get_bottom() + [0, -3, 0],
            lpf_ax.get_top() + [1, 3, 0],
            lpf_ax.get_top() + [1, 0.1, 0],
        )
        lf_bez = CubicBezier(
            lf_ax_2.get_bottom() + [0, -0.1, 0],
            lf_ax_2.get_bottom() + [0, -3, 0],
            lpf_ax.get_top() + [-1, 3, 0],
            lpf_ax.get_top() + [-1, 0.1, 0],
        )
        lf_check = (
            MathTex(r"\textbf{\checkmark}")
            .set_color(GREEN)
            .scale(2.5)
            .move_to(lf_bez.get_midpoint())
        )
        lf_check_bg = BackgroundRectangle(lf_check, buff=SMALL_BUFF, corner_radius=0.3)
        lf_check_group = Group(lf_check_bg, lf_check)
        hf_x = (
            MathTex(r"\textbf{$\times$}")
            .set_color(RED)
            .scale(2.5)
            .move_to(hf_bez.get_midpoint())
        )
        hf_x_bg = BackgroundRectangle(hf_x, buff=SMALL_BUFF, corner_radius=0.3)
        hf_x_group = Group(hf_x_bg, hf_x)

        self.play(
            LaggedStart(
                Create(lf_bez),
                GrowFromCenter(lf_check_group),
                Create(hf_bez),
                GrowFromCenter(hf_x_group),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            *[m.animate.set_stroke(opacity=0.2) for m in [hf_ax_2, hf_plot_2, hf_bez]],
            *[m.animate.set_opacity(0.2) for m in [phase_and_fc_term[0], hf_x]],
        )

        self.wait(0.5)

        phi_val = MathTex(
            r"\cos{(\phi(t)}) = \cos{(2 \pi (200 \text{ Hz}) t + \phi_0})"
        ).move_to(phase_term[0])
        phi_val[0][17:22].set_color(GREEN)

        self.play(
            LaggedStart(
                ReplacementTransform(phase_term[0], phi_val[0][17:22]),
                *[GrowFromCenter(m) for m in phi_val[0][:17]],
                *[GrowFromCenter(m) for m in phi_val[0][22:]],
                lag_ratio=0.2,
            )
        )

        self.add(phi_val)
        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(lf_bez),
                Uncreate(hf_bez),
                FadeOut(lf_check, lf_check_bg, hf_x, hf_x_bg),
                phi_val.animate.next_to(lpf_ax, UP, MED_LARGE_BUFF),
                self.camera.frame.animate.scale_to_fit_width(
                    lpf_ax.width * 1.2
                ).move_to(lpf_ax),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate(rate_func=rate_functions.ease_in_sine).shift(
                UP * fh(self, 3)
            )
        )

        # self.play(
        #     LaggedStart(
        #         self.camera.frame.animate.scale_to_fit_width(
        #             Group(phase_term, phase_and_fc_term, phase_plot).width * 1.2
        #         ).move_to(Group(phase_term, phase_and_fc_term, phase_plot)),
        #         FadeOut(
        #             cos_iden_vals_1[0][:11],
        #             cos_iden_vals_1[0][16:],
        #             cos_iden_vals_2_fs[0][:9],
        #             cos_iden_vals_2_fs[0][19:],
        #             cos_iden_final[0][:14],
        #         ),
        #         Group(sum_plot, lf_ax).animate.move_to(plot_group[1]),
        #         ReplacementTransform(cos_iden_vals_2_fs[0][9:19], phase_and_fc_term[0]),
        #         ReplacementTransform(cos_iden_vals_1[0][11:16], phase_term[0]),
        #         LaggedStart(Create(phase_ax), Create(phase_plot), lag_ratio=0.2),
        #         lag_ratio=0.5,
        #     ),
        #     run_time=3,
        # )

        # self.wait(0.5)

        # phase_term_box = SurroundingRectangle(phase_term)
        # phase_and_fc_term_box = SurroundingRectangle(phase_and_fc_term)

        # self.play(Create(phase_term_box))

        # self.wait(0.5)

        # self.play(ReplacementTransform(phase_term_box, phase_and_fc_term_box))

        self.wait(2)


class SineFilter(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        sin_iden = MathTex(
            r"\cos{(a)} \cdot \sin{(b)} = \frac{1}{2} \left[\sin{(a - b)} + \sin{(a + b)}\right]"
        )
        sin_iden[0][4].set_color(RED)
        sin_iden[0][22].set_color(RED)
        sin_iden[0][31].set_color(RED)
        sin_iden[0][11].set_color(BLUE)
        sin_iden[0][24].set_color(BLUE)
        sin_iden[0][33].set_color(BLUE)

        self.add(sin_iden)

        self.play(
            self.camera.frame.shift(DOWN * fh(self))
            .animate(rate_func=rate_functions.ease_out_sine)
            .shift(UP * fh(self))
        )

        self.wait(0.5)

        # TODO: add coloring
        sin_iden_plugged = MathTex(
            r"\cos{(2 \pi f_c t + \phi(t))} \cdot \sin{(2 \pi f_c t)} =\frac{1}{2} \left[\sin{(2 \pi f_c t + \phi(t) - 2 \pi f_c t)} + \sin{(2 \pi f_c t + \phi(t) + 2 \pi f_c t)}\right]"
        ).next_to(sin_iden, RIGHT, LARGE_BUFF * 2)
        arrow_1 = Arrow(sin_iden.get_right(), sin_iden_plugged.get_left())

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=6).move_to(sin_iden_plugged[0][-1]),
                GrowArrow(arrow_1),
                LaggedStart(*[FadeIn(m) for m in sin_iden_plugged[0]], lag_ratio=0.05),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        sum_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.2),
        ).next_to(sin_iden_plugged, RIGHT, LARGE_BUFF * 2)
        sum_plot = sum_ax.plot(
            lambda t: np.sin(2 * PI * 30 * t) + np.sin(2 * PI * 1 * t),
            color=ORANGE,
            x_range=[0, 1, 1 / 1000],
        )

        arrow_2 = Arrow(sin_iden_plugged.get_right(), sum_ax.get_left())

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=2).move_to(sum_ax),
                GrowArrow(arrow_2),
                Create(sum_ax),
                Create(sum_plot),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        lpf_block = (
            BLOCKS.get("lp_filter").copy().next_to(sum_ax, RIGHT, LARGE_BUFF * 2)
        )
        arrow_3 = Arrow(sum_ax.get_right(), lpf_block.get_left())
        lpf_label = (
            Group(
                Text("Low-Pass", font=FONT),
                Text("Filter", font=FONT),
            )
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to(lpf_block, UP)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=2).move_to(lpf_block),
                GrowArrow(arrow_3),
                GrowFromCenter(lpf_block),
                FadeIn(lpf_label, shift=DOWN),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        sin_lpf_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.4),
        ).next_to(lpf_block, RIGHT, LARGE_BUFF * 2)
        lpf_plot = sin_lpf_ax.plot(
            lambda t: np.sin(2 * PI * 1 * t), color=ORANGE, x_range=[0, 1, 1 / 200]
        )

        arrow_4 = Arrow(lpf_block.get_right(), sin_lpf_ax.get_left())

        sin_phi_val = MathTex(
            r"\sin{(\phi(t)}) = \sin{(2 \pi (200 \text{ Hz}) t + \phi_0})"
        ).next_to(sin_lpf_ax, UP)
        sin_phi_val[0][17:22].set_color(GREEN)

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=2).move_to(sin_lpf_ax),
                GrowArrow(arrow_4),
                Create(sin_lpf_ax),
                Create(lpf_plot, shift=DOWN),
                FadeIn(sin_phi_val, shift=DOWN),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        cos_lpf_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.6),
            y_length=fh(self, 0.4),
        ).next_to(sin_lpf_ax, UP, LARGE_BUFF)
        cos_lpf_plot = cos_lpf_ax.plot(
            lambda t: np.cos(2 * PI * 1 * t), color=ORANGE, x_range=[0, 1, 1 / 200]
        )
        cos_phi_val = MathTex(
            r"\cos{(\phi(t)}) = \cos{(2 \pi (200 \text{ Hz}) t + \phi_0})"
        ).next_to(cos_lpf_ax, UP)
        cos_phi_val[0][17:22].set_color(GREEN)

        lpf_group = Group(sin_lpf_ax, sin_phi_val, cos_phi_val, cos_lpf_ax)

        self.play(
            LaggedStart(
                AnimationGroup(
                    *[
                        m.animate.set_opacity(0.2)
                        for m in [
                            arrow_4,
                            arrow_3,
                            lpf_block,
                            *lpf_label,
                        ]
                    ]
                ),
                self.camera.frame.animate.scale_to_fit_height(
                    lpf_group.height * 1.2
                ).move_to(lpf_group),
                FadeIn(cos_phi_val),
                Create(cos_lpf_ax),
                Create(cos_lpf_plot),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.shift(RIGHT * fw(self)),
            Group(*lpf_group, lpf_plot, cos_lpf_plot).animate.shift(
                RIGHT * (fw(self) + 3)
            ),
        )

        self.wait(0.5)

        i_label = Text("In-Phase", font=FONT).next_to(cos_lpf_ax, LEFT, LARGE_BUFF)
        q_label = Text("Quadrature", font=FONT).next_to(sin_lpf_ax, LEFT, LARGE_BUFF)

        self.play(Write(i_label), Write(q_label))

        self.wait(2)


class IQ3D(ThreeDScene):
    def construct(self):
        iq_ax = ThreeDAxes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[0, 20 * PI, 1],
            x_length=self.camera.frame_height * 0.5,
            y_length=self.camera.frame_height * 0.5,
            z_length=self.camera.frame_height * 0.5 * 2 * PI,
            tips=False,
        ).rotate(PI, axis=UP)
        iq_ax.shift(self.camera.frame_center - iq_ax.c2p(0, 0, 0))
        self.add(iq_ax)

        theta_pi = VT(PI / 3)

        i_line = always_redraw(
            lambda: Line(
                iq_ax.c2p(0, 0, ~theta_pi),
                iq_ax.c2p(-np.cos(~theta_pi), 0, ~theta_pi),
                color=YELLOW,
            )
        )
        q_line = always_redraw(
            lambda: Line(
                iq_ax.c2p(0, 0, ~theta_pi),
                iq_ax.c2p(0, np.sin(~theta_pi), ~theta_pi),
                color=YELLOW,
            )
        )

        i_line_to_dot = always_redraw(
            lambda: DashedLine(
                iq_ax.c2p(-np.cos(~theta_pi), 0, ~theta_pi),
                iq_ax.c2p(-np.cos(~theta_pi), np.sin(~theta_pi), ~theta_pi),
                dash_length=DEFAULT_DASH_LENGTH * 2,
                color=YELLOW,
            )
        )
        q_line_to_dot = always_redraw(
            lambda: DashedLine(
                iq_ax.c2p(0, np.sin(~theta_pi), ~theta_pi),
                iq_ax.c2p(-np.cos(~theta_pi), np.sin(~theta_pi), ~theta_pi),
                dash_length=DEFAULT_DASH_LENGTH * 2,
                color=YELLOW,
            )
        )

        iq_dot = always_redraw(
            lambda: Dot(color=YELLOW).move_to(
                iq_ax.c2p(-np.cos(~theta_pi), np.sin(~theta_pi), ~theta_pi)
            )
        )

        iq_path = TracedPath(
            iq_dot.get_center,
            stroke_width=DEFAULT_STROKE_WIDTH,
            stroke_color=BLUE,
            dissipating_time=None,
        )

        theta_label = always_redraw(
            lambda: MathTex(f"\\theta = {~theta_pi / PI:.2f} \\pi")
            .next_to(iq_ax, UP, MED_LARGE_BUFF)
            .set_z(0)
            .shift(LEFT)
        )
        # self.add_fixed_in_frame_mobjects(theta_label)

        self.add(i_line, q_line, i_line_to_dot, q_line_to_dot, iq_dot, iq_path)

        self.play(
            FadeIn(theta_label, run_time=1),
            Group(i_line, q_line, iq_ax, i_line_to_dot, q_line_to_dot, iq_dot, iq_path)
            .shift(DOWN * 12)
            .animate(run_time=3)
            .shift(UP * 12),
        )

        self.wait(0.5)

        phi, theta, focal_dist, gamma, zoom = self.camera.get_value_trackers()
        self.play(
            phi.animate.increment_value(-15 * DEGREES),
            gamma.animate.increment_value(-30 * DEGREES),
            theta.animate.increment_value(-30 * DEGREES),
            run_time=2,
        )

        self.wait(1)

        self.play(theta_pi @ (20 * PI), run_time=20)

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class Cloud(MovingCameraScene):
    def construct(self):
        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).to_corner(DL, LARGE_BUFF * 1.5).shift(LEFT * 3)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .to_edge(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP + RIGHT * 2)
        )

        self.add(radar.vgroup)

        # self.play(
        #     radar.vgroup.shift(LEFT * 8).animate.shift(RIGHT * 8),
        #     cloud.shift(RIGHT * 8).animate.shift(LEFT * 8),
        # )
        to_cloud = Line(radar.radome.get_center(), cloud.get_left())
        ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=to_cloud.get_length(),
                y_length=fh(self, 0.2),
            )
            .set_opacity(0)
            .rotate(to_cloud.get_angle())
        )
        ax.shift(radar.radome.get_center() - ax.c2p(0, 0))
        rtn_ax = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[-1, 1, 1],
                tips=False,
                x_length=to_cloud.get_length(),
                y_length=fh(self, 0.2),
            )
            .set_opacity(0)
            .rotate(to_cloud.get_angle() + PI)
        )
        rtn_ax.shift(cloud.get_left() - rtn_ax.c2p(0, 0))

        phase_vt = VT(0)
        sig_x1 = VT(0)
        A = VT(1)
        f = 2
        pw = 0.4 / f
        sig = always_redraw(
            lambda: ax.plot(
                lambda t: ~A * np.sin(2 * PI * 3 * f * t),
                x_range=[max(0, ~sig_x1 - pw), min(1, ~sig_x1), 1 / 200],
                use_smoothing=False,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                color=TX_COLOR,
            ).set_z_index(-2)
        )
        rtn = always_redraw(
            lambda: rtn_ax.plot(
                lambda t: -~A * np.sin(2 * PI * 3 * f * t + ~phase_vt * PI),
                x_range=[max(0, (~sig_x1 - 1) - pw), min(1, (~sig_x1 - 1)), 1 / 200],
                use_smoothing=False,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
                color=RX_COLOR,
            ).set_z_index(-2)
        )
        self.add(sig, rtn)

        radome_background = (
            radar.radome.copy()
            .set_fill(color=BACKGROUND_COLOR, opacity=1)
            .move_to(radar.radome)
            .set_z_index(-1)
        )
        self.add(radome_background)

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(radar.radome))
        self.add(cloud)

        self.wait(0.5)

        def cam_updater(m):
            m.move_to(
                ax.c2p(max(~sig_x1 - pw / 2, 0), 0)
                if ~sig_x1 < 1
                else rtn_ax.c2p(max(~sig_x1 - 1 - pw / 2, 0), 0)
            )

        self.camera.frame.add_updater(cam_updater)

        self.play(AnimationGroup(sig_x1 @ (1.5 + pw / 2), A @ 0.5), run_time=6)

        self.wait(0.5)

        eqn = (
            MathTex(r"s(t) = A(t) \cos{(2 \pi f_c t + \phi(t))}")
            .scale(1.3)
            .next_to(rtn, UP)
        )

        self.play(LaggedStart(*[FadeIn(m) for m in eqn[0]], lag_ratio=0.03))
        self.add(eqn)

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(YELLOW) for m in eqn[0][19:23]],
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[m.animate.set_color(WHITE) for m in eqn[0][19:23][::-1]],
                lag_ratio=0.1,
            )
        )

        self.camera.frame.remove_updater(cam_updater)
        self.wait(0.5)

        both = MathTex(r"A(t)\cos{(\cdot)}, A(t)\sin{(\cdot)}")
        both.next_to(eqn[0][5:], UP, LARGE_BUFF * 2)
        brace = BraceBetweenPoints(both.get_corner(DL), both.get_corner(DR), DOWN)
        brace_arrow = Arrow(eqn[0][5:].get_top(), brace.get_bottom())

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP),
                GrowArrow(brace_arrow),
                FadeIn(brace, shift=UP / 3),
                FadeIn(both[0][:10]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeIn(both[0][10:]))

        self.wait(0.5)

        phi_box = SurroundingRectangle(eqn[0][19:23])
        inner_box = SurroundingRectangle(eqn[0][13:23])

        self.play(Create(phi_box))

        self.wait(0.5)

        self.play(ReplacementTransform(phi_box, inner_box))

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 1.2)))

        self.wait(2)


class NotebookReminder(MovingCameraScene):
    def construct(self):
        math_img = ImageMobject("./static/mpv-shot0005.jpg").scale_to_fit_width(
            fw(self, 0.4)
        )
        math_box = SurroundingRectangle(math_img, buff=0)
        math = Group(math_img, math_box).to_edge(DOWN, LARGE_BUFF)

        next_section = Text("Next section", font=FONT).to_edge(UP, LARGE_BUFF)
        next_section_bez = CubicBezier(
            next_section.get_right() + [0.1, 0, 0],
            next_section.get_right() + [2, 0, 0],
            math.get_top() + [0, 1, 0],
            math.get_top() + [0, 0.1, 0],
        )

        self.play(
            LaggedStart(
                next_section.shift(UP * 5).animate.shift(DOWN * 5),
                Create(next_section_bez),
                math.shift(DOWN * 10).animate.shift(UP * 10),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        notebook_reminder = Text("iq_sampling.ipynb ↓", font=FONT).to_edge(
            UP, MED_LARGE_BUFF
        )

        nb_img_1 = (
            ImageMobject(
                "../../../media/rf-channel-assets/notebook_thumbnails/IQ Sampling Notebook Thumbnail - 2.png"
            )
            .scale_to_fit_width(fw(self, 0.7))
            .to_edge(DOWN, LARGE_BUFF)
        )
        nb_img_1_box = SurroundingRectangle(nb_img_1, buff=0, color=ORANGE)
        nb_img_1_group = Group(nb_img_1, nb_img_1_box)
        nb_img_2 = (
            ImageMobject("./static/2025-07-05-100349_hyprshot.png")
            .scale_to_fit_width(fw(self, 0.7))
            .to_edge(DOWN, LARGE_BUFF)
        )
        nb_img_2_box = SurroundingRectangle(nb_img_2, buff=0, color=ORANGE)
        nb_img_2_group = Group(nb_img_2, nb_img_2_box)
        nb_img_3 = (
            ImageMobject("./static/2025-07-05-100445_hyprshot.png")
            .scale_to_fit_width(fw(self, 0.7))
            .to_edge(DOWN, LARGE_BUFF)
        )
        nb_img_3_box = SurroundingRectangle(nb_img_3, buff=0, color=ORANGE)
        nb_img_3_group = Group(nb_img_3, nb_img_3_box)

        Group(notebook_reminder, nb_img_1_group, nb_img_2_group, nb_img_3_group).shift(
            RIGHT * fw(self)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(RIGHT * fw(self)),
                AnimationGroup(
                    notebook_reminder.shift(UP * 5).animate.shift(DOWN * 5),
                    GrowFromCenter(nb_img_1_group),
                ),
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                nb_img_1_group.animate.shift(LEFT * fw(self)),
                nb_img_2_group.shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                nb_img_2_group.animate.shift(LEFT * fw(self)),
                nb_img_3_group.shift(RIGHT * fw(self)).animate.shift(LEFT * fw(self)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))

        self.wait(2)
