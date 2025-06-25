# iq.py

import sys

from manim import *
from MF_Tools import VT
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False

FONT = "Maple Mono CN"


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


# TODO: Something needs redone
class Intro(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar = WeatherRadarTower()
        radar.vgroup.scale(0.6).to_corner(DL, LARGE_BUFF * 1.5)

        self.play(radar.get_animation())

        self.wait(0.5)

        cloud = (
            SVGMobject("../props/static/clouds.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(1.2)
            .to_edge(RIGHT, LARGE_BUFF * 1.5)
            .shift(UP)
        )

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
                cloud.shift(RIGHT * 10).animate.shift(LEFT * 10),
                AnimationGroup(sig_x1 @ (1.5 + pw / 2), A @ 0.5),
                lag_ratio=0.5,
            ),
            run_time=4,
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rtn_ax.save_state()
        self.camera.frame.save_state()
        A.save_state()
        self.play(
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
        mag = (
            MathTex(r"\left| s(t) \right|")
            .scale(1.5)
            .next_to(mag_line, LEFT, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                Create(mag_line_d),
                Create(mag_line),
                Create(mag_line_u),
                LaggedStart(*[FadeIn(m) for m in mag[0]], lag_ratio=0.1),
                lag_ratio=0.2,
            )
        )

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
            lambda: MathTex(f"\\angle s(t) = {~phase_vt:.2f} \\pi")
            .scale(1.5)
            .next_to(phase_line, UP, MED_LARGE_BUFF)
        )

        cloud.shift(RIGHT * 3)

        self.play(
            Create(phase_line),
            Create(phase_line_l),
            Create(phase_line_r),
            GrowFromCenter(phase_label),
            self.camera.frame.animate.scale(1 / 0.8),
        )

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

        self.play(
            phase_vt @ 1.2,
            Transform(phase_line, phase_line_2),
            Transform(phase_line_l, phase_line_l_2),
            Transform(phase_line_r, phase_line_r_2),
            run_time=3,
        )

        self.wait(0.5)

        is_target = Text("?", font=FONT)
        movement = Text("movement", font=FONT)
        v_target = MathTex(r"v_{\text{target}}")
        pm_v = MathTex(r"\pm v")
        r_ll_1 = MathTex(r"\Delta R \ll 1")
        l1_group = (
            Group(is_target, movement)
            .arrange(RIGHT, LARGE_BUFF)
            .next_to(phase_label[0][1], UP, LARGE_BUFF)
        )
        Group(v_target, pm_v, r_ll_1).arrange(RIGHT, LARGE_BUFF).next_to(
            movement, UP, LARGE_BUFF
        )

        is_target_bez = CubicBezier(
            phase_label[0][1].get_top() + [0, 0.2, 0],
            phase_label[0][1].get_top() + [0, 1, 0],
            is_target.get_bottom() + [0, -1, 0],
            is_target.get_bottom() + [0, -0.1, 0],
        )
        movement_bez = CubicBezier(
            phase_label[0][1].get_top() + [0, 0.2, 0],
            phase_label[0][1].get_top() + [0, 1, 0],
            movement.get_bottom() + [0, -1, 0],
            movement.get_bottom() + [0, -0.1, 0],
        )

        v_target_bez = CubicBezier(
            movement.get_top() + [0, 0.1, 0],
            movement.get_top() + [0, 1, 0],
            v_target.get_bottom() + [0, -1, 0],
            v_target.get_bottom() + [0, -0.1, 0],
        )
        pm_v_bez = CubicBezier(
            movement.get_top() + [0, 0.1, 0],
            movement.get_top() + [0, 0.5, 0],
            pm_v.get_bottom() + [0, -0.5, 0],
            pm_v.get_bottom() + [0, -0.1, 0],
        )
        r_ll_1_bez = CubicBezier(
            movement.get_top() + [0, 0.1, 0],
            movement.get_top() + [0, 1, 0],
            r_ll_1.get_bottom() + [0, -1, 0],
            r_ll_1.get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * 2),
                Create(is_target_bez),
                Write(is_target),
                Create(movement_bez),
                Write(movement),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(v_target_bez),
                GrowFromCenter(v_target),
                Create(pm_v_bez),
                GrowFromCenter(pm_v),
                Create(r_ll_1_bez),
                GrowFromCenter(r_ll_1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    rcs_bez,
                    pm_v_bez,
                    range_bez,
                    r_ll_1_bez,
                    v_target_bez,
                    v_target,
                    movement,
                    movement_bez,
                    range_label,
                    rcs_label,
                    phase_label,
                    is_target_bez,
                    is_target,
                    mag_line,
                    mag_line_d,
                    mag_line_u,
                    phase_line,
                    phase_line_l,
                    phase_line_r,
                    mag,
                    pm_v,
                    r_ll_1,
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
            x_range=[1 / num_samples / 2, 1 - 1 / num_samples / 2],
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
        self.next_section(skip_animations=skip_animations(True))

        sample_rects1 = rx_ax.get_riemann_rectangles(
            rx_plot,
            input_sample_type="center",
            x_range=[0, 1],
            dx=1 / (num_samples),
            color=BLUE,
            show_signed_area=False,
            stroke_color=BLACK,
            fill_opacity=0.7,
        )

        self.play(
            rx_plot.animate.set_stroke(opacity=0.2),
            LaggedStart(
                *[
                    AnimationGroup(FadeIn(sr), FadeOut(s, dot))
                    for s, sr, dot in zip(samples, sample_rects1, dots)
                ],
                lag_ratio=0.1,
            ),
            run_time=2,
        )

        self.wait(0.5)

        phase_qmark = (
            Tex("$\\phi = $ ?").scale(1.5).next_to(rx_box, DOWN, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(rx, rx_box, phase_qmark).height * 1.2
                ).move_to(Group(rx, rx_box, phase_qmark)),
                LaggedStart(*[FadeIn(m) for m in phase_qmark[0]], lag_ratio=0.15),
                lag_ratio=0.25,
            )
        )

        self.wait(0.5)

        self.play(
            FadeOut(rx_box, rx, sample_rects1, v_labels, *v_backgrounds),
            rx_plot.animate.set_stroke(opacity=1),
            phase_qmark.animate.shift(DOWN * 6),
            self.camera.frame.animate.scale(0.9).shift(UP),
        )

        self.wait(0.5)

        cos = MathTex(r"\cos{(\theta)}").next_to(rx_ax, UP, LARGE_BUFF)
        cos_bez_l = CubicBezier(
            cos.get_bottom() + [0, -0.1, 0],
            cos.get_bottom() + [0, -1, 0],
            rx_ax.get_corner(UL) + [0, 1, 0],
            rx_ax.get_corner(UL) + [0, 0.1, 0],
        )
        cos_bez_r = CubicBezier(
            cos.get_bottom() + [0, -0.1, 0],
            cos.get_bottom() + [0, -1, 0],
            rx_ax.get_corner(UR) + [0, 1, 0],
            rx_ax.get_corner(UR) + [0, 0.1, 0],
        )

        self.remove(rx_plot)
        rx_f = VT(3)
        rx_x0 = VT(0)
        rx_x1 = VT(2 * PI)
        rx_plot = always_redraw(
            lambda: rx_ax.plot(
                lambda t: np.cos(~rx_f * t),
                x_range=[~rx_x0, ~rx_x1, 1 / 200],
                color=RX_COLOR,
            )
        )
        self.add(rx_plot)

        self.play(
            LaggedStart(*[FadeIn(m) for m in cos[0]], lag_ratio=0.1),
            rx_f @ 1,
            Create(cos_bez_l),
            Create(cos_bez_r),
        )

        self.wait(0.5)

        sample = Line(
            rx_ax.c2p(30 * DEGREES, 0),
            rx_ax.input_to_graph_point(30 * DEGREES, rx_plot),
            color=BLUE,
        )
        dot = Dot(sample.get_end(), color=BLUE)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(cos_bez_l),
                    Uncreate(cos_bez_r),
                ),
                Create(sample),
                Create(dot),
                lag_ratio=0.3,
            ),
        )

        self.wait(0.5)

        cos_w_inp = MathTex(r"\cos{\left(30^{\circ}\right)} \approx 0.87").next_to(
            rx_ax, UP, LARGE_BUFF
        )
        cos_w_inp.shift(cos[0][4].get_center() - cos_w_inp[0][4:7].get_center())

        val_bez = CubicBezier(
            dot.get_center() + [0, 0.1, 0],
            dot.get_center() + [0, 1, 0],
            cos_w_inp[0][4:7].get_bottom() + [0, -1, 0],
            cos_w_inp[0][4:7].get_bottom() + [0, -0.1, 0],
        )

        self.next_section(skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(cos[0][:4], cos_w_inp[0][:4]),
                    ReplacementTransform(cos[0][5], cos_w_inp[0][7]),
                ),
                Create(val_bez),
                ReplacementTransform(cos[0][4], cos_w_inp[0][4:7]),
                LaggedStart(*[FadeIn(m) for m in cos_w_inp[0][8:]], lag_ratio=0.1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        sample_390 = Line(
            rx_ax.c2p(390 * DEGREES, 0),
            rx_ax.input_to_graph_point(390 * DEGREES, rx_plot),
            color=BLUE,
        )
        dot_390 = Dot(sample_390.get_end(), color=BLUE)

        cos_w_inp_390 = MathTex(r"\cos{\left(390^{\circ}\right)} \approx 0.87").next_to(
            cos_w_inp, RIGHT, MED_LARGE_BUFF
        )

        val_bez_390 = CubicBezier(
            dot_390.get_center() + [0, 0.1, 0],
            dot_390.get_center() + [0, 1, 0],
            cos_w_inp_390[0][4:8].get_bottom() + [0, -1, 0],
            cos_w_inp_390[0][4:8].get_bottom() + [0, -0.1, 0],
        )

        rx_ax_2 = Axes(
            x_range=[0, 3 * PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=to_cloud.get_length() * 1.5,
            y_length=radar.vgroup.height,
        )
        rx_ax_2.shift(rx_ax.c2p(0, 0) - rx_ax_2.c2p(0, 0))

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.1).shift(RIGHT),
                TransformFromCopy(rx_ax, rx_ax_2),
                rx_x1 @ (3 * PI),
                Create(sample_390),
                Create(dot_390),
                Create(val_bez_390),
                LaggedStart(*[FadeIn(m) for m in cos_w_inp_390[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        rx_ax_3 = Axes(
            x_range=[-2 * PI, 3 * PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=to_cloud.get_length() * 2.5,
            y_length=radar.vgroup.height,
        )
        rx_ax_3.shift(rx_ax.c2p(0, 0) - rx_ax_3.c2p(0, 0))

        sample_n330 = Line(
            rx_ax.c2p(-330 * DEGREES, 0),
            rx_ax.input_to_graph_point(-330 * DEGREES, rx_plot),
            color=BLUE,
        )
        dot_n330 = Dot(sample_n330.get_end(), color=BLUE)

        cos_w_inp_n330 = MathTex(
            r"\cos{\left(-330^{\circ}\right)} \approx 0.87"
        ).next_to(cos_w_inp, LEFT, 2 * LARGE_BUFF)

        val_bez_n330 = CubicBezier(
            dot_n330.get_center() + [0, 0.1, 0],
            dot_n330.get_center() + [0, 1, 0],
            cos_w_inp_n330[0][4:8].get_bottom() + [0, -1, 0],
            cos_w_inp_n330[0][4:8].get_bottom() + [0, -0.1, 0],
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.2).move_to(rx_ax_3).shift(UP),
                TransformFromCopy(rx_ax, rx_ax_3),
                rx_x0 @ (-2 * PI),
                Create(sample_n330),
                Create(dot_n330),
                Create(val_bez_n330),
                LaggedStart(*[FadeIn(m) for m in cos_w_inp_n330[0]], lag_ratio=0.1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            self.camera.frame.animate.restore(),
            FadeOut(
                rx_ax_2,
                rx_ax_3,
                val_bez_390,
                val_bez_n330,
                cos_w_inp_390,
                cos_w_inp_n330,
                dot_390,
                dot_n330,
                sample_n330,
                sample_390,
            ),
            rx_x0 @ 0,
            rx_x1 @ (2 * PI),
            Uncreate(sample),
            Uncreate(dot),
            Uncreate(val_bez),
            cos_w_inp.animate.set_x(self.camera.frame.copy().restore().get_x()),
        )

        self.wait(0.5)

        qmark = Text("?", font=FONT, color=YELLOW).next_to(
            cos_w_inp[0][4:7], UP, SMALL_BUFF
        )
        self.play(Write(qmark))
        # self.play(self.camera.frame.animate.shift(UP * 10))

        self.wait(0.5)

        cos = (
            MathTex(r"\cos{(\theta)}")
            .scale(1.5)
            .move_to(self.camera.frame)
            .shift(UP * fh(self, 1))
        )

        self.play(
            Transform(cos_w_inp, cos), self.camera.frame.animate.shift(UP * fh(self))
        )

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
