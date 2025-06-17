# iq.py

import sys

from manim import *
from MF_Tools import VT

sys.path.insert(0, "..")
from props import WeatherRadarTower
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True

FONT = "Maple Mono CN"


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


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

        self.next_section(skip_animations=skip_animations(False))
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

        self.play(self.camera.frame.animate.shift(UP * 10))

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
