# pulse_compression.py

import sys

from manim import *
from MF_Tools import VT
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower, get_blocks
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False

FONT = "Maple Mono CN"

BLOCKS = get_blocks()


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


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

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(xmax @ 0.5)

        self.wait(0.5)

        pw_line = Line(ax.c2p(~xmax - pw, 1.2), ax.c2p(~xmax, 1.2))
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
                xmax_t1 @ (pw / 2),
                lag_ratio=0.4,
            ),
        )

        self.wait(0.5)

        target_dist = abs(
            (ax.p2c(target1.get_left()[0]) - ax.p2c(target2.get_left()[0]))[0]
        )

        self.play(
            xmax @ 1.5,
            xmax_t1 @ (0.5 + pw / 2),
            xmax_t2 @ (0.5 + pw / 2 - target_dist),
            run_time=3,
        )

        self.wait(0.5)

        target1_pw_line = Line(
            target1_ax.c2p(~xmax_t1, -1),
            target1_ax.c2p(~xmax_t1 - pw, -1),
        )
        target1_pw_line_l = Line(
            target1_pw_line.get_start() + UP / 6, target1_pw_line.get_start() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())
        target1_pw_line_r = Line(
            target1_pw_line.get_end() + UP / 6, target1_pw_line.get_end() + DOWN / 6
        ).rotate(target1_pw_line.get_angle())

        target2_pw_line = Line(
            target2_ax.c2p(~xmax_t2, 1),
            target2_ax.c2p(~xmax_t2 - pw, 1),
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

        self.next_section(skip_animations=skip_animations(False))

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

        self.wait(2)
