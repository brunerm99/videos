# fmcw.py

from manim import *
import numpy as np
import math

config.background_color = BLACK


class Title(Scene):
    def construct(self):
        title = Tex("F", "M", "C", "W", " Radar", font_size=DEFAULT_FONT_SIZE * 2)
        f = Text("Frequency", font_size=DEFAULT_FONT_SIZE * 0.8)
        m = Text("modulated", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            f, direction=RIGHT
        )
        c = Text("continuous", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            m, direction=RIGHT
        )
        w = Text("wave", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(c, direction=RIGHT)

        expanded = Tex(
            "Frequency",
            "-",
            "modulated ",
            "continuous ",
            "wave",
            font_size=DEFAULT_FONT_SIZE * 0.8,
        ).shift(DOWN)

        f = expanded[0]
        dash = expanded[1]
        m = expanded[2]
        c = expanded[3]
        w = expanded[4]

        self.play(Write(title))
        # self.wait(1)
        self.play(title.animate.scale(0.7).shift(UP))

        # self.play(Write(f), Write(m), Write(c), Write(w))
        self.play(Write(f), Write(dash), Write(m), Write(c), Write(w))

        f_title = title[0]
        m_title = title[1]
        c_title = title[2]
        w_title = title[3]

        f_p1 = f_title.get_bottom() - [0, 0.1, 0]
        f_p2 = f.get_top() + [0, 0.1, 0]

        f_bezier = CubicBezier(f_p1, f_p1 - [0, 1, 0], f_p2 + [0, 1, 0], f_p2)
        m_p1 = m_title.get_bottom() - [0, 0.1, 0]
        m_p2 = m.get_top()
        m_bezier = CubicBezier(m_p1, m_p1 - [0, 1, 0], m_p2 + [0, 1, 0], m_p2)
        c_p1 = c_title.get_bottom() - [0, 0.1, 0]
        c_p2 = c.get_top()
        c_bezier = CubicBezier(c_p1, c_p1 - [0, 1, 0], c_p2 + [0, 1, 0], c_p2)
        w_p1 = w_title.get_bottom() - [0, 0.1, 0]
        w_p2 = w.get_top()
        w_bezier = CubicBezier(w_p1, w_p1 - [0, 1, 0], w_p2 + [0, 1, 0], w_p2)

        self.play(
            AnimationGroup(
                f.animate.set_color(BLUE),
                m.animate.set_color(RED),
                c.animate.set_color(YELLOW),
                w.animate.set_color(GREEN),
                f_title.animate.set_color(BLUE),
                m_title.animate.set_color(RED),
                c_title.animate.set_color(YELLOW),
                w_title.animate.set_color(GREEN),
                Create(f_bezier),
                Create(m_bezier),
                Create(c_bezier),
                Create(w_bezier),
            )
        )

        self.wait(1)

        cw_box = SurroundingRectangle(VGroup(c, w))

        self.play(
            LaggedStart(
                Uncreate(f_bezier),
                Uncreate(m_bezier),
                Uncreate(c_bezier),
                Uncreate(w_bezier),
                Create(cw_box),
                lag_ratio=0.2,
            )
        )

        self.wait(2)

        self.play(FadeOut(cw_box, f, dash, m, c, w, title))


class Waveform(Scene):
    def construct(self):
        pass


class PulsedRadarTransmission(Scene):
    def construct(self):
        radar = SVGMobject(
            "./figures/weather-radar.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).scale(2)
        cloud = SVGMobject(
            "./figures/clouds.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).shift(RIGHT * 1 + UP * 2)
        tree = (
            SVGMobject(
                "./figures/tree.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=3,
            )
            .shift(RIGHT * 4 + DOWN * 1)
            .scale(1.5)
        )

        self.play(Create(radar))
        self.play(
            LaggedStart(
                radar.animate.shift(LEFT * 5 + DOWN * 2).scale(0.75),
                AnimationGroup(
                    Create(cloud, run_time=2),
                    Create(tree, run_time=2),
                ),
                lag_ratio=0.35,
            )
        )

        def create_propagation(p1, p2):
            line = Line(p1, p2)

            sine = (
                FunctionGraph(
                    lambda t: 0.5 * np.sin(5 * t),
                    x_range=[0, math.sqrt(np.sum(np.power(p2 - p1, 2)))],
                )
                .shift(p1)
                .rotate(line.get_angle(), about_point=p1)
            )

            tracing_dot = Dot(
                sine.get_start(), fill_opacity=1, radius=DEFAULT_DOT_RADIUS / 2
            )
            tracer = TracedPath(
                tracing_dot.get_center,
                dissipating_time=0.5,
                stroke_opacity=[1, 1],
                stroke_width=6,
            )
            return tracer, tracing_dot, sine

        radar_pt = radar.get_corner(UP + RIGHT)
        cloud_pt = cloud.get_corner(LEFT + DOWN)
        rc_tracer, rc_tracing_dot, rc_sine = create_propagation(radar_pt, cloud_pt)

        radar_pt2 = radar.get_corner(RIGHT)
        tree_pt = tree.get_corner(LEFT)
        rt_tracer, rt_tracing_dot, rt_sine = create_propagation(radar_pt2, tree_pt)

        self.add(rc_tracer, rt_tracer)
        self.play(
            MoveAlongPath(rc_tracing_dot, rc_sine),
            MoveAlongPath(rt_tracing_dot, rt_sine),
            rate_func=linear,
            run_time=2,
        )
        # rc_sine.reverse_points()
        # rt_sine.reverse_points()
        rc_sine_flipped = rc_sine.copy().rotate(
            math.pi, about_point=rc_sine.get_center()
        )
        rt_sine_flipped = rt_sine.copy().rotate(
            math.pi, about_point=rt_sine.get_center()
        )
        self.play(
            MoveAlongPath(rc_tracing_dot, rc_sine_flipped),
            MoveAlongPath(rt_tracing_dot, rt_sine_flipped),
            rate_func=linear,
            run_time=2,
        )
        self.remove(rc_tracing_dot, rt_tracing_dot)

        self.wait(10)


class TransmissionTest(Scene):
    def construct(self):
        p1 = np.array([-3, -1, 0])
        p2 = np.array([4, 2, 0])
        dot1 = Dot(p1)
        dot2 = Dot(p2)

        line = Line(p1, p2)

        sine = (
            FunctionGraph(
                lambda t: np.sin(5 * t),
                x_range=[0, math.sqrt(np.sum(np.power(p2 - p1, 2)))],
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
        )

        tracing_dot = Dot(sine.get_start())
        tracer = TracedPath(
            tracing_dot.get_center,
            dissipating_time=0.5,
            stroke_opacity=[1, 1],
            stroke_width=6,
        )

        self.add(dot1, dot2, line, tracing_dot, tracer, sine)

        self.wait(0.5)

        self.play(sine.animate.rotate(math.pi / 2, about_point=sine.get_center()))

        # self.play(MoveAlongPath(tracing_dot, sine), rate_func=linear, run_time=2)
        # sine.reverse_points()
        # self.play(MoveAlongPath(tracing_dot, sine), rate_func=linear, run_time=2)
        self.wait(10)
