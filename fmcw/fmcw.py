# fmcw.py

from typing import List, Callable
from manim import *
import numpy as np
import math

config.background_color = BLACK


"""Helpers"""


def create_block_diagram(
    blocks: List[Mobject],
    right_to_left: bool = False,
    row_len: int = 3,
    connector_size: int = 1,
):
    bd = VGroup(blocks[0])

    for idx, (curr, prev) in enumerate(zip(blocks[1:], blocks[:-1]), start=1):
        if idx % row_len == 0:
            right_to_left = not right_to_left
            curr.next_to(prev, buff=connector_size, direction=DOWN)
            bd.add(Line(prev.get_bottom(), curr.get_top()))
            bd.add(curr)
            continue

        if right_to_left:
            curr.next_to(prev, buff=connector_size, direction=LEFT)
            bd.add(Line(prev.get_left(), curr.get_right()))
            bd.add(curr)
        else:
            curr.next_to(prev, buff=connector_size, direction=RIGHT)
            bd.add(Line(prev.get_right(), curr.get_left()))
            bd.add(curr)

    bd.move_to(ORIGIN).scale(0.5)

    return bd


"""Scenes"""


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
        """Images"""
        radar = SVGMobject(
            "./figures/weather-radar.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).scale(2)
        radar_zoomed = (
            SVGMobject(
                "./figures/weather-radar.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            .shift(DOWN * 40)
            .scale(100)
        )
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

        """Block diagram"""
        adc, window_function, bp_filter, range_norm, product_calc, computer = [
            SVGMobject(
                f"./figures/{name}.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            for name in [
                "adc",
                "window_function",
                "filter",
                "range_norm",
                "product_calc",
                "computer",
            ]
        ]
        (
            adc_2,
            window_function_2,
            bp_filter_2,
            range_norm_2,
            product_calc_2,
            computer_2,
        ) = [
            SVGMobject(
                f"./figures/{name}.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            for name in [
                "adc",
                "window_function",
                "filter",
                "range_norm",
                "product_calc",
                "computer",
            ]
        ]

        tri = Triangle(color=WHITE).scale(0.75).rotate(PI)
        tri_line = Line(tri.get_bottom() + DOWN / 2, tri.get_top())
        tri_line_h = Line(tri_line.get_bottom(), tri_line.get_bottom() + RIGHT)
        line_dot = Dot(tri_line.get_bottom(), radius=DEFAULT_DOT_RADIUS / 2)
        antenna = VGroup(tri, tri_line, tri_line_h)
        antenna_w_inv = VGroup(
            antenna,
            line_dot,
            antenna.copy().rotate(PI, about_point=antenna.get_bottom()).set_opacity(0),
        )

        tri_2 = Triangle(color=WHITE).scale(0.75).rotate(PI)
        tri_line_2 = Line(tri_2.get_bottom() + DOWN / 2, tri_2.get_top())
        tri_line_h_2 = Line(tri_line_2.get_bottom(), tri_line_2.get_bottom() + RIGHT)
        line_dot_2 = Dot(tri_line_2.get_bottom(), radius=DEFAULT_DOT_RADIUS / 2)
        antenna_2 = VGroup(tri_2, tri_line_2, tri_line_h_2)
        antenna_w_inv_2 = VGroup(
            antenna_2,
            line_dot_2,
            antenna_2.copy()
            .rotate(PI, about_point=antenna_2.get_bottom())
            .set_opacity(0),
        )

        blocks = [
            antenna_w_inv,
            Triangle(color=WHITE).rotate(PI / 6),
            bp_filter.copy(),
            VGroup(adc, Tex("ADC").shift(RIGHT / 3)),
            # window_function,
            # bp_filter,
            # range_norm,
            # product_calc,
            computer,
        ]
        blocks_2 = [
            antenna_w_inv_2,
            Triangle(color=WHITE).rotate(PI / 6),
            bp_filter_2,
            VGroup(adc_2, Tex("ADC").shift(RIGHT / 3)),
            computer_2,
        ]
        processing_block_diagram = create_block_diagram(
            blocks, right_to_left=False, row_len=5, connector_size=1
        )
        processing_block_diagram_big = create_block_diagram(
            blocks_2, right_to_left=False, row_len=5, connector_size=4
        )

        """Animation start"""

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

        def create_propagation(
            p1, p2, waveform: Callable = lambda t: 0.5 * np.sin(5 * t), color=WHITE
        ):
            line = Line(p1, p2)
            dist = math.sqrt(np.sum(np.power(p2 - p1, 2)))

            sine = (
                FunctionGraph(
                    waveform, x_range=[0, math.sqrt(np.sum(np.power(p2 - p1, 2)))]
                )
                .shift(p1)
                .rotate(line.get_angle(), about_point=p1)
            )
            sine_flipped = (
                FunctionGraph(
                    lambda t: 0.5 * np.sin(5 * t), x_range=[0, dist], color=color
                )
                .shift(p1)
                .rotate(line.get_angle(), about_point=p1)
                .rotate(math.pi)
            )

            tracing_dot = Dot(
                sine.get_start(), fill_opacity=0, radius=DEFAULT_DOT_RADIUS / 2
            )
            tracer = TracedPath(
                tracing_dot.get_center,
                dissipating_time=1,
                stroke_opacity=[1, 1],
                stroke_width=6,
            )
            return tracer, tracing_dot, sine, sine_flipped, dist

        radar_pt = radar.get_corner(RIGHT) + UP * 3 / 2 + RIGHT / 3
        cloud_pt = cloud.get_corner(LEFT + DOWN)
        rc_tracer, rc_tracing_dot, rc_sine, rc_sine_flipped, rc_dist = (
            create_propagation(radar_pt, cloud_pt)
        )

        radar_pt2 = radar.get_corner(RIGHT) + UP * 3 / 2 + RIGHT / 3
        tree_pt = tree.get_corner(LEFT)
        rt_tracer, rt_tracing_dot, rt_sine, rt_sine_flipped, rt_dist = (
            create_propagation(radar_pt2, tree_pt)
        )

        self.add(rc_tracer, rt_tracer)

        rc_time = 1.5
        rt_time = rc_time * (rt_dist / rc_dist)
        self.play(
            LaggedStart(
                Succession(
                    MoveAlongPath(rt_tracing_dot, rt_sine, rate_func=linear),
                    MoveAlongPath(rt_tracing_dot, rt_sine_flipped, rate_func=linear),
                    run_time=rt_time,
                ),
                Succession(
                    MoveAlongPath(rc_tracing_dot, rc_sine, rate_func=linear),
                    MoveAlongPath(rc_tracing_dot, rc_sine_flipped, rate_func=linear),
                    run_time=rc_time,
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)
        self.remove(rc_tracer, rt_tracer)

        # Move into radome
        radome_center = radar.get_top() + DOWN * 2 / 3

        self.play(*[mobj.animate.shift(-radome_center) for mobj in self.mobjects])
        self.play(
            FadeOut(cloud, tree, run_time=0.1),
            Transform(radar, radar_zoomed),
            GrowFromCenter(processing_block_diagram),
        )
        self.remove(radar_zoomed)

        self.play(Transform(processing_block_diagram, processing_block_diagram_big))

        waveform_rx_amplitude = 0.2
        waveform_rx_gain = 2
        f_mult = 2
        waveform_rx = lambda t: waveform_rx_amplitude * (
            np.sin(5 * f_mult * t) + np.sin(2 * f_mult * t) + np.sin(8 * f_mult * t)
        )
        waveform_rx_amplified = (
            lambda t: waveform_rx_gain
            * waveform_rx_amplitude
            * (np.sin(5 * f_mult * t) + np.sin(2 * f_mult * t) + np.sin(8 * f_mult * t))
        )
        waveform_rx_filtered = (
            lambda t: waveform_rx_amplitude
            * waveform_rx_gain
            * 3
            * np.sin(5 * f_mult * t)
        )

        (antenna_tracer, antenna_tracing_dot, antenna_sine, *_) = create_propagation(
            antenna_2.get_top() + RIGHT * 9 + UP * 5,
            antenna_2.get_top() + UP / 2,
            waveform=waveform_rx,
            color=BLUE,
        )
        self.add(antenna_tracer)

        # Wave antenna to amp
        antenna_to_amp = processing_block_diagram_big[1]
        (antenna_to_amp_tracer, antenna_to_amp_tracing_dot, antenna_to_amp_sine, *_) = (
            create_propagation(
                antenna_to_amp.get_left(),
                antenna_to_amp.get_right(),
                waveform=waveform_rx,
                color=BLUE,
            )
        )
        self.add(antenna_to_amp_tracer)

        # Amp to filter
        amp_to_filter = processing_block_diagram_big[3]
        (amp_to_filter_tracer, amp_to_filter_tracing_dot, amp_to_filter_sine, *_) = (
            create_propagation(
                amp_to_filter.get_left(),
                amp_to_filter.get_right(),
                waveform=waveform_rx_amplified,
                color=BLUE,
            )
        )
        self.add(amp_to_filter_tracer)

        # Filter to ADC
        filter_to_adc = processing_block_diagram_big[5]
        (filter_to_adc_tracer, filter_to_adc_tracing_dot, filter_to_adc_sine, *_) = (
            create_propagation(
                filter_to_adc.get_left(),
                filter_to_adc.get_right(),
                waveform=waveform_rx_filtered,
                color=BLUE,
            )
        )
        self.add(filter_to_adc_tracer)

        self.play(
            LaggedStart(
                MoveAlongPath(
                    antenna_tracing_dot, antenna_sine, rate_func=linear, run_time=2
                ),
                MoveAlongPath(
                    antenna_to_amp_tracing_dot,
                    antenna_to_amp_sine,
                    rate_func=linear,
                    run_time=2,
                ),
                MoveAlongPath(
                    amp_to_filter_tracing_dot,
                    amp_to_filter_sine,
                    rate_func=linear,
                    run_time=2,
                ),
                MoveAlongPath(
                    filter_to_adc_tracing_dot,
                    filter_to_adc_sine,
                    rate_func=linear,
                    run_time=2,
                ),
                lag_ratio=0.9,
            )
        )

        self.wait(3)
        self.remove(antenna_tracer, antenna_to_amp_tracer)


"""Testing Animations"""


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


class RadarMoveTest(Scene):
    def construct(self):
        radar = (
            SVGMobject(
                "./figures/weather-radar.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            .scale(2)
            .shift(LEFT * 5 + DOWN * 2)
            .scale(0.75)
        )
        radar_zoomed = (
            SVGMobject(
                "./figures/weather-radar.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            .shift(DOWN * 40)
            .scale(100)
        )

        radar.shift(-(radar.get_top() + DOWN * 2 / 3))
        self.add(radar)
        self.play(Transform(radar, radar_zoomed))
        self.wait(1)


class ProcBD(Scene):
    def construct(self):
        ant_line = Line(ORIGIN, UP * 2)
        ant_tri = (
            Triangle(color=WHITE)
            .scale(0.75)
            .rotate(math.pi)
            .move_to(ant_line.get_top(), aligned_edge=UP)
        )
        antenna = VGroup(ant_line, ant_tri).shift(RIGHT * 2)

        amp = Triangle(color=WHITE).rotate(-PI / 6)
        amp_to_ant = Line(amp.get_right(), antenna.get_bottom())

        amp_ant = VGroup(antenna, amp, amp_to_ant).shift(RIGHT * 4)

        mixer = VGroup(
            Circle(color=WHITE),
            Line(LEFT / 2, RIGHT / 2).rotate(-PI / 4),
            Line(LEFT / 2, RIGHT / 2).rotate(PI / 4),
        )
        mixer_to_amp = Line(mixer.get_right(), amp.get_left())

        lo = VGroup(
            Circle(color=WHITE),
            FunctionGraph(
                lambda x: 0.25 * np.sin(-8 * x), x_range=[-0.5, 0.5], color=WHITE
            ),
        ).shift(UP * 3)
        lo_to_mixer = Line(lo.get_bottom(), mixer.get_top())

        lo_mixer_ant = VGroup(lo, lo_to_mixer, amp_ant, mixer, mixer_to_amp)

        # self.add(amp_ant, mixer_to_amp, mixer, lo)
        self.add(lo_mixer_ant.scale(0.6))


class BD(Scene):
    def construct(self):
        adc, window_function, bp_filter, range_norm, product_calc, computer = [
            SVGMobject(
                f"./figures/{name}.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.01,
            )
            for name in [
                "adc",
                "window_function",
                "filter",
                "range_norm",
                "product_calc",
                "computer",
            ]
        ]

        # Antenna
        tri = Triangle(color=WHITE).scale(0.75).rotate(PI)
        tri_line = Line(tri.get_bottom() + DOWN / 2, tri.get_top())
        tri_line_h = Line(tri_line.get_bottom(), tri_line.get_bottom() + RIGHT)
        line_dot = Dot(tri_line.get_bottom(), radius=DEFAULT_DOT_RADIUS / 2)
        antenna = VGroup(tri, tri_line, tri_line_h)
        antenna_w_inv = VGroup(
            antenna,
            line_dot,
            antenna.copy().rotate(PI, about_point=antenna.get_bottom()).set_opacity(0),
        )

        blocks = [
            antenna_w_inv,
            Triangle(color=WHITE).rotate(PI / 6),
            bp_filter.copy(),
            VGroup(adc, Tex("ADC").shift(RIGHT / 3)),
            window_function,
            bp_filter,
            range_norm,
            product_calc,
            computer,
        ][::1]

        def create_block_diagram(
            blocks: List[Mobject],
            right_to_left: bool = False,
            row_len: int = 3,
            connector_size: int = 1,
        ):
            bd = VGroup(blocks[0])

            for idx, (curr, prev) in enumerate(zip(blocks[1:], blocks[:-1]), start=1):
                if idx % row_len == 0:
                    right_to_left = not right_to_left
                    curr.next_to(prev, buff=connector_size, direction=DOWN)
                    bd.add(Line(prev.get_bottom(), curr.get_top()))
                    bd.add(curr)
                    continue

                if right_to_left:
                    curr.next_to(prev, buff=connector_size, direction=LEFT)
                    bd.add(Line(prev.get_left(), curr.get_right()))
                    bd.add(curr)
                else:
                    curr.next_to(prev, buff=connector_size, direction=RIGHT)
                    bd.add(Line(prev.get_right(), curr.get_left()))
                    bd.add(curr)

            bd.move_to(ORIGIN).scale(0.5)

            return bd

        bd = create_block_diagram(
            blocks, row_len=5, right_to_left=False, connector_size=1
        )
        # bd2 = create_block_diagram(
        #     blocks, row_len=4, right_to_left=False, connector_size=3
        # )

        # TODO: Create BlockDiagram type and override the Create animation
        # self.play(
        #     Succession(
        #         *[
        #             Create(block) if type(block) == Line else GrowFromCenter(block)
        #             for block in bd
        #         ]
        #     ),
        #     run_time=4,
        # )

        self.add(bd)

        # self.play(bp_filter.animate.shift(UP))
        # self.wait()
        # self.play(Transform(bd, bd2))
        # self.wait()
