# fmcw.py

import math
from typing import Callable, List

import cv2
import numpy as np
from manim import *
from scipy import signal, constants

# config.background_color = BLACK
BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR
# config.background_color = ManimColor.from_hex("#253f4b")
# config.background_color = ManimColor.from_hex("#183340")


TX_COLOR = BLUE
RX_COLOR = RED


"""Helpers"""


def pretty_num(n: float) -> str:
    nstr, dec = str(f"{n:.2f}").split(".")

    nstr_fmt = ",".join(
        [nstr[::-1][start : start + 3][::-1] for start in range(0, len(nstr), 3)][::-1]
    )

    return f"{nstr_fmt}.{dec}"


class WeatherRadarTower:
    def __init__(self, **kwargs):
        # super.__init__(**kwargs)

        width_scale = 2
        LINE_STYLE = dict(
            color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * width_scale * 2
        )

        leg = Line(ORIGIN, UP * 3, **LINE_STYLE)
        self.left_leg = leg.copy().shift(LEFT)
        self.right_leg = leg.copy().shift(RIGHT)
        self.middle_leg = leg.copy().shift(DOWN / 1.5)

        # self.left_start_cap = Dot(
        #     radius=DEFAULT_SMALL_DOT_RADIUS * width_scale
        # ).move_to(self.left_leg.get_start())
        # self.right_start_cap = Dot(
        #     radius=DEFAULT_SMALL_DOT_RADIUS * width_scale
        # ).move_to(self.right_leg.get_start())
        # self.middle_start_cap = Dot(
        #     radius=DEFAULT_SMALL_DOT_RADIUS * width_scale
        # ).move_to(self.middle_leg.get_start())
        # self.left_end_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        #     self.left_leg.get_end()
        # )
        # self.right_end_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        #     self.right_leg.get_end()
        # )
        # self.middle_end_cap = Dot(
        #     radius=DEFAULT_SMALL_DOT_RADIUS * width_scale
        # ).move_to(self.middle_leg.get_end())

        self.conn1_y_shift = DOWN / 2
        self.conn1 = Line(
            self.middle_leg.get_center() + self.conn1_y_shift,
            self.right_leg.get_center() + self.conn1_y_shift,
            **LINE_STYLE,
        )
        self.conn2 = Line(
            self.middle_leg.get_center() + self.conn1_y_shift,
            self.left_leg.get_center() + self.conn1_y_shift,
            **LINE_STYLE,
        )
        self.conn3 = self.conn1.copy().shift(-self.conn1_y_shift * 2)
        self.conn4 = self.conn2.copy().shift(-self.conn1_y_shift * 2)
        # conn5 = conn1.copy().shift(-conn1_y_shift * 4)
        # conn6 = conn2.copy().shift(-conn1_y_shift * 2)

        self.radome = Circle(radius=1.08, **LINE_STYLE).next_to(
            self.middle_leg,
            direction=UP,
            buff=0,
        )

        self.vgroup = VGroup(
            self.left_leg,
            self.right_leg,
            self.middle_leg,
            # self.left_start_cap,
            # self.middle_start_cap,
            # self.right_start_cap,
            # self.left_end_cap,
            # self.middle_end_cap,
            # self.right_end_cap,
            self.conn1,
            self.conn2,
            self.conn3,
            self.conn4,
            self.radome,
        ).move_to(ORIGIN)

    def get_animation(self):
        return LaggedStart(
            AnimationGroup(
                Create(self.left_leg),
                Create(self.middle_leg),
                Create(self.right_leg),
                # Create(self.left_start_cap),
                # Create(self.middle_start_cap),
                # Create(self.right_start_cap),
            ),
            AnimationGroup(
                # Create(self.left_end_cap),
                # Create(self.middle_end_cap),
                # Create(self.right_end_cap),
                Create(self.conn1),
                Create(self.conn2),
                Create(self.conn3),
                Create(self.conn4),
            ),
            Create(self.radome),
            lag_ratio=0.75,
        )

    # @override_animation(Create)
    # def _create_override(self, **kwargs):


class FMCWRadarCartoon:
    def __init__(self, text="FMCW"):
        self.rect = Rectangle(height=3, width=1.5)
        self.label = Tex(text).rotate(PI / 2)
        self.line_1 = (
            Line(ORIGIN, RIGHT / 2)
            .next_to(self.rect, direction=RIGHT, buff=0)
            .shift(UP)
        )
        self.line_2 = (
            Line(ORIGIN, RIGHT / 2)
            .next_to(self.rect, direction=RIGHT, buff=0)
            .shift(DOWN)
        )
        self.antenna_tx = (
            AnnularSector(inner_radius=1, outer_radius=1.2, angle=PI)
            .rotate(PI / 2)
            .scale(0.5)
            .next_to(self.line_1, direction=RIGHT, buff=0)
        )
        self.antenna_rx = (
            AnnularSector(inner_radius=1, outer_radius=1.2, angle=PI)
            .rotate(PI / 2)
            .scale(0.5)
            .next_to(self.line_2, direction=RIGHT, buff=0)
        )

        self.vgroup = VGroup(
            self.rect,
            self.label,
            self.line_1,
            self.line_2,
            self.antenna_tx,
            self.antenna_rx,
        ).move_to(ORIGIN)

    def get_animation(self):
        return Succession(
            GrowFromCenter(VGroup(self.rect, self.label)),
            LaggedStart(
                AnimationGroup(Create(self.line_1), Create(self.line_2)),
                AnimationGroup(
                    GrowFromCenter(self.antenna_tx), GrowFromCenter(self.antenna_rx)
                ),
                lag_ratio=0.6,
            ),
        )


def get_weather_radar_tower():
    width_scale = 2
    LINE_STYLE = dict(color=WHITE, stroke_width=DEFAULT_STROKE_WIDTH * width_scale * 2)
    leg = Line(ORIGIN, UP * 3, **LINE_STYLE)
    left_leg = leg.copy().shift(LEFT)
    right_leg = leg.copy().shift(RIGHT)
    middle_leg = leg.copy().shift(DOWN / 1.5)

    left_start_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        left_leg.get_start()
    )
    right_start_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        right_leg.get_start()
    )
    middle_start_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        middle_leg.get_start()
    )
    left_end_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        left_leg.get_end()
    )
    right_end_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        right_leg.get_end()
    )
    middle_end_cap = Dot(radius=DEFAULT_SMALL_DOT_RADIUS * width_scale).move_to(
        middle_leg.get_end()
    )

    conn1_y_shift = DOWN / 2
    conn1 = Line(
        middle_leg.get_center() + conn1_y_shift,
        right_leg.get_center() + conn1_y_shift,
        **LINE_STYLE,
    )
    conn2 = Line(
        middle_leg.get_center() + conn1_y_shift,
        left_leg.get_center() + conn1_y_shift,
        **LINE_STYLE,
    )
    conn3 = conn1.copy().shift(-conn1_y_shift * 2)
    conn4 = conn2.copy().shift(-conn1_y_shift * 2)
    # conn5 = conn1.copy().shift(-conn1_y_shift * 4)
    # conn6 = conn2.copy().shift(-conn1_y_shift * 2)

    radome = Circle(radius=1.05, **LINE_STYLE).next_to(
        middle_leg,
        direction=UP,
        buff=0,
    )

    return VGroup(
        left_leg,
        right_leg,
        middle_leg,
        left_start_cap,
        middle_start_cap,
        right_start_cap,
        left_end_cap,
        middle_end_cap,
        right_end_cap,
        conn1,
        conn2,
        conn3,
        conn4,
        radome,
    ).move_to(ORIGIN)


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


def create_title(
    s: str,
    underline_buff=MED_SMALL_BUFF,
    match_underline_width_to_text: bool = False,
):
    t = Tex(s)
    t.to_edge(UP)
    underline_width = config["frame_width"] - 2
    underline = Line(LEFT, RIGHT)
    underline.next_to(t, DOWN, buff=underline_buff)
    if match_underline_width_to_text:
        underline.match_width(t)
    else:
        underline.width = underline_width

    return VGroup(
        t,
        Line(underline.get_midpoint(), underline.get_start()),
        Line(underline.get_midpoint(), underline.get_end()),
    )


def get_title_animation(title_group, run_time=2):
    return AnimationGroup(
        FadeIn(title_group[0]),
        Create(title_group[1]),
        Create(title_group[2]),
        run_time=2,
    )


class ConveyerBelt:
    def __init__(
        self,
        time_tracker: ValueTracker,
        rotation_mult_tracker: ValueTracker,
        circ_color=RED,
        nboxes=5,
        base_shift=[0, 0, 0],
        subsequent_shift=LEFT * 3,
        radar_beam_scan_speed=1,
    ):
        self.circ_color = circ_color
        self.lcirc = Circle(radius=0.5, color=self.circ_color)
        self.lcirc_line1 = Line(
            self.lcirc.get_top(), self.lcirc.get_bottom(), color=self.circ_color
        ).rotate(PI / 4)
        self.lcirc_line2 = self.lcirc_line1.copy().rotate(PI / 2)
        self.lcirc_vg = VGroup(self.lcirc, self.lcirc_line1, self.lcirc_line2)
        self.rcirc = Circle(radius=0.5, color=self.circ_color)
        self.rcirc_line1 = Line(
            self.rcirc.get_top(), self.rcirc.get_bottom(), color=self.circ_color
        ).rotate(PI / 4)
        self.rcirc_line2 = self.rcirc_line1.copy().rotate(PI / 2)
        self.rcirc_vg = VGroup(self.rcirc, self.rcirc_line1, self.rcirc_line2)

        self.circ_vg = VGroup(self.lcirc_vg, self.rcirc_vg).arrange(
            direction=RIGHT, buff=3 * LARGE_BUFF, center=True
        )

        self.entry = (
            Rectangle(height=4, width=2, fill_color=BACKGROUND_COLOR, fill_opacity=1)
            .next_to(self.lcirc, direction=LEFT, buff=0)
            .shift(RIGHT / 2 + UP)
        )
        self.exit = (
            Rectangle(height=4, width=2, fill_color=BACKGROUND_COLOR, fill_opacity=1)
            .next_to(self.rcirc, direction=RIGHT, buff=0)
            .shift(LEFT / 2 + UP)
        )
        self.entry_invis = (
            Rectangle(
                height=4,
                width=50,
                stroke_color=BACKGROUND_COLOR,
                # stroke_color=BLUE,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            )
            .next_to(self.lcirc, direction=LEFT, buff=0)
            .shift(RIGHT / 2 + UP)
        )
        self.exit_invis = (
            Rectangle(
                height=4,
                width=50,
                stroke_color=BACKGROUND_COLOR,
                # stroke_color=BLUE,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            )
            .next_to(self.rcirc, direction=RIGHT, buff=0)
            .shift(LEFT / 2 + UP)
        )

        def circle_updater(m: Mobject):
            m.rotate(-PI * 0.75 * time_tracker.get_value())

        self.lcirc_vg.add_updater(circle_updater)
        self.rcirc_vg.add_updater(circle_updater)

        def get_belt_updater(direction=RIGHT):
            def updater(m: Mobject):
                m.shift(direction * time_tracker.get_value())

            return updater

        def get_box_updater(shift=[0, 0, 0]):
            def updater(m: Mobject):
                m.next_to(self.belt_top, direction=UP, buff=SMALL_BUFF).shift(shift)

            return updater

        self.belt_top = DashedLine(
            self.entry.get_center() + LEFT * 40,
            self.entry.get_center() + RIGHT * 40,
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 5,
            dashed_ratio=0.6,
        ).next_to(self.circ_vg, direction=UP, buff=SMALL_BUFF)
        self.belt_bot = DashedLine(
            self.exit.get_center() + LEFT * 40,
            self.exit.get_center() + RIGHT * 40,
            color=YELLOW,
            dash_length=DEFAULT_DASH_LENGTH * 5,
            dashed_ratio=0.6,
        ).next_to(self.circ_vg, direction=DOWN, buff=SMALL_BUFF)

        self.nboxes = nboxes
        self.boxes = []
        self.base_shift = base_shift
        self.subsequent_shift = subsequent_shift
        for idx in range(self.nboxes):
            shift = self.base_shift + self.subsequent_shift * idx
            box = (
                Square(side_length=1, color=GRAY_BROWN)
                .next_to(self.belt_top, direction=UP, buff=SMALL_BUFF)
                .shift(shift)
            )
            box_updater = get_box_updater(shift=shift)
            box.add_updater(box_updater)
            self.boxes.append(box)

        self.belt_top_updater = get_belt_updater(RIGHT)
        self.belt_bot_updater = get_belt_updater(LEFT)

        radar_post = Line(
            self.entry.get_corner(UR) + LEFT / 3,
            self.entry.get_corner(UR) + UP * 2 + RIGHT / 2,
        )
        self.radar_ant_dome = AnnularSector(
            inner_radius=0.8, outer_radius=0.6, angle=PI
        )
        self.radar_ant_line = Line(
            self.radar_ant_dome.get_top(),
            self.radar_ant_dome.get_top() + DOWN,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        self.radar_ant_dot = Circle(
            radius=0.1, color=WHITE, fill_color=BACKGROUND_COLOR, fill_opacity=1
        ).next_to(self.radar_ant_line, direction=DOWN, buff=0)
        self.radar_ant = VGroup(
            self.radar_ant_dome, self.radar_ant_line, self.radar_ant_dot
        )

        self.radar_ant.rotate(PI * 0.15).next_to(
            self.entry, direction=UR, buff=MED_LARGE_BUFF
        )
        self.radar_line_1 = Line(ORIGIN, RIGHT).next_to(
            self.radar_ant, direction=LEFT, buff=0
        )
        self.radar_line_2 = Line(ORIGIN, DOWN).next_to(
            self.radar_line_1, direction=DL, buff=0
        )
        self.radar_line_1_dot = Dot(self.radar_line_1.get_end())
        self.radar_line_2_dot_1 = Dot(self.radar_line_2.get_start())
        self.radar_line_2_dot_2 = Dot(self.radar_line_2.get_end())

        self.radar_beam_l = Line(
            self.radar_ant_dot.get_center(),
            self.radar_ant_dot.get_center() + DOWN * 1.5,
            color=BLUE,
        )
        self.radar_beam_r = self.radar_beam_l.copy().rotate(
            30 * DEGREES, about_point=self.radar_ant_dot.get_center()
        )

        start_angle = self.radar_beam_l.get_angle()
        self.scan_width = 60

        def radar_beam_updater(m: Mobject):
            rotation = 30 * DEGREES * radar_beam_scan_speed * time_tracker.get_value()
            if (
                self.radar_beam_l.get_angle()
                > start_angle + (self.scan_width / 2) * DEGREES
            ):
                rotation_mult_tracker.set_value(-1)
            elif (
                self.radar_beam_l.get_angle()
                < start_angle - (self.scan_width / 2) * DEGREES
            ):
                rotation_mult_tracker.set_value(1)
            m.rotate(
                rotation_mult_tracker.get_value() * rotation,
                about_point=self.radar_ant_dot.get_center(),
            )

        self.radar_beam_updater = radar_beam_updater
        self.radar_beams = VGroup(self.radar_beam_l, self.radar_beam_r)

        self.vgroup = VGroup(
            self.lcirc_vg,
            self.rcirc_vg,
            self.belt_top,
            self.belt_bot,
            *self.boxes,
            self.entry_invis,
            self.exit_invis,
            self.entry,
            self.exit,
            self.radar_line_1,
            self.radar_line_2,
            self.radar_ant,
            self.radar_line_1_dot,
            self.radar_line_2_dot_1,
            self.radar_line_2_dot_2,
        )
        self.vgroup_w_beams = VGroup(self.vgroup, self.radar_beams)

    def add_updaters(self):
        self.belt_top.add_updater(self.belt_top_updater)
        self.belt_bot.add_updater(self.belt_bot_updater)
        self.radar_beams.add_updater(self.radar_beam_updater)

    def remove_beam_updaters(self):
        self.radar_beams.remove_updater(self.radar_beam_updater)

    def remove_updaters(self):
        self.belt_top.remove_updater(self.belt_top_updater)
        self.belt_bot.remove_updater(self.belt_bot_updater)
        self.radar_beams.remove_updater(self.radar_beam_updater)


"""Scenes"""


class RadarTypesIntro(Scene):
    def construct(self):
        t_tracker = ValueTracker(0)
        rotation_mult_tracker = ValueTracker(1)
        radar_beam_scan_speed = 2

        small_radars = Tex("Small Radars").scale(1.5)
        fmcw_radar = Tex("FMCW Radar", font_size=DEFAULT_FONT_SIZE * 2)

        car = (
            SVGMobject(
                "./figures/car.svg",
                stroke_color=WHITE,
                color=WHITE,
                fill_color=WHITE,
                opacity=1,
                stroke_width=0.005,
            )
            .to_corner(UR, buff=MED_LARGE_BUFF)
            .flip()
        )

        car_radar_beam_start = car.get_left() + LEFT / 4
        car_radar_beam_l = Line(
            car_radar_beam_start, car_radar_beam_start + LEFT * 2, color=BLUE
        )
        car_radar_beam_r = car_radar_beam_l.copy().rotate(
            30 * DEGREES, about_point=car_radar_beam_start
        )
        car_radar_beams = VGroup(car_radar_beam_l, car_radar_beam_r).rotate(
            -15 * DEGREES, about_point=car_radar_beam_start
        )

        def car_radar_beam_updater(m: Mobject):
            rotation = 30 * DEGREES * radar_beam_scan_speed * t_tracker.get_value()
            m.rotate(
                rotation_mult_tracker.get_value() * rotation,
                about_point=car_radar_beam_start,
            )

        car_radar_beam_l.add_updater(car_radar_beam_updater)
        car_radar_beam_r.add_updater(car_radar_beam_updater)

        cb = ConveyerBelt(
            t_tracker,
            rotation_mult_tracker,
            circ_color=RED,
            nboxes=5,
            base_shift=ORIGIN,
            subsequent_shift=LEFT * 3,
            radar_beam_scan_speed=radar_beam_scan_speed,
        )
        cb_centered = ConveyerBelt(
            t_tracker,
            rotation_mult_tracker,
            circ_color=RED,
            nboxes=5,
            base_shift=ORIGIN,
            subsequent_shift=LEFT * 3,
            radar_beam_scan_speed=radar_beam_scan_speed,
        )
        cb.add_updaters()

        cb.vgroup_w_beams.scale(0.6).shift(DOWN * 4 + LEFT * 4)

        cb_centered.vgroup.scale(0.6).shift(DOWN * 4 + LEFT * 4).move_to(ORIGIN).scale(
            1.5
        )
        pulsed_label = (
            Tex("Pulsed Radar?")
            .next_to(cb_centered.radar_ant_dome, direction=UR, buff=MED_LARGE_BUFF)
            .shift(RIGHT)
        )
        pulsed_arrow = Arrow(
            pulsed_label.get_left(), cb_centered.radar_ant_dome.get_right()
        )
        downside_1 = Tex(
            "- High peak power", color=RED, font_size=DEFAULT_FONT_SIZE / 2
        ).next_to(pulsed_label, direction=DOWN, buff=SMALL_BUFF)
        downside_2 = Tex(
            "- Large minimum range", color=RED, font_size=DEFAULT_FONT_SIZE / 2
        ).next_to(downside_1, direction=DOWN, buff=SMALL_BUFF)
        downside_3 = Tex(
            "- Harder to get fine range resolution",
            color=RED,
            font_size=DEFAULT_FONT_SIZE / 2,
        ).next_to(downside_2, direction=DOWN, buff=SMALL_BUFF)

        cb_p1 = small_radars.get_left() - [0.1, 0, 0]
        cb_p2 = cb.exit.get_corner(UL) + [0, 0.1, 0]

        cb_bezier = CubicBezier(
            cb_p1,
            cb_p1 + [-1, 0, 0],
            cb_p2 + [0, 1, 0],
            cb_p2,
        )

        car_p1 = small_radars.get_right() + [0.1, 0, 0]
        car_p2 = car.get_bottom() - [0, 0.1, 0]

        car_bezier = CubicBezier(
            car_p1,
            car_p1 + [2, -0.5, 0],
            car_p2 + [0, -1, 0],
            car_p2,
        )

        self.play(GrowFromCenter(small_radars))

        self.wait(0.5)

        self.play(
            Create(cb_bezier),
            Create(car_bezier),
            GrowFromCenter(cb.vgroup_w_beams),
            GrowFromCenter(car),
            Create(car_radar_beam_l),
            Create(car_radar_beam_r),
        )

        self.play(
            t_tracker.animate(rate_func=sigmoid, run_time=4).increment_value(0.04)
        )
        cb.remove_beam_updaters()
        car_radar_beam_l.remove_updater(car_radar_beam_updater)
        car_radar_beam_r.remove_updater(car_radar_beam_updater)

        self.play(
            FadeOut(
                cb.radar_beams,
                small_radars,
                car,
                cb_bezier,
                car_bezier,
                car_radar_beam_l,
                car_radar_beam_r,
            ),
        )
        self.play(
            t_tracker.animate(rate_func=sigmoid, run_time=3).increment_value(0.04),
            cb.vgroup.animate.move_to(ORIGIN).scale(1.5),
            Succession(
                AnimationGroup(Create(pulsed_label), Create(pulsed_arrow)),
                FadeIn(downside_1, shift=UP / 2),
                FadeIn(downside_2, shift=UP / 2),
                FadeIn(downside_3, shift=UP / 2),
                run_time=2,
            ),
        )
        cb.remove_updaters()
        self.play(
            FadeOut(*self.mobjects, shift=DOWN * 7), FadeIn(fmcw_radar, shift=DOWN * 2)
        )

        self.wait(2)


class Title(Scene):
    def construct(self):
        but_what = Tex("but", "... ", "what ", "is ", "an")

        title = Tex("F", "M", "C", "W", " Radar", font_size=DEFAULT_FONT_SIZE * 2)
        f = Text("Frequency", font_size=DEFAULT_FONT_SIZE * 0.8)
        m = Text("modulated", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            f, direction=RIGHT
        )
        c = Text("continuous", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            m, direction=RIGHT
        )
        w = Text("wave", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(c, direction=RIGHT)

        expanded_scale = 1.2
        expanded = Tex(
            "Frequency",
            "-",
            "modulated ",
            "continuous ",
            "wave",
            font_size=DEFAULT_FONT_SIZE * expanded_scale,
        ).shift(DOWN * 1.5)

        f = expanded[0]
        dash = expanded[1]
        m = expanded[2]
        c = expanded[3]
        w = expanded[4]

        expanded_cap = Tex(
            "Frequency",
            "-",
            "Modulated ",
            "Continuous ",
            "Wave",
            font_size=DEFAULT_FONT_SIZE * expanded_scale,
        ).shift(DOWN * 1.5)

        f_cap = expanded_cap[0]
        dash_cap = expanded_cap[1]
        m_cap = expanded_cap[2]
        c_cap = expanded_cap[3]
        w_cap = expanded_cap[4]

        part_1_scale = 1.8
        part_1 = Tex(
            "Part ",
            "1",
            ": ",
            font_size=DEFAULT_FONT_SIZE * expanded_scale * part_1_scale,
        )

        but_what.next_to(
            title, direction=UP, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 3
        )

        self.add(title)
        self.wait(1)
        self.add(but_what[0])
        self.wait(0.3)
        self.play(AddTextLetterByLetter(but_what[1]), run_time=1)
        self.wait(0.3)
        self.add(but_what[2])
        self.wait(0.3)
        self.add(but_what[3])
        self.wait(0.3)
        self.add(but_what[4])

        # self.play(Write(title))
        self.wait(0.5)
        self.play(
            LaggedStart(FadeOut(but_what), title.animate.shift(UP * 1.5), lag_ratio=0.2)
        )

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

        self.wait(0.5)

        fm_box = SurroundingRectangle(VGroup(f, m))
        cw_box = SurroundingRectangle(VGroup(c, w))

        self.play(
            LaggedStart(
                Create(fm_box),
                Uncreate(f_bezier),
                Uncreate(m_bezier),
                Uncreate(c_bezier),
                Uncreate(w_bezier),
                lag_ratio=0.1,
            )
        )
        self.play(Transform(fm_box, cw_box))

        self.wait(2)

        self.play(Uncreate(cw_box))

        cw_copy = (
            VGroup(c_cap.copy(), w_cap.copy()).scale(part_1_scale).set_color(WHITE)
        )
        part_1.next_to(cw_copy, direction=LEFT)
        part_1_group = VGroup(part_1, cw_copy).move_to(ORIGIN)

        self.play(Uncreate(fm_box))
        self.play(
            LaggedStart(
                FadeOut(f, dash, m, title),
                AnimationGroup(
                    Write(part_1_group[0]),
                    Transform(VGroup(c, w), cw_copy),
                    run_time=2,
                ),
                lag_ratio=0.2,
            )
        )

        # self.wait()

        # tcw = VGroup(part_1[3], cw_copy)
        # cw_strike = Line(tcw.get_left(), tcw.get_right())
        # radar_basics = Tex("Radar Basics").scale(0.8 * 2).next_to(tcw, direction=DOWN)
        # part_1_strike = Line(part_1[1].get_left(), part_1[1].get_right())
        # part_0 = Tex("0").scale(0.8 * 2).next_to(part_1[1], direction=DOWN)
        # self.play(
        #     LaggedStart(
        #         AnimationGroup(Create(part_1_strike), Create(cw_strike)),
        #         AnimationGroup(Write(part_0), Write(radar_basics)),
        #         lag_ratio=0.4,
        #     )
        # )

        self.wait(1)

        self.play(FadeOut(*self.mobjects, shift=UP * 5))

        self.wait(2)

        # self.play(FadeOut(cw_box, f, dash, m, c, w, title))


class Waveform(Scene):
    def construct(self):
        bw_tracker = ValueTracker(1.0)
        f_tracker = ValueTracker(2.0)
        f_0_tracker = ValueTracker(6.0)

        f_1 = f_0_tracker.get_value() + bw_tracker.get_value()
        x_1 = 1

        ax_scale = 0.6
        ax = Axes(
            x_range=[0, x_1, 0.2],
            y_range=[
                f_0_tracker.get_value() - bw_tracker.get_value() / 2.0,
                f_1 + bw_tracker.get_value(),
                0.5,
            ],
            # y_range=[-1, 1],
            tips=False,
            axis_config={"include_numbers": False},
        ).scale(ax_scale)
        ax_with_numbers = Axes(
            x_range=[0, x_1, 0.2],
            y_range=[
                f_0_tracker.get_value() - bw_tracker.get_value() / 2.0,
                f_1 + bw_tracker.get_value(),
                0.5,
            ],
            # y_range=[-1, 1],
            tips=False,
            axis_config={"include_numbers": True},
        ).scale(ax_scale)

        labels = ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        tx = always_redraw(
            lambda: ax.plot(
                lambda t: (signal.sawtooth(2 * PI * f_tracker.get_value() * t) + 1)
                / 2
                * bw_tracker.get_value()
                + f_0_tracker.get_value(),
                use_smoothing=False,
                x_range=[0, x_1 - x_1 / 1000, x_1 / 1000],
                color=TX_COLOR,
            )
        )

        self.add(ax, labels, tx)
        self.wait(1)
        self.play(Transform(ax, ax_with_numbers))
        self.wait(1)
        self.play(Transform(labels[1], ax.get_y_axis_label(Tex("frequency"))))
        self.play(Transform(labels[0], ax.get_x_axis_label(Tex("time"))))
        self.wait(0.5)
        self.play(Transform(labels[1], ax.get_y_axis_label(Tex("$f$"))))
        self.play(Transform(labels[0], ax.get_x_axis_label(Tex("$t$"))))

        plot = VGroup(ax, labels, tx)
        self.play(plot.animate.shift(RIGHT * 2))

        """ f0 Variation """
        f_0_arrow = always_redraw(
            lambda: Arrow(
                ax.c2p(0, f_0_tracker.get_value()) + 2 * LEFT,
                ax.c2p(0, f_0_tracker.get_value()),
            ).shift(LEFT)
        )
        f_0_label = always_redraw(
            lambda: Tex(f"$f_0$=\\\\{f_0_tracker.get_value():.02f}MHz").next_to(
                f_0_arrow, direction=LEFT
            )
        )
        f_0_variation = VGroup(f_0_arrow, f_0_label)

        self.play(FadeIn(f_0_variation, shift=RIGHT))
        self.play(
            f_0_tracker.animate.set_value(f_0_tracker.get_value() + 1), run_time=0.8
        )
        self.play(
            f_0_tracker.animate.set_value(f_0_tracker.get_value() - 1), run_time=0.8
        )

        """ BW Variation """
        bw_brace = always_redraw(
            lambda: BraceLabel(
                Line(
                    ax.c2p(0, f_0_tracker.get_value()),
                    ax.c2p(0, f_0_tracker.get_value() + bw_tracker.get_value()),
                ),
                f"BW=\\\\{bw_tracker.get_value():.02f}MHz",  # TODO: Mess with spacing b/c values not showing
                brace_direction=LEFT,
                label_constructor=Tex,
            ).shift(LEFT)
        )

        # BW in MHz
        def r_res(bw):
            return constants.speed_of_light / (2 * bw_tracker.get_value() * 1e6)

        def r_res_updater(m: Mobject):
            m.become(
                Tex(
                    r"$R_{res}=\frac{c}{2 \cdot BW}=$",
                    f"{r_res(bw_tracker.get_value()):.2f}m",
                ).to_corner(UR)
            )

        range_resolution_eqn = Tex(
            r"$R_{res}=\frac{c}{2 \cdot BW}=$",
            f"{r_res(bw_tracker.get_value()):.2f}m",
        ).to_corner(UR)

        self.play(FadeOut(f_0_variation, shift=LEFT))
        self.play(
            FadeIn(bw_brace, shift=RIGHT), FadeIn(range_resolution_eqn, shift=DOWN)
        )
        range_resolution_eqn.add_updater(r_res_updater)

        self.play(
            bw_tracker.animate.set_value(bw_tracker.get_value() + 1), run_time=1.4
        )
        self.play(
            bw_tracker.animate.set_value(bw_tracker.get_value() - 1), run_time=1.4
        )

        """ Period Variation """
        range_resolution_eqn.remove_updater(r_res_updater)
        self.play(
            FadeOut(bw_brace, shift=LEFT),
            FadeOut(range_resolution_eqn, shift=UP),
            plot.animate.shift(LEFT * 2 + UP / 2),
        )

        # Period in ms
        def v_max():
            wavelength = constants.speed_of_light / (f_0_tracker.get_value() * 1e6)
            v_max_float = wavelength / (4 * (1e-3 * x_1 / f_tracker.get_value()))
            v_max_str = pretty_num(v_max_float)
            return v_max_str

        def v_max_updater(m: Mobject):
            m.become(
                Tex(
                    r"$v_{max} = \frac{\lambda}{4 \cdot T} = \ $",
                    f"{v_max()} ",
                    r"$\frac{m}{s}$",
                ).to_corner(UR)
            )

        v_max_eqn = Tex(
            r"$v_{max} = \frac{\lambda}{4 \cdot T} = \ $",
            f"{v_max()} ",
            r"$\frac{m}{s}$",
        ).to_corner(UR)

        period_brace = always_redraw(
            lambda: BraceLabel(
                Line(
                    ax.c2p(0, f_0_tracker.get_value()),
                    ax.c2p(x_1 / f_tracker.get_value(), f_0_tracker.get_value()),
                ),
                f"T={x_1/f_tracker.get_value():.02f}s",
                # f"T=",
                brace_direction=DOWN,
                label_constructor=Tex,
            ).shift(DOWN)
        )

        self.play(FadeIn(period_brace, shift=UP), FadeIn(v_max_eqn, shift=DOWN))
        v_max_eqn.add_updater(v_max_updater)

        self.play(
            f_tracker.animate.set_value(f_tracker.get_value() - 0.5), run_time=1.4
        )
        self.play(
            f_tracker.animate.set_value(f_tracker.get_value() + 0.5), run_time=1.4
        )

        """ Back to origin """
        next_ax_scale = 0.7

        v_max_eqn.remove_updater(v_max_updater)
        self.play(FadeOut(period_brace, shift=DOWN), FadeOut(v_max_eqn, shift=UP))
        self.play(plot.animate.move_to(ORIGIN).scale(next_ax_scale * ax_scale))

        self.wait(2)


# TODO: Combine with Waveform - slo tho
class TxAndRx(Scene):
    def construct(self):
        cw_radar = FMCWRadarCartoon()
        cw_radar.vgroup.scale(0.4).to_corner(UL, buff=MED_LARGE_BUFF)

        cloud = SVGMobject(
            "./figures/clouds.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).to_corner(UR, buff=MED_SMALL_BUFF)

        radar_to_cloud = Arrow(
            cw_radar.antenna_tx.get_right(), cloud.get_edge_center(LEFT), color=TX_COLOR
        )
        cloud_to_radar = Arrow(
            cloud.get_edge_center(LEFT) + DOWN / 2,
            cw_radar.antenna_rx.get_right(),
            color=RX_COLOR,
        )

        def get_t_shift_dist(t0: float, t1: float, graph, axes: Axes) -> np.ndarray:
            return (
                axes.input_to_graph_point(t1, graph)
                - axes.input_to_graph_point(t0, graph)
            ) * RIGHT

        f_units = "MHz"
        bw = 1.0
        f = 2.0
        f_0 = 6.0

        f_1 = f_0 + bw
        x_1 = 1

        t_shift = 0.1
        f_rx_tracker = ValueTracker(0.0)
        f_tx_tracker = ValueTracker(f_rx_tracker.get_value() + t_shift)

        def update_t(t: float):
            return [
                f_rx_tracker.animate.set_value(f_rx_tracker.get_value() + t),
                f_tx_tracker.animate.set_value(f_tx_tracker.get_value() + t),
            ]

        ax_scale = 0.6
        ax = Axes(
            x_range=[0, x_1, 0.2],
            y_range=[
                f_0 - bw / 2.0,
                f_1 + bw,
                0.5,
            ],
            tips=False,
            axis_config={"include_numbers": True},
        ).scale(ax_scale)

        func = lambda t: (signal.sawtooth(2 * PI * f * t) + 1) / 2 * bw + f_0
        tx = always_redraw(
            lambda: ax.plot(
                func,
                use_smoothing=False,
                x_range=[0, x_1, x_1 / 1000],
                color=TX_COLOR,
            )
        )
        rx_line_legend = Line(ORIGIN, RIGHT, color=RX_COLOR).next_to(
            tx, direction=UR, buff=MED_LARGE_BUFF
        )
        rx_legend = Tex("Rx", color=RX_COLOR).next_to(
            rx_line_legend, direction=LEFT, buff=SMALL_BUFF
        )
        tx_line_legend = Line(ORIGIN, RIGHT, color=TX_COLOR).next_to(
            rx_line_legend, direction=UP, buff=MED_LARGE_BUFF
        )
        tx_legend = Tex("Tx", color=TX_COLOR).next_to(
            tx_line_legend, direction=LEFT, buff=SMALL_BUFF
        )

        self.add(ax, tx)
        self.play(FadeIn(tx_line_legend, tx_legend, shift=LEFT))

        self.play(
            VGroup(ax, tx, tx_line_legend, tx_legend).animate.to_edge(
                DOWN, buff=LARGE_BUFF
            ),
            AnimationGroup(Create(cloud), cw_radar.get_animation()),
        )

        rx_line_legend.next_to(tx_line_legend, direction=DOWN, buff=MED_LARGE_BUFF)
        rx_legend.next_to(rx_line_legend, direction=LEFT, buff=SMALL_BUFF)

        rx = tx.copy().set_color(RX_COLOR)

        self.wait(1)

        self.play(Create(radar_to_cloud))

        t_shift_dist = get_t_shift_dist(t0=0, t1=t_shift, graph=tx, axes=ax)
        self.play(
            LaggedStart(
                Create(cloud_to_radar),
                AnimationGroup(
                    rx.animate.shift(t_shift_dist),
                    FadeIn(VGroup(rx_legend, rx_line_legend), shift=LEFT),
                ),
                lag_ratio=0.4,
            )
        )

        shift_start = ax.input_to_graph_point(0, tx)
        t_shift_line = Line(shift_start, shift_start + t_shift_dist)
        t_shift_brace = Brace(t_shift_line, buff=LARGE_BUFF)
        t_shift_brace_label = Tex("$t$").next_to(
            t_shift_brace, direction=DOWN, buff=SMALL_BUFF
        )

        self.play(Create(t_shift_brace), Create(t_shift_brace_label))

        f_tx_dot = Dot(ax.input_to_graph_point(f_tx_tracker.get_value(), tx))
        f_rx_dot = Dot(ax.input_to_graph_point(f_rx_tracker.get_value(), rx)).shift(
            t_shift_dist
        )

        def update_tx_dot(m: Mobject):
            m.move_to(ax.input_to_graph_point(f_tx_tracker.get_value(), tx))

        def update_rx_dot(m: Mobject):
            m.move_to(ax.input_to_graph_point(f_rx_tracker.get_value(), rx)).shift(
                t_shift_dist
            )

        f_tx_dot.add_updater(update_tx_dot)
        f_rx_dot.add_updater(update_rx_dot)

        def update_arrow(m: Mobject):
            m.next_to(
                ax.input_to_graph_point(f_tx_tracker.get_value(), tx), direction=UP
            )

        f_arrow = Arrow(ORIGIN, DOWN).next_to(
            ax.input_to_graph_point(f_tx_tracker.get_value(), tx), direction=UP
        )
        f_arrow.add_updater(update_arrow)

        f_rx_label = (
            Tex(
                r"$f_{rx}=\ $",
                f"{func(f_rx_tracker.get_value()):.02f}{f_units}",
                color=RX_COLOR,
            )
            .scale(0.8)
            .next_to(tx, direction=UP, buff=MED_LARGE_BUFF * 1.3)
        )
        f_tx_label = (
            Tex(
                r"$f_{tx}=\ $",
                f"{func(f_tx_tracker.get_value()):.02f}{f_units}",
                color=TX_COLOR,
            )
            .scale(0.8)
            .next_to(f_rx_label, direction=UP)
        )

        def update_tx_freq(m: Mobject):
            m.become(
                Tex(
                    r"$f_{tx}=\ $",
                    f"{func(f_tx_tracker.get_value()):.02f}{f_units}",
                    color=TX_COLOR,
                ).next_to(f_rx_label, direction=UP)
            )

        def update_rx_freq(m: Mobject):
            m.become(
                Tex(
                    r"$f_{rx}=\ $",
                    f"{func(f_rx_tracker.get_value()):.02f}{f_units}",
                    color=RX_COLOR,
                ).next_to(tx, direction=UP, buff=MED_LARGE_BUFF * 1.3)
            )

        f_rx_label.add_updater(update_rx_freq)
        f_tx_label.add_updater(update_tx_freq)

        self.play(
            # Uncreate(tx_label),
            # Uncreate(rx_label),
            Create(f_arrow),
            Create(f_tx_label),
            Create(f_rx_label),
            Create(f_rx_dot),
            Create(f_tx_dot),
        )

        f_inc = 0.35
        self.play(
            f_tx_tracker.animate.increment_value(f_inc),
            f_rx_tracker.animate.increment_value(f_inc),
            run_time=2,
        )

        how_do_we_derive = Tex(
            r"How do we derive range information from\\these two signals available to us?"
        ).to_edge(UP, buff=MED_LARGE_BUFF)

        f_tx_label.remove_updater(update_tx_freq)
        f_rx_label.remove_updater(update_rx_freq)
        f_tx_dot.remove_updater(update_tx_dot)
        f_rx_dot.remove_updater(update_rx_dot)
        f_arrow.remove_updater(update_arrow)

        plot_group = VGroup(
            f_arrow,
            tx,
            rx,
            ax,
            f_tx_label,
            f_rx_label,
            f_rx_dot,
            f_tx_dot,
            t_shift_brace,
            t_shift_brace_label,
            tx_line_legend,
            tx_legend,
            rx_line_legend,
            rx_legend,
        )
        self.play(
            Uncreate(cloud_to_radar),
            Uncreate(radar_to_cloud),
            FadeOut(cw_radar.vgroup, cloud),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                plot_group.animate.to_edge(DOWN, buff=MED_LARGE_BUFF),
                Write(how_do_we_derive),
            )
        )
        self.wait(1)

        self.play(
            Uncreate(how_do_we_derive),
            plot_group.animate.scale(0.6).to_edge(LEFT),
        )

        self.wait(0.5)

        screen_split = Line(DOWN, UP)
        screen_split.height = config["frame_height"] - 1

        self.play(
            Create(Line(screen_split.get_midpoint(), screen_split.get_bottom())),
            Create(Line(screen_split.get_midpoint(), screen_split.get_top())),
        )

        self.wait(0.5)

        right_center = (config["frame_width"] / 4) * RIGHT

        speed_of_light = (
            Tex(r"$c$", r"$\ \approx 3 \cdot 10^{8} \ \frac{m}{s}$")
            .to_edge(UP, buff=LARGE_BUFF)
            .shift(right_center)
        )

        unknown_time = Tex(r"$t$", r"$\ =\ $", "?", r"$\ s$").next_to(
            speed_of_light, direction=DOWN, buff=MED_LARGE_BUFF
        )
        unknown_time[2].set_color(YELLOW)

        distance_traveled = Tex(
            r"Distance traveled", r"$\ =\ $", r"$c$", r"$\ \cdot \ $", r"$t$"
        ).next_to(unknown_time, direction=DOWN, buff=MED_LARGE_BUFF)

        range_eqn = Tex(r"$R$", r"$\ =\ $", r"$\frac{c \ \cdot \  t}{2}$").move_to(
            distance_traveled
        )

        self.play(Create(speed_of_light))

        self.wait(1)

        self.play(Indicate(t_shift_brace_label, scale_factor=2))
        self.play(
            TransformFromCopy(t_shift_brace_label, unknown_time[0]),
            Create(unknown_time[1:]),
        )

        self.wait(1)

        self.play(
            LaggedStart(
                Create(distance_traveled[:2]),
                TransformFromCopy(speed_of_light[0], distance_traveled[2]),
                Create(distance_traveled[3]),
                TransformFromCopy(unknown_time[0], distance_traveled[4]),
                lag_ratio=0.5,
            )
        )

        self.wait(1)

        self.play(Transform(distance_traveled, range_eqn))

        self.wait(2)


class PulsedRadarIntro(Scene):
    def construct(self):
        radar = WeatherRadarTower()

        cap = cv2.VideoCapture("./figures/images/weather_channel.gif")
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        weather_channel_scale = 1.5
        weather_channel = (
            ImageMobject(frame)
            .scale(weather_channel_scale)
            .to_corner(UR, buff=LARGE_BUFF)
        )
        weather_channel_rect = SurroundingRectangle(weather_channel, buff=0)

        """ Weather channel animation """
        self.play(radar.get_animation())
        self.play(radar.vgroup.animate.scale(0.6).to_edge(LEFT, buff=MED_LARGE_BUFF))

        pointer = Arrow(
            radar.radome.get_corner(RIGHT),
            weather_channel.get_corner(LEFT),
        )

        self.play(
            GrowFromCenter(weather_channel),
            Create(weather_channel_rect),
            Create(pointer),
        )

        self.remove(weather_channel)
        while flag:
            flag, frame = cap.read()
            if flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                weather_channel = (
                    ImageMobject(frame)
                    .scale(weather_channel_scale)
                    .to_corner(UR, buff=LARGE_BUFF)
                )
                self.add(weather_channel)
                self.wait(0.12)
                self.remove(weather_channel)

        self.play(FadeOut(weather_channel, weather_channel_rect, pointer))

        self.wait(1)

        """ Pulsed Transmission """

        t_tracker = ValueTracker(0)

        pw = 1.5
        f = 2

        wave_buff = 0.3
        p1 = radar.radome.get_corner(RIGHT) + RIGHT * wave_buff
        p2 = (
            Dot()
            .to_edge(RIGHT, buff=LARGE_BUFF)
            .shift(radar.radome.get_corner(RIGHT) * UP)
            .get_center()
        )
        line_pulsed = Line(p1, p2)

        x_max = math.sqrt(np.sum(np.power(p2 - p1, 2)))
        ref_wave = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t), x_range=[0, x_max]
            )
            .shift(p1)
            .rotate(line_pulsed.get_angle(), about_point=p1)
        )

        rx_flip_pt_tracker = ValueTracker(ref_wave.get_y())

        x_max_tbuff = x_max * 1.2
        tx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[
                    max(
                        min(
                            (t_tracker.get_value() % (2 * x_max_tbuff) - pw),
                            x_max,
                        ),
                        0,
                    ),
                    min((t_tracker.get_value() % (2 * x_max_tbuff)), x_max),
                ],
                color=TX_COLOR,
            )
            .shift(p1)
            .rotate(line_pulsed.get_angle(), about_point=p1)
        )
        rx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t + PI),
                x_range=[
                    max(
                        x_max
                        - max((t_tracker.get_value() % (2 * x_max_tbuff)) - x_max, 0),
                        0,
                    ),
                    max(
                        x_max
                        - max(
                            (t_tracker.get_value() % (2 * x_max_tbuff)) - x_max - pw, 0
                        ),
                        0,
                    ),
                    # x_max,
                ],
                color=RX_COLOR,
            )
            .shift(p1)
            .rotate(line_pulsed.get_angle(), about_point=p1)
        )

        self.add(tx, rx)

        round_trip_time_brace = Brace(Line(p1.copy(), p2.copy()), buff=LARGE_BUFF)
        round_trip_time_brace_label = Tex("Round-trip time").next_to(
            round_trip_time_brace, direction=DOWN, buff=MED_SMALL_BUFF
        )
        range_eqn = MathTex(
            r"\text{Range} = ",
            r"\ \frac{\text{Speed of light} \cdot \text{Round-trip time}}{2}",
        ).to_corner(DR, buff=MED_LARGE_BUFF)

        range_eqn_bez_p1 = round_trip_time_brace_label.get_bottom() + [0, 0, 0]
        range_eqn_bez_p2 = range_eqn[1].get_top() + [2, 0.1, 0]

        range_eqn_bezier = CubicBezier(
            range_eqn_bez_p1,
            range_eqn_bez_p1 + [0, -1, 0],
            range_eqn_bez_p2 + [0, 1, 0],
            range_eqn_bez_p2,
        )

        runs = 5
        lag_ratio = 0.5 * 4 / runs  # Start after 2 runs
        self.play(
            LaggedStart(
                t_tracker.animate(run_time=runs * 2, rate_func=linear).increment_value(
                    x_max_tbuff * runs * 2
                ),
                LaggedStart(
                    FadeIn(
                        round_trip_time_brace, round_trip_time_brace_label, shift=UP
                    ),
                    AnimationGroup(
                        Create(range_eqn_bezier),
                        FadeIn(range_eqn, shift=UP),
                    ),
                    lag_ratio=1.3,
                ),
                lag_ratio=lag_ratio,
            )
        )

        self.play(
            FadeOut(
                round_trip_time_brace,
                round_trip_time_brace_label,
                range_eqn_bezier,
                range_eqn,
            )
        )

        t_tracker.set_value(0.0)

        """ Setup for CW """
        self.play(radar.vgroup.animate.to_corner(UL, buff=MED_LARGE_BUFF))
        line_pulsed.shift(radar.vgroup.get_y() * UP)
        p1 += radar.vgroup.get_y() * UP
        p2 += radar.vgroup.get_y() * UP
        rx_flip_pt_tracker.increment_value(radar.vgroup.get_y())

        cw_radar = FMCWRadarCartoon(text="CW")
        self.play(cw_radar.get_animation())
        self.play(cw_radar.vgroup.animate.scale(0.7).to_corner(DL, buff=MED_LARGE_BUFF))

        """ CW Transmission """
        t_cw_tracker = ValueTracker(0)

        f = 2

        # Propagation
        wave_buff = 0.3
        p1_cw = cw_radar.antenna_tx.get_edge_center(RIGHT) + RIGHT * wave_buff
        p2_cw = (
            Dot()
            .to_edge(RIGHT, buff=LARGE_BUFF)
            .shift(cw_radar.rect.get_edge_center(RIGHT) * UP)
            .get_center()
        )
        line_cw = Line(p1_cw, p2_cw)
        p1_cw_rx = cw_radar.antenna_rx.get_edge_center(RIGHT) + RIGHT * wave_buff
        p2_cw_rx = (
            Dot()
            .to_edge(RIGHT, buff=LARGE_BUFF)
            .shift(cw_radar.rect.get_edge_center(RIGHT) * UP)
            .get_center()
        )
        line_cw_rx = Line(p1_cw_rx, p2_cw_rx)

        x_max = math.sqrt(np.sum(np.power(p2_cw - p1_cw, 2)))
        line_cw_angle = line_cw.get_angle()
        tx_cw = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=np.array([-min(t_cw_tracker.get_value(), x_max), 0])
                - max(t_cw_tracker.get_value() - x_max, 0),
                color=TX_COLOR,
            )
            .shift(p1_cw)
            .rotate(line_cw_angle, about_point=p1_cw)
            .shift(
                t_cw_tracker.get_value()
                * (RIGHT * math.cos(line_cw_angle) + UP * math.sin(line_cw_angle))
            )
        )
        rx_cw = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=np.array(
                    [x_max, x_max + max(t_cw_tracker.get_value() - x_max, 0)]
                )
                + [max(t_cw_tracker.get_value() - 2 * x_max, 0), 0],
                color=RX_COLOR,
            )
            .shift(p1_cw_rx)
            .rotate(line_cw_rx.get_angle(), about_point=p1_cw_rx)
            .shift(
                max(t_cw_tracker.get_value() - x_max, 0)
                * (LEFT * math.cos(line_cw_angle) + UP * math.sin(line_cw_angle))
            )
        )

        self.add(tx_cw, rx_cw)

        runs = 3
        self.play(
            t_tracker.animate.increment_value(x_max_tbuff * runs * 2),
            t_cw_tracker.animate.set_value(x_max * runs * 2),
            run_time=runs * 2,
            rate_func=linear,
        )

        self.play(FadeOut(tx, rx, tx_cw, rx_cw))

        self.play(
            radar.vgroup.animate.move_to(ORIGIN + 4 * LEFT),
            cw_radar.vgroup.animate.move_to(ORIGIN + 4 * RIGHT),
        )

        actually_very_important = Tex(
            "it's", " actually", " very", " important"
        ).to_edge(DOWN, buff=LARGE_BUFF)
        waits = [0.3, 0.5, 0.3, 0]
        for word, wait in zip(actually_very_important, waits):
            self.add(word)
            self.wait(wait)

        self.wait(1)

        cw_benefit = Tex("+ Benefit", color=GREEN).next_to(
            cw_radar.vgroup, direction=UP, buff=LARGE_BUFF
        )
        self.play(FadeIn(cw_benefit, shift=UP), rate_func=rate_functions.ease_in_sine)
        self.play(FadeOut(cw_benefit, shift=UP), rate_func=rate_functions.ease_out_sine)

        self.wait(1)

        self.play(
            FadeOut(cw_radar.vgroup, shift=RIGHT * 3),
            FadeOut(radar.vgroup, shift=LEFT * 3),
            FadeOut(actually_very_important),
        )

        self.wait(2)


class PulsedPowerProblem(Scene):
    def construct(self):
        duty_cycle_tracker = ValueTracker(1.0)

        step = 0.001
        x_max = 6
        ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            # x_length=x_len,
            # y_length=y_len,
        )

        f = 4
        # pulsed_sine = lambda t: np.sin(2 * PI * f * t) * (
        #     (signal.square(2 * PI * t / 2, duty=1) + 1) / 2
        # )
        # sq = lambda t: (signal.square(2 * PI * t / 2, duty=1) + 1) / 2

        pulsed_graph = always_redraw(
            lambda: ax.plot(
                lambda t: np.sin(2 * PI * f * t)
                * (
                    (
                        signal.square(
                            2 * PI * t / 2,
                            duty=duty_cycle_tracker.get_value(),
                        )
                        + 1
                    )
                    / 2
                ),
                x_range=[0, x_max, step],
                use_smoothing=False,
            )
        )
        sq_graph = always_redraw(
            lambda: ax.plot(
                lambda t: (
                    signal.square(
                        2 * PI * t / 2,
                        duty=duty_cycle_tracker.get_value(),
                    )
                    + 1
                )
                / 2,
                x_range=[0, x_max - step, step],
                use_smoothing=False,
                color=YELLOW,
            )
        )

        duty_cycle = Tex(f"Duty cycle: {duty_cycle_tracker.get_value():.02f}%").to_edge(
            DOWN
        )
        self.add(duty_cycle)
        duty_cycle.add_updater(
            lambda m: m.become(
                Tex(f"Duty cycle: {duty_cycle_tracker.get_value():.02f}%").to_edge(DOWN)
            )
        )

        self.play(Create(ax))
        self.play(Create(pulsed_graph), Create(sq_graph))

        # self.play(
        #     sq_graph.animate.become(
        #         ax.plot(
        #             lambda t: (signal.square(2 * PI * t / 2, duty=0.3) + 1) / 2,
        #             x_range=[0, x_max, step],
        #             use_smoothing=False,
        #             color=YELLOW,
        #         )
        #     )
        # )

        self.play(duty_cycle_tracker.animate.increment_value(-0.7), run_time=3)

        self.wait(2)


class PropagationLoss(Scene):
    def construct(self):
        range_tracker = ValueTracker(1.0)

        step = 0.001
        x_min = 1
        x_max = 6
        f = 4
        range_ax = Axes(
            x_range=[x_min - 0.1, x_max, 1],
            y_range=[-0.1, 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
        ).scale(0.8)

        range_labels = range_ax.get_axis_labels(
            Tex("R", font_size=DEFAULT_FONT_SIZE),
            Tex("", font_size=DEFAULT_FONT_SIZE),
        )

        sine_graph = range_ax.plot(
            lambda t: np.sin(2 * PI * f * t) * (np.exp(t) ** 4),
            x_range=[0, range_tracker.get_value(), step],
            use_smoothing=False,
        )
        exp_graph = range_ax.plot(
            lambda t: (np.exp(t) ** 4),
            x_range=[0, range_tracker.get_value(), step],
            use_smoothing=False,
            color=YELLOW,
        )

        range_arrow = Arrow(
            range_ax.c2p(0, 0, 0), range_ax.c2p(range_tracker.get_value(), 0, 0)
        )
        range_arrow_label = Tex("Range").next_to(
            range_arrow, direction=DOWN, buff=SMALL_BUFF
        )

        range_attenuation_relation = Tex(r"Power $\propto \frac{1}{R^2}$").shift(
            RIGHT + 2
        )

        def sine_updater(m: Mobject):
            m.become(
                range_ax.plot(
                    lambda r: (np.sin(2 * PI * f * r) + 1) / 2 / (r**2),
                    x_range=[x_min, range_tracker.get_value(), step],
                    use_smoothing=False,
                )
            )

        def exp_updater(m: Mobject):
            m.become(
                range_ax.plot(
                    lambda r: 1 / (r**2),
                    x_range=[x_min, range_tracker.get_value(), step],
                    use_smoothing=False,
                    color=YELLOW,
                )
            )

        def range_arrow_updater(m: Mobject):
            m.become(
                Arrow(
                    range_ax.c2p(x_min, 0, 0),
                    range_ax.c2p(range_tracker.get_value(), 0, 0),
                ).shift(DOWN / 2)
            )

        def range_arrow_label_updater(m: Mobject):
            m.become(Tex("Range").next_to(range_arrow, direction=DOWN, buff=SMALL_BUFF))

        sine_graph.add_updater(sine_updater)
        exp_graph.add_updater(exp_updater)
        range_arrow.add_updater(range_arrow_updater)
        range_arrow_label.add_updater(range_arrow_label_updater)

        self.add(
            range_ax,
            range_labels,
            sine_graph,
            exp_graph,
            range_arrow,
            range_arrow_label,
            range_attenuation_relation,
        )

        self.play(range_tracker.animate.set_value(6.0), run_time=3)

        self.wait(2)

        ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-0.1, 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            # x_length=,
            # y_length=y_len,
        ).scale(0.8)

        labels = range_ax.get_axis_labels(
            Tex("t", font_size=DEFAULT_FONT_SIZE),
            Tex("", font_size=DEFAULT_FONT_SIZE),
        )

        sine_graph.remove_updater(sine_updater)
        exp_graph.remove_updater(exp_updater)
        range_arrow.remove_updater(range_arrow_updater)
        range_arrow_label.remove_updater(range_arrow_label_updater)

        self.play(
            LaggedStart(
                AnimationGroup(
                    range_attenuation_relation.animate.shift(UP * 5),
                    range_arrow_label.animate.shift(DOWN * 5),
                    range_arrow.animate.shift(DOWN * 5),
                ),
                AnimationGroup(
                    Uncreate(sine_graph),
                    Uncreate(exp_graph),
                    Transform(range_ax, ax),
                    Transform(range_labels, labels),
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(2)


# Pretty much done - just some minor issues at the end with shifting
class PulsedPowerProblemUsingUpdaters(Scene):
    def construct(self):
        duty_cycle_inc = -0.7
        step = 0.001
        x_range_min = 1
        x_max = 6
        f = 4

        duty_cycle_tracker = ValueTracker(1.0)
        range_tracker = ValueTracker(x_range_min)
        pulsed_gain_tracker = ValueTracker(1.0)
        pulsed_gain_copy_tracker = ValueTracker(1.0)

        range_ax = Axes(
            x_range=[x_range_min - 0.1, x_max, 1],
            y_range=[-0.1, 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
        ).scale(0.8)

        range_labels = range_ax.get_axis_labels(
            Tex("R", font_size=DEFAULT_FONT_SIZE),
            Tex("", font_size=DEFAULT_FONT_SIZE),
        )

        sine_graph = range_ax.plot(
            lambda t: np.sin(2 * PI * f * t) * (np.exp(t) ** 4),
            x_range=[0, range_tracker.get_value(), step],
            use_smoothing=False,
        )
        exp_graph = range_ax.plot(
            lambda t: (np.exp(t) ** 4),
            x_range=[0, range_tracker.get_value(), step],
            use_smoothing=False,
            color=YELLOW,
        )

        range_arrow = Arrow(
            range_ax.c2p(x_range_min, 0, 0),
            range_ax.c2p(range_tracker.get_value(), 0, 0),
        ).shift(DOWN / 2)

        range_arrow_label = Tex("Range").next_to(
            range_arrow, direction=DOWN, buff=SMALL_BUFF
        )

        range_attenuation_relation = (
            Tex(r"Power $\propto \frac{1}{R^2}$").shift(RIGHT + 2).scale(1.4)
        )

        def sine_updater(m: Mobject):
            m.become(
                range_ax.plot(
                    lambda r: (np.sin(2 * PI * f * r) + 1) / 2 / (r**2),
                    x_range=[x_range_min, range_tracker.get_value(), step],
                    use_smoothing=False,
                )
            )

        def exp_updater(m: Mobject):
            m.become(
                range_ax.plot(
                    lambda r: 1 / (r**2),
                    x_range=[x_range_min, range_tracker.get_value(), step],
                    use_smoothing=False,
                    color=YELLOW,
                )
            )

        def range_arrow_updater(m: Mobject):
            m.become(
                Arrow(
                    range_ax.c2p(x_range_min, 0, 0),
                    range_ax.c2p(range_tracker.get_value(), 0, 0),
                ).shift(DOWN / 2)
            )

        def range_arrow_label_updater(m: Mobject):
            m.become(Tex("Range").next_to(range_arrow, direction=DOWN, buff=SMALL_BUFF))

        sine_graph.add_updater(sine_updater)
        exp_graph.add_updater(exp_updater)
        range_arrow.add_updater(range_arrow_updater)
        range_arrow_label.add_updater(range_arrow_label_updater)

        ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-0.1, 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            # x_length=,
            # y_length=y_len,
        ).scale(0.8)

        labels = range_ax.get_axis_labels(
            Tex("t", font_size=DEFAULT_FONT_SIZE),
            Tex("", font_size=DEFAULT_FONT_SIZE),
        )

        pulsed_graph = ax.plot(
            lambda t: np.sin(2 * PI * f * t)
            * (
                (
                    signal.square(
                        2 * PI * t / 2,
                        duty=1 + duty_cycle_inc,
                    )
                    + 1
                )
                / 2
            ),
            x_range=[0, x_max, step],
            use_smoothing=False,
            color=RED,
        ).set_z_index(5)
        pulsed_graph_copy = ax.plot(
            lambda t: pulsed_gain_copy_tracker.get_value()
            * (np.sin(2 * PI * f * t) + 1)
            / 2
            * (
                (
                    signal.square(
                        2 * PI * t / 2,
                        duty=1 + duty_cycle_inc,
                    )
                    + 1
                )
                / 2
            ),
            x_range=[0, x_max - step, step],
            use_smoothing=False,
        ).set_color(WHITE)
        sq_graph = ax.plot(
            lambda t: pulsed_gain_tracker.get_value()
            * (
                signal.square(
                    2 * PI * t / 2,
                    duty=duty_cycle_tracker.get_value(),
                )
                + 1
            )
            / 2,
            x_range=[0, x_max - step, step],
            use_smoothing=False,
            color=YELLOW,
        )

        def pulsed_graph_updater(m: Mobject):
            m.become(
                ax.plot(
                    lambda t: pulsed_gain_tracker.get_value()
                    * (np.sin(2 * PI * f * t) + 1)
                    / 2
                    * (
                        (
                            signal.square(
                                2 * PI * t / 2,
                                duty=duty_cycle_tracker.get_value(),
                            )
                            + 1
                        )
                        / 2
                    ),
                    x_range=[0, x_max - step, step],
                    use_smoothing=False,
                )
                .set_color(RED)
                .set_z_index(5)
            )

        def pulsed_graph_copy_updater(m: Mobject):
            m.become(
                ax.plot(
                    lambda t: pulsed_gain_copy_tracker.get_value()
                    * (np.sin(2 * PI * f * t) + 1)
                    / 2
                    * (
                        (
                            signal.square(
                                2 * PI * t / 2,
                                duty=1 + duty_cycle_inc,
                            )
                            + 1
                        )
                        / 2
                    ),
                    x_range=[0, x_max - step, step],
                    use_smoothing=False,
                ).set_color(WHITE)
            )

        def sq_graph_updater(m: Mobject):
            m.become(
                ax.plot(
                    lambda t: pulsed_gain_tracker.get_value()
                    * (
                        signal.square(
                            2 * PI * t / 2,
                            duty=duty_cycle_tracker.get_value(),
                        )
                        + 1
                    )
                    / 2,
                    x_range=[0, x_max - step, step],
                    use_smoothing=False,
                    color=YELLOW,
                )
            )

        pulsed_graph.add_updater(pulsed_graph_updater)
        pulsed_graph_copy.add_updater(pulsed_graph_copy_updater)
        sq_graph.add_updater(sq_graph_updater)

        pulsed_graph_line_legend = (
            FunctionGraph(lambda x: 0, color=WHITE, x_range=[0, 1])
            .to_corner(UR)
            .set_z_index(2)
        )
        pulsed_graph_legend = (
            Tex("Pulses", color=WHITE)
            .next_to(pulsed_graph_line_legend, direction=LEFT, buff=SMALL_BUFF)
            .set_z_index(2)
        )
        pulsed_graph_avg_line_legend = (
            FunctionGraph(lambda x: 0, color=RED, x_range=[0, 1])
            .next_to(pulsed_graph_line_legend, direction=DOWN, buff=MED_LARGE_BUFF)
            .set_z_index(2)
        )
        pulsed_graph_avg_legend = (
            Tex(r"Pulsed Average Power", color=RED)
            .next_to(pulsed_graph_avg_line_legend, direction=LEFT, buff=SMALL_BUFF)
            .set_z_index(2)
        )
        desired_output_power_line_legend = DashedVMobject(
            FunctionGraph(lambda x: 0, color=GREEN, x_range=[0, 1]).next_to(
                pulsed_graph_avg_line_legend, direction=DOWN, buff=MED_LARGE_BUFF
            )
        ).set_z_index(2)
        desired_output_power_legend = (
            Tex(r"Desired Average Power", color=GREEN)
            .next_to(desired_output_power_line_legend, direction=LEFT, buff=SMALL_BUFF)
            .set_z_index(2)
        )

        pulsed_graph_avg_legend_background = BackgroundRectangle(
            pulsed_graph_avg_legend,
            color=config.background_color,
            fill_opacity=0.8,
        ).set_z_index(1)
        desired_output_power_legend_background = BackgroundRectangle(
            desired_output_power_legend,
            color=config.background_color,
            fill_opacity=0.8,
        ).set_z_index(1)
        pulsed_graph_line_legend_background = BackgroundRectangle(
            pulsed_graph_legend,
            color=config.background_color,
            fill_opacity=0.8,
        ).set_z_index(1)

        brace_buff = 0.3

        def duty_cycle_brace_updater(m: Mobject):
            m.become(
                BraceLabel(
                    Line(
                        ax.c2p(0, 0, 0),
                        ax.c2p(duty_cycle_tracker.get_value() * 2, 0, 0),
                    ),
                    f"Duty cycle={int(duty_cycle_tracker.get_value()*100)}\\%",
                    label_constructor=Tex,
                    buff=brace_buff,
                )
            )

        duty_cycle_brace = BraceLabel(
            Line(
                ax.c2p(0, 0, 0),
                ax.c2p(duty_cycle_tracker.get_value() * 2, 0, 0),
            ),
            f"Duty cycle={int(duty_cycle_tracker.get_value()*100)}\\%",
            label_constructor=Tex,
            buff=brace_buff,
        )
        duty_cycle_brace.add_updater(duty_cycle_brace_updater)

        period_brace = BraceLabel(
            Line(ax.c2p(0, 1, 0), ax.c2p(2, 1, 0)),
            "Period",
            brace_direction=UP,
            label_constructor=Tex,
            buff=brace_buff,
        )

        waiting_time_brace = Brace(
            Line(ax.c2p(-duty_cycle_inc, 0, 0), ax.c2p(2, 0, 0)),
            buff=brace_buff,
        )
        waiting_time_brace_label = (
            Tex("Waiting time")
            .next_to(waiting_time_brace, direction=RIGHT, buff=SMALL_BUFF)
            .shift(RIGHT / 2)
        )

        desired_output_power = 0.6
        desired_output_power_graph = DashedVMobject(
            ax.plot(lambda t: desired_output_power, x_range=[0, x_max], color=GREEN)
        )

        def avg_power_eqn_updater(m: Mobject):
            m.become(
                Tex(
                    r"For pulsed: $P_{av} = P_{peak} \cdot $",
                    r"$\underbrace{\frac{t_{on}}{t_{on}+t_{off}}}_{\text{Duty cycle}}=$ ",
                    f"${pulsed_gain_copy_tracker.get_value():.2f} [W] \\cdot $",
                    f"${int((1+duty_cycle_inc)*100)} \\%$ ",
                    f"$=$",
                    f"${pulsed_gain_copy_tracker.get_value()*(1+duty_cycle_inc):.2f} [W]$",
                )
                .scale(0.6)
                .to_edge(DOWN, buff=MED_LARGE_BUFF)
            )

        avg_power_eqn_pulsed = (
            Tex(
                r"For pulsed: $P_{av} = P_{peak} \cdot $",
                r"$\underbrace{\frac{t_{on}}{t_{on}+t_{off}}}_{\text{Duty cycle}}=$ ",
                f"${pulsed_gain_copy_tracker.get_value():.2f} [W] \\cdot $",
                f"${int((1+duty_cycle_inc)*100)} \\%$ ",
                f"$=$",
                f"${pulsed_gain_copy_tracker.get_value()*(1+duty_cycle_inc):.2f} [W]$",
            )
            .scale(0.6)
            .to_edge(DOWN, buff=MED_LARGE_BUFF)
        )

        avg_power_eqn_cw = Tex(
            r"For CW: $P_{av} =$ ",  # P_{peak} \cdot \underbrace{\frac{t_{on}}{t_{on}+t_{off}}}_{\text{Duty cycle}}=$",
            f"${desired_output_power:.2f} [W] \\ \\cdot \\  $",
            "$100 \\%$ ",
            f"$ = {desired_output_power:.2f} [W]$",
        ).scale(0.6)

        """ Animations """

        self.add(
            sine_graph,
            exp_graph,
            range_arrow,
        )
        self.play(
            Create(range_ax),
            Create(range_labels),
            FadeIn(range_attenuation_relation),
        )

        self.play(
            range_tracker.animate(run_time=3).set_value(6.0), FadeIn(range_arrow_label)
        )

        self.play(Indicate(range_attenuation_relation))

        self.wait(1)

        sine_graph.remove_updater(sine_updater)
        exp_graph.remove_updater(exp_updater)
        range_arrow.remove_updater(range_arrow_updater)
        range_arrow_label.remove_updater(range_arrow_label_updater)

        self.play(
            LaggedStart(
                AnimationGroup(
                    range_attenuation_relation.animate.shift(UP * 5),
                    range_arrow_label.animate.shift(DOWN * 5),
                    range_arrow.animate.shift(DOWN * 5),
                ),
                AnimationGroup(
                    Uncreate(sine_graph),
                    Uncreate(exp_graph),
                    Transform(range_ax, ax),
                    Transform(range_labels, labels),
                ),
                lag_ratio=0.6,
            )
        )

        # self.play(Create(ax))
        self.remove(range_attenuation_relation, range_arrow, range_arrow_label)

        self.wait(1)

        self.play(
            Create(pulsed_graph),
            Create(sq_graph),
            run_time=2,
        )

        self.wait(0.7)

        self.play(Create(duty_cycle_brace), Create(period_brace), run_time=1)

        self.wait(0.7)

        self.play(
            duty_cycle_tracker.animate.increment_value(duty_cycle_inc), run_time=3
        )

        self.wait(1)

        self.play(Create(waiting_time_brace), Create(waiting_time_brace_label))

        self.wait(1)

        duty_cycle_brace.remove_updater(duty_cycle_brace_updater)

        self.play(
            Uncreate(waiting_time_brace),
            Uncreate(waiting_time_brace_label),
            Uncreate(duty_cycle_brace),
            Uncreate(period_brace),
        )
        self.add(pulsed_graph_copy)

        self.play(
            LaggedStart(
                FadeIn(
                    VGroup(
                        pulsed_graph_avg_legend_background,
                        pulsed_graph_line_legend_background,
                        pulsed_graph_legend.set_z_index(1),
                        pulsed_graph_avg_legend.set_z_index(1),
                        pulsed_graph_line_legend,
                        pulsed_graph_avg_line_legend,
                    ),
                    shift=DOWN,
                ),
                AnimationGroup(
                    pulsed_gain_tracker.animate.set_value(
                        pulsed_gain_tracker.get_value() * (1 + duty_cycle_inc)
                    ),
                    duty_cycle_tracker.animate.increment_value(-duty_cycle_inc),
                    run_time=3,
                ),
                lag_ratio=0.7,
            )
        )

        self.wait(1)

        self.play(Uncreate(sq_graph), Create(desired_output_power_graph))

        self.play(
            FadeIn(
                VGroup(
                    desired_output_power_legend_background,
                    desired_output_power_line_legend.set_z_index(1),
                    desired_output_power_legend.set_z_index(1),
                ),
                shift=UP,
            ),
            FadeIn(avg_power_eqn_pulsed, shift=UP),
        )

        avg_power_eqn_pulsed.add_updater(avg_power_eqn_updater)

        self.play(
            pulsed_gain_copy_tracker.animate.set_value(
                desired_output_power / pulsed_gain_tracker.get_value()
            ),
            pulsed_gain_tracker.animate.set_value(desired_output_power),
            run_time=2,
        )

        self.wait()

        self.play(Circumscribe(desired_output_power_graph))

        self.wait(0.5)

        self.play(
            Circumscribe(avg_power_eqn_pulsed[1]),  # Duty cycle
            Circumscribe(avg_power_eqn_pulsed[3]),  # Duty cycle
        )

        self.wait(0.5)

        self.play(Circumscribe(avg_power_eqn_pulsed[5]))  # Desired average power

        self.wait(0.5)

        self.play(Circumscribe(pulsed_graph))

        avg_power_eqn_pulsed.remove_updater(avg_power_eqn_updater)

        self.play(
            VGroup(
                ax,
                range_ax,
                avg_power_eqn_pulsed,
                desired_output_power_graph,
                range_labels,
                labels,
                pulsed_graph_copy,
                pulsed_graph,
            )
            .set_z_index(0)
            .animate.shift(UP),
        )
        self.play(
            FadeIn(
                avg_power_eqn_cw.next_to(avg_power_eqn_pulsed, direction=DOWN), shift=UP
            )
        )

        self.wait(0.5)

        self.play(Circumscribe(avg_power_eqn_cw[2]))

        self.wait(1)

        pulsed_graph.remove_updater(pulsed_graph_updater)
        pulsed_graph_copy.remove_updater(pulsed_graph_copy_updater)
        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class CWWrapUp(Scene):
    def construct(self):
        ranging_highlight_width_tracker = ValueTracker(0.0)

        cw_radar = FMCWRadarCartoon(text="CW")
        radar = WeatherRadarTower()
        cw_radar_end = FMCWRadarCartoon(text="CW")
        cw_radar_end.vgroup.scale(0.5).to_corner(UL, buff=MED_SMALL_BUFF)

        x_max = 4
        x_len = 6
        y_len = 3
        step = 0.001
        cw_ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        pulsed_ax = Axes(
            x_range=[-0.1, x_max, 0.5],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        f = 4
        pulsed_amp = 3
        A = 0.5
        cw_sine = lambda t: A * np.sin(2 * PI * f * t)
        pulsed_sine = (
            lambda t: pulsed_amp
            * A
            * np.sin(2 * PI * f * t)
            * ((signal.square(2 * PI * t / 2, duty=0.3) + 1) / 2)
        )
        sq = (
            lambda t: pulsed_amp * A * (signal.square(2 * PI * t / 2, duty=0.3) + 1) / 2
        )

        cw_graph = cw_ax.plot(cw_sine, x_range=[0, x_max, step], use_smoothing=False)
        pulsed_graph = pulsed_ax.plot(
            pulsed_sine, x_range=[0, x_max, step], use_smoothing=False
        )
        sq_graph = pulsed_ax.plot(
            sq, x_range=[0, x_max - step, step], use_smoothing=False, color=YELLOW
        )

        graphs = (
            VGroup(VGroup(pulsed_ax, pulsed_graph, sq_graph), VGroup(cw_ax, cw_graph))
            .arrange(direction=RIGHT, buff=LARGE_BUFF * 1.5, center=True)
            .scale_to_fit_width(13)
            .to_edge(DOWN, buff=MED_LARGE_BUFF)
        )
        graph_arrow = Arrow(
            pulsed_graph.get_edge_center(RIGHT), cw_graph.get_edge_center(LEFT)
        )

        radars = VGroup(radar.vgroup.scale(0.6), cw_radar.vgroup.scale(0.8)).arrange(
            direction=RIGHT, buff=LARGE_BUFF * 1.5, center=True
        )

        wip = Text("Work in progress", color=BLACK)
        wip_bounding_box = SurroundingRectangle(
            wip, color=YELLOW, fill_color=YELLOW, fill_opacity=1, buff=SMALL_BUFF
        )

        radar_definition = (
            Tex(r"RADAR\\", "Ra", "dio", " D", "etection", " A", "nd", " R", "anging")
            .to_edge(LEFT, buff=MED_LARGE_BUFF)
            .set_z_index(1)
        )
        radar_definition[1].set_color(RED)
        radar_definition[3].set_color(RED)
        radar_definition[5].set_color(RED)
        radar_definition[7].set_color(RED)

        def ranging_highlight_width_updater(m: Mobject):
            m.become(
                Rectangle(
                    width=ranging_highlight_width_tracker.get_value(),
                    height=ranging_word.height * 1.2,
                    color=DARK_BLUE,
                    fill_color=DARK_BLUE,
                    fill_opacity=1,
                ).shift(ranging_word.get_center())
            )

        ranging_word = VGroup(radar_definition[7], radar_definition[8])
        ranging_highlight = Rectangle(
            width=ranging_highlight_width_tracker.get_value(),
            height=ranging_word.height * 1.2,
            color=DARK_BLUE,
            fill_color=DARK_BLUE,
            fill_opacity=1,
        ).shift(ranging_word.get_center())
        ranging_highlight.add_updater(ranging_highlight_width_updater)

        cloud = SVGMobject(
            "./figures/clouds.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).to_corner(UR, buff=LARGE_BUFF)

        """ Animations """

        self.play(radar.get_animation(), cw_radar.get_animation())

        self.wait(1)

        self.play(
            radar.vgroup.animate.next_to(pulsed_ax, buff=MED_LARGE_BUFF, direction=UP),
            cw_radar.vgroup.animate.next_to(cw_ax, buff=MED_LARGE_BUFF, direction=UP),
            LaggedStart(
                AnimationGroup(Create(pulsed_ax), Create(cw_ax), run_time=2),
                AnimationGroup(
                    Create(cw_graph), Create(pulsed_graph), Create(sq_graph), run_time=2
                ),
                Create(graph_arrow),
                lag_ratio=0.8,
            ),
        )

        wip_group = (
            VGroup(wip.set_z_index(1), wip_bounding_box)
            .scale_to_fit_width(cw_radar.vgroup.width * 2)
            .rotate(PI / 6)
            .shift(cw_radar.vgroup.get_center())
        )
        self.play(FadeIn(wip_group))

        self.play(
            LaggedStart(
                FadeOut(graphs, graph_arrow, radar.vgroup),
                AnimationGroup(
                    cw_radar.vgroup.animate.set_y(0),
                    wip_group.animate.set_y(0),
                    Create(radar_definition),
                ),
                lag_ratio=0.5,
            )
        )

        self.wait(1)

        self.add(ranging_highlight)
        self.play(ranging_highlight_width_tracker.animate.set_value(ranging_word.width))

        self.wait(1)

        ranging_highlight.remove_updater(ranging_highlight_width_updater)
        self.play(FadeOut(radar_definition, ranging_highlight, wip_group))

        self.play(
            cw_radar.vgroup.animate.to_corner(DL, buff=MED_LARGE_BUFF).shift(UP),
            Create(cloud),
        )

        arrow_to_cloud = Arrow(
            cw_radar.antenna_tx.get_edge_center(RIGHT),
            cloud.get_edge_center(LEFT),
            color=TX_COLOR,
        )
        cloud_to_arrow = Arrow(
            cloud.get_edge_center(LEFT) + DOWN / 2,
            cw_radar.antenna_rx.get_edge_center(RIGHT),
            color=RX_COLOR,
        )

        propagation_brace = BraceLabel(
            Line(
                cloud_to_arrow.get_end(),
                cloud_to_arrow.get_end() + arrow_to_cloud.width * RIGHT,
            ),
            "Round-trip time?",
            label_constructor=Tex,
            buff=MED_SMALL_BUFF,
        )

        self.play(Create(arrow_to_cloud))
        self.play(Create(cloud_to_arrow))
        self.play(FadeIn(propagation_brace, shift=UP))

        self.wait(1)

        self.play(
            Transform(cw_radar.vgroup, cw_radar_end.vgroup),
            FadeOut(propagation_brace, arrow_to_cloud, cloud_to_arrow, cloud),
        )

        self.wait(2)


class CWNotForRange(Scene):
    def construct(self):
        cw_radar = FMCWRadarCartoon()
        cw_radar_old = FMCWRadarCartoon("CW")
        cw_radar_old.vgroup.scale(0.5).to_corner(UL, buff=MED_SMALL_BUFF)
        cw_radar.vgroup.scale(0.5).to_corner(UL, buff=MED_SMALL_BUFF).shift(DOWN)

        cloud = SVGMobject(
            "./figures/clouds.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).to_corner(UR, buff=MED_SMALL_BUFF)

        arrow_to_cloud = Arrow(
            cw_radar.antenna_tx.get_edge_center(RIGHT),
            cloud.get_edge_center(LEFT),
            color=TX_COLOR,
        )
        cloud_rx_shift = DOWN / 2
        cloud_to_arrow = Arrow(
            cloud.get_edge_center(LEFT) + cloud_rx_shift,
            cw_radar.antenna_rx.get_edge_center(RIGHT),
            color=RX_COLOR,
        )

        def get_prop_arrow_updater(
            from_mobj,
            to_mobj,
            color=WHITE,
            edges=[RIGHT, LEFT],
            from_shift=[0, 0, 0],
            to_shift=[0, 0, 0],
        ):
            def updater(m: Mobject):
                m.become(
                    Arrow(
                        from_mobj.get_edge_center(edges[0]) + from_shift,
                        to_mobj.get_edge_center(edges[1]) + to_shift,
                        color=color,
                    )
                )

            return updater

        arrow_to_cloud_updater = get_prop_arrow_updater(
            cw_radar.antenna_tx, cloud, edges=[RIGHT, LEFT], color=TX_COLOR
        )
        cloud_to_arrow_updater = get_prop_arrow_updater(
            cloud,
            cw_radar.antenna_rx,
            edges=[LEFT, RIGHT],
            color=RX_COLOR,
            from_shift=cloud_rx_shift,
        )
        arrow_to_cloud.add_updater(arrow_to_cloud_updater)
        cloud_to_arrow.add_updater(cloud_to_arrow_updater)

        cloud_vel = Arrow(ORIGIN, LEFT * 2).next_to(cloud, direction=DOWN)
        cloud_vel_label = Tex(r"$v_{cloud}=0$").next_to(
            cloud_vel, direction=DOWN, buff=MED_SMALL_BUFF
        )
        cloud_vel_label_copy = Tex(r"$v_{cloud}=0$").next_to(
            cloud_vel, direction=DOWN, buff=MED_SMALL_BUFF
        )
        cloud_vel_label_pos = Tex(r"$v_{cloud}>0$").next_to(
            cloud_vel, direction=DOWN, buff=MED_SMALL_BUFF
        )
        cloud_vel_label_neg = Tex(r"$v_{cloud}<0$").next_to(
            cloud_vel, direction=DOWN, buff=MED_SMALL_BUFF
        )

        def cloud_vel_label_updater(m: Mobject):
            m.next_to(cloud, direction=DOWN)

        def cloud_vel_updater(m: Mobject):
            m.next_to(cloud_vel_label, direction=DOWN, buff=MED_SMALL_BUFF)

        cloud_vel.add_updater(cloud_vel_updater)
        cloud_vel_label.add_updater(cloud_vel_label_updater)
        cloud_vel_label_copy.add_updater(cloud_vel_label_updater)
        cloud_vel_label_pos.add_updater(cloud_vel_label_updater)
        cloud_vel_label_neg.add_updater(cloud_vel_label_updater)

        x_max = 4
        x_len = 6
        y_len = 3
        step = 0.001
        amp_ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        f_ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-1, 5, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        amp_labels = amp_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )
        f_labels = f_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        f = 2
        A = 1
        cw_sine = lambda t: A * np.sin(2 * PI * f * t)
        cw_graph = amp_ax.plot(cw_sine, x_range=[0, x_max, step], use_smoothing=False)

        t_shift = 1

        f_tx_graph = f_ax.plot(
            lambda t: f, x_range=[0, x_max, step], use_smoothing=False
        )

        amp_graph_group = VGroup(cw_graph, amp_ax, amp_labels)
        amp_graph_group_copy = VGroup(cw_graph.copy(), amp_ax.copy(), amp_labels.copy())
        f_graph_group = VGroup(f_ax, f_labels, f_tx_graph)

        both_graphs = (
            VGroup(amp_graph_group_copy, f_graph_group)
            .arrange(buff=MED_LARGE_BUFF)
            .scale_to_fit_width(12)
        )

        tx_line_legend = FunctionGraph(
            lambda x: 0, color=TX_COLOR, x_range=[0, 1]
        ).next_to(f_ax, direction=UP + RIGHT, aligned_edge=RIGHT)
        tx_legend = Tex("Tx", color=TX_COLOR).next_to(
            tx_line_legend, direction=LEFT, buff=SMALL_BUFF
        )
        rx_line_legend = FunctionGraph(
            lambda x: 0, color=RX_COLOR, x_range=[0, 1]
        ).next_to(tx_line_legend, direction=DOWN, buff=MED_LARGE_BUFF)
        rx_legend = Tex(r"Rx", color=RX_COLOR).next_to(
            rx_line_legend, direction=LEFT, buff=SMALL_BUFF
        )

        def get_next_to_updater(
            m2: Mobject,
            buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
            direction=RIGHT,
            aligned_edge=ORIGIN,
        ):
            def updater(m: Mobject):
                m.next_to(m2, buff=buff, direction=direction, aligned_edge=aligned_edge)

            return updater

        tx_line_legend.add_updater(
            get_next_to_updater(f_ax, direction=UP + RIGHT, aligned_edge=RIGHT)
        )
        tx_legend.add_updater(
            get_next_to_updater(tx_line_legend, direction=LEFT, buff=SMALL_BUFF)
        )
        rx_line_legend.add_updater(
            get_next_to_updater(tx_line_legend, direction=DOWN, buff=MED_LARGE_BUFF)
        )
        rx_legend.add_updater(
            get_next_to_updater(rx_line_legend, direction=LEFT, buff=SMALL_BUFF)
        )

        """ Animations """

        self.add(cw_radar_old.vgroup)

        self.play(
            LaggedStart(
                Create(VGroup(amp_ax, amp_labels)), Create(cw_graph), lag_ratio=0.8
            )
        )

        self.wait(1)

        self.play(
            LaggedStart(
                Transform(amp_graph_group, amp_graph_group_copy),
                LaggedStart(
                    Create(VGroup(f_ax, f_labels)), Create(f_tx_graph), lag_ratio=0.8
                ),
                lag_ratio=0.9,
            )
        )
        self.add(amp_graph_group_copy)
        self.remove(amp_graph_group)

        self.wait(1)

        self.play(
            f_tx_graph.animate.set_color(BLUE),
            Create(VGroup(tx_line_legend, tx_legend)),
        )

        self.wait(1)

        self.play(both_graphs.animate.to_edge(DOWN, buff=MED_SMALL_BUFF))

        f_rx_graph = f_ax.plot(
            lambda t: f,
            x_range=[t_shift, x_max + t_shift, step],
            use_smoothing=False,
            color=RX_COLOR,
        )

        # self.play(cw_radar.get_animation(), Create(cloud))
        self.play(Transform(cw_radar_old.vgroup, cw_radar.vgroup), Create(cloud))
        self.play(Create(arrow_to_cloud))
        self.play(
            Create(cloud_to_arrow),
            TransformFromCopy(f_tx_graph, f_rx_graph),
            Create(VGroup(rx_line_legend, rx_legend)),
        )

        self.wait(1)

        self.play(Create(cloud_vel_label))

        f_shift_scale = 0.5
        self.play(
            Transform(cloud_vel_label, cloud_vel_label_pos),
            GrowArrow(cloud_vel),
            run_time=0.3,
        )
        self.play(
            cloud.animate.shift(LEFT * 5),
            f_rx_graph.animate.shift(UP * f_shift_scale),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            Succession(
                AnimationGroup(
                    Transform(cloud_vel_label, cloud_vel_label_neg),
                    cloud_vel.animate.flip(axis=[0, 1, 0]),
                    run_time=0.3,
                ),
                AnimationGroup(
                    cloud.animate.shift(RIGHT * 5),
                    f_rx_graph.animate.shift(DOWN * f_shift_scale * 2),
                    run_time=3,
                ),
            )
        )

        self.wait(0.5)

        self.play(
            Succession(
                AnimationGroup(
                    Transform(cloud_vel_label, cloud_vel_label_copy),
                    FadeOut(cloud_vel),
                    run_time=0.3,
                ),
                f_rx_graph.animate.shift(UP * f_shift_scale),
            )
        )

        self.wait(1)

        arrow_to_cloud.remove_updater(arrow_to_cloud_updater)
        cloud_to_arrow.remove_updater(cloud_to_arrow_updater)
        cloud_vel.remove_updater(cloud_vel_updater)
        cloud_vel_label.remove_updater(cloud_vel_label_updater)
        cloud_vel_label_copy.remove_updater(cloud_vel_label_updater)
        cloud_vel_label_pos.remove_updater(cloud_vel_label_updater)
        cloud_vel_label_neg.remove_updater(cloud_vel_label_updater)
        self.play(
            FadeOut(
                cloud,
                arrow_to_cloud,
                cloud_to_arrow,
                cloud_vel_label,
                cw_radar_old.vgroup,
                shift=UP,
            ),
            FadeOut(amp_graph_group_copy, shift=LEFT),
        )

        self.play(
            VGroup(f_ax, f_labels, f_tx_graph, f_rx_graph).animate.move_to(ORIGIN)
        )

        self.wait(0.5)

        round_trip_brace = Brace(
            Line(f_tx_graph.get_start(), f_rx_graph.get_start()),
            buff=SMALL_BUFF,
            direction=UP,
        )
        round_trip_brace_label = (
            Tex(r"Round-trip\\time")
            .next_to(round_trip_brace, direction=UP + RIGHT, buff=MED_SMALL_BUFF)
            .shift(UP)
        )
        round_trip_brace_arrow = Arrow(
            round_trip_brace_label.get_edge_center(LEFT),
            round_trip_brace.get_edge_center(UP),
        )
        round_trip_brace_label_qmark = (
            Tex("?", color=YELLOW)
            .scale(2)
            .next_to(round_trip_brace_label, direction=RIGHT, buff=SMALL_BUFF)
        )

        self.play(
            Create(round_trip_brace),
            Create(round_trip_brace_label),
            Create(round_trip_brace_arrow),
        )

        self.wait(1)

        self.play(
            f_tx_graph.animate.set_color(WHITE),
            f_rx_graph.animate.set_color(WHITE),
            tx_legend.animate.set_color(WHITE),
            rx_legend.animate.set_color(WHITE),
            tx_line_legend.animate.set_color(WHITE),
            rx_line_legend.animate.set_color(WHITE),
        )
        self.play(FadeIn(round_trip_brace_label_qmark))

        self.wait(1)

        plot_group = VGroup(
            f_ax,
            f_labels,
            round_trip_brace,
            round_trip_brace_label,
            round_trip_brace_arrow,
            f_tx_graph,
            f_rx_graph,
            tx_legend,
            rx_legend,
            tx_line_legend,
            rx_line_legend,
            round_trip_brace_label_qmark,
        )

        self.play(plot_group.animate.shift(UP))

        self.wait(1)

        what_if_we_changed = Tex(
            r"What if we modified our signal\\to make this shift visible?"
        ).next_to(f_ax, direction=DOWN, buff=MED_LARGE_BUFF)

        self.play(Create(what_if_we_changed))

        self.wait(1)

        part_2 = Tex(
            "Part ",
            "2",
            ": ",
            "Frequency Modulation",
            font_size=DEFAULT_FONT_SIZE * 0.8 * 2,
        )

        self.play(
            FadeOut(plot_group, what_if_we_changed, shift=UP * 3),
            FadeIn(part_2, shift=UP * 3),
            run_time=2,
        )

        self.wait(2)

        self.play(
            LaggedStart(
                FadeOut(part_2[:3]),
                part_2[3].animate.move_to(ORIGIN).shift(UP),
                lag_ratio=0.4,
            )
        )

        f = Tex("frequency").shift(DOWN)
        f_of_t = Tex("frequency", "(", "time", ")").shift(DOWN)
        f_of_t_small = Tex("$f(t)$").shift(DOWN)
        self.play(FadeIn(f))
        self.play(Transform(f, f_of_t))

        self.wait(0.5)

        self.play(Transform(f, f_of_t_small))

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)


class ModulationTypes(Scene):
    def construct(self):
        x0_reveal_tracker_0 = ValueTracker(0.0)
        x1_reveal_tracker_0 = ValueTracker(0.0)
        x0_reveal_tracker_1 = ValueTracker(0.0)
        x1_reveal_tracker_1 = ValueTracker(0.0)
        x0_reveal_tracker_2 = ValueTracker(0.0)
        x1_reveal_tracker_2 = ValueTracker(0.0)
        x0_reveal_tracker_3 = ValueTracker(0.0)
        x1_reveal_tracker_3 = ValueTracker(0.0)

        no_mod_title = create_title("No Modulation")
        triangular_title = create_title("Triangular Modulation")
        fsk_title = create_title("Frequency Shift Keyring Modulation")
        sawtooth_title = create_title("Sawtooth Modulation")
        lfm_title = create_title("Linear Frequency Modulation (LFM)")
        dash_title = create_title("|")

        carrier_freq = 10  # Carrier frequency in Hz
        modulation_freq = 0.5  # Modulation frequency in Hz
        modulation_index = 20  # Modulation index
        duration = 1
        fs = 1000

        x_len = 11
        y_len = 2.2
        amp_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        f_ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-2, 30, 5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )

        amp_labels = amp_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$A$", font_size=DEFAULT_FONT_SIZE),
        )
        f_labels = f_ax.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        def get_x_reveal_updater(
            ax, func, x0_updater, x1_updater, clip_sq=False, color=WHITE
        ):
            def updater(m: Mobject):
                x1 = (
                    min(x1_updater.get_value(), duration - 1 / fs)
                    if clip_sq
                    else x1_updater.get_value()
                )
                m.become(
                    ax.plot(
                        func,
                        x_range=[x0_updater.get_value(), x1, 1 / fs],
                        use_smoothing=False,
                        color=color,
                    )
                )

            return updater

        """ Sine """
        sine_f = 4
        sine_modulating_signal = lambda t: sine_f
        sine_amp = lambda t: np.sin(2 * PI * sine_modulating_signal(t) * t)

        sine_f_graph = f_ax.plot(
            sine_modulating_signal,
            x_range=[
                x0_reveal_tracker_0.get_value(),
                x1_reveal_tracker_0.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        sine_amp_graph = amp_ax.plot(
            sine_amp,
            x_range=[
                x0_reveal_tracker_0.get_value(),
                x1_reveal_tracker_0.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )

        sine_f_graph_updater = get_x_reveal_updater(
            f_ax,
            sine_modulating_signal,
            x0_reveal_tracker_0,
            x1_reveal_tracker_0,
            clip_sq=True,
            color=TX_COLOR,
        )
        sine_amp_graph_updater = get_x_reveal_updater(
            amp_ax, sine_amp, x0_reveal_tracker_0, x1_reveal_tracker_0, color=TX_COLOR
        )
        sine_f_graph.add_updater(sine_f_graph_updater)
        sine_amp_graph.add_updater(sine_amp_graph_updater)

        """ Triangular """
        triangular_modulating_signal = lambda t: modulation_index * np.arcsin(
            np.sin(2 * np.pi * modulation_freq * t)
        )
        triangular_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(triangular_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        triangular_amp = lambda t: np.sin(2 * np.pi * triangular_modulating_cumsum(t))

        triangular_f_graph = f_ax.plot(
            triangular_modulating_signal,
            x_range=[
                x0_reveal_tracker_1.get_value(),
                x1_reveal_tracker_1.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        triangular_amp_graph = amp_ax.plot(
            triangular_amp,
            x_range=[
                x0_reveal_tracker_1.get_value(),
                x1_reveal_tracker_1.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        triangular_f_graph_updater = get_x_reveal_updater(
            f_ax,
            triangular_modulating_signal,
            x0_reveal_tracker_1,
            x1_reveal_tracker_1,
            color=TX_COLOR,
        )
        triangular_amp_graph_updater = get_x_reveal_updater(
            amp_ax,
            triangular_amp,
            x0_reveal_tracker_1,
            x1_reveal_tracker_1,
            color=TX_COLOR,
        )
        triangular_f_graph.add_updater(triangular_f_graph_updater)
        triangular_amp_graph.add_updater(triangular_amp_graph_updater)

        """ FSK """
        fsk_carrier_freq = 12
        fsk_modulation_index = 6
        fsk_modulating_signal_f = 2
        fsk_modulating_signal = (
            lambda t: fsk_modulation_index
            * signal.square(2 * PI * fsk_modulating_signal_f * t)
            + fsk_carrier_freq
        )
        fsk_amp = lambda t: np.sin(2 * PI * fsk_modulating_signal(t) * t)

        fsk_f_graph = f_ax.plot(
            fsk_modulating_signal,
            x_range=[
                x0_reveal_tracker_2.get_value(),
                x1_reveal_tracker_2.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        fsk_amp_graph = amp_ax.plot(
            fsk_amp,
            x_range=[
                x0_reveal_tracker_2.get_value(),
                x1_reveal_tracker_2.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )

        fsk_f_graph_updater = get_x_reveal_updater(
            f_ax,
            fsk_modulating_signal,
            x0_reveal_tracker_2,
            x1_reveal_tracker_2,
            clip_sq=True,
            color=TX_COLOR,
        )
        fsk_amp_graph_updater = get_x_reveal_updater(
            amp_ax, fsk_amp, x0_reveal_tracker_2, x1_reveal_tracker_2, color=TX_COLOR
        )
        fsk_f_graph.add_updater(fsk_f_graph_updater)
        fsk_amp_graph.add_updater(fsk_amp_graph_updater)

        """ Sawtooth """
        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        sawtooth_amp = lambda t: np.sin(2 * PI * sawtooth_modulating_cumsum(t))

        sawtooth_f_graph = f_ax.plot(
            sawtooth_modulating_signal,
            x_range=[
                x0_reveal_tracker_3.get_value(),
                x1_reveal_tracker_3.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )
        sawtooth_amp_graph = amp_ax.plot(
            sawtooth_amp,
            x_range=[
                x0_reveal_tracker_3.get_value(),
                x1_reveal_tracker_3.get_value(),
                1 / fs,
            ],
            use_smoothing=False,
            color=TX_COLOR,
        )

        sawtooth_f_graph_updater = get_x_reveal_updater(
            f_ax,
            sawtooth_modulating_signal,
            x0_reveal_tracker_3,
            x1_reveal_tracker_3,
            clip_sq=True,
            color=TX_COLOR,
        )
        sawtooth_amp_graph_updater = get_x_reveal_updater(
            amp_ax,
            sawtooth_amp,
            x0_reveal_tracker_3,
            x1_reveal_tracker_3,
            color=TX_COLOR,
        )
        sawtooth_f_graph.add_updater(sawtooth_f_graph_updater)
        sawtooth_amp_graph.add_updater(sawtooth_amp_graph_updater)

        amp_ax_group = VGroup(amp_ax, amp_labels, triangular_amp_graph)
        f_ax_group = VGroup(f_ax, f_labels, triangular_f_graph)

        both_graphs = (
            VGroup(amp_ax_group, f_ax_group)
            .arrange(direction=DOWN, buff=MED_LARGE_BUFF)
            .next_to(triangular_title, direction=DOWN)
        )

        self.play(get_title_animation(no_mod_title, run_time=2))

        self.play(Create(f_ax), Create(amp_ax), Create(f_labels), Create(amp_labels))
        self.add(
            sine_f_graph,
            sine_amp_graph,
            triangular_f_graph,
            triangular_amp_graph,
            fsk_f_graph,
            fsk_amp_graph,
            sawtooth_f_graph,
            sawtooth_amp_graph,
        )
        self.play(x1_reveal_tracker_0.animate.set_value(duration), run_time=2)

        self.wait(1)

        self.play(x0_reveal_tracker_0.animate.set_value(duration), run_time=2)

        self.wait(0.5)

        self.play(
            Transform(no_mod_title[0], triangular_title[0], run_time=1),
            x1_reveal_tracker_1.animate(run_time=2).set_value(duration),
        )

        self.wait(1)

        self.play(x0_reveal_tracker_1.animate.set_value(duration), run_time=2)

        self.wait(0.5)

        self.play(
            Transform(no_mod_title[0], fsk_title[0], run_time=1),
            x1_reveal_tracker_2.animate(run_time=2).set_value(duration),
        )

        self.wait(1)

        self.play(x0_reveal_tracker_2.animate.set_value(duration), run_time=2)

        self.wait(0.5)

        self.play(
            Transform(no_mod_title[0], sawtooth_title[0], run_time=1),
            x1_reveal_tracker_3.animate(run_time=2).set_value(duration),
        )

        self.wait(1)

        # self.play(Transform(no_mod_title[0]))
        sawtooth_title_copy = sawtooth_title[0].copy()
        title_y = sawtooth_title_copy.get_y()
        VGroup(sawtooth_title_copy, dash_title[0], lfm_title[0]).arrange(RIGHT).to_edge(
            UP
        ).set_y(title_y)
        self.play(
            LaggedStart(
                Transform(no_mod_title[0], sawtooth_title_copy),
                Create(VGroup(dash_title[0], lfm_title[0])),
                lag_ratio=0.7,
            )
        )

        self.wait(1)

        # self.play(x0_reveal_tracker_3.animate.set_value(duration), run_time=2)

        sine_f_graph.remove_updater(sine_f_graph_updater)
        sine_amp_graph.remove_updater(sine_amp_graph_updater)
        triangular_f_graph.remove_updater(triangular_f_graph_updater)
        triangular_amp_graph.remove_updater(triangular_amp_graph_updater)
        fsk_f_graph.remove_updater(fsk_f_graph_updater)
        fsk_amp_graph.remove_updater(fsk_amp_graph_updater)
        sawtooth_f_graph.remove_updater(sawtooth_f_graph_updater)
        sawtooth_amp_graph.remove_updater(sawtooth_amp_graph_updater)

        self.wait(2)

        """ Transforming to Waveform initial plot """
        bw_tracker = ValueTracker(1.0)
        f_tracker = ValueTracker(2.0)
        f_0_tracker = ValueTracker(6.0)

        f_1 = f_0_tracker.get_value() + bw_tracker.get_value()
        x_1 = 1

        ax_scale = 0.6
        f_ax_next = Axes(
            x_range=[0, x_1, 0.2],
            y_range=[
                f_0_tracker.get_value() - bw_tracker.get_value() / 2.0,
                f_1 + bw_tracker.get_value(),
                0.5,
            ],
            # y_range=[-1, 1],
            tips=False,
            axis_config={"include_numbers": False},
        ).scale(ax_scale)

        labels_next = f_ax_next.get_axis_labels(
            Tex("$t$", font_size=DEFAULT_FONT_SIZE),
            Tex("$f$", font_size=DEFAULT_FONT_SIZE),
        )

        tx_next = f_ax_next.plot(
            lambda t: (signal.sawtooth(2 * PI * f_tracker.get_value() * t) + 1)
            / 2
            * bw_tracker.get_value()
            + f_0_tracker.get_value(),
            use_smoothing=False,
            x_range=[0, x_1 - x_1 / 1000, x_1 / 1000],
            color=TX_COLOR,
        )

        self.play(
            LaggedStart(
                FadeOut(
                    amp_ax_group,
                    no_mod_title,
                    dash_title[0],
                    sine_amp_graph,
                    triangular_amp_graph,
                    fsk_amp_graph,
                    sawtooth_amp_graph,
                    sine_f_graph,
                    lfm_title[0],
                    shift=UP,
                ),
                AnimationGroup(
                    Transform(f_ax, f_ax_next),
                    Transform(sawtooth_f_graph, tx_next),
                    Transform(f_labels, labels_next),
                ),
                lag_ratio=0.2,
            )
        )
        self.wait(2)


class CWandPulsedWaveformComparison(Scene):
    def construct(self):
        x_max = 4
        x_len = 6
        y_len = 3
        step = 0.001
        cw_ax = Axes(
            x_range=[-0.1, x_max, 1],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_corner(UR)

        pulsed_ax = Axes(
            x_range=[-0.1, x_max, 0.5],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        ).to_corner(UL)

        f = 4
        cw_sine = lambda t: np.sin(2 * PI * f * t)
        pulsed_sine = lambda t: np.sin(2 * PI * f * t) * (
            (signal.square(2 * PI * t / 2, duty=0.3) + 1) / 2
        )
        sq = lambda t: (signal.square(2 * PI * t / 2, duty=0.3) + 1) / 2

        cw_graph = cw_ax.plot(cw_sine, x_range=[0, x_max, step], use_smoothing=False)
        pulsed_graph = pulsed_ax.plot(
            pulsed_sine, x_range=[0, x_max, step], use_smoothing=False
        )
        sq_graph = pulsed_ax.plot(
            sq, x_range=[0, x_max - step, step], use_smoothing=False, color=YELLOW
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(pulsed_ax),
                    Create(cw_ax),
                ),
                AnimationGroup(
                    Create(pulsed_graph),
                    Create(sq_graph),
                    Create(cw_graph),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.wait(2)


# Possibly move to pulsed / generic radar video
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


# I am trying to keep the wave and plot signals the same width but I want the signal to be received within the same PRI
# This means the
# Could make small wave then just shift it with its phase constantly shifting
class TransmissionTest2(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        self.add(radar.vgroup.shift(LEFT * 5 + DOWN * 2).scale(0.7))

        t_tracker = ValueTracker(0)

        pw = 1
        dwell_time = 3
        extended_dwell = 8
        n_prts = 2
        x_max = n_prts * (pw + dwell_time)
        ax = Axes(
            x_range=[-0.1, x_max + extended_dwell, 0.5],
            y_range=[-2, 2, 0.5],
            tips=False,
            axis_config={"include_numbers": True},
        ).add_coordinates()
        labels = ax.get_axis_labels(
            Tex("Time", font_size=DEFAULT_FONT_SIZE),
            Tex("Frequency", font_size=DEFAULT_FONT_SIZE),
        )

        f = 2.5
        pulsed_func = (
            lambda t: np.sin(2 * PI * f * t)
            if int(t) % (pw + dwell_time + extended_dwell) == 0
            else 0
        )
        graph = always_redraw(
            lambda: ax.plot(
                pulsed_func,
                x_range=[0, min(t_tracker.get_value(), x_max + extended_dwell), 0.001],
                use_smoothing=False,
            )
        )

        # Propagation
        wave_buff = 0.3
        p1 = radar.radome.get_corner(RIGHT) + RIGHT * wave_buff
        p2 = p1.copy() + 7 * RIGHT + 2 * UP
        line = Line(p1, p2)

        x_max_wave = math.sqrt(np.sum(np.power(p2 - p1, 2)))
        f_wave = f * x_max_wave / x_max
        # pw_wave = pw * x_max_wave / x_max
        ref_wave = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f_wave * t), x_range=[0, x_max_wave]
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
        )
        tx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f_wave * t),
                # x_range=[0, x_max_wave],
                x_range=[
                    max(
                        min(
                            (t_tracker.get_value() - pw) * x_max_wave / x_max,
                            x_max_wave,
                        ),
                        0,
                    ),
                    min(t_tracker.get_value() * x_max_wave / x_max, x_max_wave),
                ],
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
        )
        rx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f_wave * t),
                # x_range=[0, x_max_wave],
                x_range=[
                    min(
                        max(
                            (t_tracker.get_value() - x_max - pw) * x_max_wave / x_max, 0
                        ),
                        x_max_wave,
                    ),
                    min(
                        max((t_tracker.get_value() - x_max) * x_max_wave / x_max, 0),
                        x_max_wave,
                    ),
                ],
                color=BLUE,
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
            .rotate(math.pi, about_point=ref_wave.get_center())
            # .shift(DOWN)
        )

        ax_group = VGroup(ax, labels, graph).scale(0.4).to_corner(UL)

        self.add(ax_group, tx, rx)  # , Dot(p2), Dot(ref_wave.get_center()))

        self.play(t_tracker.animate.set_value(x_max * 3), run_time=4, rate_func=linear)

        self.wait(2)


# Makes a pulsed transmission
class TransmissionTest3(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        self.add(radar.vgroup.scale(0.7).to_edge(LEFT, buff=MED_LARGE_BUFF))

        t_tracker = ValueTracker(0)

        pw = 1.5
        f = 2

        # Propagation
        wave_buff = 0.3
        p1 = radar.radome.get_corner(RIGHT) + RIGHT * wave_buff
        p2 = p1.copy() + 7 * RIGHT
        p2 = (
            Dot()
            .to_edge(RIGHT, buff=LARGE_BUFF)
            .shift(radar.radome.get_corner(RIGHT) * UP)
            .get_center()
        )
        line = Line(p1, p2)

        x_max = math.sqrt(np.sum(np.power(p2 - p1, 2)))
        ref_wave = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t), x_range=[0, x_max]
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
        )

        dot1 = always_redraw(
            lambda: Dot(
                [
                    # max(
                    #     min(
                    #         (t_tracker.get_value() % (2 * x_max) - pw),
                    #         x_max,
                    #     ),
                    #     0,
                    # ),
                    min(
                        max(((t_tracker.get_value() % (2 * x_max)) - x_max - pw), 0),
                        x_max,
                    ),
                    0,
                    0,
                ],
                color=RED,
            ).shift(radar.radome.get_right())
        )
        dot2 = always_redraw(
            lambda: Dot(
                [
                    # min((t_tracker.get_value() % (2 * x_max)), x_max),
                    min(
                        max(((t_tracker.get_value() % (2 * x_max)) - x_max), 0),
                        x_max,
                    ),
                    0,
                    0,
                ],
                color=BLUE,
            ).shift(radar.radome.get_right())
        )
        x_max_tbuff = x_max * 1.2
        tx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=[
                    max(
                        min(
                            (t_tracker.get_value() % (2 * x_max_tbuff) - pw),
                            x_max,
                        ),
                        0,
                    ),
                    min((t_tracker.get_value() % (2 * x_max_tbuff)), x_max),
                ],
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
        )
        rx = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t + PI),
                x_range=[
                    min(
                        max(
                            ((t_tracker.get_value() % (2 * x_max_tbuff)) - x_max),
                            0,
                        ),
                        x_max,
                    ),
                    min(
                        max(
                            ((t_tracker.get_value() % (2 * x_max_tbuff)) - x_max + pw),
                            0,
                        ),
                        x_max,
                    ),
                ],
                color=BLUE,
            )
            .shift(p1)
            .rotate(line.get_angle(), about_point=p1)
            # .rotate(math.pi, about_point=ref_wave.get_center())
        )

        self.add(
            tx,
            rx,
            # dot1,
            # dot2,
            # Dot(p2),
            # Dot(ref_wave.get_center()),
        )

        runs = 1
        self.play(
            t_tracker.animate.increment_value(x_max_tbuff * runs * 2),
            run_time=runs * 2,
            rate_func=linear,
        )
        self.wait(1)
        runs = 3
        self.play(
            t_tracker.animate.increment_value(x_max_tbuff * runs * 2),
            run_time=runs * 2,
            rate_func=linear,
        )

        self.wait(2)


class TransmissionTest3CW(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        self.add(radar.vgroup.scale(0.7).to_edge(LEFT, buff=MED_LARGE_BUFF))

        t_cw_tracker = ValueTracker(0)

        f = 2

        # Propagation
        wave_buff = 0.3
        p1_cw = radar.radome.get_corner(RIGHT) + RIGHT * wave_buff
        p2_cw = p1_cw.copy() + 7 * RIGHT
        line_cw = Line(p1_cw, p2_cw)

        x_max = math.sqrt(np.sum(np.power(p2_cw - p1_cw, 2)))
        tx_cw = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=np.array([-min(t_cw_tracker.get_value(), x_max), 0])
                - max(t_cw_tracker.get_value() - x_max, 0),
                color=RED,
            )
            .shift(p1_cw)
            .rotate(line_cw.get_angle(), about_point=p1_cw)
            .shift(t_cw_tracker.get_value() * RIGHT)
        )
        rx_cw = always_redraw(
            lambda: FunctionGraph(
                lambda t: 0.5 * np.sin(2 * PI * f * t),
                x_range=np.array(
                    [x_max, x_max + max(t_cw_tracker.get_value() - x_max, 0)]
                )
                + [max(t_cw_tracker.get_value() - 2 * x_max, 0), 0],
                color=BLUE,
            )
            .shift(p1_cw)
            .rotate(line_cw.get_angle(), about_point=p1_cw)
            .shift(max(t_cw_tracker.get_value() - x_max, 0) * LEFT)
        )

        self.add(tx_cw, rx_cw)

        self.play(
            t_cw_tracker.animate.set_value(x_max * 4), run_time=4, rate_func=linear
        )

        self.wait(2)


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


class WeatherRadarMoveTest(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        self.add(radar.vgroup)
        # self.play(radar.get_animation())
        self.play(radar.vgroup.animate.shift(LEFT * 4))
        self.play(radar.vgroup.animate.scale(1.5))
        self.wait(2)


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


class WeatherChannel(Scene):
    def construct(self):
        radar = SVGMobject(
            "./figures/weather-radar.svg",
            stroke_color=WHITE,
            color=WHITE,
            fill_color=WHITE,
            opacity=1,
            stroke_width=0.01,
        ).scale(2)

        # weather_channel =

        cap = cv2.VideoCapture("./figures/images/weather_channel_round.gif")
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        weather_channel = ImageMobject(frame).scale(2).to_corner(UR, buff=LARGE_BUFF)
        weather_channel_rect = SurroundingRectangle(
            weather_channel, corner_radius=0.5, buff=0
        )

        self.play(Create(radar))
        self.play(radar.animate.shift(LEFT * 5 + DOWN * 2).scale(0.75))

        pointer = Arrow(
            radar.get_corner(RIGHT),
            weather_channel.get_corner(DL),
        )

        self.play(
            GrowFromCenter(weather_channel),
            Create(weather_channel_rect),
            Create(pointer),
        )

        self.remove(weather_channel)
        while flag:
            flag, frame = cap.read()
            if flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                weather_channel = (
                    ImageMobject(frame).scale(1.5).to_corner(UR, buff=LARGE_BUFF)
                )
                self.add(weather_channel)
                self.wait(0.12)
                self.remove(weather_channel)

        self.play(FadeOut(weather_channel, weather_channel_rect, pointer))

        self.wait(2)


class FlashSine(Scene):
    def construct(self):
        sine = FunctionGraph(
            lambda x: np.sin(2 * PI * 4 * x), x_range=[-3, 3, 0.01], use_smoothing=False
        )

        self.add(sine)

        self.play(Circumscribe(sine))
        self.wait()


# Possibly move to another project about pulsed / generic radar
class SignalGettingDirty(Scene):
    def construct(self):
        f = 1
        A = 1
        phi = ValueTracker(0.0)
        x_shift = ValueTracker(0.0)

        def func(t):
            output = A * np.sin(2 * PI * f * t + phi.get_value())
            if t > 8.0:
                output += A / 2 * np.sin(2 * PI * (f * 3) * t + phi.get_value())
            if t > 12.0:
                output += A / 2 * np.sin(2 * PI * (f / 3) * t + phi.get_value())
            return output

        wave = always_redraw(
            lambda: FunctionGraph(
                func,
                x_range=[-4 + x_shift.get_value(), 4 + x_shift.get_value()],
                use_smoothing=False,
            ).shift(LEFT * x_shift.get_value())
        )

        high_noise = (
            Tex(r"$\sin{(2\pi 3ft)}$").next_to(wave, direction=UP).shift(RIGHT * 4)
        )
        low_noise = (
            Tex(r"$\sin{\left(2\pi \frac{f}{3}t\right)}$")
            .next_to(wave, direction=UP)
            .shift(RIGHT * 4)
        )

        self.add(wave)
        x_shift_final = 20
        run_time = 8
        s_per_x = run_time / x_shift_final
        self.play(
            LaggedStartMap(
                (0.0, x_shift.animate.set_value(x_shift_final)),
                (4.0 * s_per_x, GrowFromCenter(high_noise)),
                rate_func=linear,
                run_time=run_time,
            )
        )

        self.wait(2)


class TriangularMod(Scene):
    def construct(self):
        carrier_freq = 10  # Carrier frequency in Hz
        modulation_freq = 0.5  # Modulation frequency in Hz
        modulation_index = 20  # Modulation index
        duration = 1  # Duration of the signal in seconds
        fs = 1000
        modulating_signal = lambda t: modulation_index * np.arcsin(
            np.sin(2 * np.pi * modulation_freq * t)
        )
        modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        amplitude = lambda t: np.sin(2 * np.pi * modulating_cumsum(t))

        ax = Axes(x_range=[0, 1], y_range=[0, 30])

        triangular_graph = ax.plot(modulating_signal, x_range=[0, duration, 1 / fs])
        graph = ax.plot(modulating_cumsum, x_range=[0, duration, 1 / fs])
        amp_graph = ax.plot(amplitude, x_range=[0, duration, 1 / fs])

        self.add(ax, graph, amp_graph, triangular_graph)


class FMCWRadarBlock(Scene):
    def construct(self):
        fmcw_radar = FMCWRadarCartoon()

        self.play(fmcw_radar.get_animation())
        self.wait(0.5)
        self.play(fmcw_radar.vgroup.animate.shift(UR * 2))
        self.wait(2)


class BGColorTest(Scene):
    def construct(self):
        fmcw_radar = FMCWRadarCartoon()
        # fmcw_radar.vgroup.scale(0.7)

        c1 = ManimColor.from_hex("#253f4b")
        c2 = ManimColor.from_hex("#183340")  # Winner
        c3 = ManimColor.from_hex("#0f2d3b")
        c4 = ManimColor.from_hex("#15455c")

        g1 = VGroup(
            SurroundingRectangle(
                fmcw_radar.vgroup.copy(),
                color=c1,
                buff=LARGE_BUFF,
                fill_color=c1,
                fill_opacity=1,
            ),
            fmcw_radar.vgroup.copy(),
        ).to_edge(LEFT, buff=0)
        g2 = VGroup(
            SurroundingRectangle(
                fmcw_radar.vgroup.copy(),
                color=c2,
                buff=LARGE_BUFF,
                fill_color=c2,
                fill_opacity=1,
            ),
            fmcw_radar.vgroup.copy(),
        ).next_to(g1, buff=0)

        g3 = VGroup(
            SurroundingRectangle(
                fmcw_radar.vgroup.copy(),
                color=c3,
                buff=LARGE_BUFF,
                fill_color=c3,
                fill_opacity=1,
            ),
            fmcw_radar.vgroup.copy(),
        ).next_to(g2, buff=0)
        g4 = VGroup(
            SurroundingRectangle(
                fmcw_radar.vgroup.copy(),
                color=c4,
                buff=LARGE_BUFF,
                fill_color=c4,
                fill_opacity=1,
            ),
            fmcw_radar.vgroup.copy(),
        ).next_to(g3, buff=0)

        self.add(VGroup(g1, g2, g3, g4).scale_to_fit_width(14).to_edge(LEFT, buff=0))


""" End screen """

""" Thumbnail """


class Thumbnail(Scene):
    def construct(self):
        text = Tex("FMCW Radar").scale(2)

        carrier_freq = 1  # Carrier frequency in Hz
        modulation_freq = 0.5  # Modulation frequency in Hz
        modulation_index = 30  # Modulation index
        duration = 1
        fs = 10000
        A = 0.2

        triangular_modulating_signal = lambda t: modulation_index * np.arcsin(
            np.sin(2 * np.pi * modulation_freq * t + PI / 2)
        )
        triangular_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(triangular_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        triangular_amp = lambda t: A * np.sin(
            2 * np.pi * triangular_modulating_cumsum(t)
        )

        triangular_f_graph = FunctionGraph(
            triangular_modulating_signal,
            x_range=[-1, 1, 1 / fs],
            use_smoothing=False,
        ).scale_to_fit_width(12)
        triangular_amp_graph = (
            FunctionGraph(
                triangular_amp,
                x_range=[0.5, 1.5, 1 / fs],
                use_smoothing=False,
                color=BLUE,
            )
            .scale_to_fit_width(13)
            .move_to(ORIGIN)
        )

        self.add(
            triangular_amp_graph,
            triangular_f_graph,
        )
