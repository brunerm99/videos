# props/block_diagram.py

from manim import *


def get_blocks(color=WHITE):
    return {
        name: SVGMobject(
            f"../props/static/{name}.svg",
            fill_color=color,
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
            "bp_filter_generic",
            "mixer",
            "phase_detector",
            "amp",
            "phase_shifter",
            "spdt_switch",
            "lp_filter",
            "hp_filter",
            "antenna",
            "splitter",
            "oscillator",
            "limiter",
        ]
    }


def get_bd_animation(bd, lagged: bool = False, lag_ratio=0.8, run_time=None):
    animations = (
        *[
            Create(block)
            if type(block) in (Line, CubicBezier)
            else GrowFromCenter(block)
            for block in bd
        ],
    )
    if lagged:
        return LaggedStart(animations, lag_ratio=lag_ratio, run_time=run_time)
    return Succession(animations, run_time=run_time)


class ShowBlocks(Scene):
    def construct(self):
        blocks = get_blocks()
        labeled_blocks = Group()
        for name, block in blocks.items():
            label = Text(name).next_to(block, direction=UP, buff=SMALL_BUFF)
            labeled_blocks.add(Group(label, block))
        self.add(
            labeled_blocks.arrange_in_grid(4, 6).scale_to_fit_width(
                config["frame_width"] - 1
            )
        )


def get_diode():
    diode_tri = Triangle(color=WHITE).rotate(PI / 3)
    diode_line = Line(LEFT * diode_tri.width / 2, RIGHT * diode_tri.width / 2).next_to(
        diode_tri, direction=DOWN, buff=0
    )
    diode_line_dot_1 = Dot(
        radius=DEFAULT_DOT_RADIUS / 4,
        fill_color=WHITE,
        fill_opacity=1,
        stroke_color=WHITE,
    ).move_to(diode_line.get_start())
    diode_line_dot_2 = Dot(
        radius=DEFAULT_DOT_RADIUS / 2,
        fill_color=WHITE,
        fill_opacity=1,
        stroke_color=WHITE,
    ).move_to(diode_line.get_end())
    diode_conn_1 = Line(diode_tri.get_top() + UP / 2, diode_tri.get_top())
    diode_conn_2 = Line(diode_tri.get_bottom() + DOWN / 2, diode_tri.get_bottom())
    diode = VGroup(
        diode_tri,
        diode_line,
        diode_line_dot_1,
        diode_line_dot_2,
        diode_conn_1,
        diode_conn_2,
    )

    return diode


def get_resistor():
    rotation = -PI / 4
    res_line_start = Line(LEFT / 2, ORIGIN).rotate(-rotation / 2)
    res_line_start_inv = Line(ORIGIN, RIGHT / 2).rotate(-rotation / 2)
    res_line_1 = VGroup(res_line_start, res_line_start_inv)
    res_line_2 = (
        Line(LEFT / 2, RIGHT / 2)
        .rotate(rotation / 2)
        .next_to(res_line_1, direction=DOWN, buff=0)
    )
    res_line_3 = (
        Line(LEFT / 2, RIGHT / 2)
        .rotate(-rotation / 2)
        .next_to(res_line_2, direction=DOWN, buff=0)
    )
    res_line_4 = (
        Line(LEFT / 2, RIGHT / 2)
        .rotate(rotation / 2)
        .next_to(res_line_3, direction=DOWN, buff=0)
    )
    res_line_end_inv = Line(LEFT / 2, ORIGIN)
    res_line_end = Line(ORIGIN, RIGHT / 2)
    res_line_5 = (
        VGroup(res_line_end, res_line_end_inv)
        .rotate(-rotation / 2)
        .next_to(res_line_4, direction=DOWN, buff=0)
    )
    res_conn_1 = Line(res_line_start.get_end() + UP / 2, res_line_start.get_end())
    res_conn_2 = Line(res_line_end.get_start() + DOWN / 2, res_line_end.get_start())
    resistor = VGroup(
        res_conn_1,
        res_line_start,
        res_line_2,
        res_line_3,
        res_line_4,
        res_line_end,
        res_conn_2,
    )

    return resistor
