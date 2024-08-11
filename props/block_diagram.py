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
        ]
    }


def get_bd_animation(bd, lagged: bool = False, lag_ratio=0.8):
    animations = (
        *[
            Create(block)
            if type(block) in (Line, CubicBezier)
            else GrowFromCenter(block)
            for block in bd
        ],
    )
    if lagged:
        return LaggedStart(animations, lag_ratio=lag_ratio)
    return Succession(animations)
