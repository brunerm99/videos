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
