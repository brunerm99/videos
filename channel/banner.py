# channel/banner.py

from manim import *
import numpy as np
import sys

sys.path.insert(0, "..")
from props import WeatherRadarTower

config["pixel_width"] = 2048
config["pixel_height"] = 1152

BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR


class Banner(Scene):
    def construct(self):
        radar = WeatherRadarTower()
        radar.vgroup.scale(0.4).to_edge(LEFT, buff=LARGE_BUFF)

        A = 0.5
        f = 2
        fs = 1000
        wave = FunctionGraph(
            lambda t: 0.5 * np.sin(2 * PI * f * t), x_range=[0, 2, 1 / fs], color=RED
        ).next_to(radar.radome, direction=RIGHT)

        title_scale = 2
        title_top = (
            Tex(r"Radar and RF").scale(title_scale).next_to(wave, direction=RIGHT)
        )
        title_bot = (
            Tex("Animations")
            .scale(title_scale)
            .next_to(title_top, direction=DOWN, buff=MED_SMALL_BUFF)
        )
        title = VGroup(title_top, title_bot)
        wave2 = wave.copy().next_to(title_top, direction=RIGHT)

        self.add(radar.vgroup, wave, wave2, title)
