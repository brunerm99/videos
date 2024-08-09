# channel/profile_picture.py

from manim import *
import numpy as np

config["pixel_width"] = 300
config["pixel_height"] = 300

BACKGROUND_COLOR = ManimColor.from_hex("#183340")
config.background_color = BACKGROUND_COLOR


class ProfilePicture(Scene):
    def construct(self):
        sigma = MathTex(
            "\lambda",
            # tex_template=TexFontTemplates.auriocus_kalligraphicus,
            tex_template=TexFontTemplates.baskervald_adf_fourier,
            color=ORANGE,
        ).scale(26)

        self.add(sigma)
