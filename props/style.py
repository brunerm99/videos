from manim import *


CITRINE = ManimColor.from_hex("#D7CF07")
MINDARO = ManimColor.from_hex("#D9E76C")
PIGMENT_GREEN = ManimColor.from_hex("#4DAA57")
LIGHT_GREEN = ManimColor.from_hex("#7CEA9C")
AUREOLIN = ManimColor.from_hex("#F4E409")
SEPIA = ManimColor.from_hex("#5B3000")


TX_COLOR = BLUE
RX_COLOR = RED
GAIN_COLOR = GREEN
IF_COLOR = ORANGE
FILTER_COLOR = GREEN


BACKGROUND_COLOR = ManimColor.from_hex("#183340")

COLOR_PALETTE = {
    "TX_COLOR": TX_COLOR,
    "RX_COLOR": RX_COLOR,
    "IF_COLOR": IF_COLOR,
    "FILTER_COLOR": FILTER_COLOR,
    "GAIN_COLOR": GAIN_COLOR,
    "BACKGROUND_COLOR": BACKGROUND_COLOR,
    "CITRINE": CITRINE,
    "MINDARO": MINDARO,
    "PIGMENT_GREEN": PIGMENT_GREEN,
    "LIGHT_GREEN": LIGHT_GREEN,
    "AUREOLIN": AUREOLIN,
    "SEPIA": SEPIA,
}


class ColorPalette(Scene):
    def construct(self):
        colors = VGroup()
        for name, color in COLOR_PALETTE.items():
            colors.add(
                VGroup(
                    Square(stroke_opacity=0, fill_opacity=1, fill_color=color),
                    Text(name).scale(0.5),
                ).arrange(UP, buff=SMALL_BUFF)
            )

        colors.arrange_in_grid(rows=4, cols=4).scale_to_fit_width(
            config["frame_width"] * 0.8
        )
        self.add(colors)
