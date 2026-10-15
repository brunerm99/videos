# arrow_styles.py
import os
import pathlib

from dotenv import load_dotenv
from manimgx import *

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = os.getenv("FONT", "").replace("Maple Mono CN", "Maple Mono")
BACKGROUND_COLOR = "#183340"


class ArrowStyles(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        length = fw(self, 0.3)
        default = Arrow(
            ORIGIN,
            RIGHT * length,
            buff=0,
            thickness=8,
            color=BLUE_B,
        )
        rounded = Arrow(
            ORIGIN,
            RIGHT * length,
            buff=0,
            thickness=8,
            tip_corner_radius=0.06,
            color=YELLOW,
        )
        bent = Arrow(
            ORIGIN,
            RIGHT * length,
            buff=0,
            thickness=8,
            path_arc=PI / 2,
            tip_corner_radius=0.06,
            color=BLUE_B,
        )
        curved = CurvedArrow(
            ORIGIN,
            RIGHT * length,
            angle=PI / 2,
            stroke_width=8,
            tip_length=0.5,
            tip_corner_radius=0.06,
            color=YELLOW,
        )
        arrows = VGroup(default, rounded, bent, curved)
        cells = VGroup()
        labels = VGroup()
        for arrow, name in zip(
            arrows,
            (
                "Arrow (default)",
                "tip_corner_radius=0.06",
                "Arrow(path_arc=PI/2)",
                "CurvedArrow",
            ),
        ):
            label = Text(name, font=FONT, font_size=26).next_to(arrow, UP, 0.5)
            labels.add(label)
            cells.add(VGroup(arrow, label))
        cells.arrange_in_grid(rows=2, buff=(1.5, 1.6)).move_to(ORIGIN)
        self.add(labels)
        self.next_section("Straight arrows", skip_animations=False)
        self.play(GrowArrow(default), GrowArrow(rounded), run_time=2)
        self.wait(0.5)
        self.next_section("Curved growth", skip_animations=False)
        self.play(GrowArrow(bent), GrowArrow(curved), run_time=2)
        self.wait(1)


class ArrowGrowth3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.set_camera_orientation(phi=65 * DEGREES, theta=-70 * DEGREES)
        arrows = VGroup(
            *(
                Arrow3D(
                    ORIGIN,
                    direction * 3,
                    shaft_radius=0.05,
                    tip_radius=0.14,
                    tip_length=0.32,
                    color=color,
                )
                for direction, color in ((RIGHT, RED), (UP, GREEN), (OUT, BLUE_B))
            )
        )
        self.next_section("3D growth", skip_animations=False)
        self.play(
            LaggedStart(*(GrowArrow(a) for a in arrows), lag_ratio=0.2), run_time=3
        )
        self.wait(1)


class SquigglyArrowGrowth(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        arrow = SquigglyLine(
            LEFT * fw(self, 0.3),
            RIGHT * fw(self, 0.3),
            amplitude=0.15,
            wavelength=1,
            seed=7,
            color=YELLOW,
            stroke_width=8,
        ).add_tip(tip_length=0.5)
        self.play(GrowArrow(arrow), run_time=3)
        self.wait(1)
