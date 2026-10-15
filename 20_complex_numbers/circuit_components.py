# circuit_components.py
import os
import pathlib

from dotenv import load_dotenv
from manimgx import *

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = os.getenv("FONT", "").replace("Maple Mono CN", "Maple Mono")
BACKGROUND_COLOR = "#183340"


class CircuitComponents(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        capacitor = Capacitor()
        resistor = Resistor()
        inductor = Inductor()
        parts = VGroup(capacitor, resistor, inductor).arrange(
            RIGHT, buff=fw(self, 0.16)
        )
        with_ports = VGroup(
            Capacitor(include_ports=True),
            Resistor(include_ports=True),
            Inductor(include_ports=True),
        ).arrange(RIGHT, buff=fw(self, 0.16))
        VGroup(parts, with_ports).arrange(DOWN, buff=fh(self, 0.18))
        labels = VGroup(
            *[
                Text(name, font=FONT, font_size=30, color="#D6EEF7").next_to(
                    part, UP, buff=fh(self, 0.08)
                )
                for name, part in zip(("Capacitor", "Resistor", "Inductor"), parts)
            ]
        )
        self.next_section("Bodies")
        self.play(LaggedStart(*(Create(part) for part in parts), lag_ratio=0.2))
        self.play(FadeIn(labels))
        self.wait(1)

        self.next_section("With ports")
        self.play(LaggedStart(*(Create(part) for part in with_ports), lag_ratio=0.2))
        self.wait(1)
