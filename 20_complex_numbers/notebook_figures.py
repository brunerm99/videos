# notebook_figures.py
import os
import pathlib

import numpy as np
from dotenv import load_dotenv
from manimgx import *
from manimgx import Capacitor as Cap

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = "Maple Mono"
# Light palette to match the public notebook page, not the video.
BACKGROUND_COLOR = "#FFFFFF"
WIRE_COLOR = "#2B3A45"
NODE_COLOR = "#D98E04"
IMPEDANCE_COLOR = "#1B9E77"
PLUS_COLOR = "#D1495B"
MINUS_COLOR = "#1E6FD9"


def sallen_key(scene, label_style):
    wire_color = WIRE_COLOR
    wire_style = dict(color=wire_color, stroke_width=4)
    unit = fh(scene, 0.11)
    center = DOWN * fh(scene, 0.08)
    node1 = center + unit * np.array([-3.2, 0.6, 0])
    node2 = center + unit * np.array([0, 0.6, 0])
    output_node = center + RIGHT * 5.5 * unit

    input_port = Circle(
        radius=0.10 * unit,
        fill_color=BACKGROUND_COLOR,
        fill_opacity=1,
        **wire_style,
    ).move_to(center + unit * np.array([-6.4, 0.6, 0]))
    output_port = input_port.copy().move_to(center + RIGHT * 6.4 * unit)
    r1 = (
        Resistor(
            length=1.25 * unit,
            amplitude=0.12 * unit,
            include_ports=True,
            **wire_style,
        )
        .rotate(-PI / 2)
        .move_to(center + unit * np.array([-4.8, 0.6, 0]))
    )
    r2 = r1.copy().move_to(center + unit * np.array([-1.6, 0.6, 0]))
    c2 = Cap(
        plate_width=fw(scene, 0.068) * unit,
        plate_thickness=fh(scene, 0.006) * unit,
        gap=fh(scene, 0.035) * unit,
        stroke_color=wire_color,
    ).move_to(center + DOWN * unit)
    c1 = c2.copy().rotate(-PI / 2).move_to(center + UP * 2.7 * unit)
    ground = (
        VGroup(
            *[
                Line(width / 2 * LEFT, width / 2 * RIGHT, **wire_style)
                for width in (0.6 * unit, 0.375 * unit, 0.15 * unit)
            ]
        )
        .arrange(DOWN, buff=0.105 * unit)
        .move_to(center + DOWN * 2.35 * unit)
    )
    amp = Polygon(
        center + unit * np.array([2.3, 1.4, 0]),
        center + unit * np.array([2.3, -1.4, 0]),
        center + RIGHT * 4.6 * unit,
        fill_color=BACKGROUND_COLOR,
        fill_opacity=1,
        **wire_style,
    )
    input_signs = Text("+\n-", font=FONT).scale(0.85 * unit)
    input_signs[0].set_color(PLUS_COLOR).move_to(
        center + unit * np.array([2.65, 0.6, 0])
    )
    input_signs[1].set_color(MINUS_COLOR).move_to(
        center + unit * np.array([2.65, -0.6, 0])
    )
    wires = VGroup(
        Line(input_port.get_right(), r1.p1, **wire_style),
        Line(r1.p2, r2.p1, **wire_style),
        Line(r2.p2, node2, **wire_style),
        Line(node2, c2.p2, **wire_style),
        Line(ground[0].get_center(), c2.p1, **wire_style),
        Line(node2, center + unit * np.array([2.3, 0.6, 0]), **wire_style),
        Line(center + RIGHT * 4.6 * unit, output_port.get_left(), **wire_style),
        VMobject(**wire_style).set_points_as_corners(
            [node1, node1 + UP * 2.1 * unit, c1.p1]
        ),
        VMobject(**wire_style).set_points_as_corners(
            [c1.p2, center + unit * np.array([5.5, 2.7, 0]), output_node]
        ),
        VMobject(**wire_style).set_points_as_corners(
            [
                center + unit * np.array([2.3, -0.6, 0]),
                center + unit * np.array([1.55, -0.6, 0]),
                center + unit * np.array([1.55, -1.95, 0]),
                center + unit * np.array([5.5, -1.95, 0]),
                output_node,
            ]
        ),
    )
    nodes = VGroup(
        *[
            Dot(point, radius=0.06 * unit, color=wire_color)
            for point in (node1, node2, output_node)
        ]
    )
    node_dots = VGroup(
        *[
            Dot(point, radius=0.1 * unit, color=NODE_COLOR).set_z_index(10)
            for point in (node1, node2)
        ]
    )
    node1_label = (
        MathTex(r"\tilde V_1", color=NODE_COLOR)
        .scale(0.9)
        .next_to(nodes[0], DOWN, MED_SMALL_BUFF)
    )
    node2_label = (
        MathTex(r"\tilde V_2", color=NODE_COLOR)
        .scale(0.9)
        .next_to(nodes[1], UP, MED_SMALL_BUFF)
    )
    input_label = (
        MathTex(r"\tilde V_{in}", color=NODE_COLOR)
        .scale(0.9)
        .next_to(input_port, UP, MED_LARGE_BUFF)
    )
    output_label = (
        MathTex(r"\tilde V_{out}", color=NODE_COLOR)
        .scale(0.9)
        .next_to(output_port, UP, MED_LARGE_BUFF)
    )

    if label_style == "components":
        labels = VGroup(
            MathTex("R_1", color=wire_color).scale(0.85).next_to(r1, UP, 0.3),
            MathTex("R_2", color=wire_color).scale(0.85).next_to(r2, UP, 0.3),
            MathTex("C_1", color=wire_color).scale(0.85).next_to(c1, UP, 0.25),
            MathTex("C_2", color=wire_color).scale(0.85).next_to(c2, LEFT, 0.3),
        )
    else:
        labels = VGroup(
            MathTex(r"Z_{R_1}=R_1", color=IMPEDANCE_COLOR)
            .scale(0.7)
            .next_to(r1, UP, 0.3),
            MathTex(r"Z_{R_2}=R_2", color=IMPEDANCE_COLOR)
            .scale(0.7)
            .next_to(r2, UP, 0.3),
            MathTex(r"Z_{C_1}=\frac{1}{j\omega C_1}", color=IMPEDANCE_COLOR)
            .scale(0.7)
            .next_to(c1, UP, 0.25),
            MathTex(r"Z_{C_2}=\frac{1}{j\omega C_2}", color=IMPEDANCE_COLOR)
            .scale(0.7)
            .next_to(c2, LEFT, 0.3),
        )

    return VGroup(
        wires,
        ground,
        input_port,
        output_port,
        r1,
        r2,
        c1,
        c2,
        amp,
        input_signs,
        nodes,
        node_dots,
        labels,
        node1_label,
        node2_label,
        input_label,
        output_label,
    )


class SallenKeyComponents(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        circuit = sallen_key(self, "components")
        self.add(circuit)
        self.camera.frame.scale_to_fit_width(circuit.width * 1.15).move_to(circuit)


class SallenKeyImpedances(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        circuit = sallen_key(self, "impedances")
        self.add(circuit)
        self.camera.frame.scale_to_fit_width(circuit.width * 1.15).move_to(circuit)
