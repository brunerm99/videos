# thumbnails.py
import pathlib

import numpy as np
from dotenv import load_dotenv
from manimgx import *
from manimgx import Capacitor as Cap

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = "Maple Mono"
BACKGROUND_COLOR = "#183340"
WIRE_COLOR = "#D6EEF7"
CIRCLE_COLOR = ORANGE
V_COLOR = YELLOW
I_COLOR = BLUE
J_COLOR = GREEN


def polar(center, radius, angle):
    return center + radius * np.array([np.cos(angle), np.sin(angle), 0])


def sine_wave(start, length, amplitude, cycles, phase=0, color=V_COLOR, width=14):
    return ParametricFunction(
        lambda t: (
            start
            + RIGHT * length * t
            + UP * amplitude * np.sin(TAU * cycles * t + phase)
        ),
        t_range=[0, 1, 1 / 300],
        color=color,
        stroke_width=width,
    ).set_cap_style(CapStyleType.ROUND)


def outlined(mobject, width=14):
    return mobject.set_stroke(BACKGROUND_COLOR, width, background=True)


def big_text(text, color=WHITE, t2c=None):
    return outlined(Text(text, font=FONT, weight=BOLD, color=color, t2c=t2c), 16)


def phasor(center, angle, radius, color, thickness=12, opacity=1):
    arrow = Arrow(
        center,
        polar(center, radius, angle),
        buff=0,
        thickness=thickness,
        tip_width_ratio=3,
        fill_color=color,
        color=color,
        max_width_to_length_ratio=1,
        max_tip_length_to_length_ratio=0.3,
        tip_corner_radius=0.08,
    )
    # The tip is its own submobject, so fade shaft and tip together.
    return arrow.set_fill(opacity=opacity).set_stroke(opacity=opacity).set_z_index(3)


def turn_arrow(center, radius, start_angle, color=WHITE, width=10):
    arc = Arc(
        radius=radius,
        start_angle=start_angle + PI / 2 * 0.15,
        angle=PI / 2 * 0.7,
        arc_center=center,
        color=color,
        stroke_width=width,
        tip_corner_radius=0.08,
    ).set_cap_style(CapStyleType.ROUND)
    arc.add_tip(tip_length=0.5, tip_width=0.46)
    return arc.set_z_index(3)


def plane_axes(center, reach):
    return VGroup(
        Line(center + LEFT * reach, center + RIGHT * reach),
        Line(center + DOWN * reach, center + UP * reach),
    ).set_stroke(WIRE_COLOR, 3, opacity=0.5)


def charge_symbol(positive, radius=0.13):
    disc = Circle(
        radius=radius,
        fill_color=RED if positive else BLUE,
        fill_opacity=1,
        stroke_width=0,
    )
    bar = Line(
        radius * 0.55 * LEFT,
        radius * 0.55 * RIGHT,
        stroke_color=BACKGROUND_COLOR,
        stroke_width=3,
    )
    symbol = VGroup(disc, bar)
    if positive:
        symbol.add(bar.copy().rotate(PI / 2))
    return symbol


def charged_capacitor(scene, scale=1, charges=7):
    cap = Cap(
        plate_width=fw(scene, 0.15) * scale,
        plate_thickness=fh(scene, 0.03) * scale,
        gap=fh(scene, 0.1) * scale,
        include_ports=True,
        stroke_color=WIRE_COLOR,
        stroke_width=6,
    )
    for plate, positive in ((cap.top_plate, True), (cap.bottom_plate, False)):
        symbols = VGroup(
            *[charge_symbol(positive, 0.115 * scale) for _ in range(charges)]
        )
        symbols.arrange(RIGHT, buff=0.07 * scale).move_to(plate).set_z_index(4)
        cap.add(symbols)
    return cap


def wire(start, end):
    return Line(start, end, color=WIRE_COLOR, stroke_width=7).set_cap_style(
        CapStyleType.ROUND
    )


def capacitor_phase_shift(scene):
    """Big capacitor on the left, its current leading its voltage on the right."""
    cap = charged_capacitor(scene, scale=1.9)
    cap.move_to(LEFT * fw(scene, 0.33) + DOWN * fh(scene, 0.03))
    top, bottom = cap.get_end(), cap.get_start()
    wires = VGroup(
        wire(top, [top[0], fh(scene, 0.5) + 0.2, 0]),
        wire(bottom, [bottom[0], -fh(scene, 0.5) - 0.2, 0]),
    )

    start = LEFT * fw(scene, 0.13) + DOWN * fh(scene, 0.06)
    length = fw(scene, 0.67)
    amplitude = fh(scene, 0.27)
    cycles = 1.25
    axis = Line(start + LEFT * 0.2, start + RIGHT * (length + 0.3)).set_stroke(
        WIRE_COLOR, 4, opacity=0.6
    )
    v_wave = sine_wave(start, length, amplitude, cycles, color=V_COLOR, width=18)
    i_wave = sine_wave(
        start, length, amplitude, cycles, phase=PI / 2, color=I_COLOR, width=18
    )

    # Current peaks at t=0, voltage a quarter cycle later.
    i_peak = start + UP * amplitude
    v_peak = start + RIGHT * length / cycles * 0.25 + UP * amplitude
    marker_top = fh(scene, 0.44)
    markers = VGroup(
        DashedLine(i_peak, [i_peak[0], marker_top, 0], dash_length=0.18),
        DashedLine(v_peak, [v_peak[0], marker_top, 0], dash_length=0.18),
    ).set_stroke(WIRE_COLOR, 5, opacity=0.8)
    for marker in markers:
        marker.set_cap_style(CapStyleType.ROUND)
    shift_y = fh(scene, 0.39)
    shift_arrow = DoubleArrow(
        [i_peak[0], shift_y, 0],
        [v_peak[0], shift_y, 0],
        buff=0,
        thickness=9,
        tip_width_ratio=3,
        fill_color=WHITE,
        color=WHITE,
        max_width_to_length_ratio=1,
        max_tip_length_to_length_ratio=0.25,
        tip_corner_radius=0.06,
    )
    shift_label = outlined(
        MathTex(r"90^\circ", color=WHITE).scale_to_fit_height(fh(scene, 0.09))
    ).next_to(shift_arrow, DOWN, buff=0.15)

    return VGroup(cap, wires, axis, v_wave, i_wave, markers, shift_arrow, shift_label)


class ThumbnailForEEs(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        title = VGroup(
            big_text("Complex"),
            big_text("Numbers"),
            big_text("for EEs", color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=fh(self, 0.035))
        title.scale_to_fit_width(fw(self, 0.47)).move_to(LEFT * fw(self, 0.25))

        center = RIGHT * fw(self, 0.26) + DOWN * fh(self, 0.06)
        radius = fh(self, 0.34)
        circle = Circle(radius=radius, color=CIRCLE_COLOR, stroke_width=11).move_to(
            center
        )
        v_arrow = phasor(center, 0, radius, V_COLOR)
        i_arrow = phasor(center, PI / 2, radius, I_COLOR)
        turn = turn_arrow(center, radius * 1.2, 0)
        times_j = outlined(
            MathTex(r"\times j", color=J_COLOR).scale_to_fit_height(fh(self, 0.13))
        )
        times_j.move_to(polar(center, radius * 0.6, PI / 4)).set_z_index(5)

        self.add(plane_axes(center, radius * 1.2), circle, v_arrow, i_arrow)
        self.add(turn, times_j, title)
        self.wait(0.1)


class ThumbnailRotation(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        j = outlined(MathTex("j", color=J_COLOR).scale_to_fit_height(fh(self, 0.52)))
        j.move_to(LEFT * fw(self, 0.3) + UP * fh(self, 0.04))

        center = RIGHT * fw(self, 0.14) + DOWN * fh(self, 0.06)
        radius = fh(self, 0.33)
        circle = Circle(radius=radius, color=CIRCLE_COLOR, stroke_width=11).move_to(
            center
        )
        one = phasor(center, 0, radius, V_COLOR)
        j_arrow = phasor(center, PI / 2, radius, I_COLOR)
        minus_one = phasor(center, PI, radius, RED)
        turns = VGroup(
            turn_arrow(center, radius * 1.2, 0),
            turn_arrow(center, radius * 1.2, PI / 2),
        )

        label_size = fh(self, 0.11)
        labels = VGroup(
            outlined(
                MathTex("1", color=V_COLOR).scale_to_fit_height(label_size)
            ).next_to(one.get_end(), DR, buff=0.2),
            outlined(
                MathTex("j", color=I_COLOR).scale_to_fit_height(label_size)
            ).next_to(j_arrow.get_end(), UP, buff=0.25),
            outlined(
                MathTex("-1", color=RED).scale_to_fit_height(label_size * 0.8)
            ).next_to(minus_one.get_end(), DL, buff=0.2),
        ).set_z_index(5)
        times_j = VGroup(
            *[
                outlined(
                    MathTex(r"\times j", color=J_COLOR).scale_to_fit_height(
                        label_size * 0.95
                    )
                ).move_to(polar(center, radius * 1.2, a) + (UP + side) * 0.75)
                for a, side in ((PI / 4, RIGHT), (3 * PI / 4, LEFT))
            ]
        ).set_z_index(5)

        self.add(plane_axes(center, radius * 1.25), circle, one, j_arrow, minus_one)
        self.add(turns, labels, times_j, j)
        self.wait(0.1)


class ThumbnailPhaseShift(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        j = outlined(MathTex("j", color=J_COLOR).scale_to_fit_height(fh(self, 0.22)))
        j.move_to(RIGHT * fw(self, 0.43) + DOWN * fh(self, 0.34))

        self.add(capacitor_phase_shift(self), j)
        self.wait(0.1)


class ThumbnailWhyJ(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        question = big_text("Why j?", t2c={"j": J_COLOR})
        question.scale_to_fit_height(fh(self, 0.14))
        question.move_to(RIGHT * fw(self, 0.3) + UP * fh(self, 0.36)).set_z_index(5)

        self.add(capacitor_phase_shift(self), question)
        self.wait(0.1)


class ThumbnailProjection(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        center = LEFT * fw(self, 0.24) + DOWN * fh(self, 0.02)
        radius = fh(self, 0.36)
        angle = 0.3 * PI
        point = polar(center, radius, angle)

        circle = Circle(radius=radius, color=CIRCLE_COLOR, stroke_width=11).move_to(
            center
        )
        # Fading trail of earlier arrow positions shows the rotation.
        trail = VGroup(
            *[
                phasor(center, angle - step * 0.22 * PI, radius, WHITE, 9, opacity)
                for step, opacity in ((3, 0.12), (2, 0.25), (1, 0.45))
            ]
        )
        # Trail fades out behind the oldest ghost; the arc stays within the lower
        # half-turn, so a gradient along UP runs along the path.
        trail_arc = Arc(
            radius=radius,
            start_angle=angle - 0.8 * PI,
            angle=0.8 * PI,
            arc_center=center,
            color=V_COLOR,
            stroke_width=11,
        ).set_z_index(1)
        trail_arc.set_sheen_direction(UP).set_stroke(opacity=[0, 1, 1])
        arm = phasor(center, angle, radius, WHITE, 10)
        dot = Dot(point, radius=0.2, color=WHITE).set_z_index(6)

        wave_start = center + RIGHT * radius * 1.32
        wave_length = fw(self, 0.5) - wave_start[0] + 0.5
        cycles = 1.15
        wave = ParametricFunction(
            lambda t: (
                wave_start
                + RIGHT * wave_length * t
                + UP * radius * np.sin(angle + TAU * cycles * t)
            ),
            t_range=[0, 1, 1 / 300],
            color=GREEN,
            stroke_width=16,
        ).set_cap_style(CapStyleType.ROUND)
        wave_axes = VGroup(
            Line(wave_start + DOWN * radius * 1.2, wave_start + UP * radius * 1.2),
            Line(wave_start + LEFT * 0.2, wave_start + RIGHT * (wave_length + 0.3)),
        ).set_stroke(WIRE_COLOR, 4, opacity=0.6)
        wave_point = wave_start + UP * radius * np.sin(angle)
        link = DashedLine(point, wave_point, dash_length=0.22).set_stroke(
            WIRE_COLOR, 6, opacity=0.9
        )
        link.set_cap_style(CapStyleType.ROUND)
        wave_dot = Dot(wave_point, radius=0.18, color=GREEN).set_z_index(6)

        j = outlined(MathTex("j", color=J_COLOR).scale_to_fit_height(fh(self, 0.14)))
        j.next_to(center + UP * radius * 1.2, LEFT, buff=0.15).set_z_index(5)

        self.add(plane_axes(center, radius * 1.2), wave_axes, circle, trail_arc)
        self.add(trail, wave, link, arm, dot, wave_dot, j)
        self.wait(0.1)


class ThumbnailCalculus(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        cap = charged_capacitor(self, scale=1.5)
        cap.rotate(PI / 2).move_to(UP * fh(self, 0.26))
        left, right = cap.get_end(), cap.get_start()
        wires = VGroup(
            wire(left, [-fw(self, 0.5) - 0.2, left[1], 0]),
            wire(right, [fw(self, 0.5) + 0.2, right[1], 0]),
        )

        derivative = MathTex(r"\frac{dV}{dt}").scale_to_fit_height(fh(self, 0.42))
        derivative[0][1].set_color(V_COLOR)
        phasor_form = MathTex(r"j", r"\omega", r"\tilde{V}").scale_to_fit_height(
            fh(self, 0.28)
        )
        phasor_form[0].set_color(J_COLOR)
        phasor_form[2].set_color(V_COLOR)
        row = VGroup(derivative, phasor_form).arrange(RIGHT, buff=fw(self, 0.2))
        row.move_to(DOWN * fh(self, 0.23))
        arrow = Arrow(
            derivative.get_right() + RIGHT * 0.35,
            phasor_form.get_left() + LEFT * 0.35,
            buff=0,
            thickness=12,
            tip_width_ratio=3,
            fill_color=WHITE,
            color=WHITE,
            max_width_to_length_ratio=1,
            max_tip_length_to_length_ratio=0.35,
            tip_corner_radius=0.1,
        )

        self.add(cap, wires, derivative, phasor_form, arrow)
        self.wait(0.1)
