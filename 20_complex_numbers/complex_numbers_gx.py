# complex_numbers_gx.py
import os
import pathlib
from collections import deque

import numpy as np
from dotenv import load_dotenv
from manimgx import *

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = os.getenv("FONT", "").replace("Maple Mono CN", "Maple Mono")
BACKGROUND_COLOR = "#183340"
SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class Capacitor(MovingCameraScene):
    def construct(self):
        wire_color = "#D6EEF7"
        self.camera.background_color = BACKGROUND_COLOR

        amplitude = VT(0)
        frequency = VT(0)
        phase = VT(0)
        capacitor_voltage = VT(0)
        time = VT(0)
        time_scale = VT(1)
        voltage_scale = VT(5)
        plot_window = 12
        voltage_history = deque([(0.0, 0.0)])

        capacitance = 1e-6
        source_resistance = 280_000
        tau = source_resistance * capacitance

        def update_circuit(mob, dt):
            dt *= ~time_scale
            if dt == 0:
                return

            decay = np.exp(-dt / tau)
            if ~frequency == 0:
                voltage = ~amplitude + (~capacitor_voltage - ~amplitude) * decay
            else:
                omega = TAU * ~frequency
                angle = ~phase + omega * dt
                ratio = omega * tau
                steady_start = (
                    ~amplitude
                    * (np.cos(~phase) + ratio * np.sin(~phase))
                    / (1 + ratio**2)
                )
                steady_end = (
                    ~amplitude
                    * (np.cos(angle) + ratio * np.sin(angle))
                    / (1 + ratio**2)
                )
                voltage = steady_end + (~capacitor_voltage - steady_start) * decay
                phase.set_value(angle % TAU)
            capacitor_voltage.set_value(voltage)
            time.increment_value(dt)
            voltage_history.append((~time, voltage))
            voltage_scale.set_value(max(~voltage_scale, abs(~amplitude), abs(voltage)))
            while (
                len(voltage_history) > 2 and voltage_history[1][0] < ~time - plot_window
            ):
                voltage_history.popleft()

        physics = Mobject().add_updater(update_circuit)
        self.add(
            amplitude,
            frequency,
            phase,
            capacitor_voltage,
            time,
            time_scale,
            voltage_scale,
            physics,
        )

        axes = Axes(
            x_range=[0, plot_window, 2],
            y_range=[-1.2, 1.2, 1],
            x_length=fw(self, 0.53),
            y_length=fh(self, 0.60),
            axis_config={
                "color": wire_color,
                "stroke_width": 2,
                "include_numbers": False,
            },
            tips=False,
        )

        def get_voltage_plot():
            start = max(0, ~time - plot_window)
            samples = np.array(voltage_history)
            if samples[0, 0] < start:
                samples[0, 1] = np.interp(start, samples[:, 0], samples[:, 1])
                samples[0, 0] = start
            if len(samples) == 1:
                samples = np.repeat(samples, 2, axis=0)
            return axes.plot_line_graph(
                samples[:, 0] - start,
                samples[:, 1] / ~voltage_scale,
                add_vertex_dots=False,
                line_color=YELLOW,
                stroke_width=4,
            )

        voltage_plot = always_redraw(get_voltage_plot)
        voltage_dot = Dot(radius=0.09, color=YELLOW).set_z_index(5)
        voltage_dot.add_updater(
            lambda mob: mob.move_to(
                axes.c2p(
                    min(~time, plot_window), voltage_history[-1][1] / ~voltage_scale
                )
            ),
            call_updater=True,
        )
        plot = VGroup(axes, voltage_plot, voltage_dot)

        def charge_symbol(positive=False, radius=0.105):
            disc = Circle(
                radius=radius,
                fill_color=RED if positive else BLUE,
                fill_opacity=1,
                stroke_width=0,
            )
            horizontal = Line(
                radius * 0.53 * LEFT,
                radius * 0.53 * RIGHT,
                stroke_color=BACKGROUND_COLOR,
                stroke_width=2,
            )
            symbol = VGroup(disc, horizontal)
            if positive:
                symbol.add(horizontal.copy().rotate(PI / 2))
            return symbol

        cap = (
            VGroup(
                *[
                    Rectangle(
                        width=fw(self, 0.15),
                        height=fh(self, 0.022),
                        fill_color="#B7C9D2",
                        fill_opacity=1,
                        stroke_color=wire_color,
                        stroke_width=2,
                    )
                    for _ in range(2)
                ]
            )
            .arrange(DOWN, buff=fh(self, 0.088))
            .set_z_index(2)
        )
        plate_charges = VGroup()
        for plate, side in zip(cap, (1, -1)):
            for positive in (True, False):
                symbols = (
                    VGroup(*[charge_symbol(positive, radius=0.10) for _ in range(9)])
                    .arrange(RIGHT, buff=0.025)
                    .move_to(plate)
                )

                def update_charge(mob, side=side, positive=positive):
                    charge = side * ~capacitor_voltage / ~voltage_scale
                    amount = charge if positive else -charge
                    mob.set_opacity(float(np.clip(amount, 0, 1)))

                symbols.add_updater(update_charge, call_updater=True)
                plate_charges.add(symbols)
        plate_charges.set_z_index(4)
        capacitor_label = (
            MathTex("C", color=wire_color).scale(0.8).next_to(cap, RIGHT, buff=0.25)
        )
        cap.add(plate_charges, capacitor_label)

        resistor_points = [[0.0, 0.0, 0.0]]
        for index, y in enumerate(np.linspace(0, fh(self, 0.11), 10)[1:-1]):
            resistor_points.append([(-1) ** index * 0.12, y, 0])
        resistor_points.append([0, fh(self, 0.11), 0])
        resistor = (
            VMobject(stroke_color=wire_color, stroke_width=4)
            .set_points_as_corners(resistor_points)
            .next_to(cap[0], UP, buff=fh(self, 0.074))
        )
        res = VGroup(resistor)
        res.add(MathTex("R_s", color=wire_color).scale(0.8).next_to(res))

        source = VGroup(
            Circle(
                radius=0.10,
                stroke_color=wire_color,
                stroke_width=4,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
            ).next_to(res[0], UP, buff=fh(self, 0.0675))
        )
        source.add(
            MathTex("V_{in}(t)", color=wire_color).next_to(source, UP, buff=0.22)
        )

        ground = (
            VGroup(
                *[
                    Line(
                        width / 2 * LEFT,
                        width / 2 * RIGHT,
                        color=wire_color,
                        stroke_width=4,
                    )
                    for width in (0.8, 0.5, 0.2)
                ]
            )
            .arrange(DOWN, buff=0.14)
            .next_to(cap[1], DOWN, buff=fh(self, 0.234))
        )

        top_wire = always_redraw(
            lambda: VGroup(
                Line(
                    cap[0].get_top(),
                    res[0].get_start(),
                    color=wire_color,
                    stroke_width=4,
                ),
                Line(
                    res[0].get_end(),
                    source[0].get_bottom(),
                    color=wire_color,
                    stroke_width=4,
                ),
            )
        )
        bottom_wire = always_redraw(
            lambda: Line(
                ground[0].get_center(),
                cap[1].get_bottom(),
                color=wire_color,
                stroke_width=4,
            )
        )
        top_path = always_redraw(
            lambda: (
                VMobject()
                .set_points_as_corners(
                    [cap[0].get_top(), *resistor.get_anchors(), source[0].get_bottom()]
                )
                .set_opacity(0)
            )
        )
        wires = VGroup(top_wire, bottom_wire, top_path)

        electrons = VGroup()
        drift_per_volt = fh(self, 0.028)
        spacing = fh(self, 0.055)
        for path in (top_path, bottom_wire):
            count = int(np.ceil(path.get_arc_length() / spacing))
            for index in range(-1, count + 1):
                electron = charge_symbol(radius=0.078)

                def update_electron(mob, path=path, index=index, count=count):
                    # Displacement follows charge; speed follows current.
                    offset = (drift_per_volt * ~capacitor_voltage / spacing) % 1
                    alpha = (index + offset) / count
                    mob.move_to(path.point_from_proportion(float(np.clip(alpha, 0, 1))))
                    edge_distance = min(alpha, 1 - alpha) * path.get_arc_length()
                    mob.set_opacity(float(np.clip(edge_distance / 0.14, 0, 1)))

                electron.add_updater(update_electron, call_updater=True)
                electrons.add(electron)
        electrons.set_z_index(3)

        circuit = VGroup(source, res, cap, ground, wires, electrons)
        VGroup(plot, circuit).arrange(RIGHT, buff=fw(self, 0.10)).move_to(ORIGIN)
        self.add(plot, circuit)

        self.next_section("Uncharged", skip_animations=skip_animations(False))
        self.wait(1)

        self.next_section("Apply DC", skip_animations=skip_animations(False))
        amplitude @= 5
        self.wait(2.4)

        self.next_section("Hold DC", skip_animations=skip_animations(False))
        self.wait(2)

        self.next_section("Apply AC", skip_animations=skip_animations(False))
        frequency @= 0.25
        self.wait(10 * tau)

        self.next_section("Steady AC", skip_animations=skip_animations(False))
        self.wait(plot_window)
