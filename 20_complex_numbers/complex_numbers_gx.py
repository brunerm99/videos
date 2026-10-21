# complex_numbers_gx.py
import pathlib
from collections import deque
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from manimgx import *
from manimgx import Capacitor as Cap
from scipy.signal import lsim

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
FONT = "Maple Mono"
BACKGROUND_COLOR = "#183340"
CAPACITOR_STATE_PATH = pathlib.Path(__file__).with_name("capacitor_state.npz")
SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class LumpedComponents(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        opening_frame = self.camera.frame.copy()
        plot, circuit = _capacitor_diagram(self, VT(0), VT(5))
        circuit.update(0).clear_updaters()

        cap = Cap(include_ports=True).scale(1.5)
        ind = Inductor(include_ports=True).scale(1.5)
        res = Resistor(include_ports=True).scale(1.5)

        lumped = VGroup(res, cap, ind).arrange(RIGHT, LARGE_BUFF * 3)

        res_label = MathTex("R").next_to(res, DOWN, LARGE_BUFF)
        cap_label = MathTex("C").next_to(cap, DOWN, LARGE_BUFF)
        ind_label = MathTex("L").next_to(ind, DOWN, LARGE_BUFF)

        self.play(
            LaggedStart(
                LaggedStart(
                    res.shift(DOWN * fh(self)).animate.shift(UP * fh(self)),
                    FadeIn(res_label),
                    lag_ratio=0.5,
                ),
                LaggedStart(
                    cap.shift(DOWN * fh(self)).animate.shift(UP * fh(self)),
                    FadeIn(cap_label),
                    lag_ratio=0.5,
                ),
                LaggedStart(
                    ind.shift(DOWN * fh(self)).animate.shift(UP * fh(self)),
                    FadeIn(ind_label),
                    lag_ratio=0.5,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    ind.animate.set_stroke(opacity=0.2).next_to(
                        self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF
                    ),
                    ind_label.animate.set_opacity(0.2).next_to(
                        ind.copy().next_to(
                            self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF
                        ),
                        DOWN,
                        LARGE_BUFF,
                    ),
                ),
                AnimationGroup(
                    cap.animate.set_stroke(opacity=0.2)
                    .set_fill(opacity=0.2)
                    .next_to(self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF)
                    .shift(LEFT),
                    cap_label.animate.set_opacity(0.2).next_to(
                        cap.copy()
                        .next_to(self.camera.frame.get_right(), LEFT, MED_LARGE_BUFF)
                        .shift(LEFT),
                        DOWN,
                        LARGE_BUFF,
                    ),
                ),
                AnimationGroup(
                    res.animate.scale(1.2).move_to(self.camera.frame),
                    res_label.animate.next_to(
                        res.copy().scale(1.2).move_to(ORIGIN),
                        DOWN,
                        LARGE_BUFF,
                    ),
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        vp = (
            Text("+", font=FONT, color=YELLOW)
            .scale(1.5)
            .next_to(res.p2, UL, SMALL_BUFF)
        )
        vn = (
            Text("-", font=FONT, color=YELLOW)
            .scale(1.5)
            .next_to(res.p1, DL, SMALL_BUFF)
        )
        v_label = MathTex("V_R").set_color(YELLOW).next_to(res, LEFT, MED_LARGE_BUFF)

        i_arrow = Arrow(UP, DOWN, color=BLUE).next_to(res, RIGHT, MED_SMALL_BUFF)
        i_label = MathTex("I_R").set_color(BLUE).next_to(i_arrow, RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                LaggedStart(
                    GrowArrow(i_arrow),
                    GrowFromCenter(i_label),
                    lag_ratio=0.2,
                ),
                LaggedStart(
                    GrowFromCenter(vp),
                    GrowFromCenter(vn),
                    GrowFromCenter(v_label),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        text_options = dict(
            font=FONT,
            typst_preamble='#set text(features: ("calt": 0))',
        )
        vpp = Text("++", **text_options).move_to(v_label, LEFT)
        ipp = Text("++", **text_options).next_to(i_label, RIGHT, SMALL_BUFF)

        self.play(
            LaggedStart(
                LaggedStart(
                    GrowFromCenter(ipp[0]),
                    GrowFromCenter(ipp[1]),
                    lag_ratio=0.2,
                ),
                LaggedStart(
                    v_label.animate.next_to(vpp, LEFT, SMALL_BUFF),
                    GrowFromCenter(vpp[0]),
                    GrowFromCenter(vpp[1]),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ohms = (
            MathTex("V_R = I_R R")
            .scale(1.5)
            .move_to(ORIGIN)
            .shift(DOWN * fh(self, 0.2))
        )
        ohms[0][:2].set_color(YELLOW)
        ohms[0][3:5].set_color(BLUE)

        ohms_law = (
            Text("Ohm's law", font=FONT).scale(0.7).next_to(ohms, DOWN, MED_LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                FadeOut(vpp, ipp),
                Group(res, vn, vp, i_arrow).animate.shift(UP * fh(self, 0.15)),
                ReplacementTransform(v_label[0], ohms[0][:2], path_arc=PI / 6),
                GrowFromCenter(ohms[0][2]),
                ReplacementTransform(res_label[0], ohms[0][5], path_arc=PI / 3),
                ReplacementTransform(i_label[0], ohms[0][3:5], path_arc=PI / 6),
                Write(ohms_law),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                AnimationGroup(
                    res.animate.set_stroke(opacity=0.2).shift(LEFT * 5),
                    VGroup(vn, vp, i_arrow, ohms_law, ohms)
                    .animate.set_opacity(0.2)
                    .shift(LEFT * 5),
                ),
                self.camera.frame.animate.move_to(
                    Group(cap, ind, cap_label, ind_label)
                ),
                AnimationGroup(
                    cap.animate.set_opacity(1).shift(LEFT * 0.6),
                    cap_label.animate.set_opacity(1).shift(LEFT * 0.6),
                ),
                AnimationGroup(
                    ind.animate.set_stroke(opacity=1).shift(RIGHT * 0.6),
                    ind_label.animate.set_opacity(1).shift(RIGHT * 0.6),
                ),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        vcp = (
            Text("+", font=FONT, color=YELLOW)
            .scale(1.5)
            .next_to(cap.p2, UL, SMALL_BUFF)
        )
        vcn = (
            Text("-", font=FONT, color=YELLOW)
            .scale(1.5)
            .next_to(cap.p1, DL, SMALL_BUFF)
        )
        vc_label = MathTex("V_C").set_color(YELLOW).next_to(cap, LEFT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                Group(ind, ind_label).animate.shift(RIGHT * fw(self)),
                self.camera.frame.animate.move_to(Group(cap, cap_label)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        ic_arrow = (
            Arrow(UP, DOWN / 2, color=BLUE)
            .next_to(cap.top_plate, UP, SMALL_BUFF)
            .shift(RIGHT * cap.top_plate.width * 0.3)
        )
        ic_label = (
            MathTex("I_C").set_color(BLUE).next_to(ic_arrow, RIGHT, MED_SMALL_BUFF)
        )

        self.play(
            self.camera.frame.animate.scale_to_fit_height(cap.height * 2.5).move_to(
                cap
            ),
            LaggedStart(
                GrowArrow(ic_arrow, rate_func=ease_out_bounce),
                GrowFromCenter(ic_label),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(vcp),
                GrowFromCenter(vcn),
                GrowFromCenter(vc_label),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)
        self.next_section("Capacitor circuit", skip_animations=skip_animations(False))

        source, target_cap, ground, wires, electrons = circuit
        plot_vc = plot[2][0][:2]
        plot_rest = VGroup(*plot[:2], plot[2][0][2:])
        self.remove(
            res,
            res_label,
            vn,
            vp,
            v_label,
            i_arrow,
            i_label,
            ohms_law,
            ohms,
            ind,
            ind_label,
        )
        self.play(
            AnimationGroup(
                self.camera.frame.animate.become(opening_frame),
                ReplacementTransform(cap.top_plate, target_cap.top_plate),
                ReplacementTransform(cap.bottom_plate, target_cap.bottom_plate),
                ReplacementTransform(cap[2], wires[1]),
                ReplacementTransform(cap[3], wires[0]),
                ReplacementTransform(cap_label, target_cap[-1], path_arc=PI / 2),
                run_time=2,
            ),
            FadeOut(VGroup(vcp, vcn, ic_arrow, ic_label), run_time=0.6),
            ReplacementTransform(vc_label[0], plot_vc, run_time=2),
            Succession(Wait(1.2), FadeIn(plot_rest, run_time=0.8)),
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(source[0]),
                    FadeIn(source[1]),
                    Create(ground),
                ),
                FadeIn(electrons),
                lag_ratio=0.2,
            ),
            run_time=1.5,
        )
        self.remove(cap, cap_label)
        self.add(plot, circuit)
        self.wait(0.5)


def _capacitor_diagram(
    scene, capacitor_voltage, voltage_scale, plot_window=12, y_range=None
):
    wire_color = "#D6EEF7"
    axes = Axes(
        x_range=[0, plot_window, 2],
        y_range=y_range if y_range is not None else [-1.2, 1.2, 0.5],
        x_length=fw(scene, 0.53),
        y_length=fh(scene, 0.60),
        axis_config={
            "color": wire_color,
            "stroke_width": 2,
            "include_numbers": False,
        },
        tips=False,
    )

    voltage_dot = Dot(radius=0.09, color=YELLOW).set_z_index(5)
    voltage_dot.move_to(axes.c2p(0, 0))
    voltage_label = (
        MathTex("V_C(t)", color=YELLOW)
        .scale(0.8)
        .rotate(PI / 2)
        .next_to(axes.c2p(0, 0), LEFT, MED_SMALL_BUFF)
    )
    plot = VGroup(axes, voltage_dot, voltage_label)

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

    cap = Cap(
        plate_width=fw(scene, 0.15),
        plate_thickness=fh(scene, 0.022),
        gap=fh(scene, 0.088),
        stroke_color=wire_color,
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

    source = VGroup(
        Circle(
            radius=0.10,
            stroke_color=wire_color,
            stroke_width=4,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        ).next_to(cap[0], UP, buff=fh(scene, 0.25))
    )
    source.add(MathTex("V_{in}(t)", color=YELLOW).next_to(source, UP, buff=0.22))

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
        .next_to(cap[1], DOWN, buff=fh(scene, 0.234))
    )

    top_wire = always_redraw(
        lambda: Line(
            cap[0].get_top(),
            source[0].get_bottom(),
            color=wire_color,
            stroke_width=4,
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
    wires = VGroup(top_wire, bottom_wire)

    electrons = VGroup()
    drift_per_volt = fh(scene, 0.028)
    spacing = fh(scene, 0.055)
    for path in (top_wire, bottom_wire):
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

    circuit = VGroup(source, cap, ground, wires, electrons)
    VGroup(plot, circuit).arrange(RIGHT, buff=fw(scene, 0.10)).move_to(ORIGIN)
    return plot, circuit


def _capacitor_simulation(scene, plot_window=12, y_range=None):
    amplitude = VT(0)
    frequency = VT(0)
    phase = VT(0)
    source_voltage = VT(0)
    capacitor_voltage = VT(0)
    time = VT(0)
    time_scale = VT(1)
    voltage_scale = VT(5)
    history = deque([(0.0, 0.0)])

    def update_circuit(mob, dt):
        dt *= ~time_scale
        if dt == 0:
            return

        if ~frequency == 0:
            voltage = ~amplitude
        else:
            angle = ~phase + TAU * ~frequency * dt
            voltage = ~amplitude * np.cos(angle)
            phase.set_value(angle % TAU)
        source_voltage.set_value(voltage)
        capacitor_voltage.set_value(voltage)
        time.increment_value(dt)
        history.append((~time, ~source_voltage))
        voltage_scale.set_value(max(~voltage_scale, abs(~amplitude), abs(voltage)))
        while len(history) > 2 and history[1][0] < ~time - plot_window:
            history.popleft()

    physics = Mobject().add_updater(update_circuit)
    scene.add(
        amplitude,
        frequency,
        phase,
        source_voltage,
        capacitor_voltage,
        time,
        time_scale,
        voltage_scale,
        physics,
    )

    plot, circuit = _capacitor_diagram(
        scene, capacitor_voltage, voltage_scale, plot_window, y_range
    )
    voltage_axes, voltage_dot, v_label = plot
    axes_width = voltage_axes.width
    label_height = v_label.height

    v_label.add_updater(
        lambda mob: mob.scale_to_fit_height(
            label_height * voltage_axes.width / axes_width
        ).next_to(voltage_axes, LEFT, buff=MED_SMALL_BUFF)
    )

    def get_trace(axes, gain=1, color=YELLOW):
        start = max(0, ~time - plot_window)
        samples = np.array(history)
        if samples[0, 0] < start:
            samples[0, 1] = np.interp(start, samples[:, 0], samples[:, 1])
            samples[0, 0] = start
        if len(samples) == 1:
            samples = np.repeat(samples, 2, axis=0)
        return axes.plot_line_graph(
            samples[:, 0] - start,
            gain * samples[:, 1] / (~voltage_scale if y_range is None else 1),
            add_vertex_dots=False,
            line_color=color,
            stroke_width=4 * axes.width / axes_width,
        )

    def place_dot(dot, axes, gain=1):
        return dot.scale_to_fit_width(0.18 * axes.width / axes_width).move_to(
            axes.c2p(
                min(~time, plot_window),
                gain * ~source_voltage / (~voltage_scale if y_range is None else 1),
            )
        )

    voltage_plot = always_redraw(lambda: get_trace(voltage_axes))
    voltage_dot.add_updater(lambda mob: place_dot(mob, voltage_axes), call_updater=True)
    plot.add(voltage_plot)

    return SimpleNamespace(
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        source_voltage=source_voltage,
        capacitor_voltage=capacitor_voltage,
        time=time,
        time_scale=time_scale,
        voltage_scale=voltage_scale,
        history=history,
        plot=plot,
        circuit=circuit,
        cap=circuit[1],
        trace=get_trace,
        place_dot=place_dot,
    )


class Charging(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        plot_window = 12
        sim = _capacitor_simulation(self, plot_window)
        amplitude, frequency = sim.amplitude, sim.frequency
        plot, circuit = sim.plot, sim.circuit
        cap = self.cap = sim.cap
        top_plate = self.top_plate = cap.top_plate
        bottom_plate = self.bottom_plate = cap.bottom_plate
        voltage_axes, _, v_label, _ = plot
        self.voltage_axes = voltage_axes

        self.add(plot, circuit)

        self.next_section("Uncharged", skip_animations=skip_animations(False))
        self.wait(1)

        self.wait(0.5)

        self.next_section("Apply DC", skip_animations=skip_animations(False))
        self.play(amplitude @ 5, run_time=1)
        self.wait(0.4)

        self.wait(0.5)

        poscharge_line = SquigglyLine(
            top_plate.get_left() + LEFT * 1.5 + UP,
            top_plate.get_left() + RIGHT * 0.2 + UP * 0.2,
            amplitude=0.1,
            color=RED,
            seed=1,
            wavelength=1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )
        poscharge_label = (
            Paragraph("positive", "charge", font=FONT)
            .set_color(RED)
            .scale(0.4)
            .next_to(poscharge_line.get_start(), LEFT, SMALL_BUFF)
        )

        negcharge_line = SquigglyLine(
            bottom_plate.get_left() + LEFT * 1.5 + DOWN,
            bottom_plate.get_left() + RIGHT * 0.2 + DOWN * 0.3,
            color=BLUE,
            amplitude=0.1,
            seed=2,
            wavelength=1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )
        negcharge_label = (
            Paragraph("negative", "charge", font=FONT)
            .set_color(BLUE)
            .scale(0.4)
            .next_to(negcharge_line.get_start(), LEFT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    Write(poscharge_label),
                    GrowArrow(poscharge_line),
                    lag_ratio=0.2,
                ),
                LaggedStart(
                    Write(negcharge_label),
                    GrowArrow(negcharge_line),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.4,
            )
        )

        self.next_section("Hold DC", skip_animations=skip_animations(False))
        self.wait(1)
        self.play(
            FadeOut(
                negcharge_line,
                negcharge_label,
                poscharge_line,
                poscharge_label,
            ),
            run_time=1,
        )
        self.wait(2)

        self.next_section("Apply AC", skip_animations=skip_animations(False))

        i_arrow = Arrow(UP * 2, DOWN * 2, color=BLUE).next_to(cap, LEFT)
        i_label = (
            MathTex(r"I_{C}(t)")
            .set_color(BLUE)
            .scale(0.8)
            .next_to(i_arrow, LEFT, SMALL_BUFF * 0.1)
        )

        frequency @= 0.25
        self.wait(0.5)
        self.play(GrowArrow(i_arrow), FadeIn(i_label))
        self.wait(2.8 - 1 - 0.5)

        self.next_section("Steady AC", skip_animations=skip_animations(False))
        self.wait(plot_window)

        self.next_section("increase f", skip_animations=skip_animations(False))

        self.play(frequency * 5, run_time=3)

        self.wait(2)

        self.play(frequency / 5, run_time=1)

        self.wait(1)

        self.next_section("scale", skip_animations=skip_animations(False))

        self.play(
            voltage_axes.animate.stretch(0.6, 1).shift(UP * voltage_axes.height / 3),
            run_time=2,
        )

        self.wait(0.5)

        self.next_section("Equation", skip_animations=skip_animations(False))

        cap_label = cap[-1]
        propto = (
            MathTex("I_C(t)", r"\propto", r"\frac{d V_C(t) }{dt}")
            .scale(1.4)
            .next_to(voltage_axes, DOWN, LARGE_BUFF)
        )
        propto[0].set_color(BLUE)
        propto[2][1:6].set_color(YELLOW)

        equation = (
            MathTex("I_C(t)", "=", r"C \frac{d V_C(t) }{dt}").scale(1.4).move_to(propto)
        )
        equation[0].set_color(BLUE)
        equation[2][2:7].set_color(YELLOW)

        self.play(
            LaggedStart(
                TransformFromCopy(
                    i_label[0], propto[0][:5], path_arc=PI / 3, introducer=True
                ),
                GrowFromCenter(propto[1]),
                TransformFromCopy(
                    v_label[0], propto[2][1:6], path_arc=PI / 3, introducer=True
                ),
                LaggedStart(
                    GrowFromCenter(propto[2][0]),
                    GrowFromCenter(propto[2][6]),
                    GrowFromCenter(propto[2][7]),
                    GrowFromCenter(propto[2][8]),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                propto[2][0]
                .animate(rate_func=there_and_back)
                .set_color(GREEN)
                .scale(1.5),
                propto[2][8]
                .animate(rate_func=there_and_back)
                .set_color(GREEN)
                .scale(1.5),
                propto[2][6]
                .animate(rate_func=there_and_back)
                .set_color(GREEN)
                .scale(1.5),
                propto[2][7]
                .animate(rate_func=there_and_back)
                .set_color(GREEN)
                .scale(1.5),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(propto[0], equation[0]),
                ReplacementTransform(propto[1], equation[1]),
                ReplacementTransform(propto[2], equation[2][1:]),
                TransformFromCopy(
                    cap_label, equation[2][0], path_arc=PI / 3, introducer=True
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        cap_word = (
            Text("capacitance", font=FONT)
            .next_to(equation[2][0], DOWN, LARGE_BUFF)
            .shift(LEFT * 0.8)
        )
        cap_line = SquigglyLine(
            cap_word.get_top() + UP * 0.1,
            equation[2][0].get_corner(DL) + DOWN * 0.2,
            seed=1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.6,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.6,
        )

        self.play(
            LaggedStart(
                Write(cap_word),
                GrowArrow(cap_line),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(cap_line, cap_word))

        self.wait(0.5)

        cap_label = Text("Capacitor:", font=FONT).next_to(
            self.camera.frame.copy().shift(DOWN * fh(self)).get_left(),
            RIGHT,
            MED_LARGE_BUFF,
        )
        self.add(cap_label)

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(DOWN * fh(self)),
                equation.animate.move_to(
                    self.camera.frame.copy().shift(DOWN * fh(self))
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        res_current = (
            MathTex(r"I_{R}(t) = \frac{V_{R}(t)}{R}")
            .scale(1.4)
            .move_to(equation, LEFT)
            .shift(DOWN * fh(self, 0.2))
        )
        res_current[0][:5].set_color(BLUE)
        res_current[0][6:11].set_color(YELLOW)
        res_label = (
            Text("Resistor:", font=FONT)
            .move_to(cap_label, LEFT)
            .set_y(res_current.get_y())
        )

        self.play(
            LaggedStart(
                Group(cap_label, equation).animate.shift(UP * fh(self, 0.2)),
                Group(res_label, res_current)
                .shift(DOWN * fh(self))
                .animate.shift(UP * fh(self)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            Group(res_label, res_current).animate.shift(DOWN * fh(self)),
            cap_label.animate.shift(LEFT * fw(self)),
            equation.animate.move_to(self.camera.frame),
        )

        self.wait(1)

        # np.savez(
        #     CAPACITOR_STATE_PATH,
        #     amplitude=~amplitude,
        #     frequency=~frequency,
        #     phase=~sim.phase,
        #     time=~sim.time,
        #     voltage_scale=~sim.voltage_scale,
        #     history=np.array(sim.history),
        # )


class CapacitorEquationV2(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        equation = MathTex(r"I_C(t) = C \frac{d V_C(t) }{dt}").scale(1.4)
        equation[0][:5].set_color(BLUE)
        equation[0][8:13].set_color(YELLOW)
        self.add(equation)

        self.wait(0.5)

        v_eqn = (
            MathTex(r"V_0 \cos{(2 \pi f t)}")
            .scale(1.4)
            .set_color(YELLOW)
            .next_to(equation, UP, LARGE_BUFF)
            .shift(RIGHT * fw(self, 0.15))
        )
        v_line = SquigglyLine(
            equation[0][8:13].get_top() + UP * 0.1,
            v_eqn.get_bottom() + DOWN * 0.1,
            color=YELLOW,
            amplitude=0.2,
            seed=1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )

        self.play(
            LaggedStart(
                GrowArrow(v_line),
                GrowFromCenter(v_eqn[0][:2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            GrowFromCenter(v_eqn[0][2:6]),
            GrowFromCenter(v_eqn[0][-1]),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(v_eqn[0][6]),
                GrowFromCenter(v_eqn[0][7]),
                GrowFromCenter(v_eqn[0][8]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        self.play(GrowFromCenter(v_eqn[0][9]))

        self.wait(0.5)

        vw_eqn = (
            MathTex(r"V_0 \cos{(\omega t)}").scale(1.4).set_color(YELLOW).move_to(v_eqn)
        )
        for c in vw_eqn[0]:
            c.save_state()
        vw_eqn[0][:6].move_to(v_eqn[0][:6])
        vw_eqn[0][-2:].move_to(v_eqn[0][-2:])

        self.play(
            LaggedStart(
                ShrinkToCenter(v_eqn[0][6:-2], rate_func=ease_in_back, run_time=0.3),
                GrowFromCenter(vw_eqn[0][6], rate_func=ease_out_back, run_time=0.3),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.remove(v_eqn, *v_eqn[0])
        self.add(*vw_eqn[0])
        #
        self.play(*[c.animate.restore() for c in vw_eqn[0]])

        self.wait(0.5)

        i_eqn = (
            MathTex(r"I_C(t) = C \frac{d V_0 \cos{(\omega t)} }{dt}")
            .scale(1.4)
            .move_to(equation)
        )
        i_eqn[0][:5].set_color(BLUE)
        i_eqn[0][8:17].set_color(YELLOW)

        self.play(
            LaggedStart(
                ReplacementTransform(equation[0][:8], i_eqn[0][:8]),
                ReplacementTransform(equation[0][13:], i_eqn[0][17:]),
                FadeOut(v_line),
                ShrinkToCenter(equation[0][8:13]),
                ReplacementTransform(vw_eqn[0], i_eqn[0][8:17]),
                self.camera.frame.animate.move_to(i_eqn),
                lag_ratio=0.25,
            )
        )
        self.remove(equation, *equation, vw_eqn, *vw_eqn, i_eqn, *i_eqn)
        self.add(i_eqn)

        self.wait(0.5)

        i_ddt1_eqn = (
            MathTex(r"I_C(t) = - \omega C V_0 \sin{(\omega t)}")
            .scale(1.4)
            .move_to(i_eqn)
        )
        i_ddt1_eqn[0][:5].set_color(BLUE)
        i_ddt1_eqn[0][9:18].set_color(YELLOW)

        self.next_section("Take the derivative", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(i_eqn[0][:6], i_ddt1_eqn[0][:6]),
                    ReplacementTransform(i_eqn[0][6], i_ddt1_eqn[0][8]),
                    ReplacementTransform(i_eqn[0][8:10], i_ddt1_eqn[0][9:11]),
                    ReplacementTransform(i_eqn[0][13:17], i_ddt1_eqn[0][14:18]),
                    i_eqn[0][10:13].animate.move_to(i_ddt1_eqn[0][11:14]),
                    FadeOut(
                        VGroup(i_eqn[0][7], *i_eqn[0][17:]),
                        shift=DOWN * 0.15,
                    ),
                    run_time=0.8,
                ),
                Succession(
                    TransformFromCopy(
                        i_eqn[0][14],
                        i_ddt1_eqn[0][7],
                        path_arc=-PI / 2,
                        introducer=True,
                        run_time=1.2,
                    ),
                    AnimationGroup(
                        ReplacementTransform(i_eqn[0][10:13], i_ddt1_eqn[0][11:14]),
                        GrowFromCenter(i_ddt1_eqn[0][6]),
                        lag_ratio=0.15,
                        run_time=0.9,
                    ),
                ),
                lag_ratio=0.25,
            )
        )
        self.remove(i_eqn, *i_eqn, i_ddt1_eqn, *i_ddt1_eqn)
        self.add(i_ddt1_eqn)
        self.wait(1)

        plot_window = 12
        sim = _capacitor_simulation(self, plot_window, y_range=[-6, 6, 2])
        sim.time_scale @= 0
        if CAPACITOR_STATE_PATH.exists():
            with np.load(CAPACITOR_STATE_PATH) as state:
                for name in (
                    "amplitude",
                    "frequency",
                    "phase",
                    "time",
                    "voltage_scale",
                ):
                    getattr(sim, name).set_value(float(state[name]))
                history = [tuple(sample) for sample in state["history"]]
        else:
            sim.amplitude @= 5
            sim.frequency @= 0.25
            sim.time @= plot_window
            sim.phase @= (TAU * ~sim.frequency * plot_window) % TAU
            times = np.arange(0, plot_window + 1 / 60, 1 / 60)
            history = list(
                zip(times, ~sim.amplitude * np.cos(TAU * ~sim.frequency * times))
            )
        sim.history.clear()
        sim.history.extend(history)
        sim.source_voltage @= history[-1][1]
        sim.capacitor_voltage @= history[-1][1]

        plot, circuit = sim.plot, sim.circuit
        voltage_axes, _, v_label, _ = plot
        self.voltage_axes = voltage_axes
        cap = self.cap = sim.cap
        self.top_plate, self.bottom_plate = cap.top_plate, cap.bottom_plate

        return_height = fh(self)
        axes_height = voltage_axes.height
        voltage_axes.stretch(0.6, 1).shift(UP * axes_height / 3)
        Group(plot, circuit).shift(UP * return_height)
        equation_target = i_ddt1_eqn.copy().next_to(voltage_axes, DOWN, LARGE_BUFF)
        plot.update(0)
        circuit.update(0)

        i_arrow = Arrow(UP * 2, DOWN * 2, color=BLUE).next_to(cap, LEFT)
        i_label = (
            MathTex(r"I_C(t)", color=BLUE)
            .scale(0.8)
            .next_to(i_arrow, LEFT, SMALL_BUFF * 0.1)
        )
        current_opacity = VT(0)
        capacitance = self.capacitance = VT(0.25)  # F
        axes_width = voltage_axes.width

        def current_samples():
            samples = np.array(sim.history)
            if len(samples) < 2:
                return samples[:, 0], np.zeros(len(samples))
            slope = np.gradient(
                samples[:, 1], samples[:, 0], edge_order=min(2, len(samples) - 1)
            )
            return samples[:, 0], ~capacitance * slope

        def current_curve():
            times, values = current_samples()
            start = max(0, ~sim.time - plot_window)
            if times[0] < start:
                values[0] = np.interp(start, times, values)
                times[0] = start
            return voltage_axes.plot_line_graph(
                times - start,
                values,
                add_vertex_dots=False,
                line_color=BLUE,
                stroke_width=4 * voltage_axes.width / axes_width,
            ).set_stroke(opacity=~current_opacity)

        current_trace = always_redraw(current_curve).set_z_index(2)
        current_dot = Dot(radius=0.09, color=BLUE).set_z_index(6)

        def place_current_dot(mob):
            _, values = current_samples()
            mob.scale_to_fit_width(0.18 * voltage_axes.width / axes_width).move_to(
                voltage_axes.c2p(min(~sim.time, plot_window), values[-1])
            ).set_opacity(~current_opacity)

        current_dot.add_updater(place_current_dot, call_updater=True)
        vi_label = (
            MathTex(r"V_C(t),\, I_C(t)")
            .scale(0.8)
            .rotate(PI / 2)
            .next_to(voltage_axes, LEFT, MED_SMALL_BUFF)
        )
        vi_label[0][:5].set_color(YELLOW)
        vi_label[0][6:].set_color(BLUE)
        self.current_trace, self.current_dot = current_trace, current_dot
        self.add(
            plot,
            circuit,
            i_arrow,
            i_label,
            current_opacity,
            capacitance,
            current_trace,
            current_dot,
        )

        self.next_section(
            "Return to the capacitor", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(UP * return_height),
                i_ddt1_eqn.animate.move_to(equation_target),
                lag_ratio=0.15,
            ),
            run_time=2,
        )
        v_label.clear_updaters()
        self.play(
            LaggedStart(
                ReplacementTransform(v_label[0], vi_label[0][:5]),
                GrowFromCenter(vi_label[0][5]),
                TransformFromCopy(
                    i_ddt1_eqn[0][:5],
                    vi_label[0][6:],
                    path_arc=-PI / 3,
                    introducer=True,
                ),
                current_opacity.animate(run_time=2).set_value(1),
                lag_ratio=0.2,
            ),
        )
        self.remove(v_label, *v_label, vi_label, *vi_label)
        plot.remove(v_label)
        self.add(vi_label)
        vi_label.add_updater(
            lambda mob: mob.next_to(voltage_axes, LEFT, MED_SMALL_BUFF)
        )
        self.play(sim.time_scale @ 1, run_time=0.5)
        self.wait(8)

        self.next_section("Expand the circuit", skip_animations=skip_animations(False))
        sim.time_scale @= 0
        for mob in (plot, circuit, current_trace, current_dot, vi_label):
            mob.clear_updaters()

        source, _, ground, wires, electrons = circuit
        top_wire, bottom_wire = wires
        plate_charges, cap_label = cap[2:]
        self.remove(circuit)
        self.add(source, top_wire, bottom_wire, cap, ground, electrons)

        self.play(
            LaggedStart(
                FadeOut(plot, current_trace, current_dot, vi_label, run_time=0.6),
                FadeOut(i_ddt1_eqn, i_arrow, i_label, run_time=0.5),
                FadeOut(plate_charges, electrons, run_time=0.45),
                lag_ratio=0.3,
            )
        )
        cap.remove(plate_charges)

        wire_color = "#D6EEF7"
        wire_style = dict(color=wire_color, stroke_width=4)
        center = self.camera.frame.get_center()
        unit = fh(self, 0.125)
        input_point = center + unit * np.array([-6.4, 0.6, 0])
        node1 = center + unit * np.array([-3.2, 0.6, 0])
        node2 = center + unit * np.array([0, 0.6, 0])
        output_node = center + unit * np.array([5.5, 0, 0])
        output_point = center + unit * np.array([6.4, 0, 0])

        cap_target = Cap(
            plate_width=fw(self, 0.068),
            plate_thickness=fh(self, 0.006),
            gap=fh(self, 0.035),
            stroke_color=wire_color,
        ).move_to(center + DOWN * unit)
        c2_label = (
            MathTex("C_2", color=wire_color)
            .scale(0.8)
            .next_to(cap_target, LEFT, MED_LARGE_BUFF)
        )
        source_target = source.copy()
        source_target[0].move_to(input_point)
        source_target[1].scale(0.9).next_to(source_target[0], UP, MED_LARGE_BUFF)
        ground_target = ground.copy().scale(0.75).move_to(center + DOWN * 2.35 * unit)
        top_wire_target = VMobject(**wire_style).set_points_as_corners(
            [cap_target.p2, node2, source_target[0].get_right()]
        )
        bottom_wire_target = Line(
            ground_target[0].get_center(), cap_target.p1, **wire_style
        )

        self.play(
            self.camera.frame.animate(run_time=1.6)
            .scale(1.18)
            .move_to(center + UP * 0.4 * unit),
            AnimationGroup(
                Transform(cap.top_plate, cap_target.top_plate),
                Transform(cap.bottom_plate, cap_target.bottom_plate),
                ReplacementTransform(cap_label, c2_label),
                Transform(source, source_target),
                Transform(ground, ground_target),
                Transform(top_wire, top_wire_target),
                Transform(bottom_wire, bottom_wire_target),
                run_time=1.4,
            ),
        )
        cap.remove(cap_label).add(c2_label)
        self.add(cap)
        self.wait(0.175)

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
        r1_label = (
            MathTex("R_1", color=wire_color).scale(0.8).next_to(r1, UP, MED_LARGE_BUFF)
        )
        r2_label = (
            MathTex("R_2", color=wire_color).scale(0.8).next_to(r2, UP, MED_LARGE_BUFF)
        )
        input_wire = Line(source[0].get_right(), r1.p1, **wire_style)
        middle_wire = Line(r1.p2, r2.p1, **wire_style)
        right_wire = Line(r2.p2, node2, **wire_style)
        cap_wire = Line(node2, cap.p2, **wire_style)
        r1_wire = Line(r1.p1, r1.p2, **wire_style)
        r2_wire = Line(r2.p1, r2.p2, **wire_style)
        self.remove(top_wire)
        self.add(input_wire, r1_wire, middle_wire, r2_wire, right_wire, cap_wire)

        self.next_section("Add the resistors", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                LaggedStart(
                    ReplacementTransform(r1_wire, r1, run_time=0.7),
                    FadeIn(r1_label, shift=UP * 0.15 * unit, run_time=0.35),
                    lag_ratio=0.6,
                ),
                LaggedStart(
                    ReplacementTransform(r2_wire, r2, run_time=0.7),
                    FadeIn(r2_label, shift=UP * 0.15 * unit, run_time=0.35),
                    lag_ratio=0.6,
                ),
                lag_ratio=0.55,
            )
        )
        self.wait(0.175)

        amp = Polygon(
            center + unit * np.array([2.3, 1.4, 0]),
            center + unit * np.array([2.3, -1.4, 0]),
            center + unit * np.array([4.6, 0, 0]),
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
            **wire_style,
        )
        plus = (
            Text("+", font=FONT, color=RED)
            .scale(0.65)
            .move_to(center + unit * np.array([2.65, 0.6, 0]))
        )
        minus = (
            Text("-", font=FONT, color=BLUE)
            .scale(0.65)
            .move_to(center + unit * np.array([2.65, -0.6, 0]))
        )
        amp_input = Line(node2, center + unit * np.array([2.3, 0.6, 0]), **wire_style)
        output_port = source[0].copy().move_to(output_point)
        output_wire = Line(
            center + RIGHT * 4.6 * unit, output_port.get_left(), **wire_style
        )
        output_label = (
            MathTex("V_{out}(t)", color=YELLOW)
            .scale(0.9)
            .next_to(output_port, UP, MED_LARGE_BUFF)
        )
        node1_dot = Dot(node1, radius=0.06 * unit, color=wire_color)
        node2_dot = Dot(node2, radius=0.06 * unit, color=wire_color)
        output_dot = Dot(output_node, radius=0.06 * unit, color=wire_color)

        self.next_section("Add the op amp", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                AnimationGroup(
                    Create(amp_input, run_time=0.45),
                    GrowFromCenter(node2_dot, run_time=0.25),
                ),
                LaggedStart(
                    Create(amp, run_time=0.7),
                    FadeIn(plus, minus, run_time=0.3),
                    lag_ratio=0.65,
                ),
                LaggedStart(
                    Create(output_wire, run_time=0.5),
                    GrowFromCenter(output_port, run_time=0.25),
                    FadeIn(output_label, run_time=0.35),
                    lag_ratio=0.6,
                ),
                lag_ratio=0.65,
            )
        )
        self.wait(0.175)

        follower_wire = VMobject(**wire_style).set_points_as_corners(
            [
                center + unit * np.array([2.3, -0.6, 0]),
                center + unit * np.array([1.55, -0.6, 0]),
                center + unit * np.array([1.55, -1.95, 0]),
                center + unit * np.array([5.5, -1.95, 0]),
                output_node,
            ]
        )
        c1 = (
            cap_target.copy()
            .rotate(-PI / 2)
            .move_to(center + unit * np.array([0, 2.7, 0]))
        )
        c1_label = (
            MathTex("C_1", color=wire_color).scale(0.8).next_to(c1, UP, MED_LARGE_BUFF)
        )
        c1_left_wire = VMobject(**wire_style).set_points_as_corners(
            [node1, node1 + UP * 2.1 * unit, c1.p1]
        )
        c1_right_wire = VMobject(**wire_style).set_points_as_corners(
            [c1.p2, center + unit * np.array([5.5, 2.7, 0]), output_node]
        )

        self.next_section(
            "Complete the Sallen-Key filter", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                LaggedStart(
                    Create(follower_wire, run_time=0.8),
                    GrowFromCenter(output_dot, run_time=0.225),
                    lag_ratio=0.8,
                ),
                Succession(
                    AnimationGroup(
                        GrowFromCenter(node1_dot, run_time=0.225),
                        Create(c1_left_wire, run_time=0.55),
                    ),
                    LaggedStart(
                        GrowFromCenter(c1, run_time=0.4),
                        FadeIn(c1_label, run_time=0.3),
                        Create(c1_right_wire, run_time=0.75),
                        lag_ratio=0.5,
                    ),
                ),
                lag_ratio=0.75,
            )
        )
        self.wait(1)

        phi1 = (
            MathTex(r"\phi_1")
            .set_color(YELLOW)
            .scale(1.4)
            .next_to(node1_dot, DOWN, MED_SMALL_BUFF)
        )
        phi2 = (
            MathTex(r"\phi_2")
            .set_color(YELLOW)
            .scale(1.4)
            .next_to(node2_dot, UP, MED_SMALL_BUFF)
        )
        phi3 = (
            MathTex(r"\phi_3")
            .set_color(YELLOW)
            .scale(1.4)
            .next_to(output_dot, DL, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    node1_dot.animate.scale(1.7).set_color(YELLOW).set_z_index(10),
                    FadeIn(phi1, shift=DOWN * 0.5),
                    lag_ratio=0.2,
                ),
                LaggedStart(
                    node2_dot.animate.scale(1.7).set_color(YELLOW).set_z_index(10),
                    FadeIn(phi2, shift=UP * 0.5),
                    lag_ratio=0.2,
                ),
                LaggedStart(
                    output_dot.animate.scale(1.7).set_color(YELLOW).set_z_index(10),
                    FadeIn(phi3, shift=DL * 0.5),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        sin1 = MathTex(r"\sin{(\omega t + \phi_1)}")
        sin1[0][-3:-1].set_color(YELLOW)
        sin2 = MathTex(r"\sin{(\omega t + \phi_2)}")
        sin2[0][-3:-1].set_color(YELLOW)
        sin3 = MathTex(r"\sin{(\omega t + \phi_3)}")
        sin3[0][-3:-1].set_color(YELLOW)
        sins = (
            Group(sin1, sin2, sin3)
            .arrange(RIGHT, LARGE_BUFF)
            .scale(1.4)
            .move_to(self.camera.frame.get_bottom())
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale(1.1).shift(DOWN * 1),
                LaggedStart(
                    *[FadeIn(m) for m in sin1[0][:-3]],
                    TransformFromCopy(phi1[0], sin1[0][-3:-1], introducer=True),
                    FadeIn(sin1[0][-1]),
                    lag_ratio=0.1,
                ),
                LaggedStart(
                    *[FadeIn(m) for m in sin2[0][:-3]],
                    TransformFromCopy(phi2[0], sin2[0][-3:-1], introducer=True),
                    FadeIn(sin2[0][-1]),
                    lag_ratio=0.1,
                ),
                LaggedStart(
                    *[FadeIn(m) for m in sin3[0][:-3]],
                    TransformFromCopy(phi3[0], sin3[0][-3:-1], introducer=True),
                    FadeIn(sin3[0][-1]),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )
        content = Group(
            source,
            cap,
            ground,
            bottom_wire,
            input_wire,
            middle_wire,
            right_wire,
            cap_wire,
            r1,
            r2,
            r1_label,
            r2_label,
            amp,
            plus,
            minus,
            amp_input,
            output_port,
            output_wire,
            output_label,
            follower_wire,
            c1,
            c1_label,
            c1_left_wire,
            c1_right_wire,
            phi1,
            phi2,
            phi3,
            sins,
            node1_dot,
            node2_dot,
            output_dot,
        ).set_z_index(4)
        self.clear()
        self.add(content)

        w, h = fw(self), fh(self)
        screen_center = self.camera.frame.get_center()
        title_h = h * 0.095
        glass_center = screen_center + UP * title_h / 2
        glass_w, glass_h = w + h * 0.12, h * 1.21
        glass_radius = h * 0.035

        def bow(points):
            p = np.array(points) - glass_center
            x, y = p[..., 0] / (glass_w / 2), p[..., 1] / (glass_h / 2)
            p[..., 0] *= 1 - 0.035 * y * y
            p[..., 1] *= 1 - 0.035 * x * x
            return p + glass_center

        glass = (
            RoundedRectangle(
                width=glass_w,
                height=glass_h,
                corner_radius=glass_radius,
                fill_color="#10262E",
                fill_opacity=1,
                stroke_color="#263A3D",
                stroke_width=3,
            )
            .move_to(glass_center)
            .set_z_index(-5)
        )
        window_rim = (
            Rectangle(
                width=w + h * 0.028,
                height=h + title_h + h * 0.028,
                fill_color="#9DAEA7",
                fill_opacity=1,
                stroke_width=0,
            )
            .move_to(glass_center)
            .set_z_index(-2)
        )
        client = (
            Rectangle(
                width=w,
                height=h,
                fill_color=BACKGROUND_COLOR,
                fill_opacity=1,
                stroke_width=0,
            )
            .move_to(screen_center)
            .set_z_index(-1)
        )
        title_bar = (
            Rectangle(
                width=w,
                height=title_h,
                fill_color="#BDC9C0",
                fill_opacity=1,
                stroke_width=0,
            )
            .next_to(client, UP, buff=0)
            .set_z_index(2)
        )
        computing = (
            Text(
                "Computing...",
                font=FONT,
                color=BACKGROUND_COLOR,
                typst_preamble='#set text(features: ("calt": 0, "liga": 0))',
            )
            .scale_to_fit_height(title_h * 0.53)
            .move_to(title_bar)
            .align_to(title_bar, LEFT)
            .shift(RIGHT * title_h * 0.45)
            .set_z_index(5)
        )
        computing[-2:].set_opacity(0)
        buttons = (
            VGroup(
                *[
                    Rectangle(
                        width=title_h * 0.63,
                        height=title_h * 0.63,
                        fill_color="#D7DED4",
                        fill_opacity=1,
                        stroke_color="#4A6566",
                        stroke_width=1.4,
                    )
                    for _ in range(2)
                ]
            )
            .arrange(RIGHT, buff=title_h * 0.16)
            .move_to(title_bar)
            .align_to(title_bar, RIGHT)
            .shift(LEFT * title_h * 0.22)
            .set_z_index(5)
        )
        minimize = (
            Line(LEFT, RIGHT, color="#244449", stroke_width=1.5)
            .scale_to_fit_width(title_h * 0.32)
            .move_to(buttons[0])
            .shift(DOWN * title_h * 0.12)
            .set_z_index(6)
        )
        close = (
            VGroup(
                Line(UL, DR, color="#244449", stroke_width=1.5),
                Line(DL, UR, color="#244449", stroke_width=1.5),
            )
            .scale_to_fit_width(title_h * 0.29)
            .move_to(buttons[1])
            .set_z_index(6)
        )
        screen = Group(
            glass,
            window_rim,
            client,
            title_bar,
            computing,
            buttons,
            minimize,
            close,
            content,
        )
        crt_start = self.time
        crt_time = VT(0).add_updater(lambda mob: mob.set_value(self.time - crt_start))
        crt_opacity = VT(0)
        crt_bend = VT(0)

        def update_computing(mob):
            count = 1 + int(~crt_time / 0.5) % 3
            for index, dot in enumerate(mob[-3:]):
                dot.set_opacity(float(index < count))

        computing.add_updater(update_computing, call_updater=True)

        def screen_row(y):
            dy = max(0, abs(y) - glass_h / 2 + glass_radius)
            extent = (
                glass_w / 2 - glass_radius + np.sqrt(max(0, glass_radius**2 - dy**2))
            )
            bend = 0.035 * ~crt_bend
            x = extent * (1 - bend * (y / (glass_h / 2)) ** 2)
            q = bend * (extent / (glass_w / 2)) ** 2
            return glass_center + np.array(
                [
                    [-x, y * (1 - q), 0],
                    [-x / 3, y * (1 + q / 3), 0],
                    [x / 3, y * (1 + q / 3), 0],
                    [x, y * (1 - q), 0],
                ]
            )

        scan_count = int(np.ceil(glass_h / (h * 0.007)))
        scan_spacing = glass_h / scan_count
        scanlines = VGroup(
            *[
                VMobject(stroke_color="#020D14", stroke_width=0.8)
                for _ in range(scan_count)
            ]
        ).set_z_index(41)

        def update_scanlines(mob):
            offset = ~crt_time * scan_spacing * 0.65
            for index, line in enumerate(mob):
                y = (index * scan_spacing - offset) % glass_h - glass_h / 2
                edge = min(1, (glass_h / 2 - abs(y)) / (h * 0.012))
                line.set_points(screen_row(y)).set_stroke(
                    opacity=0.24 * ~crt_opacity * edge
                )

        scanlines.add_updater(update_scanlines, call_updater=True)
        band_offsets = np.linspace(-h * 0.08, h * 0.08, 24)
        band_height = band_offsets[1] - band_offsets[0]
        rolling_band = VGroup(
            *[
                VMobject(fill_color="#D6EFE3", fill_opacity=0, stroke_width=0)
                for _ in band_offsets
            ]
        ).set_z_index(40)

        def update_band(mob):
            top = glass_h / 2 + h * 0.08
            band_y = top - ((top - h * 0.14 + ~crt_time * h * 0.12) % (2 * top))
            for strip, offset in zip(mob, band_offsets):
                y = band_y + offset
                if abs(y) >= glass_h / 2:
                    strip.set_fill(opacity=0)
                    continue
                upper = screen_row(min(y + band_height / 2, glass_h / 2))
                lower = screen_row(max(y - band_height / 2, -glass_h / 2))[::-1]
                strip.set_points(
                    np.vstack(
                        (
                            upper,
                            np.linspace(upper[-1], lower[0], 4),
                            lower,
                            np.linspace(lower[-1], upper[0], 4),
                        )
                    )
                ).set_fill(
                    opacity=0.045
                    * ~crt_opacity
                    * np.exp(-4 * (offset / (h * 0.08)) ** 2)
                )

        rolling_band.add_updater(update_band, call_updater=True)
        self.add(crt_time, crt_opacity, crt_bend, screen, rolling_band, scanlines)
        self.next_section("Computing window", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(glass_h * 1.13).move_to(
                    glass_center
                ),
                AnimationGroup(
                    screen.animate(suspend_mobject_updating=False).apply_function(bow),
                    crt_bend @ 1,
                ),
                lag_ratio=0.1,
            ),
            crt_opacity @ 1,
            run_time=2.2,
        )
        self.next_section("Computing", skip_animations=skip_animations(False))
        self.play(
            *[
                dot.animate.set_color(wire_color)
                for dot in (node1_dot, node2_dot, output_dot)
            ],
            run_time=0.25,
        )
        self.wait(5.75)

        mouse_x = VT(screen_center[0] + w * 0.55)
        mouse_y = VT(node2[1] - h * 0.08)
        mouse_size = VT(1)
        mouse_vertices = (
            h
            * 0.065
            * np.array(
                [
                    [0, 0, 0],
                    [0, -1, 0],
                    [0.25, -0.75, 0],
                    [0.49, -1.22, 0],
                    [0.70, -1.12, 0],
                    [0.46, -0.66, 0],
                    [0.83, -0.66, 0],
                ]
            )
        )

        def mouse_pointer():
            points = mouse_vertices * ~mouse_size + np.array([~mouse_x, ~mouse_y, 0])
            edge = screen_center[0] + w / 2
            clipped = []
            for start, end in zip(points, np.roll(points, -1, axis=0)):
                if start[0] <= edge:
                    clipped.append(start)
                if (start[0] <= edge) != (end[0] <= edge):
                    clipped.append(
                        start + (end - start) * (edge - start[0]) / (end[0] - start[0])
                    )
            return (
                Polygon(
                    *(clipped if len(clipped) >= 3 else points),
                    fill_color="#F4F4E9",
                    fill_opacity=1,
                    stroke_color="#071318",
                    stroke_width=2.5,
                )
                .set_opacity(float(len(clipped) >= 3))
                .apply_function(bow)
                .set_z_index(30)
            )

        mouse = always_redraw(mouse_pointer).set_z_index(30)
        self.add(mouse_x, mouse_y, mouse_size, mouse)
        self.next_section(
            "Select capacitor node", skip_animations=skip_animations(False)
        )
        self.play(
            mouse_x @ node2[0],
            mouse_y @ node2[1],
            run_time=1.35,
            rate_func=ease_in_out_cubic,
        )
        self.wait(0.2)
        click_ring = (
            Circle(radius=h * 0.023, color=YELLOW, stroke_width=2.5)
            .move_to(node2)
            .apply_function(bow)
            .set_z_index(26)
        )
        self.add(click_ring)
        self.play(
            mouse_size @ 0.88,
            node2_dot.animate.set_color(YELLOW),
            run_time=0.12,
        )
        self.play(
            mouse_size @ 1,
            click_ring.animate.scale(2.1).set_stroke(opacity=0),
            run_time=0.25,
        )
        self.remove(click_ring)
        self.wait(0.15)

        probe_w, probe_h = w * 0.72, h * 0.55
        probe_center = screen_center + RIGHT * w * 0.09 + DOWN * h * 0.18
        probe_title_h = title_h * 0.68
        probe_rim = (
            Rectangle(
                width=probe_w,
                height=probe_h,
                fill_color="#9DAEA7",
                fill_opacity=1,
                stroke_color="#071318",
                stroke_width=3,
            )
            .move_to(probe_center)
            .set_z_index(20)
        )
        probe_body = (
            Rectangle(
                width=probe_w - h * 0.035,
                height=probe_h - probe_title_h - h * 0.035,
                fill_color="#102C38",
                fill_opacity=1,
                stroke_width=0,
            )
            .move_to(probe_center + DOWN * probe_title_h / 2)
            .set_z_index(21)
        )
        probe_title_bar = (
            Rectangle(
                width=probe_body.width,
                height=probe_title_h,
                fill_color="#BDC9C0",
                fill_opacity=1,
                stroke_width=0,
            )
            .next_to(probe_body, UP, buff=0)
            .set_z_index(21)
        )
        probe_title = (
            Text("Voltage probe - C2", font=FONT, color=BACKGROUND_COLOR)
            .scale_to_fit_height(probe_title_h * 0.51)
            .move_to(probe_title_bar)
            .align_to(probe_title_bar, LEFT)
            .shift(RIGHT * probe_title_h * 0.35)
            .set_z_index(22)
        )
        probe_close_box = (
            Rectangle(
                width=probe_title_h * 0.68,
                height=probe_title_h * 0.68,
                fill_color="#D7DED4",
                fill_opacity=1,
                stroke_color="#4A6566",
                stroke_width=1.2,
            )
            .move_to(probe_title_bar)
            .align_to(probe_title_bar, RIGHT)
            .shift(LEFT * probe_title_h * 0.18)
            .set_z_index(22)
        )
        probe_close = (
            VGroup(
                Line(UL, DR, color="#244449", stroke_width=1.3),
                Line(DL, UR, color="#244449", stroke_width=1.3),
            )
            .scale_to_fit_width(probe_title_h * 0.32)
            .move_to(probe_close_box)
            .set_z_index(22)
        )
        probe_shell = Group(
            probe_rim,
            probe_body,
            probe_title_bar,
            probe_title,
            probe_close_box,
            probe_close,
        ).apply_function(bow)

        probe_duration = 12
        probe_axes = Axes(
            x_range=[0, probe_duration, 2],
            y_range=[-6, 6, 1],
            x_length=probe_w * 0.85,
            y_length=probe_h * 0.51,
            axis_config={
                "include_ticks": False,
                "color": "#91ABA9",
                "stroke_width": 1.2,
            },
            tips=False,
        ).move_to(probe_center + RIGHT * h * 0.015 + DOWN * h * 0.035)
        probe_grid = VGroup(
            *[
                Line(
                    probe_axes.c2p(t, -6),
                    probe_axes.c2p(t, 6),
                    color="#91ABA9",
                    stroke_width=0.8,
                    stroke_opacity=0.22,
                )
                for t in range(0, probe_duration + 1, 2)
            ],
            *[
                Line(
                    probe_axes.c2p(0, value),
                    probe_axes.c2p(probe_duration, value),
                    color="#91ABA9",
                    stroke_width=0.8,
                    stroke_opacity=0.22,
                )
                for value in (-5, 5)
            ],
        )
        probe_ticks = VGroup(
            *[
                Text(str(t), font=FONT, font_size=20, color="#BDC9C0").next_to(
                    probe_axes.c2p(t, -6), DOWN, buff=h * 0.012
                )
                for t in range(0, probe_duration + 1, 2)
            ],
            *[
                Text(str(value), font=FONT, font_size=20, color="#BDC9C0").next_to(
                    probe_axes.c2p(0, value), LEFT, buff=h * 0.015
                )
                for value in (-5, 0, 5)
            ],
        )
        probe_units = VGroup(
            MathTex("V", color="#BDC9C0")
            .scale_to_fit_height(h * 0.026)
            .next_to(probe_axes.c2p(0, 6), UL, buff=h * 0.012),
            MathTex("t", color="#BDC9C0")
            .scale_to_fit_height(h * 0.026)
            .next_to(probe_axes, DOWN, buff=h * 0.035),
        )
        probe_legend = VGroup(
            VGroup(
                Line(LEFT, RIGHT, color=YELLOW, stroke_width=3).scale_to_fit_width(
                    h * 0.065
                ),
                MathTex(r"V_{in}(t)", color=YELLOW).scale_to_fit_height(h * 0.035),
            ).arrange(RIGHT, buff=h * 0.018),
            VGroup(
                Line(LEFT, RIGHT, color=BLUE, stroke_width=3).scale_to_fit_width(
                    h * 0.065
                ),
                MathTex(r"V_{C_2}(t)", color=BLUE).scale_to_fit_height(h * 0.035),
            ).arrange(RIGHT, buff=h * 0.018),
        )
        probe_legend.arrange(RIGHT, buff=h * 0.10).move_to(
            probe_center + UP * (probe_h / 2 - probe_title_h - h * 0.06)
        )
        probe_origin = probe_axes.c2p(0, 0)
        probe_x_unit = probe_axes.c2p(1, 0) - probe_origin
        probe_y_unit = probe_axes.c2p(0, 1) - probe_origin

        def voltage_point(t, voltage):
            return bow(probe_origin + t * probe_x_unit + voltage * probe_y_unit)

        r1_value = r2_value = 10e3  # ohm
        c1_value, c2_value = 330e-6, 22e-6  # F
        probe_amplitude, probe_frequency = ~sim.amplitude, ~sim.frequency
        probe_times = np.linspace(0, probe_duration, 721)
        probe_input = probe_amplitude * np.cos(TAU * probe_frequency * probe_times)
        _, probe_voltage, _ = lsim(
            (
                [1],
                [
                    r1_value * r2_value * c1_value * c2_value,
                    c2_value * (r1_value + r2_value),
                    1,
                ],
            ),
            U=probe_input,
            T=probe_times,
            X0=0,
        )
        input_curve = VMobject(color=YELLOW, stroke_width=3).set_points_smoothly(
            bow(probe_axes.c2p(probe_times, probe_input).T)
        )
        node_curve = VMobject(color=BLUE, stroke_width=3).set_points_smoothly(
            bow(probe_axes.c2p(probe_times, probe_voltage).T)
        )
        probe_plot = (
            Group(probe_grid, probe_axes, probe_ticks, probe_units, probe_legend)
            .apply_function(bow)
            .set_z_index(23)
        )

        self.next_section("Open voltage probe", skip_animations=skip_animations(False))
        self.play(
            GrowFromPoint(probe_shell, node2_dot.get_center(), run_time=0.55),
            run_time=0.55,
            rate_func=ease_out_cubic,
        )
        self.play(
            FadeIn(probe_plot),
            run_time=0.4,
        )
        probe_time = VT(0)
        input_trace = always_redraw(
            lambda: (
                input_curve.copy()
                .pointwise_become_partial(input_curve, 0, ~probe_time / probe_duration)
                .set_z_index(24)
            )
        )
        node_trace = always_redraw(
            lambda: (
                node_curve.copy()
                .pointwise_become_partial(node_curve, 0, ~probe_time / probe_duration)
                .set_z_index(24)
            )
        )
        input_probe_dot = always_redraw(
            lambda: Dot(
                voltage_point(
                    ~probe_time,
                    probe_amplitude * np.cos(TAU * probe_frequency * ~probe_time),
                ),
                radius=h * 0.006,
                color=YELLOW,
            ).set_z_index(25)
        )
        node_probe_dot = always_redraw(
            lambda: Dot(
                voltage_point(
                    ~probe_time,
                    np.interp(~probe_time, probe_times, probe_voltage),
                ),
                radius=h * 0.006,
                color=BLUE,
            ).set_z_index(25)
        )
        sample_step = 0.5
        sample_times = np.arange(0, probe_duration + sample_step / 2, sample_step)
        sample_input = np.interp(sample_times, probe_times, probe_input)
        sample_voltage = np.interp(sample_times, probe_times, probe_voltage)
        samples = VGroup()
        for t, input_value, node_value in zip(
            sample_times, sample_input, sample_voltage
        ):
            sample = VGroup()
            zero = probe_origin + t * probe_x_unit
            for value, color in ((input_value, YELLOW), (node_value, BLUE)):
                stem = (
                    Line(
                        zero,
                        zero + value * probe_y_unit,
                        color=color,
                        stroke_width=1.3,
                        stroke_opacity=0.6,
                    )
                    .apply_function(bow)
                    .set_z_index(23.5)
                )
                dot = Dot(
                    voltage_point(t, value), radius=h * 0.005, color=color
                ).set_z_index(25)
                sample.add(stem, dot)
            samples.add(sample)

        self.add(probe_time, input_trace, node_trace, input_probe_dot, node_probe_dot)
        self.next_section("Plot node voltage", skip_animations=skip_animations(False))
        probe_run_time = 3.6
        self.play(
            probe_time.animate(run_time=probe_run_time, rate_func=linear).set_value(
                probe_duration
            ),
            LaggedStart(
                *[
                    GrowFromPoint(
                        sample,
                        voltage_point(t, 0),
                        run_time=sample_step * probe_run_time / probe_duration,
                        rate_func=ease_out_cubic,
                    )
                    for t, sample in zip(sample_times, samples)
                ],
                lag_ratio=1,
            ),
        )
        self.wait(2)

        settled_start = 7
        view_start = VT(0)
        selection_end = VT(settled_start)
        selection_opacity = VT(0)
        drag_start = probe_origin + settled_start * probe_x_unit + 1.5 * probe_y_unit
        drag_end = probe_origin + probe_duration * probe_x_unit + 1.5 * probe_y_unit

        def selected_region():
            scale = probe_duration / (probe_duration - ~view_start)
            left = probe_origin + (settled_start - ~view_start) * scale * probe_x_unit
            right = probe_origin + (~selection_end - ~view_start) * scale * probe_x_unit
            return (
                VGroup(
                    Polygon(
                        left - 6 * probe_y_unit,
                        right - 6 * probe_y_unit,
                        right + 6 * probe_y_unit,
                        left + 6 * probe_y_unit,
                        fill_color=BLUE,
                        fill_opacity=0.18 * ~selection_opacity,
                        stroke_width=0,
                    ),
                    *[
                        Line(
                            point - 6 * probe_y_unit,
                            point + 6 * probe_y_unit,
                            color=BLUE,
                            stroke_width=1.8,
                            stroke_opacity=~selection_opacity,
                        )
                        for point in (left, right)
                    ],
                )
                .apply_function(bow)
                .set_z_index(26)
            )

        selection = always_redraw(selected_region)
        self.add(selection_end, selection_opacity, selection)
        self.next_section(
            "Select settled interval", skip_animations=skip_animations(False)
        )
        self.play(
            mouse_x @ drag_start[0],
            mouse_y @ drag_start[1],
            run_time=1.0,
            rate_func=ease_in_out_cubic,
        )
        self.wait(0.2)
        self.play(mouse_size @ 0.88, selection_opacity @ 1, run_time=0.12)
        self.play(
            mouse_x @ drag_end[0],
            selection_end @ probe_duration,
            run_time=1.6,
            rate_func=linear,
        )
        self.play(mouse_size @ 1, run_time=0.12)
        self.wait(0.35)

        tick_times = sorted(
            set(range(0, probe_duration + 1, 2))
            | set(range(settled_start, probe_duration + 1))
        )
        tick_templates = [
            Text(str(t), font=FONT, font_size=20, color="#BDC9C0") for t in tick_times
        ]
        zoom_ticks = VGroup(*[label.copy() for label in tick_templates]).set_z_index(23)
        zoom_grid = VGroup(*[Line(UP, DOWN) for _ in tick_times]).set_z_index(23)

        def view_point(t, voltage):
            x = (t - ~view_start) * probe_duration / (probe_duration - ~view_start)
            return probe_origin + x * probe_x_unit + voltage * probe_y_unit

        def update_probe_view(_):
            times = np.linspace(~view_start, probe_duration, len(probe_times))
            x = np.linspace(0, probe_duration, len(probe_times))
            for trace, values in (
                (input_trace, probe_amplitude * np.cos(TAU * probe_frequency * times)),
                (node_trace, np.interp(times, probe_times, probe_voltage)),
            ):
                trace.set_points_smoothly(
                    bow(
                        probe_origin
                        + x[:, None] * probe_x_unit
                        + values[:, None] * probe_y_unit
                    )
                )
            for t, template, label, line in zip(
                tick_times, tick_templates, zoom_ticks, zoom_grid
            ):
                opacity = 1 if t % 2 == 0 else ~view_start / settled_start
                if t < ~view_start:
                    label.set_opacity(0)
                    line.set_opacity(0)
                    continue
                label.become(
                    template.copy()
                    .next_to(view_point(t, -6), DOWN, buff=h * 0.012)
                    .apply_function(bow)
                    .set_opacity(opacity)
                    .set_z_index(23)
                )
                line.become(
                    Line(
                        view_point(t, -6),
                        view_point(t, 6),
                        color="#91ABA9",
                        stroke_width=0.8,
                        stroke_opacity=0.22 * opacity,
                    )
                    .apply_function(bow)
                    .set_z_index(23)
                )
            for t, input_value, node_value, sample in zip(
                sample_times, sample_input, sample_voltage, samples
            ):
                if t < ~view_start:
                    sample.set_opacity(0)
                    continue
                for index, value, color in (
                    (0, input_value, YELLOW),
                    (2, node_value, BLUE),
                ):
                    sample[index].become(
                        Line(
                            view_point(t, 0),
                            view_point(t, value),
                            color=color,
                            stroke_width=1.3,
                            stroke_opacity=0.6,
                        )
                        .apply_function(bow)
                        .set_z_index(23.5)
                    )
                    sample[index + 1].move_to(bow(view_point(t, value)))

        for mob in (input_trace, node_trace, input_probe_dot, node_probe_dot):
            mob.clear_updaters()
        probe_ticks.remove(*probe_ticks[: probe_duration // 2 + 1])
        probe_grid.remove(*probe_grid[: probe_duration // 2 + 1])
        zoom_content = Group(input_trace, node_trace, samples, zoom_ticks, zoom_grid)
        update_probe_view(zoom_content)
        self.add(view_start, zoom_content)
        self.next_section(
            "Fit selected interval", skip_animations=skip_animations(False)
        )
        self.play(
            view_start @ settled_start,
            UpdateFromFunc(zoom_content, update_probe_view),
            run_time=1.3,
            rate_func=ease_in_out_cubic,
        )
        self.play(selection_opacity @ 0, run_time=0.3)
        self.remove(selection, selection_end, selection_opacity)
        probe_axes.x_range = [settled_start, probe_duration, 1]
        probe_axes.x_axis.x_range = np.array(probe_axes.x_range, dtype=float)
        probe_axes.x_axis.x_min, probe_axes.x_axis.x_max, probe_axes.x_axis.x_step = (
            probe_axes.x_range
        )
        probe_axes.x_axis.unit_size = probe_axes.x_axis.get_unit_size()
        self.wait(2)

        z = (
            MathTex(r"z = a + jb")
            .scale(2)
            .move_to(self.camera.frame)
            .shift(LEFT * fw(self))
        )
        self.add(z)

        self.play(self.camera.frame.animate.shift(LEFT * fw(self)))

        self.wait(0.5)

        human = (
            ImageMobject("../props/static/human_outline_white.png")
            .scale_to_fit_height(fh(self, 0.6))
            .next_to(z, DOWN, fh(self, 0.14))
            .shift(LEFT * fw(self, 0.25))
        )
        camera_target = (
            self.camera.frame.copy().scale(1.2).set_y(Group(z, human).get_y())
        )

        mouse.clear_updaters()
        crt = Group(
            screen,
            probe_shell,
            probe_plot,
            zoom_content,
            input_probe_dot,
            node_probe_dot,
            mouse,
        )
        crt_center, crt_height = glass.get_center(), glass.height
        original_screen_row = screen_row

        def screen_row(y):
            return (
                original_screen_row(y) - crt_center
            ) * glass.height / crt_height + glass.get_center()

        crt_target = (
            crt.copy()
            .scale_to_fit_height(human.height)
            .move_to(human)
            .set_x(2 * z.get_x() - human.get_x())
        )
        crt.scale_to_fit_width(crt_target.width).move_to(crt_target).set_x(
            camera_target.get_right()[0] + crt_target.width / 2 + LARGE_BUFF
        )
        self.add(crt)

        human_bez = CubicBezier(
            z.get_bottom() + [0, -0.1, 0],
            z.get_bottom() + [0, -1.5, 0],
            human.get_top() + [0, 1.5, 0],
            human.get_top() + [0, 0.1, 0],
            stroke_width=3,
        )
        screen_bez = CubicBezier(
            z.get_bottom() + [0, -0.1, 0],
            z.get_bottom() + [0, -1.5, 0],
            crt_target.get_top() + [0, 1.5, 0],
            crt_target.get_top() + [0, 0.1, 0],
            stroke_width=3,
        )

        self.next_section("Human and computer", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=1.5).become(camera_target),
                z.animate.scale(1.5).move_to(z, DOWN),
                Create(human_bez),
                human.shift(DOWN * fh(self)).animate.shift(UP * fh(self)),
                Create(screen_bez),
                crt.animate(run_time=1.5, suspend_mobject_updating=False).move_to(
                    crt_target
                ),
                lag_ratio=0.3,
            )
        )
        self.wait(2)

        self.play(self.camera.frame.animate.shift(DOWN * fh(self)))


class Impedance(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        cap_relation = MathTex(r"V_C \sim I_C").scale(2)
        cap_relation[0][:2].set_color(YELLOW)
        cap_relation[0][3:5].set_color(BLUE)
        res_relation = MathTex(r"V_R = I_R R").scale(2).shift(DOWN * fh(self))
        res_relation[0][:2].set_color(YELLOW)
        res_relation[0][3:5].set_color(BLUE)

        self.play(
            LaggedStart(
                *[PopIn(m) for m in cap_relation[0]],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(Group(cap_relation, res_relation).animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        cap_ohms = MathTex(r"V_C = I_C ?").scale(2).move_to(cap_relation)
        cap_ohms[0][:2].set_color(YELLOW)
        cap_ohms[0][3:5].set_color(BLUE)

        self.play(
            LaggedStart(
                ReplacementTransform(cap_relation[0][:2], cap_ohms[0][:2]),
                FadeOut(cap_relation[0][2], shift=UP / 2, run_time=0.5),
                FadeIn(cap_ohms[0][2], shift=UP / 2, run_time=0.5),
                ReplacementTransform(cap_relation[0][3:5], cap_ohms[0][3:5]),
                GrowFromCenter(cap_ohms[0][-1]),
                lag_ratio=0.1,
            )
        )

        self.wait(0.5)

        cap_ohms_z = MathTex(r"V_C = I_C Z").scale(2).move_to(cap_ohms)
        cap_ohms_z[0][:2].set_color(YELLOW)
        cap_ohms_z[0][3:5].set_color(BLUE)

        r_arrow = Arrow(
            res_relation.get_right() + [0, 0, 0],
            cap_ohms_z.get_right() + [-0.2, 0.1, 0],
            path_arc=PI,
        )

        self.play(
            LaggedStart(
                GrowArrow(r_arrow),
                LaggedStart(
                    ReplacementTransform(cap_ohms[0][:-1], cap_ohms_z[0][:-1]),
                    FadeOut(cap_ohms[0][-1], shift=UP / 2, run_time=0.5),
                    FadeIn(cap_ohms_z[0][-1], shift=UP / 2, run_time=0.5),
                    lag_ratio=0.1,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        curr_z = MathTex(r"I_C = \frac{V_C}{Z}").scale(2).move_to(cap_ohms_z)
        curr_z[0][:2].set_color(BLUE)
        curr_z[0][3:5].set_color(YELLOW)

        ohms_time = (
            MathTex(r"I_C(t) = C \frac{d V_C(t) }{dt}")
            .scale(2)
            .next_to(curr_z, UL, LARGE_BUFF * 1.4)
            .shift(RIGHT * 2.5)
        )
        ohms_time[0][:5].set_color(BLUE)
        ohms_time[0][8:13].set_color(YELLOW)

        # self.add(ohms_time)

        arrow_start = ohms_time.get_right() + RIGHT * 0.3
        arrow_end = curr_z.get_left() + LEFT * 0.25 + UP * 0.1
        arrow_mid = (arrow_start + arrow_end) / 2 + LEFT * 0.4 + UP * 0.25
        impedance_arrow = TipableVMobject(
            color=GREEN, stroke_width=3, tip_corner_radius=0.02
        )
        impedance_arrow.start_new_path(arrow_start)
        impedance_arrow.add_cubic_bezier_curve_to(
            arrow_start + RIGHT * 3.1,
            arrow_mid + RIGHT * 2.25 + UP * 0.4,
            arrow_mid,
        )
        impedance_arrow.add_cubic_bezier_curve_to(
            arrow_mid + LEFT * 2.25 + DOWN * 0.4,
            arrow_end + LEFT * 3.5,
            arrow_end,
        )
        impedance_arrow.add_tip(tip_length=0.2, tip_width=0.2)

        self.play(
            FadeOut(r_arrow),
            res_relation.animate.shift(DOWN * fh(self)),
            self.camera.frame.animate(run_time=2).move_to(Group(curr_z, ohms_time)),
            LaggedStart(
                ReplacementTransform(cap_ohms_z[0][5], curr_z[0][5]),
                ReplacementTransform(
                    cap_ohms_z[0][3:5], curr_z[0][:2], path_arc=-PI / 2
                ),
                ReplacementTransform(
                    cap_ohms_z[0][:2], curr_z[0][3:5], path_arc=-PI / 2
                ),
                ReplacementTransform(cap_ohms_z[0][2], curr_z[0][2]),
                Create(curr_z[0][6]),
                LaggedStart(*[FadeIn(m) for m in ohms_time[0]], lag_ratio=0.08),
                GrowArrow(impedance_arrow, run_time=1.5),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        x = (
            MathTex(r"x")
            .scale(3)
            .next_to(self.camera.frame.get_corner(DL), UR, LARGE_BUFF * 2.7)
        )

        xsin = MathTex(r"x V_C(t)").scale(2).move_to(self.camera.frame.get_bottom())
        xsin[0][1:].set_color(YELLOW)

        xsin_dt = MathTex(r"x V_C(t) = \frac{d V_C(t)}{dt}").scale(2).move_to(xsin)
        xsin_dt[0][1:6].set_color(YELLOW)
        xsin_dt[0][8:13].set_color(YELLOW)

        self.play(GrowFromCenter(x))

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(x[0], xsin[0][0], introducer=True),
                TransformFromCopy(ohms_time[0][8:13], xsin[0][1:], introducer=True),
                self.camera.frame.animate.scale_to_fit_height(
                    Group(xsin, ohms_time, curr_z).height * 1.2
                ).move_to(Group(xsin, ohms_time, curr_z)),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ReplacementTransform(xsin[0], xsin_dt[0][:6], introducer=True),
                GrowFromCenter(xsin_dt[0][6]),
                TransformFromCopy(ohms_time[0][7:], xsin_dt[0][7:], introducer=True),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.remove(res_relation)

        self.play(
            xsin_dt[0][7:]
            .animate.scale_to_fit_width(fw(self, 0.2))
            .move_to(self.camera.frame.copy().shift(DOWN * fh(self))),
            self.camera.frame.animate.shift(DOWN * fh(self)),
        )

        self.wait(2)


class DerivToComplex(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        dv = MathTex(r"\frac{d V_C(t)}{dt}").scale_to_fit_width(fw(self, 0.2))
        dv[0][1:6].set_color(YELLOW)

        self.add(dv)

        self.wait(0.5)

        dv_full = MathTex(r"\frac{d V_0 \cos{(\omega t)}  }{dt}").scale(2)
        dv_full[0][1:10].set_color(YELLOW)

        self.play(
            LaggedStart(
                ReplacementTransform(dv[0][0], dv_full[0][0]),
                FadeOut(dv[0][1:6]),
                ReplacementTransform(dv[0][6:], dv_full[0][10:]),
                GrowFromCenter(dv_full[0][1:10]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        dv_solve = (
            MathTex(
                r"\frac{d V_0 \cos{(\omega t)}  }{dt}"
                r"= - \omega V_0 \sin{(\omega t)}"
            )
            .scale(2)
            .move_to(dv_full)
        )
        dv_solve[0][1:10].set_color(YELLOW)

        self.play(
            LaggedStart(
                ReplacementTransform(dv_full[0][:13], dv_solve[0][:13]),
                LaggedStart(*[PopIn(m) for m in dv_solve[0][13:]], lag_ratio=0.2),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        omega_arrow = Arrow(
            dv_solve[0][7].get_bottom() + [0, -0.1, 0],
            dv_solve[0][15].get_bottom() + [0, -0.1, 0],
            path_arc=PI,
            color=GREEN,
        )

        self.play(GrowArrow(omega_arrow))

        self.wait(0.5)

        cos_arrow_1 = Arrow(
            dv_solve[0][3:6].get_top() + [0, -0.1, 0],
            dv_solve[0][14].get_top() + [0, 0, 0],
            path_arc=-PI * 0.7,
            color=GREEN,
        )
        cos_arrow_2 = Arrow(
            dv_solve[0][3:6].get_top() + [0, -0.1, 0],
            dv_solve[0][18:21].get_top() + [0, 0, 0],
            path_arc=-PI * 0.6,
            color=GREEN,
        )

        self.play(
            LaggedStart(
                FadeOut(omega_arrow),
                GrowArrow(cos_arrow_1),
                GrowArrow(cos_arrow_2),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(cos_arrow_1, cos_arrow_2),
                dv_solve[0][15]
                .animate(rate_func=there_and_back_with_pause, run_time=2)
                .shift(UP * 0.3)
                .scale(1.2)
                .set_color(GREEN),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        phase_shift = (
            MathTex(r"\cos{(\omega t + 90^\circ)} = - \sin{(\omega t)}")
            .scale(2)
            .next_to(self.camera.frame.get_top(), DOWN, LARGE_BUFF)
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    dv_solve[0][3:9],
                    phase_shift[0][:6],
                    path_arc=-PI / 4,
                    introducer=True,
                ),
                TransformFromCopy(
                    dv_solve[0][9], phase_shift[0][10], path_arc=PI / 4, introducer=True
                ),
                TransformFromCopy(dv_solve[0][13], phase_shift[0][11], introducer=True),
                TransformFromCopy(dv_solve[0][14], phase_shift[0][12], introducer=True),
                TransformFromCopy(
                    dv_solve[0][18:], phase_shift[0][13:], introducer=True
                ),
                dv_solve.animate.set_opacity(0.2).shift(DOWN * 2),
                lag_ratio=0.3,
            ),
        )
        self.play(
            LaggedStart(
                *[PopIn(m) for m in phase_shift[0][6:10]],
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    m.animate(rate_func=there_and_back, run_time=0.4)
                    .scale(1.2)
                    .set_color(GREEN)
                    for m in phase_shift[0][6:10]
                ],
                lag_ratio=0.3,
            ),
        )


class CapacitorEquation(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        plot_window = 12
        sim = _capacitor_simulation(self, plot_window)
        plot, circuit = sim.plot, sim.circuit
        voltage_axes, _, v_label, _ = plot
        self.voltage_axes = voltage_axes

        if CAPACITOR_STATE_PATH.exists():
            with np.load(CAPACITOR_STATE_PATH) as state:
                for name in (
                    "amplitude",
                    "frequency",
                    "phase",
                    "time",
                    "voltage_scale",
                ):
                    getattr(sim, name).set_value(float(state[name]))
                history = [tuple(sample) for sample in state["history"]]
        else:
            print(
                f"{CAPACITOR_STATE_PATH.name} missing; render Capacitor first for a seamless cut"
            )
            sim.amplitude @= 5
            sim.frequency @= 0.25
            sim.time @= plot_window
            sim.phase @= (TAU * ~sim.frequency * plot_window) % TAU
            times = np.arange(0, plot_window + 1 / 60, 1 / 60)
            history = list(
                zip(times, ~sim.amplitude * np.cos(TAU * ~sim.frequency * times))
            )
        sim.history.clear()
        sim.history.extend(history)
        sim.source_voltage @= history[-1][1]
        sim.capacitor_voltage @= history[-1][1]

        axes_height = voltage_axes.height
        voltage_axes.stretch(0.6, 1).shift(UP * axes_height / 3)

        i_arrow = Arrow(UP * 2, DOWN * 2, color=BLUE).next_to(circuit[1], LEFT)
        i_label = (
            MathTex(r"I_{C}(t)")
            .set_color(BLUE)
            .scale(0.8)
            .next_to(i_arrow, LEFT, SMALL_BUFF * 0.1)
        )

        equation = (
            MathTex("I_C(t)", "=", r"C \frac{d V_C(t) }{dt}")
            .scale(1.4)
            .next_to(voltage_axes, DOWN, LARGE_BUFF)
        )
        equation[0].set_color(BLUE)
        equation[2][2:7].set_color(YELLOW)

        plot.update(0)
        circuit.update(0)
        self.add(plot, circuit, i_arrow, i_label, equation)

        self.next_section("Pan", skip_animations=skip_animations(False))

        self.play(
            LaggedStart(
                AnimationGroup(
                    equation.animate.scale_to_fit_width(fw(self, 0.35))
                    .set_x((ORIGIN + LEFT * fw(self, 2))[0])
                    .shift(RIGHT * fw(self, 0.3)),
                    voltage_axes.animate.scale_to_fit_width(fw(self, 0.5))
                    .set_x((ORIGIN + LEFT * fw(self, 2))[0])
                    .shift(RIGHT * fw(self, 0.3)),
                ),
                self.camera.frame.animate.shift(LEFT * fw(self, 2)).scale(1.2),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(2)

        self.next_section("Resistor plot", skip_animations=skip_animations(False))

        frame_x = self.camera.frame.get_x()
        resistor_axes = voltage_axes.copy().set_x(2 * frame_x - voltage_axes.get_x())
        resistor_label = (
            MathTex("V_R(t)", color=YELLOW)
            .rotate(PI / 2)
            .scale_to_fit_height(v_label.height)
            .next_to(resistor_axes, LEFT, MED_SMALL_BUFF)
        )
        resistor_trace = always_redraw(lambda: sim.trace(resistor_axes))
        resistor_dot = (
            Dot(radius=0.09, color=YELLOW)
            .set_z_index(5)
            .add_updater(lambda mob: sim.place_dot(mob, resistor_axes))
        )

        ohms = MathTex("I_R(t)", "=", r"\frac{1}{R}", "V_R(t)").scale(1.4)
        ohms[0].set_color(BLUE)
        ohms[3].set_color(YELLOW)
        ohms.scale(equation[0].height / ohms[0].height).move_to(equation).set_x(
            resistor_axes.get_x()
        )

        VGroup(resistor_axes, resistor_label).shift(LEFT * fw(self, 0.6))
        self.add(resistor_axes, resistor_label, resistor_trace, resistor_dot)
        self.play(
            VGroup(resistor_axes, resistor_label).animate.shift(RIGHT * fw(self, 0.6)),
            run_time=1.5,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(ohms[0]),
                GrowFromCenter(ohms[1]),
                GrowFromCenter(ohms[2]),
                TransformFromCopy(
                    resistor_label[0], ohms[3], path_arc=PI / 3, introducer=True
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.next_section("Scaled current", skip_animations=skip_animations(False))

        current_gain = VT(1)
        current_opacity = VT(0)
        current_trace = always_redraw(
            lambda: sim.trace(resistor_axes, ~current_gain, BLUE).set_stroke(
                opacity=~current_opacity
            )
        )
        current_dot = (
            Dot(radius=0.09, color=BLUE)
            .set_z_index(6)
            .add_updater(
                lambda mob: sim.place_dot(
                    mob, resistor_axes, ~current_gain
                ).set_opacity(~current_opacity)
            )
        )
        self.add(current_trace, current_dot)

        self.play(current_opacity @ 1, Indicate(ohms[0]), run_time=1.5)

        self.wait(0.5)

        self.play(current_gain @ 0.5, Indicate(ohms[2]), run_time=2)

        self.wait(2)

        # self.embed()
