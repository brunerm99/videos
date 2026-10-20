# complex_numbers_gx.py
import pathlib
from collections import deque
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from manimgx import *
from manimgx import Capacitor as Cap

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


def _capacitor_diagram(scene, capacitor_voltage, voltage_scale, plot_window=12):
    wire_color = "#D6EEF7"
    axes = Axes(
        x_range=[0, plot_window, 2],
        y_range=[-1.2, 1.2, 0.5],
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


def _capacitor_simulation(scene, plot_window=12):
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
        scene, capacitor_voltage, voltage_scale, plot_window
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
            gain * samples[:, 1] / ~voltage_scale,
            add_vertex_dots=False,
            line_color=color,
            stroke_width=4 * axes.width / axes_width,
        )

    def place_dot(dot, axes, gain=1):
        return dot.scale_to_fit_width(0.18 * axes.width / axes_width).move_to(
            axes.c2p(min(~time, plot_window), gain * ~source_voltage / ~voltage_scale)
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
        voltage_axes, _, v_label, _ = plot
        self.voltage_axes = voltage_axes

        self.add(plot, circuit)

        self.next_section("Uncharged", skip_animations=skip_animations(False))
        self.wait(1)

        electron_line = SquigglyLine(
            circuit[-1][3].get_left() + LEFT * 1.5,
            circuit[-1][3].get_left() + LEFT * 0.2,
            color=BLUE,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )
        electron_label = (
            Text("electrons", font=FONT)
            .set_color(BLUE)
            .scale(0.4)
            .next_to(electron_line, LEFT, SMALL_BUFF)
        )

        self.play(GrowArrow(electron_line), FadeIn(electron_label))

        self.wait(1)

        self.play(FadeOut(electron_label, electron_line))

        self.wait(0.5)

        self.next_section("Apply DC", skip_animations=skip_animations(False))
        self.play(amplitude @ 5, run_time=1)
        self.wait(0.4)

        self.next_section("Hold DC", skip_animations=skip_animations(False))
        self.wait(2)

        self.next_section("Apply AC", skip_animations=skip_animations(False))

        i_arrow = Arrow(UP * 2, DOWN * 2, color=BLUE).next_to(circuit[1], LEFT)
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

        cap_label = circuit[1][-1]
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

        np.savez(
            CAPACITOR_STATE_PATH,
            amplitude=~amplitude,
            frequency=~frequency,
            phase=~sim.phase,
            time=~sim.time,
            voltage_scale=~sim.voltage_scale,
            history=np.array(sim.history),
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
