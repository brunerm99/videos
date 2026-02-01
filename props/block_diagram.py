# props/block_diagram.py
from manim import *


def get_blocks(color=WHITE):
    return {
        name: SVGMobject(
            f"../props/static/{name}.svg",
            fill_color=color,
            opacity=1,
            stroke_width=0.01,
        )
        for name in [
            "adc",
            "window_function",
            "filter",
            "range_norm",
            "product_calc",
            "computer",
            "bp_filter_generic",
            "mixer",
            "phase_detector",
            "amp",
            "phase_shifter",
            "spdt_switch",
            "lp_filter",
            "hp_filter",
            "antenna",
            "splitter",
            "oscillator",
            "limiter",
            "attenuator",
        ]
    }


def get_bd_animation(bd, lagged: bool = False, lag_ratio=0.8, run_time=None):
    animations = (
        *[
            Create(block)
            if type(block) in (Line, CubicBezier)
            else GrowFromCenter(block)
            for block in bd
        ],
    )
    if lagged:
        return LaggedStart(animations, lag_ratio=lag_ratio, run_time=run_time)
    return Succession(animations, run_time=run_time)


class ShowBlocks(Scene):
    def construct(self):
        blocks = get_blocks()
        labeled_blocks = Group()
        for name, block in blocks.items():
            label = Text(name).next_to(block, direction=UP, buff=SMALL_BUFF)
            labeled_blocks.add(Group(label, block))
        self.add(
            labeled_blocks.arrange_in_grid(4, 6).scale_to_fit_width(
                config["frame_width"] - 1
            )
        )


def get_diode():
    diode_tri = Triangle(color=WHITE).rotate(PI / 3)
    diode_line = Line(LEFT * diode_tri.width / 2, RIGHT * diode_tri.width / 2).next_to(
        diode_tri, direction=DOWN, buff=0
    )
    diode_line_dot_1 = Dot(
        radius=DEFAULT_DOT_RADIUS / 4,
        fill_color=WHITE,
        fill_opacity=1,
        stroke_color=WHITE,
    ).move_to(diode_line.get_start())
    diode_line_dot_2 = Dot(
        radius=DEFAULT_DOT_RADIUS / 2,
        fill_color=WHITE,
        fill_opacity=1,
        stroke_color=WHITE,
    ).move_to(diode_line.get_end())
    diode_conn_1 = Line(diode_tri.get_top() + UP / 2, diode_tri.get_top())
    diode_conn_2 = Line(diode_tri.get_bottom() + DOWN / 2, diode_tri.get_bottom())
    diode = VGroup(
        diode_tri,
        diode_line,
        diode_line_dot_1,
        diode_line_dot_2,
        diode_conn_1,
        diode_conn_2,
    )

    return diode


class Resistor(VMobject):
    def __init__(
        self, width=1, label=None, direction=DOWN, stroke_width_mult=2, **kwargs
    ):
        super().__init__(**kwargs)
        self._direction = direction

        rotation = -PI / 4
        res_line_start = Line(
            LEFT / 2, ORIGIN, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        ).rotate(-rotation / 2)
        res_line_start_inv = Line(
            ORIGIN, RIGHT / 2, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        ).rotate(-rotation / 2)
        res_line_1 = VGroup(res_line_start, res_line_start_inv)
        res_line_2 = (
            Line(
                LEFT / 2,
                RIGHT / 2,
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            )
            .rotate(rotation / 2)
            .next_to(res_line_1, direction=DOWN, buff=0)
        )
        res_line_3 = (
            Line(
                LEFT / 2,
                RIGHT / 2,
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            )
            .rotate(-rotation / 2)
            .next_to(res_line_2, direction=DOWN, buff=0)
        )
        res_line_4 = (
            Line(
                LEFT / 2,
                RIGHT / 2,
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            )
            .rotate(rotation / 2)
            .next_to(res_line_3, direction=DOWN, buff=0)
        )
        res_line_end_inv = Line(
            LEFT / 2, ORIGIN, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        )
        res_line_end = Line(
            ORIGIN, RIGHT / 2, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        )
        res_line_5 = (
            VGroup(res_line_end, res_line_end_inv)
            .rotate(-rotation / 2)
            .next_to(res_line_4, direction=DOWN, buff=0)
        )
        res_conn_1 = Line(
            res_line_start.get_end() + UP / 2,
            res_line_start.get_end(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        res_conn_2 = Line(
            res_line_end.get_start() + DOWN / 2,
            res_line_end.get_start(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        self.main_body = VGroup(
            res_conn_1,
            res_line_start,
            res_line_2,
            res_line_3,
            res_line_4,
            res_line_end,
            res_conn_2,
        ).scale_to_fit_height(width)

        self.add(self.main_body)

        if label is not None:
            self.label = (
                Tex(str(label) + r" $\Omega$")
                .scale(0.5)
                .next_to(self.main_body, self._direction, buff=0.1)
            )
            self.add(self.label)
        else:
            self.label = None

    def get_terminals(self, val):
        if val == "top":
            return self.main_body[0].get_start()
        elif val == "bottom":
            return self.main_body[-1].get_end()

    def center(self):
        self.shift(
            DOWN * self.main_body.get_center()[1] + LEFT * self.main_body.get_center()
        )
        return self

    def rotate(self, angle, *args, **kwargs):
        kwargs.setdefault("about_point", self.main_body.get_center())
        super().rotate(angle, *args, **kwargs)
        if self.label is not None:
            self.label.rotate(-angle).next_to(self.main_body, self._direction, buff=0.1)
        return self


def get_splitter(width, n=2):
    box = RoundedRectangle(
        width=width,
        height=width * (n / 2),
        stroke_width=DEFAULT_STROKE_WIDTH * 2,
    ).set_z_index(1)
    out = box.get_edge_center(RIGHT)
    jump = 1 / n

    def get_bez(idx):
        start = box.get_corner(UL) + DOWN * box.height * (idx * jump + jump / 2)
        return CubicBezier(
            start,
            start + [box.width / 2, 0, 0],
            out + [-box.width / 2, 0, 0],
            out,
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

    bezs = [get_bez(idx) for idx in range(n)]

    return Group(box, *bezs)


def get_phase_shifter(width):
    box = RoundedRectangle(
        width=width,
        height=width,
        stroke_width=DEFAULT_STROKE_WIDTH * 2,
    ).set_z_index(1)
    phi = MathTex(r"\Large \phi").scale_to_fit_height(box.height * 0.7).move_to(box)
    arrow = Arrow(
        box.get_corner(DL),
        box.get_corner(UR),
        buff=MED_LARGE_BUFF,
        color=BLUE,
        stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
    )
    return Group(box, phi, arrow)


def get_filt_block(width, passband="band"):
    filt_box = RoundedRectangle(
        width=width,
        height=width,
        stroke_width=DEFAULT_STROKE_WIDTH * 2,
    ).set_z_index(1)
    ax = (
        Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            tips=False,
            x_length=filt_box.width * 0.9,
            y_length=filt_box.height * 0.9,
            x_axis_config={"include_ticks": False},
            y_axis_config={"include_ticks": False},
        )
        .set_opacity(0)
        .move_to(filt_box)
    )
    A = 0.4
    diff = 0.5
    mid = ax.plot(
        lambda t: A * np.sin(2 * PI * t),
        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        color=BLUE if passband == "band" else RED,
    )
    hi = ax.plot(
        lambda t: A * np.sin(2 * PI * t) + diff,
        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        color=BLUE if passband == "high" else RED,
    )
    lo = ax.plot(
        lambda t: A * np.sin(2 * PI * t) - diff,
        stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        color=BLUE if passband == "low" else RED,
    )
    return Group(filt_box, mid, hi, lo)


def get_amp(width, stroke_width_mult=2):
    amp_box = RoundedRectangle(
        width=width,
        height=width,
        stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        corner_radius=width * 0.25,
    ).set_z_index(1)
    amp_tri = (
        Triangle(stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult, color=GREEN)
        .scale_to_fit_width(amp_box.width * 0.7)
        .rotate(PI / 6)
        .move_to(amp_box)
        .set_z_index(1)
    )
    return Group(amp_box, amp_tri)


class Fet(VMobject):
    def __init__(
        self, width=1, stroke_width_mult=2, mode="depletion", channel="n", **kwargs
    ):
        super().__init__(**kwargs)

        center_line = Line(
            UP, DOWN, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        )

        right_line_buff = 0
        right_line_len = center_line.height / 3
        if mode == "enhancement":
            right_line_buff = center_line.height * 0.2
            right_line_len = center_line.height * 0.2

        l1 = Line(
            center_line.get_top(),
            center_line.get_top() + DOWN * right_line_len,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        l2 = l1.copy().next_to(l1, DOWN, right_line_buff)
        l3 = l1.copy().next_to(l2, DOWN, right_line_buff)
        right_line = VGroup(l1, l2, l3)
        right_line.next_to(center_line, RIGHT, MED_SMALL_BUFF)

        self.gate = Line(
            center_line.get_bottom(),
            center_line.get_bottom() + LEFT * center_line.height * 0.5,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        drain_start = Line(
            l1.get_midpoint(),
            l1.get_midpoint() + RIGHT * self.gate.width * 0.5,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        self.drain = Line(
            drain_start.get_end(),
            drain_start.get_end() + UP * self.gate.width,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        source_start = Line(
            l3.get_midpoint(),
            l3.get_midpoint() + RIGHT * self.gate.width * 0.5,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        self.source = Line(
            source_start.get_end(),
            source_start.get_end() + DOWN * self.gate.width,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        if channel == "n":
            source_dir = Arrow(
                l2.get_midpoint() + RIGHT * self.gate.width * 0.5,
                l2.get_midpoint(),
                buff=0,
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
                max_stroke_width_to_length_ratio=999,
                max_tip_length_to_length_ratio=0.5,
            )
        elif channel == "p":
            source_dir = Arrow(
                l2.get_midpoint(),
                l2.get_midpoint() + RIGHT * self.gate.width * 0.5,
                buff=0,
                max_stroke_width_to_length_ratio=999,
                max_tip_length_to_length_ratio=0.5,
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            )
        else:
            raise ValueError("channel must be 'p' or 'n'")

        from_source_dir = Line(
            [source_start.get_right()[0], source_dir.get_y(), 0],
            source_start.get_right(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        self.main_body = VGroup(
            center_line,
            right_line,
            self.gate,
            drain_start,
            self.drain,
            source_start,
            self.source,
            source_dir,
            from_source_dir,
        ).scale_to_fit_width(width)

        self.add(self.main_body)

    def get_terminals(self, val):
        if val == "drain":
            return self.drain.get_top()
        elif val == "source":
            return self.source.get_bottom()
        elif val == "gate":
            return self.gate.get_left()


class Bjt(VMobject):
    def __init__(self, width=1, stroke_width_mult=2, **kwargs):
        super().__init__(**kwargs)

        center_line = Line(
            UP, DOWN, stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult
        )
        base_line = Line(
            center_line.get_center() + LEFT * 0.5,
            center_line.get_center(),
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        base = Circle(
            radius=center_line.height * 0.05,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            color=WHITE,
        ).next_to(base_line, LEFT, 0)
        collector = Line(
            center_line.get_center() + UP * center_line.height / 8,
            center_line.get_top() + RIGHT * 0.9,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        emitter = Arrow(
            center_line.get_center() + DOWN * center_line.height / 8,
            center_line.get_bottom() + RIGHT * 0.9,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            buff=0,
            max_stroke_width_to_length_ratio=100,
        )

        self.main_body = VGroup(
            base,
            collector,
            emitter,
            center_line,
            base_line,
        ).scale_to_fit_width(width)

        self.base = base
        self.collector = collector
        self.emitter = emitter

        self.add(self.main_body)

    def get_terminals(self, val):
        if val == "base":
            return self.base.get_center()
        elif val == "collector":
            return self.collector.get_end()
        elif val == "emitter":
            return self.emitter.get_end()


class Inductor(VMobject):
    def __init__(
        self, width=1, label=None, direction=DOWN, stroke_width_mult=2, **kwargs
    ):
        super().__init__(**kwargs)
        self._direction = direction

        self.main_body = (
            ParametricFunction(
                (lambda t: ((np.cos(t) / 1.94) + (t / (2.21 * PI)), -np.sin(t), 0)),
                t_range=(-PI, 8 * PI),
                stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
            )
            .scale_to_fit_width(width)
            .center()
        )

        self.add(self.main_body)

        if not label is None:
            self.label = (
                Tex(str(label) + " H")
                .scale(0.5)
                .next_to(self.main_body, self._direction, buff=0.1)
            )
            self.add(self.label)
        else:
            self.label = None

    def get_anchors(self):
        return [self.main_body.get_start(), self.main_body.get_end()]

    def get_terminals(self, val):
        if val == "left":
            return self.main_body.get_start()
        elif val == "right":
            return self.main_body.get_end()

    def center(self):
        self.shift(
            DOWN * self.main_body.get_center()[1] + LEFT * self.main_body.get_center()
        )

        return self

    def rotate(self, angle, *args, **kwargs):
        super().rotate(angle, about_point=self.main_body.get_center(), *args, **kwargs)
        if not self.label == None:
            self.label.rotate(-angle).next_to(self.main_body, self._direction, buff=0.1)

        return self


class Capacitor(VMobject):
    def __init__(
        self, width, label=None, direction=DOWN, stroke_width_mult=2, **kwargs
    ):
        # initialize the vmobject
        super().__init__(**kwargs)
        self._direction = direction

        self.left_plate = Line(
            ORIGIN,
            UP * width * 0.75,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        self.right_plate = Line(
            ORIGIN,
            UP * width * 0.75,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        ).next_to(self.left_plate, RIGHT, width * 0.25)
        self.p1 = Line(
            self.left_plate.get_center(),
            self.left_plate.get_center() + LEFT * 0.375,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )
        self.p2 = Line(
            self.right_plate.get_center(),
            self.right_plate.get_center() + RIGHT * 0.375,
            stroke_width=DEFAULT_STROKE_WIDTH * stroke_width_mult,
        )

        self.main_body = VGroup(self.p1, self.left_plate, self.right_plate, self.p2)

        self.add(self.main_body)

        # check if lebel is present.
        if not label is None:
            self.label = (
                Tex(str(label) + "F")
                .scale(0.5)
                .next_to(self.main_body, self._direction, buff=0.1)
            )
            self.add(self.label)

    def center(self):
        self.shift(
            DOWN * self.main_body.get_center()[1] + LEFT * self.main_body.get_center()
        )

        return self

    def rotate(self, angle, *args, **kwargs):
        super().rotate(angle, about_point=self.main_body.get_center(), *args, **kwargs)
        if not self.label == None:
            self.label.rotate(-angle).next_to(self.main_body, self._direction, buff=0.1)

        return self


class Ground(VMobject):
    def __init__(self, ground_type="ground", label=None, **kwargs):
        # initialize the vmobject
        super().__init__(**kwargs)

        if ground_type == "ground":
            self.main_body = VGroup(Polygon([0, 0, 0], [2, 0, 0], [1, -1, 0]))
            if not label is None and label == "D" or label == "A":
                self.main_body.add(Text(label).move_to(self.main_body))
                # 'D' or 'A' for digital vs analog ground
                pass

        elif ground_type == "earth":
            self.main_body = VGroup(
                Line([0, 0, 0], [2, 0, 0]),
                Line([(1 / 3), -(1 / 3), 0], [(5 / 3), -(1 / 3), 0]),
                Line([(2 / 3), -(2 / 3), 0], [(4 / 3), -(2 / 3), 0]),
            )

        # tail for ground:
        self.add(self.main_body)

        # Scale down to match the scale of other electrical mobjects
        self.main_body.set_color(WHITE)
        self.main_body.stroke_opacity = 1

        self.main_body.center().scale(0.25).center()

    def get_terminals(self, *args):
        if len(self.main_body) != 3:
            return self.main_body[0].point_from_proportion(1 / (2 + 2 * np.sqrt(2)))
        else:
            return self.main_body[0].point_from_proportion(0.5)
