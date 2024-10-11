# drop_shape.py

from manim import *
from MF_Tools import VT

PRECIP_COLOR = BLUE
H_COLOR = GREEN
V_COLOR = ORANGE


class DropShape(Scene):
    def construct(self):
        Deq_min = 1.5
        Deq_max = 6
        Deq = VT(Deq_min)

        def c1(Deq):
            return (1 / np.pi) * (0.02914 * Deq**2 + 0.9263 * Deq + 0.07791)

        def c2(Deq):
            return -0.01938 * Deq**2 + 0.4698 * Deq + 0.09538

        def c3(Deq):
            return -0.06123 * Deq**3 + 1.3880 * Deq**2 - 10.41 * Deq + 28.34

        def c4(Deq):
            if Deq > 4:
                return -0.01352 * Deq**3 + 0.2014 * Deq**2 - 0.8964 * Deq + 1.226
            elif 1.5 <= Deq <= 4:
                return 0
            else:
                return np.nan

        def x_function(Deq, y):
            c1_val = c1(Deq)
            c2_val = c2(Deq)
            c3_val = c3(Deq)
            c4_val = c4(Deq)

            if np.abs(y / c2_val) > 1:
                return np.nan

            term1 = np.sqrt(1 - (y / c2_val) ** 2)
            term2 = np.arccos(y / (c3_val * c2_val))
            term3 = c4_val * (y / c2_val) ** 2 + 1

            return c1_val * term1 * term2 * term3

        y_values = np.linspace(-2.5, 2.5, 10_000)

        x_values_positive = [x_function(~Deq, y) for y in y_values]
        x_values_negative = [-x_function(~Deq, y) for y in y_values]

        xvn = np.array(x_values_negative)
        ind = np.where(~np.isnan(xvn))
        xvn = xvn[ind]
        yvn = np.array(y_values)[ind]

        xvp = np.array(x_values_positive)
        ind = np.where(~np.isnan(xvp))
        xvp = xvp[ind]
        yvp = np.array(y_values)[ind]

        rain_ax = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={
                "stroke_color": LIGHT_GRAY,
                "stroke_opacity": 0.6,
            },
        )

        hail_ax = rain_ax.copy()

        rain_label = Tex("Rain Drop").next_to(rain_ax, UP)
        hail_label = Tex("Hail").next_to(hail_ax, UP)

        rain_group = VGroup(rain_label, rain_ax)
        hail_group = VGroup(hail_label, hail_ax)
        VGroup(rain_group, hail_group).arrange(RIGHT, LARGE_BUFF).to_edge(UP)

        hail_plot = always_redraw(
            lambda: Circle(
                (hail_ax.c2p(1, 0)[0] - hail_ax.c2p(0, 0)[0]) * ~Deq / 2,
                color=PRECIP_COLOR,
            ).move_to(hail_ax.c2p(0, 0))
        )

        lplot = rain_ax.plot_line_graph(
            xvn,
            yvn,
            line_color=PRECIP_COLOR,
            add_vertex_dots=False,
        )
        rplot = rain_ax.plot_line_graph(
            xvp,
            yvp,
            line_color=PRECIP_COLOR,
            add_vertex_dots=False,
        )

        def get_plot_updater(scalar=1):
            def updater(m: Mobject):
                x_values = np.array([scalar * x_function(~Deq, y) for y in y_values])
                ind = np.where(~np.isnan(x_values))
                xv = x_values[ind]
                yv = np.array(y_values)[ind]
                m.become(
                    rain_ax.plot_line_graph(
                        xv, yv, add_vertex_dots=False, line_color=PRECIP_COLOR
                    )
                )

            return updater

        lplot.add_updater(get_plot_updater(scalar=-1))
        rplot.add_updater(get_plot_updater(scalar=1))

        Deq_nl = NumberLine(
            x_range=[Deq_min, Deq_max, 0.5],
            include_numbers=True,
            include_tip=False,
            length=config["frame_width"] * 0.6,
        ).to_edge(DOWN)

        Deq_label = Tex(r"$D_{eq}$ | Equivalent Drop Diameter (mm)").next_to(Deq_nl, UP)

        Deq_dot = always_redraw(lambda: Dot(Deq_nl.n2p(~Deq)))

        # self.add(
        #     rain_ax,
        #     lplot,
        #     rplot,
        #     Deq_nl,
        #     Deq_dot,
        #     hail_plot,
        #     hail_ax,
        #     Deq_label,
        #     rain_label,
        #     hail_label,
        # )

        self.play(
            Create(rain_ax),
            Create(hail_ax),
            Create(Deq_nl),
            FadeIn(Deq_label),
        )
        self.play(
            FadeIn(rain_label, hail_label, shift=DOWN),
            Create(lplot),
            Create(rplot),
            Create(hail_plot),
            Create(Deq_dot),
        )

        self.next_section(skip_animations=False)
        self.play(Deq @ Deq_max, run_time=6)

        self.next_section(skip_animations=False)
        self.wait(1)

        rain_Zh = Line(rain_ax.c2p(0, 0), rain_ax.c2p(4, 0), color=H_COLOR).add_tip()
        rain_Zv = Line(rain_ax.c2p(0, 0), rain_ax.c2p(0, 2.6), color=V_COLOR).add_tip()

        rain_Zh_label = MathTex("Z_H", color=H_COLOR).next_to(rain_Zh, RIGHT)
        rain_Zv_label = MathTex("Z_V", color=V_COLOR).next_to(rain_Zv, UP)

        hail_Zh = Line(hail_ax.c2p(0, 0), hail_ax.c2p(4, 0), color=H_COLOR).add_tip()
        hail_Zv = Line(hail_ax.c2p(0, 0), hail_ax.c2p(0, 4), color=V_COLOR).add_tip()

        hail_Zh_label = MathTex("Z_H", color=H_COLOR).next_to(hail_Zh, RIGHT)
        hail_Zv_label = (
            MathTex("Z_V", color=V_COLOR).next_to(hail_Zv, UP).shift(RIGHT + DOWN / 2)
        )

        self.play(FadeOut(rain_ax, hail_ax, Deq_dot, Deq_nl, Deq_label))
        self.play(
            LaggedStart(
                Create(rain_Zh),
                Create(rain_Zh_label),
                Create(rain_Zv),
                Create(rain_Zv_label),
                Create(hail_Zh),
                Create(hail_Zh_label),
                Create(hail_Zv),
                Create(hail_Zv_label),
                lag_ratio=0.5,
            )
        )

        self.wait(1)

        rain_zdr = MathTex("Z_H", ">", "Z_V").scale(1.2).next_to(rain_ax, DOWN)
        hail_zdr = MathTex("Z_H", "=", "Z_V").scale(1.2).next_to(hail_ax, DOWN)
        rain_zdr[0].set_color(H_COLOR)
        rain_zdr[2].set_color(V_COLOR)
        hail_zdr[0].set_color(H_COLOR)
        hail_zdr[2].set_color(V_COLOR)

        self.play(
            LaggedStart(
                TransformFromCopy(rain_Zh_label, rain_zdr[0]),
                Create(rain_zdr[1]),
                TransformFromCopy(rain_Zv_label, rain_zdr[2]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(hail_Zh_label, hail_zdr[0]),
                Create(hail_zdr[1]),
                TransformFromCopy(hail_Zv_label, hail_zdr[2]),
                lag_ratio=0.4,
            )
        )

        self.wait(2)
