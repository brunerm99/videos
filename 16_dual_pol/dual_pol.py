# dual_pol.py
import os
import sys
from random import shuffle
from turtle import width

import pandas as pd
import skrf as rf
from dotenv import load_dotenv
from manim import *
from manim.utils.color.X11 import BROWN1
from MF_Tools import VT
from networkx import center
from numpy.fft import fft, fftshift
from scipy import signal
from scipy.interpolate import PchipInterpolator, interp1d

sys.path.insert(0, "..")
from props import (
    Bjt,
    Capacitor,
    Fet,
    Inductor,
    Resistor,
    VideoMobject,
    WeatherRadarTower,
    cubic_bezier,
    ease_in_out_elastic,
    ease_out_elastic,
    get_amp,
    get_blocks,
    get_filt_block,
    get_phase_shifter,
    get_splitter,
)
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False

load_dotenv("../.env")
FONT = os.getenv("FONT", "")

BLOCKS = get_blocks()

GOOD = BLUE
OK = GREY
BAD = RED
TARGET1_COLOR = GREEN
TARGET2_COLOR = ORANGE
TARGET3_COLOR = BLUE
INPUT_COLOR = BLUE
OUTPUT_COLOR = ORANGE
GAIN_COLOR = GREEN
PAE_COLOR = YELLOW
PRECIP_COLOR = BLUE
HPOL_TX_COLOR = BLUE
VPOL_TX_COLOR = RED
HPOL_RX_COLOR = PURPLE
VPOL_RX_COLOR = ORANGE


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale


def lin2db(x):
    return 10 * np.log10(x)


class Idea3D(ThreeDScene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[-2, 2, 0.5],
            x_length=20,
            y_length=10,
            z_length=20,
            tips=False,
            x_axis_config=dict(
                # stroke_color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            y_axis_config=dict(
                # stroke_color=RED,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
            z_axis_config=dict(
                # stroke_color=ORANGE,
                stroke_width=DEFAULT_STROKE_WIDTH * 1,
                stroke_opacity=0.5,
            ),
        )
        self.add(axes)

        target_dist = 40
        u0 = VT(-2)
        u1 = VT(2)
        u0_rx = VT(target_dist)
        u1_rx = VT(target_dist + 4)
        hpol_tx_opacity = VT(1)
        hpol_rx_opacity = VT(0)
        vpol_tx_opacity = VT(0)
        vpol_rx_opacity = VT(0)
        hpol_tx_amp = VT(1)
        vpol_tx_amp = VT(1)
        hpol_rx_amp = VT(1)
        vpol_rx_amp = VT(0.6)
        hpol_rotation = VT(0)
        hpol_tx_highlight_opacity = VT(0)
        hpol_tx_highlight_x0 = VT(0)
        hpol_tx_highlight_x1 = VT(0)
        vpol_tx_highlight_x0 = VT(0)
        vpol_tx_highlight_x1 = VT(0)
        vpol_tx_width = VT(0)
        vpol_tx_rotation = VT(-PI / 2)
        hpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=HPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~hpol_rotation, RIGHT)
        ).set_shade_in_3d(True)
        hpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_tx_amp * np.sin(2 * PI * u), 0),
                color=YELLOW,
                t_range=(~hpol_tx_highlight_x0, ~hpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_tx_highlight = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=YELLOW,
                t_range=(~vpol_tx_highlight_x0, ~vpol_tx_highlight_x1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2.5,
                stroke_opacity=~hpol_tx_highlight_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_tx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_tx_amp * np.sin(2 * PI * u)),
                color=VPOL_TX_COLOR,
                t_range=(~u0, ~u1, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * ~vpol_tx_width,
                stroke_opacity=~hpol_tx_opacity,
            ).rotate(~vpol_tx_rotation, RIGHT)
        ).set_shade_in_3d(True)
        hpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, ~hpol_rx_amp * np.sin(2 * PI * u), 0),
                color=HPOL_RX_COLOR,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~hpol_rx_opacity,
            )
        ).set_shade_in_3d(True)
        vpol_rx = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda u: (u, 0, ~vpol_rx_amp * np.sin(2 * PI * u)),
                color=VPOL_RX_COLOR,
                t_range=(~u0_rx, ~u1_rx, 1 / 100),
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
                stroke_opacity=~vpol_rx_opacity,
            )
        ).set_shade_in_3d(True)

        self.add(
            hpol_tx,
            hpol_rx,
            vpol_tx,
            vpol_rx,
            hpol_tx_highlight,
            vpol_tx_highlight,
        )

        self.set_camera_orientation(
            zoom=0.4,
            theta=-90 * DEGREES,
            phi=0 * DEGREES,
            gamma=0,
        )

        self.wait(0.5)

        axes.save_state()
        self.move_camera(
            zoom=0.6,
            theta=-160 * DEGREES,
            phi=80 * DEGREES,
            gamma=0,
            run_time=3,
            added_anims=[
                u0 + 2,
                u1 + 2,
                axes.animate.shift(IN * 5 + UP * 5),
            ],
        )

        self.wait(0.5)

        sp1 = Sphere(
            axes.c2p(target_dist, 0, 0),
            radius=9,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp2 = Sphere(
            axes.c2p(target_dist, 2, -1),
            radius=8,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp3 = Sphere(
            axes.c2p(target_dist, -1, -1.2),
            radius=8,
            fill_opacity=0.3,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        ).set_shade_in_3d(True)
        sp4 = Sphere(
            axes.c2p(30, 3, -2),
            radius=9,
            fill_opacity=0.5,
            color=WHITE,
            fill_color=WHITE,
            stroke_opacity=0,
            resolution=(20, 20),
        )
        sp1.set_color(WHITE)
        sp2.set_color(WHITE)
        sp3.set_color(WHITE)
        sp4.set_color(WHITE)

        self.play(LaggedStart(FadeIn(sp1), FadeIn(sp2), FadeIn(sp3), lag_ratio=0.2))

        self.wait(0.5)

        self.play(hpol_rotation @ (2 * PI), run_time=3)

        self.wait(0.5)

        self.play(
            LaggedStart(
                hpol_tx_highlight_opacity @ 1,
                hpol_tx_highlight_x1 @ (~u1),
                hpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                vpol_tx_width @ 2,
                vpol_tx_rotation @ 0,
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        hpol_tx_highlight_x0 @= 0
        hpol_tx_highlight_x1 @= 0

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                hpol_tx_highlight_x1 @ (~u1),
                hpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                vpol_tx_highlight_x1 @ (~u1),
                vpol_tx_highlight_x0 @ (~u1),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                AnimationGroup(u0 + target_dist, u1 + target_dist),
                AnimationGroup(hpol_tx_opacity @ 0, vpol_tx_opacity @ 0),
                AnimationGroup(
                    hpol_rx_opacity @ 1,
                    vpol_rx_opacity @ 1,
                    u0_rx - target_dist,
                    u1_rx - target_dist,
                ),
                lag_ratio=0.3,
            ),
            run_time=10,
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(True))

        self.play(
            LaggedStart(
                sp1.animate.set_fill(opacity=0),
                sp3.animate.set_fill(opacity=0),
                sp2.animate.set_y(axes.c2p(0, 0, 0)[1]).set_z(axes.c2p(0, 0, 0)[2]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.move_camera(
            phi=PI / 2,
            theta=PI,
            # zoom=0.25,
            added_anims=[
                Group(sp3, sp2, sp1).animate.shift(-(IN * 5 + UP * 5)),
                axes.animate.set_stroke(opacity=0.3).shift(-(IN * 5 + UP * 5)),
                hpol_rx_opacity @ 0.7,
                vpol_rx_opacity @ 0.7,
            ],
        )

        self.wait(0.5)
        self.next_section(skip_animations=skip_animations(False))

        hv_relation = MathTex(r"\lvert H \rvert = \lvert V \rvert")
        hv_relation[0][1].set_color(HPOL_RX_COLOR)
        hv_relation[0][5].set_color(VPOL_RX_COLOR)
        for char in hv_relation[0]:
            char.set_opacity(0)
        self.add_fixed_in_frame_mobjects(hv_relation)
        hv_relation.scale(2).to_corner(DL, LARGE_BUFF)

        self.play(
            LaggedStart(
                vpol_rx_amp @ 1,
                LaggedStart(
                    *[m.animate.set_opacity(1) for m in hv_relation[0]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            sp2.animate.stretch(1.5, dim=1),
            vpol_rx_amp @ (~vpol_rx_amp / 1.5),
            run_time=3,
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                axes.animate.shift([0, 0, 20]),
                Group(sp2, sp3, sp1).animate.shift([0, 0, 100]),
                hv_relation.animate.set_opacity(0),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


class Idea2D(MovingCameraScene):
    def construct(self): ...


class DropShape(Scene):
    def construct(self):
        Deq_min = 1.5
        Deq_max = 6
        Deq_target = 5.25
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

        f_vn = interp1d(xvn, yvn, fill_value="extrapolate")
        f_vp = interp1d(xvp, yvp, fill_value="extrapolate")

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

        # Deq_nl = NumberLine(
        #     x_range=[Deq_min, Deq_max, 0.5],
        #     include_numbers=True,
        #     include_tip=False,
        #     length=config["frame_width"] * 0.6,
        # ).to_edge(DOWN)

        # def clamp_drop_y(y):
        #     drop_half_height = c2(~Deq)
        #     return np.clip(y, -0.92 * drop_half_height, 0.92 * drop_half_height)

        # def drop_half_width(y):
        #     return x_function(~Deq, clamp_drop_y(y))

        # def build_updraft_streamline(
        #     side,
        #     start_x_factor,
        #     lower_gap,
        #     upper_gap,
        #     top_gap,
        #     start_pad,
        #     end_pad,
        #     lower_y_factor,
        #     upper_y_factor,
        #     stroke_width,
        #     stroke_opacity,
        #     tip_scale,
        # ):
        #     drop_half_height = c2(~Deq)
        #     max_half_width = drop_half_width(0)
        #     lower_y = clamp_drop_y(lower_y_factor * drop_half_height)
        #     mid_lower_y = clamp_drop_y(0.55 * lower_y_factor * drop_half_height)
        #     shoulder_y = clamp_drop_y(-0.08 * drop_half_height)
        #     upper_y = clamp_drop_y(upper_y_factor * drop_half_height)
        #     mid_upper_y = clamp_drop_y(0.5 * (upper_y + 0.82 * drop_half_height))
        #     top_y = clamp_drop_y(0.82 * drop_half_height)
        #     start_x = 0.3 * start_x_factor * max_half_width + 0.7 * (
        #         drop_half_width(lower_y) + 0.35 * lower_gap
        #     )
        #     end_x = drop_half_width(top_y) + 0.75 * top_gap

        #     y_knots = np.array(
        #         [
        #             -drop_half_height - start_pad,
        #             lower_y,
        #             mid_lower_y,
        #             shoulder_y,
        #             upper_y,
        #             mid_upper_y,
        #             top_y,
        #             drop_half_height + end_pad,
        #         ]
        #     )
        #     x_knots = np.array(
        #         [
        #             start_x,
        #             drop_half_width(lower_y) + lower_gap,
        #             drop_half_width(mid_lower_y) + 0.7 * lower_gap + 0.3 * upper_gap,
        #             drop_half_width(shoulder_y) + upper_gap,
        #             drop_half_width(upper_y) + upper_gap,
        #             drop_half_width(mid_upper_y) + 0.45 * top_gap + 0.55 * upper_gap,
        #             drop_half_width(top_y) + top_gap,
        #             end_x,
        #         ]
        #     )
        #     x_interp = PchipInterpolator(y_knots, x_knots)

        #     curve = ParametricFunction(
        #         lambda y: rain_ax.c2p(side * float(x_interp(y)), y),
        #         t_range=[y_knots[0], y_knots[-1], (y_knots[-1] - y_knots[0]) / 120],
        #     ).set_stroke(
        #         WHITE,
        #         width=stroke_width,
        #         opacity=stroke_opacity,
        #     )
        #     if tip_scale <= 0:
        #         return curve

        #     tip_anchor = rain_ax.c2p(side * x_knots[-1], y_knots[-1])
        #     tail_anchor = rain_ax.c2p(side * x_knots[-2], y_knots[-2])
        #     tangent = normalize(tip_anchor - tail_anchor)
        #     tip = Triangle(
        #         fill_color=WHITE,
        #         fill_opacity=stroke_opacity,
        #         stroke_width=0,
        #     ).scale(tip_scale)
        #     tip.rotate(angle_of_vector(tangent) - PI / 2)
        #     tip.shift(tip_anchor - tip.get_boundary_point(tangent))
        #     return VGroup(curve, tip)

        # streamline_specs = [
        #     dict(
        #         start_x_factor=0.24,
        #         lower_gap=0.34,
        #         upper_gap=0.34,
        #         top_gap=0.44,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=2.5,
        #         stroke_opacity=0.58,
        #         tip_scale=0,
        #     ),
        #     dict(
        #         start_x_factor=0.48,
        #         lower_gap=0.56,
        #         upper_gap=0.56,
        #         top_gap=0.72,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=2.2,
        #         stroke_opacity=0.42,
        #         tip_scale=0,
        #     ),
        #     dict(
        #         start_x_factor=0.72,
        #         lower_gap=0.78,
        #         upper_gap=0.78,
        #         top_gap=1,
        #         start_pad=3,
        #         end_pad=3,
        #         lower_y_factor=-0.5,
        #         upper_y_factor=0.1,
        #         stroke_width=1.8,
        #         stroke_opacity=0.24,
        #         tip_scale=0,
        #     ),
        # ]
        # wind_lines = always_redraw(
        #     lambda: VGroup(
        #         *(
        #             build_updraft_streamline(side=side, **spec)
        #             for spec in streamline_specs
        #             for side in (-1, 1)
        #         ),
        #     ).set_z_index(-1)
        # )

        self.add(
            # wind_lines,
            lplot,
            rplot,
        )

        # # self.play(
        # #     Create(rain_ax),
        # #     Create(lplot),
        # #     Create(rplot),
        # # )

        # b1 = always_redraw(
        #     lambda: CubicBezier(
        #         rain_ax.i2gp(0.1, rplot),
        #         rain_ax.i2gp(0.1, rplot),
        #     )
        # )
        self.add(Dot(rain_ax.c2p(0.1, f_vp(0.1))))

        self.play(Deq @ Deq_target, run_time=6)

        self.wait(2)
