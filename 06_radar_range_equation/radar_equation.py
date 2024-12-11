# radar_equation.py

import sys
import warnings

from manim import *
from random import shuffle
from MF_Tools import TransformByGlyphMap, VT
from scipy.interpolate import interp1d
from scipy.constants import c

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props import WeatherRadarTower
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def compute_af_1d(weights, d_x, k_0, u, u_0):
    n = np.arange(weights.size)
    AF = np.sum(
        weights[:, None] * np.exp(1j * n[:, None] * d_x * k_0 * (u - u_0)), axis=0
    )
    AF /= AF.max()
    return AF


class EquationIntro(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        radar_eqn_messy = MathTex(
            r"P_r = \frac{P_t G_t \sigma A_e}{16 \pi^2 R^4}",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )
        radar_eqn = MathTex(
            r"P_r = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )

        self.play(LaggedStart(GrowFromCenter(m) for m in radar_eqn_messy[0]))

        # self.play(
        #     LaggedStart(
        #         GrowFromCenter(radar_eqn[0][0:2]),
        #         GrowFromCenter(radar_eqn[0][2]),
        #         GrowFromCenter(radar_eqn[0][3:7]),
        #         GrowFromCenter(radar_eqn[0][7]),
        #         GrowFromCenter(radar_eqn[0][8:12]),
        #         GrowFromCenter(radar_eqn[0][12]),
        #         GrowFromCenter(radar_eqn[0][13]),
        #         GrowFromCenter(radar_eqn[0][14]),
        #         GrowFromCenter(radar_eqn[0][15:19]),
        #         GrowFromCenter(radar_eqn[0][19]),
        #         GrowFromCenter(radar_eqn[0][20:22]),
        #         lag_ratio=0.15,
        #     )
        # )

        self.wait(0.5)

        weather_radar = WeatherRadarTower()
        weather_radar.vgroup.scale(0.6).to_edge(LEFT, LARGE_BUFF).shift(DOWN)
        cloud = SVGMobject(
            "../props/static/clouds.svg", fill_color=WHITE, stroke_color=WHITE
        ).to_edge(RIGHT, LARGE_BUFF)

        weather_ex = Group(weather_radar.vgroup, cloud).next_to(
            [0, -config.frame_height / 2, 0], DOWN
        )
        weather_ex_old_y = weather_ex.get_y()

        self.play(Group(radar_eqn_messy, weather_ex).animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        beam_u = Line(
            weather_radar.radome.get_right(),
            cloud.get_corner(UL),
            SMALL_BUFF,
            color=TX_COLOR,
        )
        beam_l = Line(
            weather_radar.radome.get_right(),
            cloud.get_corner(DL),
            SMALL_BUFF,
            color=TX_COLOR,
        )
        radar_return = Arrow(
            cloud.get_left(), weather_radar.radome.get_right(), color=RX_COLOR
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(beam_u), Create(beam_l)),
                GrowArrow(radar_return),
                lag_ratio=0.3,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            Group(weather_ex, beam_u, beam_l, radar_return).animate.set_y(
                weather_ex_old_y
            ),
            radar_eqn_messy.animate.move_to(ORIGIN),
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                radar_eqn_messy,
                radar_eqn,
                ([0, 1, 2], [0, 1, 2]),
                ([3, 4, 5, 6], [3, 4, 5, 6]),
                ([10], [7]),
                ([11, 12], [8]),
                ([13], [9]),
                ([14], ShrinkToCenter),
                ([15], [10]),
                ([16], [11]),
                ([7], [13]),
                ([8, 9], [20, 21]),
                ([13], [16]),
                ([14], [18]),
                ([11, 12], [15]),
                ([15], [17]),
                ([16], [18]),
                (GrowFromCenter, [14], {"delay": 0.4}),
                (GrowFromCenter, [12], {"delay": 0.4}),
                (GrowFromCenter, [19], {"delay": 0.4}),
            )
        )

        self.wait(0.5)

        section_1 = radar_eqn[0][3:12]
        section_2 = radar_eqn[0][13:19]
        section_3 = radar_eqn[0][20:22]

        section_1_color = GREEN
        section_2_color = BLUE
        section_3_color = YELLOW

        section_1_label = Tex(
            "1", font_size=DEFAULT_FONT_SIZE * 1.5, color=section_1_color
        ).next_to(section_1, UP, MED_LARGE_BUFF)
        section_2_label = (
            Tex("2", font_size=DEFAULT_FONT_SIZE * 1.5, color=section_2_color)
            .next_to(section_2, UP)
            .set_y(section_1_label.get_y())
        )
        section_3_label = (
            Tex("3", font_size=DEFAULT_FONT_SIZE * 1.5, color=section_3_color)
            .next_to(section_3, UP)
            .set_y(section_1_label.get_y())
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    section_1.animate.set_color(section_1_color),
                    GrowFromCenter(section_1_label),
                ),
                AnimationGroup(
                    section_2.animate.set_color(section_2_color),
                    GrowFromCenter(section_2_label),
                ),
                AnimationGroup(
                    section_3.animate.set_color(section_3_color),
                    GrowFromCenter(section_3_label),
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            radar_eqn[0][:3].animate.set_opacity(0.2),
            radar_eqn[0][12:].animate.set_opacity(0.2),
            section_2_label.animate.set_opacity(0.2),
            section_3_label.animate.set_opacity(0.2),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(section_1_label, section_2_label, section_3_label),
                radar_eqn.animate.scale(1 / 1.5).to_edge(UP, MED_LARGE_BUFF),
                lag_ratio=0.3,
            )
        )

        self.wait(2)


class Section1(MovingCameraScene):
    def construct(self):
        radar_eqn = (
            MathTex(
                r"P_r = \frac{P_t G_t}{4 \pi R^2} \cdot \frac{\sigma}{4 \pi R^2} \cdot A_e",
                font_size=DEFAULT_FONT_SIZE * 1.5,
            )
            .scale(1 / 1.5)
            .to_edge(UP, MED_LARGE_BUFF)
        )
        section_1_color = GREEN
        section_2_color = BLUE
        section_3_color = YELLOW

        section_1 = radar_eqn[0][3:12].set_color(section_1_color)
        section_2 = radar_eqn[0][13:19].set_color(section_2_color)
        section_3 = radar_eqn[0][20:22].set_color(section_3_color)
        radar_eqn[0][:3].set_opacity(0.2)
        radar_eqn[0][12:].set_opacity(0.2)

        self.add(radar_eqn)

        self.next_section(skip_animations=skip_animations(True))

        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=config.frame_width * 0.6,
            y_length=config.frame_height * 0.4,
        )
        sig = ax.plot(
            lambda t: np.sin(2 * PI * 2 * t), x_range=[0, 1, 1 / 100], color=TX_COLOR
        )
        plot_group = Group(ax, sig).to_edge(DOWN, LARGE_BUFF)

        self.play(LaggedStart(Create(ax), Create(sig), lag_ratio=0.4))

        self.wait(0.5)

        pt_val = (
            Tex(r"$P_t = 1$ kW").to_edge(LEFT, LARGE_BUFF).set_y(plot_group.get_y())
        )
        pt_val[0][:2].set_color(section_1_color)

        self.play(
            LaggedStart(
                plot_group.animate.to_edge(RIGHT, MED_LARGE_BUFF),
                TransformFromCopy(radar_eqn[0][3:5], pt_val[0][:2], path_arc=-PI / 3),
                GrowFromCenter(pt_val[0][2]),
                GrowFromCenter(pt_val[0][3]),
                GrowFromCenter(pt_val[0][4:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            plot_group.animate.shift(DOWN * 8), pt_val.animate.to_edge(DOWN).set_x(0)
        )
        self.remove(plot_group)

        self.wait(0.5)

        point_source = Dot().set_z_index(-1)
        reflector = (
            ArcBetweenPoints(
                point_source.get_center() + UP,
                point_source.get_center() + DOWN,
                stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
            )
            .move_to(point_source, LEFT)
            .set_z_index(-1)
        )
        iso_pattern = Circle(radius=config.frame_width, color=TX_COLOR)

        self.play(Create(point_source))

        self.wait(0.5)

        self.play(
            Broadcast(
                iso_pattern,
                focal_point=point_source.get_center(),
                final_opacity=1,
                n_mobs=3,
                lag_ratio=0.4,
                run_time=5,
            )
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        target_shift = VT(0)

        target_sq = Square(side_length=1.5).to_edge(RIGHT).set_y(point_source.get_y())
        target_label = Tex("Target").move_to(target_sq)
        target = Group(target_sq, target_label).shift(LEFT * 3)

        # target.add_updater(lambda m: m.shift(RIGHT * ~target_shift))

        self.play(target.shift(RIGHT * 7).animate.shift(LEFT * 7))

        self.wait(0.5)

        iso_pattern_target = always_redraw(
            lambda: Circle(
                radius=(target.get_left() - point_source.get_center())[0],
                color=TX_COLOR,
            ).move_to(point_source)
        )

        self.play(GrowFromCenter(iso_pattern_target))

        self.wait(0.5)

        section_at_target = always_redraw(
            lambda: SurroundingRectangle(
                Line(target.get_corner(DL), target.get_corner(UL))
            )
        )

        self.play(Create(section_at_target))

        self.wait(0.5)

        range_arrow = always_redraw(
            lambda: Arrow(point_source.get_center(), iso_pattern_target.get_right())
        )
        range_label = always_redraw(
            lambda: MathTex("R", color=section_1_color).next_to(
                range_arrow, UP, MED_LARGE_BUFF
            )
        )

        self.play(
            TransformFromCopy(radar_eqn[0][10], range_label), GrowArrow(range_arrow)
        )

        self.wait(0.5)

        self.play(target.animate.shift(RIGHT * 3), run_time=2)

        self.wait(0.5)

        self.play(Uncreate(section_at_target))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        sphere_dotted_line = DashedLine(
            iso_pattern_target.get_right(),
            iso_pattern_target.get_left(),
            path_arc=PI / 6,
            dash_length=DEFAULT_DASH_LENGTH * 4,
        )
        sphere_dotted_line_back = Line(
            iso_pattern_target.get_left(),
            iso_pattern_target.get_right(),
            path_arc=PI / 6,
        )

        self.play(Create(sphere_dotted_line), rate_func=rate_functions.ease_in_sine)
        self.play(
            Create(sphere_dotted_line_back), rate_func=rate_functions.ease_out_sine
        )

        self.wait(0.5)

        p_at_target = MathTex(r"P_{\text{at target}} = \frac{P_t}{4 \pi R^2}").next_to(
            pt_val, UP
        )
        p_at_target[0][10:].set_color(section_1_color)
        p_label = p_at_target[0][:9]
        p_label.save_state()

        self.play(GrowFromCenter(p_label.set_x(pt_val.get_x())))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(p_label.animate.restore(), GrowFromCenter(p_at_target[0][9]))

        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(pt_val[0][:2], p_at_target[0][10:12]),
                GrowFromCenter(p_at_target[0][12]),
                GrowFromCenter(p_at_target[0][13:15]),
                ReplacementTransform(
                    range_label[0], p_at_target[0][15], path_arc=PI / 6
                ),
                GrowFromCenter(p_at_target[0][16]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(range_arrow, sphere_dotted_line, sphere_dotted_line_back))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        n_elem = 17  # Must be odd
        weight_trackers = [VT(0) for _ in range(n_elem)]
        weight_trackers[n_elem // 2] @= 1

        f_0 = 10e9
        wavelength_0 = c / f_0
        k_0 = 2 * PI / wavelength_0
        d_x = wavelength_0 / 2

        steering_angle = VT(0)
        theta = np.linspace(-PI, PI, 1000)
        u = np.sin(theta)

        r_min = -30
        x_len = (target.get_left() - point_source.get_center())[0] * 2
        ax = Axes(
            x_range=[r_min, -r_min, r_min / 8],
            y_range=[r_min, -r_min, r_min / 8],
            tips=False,
            axis_config={
                "include_numbers": False,
            },
            x_length=x_len,
            y_length=x_len,
        )

        AF_scale = VT(1)

        def get_af():
            u_0 = np.sin(~steering_angle * PI / 180)
            weights = np.array([~w for w in weight_trackers])
            AF = compute_af_1d(weights, d_x, k_0, u, u_0)
            AF_log = np.clip(20 * np.log10(np.abs(AF)) - r_min, 0, None) * ~AF_scale
            f_AF = interp1d(u * PI, AF_log, fill_value="extrapolate")
            plot = ax.plot_polar_graph(
                r_func=f_AF, theta_range=[-PI, PI, 2 * PI / 200], color=TX_COLOR
            )
            return plot

        AF_plot = always_redraw(get_af)

        self.add(AF_plot)
        self.remove(iso_pattern_target)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        weight_trackers[: n_elem // 2][::-1][n] @ 1,
                        weight_trackers[n_elem // 2 + 1 :][n] @ 1,
                    )
                    for n in range(n_elem // 2)
                ],
                lag_ratio=0.3,
            ),
            ReplacementTransform(point_source, reflector),
            run_time=4,
        )

        self.wait(0.5)

        eqns = Group(pt_val, p_at_target)

        self.play(
            eqns.animate.arrange(UP, aligned_edge=LEFT).to_corner(DL, MED_LARGE_BUFF)
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        gt_val = MathTex(
            r"G_t &= 1,000 \text{ or } 10,000\\ &= 30 \text{ dBi or } 40 \text{ dBi}"
        ).next_to(eqns, UP, aligned_edge=LEFT)
        gt_val_chosen = MathTex(r"G_t = 30 \text{ dBi}").move_to(
            gt_val, aligned_edge=LEFT
        )
        gt_val[0][:2].set_color(section_1_color)
        gt_val_chosen[0][:2].set_color(section_1_color)
        # eqns.add(gt_val)

        self.play(TransformFromCopy(radar_eqn[0][5:7], gt_val[0][:2], path_arc=-PI / 6))

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(gt_val[0][2]),
                GrowFromCenter(gt_val[0][3:8]),
                GrowFromCenter(gt_val[0][8:10]),
                GrowFromCenter(gt_val[0][10:16]),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(gt_val[0][16]),
                GrowFromCenter(gt_val[0][17:19]),
                GrowFromCenter(gt_val[0][19:22]),
                GrowFromCenter(gt_val[0][22:24]),
                GrowFromCenter(gt_val[0][24:26]),
                GrowFromCenter(gt_val[0][26:]),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        # fmt:off
        self.play(
            TransformByGlyphMap(
                gt_val,
                gt_val_chosen,
                ([0, 1, 2], [0, 1, 2]),
                ([3,4,5,6,7,8,9,10,11,12,13,14,15,16,22,23,24,25,26,27,28], FadeOut),
                ([17,18,19,20,21], [3,4,5,6,7],{"delay": 0.3})
            )
        )
        # fmt:on

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        p_at_target_full = MathTex(
            r"P_{\text{at target}} = \frac{P_t G_t}{4 \pi R^2}"
        ).move_to(p_at_target, aligned_edge=LEFT)
        p_at_target_full[0][10:].set_color(section_1_color)

        def get_transform_func(from_var, func=TransformFromCopy, path_arc=PI):
            def transform_func(m, **kwargs):
                return func(from_var, m, path_arc=path_arc, **kwargs)

            return transform_func

        self.play(
            TransformByGlyphMap(
                p_at_target,
                p_at_target_full,
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                ([10, 11], [10, 11]),
                ([12], [14]),
                ([13, 14], [15, 16]),
                ([15, 16], [17, 18]),
                (get_transform_func(gt_val_chosen[0][:2]), [12, 13]),
            ),
            run_time=2,
        )

        self.wait(0.5)

        eqns.remove(p_at_target)
        eqns.add(gt_val_chosen)

        self.play(
            eqns.animate.arrange(UP, aligned_edge=LEFT).to_corner(DL, MED_LARGE_BUFF),
            p_at_target_full.animate.set_x(0).to_edge(DOWN, MED_LARGE_BUFF),
        )

        self.wait(0.5)

        box = SurroundingRectangle(pt_val)
        gain_box = SurroundingRectangle(gt_val_chosen)

        self.play(Create(box))

        self.wait(0.5)

        self.play(Transform(box, gain_box))

        self.wait(0.5)

        range_val = MathTex(r"R = 1 \text{ km}").next_to(
            gt_val_chosen, UP, aligned_edge=LEFT
        )
        range_val[0][0].set_color(section_1_color)

        range_box = SurroundingRectangle(range_val)

        self.play(range_val.shift(LEFT * 5).animate.shift(RIGHT * 5))

        self.wait(0.5)

        self.play(Transform(box, range_box))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        p_at_target_full_val = MathTex(
            r"P_{\text{at target}} = \frac{P_t G_t}{4 \pi R^2} \approx 0.08 \ \frac{\text{W}}{\text{m}^2}"
        ).move_to(p_at_target_full, aligned_edge=LEFT)
        p_at_target_full_val[0][10:-9].set_color(section_1_color)

        self.play(GrowFromCenter(p_at_target_full_val[0][-9:]))

        self.wait(0.5)

        self.play(Uncreate(box))

        self.wait(0.5)

        p_at_target_group = Group(
            p_at_target_full[0][10:], p_at_target_full_val[0][-9:]
        )

        self.play(
            section_1.animate.set_opacity(0.2),
            FadeOut(p_at_target_full[0][:10]),
            p_at_target_group.animate.scale(0.8).to_corner(UL),
            AF_scale @ 0,
        )
        # self.remove(AF_plot)

        self.wait(0.5)

        self.play(section_2.animate.set_opacity(1))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        box = SurroundingRectangle(radar_eqn[0][15:19])

        self.play(Create(box))

        self.wait(0.5)

        range_arrow = Arrow(target.get_left(), point_source.get_center())
        range_label = MathTex("R", color=section_2_color).next_to(range_arrow, UP)

        self.play(
            GrowArrow(range_arrow), TransformFromCopy(radar_eqn[0][17], range_label[0])
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rcs_box = SurroundingRectangle(radar_eqn[0][13])

        self.play(Transform(box, rcs_box))

        self.wait(0.5)

        rcs_label = Tex(r"$\sigma$ | radar cross section [m$^2$]").next_to(
            point_source, DOWN, LARGE_BUFF
        )
        rcs_label[0][0].set_color(section_2_color)

        self.play(
            LaggedStart(
                TransformFromCopy(radar_eqn[0][13], rcs_label[0][0], path_arc=PI / 4),
                GrowFromCenter(rcs_label[0][1]),
                GrowFromCenter(rcs_label[0][2:7]),
                GrowFromCenter(rcs_label[0][7:12]),
                GrowFromCenter(rcs_label[0][12:19]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in rcs_label[0][-4:]], lag_ratio=0.2)
        )

        self.wait(0.5)

        self.play(Uncreate(box))

        self.wait(0.5)

        self.camera.frame.save_state()

        self.play(self.camera.frame.animate.shift(DOWN * (config.frame_height * 1.2)))

        self.wait(0.5)

        self.play(self.camera.frame.animate.restore())

        self.wait(0.5)

        self.play(target_sq.animate.stretch(2, 1))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        rcs_label_val = MathTex(r"\sigma = 1 \text{ m}^2").move_to(rcs_label)
        rcs_label_val[0][0].set_color(section_2_color)

        # TODO: Fix the m/s -> m^2 issue
        self.play(
            TransformByGlyphMap(
                rcs_label,
                rcs_label_val,
                ([2, 3, 4, 5, 6], ShrinkToCenter, {"delay": 0.2}),
                ([7, 8, 9, 10, 11], ShrinkToCenter, {"delay": 0.4}),
                ([12, 13, 14, 15, 16, 17, 18], ShrinkToCenter, {"delay": 0.6}),
                ([19], ShrinkToCenter, {"delay": 0.6}),
                ([22], ShrinkToCenter, {"delay": 0.6}),
                ([0], [0], {"delay": 1}),
                ([1], [1], {"delay": 1}),
                ([20, 21], [3, 4], {"delay": 0.8}),
                (GrowFromCenter, [2], {"delay": 1}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        eqns.add(range_val, rcs_label_val)

        section_2_copy = section_2.copy().move_to(rcs_label_val)

        self.play(
            eqns.animate.arrange(UP, aligned_edge=LEFT).to_corner(DL),
            TransformFromCopy(rcs_label_val[0][0], section_2_copy[0][0]),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            LaggedStart(
                TransformFromCopy(section_2[2:4], section_2_copy[2:4], path_arc=PI / 6),
                TransformFromCopy(range_label[0], section_2_copy[4]),
                GrowFromCenter(section_2_copy[1]),
                GrowFromCenter(section_2_copy[-1]),
                lag_ratio=0.4,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        section_2_units = MathTex(r"\left[ \frac{m^2}{m^2} \right]").next_to(
            section_2_copy
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in section_2_units[0]]))

        self.wait(0.5)

        self.play(ShrinkToCenter(section_2_units))

        self.wait(0.5)

        section_2_val = MathTex(
            r"\frac{\sigma}{4 \pi R^2} = -71 \text{ dB} = 8 \cdot 10^{-8}"
        ).move_to(section_2_copy)
        section_2_val[0][:6].set_color(section_2_color)

        self.play(
            LaggedStart(
                ReplacementTransform(section_2_copy[:6], section_2_val[0][:6]),
                GrowFromCenter(section_2_val[0][6]),
                GrowFromCenter(section_2_val[0][7:10]),
                GrowFromCenter(section_2_val[0][10:12]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                GrowFromCenter(section_2_val[0][12]),
                GrowFromCenter(section_2_val[0][13]),
                GrowFromCenter(section_2_val[0][14:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            section_2_val[0][:12]
            .animate.scale(0.8)
            .next_to(p_at_target_group, DOWN, aligned_edge=LEFT),
            ShrinkToCenter(section_2_val[0][12:]),
        )

        self.wait(0.5)

        self.play(FadeOut(range_arrow, range_label))

        self.wait(0.5)

        self.play(section_2.animate.set_opacity(0.2))

        self.wait(0.5)

        self.play(section_3.animate.set_opacity(1))

        self.wait(0.5)

        how_effective = Tex("how effective?").next_to(point_source, DOWN, LARGE_BUFF)
        how_effective_curve = CubicBezier(
            how_effective.get_right() + [0.1, 0, 0],
            how_effective.get_right() + [1, 0, 0],
            point_source.get_center() + [1, 0, 0],
            point_source.get_center() + [0.3, 0, 0],
        )

        self.play(
            LaggedStart(
                Write(how_effective), Create(how_effective_curve), lag_ratio=0.4
            )
        )

        self.wait(0.5)

        self.play(ShrinkToCenter(how_effective), Uncreate(how_effective_curve))

        self.wait(0.5)

        gain_ae_eqn = MathTex(r"G = \frac{4 \pi A_e}{\lambda^2}").to_edge(
            DOWN, LARGE_BUFF
        )
        gain_ae_eqn[0][4:6].set_color(section_3_color)
        ae_eqn = MathTex(r"A_e = \frac{G \lambda^2}{4 \pi}").move_to(gain_ae_eqn)
        ae_eqn[0][:2].set_color(section_3_color)

        self.wait(0.5)

        self.play(gain_ae_eqn.shift(DOWN * 6).animate.shift(UP * 6))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(
            TransformByGlyphMap(
                gain_ae_eqn,
                ae_eqn,
                ([0], [3], {"delay": 0.4, "path_arc": -PI / 3}),
                ([1], [2]),
                ([2, 3], [7, 8], {"path_arc": PI / 2}),
                ([4, 5], [0, 1], {"delay": 0.4, "path_arc": PI / 2}),
                ([7, 8], [4, 5], {"delay": 0.4, "path_arc": PI / 2}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ae_eqn_gt = MathTex(r"A_e = \frac{G_t \lambda^2}{4 \pi}").move_to(gain_ae_eqn)
        ae_eqn_gt[0][3:5].set_color(section_1_color)
        ae_eqn_gt[0][:2].set_color(section_3_color)

        self.play(
            TransformByGlyphMap(
                ae_eqn,
                ae_eqn_gt,
                ([0, 1, 2], [0, 1, 2]),
                ([3], ShrinkToCenter),
                (get_transform_func(gt_val_chosen[0][:2], path_arc=PI / 6), [3, 4]),
                ([4, 5, 6, 7, 8], [5, 6, 7, 8, 9]),
            )
        )

        self.wait(0.5)

        f_val = MathTex(r"f = 3 \text{ GHz}").next_to(
            rcs_label_val, UP, aligned_edge=LEFT
        )
        f_box = SurroundingRectangle(f_val)

        self.play(f_val.shift(LEFT * 6).animate.shift(RIGHT * 6))
        self.play(Create(f_box))

        self.wait(0.5)

        self.play(Uncreate(f_box))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ae_eqn_gt_val = MathTex(
            r"A_e = \frac{G_t \lambda^2}{4 \pi} = 0.79 \text{ m}^2"
        ).move_to(ae_eqn_gt, LEFT)
        ae_eqn_gt_val[0][:2].set_color(section_3_color)
        ae_eqn_gt_val[0][3:5].set_color(section_1_color)

        self.play(
            LaggedStart(
                GrowFromCenter(ae_eqn_gt_val[0][-7]),
                GrowFromCenter(ae_eqn_gt_val[0][-6:-2]),
                GrowFromCenter(ae_eqn_gt_val[0][-2:]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        m2_box = SurroundingRectangle(ae_eqn_gt_val[0][-2:])

        self.play(Create(m2_box))

        self.wait(0.5)

        self.play(Uncreate(m2_box))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        ae_eqn_gt_val_corner = (
            MathTex(r"A_e = 0.79 \text{ m}^2")
            .scale(0.8)
            .next_to(section_2_val[0][:12], DOWN, aligned_edge=LEFT)
        )
        ae_eqn_gt_val_corner[0][:2].set_color(section_3_color)

        self.add(ae_eqn_gt_val[0][:10])
        self.remove(*ae_eqn_gt[0])

        self.play(
            TransformByGlyphMap(
                ae_eqn_gt_val,
                ae_eqn_gt_val_corner,
                ([0, 1, 2], [0, 1, 2], {"delay": 0.4}),
                ([3, 4, 5, 6, 7, 8, 9, 10], ShrinkToCenter),
                ([11, 12, 13, 14, 15, 16], [3, 4, 5, 6, 7, 8], {"delay": 0.4}),
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(radar_eqn[0].animate.set_opacity(1))

        self.wait(0.5)

        section_2_val_lin = (
            MathTex(r"\frac{\sigma}{4 \pi R^2} = 8 \cdot 10^{-8}")
            .scale(0.8)
            .move_to(section_2_val[0][1], LEFT)
        )
        # section_2_val_lin[0][1].move_to(section_2_val[0][7], LEFT)
        section_2_val_lin[0][:6].set_color(section_2_color)

        self.play(
            # ReplacementTransform(section_2_val[:7], section_2_val_lin[:7]),
            ReplacementTransform(section_2_val[0][7:12], section_2_val_lin[0][7:]),
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        self.play(ShrinkToCenter(reflector), ShrinkToCenter(target))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        radar_eqn_val = (
            MathTex(
                r"P_r &= 0.08 \cdot 8 \cdot 10^{-8} \cdot 0.79\\ &= -83 \text{ dBW} = 5 \cdot 10^{-9} \text{ W}",
            )
            .move_to(radar_eqn, LEFT)
            .shift(DOWN * 2)
        )
        radar_eqn_val[0][3:7].set_color(section_1_color)
        radar_eqn_val[0][8:14].set_color(section_2_color)
        radar_eqn_val[0][15:19].set_color(section_3_color)

        self.play(
            LaggedStart(
                TransformFromCopy(radar_eqn[0][:2], radar_eqn_val[0][:2]),
                TransformFromCopy(radar_eqn[0][2], radar_eqn_val[0][2]),
                TransformFromCopy(
                    p_at_target_full_val[0][-8:-4],
                    radar_eqn_val[0][3:7],
                    path_arc=PI / 6,
                ),
                TransformFromCopy(radar_eqn[0][12], radar_eqn_val[0][7]),
                TransformFromCopy(
                    section_2_val_lin[0][-6:],
                    radar_eqn_val[0][8:14],
                    path_arc=PI / 6,
                ),
                TransformFromCopy(radar_eqn[0][19], radar_eqn_val[0][14]),
                TransformFromCopy(
                    ae_eqn_gt_val_corner[0][-6:-2],
                    radar_eqn_val[0][15:19],
                    path_arc=-PI / 6,
                ),
                lag_ratio=0.5,
            )
        )

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        box = SurroundingRectangle(pt_val)
        f_box = SurroundingRectangle(f_val)
        gt_box = SurroundingRectangle(gt_val_chosen)
        rcs_box = SurroundingRectangle(rcs_label_val)
        r_box = SurroundingRectangle(range_val)

        self.play(Create(box))

        self.wait(0.5)

        self.play(Transform(box, f_box))

        self.wait(0.5)

        self.play(Transform(box, gt_box))

        self.wait(0.5)

        self.play(Transform(box, rcs_box))

        self.wait(0.5)

        self.play(Transform(box, r_box))

        self.wait(0.5)

        self.play(
            Uncreate(box),
            LaggedStart(*[GrowFromCenter(m) for m in radar_eqn_val[0][-15:-8]]),
        )

        self.wait(0.5)

        self.play(LaggedStart(*[GrowFromCenter(m) for m in radar_eqn_val[0][-8:]]))

        self.next_section(skip_animations=skip_animations(True))
        self.wait(0.5)

        radar_eqn_solo = radar_eqn.copy().shift(UP * 10)

        self.play(
            self.camera.frame.animate.move_to(radar_eqn_solo),
            radar_eqn.animate.move_to(radar_eqn_solo),
        )
        self.play(self.camera.frame.animate.scale(0.6))

        self.wait(0.5)

        objs = [*radar_eqn[0]]
        shuffle(objs)

        self.play(LaggedStart(*[ShrinkToCenter(m) for m in objs], lag_ratio=0.05))

        self.wait(2)


class Outro(Scene):
    def construct(self):
        title = Tex(
            "The ",
            "Animated ",
            "Radar ",
            "Cheatsheet",
            font_size=DEFAULT_FONT_SIZE * 1.5,
        )

        self.play(LaggedStart(*[GrowFromCenter(m) for m in title], lag_ratio=0.15))

        self.wait(0.5)

        thumbnail_1_img = ImageMobject(
            "./media/images/radar_equation/thumbnails/Thumbnail.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        thumbnail_1_box = SurroundingRectangle(thumbnail_1_img, buff=0)
        thumbnail_1 = Group(thumbnail_1_img, thumbnail_1_box)
        title_1 = Tex(r"1: The Radar\\Range Equation")
        thumbnail_1_group = (
            Group(thumbnail_1, title_1)
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to([0, -config.frame_height / 2, 0], DOWN)
        )

        thumbnail_2_img = ImageMobject(
            "./media/images/radar_equation/thumbnails/Thumbnail.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        thumbnail_2_box = SurroundingRectangle(thumbnail_2_img, buff=0)
        thumbnail_2 = Group(thumbnail_2_img, thumbnail_2_box)
        title_2 = Tex("2: TBA")
        thumbnail_2_group = (
            Group(thumbnail_2, title_2)
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to([0, -config.frame_height / 2, 0], DOWN)
        )

        thumbnail_3_img = ImageMobject(
            "./media/images/radar_equation/thumbnails/Thumbnail.png"
        ).scale_to_fit_width(config.frame_width * 0.3)
        thumbnail_3_box = SurroundingRectangle(thumbnail_3_img, buff=0)
        thumbnail_3 = Group(thumbnail_3_img, thumbnail_3_box)
        title_3 = Tex("3: TBA")
        thumbnail_3_group = (
            Group(thumbnail_3, title_3)
            .arrange(DOWN, MED_SMALL_BUFF)
            .next_to([0, -config.frame_height / 2, 0], DOWN)
        )

        self.play(Group(title, thumbnail_1_group).animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        thumbnail_y = thumbnail_1_group.get_y()

        thumbnail_2_group.next_to([config.frame_width / 2, thumbnail_y, 0], RIGHT)
        thumbnail_3_group.next_to([config.frame_width / 2, thumbnail_y, 0], RIGHT)

        self.play(
            Group(thumbnail_1_group, thumbnail_2_group)
            .animate.arrange(RIGHT)
            .set_y(thumbnail_y)
        )

        self.wait(0.5)

        self.play(
            Group(thumbnail_1_group, thumbnail_2_group, thumbnail_3_group)
            .animate.arrange(RIGHT, aligned_edge=UP)
            .set_y(thumbnail_y)
        )

        self.wait(2)


class RCS(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI / 2, theta=-PI / 2)
        v = VT(TAU)
        sphere1 = Sphere(
            center=(0, 0, 0),
            radius=1,
            resolution=(20, 20),
            u_range=[0.001, 2 * PI - 0.001],
            v_range=[0, PI],
            color=BLUE,
            fill_opacity=1,
        )
        sphere1.set_fill(color=BLUE, opacity=1)
        sphere1.set_opacity(1)

        circle = Circle(radius=1, color=YELLOW)
        # .rotate(
        #     PI / 2, about_point=[0, 0, 0], axis=[1, 0, 0]
        # )

        # self.add(sphere1)
        self.begin_ambient_camera_rotation(rate=0.2, about="theta")

        self.wait(0.5, frozen_frame=False)

        self.play(sphere1.shift(IN * 6).animate.shift(OUT * 6))

        # self.play(v @ PI)

        self.wait(0.5, frozen_frame=False)

        self.add_fixed_in_frame_mobjects(circle)
        self.remove(circle)

        self.play(Create(circle))

        self.wait(0.5)

        area = MathTex(r"A = \pi r^2")
        self.add_fixed_in_frame_mobjects(area)
        area.next_to(circle, DOWN)
        self.remove(area)

        self.play(
            circle.animate.set_fill(color=GREEN, opacity=0.5),
            LaggedStart(*[GrowFromCenter(m) for m in area[0]], lag_ratio=0.07),
        )

        self.wait(0.5)

        person = (
            ImageMobject("../props/static/person.png")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale_to_fit_height(config.frame_height * 0.4)
        )
        self.add_fixed_in_frame_mobjects(person)
        person.next_to(person, LEFT * 0.5)
        self.remove(person)

        person_area = MathTex(r"A = \text{?}")
        self.add_fixed_in_frame_mobjects(person_area)
        person_area.next_to(person, DOWN)
        self.remove(person_area)

        self.play(person.shift(LEFT * 8).animate.shift(RIGHT * 8))

        self.wait(0.5)

        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in person_area[0]], lag_ratio=0.07),
        )

        self.wait(0.5)

        self.play(
            person.animate.shift(LEFT * 8),
            circle.animate.shift(DOWN * 8),
            sphere1.animate.shift(IN * 6),
            LaggedStart(*[ShrinkToCenter(m) for m in person_area[0]], lag_ratio=0.07),
            LaggedStart(*[ShrinkToCenter(m) for m in area[0]], lag_ratio=0.07),
        )

        self.wait(2)


class AntennaTransform(Scene):
    def construct(self):
        point_source = Dot()
        reflector = ArcBetweenPoints(
            point_source.get_center() + UP,
            point_source.get_center() + DOWN,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        ).move_to(point_source, LEFT)
        reflector_bot = ArcBetweenPoints(
            point_source.get_center(),
            reflector.get_end(),
            angle=PI / 4,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        reflector_top = ArcBetweenPoints(
            point_source.get_center(),
            reflector.get_start(),
            angle=-PI / 4,
            stroke_width=DEFAULT_STROKE_WIDTH * 1.5,
        )
        self.add(
            point_source,
            reflector,
            reflector_bot.shift(RIGHT),
            reflector_top.shift(RIGHT),
        )

        self.wait(0.5)

        # self.play(
        #     MoveAlongPath(
        #         point_source,
        #     )
        #     # Transform(
        #     #     point_source,
        #     #     reflector,
        #     # )
        # )

        self.wait(2)


""" Thumbnail """


class Thumbnail(Scene):
    def construct(self):
        self.add(Tex("placeholder"))
