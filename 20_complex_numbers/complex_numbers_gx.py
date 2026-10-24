# complex_numbers_gx.py
import json
import pathlib
import pickle
import textwrap
from collections import deque
from copy import deepcopy
from functools import partial
from random import shuffle
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


class CommentIntro(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        comments = [
            (
                "@MindlessUser",
                "PLEASE do the video about complex numbers! Your videos are amazing!!!",
                (-0.32, 0.05),
            ),
            (
                "@victorscarpes",
                "Please make a video on complex numbers.",
                (0.32, 0.21),
            ),
            (
                "@odhinnsrunes",
                "...Yes, I'd be interested in a video on complex numbers...",
                (0.00, 0.04),
            ),
            (
                "@jagdish_atm_science",
                "...Please upload a detailed video on the complex [number] application...",
                (-0.32, -0.26),
            ),
            (
                "@ellisbender5839",
                "I would appreciate a video on imaginary numbers and their "
                "application in this video. Pt2??",
                (0.32, -0.15),
            ),
            (
                "@PremKumar-u2c5x",
                "Video on Complex Numbers pls",
                (-0.32, 0.30),
            ),
            (
                "@wangbing-f5s",
                "...I look forward to your future work on the application "
                "of complex numbers in RF engineering...",
                (0.00, -0.31),
            ),
            (
                "@-ion",
                "...Another video covering it with complex numbers would be "
                "a good addition.",
                (0.00, 0.32),
            ),
        ]
        bubbles = VGroup()
        for index, (author, comment, (x, y)) in enumerate(comments):
            handle = Text(author, font=FONT, font_size=20, color=BLUE)
            quote = Text(
                textwrap.fill(comment, width=26),
                font=FONT,
                font_size=24,
                line_spacing=0.2,
                color="#D6EEF7",
                t2c={
                    "complex": GREEN,
                    "Complex": GREEN,
                    "number": GREEN,
                    "Numbers": GREEN,
                    "numbers": GREEN,
                    "imaginary": GREEN,
                },
            )
            content = VGroup(handle, quote).arrange(
                DOWN, aligned_edge=LEFT, buff=fh(self, 0.018)
            )
            if content.width > fw(self, 0.26):
                content.scale_to_fit_width(fw(self, 0.26))

            body = RoundedRectangle(
                width=content.width + fw(self, 0.033),
                height=content.height + fh(self, 0.06),
                corner_radius=fh(self, 0.035),
                fill_color="#244957",
                fill_opacity=1,
                stroke_width=0,
            )
            tail_base = body.get_bottom() + LEFT * body.width * 0.25
            tail = Polygon(
                tail_base + UP * fh(self, 0.015),
                tail_base + DOWN * fh(self, 0.035) + LEFT * fw(self, 0.008),
                tail_base + RIGHT * fw(self, 0.026) + UP * fh(self, 0.015),
                fill_color="#244957",
                fill_opacity=1,
                stroke_width=0,
            ).round_corners(fh(self, 0.008))
            content.move_to(body)
            bubble = VGroup(body, tail, content).move_to(
                RIGHT * fw(self, x) + UP * fh(self, y)
            )
            direction = UP + RIGHT * (0.07 if index % 2 else -0.07)
            always_shift(bubble, direction, rate=fh(self, 0.05 + 0.003 * (index % 3)))
            bubbles.add(bubble)

        bubbles.shift(DOWN)

        self.next_section("Comment bubbles", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                *[
                    PopIn(bubble, rate_func=ease_out_back, run_time=0.5)
                    for bubble in bubbles
                ],
                lag_ratio=0.3,
            )
        )
        self.next_section("Floating comments", skip_animations=skip_animations(False))
        self.wait(2)

        self.play(
            LaggedStart(
                *[
                    ShrinkToCenter(bubble, run_time=0.6, rate_func=ease_in_back)
                    for bubble in bubbles
                ],
                lag_ratio=0.3,
            )
        )

        thumbnail_dir = pathlib.Path(__file__).with_name("static") / "other_videos"
        videos = json.loads((thumbnail_dir / "metadata.json").read_text())
        positions = [
            (-0.32, 0.31),
            (0, 0.31),
            (0.32, 0.31),
            (-0.16, -0.18),
            (0.16, -0.18),
        ]
        video_groups = Group()
        for video, (x, y) in zip(videos, positions, strict=True):
            thumbnail = ImageMobject(thumbnail_dir / video["thumbnail_file"])
            if video["width"] / video["height"] < 16 / 9:
                # YouTube's SD thumbnail includes black bars above and below.
                crop = (1 - video["width"] / video["height"] * 9 / 16) / 2
                thumbnail.uvs = np.array(
                    [[0, crop], [1, crop], [0, 1 - crop], [1, 1 - crop]]
                )
            thumbnail.scale_to_fit_width(fw(self, 0.274))
            thumbnail.stretch_to_fit_height(thumbnail.width * 9 / 16)
            thumbnail.move_to(RIGHT * fw(self, x) + UP * fh(self, y))
            outline = SurroundingRectangle(
                thumbnail, buff=0, color=GREEN, stroke_width=3
            )
            title = Paragraph(
                *textwrap.wrap(video["title"], width=28),
                alignment="center",
                font=FONT,
                font_size=18,
                line_spacing=0.12,
                color=WHITE,
            )
            if title.width > thumbnail.width:
                title.scale_to_fit_width(thumbnail.width)
            title.next_to(thumbnail, DOWN, buff=fh(self, 0.022))
            author = Paragraph(
                *textwrap.wrap(video["author"], width=32),
                alignment="center",
                font=FONT,
                font_size=16,
                line_spacing=0.15,
                color=BLUE,
            )
            author.next_to(title, DOWN, buff=fh(self, 0.016))
            video_groups.add(Group(thumbnail, outline, title, author))

        self.next_section(
            "Other complex number videos", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                *[FadeIn(video_group, run_time=0.6) for video_group in video_groups],
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        video_groups_list = list(video_groups)
        shuffle(video_groups_list)

        self.play(
            LaggedStart(
                *[
                    ShrinkToCenter(video_group, run_time=0.6, rate_func=ease_in_back)
                    for video_group in video_groups_list
                ],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.35),
            y_length=fh(self, 0.3),
        ).next_to(ORIGIN, LEFT, LARGE_BUFF)
        sine = ax.plot(
            lambda x: np.sin(TAU * 3 * x),
            color=BLUE,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        )

        self.play(Create(sine))

        self.wait(0.5)

        amp = (
            Polygon(
                [-1, 1, 0],
                [-1, -1, 0],
                [1, 0, 0],
                color=GREEN,
                stroke_width=6,
            )
            .round_corners(0.13)
            .scale_to_fit_height(fh(self, 0.18))
            .move_to(RIGHT * fw(self, 0.10) + UP * fh(self, 0.32))
        )
        filter_box = RoundedRectangle(
            width=amp.height,
            height=amp.height,
            corner_radius=fh(self, 0.02),
            stroke_width=6,
        )
        filter_waves = VGroup()
        for offset, color in ((0.24, RED), (0, BLUE), (-0.24, RED)):
            filter_waves.add(
                ParametricFunction(
                    lambda t, offset=offset: np.array(
                        [
                            (t - 0.5) * filter_box.width * 0.70,
                            (offset + 0.10 * np.sin(TAU * t)) * filter_box.height,
                            0,
                        ]
                    ),
                    t_range=[0, 1, 0.01],
                    color=color,
                    stroke_width=4,
                )
            )
        filt = (
            VGroup(filter_box, filter_waves)
            .move_to(RIGHT * fw(self, 0.30) + UP * fh(self, 0.32))
            .set_z_index(1)
        )
        amp_to_filter = Line(
            amp.get_right(), filt.get_left(), stroke_width=4
        ).set_z_index(0)

        dish_arc = (
            Arc(
                radius=fh(self, 0.14),
                start_angle=3 * PI / 4,
                angle=PI / 2,
                color=WHITE,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
            .rotate(PI / 12)
            .move_to(RIGHT * fw(self, 0.22) + UP * fh(self, 0.065))
        )
        mount = dish_arc.point_from_proportion(0.5)
        pedestal_top = mount + DOWN * fh(self, 0.075)
        pedestal = Polygon(
            pedestal_top + LEFT * fw(self, 0.012),
            pedestal_top + RIGHT * fw(self, 0.012),
            pedestal_top + DOWN * fh(self, 0.055) + RIGHT * fw(self, 0.04),
            pedestal_top + DOWN * fh(self, 0.055) + LEFT * fw(self, 0.04),
            color=WHITE,
            stroke_width=DEFAULT_STROKE_WIDTH * 2,
        ).round_corners(fh(self, 0.007))
        dish = VGroup(
            dish_arc,
            Line(mount, pedestal_top, stroke_width=DEFAULT_STROKE_WIDTH * 2),
            pedestal,
        )
        wave_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=fw(self, 0.20),
            y_length=fh(self, 0.13),
        ).next_to(dish_arc, RIGHT, SMALL_BUFF)
        wave_ax.rotate(PI / 12, about_point=wave_ax.c2p(0, 0))
        wave_phase = VT(0)
        transmitted_wave = always_redraw(
            lambda: wave_ax.plot(
                lambda x: np.sin(TAU * (2.5 * x - ~wave_phase)),
                x_range=[0, max(0.0001, min(~wave_phase / 2.5, 1)), 0.005],
                color=BLUE,
                stroke_width=DEFAULT_STROKE_WIDTH * 2,
            )
        )

        iq_clip = VideoMobject(pathlib.Path(__file__).with_name("iq_video_clip.mp4"))
        iq_clip.scale_to_fit_width(fw(self, 0.29)).move_to(
            RIGHT * fw(self, 0.14) + DOWN * fh(self, 0.28)
        )
        iq_outline = SurroundingRectangle(iq_clip, buff=0, color=GREEN, stroke_width=3)

        branch_start = sine.get_end() + RIGHT * MED_SMALL_BUFF
        branches = VGroup()
        for end in (
            amp.get_left() + LEFT * MED_SMALL_BUFF,
            dish_arc.get_left() + LEFT * MED_SMALL_BUFF,
            iq_clip.get_left() + LEFT * MED_SMALL_BUFF,
        ):
            branch = TipableVMobject(
                color=GREEN, stroke_width=3, tip_corner_radius=0.02
            )
            branch.start_new_path(branch_start)
            branch.add_cubic_bezier_curve_to(
                branch_start + RIGHT * fw(self, 0.07),
                end + LEFT * fw(self, 0.07),
                end,
            )
            branch.add_tip(tip_length=0.2, tip_width=0.2)
            branches.add(branch)

        self.next_section(
            "Amplifier and filter", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                GrowArrow(branches[0]),
                GrowFromCenter(amp),
                Create(amp_to_filter),
                GrowFromCenter(filt),
                lag_ratio=0.3,
                run_time=1.8,
            )
        )
        self.wait(0.5)

        self.next_section("Transmitting dish", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(GrowArrow(branches[1]), Create(dish), lag_ratio=0.3),
            run_time=1.2,
        )
        wave_phase.add_updater(lambda mob, dt: mob.increment_value(dt))
        self.add(wave_phase, transmitted_wave)
        self.wait(1.4)

        self.next_section("IQ video", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                GrowArrow(branches[2]),
                Group(iq_clip, iq_outline)
                .shift(RIGHT * fw(self))
                .animate.shift(LEFT * fw(self)),
                lag_ratio=0.3,
            ),
            run_time=1.2,
        )

        self.wait(iq_clip.duration - 3)

        self.wait(0.5)

        complex_numbers = (
            Text("Complex Numbers", font=FONT)
            .move_to(ORIGIN)
            .shift(DOWN * fh(self) + UP * fh(self, 0.25))
        )
        z_eqn = (
            MathTex(r"z = x + j y")
            .move_to(self.camera.frame)
            .shift(DOWN * fh(self))
            .scale(1.5)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=2).shift(DOWN * fh(self)),
                LaggedStart(
                    *[PopIn(m) for m in [*complex_numbers, *z_eqn[0]]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.6,
            ),
        )

        self.remove(iq_clip)

        self.wait(0.5)

        xy_pair = MathTex(r"(x,y)").scale(1.5)
        z_xy = DoubleArrow(LEFT, RIGHT)
        z_like_group = (
            Group(z_eqn.copy(), z_xy, xy_pair)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .move_to(z_eqn)
        )

        self.play(
            LaggedStart(
                Transform(z_eqn[0], z_like_group[0][0]),
                GrowArrow(z_xy),
                LaggedStart(*[PopIn(m) for m in xy_pair], lag_ratio=0.1),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                z_eqn[0][2].animate.set_color(YELLOW),
                xy_pair[0][1].animate.set_color(YELLOW),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                z_eqn[0][5].animate.set_color(BLUE),
                xy_pair[0][3].animate.set_color(BLUE),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        sqrtn1 = (
            MathTex(r"\sqrt{-1}")
            .next_to(z_eqn[0][4], DOWN, LARGE_BUFF)
            .shift(LEFT * 1)
            .set_color(ORANGE)
        )
        j_arrow = Arrow(
            z_eqn[0][4].get_bottom(),
            sqrtn1.get_corner(UR),
            buff=SMALL_BUFF,
            path_arc=-PI / 3,
            color=ORANGE,
        )

        self.play(
            LaggedStart(
                z_eqn[0][4].animate.set_color(ORANGE),
                GrowArrow(j_arrow),
                PopIn(sqrtn1),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        q1 = (
            Text("?", font=FONT, weight=BOLD)
            .scale(1.2)
            .rotate(20 * DEGREES)
            .next_to(z_eqn, UP, MED_SMALL_BUFF)
            .shift(LEFT)
        )

        q2 = (
            Text("?", font=FONT, weight=BOLD)
            .scale(1.2)
            .rotate(-20 * DEGREES)
            .next_to(sqrtn1, RIGHT, MED_SMALL_BUFF)
            .shift(DOWN * 0.2)
        )

        self.play(PopIn(q1))

        self.wait(0.5)

        self.play(PopIn(q2))

        self.wait(0.5)

        circuit = Group(amp, amp_to_filter, filt).next_to(self.camera.frame, DOWN)

        arrow_start = xy_pair.get_right() + RIGHT * 0.3
        arrow_end = amp.get_left() + LEFT * 0.25 + UP * 0.1
        arrow_mid = (arrow_start + arrow_end) / 2 + UP * 0.1
        amp_arrow = TipableVMobject(color=GREEN, stroke_width=3, tip_corner_radius=0.02)
        amp_arrow.start_new_path(arrow_start)
        amp_arrow.add_cubic_bezier_curve_to(
            arrow_start + RIGHT * 3.1 + DOWN * 0.8,
            arrow_mid + RIGHT * 3.5 + UP * 0.15,
            arrow_mid,
        )
        amp_arrow.add_cubic_bezier_curve_to(
            arrow_mid + LEFT * 3.5 + DOWN * 0.15,
            arrow_end + LEFT * 3.5,
            arrow_end,
        )
        amp_arrow.add_tip(tip_length=0.2, tip_width=0.2)

        q3 = (
            Text("?", font=FONT, weight=BOLD)
            .scale(1.2)
            .rotate(10 * DEGREES)
            .next_to(circuit, UP, MED_SMALL_BUFF)
            .shift(LEFT * 1.5)
        )

        self.play(
            self.camera.frame.animate.move_to(Group(z_eqn, xy_pair, circuit)),
            GrowArrow(amp_arrow),
        )

        self.play(PopIn(q3))

        self.wait(0.5)

        self.play(
            LaggedStart(
                q1.animate(rate_func=there_and_back).scale(2).set_color(YELLOW),
                q2.animate(rate_func=there_and_back).scale(2).set_color(YELLOW),
                q3.animate(rate_func=there_and_back).scale(2).set_color(YELLOW),
                lag_ratio=0.2,
            )
        )

        self.next_section(
            "Resources in the description", skip_animations=skip_animations(False)
        )
        questions = Group(
            complex_numbers,
            z_eqn,
            z_xy,
            xy_pair,
            sqrtn1,
            j_arrow,
            q1,
            q2,
            circuit,
            amp_arrow,
            q3,
        )
        aspect = self.camera.frame.width / self.camera.frame.height
        frame_height = max(questions.height / 0.82, questions.width / (aspect * 0.51))
        frame_center = (
            questions.get_center()
            + RIGHT * frame_height * aspect * 0.23
            + DOWN * frame_height * 0.025
        )
        self.play(
            self.camera.frame.animate.scale_to_fit_height(frame_height).move_to(
                frame_center
            ),
            run_time=1.2,
        )

        resource_center = frame_center + RIGHT * fw(self, 0.285)
        notebook_panel = RoundedRectangle(
            width=fw(self, 0.35),
            height=fh(self, 0.245),
            corner_radius=fh(self, 0.015),
            fill_color="#244957",
            fill_opacity=1,
            color=BLUE,
            stroke_width=3,
        ).move_to(resource_center + UP * fh(self, 0.24))
        notebook_title = Text("complex_numbers.py", font=FONT, color=WHITE)
        notebook_title.scale_to_fit_height(fh(self, 0.032)).move_to(
            notebook_panel.get_top() + DOWN * fh(self, 0.035)
        )
        notebook_axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1, 1, 1],
            x_length=fw(self, 0.3),
            y_length=fh(self, 0.13),
            tips=False,
            axis_config={"include_ticks": False, "stroke_opacity": 0.4},
        ).move_to(notebook_panel.get_center() + DOWN * fh(self, 0.025))
        notebook_plot = VGroup(
            notebook_axes,
            notebook_axes.plot(
                lambda t: np.cos(TAU * 3 * t), color=ORANGE, stroke_width=4
            ),
            notebook_axes.plot(
                lambda t: np.sin(TAU * 3 * t), color=BLUE, stroke_width=4
            ),
        )
        notebook_label = Text("Python notebook", font=FONT, color=BLUE)
        notebook_label.scale_to_fit_height(fh(self, 0.032)).next_to(
            notebook_panel, DOWN, buff=fh(self, 0.018)
        )
        notebook_card = VGroup(
            notebook_panel, notebook_title, notebook_plot, notebook_label
        )

        book_cover = RoundedRectangle(
            width=fw(self, 0.075),
            height=fh(self, 0.15),
            corner_radius=fh(self, 0.008),
            fill_color="#244957",
            fill_opacity=1,
            color=ORANGE,
            stroke_width=3,
        )
        book_back = book_cover.copy().shift(
            LEFT * fw(self, 0.009) + UP * fh(self, 0.012)
        )
        book_spine = Line(
            book_cover.get_corner(UL) + RIGHT * fw(self, 0.01),
            book_cover.get_corner(DL) + RIGHT * fw(self, 0.01),
            color=ORANGE,
            stroke_width=3,
        )
        book_symbol = MathTex("j", color=ORANGE).scale_to_fit_height(fh(self, 0.065))
        book_symbol.move_to(book_cover).shift(RIGHT * fw(self, 0.004))
        books_icon = VGroup(book_back, book_cover, book_spine, book_symbol)

        article_page = RoundedRectangle(
            width=fw(self, 0.075),
            height=fh(self, 0.15),
            corner_radius=fh(self, 0.006),
            fill_color="#D6EEF7",
            fill_opacity=1,
            stroke_width=0,
        )
        article_back = (
            article_page.copy()
            .set_fill("#8FA3AB")
            .shift(RIGHT * fw(self, 0.008) + UP * fh(self, 0.012))
        )
        article_lines = VGroup(
            *[
                Line(
                    article_page.get_corner(UL) + [fw(self, 0.012), -fh(self, y), 0],
                    article_page.get_corner(UL) + [fw(self, width), -fh(self, y), 0],
                    color=BACKGROUND_COLOR,
                    stroke_width=4 if index == 0 else 2,
                )
                for index, (y, width) in enumerate(
                    [
                        (0.025, 0.06),
                        (0.05, 0.062),
                        (0.067, 0.056),
                        (0.084, 0.062),
                        (0.101, 0.06),
                        (0.118, 0.044),
                    ]
                )
            ]
        )
        articles_icon = VGroup(article_back, article_page, article_lines)

        video_front = ImageMobject(thumbnail_dir / "-j8PzkZ70Lg.jpg")
        video_front.scale_to_fit_width(fw(self, 0.115))
        video_back = ImageMobject(thumbnail_dir / "TLWE388JWGs.jpg")
        video_back.scale_to_fit_width(video_front.width).shift(
            RIGHT * fw(self, 0.008) + UP * fh(self, 0.02)
        )
        videos_icon = Group(
            video_back,
            SurroundingRectangle(video_back, buff=0, color=GREEN, stroke_width=2),
            video_front,
            SurroundingRectangle(video_front, buff=0, color=GREEN, stroke_width=2),
        )

        reading_cards = Group()
        for icon, title, color, x in [
            (books_icon, "Books", ORANGE, -0.12),
            (articles_icon, "Articles", WHITE, 0),
            (videos_icon, "Videos", GREEN, 0.12),
        ]:
            icon.move_to(resource_center + RIGHT * fw(self, x) + DOWN * fh(self, 0.1))
            label = Text(title, font=FONT, color=color).scale_to_fit_height(
                fh(self, 0.03)
            )
            label.move_to(resource_center + RIGHT * fw(self, x) + DOWN * fh(self, 0.22))
            reading_cards.add(Group(icon, label))

        description = Text("In the description", font=FONT, color=WHITE)
        description.scale_to_fit_height(fh(self, 0.033)).move_to(
            resource_center + DOWN * fh(self, 0.325)
        )
        description_arrow = SquigglyLine(
            resource_center + DOWN * fh(self, 0.365),
            resource_center + DOWN * fh(self, 0.46),
            amplitude=fw(self, 0.006),
            color=ORANGE,
            stroke_width=5,
            tip_corner_radius=0.02,
        ).add_tip(tip_length=fh(self, 0.025), tip_width=fw(self, 0.018))

        self.play(FadeIn(notebook_card, shift=UP * fh(self, 0.035)), run_time=0.8)
        self.wait(0.4)
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=UP * fh(self, 0.035)) for card in reading_cards],
                lag_ratio=0.3,
            ),
            run_time=1.5,
        )
        self.play(
            LaggedStart(
                PopIn(description), GrowArrow(description_arrow), lag_ratio=0.35
            )
        )
        self.wait(2)

        self.next_section("ECE103", skip_animations=skip_animations(False))
        self.play(self.camera.frame.animate.shift(DOWN * fh(self, 1.1)), run_time=1.2)
        self.wait(0.25)

        course = Text("ECE103", font=FONT).scale_to_fit_height(fh(self, 0.1))
        course.move_to(self.camera.frame)
        self.play(
            LaggedStart(
                *[PopIn(character, run_time=0.25) for character in course],
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(course))

        self.wait(0.5)


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
    voltage_label.next_to(axes, LEFT, MED_SMALL_BUFF)
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
        self.wait(6)

        self.play(
            FadeOut(
                negcharge_line,
                negcharge_label,
                poscharge_line,
                poscharge_label,
            ),
            run_time=1,
        )

        self.wait(8.5)

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
        self.wait(plot_window + 2.5)

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

        self.wait(8.25)

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
        ).set_z_index(1)
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

        self.wait(4.5)

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
            run_time=0.85,
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
        self.wait(1.2)

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
        frame_shift = DOWN * fh(self)

        self.play(
            xsin_dt[0][7:]
            .animate.scale_to_fit_width(fw(self, 0.2))
            .move_to(self.camera.frame)
            .shift(frame_shift),
            self.camera.frame.animate.shift(frame_shift),
        )

        self.wait(2)


class ShiftProblem(MovingCameraScene):
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

        self.wait(0.5)

        self.play(self.camera.frame.animate.shift(UP * fh(self)))


class TwoPoints(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        max_time = 1
        capacitance = 1
        omega = TAU
        current_peak = capacitance * omega
        x = VT(0)
        follow_time = VT(1)
        fixed_window_start = VT(0)
        wrap_time = False

        def window_start():
            return (
                ~follow_time * max(0, ~x - max_time / 2)
                + (1 - ~follow_time) * ~fixed_window_start
            )

        def plot_point(plot_axes, t, value):
            plot_time = t - window_start()
            if wrap_time:
                plot_time %= TAU / omega
            return plot_axes.c2p(plot_time, value)

        def voltage(t, phi0=0):
            return np.sin(omega * t + phi0)

        def current(t, phi0=0):
            return current_peak * np.cos(omega * t + phi0)

        wire_color = "#D6EEF7"
        axis_config = {
            "color": wire_color,
            "stroke_width": 2,
            "include_numbers": False,
        }
        axes = Axes(
            x_range=[0, max_time, 0.5],
            y_range=[-1, 1, 0.5],
            x_length=fw(self, 0.7),
            y_length=fh(self, 0.35),
            axis_config=axis_config,
            x_axis_config={"include_ticks": False},
            tips=False,
        ).shift(UP * fh(self, 0.19))
        current_axes = Axes(
            x_range=[0, max_time, 0.5],
            y_range=[-current_peak, current_peak, current_peak / 2],
            x_length=fw(self, 0.7),
            y_length=fh(self, 0.35),
            axis_config=axis_config,
            x_axis_config={"include_ticks": False},
            tips=False,
        ).shift(DOWN * fh(self, 0.23))
        voltage_label = (
            MathTex(r"V_C(t)", color=YELLOW)
            .scale(0.7)
            .rotate(PI / 2)
            .next_to(axes.y_axis, LEFT, fw(self, 0.055))
        )
        voltage_spelled = (
            Text("Voltage", font=FONT)
            .rotate(PI / 2)
            .scale(0.5)
            .set_color(YELLOW)
            .next_to(voltage_label, RIGHT, MED_SMALL_BUFF)
        )
        current_label = (
            MathTex(r"I_C(t)", color=BLUE)
            .scale(0.7)
            .rotate(PI / 2)
            .next_to(current_axes.y_axis, LEFT, fw(self, 0.055))
        )
        current_spelled = (
            Text("Current", font=FONT)
            .scale(0.6)
            .rotate(PI / 2)
            .set_color(BLUE)
            .next_to(current_label, RIGHT, MED_SMALL_BUFF)
        )
        self.add(
            axes,
            current_axes,
            voltage_label,
            current_label,
            current_spelled,
            voltage_spelled,
        )

        def time_ticks():
            start = window_start()
            times = np.arange(np.ceil(start / 0.5) * 0.5, start + max_time + 1e-8, 0.5)
            return VGroup(
                *[
                    Line(
                        plot_point(plot_axes, t, 0) + DOWN * 0.1,
                        plot_point(plot_axes, t, 0) + UP * 0.1,
                        color=wire_color,
                        stroke_width=2,
                    )
                    for plot_axes in (axes, current_axes)
                    for t in times
                    if t > start + 1e-8
                ]
            )

        time_tick_marks = always_redraw(time_ticks)
        self.add(time_tick_marks)

        def visible_curve(plot_axes, function, color, opacity=1):
            start = window_start()
            return plot_axes.plot(
                lambda t: function(t + start),
                x_range=[0, max_time, 1 / 200],
                color=color,
                stroke_opacity=opacity,
            )

        v_min = VT(0)
        v_max = VT(max_time * 2)
        voltage_opacity = VT(0.2)
        voltage_background = always_redraw(
            lambda: visible_curve(axes, voltage, YELLOW, opacity=~voltage_opacity)
        )

        def selected_voltage():
            offset = window_start()
            start = np.clip((~v_min - offset) / max_time, 0, 1)
            stop = np.clip((~v_max - offset) / max_time, start, 1)
            curve = visible_curve(axes, voltage, YELLOW)
            return curve.pointwise_become_partial(curve, start, stop).set_z_index(1)

        v_plot = always_redraw(selected_voltage)
        self.add(voltage_background, v_plot)

        self.play(self.camera.frame.shift(DOWN * fh(self)).animate.shift(UP * fh(self)))

        voltage_point_rad_scale = VT(1)
        current_point_rad_scale = VT(1)
        current_point_180deg_rad_scale = VT(1)

        current_point = always_redraw(
            lambda: Dot(
                plot_point(current_axes, ~x, current(~x)),
                radius=fh(self, 0.012) * ~current_point_rad_scale,
                color=BLUE,
            ).set_z_index(3)
        )

        def drawn_current():
            curve = visible_curve(current_axes, current, BLUE)
            stop = np.clip((~x - window_start()) / max_time, 0, 1)
            return curve.pointwise_become_partial(curve, 0, stop).set_z_index(2)

        current_trace = always_redraw(drawn_current)
        point_link = always_redraw(
            lambda: DashedLine(
                plot_point(axes, ~x, voltage(~x)),
                plot_point(current_axes, ~x, current(~x)),
                color=BLUE,
                stroke_width=2,
                stroke_opacity=0.6,
                dash_length=fh(self, 0.012),
            ).set_z_index(1)
        )

        slope_line_opacity = VT(1)

        def slope_line():
            slope = omega * np.cos(omega * ~x)
            point = plot_point(axes, ~x, voltage(~x))
            return (
                Line(
                    point,
                    plot_point(axes, ~x + 1, voltage(~x) + slope),
                    color=BLUE,
                    stroke_width=5,
                    stroke_opacity=~slope_line_opacity,
                )
                .set_length(fw(self, 0.13))
                .move_to(point)
                .set_z_index(2)
            )

        current_slope = always_redraw(slope_line)
        self.next_section("Voltage slope", skip_animations=skip_animations(False))
        self.play(
            FadeIn(
                current_slope,
                # voltage_point,
                current_point,
                point_link,
            ),
            run_time=0.5,
        )
        self.add(current_trace)

        self.wait(0.5)

        self.play(x @ (max_time), run_time=6, rate_func=linear)

        self.wait(0.5)

        angle = current_slope.get_angle()
        midpoint = current_slope.get_midpoint()

        slope_direction = Arrow(
            midpoint,
            midpoint + 2 * (RIGHT * np.cos(angle) + UP * np.sin(angle)),
            buff=0,
        ).set_z_index(10)

        self.play(GrowArrow(slope_direction))

        self.wait(0.5)

        self.play(FadeOut(slope_direction))

        self.wait(0.5)

        voltage_point = always_redraw(
            lambda: Dot(
                plot_point(axes, ~x, voltage(~x)),
                radius=fh(self, 0.012) * ~voltage_point_rad_scale,
                color=YELLOW,
            ).set_z_index(3)
        )

        self.play(
            LaggedStart(
                slope_line_opacity.animate(run_time=1.5).set_value(0),
                LaggedStart(
                    AnimationGroup(
                        v_min @ (~x - 0.005),
                        v_max @ (~x + 0.005),
                        run_time=2,
                    ),
                    PopIn(voltage_point),
                    lag_ratio=0.9,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        option1_label = (
            Text("Option 1", font=FONT)
            .scale(0.6)
            .next_to(current_point, RIGHT, LARGE_BUFF * 1.5)
        )
        option1_line = SquigglyLine(
            current_point.get_right() + [0.1, 0, 0],
            option1_label.get_left() + [-0.1, 0, 0],
            seed=1,
            amplitude=0.1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )

        self.play(
            LaggedStart(
                GrowArrow(option1_line),
                FadeIn(option1_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        current_180deg_opacity = VT(1)

        def drawn_current_180deg():
            curve = visible_curve(
                current_axes,
                partial(current, phi0=PI),
                BLUE,
                opacity=~current_180deg_opacity,
            )
            stop = np.clip((~x - window_start()) / max_time, 0, 1)
            return curve.pointwise_become_partial(curve, 0, stop).set_z_index(2)

        current_180deg = always_redraw(drawn_current_180deg)

        current_point_180deg = always_redraw(
            lambda: Dot(
                plot_point(current_axes, ~x, current(~x, phi0=PI)),
                radius=fh(self, 0.012) * ~current_point_180deg_rad_scale,
                color=BLUE,
                fill_opacity=~current_180deg_opacity,
                stroke_opacity=~current_180deg_opacity,
            ).set_z_index(3)
        )

        option2_label = (
            Text("Option 2", font=FONT)
            .scale(0.6)
            .next_to(current_point_180deg, RIGHT, LARGE_BUFF * 1.5)
        )
        option2_line = SquigglyLine(
            current_point_180deg.get_right() + [0.1, 0, 0],
            option2_label.get_left() + [-0.1, 0.08, 0],
            seed=2,
            amplitude=0.1,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )

        self.play(
            LaggedStart(
                Create(current_180deg),
                PopIn(current_point_180deg),
                LaggedStart(
                    GrowArrow(option2_line),
                    FadeIn(option2_label),
                    lag_ratio=0.3,
                ),
                lag_ratio=0.6,
            )
        )

        self.wait(0.5)

        v_min @= ~x - 0.6
        v_max @= ~x - 0.5

        self.play(v_min @ (~x + 0.8), v_max @ (~x + 0.8), run_time=2)

        self.wait(0.5)

        voltage_background_180deg = always_redraw(
            lambda: visible_curve(axes, partial(voltage, phi0=PI), YELLOW).set_stroke(
                opacity=0.2
            )
        )

        self.play(Create(voltage_background_180deg))

        self.wait(0.5)

        self.play(
            LaggedStart(
                Uncreate(voltage_background_180deg),
                FadeOut(option1_line),
                FadeOut(option1_label),
                FadeOut(option2_line),
                FadeOut(option2_label),
                lag_ratio=0.2,
            ),
            run_time=1.3,
        )

        self.wait(0.5)

        self.play(x + 0.125, run_time=2)

        self.wait(0.5)

        plus = Text("+", font=FONT).next_to(current_point, RIGHT, MED_SMALL_BUFF)
        minus = Text("-", font=FONT).next_to(
            current_point_180deg, RIGHT, MED_SMALL_BUFF
        )

        self.play(
            LaggedStart(
                PopIn(plus),
                PopIn(minus),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(plus),
                FadeOut(minus),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            voltage_point_rad_scale.animate(
                rate_func=there_and_back_with_pause, run_time=1.5
            ).set_value(2)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                current_point_rad_scale.animate(rate_func=there_and_back).set_value(2),
                current_point_180deg_rad_scale.animate(
                    rate_func=there_and_back
                ).set_value(2),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)
        self.next_section(
            "Quarter-cycle lookback", skip_animations=skip_animations(False)
        )

        lookback_angle = VT(0)
        lagged_shift = VT(0)
        lagged_axes = axes

        def lookback_time():
            return ~x - ~lookback_angle * DEGREES / omega

        lookback_point = always_redraw(
            lambda: Dot(
                plot_point(
                    lagged_axes,
                    lookback_time() + ~lagged_shift,
                    voltage(lookback_time()),
                ),
                radius=fh(self, 0.012),
                color=GREEN,
            ).set_z_index(4)
        )
        lookback_arrow = always_redraw(
            lambda: Arrow(
                plot_point(axes, ~x, voltage(~x)),
                plot_point(axes, lookback_time() + ~lagged_shift, voltage(~x)),
                buff=0,
                thickness=2,
                tip_corner_radius=0.02,
                color=GREEN,
            ).set_z_index(2)
        )
        lookback_guide = always_redraw(
            lambda: DashedLine(
                plot_point(axes, lookback_time() + ~lagged_shift, voltage(~x)),
                plot_point(
                    axes, lookback_time() + ~lagged_shift, voltage(lookback_time())
                ),
                color=GREEN,
                stroke_width=2,
                dash_length=fh(self, 0.012),
            ).set_z_index(2)
        )
        lookback_label = always_redraw(
            lambda: (
                DecimalNumber(
                    ~lookback_angle,
                    num_decimal_places=0,
                    unit=r"^\circ",
                    color=GREEN,
                )
                .scale(0.6)
                .next_to(lookback_arrow, UP, MED_SMALL_BUFF)
                .set_z_index(4)
            )
        )

        self.add(lookback_arrow, lookback_guide)

        self.play(
            FadeIn(lookback_point, lookback_label),
            current_180deg_opacity @ 0.2,
            run_time=0.5,
        )

        self.wait(0.5)

        self.play(lookback_angle @ 90, run_time=3)

        self.wait(0.5)

        self.play(x @ 0.5, run_time=3)

        self.wait(0.5)

        readout_y = axes.get_top()[1] + fh(self, 0.095)
        vt0_anchor = [axes.c2p(max_time * 0.96, 0)[0], readout_y, 0]
        vtback_anchor = [axes.c2p(max_time * 0.46, 0)[0], readout_y, 0]

        def voltage_value(t):
            value = f"{voltage(t):z.2f}"
            return value if value.startswith("-") else r"\phantom{-}" + value

        vt0 = always_redraw(
            lambda: (
                MathTex(rf"V_C(t_0) = {voltage_value(~x)} \text{{ V}}")
                .scale(0.7)
                .set_color(YELLOW)
                .move_to(vt0_anchor, aligned_edge=RIGHT)
            )
        )
        vtback = always_redraw(
            lambda: (
                MathTex(
                    rf"V_C(t_0 - {~lookback_angle:.0f}^\circ) = "
                    rf"{voltage_value(lookback_time())} \text{{ V}}"
                )
                .scale(0.7)
                .set_color(GREEN)
                .move_to(vtback_anchor, aligned_edge=RIGHT)
            )
        )
        vt0_key = Dot(radius=fh(self, 0.009), color=YELLOW).next_to(
            vt0, LEFT, MED_SMALL_BUFF
        )
        vtback_key = Dot(radius=fh(self, 0.009), color=GREEN).next_to(
            vtback, LEFT, MED_SMALL_BUFF
        )

        self.play(
            PopIn(vt0_key),
            LaggedStart(*[PopIn(m) for m in vt0[0]], lag_ratio=0.1),
        )
        self.remove(*vt0[0])
        self.add(vt0)

        self.wait(0.5)

        self.play(
            PopIn(vtback_key),
            LaggedStart(*[PopIn(m) for m in vtback[0]], lag_ratio=0.1),
        )
        self.remove(*vtback[0])
        self.add(vtback)

        self.wait(0.5)

        self.play(x + 0.5, run_time=4)

        self.wait(0.5)

        self.remove(v_plot)

        self.play(x + 1, run_time=10)

        self.wait(0.5)
        self.next_section("Lagged voltage", skip_animations=skip_animations(False))

        lagged_voltage = always_redraw(
            lambda: visible_curve(
                lagged_axes, lambda t: voltage(t - ~lagged_shift), GREEN
            ).set_z_index(1)
        )

        self.play(voltage_opacity @ 1, run_time=0.8)
        self.wait(0.3)
        self.play(
            FadeIn(lagged_voltage),
            FadeOut(lookback_arrow, lookback_label),
            run_time=0.6,
        )
        self.play(lagged_shift @ (~lookback_angle * DEGREES / omega), run_time=3)
        self.wait(0.5)

        self.next_section(
            "Voltage coordinate pair", skip_animations=skip_animations(False)
        )

        vt0.clear_updaters()
        vtback.clear_updaters()
        self.remove(vt0, vtback)
        # Sign changes can leave duplicate padding glyphs in always_redraw.
        vt0 = (
            MathTex(rf"V_C(t_0) = {voltage_value(~x)} \text{{ V}}")
            .scale(0.7)
            .set_color(YELLOW)
            .move_to(vt0_anchor, aligned_edge=RIGHT)
        )
        vtback = (
            MathTex(
                rf"V_C(t_0 - {~lookback_angle:.0f}^\circ) = "
                rf"{voltage_value(lookback_time())} \text{{ V}}"
            )
            .scale(0.7)
            .set_color(GREEN)
            .move_to(vtback_anchor, aligned_edge=RIGHT)
        )
        now_readout = vt0[0][:6]
        lagged_readout = vtback[0][:10]
        readout_values = VGroup(vt0[0][6:], vtback[0][10:])
        self.add(now_readout, lagged_readout, readout_values)

        self.play(
            FadeOut(
                current_axes,
                current_label,
                current_trace,
                current_180deg,
                current_point,
                current_point_180deg,
                point_link,
                time_tick_marks,
                lookback_guide,
                voltage_label,
                readout_values,
                vt0_key,
                vtback_key,
            ),
            run_time=0.5,
        )
        self.remove(current_slope)

        source_axes = axes.copy()
        source_origin = source_axes.c2p(0, 0)
        source_center = source_axes.c2p(max_time / 2, 0)
        source_time_length = np.linalg.norm(
            source_axes.c2p(max_time, 0) - source_origin
        )
        source_voltage_length = np.linalg.norm(
            source_axes.c2p(0, 1) - source_axes.c2p(0, -1)
        )
        radius = fh(self, 0.2)
        time_axis_length = fh(self, 0.7)
        circle_center = self.camera.frame.get_center() + UP * fh(self, 2)

        circle_axes = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            x_length=2 * radius,
            y_length=2 * radius,
            axis_config={**axis_config, "stroke_opacity": 0.4},
            tips=False,
        )
        circle_axes.shift(circle_center - circle_axes.c2p(0, 0))
        lagged_axes = Axes(
            x_range=[0, max_time, 0.5],
            y_range=[-1, 1, 0.5],
            x_length=time_axis_length,
            y_length=2 * radius,
            axis_config=axis_config,
            x_axis_config={"include_ticks": False},
            tips=False,
        )
        lagged_axes.shift(circle_axes.c2p(1, 0) - lagged_axes.c2p(0, 0))
        now_axes = Axes(
            x_range=[0, max_time, 0.5],
            y_range=[-1, 1, 0.5],
            x_length=time_axis_length,
            y_length=2 * radius,
            axis_config=axis_config,
            x_axis_config={"include_ticks": False},
            tips=False,
        ).rotate(-PI / 2)
        now_axes.shift(circle_axes.c2p(0, -1) - now_axes.c2p(0, 0))
        now_center = now_axes.c2p(max_time / 2, 0)

        now_label = (
            MathTex(r"V_C(t_0)", color=YELLOW)
            .scale(0.8)
            .next_to(now_axes, LEFT, MED_LARGE_BUFF)
        )
        lagged_label = (
            MathTex(r"V_C(t_0 - 90^\circ)", color=GREEN)
            .scale(0.8)
            .next_to(lagged_axes, UP, MED_LARGE_BUFF)
        )
        circle = Circle(
            radius=radius, color=WHITE, stroke_width=2, stroke_opacity=0.2
        ).move_to(circle_center)
        layout = VGroup(circle_axes, now_axes, lagged_axes, now_label, lagged_label)
        layout_center = layout.get_center().copy()
        layout_height = layout.height * 1.15

        def separate_voltage_axes(mob, alpha):
            pose = source_axes.copy()
            pose.stretch(
                interpolate(1, time_axis_length / source_time_length, alpha),
                0,
                about_point=source_center,
            )
            pose.stretch(
                interpolate(1, 2 * radius / source_voltage_length, alpha),
                1,
                about_point=source_center,
            )
            pose.rotate(-alpha * PI / 2, about_point=source_center)
            pose.shift(alpha * (now_center - source_center))
            mob.x_axis.become(pose.x_axis)
            mob.y_axis.become(pose.y_axis)

        circle_start = (np.ceil((omega * ~x - PI / 2) / TAU) * TAU + PI / 2) / omega
        fixed_window_start @= circle_start
        label_detour = LEFT * fw(self, 0.32) + UP * fh(self, 0.12)

        def now_label_path(start, end, alpha):
            return interpolate(start, end, alpha) + np.sin(PI * alpha) * label_detour

        self.play(
            LaggedStart(
                AnimationGroup(
                    TransformFromCopy(source_axes, lagged_axes, introducer=True),
                    UpdateFromAlphaFunc(axes, separate_voltage_axes),
                    x @ circle_start,
                    follow_time @ 0,
                    self.camera.frame.animate.scale_to_fit_height(
                        layout_height
                    ).move_to(layout_center),
                    *[
                        ReplacementTransform(source, target)
                        for source, target in zip(lagged_readout, lagged_label[0])
                    ],
                    *[
                        ReplacementTransform(source, target, path_func=now_label_path)
                        for source, target in zip(now_readout, now_label[0])
                    ],
                    run_time=3.5,
                ),
                FadeIn(circle_axes, circle, run_time=1),
                lag_ratio=0.8,
            ),
        )
        self.remove(now_readout, lagged_readout, *now_label[0], *lagged_label[0])
        self.add(now_label, lagged_label)
        wrap_time = True

        def coordinate_pair(t):
            return circle_axes.c2p(voltage(t), voltage(t - ~lagged_shift))

        pair_point = always_redraw(
            lambda: Dot(
                coordinate_pair(~x), radius=fh(self, 0.013), color=ORANGE
            ).set_z_index(5)
        )
        projections = always_redraw(
            lambda: VGroup(
                DashedLine(
                    voltage_point.get_center(),
                    coordinate_pair(~x),
                    color=YELLOW,
                    stroke_width=2,
                    dash_length=fh(self, 0.012),
                ),
                DashedLine(
                    lookback_point.get_center(),
                    coordinate_pair(~x),
                    color=GREEN,
                    stroke_width=2,
                    dash_length=fh(self, 0.012),
                ),
            ).set_z_index(1)
        )
        self.play(
            LaggedStart(
                Create(projections),
                PopIn(pair_point),
                lag_ratio=0.3,
            ),
            run_time=0.8,
        )

        circle_trace = always_redraw(
            lambda: ParametricFunction(
                coordinate_pair,
                t_range=[circle_start, max(circle_start + 1e-6, ~x), 1 / 200],
                color=ORANGE,
                stroke_width=3,
            ).set_z_index(2)
        )
        self.add(circle_trace)

        self.next_section("Trace the circle", skip_animations=skip_animations(False))

        v0_radius_line = Arrow(circle_axes.c2p(0, 0), circle_axes.c2p(0, 1), buff=0)
        v0_radius_label = MathTex("V_0").next_to(v0_radius_line, RIGHT, SMALL_BUFF)

        final_phase = 2 * TAU + PI / 3
        self.play(
            LaggedStart(
                x.animate(
                    run_time=6 * final_phase / PI, rate_func=linear
                ).increment_value(final_phase / omega),
                Succession(
                    AnimationGroup(GrowArrow(v0_radius_line), FadeIn(v0_radius_label)),
                    Wait(2),
                    AnimationGroup(FadeOut(v0_radius_line), FadeOut(v0_radius_label)),
                ),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        coords = MathTex(r"x_0,y_0").next_to(pair_point, UP, MED_SMALL_BUFF)
        coords[0][:2].set_color(YELLOW)
        coords[0][3:].set_color(GREEN)

        self.camera.frame.save_state()
        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(circle_axes, coords).height * 1.2
                ).move_to(Group(circle_axes, coords)),
                LaggedStart(*[PopIn(m) for m in coords[0]], lag_ratio=0.25),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        real_axis_arrow = Arrow(
            circle_axes.c2p(0, 0),
            circle_axes.c2p(1, 0),
            buff=0,
            color=YELLOW,
        )
        real_axis_label = (
            MathTex(r"\Re")
            .set_color(YELLOW)
            .next_to(real_axis_arrow, DOWN, SMALL_BUFF)
            .shift(LEFT * real_axis_arrow.get_width() * 0.3)
        )

        self.play(
            LaggedStart(
                GrowArrow(real_axis_arrow),
                PopIn(real_axis_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        real_axes_box = SurroundingRectangle(
            now_axes, buff=SMALL_BUFF, color=YELLOW
        ).set_z_index(11)

        real_label = (
            MathTex(r"\Re \{ \tilde V_C \}", color=YELLOW)
            .scale(0.8)
            .move_to(now_label, RIGHT)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(circle_axes, coords, now_axes).height * 1.2
                ).move_to(Group(circle_axes, coords, now_axes)),
                Create(real_axes_box),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                ShrinkToCenter(now_label[0][5]),
                ShrinkToCenter(now_label[0][4]),
                ShrinkToCenter(now_label[0][3]),
                ShrinkToCenter(now_label[0][2]),
                AnimationGroup(
                    ReplacementTransform(now_label[0][0], real_label[0][2]),
                    ReplacementTransform(now_label[0][1], real_label[0][4]),
                ),
                PopIn(real_label[0][5]),
                PopIn(real_label[0][3]),
                PopIn(real_label[0][0]),
                PopIn(real_label[0][1]),
                # PopIn(real_label[0][1]),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        imag_axis_arrow = Arrow(
            circle_axes.c2p(0, 0),
            circle_axes.c2p(0, 1),
            buff=0,
            color=GREEN,
        )
        imag_axis_label = (
            MathTex(r"\Im").set_color(GREEN).next_to(imag_axis_arrow, LEFT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                Uncreate(real_axes_box),
                self.camera.frame.animate.scale_to_fit_height(
                    Group(circle_axes, coords).height * 1.2
                ).move_to(Group(circle_axes, coords)),
                GrowArrow(imag_axis_arrow),
                PopIn(imag_axis_label),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        imag_axes_box = SurroundingRectangle(
            lagged_axes, buff=SMALL_BUFF, color=GREEN
        ).set_z_index(11)

        imag_label = (
            MathTex(r"\Im \{ \tilde V_C \}", color=GREEN)
            .scale(0.8)
            .move_to(lagged_label, RIGHT)
        )

        # self.add(imag_label, index_labels(imag_label[0]), index_labels(lagged_label[0]))

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_width(
                    Group(circle_axes, coords, lagged_axes).width * 1.1
                ).move_to(Group(circle_axes, coords, lagged_axes)),
                Create(imag_axes_box),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[ShrinkToCenter(m) for m in lagged_label[0][2:10]],
                AnimationGroup(
                    ReplacementTransform(lagged_label[0][0], imag_label[0][2]),
                    ReplacementTransform(lagged_label[0][1], imag_label[0][4]),
                ),
                PopIn(imag_label[0][5]),
                PopIn(imag_label[0][3]),
                PopIn(imag_label[0][0]),
                PopIn(imag_label[0][1]),
                lag_ratio=0.2,
            ),
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(circle_axes, coords).height * 1.3
                ).move_to(Group(circle_axes, coords)),
                Uncreate(imag_axes_box),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        complex_coords = MathTex(r"x_0 + j y_0").move_to(coords)
        complex_coords[0][:2].set_color(YELLOW)
        complex_coords[0][4:].set_color(GREEN)

        self.play(
            LaggedStart(
                FadeOut(coords[0][2]),
                ReplacementTransform(coords[0][:2], complex_coords[0][:2]),
                ReplacementTransform(coords[0][3:], complex_coords[0][4:]),
                PopIn(complex_coords[0][2]),
                PopIn(complex_coords[0][3]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        angle_line = always_redraw(
            lambda: Line(circle_axes.c2p(0, 0), pair_point.get_center(), color=ORANGE)
        )

        self.play(
            LaggedStart(
                LaggedStart(
                    FadeOut(imag_axis_arrow),
                    FadeOut(real_axis_arrow),
                    FadeOut(imag_axis_label),
                    FadeOut(real_axis_label),
                    lag_ratio=0.07,
                ),
                Create(angle_line),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        phase_arc = always_redraw(
            lambda: Arc(
                radius=angle_line.get_length() * 0.25,
                start_angle=0,
                angle=((~x - circle_start) * omega) % TAU,
                arc_center=circle_axes.c2p(0, 0),
            ).set_z_index(-1)
        )

        def position_phase_phi(mob):
            midpoint = phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            mob.move_to(midpoint + outward * (mob.height / 2 + SMALL_BUFF))

        phase_phi = MathTex(r"\phi").scale(1).set_z_index(3)
        phase_phi.add_updater(position_phase_phi, call_updater=True)

        self.play(Create(phase_arc), PopIn(phase_phi))

        self.wait(0.5)

        phase_real_line = always_redraw(
            lambda: DashedLine(
                plot_point(now_axes, ~x, -1),
                plot_point(now_axes, ~x, 1),
                color=WHITE,
                dash_length=DEFAULT_DASH_LENGTH * 2,
            )
        )
        phase_real_phi = MathTex(r"\phi").add_updater(
            lambda mob: mob.next_to(phase_real_line, RIGHT, MED_SMALL_BUFF),
            call_updater=True,
        )

        phase_phi_arrow = always_redraw(
            lambda: Arrow(
                phase_phi.get_right(),
                phase_real_phi.get_top(),
                path_arc=-PI / 3,
                buff=SMALL_BUFF,
            ).set_z_index(15)
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.scale_to_fit_height(
                    Group(circle_axes, coords, now_axes).height * 1.2
                ).move_to(Group(circle_axes, coords, now_axes)),
                GrowArrow(phase_phi_arrow),
                FadeIn(phase_real_line),
                FadeIn(phase_real_phi),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        phase_real_value = always_redraw(
            lambda: MathTex(
                rf"= {((~x - circle_start) * omega % TAU) / DEGREES:.0f}^\circ"
            ).next_to(phase_real_phi, RIGHT, SMALL_BUFF)
        )

        self.play(
            LaggedStart(*[PopIn(glyph) for glyph in phase_real_value[0]], lag_ratio=0.1)
        )

        self.remove(*phase_real_value[0])

        self.add(phase_real_value)

        self.wait(0.5)

        self.play(FadeOut(phase_phi_arrow))

        self.wait(0.5)

        self.next_section(
            "Current leads voltage", skip_animations=skip_animations(False)
        )

        leading_phase = VT(0)

        def leading_time():
            return ~x + ~leading_phase * DEGREES / omega

        leading_line = always_redraw(
            lambda: Line(
                circle_axes.c2p(0, 0),
                coordinate_pair(leading_time()),
                color=ORANGE,
                stroke_opacity=0.6,
            ).set_z_index(2)
        )
        leading_point = always_redraw(
            lambda: (
                Dot(
                    coordinate_pair(leading_time()),
                    radius=pair_point.width / 2,
                    color=ORANGE,
                )
                .set_opacity(0.6)
                .set_z_index(4)
            )
        )
        leading_real_point = always_redraw(
            lambda: (
                Dot(
                    plot_point(now_axes, leading_time(), voltage(leading_time())),
                    radius=voltage_point.width / 2,
                    color=YELLOW,
                )
                .set_opacity(0.6)
                .set_z_index(4)
            )
        )
        leading_imag_point = always_redraw(
            lambda: (
                Dot(
                    plot_point(
                        lagged_axes,
                        leading_time(),
                        voltage(leading_time() - ~lagged_shift),
                    ),
                    radius=lookback_point.width / 2,
                    color=GREEN,
                )
                .set_opacity(0.6)
                .set_z_index(4)
            )
        )
        leading_projections = always_redraw(
            lambda: (
                VGroup(
                    DashedLine(
                        leading_point.get_center(),
                        leading_real_point.get_center(),
                        color=YELLOW,
                        stroke_width=2,
                        dash_length=fh(self, 0.012),
                    ),
                    DashedLine(
                        leading_point.get_center(),
                        leading_imag_point.get_center(),
                        color=GREEN,
                        stroke_width=2,
                        dash_length=fh(self, 0.012),
                    ),
                )
                .set_stroke(opacity=0.4 * np.clip((~leading_phase - 15) / 15, 0, 1))
                .set_z_index(1)
            )
        )
        leading_phase_arc = always_redraw(
            lambda: Arc(
                radius=angle_line.get_length() * 0.6,
                start_angle=angle_line.get_angle(),
                angle=~leading_phase * DEGREES,
                arc_center=circle_axes.c2p(0, 0),
                stroke_opacity=0.7,
            ).set_z_index(3)
        )

        def leading_phase_label():
            label = MathTex(rf"+{~leading_phase:.0f}^\circ").scale(0.7)
            midpoint = leading_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            return label.move_to(
                midpoint + outward * (label.height / 2 + SMALL_BUFF)
            ).set_z_index(5)

        leading_phase_value = always_redraw(leading_phase_label)

        self.add(
            leading_line,
            leading_point,
            leading_real_point,
            leading_imag_point,
            leading_projections,
            leading_phase_arc,
        )
        self.play(
            LaggedStart(
                PopIn(leading_phase_value),
                leading_phase.animate(run_time=3).set_value(90),
                lag_ratio=0.3,
            )
        )
        self.wait(0.5)

        phase_current_line = always_redraw(
            lambda: DashedLine(
                plot_point(now_axes, leading_time(), -1),
                plot_point(now_axes, leading_time(), 1),
                color=WHITE,
                stroke_opacity=0.6,
                dash_length=DEFAULT_DASH_LENGTH * 2,
            ).set_z_index(1)
        )
        phase_current_phi = always_redraw(
            lambda: MathTex(
                r"\phi_{\text{current}} = "
                rf"{((leading_time() - circle_start) * omega % TAU) / DEGREES:.0f}^\circ"
            ).next_to(phase_current_line, RIGHT, MED_SMALL_BUFF)
        )

        self.play(
            LaggedStart(
                Create(phase_current_line),
                LaggedStart(
                    *[PopIn(glyph) for glyph in phase_current_phi[0]],
                    lag_ratio=0.1,
                ),
                lag_ratio=0.3,
            )
        )
        self.remove(*phase_current_phi[0])
        self.add(phase_current_phi)

        self.wait(0.5)

        self.next_section(
            "Ohm's law needs rotation", skip_animations=skip_animations(False)
        )

        ohms_res = MathTex(r"I_R = \frac{V_R}{R}").scale(1.5)
        ohms_cap = MathTex(r"I_C = \frac{V_C}{Z}").scale(1.5)
        for equation in (ohms_res, ohms_cap):
            equation[0][:2].set_color(BLUE)
            equation[0][3:5].set_color(YELLOW)

        phase_plots = Group(
            circle_axes,
            now_axes,
            lagged_axes,
            complex_coords,
            real_label,
            imag_label,
            phase_real_phi,
            phase_real_value,
            phase_current_phi,
            leading_phase_value,
        )
        ohms_equations = (
            VGroup(ohms_res, ohms_cap)
            .arrange(DOWN, buff=fh(self, 0.11))
            .next_to(phase_plots, LEFT, fw(self, 0.055))
        )
        ohms_arrow = Arrow(
            ohms_res.get_bottom(),
            ohms_cap.get_top(),
            buff=MED_SMALL_BUFF,
            stroke_width=3,
        )

        phase_arrow_start = leading_phase_value.get_left() + LEFT * SMALL_BUFF
        phase_arrow_end = ohms_cap[0][5].get_right() + RIGHT * MED_SMALL_BUFF
        phase_impedance_arrow = TipableVMobject(
            color=ORANGE, stroke_width=3, tip_corner_radius=0.02
        ).set_z_index(6)
        phase_impedance_arrow.start_new_path(phase_arrow_start)
        phase_impedance_arrow.add_cubic_bezier_curve_to(
            phase_arrow_start + LEFT * fw(self, 0.15),
            phase_arrow_end + RIGHT * fw(self, 0.1),
            phase_arrow_end,
        )
        phase_impedance_arrow.add_tip(tip_length=0.2, tip_width=0.2)

        ohms_layout = Group(phase_plots, ohms_equations, phase_impedance_arrow)
        ohms_frame_height = 1.15 * max(
            ohms_layout.height,
            ohms_layout.width * self.camera.frame.height / self.camera.frame.width,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate(run_time=1.5)
                .scale_to_fit_height(ohms_frame_height)
                .move_to(ohms_layout),
                LaggedStart(*[PopIn(glyph) for glyph in ohms_res[0]], lag_ratio=0.1),
                lag_ratio=0.25,
            )
        )
        self.remove(*ohms_res[0])
        self.add(ohms_res)

        self.play(
            LaggedStart(
                GrowArrow(ohms_arrow, run_time=0.7),
                LaggedStart(
                    *[
                        TransformFromCopy(source, target, introducer=True)
                        for source, target in zip(ohms_res[0], ohms_cap[0])
                    ],
                    lag_ratio=0.08,
                ),
                lag_ratio=0.4,
            )
        )
        self.remove(*ohms_cap[0])
        self.add(ohms_cap)

        self.wait(0.3)
        self.play(
            GrowArrow(phase_impedance_arrow),
            ohms_cap[0][5].animate.set_color(ORANGE),
            run_time=1.5,
        )
        self.wait(0.5)

        waveform_opacity = VT(1)
        for mob in (
            axes,
            lagged_axes,
            voltage_background,
            lagged_voltage,
        ):
            mob.add_updater(lambda mob: mob.set_stroke(opacity=~waveform_opacity))
        for mob in (
            voltage_point,
            lookback_point,
            *real_label[0],
            *imag_label[0],
        ):
            mob.add_updater(lambda mob: mob.set_opacity(~waveform_opacity))
        for mob in (leading_real_point, leading_imag_point):
            mob.add_updater(lambda mob: mob.set_opacity(0.6 * ~waveform_opacity))

        circle_view = Group(circle_axes, complex_coords)
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(
                        phase_impedance_arrow,
                        ohms_res,
                        ohms_cap,
                        ohms_arrow,
                        projections,
                        leading_projections,
                        phase_real_line,
                        phase_real_phi,
                        phase_real_value,
                        phase_current_line,
                        phase_current_phi,
                    ),
                    waveform_opacity @ 0.2,
                    run_time=0.8,
                ),
                self.camera.frame.animate(run_time=1.8)
                .scale_to_fit_height(circle_view.height * 1.2)
                .move_to(circle_view),
                lag_ratio=0.15,
            )
        )

        self.wait(0.5)

        cut_state = deepcopy(
            {
                "mobjects": self.mobjects,
                "camera_center": self.camera.frame.get_center(),
                "camera_height": self.camera.frame.height,
                "x": ~x,
                "omega": omega,
                "circle_start": circle_start,
                "lagged_shift": ~lagged_shift,
                "circle_axes": circle_axes,
                "now_axes": now_axes,
                "lagged_axes": lagged_axes,
                "coords": coords,
                "complex_coords": complex_coords,
                "angle_line": angle_line,
                "pair_point": pair_point,
                "leading_line": leading_line,
                "leading_point": leading_point,
                "leading_real_point": leading_real_point,
                "leading_imag_point": leading_imag_point,
                "leading_phase_arc": leading_phase_arc,
                "leading_phase_value": leading_phase_value,
                "leading_phase": ~leading_phase,
            }
        )
        for mob in Group(*cut_state["mobjects"]).get_family():
            mob.clear_updaters(recursive=False)
            if isinstance(mob, ParametricFunction):
                mob.function = None
                mob.underlying_function = None
        cut_path = pathlib.Path(__file__).with_name(".cache") / "two_points_cut.pkl"
        cut_path.parent.mkdir(exist_ok=True)
        cut_path.write_bytes(pickle.dumps(cut_state))


class TwoPointsP2(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        cut_path = pathlib.Path(__file__).with_name(".cache") / "two_points_cut.pkl"
        if not cut_path.exists():
            raise FileNotFoundError("Render TwoPoints once to create its ending state.")
        state = pickle.loads(cut_path.read_bytes())
        self.camera.frame.scale_to_fit_height(state["camera_height"]).move_to(
            state["camera_center"]
        )
        self.add(*state["mobjects"])
        x = VT(state["x"])
        omega = state["omega"]
        circle_start = state["circle_start"]
        lagged_shift = state["lagged_shift"]
        circle_axes = state["circle_axes"]
        now_axes = state["now_axes"]
        lagged_axes = state["lagged_axes"]
        coords = state["coords"]
        complex_coords = state["complex_coords"]
        angle_line = state["angle_line"]
        pair_point = state["pair_point"]
        leading_line = state["leading_line"]
        leading_point = state["leading_point"]
        leading_real_point = state["leading_real_point"]
        leading_imag_point = state["leading_imag_point"]
        leading_phase_arc = state["leading_phase_arc"]
        leading_phase_value = state["leading_phase_value"]
        leading_phase = VT(state["leading_phase"])

        def voltage(t):
            return np.sin(omega * t)

        def coordinate_pair(t):
            return circle_axes.c2p(voltage(t), voltage(t - lagged_shift))

        def plot_point(plot_axes, t, value):
            return plot_axes.c2p((t - circle_start) % (TAU / omega), value)

        def leading_time():
            return ~x + ~leading_phase * DEGREES / omega

        leading_line.add_updater(
            lambda mob: mob.put_start_and_end_on(
                circle_axes.c2p(0, 0), coordinate_pair(leading_time())
            )
        )
        leading_point.add_updater(
            lambda mob: mob.move_to(coordinate_pair(leading_time()))
        )
        leading_real_point.add_updater(
            lambda mob: mob.move_to(
                plot_point(now_axes, leading_time(), voltage(leading_time()))
            )
        )
        leading_imag_point.add_updater(
            lambda mob: mob.move_to(
                plot_point(
                    lagged_axes, leading_time(), voltage(leading_time() - lagged_shift)
                )
            )
        )
        leading_phase_arc.add_updater(
            lambda mob: mob.become(
                Arc(
                    radius=angle_line.get_length() * 0.6,
                    start_angle=angle_line.get_angle(),
                    angle=~leading_phase * DEGREES,
                    arc_center=circle_axes.c2p(0, 0),
                    stroke_opacity=0.7,
                ).set_z_index(3)
            )
        )

        self.next_section("Can multiplication rotate?", skip_animations=False)
        self.wait(0.5)

        def get_phase_shift_times_label():
            label = MathTex(r"\times ?").scale(0.7)
            midpoint = leading_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            return label.move_to(
                midpoint + outward * (label.height / 2 + SMALL_BUFF)
            ).set_z_index(5)

        phase_shift_times_label = always_redraw(get_phase_shift_times_label)

        leading_phase_value.clear_updaters()
        self.play(
            LaggedStart(
                AnimationGroup(
                    ShrinkToCenter(leading_phase_value, remover=True),
                    PopIn(phase_shift_times_label),
                ),
                leading_phase.animate(
                    rate_func=lambda t: np.sin(TAU * ease_in_out_sine(t)),
                    run_time=6,
                ).increment_value(45),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)
        self.next_section(
            "Voltage at 60 degrees", skip_animations=skip_animations(False)
        )

        self.remove(coords, *coords, complex_coords, *complex_coords)
        self.add(complex_coords)
        phase_shift_times_label.clear_updaters()
        self.play(
            leading_phase.animate(rate_func=ease_in_out_sine).set_value(0),
            ShrinkToCenter(phase_shift_times_label, remover=True),
            run_time=1.6,
        )

        trig_coords = (
            MathTex(r"V_0\cos(60^\circ) + j V_0\sin(60^\circ)")
            .scale(0.8)
            .next_to(pair_point, UP, MED_SMALL_BUFF)
        )
        trig_coords[0][:10].set_color(YELLOW)
        trig_coords[0][12:].set_color(GREEN)
        trig_view = Group(circle_axes, trig_coords)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                trig_view.height * 1.2
            ).move_to(trig_view),
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(complex_coords[0][:2], trig_coords[0][:2]),
                    ReplacementTransform(complex_coords[0][2:4], trig_coords[0][10:12]),
                    ReplacementTransform(complex_coords[0][4:], trig_coords[0][12:14]),
                ),
                LaggedStart(
                    *[PopIn(glyph) for glyph in trig_coords[0][2:10]],
                    *[PopIn(glyph) for glyph in trig_coords[0][14:]],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.35,
                run_time=1.4,
            ),
        )
        self.remove(complex_coords, *complex_coords, *trig_coords[0])
        self.add(trig_coords)
        self.wait(0.5)

        evaluated_coords = (
            MathTex(r"\frac{V_0}{2} + j\frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .next_to(pair_point, UP, MED_SMALL_BUFF)
        )
        evaluated_coords[0][:4].set_color(YELLOW)
        evaluated_coords[0][6:].set_color(GREEN)
        evaluated_view = Group(circle_axes, evaluated_coords)
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                evaluated_view.height * 1.2
            ).move_to(evaluated_view),
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(trig_coords[0][:2], evaluated_coords[0][:2]),
                    ReplacementTransform(
                        trig_coords[0][10:12], evaluated_coords[0][4:6]
                    ),
                    ReplacementTransform(
                        trig_coords[0][12:14], evaluated_coords[0][6:8]
                    ),
                    FadeOut(trig_coords[0][2:10], trig_coords[0][14:]),
                ),
                LaggedStart(
                    *[PopIn(evaluated_coords[0][i]) for i in (2, 3, 8, 9, 10, 11, 12)],
                    lag_ratio=0.08,
                ),
                lag_ratio=0.4,
                run_time=1.4,
            ),
        )
        self.remove(trig_coords, *trig_coords, *evaluated_coords[0])
        self.add(evaluated_coords)
        complex_coords = evaluated_coords
        self.wait(0.5)
        self.next_section(
            "Multiplying by minus one", skip_animations=skip_animations(False)
        )

        shifted_coords = complex_coords.copy().set_opacity(0.6)
        original_coords_target = complex_coords.copy().next_to(
            shifted_coords, UP, MED_SMALL_BUFF
        )
        shifted_coords_end = shifted_coords.copy().next_to(
            coordinate_pair(~x + PI / omega),
            DOWN,
            MED_SMALL_BUFF + leading_point.width / 2,
        )
        negative_view = Group(
            circle_axes, original_coords_target, shifted_coords, shifted_coords_end
        )
        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(complex_coords, original_coords_target),
                    self.camera.frame.animate.scale_to_fit_height(
                        negative_view.height * 1.2
                    ).move_to(negative_view),
                    run_time=1.6,
                ),
                TransformFromCopy(
                    complex_coords, shifted_coords, introducer=True, run_time=0.9
                ),
                lag_ratio=0.5,
            )
        )
        self.wait(0.5)

        multiplied_coords = (
            MathTex(r"-1\left(\frac{V_0}{2} + j\frac{V_0\sqrt{3}}{2}\right)")
            .scale(0.7)
            .move_to(shifted_coords)
            .set_opacity(0.6)
        )
        multiplied_coords[0][3:7].set_color(YELLOW)
        multiplied_coords[0][9:16].set_color(GREEN)
        self.play(
            LaggedStart(
                ReplacementTransform(
                    shifted_coords[0], multiplied_coords[0][3:16], run_time=0.7
                ),
                AnimationGroup(
                    PopIn(multiplied_coords[0][2]),
                    PopIn(multiplied_coords[0][16]),
                ),
                LaggedStart(
                    *[PopIn(glyph) for glyph in multiplied_coords[0][:2]],
                    lag_ratio=0.15,
                ),
                lag_ratio=0.8,
            )
        )
        self.remove(shifted_coords, *shifted_coords, *multiplied_coords[0])
        self.add(multiplied_coords)
        self.wait(0.5)

        negated_coords = (
            MathTex(r"-\frac{V_0}{2} - j\frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .move_to(multiplied_coords)
            .set_opacity(0.6)
        )
        negated_coords[0][1:5].set_color(YELLOW)
        negated_coords[0][7:].set_color(GREEN)

        def position_phase_multiplier(mob):
            midpoint = leading_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            offset = (
                abs(outward[0]) * mob.width / 2
                + abs(outward[1]) * mob.height / 2
                + SMALL_BUFF / 2
            )
            mob.move_to(midpoint + outward * offset)

        phase_multiplier = MathTex(r"\times -1").scale(0.5).set_z_index(5)
        position_phase_multiplier(phase_multiplier)

        leading_phase_color = VT(0)

        def negative_phase_label():
            label = (
                MathTex(rf"+{~leading_phase:.0f}^\circ")
                .scale(0.7)
                .set_color(interpolate_color(WHITE, RED, ~leading_phase_color))
            )
            midpoint = leading_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            inset = (
                abs(outward[0]) * label.width / 2
                + abs(outward[1]) * label.height / 2
                + SMALL_BUFF
            )
            return (
                label.move_to(midpoint - outward * inset)
                .set_opacity(np.clip((~leading_phase - 20) / 25, 0, 1))
                .set_z_index(5)
            )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(
                        multiplied_coords[0][3:7], negated_coords[0][1:5]
                    ),
                    ReplacementTransform(
                        multiplied_coords[0][8:16], negated_coords[0][6:14]
                    ),
                    FadeOut(
                        multiplied_coords[0][2],
                        multiplied_coords[0][7],
                        multiplied_coords[0][16],
                    ),
                ),
                TransformFromCopy(
                    multiplied_coords[0][0], negated_coords[0][0], introducer=True
                ),
                TransformFromCopy(
                    multiplied_coords[0][0], negated_coords[0][5], introducer=True
                ),
                AnimationGroup(
                    ReplacementTransform(
                        multiplied_coords[0][:2], phase_multiplier[0][1:]
                    ),
                    PopIn(phase_multiplier[0][0]),
                ),
                lag_ratio=0.2,
                run_time=1.5,
            ),
        )
        self.remove(multiplied_coords, *multiplied_coords, *negated_coords[0])
        self.remove(*phase_multiplier[0])
        self.add(negated_coords, phase_multiplier)
        phase_multiplier.add_updater(position_phase_multiplier)

        def follow_shifted_point(mob):
            radial_adjustment = (angle_line.get_angle() - PI / 2) * ease_in_out_sine(
                np.clip(~leading_phase / 180, 0, 1)
            )
            direction = rotate_vector(UP, ~leading_phase * DEGREES + radial_adjustment)
            gap = leading_point.width / 2 + MED_SMALL_BUFF
            offset = np.array([mob.width / 2 + gap, mob.height / 2 + gap, 0])
            mob.move_to(leading_point.get_center() + direction * offset)

        negated_coords.add_updater(follow_shifted_point, call_updater=True)
        leading_phase_value = always_redraw(negative_phase_label)
        self.add(leading_phase_value)
        original_coords_end = complex_coords.copy().next_to(
            pair_point, UP, MED_SMALL_BUFF
        )
        negated_coords_end = negated_coords.copy().clear_updaters()
        end_direction = unit(coordinate_pair(~x + PI / omega) - circle_axes.c2p(0, 0))
        end_gap = leading_point.width / 2 + MED_SMALL_BUFF
        negated_coords_end.move_to(
            coordinate_pair(~x + PI / omega)
            + end_direction
            * np.array(
                [
                    negated_coords.width / 2 + end_gap,
                    negated_coords.height / 2 + end_gap,
                    0,
                ]
            )
        )
        rotation_view = Group(circle_axes, original_coords_end, negated_coords_end)
        self.play(
            LaggedStart(
                leading_phase.animate(run_time=3, rate_func=ease_in_out_sine).set_value(
                    180
                ),
                AnimationGroup(
                    Transform(complex_coords, original_coords_end),
                    self.camera.frame.animate.scale_to_fit_height(
                        rotation_view.height * 1.2
                    ).move_to(rotation_view),
                    run_time=1.4,
                ),
                lag_ratio=0.35,
            )
        )

        self.wait(0.5)

        neg1_shift_mult = MathTex(r"\times -1")
        neg1_shift_phase = MathTex(r"+180^\circ")
        neg1_shift_arrow = Arrow(LEFT * 0.5, RIGHT)
        neg1_shift = (
            Group(neg1_shift_mult, neg1_shift_arrow, neg1_shift_phase)
            .arrange(RIGHT, SMALL_BUFF)
            .scale(0.7)
            .next_to(circle_axes, LEFT, LARGE_BUFF)
            .shift(UP * 2)
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    phase_multiplier[0], neg1_shift_mult[0], path_arc=PI / 3
                ),
                GrowArrow(neg1_shift_arrow),
                TransformFromCopy(
                    leading_phase_value[0],
                    neg1_shift_phase[0],
                    introducer=True,
                    path_arc=PI / 3,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        want90 = (
            Group(Text("want", font=FONT).scale(0.7), MathTex(r"90^\circ"))
            .arrange(RIGHT, MED_SMALL_BUFF, aligned_edge=DOWN)
            .next_to(circle_axes, LEFT, LARGE_BUFF)
            .shift(DOWN * 0.5)
        )

        want90_arrow = Arrow(
            leading_phase_value.get_corner(DL),
            want90.get_right(),
            buff=SMALL_BUFF,
            path_arc=-PI / 3,
        )

        self.play(
            LaggedStart(
                leading_phase_color @ 1,
                GrowArrow(want90_arrow),
                FadeIn(want90),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(want90_arrow, want90), leading_phase_color @ 0)

        self.next_section("Two quarter turns", skip_animations=skip_animations(False))
        negated_coords.clear_updaters()
        phase_multiplier.clear_updaters()
        self.play(
            LaggedStart(
                FadeOut(negated_coords, phase_multiplier, run_time=0.5),
                leading_phase.animate(
                    run_time=1.8, rate_func=ease_in_out_sine
                ).set_value(0),
                lag_ratio=0.3,
            )
        )
        self.wait(0.3)

        quarter_multiplier = MathTex(r"\times ?").scale(0.5).set_z_index(5)
        quarter_multiplier.add_updater(position_phase_multiplier)
        quarter_multiplier.add_updater(
            lambda mob: mob.set_opacity(np.clip((~leading_phase - 20) / 25, 0, 1)),
            call_updater=True,
        )
        self.add(quarter_multiplier)
        self.play(
            leading_phase.animate(run_time=2, rate_func=ease_in_out_sine).set_value(90)
        )
        self.wait(0.5)

        second_phase = VT(0)

        def second_time():
            return leading_time() + ~second_phase * DEGREES / omega

        second_line = leading_line.copy().clear_updaters()
        second_point = leading_point.copy().clear_updaters()
        second_line.add_updater(
            lambda mob: mob.put_start_and_end_on(
                circle_axes.c2p(0, 0), coordinate_pair(second_time())
            )
        )
        second_point.add_updater(
            lambda mob: mob.move_to(coordinate_pair(second_time()))
        )
        second_real_point = leading_real_point.copy().clear_updaters()
        second_imag_point = leading_imag_point.copy().clear_updaters()
        second_real_point.add_updater(
            lambda mob: mob.move_to(
                plot_point(now_axes, second_time(), voltage(second_time()))
            )
        )
        second_imag_point.add_updater(
            lambda mob: mob.move_to(
                plot_point(
                    lagged_axes, second_time(), voltage(second_time() - lagged_shift)
                )
            )
        )
        second_phase_arc = always_redraw(
            lambda: Arc(
                radius=angle_line.get_length() * 0.6,
                start_angle=angle_line.get_angle() + ~leading_phase * DEGREES,
                angle=~second_phase * DEGREES,
                arc_center=circle_axes.c2p(0, 0),
                stroke_opacity=0.7,
            ).set_z_index(3)
        )

        def second_phase_label():
            label = MathTex(rf"+{~second_phase:.0f}^\circ").scale(0.7)
            midpoint = second_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            inset = (
                abs(outward[0]) * label.width / 2
                + abs(outward[1]) * label.height / 2
                + SMALL_BUFF
            )
            return (
                label.move_to(midpoint - outward * inset)
                .set_opacity(np.clip((~second_phase - 20) / 25, 0, 1))
                .set_z_index(5)
            )

        second_phase_value = always_redraw(second_phase_label)
        second_multiplier = quarter_multiplier.copy().clear_updaters()

        def position_second_multiplier(mob):
            midpoint = second_phase_arc.get_midpoint()
            outward = unit(midpoint - circle_axes.c2p(0, 0))
            offset = (
                abs(outward[0]) * mob.width / 2
                + abs(outward[1]) * mob.height / 2
                + SMALL_BUFF / 2
            )
            mob.move_to(midpoint + outward * offset)
            mob.set_opacity(np.clip((~second_phase - 20) / 25, 0, 1))

        second_multiplier.add_updater(position_second_multiplier, call_updater=True)
        self.add(
            second_line,
            second_point,
            second_real_point,
            second_imag_point,
            second_phase_arc,
            second_phase_value,
            second_multiplier,
        )
        self.play(
            second_phase.animate(run_time=2, rate_func=ease_in_out_sine).set_value(90)
        )

        self.wait(0.5)

        qmark_shift_mult = MathTex(r"\times ? \times ?")
        qmark_shift_phase = MathTex(r"+90^\circ + 90^\circ")
        qmark_shift_arrow = Arrow(LEFT * 0.5, RIGHT)
        qmark_shift = (
            Group(qmark_shift_mult, qmark_shift_arrow, qmark_shift_phase)
            .arrange(RIGHT, SMALL_BUFF)
            .scale(0.7)
            .next_to(neg1_shift, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    quarter_multiplier[0],
                    qmark_shift_mult[0][:2],
                    path_arc=PI / 3,
                    introducer=True,
                ),
                TransformFromCopy(
                    second_multiplier[0],
                    qmark_shift_mult[0][2:4],
                    path_arc=PI / 3,
                    introducer=True,
                ),
                GrowArrow(qmark_shift_arrow),
                TransformFromCopy(
                    leading_phase_value[0],
                    qmark_shift_phase[0][:4],
                    introducer=True,
                    path_arc=PI / 3,
                ),
                TransformFromCopy(
                    second_phase_value[0],
                    qmark_shift_phase[0][4:],
                    introducer=True,
                    path_arc=PI / 3,
                ),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        qmark_sq = MathTex(r"?^2 = -1")
        qmark = MathTex(r"\Rightarrow ? = \sqrt{-1}")
        qmark_j = MathTex(r"\Rightarrow ? = j")
        qmark_group = (
            Group(qmark_sq, qmark, qmark_j)
            .arrange(DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)
            .next_to(qmark_shift, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        )

        self.play(
            LaggedStart(
                TransformFromCopy(
                    qmark_shift_mult[0][1], qmark_sq[0][0], introducer=True
                ),
                TransformFromCopy(
                    qmark_shift_mult[0][3], qmark_sq[0][1], introducer=True
                ),
                PopIn(qmark_sq[0][2]),
                TransformFromCopy(
                    neg1_shift_mult[0][1:], qmark_sq[0][3:], introducer=True
                ),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.play(LaggedStart(*[PopIn(m) for m in qmark[0]], lag_ratio=0.15))

        self.wait(0.5)

        j_box = SurroundingRectangle(qmark_j, buff=SMALL_BUFF, color=GREEN)

        self.play(
            LaggedStart(*[PopIn(m) for m in qmark_j[0]], Create(j_box), lag_ratio=0.15)
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(qmark),
                FadeOut(qmark_sq),
                FadeOut(qmark_shift),
                FadeOut(neg1_shift),
                Group(qmark_j, j_box).animate.set_y(circle_axes.c2p(0, 0)[1]),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        self.next_section(
            "Return to the original point", skip_animations=skip_animations(False)
        )
        self.play(second_phase @ 0, run_time=1.4, rate_func=ease_in_out_sine)
        self.remove(
            second_line,
            second_point,
            second_real_point,
            second_imag_point,
            second_phase_arc,
            second_phase_value,
            second_multiplier,
        )
        self.play(leading_phase @ 0, run_time=1.4, rate_func=ease_in_out_sine)
        self.remove(quarter_multiplier, leading_phase_value)
        leading_phase_value = always_redraw(negative_phase_label)
        self.add(leading_phase_value)
        self.wait(0.5)

        work_center = circle_axes.c2p(0, 0) + LEFT * fw(self, 0.43)
        working_coords = complex_coords.copy().move_to(work_center)
        identity_target = qmark_j.copy().next_to(working_coords, UP, LARGE_BUFF)
        identity_box_target = SurroundingRectangle(
            identity_target, buff=SMALL_BUFF, color=GREEN
        )
        first_coords = (
            MathTex(r"-\frac{V_0\sqrt{3}}{2} + j\frac{V_0}{2}")
            .scale(0.7)
            .move_to(work_center)
        )
        first_coords[0][:8].set_color(YELLOW)
        first_coords[0][10:].set_color(GREEN)
        first_coords_end = first_coords.copy().next_to(
            coordinate_pair(~x + PI / 2 / omega),
            UP,
            MED_SMALL_BUFF + leading_point.height / 2,
            aligned_edge=RIGHT,
        )
        algebra_view = Group(
            circle_axes,
            complex_coords,
            identity_box_target,
            working_coords,
            first_coords_end,
        )
        self.play(
            self.camera.frame.animate.scale_to_fit_height(
                max(algebra_view.height * 1.15, algebra_view.width * 9 / 16 * 1.15)
            ).move_to(algebra_view),
            Transform(qmark_j, identity_target),
            Transform(j_box, identity_box_target),
            run_time=1.2,
        )

        self.next_section(
            "Multiply the original coordinates by j",
            skip_animations=skip_animations(False),
        )
        self.play(TransformFromCopy(complex_coords, working_coords), run_time=1)
        j_product = (
            MathTex(r"j\left(\frac{V_0}{2} + j\frac{V_0\sqrt{3}}{2}\right)")
            .scale(0.7)
            .move_to(work_center)
        )
        j_product[0][0].set_color(ORANGE)
        j_product[0][2:6].set_color(YELLOW)
        j_product[0][8:15].set_color(GREEN)
        self.play(
            LaggedStart(
                ReplacementTransform(working_coords[0], j_product[0][2:15]),
                AnimationGroup(
                    GrowFromCenter(j_product[0][1]), GrowFromCenter(j_product[0][15])
                ),
                TransformFromCopy(qmark_j[0][-1], j_product[0][0], introducer=True),
                lag_ratio=0.2,
                run_time=1.2,
            )
        )
        self.remove(working_coords, *working_coords, *j_product[0])
        self.add(j_product)
        self.wait(0.5)

        self.next_section(
            "Distribute j to both terms", skip_animations=skip_animations(False)
        )
        distributed = (
            MathTex(r"j\frac{V_0}{2} + j\cdot j\frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .move_to(work_center)
        )
        distributed[0][0].set_color(ORANGE)
        distributed[0][1:5].set_color(GREEN)
        distributed[0][6].set_color(ORANGE)
        distributed[0][9:].set_color(GREEN)
        outer_j = j_product[0][0]
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(j_product[0][2:6], distributed[0][1:5]),
                    ReplacementTransform(j_product[0][6], distributed[0][5]),
                    ReplacementTransform(j_product[0][7:15], distributed[0][8:]),
                    FadeOut(j_product[0][1], j_product[0][15]),
                ),
                TransformFromCopy(
                    outer_j,
                    distributed[0][0],
                    path_arc=-PI / 3,
                    introducer=True,
                ),
                AnimationGroup(
                    ReplacementTransform(
                        outer_j,
                        distributed[0][6],
                        path_arc=-PI / 3,
                    ),
                    GrowFromCenter(distributed[0][7]),
                ),
                lag_ratio=0.3,
                run_time=1.6,
            )
        )
        self.remove(j_product, *j_product, outer_j, *distributed[0])
        self.add(distributed)
        self.wait(0.5)

        self.next_section(
            "The imaginary j becomes j squared", skip_animations=skip_animations(False)
        )
        squared = (
            MathTex(r"j\frac{V_0}{2} + j^2\frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .move_to(work_center)
        )
        squared[0][0].set_color(ORANGE)
        squared[0][1:5].set_color(GREEN)
        squared[0][6:8].set_color(ORANGE)
        squared[0][8:].set_color(GREEN)
        self.play(
            ReplacementTransform(distributed[0][:7], squared[0][:7]),
            ReplacementTransform(distributed[0][8], squared[0][7], path_arc=PI / 3),
            ReplacementTransform(distributed[0][9:], squared[0][8:]),
            FadeOut(distributed[0][7]),
            run_time=0.8,
        )
        self.remove(distributed, *distributed, *squared[0])
        self.add(squared)
        self.wait(0.5)

        self.next_section(
            "j squared is minus one", skip_animations=skip_animations(False)
        )
        j_identity = MathTex(r"j^2 = -1").scale(0.8).move_to(qmark_j)
        j_identity[0][:2].set_color(ORANGE)
        self.play(
            TransformFromCopy(squared[0][6:8], j_identity[0][:2]),
            ReplacementTransform(qmark_j[0][2], j_identity[0][2]),
            FadeOut(qmark_j[0][:2], qmark_j[0][3]),
            GrowFromCenter(j_identity[0][3:]),
            Transform(
                j_box, SurroundingRectangle(j_identity, buff=SMALL_BUFF, color=GREEN)
            ),
            run_time=0.8,
        )
        self.remove(qmark_j, *qmark_j, *j_identity[0])
        self.add(j_identity)
        negative_product = (
            MathTex(r"j\frac{V_0}{2} + (-1)\frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .move_to(work_center)
        )
        negative_product[0][1:5].set_color(GREEN)
        negative_product[0][6:].set_color(YELLOW)
        self.play(
            ReplacementTransform(squared[0][:6], negative_product[0][:6]),
            FadeOut(squared[0][6:8]),
            TransformFromCopy(j_identity[0][3:], negative_product[0][7:9]),
            GrowFromCenter(negative_product[0][6]),
            GrowFromCenter(negative_product[0][9]),
            ReplacementTransform(squared[0][8:], negative_product[0][10:]),
            run_time=1,
        )
        self.remove(squared, *squared, *negative_product[0])
        self.add(negative_product)
        self.wait(0.5)

        simplified = (
            MathTex(r"j\frac{V_0}{2} - \frac{V_0\sqrt{3}}{2}")
            .scale(0.7)
            .move_to(work_center)
        )
        simplified[0][1:5].set_color(GREEN)
        simplified[0][5:].set_color(YELLOW)
        self.play(
            ReplacementTransform(negative_product[0][:5], simplified[0][:5]),
            ReplacementTransform(negative_product[0][7], simplified[0][5]),
            ReplacementTransform(negative_product[0][10:], simplified[0][6:]),
            FadeOut(negative_product[0][5:7], negative_product[0][8:10]),
            run_time=0.8,
        )
        self.remove(negative_product, *negative_product, *simplified[0])
        self.add(simplified)
        self.wait(0.5)

        self.next_section("Real part first", skip_animations=skip_animations(False))
        self.play(
            ReplacementTransform(simplified[0][5:], first_coords[0][:8], path_arc=-PI),
            ReplacementTransform(simplified[0][:5], first_coords[0][9:], path_arc=-PI),
            GrowFromCenter(first_coords[0][8]),
            run_time=1.2,
        )
        self.remove(simplified, *simplified, *first_coords[0])
        self.add(first_coords)
        self.wait(0.5)

        self.next_section(
            "j rotates the point by 90 degrees", skip_animations=skip_animations(False)
        )
        first_point_coords = first_coords_end.copy()
        quarter_multiplier = MathTex(r"\times j").scale(0.5).set_z_index(5)
        quarter_multiplier.add_updater(position_phase_multiplier)
        quarter_multiplier.add_updater(
            lambda mob: mob.set_opacity(np.clip((~leading_phase - 20) / 25, 0, 1)),
            call_updater=True,
        )
        self.add(quarter_multiplier)
        self.play(
            leading_phase @ 90,
            TransformFromCopy(first_coords, first_point_coords, path_arc=-PI / 6),
            run_time=2,
            rate_func=ease_in_out_sine,
        )
        self.next_section(
            "The phase component of the derivative",
            skip_animations=skip_animations(False),
        )
        leading_phase_arc.clear_updaters()
        self.play(
            leading_phase_arc.animate.set_stroke(color=GREEN, width=8, opacity=1),
            run_time=0.6,
        )
        self.wait(0.8)

        self.next_section(
            "Differentiation becomes multiplication by j omega",
            skip_animations=skip_animations(False),
        )
        dvdt = MathTex(r"\frac{d V(t)}{dt}")
        dvdt[0][1:5].set_color(YELLOW)
        dvdt_arrow = Arrow(LEFT, RIGHT)
        jw = MathTex(r"j \omega \tilde V")
        jw[0][2:].set_color(YELLOW)
        dvdt_group = (
            Group(dvdt, dvdt_arrow, jw)
            .arrange(RIGHT, MED_SMALL_BUFF)
            .move_to(self.camera.frame)
            .shift(UP * fh(self))
        )
        dvdt.save_state()
        dvdt.set_x(self.camera.frame.get_x())

        self.play(self.camera.frame.animate.shift(UP * fh(self)))
        self.remove(*self.mobjects)
        self.play(FadeIn(dvdt), run_time=0.5)

        self.play(
            LaggedStart(
                dvdt.animate.restore(),
                GrowArrow(dvdt_arrow),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        complex_arrow = Arrow(
            dvdt[0][1:5].get_top(),
            jw[0][2:].get_top(),
            buff=SMALL_BUFF,
            path_arc=-PI / 2,
            color=YELLOW,
        )

        self.play(
            LaggedStart(
                GrowArrow(complex_arrow),
                PopIn(jw[0][2:]),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeOut(complex_arrow),
                LaggedStart(
                    PopIn(jw[0][0]),
                    PopIn(jw[0][1]),
                    lag_ratio=0.15,
                ),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))


class CapExample(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        cap = (
            Cap(include_ports=True)
            .scale(1.5)
            .next_to(self.camera.frame.get_left(), RIGHT, LARGE_BUFF)
        )

        current = MathTex(r"I_C(t) = C \frac{d V_C(t)}{dt}")
        current[0][:5].set_color(BLUE)
        current[0][8:13].set_color(YELLOW)

        self.play(
            LaggedStart(
                cap.shift(LEFT * fw(self)).animate.shift(RIGHT * fw(self)),
                LaggedStart(*[PopIn(m) for m in current[0]], lag_ratio=0.15),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        sinusoid_callout = (
            Text("sinusoid", font=FONT)
            .next_to(current[0][8:13], UP, LARGE_BUFF)
            .shift(RIGHT * 3)
        )
        sinusoid_arrow = SquigglyLine(
            current[0][8:13].get_corner(UR) + [0, 0.1, 0],
            sinusoid_callout.get_corner(DL) + [-0.1, 0.1, 0],
            amplitude=0.1,
            seed=3,
        ).add_tip(
            tip_length=DEFAULT_ARROW_TIP_LENGTH * 0.5,
            tip_width=DEFAULT_ARROW_TIP_LENGTH * 0.5,
        )

        self.play(
            LaggedStart(
                GrowArrow(sinusoid_arrow),
                PopIn(sinusoid_callout),
                lag_ratio=0.2,
            )
        )

        self.wait(0.5)

        complex_current = MathTex(r"\tilde I_C = j \omega C \tilde V_C").next_to(
            current, DOWN, SMALL_BUFF
        )
        complex_current[0][:3].set_color(BLUE)
        complex_current[0][7:].set_color(YELLOW)

        self.play(
            LaggedStart(
                FadeOut(sinusoid_arrow, sinusoid_callout),
                current.animate.shift(UP * fh(self, 0.25)),
                lag_ratio=0.3,
            )
        )

        arrow_start = current.get_right() + RIGHT * MED_SMALL_BUFF
        arrow_end = complex_current.get_left() + LEFT * MED_SMALL_BUFF
        arrow_mid = (arrow_start + arrow_end) / 2 + UP * 0.1
        current_arrow = TipableVMobject(
            color=GREEN, stroke_width=3, tip_corner_radius=0.02
        )
        current_arrow.start_new_path(arrow_start)
        current_arrow.add_cubic_bezier_curve_to(
            arrow_start + RIGHT * 3.1 + DOWN * 0.8,
            arrow_mid + RIGHT * 3.5 + UP * 0.15,
            arrow_mid,
        )
        current_arrow.add_cubic_bezier_curve_to(
            arrow_mid + LEFT * 3.5 + DOWN * 0.15,
            arrow_end + LEFT * 3.5,
            arrow_end,
        )
        current_arrow.add_tip(tip_length=0.2, tip_width=0.2)

        self.next_section(
            "Derivative to complex current", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                GrowArrow(current_arrow, run_time=1.5),
                LaggedStart(*[PopIn(m) for m in complex_current[0]], lag_ratio=0.15),
                lag_ratio=0.7,
            )
        )
        self.wait(0.5)

        self.next_section(
            "Write the complex voltage", skip_animations=skip_animations(False)
        )
        voltage = (
            MathTex(r"\tilde V_C = V_0\cos(\omega t) + j V_0\sin(\omega t)")
            .scale(0.9)
            .move_to(DOWN * fh(self, 0.2))
        )
        voltage[0][:3].set_color(YELLOW)
        voltage[0][4:13].set_color(YELLOW)
        voltage[0][15:].set_color(GREEN)
        self.play(FadeOut(cap, current_arrow), complex_current.animate.move_to(ORIGIN))
        self.play(
            LaggedStart(
                TransformFromCopy(
                    complex_current[0][7:], voltage[0][:3], introducer=True
                ),
                LaggedStart(
                    *[PopIn(glyph) for glyph in voltage[0][3:]], lag_ratio=0.06
                ),
                lag_ratio=0.3,
                run_time=1.4,
            )
        )
        self.remove(*voltage[0])
        self.add(voltage)
        self.wait(0.5)

        self.next_section(
            "Substitute the voltage", skip_animations=skip_animations(False)
        )
        expanded = MathTex(
            r"\tilde I_C = j\omega C\left(V_0\cos(\omega t)+j V_0\sin(\omega t)\right)"
        ).scale(0.95)
        expanded[0][:3].set_color(BLUE)
        expanded[0][4:7].set_color(ORANGE)
        expanded[0][8:17].set_color(YELLOW)
        expanded[0][19:28].set_color(GREEN)
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(complex_current[0][:7], expanded[0][:7]),
                    FadeOut(complex_current[0][7:]),
                    GrowFromCenter(expanded[0][7]),
                    GrowFromCenter(expanded[0][28]),
                ),
                TransformFromCopy(voltage[0][4:], expanded[0][8:28], introducer=True),
                lag_ratio=0.25,
                run_time=1.3,
            )
        )
        self.remove(complex_current, *complex_current, *expanded[0])
        self.add(expanded)
        self.wait(0.5)

        self.next_section(
            "Distribute j omega C", skip_animations=skip_animations(False)
        )
        distributed = MathTex(
            r"\tilde I_C = j\omega C V_0\cos(\omega t) + j\cdot j\omega C V_0\sin(\omega t)"
        ).scale(0.95)
        distributed[0][:3].set_color(BLUE)
        distributed[0][4].set_color(ORANGE)
        distributed[0][5:16].set_color(GREEN)
        distributed[0][17].set_color(ORANGE)
        distributed[0][20:].set_color(GREEN)
        factor = expanded[0][4:7]
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(expanded[0][:4], distributed[0][:4]),
                    ReplacementTransform(expanded[0][8:17], distributed[0][7:16]),
                    ReplacementTransform(expanded[0][17], distributed[0][16]),
                    ReplacementTransform(expanded[0][18], distributed[0][19]),
                    ReplacementTransform(expanded[0][19:28], distributed[0][22:]),
                    FadeOut(expanded[0][7], expanded[0][28], voltage),
                ),
                TransformFromCopy(
                    factor, distributed[0][4:7], path_arc=-PI / 3, introducer=True
                ),
                AnimationGroup(
                    ReplacementTransform(
                        factor[0], distributed[0][17], path_arc=-PI / 3
                    ),
                    ReplacementTransform(
                        factor[1:], distributed[0][20:22], path_arc=-PI / 3
                    ),
                    GrowFromCenter(distributed[0][18]),
                ),
                lag_ratio=0.3,
                run_time=1.6,
            )
        )
        self.remove(expanded, *expanded, factor, *factor, *distributed[0])
        self.add(distributed)
        self.wait(0.5)

        self.next_section(
            "Combine the two j factors", skip_animations=skip_animations(False)
        )
        squared = MathTex(
            r"\tilde I_C = j\omega C V_0\cos(\omega t) + j^2\omega C V_0\sin(\omega t)"
        ).scale(0.95)
        squared[0][:3].set_color(BLUE)
        squared[0][5:16].set_color(GREEN)
        squared[0][17:19].set_color(ORANGE)
        squared[0][19:].set_color(GREEN)
        self.play(
            ReplacementTransform(distributed[0][:18], squared[0][:18]),
            ReplacementTransform(distributed[0][19], squared[0][18], path_arc=PI / 3),
            ReplacementTransform(distributed[0][20:], squared[0][19:]),
            FadeOut(distributed[0][18]),
            run_time=0.8,
        )
        self.remove(distributed, *distributed, *squared[0])
        self.add(squared)
        self.wait(0.5)

        self.next_section(
            "j squared makes the sine term negative",
            skip_animations=skip_animations(False),
        )
        j_identity = MathTex(r"j^2=-1").scale(0.8).next_to(squared, UP, MED_LARGE_BUFF)
        j_identity[0][:2].set_color(ORANGE)
        self.play(
            TransformFromCopy(squared[0][17:19], j_identity[0][:2]),
            GrowFromCenter(j_identity[0][2:]),
            run_time=0.8,
        )
        self.remove(*j_identity[0])
        self.add(j_identity)
        negative = MathTex(
            r"\tilde I_C = j\omega C V_0\cos(\omega t) - \omega C V_0\sin(\omega t)"
        ).scale(0.95)
        negative[0][:3].set_color(BLUE)
        negative[0][5:16].set_color(GREEN)
        negative[0][16:].set_color(BLUE)
        self.play(
            ReplacementTransform(squared[0][:16], negative[0][:16]),
            ReplacementTransform(squared[0][19:], negative[0][17:]),
            FadeOut(squared[0][16:19]),
            TransformFromCopy(j_identity[0][3], negative[0][16]),
            run_time=1,
        )
        self.remove(squared, *squared, *negative[0])
        self.add(negative)
        self.wait(0.5)

        self.next_section(
            "Real current followed by imaginary current",
            skip_animations=skip_animations(False),
        )
        result = MathTex(
            r"\tilde I_C = -\omega C V_0\sin(\omega t) + j\omega C V_0\cos(\omega t)"
        ).scale(0.95)
        result[0][:3].set_color(BLUE)
        result[0][4:16].set_color(BLUE)
        result[0][18:].set_color(GREEN)
        self.play(FadeOut(j_identity), run_time=0.3)
        self.play(
            ReplacementTransform(negative[0][:4], result[0][:4]),
            ReplacementTransform(negative[0][16:], result[0][4:16], path_arc=-PI / 2),
            ReplacementTransform(negative[0][4:16], result[0][17:], path_arc=-PI / 2),
            GrowFromCenter(result[0][16]),
            run_time=1.3,
        )
        self.remove(negative, *negative, *result[0])
        self.add(result)
        self.wait(0.5)

        self.next_section(
            "Check against the time derivative", skip_animations=skip_animations(False)
        )
        time_substitution = MathTex(r"I_C(t)=C\frac{d V_0\cos(\omega t)}{dt}").move_to(
            current
        )
        time_substitution[0][:5].set_color(BLUE)
        time_substitution[0][8:17].set_color(YELLOW)
        self.play(
            ReplacementTransform(current[0][:8], time_substitution[0][:8]),
            ReplacementTransform(current[0][13:], time_substitution[0][17:]),
            FadeOut(current[0][8:13]),
            TransformFromCopy(result[0][20:], time_substitution[0][8:17]),
            run_time=1.1,
        )
        self.remove(current, *current, *time_substitution[0])
        self.add(time_substitution)
        self.wait(0.5)
        physical_current = MathTex(r"I_C(t)=-\omega C V_0\sin(\omega t)").move_to(
            current
        )
        physical_current[0][:5].set_color(BLUE)
        physical_current[0][6:].set_color(BLUE)
        self.play(
            LaggedStart(
                TransformFromCopy(
                    time_substitution[0][14],
                    physical_current[0][7],
                    path_arc=PI / 3,
                    introducer=True,
                ),
                AnimationGroup(
                    ReplacementTransform(
                        time_substitution[0][:6], physical_current[0][:6]
                    ),
                    ReplacementTransform(
                        time_substitution[0][6], physical_current[0][8]
                    ),
                    ReplacementTransform(
                        time_substitution[0][8:10], physical_current[0][9:11]
                    ),
                    ReplacementTransform(
                        time_substitution[0][10:13], physical_current[0][11:14]
                    ),
                    ReplacementTransform(
                        time_substitution[0][13:17], physical_current[0][14:]
                    ),
                    FadeOut(time_substitution[0][7], time_substitution[0][17:]),
                    GrowFromCenter(physical_current[0][6]),
                ),
                lag_ratio=0.2,
                run_time=1.2,
            )
        )
        self.remove(time_substitution, *time_substitution, *physical_current[0])
        self.add(physical_current)
        real_box = SurroundingRectangle(result[0][4:16], color=BLUE, buff=SMALL_BUFF)
        derivative_box = SurroundingRectangle(
            physical_current[0][6:], color=BLUE, buff=SMALL_BUFF
        )
        self.play(Create(real_box), Create(derivative_box), run_time=0.7)
        self.wait(0.8)

        self.next_section(
            "The real part is the physical current",
            skip_animations=skip_animations(False),
        )
        real_brace = Brace(result[0][4:16], DOWN, color=BLUE, buff=MED_SMALL_BUFF)
        real_label = (
            MathTex(r"I_C(t)", color=BLUE)
            .scale(0.8)
            .next_to(real_brace, DOWN, SMALL_BUFF)
        )
        real_caption = Text(
            "physical current", font=FONT, font_size=24, color=BLUE
        ).next_to(real_label, DOWN, MED_SMALL_BUFF)
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(real_box, derivative_box), GrowFromCenter(real_brace)
                ),
                TransformFromCopy(
                    physical_current[0][:5], real_label[0], introducer=True
                ),
                FadeIn(real_caption),
                lag_ratio=0.2,
                run_time=1.1,
            )
        )
        self.remove(*real_label[0])
        self.add(real_label)
        self.wait(0.5)

        self.next_section(
            "The imaginary part is a quarter cycle earlier",
            skip_animations=skip_animations(False),
        )
        imag_brace = Brace(result[0][18:], DOWN, color=GREEN, buff=MED_SMALL_BUFF)
        imag_label = (
            MathTex(r"I_C\left(t-\frac{T}{4}\right)", color=GREEN)
            .scale(0.8)
            .next_to(imag_brace, DOWN, SMALL_BUFF)
        )
        period = (
            MathTex(r"T=\frac{2\pi}{\omega}")
            .scale(0.7)
            .next_to(imag_label, DOWN, MED_LARGE_BUFF)
        )
        self.play(
            LaggedStart(
                GrowFromCenter(imag_brace),
                AnimationGroup(
                    TransformFromCopy(
                        real_label[0][:4], imag_label[0][:4], introducer=True
                    ),
                    TransformFromCopy(
                        real_label[0][4], imag_label[0][8], introducer=True
                    ),
                    GrowFromCenter(imag_label[0][4:8]),
                ),
                FadeIn(period),
                lag_ratio=0.2,
                run_time=1.4,
            )
        )
        self.remove(*imag_label[0])
        self.add(imag_label)
        self.wait(1)

        self.next_section(
            "Rearrange the complex current law", skip_animations=skip_animations(False)
        )
        compact = MathTex(r"\tilde I_C=j\omega C\tilde V_C").scale(1.3)
        compact[0][:3].set_color(BLUE)
        compact[0][7:].set_color(YELLOW)
        resistor_law = MathTex(r"V=IR").scale(1.2).move_to(UP * fh(self, 0.28))
        resistor_law[0][0].set_color(YELLOW)
        resistor_law[0][2].set_color(BLUE)
        resistor_law[0][3].set_color(RED)
        self.play(
            FadeOut(
                result,
                physical_current,
                real_brace,
                real_label,
                real_caption,
                imag_brace,
                imag_label,
                period,
            ),
            run_time=0.7,
        )
        self.play(
            FadeIn(compact),
            LaggedStart(*[PopIn(glyph) for glyph in resistor_law[0]], lag_ratio=0.15),
        )
        self.wait(0.5)

        self.next_section(
            "Voltage equals current times a constant",
            skip_animations=skip_animations(False),
        )
        factored = MathTex(r"\tilde V_C=\tilde I_C\frac{1}{j\omega C}").scale(1.3)
        factored[0][:3].set_color(YELLOW)
        factored[0][4:7].set_color(BLUE)
        factored[0][7:].set_color(GREEN)
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(compact[0][7:], factored[0][:3], path_arc=PI),
                    ReplacementTransform(compact[0][3], factored[0][3]),
                    ReplacementTransform(compact[0][:3], factored[0][4:7], path_arc=PI),
                    ReplacementTransform(compact[0][4:7], factored[0][8:11]),
                ),
                AnimationGroup(
                    GrowFromCenter(factored[0][7]),
                    GrowFromCenter(factored[0][11]),
                ),
                lag_ratio=0.6,
                run_time=1.2,
            )
        )
        self.remove(compact, *compact, *factored[0])
        self.add(factored)
        self.wait(0.5)

        self.next_section("Name the impedance", skip_animations=skip_animations(False))
        impedance_brace = Brace(factored[0][7:], DOWN, color=GREEN, buff=MED_SMALL_BUFF)
        impedance_label = MathTex("Z", color=GREEN).next_to(
            impedance_brace, DOWN, SMALL_BUFF
        )
        self.play(
            LaggedStart(
                GrowFromCenter(impedance_brace),
                GrowFromCenter(impedance_label),
                lag_ratio=0.2,
            )
        )
        self.wait(0.5)
        impedance = (
            MathTex(r"Z=\frac{1}{j\omega C}", color=GREEN)
            .scale(0.95)
            .move_to(DOWN * fh(self, 0.33))
        )
        self.play(
            TransformFromCopy(impedance_label[0], impedance[0][:1], path_arc=-PI / 3),
            GrowFromCenter(impedance[0][1]),
            TransformFromCopy(factored[0][7:], impedance[0][2:], path_arc=PI),
            run_time=1,
        )
        self.remove(*impedance[0])
        self.add(impedance)
        self.wait(0.5)

        self.next_section(
            "Ohms law with impedance", skip_animations=skip_animations(False)
        )
        ohms_law = MathTex(r"\tilde V_C=\tilde I_C Z").scale(1.8)
        ohms_law[0][:3].set_color(YELLOW)
        ohms_law[0][4:7].set_color(BLUE)
        ohms_law[0][7].set_color(GREEN)
        self.play(FadeOut(factored[0][7:], impedance_brace), run_time=0.3)
        self.play(
            ReplacementTransform(factored[0][:7], ohms_law[0][:7]),
            ReplacementTransform(
                impedance_label[0][0], ohms_law[0][7], path_arc=PI / 3
            ),
            run_time=0.9,
        )
        self.remove(
            factored, *factored, impedance_label, *impedance_label, *ohms_law[0]
        )
        self.add(ohms_law)
        self.wait(1)

        self.next_section(
            "Return to the Sallen-Key circuit", skip_animations=skip_animations(False)
        )
        wire_color = "#D6EEF7"
        wire_style = dict(color=wire_color, stroke_width=4)
        unit = fh(self, 0.11)
        center = DOWN * fh(self, 0.08)
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
        input_label = (
            MathTex(r"\tilde V_{in}", color=YELLOW)
            .scale(0.8)
            .next_to(input_port, UP, MED_LARGE_BUFF)
        )
        output_label = (
            MathTex(r"\tilde V_{out}", color=YELLOW)
            .scale(0.8)
            .next_to(output_port, UP, MED_LARGE_BUFF)
        )
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
            plate_width=fw(self, 0.068) * unit,
            plate_thickness=fh(self, 0.006) * unit,
            gap=fh(self, 0.035) * unit,
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
        input_signs[0].set_color(RED).move_to(center + unit * np.array([2.65, 0.6, 0]))
        input_signs[1].set_color(BLUE).move_to(
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
        circuit = VGroup(
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
        )
        r1_label = MathTex("R_1", color=wire_color).scale(0.75).next_to(r1, UP, 0.3)
        r2_label = MathTex("R_2", color=wire_color).scale(0.75).next_to(r2, UP, 0.3)
        c1_label = MathTex("C_1", color=wire_color).scale(0.75).next_to(c1, UP, 0.25)
        c2_label = MathTex("C_2", color=wire_color).scale(0.75).next_to(c2, LEFT, 0.3)
        circuit_law = (
            MathTex(r"\tilde V=\tilde I Z")
            .scale(1.2)
            .move_to(RIGHT * fw(self, 0.29) + UP * fh(self, 0.37))
        )
        circuit_law[0][:2].set_color(YELLOW)
        circuit_law[0][3:5].set_color(BLUE)
        circuit_law[0][5].set_color(GREEN)
        self.play(FadeOut(resistor_law), run_time=0.3)
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(ohms_law[0][:2], circuit_law[0][:2]),
                    ReplacementTransform(ohms_law[0][3], circuit_law[0][2]),
                    ReplacementTransform(ohms_law[0][4:6], circuit_law[0][3:5]),
                    ReplacementTransform(ohms_law[0][7], circuit_law[0][5]),
                    FadeOut(ohms_law[0][2], ohms_law[0][6]),
                    impedance.animate.scale(0.9).move_to(
                        LEFT * fw(self, 0.28) + DOWN * fh(self, 0.32)
                    ),
                ),
                FadeIn(
                    circuit,
                    input_label,
                    output_label,
                    r1_label,
                    r2_label,
                    c1_label,
                    c2_label,
                ),
                lag_ratio=0.6,
                run_time=1.6,
            )
        )
        self.remove(ohms_law, *ohms_law, *circuit_law[0])
        self.add(circuit_law)
        self.wait(0.5)

        self.next_section("Resistor impedances", skip_animations=skip_animations(False))
        r1_impedance = (
            MathTex(r"Z_{R_1}=R_1", color=GREEN).scale(0.65).next_to(r1, UP, 0.3)
        )
        r2_impedance = (
            MathTex(r"Z_{R_2}=R_2", color=GREEN).scale(0.65).next_to(r2, UP, 0.3)
        )
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        TransformFromCopy(old[0], new[0][1:3], introducer=True),
                        ReplacementTransform(old[0], new[0][4:]),
                        TransformFromCopy(impedance[0][0], new[0][0], introducer=True),
                        GrowFromCenter(new[0][3]),
                    )
                    for old, new in ((r1_label, r1_impedance), (r2_label, r2_impedance))
                ],
                lag_ratio=0.25,
                run_time=1.2,
            )
        )
        self.remove(
            r1_label, *r1_label, r2_label, *r2_label, *r1_impedance[0], *r2_impedance[0]
        )
        self.add(r1_impedance, r2_impedance)
        self.wait(0.4)

        c1_impedance = (
            MathTex(r"Z_{C_1}=\frac{1}{j\omega C_1}", color=GREEN)
            .scale(0.7)
            .next_to(c1, UP, 0.25)
        )
        c2_impedance = (
            MathTex(r"Z_{C_2}=\frac{1}{j\omega C_2}", color=GREEN)
            .scale(0.7)
            .next_to(c2, LEFT, 0.3)
        )
        for index, (old, new) in enumerate(
            ((c1_label, c1_impedance), (c2_label, c2_impedance)), start=1
        ):
            self.next_section(
                f"Capacitor C{index} impedance", skip_animations=skip_animations(False)
            )
            arc = PI / 2 if index == 1 else 0
            self.play(
                TransformFromCopy(old[0], new[0][1:3]),
                ReplacementTransform(old[0], new[0][7:9]),
                TransformFromCopy(impedance[0][0], new[0][0], path_arc=arc),
                TransformFromCopy(impedance[0][1], new[0][3], path_arc=arc),
                TransformFromCopy(impedance[0][2:5], new[0][4:7], path_arc=arc),
                TransformFromCopy(impedance[0][6], new[0][9], path_arc=arc),
                run_time=1.1,
            )
            self.remove(old, *old, *new[0])
            self.add(new)
            self.wait(0.4)
        self.play(FadeOut(impedance), run_time=0.5)

        self.wait(0.5)

        nb = Text("complex_numbers.py", font=FONT).move_to(
            self.camera.frame.get_bottom()
        )

        self.play(self.camera.frame.animate.scale(1.1).shift(DOWN * 0.5), Write(nb))

        self.wait(0.5)

        self.next_section("Inductor impedance", skip_animations=skip_animations(False))
        comparison_center = self.camera.frame.get_center() + UP * fh(self)
        comparison_cap = (
            Cap(include_ports=True)
            .rotate(-PI / 2)
            .scale(1.5)
            .move_to(comparison_center + LEFT * fw(self, 0.2) + UP * fh(self, 0.1))
        )
        comparison_ind = (
            Inductor(include_ports=True, **wire_style)
            .rotate(-PI / 2)
            .scale(1.5)
            .move_to(comparison_center + RIGHT * fw(self, 0.2) + UP * fh(self, 0.1))
        )
        comparison_c_label = MathTex("C").next_to(comparison_cap, UP, MED_LARGE_BUFF)
        comparison_l_label = MathTex("L").next_to(comparison_ind, UP, MED_LARGE_BUFF)
        comparison_c_impedance = (
            MathTex(r"Z_C=\frac{1}{j\omega C}", color=GREEN)
            .scale(1.2)
            .next_to(comparison_cap, DOWN, LARGE_BUFF)
        )
        comparison_l_impedance = (
            MathTex(r"Z_L=j\omega L", color=GREEN)
            .scale(1.2)
            .next_to(comparison_ind, DOWN, LARGE_BUFF)
            .match_y(comparison_c_impedance)
        )
        self.play(self.camera.frame.animate.shift(UP * fh(self)))
        self.play(
            LaggedStart(
                FadeIn(comparison_cap, comparison_c_label, comparison_c_impedance),
                AnimationGroup(Create(comparison_ind), PopIn(comparison_l_label)),
                lag_ratio=0.35,
                run_time=1.4,
            )
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                AnimationGroup(
                    PopIn(comparison_l_impedance[0][0]),
                    TransformFromCopy(
                        comparison_l_label[0][0],
                        comparison_l_impedance[0][1],
                        introducer=True,
                    ),
                    GrowFromCenter(comparison_l_impedance[0][2]),
                ),
                LaggedStart(
                    PopIn(comparison_l_impedance[0][3]),
                    PopIn(comparison_l_impedance[0][4]),
                    TransformFromCopy(
                        comparison_l_label[0][0],
                        comparison_l_impedance[0][5],
                        introducer=True,
                    ),
                    lag_ratio=0.2,
                ),
                lag_ratio=0.4,
                run_time=1.6,
            )
        )
        self.remove(*comparison_l_impedance[0])
        self.add(comparison_l_impedance)
        self.wait(0.8)

        self.next_section(
            "Add an inductor to the circuit", skip_animations=skip_animations(False)
        )
        old_input = input_port.get_center()
        extended_input = input_port.copy().shift(LEFT * 3.2 * unit)
        series_ind = (
            Inductor(
                length=1.25 * unit,
                radius=0.2 * unit,
                include_ports=True,
                **wire_style,
            )
            .rotate(-PI / 2)
            .move_to((old_input + extended_input.get_center()) / 2)
        )
        series_ind.shift(UP * (old_input[1] - series_ind.p1[1]))
        inductor_wires = VGroup(
            Line(extended_input.get_right(), series_ind.p1, **wire_style),
            Line(series_ind.p2, wires[0].get_start(), **wire_style),
        )
        l1_impedance = (
            MathTex(r"Z_{L_1}=j\omega L_1", color=GREEN)
            .scale(0.7)
            .next_to(series_ind, UP, 0.35)
        )
        extended_input_label = input_label.copy().next_to(
            extended_input, DOWN, MED_LARGE_BUFF
        )
        circuit_view = Group(
            circuit,
            input_label,
            output_label,
            r1_impedance,
            r2_impedance,
            c1_impedance,
            c2_impedance,
            circuit_law,
        )
        self.remove(*circuit_view)
        self.add(circuit_view)
        input_port.become(extended_input)
        input_label.become(extended_input_label)
        extended_circuit = Group(
            circuit_view,
            series_ind,
            l1_impedance,
        )
        extended_frame_height = 1.12 * max(
            extended_circuit.height,
            extended_circuit.width * self.camera.frame.height / self.camera.frame.width,
        )
        input_label.set_opacity(0)
        output_label.set_opacity(0)
        self.play(
            FadeOut(
                comparison_cap,
                comparison_c_label,
                comparison_c_impedance,
                comparison_l_label,
                nb,
            ),
            self.camera.frame.animate.scale_to_fit_height(
                extended_frame_height
            ).move_to(extended_circuit),
            Transform(comparison_ind, series_ind),
            ReplacementTransform(comparison_l_impedance[0][:2], l1_impedance[0][:2]),
            ReplacementTransform(comparison_l_impedance[0][2:], l1_impedance[0][3:7]),
            run_time=1.8,
        )
        self.play(
            Create(inductor_wires),
            input_label.animate.set_opacity(1),
            output_label.animate.set_opacity(1),
            GrowFromCenter(l1_impedance[0][2]),
            GrowFromCenter(l1_impedance[0][7]),
            run_time=0.7,
        )
        self.remove(comparison_l_impedance, *comparison_l_impedance, *l1_impedance[0])
        self.add(l1_impedance)

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))


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


class Wrapup(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        cn = Text("Complex Numbers", font=FONT)

        spring_label = Text(
            "Box on a spring", font=FONT, font_size=34, color=ORANGE
        ).rotate(-6 * DEGREES)
        wall = VGroup(
            Line([-1.6, -0.45, 0], [-1.6, 0.45, 0]),
            *[
                Line([-1.6, y, 0], [-1.78, y - 0.14, 0])
                for y in np.linspace(-0.3, 0.45, 5)
            ],
        )
        coil = ParametricFunction(
            lambda t: [-1.4 + 1.8 * t, 0.14 * np.sin(12 * PI * t), 0],
            t_range=[0, 1, 0.005],
        )
        box = RoundedRectangle(
            width=0.8,
            height=0.7,
            corner_radius=0.06,
            fill_opacity=0.15,
        ).move_to([1, 0, 0])
        spring_lead = Line([0.4, 0, 0], box.get_left())
        spring_diagram = VGroup(
            wall,
            Line([-1.6, 0, 0], [-1.4, 0, 0]),
            coil,
            spring_lead,
            box,
            Line([-1.6, -0.4, 0], [1.6, -0.4, 0]),
        ).set_stroke(width=3)
        spring_diagram.scale_to_fit_width(fw(self, 0.22))
        spring = (
            VGroup(spring_diagram, spring_label)
            .arrange(DOWN, buff=fh(self, 0.025))
            .set_color(ORANGE)
        )
        control_label = Text(
            "Control theory", font=FONT, font_size=34, color=BLUE
        ).rotate(7 * DEGREES)
        summer = Circle(radius=0.14).move_to([-1.2, 0, 0])
        controller = RoundedRectangle(
            width=0.6, height=0.5, corner_radius=0.06, fill_opacity=0.12
        ).move_to([-0.4, 0, 0])
        plant = controller.copy().move_to([0.55, 0, 0])
        control_diagram = VGroup(
            summer,
            controller,
            plant,
            *[
                Line(start, end, tip_corner_radius=0.02).add_tip(
                    tip_length=0.1, tip_width=0.085
                )
                for start, end in [
                    ([-1.9, 0, 0], summer.get_left()),
                    (summer.get_right(), controller.get_left()),
                    (controller.get_right(), plant.get_left()),
                    (plant.get_right(), [1.55, 0, 0]),
                    ([-1.2, -0.6, 0], summer.get_bottom()),
                ]
            ],
            VMobject().set_points_as_corners(
                [[1.2, 0, 0], [1.2, -0.6, 0], [-1.2, -0.6, 0]]
            ),
            Dot([1.2, 0, 0], radius=0.035),
            Line([-1.25, 0, 0], [-1.15, 0, 0]),
            Line([-1.2, -0.05, 0], [-1.2, 0.05, 0]),
            Line([-1.45, -0.24, 0], [-1.33, -0.24, 0]),
        ).set_stroke(width=3)
        control_diagram.scale_to_fit_width(fw(self, 0.22))
        control = (
            VGroup(control_diagram, control_label)
            .arrange(DOWN, buff=fh(self, 0.025))
            .set_color(BLUE)
        )

        quantum_label = Text(
            "Quantum mechanics", font=FONT, font_size=34, color=YELLOW
        ).rotate(-4 * DEGREES)
        qubit = VGroup(
            Circle(radius=0.5, fill_opacity=0.05),
            Ellipse(width=1, height=0.28).set_stroke(opacity=0.6),
            Ellipse(width=0.28, height=1).set_stroke(opacity=0.4),
            DashedLine(DOWN * 0.65, UP * 0.65, dash_length=0.05).set_stroke(
                opacity=0.5
            ),
            Arrow(ORIGIN, [0.3, 0.4, 0], buff=0, tip_length=0.12, tip_width=0.1),
            Dot(ORIGIN, radius=0.035),
        ).set_stroke(width=3)
        qubit.scale_to_fit_height(fh(self, 0.13))
        quantum = (
            VGroup(qubit, quantum_label)
            .arrange(DOWN, buff=fh(self, 0.025))
            .set_color(YELLOW)
        )
        waves = (
            Paragraph(
                "Electromagnetic",
                "waves",
                alignment="center",
                font=FONT,
                font_size=34,
                color=GREEN,
            )
            .set_color(GREEN)
            .rotate(6 * DEGREES)
        )
        transmission = (
            Paragraph(
                "Transmission",
                "lines",
                alignment="center",
                font=FONT,
                font_size=34,
                color=PURPLE,
            )
            .set_color(PURPLE)
            .rotate(-7 * DEGREES)
        )
        spring.move_to(LEFT * fw(self, 0.25) + UP * fh(self, 0.28))
        control.move_to(RIGHT * fw(self, 0.25) + UP * fh(self, 0.28))
        quantum.move_to(DOWN * fh(self, 0.34))
        waves.move_to(LEFT * fw(self, 0.3) + DOWN * fh(self, 0.17))
        transmission.move_to(RIGHT * fw(self, 0.3) + DOWN * fh(self, 0.17))

        spring_time = VT(0)
        box_center = box.get_center().copy()
        coil_points = coil.points.copy()
        coil_start = coil.get_start().copy()
        coil_length = coil.get_end()[0] - coil_start[0]
        lead_points = spring_lead.points.copy()

        def oscillate_spring(mob):
            offset = fw(self, 0.008) * np.sin(TAU * ~spring_time / 1.8)
            box.move_to(box_center + RIGHT * offset)
            points = coil_points.copy()
            points[:, 0] += (points[:, 0] - coil_start[0]) / coil_length * offset
            coil.set_points(points)
            spring_lead.set_points(lead_points + RIGHT * offset)

        topics = VGroup(spring, control, quantum, waves, transmission)
        connections = VGroup(
            *[
                SquigglyLine(
                    start,
                    end,
                    amplitude=fh(self, 0.025),
                    wavelength=fh(self, 0.18),
                    seed=index + 3,
                    color=topic.get_color(),
                    stroke_width=3,
                    tip_corner_radius=0.02,
                ).add_tip(
                    tip_length=fh(self, 0.022),
                    tip_width=fh(self, 0.018),
                )
                for index, (topic, start, end) in enumerate(
                    [
                        (
                            spring,
                            spring.get_bottom() + DOWN * fh(self, 0.025),
                            cn.get_top()
                            + LEFT * cn.width * 0.33
                            + UP * fh(self, 0.035),
                        ),
                        (
                            control,
                            control.get_bottom() + DOWN * fh(self, 0.025),
                            cn.get_top()
                            + RIGHT * cn.width * 0.33
                            + UP * fh(self, 0.035),
                        ),
                        (
                            quantum,
                            quantum.get_top() + UP * fh(self, 0.025),
                            cn.get_bottom() + DOWN * fh(self, 0.035),
                        ),
                        (
                            waves,
                            waves.get_top() + UP * fh(self, 0.025),
                            cn.get_left() + LEFT * fw(self, 0.018),
                        ),
                        (
                            transmission,
                            transmission.get_top() + UP * fh(self, 0.025),
                            cn.get_right() + RIGHT * fw(self, 0.018),
                        ),
                    ]
                )
            ]
        )

        self.next_section("Complex numbers", skip_animations=skip_animations(False))
        self.play(LaggedStart(*[PopIn(m) for m in cn], lag_ratio=0.1))
        self.wait(0.5)

        self.next_section("Other applications", skip_animations=skip_animations(False))
        self.play(
            LaggedStart(
                *[
                    LaggedStart(PopIn(topic), GrowArrow(connection), lag_ratio=0.35)
                    for topic, connection in zip(topics[:3], connections[:3])
                ],
                lag_ratio=0.45,
            ),
            run_time=3,
        )
        spring_time.add_updater(lambda mob, dt: mob.increment_value(dt))
        self.add(spring_time)
        spring_diagram.add_updater(oscillate_spring)
        self.wait(2)

        self.next_section(
            "Waves and transmission lines", skip_animations=skip_animations(False)
        )
        self.play(
            LaggedStart(
                *[
                    LaggedStart(PopIn(topic), GrowArrow(connection), lag_ratio=0.35)
                    for topic, connection in zip(topics[3:], connections[3:])
                ],
                lag_ratio=0.5,
            ),
            run_time=2.5,
        )
        self.wait(2)
