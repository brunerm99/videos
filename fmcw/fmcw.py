# fmcw.py

from manim import *


class Title(Scene):
    def construct(self):
        title = Tex("F", "M", "C", "W", " Radar", font_size=DEFAULT_FONT_SIZE * 2)
        f = Text("Frequency", font_size=DEFAULT_FONT_SIZE * 0.8)
        m = Text("modulated", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            f, direction=RIGHT
        )
        c = Text("continuous", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(
            m, direction=RIGHT
        )
        w = Text("wave", font_size=DEFAULT_FONT_SIZE * 0.8).next_to(c, direction=RIGHT)

        expanded = Tex(
            "Frequency",
            "-",
            "modulated ",
            "continuous ",
            "wave",
            font_size=DEFAULT_FONT_SIZE * 0.8,
        ).shift(DOWN)

        f = expanded[0]
        dash = expanded[1]
        m = expanded[2]
        c = expanded[3]
        w = expanded[4]

        self.play(Write(title))
        # self.wait(1)
        self.play(title.animate.scale(0.6).shift(UP * 2))

        # self.play(Write(f), Write(m), Write(c), Write(w))
        self.play(Write(f), Write(dash), Write(m), Write(c), Write(w))

        f_title = title[0]
        m_title = title[1]
        c_title = title[2]
        w_title = title[3]

        f_p1 = f_title.get_bottom() - [0, 0.1, 0]
        f_p2 = f.get_top()
        f_bezier = CubicBezier(f_p1, f_p1 - [0, 1, 0], f_p2 + [0, 1, 0], f_p2)
        m_p1 = m_title.get_bottom() - [0, 0.1, 0]
        m_p2 = m.get_top()
        m_bezier = CubicBezier(m_p1, m_p1 - [0, 1, 0], m_p2 + [0, 1, 0], m_p2)
        c_p1 = c_title.get_bottom() - [0, 0.1, 0]
        c_p2 = c.get_top()
        c_bezier = CubicBezier(c_p1, c_p1 - [0, 1, 0], c_p2 + [0, 1, 0], c_p2)
        w_p1 = w_title.get_bottom() - [0, 0.1, 0]
        w_p2 = w.get_top()
        w_bezier = CubicBezier(w_p1, w_p1 - [0, 1, 0], w_p2 + [0, 1, 0], w_p2)

        self.play(
            AnimationGroup(
                f.animate.set_color(BLUE),
                m.animate.set_color(RED),
                c.animate.set_color(YELLOW),
                w.animate.set_color(GREEN),
                f_title.animate.set_color(BLUE),
                m_title.animate.set_color(RED),
                c_title.animate.set_color(YELLOW),
                w_title.animate.set_color(GREEN),
                Create(f_bezier),
                Create(m_bezier),
                Create(c_bezier),
                Create(w_bezier),
            )
        )

        self.wait(10)
