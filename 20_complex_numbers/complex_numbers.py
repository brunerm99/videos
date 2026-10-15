import numpy as np
from manimlib import *

BACKGROUND_COLOR = "#183340"
GROUND_COLOR = GREY_B
SPRING_COLOR = YELLOW
BOX_COLOR = BLUE


def get_spring(start, end, coils=9, radius=0.30):
    """Draw rounded, overlapping turns like a side view of a coil spring."""
    start = np.array(start)
    end = np.array(end)
    lead_length = min(0.35, 0.12 * (end[0] - start[0]))
    coil_start = start + lead_length * RIGHT
    coil_end = end - lead_length * RIGHT

    pitch = (coil_end[0] - coil_start[0]) / coils
    angles = np.linspace(0, TAU * coils, 32 * coils + 1)
    points = [
        coil_start
        + (pitch * angle / TAU + 0.30 * pitch * (np.cos(angle) - 1)) * RIGHT
        + radius * np.sin(angle) * UP
        for angle in angles
    ]

    spring = VMobject()
    spring.set_points_smoothly([start, coil_start, *points[1:-1], coil_end, end])
    spring.set_stroke(SPRING_COLOR, width=5)
    return spring


class SpringOscillation(InteractiveScene):
    default_camera_config = {"background_color": BACKGROUND_COLOR, "fps": 60}

    def construct(self):
        ground_y = -1.7
        wall_x = -5.4
        equilibrium_x = 0.8
        amplitude = 1.5
        oscillation_time = 6
        cycles = 3

        ground = Line(
            [wall_x, ground_y, 0],
            [6.2, ground_y, 0],
            stroke_color=GROUND_COLOR,
            stroke_width=6,
        )
        wall = Line(
            [wall_x, ground_y, 0],
            [wall_x, 1.8, 0],
            stroke_color=GROUND_COLOR,
            stroke_width=6,
        )

        wall_hatching = VGroup(
            *[
                Line(
                    [wall_x - 0.42, y + 0.22, 0],
                    [wall_x, y, 0],
                    stroke_color=GROUND_COLOR,
                    stroke_width=3,
                )
                for y in np.arange(ground_y + 0.15, 1.8, 0.38)
            ]
        )
        ground_hatching = VGroup(
            *[
                Line(
                    [x - 0.2, ground_y - 0.35, 0],
                    [x + 0.2, ground_y, 0],
                    stroke_color=GROUND_COLOR,
                    stroke_width=3,
                )
                for x in np.arange(wall_x + 0.25, 6.2, 0.55)
            ]
        )

        box = Square(
            side_length=1.35,
            fill_color=BOX_COLOR,
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=4,
        )
        box.set_x(equilibrium_x)
        box.set_y(ground_y + box.get_height() / 2)

        spring_anchor = np.array([wall_x, box.get_y(), 0])
        spring = always_redraw(lambda: get_spring(spring_anchor, box.get_left()))

        self.add(ground_hatching, wall_hatching, ground, wall, spring, box)
        self.wait(0.4)

        # Pull the mass to the right, then release it from rest.
        self.play(
            box.animate.set_x(equilibrium_x + amplitude),
            run_time=1.2,
            rate_func=smooth,
        )

        def oscillate(mob, alpha):
            phase = TAU * cycles * alpha
            mob.set_x(equilibrium_x + amplitude * np.cos(phase))

        self.play(
            UpdateFromAlphaFunc(box, oscillate),
            run_time=oscillation_time,
            rate_func=linear,
        )


if __name__ == "__main__":
    from manimlib.__main__ import main

    main()
