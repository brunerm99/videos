# short.py

import sys
import warnings

import numpy as np
from numpy.fft import fft, fftshift
from manim import *
from scipy.interpolate import interp1d, bisplrep, bisplev
from scipy.constants import c
from scipy import signal
from MF_Tools import VT, TransformByGlyphMap


warnings.filterwarnings("ignore")
sys.path.insert(0, "../../")

from props import VideoMobject
from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR
config.pixel_height = 1920
config.pixel_width = 1080
config.frame_height = 14
config.frame_width = 9

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class Short(Scene):
    def construct(self):
        self.next_section(skip_animations=skip_animations(True))
        video = (
            VideoMobject(
                "./static/af_2d_anim_rotating_w_steering_2.mp4", speed=1, loop=False
            )
            .scale_to_fit_width(config.frame_width * 1.1)
            .next_to([0, -config.frame_height / 2, 0], DOWN)
            .set_z_index(-2)
        )
        self.add(video)

        ap = (
            Tex(r"Antenna\\Pattern", color=ORANGE)
            .scale_to_fit_width(config.frame_width * 0.4)
            .to_edge(UP, LARGE_BUFF)
        )

        ap_bez = CubicBezier(
            ap.get_corner(DR) + [0.1, -0.1, 0],
            ap.get_corner(DR) + [1, -1, 0],
            video.copy().set_y(0).get_center() + [2, 1.5, 0] + [1, 1, 0],
            video.copy().set_y(0).get_center() + [2, 1.5, 0],
            color=ORANGE,
        )
        ap_tip = (
            Triangle(color=ORANGE, fill_color=ORANGE, fill_opacity=1)
            .scale(0.2)
            .rotate(15 * DEGREES)
            .move_to(ap_bez.get_end())
        )

        self.play(
            LaggedStart(
                video.animate.set_y(0),
                Write(ap),
                Create(ap_bez),
                GrowFromCenter(ap_tip),
                lag_ratio=0.3,
            )
        )

        self.wait(7)

        nls = (
            Group(
                *[
                    NumberLine(x_range=[-1, 1, 0.25], length=config.frame_height * 0.3)
                    .rotate(PI / 2)
                    .set_z_index(-1)
                    for _ in range(4)
                ]
            )
            .arrange(RIGHT, LARGE_BUFF)
            .to_edge(DOWN, MED_LARGE_BUFF)
        )

        def get_updater(nl, vt):
            def updater(m):
                m.move_to(nl.n2p(~vt))

            return updater

        trackers = [VT(0) for _ in nls]
        dots = Group()
        phis = Group()
        for idx, (vt, nl) in enumerate(zip(trackers, nls)):
            dot = Dot().scale(1.5).set_color(ORANGE)
            updater = get_updater(nl, vt)
            dot.add_updater(updater)
            dots.add(dot)
            phis.add(
                MathTex(f"\\phi_{{{idx}}}", font_size=DEFAULT_FONT_SIZE * 1.5).next_to(
                    nl, DOWN
                )
            )

        self.play(
            *[Create(m) for m in [*nls, *dots]],
            LaggedStart(*[GrowFromCenter(m) for m in phis], lag_ratio=0.2),
            FadeOut(ap_bez, ap_tip),
        )

        self.wait(0.5)

        self.play(
            trackers[1] @ -0.25, trackers[2] @ -0.5, trackers[3] @ -0.75, run_time=15
        )

        self.next_section(skip_animations=skip_animations(False))
        self.wait(0.5)

        self.play(*[m.animate.rotate(PI / 2) for m in nls])
        self.play(
            nls.animate.arrange(DOWN, LARGE_BUFF * 0.8).to_edge(DOWN, LARGE_BUFF),
            *[vt @ 0 for vt in trackers],
            *[
                phi.animate.next_to(nl, RIGHT)
                for phi, nl in zip(
                    phis,
                    nls.copy()
                    .arrange(DOWN, LARGE_BUFF * 0.8)
                    .to_edge(DOWN, LARGE_BUFF),
                )
            ],
        )
        self.play(
            trackers[1] @ 0.25, trackers[2] @ 0.5, trackers[3] @ 0.75, run_time=15
        )

        self.wait(1)

        self.play(FadeOut(phis, dots, nls))

        self.wait(15)

        thumbnail_img = ImageMobject(
            "../../05_phased_array/media/images/phased_array/thumbnails/Thumbnail1.png"
        ).scale_to_fit_width(config.frame_width * 0.8)
        thumbnail_box = SurroundingRectangle(thumbnail_img, buff=0)
        thumbnail = Group(thumbnail_box, thumbnail_img).to_edge(DOWN, LARGE_BUFF)

        self.play(
            LaggedStart(
                FadeOut(*self.mobjects),
                thumbnail.shift(DOWN * 10).animate.shift(UP * 10),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        profile_pic = (
            ImageMobject(
                "../../../../media/rf_channel_assets/profile_pictures/Raccoon_Coding_Retro_Channel_Colors.jpg"
            )
            .scale_to_fit_width(config.frame_width * 0.4)
            .next_to(thumbnail, UP, LARGE_BUFF * 2)
        )
        mb = Tex("Marshall Bruner", font_size=DEFAULT_FONT_SIZE * 2).next_to(
            profile_pic, DOWN
        )

        self.play(
            LaggedStart(
                GrowFromCenter(profile_pic),
                Write(mb),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)
