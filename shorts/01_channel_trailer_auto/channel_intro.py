# shorts/channel_intro.py

import sys
from manim import *
from scipy import signal

sys.path.insert(0, "../..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR, IF_COLOR

config.background_color = BACKGROUND_COLOR

config.pixel_height = 1920
config.pixel_width = 1080


class Intro(Scene):
    def construct(self):
        car1 = (
            SVGMobject("../../props/static/car.svg").set_fill(WHITE)
            # .scale_to_fit_width(config.frame_width * 3)
        )
        gas_station = (
            SVGMobject("../../props/static/Icon 10.svg")
            .set_fill(WHITE)
            .set_color(WHITE)
            .scale(2)
            # .scale_to_fit_width(config.frame_width * 3)
        )
        car2 = (
            (
                SVGMobject("../../props/static/car.svg").set_fill(WHITE)
                # .scale_to_fit_width(config.frame_width * 3)
            )
            .next_to(gas_station, LEFT, MED_LARGE_BUFF, aligned_edge=DOWN)
            .flip()
        )

        obst_group = Group(car2, gas_station).shift(RIGHT * 15)
        scene_group = Group(car1, obst_group)

        self.wait(0.5)

        self.play(car1.shift(LEFT * 10).animate.shift(RIGHT * 10))

        self.wait(0.5)

        self.play(
            scene_group.animate.arrange(RIGHT, LARGE_BUFF * 1.5, aligned_edge=DOWN)
        )

        self.wait(0.5)

        beam_pencil = Line(
            car1.get_right() + [0.1, 0, 0],
            car2.get_left() + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        beam_u = Line(
            car1.get_right() + [0.1, 0, 0],
            gas_station.get_corner(UL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )
        beam_l = Line(
            car1.get_right() + [0.1, 0, 0],
            car2.get_corner(DL) + [-0.1, 0, 0],
            color=TX_COLOR,
        )

        self.play(Create(beam_pencil))

        self.wait(0.5)

        self.play(
            TransformFromCopy(beam_pencil, beam_l),
            ReplacementTransform(beam_pencil, beam_u),
        )

        # self.wait(0.5)

        # self.play(car2.animate.next_to(car1, SMALL_BUFF, aligned_edge=DOWN))

        self.wait(0.5)

        self.play(
            car1.animate.shift(LEFT * 15),
            obst_group.animate.shift(RIGHT * 15),
            Uncreate(beam_l),
            Uncreate(beam_u),
        )

        self.wait(0.5)

        carrier_freq = 10
        duration = 1
        fs = 1000

        x_len = config["frame_width"] * 0.8
        y_len = config["frame_height"] * 0.4

        ax = Axes(
            x_range=[-0.1, duration, duration / 4],
            y_range=[-2, 30, 5],
            tips=False,
            axis_config={"include_numbers": False},
            x_length=x_len,
            y_length=y_len,
        )
        ax_labels = ax.get_axis_labels(MathTex("t"), MathTex("f"))

        sawtooth_carrier_freq = 14
        sawtooth_modulation_index = 12
        sawtooth_modulating_signal_f = 2
        sawtooth_modulating_signal = (
            lambda t: sawtooth_modulation_index
            * signal.sawtooth(2 * PI * sawtooth_modulating_signal_f * t)
            + sawtooth_carrier_freq
        )
        sawtooth_modulating_cumsum = (
            lambda t: carrier_freq
            + np.sum(sawtooth_modulating_signal(np.arange(0, t, 1 / fs))) / fs
        )

        tx = ax.plot(
            sawtooth_modulating_signal,
            x_range=[0, duration, 1 / fs],
            use_smoothing=False,
            color=TX_COLOR,
        )
        rx = ax.plot(
            sawtooth_modulating_signal,
            x_range=[0, duration, 1 / fs],
            use_smoothing=False,
            color=RX_COLOR,
        ).shift(RIGHT)

        plot_group = Group(ax, ax_labels, tx, rx)

        self.play(plot_group.shift(UP * 20).animate.move_to(ORIGIN))

        self.wait(0.5)

        cfar = (
            ImageMobject("../../03_cfar/media/images/cfar/Designer_Sectioned.png")
            .scale_to_fit_width(plot_group.width)
            .shift(DOWN * 20)
        )

        self.play(Group(plot_group, cfar).animate.arrange(DOWN, LARGE_BUFF))

        self.wait(0.5)

        range_doppler = (
            ImageMobject("./static/range_doppler_3d.png")
            .scale_to_fit_width(cfar.width * 0.8)
            .shift(UP * 20)
        )

        self.play(
            Group(range_doppler, plot_group, cfar).animate.arrange(DOWN, LARGE_BUFF)
        )

        self.wait(0.5)

        self.play(
            range_doppler.animate.shift(LEFT * 20),
            plot_group.animate.shift(RIGHT * 20),
            cfar.animate.shift(LEFT * 20),
        )

        self.wait(0.5)

        thumbnail1 = (
            ImageMobject("../../01_fmcw/media/images/fmcw/thumbnails/comparison.png")
            .scale_to_fit_width(cfar.width)
            .shift(DOWN * 20)
        )
        thumbnail2 = (
            ImageMobject(
                "../../02_fmcw_implementation/media/images/fmcw_implementation/Thumbnail_Option_1.png"
            )
            .scale_to_fit_width(cfar.width)
            .shift(DOWN * 20)
        )
        thumbnail3 = (
            ImageMobject("../../03_cfar/media/images/cfar/thumbnails/Thumbnail_1.png")
            .scale_to_fit_width(cfar.width)
            .shift(DOWN * 20)
        )

        thumbnail_group = Group(
            thumbnail1.copy(), thumbnail2.copy(), thumbnail3.copy()
        ).arrange(DOWN, LARGE_BUFF)

        self.play(
            LaggedStart(
                thumbnail1.animate.move_to(thumbnail_group[0]),
                thumbnail2.animate.move_to(thumbnail_group[1]),
                thumbnail3.animate.move_to(thumbnail_group[2]),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        cfar_nb = ImageMobject("./static/cfar_nb.png").scale_to_fit_width(cfar.width)
        rd_nb = ImageMobject("./static/range_doppler_nb.png").scale_to_fit_width(
            cfar.width
        )

        self.play(
            Group(thumbnail1, thumbnail2, thumbnail3).animate.shift(LEFT * 20),
            rd_nb.shift(RIGHT * 20).animate.shift(LEFT * 20),
        )

        self.wait(0.5)

        self.play(
            rd_nb.animate.shift(LEFT * 20),
            cfar_nb.shift(RIGHT * 20).animate.shift(LEFT * 20),
        )

        self.wait(0.5)

        profile_pic = ImageMobject(
            "../../../../media/rf_channel_assets/profile_pictures/Raccoon_Coding_Retro_Channel_Colors.jpg"
        ).scale_to_fit_width(cfar.width * 0.3)
        mb = Tex("Marshall Bruner", font_size=DEFAULT_FONT_SIZE * 2)

        self.play(
            LaggedStart(
                cfar_nb.animate.shift(LEFT * 20),
                GrowFromCenter(profile_pic),
                lag_ratio=0.4,
            )
        )

        self.wait(0.5)

        self.play(
            Group(profile_pic, mb.shift(DOWN * 20)).animate.arrange(DOWN, LARGE_BUFF)
        )

        self.wait(0.5)

        self.play(FadeOut(*self.mobjects))

        self.wait(2)
