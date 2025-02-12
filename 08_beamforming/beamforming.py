# beamforming.py

import warnings
import sys
from manim import *
from MF_Tools import VT
import numpy as np
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

from props.style import BACKGROUND_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def get_transform_func(from_var, func=TransformFromCopy):
    def transform_func(m, **kwargs):
        return func(from_var, m, **kwargs)

    return transform_func


class SincTest(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            tips=False,
        )

        t = np.linspace(0.001, 1, 1000)

        def plot_func():
            sinc = np.sin(2 * PI * 10 * t) / ((2 * PI * 10 * t) ** ~exp)
            sinc /= sinc.max()
            f = interp1d(t, sinc)
            return ax.plot(f, x_range=[0.001, 1, 0.001])

        exp = VT(0)
        plot = always_redraw(plot_func)
        self.add(ax, plot)

        self.wait(0.5)

        self.play(exp @ 1, run_time=4)

        self.wait(2)
