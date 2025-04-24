# aliasing.py


from manim import *
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.interpolate import interp1d
import sys
from MF_Tools import VT, TransformByGlyphMap
from numpy.fft import fft, fftshift, fft2

import matplotlib

matplotlib.use("Agg")
from matplotlib.pyplot import get_cmap

sys.path.insert(0, "..")

from props import WeatherRadarTower, VideoMobject
from props.style import BACKGROUND_COLOR, TX_COLOR, RX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = True


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


class ChangingFS(Scene):
    def construct(self):
        fft_len = 1024
        freq = np.linspace(-PI, PI, fft_len)
        stop_time = 3

        fs = VT(10)
        f1 = VT(2)
        f2 = VT(4)
        abs_vt = VT(0)

        ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.7,
        )

        def plot_X_k():
            t = np.arange(0, stop_time, 1 / ~fs)
            freqs = [~f1, ~f2]
            x_n = np.sum([np.sin(2 * PI * f * t) for f in freqs], axis=0)

            X_k = ((1 - ~abs_vt) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                (~abs_vt) * fft(x_n, fft_len) / (t.size / 2)
            )
            f_X_k = interp1d(freq, np.real(fftshift(X_k)), fill_value="extrapolate")
            return ax.plot(f_X_k, x_range=[-PI, PI, PI / 200], color=ORANGE)

        plot = always_redraw(plot_X_k)

        self.add(ax, plot)

        self.play(f1 @ 8, run_time=20)

        self.wait(0.5)

        self.play(abs_vt @ 1, f1 @ 2, run_time=2)

        self.wait(0.5)

        self.play(f1 @ 8, run_time=20)

        self.wait(2)


class SimpleSignal(Scene):
    def construct(self):
        fft_len = 1024
        freq = np.linspace(-PI, PI, fft_len)
        stop_time = 3
        fs = 10

        f1 = VT(4)
        f2 = VT(8)
        abs_vt = VT(0)

        ax = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[-1, 1, 0.5],
            tips=False,
            x_length=config.frame_width * 0.8,
            y_length=config.frame_height * 0.7,
        )

        def plot_X_k():
            t = np.arange(0, stop_time, 1 / fs)
            x_n = np.sin(2 * PI * 3 * t)
            freqs = [~f1, ~f2]
            x_n = np.sum([np.sin(2 * PI * f * t) for f in freqs], axis=0)

            X_k = ((1 - ~abs_vt) * np.abs(fft(x_n, fft_len)) / (t.size / 2)) + (
                (~abs_vt) * fft(x_n, fft_len) / (t.size / 2)
            )
            f_X_k = interp1d(freq, np.real(X_k), fill_value="extrapolate")
            return ax.plot(f_X_k, x_range=[-PI, PI, PI / 200], color=ORANGE)

        plot = always_redraw(plot_X_k)

        self.add(ax, plot)

        self.play(abs_vt @ 1, run_time=2)

        self.wait(2)
