# props/helpers.py

import numpy as np
from numpy.fft import fft, fftshift
from scipy import signal
from manim import *


def get_plot_values(
    frequencies,
    power_norms,
    fs,
    stop_time,
    ports=["1", "2", "3", "noise"],
    y_min=None,
    noise_power_db=-20,
    noise_seed=0,
    f_max=None,
):
    N = int(fs * stop_time)
    fft_len = N * 4
    freq = np.linspace(-fs / 2, fs / 2, fft_len)

    t = np.linspace(0, stop_time, N)

    f_max = fs / 2 if f_max is None else f_max

    np.random.seed(int(noise_seed))
    noise = np.random.normal(loc=0, scale=10 ** (noise_power_db / 10), size=t.size)

    summed_signals = (
        sum(
            [
                np.sin(2 * PI * f * t) * (10 ** (power_norm / 10))
                for f, power_norm in zip(frequencies, power_norms)
            ]
        )
        + noise
    )

    blackman_window = signal.windows.blackman(N)
    summed_signals *= blackman_window

    X_k = fftshift(fft(summed_signals, fft_len))
    X_k /= N / 2
    X_k = np.abs(X_k)
    X_k_log = 10 * np.log10(X_k)

    indices = np.where((freq > 0) & (freq < f_max))
    x_values = freq[indices]
    y_values = X_k_log[indices]

    if y_min is not None:
        y_values[y_values < y_min] = y_min
        y_values -= y_min

    return dict(x_values=x_values, y_values=y_values)
