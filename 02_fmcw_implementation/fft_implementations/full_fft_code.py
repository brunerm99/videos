import numpy as np
from numpy.fft import fft, fftshift
from scipy import signal

stop_time = 4
fs = 1000
N = fs * stop_time

window = signal.windows.blackman(N)
x_n_windowed = x_n * window

fft_len = 1024 * 10

X_k = np.abs(fft(x_n_windowed, fft_len) / (N / 2))
X_k = 10 * np.log10(fftshift(X_k))

freq = np.linspace(-fs / 2, fs / 2, fft_len)
