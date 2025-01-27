import numpy as np
from numpy.fft import fft, fftshift
from scipy import signal

stop_time = 4
fs = 1000
N = fs * stop_time

window = signal.windows.blackman(N)
x_n_windowed = x_n * window

fft_len = N
X_k = fftshift(fft(x_n_windowed, fft_len))
X_k /= N / 2
X_k = np.abs(X_k)
X_k = 10 * np.log10(X_k)

freq = np.linspace(-fs / 2, fs / 2, fft_len)
