import numpy as np
from numpy.fft import fft, fftshift
from scipy import signal

stop_time = 4
fs = 1000
N = fs * stop_time
