# range_doppler_plot.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fft2, fftshift
from scipy.constants import c, pi
from scipy import signal
from matplotlib import animation

np.random.seed(0)


def compute_phase_diff(v):
    time_from_vel = 2 * (v * Tc) / c
    return 2 * pi * f * time_from_vel


def compute_f_beat(R):
    return (2 * R * bw) / (c * Tc)


def db_to_lin(x):
    return 10 ** (x / 10)


# Radar setup for doppler stuff
f = 77e9  # Hz
Tc = 40e-6  # chirp time - s
bw = 1.6e9  # bandwidth - Hz
chirp_rate = bw / Tc  # Hz/s

wavelength = c / f
M = 40  # number of chirps in coherent processing interval (CPI)

# Target
R = 20  # m
v = 10  # m/s
f_beat = compute_f_beat(R)
phase_diff = compute_phase_diff(v)
max_time = 20 / f_beat
N = 10000
Ts = max_time / N
fs = 1 / Ts

t = np.arange(0, max_time, 1 / fs)

window = signal.windows.blackman(N)

fft_len = N * 8
max_vel = wavelength / (4 * Tc)
vel_res = wavelength / (2 * M * Tc)
rmax = c * Tc * fs / (2 * bw)
n_ranges = np.linspace(-rmax / 2, rmax / 2, N)
ranges = np.linspace(-rmax / 2, rmax / 2, fft_len)


targets = [(20, 8, 0), (25, 10, 0)]
cpi = np.array(
    [
        (
            np.sum(
                [
                    np.sin(2 * pi * compute_f_beat(r) * t + m * compute_phase_diff(v))
                    * db_to_lin(p)
                    for r, v, p in targets
                ],
                axis=0,
            )
            + np.random.normal(0, 0.1, N)
        )
        * window
        for m in range(M)
    ]
)

range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)

extent = [-max_vel, max_vel, ranges.min(), ranges.max()]

fig, ax = plt.subplots(figsize=(8, 8))


target2_pos = np.linspace(22, 18, 100)

noise = np.random.normal(0, 0.1, N)
noise = np.random.normal(0, 0.1, cpi.shape)


def update(frame, p, ax):
    print(type(p))
    if frame < target2_pos.size:
        p[0].remove()
        targets = [(20, 8, 0), (target2_pos[frame], 10, 0)]
        cpi = (
            np.array(
                [
                    (
                        np.sum(
                            [
                                np.sin(
                                    2 * pi * compute_f_beat(r) * t
                                    + m * compute_phase_diff(v)
                                )
                                * db_to_lin(p)
                                for r, v, p in targets
                            ],
                            axis=0,
                        )
                    )
                    * window
                    for m in range(M)
                ]
            )
            + noise
        )

        range_doppler = fftshift(np.abs(fft2(cpi.T))) / (N / 2)
        p[0] = ax.imshow(
            10 * np.log10(range_doppler),
            aspect="auto",
            extent=extent,
            origin="lower",
            vmax=2,
            vmin=-25,
            cmap="coolwarm",
        )


range_doppler_plot = [
    ax.imshow(
        10 * np.log10(range_doppler),
        aspect="auto",
        extent=extent,
        origin="lower",
        vmax=2,
        vmin=-25,
        cmap="coolwarm",
    )
]
ax.set_ylim([0, 40])
ax.set_axis_off()
fig.patch.set_facecolor("#183340")
ax.set_facecolor("#183340")
plt.gca().invert_xaxis()

ani = animation.FuncAnimation(
    fig,
    update,
    frames=target2_pos.size + 30,
    interval=30,
    fargs=(range_doppler_plot, ax),
)
ani.save(
    "static/target_moving_constant_noise.mp4",
    bitrate=-1,
    dpi=300,
    progress_callback=lambda c, l: print(f"Saved: {c}/{l}"),
    savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
)
