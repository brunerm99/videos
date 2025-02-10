# af_2d_plot.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, pi
from matplotlib import cm
import matplotlib.animation as animation


plt.style.use("ggplot")

f_0 = 10e9  # center frequency for computing phase shifts (GHz)
wavelength_0 = c / f_0  # m
k_0 = 2 * pi / wavelength_0  # wave number
d_x = wavelength_0 / 2  # element spacing (m)

N = 9  # number of elements in horizontal dimension
M = 9  # number of elements in vertical dimension

n = np.arange(N)  # array for summing over n
m = np.arange(M)  # array for summing over m

d_x = wavelength_0 / 2  # x spacing
d_y = wavelength_0 / 2  # y spacing

steering_angle_theta = 0  # theta steering angle
steering_angle_phi = 0  # phi steering angle

u_0 = np.sin(steering_angle_theta * pi / 180)  # theta steering angle in sine space
v_0 = np.sin(steering_angle_phi * pi / 180)  # phi steering angle in sine space

npts = 1000
theta = np.linspace(-pi, pi, npts)
phi = np.linspace(-pi, pi, npts)

window_n = np.ones(N)
window_m = np.ones(M)

u2 = np.sin(theta) * np.cos(phi)
v2 = np.sin(theta) * np.sin(phi)

u2 = np.linspace(-1, 1, npts)
v2 = np.linspace(-1, 1, npts)

U, V = np.meshgrid(u2, v2)  # mesh grid of sine space


def compute_af_2d(weights_n, weights_m, d_x, d_y, k_0, u_0, v_0):
    AF_m = np.sum(
        weights_n[:, None, None]
        * np.exp(1j * n[:, None, None] * d_x * k_0 * (U - u_0)),
        axis=0,
    )
    AF_n = np.sum(
        weights_m[:, None, None]
        * np.exp(1j * m[:, None, None] * d_y * k_0 * (V - v_0)),
        axis=0,
    )

    AF = AF_m * AF_n / (M * N)
    return AF


AF = compute_af_2d(window_n, window_m, d_x, d_y, k_0, u_0, v_0)

z_min = -25
z_max = 0

AF_rect_log = 10 * np.log10(np.abs(AF))
AF_mask = np.where(AF_rect_log > z_min, AF_rect_log, np.nan)

x = AF_mask.shape[0] // 2
y = AF_mask.shape[1] // 2

frame = 1
max_frame = x


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")


def update(frame, p, ax):
    print(type(p))
    if frame < max_frame:
        p[0].remove()
        p[0] = ax.plot_surface(
            U[x - frame : x + frame, y - frame : y + frame],
            V[x - frame : x + frame, y - frame : y + frame],
            AF_mask[x - frame : x + frame, y - frame : y + frame],
            cmap=cm.coolwarm,
            antialiased=False,
        )
    ax.view_init(elev=30, azim=-50 + frame * 0.5, roll=0)


plot = [
    ax.plot_surface(
        U[x - frame : x + frame, y - frame : y + frame],
        V[x - frame : x + frame, y - frame : y + frame],
        AF_mask[x - frame : x + frame, y - frame : y + frame],
        cmap=cm.coolwarm,
        antialiased=False,
    ),
]
ax.set_zlim(z_min, z_max)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.view_init(elev=30, azim=-50, roll=0)
ax.set_box_aspect(None, zoom=0.85)
fig.patch.set_facecolor("#183340")
ax.set_facecolor("#183340")
ax.set_axis_off()


ani = animation.FuncAnimation(
    fig, update, frames=max_frame * 3, interval=30, fargs=(plot, ax)
)
ani.save(
    "static/af_2d_anim_rotating.mp4",
    bitrate=-1,
    dpi=300,
    progress_callback=lambda c, l: print(f"Saved: {c}/{l}"),
    savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
)
# plt.show()

plt.close(fig)
