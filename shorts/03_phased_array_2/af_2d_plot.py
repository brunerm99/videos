# af_2d_plot.py

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, pi
from matplotlib import cm
import matplotlib.animation as animation
from scipy import signal
from manim import rate_functions


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

npts = 500
theta = np.linspace(-pi, pi, npts)
phi = np.linspace(-pi, pi, npts)

weights_n = np.ones(N)
weights_m = np.ones(M)
# weights_n[weights_n.size // 2] = 1
# weights_m[weights_m.size // 2] = 1

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


max_frame_window = 500

window_n = signal.windows.taylor(N, nbar=5, sll=40)
window_m = signal.windows.taylor(N, nbar=5, sll=40)
steps_n = (window_n - weights_n) / max_frame_window
steps_m = (window_m - weights_m) / max_frame_window


z_min = -50
z_max = 0


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")


pause = 10


t = np.vectorize(rate_functions.ease_in_out_sine)(np.linspace(0, 1, max_frame_window))
steps_over_t_n = (window_n - weights_n)[:, None] * t
steps_over_t_m = (window_m - weights_m)[:, None] * t

frame = 0

AF = compute_af_2d(
    weights_n + steps_over_t_n[:, frame],
    weights_m + steps_over_t_m[:, frame],
    d_x,
    d_y,
    k_0,
    u_0,
    v_0,
)

AF_rect_log = 10 * np.log10(np.abs(AF))
AF_mask = np.where(AF_rect_log > z_min, AF_rect_log, np.nan)

x = AF_mask.shape[0] // 2
y = AF_mask.shape[1] // 2
max_frame_uv = x

max_frame_steer = 500

t_steer = np.vectorize(rate_functions.ease_in_out_sine)(
    np.linspace(0, 1, max_frame_steer)
)
u_0_1 = np.pi / 6
v_0_1 = 0
u_step_over_t = (u_0_1 - u_0) * t_steer
v_step_over_t = (v_0_1 - v_0) * t_steer

u_0_2 = np.pi / 10
v_0_2 = -np.pi / 10
u2_step_over_t = (u_0_2 - u_0_1) * t_steer
v2_step_over_t = (v_0_2 - v_0_1) * t_steer


def update(frame, p, ax):
    if frame < max_frame_uv:
        p[0].remove()
        p[0] = ax.plot_surface(
            U[x - frame : x + frame, y - frame : y + frame],
            V[x - frame : x + frame, y - frame : y + frame],
            AF_mask[x - frame : x + frame, y - frame : y + frame],
            cmap=cm.coolwarm,
            antialiased=False,
        )
    elif (
        frame > max_frame_uv + pause and frame < max_frame_steer + max_frame_uv + pause
    ):
        frame_norm = frame - (max_frame_uv + pause)
        # print(frame_norm)
        p[0].remove()

        new_data = 10 * np.log10(
            np.abs(
                compute_af_2d(
                    weights_n,  # + steps_over_t_n[:, max_frame_window - 1],
                    weights_m,  # + steps_over_t_m[:, max_frame_window - 1],
                    d_x,
                    d_y,
                    k_0,
                    u_0 + u_step_over_t[frame_norm],
                    v_0 + v_step_over_t[frame_norm],
                )
            )
        )

        new_data = np.where(new_data > z_min, new_data, np.nan)
        p[0] = ax.plot_surface(U, V, new_data, cmap=cm.coolwarm, antialiased=False)
    elif (
        frame > max_frame_steer + max_frame_uv + pause * 2
        and frame < max_frame_steer + max_frame_uv + pause * 2 + max_frame_steer
    ):
        frame_norm = frame - (max_frame_window + max_frame_uv + pause * 2)
        # print(frame_norm)
        p[0].remove()

        new_data = 10 * np.log10(
            np.abs(
                compute_af_2d(
                    weights_n,  # + steps_over_t_n[:, max_frame_window - 1],
                    weights_m,  # + steps_over_t_m[:, max_frame_window - 1],
                    d_x,
                    d_y,
                    k_0,
                    u_0_1 + u2_step_over_t[frame_norm],
                    v_0_1 + v2_step_over_t[frame_norm],
                )
            )
        )
        new_data = np.where(new_data > z_min, new_data, np.nan)
        p[0] = ax.plot_surface(U, V, new_data, cmap=cm.coolwarm, antialiased=False)
    elif (
        frame > max_frame_window + max_frame_uv + pause * 3 + max_frame_steer
        and frame < max_frame_window + max_frame_uv + pause * 3 + max_frame_steer * 2
    ):
        frame_norm = frame - (
            max_frame_window + max_frame_uv + pause * 3 + max_frame_steer
        )
        # print(frame_norm)
        p[0].remove()

        new_data = 10 * np.log10(
            np.abs(
                compute_af_2d(
                    weights_n + steps_over_t_n[:, frame_norm],
                    weights_m + steps_over_t_m[:, frame_norm],
                    d_x,
                    d_y,
                    k_0,
                    u_0_2,
                    v_0_2,
                )
            )
        )
        new_data = np.where(new_data > z_min, new_data, np.nan)
        p[0] = ax.plot_surface(U, V, new_data, cmap=cm.coolwarm, antialiased=False)
    # ax.view_init(elev=30, azim=-50 + frame * 0.25, roll=0)


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
    fig,
    update,
    frames=max_frame_window + pause * 4 + max_frame_uv + max_frame_steer * 2,
    interval=30,
    fargs=(plot, ax),
)
ani.save(
    "static/af_2d_anim_rotating_w_steering_2.mp4",
    bitrate=-1,
    dpi=300,
    progress_callback=lambda c, l: print(f"Saved: {c}/{l}"),
    savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
)
# plt.show()

plt.close(fig)
