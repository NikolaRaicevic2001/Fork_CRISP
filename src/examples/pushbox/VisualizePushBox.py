#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.animation import FuncAnimation

# ---------------- user-editable constants ----------------
N           = 100               # time steps
num_state   = 3                 # [px, py, theta]
num_control = 6                 # [cx, cy, λ1-λ4]  (only cx, cy plotted)
dt          = 0.02              # 20 ms
a, b        = 0.5, 0.25         # box half-sizes → diagonal = 2·r
num_segments    = 18            # number of segments for the goal circle
theta_seg       = 12 * 2 * np.pi / num_segments  
goal_state      = np.array([3 * np.cos(theta_seg), 3 * np.sin(theta_seg), theta_seg])  

# ---------------- load data from CSV --------------------
csv_file   = Path(__file__).resolve().parent / "results_pushbox.csv" 
flat   = np.loadtxt(csv_file, dtype=float)          # 900 × 1
data   = flat.reshape(N, num_state + num_control)

px, py, theta   = data[:, 0], data[:, 1], data[:, 2]
cx,  cy         = data[:, 3], data[:, 4]
lam             = data[:, 5:]  
t               = np.arange(N) * dt

# ---------- STATIC PLOTS ----------
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

ax[0].plot(t, px,     label="px  [m]")
ax[0].plot(t, py,     label="py  [m]")
ax[0].plot(t, theta,  label="θ   [rad]")
ax[0].set_ylabel("states")
ax[0].legend()

ax[1].plot(t, cx, label="cx  [m]")
ax[1].plot(t, cy, label="cy  [m]")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("contact point")
ax[1].legend()

fig.tight_layout()

# ---------- SIMPLE CARTOON ANIMATION ----------
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_aspect("equal")
ax2.set_xlim(px.min()-1, px.max()+1)
ax2.set_ylim(py.min()-1, py.max()+1)
box,   = ax2.plot([], [], 'k-', lw=2, label="box")
center,= ax2.plot([], [], 'bo', ms=4, label="box center")
goal,  = ax2.plot([], [], 'r--', ms=4, label="goal")
ax2.set_title("Push Box Animation")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.legend()

def frame(k):
    # Rectangular corners (world frame)
    c, s = np.cos(theta[k]), np.sin(theta[k])
    R    = np.array([[c, -s],[s,  c]])
    corners = np.array([[-a, -b],[ a, -b],[ a, b],[-a, b],[-a, -b]]).T          
    world  = R @ corners + np.array([[px[k]],[py[k]]])

    # Goal configuration (world frame)
    R_goal = np.array([[np.cos(goal_state[2]), -np.sin(goal_state[2])],[np.sin(goal_state[2]), np.cos(goal_state[2])]])
    goal_corner = R_goal @ corners + np.array([[goal_state[0]], [goal_state[1]]])

    # Update plot elements
    box.set_data(world[0], world[1])
    center.set_data([px[k]], [py[k]])
    goal.set_data(goal_corner[0], goal_corner[1])
    return box, center

ani = FuncAnimation(fig2, frame, frames=N, interval=dt*1000, blit=True)

plt.show()
