#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ──────── PARAMETERS ──────────────────────────────────────────────────────
R              = 0.4            # radius of the circle  
dt             = 0.02           # time-step              
N              = 100            # number of time-steps
num_state      = 2              # [px, py]
num_control    = 3              # [cx, cy, λ]  -> we plot cx, cy
gradient_method  = "FD"         # "AD" or "FD" – only used in file-name
csv_file = ( Path(__file__).resolve().parent / "results" / f"results_pushcircle_sdf_{gradient_method}.csv" )

# goal configuration (in world frame)
goal_state   = np.array([-0.5, -1.0])  

# ──────── LOAD CSV ────────────────────────────────────────────────────────
flat  = np.loadtxt(csv_file, dtype=float)
data  = flat.reshape(N, num_state + num_control)
px, py        = data[:, 0], data[:, 1]
cx, cy        = data[:, 2], data[:, 3]
lam           = data[:, 4]                   
t             = np.arange(N) * dt

# ──────── STATIC PLOTS ────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax[0].plot(t, px, label="px [m]")
ax[0].plot(t, py, label="py [m]")
ax[0].set_ylabel("state")
ax[0].legend()

ax[1].plot(t, cx, label="cx [m]")
ax[1].plot(t, cy, label="cy [m]")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("contact point")
ax[1].legend()

fig.tight_layout()
fig.savefig(f"results/figures_pushcircle_{gradient_method}.png", dpi=120, bbox_inches='tight')


# ──────── CARTOON ANIMATION ───────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_aspect("equal")
ax2.set_xlim(px.min()-1, px.max()+1)
ax2.set_ylim(py.min()-1, py.max()+1)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title("Push-Circle animation")

circle_artist   = Circle((px[0], py[0]), R, ec='k', fc='none', lw=2)
contact_artist, = ax2.plot([], [], 'rx', ms=6, label="contact")
goal_artist     = Circle(goal_state, R, ec='r', fc='none', ls='--')
ax2.add_patch(circle_artist)
ax2.add_patch(goal_artist)
ax2.legend()

def update(k):
    circle_artist.center = (px[k], py[k])
    contact_artist.set_data([px[k] + cx[k]], [py[k] + cy[k]])
    return circle_artist, contact_artist


ani = FuncAnimation(fig2, update, frames=N, interval=dt*1000, blit=True)
gif_name = f"results/animation_pushcircle_{gradient_method}.gif"
ani.save(gif_name, writer='pillow', fps=int(1/dt), dpi=120)

plt.show()
