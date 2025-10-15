#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ─────── Params ─────────────────────────────────
R           = 0.05          # circle radius
dt          = 0.02
N           = 100
num_state   = 2             # [px, py]
num_control = 3             # [cx, cy, λ]
CONTACT_EPS = 1e-8

# File + goal
csv_file   = (Path(__file__).resolve().parent / "results" / "results_pushcircle_sdf.csv")
goal_state = np.array([1.0, 1.0])  

# Optional tags for metrics filename
example = "pushcircle"
method  = "sdf"

# Ensure folders
(Path(__file__).resolve().parent / "results").mkdir(exist_ok=True, parents=True)

# ─────── Load data ────────────────────────────────────────────────────────
flat = np.loadtxt(csv_file, dtype=float)
data = flat.reshape(N, num_state + num_control)
px, py = data[:, 0], data[:, 1]
cx, cy = data[:, 2], data[:, 3]
lam    = data[:, 4]
t      = np.arange(N) * dt

# World-frame contact points 
world_cx = px + cx
world_cy = py + cy

# ─────── CONTACT PATH METRICS ─────────────────────────────────────────────
# Step-to-step distances of the contact point (world frame)
step_dx   = np.diff(world_cx)
step_dy   = np.diff(world_cy)
step_dist = np.hypot(step_dx, step_dy)
total_contact_path_m = float(step_dist.sum())

# Only count when contact is "active" (norm(lam)>eps); handle scalar lam per step
lam_arr = lam[:, None] if lam.ndim == 1 else lam
contact_force_norm = np.linalg.norm(lam_arr, axis=1)
active = contact_force_norm > CONTACT_EPS

# We require contact active on BOTH ends of the step
active_pairs = active[:-1] & active[1:]
total_contact_path_active_m = float(step_dist[active_pairs].sum())

active_time_s = float(active_pairs.sum() * dt)
avg_contact_speed_active_mps = (float(total_contact_path_active_m / active_time_s) if active_time_s > 0 else float("nan"))

print(f"[Metrics] Total contact path (all steps):   {total_contact_path_m:.6f} m")
print(f"[Metrics] Total contact path (active only): {total_contact_path_active_m:.6f} m")
print(f"[Metrics] Avg contact speed while active:   {avg_contact_speed_active_mps:.6f} m/s")

metrics = {
    "total_contact_path_m": total_contact_path_m,
    "total_contact_path_active_m": total_contact_path_active_m,
    "avg_contact_speed_active_mps": avg_contact_speed_active_mps,
    "active_time_s": active_time_s,
    "dt": float(dt),
    "N": int(N),
    "example": example,
    "method": method,
    "contact_eps": CONTACT_EPS,
}
with open(f"results/metrics_{example}_{method}.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ─────── Static plots (states, contact, lambda) ───────────────────────────
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

axs[0].plot(t, px, label="px [m]")
axs[0].plot(t, py, label="py [m]")
axs[0].set_ylabel("state")
axs[0].legend(loc="best")

axs[1].plot(t, cx, label="cx (body->world offset) [m]")
axs[1].plot(t, cy, label="cy (body->world offset) [m]")
axs[1].set_ylabel("contact (body frame)")
axs[1].legend(loc="best")

axs[2].plot(t, lam, label="λ")
axs[2].axhline(CONTACT_EPS, ls="--", lw=1, color="k", alpha=0.4)
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("contact force")
axs[2].legend(loc="best")

fig.tight_layout()
fig.savefig("results/figures_pushcircle.png", dpi=140, bbox_inches="tight")

# ─────── Enhanced “cartoon” visualization + animation ─────────────────────
fig2, ax2 = plt.subplots(figsize=(7.5, 6))
ax2.set_aspect("equal")

# nice margins
xmin = min(px.min(), world_cx.min(), goal_state[0]) - 0.4
xmax = max(px.max(), world_cx.max(), goal_state[0]) + 0.4
ymin = min(py.min(), world_cy.min(), goal_state[1]) - 0.4
ymax = max(py.max(), world_cy.max(), goal_state[1]) + 0.4
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

# Artists (pre-create so legend is stable)
circle_artist = Circle((px[0], py[0]), R, ec="k", fc="none", lw=2, label="circle")
goal_artist   = Circle((goal_state[0], goal_state[1]), R, ec="r", fc="none", ls="--", lw=2, label="goal")

center_traj, = ax2.plot(px, py, color="C0", lw=1.5, alpha=0.6, label="center trajectory")
contact_path, = ax2.plot([], [], color="gold", lw=2, alpha=0.9, label="contact path")
contact_point, = ax2.plot([], [], "ro", ms=5, label="contact point")

ax2.add_patch(circle_artist)
ax2.add_patch(goal_artist)

# circle center marker (current)
center_marker, = ax2.plot([], [], "bo", ms=4, label="circle center")

# Create custom legend handles (round markers for circle + goal)
legend_handles = [
    Line2D([], [], marker='o', color='k', lw=2, markersize=10, markerfacecolor='none', label='circle'),
    Line2D([], [], marker='o', color='r', lw=2, markersize=10, markerfacecolor='none', ls='--', label='goal'),
    Line2D([], [], color='C0', lw=1.5, alpha=0.6, label='center trajectory'),
    Line2D([], [], color='gold', lw=2, alpha=0.9, label='contact path'),
    Line2D([], [], marker='o', color='r', lw=0, markersize=6, label='contact point'),
    Line2D([], [], marker='o', color='b', lw=0, markersize=6, label='circle center'),
]
ax2.set_title("Push-Circle — Center, Contact, and Goal")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.legend(handles=legend_handles, loc='upper left', frameon=True)

def frame(k):
    # update circle (current pose)
    circle_artist.center = (px[k], py[k])
    center_marker.set_data([px[k]], [py[k]])

    # update cumulative contact path up to k and the current contact point
    contact_path.set_data(world_cx[:k+1], world_cy[:k+1])
    contact_point.set_data([world_cx[k]], [world_cy[k]])

    return circle_artist, center_marker, contact_path, contact_point

ani = FuncAnimation(fig2, frame, frames=N, interval=dt*1000, blit=True)
gif_name = f"results/animation_{example}_{method}.gif"
ani.save(gif_name, writer="pillow", fps=int(round(1/dt)), dpi=120)

plt.show()
