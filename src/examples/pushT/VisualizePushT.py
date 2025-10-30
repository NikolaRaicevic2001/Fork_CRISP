#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import json

# ---------------- PARAMETERS ----------------
l  = 0.05                       # base length unit
dt = 0.05                       # must match the solver's dt
N  = 50                         # must match the solver's horizon

num_state   = 19                # [px, py, theta, cx, cy, v1,w1, ..., v7,w7] = 19
num_control = 10                # [lambda1..lambda8, c_theta, s_theta] = 10
D = num_state + num_control

# Input / output paths
csv_file = Path(__file__).resolve().parent / "results" / "results_pushT_original.csv"
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD & SLICE ----------------
flat = np.loadtxt(csv_file, dtype=float)
assert flat.size % D == 0, f"CSV length {flat.size} not divisible by {D}."
rows = flat.size // D
if rows != N:
    print(f"[warn] CSV implies N={rows}, overriding configured N={N}. Using N={rows}.")
    N = rows

data = flat.reshape(N, D)

# State fields (indices mirror your C++ code)
px      = data[:, 0]
py      = data[:, 1]
theta   = data[:, 2]
cx      = data[:, 3]
cy      = data[:, 4]
# v1..w7 live at 5..18 but not needed for basic viz

# Control fields
lam     = data[:, 19:27]             # λ1..λ8 (shape N×8)
c_theta = data[:, 27]
s_theta = data[:, 28]

# Prefer the explicit theta state; fall back to atan2 if needed.
theta_from_cs = np.arctan2(s_theta, c_theta)
# sanity blend (optional): if |cos-sin| inconsistent, use fallback
bad = (np.abs(np.cos(theta) - c_theta) + np.abs(np.sin(theta) - s_theta)) > 1e-2
theta_vis = np.where(bad, theta_from_cs, theta)

t = np.arange(N) * dt

# Contact point in world frame
c = np.cos(theta_vis)
s = np.sin(theta_vis)
world_cx = px + c * cx - s * cy
world_cy = py + s * cx + c * cy

# ---------------- METRICS ----------------
CONTACT_EPS = 1e-6
step_dx = np.diff(world_cx)
step_dy = np.diff(world_cy)
step_dist = np.hypot(step_dx, step_dy)
total_contact_path_m = float(step_dist.sum())

force_norm = np.linalg.norm(lam, axis=1)  # N×8 -> row-wise norm
active = force_norm > CONTACT_EPS
active_pairs = active[:-1] & active[1:]
total_contact_path_active_m = float(step_dist[active_pairs].sum())
active_time_s = active_pairs.sum() * dt
avg_contact_speed_active_mps = (total_contact_path_active_m / active_time_s) if active_time_s > 0 else np.nan

print(f"[Metrics] Total contact path (all steps):   {total_contact_path_m:.6f} m")
print(f"[Metrics] Total contact path (active only): {total_contact_path_active_m:.6f} m")
print(f"[Metrics] Avg contact speed while active:   {avg_contact_speed_active_mps:.6f} m/s")

with open(results_dir / "metrics_pushT_original.json", "w") as f:
    json.dump({
        "total_contact_path_m": total_contact_path_m,
        "total_contact_path_active_m": total_contact_path_active_m,
        "avg_contact_speed_active_mps": float(avg_contact_speed_active_mps),
        "dt": float(dt),
        "N": int(N),
        "example": "pushT",
        "method": "original"
    }, f, indent=2)

# ---------------- T-SHAPE OUTLINE (body frame) ----------------
stem_w = 1.0 * l
stem_h = 4.0 * l
bar_w  = 4.0 * l
bar_h  = 1.0 * l

T_outline = np.array([
    [-stem_w/2, -stem_h],
    [ stem_w/2, -stem_h],
    [ stem_w/2,  0.0],
    [ bar_w/2,   0.0],
    [ bar_w/2,   bar_h],
    [-bar_w/2,   bar_h],
    [-bar_w/2,   0.0],
    [-stem_w/2,  0.0],
    [-stem_w/2, -stem_h],
]).T

def transform_body_polygon(x_body, y_body, px_k, py_k, theta_k):
    c, s = np.cos(theta_k), np.sin(theta_k)
    R = np.array([[c, -s],
                  [s,  c]])
    return R @ np.vstack([x_body, y_body]) + np.array([[px_k],[py_k]])

# ---------------- GOAL ----------------
goal_px, goal_py = 0.036, -0.143
goal_theta = -2.637
goal_world = transform_body_polygon(T_outline[0], T_outline[1], goal_px, goal_py, goal_theta)

# ---------------- STATIC PLOTS ----------------
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 5))

ax[0].plot(t, px,    label="px  [m]")
ax[0].plot(t, py,    label="py  [m]")
ax[0].plot(t, theta_vis, label="θ   [rad]")
ax[0].axhline(goal_px, color='r', ls='--', lw=1, alpha=0.5, label='goal px')
ax[0].set_ylabel("states")
ax[0].legend()

ax[1].plot(t, world_cx, label="cx  [m]")
ax[1].plot(t, world_cy, label="cy  [m]")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("contact point")
ax[1].legend()

for j in range(lam.shape[1]):
    ax[2].plot(t, lam[:, j], label=f"λ{j+1}")
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("contact forces [N]")
ax[2].legend(ncol=4, fontsize=8)

fig.tight_layout()
fig.savefig(results_dir / "figures_pushT_original.png", dpi=110, bbox_inches='tight')

# ---------------- ANIMATION ----------------
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_aspect("equal", adjustable="box")

pad = 0.25
xmin = min(px.min(), world_cx.min(), goal_px) - pad
xmax = max(px.max(), world_cx.max(), goal_px) + pad
ymin = min(py.min(), world_cy.min(), goal_py) - pad
ymax = max(py.max(), world_cy.max(), goal_py) + pad
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

# Artists
t_body,       = ax2.plot([], [], 'k-', lw=2, label="T-shape")
center,       = ax2.plot([], [], 'bo', ms=4, label="center")
full_c_traj,  = ax2.plot(world_cx, world_cy, '--', alpha=0.3, label="contact traj")
c_path,       = ax2.plot([], [], '-', lw=2, alpha=0.9, label="contact path")
c_point,      = ax2.plot([], [], 'ro', ms=6, label="contact point")
goal_shape,   = ax2.plot(goal_world[0], goal_world[1], 'r--', lw=2, label="goal")

ax2.set_title("Push T Animation")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.legend()

def frame(k):
    world = transform_body_polygon(T_outline[0], T_outline[1], px[k], py[k], theta_vis[k])
    t_body.set_data(world[0], world[1])
    center.set_data([px[k]], [py[k]])
    c_path.set_data(world_cx[:k+1], world_cy[:k+1])
    c_point.set_data([world_cx[k]], [world_cy[k]])
    return t_body, center, c_path, c_point, goal_shape

ani = FuncAnimation(fig2, frame, frames=N, interval=dt*1000, blit=True)
gif_path = results_dir / "animation_pushT_original.gif"
ani.save(gif_path, writer='pillow', fps=max(1, int(round(1/dt))), dpi=110)

plt.show()
