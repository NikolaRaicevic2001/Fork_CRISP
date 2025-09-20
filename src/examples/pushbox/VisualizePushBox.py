#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib.animation import FuncAnimation
from pathlib import Path

# ---------------- PARAMETERS ----------------
# Define model parameters for pushbox
a = 0.05                       # half width of the box
b = 0.05                       # half height of the box
dt = 0.1                       # time step (100 ms)
N  = 100                       # number of time steps
num_state   = 3                # STATE  (3) : [px, py, θ]
CONTACT_EPS = 1e-6
example_name = "pushbox"    

if example_name == "pushbox":
    method = "actual"
    num_control = 6             # CONTROL (6) : [cx, cy, λ1-λ4]  (cx, cy plotted)
    csv_file   = Path(__file__).resolve().parent / "results" / f"results_{example_name}_{method}.csv" 
elif example_name == "pushbox_single":
    example_name = "pushbox"
    method = "single"
    num_control = 5             
    csv_file   = Path(__file__).resolve().parent / "results" / f"results_{example_name}_{method}.csv"
elif example_name == "pushbox_sdf":
    method = "roundedsmooth"
    num_control = 3             # CONTROL (3) : [cx, cy, λ]  (cx, cy plotted)
    csv_file   = Path(__file__).resolve().parent / "results" / f"results_{example_name}_{method}.csv" 

# Goal configuration (in world frame)
num_segments    = 18            # number of segments for the goal circle
theta_seg       = 12 * 2 * np.pi / num_segments  
goal_state      = np.array([2 * np.cos(theta_seg), 2 * np.sin(theta_seg), theta_seg])  
goal_state      = np.array([0.4, 0.3, 0.0])
# goal_state      = np.array([-1, -1.73205, 4.18879])


# ---------------- load data from CSV --------------------
flat   = np.loadtxt(csv_file, dtype=float)          # 900 × 1
data   = flat.reshape(N, num_state + num_control)

px, py, theta   = data[:, 0], data[:, 1], data[:, 2]
cx,  cy         = data[:, 3], data[:, 4]
lam             = data[:, 5]  
t               = np.arange(N) * dt

world_cx = px +  np.cos(theta)*cx  -  np.sin(theta)*cy
world_cy = py +  np.sin(theta)*cx  +  np.cos(theta)*cy

# ---------- CONTACT PATH METRICS ----------
# Step-to-step distances 
step_dx = np.diff(world_cx)
step_dy = np.diff(world_cy)
step_dist = np.hypot(step_dx, step_dy)            
total_contact_path_m = float(step_dist.sum())

# Only count distance when contact is "active": lam can be (N, 1) or (N, 4) etc.; take row-wise norm
lam_arr = lam if lam.ndim == 2 else lam[:, None]
contact_force_norm = np.linalg.norm(lam_arr, axis=1)
active = contact_force_norm > CONTACT_EPS
active_pairs = active[:-1] & active[1:]
total_contact_path_active_m = float(step_dist[active_pairs].sum())

# Average speed of the contact point while active 
active_time_s = active_pairs.sum() * dt
avg_contact_speed_active_mps = (float(total_contact_path_active_m / active_time_s) if active_time_s > 0 else float("nan"))

print(f"[Metrics] Total contact path (all steps): {total_contact_path_m:.6f} m")
print(f"[Metrics] Total contact path (active only): {total_contact_path_active_m:.6f} m")
print(f"[Metrics] Avg contact speed while active: {avg_contact_speed_active_mps:.6f} m/s")

metrics = {
    "total_contact_path_m": total_contact_path_m,
    "total_contact_path_active_m": total_contact_path_active_m,
    "avg_contact_speed_active_mps": avg_contact_speed_active_mps,
    "dt": float(dt),
    "N": int(N),
    "example_name": example_name,
    "method": method,
}
Path("results").mkdir(exist_ok=True, parents=True)
with open(f"results/metrics_{example_name}_{method}.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------- STATIC PLOTS ----------
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 5))

ax[0].plot(t, px,     label="px  [m]")
ax[0].plot(t, py,     label="py  [m]")
ax[0].plot(t, theta,  label="θ   [rad]")
ax[0].set_ylabel("states")
ax[0].legend()

ax[1].plot(t, world_cx, label="cx  [m]")
ax[1].plot(t, world_cy, label="cy  [m]")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("contact point")
ax[1].legend()

ax[2].plot(t, lam, label="λ  [N]")
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("contact forces")
ax[2].legend()

fig.tight_layout()
fig.savefig(f"results/figures_{example_name}_{method}.png", dpi=100, bbox_inches='tight')

# ---------- SIMPLE CARTOON ANIMATION ----------
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_aspect("equal")
ax2.set_xlim(px.min()-1.0, px.max()+1.5)
ax2.set_ylim(py.min()-1.0, py.max()+1.5)

# Artists
box,   = ax2.plot([], [], 'k-', lw=2, label="box")
center,= ax2.plot([], [], 'bo', ms=4, label="box center")
ax2.plot(world_cx, world_cy, '--', color='gold', alpha=0.3, label='final trajectory')
contact_path,    = ax2.plot([], [], '-', lw=2, color='y', alpha=0.9, label="contact path")
contact_points,  = ax2.plot([], [], 'ro', ms=6, label="contact point")

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
    R_goal = np.array([[np.cos(goal_state[2]), -np.sin(goal_state[2])],
                       [np.sin(goal_state[2]),  np.cos(goal_state[2])]])
    goal_corner = R_goal @ corners + np.array([[goal_state[0]], [goal_state[1]]])

    # Update plot elements
    box.set_data(world[0], world[1])
    center.set_data([px[k]], [py[k]])

    # Update cumulative path up to frame k, and current contact point
    contact_path.set_data(world_cx[:k+1], world_cy[:k+1])
    contact_points.set_data([world_cx[k]], [world_cy[k]])

    goal.set_data(goal_corner[0], goal_corner[1])
    return box, center, contact_path, contact_points, goal

ani = FuncAnimation(fig2, frame, frames=N, interval=dt*1000, blit=True)
ani.save(f"results/animation_{example_name}_{method}.gif", writer='pillow', fps=1/dt, dpi=100)

plt.show()
