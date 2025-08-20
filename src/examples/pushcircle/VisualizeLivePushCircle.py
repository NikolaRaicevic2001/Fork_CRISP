#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing import shared_memory

# ── Hyperparameters ──────────────────────────────────────────────────────────
R           = 0.10      # circle radius (meters)
dt          = 0.02      # time step (seconds)
N           = 200       # horizon length
num_state   = 2         # [px, py]
num_control = 3         # [cx, cy, lambda]
SHM_PLAN    = "CRISP_publisher"
SHM_FINAL   = "CRISP_final_state"
SHM_INIT    = "CRISP_initial_state"

count = 0
init_positions_list = np.array([[0.0, 0.0], [0.0, 0.2], [0.2, 0.2], [0.2, 0.0]], dtype=np.float64)
goal_state = np.array([1.0, 1.0], dtype=np.float64)

# ── Shared memory attach ─────────────────────────────────────────────────────
crisp_shm = shared_memory.SharedMemory( name=SHM_PLAN, create=False, size=np.prod((N, num_state + num_control)) * np.dtype(np.float64).itemsize )
crisp_shared = np.ndarray((N, num_state + num_control), dtype=np.float64, buffer=crisp_shm.buf)

crispFinalState_shm = shared_memory.SharedMemory( name=SHM_FINAL, create=False, size=num_state * np.dtype(np.float64).itemsize)
crispFinalState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispFinalState_shm.buf)

crispInitialState_shm = shared_memory.SharedMemory( name=SHM_INIT, create=False, size=num_state * np.dtype(np.float64).itemsize )
crispInitialState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispInitialState_shm.buf)

# ── Interactive figure: workspace (top) + time series (bottom) ───────────────
plt.ion()
steps = 0
t = np.arange(N) * dt
fig, (ax_xy, ax_ts) = plt.subplots( 2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2]}, constrained_layout=True)

# ── Top: XY workspace ────────────────────────────────────────────────────────
ax_xy.set_title("Circle workspace")
ax_xy.set_xlabel("x [m]")
ax_xy.set_ylabel("y [m]")
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.grid(True, alpha=0.25)

# Initial circle at initial pose (dynamic center)
init_xy = np.array([crispInitialState_share[0], crispInitialState_share[1]], dtype=float)
initial_circle = Circle((init_xy[0], init_xy[1]), R, fill=False, linewidth=2.0)
ax_xy.add_patch(initial_circle)
last_init_xy = init_xy.copy()

# Optional goal marker
ax_xy.plot([goal_state[0]], [goal_state[1]], marker="*", markersize=10, label="goal")

# Paths
line_xy_path,      = ax_xy.plot([], [], linewidth=2.0, label="center path (px, py)")
line_contact_path, = ax_xy.plot([], [], linestyle="--", linewidth=1.6, alpha=0.9, label="contact path (cx, cy)")

# Faded mini-circles (every 10th center point)
mini_patches = []
ax_xy.legend(loc="best")

# ── Bottom: time-series (px, py, cx, cy) ─────────────────────────────────────
ax_ts.set_title("States & contacts vs. time")
(ax_px,) = ax_ts.plot(t, np.zeros_like(t), label="px", linewidth=1.8)
(ax_py,) = ax_ts.plot(t, np.zeros_like(t), label="py", linewidth=1.8)
(ax_cx,) = ax_ts.plot(t, np.zeros_like(t), label="cx", linewidth=1.8)
(ax_cy,) = ax_ts.plot(t, np.zeros_like(t), label="cy", linewidth=1.8)
ax_ts.set_xlabel("time [s]")
ax_ts.set_ylabel("value")
ax_ts.grid(True, alpha=0.25)
ax_ts.legend(loc="best")

fig.canvas.draw()
fig.canvas.flush_events()

def refresh_initial_circle_if_needed():
    """If the initial position changed in shared memory, move the circle center. Updates arrays IN-PLACE to avoid global/nonlocal. """
    cur = np.array(crispInitialState_share[:], dtype=float)
    if not np.allclose(cur, last_init_xy):
        initial_circle.center = (cur[0], cur[1])
        init_xy[:] = cur          
        last_init_xy[:] = cur     

def update_xy(px, py, cx, cy):
    """Update the workspace view: main paths + faded mini-circles + autoscale."""
    # Update main paths
    line_xy_path.set_data(px, py)
    line_contact_path.set_data(cx, cy)

    # Remove old mini-circles
    for p in mini_patches:
        try:
            p.remove()
        except Exception:
            pass
    mini_patches.clear()

    # Add mini-circles at every 10th (px,py) with increasing opacity
    if len(px) > 0:
        idxs = list(range(0, len(px), 10))
        if idxs:
            alphas = np.linspace(0.25, 0.85, len(idxs))
            for i, a in zip(idxs, alphas):
                mini = Circle((px[i], py[i]), R * 0.25, alpha=a)
                ax_xy.add_patch(mini)
                mini_patches.append(mini)

        # Autoscale around data + current initial + goal
        x_all = np.concatenate([px, [init_xy[0], goal_state[0]]])
        y_all = np.concatenate([py, [init_xy[1], goal_state[1]]])
        xmin, xmax = np.min(x_all), np.max(x_all)
        ymin, ymax = np.min(y_all), np.max(y_all)
        pad = 0.2 + 0.1 * max(xmax - xmin, ymax - ymin)
        ax_xy.set_xlim(xmin - pad, xmax + pad)
        ax_xy.set_ylim(ymin - pad, ymax + pad)

try:
    while True:
        steps += 1
        print(f"[VisualizeLivePushBox] Step {steps} at {time.strftime('%H:%M:%S')}")

        # Current plan (N x 5): [px, py, cx, cy, lambda]
        plan = crisp_shared.reshape((N, num_state + num_control))
        px = plan[:, 0]
        py = plan[:, 1]
        cx = plan[:, 2]
        cy = plan[:, 3]
        # lam = plan[:, 4]  # available if needed

        # Check if someone updated the initial pose; move the initial circle if so
        refresh_initial_circle_if_needed()

        # Update time-series
        ax_px.set_ydata(px)
        ax_py.set_ydata(py)
        ax_cx.set_ydata(cx)
        ax_cy.set_ydata(cy)
        ax_ts.relim()
        ax_ts.autoscale_view()

        # Update workspace
        update_xy(px, py, cx, cy)

        # Draw (non-blocking)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.05)  # ~20 Hz

        # Example test write at every 100th step
        if steps % 100 == 0:
            if count < len(init_positions_list):
                crispInitialState_share[:] = init_positions_list[count]
                count += 1
                print(f"[VisualizeLivePushBox] Initial state updated: {crispInitialState_share}")

except KeyboardInterrupt:
    pass
finally:
    crisp_shm.close()
    crispFinalState_shm.close()
    crispInitialState_shm.close()
    try:
        plt.close(fig)
    except Exception:
        pass
