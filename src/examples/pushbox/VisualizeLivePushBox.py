#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import shared_memory

# Hyperparameters
a = 0.05                         # half width of the box
b = 0.05                        # half height of the box
dt = 0.02                       # time step (20 ms) 
num_state = 3                   # STATE  (3) :  [px, py, θ]
num_control = 6                 # CONTROL (6) : [cx, cy, λ1-λ4]  
N = 100                         # number of time steps      
steps = 0                       # number of steps taken
count = 0                       # counter for initial positions
initial_position_list = np.array([[1,1,45 * np.pi / 180],[0.5,0.5,0]], dtype=np.float64)
variableNum = N * (num_state + num_control)

# Setting up the listener for shared memory
crisp_shm = shared_memory.SharedMemory(name="CRISP_publisher", create=False, size=np.prod((N, num_state + num_control)) * np.dtype(np.float64).itemsize)
crisp_shared = np.ndarray((N, num_state + num_control), dtype=np.float64, buffer=crisp_shm.buf)

crispFinalState_shm = shared_memory.SharedMemory(name="CRISP_final_state", create=False, size=num_state * np.dtype(np.float64).itemsize)
crispFinalState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispFinalState_shm.buf)

crispInitialState_shm = shared_memory.SharedMemory(name="CRISP_initial_state", create=False, size=num_state * np.dtype(np.float64).itemsize)
crispInitialState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispInitialState_shm.buf)

# --- geometry helper ---
def box_corners(px_k, py_k, th_k, a, b):
    c, s = np.cos(th_k), np.sin(th_k)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[-a, -b], [ a, -b], [ a,  b], [-a,  b], [-a, -b]]).T
    world = R @ corners + np.array([[px_k], [py_k]])
    return world  

# Turn on interactive mode
plt.ion()

# --- Figure showing path + boxes ---
fig_xy, ax_xy = plt.subplots(figsize=(6, 6))
ax_xy.set_aspect("equal")

path_line,        = ax_xy.plot([], [], "-",  lw=2,           label="COM path")
contact_path_line,= ax_xy.plot([], [], "-",  lw=1.5, color="y", alpha=0.9, label="contact path")
contact_pt,       = ax_xy.plot([], [], "yo", ms=4,           label="contact point (current)")
box_line,         = ax_xy.plot([], [], "k-", lw=2,           label="box (current)")
center_pt,        = ax_xy.plot([], [], "bo", ms=4,           label="center (current)")
init_box_line,    = ax_xy.plot([], [], "r-",  lw=2, alpha=0.8, label="initial pose")
goal_box_line,    = ax_xy.plot([], [], "r--", lw=2, alpha=0.9, label="goal pose")
init_center,      = ax_xy.plot([], [], "rs", ms=5)
goal_center,      = ax_xy.plot([], [], "r^", ms=5)

ax_xy.set_xlabel("x [m]")
ax_xy.set_ylabel("y [m]")
ax_xy.legend(loc="best")
fig_xy.canvas.draw()
fig_xy.canvas.flush_events()

# --- Figure showing states and controls ---
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,5), sharex=False, sharey=False)
t = np.arange(N)*dt

line_px,    = ax1.plot(t, np.zeros_like(t), label="px")
line_py,    = ax1.plot(t, np.zeros_like(t), label="py")
line_theta, = ax1.plot(t, np.zeros_like(t), label="θ")
ax1.set_ylabel("states")
ax1.legend(loc="best")

line_cx,  = ax2.plot(t, np.zeros_like(t), label="cx")
line_cy,  = ax2.plot(t, np.zeros_like(t), label="cy")
ax2.set_ylabel("contact")
ax2.set_xlabel("time [s]")
ax2.legend(loc="best")
fig.canvas.draw()
fig.canvas.flush_events()

# Starting the real-time loop
try:
    while True:
        steps += 1
        print(f"[VisualizeLivePushBox] Step {steps} at {time.strftime('%H:%M:%S')}")
        
        # Loading the crisp solution from shared memory
        final_state     = crispFinalState_share
        crisp_solution  = crisp_shared.reshape((N, num_state + num_control))
        px, py, theta   = crisp_solution[:,0], crisp_solution[:,1], crisp_solution[:,2]
        cx, cy          = crisp_solution[:,3], crisp_solution[:,4]
        lam             = crisp_solution[:, 5:]  
        print(f"[VisualizeLivePushBox] Received solution with shape {crisp_solution.shape} at {time.strftime('%H:%M:%S')}")
        print(f"[VisualizeLivePushBox] px: {px.shape}, py: {py.shape}, theta: {theta.shape}, cx: {cx.shape}, cy: {cy.shape}, lam: {lam.shape}")
        print(f"[VisualizeLivePushBox] Received final state: {final_state}")

        # --- Figure 1 Update ---
        # Pick a "current" index to show the moving box (animate along the solved path)
        k_show = steps % N

        # World-frame contact point (matches your offline script)
        world_cx = px +  np.cos(theta)*cx  -  np.sin(theta)*cy
        world_cy = py +  np.sin(theta)*cx  +  np.cos(theta)*cy
        print(f"[VisualizeLivePushBox] First few body contact points: ({cx[:5]}, {cy[:5]})")
        print(f"[VisualizeLivePushBox] First few world contact points: ({world_cx[:5]}, {world_cy[:5]})")

        # Update full paths
        path_line.set_data(px, py)
        contact_path_line.set_data(world_cx, world_cy)
        contact_pt.set_data([world_cx[k_show]], [world_cy[k_show]])

        # Update current box + center
        W = box_corners(px[k_show], py[k_show], theta[k_show], a, b)
        box_line.set_data(W[0], W[1])
        center_pt.set_data([px[k_show]], [py[k_show]])

        # Initial pose (solid red)
        sx, sy, sth = float(crispInitialState_share[0]), float(crispInitialState_share[1]), float(crispInitialState_share[2])
        W0 = box_corners(sx, sy, sth, a, b)
        init_box_line.set_data(W0[0], W0[1])
        init_center.set_data([px[0]], [py[0]])

        # Goal pose (dashed red) — use the goal from shared memory if present
        gx, gy, gth = float(crispFinalState_share[0]), float(crispFinalState_share[1]), float(crispFinalState_share[2])
        Wg = box_corners(gx, gy, gth, a, b)
        goal_box_line.set_data(Wg[0], Wg[1])
        goal_center.set_data([gx], [gy])

        # Keep XY axes comfortably framed
        pad = max(a, b) + 0.2
        xmin, xmax = np.nanmin(px) - pad, np.nanmax(px) + pad
        ymin, ymax = np.nanmin(py) - pad, np.nanmax(py) + pad
        ax_xy.set_xlim(xmin - 0.5, xmax + 0.5)
        ax_xy.set_ylim(ymin - 0.5, ymax + 0.5)

        # Redraw (non-blocking)
        fig_xy.canvas.draw()
        fig_xy.canvas.flush_events()

        # --- Figure 2 update ---
        # update the lines
        line_px.set_ydata(px)
        line_py.set_ydata(py)
        line_theta.set_ydata(theta)
        line_cx.set_ydata(cx)
        line_cy.set_ydata(cy)

        # allow each axis to rescale itself
        ax1.relim()            
        ax1.autoscale_view()  
        ax2.relim()
        ax2.autoscale_view()

        # redraw non-blocking
        fig.canvas.draw()
        fig.canvas.flush_events()

        # if steps%100 == 0:
        #     if count < len(initial_position_list):
        #         crispInitialState_share[:] = initial_position_list[count]
        #         count += 1
        #         print(f"[VisualizeLivePushBox] Initial state updated: {crispInitialState_share}")

        time.sleep(0.05)  # 20 Hz
except KeyboardInterrupt:
    pass
finally:
    crisp_shm.close()
    crispFinalState_shm.close()
    crispInitialState_shm.close()
