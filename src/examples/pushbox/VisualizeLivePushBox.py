#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from multiprocessing import shared_memory
from matplotlib.animation import FuncAnimation

# Hyperparameters
a = 0.5                         # half width of the box
b = 0.25                        # half height of the box
dt = 0.02                       # time step (20 ms) 
num_state = 3                   # STATE  (3) : [px, py, θ]
num_control = 6                 # CONTROL (6) : [cx, cy, λ1-λ4, λ5]  (cx, cy plotted)
N = 100                         # number of time steps      
steps = 0                       # number of steps taken
variableNum = N * (num_state + num_control)

# Setting up the listener for shared memory
crisp_shm = shared_memory.SharedMemory(name="CRISP_publisher", create=False, size=np.prod((N, num_state + num_control)) * np.dtype(np.float64).itemsize)
crisp_shared = np.ndarray((N, num_state + num_control), dtype=np.float64, buffer=crisp_shm.buf)

crispFinalState_shm = shared_memory.SharedMemory(name="CRISP_final_state", create=False, size=num_state * np.dtype(np.float64).itemsize)
crispFinalState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispFinalState_shm.buf)

crispInitialState_shm = shared_memory.SharedMemory(name="CRISP_initial_state", create=False, size=num_state * np.dtype(np.float64).itemsize)
crispInitialState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispInitialState_shm.buf)

# Turn on interactive mode
plt.ion()

# Build your figure once
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,5), sharex=False, sharey=False)
t = np.arange(N)*dt

line_px,    = ax1.plot(t, np.zeros_like(t), label="px")
line_py,    = ax1.plot(t, np.zeros_like(t), label="py")
line_theta, = ax1.plot(t, np.zeros_like(t), label="θ")
ax1.set_ylabel("states")
ax1.legend()

line_cx,  = ax2.plot(t, np.zeros_like(t), label="cx")
line_cy,  = ax2.plot(t, np.zeros_like(t), label="cy")
ax2.set_ylabel("contact")
ax2.set_xlabel("time [s]")
ax2.legend()

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

        if steps%100 == 0:
            crispInitialState_share[:] = np.array([1,1,45 * np.pi / 180], dtype=np.float64)  
            print(f"[VisualizeLivePushBox] Initial state updated: {crispInitialState_share}")

        time.sleep(0.05)  # 20 Hz
except KeyboardInterrupt:
    pass
finally:
    crisp_shm.close()
    crispFinalState_shm.close()
    crispInitialState_shm.close()
