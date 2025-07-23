# Import necessary libraries
import os
import sys
import time
import numpy as np

from pathlib import Path
from multiprocessing import shared_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../build/core')) # Add the path to generated python bindings
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent      # Adjust the path to your project root
import pyCRISP

# Set the hyperparameters
num_state = 3
num_control = 6
N = 100
variableNum = N * (num_state + num_control)

# Set problem parameters
x_initial_guess = np.zeros(variableNum, dtype=np.float64)  
x0 = np.zeros(num_state)
num_segments = 18
theta = 12 * 2 * np.pi / num_segments
xf = np.array([3 * np.cos(theta), 3 * np.sin(theta), theta], dtype=np.float64)

# Create the shared-memory block
shm_name = "CRISP_publisher"
try:
    existing = shared_memory.SharedMemory(name=shm_name)
    existing.close()
    existing.unlink()
except FileNotFoundError:
    pass
crispSol_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=variableNum * np.dtype(np.float64).itemsize)
crispSol_share = np.ndarray((variableNum,), dtype=np.float64, buffer=crispSol_shm.buf)

shm_name = "CRISP_final_state"
try:
    existing = shared_memory.SharedMemory(name=shm_name)
    existing.close()
    existing.unlink()
except FileNotFoundError:
    pass
crispFinalState_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_state * np.dtype(np.float64).itemsize)
crispFinalState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispFinalState_shm.buf)
crispFinalState_share[:] = xf

shm_name = "CRISP_initial_state"
try:
    existing = shared_memory.SharedMemory(name=shm_name)
    existing.close()
    existing.unlink()
except FileNotFoundError:
    pass
crispInitialState_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_state * np.dtype(np.float64).itemsize)
crispInitialState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispInitialState_shm.buf)
crispInitialState_share[:] = np.zeros(num_state, dtype=np.float64)

# Create optimization problem
problemName = "Pushbox"
folderName = "model"
problem = pyCRISP.OptimizationProblem(variableNum, "Pushbox")
obj = pyCRISP.ObjectiveFunction(variableNum, num_state, problemName, folderName, "pushboxObjective")
dynamic = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushboxDynamicConstraints")
contact = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushboxContactConstraints")
initial = pyCRISP.ConstraintFunction(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints")
problem.add_objective(obj)
problem.add_equality_constraint(dynamic)
problem.add_inequality_constraint(contact)
problem.add_equality_constraint(initial)

# Initialize the solver
params = pyCRISP.SolverParameters()
solver = pyCRISP.SolverInterface(problem, params)
print(f"[pyCRISP] Problem {problemName} created with {variableNum} variables and {N} time steps.")

# Set the parameters for those parametric functions
solver.set_problem_parameters("pushboxObjective", xf)               
solver.set_problem_parameters("pushboxInitialConstraints", x0)
print(f"[pyCRISP] Problem parameters set with initial states {x0} and final states {xf}.")
solver.set_hyper_parameters("max_iter", np.array([100]))
solver.initialize(x_initial_guess)
solver.solve()

# Get the solution
solution = solver.get_solution()
print(f"[pyCRISP] Solution obtained: {solution.shape} , type of the solution {solution.dtype}")

try:
    while True:
        initial_state = crispInitialState_share
        final_state = crispFinalState_share
        solver.set_problem_parameters("pushboxObjective", final_state)
        solver.set_problem_parameters("pushboxInitialConstraints", initial_state)
        solver.reset_problem(x_initial_guess)
        solver.solve()
        solution = solver.get_solution()

        crispSol_share[:] = solution
        print(f"[Solver] published new trajectory @ {time.strftime('%H:%M:%S')}")
        time.sleep(0.5)  
except KeyboardInterrupt:
    print("[Solver] interrupted by user, cleaning up…")
finally:
    crispSol_shm.close()
    crispSol_shm.unlink()
