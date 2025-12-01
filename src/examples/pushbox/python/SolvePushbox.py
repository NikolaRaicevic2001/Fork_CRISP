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
a = 0.05                       # half width of the box
b = 0.05                       # half height of the box
num_state = 3
num_control = 6
N = 100
variableNum = N * (num_state + num_control)

# Set problem parameters
x_initial_guess = np.zeros(variableNum, dtype=np.float64)  
x0 = np.array([0.5, -0.3, 0.0], dtype=np.float64)       # Initial state [px, py, θ]
xf = np.array([0.5, 0.3, 0.0], dtype=np.float64)        # Final state [px, py, θ]
xee = np.array([0.0, -4*b], dtype=np.float64)           # End effector state [cx, cy]

# -------------- Helper function to create or replace shared memory --------------
def create_or_replace_shm(name, size):
    try:
        tmp = shared_memory.SharedMemory(name=name)
        tmp.close(); tmp.unlink()
    except FileNotFoundError:
        pass
    return shared_memory.SharedMemory(name=name, create=True, size=size)
# ---------------------------------------------------------------------------------

# Create the shared-memory block
crispSol_shm = create_or_replace_shm("CRISP_publisher", variableNum * np.dtype(np.float64).itemsize)
crispSol_share = np.ndarray((variableNum,), dtype=np.float64, buffer=crispSol_shm.buf)

crispFinalState_shm = create_or_replace_shm("CRISP_final_state", num_state * np.dtype(np.float64).itemsize)
crispFinalState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispFinalState_shm.buf)
crispFinalState_share[:] = xf

crispInitialState_shm = create_or_replace_shm("CRISP_initial_state", num_state * np.dtype(np.float64).itemsize)
crispInitialState_share = np.ndarray((num_state,), dtype=np.float64, buffer=crispInitialState_shm.buf)
crispInitialState_share[:] = x0

crispInitialStateEE_shm = create_or_replace_shm("CRISP_initial_state_ee", 2 * np.dtype(np.float64).itemsize)
crispInitialStateEE_share = np.ndarray((2,), dtype=np.float64, buffer=crispInitialStateEE_shm.buf)
crispInitialStateEE_share[:] = xee

# Create optimization problem
problemName = "Pushbox"
folderName = "model"
problem = pyCRISP.OptimizationProblem(variableNum, problemName)
obj     = pyCRISP.ObjectiveFunction(variableNum, num_state, problemName, folderName, "pushboxObjective")
dynamic = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushboxDynamicConstraints")
contact = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushboxContactConstraints")
contactSingleForce = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushboxContactSingleForceConstraints")
initial = pyCRISP.ConstraintFunction(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints")
# initial_ee = pyCRISP.ConstraintFunction(variableNum, 2, problemName, folderName, "pushboxInitialConstraintsEndEffector")
problem.add_objective(obj)
problem.add_equality_constraint(dynamic)
problem.add_inequality_constraint(contact)
problem.add_inequality_constraint(contactSingleForce)
problem.add_equality_constraint(initial)
# problem.add_equality_constraint(initial_ee)

# Initialize the solver
params = pyCRISP.SolverParameters()
solver = pyCRISP.SolverInterface(problem, params)
print(f"[pyCRISP] Problem {problemName} created with {variableNum} variables and {N} time steps.")

# Set the parameters for those parametric functions
solver.set_problem_parameters("pushboxObjective", xf)               
solver.set_problem_parameters("pushboxInitialConstraints", x0)
# solver.set_problem_parameters("pushboxInitialConstraintsEndEffector", xee)
print(f"[pyCRISP] Problem parameters set with initial states {x0} and final states {xf} and end-effector states {xee}.")
solver.set_hyper_parameters("maxIterations", np.array([1000]))              # maximum number of iterations for the outer loop
# solver.set_hyper_parameters("trustRegionInitRadius", np.array([1.0]))       # initial trust region radius
# solver.set_hyper_parameters("trustRegionMaxRadius", np.array([10.0]))       # maximum trust region radius
# solver.set_hyper_parameters("mu", np.array([1e1]))                          # penalty parameter
solver.set_hyper_parameters("muMax", np.array([1e8]))                       # maximum penalty parameter
# solver.set_hyper_parameters("etaLow", np.array([0.25]))                     # low threshold for reduction ratio
# solver.set_hyper_parameters("etaHigh", np.array([0.75]))                    # high threshold for reduction ratio
solver.set_hyper_parameters("trailTol", np.array([1e-5]))                   # tolerance for the outer iterations
solver.set_hyper_parameters("trustRegionTol", np.array([1e-5]))             # tolerance for the trust region
solver.set_hyper_parameters("constraintTol", np.array([1e-6]))              # tolerance for the maximum constraints violation
# solver.set_hyper_parameters("verbose", np.array([0]))                       # verbose level
solver.set_hyper_parameters("WeightedMode", np.array([1]))                  # 0: no weighted, 1: weighted
solver.set_hyper_parameters("WeightedTolFactor", np.array([10.0]))          # factor for the weighted mode
# solver.set_hyper_parameters("secondOrderCorrection", np.array([1]))         # 0: no second order correction, 1: second order correction

solver.initialize(x_initial_guess)
solver.solve()

# Get the solution
solution = solver.get_solution()
print(f"[pyCRISP] Solution obtained: {solution.shape} , type of the solution {solution.dtype}")

try:
    while True:
        initial_state = crispInitialState_share
        initial_state_ee = crispInitialStateEE_share
        final_state = crispFinalState_share
        x_initial_guess = solution
        solver.set_problem_parameters("pushboxObjective", final_state)
        solver.set_problem_parameters("pushboxInitialConstraints", initial_state)
        # solver.set_problem_parameters("pushboxInitialConstraintsEndEffector", initial_state_ee)
        solver.reset_problem(x_initial_guess)
        solver.solve()
        solution = solver.get_solution()

        crispSol_share[:] = solution
        print(f"[Solver] published new trajectory @ {time.strftime('%H:%M:%S')}")
        print(f"[pyCRISP] Initial state: {initial_state}, Initial End Effector state: {initial_state_ee}, Final state: {final_state}")
except KeyboardInterrupt:
    print("[Solver] interrupted by user, cleaning up…")
finally:
    crispSol_shm.close()
    crispSol_shm.unlink()
    crispFinalState_shm.close()
    crispFinalState_shm.unlink()
    crispInitialState_shm.close()
    crispInitialState_shm.unlink()
