import numpy as np
import os
import sys
# add the generated python bindings to the path, defalut path is path/to/build/core
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../build/core')) # Add the path to generated python bindings
import pyCRISP

# In the python example, the workflow is similar to the C++ example. 
# Except that we don't need to specify the function handles (obj, constraints) as we assume the autodifferentiation has already been generated with the following naming convention:
# problem name: "PushbotSwingUp"; folder name: "model"; function name: "pushbotObjective", "pushBotDynamicConstraints", "pushBotContactConstraints", "pushBotInitialConstraints".
# the obj and constraints functions can be defined in both parametric and non-parametric ways. The parametric way is useful for dynamically adjusting problem parameter like the tracking ref, terminal states.

# If your problem itself is not changed (the dynamics or constraints themself), the autodifferentiation function library would be loaded automatically and only once. And then you are able to set different problem parameters and solver hyperparameters in the python interface for resolve the problem.
# The following example shows how we set up and solve the pushbot swing up problem using the python interface.


num_state = 4
num_control = 3
N = 100

# 1. Create optimization problem. 
# Notice that the problem name, folder name, and function name should be the same as the ones in the model file for the system to load the autodifferentiation functions
variableNum = N * (num_state + num_control)
problemName = "PushbotSwingUp"
folderName = "model"
problem = pyCRISP.OptimizationProblem(N * (num_state + num_control), "PushbotSwingUp")


# 2. Create objective and constraints objects. In py interface, we don't need to specify the function handles as we assume the autodifferentiation has already been generated
# If the function is parameterized, the second argument should be the number of parameters
obj = pyCRISP.ObjectiveFunction(variableNum, num_state, problemName, folderName, "pushbotObjective")
dynamic = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushBotDynamicConstraints")
contact = pyCRISP.ConstraintFunction(variableNum, problemName, folderName, "pushBotContactConstraints")
initial = pyCRISP.ConstraintFunction(variableNum, num_state, problemName, folderName, "pushBotInitialConstraints")

# 3. Add obj and constraints to the problem
problem.add_objective(obj)
problem.add_equality_constraint(dynamic)
problem.add_inequality_constraint(contact)
problem.add_equality_constraint(initial)

# 4. Set problem parameters, as the functions can be defined in both parametric and non-parametric ways, the problem parameters can also be adjusted dynamically after creating the solver object.
x_initial_states = np.zeros(num_state)
x_final_states = np.array([0, 0, 0, 0])

# Read initial guess from a file, you can also set it to your own test values.
x_initial_guess = np.loadtxt('/home/workspace/src/examples/pushbot/initial_guess_pushbot_example.txt')
x_initial_states[:] = x_initial_guess[:num_state]


# 5. Initialize the solver hyperparameters, and you can change the hyperparameters here or after 
params = pyCRISP.SolverParameters()
solver = pyCRISP.SolverInterface(problem, params)

# set the parameters for those parametric functions: mandatory
solver.set_problem_parameters("pushbotObjective", x_final_states) # the objective parameters are the terminal states for calculating the terminal cost
solver.set_problem_parameters("pushBotInitialConstraints", x_initial_states)
# set the hyperparameters for the solver: optional
# params.set_hyper_parameters("verbose", np.zeros(1))
solver.initialize(x_initial_guess)
solver.solve()

# 7. Get the solution. Note that the solution and all intermediate results are automatically saved to path/to/build/results folder in .mat format for further analysis and visualization in matlab.
solution = solver.get_solution()

# 8. Optional: 
# Dynamically change the problem parameters and solver hyperparameters and solve the problem again.
# solver.set_hyper_parameters("max_iter", np.array([500]))
# solver.set_problem_parameters("pushbotObjective", np.array([0.5, 0, 0, 0]))
# x_initial_new = 0.1 * np.random.rand(variableNum)
# solver.reset(x_initial_new)
# solver.solve()
# solution = solver.get_solution()
