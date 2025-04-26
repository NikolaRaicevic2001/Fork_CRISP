
using Pkg
Pkg.activate(".")
using JuMP
using Gurobi
using LinearAlgebra
using MAT
using Printf

# Constants for pushT
const l = 0.05
const m = 1
const mu = 0.4
const g = 9.8
const r = 2.8 * l
const c = 0.4
const dc = 37 / 14
const x1 = -2 * l
const x2 = -0.5 * l
const x3 = 0.5 * l
const x4 = 2 * l
const y1 = -dc * l
const y2 = (3 - dc) * l
const y3 = (4 - dc) * l
const Q = Diagonal([10.0, 10.0, 0, 0, 10.0, 10.0, 0, 0, 0, 0])
const R = Diagonal([1.0, 1.0])

# All continuous states = [sx sy px py rc rs fc fs Fx Fy] 
# All binary states = [lambda1 - lambda8]
const N = 4# 10, 0.1 should be ok, but I use 50 steps in mine.
const dt = 0.1
const num_continuous_states = 10
const num_binary_states = 8

# Helper function to extract continuous state and discrete states for each time step
function extract_vars(x_continuous, x_discrete, i)
    base_idx_continuous = (i - 1) * num_continuous_states
    base_idx_discrete = (i - 1) * num_binary_states
    state_continuous = x_continuous[base_idx_continuous + 1:base_idx_continuous + num_continuous_states]
    state_discrete = x_discrete[base_idx_discrete + 1:base_idx_discrete + num_binary_states]
    return state_continuous, state_discrete
end

# Define dynamics constraints for the pushbox
function add_dynamics_constraints!(model, x_continuous, x_discrete)
    for k in 1:(N - 1)  # Loop over 1:N-1
        # Extract current and next states
        curr_state, _ = extract_vars(x_continuous, x_discrete, k)
        next_state, _ = extract_vars(x_continuous, x_discrete, k + 1)

        # Current and next continuous states
        sx_k, sy_k, px_k, py_k, rc_k, rs_k, fc_k, fs_k, Fx_k, Fy_k = curr_state
        sx_next, sy_next, px_next, py_next, rc_next, rs_next, fc_next, fs_next, Fx_next, Fy_next = next_state

        # Dynamics equations
        @constraint(model, sx_next == sx_k + dt * (rc_k * Fx_k - rs_k * Fy_k))
        @constraint(model, sy_next == sy_k + dt * (rs_k * Fx_k + rc_k * Fy_k))
        @constraint(model, fs_k == dt * (1 / (c * r)) * (-py_k * Fx_k + px_k * Fy_k))
        @constraint(model, rc_next == rc_k * fc_k - rs_k * fs_k)
        @constraint(model, rs_next == rs_k * fc_k + rc_k * fs_k)
        @constraint(model, rc_k^2 + rs_k^2 == 1)
        @constraint(model, fc_k^2 + fs_k^2 == 1)
        
        # for the final step
        if k == N-1
            @constraint(model, rc_next^2 + rs_next^2 == 1)
        end

    end
end

# Define contact constraints for the pushbox
function add_contact_constraints!(model, x_continuous, x_discrete)
    for i in 1:(N - 1)
        _, current_discrete = extract_vars(x_continuous, x_discrete, i)
        lambda = current_discrete  # Binary variables λ₁ to λ₈

        # Ensure only one λ is active at any time
        @constraint(model, sum(lambda) == 1)

        # Define continuous variables for the current time step
        current_state, _ = extract_vars(x_continuous, x_discrete, i)
        px, py, Fx, Fy = current_state[3], current_state[4], current_state[9], current_state[10]

        # Constraints for λ₁
        @constraint(model, lambda[1] * (py - y3) + lambda[2] * (px - x4) + lambda[3] * (py - y2) + lambda[4] * (px - x3) + lambda[5] * (py - y1) + lambda[6] * (px - x2) + lambda[7] * (py - y2) + lambda[8] * (px - x1) + lambda[1] * Fx + lambda[2] * Fy + lambda[3] * Fx + lambda[4] * Fy + lambda[5] * Fx + lambda[6] * Fy + lambda[7] * Fx + lambda[8] * Fy == 0)
        @constraint(model, lambda[1] * (px - x1) >= 0)
        @constraint(model, lambda[1] * (x4 - px) >= 0)
        @constraint(model, lambda[1] * (-Fy) >= 0)

        # Constraints for λ₂ to λ₈ (similar logic)
        @constraint(model, lambda[2] * (py - y2) >= 0)
        @constraint(model, lambda[2] * (y3 - py) >= 0)
        @constraint(model, lambda[2] * (-Fx) >= 0)

        @constraint(model, lambda[3] * (px - x3) >= 0)
        @constraint(model, lambda[3] * (x4 - px) >= 0)
        @constraint(model, lambda[3] * Fy >= 0)

        @constraint(model, lambda[4] * (py - y1) >= 0)
        @constraint(model, lambda[4] * (y2 - py) >= 0)
        @constraint(model, lambda[4] * (-Fx) >= 0)

        @constraint(model, lambda[5] * (px - x2) >= 0)
        @constraint(model, lambda[5] * (x3 - px) >= 0)
        @constraint(model, lambda[5] * (Fy) >= 0)

        @constraint(model, lambda[6] * (py - y1) >= 0)
        @constraint(model, lambda[6] * (y2 - py) >= 0)
        @constraint(model, lambda[6] * (Fx) >= 0)

        @constraint(model, lambda[7] * (px - x1) >= 0)
        @constraint(model, lambda[7] * (x2 - px) >= 0)
        @constraint(model, lambda[7] * Fy >= 0)

        @constraint(model, lambda[8] * (py - y2) >= 0)
        @constraint(model, lambda[8] * (y3 - py) >= 0)
        @constraint(model, lambda[8] * (Fx) >= 0)
    end
end

# Add initial state constraint
function add_initial_constraints!(model, x_continuous, x_discrete)
    initial_state, _ = extract_vars(x_continuous, x_discrete, 1)

    # Example initial conditions
    @constraint(model, initial_state[1] == 0)  # Initial sx
    @constraint(model, initial_state[2] == 0)  # Initial sy
    @constraint(model, initial_state[5] == 1)  # Initial rc:cos(0)
    @constraint(model, initial_state[6] == 0)  # Initial rs:sin(0)
end

# Set the objective function for the pushbox
function set_objective!(model, x_continuous, x_discrete, target_state)
    tracking_cost = 0.0
    control_cost = 0.0

    final_state, _ = extract_vars(x_continuous, x_discrete, N)
    tracking_error = final_state - target_state
    tracking_cost += tracking_error' * Q * tracking_error

    for i in 1:(N - 1)
        current_state, _ = extract_vars(x_continuous, x_discrete, i)
        Fx, Fy = current_state[9], current_state[10]
        control_cost += [Fx, Fy]' * R * [Fx, Fy]
    end

    @objective(model, Min, tracking_cost + control_cost)
end 

# Solve for each terminal state
result_dir = "results"
radius = 0.5

# Create results directory if it does not exist
if !isdir(result_dir)
    mkdir(result_dir)
end

for i in 0:20:0
    # Define the terminal state vector
    terminal_state = [
        radius * cos(i * π / 180),
        radius * sin(i * π / 180),
        0, 0,
        cos(i * π / 180),
        sin(i * π / 180),
        0, 0,
        0, 0
    ]
    result_filename = "$(result_dir)/result_$(lpad(i, 3, '0')).mat"

    println("Solving pushT with terminal state: $terminal_state")
    
    # Initialize model using Gurobi as the optimizer
    model = Model(Gurobi.Optimizer)
    
    # Set solver options (uncomment any options you need)
    # set_optimizer_attribute(model, "OutputFlag", 1)
    # set_optimizer_attribute(model, "MIPGap", 1e-4)
    # set_optimizer_attribute(model, "print_level", 3)
    set_optimizer_attribute(model, "IterationLimit", 100000000)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-2)


    # Define variables: continuous and binary
    @variable(model, x_continuous[1:N * num_continuous_states])
    @variable(model, x_discrete[1:N * num_binary_states], Bin)

    # Set initial guess for continuous variables
    initial_guess = zeros(N * num_continuous_states)
    for j in 1:length(initial_guess)
        set_start_value(x_continuous[j], initial_guess[j])
    end

    # Define objective and constraints
    set_objective!(model, x_continuous, x_discrete, terminal_state)
    add_dynamics_constraints!(model, x_continuous, x_discrete)
    add_contact_constraints!(model, x_continuous, x_discrete)
    add_initial_constraints!(model, x_continuous, x_discrete)

    # Solve the optimization problem
    optimize!(model)
    status = termination_status(model)
    println("Status: $status")
    
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        x_sol_continuous = value.(x_continuous)
        x_sol_discrete = value.(x_discrete)
        objvalue = objective_value(model)
        solving_time = solve_time(model)

        println("Solved for terminal state $i with objective: $objvalue")
        
        # Save results to file
        results_dict = Dict(
            "status" => string(status),
            "xopt_continuous" => x_sol_continuous,
            "xopt_discrete" => x_sol_discrete,
            "objective" => objvalue,
            "solving_time" => solving_time
        )
        matwrite(result_filename, results_dict)
        println("Results saved to $result_filename")
    else
        println("Failed to solve for terminal state $i, status: $status")
    end
end
