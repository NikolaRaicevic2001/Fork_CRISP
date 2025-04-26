function [violations, objValue] = checkSolutionFeasibility(Q_state, R_control, L, R, A, lb, ub, zOpt, x_final, N, num_state, num_control)
    % INPUTS:
    % Q          - Cost matrix for the quadratic objective
    % L, R       - Complementarity matrices
    % A          - Linear constraint matrix
    % lb, ub     - Linear constraint bounds
    % zOpt       - Optimal solution
    % x_final    - Final desired state
    % N          - Number of time steps
    % num_state  - Number of states
    % num_control- Number of controls

    % OUTPUTS:
    % violations - Struct containing the values of constraint violations
    % objValue   - True objective value (tracking error + control effort)

    %% Initialize violations structure
    violations = struct();
    
    % 1. Linear constraints: lb <= Ax <= ub
    Ax = A * zOpt;
    violations.lb_violation = max(lb - Ax, 0); % Positive values indicate violation of lb
    violations.ub_violation = max(Ax - ub, 0); % Positive values indicate violation of ub

    % 2. Complementarity constraints: Lx >= 0, Rx >= 0, x'L'Rx = 0
    Lx = L * zOpt;
    Rx = R * zOpt;
    violations.L_violation = min(Lx, 0); % Negative values indicate violation of Lx >= 0
    violations.R_violation = min(Rx, 0); % Negative values indicate violation of Rx >= 0
    violations.complementarity_violation = zOpt' * L' * R * zOpt; % Should be close to 0

    %% Compute true objective value
    % Extract state and control variables from zOpt
    xOpt = zeros(num_state, N);
    uOpt = zeros(num_control, N);
    for k = 1:N
        idx = (k-1)*(num_state+num_control)+1;
        xOpt(:,k) = zOpt(idx:idx+num_state-1);
        uOpt(:,k) = zOpt(idx+num_state:idx+num_state+num_control-1);
    end

    % Tracking error
    tracking_error = xOpt(:,end) - x_final;
    tracking_cost = tracking_error' * Q_state * tracking_error;

    % Control effort
    control_cost = 0;
    for k = 1:(N-1)
        u_k = uOpt(:,k);
        control_cost = control_cost + u_k' * R_control * u_k;
    end

    % Total objective value
    objValue = tracking_cost + control_cost;

    %% Print violations for debugging
    fprintf('Constraint Violations:\n');
    fprintf('  lb_violation (size %d): max %.6f\n', length(violations.lb_violation), max(violations.lb_violation));
    fprintf('  ub_violation (size %d): max %.6f\n', length(violations.ub_violation), max(violations.ub_violation));
    fprintf('  L_violation (size %d): max %.6f\n', length(violations.L_violation), max(abs(violations.L_violation)));
    fprintf('  R_violation (size %d): max %.6f\n', length(violations.R_violation), max(abs(violations.R_violation)));
    fprintf('  Complementarity violation: %.6f\n', violations.complementarity_violation);
    fprintf('True Objective Value: %.6f\n', objValue);
end
