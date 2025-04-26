function [time, xOpt, uOpt, stats, violation, objvalue] = solveLCQP_WaiterProblem(x_init, x_final, N, dt)
    %% Parameters
    m1 = 2.0; m2 = 1.0; g_const = 9.81; mu1 = 0.1; mu2 = 1; l0 = 7.0;
    pusherSize = 0.1;
    num_state = 8; % [x1, x2, x1_dot, x2_dot, v, w, p, q]
    num_control = 5; % [lambda_N, u, fp, ft, N]
    num_dynamic_constraints_per_step = 7; % equality
    num_contact_constraints_per_step = 4; % inequality
    num_complementarity_constraints_per_step = 6;

    %% Dimensions
    n_var = (num_state + num_control) * N;
    neq = num_dynamic_constraints_per_step * (N-1) + num_state + 1; % Dynamics + Initial constraints
    nin = num_contact_constraints_per_step * N; % Inequality constraints
    n_compl = num_complementarity_constraints_per_step * N; % Complementarity constraints

    %% Cost Matrices
    Q_state = eps * ones(num_state); % Penalize state deviation
    Q_state(1,1) = 100; Q_state(2,2) = 100; Q_state(3,3) = 100; Q_state(4,4) = 100;
    R_control = eps * ones(num_control); % Penalize control effort
    R_control(1,1) = 0.01; R_control(2,2) = 0.01;

    %% Initialize Matrices
    Q = eps * ones(n_var); g = zeros(n_var,1);
    A = zeros(neq + nin, n_var); lb = zeros(neq + nin,1); ub = zeros(neq + nin,1);
    L = zeros(n_compl, n_var); R = zeros(n_compl, n_var);

    %% Build Objective Matrices Q and g
    for k = 1:N
        idx_state = (k-1)*(num_state+num_control)+1 : (k-1)*(num_state+num_control)+num_state;       
        idx_control = idx_state(end)+1 : idx_state(end)+num_control;
        if k == N
            Q(idx_state, idx_state) = Q_state;
            g(idx_state) = -x_final' * Q_state;
        end
        if k < N
            Q(idx_control, idx_control) = R_control;
        end
    end

    %% Linear Constraints: Dynamics, Initial, Slack Constraints
    eq_idx = 1;
    for k = 1:N
        idx_k = (k-1)*(num_state+num_control)+1;
        idx_k1 = idx_k + num_state + num_control;

        % States at current and next timestep
        xk = idx_k : idx_k+num_state-1;
        uk = idx_k+num_state : idx_k+num_state+num_control-1;
        xk1 = idx_k1 : idx_k1+num_state-1;

        % Initial condition
        if k == 1
            A(xk, xk) = eye(num_state);
            lb(xk) = x_init;
            ub(xk) = x_init;
            eq_idx = eq_idx + num_state;
        end

        % Dynamics equations and slack constraints
        if k < N
            % Dynamics: Euler discretization
            A(eq_idx, xk1(1)) = 1; A(eq_idx, xk(1)) = -1; A(eq_idx, xk1(3)) = -dt;
            eq_idx = eq_idx + 1;
            A(eq_idx, xk1(2)) = 1; A(eq_idx, xk(2)) = -1; A(eq_idx, xk1(4)) = -dt;
            eq_idx = eq_idx + 1;

            A(eq_idx, xk1(3)) = 1; A(eq_idx, xk(3)) = -1; A(eq_idx, uk(3)) = -dt/m1;A(eq_idx, uk(4)) = dt/m1;
            eq_idx = eq_idx + 1;
            A(eq_idx, xk1(4)) = 1; A(eq_idx, xk(4)) = -1; A(eq_idx, uk(2)) = -dt/m2; A(eq_idx, uk(3)) = dt/m2;
            eq_idx = eq_idx + 1;

            A(eq_idx, xk(4)) = 1; A(eq_idx, xk(3)) = -1; A(eq_idx, xk(8)) = 1; A(eq_idx, xk(7)) = -1;
            eq_idx = eq_idx + 1;

            A(eq_idx, xk(3)) = 1; A(eq_idx, xk(5)) = -1; A(eq_idx, xk(6)) = 1;
            eq_idx = eq_idx + 1;

            A(eq_idx, uk(5)) = 1; A(eq_idx, uk(1)) = 1; lb(eq_idx) = m1 * g_const; ub(eq_idx) = m1 * g_const;
            eq_idx = eq_idx + 1;

            % inequalites
            A(eq_idx, xk(2)) = 1; % x2 >= pusherSize
            lb(eq_idx) = pusherSize; ub(eq_idx) = inf;
            eq_idx = eq_idx + 1;

            A(eq_idx, uk(1)) = 1; % lambdaN >= 0, <= 10
            ub(eq_idx) = 10;
            eq_idx = eq_idx + 1;

            A(eq_idx, uk(5)) = 1; % N >= 0
            ub(eq_idx) = inf;
            eq_idx = eq_idx + 1;

            A(eq_idx, xk(2)) = -1; A(eq_idx, xk(1)) = 1; % l - (x2 - x1) >= 0
            lb(eq_idx) = -l0; ub(eq_idx) = inf;
            eq_idx = eq_idx + 1;

        end
        if k == N
            A(eq_idx, uk(5)) = 1; A(eq_idx, uk(1)) = 1; lb(eq_idx) = m1 * g_const; ub(eq_idx) = m1 * g_const;
            eq_idx = eq_idx + 1;
            % inequalites
            A(eq_idx, xk(2)) = 1; % x2 >= pusherSize
            lb(eq_idx) = pusherSize; ub(eq_idx) = inf;
            eq_idx = eq_idx + 1;

            A(eq_idx, uk(1)) = 1; % lambdaN >= 0, <= 10
            ub(eq_idx) = 10;
            eq_idx = eq_idx + 1;

            A(eq_idx, uk(5)) = 1; % N >= 0
            ub(eq_idx) = inf;
            eq_idx = eq_idx + 1;

            A(eq_idx, xk(2)) = -1; A(eq_idx, xk(1)) = 1; % l - (x2 - x1) >= 0
            lb(eq_idx) = -l0; ub(eq_idx) = inf;
        end

    end

    %% Complementarity Constraints
    compl_idx = 1;
    for k = 1:N
        idx_k = (k-1)*(num_state+num_control)+1;
        z_idx = idx_k + 4; w_idx = idx_k + 5; p_idx = idx_k + 6; q_idx = idx_k + 7;
        % States at current and next timestep
        uk = idx_k+num_state : idx_k+num_state+num_control-1;

        % Complementarity conditions
        L(compl_idx, z_idx) = 1; R(compl_idx, w_idx) = 1; % 0 <= z ⊥ w >= 0
        compl_idx = compl_idx + 1;

        L(compl_idx, p_idx) = 1; R(compl_idx, q_idx) = 1; % 0 <= p ⊥ q >= 0
        compl_idx = compl_idx + 1;

        % Friction-related complementarity
        L(compl_idx, z_idx) = 1; R(compl_idx, uk(5)) = mu1;R(compl_idx,uk(4)) = -1; % 0 <= plateN*mu1 - lambdap ⊥ z >= 0
        compl_idx = compl_idx + 1;
        % w \perp lambdap_i + mu1*plateN
        L(compl_idx, w_idx) = 1; R(compl_idx, uk(4)) = 1; R(compl_idx, uk(5)) = mu1; % 0 <= lambdap + mu1*plateN ⊥ w >= 0
        compl_idx = compl_idx + 1;
        L(compl_idx, p_idx) = 1; R(compl_idx, uk(1)) = mu2; R(compl_idx, uk(3)) = -1; % 0 <= mu2*lambdaN - lambdaf ⊥ p >= 0
        compl_idx = compl_idx + 1;
        L(compl_idx, q_idx) = 1; R(compl_idx, uk(3)) = 1; R(compl_idx, uk(1)) = mu2; % 0 <= lambdaf + mu2*lambdaN ⊥ q >= 0
        compl_idx = compl_idx + 1;
    end

    %% Solve LCQP
    params.x0 = zeros(n_var,1); 
    params.initialPenaltyParameter = 0.01;
    params.penaltyUpdateFactor = 2;
    params.solveZeroPenaltyFirst = true;
    params.printLevel = 2;
    % Use sparse solver
    params.qpSolver = 1;
    if (~issparse(Q))
        Q = sparse(Q);
        L = sparse(L);
        R = sparse(R);
        A = sparse(A);
    end

    [zOpt,~,stats] = LCQPow(Q, g, L, R, [], [], [], [], A, lb, ub, params);
    %% check solution quality
    [violation, objvalue] = checkSolutionFeasibility(Q_state, R_control, L, R, A, lb, ub,zOpt,x_final,N,num_state,num_control);
    %% Extract Results
    xOpt = zeros(num_state, N);
    uOpt = zeros(num_control, N);
    for k = 1:N
        idx = (k-1)*(num_state+num_control)+1;
        xOpt(:,k) = zOpt(idx:idx+num_state-1);
        uOpt(:,k) = zOpt(idx+num_state:idx+num_state+num_control-1);
    end
    time = dt * (0:N-1);
end
