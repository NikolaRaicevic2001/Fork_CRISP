function [time_solve, xOpt, uOpt, stats,violation,objvalue] = solveLCQP_CartProblem(x_init, x_final, N, dt)
    %% Parameters
    m1 =1.0; m2 = 2.0; g_const = 9.81; mu = 0.2; l=1.0;
    num_dynamic_constraints_per_step = 5;
    %% Cost matrices
    % Q_state = diag([1, 1, 1, 1, eps, eps]);
    Q_state = eps * ones(6);
    Q_state(1,1) = 1e5;Q_state(2,2) = 1e5;Q_state(3,3) = 10;Q_state(4,4) = 10;
    R_control = eps * ones(4);
    R_control(1,1) = 1e-4; R_control(2,2) = 1e-4;

    %% Dimensions: add two slack variables for the complementarity constraints.
    n_state = 6; n_control = 4; n_var = (n_state + n_control)*N;
    n_eq = n_state + (num_dynamic_constraints_per_step + 4) *(N-1) + 4; % dynamics + initial + relative distance of x1 x2 + slack constraints
    n_compl = 3*N; % complementarity constraints per timestep

    %% Initialize matrices
    Q = eps*ones(n_var); g = zeros(n_var,1);
    A = zeros(n_eq, n_var); lb = zeros(n_eq,1); ub = zeros(n_eq,1);
    L = zeros(n_compl, n_var); R = zeros(n_compl, n_var);
    % lbL = zeros(n_compl,1); ubL = inf*ones(n_compl,1);
    % lbR = zeros(n_compl,1); ubR = inf*ones(n_compl,1);


    %% Build objective matrices Q and g
    for k = 1:N
        idx_state = (k-1)*(n_state+n_control)+1 : (k-1)*(n_state+n_control)+n_state;       
        idx_control = idx_state(end)+1 : idx_state(end)+n_control;
        if k == N
            Q(idx_state, idx_state) = Q_state;
            g(idx_state) = -x_final' * Q_state;
        end
        if k < N
            Q(idx_control, idx_control) = R_control;
        end
         % Linear term due to target state
    end

    %% linear constraints: initial, (dynamics + relative distance of x1 x2) * (N-1), relative distance of x1 x2
    eq_idx = 1;
    for k = 1:N
        idx_k = (k-1)*(n_state+n_control)+1;
        idx_k1 = idx_k + n_state + n_control;

        % States at current and next timestep
        xk = idx_k : idx_k+n_state-1;
        uk = idx_k+n_state : idx_k+n_state+n_control-1;
        xk1 = idx_k1 : idx_k1+n_state-1;
        % initial condition
        if k == 1
            A(xk, xk) = eye(n_state);
            lb(xk) = x_init;
            ub(xk) = x_init;
            eq_idx = eq_idx + n_state;
        end

        % Dynamics equations (Euler discretization) and relative distance constraints and slack constraints
        if k < N
            % dynamics 
            A(eq_idx, xk1(1)) = 1; A(eq_idx, xk(1)) = -1; A(eq_idx, xk1(3)) = -dt;
            eq_idx = eq_idx +1;
            A(eq_idx, xk1(2)) = 1; A(eq_idx, xk(2)) = -1; A(eq_idx, xk1(4)) = -dt;
            eq_idx = eq_idx +1;

            A(eq_idx, xk1(3)) = 1; A(eq_idx, xk(3)) = -1; A(eq_idx, uk(1)) = -dt/m1;
            eq_idx = eq_idx +1;
            A(eq_idx, xk1(4)) = 1; A(eq_idx, xk(4)) = -1; A(eq_idx, uk(2)) = -dt/m2; A(eq_idx, uk(1)) = dt/m2;
            eq_idx = eq_idx +1;

            A(eq_idx, xk(3)) = 1; A(eq_idx, xk(4)) = -1; A(eq_idx, xk(5)) = -1; A(eq_idx, xk(6)) = 1;
            eq_idx = eq_idx +1;

            % relative distance inequality constraints
            %  x1 -x2 > = -l; -x1 + x2 >= -l
            A(eq_idx, xk(1)) = 1; A(eq_idx, xk(2)) = -1; lb(eq_idx) = -l; ub(eq_idx) = inf;
            eq_idx = eq_idx +1;
            A(eq_idx, xk(1)) = -1; A(eq_idx, xk(2)) = 1; lb(eq_idx) = -l; ub(eq_idx) = inf;
            eq_idx = eq_idx +1;
            % slack constraints: mu*m1*g - f = s1; f + mu*m1*g = s2
            A(eq_idx, uk(3)) = 1; A(eq_idx, uk(1)) = 1; lb(eq_idx) = mu*m1*g_const; ub(eq_idx) = mu*m1*g_const;
            eq_idx = eq_idx +1;
            A(eq_idx, uk(4)) = 1; A(eq_idx, uk(1)) = -1; lb(eq_idx) = mu*m1*g_const; ub(eq_idx) = mu*m1*g_const;
            eq_idx = eq_idx +1;
        end

        if k == N
            % relative distance inequality constraints
            %  x1 -x2 > = -l; -x1 + x2 >= -l
            A(eq_idx, xk(1)) = 1; A(eq_idx, xk(2)) = -1; lb(eq_idx) = -l; ub(eq_idx) = inf;
            eq_idx = eq_idx +1;
            A(eq_idx, xk(1)) = -1; A(eq_idx, xk(2)) = 1; lb(eq_idx) = -l; ub(eq_idx) = inf;
            eq_idx = eq_idx +1;
            % slack constraints: mu*m1*g - f = s1; f + mu*m1*g = s2
            A(eq_idx, uk(3)) = 1; A(eq_idx, uk(1)) = 1; lb(eq_idx) = mu*m1*g_const; ub(eq_idx) = mu*m1*g_const;
            eq_idx = eq_idx +1;
            A(eq_idx, uk(4)) = 1; A(eq_idx, uk(1)) = -1; lb(eq_idx) = mu*m1*g_const; ub(eq_idx) = mu*m1*g_const;
        end
    end


    %% Complementarity constraints
    compl_idx = 1;
    for k = 1:N
        idx = (k-1)*(n_state+n_control)+1;
        v_idx = idx+4; w_idx = idx+5; s1_idx = idx + 8; s2_idx = idx + 9;

        % 0 <= v ⊥ w >= 0
        L(compl_idx,v_idx)=1; R(compl_idx,w_idx)=1; compl_idx=compl_idx+1;

        % 0 <= mu*m1*g - f ⊥ w >= 0
        L(compl_idx,s1_idx)=1; R(compl_idx,w_idx)=1; compl_idx=compl_idx+1;
        
        
        % 0 <= f + mu*m1*g ⊥ v >= 0
        L(compl_idx,s2_idx)=1; R(compl_idx,v_idx)=1; compl_idx=compl_idx+1;
    end

    %% Solve LCQP
    % params = struct('maxIter', 500, 'tol', 1e-6);
    % Algorithm parameters
    params.x0 = zeros(n_var,1); 
    params.initialPenaltyParameter = 0.001;
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
    tic;
    [zOpt,~,stats] = LCQPow(Q,g,L,R,[],[],[],[],A,lb,ub,params);
    time_solve = toc;
    [violation, objvalue] = checkSolutionFeasibility(Q_state, R_control, L, R, A, lb, ub,zOpt,x_final,N,n_state,n_control);
    % [xOpt,~,stats] = LCQPow(Q, g, L, R, [], [], [], [], [], [], [], params);
    % disp(stats.qp_exit_flag);
    %% Extract results
    xOpt = zeros(n_state,N);
    uOpt = zeros(n_control,N);
    for k = 1:N
        idx = (k-1)*(n_state+n_control)+1;
        xOpt(:,k) = zOpt(idx:idx+n_state-1);
        uOpt(:,k) = zOpt(idx+n_state:idx+n_state+n_control-1);
    end
    time = dt*(0:N-1);
end
