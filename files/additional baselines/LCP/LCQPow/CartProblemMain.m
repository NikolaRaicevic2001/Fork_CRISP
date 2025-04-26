clear;
addpath("/home/workspace/LCQPow/build/lib/");

% Example usage:
N = 200; dt = 0.02;
% construct different initial and final states
x2_initial = 3;
x2_final = 0;
x1_initial = [x2_initial,x2_initial-0.5,x2_initial+0.5];
x1_final = [x2_final,x2_final-0.5,x2_final+0.5];
final_state = [];
initial_state = [];
for i = 1:3
    for j = 1:3
        initial = [x1_initial(i),x2_initial,0,0,0,0];
        initial_state =[initial_state;initial];
        final = [x1_final(j),x2_final,0,0,0,0];
        final_state = [final_state;final];
        initial = [x1_initial(i),x2_initial,-3,-3,0,0];
        initial_state =[initial_state;initial];
        final = [x1_final(j),x2_final,0,0,0,0];
        final_state = [final_state;final];
    end
end
for i = 1:3
    for j = 1:3
        initial = [x1_initial(i),x2_initial,-4,-4,0,0];
        initial_state =[initial_state;initial];
        final = [x1_final(j),x2_final,-2,-2,0,0];
        final_state = [final_state;final];
    end
end
trackingerror = [];
    % fprintf('  lb_violation (size %d): max %.6f\n', length(violations.lb_violation), max(violations.lb_violation));
    % fprintf('  ub_violation (size %d): max %.6f\n', length(violations.ub_violation), max(violations.ub_violation));
    % fprintf('  L_violation (size %d): max %.6f\n', length(violations.L_violation), max(abs(violations.L_violation)));
    % fprintf('  R_violation (size %d): max %.6f\n', length(violations.R_violation), max(abs(violations.R_violation)));
    % fprintf('  Complementarity violation: %.6f\n', violations.complementarity_violation);
violation.lb_violation = [];
violation.ub_violation = [];
violation.L_violation = [];
violation.R_violation = [];
violation.complementarity_violation = [];
solvetime = [];
for k = 1:size(final_state,1)
    [time, xOpt, uOpt, stats,violations,objvalue] = solveLCQP_CartProblem(initial_state(k,:)', final_state(k,:)', N, dt);
    xend = xOpt(:,end);
    error = final_state(k,1:4)' - xend(1:4);
    errornorm = norm(error);
    trackingerror = [trackingerror;errornorm];
    solvetime = [solvetime;time];
    violation.lb_violation = [violation.lb_violation;max(violations.lb_violation)];
    violation.ub_violation = [violation.ub_violation;max(violations.ub_violation)];
    violation.L_violation = [violation.L_violation;max(abs(violations.L_violation))];
    violation.R_violation = [violation.R_violation;max(abs(violations.R_violation))];
    violation.complementarity_violation = [violation.complementarity_violation;violations.complementarity_violation];

end


% % % Plot results
% figure;
% subplot(3,1,1); plot(time,xOpt(1:2,:)'); legend('x1','x2'); title('Positions');
% subplot(3,1,2); plot(time,xOpt(3:4,:)'); legend('dx1','dx2'); title('Velocities');
% subplot(3,1,3); plot(time,uOpt(1:2,:)'); legend('f','u'); title('Control Inputs');
