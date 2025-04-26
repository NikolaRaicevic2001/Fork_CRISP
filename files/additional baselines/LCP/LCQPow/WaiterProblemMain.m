clear;
addpath("/home/workspace/LCQPow/build/lib/");

% Example usage:
N = 100; dt = 0.05;
x_init = [-6; 0.1; 0; 0; 0; 0;0;0];
x_final = [0.1; 0.1; 2; 2; 0; 0;0;0];

[time, xOpt, uOpt, stats, violation, objvalue] = solveLCQP_WaiterProblem(x_init, x_final, N, dt);

% % Plot results
figure;
subplot(3,1,1); plot(time,xOpt(1:2,:)'); legend('x1','x2'); title('Positions');
subplot(3,1,2); plot(time,xOpt(3:4,:)'); legend('dx1','dx2'); title('Velocities');
subplot(3,1,3); plot(time,uOpt(1:2,:)'); legend('f','u'); title('Control Inputs');
