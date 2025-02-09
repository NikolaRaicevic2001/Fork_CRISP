#include "solver_core/SolverInterface.h"
#include "common/MatlabHelper.h"
#include <chrono>
#include "math.h"

using namespace ContactSolver;

// Define model model parameters for pushbot
const double dt = 0.02;
const size_t N = 100; // number of time steps
const size_t num_state = 4;  
const size_t num_control = 3;
const double mc = 1.0;
const double mp = 0.1;
const double l = 0.8;
const double g = 9.8;
const double d1 = 1.0;
const double d2 = 1.0;
const double k1 = 200.0;
const double k2 = 200.0;

// allstates = [x,x_dot,theta,theta_dot,u,lamda1,lamda2]
// define the objective function handle where the final desired state is the parameter
ad_function_with_param_t pushbotObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    ad_matrix_t Q(num_state, num_state);
    Q.setZero();
    Q(0,0) = 100;
    Q(1, 1) = 100;
    Q(2, 2) = 100;
    Q(3, 3) = 100;
    ad_matrix_t R(num_control, num_control);
    R.setZero();
    R(0, 0) = 0.001;

    for (size_t i = 0; i < N; ++i) {
        ad_vector_t state(num_state);
        for (size_t j = 0; j < num_state; ++j)
            state(j) = x(i * (num_state + num_control) + j);

        ad_vector_t control(num_control);
        for (size_t j = 0; j < num_control; ++j)
            control(j) = x(i * (num_state + num_control) + num_state + j);
        
        // terminal cost
        if (i == N - 1) {
            ad_vector_t tracking_error = state - p;
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        if (i < N - 1) {
            ad_vector_t control_error = control;
            control_cost += control_error.transpose() * R * control_error;
        }
    }

    y(0) = tracking_cost + control_cost;
};



// Dynamic constraints
ad_function_t pushBotDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t x_i = x[idx + 0];
        ad_scalar_t theta_i = x[idx + 1];
        ad_scalar_t x_dot_i = x[idx + 2];
        ad_scalar_t theta_dot_i = x[idx + 3];
        ad_scalar_t u_i = x[idx + 4];
        ad_scalar_t lamda1_i = x[idx + 5];
        ad_scalar_t lamda2_i = x[idx + 6];

        ad_scalar_t x_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t theta_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t x_dot_next = x[idx + (num_state + num_control) + 2];
        ad_scalar_t theta_dot_next = x[idx + (num_state + num_control) + 3];

        ad_scalar_t x_dot_dot = (lamda2_i - lamda1_i + u_i + lamda1_i * cos(theta_i) * cos(theta_i)
                               - lamda2_i * cos(theta_i) * cos(theta_i) - g * mp * cos(theta_i) * sin(theta_i)
                               + l * mp * theta_dot_i * theta_dot_i * sin(theta_i))
                              / (-mp * cos(theta_i) * cos(theta_i) + mc + mp);

        ad_scalar_t theta_dot_dot = -(lamda1_i * mc * cos(theta_i) - lamda2_i * mc * cos(theta_i)
                                    + mp * u_i * cos(theta_i) - g * mp * mp * sin(theta_i)
                                    - g * mc * mp * sin(theta_i) + l * mp * mp * theta_dot_i * theta_dot_i
                                    * cos(theta_i) * sin(theta_i))
                                  / (l * mp * (-mp * cos(theta_i) * cos(theta_i) + mc + mp));

        // Implicit state update
        y.segment(i * num_state, num_state) << x_next - x_i - x_dot_next * dt,
                                                theta_next - theta_i - theta_dot_next * dt,
                                                x_dot_next - x_dot_i - x_dot_dot * dt,
                                                theta_dot_next - theta_dot_i - theta_dot_dot * dt;
    }
};

// contact constraints:f>=0 g >= 0, -fg>=0
ad_function_t pushBotContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 6);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);

        ad_scalar_t x_i = x[idx + 0];
        ad_scalar_t theta_i = x[idx + 1];
        ad_scalar_t lamda1_i = x[idx + 5];
        ad_scalar_t lamda2_i = x[idx + 6];

        y.segment(i * 6, 6) << lamda1_i,
                            lamda2_i,
                            d1 - x_i - l * sin(theta_i) + lamda1_i / k1,
                            d2 + x_i + l * sin(theta_i) + lamda2_i / k2,
                            -(lamda1_i * (d1 - x_i - l * sin(theta_i) + lamda1_i / k1)),
                            -(lamda2_i * (d2 + x_i + l * sin(theta_i) + lamda2_i / k2));
    }
};

// initial constraints:
ad_function_with_param_t pushBotInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(4);
    y.segment(0, 4) << x[0] - p[0],
                    x[1] - p[1],
                    x[2] - p[2],
                    x[3] - p[3];
};

int main() {
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushbotSwingUp";
    std::string folderName = "model";
    OptimizationProblem pushbotProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushbotObjective", pushbotObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushBotDynamicConstraints", pushBotDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushBotContactConstraints", pushBotContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushBotInitialConstraints", pushBotInitialConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushbotProblem.addObjective(obj);
    pushbotProblem.addEqualityConstraint(dynamics);
    pushbotProblem.addInequalityConstraint(contact);
    pushbotProblem.addEqualityConstraint(initial);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);

    // --------------------- read matlab files for initial guess --------------------- //
    // change the following matfile path to your own path
    const char* variableName = "z_record";
    size_t num_experiments = 30;
    SolverParameters params;
    SolverInterface solver(pushbotProblem, params);
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setHyperParameters("WeightedTolFactor", vector_t::Constant(1, 10));
    // solver.serHyperParameters("mu", vector_t::Constant(1, 100));
    std::string resultFolderPrefix = "/home/workspace/src/examples/pushbot/experiments/results_test/";
    for (size_t i = 1; i <= num_experiments; ++i) {
        // stop for 1s for storing the results in different names
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::string matFileName = "/home/workspace/src/examples/pushbot/experiments/initial_guess_0" + std::to_string(i) + ".mat";
        MatlabHelper::readVariableFromMatFile(matFileName.c_str(), variableName, xInitialGuess);
        xInitialStates << xInitialGuess[0], xInitialGuess[1], xInitialGuess[2], xInitialGuess[3];
        if (i == 1) {
            // initialize the solver interface with the problem
            xFinalStates << 0,0,0,0;
            solver.setProblemParameters("pushbotObjective", xFinalStates);
            solver.setProblemParameters("pushBotInitialConstraints", xInitialStates);
            solver.initialize(xInitialGuess);
            solver.solve();
            solver.getSolution();
            solver.saveResults(resultFolderPrefix);
        } else {
            // adjust parameter dynamically and re-solve the problem
            solver.setProblemParameters("pushBotInitialConstraints", xInitialStates);
            solver.resetProblem(xInitialGuess);
            solver.solve();
            solver.getSolution();
            solver.saveResults(resultFolderPrefix);
        }
    }
}
    // ------------ simple running example -------------//
    // SolverParameters params;
    // SolverInterface solver(pushbotProblem, params);
    // // set the parameters for those parametric functions: mandatory
    // solver.setProblemParameters("pushbotObjective", xFinalStates);
    // solver.setProblemParameters("pushBotInitialConstraints", xInitialStates);
    // // set the hyperparameters for the solver: optional
    // solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
    // // initialize the solve with initial guess and solve the problem
    // solver.initialize(xInitialGuess);
    // solver.solve();
    // solver.getSolution();
