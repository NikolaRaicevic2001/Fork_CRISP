#include "solver_core/SolverInterface.h"
#include "common/MatlabHelper.h"
#include <chrono>
#include "math.h"

using namespace ContactSolver;

// Define model model parameters for cart transpotation
const scalar_t m1 = 2.0;
const scalar_t m2 = 1.0;
const scalar_t mu1 = 0.1;
const scalar_t mu2 = 1;
const scalar_t g = 9.81;
const scalar_t l = 7;
const scalar_t dt = 0.05;
const size_t N = 100; // number of time steps
const scalar_t pusherSize = 0.1;

const size_t num_state = 8;
const size_t num_control = 5;

const size_t num_dynamic_constraints_per_step = 7;

// all states = [x1, x2, x1_dot, x2_dot, v, w, p,q, lambdaN, u, lambdaf,lambdap, plateN]
// Define the dynamics:
ad_function_t waiterDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y)
{
    y.resize((N - 1) * num_dynamic_constraints_per_step);
    for (size_t i = 0; i < N - 1; ++i)
    {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t x1_i = x[idx + 0];
        ad_scalar_t x2_i = x[idx + 1];
        ad_scalar_t x1_dot_i = x[idx + 2];
        ad_scalar_t x2_dot_i = x[idx + 3];
        ad_scalar_t v_i = x[idx + 4];
        ad_scalar_t w_i = x[idx + 5];
        ad_scalar_t p_i = x[idx + 6];
        ad_scalar_t q_i = x[idx + 7];
        ad_scalar_t lambdaN_i = x[idx + 8];
        ad_scalar_t u_i = x[idx + 9];
        ad_scalar_t lambdaf_i = x[idx + 10];
        ad_scalar_t lambdap_i = x[idx + 11];
        ad_scalar_t plateN_i = x[idx + 12];

        ad_scalar_t x1_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t x2_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t x1_dot_next = x[idx + (num_state + num_control) + 2];
        ad_scalar_t x2_dot_next = x[idx + (num_state + num_control) + 3];
        ad_scalar_t v_next = x[idx + (num_state + num_control) + 4];
        ad_scalar_t w_next = x[idx + (num_state + num_control) + 5];
        ad_scalar_t p_next = x[idx + (num_state + num_control) + 6];
        ad_scalar_t q_next = x[idx + (num_state + num_control) + 7];
        ad_scalar_t lambdaN_next = x[idx + (num_state + num_control) + 8];
        ad_scalar_t u_next = x[idx + (num_state + num_control) + 9];
        ad_scalar_t lambdaf_next = x[idx + (num_state + num_control) + 10];
        ad_scalar_t lambdap_next = x[idx + (num_state + num_control) + 11];
        ad_scalar_t plateN_next = x[idx + (num_state + num_control) + 12];

        ad_scalar_t x1_dot_dot = (1/m1) * (lambdaf_i - lambdap_i);
        ad_scalar_t x2_dot_dot = (1/m2) * (u_i - lambdaf_i);

        y.segment(i * num_dynamic_constraints_per_step, num_dynamic_constraints_per_step) << x1_next - x1_i - x1_dot_next * dt,
                                                                                            x2_next - x2_i - x2_dot_next * dt,
                                                                                            x1_dot_next - x1_dot_i - x1_dot_dot * dt,
                                                                                            x2_dot_next - x2_dot_i - x2_dot_dot * dt,
                                                                                            x2_dot_i - x1_dot_i - p_i + q_i,
                                                                                            x1_dot_i - v_i + w_i,
                                                                                            plateN_i + lambdaN_i -m1*g;
    }
    std::cout << "dynamic constraints" << std::endl;
};

// Define contact constraints for cart transpotation
ad_function_t waiterContactConstraints = [](const ad_vector_t& x, ad_vector_t& y)
{
    y.resize(N * 19);
    for (size_t i = 0; i < N; ++i)
    {
        size_t idx = i * (num_state + num_control);

        ad_scalar_t x1_i = x[idx + 0];
        ad_scalar_t x2_i = x[idx + 1];
        ad_scalar_t x1_dot_i = x[idx + 2];
        ad_scalar_t x2_dot_i = x[idx + 3];
        ad_scalar_t v_i = x[idx + 4];
        ad_scalar_t w_i = x[idx + 5];
        ad_scalar_t p_i = x[idx + 6];
        ad_scalar_t q_i = x[idx + 7];
        ad_scalar_t lambdaN_i = x[idx + 8];
        ad_scalar_t u_i = x[idx + 9];
        ad_scalar_t lambdaf_i = x[idx + 10];
        ad_scalar_t lambdap_i = x[idx + 11];
        ad_scalar_t plateN_i = x[idx + 12];


        y.segment(i * 19, 19) << v_i,
                                w_i,
                                p_i,
                                q_i,
                                plateN_i,
                                lambdaN_i,
                                x2_i-pusherSize,
                                l - (x2_i-x1_i),
                                m1*g*l - lambdaN_i*(x2_i - x1_i + l),
                                -v_i * w_i, 
                                -p_i * q_i,
                                plateN_i * mu1 - lambdap_i,
                                lambdap_i + mu1 * plateN_i,
                                mu2*lambdaN_i - lambdaf_i,
                                lambdaf_i + mu2*lambdaN_i,
                                -v_i * (plateN_i * mu1 - lambdap_i),
                                -w_i * (lambdap_i + mu1 * plateN_i),
                                -p_i * (mu2*lambdaN_i - lambdaf_i),
                                -q_i * (lambdaf_i + mu2*lambdaN_i);
    }
    std::cout << "contact constraints" << std::endl;
};


// Define initial constraints for cart transpotation
ad_function_with_param_t waiterInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y)
{
    y.resize(num_state);
    y.segment(0, num_state) << x[0] - p[0],
                    x[1] - p[1],
                    x[2] - p[2],
                    x[3] - p[3],
                    x[4] - p[4],
                    x[5] - p[5],
                    x[6] - p[6],
                    x[7] - p[7];
    std::cout << "initial constraints" << std::endl;
};

// Define objective
ad_function_with_param_t waiterObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y)
{
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i)
    {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t x1_i = x[idx + 0];
        ad_scalar_t x2_i = x[idx + 1];
        ad_scalar_t x1_dot_i = x[idx + 2];
        ad_scalar_t x2_dot_i = x[idx + 3];
        ad_scalar_t v_i = x[idx + 4];
        ad_scalar_t w_i = x[idx + 5];
        ad_scalar_t p_i = x[idx + 6];
        ad_scalar_t q_i = x[idx + 7];
        ad_scalar_t lambdaN_i = x[idx + 8];
        ad_scalar_t u_i = x[idx + 9];
        ad_scalar_t lambdaf_i = x[idx + 10];
        ad_scalar_t lambdap_i = x[idx + 11];
        ad_scalar_t plateN_i = x[idx + 12];


        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        Q(2, 2) = 100;
        Q(3, 3) = 100; 
        if (i == N - 1)
        {
            ad_vector_t tracking_error(num_state);
            tracking_error << x1_i - p[0],
                            x2_i - p[1],
                            x1_dot_i - p[2],
                            x2_dot_i - p[3],
                            v_i - p[4],
                            w_i - p[5],
                            p_i - p[6],
                            q_i - p[7];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }
        ad_matrix_t R(num_control, num_control);
        R.setZero();
        R(0, 0) = 0.0001;
        R(1, 1) = 0.0001;

        if (i < N - 1)
        {
            ad_vector_t control_error(num_control);
            control_error << lambdaN_i,
                            u_i,
                            lambdaf_i,
                            lambdap_i,
                            plateN_i;

            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
    std::cout << "objective function" << std::endl;
};


int main()
{
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "WaiterProblem";
    std::string folderName = "model";
    OptimizationProblem WaiterProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "WaiterObjective", waiterObjective, true);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "WaiterDynamicConstraints", waiterDynamicConstraints, true);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName,  "WaiterContactConstraints", waiterContactConstraints, true);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "WaiterInitialConstraints", waiterInitialConstraints, true);

    WaiterProblem.addObjective(obj);
    WaiterProblem.addEqualityConstraint(dynamics);
    WaiterProblem.addEqualityConstraint(initial);
    WaiterProblem.addInequalityConstraint(contact);
    
    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    xInitialGuess.setZero();
    // define the initial states
    xInitialStates << -6.0, 0.1, 0.0, 0.0, 0, 0, 0, 0;
    // // set random initial guess beteween -0.01 and 0.01
    // vector_t lambdaN_initial(N);
    // lambdaN_initial.setConstant(10.0);
    // xInitialGuess.segment(0, 8) = xInitialStates;
    // for (size_t i = 0; i < N; ++i) {
    //     size_t idx = i * (num_state + num_control);
    //     xInitialGuess[idx + 8] = lambdaN_initial[i];
    // }
    // xInitialGuess.segment(0, 8) = xInitialStates;
    xFinalStates << pusherSize, pusherSize, 2.0, 2.0, 0, 0, 0, 0;

    SolverParameters params;
    SolverInterface solver(WaiterProblem, params);
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    // solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-4));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-3));
    // solver.setHyperParameters("muMax", vector_t::Constant(1, 1e8));
    solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
    solver.setProblemParameters("WaiterInitialConstraints", xInitialStates);
    solver.setProblemParameters("WaiterObjective", xFinalStates);

    solver.initialize(xInitialGuess);
    solver.solve();
    solver.getSolution();
    solver.saveResults("/home/workspace/src/examples/server/experiments/results_exp");
}    




