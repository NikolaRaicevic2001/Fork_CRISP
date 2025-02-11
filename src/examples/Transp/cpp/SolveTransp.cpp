#include "solver_core/SolverInterface.h"
// #include "common/MatlabHelper.h"
#include <chrono>
#include "math.h"

using namespace CRISP;

// Define model model parameters for cart transpotation
const scalar_t m1 = 1.0;
const scalar_t m2 = 2.0;
const scalar_t mu = 0.2;
const scalar_t g = 9.81;
const scalar_t l = 1;
const scalar_t dt = 0.02;
const size_t N = 200; // number of time steps

const size_t num_state = 6;
const size_t num_control = 2;

const size_t num_dynamic_constraints_per_step = 5;

// all states = [x1, x2, x1_dot, x2_dot, v, w, f, u]
// Define the dynamics:
ad_function_t cartTranspDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y)
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
        ad_scalar_t f_i = x[idx + 6];
        ad_scalar_t u_i = x[idx + 7];

        ad_scalar_t x1_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t x2_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t x1_dot_next = x[idx + (num_state + num_control) + 2];
        ad_scalar_t x2_dot_next = x[idx + (num_state + num_control) + 3];
        ad_scalar_t v_next = x[idx + (num_state + num_control) + 4];
        ad_scalar_t w_next = x[idx + (num_state + num_control) + 5];
        ad_scalar_t f_next = x[idx + (num_state + num_control) + 6];
        ad_scalar_t u_next = x[idx + (num_state + num_control) + 7];

        ad_scalar_t x1_dot_dot = (1/m1) * f_i;
        ad_scalar_t x2_dot_dot = (1/m2) * (u_i - f_i);

        y.segment(i * num_dynamic_constraints_per_step, num_dynamic_constraints_per_step) << x1_next - x1_i - x1_dot_next * dt,
                                                                                            x2_next - x2_i - x2_dot_next * dt,
                                                                                            x1_dot_next - x1_dot_i - x1_dot_dot * dt,
                                                                                            x2_dot_next - x2_dot_i - x2_dot_dot * dt,
                                                                                            x1_dot_i - x2_dot_i - v_i + w_i;
    }
};

// Define contact constraints for cart transpotation
ad_function_t cartTranspContactConstraints = [](const ad_vector_t& x, ad_vector_t& y)
{
    y.resize(N * 9);
    for (size_t i = 0; i < N; ++i)
    {
        size_t idx = i * (num_state + num_control);

        ad_scalar_t x1_i = x[idx + 0];
        ad_scalar_t x2_i = x[idx + 1];
        ad_scalar_t x1_dot_i = x[idx + 2];
        ad_scalar_t x2_dot_i = x[idx + 3];
        ad_scalar_t v_i = x[idx + 4];
        ad_scalar_t w_i = x[idx + 5];
        ad_scalar_t f_i = x[idx + 6];
        ad_scalar_t u_i = x[idx + 7];

        y.segment(i * 9, 9) << v_i,
                            w_i,
                            -v_i * w_i,
                            mu * m1 * g - f_i,
                            f_i + mu * m1 *g,
                            -w_i * (mu * m1 * g - f_i),
                            -v_i * (f_i + mu * m1 * g),
                            x1_i - x2_i + l,
                            l - (x1_i - x2_i);
    }
};

// Define initial constraints for cart transpotation
ad_function_with_param_t cartTranspInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y)
{
    y.resize(num_state);
    y.segment(0, num_state) << x[0] - p[0],
                    x[1] - p[1],
                    x[2] - p[2],
                    x[3] - p[3],
                    x[4] - p[4],
                    x[5] - p[5];
};

// Define objective
ad_function_with_param_t cartTranspObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y)
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
        ad_scalar_t f_i = x[idx + 6];
        ad_scalar_t u_i = x[idx + 7];

        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100000;
        Q(1, 1) = 100000;
        Q(2, 2) = 10;
        Q(3, 3) = 10;
        Q(4, 4) = 0;
        Q(5, 5) = 0;

        ad_matrix_t R(num_control, num_control);
        R.setZero();
        R(0, 0) = 0.0001;
        R(1, 1) = 0.0001;
        if (i == N - 1)
        {
            ad_vector_t tracking_error(num_state);
            tracking_error << x1_i - p[0],
                            x2_i - p[1],
                            x1_dot_i - p[2],
                            x2_dot_i - p[3],
                            v_i - p[4],
                            w_i - p[5];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }
        if (i < N - 1)
        {
            ad_vector_t control_error(num_control);
            control_error << f_i,
                            u_i;
            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};


int main()
{
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "CartTransp";
    std::string folderName = "model";
    OptimizationProblem cartTranspProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "cartTranspObjective", cartTranspObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "cartTranspDynamicConstraints", cartTranspDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "cartTranspContactConstraints", cartTranspContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "cartTranspInitialConstraints", cartTranspInitialConstraints);

    cartTranspProblem.addObjective(obj);
    cartTranspProblem.addEqualityConstraint(dynamics);
    cartTranspProblem.addEqualityConstraint(initial);
    cartTranspProblem.addInequalityConstraint(contact);

    SolverParameters params;
    SolverInterface solver(cartTranspProblem, params);
    // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 1));
    // solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
    // solver.setHyperParameters("muMax", vector_t::Constant(1, 1e8));

    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    // zero initial guess
    xInitialGuess.setZero();
    scalar_t x2_initial = 3.0;  
    scalar_t x2_final = 0.0; 
    vector_t xInitial(num_state); 
    xInitial<< x2_initial + 0.5, x2_initial, -4.0, -4.0, 0.0, 0.0; // initial state: cargo pose, cart pose, cargo vel, cart vel
    vector_t xFinal(num_state); 
    xFinal << x2_final - 0.5, x2_final, -2.0, -2.0, 0.0, 0.0;
    solver.setProblemParameters("cartTranspInitialConstraints", xInitial);
    solver.setProblemParameters("cartTranspObjective", xFinal);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();
}