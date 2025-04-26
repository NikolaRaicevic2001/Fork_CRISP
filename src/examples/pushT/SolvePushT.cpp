#include "solver_core/SolverInterface.h"
#include <chrono>
#include "math.h"

using namespace CRISP;

// Define model parameters for pushT
const scalar_t l = 0.05;
const scalar_t m = 1;
const scalar_t mu = 0.4;
const scalar_t g = 9.8;
const scalar_t r = 2.8 * l;
const scalar_t c = 0.4;
const scalar_t dc = 2.6429;
const scalar_t dt = 0.05;
const size_t N = 50; // number of time steps
const size_t num_state = 19;
const size_t num_control = 10;

// define the dynamics constraints
ad_function_t pushTDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 12 + 9);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];
        ad_scalar_t c_theta = x[idx + 27];
        ad_scalar_t s_theta = x[idx + 28];

        if (i < N-1 ){
        ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next = x[idx + (num_state + num_control) + 2];

        ad_scalar_t px_dot = (1/(mu*m*g))*(cos(theta_i)*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) - sin(theta_i)*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));
        ad_scalar_t py_dot = (1/(mu*m*g))*(sin(theta_i)*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) + cos(theta_i)*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));
        ad_scalar_t theta_dot = (1/(mu*m*g*c*r))*(-cy_i*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) + cx_i*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));

        // Explicit State Update
        y.segment(i * 12, 12) << px_next - px_i - px_dot * dt,
                                py_next - py_i - py_dot * dt,
                                theta_next - theta_i - theta_dot * dt,
                                (cx_i - 2*l) - v1_i + w1_i,
                                (cy_i - (4-dc)*l) - v2_i + w2_i,
                                (cy_i - (3-dc)*l) - v3_i + w3_i,
                                (cx_i - 0.5*l) - v4_i + w4_i,
                                (cy_i + dc*l) - v5_i + w5_i,
                                (cx_i + 0.5*l) - v6_i + w6_i,
                                (cx_i + 2*l) - v7_i + w7_i,
                                c_theta - cos(theta_i),
                                s_theta - sin(theta_i);
        }
        else{
            y.segment(i * 12, 9) << cx_i - 2*l - v1_i + w1_i,
                                    cy_i - (4-dc)*l - v2_i + w2_i,
                                    (cy_i - (3-dc)*l) - v3_i + w3_i,
                                    (cx_i - 0.5*l) - v4_i + w4_i,
                                    (cy_i + dc*l) - v5_i + w5_i,
                                    (cx_i + 0.5*l) - v6_i + w6_i,
                                    (cx_i + 2*l) - v7_i + w7_i,
                                    c_theta - cos(theta_i),
                                    s_theta - sin(theta_i);
        }
    }
};

// contact implicit constraints for pushT
ad_function_t pushTContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize(N * 41);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];

        y.segment(i * 41, 41) << v1_i,
                            w1_i,
                            -v1_i * w1_i,
                            v2_i,
                            w2_i,
                            -v2_i * w2_i,
                            v3_i,
                            w3_i,
                            -v3_i * w3_i,
                            v4_i,
                            w4_i,
                            -v4_i * w4_i,
                            v5_i,
                            w5_i,
                            -v5_i * w5_i,
                            v6_i,
                            w6_i,
                            -v6_i * w6_i,
                            v7_i,
                            w7_i,
                            -v7_i * w7_i,
                            cx_i + 2*l,
                            2*l - cx_i,
                            cy_i + dc * l,
                            (4-dc)*l - cy_i,
                            -lambda1_i,
                            -lambda2_i,
                             lambda3_i,
                            -lambda4_i,
                            lambda5_i,
                            lambda6_i,
                            lambda7_i,
                            lambda8_i,
                            -(-lambda1_i)*((4-dc)*l - cy_i),
                            -(-lambda2_i)*(v1_i + w1_i + v2_i + w2_i + v3_i + w3_i- l),
                            -(lambda3_i)*(v1_i + w1_i + v3_i + w3_i + v4_i + w4_i - 1.5*l),
                            -(-lambda4_i)*(v3_i + w3_i + v4_i + w4_i + v5_i + w5_i - 3.0*l),
                            -(lambda5_i)*(v4_i + w4_i + v5_i + w5_i + v6_i + w6_i - l),
                            -(lambda6_i)*(v3_i + w3_i + v5_i + w5_i + v6_i + w6_i - 3.0*l),
                            -(lambda7_i)*(v3_i + w3_i + v6_i + w6_i + v7_i + w7_i - 1.5*l),
                            -(lambda8_i)*(v2_i + w2_i + v3_i + w3_i + v7_i + w7_i - l);
    }
};

// allow only one contact force at a time
ad_function_t pushTContactSingleForceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 28);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];

        y.segment(i * 28, 28) << -(-lambda1_i * (-lambda2_i)),
                            -(-lambda1_i * lambda3_i),
                            -(-lambda1_i * (-lambda4_i)),
                            -(-lambda1_i * lambda5_i),
                            -(-lambda1_i * (lambda6_i)),
                            -(-lambda1_i * (lambda7_i)),
                            -(-lambda1_i * (lambda8_i)),
                            -(-lambda2_i * lambda3_i),
                            -(-lambda2_i * (-lambda4_i)),
                            -(-lambda2_i * lambda5_i),
                            -(-lambda2_i * (lambda6_i)),
                            -(-lambda2_i * (lambda7_i)),
                            -(-lambda2_i * (lambda8_i)),
                            -(lambda3_i * (-lambda4_i)),
                            -(lambda3_i * lambda5_i),
                            -(lambda3_i * (lambda6_i)),
                            -(lambda3_i * (lambda7_i)),
                            -(lambda3_i * (lambda8_i)),
                            -(-lambda4_i * lambda5_i),
                            -(-lambda4_i * (lambda6_i)),
                            -(-lambda4_i * (lambda7_i)),
                            -(-lambda4_i * (lambda8_i)),
                            -(lambda5_i * (lambda6_i)),
                            -(lambda5_i * (lambda7_i)),
                            -(lambda5_i * (lambda8_i)),
                            -(lambda6_i * (lambda7_i)),
                            -(lambda6_i * (lambda8_i)),
                            -(lambda7_i * (lambda8_i));

    }
};

// initial constraints
ad_function_with_param_t pushTInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(4);
    y.segment(0, 4) << x[0] - p[0],
                    x[1] - p[1],
                    x[27] - p[2],
                    x[28] - p[3];
};

// cost function for pushT
ad_function_with_param_t pushTObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];
        ad_scalar_t c_theta = x[idx + 27];
        ad_scalar_t s_theta = x[idx + 28];
        ad_matrix_t Q(4, 4);
        ad_matrix_t Q_final(4, 4);
        Q.setZero();
        Q(0, 0) = 1;
        Q(1, 1) = 1;
        Q(2, 2) = 1;
        Q(3, 3) = 1;
        Q_final.setZero();
        Q_final(0, 0) = 100;
        Q_final(1, 1) = 100;
        Q_final(2, 2) = 100;
        Q_final(3, 3) = 100;
        ad_matrix_t R(num_control-2, num_control-2);
        R.setZero();
        R(0, 0) = 0.01;
        R(1, 1) = 0.01;
        R(2, 2) = 0.01;
        R(3, 3) = 0.01;
        R(4, 4) = 0.01;
        R(5, 5) = 0.01;
        R(6, 6) = 0.01;
        R(7, 7) = 0.01;

        if (i == N - 1) {
            ad_vector_t tracking_error(4);

            tracking_error << px_i - p[0],
                            py_i - p[1],
                            c_theta - p[2],
                            s_theta - p[3];

            tracking_cost += tracking_error.transpose() * Q_final * tracking_error;
        }

        if (i < N - 1) {
            ad_vector_t control_error(num_control-2);
            control_error << lambda1_i,
                            lambda2_i,
                            lambda3_i,
                            lambda4_i,
                            lambda5_i,
                            lambda6_i,
                            lambda7_i,
                            lambda8_i;
            control_cost += control_error.transpose() * R * control_error;
            ad_vector_t tracking_error(4);
            tracking_error << px_i - p[0],
                            py_i - p[1],
                            c_theta - p[2],
                            s_theta - p[3];
    
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushT";
    std::string folderName = "model";
    OptimizationProblem pushTProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, 4, problemName, folderName, "pushTObjective", pushTObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTDynamicConstraints", pushTDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTContactConstraints", pushTContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, 4, problemName, folderName, "pushTInitialConstraints", pushTInitialConstraints);
    auto contactSingleForce = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTContactSingleForceConstraints", pushTContactSingleForceConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushTProblem.addObjective(obj);
    pushTProblem.addEqualityConstraint(dynamics);
    pushTProblem.addEqualityConstraint(initial);
    pushTProblem.addInequalityConstraint(contact);
    pushTProblem.addInequalityConstraint(contactSingleForce);


    // problem parameters
    vector_t xInitialStates(4);
    vector_t xFinalStates(4);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20 degree
    // xInitialStates << 0, 0, 1, 0;
    xInitialStates << 0.24722, 0.0141359, cos(-2.95844),sin(-2.95844);
    // set random initial guess beteween -0.01 and 0.01
    xInitialGuess.setZero();
    // xInitialGuess.setRandom();
    // xInitialGuess = 0.001 * xInitialGuess;
    // xInitialGuess.setZero();
    SolverParameters params;
    SolverInterface solver(pushTProblem, params);
    // feel free to change the hyperparameters for the solver to obtain different performance
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 1));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-3));
    // solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
    xFinalStates << 0.036, -0.143, cos(-2.637), sin(-2.637);
    solver.setProblemParameters("pushTInitialConstraints", xInitialStates);
    solver.setProblemParameters("pushTObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();
    std::cout << "Finish solving the problem with final state: " << xFinalStates.transpose() << std::endl;
    }