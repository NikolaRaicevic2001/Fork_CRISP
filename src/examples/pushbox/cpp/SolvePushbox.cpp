#include "solver_core/SolverInterface.h"
#include "utils/utils.h"

#include <filesystem>
#include <chrono>
#include <random>   
#include "math.h"

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.05;
const scalar_t b = 0.05;
const scalar_t m = 1.0;
const scalar_t mu = 0.5;
const scalar_t g = 9.8;
const scalar_t r = sqrt(a * a + b * b);
const scalar_t c = 0.4; 
const scalar_t dt = 0.02;
const size_t N = 100;                   
const size_t num_state = 3;
const size_t num_control = 6;

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();    

// Define the dynamics constraints
ad_function_t pushboxDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i        = x[idx + 0];
        ad_scalar_t py_i        = x[idx + 1];
        ad_scalar_t theta_i     = x[idx + 2];
        ad_scalar_t cx_i        = x[idx + 3];
        ad_scalar_t cy_i        = x[idx + 4];
        ad_scalar_t lambda1_i   = x[idx + 5];
        ad_scalar_t lambda2_i   = x[idx + 6];
        ad_scalar_t lambda3_i   = x[idx + 7];
        ad_scalar_t lambda4_i   = x[idx + 8];

        ad_scalar_t px_next     = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next     = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next  = x[idx + (num_state + num_control) + 2];

        ad_scalar_t px_dot      = (1/(mu*m*g))*(cos(theta_i)*(lambda2_i + lambda4_i) - sin(theta_i)*(lambda1_i + lambda3_i));
        ad_scalar_t py_dot      = (1/(mu*m*g))*(sin(theta_i)*(lambda2_i + lambda4_i) + cos(theta_i)*(lambda1_i + lambda3_i));
        ad_scalar_t theta_dot   = (1/(mu*m*g*c*r))*(-cy_i*(lambda2_i + lambda4_i) + cx_i*(lambda1_i + lambda3_i));

        // Explicit State Update
        y.segment(i * num_state, num_state) <<  px_next - px_i - px_dot * dt,
                                                py_next - py_i - py_dot * dt,
                                                theta_next - theta_i - theta_dot * dt;
    }
};

ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    const scalar_t eps = 1e-12;

    auto sabs = [&](const ad_scalar_t& z) {// smooth |z|
        return sqrt(z * z + eps);
    };
    auto smax2 = [&](const ad_scalar_t& u, const ad_scalar_t& v) { // smooth max(u,v)
        ad_scalar_t d = u - v;
        return 0.5 * (u + v + sqrt(d * d + eps));
    };

    y.resize((N - 1) * 17);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i        = x[idx + 0];
        ad_scalar_t py_i        = x[idx + 1];
        ad_scalar_t theta_i     = x[idx + 2];
        ad_scalar_t cx_i        = x[idx + 3];
        ad_scalar_t cy_i        = x[idx + 4];
        ad_scalar_t lambda1_i   = x[idx + 5];
        ad_scalar_t lambda2_i   = x[idx + 6];
        ad_scalar_t lambda3_i   = x[idx + 7];
        ad_scalar_t lambda4_i   = x[idx + 8];

        // nonnegative force magnitudes on the 4 faces
        ad_scalar_t f1 =  lambda1_i;   // bottom face
        ad_scalar_t f2 =  lambda2_i;   // left face
        ad_scalar_t f3 = -lambda3_i;   // top face   (note sign)
        ad_scalar_t f4 = -lambda4_i;   // right face (note sign)

        // normal “gaps” (zero on each face), square them to get smooth ψ^n >= 0
        ad_scalar_t psi1_n = (cy_i + b)*(cy_i + b); // y = -b
        ad_scalar_t psi2_n = (cx_i + a)*(cx_i + a); // x = -a
        ad_scalar_t psi3_n = (b - cy_i)*(b - cy_i); // y = +b
        ad_scalar_t psi4_n = (a - cx_i)*(a - cx_i); // x = +a

        // tangential “overrun” (linear)
        ad_scalar_t tx_right =  cx_i - a;   // >0 if cx >  a
        ad_scalar_t tx_left  = -cx_i - a;   // >0 if cx < -a
        ad_scalar_t ty_top   =  cy_i - b;   // >0 if cy >  b
        ad_scalar_t ty_bot   = -cy_i - b;   // >0 if cy < -b

        // “outside or on boundary” guard (forbid interior):
        // ax = |cx|-a, ay = |cy|-b ; outside iff max(ax, ay) >= 0
        ad_scalar_t ax = sabs(cx_i) - a;
        ad_scalar_t ay = sabs(cy_i) - b;
        ad_scalar_t outside_or_on = smax2(ax, ay); // must be >= 0

        // pack (>=0 is feasible)
        y.segment(i * 17, 17) <<
            f1, f2, f3, f4,
            -(f1 * psi1_n),
            -(f2 * psi2_n),
            -(f3 * psi3_n),
            -(f4 * psi4_n),
            -(f1 * tx_right),  // if f1>0 ⇒ cx ≤ a
            -(f1 * tx_left),   // if f1>0 ⇒ cx ≥ -a
            -(f2 * ty_top),    // if f2>0 ⇒ cy ≤ b
            -(f2 * ty_bot),    // if f2>0 ⇒ cy ≥ -b
            -(f3 * tx_right),  // if f3>0 ⇒ cx ≤ a
            -(f3 * tx_left),   // if f3>0 ⇒ cx ≥ -a
            -(f4 * ty_top),    // if f4>0 ⇒ cy ≤ b
            -(f4 * ty_bot),    // if f4>0 ⇒ cy ≥ -b
            outside_or_on;
    }
};

// allow only one contact force at a time
ad_function_t pushboxContactSingleForceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 6);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t lambda1_i = x[idx + 5];
        ad_scalar_t lambda2_i = x[idx + 6];
        ad_scalar_t lambda3_i = x[idx + 7];
        ad_scalar_t lambda4_i = x[idx + 8];

        y.segment(i * 6, 6) <<  -(lambda1_i * lambda2_i),
                                -(lambda1_i * (-lambda3_i)),
                                -(lambda1_i * (-lambda4_i)),
                                -(lambda2_i * (-lambda3_i)),
                                -(lambda2_i * (-lambda4_i)),
                                -(-lambda3_i * (-lambda4_i));
    }
};

// initial constraints
ad_function_with_param_t pushboxInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(3);
    y.segment(0, 3) << x[0] - p[0], x[1] - p[1], x[2] - p[2];
};

// ad_function_with_param_t pushboxInitialConstraintsEndEffector = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
//     y.resize(2);
//     y.segment(0, 2) << x[3] - p[0], x[4] - p[1];
// };

// cost function for pushbox
ad_function_with_param_t pushboxObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        size_t idx_next = (i+1) * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t lambda1_i = x[idx + 5];
        ad_scalar_t lambda2_i = x[idx + 6];
        ad_scalar_t lambda3_i = x[idx + 7];
        ad_scalar_t lambda4_i = x[idx + 8];
        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        Q(2, 2) = 100;
        ad_matrix_t P(num_state, num_state);
        P.setZero();
        P(0, 0) = 0.01;
        P(1, 1) = 0.01;
        P(2, 2) = 0.01;
        ad_matrix_t M(2, 2);
        M.setZero();
        M(0, 0) = 0.05;
        M(1, 1) = 0.05;
        ad_matrix_t R(4, 4);
        R.setZero();
        R(0, 0) = 0.0001;
        R(1, 1) = 0.0001;
        R(2, 2) = 0.0001;
        R(3, 3) = 0.0001;

        // Penalize the tracking error at the final time step
        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0], py_i - p[1], theta_i - p[2];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        // // Penalize large distance traveled by the box
        // if (i < N - 1) {
        //     ad_vector_t tracking_error_whole(num_state);
        //     tracking_error_whole << px_i - p[0], py_i - p[1], theta_i - p[2];
        //     tracking_cost += tracking_error_whole.transpose() * P * tracking_error_whole;
        // }

        // // Penalize the difference between the contact point to prevent large jumps
        // if (i < N - 1) {
        //     ad_vector_t contact_point_diff(2);
        //     ad_scalar_t cx_next, cy_next;
        //     cx_next = x[idx_next + 3];
        //     cy_next = x[idx_next + 4];
        //     contact_point_diff << cx_i - cx_next, cy_i - cy_next;
        //     control_cost += contact_point_diff.transpose() * M * contact_point_diff;
        // }

        // Penalize the contact forces to prevent excessive forces
        if (i < N - 1) {
            ad_vector_t control_error(4);
            control_error << lambda1_i, lambda2_i, lambda3_i, lambda4_i;
            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "Pushbox";
    std::string folderName = "model";
    OptimizationProblem pushboxProblem(variableNum, problemName);

    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxDynamicConstraints", pushboxDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    auto contactSingleForce = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactSingleForceConstraints", pushboxContactSingleForceConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);
    // auto initial_ee = std::make_shared<ConstraintFunction>(variableNum, 2, problemName, folderName, "pushboxInitialConstraintsEndEffector", pushboxInitialConstraintsEndEffector);
    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective", pushboxObjective);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addInequalityConstraint(contact);
    pushboxProblem.addInequalityConstraint(contactSingleForce);
    pushboxProblem.addEqualityConstraint(initial);
    // pushboxProblem.addEqualityConstraint(initial_ee);
    pushboxProblem.addObjective(obj);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xInitialStates_ee(2);
    vector_t xOptimal(variableNum);

    // xInitialStates << 0.4, 0.0, 0.0;
    xInitialStates << 0.35766736, 0.08357876, 0.42412436;       // Suboptimal initial condition
    // xInitialStates << 0.35766736, 0.08357876, 1.42412436;    // Suboptimal initial condition
    xInitialStates_ee << 0.0, -4*b;
    xFinalStates << 0.4, 0.3, 0.0;
    // xInitialGuess.setZero();
    xInitialGuess = makeRandomFirstGuess(N, num_state, num_control, a, b, /*seed=*/100);
    std::cout << "Initial State: " << xInitialStates.transpose() << std::endl;
    std::cout << "Initial State End Effector: " << xInitialStates_ee.transpose() << std::endl;
    std::cout << "Final State: " << xFinalStates.transpose() << std::endl;

    // solver parameters
    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);
    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    // solver.setProblemParameters("pushboxInitialConstraintsEndEffector", xInitialStates_ee);
    // solver.setHyperParameters("trustRegionInitRadius", vector_t::Constant(1, 1.0));
    // solver.setHyperParameters("trustRegionMaxRadius", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("etaLow", vector_t::Constant(1, 0.25));
    // solver.setHyperParameters("etaHigh", vector_t::Constant(1, 0.75));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 10.0));
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e10));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-7));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setHyperParameters("WeightedTolFactor", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("secondOrderCorrection", vector_t::Constant(1, 1));

    solver.setProblemParameters("pushboxObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();

    ad_vector_t eq_values;
    ad_vector_t ineq_values; 
    ad_vector_t solution_ad = xOptimal.cast<ad_scalar_t>();
    pushboxDynamicConstraints(solution_ad, eq_values);
    pushboxContactConstraints(solution_ad, ineq_values);

    // Print Inequality Constraint Values
    std::cout << "Inequality Constraint Values (should be >= 0):" << std::endl;
    for (size_t i = 0; i < 20; ++i) {
        std::cout << "Constraint " << i << ": " << ineq_values[i] << std::endl;
    }

    // Print Equality Constraint Values
    std::cout << "Equality Constraint Values (should be ~0):" << std::endl;
    for (size_t i = 0; i < 20; ++i) {
        std::cout << "Constraint " << i << ": " << eq_values[i] << std::endl;
    }

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_analytical.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();                              

}


