#include "solver_core/SolverInterface.h"
// #include "common/MatlabHelper.h"
#include <filesystem>
#include <chrono>
#include <random>   

#include "math.h"

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.048;
const scalar_t b = 0.048;
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

// Warm start
static inline double clamp(double v, double lo, double hi){ return std::max(lo, std::min(hi, v));}
vector_t makeRandomInit(const vector_t& x0, const vector_t& xg, const size_t variableNum,
                        unsigned seed = 42,
                        double pos_sigma_frac = 0.02,  // position jitter as fraction of path length
                        double th_sigma = 0.05,        // rad jitter for theta
                        double force_scale = 0.5,      // N-scale for lambda magnitudes
                        double edge_margin = 1e-3)     // keep (cx,cy) off the exact edges
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    vector_t x(variableNum);
    x.setZero();

    // Path length to scale position noise
    double L = std::hypot(xg[0] - x0[0], xg[1] - x0[1]);
    double pos_sigma = pos_sigma_frac * std::max(1.0, L);

    // 1) States: straight-line + small jitter (no jitter at endpoints)
    for (size_t k = 0; k < N; ++k) {
        double alpha = (N == 1) ? 1.0 : double(k) / double(N - 1);
        double jx = 0.0, jy = 0.0, jt = 0.0;
        if (k != 0 && k != N - 1) {            // keep endpoints exact
            jx = pos_sigma * N01(rng);
            jy = pos_sigma * N01(rng);
            jt = th_sigma  * N01(rng);
        }
        size_t idx = k * (num_state + num_control);
        x[idx + 0] = (1 - alpha) * x0[0] + alpha * xg[0] + jx;   // px
        x[idx + 1] = (1 - alpha) * x0[1] + alpha * xg[1] + jy;   // py
        x[idx + 2] = (1 - alpha) * x0[2] + alpha * xg[2] + jt;   // theta
    }

    // 2) Contacts + forces: simple, sign-correct, single active face per k
    for (size_t k = 0; k < N; ++k) {
        size_t idx = k * (num_state + num_control);

        // Random contact point inside box with a small safety margin
        double cx = (2.0 * U01(rng) - 1.0) * (a - edge_margin);
        double cy = (2.0 * U01(rng) - 1.0) * (b - edge_margin);

        // Pick one face to activate; draw a nonnegative magnitude
        int face = int(4.0 * U01(rng));      // 0..3
        double mag = force_scale * std::fabs(N01(rng));

        double l1 = 0.0, l2 = 0.0, l3 = 0.0, l4 = 0.0;
        switch (face) {
            case 0: l1 = mag;     break;     // λ1 ≥ 0
            case 1: l2 = mag;     break;     // λ2 ≥ 0
            case 2: l3 = -mag;    break;     // λ3 ≤ 0
            default:l4 = -mag;    break;     // λ4 ≤ 0
        }

        // Optional: taper forces to zero at the very ends
        if (k == 0 || k == N - 1) l1 = l2 = l3 = l4 = 0.0;

        x[idx + 3] = clamp(cx, -a + edge_margin,  a - edge_margin);   // cx
        x[idx + 4] = clamp(cy, -b + edge_margin,  b - edge_margin);   // cy
        x[idx + 5] = l1;
        x[idx + 6] = l2;
        x[idx + 7] = l3;
        x[idx + 8] = l4;
    }

    return x;
}

// define the dynamics constraints
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

// contact implicit constraints for pushbox
ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 12);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t lambda1_i = x[idx + 5];
        ad_scalar_t lambda2_i = x[idx + 6];
        ad_scalar_t lambda3_i = x[idx + 7];
        ad_scalar_t lambda4_i = x[idx + 8];

        y.segment(i * 12, 12) << lambda1_i,
                            lambda2_i,
                            -lambda3_i,
                            -lambda4_i,
                            cy_i + b,
                            cx_i + a,
                            b - cy_i,
                            a - cx_i,
                            -(lambda1_i)*(cy_i+b),
                            -(lambda2_i)*(cx_i+a),
                            -(-lambda3_i)*(b-cy_i),
                            -(-lambda4_i)*(a-cx_i);
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

        y.segment(i * 6, 6) << -(lambda1_i * lambda2_i),
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

// cost function for pushbox
ad_function_with_param_t pushboxObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
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
        ad_scalar_t lambda1_i = x[idx + 5];
        ad_scalar_t lambda2_i = x[idx + 6];
        ad_scalar_t lambda3_i = x[idx + 7];
        ad_scalar_t lambda4_i = x[idx + 8];
        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        Q(2, 2) = 100;
        ad_matrix_t R(4, 4);
        R.setZero();
        R(0, 0) = 0.001;
        R(1, 1) = 0.001;
        R(2, 2) = 0.001;
        R(3, 3) = 0.001;

        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);

            tracking_error << px_i - p[0], py_i - p[1], theta_i - p[2];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;

        }

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

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective", pushboxObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxDynamicConstraints", pushboxDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);
    auto contactSingleForce = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactSingleForceConstraints", pushboxContactSingleForceConstraints);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addObjective(obj);
    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addEqualityConstraint(initial);
    pushboxProblem.addInequalityConstraint(contact);
    pushboxProblem.addInequalityConstraint(contactSingleForce);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20 degree
    xInitialStates << 0.4, 0.0, 0;
    // set zero initial guess
    xInitialGuess.setZero();
    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);
    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e8));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-6));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));

    size_t num_segments = 18;
    scalar_t theta = 12 * 2 * M_PI / num_segments;
    xFinalStates << 2*cos(theta), 2*sin(theta), theta;
    xFinalStates << 0.4, 0.3, 0.0;
    std::cout << "Initial State: " << xInitialStates.transpose() << std::endl;
    std::cout << "Final State: " << xFinalStates.transpose() << std::endl;
    // xInitialGuess = makeRandomInit(xInitialStates, xFinalStates, variableNum, /*seed=*/110, /*pos_sigma_frac=*/0.02, /*th_sigma=*/0.01, /*force_scale=*/0.5, /*edge_margin=*/1e-3);

    solver.setProblemParameters("pushboxObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_actual.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();                              

}


