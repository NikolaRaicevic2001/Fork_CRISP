#include "solver_core/SolverInterface.h"

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
const size_t num_control = 5;

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();    

// Helper Functions
static inline double clamp(double v, double lo, double hi){ return std::max(lo, std::min(hi, v));}

// Function that generates a random vector of size num_state+num_control and then repeats it N times to form the initial guess
vector_t makeRandomFirstGuess(const size_t N, const size_t num_state, const size_t num_control, const scalar_t a, const scalar_t b, const unsigned seed = 40)
{ 
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Initial Guess Vector
    vector_t x(N*(num_state + num_control));

    x.setZero();

    // Random vector of size num_state+num_control
    vector_t random_vector(num_state + num_control);
    random_vector.setZero();

    random_vector[0] = U01(rng) * 4.0 - 2.0;   // px in [-2, 2]
    random_vector[1] = U01(rng) * 4.0 - 2.0;   // py in [-2, 2]
    random_vector[2] = U01(rng) * 1.0 * M_PI;  // theta in [0, pi]

    // Random contact point inside box with a small safety margin
    double edge_margin = 1e-3;
    double cx = (2.0 * U01(rng) - 1.0) * (a - edge_margin);
    double cy = (2.0 * U01(rng) - 1.0) * (b - edge_margin);
    random_vector[3] = clamp(cx, -a + edge_margin,  a - edge_margin);   // cx
    random_vector[4] = clamp(cy, -b + edge_margin,  b - edge_margin);   // cy

    // Random force and epigraph values
    random_vector[5] = N01(rng);   // lambda1
    random_vector[6] = N01(rng);   // ux_i
    random_vector[7] = N01(rng);   // uy_i

    // Print the generated random vector
    std::cout << "Generated random vector: " << random_vector.transpose() << std::endl;
    
    // Repeat the random vector N times to form the initial guess
    for (size_t i = 0; i < N; ++i) {
        x.segment(i * (num_state + num_control), num_state + num_control) = random_vector; 
    }

    return x;
}

// Define the dynamics constraints
ad_function_t pushboxDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    const ad_scalar_t eps = ad_scalar_t(1e-12);

    auto sgn_smooth = [&](const ad_scalar_t& z) {
        // smooth sign to keep derivatives well-behaved at 0
        return z / CppAD::sqrt(z * z + eps);
    };

    y.resize((N-1) * num_state);

    for (size_t i = 0; i < N - 1; ++i) {
        // Extract state and control for current and next time steps
        const size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i        = x[idx + 0];
        ad_scalar_t py_i        = x[idx + 1];
        ad_scalar_t theta_i     = x[idx + 2];
        ad_scalar_t cx_i        = x[idx + 3];
        ad_scalar_t cy_i        = x[idx + 4];
        ad_scalar_t lambda_i    = x[idx + 5];
        ad_scalar_t ux_i        = x[idx + 6];
        ad_scalar_t uy_i        = x[idx + 7];

        ad_scalar_t px_next     = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next     = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next  = x[idx + (num_state + num_control) + 2];

        // Face weights (swap logic): on x-face -> uy>0, on y-face -> ux>0
        const ad_scalar_t denom = ux_i + uy_i + eps;
        const ad_scalar_t wx = uy_i / denom; // weight for x-normal (x-face)
        const ad_scalar_t wy = ux_i / denom; // weight for y-normal (y-face)

        // Outward normal signs in box frame
        const ad_scalar_t nx = -sgn_smooth(cx_i); // +1 on right face, -1 on left
        const ad_scalar_t ny = -sgn_smooth(cy_i); // +1 on top face,   -1 on bottom

        // Body-frame contact force from single λ
        const ad_scalar_t fx_b = lambda_i * wx * nx;
        const ad_scalar_t fy_b = lambda_i * wy * ny;

        // Rotate force to world and Torque about COM (lever arm in box frame)
        const ad_scalar_t fx_w =  cos(theta_i) * fx_b - sin(theta_i) * fy_b;
        const ad_scalar_t fy_w =  sin(theta_i) * fx_b + cos(theta_i) * fy_b;
        const ad_scalar_t tau  = cx_i * fy_b - cy_i * fx_b;

        // Scaling
        const ad_scalar_t inv = ad_scalar_t(1.0) / (mu * m * g);
        const ad_scalar_t px_dot    = inv * fx_w;
        const ad_scalar_t py_dot    = inv * fy_w;
        const ad_scalar_t theta_dot = inv * (tau / (c * r));

        // Explicit Euler defects
        y.segment(i * num_state, num_state) <<
            (px_next    - px_i    - px_dot    * dt),
            (py_next    - py_i    - py_dot    * dt),
            (theta_next - theta_i - theta_dot * dt);
    }
};

ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 8);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i        = x[idx + 0];
        ad_scalar_t py_i        = x[idx + 1];
        ad_scalar_t theta_i     = x[idx + 2];
        ad_scalar_t cx_i        = x[idx + 3];
        ad_scalar_t cy_i        = x[idx + 4];
        ad_scalar_t lambda_i    = x[idx + 5];
        ad_scalar_t ux_i        = x[idx + 6];
        ad_scalar_t uy_i        = x[idx + 7];

        // signed "inside-ness" per coord: >0 inside, =0 on face, <0 strictly outside
        const ad_scalar_t sx = 1.0 - (cx_i / a)*(cx_i / a);
        const ad_scalar_t sy = 1.0 - (cy_i / b)*(cy_i / b);

        // pack (>=0 is feasible)
        y.segment(i * 8, 8) <<
            ux_i,
            uy_i,
            (ux_i - sx),
            (uy_i - sy),
            -(ux_i * uy_i),
            lambda_i,
            (lambda_i * sx),
            (lambda_i * sy);
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
        ad_scalar_t px_i        = x[idx + 0];
        ad_scalar_t py_i        = x[idx + 1];
        ad_scalar_t theta_i     = x[idx + 2];
        ad_scalar_t cx_i        = x[idx + 3];
        ad_scalar_t cy_i        = x[idx + 4];
        ad_scalar_t lambda_i    = x[idx + 5];
        ad_scalar_t ux_i        = x[idx + 6];
        ad_scalar_t uy_i        = x[idx + 7];
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
        ad_matrix_t R(1, 1);
        R.setZero();
        R(0, 0) = 0.0001;

        // Penalize the tracking error at the final time step
        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0], py_i - p[1], theta_i - p[2];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        // Penalize large distance traveled by the box
        if (i < N - 1) {
            ad_vector_t tracking_error_whole(num_state);
            tracking_error_whole << px_i - p[0], py_i - p[1], theta_i - p[2];
            tracking_cost += tracking_error_whole.transpose() * P * tracking_error_whole;
        }

        // Penalize the difference between the contact point to prevent large jumps
        if (i < N - 1) {
            ad_vector_t contact_point_diff(2);
            ad_scalar_t cx_next, cy_next;
            cx_next = x[idx_next + 3];
            cy_next = x[idx_next + 4];
            contact_point_diff << cx_i - cx_next, cy_i - cy_next;
            control_cost += contact_point_diff.transpose() * M * contact_point_diff;
        }

        // Penalize the contact forces to prevent excessive forces
        if (i < N - 1) {
            ad_vector_t control_error(1);
            control_error << lambda_i;
            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushboxSingleForce";
    std::string folderName = "model";
    OptimizationProblem pushboxProblem(variableNum, problemName);

    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxDynamicConstraints", pushboxDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);
    // auto initial_ee = std::make_shared<ConstraintFunction>(variableNum, 2, problemName, folderName, "pushboxInitialConstraintsEndEffector", pushboxInitialConstraintsEndEffector);
    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective", pushboxObjective);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addInequalityConstraint(contact);
    pushboxProblem.addEqualityConstraint(initial);
    // pushboxProblem.addEqualityConstraint(initial_ee);
    pushboxProblem.addObjective(obj);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xInitialStates_ee(2);
    vector_t xOptimal(variableNum);

    xInitialStates << 0.4, 0.0, 0.0;
    // xInitialStates << 0.35766736, 0.08357876, 0.42412436;  // Suboptimal initial condition
    // xInitialStates << 0.35766736, 0.08357876, 1.42412436;  // Suboptimal initial condition
    xInitialStates_ee << 0.0, -4*b;
    xInitialGuess.setZero();
    // xInitialGuess = makeRandomFirstGuess(N, num_state, num_control, a, b, /*seed=*/90);
    xFinalStates << 0.4, 0.3, 0.0;
    std::cout << "Initial State: " << xInitialStates.transpose() << std::endl;
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

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_single_force.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();                              

}


