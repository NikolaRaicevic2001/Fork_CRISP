#include "solver_core/SolverInterface.h"
#include <filesystem>
#include <chrono>
#include <random>   

#include "math.h"

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.1;
const scalar_t b = 0.1;
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

// Function that generates a random vector of size num_state+num_control and then repeats it N times to form the initial guess
static inline double clamp(double v, double lo, double hi){ return std::max(lo, std::min(hi, v));}
vector_t makeRandomFirstGuess(const size_t N, const size_t num_state, const size_t num_control, unsigned seed = 40)
{ 
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Initial Guess Vector
    vector_t x(N*(num_state + num_control));
    x.setZero();

    // Random vector of size num_state + num_control
    vector_t random_vector(num_state + num_control);
    random_vector.setZero();

    random_vector[0] = U01(rng) * 4.0 - 2.0;   // px in [-2, 2]
    random_vector[1] = U01(rng) * 4.0 - 2.0;   // py in [-2, 2]
    random_vector[2] = U01(rng) * 2.0 * M_PI;  // theta in [0, 2pi]

    // Random contact point inside box with a small safety margin
    double edge_margin = 1e-3;
    double cx = (2.0 * U01(rng) - 1.0) * (a - edge_margin);
    double cy = (2.0 * U01(rng) - 1.0) * (b - edge_margin);
    random_vector[3] = clamp(cx, -a + edge_margin,  a - edge_margin);   // cx
    random_vector[4] = clamp(cy, -b + edge_margin,  b - edge_margin);   // cy

    // Random force values
    random_vector[5] = N01(rng);   // lambda1
    random_vector[6] = N01(rng);   // lambda2
    random_vector[7] = N01(rng);   // lambda3
    random_vector[8] = N01(rng);   // lambda4

    // Print the generated random vector
    std::cout << "Generated random vector: " << random_vector.transpose() << std::endl;
    
    // Repeat the random vector N times to form the initial guess
    for (size_t i = 0; i < N; ++i) {
        x.segment(i * (num_state + num_control), num_state + num_control) = random_vector; 
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

//     ad_function_with_param_t pushboxInitialConstraintsEE = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
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
        ad_matrix_t R(4, 4);
        R.setZero();
        R(0, 0) = 0.001;
        R(1, 1) = 0.001;
        R(2, 2) = 0.001;
        R(3, 3) = 0.001;
        ad_matrix_t M(2, 2);
        M.setZero();
        M(0, 0) = 0.0001;
        M(1, 1) = 0.0001;

        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);

            tracking_error << px_i - p[0], py_i - p[1], theta_i - p[2];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;

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
    // auto initial_ee = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraintsEE", pushboxInitialConstraintsEE);
    auto contactSingleForce = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactSingleForceConstraints", pushboxContactSingleForceConstraints);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addObjective(obj);
    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addEqualityConstraint(initial);
    // pushboxProblem.addEqualityConstraint(initial_ee);
    pushboxProblem.addInequalityConstraint(contact);
    pushboxProblem.addInequalityConstraint(contactSingleForce);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xInitialStates_ee(2);
    vector_t xOptimal(variableNum);
    // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20 degree
    xInitialStates << 0.4, 0.0, 0;
    // xInitialStates_ee << a, a;
    // xInitialStates << 0.35766736, 0.08357876, 1.42412436;  // Suboptimal initial condition

    // set zero initial guess
    xInitialGuess.setZero();
    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);
    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    // solver.setHyperParameters("trustRegionInitRadius", vector_t::Constant(1, 1.0));
    // solver.setHyperParameters("trustRegionMaxRadius", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("etaLow", vector_t::Constant(1, 0.25));
    // solver.setHyperParameters("etaHigh", vector_t::Constant(1, 0.75));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 10.0));
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e8));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-6));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setHyperParameters("WeightedTolFactor", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("secondOrderCorrection", vector_t::Constant(1, 1));

    size_t num_segments = 18;
    scalar_t theta = 12 * 2 * M_PI / num_segments;
    xFinalStates << 2*cos(theta), 2*sin(theta), theta;
    xFinalStates << 0.4, 0.3, 0.0;
    std::cout << "Initial State: " << xInitialStates.transpose() << std::endl;
    std::cout << "Final State: " << xFinalStates.transpose() << std::endl;
    xInitialGuess = makeRandomFirstGuess(N, num_state, num_control, /*seed=*/40);

    solver.setProblemParameters("pushboxObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_actual.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();                              

}


