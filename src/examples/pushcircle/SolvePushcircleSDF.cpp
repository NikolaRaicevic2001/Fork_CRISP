#include "solver_core/SolverInterface.h"

#include <cppad/cppad.hpp>   
#include <filesystem>
#include <chrono>
#include "math.h"

using namespace CRISP;

// Define model parameters for circle
const scalar_t R = 0.4;                 // radius of the circle
const scalar_t m = 1;                   // mass of the circle
const scalar_t mu = 0.5;                // friction coefficient
const scalar_t g = 9.8;                 // gravitational acceleration  
const scalar_t dt = 0.02;               // time step size
const size_t N = 100;                   // number of time steps
const size_t num_state = 2;             // STATE  (2) : [px, py]
const size_t num_control = 3;           // CONTROL (3) : [cx, cy, λ ]

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();    

// Signed distance function for circle
template<class T>
inline T sdfCircle(const Eigen::Matrix<T,2,1>& p,  T radius)
{
    return CppAD::sqrt(p.x()*p.x() + p.y()*p.y()) - radius;
}

// Gradient of the SDF using analytical formula
template<class T>
inline Eigen::Matrix<T,2,1> sdfCircle_Grad(const Eigen::Matrix<T,2,1>& p, T radius)
{
    /* distance from centre ------------------------------------ */
    T len = CppAD::sqrt(p.x()*p.x() + p.y()*p.y());

    /* avoid division-by-zero at the centre -------------------- */
    T inv = CppAD::CondExpGt(len, T(0), T(1)/len, T(0));

    Eigen::Matrix<T,2,1> n;
    n << p.x()*inv, p.y()*inv;          // p / |p|
    return n;
}

// Numerical gradient of the SDF using finite differences
template<class T>
Eigen::Matrix<T,2,1> sdfCircle_FDGrad(const Eigen::Matrix<T,2,1>& p, T radius, double eps = 1e-6)
{
    Eigen::Matrix<T,2,1> g;
    T f0 = sdfCircle(p, radius);

    for (int k = 0; k < 2; ++k) {
        Eigen::Matrix<T,2,1> ph = p;
        ph(k) += eps;
        g(k)   = (sdfCircle(ph, radius) - f0) / eps;
    }

    // Normalize the gradient
    return g / CppAD::sqrt(g.squaredNorm()); 
}

// define the dynamics constraints
ad_function_t pushcircleDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t cx_i    = x[idx + 2];
        ad_scalar_t cy_i    = x[idx + 3];
        ad_scalar_t lam_i   = x[idx + 4];

        ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];

        auto g_i = sdfCircle(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), ad_scalar_t(R));
        auto n_i = sdfCircle_FDGrad(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), ad_scalar_t(R));

        ad_scalar_t Fx = lam_i * n_i.x();
        ad_scalar_t Fy = lam_i * n_i.y();

        ad_scalar_t px_dot    =  Fx / (mu * m * g);
        ad_scalar_t py_dot    =  Fy / (mu * m * g);

        // Explicit State Update
        y.segment(i * num_state, num_state) << 
                                px_next - px_i - px_dot * dt, 
                                py_next - py_i - py_dot * dt; 
    }
};

// contact implicit constraints for pushcircle
ad_function_t pushcircleContactConstraints = [](const ad_vector_t& x, ad_vector_t& y)
{
    y.resize((N-1)*3);
    for (size_t i=0; i<N-1; ++i)
    {
        size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t cx_i    = x[idx + 2];
        ad_scalar_t cy_i    = x[idx + 3];
        ad_scalar_t lam_i   = x[idx + 4];

        auto g_i  = sdfCircle(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), ad_scalar_t(R));

        y.segment(i*3,3) << lam_i,          // λ ≥ 0  (handled as inequality)
                           g_i,             // g ≥ 0
                          -g_i*lam_i;       // -λ·g ≥ 0   ⇒ complementarity
    }
};

// initial constraints
ad_function_with_param_t pushcircleInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(2);
    y.segment(0, 2) << x[0] - p[0], x[1] - p[1];
};

// cost function for pushcircle
ad_function_with_param_t pushcircleObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t cx_i    = x[idx + 2];
        ad_scalar_t cy_i    = x[idx + 3];
        ad_scalar_t lam_i   = x[idx + 4];

        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        ad_matrix_t R(1, 1);
        R.setZero();
        R(0, 0) = 0.001;

        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0], py_i - p[1];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        if (i < N - 1) {
            ad_vector_t control_error(1);
            control_error << lam_i;
            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushcircleSDF";
    std::string folderName = "model";
    OptimizationProblem pushcircleProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushcircleObjective", pushcircleObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushcircleDynamicConstraints", pushcircleDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushcircleContactConstraints", pushcircleContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushcircleInitialConstraints", pushcircleInitialConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //
    pushcircleProblem.addObjective(obj);
    pushcircleProblem.addEqualityConstraint(dynamics);
    pushcircleProblem.addEqualityConstraint(initial);
    pushcircleProblem.addInequalityConstraint(contact);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20 degree
    xInitialStates << 0, 0;
    // set zero initial guess
    xInitialGuess.setZero();
    for (size_t k = 2; k < xInitialGuess.size(); k += (num_state+num_control))
    {
        xInitialGuess[k]   =  R;   // cx_i
        xInitialGuess[k+1] =  0;   // cy_i
    }

    SolverParameters params;
    SolverInterface solver(pushcircleProblem, params);
    solver.setProblemParameters("pushcircleInitialConstraints", xInitialStates);
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("verbose", vector_t::Constant(1, 1));  
        xFinalStates << -0.5, -1.0;
        solver.setProblemParameters("pushcircleObjective", xFinalStates);
        solver.initialize(xInitialGuess);
        solver.solve();
        xOptimal = solver.getSolution();

        std::ofstream log(PROJECT_ROOT / "src/examples/pushcircle/results/results_pushcircle_sdf_FD.csv");
        for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
        log.close();                              

    }


