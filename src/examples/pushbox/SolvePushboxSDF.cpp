#include "solver_core/SolverInterface.h"

#include <chrono>
#include <filesystem>
#include "math.h"

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.5;
const scalar_t b = 0.25;
const scalar_t m = 1;
const scalar_t mu = 0.5;
const scalar_t g = 9.8;
const scalar_t r = sqrt(a * a + b * b);
const scalar_t c = 0.4;
const scalar_t dt = 0.02;
const size_t N = 100;               // number of time steps
const size_t num_state = 3;         // STATE  (3) : [px, py, θ]
const size_t num_control = 3;       // CONTROL (3) : [cx, cy, λ ]

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();    

// Signed distance
#include <cppad/cppad.hpp>   // already pulled in via BasicTypes.h

/* ---------- exact SDF of an axis–aligned box -------------------- */
template<class T>
inline T sdfBox(const Eigen::Matrix<T,2,1>& p, const Eigen::Matrix<T,2,1>& half)
{
    T dx = CppAD::abs(p.x()) - half.x();
    T dy = CppAD::abs(p.y()) - half.y();

    /* outside term: ‖max(d,0)‖₂ ---------------------------------- */
    T ax = CppAD::CondExpGt(dx, T(0), dx, T(0));   // max(dx,0)
    T ay = CppAD::CondExpGt(dy, T(0), dy, T(0));   // max(dy,0)
    T outside = CppAD::sqrt(ax*ax + ay*ay);

    /* inside term: min(max(dx,dy),0) ------------------------------ */
    T dmax   = CppAD::CondExpGt(dx, dy, dx, dy);   // max(dx,dy)
    T inside = CppAD::CondExpLt(dmax, T(0), dmax, T(0));

    return outside + inside;
}

// gradient of the SDF using analytical formula
template<class T>
inline Eigen::Matrix<T,2,1> sdfBox_Grad(const Eigen::Matrix<T,2,1>& p, const Eigen::Matrix<T,2,1>& half)
{
    /* distances to the box faces --------------------------------- */
    T dx = CppAD::abs(p.x()) - half.x();
    T dy = CppAD::abs(p.y()) - half.y();

    /* helper: max(d,0) ------------------------------------------- */
    T ax = CppAD::CondExpGt(dx, T(0), dx, T(0));
    T ay = CppAD::CondExpGt(dy, T(0), dy, T(0));
    T g  = CppAD::sqrt(ax*ax + ay*ay);               // ‖clamped‖₂

    /* sign(p) implemented with CondExp --------------------------- */
    T sx = CppAD::CondExpGe(p.x(), T(0), T(1), T(-1));
    T sy = CppAD::CondExpGe(p.y(), T(0), T(1), T(-1));

    /* choose which face gives the normal ------------------------- */
    Eigen::Matrix<T,2,1> n;
    n.x() = CppAD::CondExpGt(dx, dy, sx, T(0));      // if |dx|>|dy| use x-face
    n.y() = CppAD::CondExpGt(dx, dy, T(0), sy);      // else use y-face

    /* if the point is *outside*, override by the clamped vector --- */
    n.x() = CppAD::CondExpGt(g, T(0), ax/g, n.x());
    n.y() = CppAD::CondExpGt(g, T(0), ay/g, n.y());
    return n;
}

// numerical gradient of the SDF using finite differences
template<class T>
Eigen::Matrix<T,2,1> sdfBox_FDGrad(const Eigen::Matrix<T,2,1>& p, const Eigen::Matrix<T,2,1>& half, T eps = T(1e-6))
{
    Eigen::Matrix<T,2,1> g;
    auto f0 = sdfBox(p, half);
    for (int k=0; k<2; ++k)
    {
        Eigen::Matrix<T,2,1> ph = p;
        ph(k) += eps;
        g(k) = (sdfBox(ph, half) - f0) / eps;
    }
    return g / CppAD::sqrt(g.squaredNorm());              
}

// define the dynamics constraints
ad_function_t pushboxDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t th_i    = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lam_i   = x[idx + 5];

        ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next = x[idx + (num_state + num_control) + 2];

        auto g_i = sdfBox(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), Eigen::Matrix<ad_scalar_t,2,1>(a,b));
        auto n_i = sdfBox_Grad(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), Eigen::Matrix<ad_scalar_t,2,1>(a,b));
        
        ad_scalar_t c = cos(th_i),  s = sin(th_i);
        ad_scalar_t Fx = lam_i * (  c*n_i.x() - s*n_i.y() );
        ad_scalar_t Fy = lam_i * (  s*n_i.x() + c*n_i.y() );
        ad_scalar_t torque_z = lam_i * (cx_i*n_i.y() - cy_i*n_i.x());

        ad_scalar_t px_dot    =  Fx / (mu * m * g);
        ad_scalar_t py_dot    =  Fy / (mu * m * g);
        ad_scalar_t th_dot    =  torque_z / (mu * m * g * c * r);

        // Explicit State Update
        y.segment(i * num_state, num_state) << 
                                px_next - px_i - px_dot * dt, 
                                py_next - py_i - py_dot * dt, 
                                theta_next - th_i - th_dot * dt;
    }
};

// contact implicit constraints for pushbox
ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y)
{
    y.resize((N-1)*3);
    for (size_t i=0; i<N-1; ++i)
    {
        size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lam_i   = x[idx + 5];

        auto g_i  = sdfBox(Eigen::Matrix<ad_scalar_t,2,1>(cx_i,cy_i), Eigen::Matrix<ad_scalar_t,2,1>(a,b));

        y.segment(i*3,3) << lam_i,          // λ ≥ 0  (handled as inequality)
                           g_i,             // g ≥ 0
                          -g_i*lam_i;       // -λ·g ≥ 0   ⇒ complementarity
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
        // Extract state and control for current and next time steps
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t th_i    = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lam_i   = x[idx + 5];

        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        Q(2, 2) = 100;
        ad_matrix_t R(1, 1);
        R.setZero();
        R(0, 0) = 0.001;

        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);

            tracking_error << px_i - p[0], py_i - p[1], th_i - p[2];
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
    std::string problemName = "PushboxSDF";
    std::string folderName = "model";
    OptimizationProblem pushboxProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective", pushboxObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxDynamicConstraints", pushboxDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //
    pushboxProblem.addObjective(obj);
    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addEqualityConstraint(initial);
    pushboxProblem.addInequalityConstraint(contact);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20 degree
    xInitialStates << 0, 0, 0;
    // set zero initial guess
    xInitialGuess.setZero();
    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);
    // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    size_t num_segments = 18;
        scalar_t theta = 12 * 2 * M_PI / num_segments;
        xFinalStates << 3*cos(theta), 3*sin(theta), theta;
        solver.setProblemParameters("pushboxObjective", xFinalStates);
        solver.initialize(xInitialGuess);
        solver.solve();
        xOptimal = solver.getSolution();

        std::ofstream log(PROJECT_ROOT / "src/examples/pushbox/results/results_pushbox_sdf_AD.csv");
        for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
        log.close();                              

    }


