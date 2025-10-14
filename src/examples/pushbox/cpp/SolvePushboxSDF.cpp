#include "solver_core/SolverInterface.h"
#include "sdf/RoundBox.h"

#include <cmath>                    // use <cmath>, not <math.h>
#include <chrono>                   // use <chrono> for timing
#include <iomanip>                  // use <iomanip> for formatting
#include <filesystem>               // use <filesystem> for path manipulation
#include <cppad/cppad.hpp>          // CppAD
#include <cppad/cg/cppadcg.hpp>     // ensure CG is visible

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.05;
const scalar_t b = 0.05;
const scalar_t m = 1;
const scalar_t mu = 0.5;
const scalar_t g = 9.8;
const scalar_t r = std::sqrt(a * a + b * b);    
const scalar_t c = 0.4;                     
const scalar_t dt = 0.1;
const size_t   N = 100;                               // number of time steps
const size_t   num_state = 3;                         // STATE  (3) : [px, py, θ]
const size_t   num_control = 3;                       // CONTROL (3) : [cx, cy, λ ]

// SDF function parameters
constexpr scalar_t ROUND_R = 0.01;
constexpr scalar_t CONTACT_EPS = 1e-6;

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();

// ---------- Atomic for SDF: x=[cx, cy] -> y=[d]; (hx,hy,r) captured ----------
template <class Base>
class AtomicBoxRoundedSDF : public CppAD::atomic_four<Base> {
public:
    explicit AtomicBoxRoundedSDF(Base hx, Base hy, Base r, Base beta = Base(40), Base eps = Base(1e-12), Base sgn_eps = Base(1e-12))
    : CppAD::atomic_four<Base>("box_rounded_sdf_x2"), hx_(hx), hy_(hy), r_(r), beta_(beta), eps_(eps), sgn_eps_(sgn_eps) {}

private:
    Base hx_, hy_, r_, beta_, eps_, sgn_eps_;

    // x = [cx, cy]  ->  y = [d]
    bool for_type(size_t /*call_id*/,
                  const CppAD::vector<CppAD::ad_type_enum>& type_x,
                  CppAD::vector<CppAD::ad_type_enum>& type_y) override
    {
        type_y[0] = type_x[0];
        return true;
    }

    bool forward(size_t /*call_id*/, const CppAD::vector<bool>& /*select_y*/, size_t order_low, size_t order_up, const CppAD::vector<Base>& taylor_x, CppAD::vector<Base>& taylor_y) override
    {
        const size_t q = order_up + 1;
        const Base cx0 = taylor_x[0*q + 0];
        const Base cy0 = taylor_x[1*q + 0];

        Eigen::Matrix<Base,2,1> P(cx0, cy0);
        Eigen::Matrix<Base,2,1> H; H << hx_, hy_;

        // 0th order value
        Base d0 = CRISP::sdf::BoxRoundedSmoothSDF<Base>(P, H, r_, beta_, eps_);
        if (order_low <= 0) taylor_y[0*q + 0] = d0;
        if (order_up < 1)   return true;

        // 1st order: y' = grad_p^T * x'
        const Base dcx = taylor_x[0*q + 1];
        const Base dcy = taylor_x[1*q + 1];

        Eigen::Matrix<Base,2,1> grad_p = CRISP::sdf::BoxRoundedSmoothGrad<Base>(P, H, r_, beta_, eps_, sgn_eps_); // ∇_p d
        taylor_y[0*q + 1] = grad_p.x()*dcx + grad_p.y()*dcy;
        return true;
    }

    bool reverse(size_t /*call_id*/,
                 const CppAD::vector<bool>& /*select_x*/,
                 size_t order_up,
                 const CppAD::vector<Base>& taylor_x,
                 const CppAD::vector<Base>& /*taylor_y*/,
                 CppAD::vector<Base>& partial_x,
                 const CppAD::vector<Base>& partial_y) override
    {
        if (order_up != 0) return false; // first-order reverse only
        const size_t q = 1;

        const Base cx0 = taylor_x[0*q + 0];
        const Base cy0 = taylor_x[1*q + 0];

        Eigen::Matrix<Base,2,1> P(cx0, cy0);
        Eigen::Matrix<Base,2,1> H; H << hx_, hy_;
        Eigen::Matrix<Base,2,1> grad_p = CRISP::sdf::BoxRoundedSmoothGrad<Base>(P, H, r_, beta_, eps_, sgn_eps_);

        const Base pd = partial_y[0*q + 0];
        partial_x[0*q + 0] += grad_p.x() * pd; // ∂d/∂cx * pd
        partial_x[1*q + 0] += grad_p.y() * pd; // ∂d/∂cy * pd
        return true;
    }

    bool jac_sparsity(size_t /*call_id*/,
                      bool /*dependency*/,
                      const CppAD::vector<bool>& /*ident_zero_x*/,
                      const CppAD::vector<bool>& select_x,
                      const CppAD::vector<bool>& select_y,
                      CppAD::sparse_rc< CppAD::vector<size_t> >& pattern_out) override
    {
        // Jacobian is 1x2 (dense if both inputs selected)
        const size_t m = 1, n = 2;
        size_t nnz = 0;
        if (select_y[0]) for (size_t j=0;j<n;++j) if (select_x[j]) ++nnz;
        pattern_out.resize(m, n, nnz);
        size_t k = 0;
        if (select_y[0]) for (size_t j=0;j<n;++j) if (select_x[j]) pattern_out.set(k++, 0, j);
        return true;
    }
};

using CGD = CppAD::cg::CG<double>;
static AtomicBoxRoundedSDF<CGD>    g_sdf_d( CGD(a), CGD(b), CGD(ROUND_R), CGD(40), CGD(1e-12), CGD(1e-12) );
static AtomicBoxRoundedSDF<double> g_sdf_d_dbg( a, b, ROUND_R, 40.0, 1e-12, 1e-12 );

// -------------------------- Dynamics constraints -----------------------------
ad_function_t pushboxDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;

    // Build 'half' without parens to avoid most-vexing-parse
    V2ad half; half << ad_scalar_t(a), ad_scalar_t(b);

    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        const size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lambda_i = x[idx + 5];

        ad_scalar_t px_next     = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next     = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next  = x[idx + (num_state + num_control) + 2];

        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        const V2ad n_i = -sdg.n;

        const ad_scalar_t Fx = lambda_i * (CppAD::cos(theta_i)*n_i.x() - CppAD::sin(theta_i)*n_i.y());
        const ad_scalar_t Fy = lambda_i * (CppAD::sin(theta_i)*n_i.x() + CppAD::cos(theta_i)*n_i.y());
        const ad_scalar_t torque_z = lambda_i * (cx_i*n_i.y() - cy_i*n_i.x());

        const ad_scalar_t denom_lin = ad_scalar_t(mu * m * g);
        const ad_scalar_t denom_ang = ad_scalar_t(mu * m * g * c * r);

        const ad_scalar_t px_dot  = Fx / denom_lin;
        const ad_scalar_t py_dot  = Fy / denom_lin;
        const ad_scalar_t theta_dot  = torque_z / denom_ang;

        // Explicit Euler defects
        y.segment(i * num_state, num_state) <<
            (px_next    - px_i    - px_dot    * dt),
            (py_next    - py_i    - py_dot    * dt),
            (theta_next - theta_i - theta_dot * dt);
    }
};

// ----------------------- Contact-implicit constraints ------------------------
ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y){
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;
    V2ad half; half << ad_scalar_t(a), ad_scalar_t(b);

    y.resize((N-1)*1);
    for (size_t i=0; i<N-1; ++i)
    {
        const size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lambda_i= x[idx + 5];

        // std::vector<ad_scalar_t> xin(2), yout(1);
        // xin[0] = cx_i; xin[1] = cy_i;
        // g_sdf_d(xin, yout);
        // ad_scalar_t g_i = yout[0];

        // V2ad p_i; p_i << cx_i, cy_i;
        // const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R), ad_scalar_t(40), ad_scalar_t(1e-12), ad_scalar_t(1e-12));
        // const ad_scalar_t g_i = sdg.d;

        y.segment(i*1,1) << lambda_i;                   // λ ≥ 0
                            // g_i,                        // g ≥ 0
                            // CONTACT_EPS-g_i*lambda_i;   // ε - λ*g ≥ 0
    }
};

// ------------------------------ Initial constraint ---------------------------
ad_function_with_param_t pushboxInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(3);
    y.segment(0, 3) << x[0] - p[0], x[1] - p[1], x[2] - p[2];
};

ad_function_t pushboxStayOnSurfaceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;
    V2ad half; half << ad_scalar_t(a), ad_scalar_t(b);

    y.resize(N*1);
    for (size_t i=0; i<N; ++i)
    {
        const size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lambda_i= x[idx + 5];

        // std::vector<ad_scalar_t> xin(2), yout(1);
        // xin[0] = cx_i; xin[1] = cy_i;
        // g_sdf_d(xin, yout);
        // ad_scalar_t g_i = yout[0];

        // V2ad p_i; p_i << cx_i, cy_i;
        // ad_scalar_t g_i = CRISP::sdf::BoxRoundedSmoothSDF<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R), ad_scalar_t(40), ad_scalar_t(1e-12));

        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R), ad_scalar_t(40), ad_scalar_t(1e-12), ad_scalar_t(1e-12));
        const ad_scalar_t g_i = sdg.d;

        y.segment(i*1,1) << g_i;                        // g = 0
    }
};

// --------------------------------- Objective ---------------------------------
ad_function_with_param_t pushboxObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        const size_t idx  = i * (num_state + num_control);
        size_t idx_next = (i+1) * (num_state + num_control);
        ad_scalar_t px_i  = x[idx + 0];
        ad_scalar_t py_i  = x[idx + 1];
        ad_scalar_t theta_i  = x[idx + 2];
        ad_scalar_t cx_i  = x[idx + 3];
        ad_scalar_t cy_i  = x[idx + 4];
        ad_scalar_t lambda_i = x[idx + 5];
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

// ---------------------------------- Main -------------------------------------
int main() {
    const size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushboxSDF";
    std::string folderName = "model";
    OptimizationProblem pushboxProblem(variableNum, problemName);

    auto obj      = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective", pushboxObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxDynamicConstraints", pushboxDynamicConstraints);
    auto initial  = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);
    auto contact_surface = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxStayOnSurfaceConstraints", pushboxStayOnSurfaceConstraints);
    auto contact  = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addObjective(obj);
    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addEqualityConstraint(initial);
    pushboxProblem.addEqualityConstraint(contact_surface);
    pushboxProblem.addInequalityConstraint(contact);

    // parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);

    // xInitialStates << 0.4, 0.0, 0.0;
    xInitialStates << 0.35766736, 0.08357876, 0.42412436;       // Suboptimal initial condition
    // xInitialStates << 0.35766736, 0.08357876, 1.42412436;    // Suboptimal initial condition
    xInitialGuess.setZero();
    xFinalStates << 0.4, 0.3, 0.0;
    std::cout << "Initial State: " << xInitialStates.transpose() << std::endl;
    std::cout << "Final State: " << xFinalStates.transpose() << std::endl;

    // Setting initial guess
    const size_t num_segments = 18;
    const scalar_t theta = 12 * 2 * M_PI / num_segments;
    for (size_t i = 0; i < N; ++i) {
        const size_t idx = i * (num_state + num_control);
        const scalar_t alpha = static_cast<scalar_t>(i) / static_cast<scalar_t>(N-1);
        xInitialGuess[idx + 0] = alpha * (2*std::cos(theta));   // px
        xInitialGuess[idx + 1] = alpha * (2*std::sin(theta));   // py
        xInitialGuess[idx + 2] = alpha * theta;                 // th
        xInitialGuess[idx + 3] = a + 0.01;                      // cx on right face
        xInitialGuess[idx + 4] = 0;                             // cy
        xInitialGuess[idx + 5] = 0;                             // λ
    }

    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);

    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    // solver.setHyperParameters("trustRegionInitRadius", vector_t::Constant(1, 1.0));
    // solver.setHyperParameters("trustRegionMaxRadius", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("etaLow", vector_t::Constant(1, 0.25));
    // solver.setHyperParameters("etaHigh", vector_t::Constant(1, 0.75));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 10.0));
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e25));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-7));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setHyperParameters("WeightedTolFactor", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("secondOrderCorrection", vector_t::Constant(1, 1));

    // choose a final target on a circle
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

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_sdf_roundedsmooth.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();
}

