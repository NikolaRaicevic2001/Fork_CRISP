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
const size_t N = 100;                               // number of time steps
const size_t num_state = 3;                         // STATE  (3) : [px, py, θ]
const size_t num_control = 3;                       // CONTROL (3) : [cx, cy, λ ]

// SDF function parameters
constexpr scalar_t ROUND_R = 0.01;
constexpr scalar_t CONTACT_EPS = 1e-6;

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();

// ---------- Atomic for SDF that returns [d, nx, ny] with custom derivatives ----------
template <class Base>
class AtomicSdf : public CppAD::atomic_four<Base> {
public:
    AtomicSdf() : CppAD::atomic_four<Base>("sdf_box_rounded_smooth") {}

    // Smooth Heaviside function
    static inline Base smooth_heaviside(Base t, Base eps) {
        // 0.5 * (1 + tanh(t / eps)), eps > 0
        return Base(0.5) * ( Base(1) + CppAD::tanh( t / eps ) );
    }

    // Evaluate SDF and its Jacobian at (cx, cy)
    static inline void eval_and_jac(const Base& cx, const Base& cy,Base& d, Base& nx, Base& ny,Base J[3][2])
    {
        // Values from your SDF
        Eigen::Matrix<Base,2,1> P(cx, cy);
        Eigen::Matrix<Base,2,1> H; H << Base(a), Base(b);
        auto sd = CRISP::sdf::sdfBoxRoundedSmooth<Base>(P, H, Base(ROUND_R));

        d  = sd.d;
        nx = sd.n.x();
        ny = sd.n.y();

        // Smooth corner mask
        const Base wx  = CppAD::abs(cx) - Base(a - ROUND_R);
        const Base wy  = CppAD::abs(cy) - Base(b - ROUND_R);
        const Base eps = Base(1e-3);
        const Base sx  = smooth_heaviside(wx, eps);
        const Base sy  = smooth_heaviside(wy, eps);
        const Base mask = sx * sy;

        // Analytic Jacobian
        // ∂d/∂p = n
        J[0][0] = nx;                 // ∂d/∂cx
        J[0][1] = ny;                 // ∂d/∂cy

        // ∂n/∂p = mask * (1/R) * (I - n n^T)
        const Base k    = Base(1) / Base(ROUND_R);
        const Base nnxx = nx*nx;
        const Base nnyy = ny*ny;
        const Base nnxy = nx*ny;

        J[1][0] = mask * k * (Base(1) - nnxx);    // ∂nx/∂cx
        J[1][1] = mask * k * (-nnxy);             // ∂nx/∂cy
        J[2][0] = mask * k * (-nnxy);             // ∂ny/∂cx
        J[2][1] = mask * k * (Base(1) - nnyy);    // ∂ny/∂cy
    }

private:
  bool for_type(size_t /*call_id*/, const CppAD::vector<CppAD::ad_type_enum>& type_x, CppAD::vector<CppAD::ad_type_enum>& type_y) override 
    { // x = [cx, cy]  ->  y = [d, nx, ny]
    type_y[0] = type_x[0];
    type_y[1] = type_x[0];
    type_y[2] = type_x[0];
    return true;
    }

    bool forward(size_t /*call_id*/, const CppAD::vector<bool>& /*select_y*/, size_t order_low, size_t order_up, const CppAD::vector<Base>& taylor_x, CppAD::vector<Base>& taylor_y) override
    { // forward(order_low, order_up): inputs taylor_x, outputs taylor_y, all in Base
        const size_t q = order_up + 1;
        const Base cx0 = taylor_x[0*q + 0];
        const Base cy0 = taylor_x[1*q + 0];

        Base d0, nx0, ny0; Base J[3][2];
        eval_and_jac(cx0, cy0, d0, nx0, ny0, J);

        // order 0
        if (order_low <= 0) {
            taylor_y[0*q + 0] = d0;
            taylor_y[1*q + 0] = nx0;
            taylor_y[2*q + 0] = ny0;
        }
        if (order_up < 1) return true;

        // order 1: y' = J * x'
        const Base dcx = taylor_x[0*q + 1];
        const Base dcy = taylor_x[1*q + 1];

        taylor_y[0*q + 1] = J[0][0]*dcx + J[0][1]*dcy; // d'
        taylor_y[1*q + 1] = J[1][0]*dcx + J[1][1]*dcy; // nx'
        taylor_y[2*q + 1] = J[2][0]*dcx + J[2][1]*dcy; // ny'
        return true;
    }

    bool reverse(size_t /*call_id*/,
                const CppAD::vector<bool>& /*select_x*/,
                size_t order_up,
                const CppAD::vector<Base>& taylor_x, const CppAD::vector<Base>& taylor_y,
                CppAD::vector<Base>& partial_x, const CppAD::vector<Base>& partial_y) override
    { // reverse(order_up, tx, ty, px, py): accumulate px += J^T * py
        // We implement first-order reverse (order_up == 0)
        if (order_up != 0) return false;
        const size_t q = order_up + 1;

        // base point
        const Base cx0 = taylor_x[0*q + 0];
        const Base cy0 = taylor_x[1*q + 0];

        // recompute y and J at base (or reuse ty[.,0] for y if you prefer)
        Base d0, nx0, ny0; Base J[3][2];
        eval_and_jac(cx0, cy0, d0, nx0, ny0, J);

        // seeds on outputs (order 0)
        const Base pd  = partial_y[0*q + 0];
        const Base pnx = partial_y[1*q + 0];
        const Base pny = partial_y[2*q + 0];

        // px += J^T * py
        partial_x[0*q + 0] += J[0][0]*pd + J[1][0]*pnx + J[2][0]*pny; // ∂H/∂cx
        partial_x[1*q + 0] += J[0][1]*pd + J[1][1]*pnx + J[2][1]*pny; // ∂H/∂cy
        return true;
    }

    bool jac_sparsity(
        size_t /*call_id*/,
        bool /*dependency*/,
        const CppAD::vector<bool>& /*ident_zero_x*/,
        const CppAD::vector<bool>& select_x,
        const CppAD::vector<bool>& select_y,
        CppAD::sparse_rc< CppAD::vector<size_t> >& pattern_out
    ) override
    { // Provide Jacobian sparsity pattern (dense 3x2, filtered by selections)
        const size_t n = 2; // inputs: cx, cy
        const size_t m = 3; // outputs: d, nx, ny

        // count nnz according to selections
        size_t nnz = 0;
        for (size_t i = 0; i < m; ++i) if (select_y[i])
            for (size_t j = 0; j < n; ++j) if (select_x[j])
                ++nnz;

        pattern_out.resize(m, n, nnz);
        size_t k = 0;
        for (size_t i = 0; i < m; ++i) if (select_y[i])
            for (size_t j = 0; j < n; ++j) if (select_x[j]) {
                pattern_out.set(k, i, j); // (k-th nonzero) at row i, col j
                ++k;
            }
        return true;
    }

};

using CGD = CppAD::cg::CG<double>;
static AtomicSdf<CGD>    g_sdf_dn;   // used by the solver (AD<CG<double>>)
static AtomicSdf<double> g_sdf_dbg;  // used only for logging (AD<double>)

// -------------------------- Helper Functions --------------------------
static inline std::pair<double,double>
get_cx_cy_from_solution(const vector_t& sol, size_t i) 
{ // Pull (cx, cy) at knot i from the flat decision vector
    const size_t stride = (num_state + num_control);
    const size_t base   = i * stride;
    return { sol[base + 3], sol[base + 4] }; // [cx, cy]
}

static inline void direct_sdf_eval_and_jac(double cx, double cy, double& d, double& nx, double& ny, double J[3][2])
{ // Evaluate the SDF directly (no atomic) via a tiny AD<double> tape. Returns y=[d,nx,ny] and J=[3x2] wrt [cx,cy].
    using ADd = CppAD::AD<double>;

    // Independent vars
    CppAD::vector<ADd> ax(2); ax[0] = cx; ax[1] = cy;
    CppAD::Independent(ax);

    // Call the templated SDF with AD<double>
    Eigen::Matrix<ADd,2,1> P_ad, H_ad;
    P_ad << ax[0], ax[1];
    H_ad << ADd(a), ADd(b);
    auto sd_ad = CRISP::sdf::sdfBoxRoundedSmooth<ADd>(P_ad, H_ad, ADd(ROUND_R));

    // Outputs: [d, nx, ny]
    CppAD::vector<ADd> ay(3);
    ay[0] = sd_ad.d;
    ay[1] = sd_ad.n.x();
    ay[2] = sd_ad.n.y();

    // Build tape
    CppAD::ADFun<double> f;
    f.Dependent(ax, ay);
    f.optimize(); 

    // Values at (cx,cy)
    CppAD::vector<double> x0(2); x0[0] = cx; x0[1] = cy;
    CppAD::vector<double> y0 = f.Forward(0, x0);
    d  = y0[0]; nx = y0[1]; ny = y0[2];

    // Jacobian (row-major 3x2)
    CppAD::vector<double> Jflat = f.Jacobian(x0);
    J[0][0] = Jflat[0*2 + 0];  J[0][1] = Jflat[0*2 + 1];  // ∂d/∂cx,  ∂d/∂cy
    J[1][0] = Jflat[1*2 + 0];  J[1][1] = Jflat[1*2 + 1];  // ∂nx/∂cx, ∂nx/∂cy
    J[2][0] = Jflat[2*2 + 0];  J[2][1] = Jflat[2*2 + 1];  // ∂ny/∂cx, ∂ny/∂cy
}

// Dump direct (non-atomic) outputs + gradients for all knots to CSV
static void dump_atomic_gradients_csv(const vector_t& xsol, const std::string& csv_path, size_t max_rows = N, bool atomic = true)
{ // Dump atomic outputs + gradients for all knots to CSV
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Failed to open '" << csv_path << "' for writing\n";
        return;
    }
    csv << std::setprecision(17);
    csv << "i,cx,cy,d,nx,ny,ddcx,ddcy,dnxcx,dnxcy,dnycx,dnycy\n";

    for (size_t i = 0; i < std::min(max_rows, N); ++i) {
        auto [cx, cy] = get_cx_cy_from_solution(xsol, i);
        double d, nx, ny, J[3][2];

        if (atomic) {
            AtomicSdf<double>::eval_and_jac(cx, cy, d, nx, ny, J);
        } else {
            direct_sdf_eval_and_jac(cx, cy, d, nx, ny, J);
        }

        csv << i << "," << cx << "," << cy << "," 
            << d  << "," << nx  << "," << ny  << ","
            << J[0][0] << "," << J[0][1] << ","
            << J[1][0] << "," << J[1][1] << ","
            << J[2][0] << "," << J[2][1] << "\n";

        if (atomic) {
            if (i<2) {
                std::cout << "Atomic eval at i = " << i << ": d=" << d << ", n=[" << nx << "," << ny << "]\n";
                std::cout << "Atomic Jacobian : ddcx=" << J[0][0] << ", ddcy=" << J[0][1] << "\n";
                std::cout << "                : dnxcx=" << J[1][0] << ", dnxcy=" << J[1][1] << "\n";
                std::cout << "                : dnycx=" << J[2][0] << ", dnycy=" << J[2][1] << "\n";
            }
        } else {
            if (i<2) {
                std::cout << "Direct eval at i = " << i << ": d=" << d << ", n=[" << nx << "," << ny << "]\n";
                std::cout << "Direct Jacobian : ddcx=" << J[0][0] << ", ddcy=" << J[0][1] << "\n";
                std::cout << "                : dnxcx=" << J[1][0] << ", dnxcy=" << J[1][1] << "\n";
                std::cout << "                : dnycx=" << J[2][0] << ", dnycy=" << J[2][1] << "\n";
            }
        }
    }
    std::cerr << "Wrote atomic gradient log to " << csv_path << "\n";
}

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

        // V2ad p_i; p_i << cx_i, cy_i;
        // const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        // const V2ad n_i = -sdg.n;

        std::vector<ad_scalar_t> xin(2), yout(3);
        xin[0] = cx_i; xin[1] = cy_i;
        g_sdf_dn(xin, yout);
        ad_scalar_t d_i  = yout[0];
        ad_scalar_t nx_i = yout[1];
        ad_scalar_t ny_i = yout[2];
        V2ad n_i; n_i << -nx_i, -ny_i;

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

    y.resize((N-1)*3);
    for (size_t i=0; i<N-1; ++i)
    {
        const size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lambda_i= x[idx + 5];

        // V2ad p_i; p_i << cx_i, cy_i;
        // const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        // const ad_scalar_t g_i = sdg.d;

        std::vector<ad_scalar_t> xin(2), yout(3);
        xin[0] = cx_i; xin[1] = cy_i;
        g_sdf_dn(xin, yout);
        ad_scalar_t d_i  = yout[0];
        ad_scalar_t nx_i = yout[1];
        ad_scalar_t ny_i = yout[2];
        V2ad n_i; n_i << -nx_i, -ny_i;
        const ad_scalar_t g_i = d_i;

        y.segment(i*3,3) << lambda_i,                   // λ ≥ 0
                            g_i,                        // g ≥ 0
                            CONTACT_EPS-g_i*lambda_i;   // ε - λ * g ≥ 0
    }
};

// ------------------------------ Initial constraint ---------------------------
ad_function_with_param_t pushboxInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(3);
    y.segment(0, 3) << x[0] - p[0], x[1] - p[1], x[2] - p[2];
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
    auto contact  = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints", pushboxContactConstraints);
    auto initial  = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "pushboxInitialConstraints", pushboxInitialConstraints);
    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushboxProblem.addObjective(obj);
    pushboxProblem.addEqualityConstraint(dynamics);
    pushboxProblem.addEqualityConstraint(initial);
    pushboxProblem.addInequalityConstraint(contact);

    // parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);

    xInitialStates << 0.4, 0.0, 0.0;
    // xInitialStates << 0.35766736, 0.08357876, 0.42412436;  // Suboptimal initial condition
    // xInitialStates << 0.35766736, 0.08357876, 1.42412436;  // Suboptimal initial condition
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
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e10));
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

    std::cout << "<====== Comparing Atomic vs Direct SDF evaluations and gradients =======>\n";
    const auto csv_path = (PROJECT_ROOT / "examples/pushbox/results/results_pushbox_sdf_atomic_gradients.csv").string();
    dump_atomic_gradients_csv(xOptimal, csv_path, N, true);
    const auto csv_path_direct = (PROJECT_ROOT / "examples/pushbox/results/results_pushbox_sdf_direct_gradients.csv").string();
    dump_atomic_gradients_csv(xOptimal, csv_path_direct, N, false);
    std::cout << "<===========================================================================>\n";

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

