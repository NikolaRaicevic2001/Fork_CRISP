#include "solver_core/SolverInterface.h"
#include "sdf/RoundBox.h"

#include <cmath>                // use <cmath>, not <math.h>
#include <chrono>               // use <chrono> for timing
#include <iomanip>              // use <iomanip> for formatting
#include <filesystem>           // use <filesystem> for path manipulation
#include <cppad/cppad.hpp>      // for CppAD::cos/sin with AD

using namespace CRISP;

// Define model parameters for pushbox
const scalar_t a = 0.1;
const scalar_t b = 0.1;
const scalar_t m = 1;
const scalar_t mu = 0.5;
const scalar_t g = 9.8;
const scalar_t r = std::sqrt(a * a + b * b);    
const scalar_t c = 0.4;                     
const scalar_t dt = 0.1;
const size_t N = 100;                               // number of time steps
const size_t num_state = 3;                         // STATE  (3) : [px, py, θ]
const size_t num_control = 3;                       // CONTROL (3) : [cx, cy, λ ]

const size_t num_segments = 18;
const scalar_t theta = 12 * 2 * M_PI / num_segments;

// SDF rounding radius (meters)
constexpr scalar_t ROUND_R = 0.01;
constexpr scalar_t EPS_COMP = 1e-9;
constexpr scalar_t W_PENETRATION = 1e-3;

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();

// -------------------------- Helper Functions ---------------------------------
static inline ad_scalar_t pospart(const ad_scalar_t& z) {
    return CppAD::CondExpGt(z, ad_scalar_t(0), z, ad_scalar_t(0));
}

auto printIneq = [](const vector_t& v, const char* tag) {
    using std::cout; using std::endl;
    cout << tag << " size=" << v.size() << "   max(v) = " << v.array().maxCoeff() << "   min(v) = " << v.array().minCoeff() << "   (-v).maxCoeff = " << (-v).array().maxCoeff() << endl;

    // Show first few 3-tuples [λ_i, g_i, ε - λ_i*g_i]
    const Eigen::Index blocks = v.size() / 3;
    const Eigen::Index show   = std::min<Eigen::Index>(3, blocks);
    for (Eigen::Index i = 0; i < show; ++i) {
        cout << "  ineq[" << i << "] = " << v.segment<3>(3*i).transpose() << endl;
    }
    if (blocks > 0) {
        const Eigen::Index last = blocks - 1;
        cout << "  ineq[last] = " << v.segment<3>(3*last).transpose() << endl;
    }
};

auto printEq = [](const vector_t& v, const char* tag, int N) {
    using std::cout; using std::endl;
    cout << tag << " size=" << v.size() << "   ||eq||_inf = " << v.cwiseAbs().maxCoeff() << endl;

    // Argmax (index of the largest absolute entry)
    Eigen::Index imax;
    const double vmax = v.cwiseAbs().maxCoeff(&imax);

    // Decode index -> (block, component)
    // Layout: [0..2] = init(3), then (N-1) dynamics blocks of size 3
    static const char* comp_name[3] = {"px","py","theta"};
    bool is_init = (imax < 3);
    Eigen::Index blk = -1, comp = (is_init ? imax : (imax - 3) % 3);
    if (!is_init) blk = (imax - 3) / 3; // blk in [0 .. N-2]

    cout << "  worst entry: idx=" << imax << "  value=" << v(imax) << "  |value|=" << vmax << endl;

    if (is_init) {
        cout << "  => corresponds to INIT constraint, component " << comp_name[comp] << endl;
        cout << "  init residual:  " << v.segment(0,3).transpose() << endl;
    } else {
        cout << "  => corresponds to DYNAMICS constraint at step k=" << blk
             << ", component " << comp_name[comp] << endl;
        cout << "  dyn[" << blk << "] residual: " << v.segment(3 + 3*blk, 3).transpose() << endl;
    }

    // Also show first, and last block for context
    if (v.size() >= 6) {
        cout << "  dyn[0] residual : " << v.segment(3,3).transpose() << endl;
        cout << "  dyn[last] resid : " << v.tail(3).transpose() << endl;
    }
};
// ---------------------------------------------------------------------------

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
        ad_scalar_t th_i    = x[idx + 2];
        ad_scalar_t cx_i    = x[idx + 3];
        ad_scalar_t cy_i    = x[idx + 4];
        ad_scalar_t lam_i   = x[idx + 5];

        ad_scalar_t px_next     = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next     = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next  = x[idx + (num_state + num_control) + 2];

        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        const V2ad n_i = -sdg.n;

        const ad_scalar_t cth = CppAD::cos(th_i);
        const ad_scalar_t sth = CppAD::sin(th_i);

        const ad_scalar_t Fx       = lam_i * (cth*n_i.x() - sth*n_i.y());
        const ad_scalar_t Fy       = lam_i * (sth*n_i.x() + cth*n_i.y());
        const ad_scalar_t torque_z = lam_i * (cx_i*n_i.y() - cy_i*n_i.x());

        const ad_scalar_t denom_lin = ad_scalar_t(mu * m * g);
        const ad_scalar_t denom_ang = ad_scalar_t(mu * m * g * c * r);

        const ad_scalar_t px_dot  = Fx / denom_lin;
        const ad_scalar_t py_dot  = Fy / denom_lin;
        const ad_scalar_t th_dot  = torque_z / denom_ang;

        y.segment(i * num_state, num_state) <<  px_next - px_i - px_dot * ad_scalar_t(dt),
                                                py_next - py_i - py_dot * ad_scalar_t(dt),
                                                theta_next - th_i - th_dot * ad_scalar_t(dt);
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
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i  = x[idx + 3];
        ad_scalar_t cy_i  = x[idx + 4];
        ad_scalar_t lam_i = x[idx + 5];

        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        const ad_scalar_t g_i = sdg.d;

        y.segment(i*3,3) << lam_i,                              // λ ≥ 0
                            g_i,                                // g ≥ 0
                            ad_scalar_t(EPS_COMP) - g_i*lam_i;  // λ g ≤ ε
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
    V2ad half; half << ad_scalar_t(a), ad_scalar_t(b);
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);

    for (size_t i = 0; i < N; ++i) {
        const size_t idx  = i * (num_state + num_control);
        ad_scalar_t px_i  = x[idx + 0];
        ad_scalar_t py_i  = x[idx + 1];
        ad_scalar_t th_i  = x[idx + 2];
        ad_scalar_t cx_i  = x[idx + 3];
        ad_scalar_t cy_i  = x[idx + 4];
        ad_scalar_t lam_i = x[idx + 5];

        ad_matrix_t Q(num_state, num_state);
        Q.setZero(); Q(0,0)=100; Q(1,1)=100; Q(2,2)=100;

        V2ad p_i; p_i << cx_i, cy_i;
        auto sdf = CRISP::sdf::sdfBoxRoundedSmooth<ad_scalar_t>(p_i, half, ad_scalar_t(ROUND_R));
        ad_scalar_t g_i = sdf.d;
        ad_scalar_t pen = pospart(-g_i);
        tracking_cost += ad_scalar_t(W_PENETRATION) * pen * pen;

        // terminal tracking
        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0], py_i - p[1], th_i - p[2];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        // control effort (through λ)
        if (i < N - 1) {
            control_cost += ad_scalar_t(0.001) * lam_i * lam_i;
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

    auto obj      = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective",          pushboxObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum,          problemName, folderName, "pushboxDynamicConstraints",  pushboxDynamicConstraints);
    auto contact  = std::make_shared<ConstraintFunction>(variableNum,          problemName, folderName, "pushboxContactConstraints",  pushboxContactConstraints);
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

    xInitialStates << 0, 0, 0;

    // Setting initial guess
    xInitialGuess.setZero();
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

    {
        using V2d = Eigen::Matrix<double,2,1>;
        const V2d half_d(a, b);

        Eigen::SparseMatrix<double> Js = contact->getGradient(xInitialGuess);

        // Prepare output
        const auto out = PROJECT_ROOT / "examples/pushbox/results/results_pushbox_sdf_roundedsmooth_gradcheck.csv";
        std::ofstream os(out);
        os << std::setprecision(17);
        os << "i,cx,cy,jac_nx_raw,jac_ny_raw,jac_nx,jac_ny,sdf_nx,sdf_ny,dot,angle_deg\n";

        const size_t vars_per_knot = num_state + num_control; // 6
        const size_t Ncols = Js.cols();                       // N*6
        const size_t N     = Ncols / vars_per_knot;

        for (size_t i = 0; i < N - 1; ++i) {
            const size_t row_g = 3*i + 1;                 // g_i row
            const size_t col_cx = i*vars_per_knot + 3;    // cx_i column
            const size_t col_cy = i*vars_per_knot + 4;    // cy_i column

            // Extract raw partials (gap wrt cx, cy)
            const double jx_raw = Js.coeff(row_g, col_cx);
            const double jy_raw = Js.coeff(row_g, col_cy);

            // Normalize Jacobian gradient
            const double nrm = std::sqrt(jx_raw*jx_raw + jy_raw*jy_raw) + 1e-12;
            const double jnx = jx_raw / nrm;
            const double jny = jy_raw / nrm;

            // Evaluate SDF normal at (cx,cy) using the SAME SDF variant as the constraint
            const double cx = static_cast<double>(xInitialGuess[i*vars_per_knot + 3]);
            const double cy = static_cast<double>(xInitialGuess[i*vars_per_knot + 4]);
            const V2d p(cx, cy);
            const auto sdg = CRISP::sdf::sdfBoxRoundedSmooth<double>(p, half_d, double(ROUND_R));
            const double snx = sdg.n.x(), sny = sdg.n.y();

            // Direction agreement
            double dot = jnx*snx + jny*sny;
            dot = std::max(-1.0, std::min(1.0, dot));
            const double angle_deg = std::acos(dot) * 180.0 / M_PI;

            os << i << ',' << cx << ',' << cy << ',' << jx_raw << ',' << jy_raw << ',' << jnx << ',' << jny << ',' << snx << ',' << sny << ',' << dot << ',' << angle_deg << '\n';
        }
        os.close();
    }

    SolverParameters params;
    SolverInterface solver(pushboxProblem, params);

    solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e12));
    solver.setHyperParameters("trailTol",       vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    // solver.setHyperParameters("WeightedMode",   vector_t::Constant(1, 1));

    // choose a final target on a circle
    xFinalStates << 2*std::cos(theta), 2*std::sin(theta), theta;
    solver.setProblemParameters("pushboxObjective", xFinalStates);

    // Check constraints at initial guess
    const vector_t ineq_init = pushboxProblem.evaluateInequalityConstraints(xInitialGuess);
    printIneq(ineq_init, "[INEQ @ init]");
    const vector_t eq_init = pushboxProblem.evaluateEqualityConstraints(xInitialGuess);
    printEq(eq_init, "[EQ   @ init]", static_cast<int>(N));

    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();

    // Check constraints at solution
    const vector_t ineq_opt = pushboxProblem.evaluateInequalityConstraints(xOptimal);
    printIneq(ineq_opt, "[INEQ @ opt]");
    const vector_t eq_opt = pushboxProblem.evaluateEqualityConstraints(xOptimal);
    printEq(eq_opt, "[EQ   @ opt]", static_cast<int>(N));

    std::ofstream log(PROJECT_ROOT / "examples/pushbox/results/results_pushbox_sdf_roundedsmooth.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();
}

