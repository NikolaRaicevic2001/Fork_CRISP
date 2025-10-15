#include "solver_core/SolverInterface.h"

#include <cppad/cppad.hpp>   
#include <filesystem>
#include <chrono>
#include "math.h"

#include "sdf/Circle.h"

using namespace CRISP;

// Define model parameters for circle
const scalar_t R = 0.05;                // radius of the circle
const scalar_t m = 1;                   // mass of the circle
const scalar_t mu = 0.5;                // friction coefficient
const scalar_t g = 9.8;                 // gravitational acceleration  
const scalar_t dt = 0.02;               // time step size
const size_t N = 100;                   // number of time steps
const size_t num_state = 2;             // STATE  (2) : [px, py]
const size_t num_control = 3;           // CONTROL (3) : [cx, cy, λ ]

// Global variables for the problem
static const std::filesystem::path PROJECT_ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();    

// ---------- Atomics for circle: SDF and GRAD (custom Jacobians) -------------
// x=[cx, cy] -> y=[d]
template <class Base>
class AtomicCircleSDF : public CppAD::atomic_four<Base> {
public:
    explicit AtomicCircleSDF(Base radius, Base eps = Base(0))
    : CppAD::atomic_four<Base>("circle_sdf_x2"), R_(radius), eps_(eps) {}

private:
    Base R_, eps_;

    bool for_type(size_t, const CppAD::vector<CppAD::ad_type_enum>& tx,
                  CppAD::vector<CppAD::ad_type_enum>& ty) override { ty[0]=tx[0]; return true; }

    bool forward(size_t, const CppAD::vector<bool>&, size_t ol, size_t ou,
                 const CppAD::vector<Base>& tX, CppAD::vector<Base>& tY) override {
        const size_t q = ou + 1;
        const Base cx0 = tX[0*q + 0], cy0 = tX[1*q + 0];
        Eigen::Matrix<Base,2,1> p(cx0, cy0);

        const Base d0 = CRISP::sdf::sdfCircle(p, R_, eps_);
        if (ol <= 0) tY[0*q + 0] = d0;
        if (ou < 1)  return true;

        const Base dcx = tX[0*q + 1], dcy = tX[1*q + 1];
        const auto g = CRISP::sdf::gradCircle(p, R_, eps_);
        tY[0*q + 1] = g.x()*dcx + g.y()*dcy;
        return true;
    }

    bool reverse(size_t, const CppAD::vector<bool>&, size_t ou,
                 const CppAD::vector<Base>& tX, const CppAD::vector<Base>&,
                 CppAD::vector<Base>& pX, const CppAD::vector<Base>& pY) override {
        if (ou != 0) return false;
        const size_t q = 1;
        const Base cx0 = tX[0*q + 0], cy0 = tX[1*q + 0];
        Eigen::Matrix<Base,2,1> p(cx0, cy0);
        const auto g = CRISP::sdf::gradCircle(p, R_, eps_);
        const Base pd = pY[0*q + 0];
        pX[0*q + 0] += g.x() * pd;
        pX[1*q + 0] += g.y() * pd;
        return true;
    }

    bool jac_sparsity(size_t, bool, const CppAD::vector<bool>&,
                      const CppAD::vector<bool>& sel_x, const CppAD::vector<bool>& sel_y,
                      CppAD::sparse_rc< CppAD::vector<size_t> >& pat) override {
        size_t nnz = (sel_y[0]? (sel_x[0]?1:0) + (sel_x[1]?1:0) : 0);
        pat.resize(1, 2, nnz);
        size_t k=0; if (sel_y[0]) { if (sel_x[0]) pat.set(k++,0,0); if (sel_x[1]) pat.set(k++,0,1); }
        return true;
    }
};

// x=[cx, cy] -> y=[nx, ny]
template <class Base>
class AtomicCircleGrad : public CppAD::atomic_four<Base> {
public:
    explicit AtomicCircleGrad(Base radius, Base eps = Base(0))
    : CppAD::atomic_four<Base>("circle_grad_x2"), R_(radius), eps_(eps) {}

private:
    Base R_, eps_;

    bool for_type(size_t, const CppAD::vector<CppAD::ad_type_enum>& tx,
                  CppAD::vector<CppAD::ad_type_enum>& ty) override { ty[0]=tx[0]; ty[1]=tx[0]; return true; }

    bool forward(size_t, const CppAD::vector<bool>&, size_t ol, size_t ou,
                 const CppAD::vector<Base>& tX, CppAD::vector<Base>& tY) override {
        const size_t q = ou + 1;
        const Base cx0 = tX[0*q + 0], cy0 = tX[1*q + 0];
        Eigen::Matrix<Base,2,1> p(cx0, cy0);

        const auto g0 = CRISP::sdf::gradCircle(p, R_, eps_);
        if (ol <= 0) { tY[0*q + 0] = g0.x(); tY[1*q + 0] = g0.y(); }
        if (ou < 1)  return true;

        const Base dcx = tX[0*q + 1], dcy = tX[1*q + 1];
        const auto H = CRISP::sdf::hessianCircle(p, R_, eps_);
        tY[0*q + 1] = H(0,0)*dcx + H(0,1)*dcy;
        tY[1*q + 1] = H(1,0)*dcx + H(1,1)*dcy;
        return true;
    }

    bool reverse(size_t, const CppAD::vector<bool>&, size_t ou,
                 const CppAD::vector<Base>& tX, const CppAD::vector<Base>&,
                 CppAD::vector<Base>& pX, const CppAD::vector<Base>& pY) override {
        if (ou != 0) return false;
        const size_t q = 1;
        const Base cx0 = tX[0*q + 0], cy0 = tX[1*q + 0];
        Eigen::Matrix<Base,2,1> p(cx0, cy0);
        const auto H = CRISP::sdf::hessianCircle(p, R_, eps_);
        const Base pnx = pY[0*q + 0], pny = pY[1*q + 0];
        // px += H^T * py  (H is symmetric)
        pX[0*q + 0] += H(0,0)*pnx + H(0,1)*pny;
        pX[1*q + 0] += H(1,0)*pnx + H(1,1)*pny;
        return true;
    }

    bool jac_sparsity(size_t, bool, const CppAD::vector<bool>&,
                      const CppAD::vector<bool>& sel_x, const CppAD::vector<bool>& sel_y,
                      CppAD::sparse_rc< CppAD::vector<size_t> >& pat) override {
        size_t nnz=0; for (int i=0;i<2;++i) if (sel_y[i]) for (int j=0;j<2;++j) if (sel_x[j]) ++nnz;
        pat.resize(2,2,nnz);
        size_t k=0; for (int i=0;i<2;++i) if (sel_y[i]) for (int j=0;j<2;++j) if (sel_x[j]) pat.set(k++, i, j);
        return true;
    }
};

using CGD = CppAD::cg::CG<double>;
static AtomicCircleSDF<CGD>    g_circle_sdf_cg( CGD(R), CGD(1e-12) );
static AtomicCircleSDF<double> g_circle_sdf_dbg( R, 1e-12 );

static AtomicCircleGrad<CGD>    g_circle_grad_cg( CGD(R), CGD(1e-12) );
static AtomicCircleGrad<double> g_circle_grad_dbg( R, 1e-12 );

// ----------------------- Dynamics constraints ------------------------
// define the dynamics constraints
ad_function_t pushcircleDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;
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

        // // n = grad(d) at p_i  (unit outward normal)
        // std::vector<ad_scalar_t> xin(2), yout(2);
        // xin[0] = p_i.x();  xin[1] = p_i.y();
        // g_circle_grad_cg(xin, yout);
        // V2ad n_i; n_i << -yout[0], -yout[1];   // inward normal for pushing

        // outward unit normal at the surface; pushing uses inward
        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdgCircle<ad_scalar_t>(p_i, ad_scalar_t(R));
        V2ad n_i = -sdg.n;   // inward normal for pushing

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

// ------------------- Contact Implicit Constraints --------------------
// contact implicit constraints for pushcircle
ad_function_t pushcircleContactConstraints = [](const ad_vector_t& x, ad_vector_t& y){
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;
    y.resize((N-1)*1);

    for (size_t i=0; i<N-1; ++i)
    {
        size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t cx_i    = x[idx + 2];
        ad_scalar_t cy_i    = x[idx + 3];
        ad_scalar_t lam_i   = x[idx + 4];

        y.segment(i*1,1) << lam_i;          // λ ≥ 0  (handled as inequality)
                            // g_i,            // g ≥ 0
                            // -g_i*lam_i;     // -λ·g ≥ 0   ⇒ complementarity
    }
};

// initial constraints
ad_function_with_param_t pushcircleInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(2);
    y.segment(0, 2) << x[0] - p[0], x[1] - p[1];
};

// Stay on the surface constraint
ad_function_t pushcircleStayOnSurfaceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    using V2ad = Eigen::Matrix<ad_scalar_t,2,1>;

    y.resize(N*1);
    for (size_t i=0; i<N; ++i)
    {
        size_t idx = i*(num_state+num_control);
        ad_scalar_t px_i    = x[idx + 0];
        ad_scalar_t py_i    = x[idx + 1];
        ad_scalar_t cx_i    = x[idx + 2];
        ad_scalar_t cy_i    = x[idx + 3];
        ad_scalar_t lam_i   = x[idx + 4];

        V2ad p_i; p_i << cx_i, cy_i;
        const auto sdg = CRISP::sdf::sdgCircle<ad_scalar_t>(p_i, ad_scalar_t(R));
        ad_scalar_t g_i = sdg.d;

        // std::vector<ad_scalar_t> xin(2), yout(1);
        // xin[0] = p_i.x(); xin[1] = p_i.y();
        // g_circle_sdf_cg(xin, yout);
        // ad_scalar_t g_i = yout[0];

        y.segment(i*1,1) << g_i;                        // g = 0
    }
};

// cost function for pushcircle
ad_function_with_param_t pushcircleObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        const size_t idx = i * (num_state + num_control);
        size_t idx_next = (i+1) * (num_state + num_control);
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
        ad_matrix_t P(num_state, num_state);
        P.setZero();
        P(0, 0) = 0.01;
        P(1, 1) = 0.01;
        ad_matrix_t R(1, 1);
        R.setZero();
        R(0, 0) = 0.0001;

        // Penalize the tracking error at the final time step
        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0], py_i - p[1];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        // Penalize large distance traveled by the box
        if (i < N - 1) {
            ad_vector_t tracking_error_whole(num_state);
            tracking_error_whole << px_i - p[0], py_i - p[1];
            tracking_cost += tracking_error_whole.transpose() * P * tracking_error_whole;
        }

        // Penalize the contact forces to prevent excessive forces
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
    auto stayOnSurface = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushcircleStayOnSurfaceConstraints", pushcircleStayOnSurfaceConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //
    pushcircleProblem.addObjective(obj);
    pushcircleProblem.addEqualityConstraint(dynamics);
    pushcircleProblem.addInequalityConstraint(contact);
    pushcircleProblem.addEqualityConstraint(initial);
    pushcircleProblem.addEqualityConstraint(stayOnSurface);

    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    
    xInitialStates << 0.5, 0.2;
    // xInitialStates << 0.3, 0.3;
    // xInitialStates << -0.3, 0.5;

    // set zero initial guess
    xInitialGuess.setZero();
    for (size_t k = 2; k < xInitialGuess.size(); k += (num_state+num_control))
    {
        xInitialGuess[k]   =  R+0.01;   // cx_i
        xInitialGuess[k+1] =  0;   // cy_i
    }

    SolverParameters params;
    SolverInterface solver(pushcircleProblem, params);
    solver.setProblemParameters("pushcircleInitialConstraints", xInitialStates);
    // solver.setHyperParameters("maxIterations", vector_t::Constant(1, 15000));
    // solver.setHyperParameters("trustRegionInitRadius", vector_t::Constant(1, 1.0));
    // solver.setHyperParameters("trustRegionMaxRadius", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("etaLow", vector_t::Constant(1, 0.25));
    // solver.setHyperParameters("etaHigh", vector_t::Constant(1, 0.75));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 10.0));
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e30));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-7));
    solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    solver.setHyperParameters("WeightedTolFactor", vector_t::Constant(1, 10.0));
    // solver.setHyperParameters("secondOrderCorrection", vector_t::Constant(1, 1));

    xFinalStates << 1.0, 1.0;
    solver.setProblemParameters("pushcircleObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    xOptimal = solver.getSolution();

    std::ofstream log(PROJECT_ROOT / "examples/pushcircle/results/results_pushcircle_sdf.csv");
    for (size_t k = 0; k < xOptimal.size(); ++k) log << xOptimal[k] << '\n';
    log.close();                              

    }



