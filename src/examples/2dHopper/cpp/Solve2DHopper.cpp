#include "solver_core/SolverInterface.h"
#include "common/MatlabHelper.h"
#include <chrono>
#include "math.h"

using namespace ContactSolver;

// Define model model parameters for pushbox
const scalar_t m = 1.0;
const scalar_t l_0 = 1.0;
const scalar_t r_0 = 0.8;
const scalar_t g = 9.81;
const scalar_t dt = 0.02;
const size_t N = 200; // number of time steps
const size_t num_state = 8;
const size_t num_control = 3;
const size_t num_dynamic_constraints_per_step = 9;
// full states of 2D hopper:
// v = [px, py, qx, qy, theta, r, px_dot, py_dot, u1, u2, lambda]
ad_function_t HopperDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y){
    y.resize((N - 1) * num_dynamic_constraints_per_step + 2);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t qx_i = x[idx + 2];
        ad_scalar_t qy_i = x[idx + 3];
        ad_scalar_t theta_i = x[idx + 4];
        ad_scalar_t r_i = x[idx + 5];
        ad_scalar_t px_dot_i = x[idx + 6];
        ad_scalar_t py_dot_i = x[idx + 7];
        ad_scalar_t u1_i = x[idx + 8];
        ad_scalar_t u2_i = x[idx + 9];
        ad_scalar_t lambda_i = x[idx + 10];

        ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t qx_next = x[idx + (num_state + num_control) + 2];
        ad_scalar_t qy_next = x[idx + (num_state + num_control) + 3];
        ad_scalar_t theta_next = x[idx + (num_state + num_control) + 4];
        ad_scalar_t r_next = x[idx + (num_state + num_control) + 5];
        ad_scalar_t px_dot_next = x[idx + (num_state + num_control) + 6];
        ad_scalar_t py_dot_next = x[idx + (num_state + num_control) + 7];
        ad_scalar_t u1_next = x[idx + (num_state + num_control) + 8];
        ad_scalar_t u2_next = x[idx + (num_state + num_control) + 9];
        ad_scalar_t lambda_next = x[idx + (num_state + num_control) + 10];

        ad_scalar_t constraint1 = px_next - px_i - px_dot_next * dt;
        ad_scalar_t constraint2 = py_next - py_i - py_dot_next * dt;
        ad_scalar_t constraint3 = (m*(px_dot_next-px_dot_i) + u2_i*sin(theta_i)*dt);
        ad_scalar_t constraint4 = (m*(py_dot_next-py_dot_i) - u2_i*cos(theta_i)*dt + m*g*dt);
        ad_scalar_t constraint6 = (l_0-r_i)*cos(theta_i) - py_i + qy_i;
        ad_scalar_t constraint7 = (l_0-r_i)*sin(theta_i) - qx_i + px_i;
        ad_scalar_t constraint8 =  r_i * (qx_next - qx_i);
        ad_scalar_t constraint9 = r_i * (qy_next - qy_i);
        ad_scalar_t constraint10 =  qy_i * (theta_next - theta_i - u1_i * dt);

        y.segment(i * num_dynamic_constraints_per_step, num_dynamic_constraints_per_step) << constraint1,
                                                                                            constraint2,
                                                                                            constraint3,
                                                                                            constraint4,
                                                                                            constraint6,
                                                                                            constraint7,
                                                                                            constraint8,
                                                                                            constraint9,
                                                                                            constraint10;
        if (i == N-2){
            ad_scalar_t constraint_final1 = (l_0-r_next)*cos(theta_next) - py_next + qy_next;
            ad_scalar_t constraint_final2 = (l_0-r_next)*sin(theta_next) - qx_next + px_next;
            y.segment((i+1) * num_dynamic_constraints_per_step, 2) << constraint_final1,
                                                                    constraint_final2;
        
        }
    }
    std::cout << "dynamic constraints" << std::endl;
};

// Define contact constraints for 2D hopper
ad_function_t HopperContactConstraints = [](const ad_vector_t& x, ad_vector_t& y){
    y.resize(N * 9);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);

        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t qx_i = x[idx + 2];
        ad_scalar_t qy_i = x[idx + 3];
        ad_scalar_t theta_i = x[idx + 4];
        ad_scalar_t r_i = x[idx + 5];
        ad_scalar_t px_dot_i = x[idx + 6];
        ad_scalar_t py_dot_i = x[idx + 7];
        ad_scalar_t u1_i = x[idx + 8];
        ad_scalar_t u2_i = x[idx + 9];
        ad_scalar_t lambda_i = x[idx + 10];

        y.segment(i * 9, 9) << r_i,
                            qy_i,
                            u2_i,
                            -r_i * qy_i,
                            -u2_i * qy_i,
                            -(u1_i)*(u1_i) * r_i,
                            r_0 - r_i,
                            (qy_i + r_i - 0.1) + lambda_i,
                            lambda_i;
    }
    std::cout << "contact constraints" << std::endl;
};

// // Define lambda equality constraints for 2D hopper
// ad_function_t HopperLambdaConstraints = [](const ad_vector_t& x, ad_vector_t& y){
//     y.resize(N);
//     for (size_t i = 0; i < N; ++i) {
//         size_t idx = i * (num_state + num_control);
//         ad_scalar_t py_i = x[idx + 1];
//         ad_scalar_t qy_i = x[idx + 3];
//         ad_scalar_t theta_i = x[idx + 4];
//         ad_scalar_t r_i = x[idx + 5];
//         ad_scalar_t lambda_i = x[idx + 10];
//         y[i] = lambda_i * r_i + (0.01-lambda_i) * (py_i - (l_0 - r_i) * cos(theta_i));
//     }
//         std::cout << "lambda constraints" << std::endl;
// };


// Define initial constraints for 2D hopper
ad_function_with_param_t HopperInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y){
    y.resize(num_state);
    y.segment(0, num_state) << x[0] - p[0],
                    x[1] - p[1],
                    x[2] - p[2],
                    x[3] - p[3],
                    x[4] - p[4],
                    x[5] - p[5],
                    x[6] - p[6],
                    x[7] - p[7];
    std::cout << "initial constraints" << std::endl;
};

// ad_function_with_param_t HopperObjective02 = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
//     y.resize(1);
//     y[0] = 0.0;

//     ad_scalar_t tracking_cost(0.0);
//     ad_scalar_t control_cost(0.0);

//     Eigen::VectorXd ref_px = Eigen::VectorXd::Zero(N);
//     Eigen::VectorXd ref_py(N);
//     ref_py.setConstant(1.0);
//     ref_py.head(20).setLinSpaced(20, 1.5, 0.2);
//     ref_py.segment(20, 40).setLinSpaced(40, 0.2, 2.5);
//     ref_py.segment(60, 30).setLinSpaced(30, 2.5, 1.0); 
    
//     Eigen::VectorXd ref_theta(N);
//     ref_theta.setZero();
//     // ref_theta.head(40).setZero();
//     ref_theta.segment(50, 30).setLinSpaced(30, 0.0, 2 * M_PI);

//     // 成本权重
//     ad_matrix_t Q(num_state, num_state);
//     Q.setZero();
//     Q(0, 0) = 1;
//     Q(1, 1) = 1;
//     Q(4, 4) = 1;

//     ad_matrix_t R(num_control, num_control);
//     R.setZero();
//     R(0, 0) = 0.001;
//     R(1, 1) = 0.001;

//     // 遍历每个时间步长
//     for (size_t i = 0; i < N; ++i) {
//         size_t idx = i * (num_state + num_control);
//         ad_scalar_t px_i = x[idx + 0];
//         ad_scalar_t py_i = x[idx + 1];
//         ad_scalar_t theta_i = x[idx + 4];
//         ad_scalar_t u1_i = x[idx + 8];
//         ad_scalar_t u2_i = x[idx + 9];
//         ad_vector_t tracking_error(num_state);
//         tracking_error.setZero();
//         tracking_error[0] = px_i - ref_px[i];
//         tracking_error[1] = py_i - ref_py[i];
//         tracking_error[4] = theta_i - ref_theta[i];

//         tracking_cost += tracking_error.transpose() * Q * tracking_error;

//         ad_vector_t control_error(num_control);
//         control_error << u1_i, u2_i;
//         control_cost += control_error.transpose() * R * control_error;
//     }

//     // 计算总成本
//     y[0] = tracking_cost + control_cost;

// };



// Define objective function for 2D hopper
ad_function_with_param_t HopperObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y){
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t qx_i = x[idx + 2];
        ad_scalar_t qy_i = x[idx + 3];
        ad_scalar_t theta_i = x[idx + 4];
        ad_scalar_t r_i = x[idx + 5];
        ad_scalar_t px_dot_i = x[idx + 6];
        ad_scalar_t py_dot_i = x[idx + 7];
        ad_scalar_t u1_i = x[idx + 8];
        ad_scalar_t u2_i = x[idx + 9];
        ad_scalar_t lambda_i = x[idx + 10];
        ad_matrix_t Q(num_state, num_state);
        Q.setZero();
        Q(0, 0) = 100;
        Q(1, 1) = 100;
        Q(2, 2) = 100;
        Q(3, 3) = 100;
        Q(4, 4) = 100;
        Q(5, 5) = 100;
        Q(6, 6) = 100;
        Q(7, 7) = 100;

        ad_matrix_t R(num_control, num_control);
        R.setZero();
        R(0, 0) = 0.001;
        R(1, 1) = 0.00001;
        R(2, 2) = 100;

        if (i == N - 1) {
            ad_vector_t tracking_error(num_state);
            tracking_error << px_i - p[0],
                            py_i - p[1],
                            qx_i - p[2],
                            qy_i - p[3],
                            theta_i - p[4],
                            r_i - p[5],
                            px_dot_i - p[6],
                            py_dot_i - p[7];
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }

        ad_vector_t control_error(num_control);
        control_error << u1_i,
                        u2_i,
                        lambda_i;
        control_cost += control_error.transpose() * R * control_error;
    }
    y[0] = tracking_cost + control_cost;
    std::cout << "objective function" << std::endl;
};


int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "HopperProblem";
    std::string folderName = "model";
    OptimizationProblem HopperProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "HopperObjective", HopperObjective);
    // auto obj02 = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "HopperObjective02", HopperObjective02);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "HopperDynamicConstraints", HopperDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName,  "HopperContactConstraints", HopperContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName, "HopperInitialConstraints", HopperInitialConstraints);
    // auto lambda = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "HopperLambdaConstraints", HopperLambdaConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    HopperProblem.addObjective(obj);
    // HopperProblem.addObjective(obj02);
    HopperProblem.addEqualityConstraint(dynamics);
    HopperProblem.addEqualityConstraint(initial);
    // HopperProblem.addEqualityConstraint(lambda);
    HopperProblem.addInequalityConstraint(contact);
    
    // problem parameters
    vector_t xInitialStates(num_state);
    vector_t xFinalStates(num_state);
    vector_t xInitialGuess(variableNum);
    // define the initial states
    // xInitialStates << 0.0, l_0 + 0.5, 0.0, 0.5, 0, 0, 0, 0;
    xInitialStates << 0.0, l_0 + 0.5, 0.0, 0.5, 0, 0, 0, 0;
    // set random initial guess beteween -0.01 and 0.01
    xInitialGuess.setZero();
    // SizeVector lambda_idx_list;
    // for (size_t i = 0; i < N; ++i) {
    //     size_t lambda_idx = i * (num_state + num_control) + 10;
    //     lambda_idx_list.push_back(lambda_idx); 
    // }
    // set initial guess of lambda to 0.01,0,0.01,0,0.01 each 20 steps
    // for (size_t j = 0; j < lambda_idx_list.size(); ++j) {
    //     scalar_t value_lambda = 0.0;
    //     if (j % 20 == 0) {
    //         if (value_lambda == 0.0) {
    //             value_lambda = 0.01;
    //         }
    //         else {
    //             value_lambda = 0.0;
    //         }
    //     }
    //     xInitialGuess[lambda_idx_list[j]] = value_lambda;
    // }
    // xInitialGuess.segment(0, 8) = xInitialStates;
    xFinalStates << 0, l_0 + 1.5, 0, 1.5, 0, 0, 0, 0;
    xFinalStates << 3, l_0, 3, 0, 0, 0, 0, 0;
    
    // Eigen::VectorXd ref_px = Eigen::VectorXd::Zero(N);
    // Eigen::VectorXd ref_py(N);
    // ref_py.setConstant(1.0);
    // ref_py.head(20).setLinSpaced(20, 1.5, 0.2);
    // ref_py.segment(20, 40).setLinSpaced(40, 0.2, 2.5);
    // ref_py.segment(60, 30).setLinSpaced(30, 2.5, 1.0); 
    // vector_t ref_px(N);
    // ref_px.setZero();
    // vector_t ref_py(N);
    // ref_py.setOnes();
    // // interpolate the parobolic reference trajectory between initial and final states
    // ref_px.segment(20,60).setLinSpaced(60, 0.0, 0.5);
    // ref_px.segment(80,20).setConstant(0.5);
    // ref_py.segment(20,30).setLinSpaced(30, 1.0, 2.5);
    // ref_py.segment(50,50).setLinSpaced(50, 2.5, 1.0);

    // for (size_t i = 0; i < N; ++i) {
    //     size_t idx = i * (num_state + num_control);
    //     xInitialGuess[idx + 0] = ref_px[i];
    //     xInitialGuess[idx + 1] = ref_py[i];
    // }

    
    // Eigen::VectorXd ref_theta(N);
    // ref_theta.setZero();
    // // ref_theta.head(40).setZero();
    // ref_theta.segment(50, 30).setLinSpaced(30, 0.0, 2 * M_PI);
    // for (size_t i = 0; i < N; ++i) {
    //     size_t idx = i * (num_state + num_control);
    //     xInitialGuess[idx + 0] = ref_px[i];
    //     xInitialGuess[idx + 1] = ref_py[i];
    //     xInitialGuess[idx + 4] = ref_theta[i];
    // }


    const char* variableName = "z_record";
    std::string matFileName = "/home/workspace/src/examples/2dHopper/ylhopper_4.mat";
    MatlabHelper::readVariableFromMatFile(matFileName.c_str(), variableName, xInitialGuess);   
    std::cout << "Initial guess: " << xInitialGuess.transpose() << std::endl;
    SolverParameters params;
    SolverInterface solver(HopperProblem, params);
    // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("mu", vector_t::Constant(1,10));
    solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
    solver.setProblemParameters("HopperInitialConstraints", xInitialStates);
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("muMax", vector_t::Constant(1, 1e8));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 5e-3));
    solver.setHyperParameters("maxIterations", vector_t::Constant(1, 20000));
    std::string resultFolderPrefix = "/home/workspace/src/examples/2dHopper/experiments/results_exp";
    solver.setProblemParameters("HopperObjective", xFinalStates);
    solver.initialize(xInitialGuess);
    solver.solve();
    solver.getSolution();
    solver.saveResults(resultFolderPrefix);
    std::cout << "Finish solving the problem with final state: " << xFinalStates.transpose() << std::endl;
    }
