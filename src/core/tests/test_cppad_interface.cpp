#include "cppad_core/CppAdInterface.h"
#include <iostream>

using namespace CRISP;
void testCppAdInterface() {
    size_t variableDim = 3;
    size_t parameterDim = 2;
    std::string modelName = "testModel";
    std::string folderName = "models"; // This can be a relative path
    std::string functionName = "testFunction";
    // Define a simple function for testing
    ad_function_with_param_t function = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
        y.resize(2);
        y(0) = x(0) * x(1) * x(2) + p(0) * x(0) + p(1) * x(1);
        y(1) = x(0) + x(1) + x(2);
    };
    bool regenerateLibrary = true;
    // Create an instance of CppAdInterface
    CppAdInterface interface(variableDim, parameterDim, modelName, folderName, functionName, function, CppAdInterface::ModelInfoLevel::SECOND_ORDER, regenerateLibrary);
     
    // Test computeSparseJacobian
    vector_t x(variableDim);
    x << 1.0, 2.0, 3.0;
    vector_t p(parameterDim);
    p << 4.0, 5.0;
    // print vector x and p
    std::cout << "x: " << x << std::endl;
    std::cout << "p: " << p << std::endl;

    sparse_matrix_t jacobian = interface.computeSparseJacobian(x, p);
    std::cout << "Jacobian (triplet format):" << std::endl;
    interface.printSparsityMatrix(jacobian);

    // Test computeSparseHessian
    sparse_matrix_t hessian = interface.computeSparseHessian(x, p);
    std::cout << "Hessian (triplet format):" << std::endl;
    interface.printSparsityMatrix(hessian);

    // Test computeFunctionValue
    vector_t y = interface.computeFunctionValue(x, p);
    std::cout << "Function value: " << y[0] << std::endl;
    std::cout << "Function value: " << y[1] << std::endl;


    // Print sparsity patterns
    interface.printSparsityPatterns();
}

int main() {
    testCppAdInterface();
    return 0;
}