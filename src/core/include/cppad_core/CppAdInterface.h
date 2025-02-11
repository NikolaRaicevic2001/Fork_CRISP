#ifndef CPP_AD_INTERFACE_H
#define CPP_AD_INTERFACE_H

#include "common/BasicTypes.h"
#include <string>
#include <memory>

namespace CRISP {


class CppAdInterface {
public:
    enum class ModelInfoLevel {
        ZERO_ORDER,
        FIRST_ORDER,
        SECOND_ORDER
    };
    CppAdInterface(size_t variableDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   const ad_function_t& function, ModelInfoLevel infoLevel = ModelInfoLevel::SECOND_ORDER, bool regenerateLibrary = true);

    CppAdInterface(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   const ad_function_with_param_t& function, ModelInfoLevel infoLevel = ModelInfoLevel::SECOND_ORDER, bool regenerateLibrary = true);

    CppAdInterface(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   ModelInfoLevel infoLevel = ModelInfoLevel::SECOND_ORDER, bool regenerateLibrary = true); // for py binding constructor, no need to pass function, read from generated .so file
    
    CppAdInterface(size_t variableDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   ModelInfoLevel infoLevel = ModelInfoLevel::SECOND_ORDER, bool regenerateLibrary = true); // for py binding constructor, no need to pass function, read from generated .so file

    sparse_matrix_t computeSparseJacobian(const vector_t& x);
    sparse_matrix_t computeSparseJacobian(const vector_t& x, const vector_t& p);
    triplet_vector_t computeSparseJacobianTriplet(const vector_t& x);
    triplet_vector_t computeSparseJacobianTriplet(const vector_t& x, const vector_t& p);
    CSRSparseMatrix computeSparseJacobianCSR(const vector_t& x);
    CSRSparseMatrix computeSparseJacobianCSR(const vector_t& x, const vector_t& p);
    sparse_matrix_t computeSparseHessian(const vector_t& x);
    sparse_matrix_t computeSparseHessian(const vector_t& x, const vector_t& p);
    triplet_vector_t computeSparseHessianTriplet(const vector_t& x);
    triplet_vector_t computeSparseHessianTriplet(const vector_t& x, const vector_t& p);
    CSRSparseMatrix computeSparseHessianCSR(const vector_t& x);
    CSRSparseMatrix computeSparseHessianCSR(const vector_t& x, const vector_t& p);

    vector_t computeFunctionValue(const vector_t& x);
    vector_t computeFunctionValue(const vector_t& x, const vector_t& p);
    
    void printSparsityPatterns() const;
    void printSparsityMatrix(const sparse_matrix_t& matrix) const;
    void printSparsityMatrixFromTriplets(const triplet_vector_t& triplets) const;

    size_t getFunDim() const {
        return funDim_;
    }

    bool isParameterized() const {
        return isParameterized_;
    }

    size_t getNumNonZerosJacobian() const {
        return nnzJacobian_;
    }

    size_t getNumNonZerosHessian() const {
        return nnzHessian_;
    }
private:
    bool isParameterized_;
    bool regenerateLibrary_;

    size_t variableDim_;
    size_t parameterDim_;
    size_t funDim_;
    size_t nnzJacobian_;
    size_t nnzHessian_;

    std::string modelName_;
    std::string folderName_;
    std::string functionName_;
    std::string libraryFolder_;
    std::string libraryName_;

    ad_function_t functionNoParam_;
    ad_function_with_param_t functionWithParam_;

    std::unique_ptr<CppAD::cg::DynamicLib<scalar_t>> dynamicLib_;
    std::unique_ptr<CppAD::cg::GenericModel<scalar_t>> model_;
    ModelInfoLevel infoLevel_;

    CppAD::sparse_rc<SizeVector> jacobianSparsity_;   // Jacobian Sparsity Pattern
    CppAD::sparse_rc<SizeVector> hessianSparsity_;   // Hessian Sparsity Pattern

    void initializeModel();
    bool isLibraryAvailable() const;
    void loadModel();
    void generateLibrary();
};
}

#endif // CPP_AD_INTERFACE_H