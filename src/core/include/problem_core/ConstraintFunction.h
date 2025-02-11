#ifndef CONSTRAINT_FUNCTION_H
#define CONSTRAINT_FUNCTION_H

#include "common/ValueFunction.h"

// Only first order information is needed for constraints
namespace CRISP {
    class ConstraintFunction : public ValueFunction {
    public:
        enum class SpecifiedFunctionLevel {
            NONE,
            VALUE,
            GRADIENT
        };
        ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, const ad_function_t& function, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE):
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false){
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, function, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            funDim_ = cppadInterface_->getFunDim();        
        }

        ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, const ad_function_with_param_t& function,  bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE): 
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true){
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName, function, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            parameterDim_ = parameterDim;
            funDim_ = cppadInterface_->getFunDim();
        }

        //  for pybind
        ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE):
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false){

            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            funDim_ = cppadInterface_->getFunDim();
        }

        ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName,
                           const std::string& functionName, bool regenerateLibrary = false,
                           CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER, SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE): 
                           specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true){
            //   convert the eigen vector function to ad function
            cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName, infoLevel, regenerateLibrary);
            nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
            variableDim_ = variableDim;
            parameterDim_ = parameterDim;
            funDim_ = cppadInterface_->getFunDim();
        }


        //  ------------------------ Get function information from ad or user defined functions ------------------------ //
        vector_t getValue(const vector_t& x, const vector_t& params) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
                if (valueFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : valueFunctionWithParam_(x, params);
                } else {
                    throw std::runtime_error("No value function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeFunctionValue(x, params);
        }

        vector_t getValue(const vector_t& x) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
                if (valueFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : valueFunction_(x);
                } else {
                    throw std::runtime_error("No value function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeFunctionValue(x);
        }

        triplet_vector_t getGradientTriplet(const vector_t& x, const vector_t& params) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianTriplet(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianTriplet(x, params);
        }

        triplet_vector_t getGradientTriplet(const vector_t& x) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianTriplet(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianTriplet(x);
        }

        sparse_matrix_t getGradient(const vector_t& x, const vector_t& params) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : gradientFunctionWithParam_(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobian(x, params);
        }

        sparse_matrix_t getGradient(const vector_t& x) override {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : gradientFunction_(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobian(x);
        }

        CSRSparseMatrix getGradientCSR(const vector_t& x, const vector_t& params) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunctionWithParam_ != nullptr) {
                    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianCSR(x, params);
                } else {
                    throw std::runtime_error("No gradient function with parameters specified.");
                }
            }
            return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.") : cppadInterface_->computeSparseJacobianCSR(x, params);
        }

        CSRSparseMatrix getGradientCSR(const vector_t& x) {
            if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
                if (gradientFunction_ != nullptr) {
                    return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianCSR(x);
                } else {
                    throw std::runtime_error("No gradient function specified.");
                }
            }
            return isParameterized_ ? throw std::runtime_error("Parameters are required.") : cppadInterface_->computeSparseJacobianCSR(x);
        }

        SpecifiedFunctionLevel getSpecifiedFunctionLevel() const {
            return specifiedFunctionLevel_;
        }

        bool isParameterized() const {
            return isParameterized_;
        }

        size_t getVariableDim() const {
            return variableDim_;
        }

        size_t getFunDim() const {
            return funDim_;
        }

        size_t getParameterDim() const {
            return parameterDim_;
        }

        size_t getNumNonZerosJacobian() const {
            return nnzJacobian_;
        }


        const std::string& getFunctionName() const {
            return functionName_;
        }


private:
    SpecifiedFunctionLevel specifiedFunctionLevel_;
    size_t variableDim_ = 0;
    size_t parameterDim_ = 0;
    size_t funDim_ = 0;
    size_t nnzJacobian_ = 0;
    std::string functionName_;
    bool isParameterized_ = false;

};

} // namespace CRISP

# endif // CONSTRAINT_FUNCTION_H

