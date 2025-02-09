#ifndef VALUE_FUNCTION_H
#define VALUE_FUNCTION_H

#include "cppad_core/CppAdInterface.h"

namespace ContactSolver {


class ValueFunction {
public:
    virtual ~ValueFunction() = default;
 
    // ------------------------ Get function information ------------------------ //
    // nonparametric
    virtual vector_t getValue(const vector_t& x) = 0;
    virtual sparse_matrix_t getGradient(const vector_t& x) = 0;

    // parametric
    virtual vector_t getValue(const vector_t& x, const vector_t& p) = 0;
    virtual sparse_matrix_t getGradient(const vector_t& x, const vector_t& p) = 0;


    // ------------------------ Set user specified function information ------------------------ //
    void setValueFunction(const std::function<vector_t(const vector_t&)>& valueFunction) {
        valueFunction_ = valueFunction;
    }

    void setGradientFunction(const std::function<sparse_matrix_t(const vector_t&)>& gradientFunction) {
        gradientFunction_ = gradientFunction;
    }

    void setValueFunctionWithParam(const std::function<vector_t(const vector_t&, const vector_t&)>& valueFunctionWithParam) {
        valueFunctionWithParam_ = valueFunctionWithParam;
    }

    void setGradientFunctionWithParam(const std::function<sparse_matrix_t(const vector_t&, const vector_t&)>& gradientFunctionWithParam) {
        gradientFunctionWithParam_ = gradientFunctionWithParam;
    }


protected:
    std::unique_ptr<CppAdInterface> cppadInterface_;
    // User specified function information.
    std::function<vector_t(const vector_t&)> valueFunction_;
    std::function<sparse_matrix_t(const vector_t&)> gradientFunction_;
    std::function<vector_t(const vector_t&, const vector_t&)> valueFunctionWithParam_;
    std::function<sparse_matrix_t(const vector_t&, const vector_t&)> gradientFunctionWithParam_;
};
} // namespace ContactSolver

#endif // VALUE_FUNCTION_H