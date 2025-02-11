#ifndef SOLVER_PARAMETER_H
#define SOLVER_PARAMETER_H

#include "common/BasicTypes.h"
#include "common/ParametersManager.h"
#include <yaml-cpp/yaml.h>

namespace CRISP {
class SolverParameters : public ParametersManager {
public:
    // load default
    SolverParameters(){
        setDefaultParameters();
    };
    // load from yaml file
    SolverParameters(const std::string& filename) {
        setDefaultParameters();
        loadParametersFromFile(filename);
    }
private:
    void setDefaultParameters() {
        // ------------------parameters for outer iterations ------------------ //
        setParameters("maxIterations", vector_t::Constant(1, 5000)); // maximum number of iterations for the outer loop
        setParameters("trustRegionInitRadius", vector_t::Constant(1, 1.0)); // initial trust region radius
        setParameters("trustRegionMaxRadius", vector_t::Constant(1, 10.0)); // maximum trust region radius
        setParameters("mu", vector_t::Constant(1, 1e1)); // penalty
        setParameters("muMax", vector_t::Constant(1, 1e8)); // maximum penalty
        setParameters("etaLow", vector_t::Constant(1, 0.25)); // low threshold for reduction ratio
        setParameters("etaHigh", vector_t::Constant(1, 0.75)); // high threshold for reduction ratio
        setParameters("trailTol", vector_t::Constant(1, 1e-4)); // tolerance for the outer iterations
        setParameters("trustRegionTol", vector_t::Constant(1, 1e-4)); // tolerance for the trust region
        setParameters("constraintTol", vector_t::Constant(1, 1e-6)); // tolerance for the max constraints violation
        setParameters("verbose", vector_t::Constant(1, 0)); // verbose level
        setParameters("WeightedMode", vector_t::Constant(1, 0)); // 0: no weighted, 1: weighted
        setParameters("WeightedTolFactor", vector_t::Constant(1, 10.0)); // factor for the weighted mode
        setParameters("secondOrderCorrection", vector_t::Constant(1, 1)); // 0: no second order correction, 1: second order correction
        // ------------------parameters for inner iterations ------------------ //
        // to be added for inner convex QP solver.
    }

    bool loadParametersFromFile(const std::string& fileName) {
        try {
            YAML::Node config = YAML::LoadFile(fileName);
            for (const auto& it : config) {
                std::string name = it.first.as<std::string>();
                double value = it.second.as<scalar_t>();
                setParameters(name, vector_t::Constant(1, value));
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to read YAML file " << fileName << ": " << e.what() << std::endl;
            return false;
        }
    }

    void printParameters() {
        auto parameters_ = getParametersMap();
        for (const auto& it : parameters_) {
            std::cout << it.first << ": " << it.second(0) << std::endl;
        }
    }
};
} // namespace CRISP

#endif // SOLVER_PARAMETER_H