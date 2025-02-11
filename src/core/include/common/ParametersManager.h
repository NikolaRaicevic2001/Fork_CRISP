#ifndef PARAMETERS_MANAGER_H
#define PARAMETERS_MANAGER_H

#include "common/BasicTypes.h"
#include <string>
#include <unordered_map>


namespace CRISP {
class ParametersManager {
public:

    ParametersManager() = default;
    // Set parameters associated with a specific name
    void setParameters(const std::string& name, const vector_t& params) {
        parameters_[name] = params;
    }

    // Retrieve parameters associated with a specific name
    vector_t getParameters(const std::string& name) const {
        auto it = parameters_.find(name);
        if (it != parameters_.end()) {
            return it->second;
        }
        throw std::runtime_error("Parameters not found for: " + name);
    }
    
    std::unordered_map<std::string, vector_t> getParametersMap() const {
        return parameters_;
    }

private:
    std::unordered_map<std::string, vector_t> parameters_;
};
}

#endif // PARAMETERS_MANAGER_H