# ifndef MATLAB_HELPER_H
# define MATLAB_HELPER_H
#include "mat.h"
#include "common/BasicTypes.h" 

namespace ContactSolver {
class MatlabHelper {
public:
    // read a variable from a .mat file
    static void readVariableFromMatFile(const char* matFileName, const char* variableName, vector_t& xInitialGuess) 
    { 
        size_t variableNum = xInitialGuess.size();
        // std::cout << "variableNum: " << variableNum << std::endl;
        // Open the MAT-file
        MATFile* matFile = matOpen(matFileName, "r");
        if (!matFile) {
            std::cerr << "Error opening MAT-file: " << matFileName << std::endl;
        }

        // Read the desired variable from the file
        mxArray* dataArray = matGetVariable(matFile, variableName);
        if (!dataArray) {   
            std::cerr << "Variable " << variableName << " not found in MAT-file: " << matFileName << std::endl;
            matClose(matFile);
        }

        // Ensure the variable is in the expected shape and size
        if (!mxIsDouble(dataArray) || mxGetNumberOfElements(dataArray) != variableNum) {
            std::cerr << "Variable " << variableName << " is not of the expected size. " << std::endl;
            mxDestroyArray(dataArray);
            matClose(matFile);
        }

        // Map the data from MATLAB array to Eigen vector
        double* matData = mxGetPr(dataArray);
        for (size_t i = 0; i < xInitialGuess.size(); ++i) {
            xInitialGuess[i] = matData[i];
        }

        // Clean up
        mxDestroyArray(dataArray);
        matClose(matFile);

    }
    
    static vector_t readVariableFromMatFilePy(const char* matFileName, const char* variableName) 
    { 
        size_t variableNum = 350;
        vector_t xInitialGuess(variableNum);
        std::cout << "variableNum: " << variableNum << std::endl;
        // Open the MAT-file
        MATFile* matFile = matOpen(matFileName, "r");
        if (!matFile) {
            std::cerr << "Error opening MAT-file: " << matFileName << std::endl;
        }

        // Read the desired variable from the file
        mxArray* dataArray = matGetVariable(matFile, variableName);
        if (!dataArray) {   
            std::cerr << "Variable " << variableName << " not found in MAT-file: " << matFileName << std::endl;
            matClose(matFile);
        }

        // Ensure the variable is in the expected shape and size
        if (!mxIsDouble(dataArray) || mxGetNumberOfElements(dataArray) != variableNum) {
            std::cerr << "Variable " << variableName << " is not of the expected size." << std::endl;
            mxDestroyArray(dataArray);
            matClose(matFile);
        }

        // Map the data from MATLAB array to Eigen vector
        double* matData = mxGetPr(dataArray);
        for (size_t i = 0; i < xInitialGuess.size(); ++i) {
            xInitialGuess[i] = matData[i];
        }
        return xInitialGuess;
        std::cout << "xInitialGuess: " << xInitialGuess << std::endl;
        
        // Clean up
        mxDestroyArray(dataArray);
        matClose(matFile);

    }
};
} // namespace ContactSolver

# endif // MATLAB_HELPER_H