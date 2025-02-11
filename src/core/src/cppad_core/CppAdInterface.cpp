#include "cppad_core/CppAdInterface.h"
#include <boost/filesystem.hpp>
#include <iostream>

namespace CRISP {
CppAdInterface::CppAdInterface(size_t variableDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                               const ad_function_t& function, ModelInfoLevel infoLevel, bool regenerateLibrary)
    : variableDim_(variableDim), parameterDim_(0), modelName_(modelName), folderName_(folderName), functionName_(functionName),
      functionNoParam_(function), isParameterized_(false), infoLevel_(infoLevel), regenerateLibrary_(regenerateLibrary) {
    libraryFolder_ = folderName_ + '/' + modelName_ + '/' + "auto_generated";
    libraryName_ = libraryFolder_ + '/' + modelName_ + "_" + functionName_ + "_cppad_cg_model"; 
    if (isLibraryAvailable() && !regenerateLibrary_) {
        loadModel();
    } else {
        initializeModel();
    }
}

CppAdInterface::CppAdInterface(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                               const ad_function_with_param_t& function, ModelInfoLevel infoLevel, bool regenerateLibrary)
    : variableDim_(variableDim), parameterDim_(parameterDim), modelName_(modelName), folderName_(folderName), functionName_(functionName),
      functionWithParam_(function), isParameterized_(true), infoLevel_(infoLevel), regenerateLibrary_(regenerateLibrary) {
    libraryFolder_ = folderName_ + '/' + modelName_ + '/' + "auto_generated";
    libraryName_ = libraryFolder_ + '/' + modelName_ + "_" + functionName_ + "_cppad_cg_model"; 
    if (isLibraryAvailable() && !regenerateLibrary_) {
        loadModel();
    } else {
        initializeModel();
    }
}

CppAdInterface::CppAdInterface(size_t variableDim, size_t parameterDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   ModelInfoLevel infoLevel, bool regenerateLibrary): variableDim_(variableDim), parameterDim_(parameterDim), modelName_(modelName), folderName_(folderName), functionName_(functionName),
                     isParameterized_(true), infoLevel_(infoLevel), regenerateLibrary_(regenerateLibrary) {
    libraryFolder_ = folderName_ + '/' + modelName_ + '/' + "auto_generated";
    libraryName_ = libraryFolder_ + '/' + modelName_ + "_" + functionName_ + "_cppad_cg_model";
    if (isLibraryAvailable()) {
        loadModel();
    } else {
        std::cerr << "Model " << modelName_ << '_' << functionName_ << " is not available, " << "check your function name or you should generate the auto differential library with cppad first." << std::endl;
    }
}

CppAdInterface::CppAdInterface(size_t variableDim, const std::string& modelName, const std::string& folderName, const std::string& functionName,
                   ModelInfoLevel infoLevel, bool regenerateLibrary): variableDim_(variableDim), parameterDim_(0), modelName_(modelName), folderName_(folderName), functionName_(functionName),
                     isParameterized_(false), infoLevel_(infoLevel), regenerateLibrary_(regenerateLibrary) {
    libraryFolder_ = folderName_ + '/' + modelName_ + '/' + "auto_generated";
    libraryName_ = libraryFolder_ + '/' + modelName_ + "_" + functionName_ + "_cppad_cg_model";
    if (isLibraryAvailable()) {
        loadModel();
    } else {
        std::cerr << "Model " << modelName_ << '_' << functionName_ << " is not available, " << "check your function name or you should generate the auto differential library with cppad first." << std::endl;
    }
}

bool CppAdInterface::isLibraryAvailable() const {
    std::string file_name_ext = libraryName_ + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    return boost::filesystem::exists(file_name_ext);
}

void CppAdInterface::loadModel() {
    // Load the dynamic library
    std::string file_name_ext = libraryName_ + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    dynamicLib_ = std::make_unique<CppAD::cg::LinuxDynamicLib<scalar_t>>(file_name_ext);
    model_ = dynamicLib_->model(modelName_);
    funDim_ = model_->Range();
    // get non-zero numbers of the jacobian and hessian
    if (model_->isJacobianSparsityAvailable()) {
        auto jacobianSparsityPattern = model_->JacobianSparsitySet();
        nnzJacobian_ = 0;
        if (isParameterized_){
            for (size_t i = 0; i < jacobianSparsityPattern.size(); ++i) {
                for (const auto& col : jacobianSparsityPattern[i]) {
                    if (col < variableDim_) {
                        ++nnzJacobian_;
                    }
                }
            }
        }
        else {
            for (const auto& row : jacobianSparsityPattern) {
                nnzJacobian_ += row.size();
            }
        }
    }
    if (model_->isHessianSparsityAvailable()) {
        auto hessianSparsityPattern = model_->HessianSparsitySet();
        nnzHessian_ = 0;
        if (isParameterized_){
            for (size_t i = 0; i < hessianSparsityPattern.size(); ++i) {
                for (const auto& col : hessianSparsityPattern[i]) {
                    if (col < variableDim_ && i < variableDim_) {
                        ++nnzHessian_;
                    }
                }
            }
        }
        else {
            for (const auto& row : hessianSparsityPattern) {
                nnzHessian_ += row.size();
            }
        }
    }

    // print if the model named modelName_ is loaded
    std::cout<<"Model "<<modelName_<<'_'<< functionName_ <<" is loaded."<<std::endl;
}

void CppAdInterface::initializeModel() {
    if (isParameterized_) {
        ad_vector_t ax(variableDim_);
        ad_vector_t ap(parameterDim_);
        ax.setOnes();
        ap.setOnes();

        // Concatenate variables and parameters into a single vector
        ad_vector_t axp(variableDim_ + parameterDim_);
        axp.segment(0, variableDim_) = ax;
        axp.segment(variableDim_, parameterDim_) = ap;

        // Declare axp as independent variables
        CppAD::Independent(axp);

        // dependent variable vector
        ad_vector_t ay;

        // Call the function with concatenated inputs
        functionWithParam_(axp.segment(0, variableDim_), axp.segment(variableDim_, parameterDim_), ay);
        funDim_ = ay.size();

        CppAD::ADFun<cg_scalar_t> cg_fun(axp, ay);
        cg_fun.optimize();

        CppAD::cg::ModelCSourceGen<scalar_t> cgen(cg_fun, modelName_);
        
        // Set options based on infoLevel_
        switch (infoLevel_) {
            case ModelInfoLevel::FIRST_ORDER:
                cgen.setCreateSparseJacobian(true);
                {
                    CppAD::sparse_rc<SizeVector> eye_sparsity(variableDim_ + parameterDim_, variableDim_ + parameterDim_, variableDim_ + parameterDim_);
                    for (size_t k = 0; k < variableDim_ + parameterDim_; ++k)
                        eye_sparsity.set(k, k, k);
                    bool transpose = false;
                    bool dependency = false;
                    bool internal_bool = false;
                    // Compute the sparsity pattern of the Jacobian
                    cg_fun.for_jac_sparsity(eye_sparsity, transpose, dependency, internal_bool, jacobianSparsity_);
                    nnzJacobian_ = 0;
                    // count non-zeros in the jacobian without the parameter part
                    for (size_t i = 0; i < jacobianSparsity_.nnz(); ++i) {
                        if (jacobianSparsity_.col()[i] < variableDim_) {
                            ++nnzJacobian_;
                        }
                    }
                }
                break;
            case ModelInfoLevel::SECOND_ORDER:
                cgen.setCreateSparseJacobian(true);
                cgen.setCreateSparseHessian(true);
                {
                    // Initialize pattern_in with an identity pattern
                    CppAD::sparse_rc<SizeVector> eye_sparsity(variableDim_ + parameterDim_, variableDim_ + parameterDim_, variableDim_ + parameterDim_);
                    for (size_t k = 0; k < variableDim_ + parameterDim_; ++k)
                        eye_sparsity.set(k, k, k);
                    bool transpose = false;
                    bool dependency = false;
                    bool internal_bool = false;
                    // Compute the sparsity pattern of the Jacobian
                    cg_fun.for_jac_sparsity(eye_sparsity, transpose, dependency, internal_bool, jacobianSparsity_);
                    nnzJacobian_ = 0;
                    // count non-zeros in the jacobian without the parameter part
                    for (size_t i = 0; i < jacobianSparsity_.nnz(); ++i) {
                        if (jacobianSparsity_.col()[i] < variableDim_) {
                            ++nnzJacobian_;
                        }
                    }
                    // Compute the sparsity pattern of the Hessian for the first component
                    std::vector<bool> select_range(ay.size(), false);
                    select_range[0] = true;  // Only the first element for (F_1)
                    cg_fun.rev_hes_sparsity(select_range, transpose, internal_bool, hessianSparsity_);
                    nnzHessian_ = 0;
                    for (size_t i = 0; i < hessianSparsity_.nnz(); ++i) {
                        if (hessianSparsity_.col()[i] < variableDim_ && hessianSparsity_.row()[i] < variableDim_) {
                            ++nnzHessian_;
                        }
                    }
                }
                break;
            case ModelInfoLevel::ZERO_ORDER:
            default:
                // Do nothing
                break;
        }

        CppAD::cg::ModelLibraryCSourceGen<scalar_t> libcgen(cgen);
        CppAD::cg::GccCompiler<scalar_t> compiler;
        CppAD::cg::DynamicModelLibraryProcessor<scalar_t> proc(libcgen, libraryName_);

        if (!boost::filesystem::exists(libraryFolder_)) {
            boost::filesystem::create_directories(libraryFolder_);
        }

        dynamicLib_ = proc.createDynamicLibrary(compiler);
        model_ = dynamicLib_->model(modelName_);
    } else {
        ad_vector_t ax(variableDim_);
        // Initialize these vectors with ones to avoid division by zero in the function
        ax.setOnes();

        CppAD::Independent(ax);
        ad_vector_t ay;
        functionNoParam_(ax, ay);
        funDim_ = ay.size();
        CppAD::ADFun<cg_scalar_t> cg_fun(ax, ay);

        CppAD::cg::ModelCSourceGen<scalar_t> cgen(cg_fun, modelName_);
        
        // Set options based on infoLevel_
        switch (infoLevel_) {
            case ModelInfoLevel::FIRST_ORDER:
                cgen.setCreateSparseJacobian(true);
                {
                    CppAD::sparse_rc<SizeVector> eye_sparsity(variableDim_, variableDim_, variableDim_);
                    for (size_t k = 0; k < variableDim_; ++k)
                        eye_sparsity.set(k, k, k);
                    bool transpose = false;
                    bool dependency = false;
                    bool internal_bool = false;
                    // Compute the sparsity pattern of the Jacobian
                    cg_fun.for_jac_sparsity(eye_sparsity, transpose, dependency, internal_bool, jacobianSparsity_);
                    nnzJacobian_ = jacobianSparsity_.nnz();
                }
                break;
            case ModelInfoLevel::SECOND_ORDER:
                cgen.setCreateSparseJacobian(true);
                cgen.setCreateSparseHessian(true);
                {
                    // Initialize pattern_in with an identity pattern
                    CppAD::sparse_rc<SizeVector> eye_sparsity(variableDim_, variableDim_, variableDim_);
                    for (size_t k = 0; k < variableDim_; ++k)
                        eye_sparsity.set(k, k, k);
                    bool transpose = false;
                    bool dependency = false;
                    bool internal_bool = false;
                    // Compute the sparsity pattern of the Jacobian
                    cg_fun.for_jac_sparsity(eye_sparsity, transpose, dependency, internal_bool, jacobianSparsity_);
                    nnzJacobian_ = jacobianSparsity_.nnz();
                    // Compute the sparsity pattern of the Hessian for the first component
                    std::vector<bool> select_range(ay.size(), false);
                    select_range[0] = true;  // Only the first element for (F_1)
                    cg_fun.rev_hes_sparsity(select_range, transpose, internal_bool, hessianSparsity_);
                    nnzHessian_ = hessianSparsity_.nnz();
                }
                break;
            case ModelInfoLevel::ZERO_ORDER:
            default:
                // Do nothing
                break;
        }

        CppAD::cg::ModelLibraryCSourceGen<scalar_t> libcgen(cgen);
        CppAD::cg::GccCompiler<scalar_t> compiler;
        CppAD::cg::DynamicModelLibraryProcessor<scalar_t> proc(libcgen, libraryName_);

        if (!boost::filesystem::exists(libraryFolder_)) {
            boost::filesystem::create_directories(libraryFolder_);
        }

        dynamicLib_ = proc.createDynamicLibrary(compiler);
        model_ = dynamicLib_->model(modelName_);
    }
}

sparse_matrix_t CppAdInterface::computeSparseJacobian(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(x_std, jac, row, col);
    // from cppad sparse to eigen sparse
    triplet_vector_t tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        tripletList.emplace_back(row[i], col[i], jac[i]);
    }
    sparse_matrix_t sparseJacobian(funDim_, variableDim_);
    sparseJacobian.setFromTriplets(tripletList.begin(), tripletList.end());
    return sparseJacobian;
}

triplet_vector_t CppAdInterface::computeSparseJacobianTriplet(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(x_std, jac, row, col);
    triplet_vector_t tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        tripletList.emplace_back(row[i], col[i], jac[i]);
    }
    return tripletList;
}

CSRSparseMatrix CppAdInterface::computeSparseJacobianCSR(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(x_std, jac, row, col); // automatically in row major format
    CSRSparseMatrix jacobianCSR(funDim_, row.size());
    // copy the data to the eigen csr format for fast processing of sparse matrix
    size_t numNonZeros = 0;
    size_t currentRow = 0;


    for (size_t i = 0; i < row.size(); ++i) {
        numNonZeros++;
        if (row[i] != currentRow) {
            while (currentRow < row[i]) {
                jacobianCSR.outerIndex[currentRow + 1] = numNonZeros - 1;
                currentRow++;
            }
        }
    }
    while (currentRow < funDim_) {
        jacobianCSR.outerIndex[currentRow + 1] = numNonZeros;
        currentRow++;
    }

    // directly copy the std::vector col to jacobianCSR.innerIndices
    std::memcpy(jacobianCSR.innerIndices.data(), col.data(), col.size() * sizeof(size_t));
    // directly copy the std::vector jac to jacobianCSR.values
    std::memcpy(jacobianCSR.values.data(), jac.data(), jac.size() * sizeof(scalar_t));

    return jacobianCSR;
}


sparse_matrix_t CppAdInterface::computeSparseJacobian(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }
    // construct the std vector from the eigen vector
    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    xp_std.insert(xp_std.begin(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());

    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(xp_std, jac, row, col);

    std::vector<Eigen::Triplet<scalar_t>> tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
    if (col[i] < variableDim_) {  // Only include entries with respect to the variables
        tripletList.emplace_back(row[i], col[i], jac[i]);
    }
    }

    sparse_matrix_t sparseJacobian(funDim_, variableDim_);
    sparseJacobian.setFromTriplets(tripletList.begin(), tripletList.end());

    return sparseJacobian;
}

triplet_vector_t CppAdInterface::computeSparseJacobianTriplet(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    xp_std.insert(xp_std.end(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());

    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(xp_std, jac, row, col);

    triplet_vector_t tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        if (col[i] < variableDim_) {  // Only include entries with respect to the variables
            tripletList.emplace_back(row[i], col[i], jac[i]);
        }
    }
    return tripletList;
}

CSRSparseMatrix CppAdInterface::computeSparseJacobianCSR(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    xp_std.insert(xp_std.end(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());

    ValueVector jac;
    SizeVector row, col;
    model_->SparseJacobian(xp_std, jac, row, col); // automatically in row major format
    CSRSparseMatrix jacobianCSR(funDim_, nnzJacobian_); // nnzJacobian_ is already computed in the constructor to preclude the parameter part
    // copy the data to the CSR format
    size_t numNonZeros = 0;
    size_t currentRow = 0;
    for (size_t i = 0; i < row.size(); ++i) {
        if (col[i] < variableDim_) {  // Only include entries with respect to the variables
            numNonZeros++;
            if (row[i] == currentRow) {
                jacobianCSR.innerIndices[numNonZeros - 1] = col[i];
                jacobianCSR.values[numNonZeros - 1] = jac[i];
            } else {
                while (currentRow < row[i]) {
                    jacobianCSR.outerIndex[currentRow + 1] = numNonZeros - 1;
                    currentRow++;
                }
                jacobianCSR.innerIndices[numNonZeros - 1] = col[i];
                jacobianCSR.values[numNonZeros - 1] = jac[i];
            }
        }
    }
    while (currentRow < funDim_) {
        jacobianCSR.outerIndex[currentRow + 1] = numNonZeros;
        currentRow++;
    }

    return jacobianCSR;
}



sparse_matrix_t CppAdInterface::computeSparseHessian(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector w(model_->Range(), 1.0); // We only need the objective hessian, so range would be 1.
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(x_std, w, hes, row, col);
    std::vector<Eigen::Triplet<scalar_t>> tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        tripletList.emplace_back(row[i], col[i], hes[i]);
    }
    sparse_matrix_t sparseHessian(variableDim_, variableDim_);
    sparseHessian.setFromTriplets(tripletList.begin(), tripletList.end());
    return sparseHessian;
}

triplet_vector_t CppAdInterface::computeSparseHessianTriplet(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector w(model_->Range(), 1.0); // We only need the objective hessian, so range would be 1.
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(x_std, w, hes, row, col);
    triplet_vector_t tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        tripletList.emplace_back(row[i], col[i], hes[i]);
    }
    return tripletList;
}

CSRSparseMatrix CppAdInterface::computeSparseHessianCSR(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    } else {
        // initialize the eigen vector, but check size first
        if (x.size() != variableDim_) {
            throw std::runtime_error("Input vector size does not match the variable dimension.");
        }
    }
    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector w(model_->Range(), 1.0); // We only need the objective hessian, so range would be 1.
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(x_std, w, hes, row, col);
    CSRSparseMatrix hessianCSR(variableDim_, row.size());
    // copy the data to the CSR format
    size_t numNonZeros = 0;
    size_t currentRow = 0;
    for (size_t i = 0; i < row.size(); ++i) {
        numNonZeros++;
        if (row[i] == currentRow) {
            // hessianCSR.innerIndices[i] = col[i];
        } else {
            while (currentRow < row[i]) {
                hessianCSR.outerIndex[currentRow + 1] = numNonZeros - 1;
                currentRow++;
            }
            // hessianCSR.innerIndices[i] = col[i];
        }
    }
    while (currentRow < variableDim_) {
        hessianCSR.outerIndex[currentRow + 1] = numNonZeros;
        currentRow++;
    }
    std::memcpy(hessianCSR.innerIndices.data(), col.data(), col.size() * sizeof(size_t));
    std::memcpy(hessianCSR.values.data(), hes.data(), hes.size() * sizeof(scalar_t));

    return hessianCSR;
}



sparse_matrix_t CppAdInterface::computeSparseHessian(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    // Convert Eigen::VectorXd to std::vector
    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    ValueVector w(model_->Range(), 1.0); // Assuming w is a vector of ones with size equal to the range of the model
    xp_std.insert(xp_std.begin(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(xp_std, w, hes, row, col);


    // Filter out the Hessian entries that correspond to the parameters
    std::vector<Eigen::Triplet<scalar_t>> tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i] < variableDim_ && col[i] < variableDim_) {  // Only include entries with respect to the variables
            tripletList.emplace_back(row[i], col[i], hes[i]);
        }
    }
    sparse_matrix_t sparseHessian(variableDim_, variableDim_);
    sparseHessian.setFromTriplets(tripletList.begin(), tripletList.end());
    return sparseHessian;
}

triplet_vector_t CppAdInterface::computeSparseHessianTriplet(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    // Convert Eigen::VectorXd to std::vector
    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    ValueVector w(model_->Range(), 1.0); // Assuming w is a vector of ones with size equal to the range of the model
    xp_std.insert(xp_std.end(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(xp_std, w, hes, row, col);

    // Filter out the Hessian entries that correspond to the parameters
    triplet_vector_t tripletList;
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i] < variableDim_ && col[i] < variableDim_) {  // Only include entries with respect to the variables
            tripletList.emplace_back(row[i], col[i], hes[i]);
        }
    }
    return tripletList;
}

CSRSparseMatrix CppAdInterface::computeSparseHessianCSR(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    // Convert Eigen::VectorXd to std::vector
    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    ValueVector w(model_->Range(), 1.0); // Assuming w is a vector of ones with size equal to the range of the model
    xp_std.insert(xp_std.end(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());
    ValueVector hes;
    SizeVector row, col;
    model_->SparseHessian(xp_std, w, hes, row, col); // automatically in row major format.
    // convert the data to eigen CSR format for fast sparse matrixprocessing
    CSRSparseMatrix hessianCSR(variableDim_, nnzHessian_);
    // copy the data to the CSR format
    size_t numNonZeros = 0;
    size_t currentRow = 0;
    for (size_t i = 0; i < row.size(); ++i) {
        if (col[i] < variableDim_ && row[i] < variableDim_) {  // Only include entries with respect to the variables
            numNonZeros++;
            if (row[i] == currentRow) {
                hessianCSR.innerIndices[numNonZeros - 1] = col[i];
                hessianCSR.values[numNonZeros - 1] = hes[i];
            } else {
                while (currentRow < row[i]) {
                    hessianCSR.outerIndex[currentRow + 1] = numNonZeros - 1;
                    currentRow++;
                }
                hessianCSR.innerIndices[numNonZeros - 1] = col[i];
                hessianCSR.values[numNonZeros - 1] = hes[i];
            }
        }
    }
    while (currentRow < variableDim_) {
        hessianCSR.outerIndex[currentRow + 1] = numNonZeros;
        currentRow++;
    }
    return hessianCSR;
}

vector_t CppAdInterface::computeFunctionValue(const vector_t& x) {
    if (isParameterized_) {
        throw std::runtime_error("Parameter vector required.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    ValueVector x_std(x.data(), x.data() + x.size());
    ValueVector y(model_->Range());
    model_->ForwardZero(x_std, y);

    vector_t y_eigen(y.size());
    std::memcpy(y_eigen.data(), y.data(), y.size() * sizeof(scalar_t));
    return y_eigen;
}

vector_t CppAdInterface::computeFunctionValue(const vector_t& x, const vector_t& p) {
    if (!isParameterized_) {
        throw std::runtime_error("This model is not parameterized.");
    }

    if (x.size() != variableDim_) {
        throw std::runtime_error("Input vector size does not match the variable dimension.");
    }

    if (p.size() != parameterDim_) {
        throw std::runtime_error("Parameter vector size does not match the parameter dimension.");
    }

    ValueVector xp_std;
    xp_std.reserve(variableDim_ + parameterDim_);
    xp_std.insert(xp_std.end(), x.data(), x.data() + x.size());
    xp_std.insert(xp_std.end(), p.data(), p.data() + p.size());
    // print the value of xp_std vector using iterator
    // std::cout<<"value of xp_std: ";
    // for(auto it = xp_std.begin(); it != xp_std.end(); ++it){
    //     std::cout<<*it<<" ";
    // } 


    ValueVector y(model_->Range());
    model_->ForwardZero(xp_std, y);
    // print the value of y
    // std::cout<<"value of y: "<<y[0]<<std::endl;

    vector_t y_eigen(y.size());
    std::memcpy(y_eigen.data(), y.data(), y.size() * sizeof(scalar_t));
    return y_eigen;
}


void CppAdInterface::printSparsityPatterns() const {
    if (infoLevel_ == ModelInfoLevel::ZERO_ORDER) {
        std::cout << "Model is zero order." << std::endl;
        return;
    }

    std::cout << "Jacobian Sparsity Pattern:" << std::endl;
    for (size_t k = 0; k < jacobianSparsity_.nnz(); ++k) {
        size_t row = jacobianSparsity_.row()[k];
        size_t col = jacobianSparsity_.col()[k];
        std::cout << "(" << row << ", " << col << ")" << std::endl;
    }

    if (infoLevel_ == ModelInfoLevel::SECOND_ORDER) {
        std::cout << "Hessian Sparsity Pattern:" << std::endl;
        for (size_t k = 0; k < hessianSparsity_.nnz(); ++k) {
            size_t row = hessianSparsity_.row()[k];
            size_t col = hessianSparsity_.col()[k];
            std::cout << "(" << row << ", " << col << ")" << std::endl;
        }
    }
}

void CppAdInterface::printSparsityMatrix(const sparse_matrix_t& matrix) const {
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (sparse_matrix_t::InnerIterator it(matrix, k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ", " << it.value() << ")" << std::endl;
        }
    }
}

void CppAdInterface::printSparsityMatrixFromTriplets(const triplet_vector_t& triplets) const {
    for (size_t i = 0; i < triplets.size(); ++i) {
        std::cout << "(" << triplets[i].row() << ", " << triplets[i].col() << ", " << triplets[i].value() << ")" << std::endl;
    }


}
} // namespace CRISP