# add the solver as a library, the solver only has one cpp file and others are header files only.
add_library(ContactSolver
  src/cppad_core/CppAdInterface.cpp
)

target_include_directories(ContactSolver PUBLIC
  include
  ${MATLAB_ROOT}/extern/include                    
  ${EIGEN3_INCLUDE_DIR}     
  ${Boost_INCLUDE_DIRS}
  ${CPPAD_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}      
)


target_link_libraries(ContactSolver
  ${Boost_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${CPPAD_LIBRARIES}
  yaml-cpp
  piqp::piqp # the piqp library (header file library)
  mat   # MAT-file API library
  mx    # MATLAB array library
)

# level 1 test, test the cppad interface: PASSED
add_executable(test_core_cppad tests/test_cppad_interface.cpp)


target_link_libraries(test_core_cppad
  ContactSolver
  -ldl
)

# level 2 test, test the core functions for constructing the optimization problem: PASSED
add_executable(test_core_optimizationProblem tests/test_optimizationProblem.cpp)

target_link_libraries(test_core_optimizationProblem
  ContactSolver
  -ldl
)

# level 3 test, test the core functions for solving the optimization problem: PASSED
add_executable(test_core_solver tests/test_solver.cpp)
target_link_libraries(test_core_solver
  ContactSolver
  -ldl
)

# test pybind
# pybind11_add_module(pybind_test python/testPybind.cpp)
# install(TARGETS pybind_test DESTINATION ${CMAKE_SOURCE_DIR}/python)

pybind11_add_module(pyContactSolver python/pybindSolver.cpp)
# Linking the static library
target_link_libraries(pyContactSolver PRIVATE ContactSolver)
# Ensure position-independent code for the shared module
set_target_properties(ContactSolver pyContactSolver PROPERTIES POSITION_INDEPENDENT_CODE ON)


