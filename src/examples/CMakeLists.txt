add_executable(pushbot_example pushbot/cpp/SolvePushbot.cpp)

target_link_libraries(pushbot_example
  ContactSolver
  -ldl
)

add_executable(pushboxObs_example pushboxObs/pushboxObs.cpp)

target_link_libraries(pushboxObs_example
  ContactSolver
  -ldl
)

add_executable(planerManip_example planerManip/SolvePlanerManip.cpp)

target_link_libraries(planerManip_example
  ContactSolver
  -ldl
)

add_executable(pushbox_example pushbox/SolvePushbox.cpp)

target_link_libraries(pushbox_example
  ContactSolver
  -ldl
)

add_executable(hopper_example 2dHopper/cpp/Solve2DHopper.cpp)

target_link_libraries(hopper_example
  ContactSolver
  -ldl
)

add_executable(cartTransp_example cartTransp/cpp/SolveCartTransp.cpp)

target_link_libraries(cartTransp_example
  ContactSolver
  -ldl
)

add_executable(waiter_example server/cpp/SolveServer.cpp)

target_link_libraries(waiter_example
  ContactSolver
  -ldl
)