add_executable(pushbot_example pushbot/cpp/SolvePushbot.cpp)

target_link_libraries(pushbot_example
  CRISP
)

add_executable(pushbox_example pushbox/SolvePushbox.cpp)

target_link_libraries(pushbox_example
  CRISP
)

add_executable(hopper_example 2dHopper/cpp/Solve2DHopper.cpp)

target_link_libraries(hopper_example
  CRISP
)

add_executable(cartTransp_example Transp/cpp/SolveTransp.cpp)

target_link_libraries(cartTransp_example
  CRISP
)

add_executable(waiter_example waiter/cpp/SolveWaiter.cpp)

target_link_libraries(waiter_example
  CRISP
)