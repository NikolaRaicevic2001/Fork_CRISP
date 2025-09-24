# build some examples
add_executable(pushbot_example pushbot/cpp/SolvePushbot.cpp)
target_link_libraries(pushbot_example
  CRISP
)

add_executable(pushbox_example pushbox/cpp/SolvePushbox.cpp)
target_link_libraries(pushbox_example
  CRISP
)
add_executable(pushboxsingleforce_example pushbox/cpp/SolvePushboxSingleForce.cpp)
target_link_libraries(pushboxsingleforce_example
  CRISP
)
add_executable(pushboxsdf_example pushbox/cpp/SolvePushboxSDF.cpp)
target_link_libraries(pushboxsdf_example
  CRISP
)

add_executable(pushcirclesdf_example pushcircle/cpp/SolvePushcircleSDF.cpp)
target_link_libraries(pushcirclesdf_example
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

add_executable(pushT_example pushT/SolvePushT.cpp)
target_link_libraries(pushT_example
  CRISP
)

add_executable(sdf_viewer_example sdf_viewer/cpp/sdf_viewer.cpp)
target_link_libraries(sdf_viewer_example
  CRISP
)