add_executable(GoPlayer GoPlayer.cpp)

target_link_libraries(GoPlayer PRIVATE board)
target_link_libraries(GoPlayer PRIVATE comms)