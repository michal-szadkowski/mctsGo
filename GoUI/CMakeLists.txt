add_executable(GoUI GoUI.cpp)

target_link_libraries(GoUI PRIVATE board)
target_link_libraries(GoUI PRIVATE comms)