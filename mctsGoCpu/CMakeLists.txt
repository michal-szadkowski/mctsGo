add_executable(CpuGo CpuGo.cpp)

target_link_libraries(CpuGo PRIVATE board)
target_link_libraries(CpuGo PRIVATE comms)
