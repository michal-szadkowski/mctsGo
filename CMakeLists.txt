cmake_minimum_required(VERSION 3.18)
project(MCTSGo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../bin )

add_subdirectory(boardUtils)
add_subdirectory(GoUI)
add_subdirectory(GoPlayer)
add_subdirectory(mctsGoCpu)

