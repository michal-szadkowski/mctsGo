add_library(board STATIC board.cpp board.h)
add_library(comms STATIC comms.cpp comms.h)

target_include_directories(board PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(comms PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")