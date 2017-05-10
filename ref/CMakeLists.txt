cmake_minimum_required(VERSION 2.8)

project(flow_ref)

message(STATUS "Architecture: ${ARCH}")

if (ARCH STREQUAL "x86")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2")

elseif (ARCH STREQUAL "ARM")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wno-unknown-pragmas -Wall -std=c++11 -fopenmp")  #-Wall
  # For ghc machines
  set(OpenCV_DIR "/afs/cs/academic/class/15418-s17/public/sw/opencv/build")

else ()

  message(STATUS "specify using -DARCH=x86 or -DARCH=ARM")
  message(FATAL_ERROR "aborting: missing required architecture")

endif ()


find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})

# Samples
add_executable(flow_ref jetson_flow_ref.cpp)
target_link_libraries(flow_ref ${OpenCV_LIBRARIES})
