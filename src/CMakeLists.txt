cmake_minimum_required (VERSION 2.8)
project (flow)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wall -std=c++11 -msse4")  #-Wall
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wall -msse4")  #-Wall

# For ghc machines
set(OpenCV_DIR "/afs/cs/academic/class/15418-s17/public/sw/opencv/build")
set(Eigen3_DIR "/afs/cs.cmu.edu/academic/class/15418-s17/public/sw/eigen/build")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

# On ghc, include DIR from FIND_PACKAGE is set wrong
set(EIGEN3_INCLUDE_DIR "/afs/cs.cmu.edu/academic/class/15418-s17/public/sw/eigen/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

# Add extra cuda libraries
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Eigen3 library status:")
message(STATUS "    version: ${EIGEN3_VERSION_STRING}")
message(STATUS "    include path: ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    nppi path: ${CUDA_nppi_LIBRARY}")

################################################################################
# CUDA                                                                         #
################################################################################

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -arch=compute_61 -code=sm_61)

set(KERNELS
  kernels/sobel.cpp)

set(COMMON
  common/RgbMat.cpp)

set(CODEFILES
  run_dense.cpp
  oflow.cpp
  patch.cpp
  patchgrid.cpp
  refine_variational.cpp
  FDF1.0.1/image.c
  FDF1.0.1/opticalflow_aux.c
  FDF1.0.1/solver.c)

# GrayScale, Optical Flow
cuda_add_executable(flow ${COMMON} ${CODEFILES} ${KERNELS})
set_target_properties (flow PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET flow APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=3") # use RGB image
target_link_libraries(flow ${OpenCV_LIBS})

# CUDA sandbox
set(SANDBOX_FILES
  kernels/sobel.cpp
  # sandbox/RgbMatTest.cpp
  sandbox/sandbox.cpp)
cuda_add_executable(sandbox ${COMMON} ${SANDBOX_FILES})
target_link_libraries(sandbox ${OpenCV_LIBS})

