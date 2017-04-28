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

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 --gpu-architecture compute_35)

set(CUFILES
  kernels/hello.cu)

set(CODEFILES
  run_dense.cpp
  oflow.cpp
  patch.cpp
  patchgrid.cpp
  refine_variational.cpp
  FDF1.0.1/image.c
  FDF1.0.1/opticalflow_aux.c
  FDF1.0.1/solver.c)

# add_executable (flow ${CODEFILES})

# GrayScale, Optical Flow
cuda_add_executable(flow ${CODEFILES} ${CUFILES})
target_link_libraries(flow ${OpenCV_LIBS})

# CUDA sandbox
set(SANDBOX_FILES
  sandbox/sandbox.cpp)
cuda_add_executable(sandbox ${SANDBOX_FILES})
target_link_libraries(sandbox ${OpenCV_LIBS})



# # # UNCOMMENT THIS IF YOU WANT TO USE OPENMP PARALLELIZATION
# add_definitions(-DWITH_OPENMP=true)
# FIND_PACKAGE( OpenMP REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# #
# # ENABLE PARALLEL FLOW AGGREGATION, CAN LEAD TO DATA RACES, BUT IN PRACTICE HAS ONLY A WEAK NEGATIVE EFFECT ON THE RESULT, [affects only PatGridClass::AggregateFlowDense() ]
# # add_definitions(-DUSE_PARALLEL_ON_FLOWAGGR)

