cmake_minimum_required (VERSION 2.8)
project (flow)

message(STATUS "Architecture: ${ARCH}")

# On ghc machines
if (ARCH STREQUAL "x86")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wno-unknown-pragmas -Wall -std=c++11 -msse4 -fopenmp")  #-Wall
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3 -Wno-unknown-pragmas -Wall -msse4 -fopenmp")  #-Wall

  # For ghc machines
  set(OpenCV_DIR "/afs/cs/academic/class/15418-s17/public/sw/opencv/build")
  set(Eigen3_DIR "/afs/cs.cmu.edu/academic/class/15418-s17/public/sw/eigen/build")

# On NVIDIA Jetson
elseif (ARCH STREQUAL "ARM")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wno-unknown-pragmas -Wall -std=c++11 -fopenmp")  #-Wall
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3 -Wno-unknown-pragmas -Wall -fopenmp")  #-Wall

else()
  message(STATUS "specify using -DARCH=x86 or -DARCH=ARM")
  message(FATAL_ERROR "aborting: missing required architecture")
endif ()


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

# On ghc, include DIR from FIND_PACKAGE is set wrong
if (ARCH STREQUAL "x86")
  set(EIGEN3_INCLUDE_DIR "/afs/cs.cmu.edu/academic/class/15418-s17/public/sw/eigen/include/eigen3")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# Add extra cuda libraries
set(CUDA_CUSOLVER_LIBRARIES
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusolver.so"
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so.8.0")
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES})

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
message(STATUS "    toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    nppi path: ${CUDA_nppi_LIBRARY}")
message(STATUS "    cuBLAS path: ${CUDA_CUBLAS_LIBRARIES}")
message(STATUS "    cuSOLVER path: ${CUDA_CUSOLVER_LIBRARIES}")

################################################################################
# CUDA                                                                         #
################################################################################

set(CUDA_VERBOSE_BUILD ON)

if (ARCH STREQUAL "x86")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -arch=compute_61 -code=sm_61)
endif()

if (ARCH STREQUAL "ARM")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -arch=compute_62 -code=sm_62)
endif()

set(KERNELS
  kernels/flowUtil.cu
  kernels/densify.cu
  kernels/interpolate.cu
  kernels/extract.cu
  kernels/optimize.cu
  kernels/warmup.cpp
  # kernels/sobel.cpp
  kernels/sobelUnifiedMem.cpp
  kernels/pyramid.cpp
  kernels/pad.cpp
  kernels/resizeGrad.cpp
  kernels/resize.cpp)


set(COMMON
  common/RgbMat.cpp)

set(CODEFILES
  run_dense.cpp
  oflow.cpp
  patchgrid.cpp
  refine_variational.cpp
  FDF1.0.1/image.cpp
  )

# GrayScale, Optical Flow
cuda_add_executable(flow ${COMMON} ${CODEFILES} ${KERNELS})
set_target_properties (flow PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET flow APPEND PROPERTY COMPILE_DEFINITIONS "VECTOR_WIDTH=4") # 8 wide SIMD (4 floats)
set_property(TARGET flow APPEND PROPERTY COMPILE_DEFINITIONS "UNIFIED_MEM=1") # Use zero copy memory
set_property(TARGET flow APPEND PROPERTY COMPILE_DEFINITIONS "VERBOSE=0")
# set_property(TARGET flow APPEND PROPERTY COMPILE_DEFINITIONS "VECTOR_WIDTH=1") # no SIMD
target_link_libraries(flow ${OpenCV_LIBS})

# CUDA sandbox
# set(SANDBOX_FILES
#   # sandbox/process_sobel.cpp
#   sandbox/process_sobel.cpp
#   # sandbox/process_resize.cpp
#   # sandbox/process_resizeGrad.cpp
#   # sandbox/process_pad.cpp
#   # sandbox/RgbMatTest.cpp
#   sandbox/sandbox.cpp)
# cuda_add_executable(sandbox ${COMMON} ${KERNELS} ${SANDBOX_FILES})
# target_link_libraries(sandbox ${OpenCV_LIBS})
