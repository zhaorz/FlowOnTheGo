// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/cuda_helper.h"
#include "../common/timer.h"

#include "cublasDot.h"

using namespace timer;

namespace cu {

  /* Computes <A,B> */
  float cublasDot(
      float* A, float* B, int N) {

    std::cout << "[start] cublasDot" << std::endl;
    auto start_total = now();

    cublasHandle_t handle;
    CUBLAS_CHECK( cublasCreate(&handle) );

    float* d_A, *d_B;

    checkCudaErrors( cudaMalloc((void**) &d_A, N * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_B, N * sizeof(float)) );

    // transfer to device
    auto start_copy = now();
    CUBLAS_CHECK (
        cublasSetVector(N, sizeof(float),
          A, 1, d_A, 1) );
    CUBLAS_CHECK (
        cublasSetVector(N, sizeof(float),
          B, 1, d_B, 1) );
    calc_print_elapsed("cublas SetVector H->D", start_copy);


    float sum;


    auto start_dot = now();
    CUBLAS_CHECK (
        cublasSdot(handle, N,
          d_A, 1, d_B, 1, &sum) );
    calc_print_elapsed("cublas Dot", start_dot);


    calc_print_elapsed("cublas Dot total", start_total);

    return sum;

  }

}


