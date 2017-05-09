// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#include "../common/timer.h"
#include "../common/cuda_helper.h"

#include "dot.h"

using namespace timer;

__global__ void kernelMultiply(float* d_A, float* d_B, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    d_B[tid] = d_A[tid] * d_B[tid];
}

namespace cu {

  /* Computes <A,B> */
  float dot(
      float* A, float* B, int N) {

    std::cout << "[start] dot" << std::endl;
    auto start_total = now();

    // transfer to device
    auto start_copy = now();
    float *d_A, *d_B;
    checkCudaErrors( cudaMalloc((void**) &d_A, N * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_B, N * sizeof(float)) );
    checkCudaErrors( cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice) );
    calc_print_elapsed("cudaMalloc, cudaMemcpy copy H->D", start_copy);

    // B = A * B
    auto start_mult = now();

    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, N);

    calc_print_elapsed("kernel A * B", start_mult);


    auto start_copyback = now();
    float* tmp = new float[N];
    checkCudaErrors( cudaMemcpy(tmp, d_B, N * sizeof(float), cudaMemcpyDeviceToHost) );
    calc_print_elapsed("cudaMemcpy D->H", start_copyback);

    auto start_reduce = now();
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += tmp[i];
    }
    calc_print_elapsed("CPU reduce(sum)", start_reduce);


    calc_print_elapsed("dot total", start_total);

    return sum;

  }

}


