/**
 * Implements an element wise square kernel
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "squareElem.h"

__global__ void kernelSquareElem(float* a, float* b, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N)
    b[tid] = (a[tid]*a[tid]);
}

namespace cu {

  void squareElem(
      float* a, float* b, int N) {

    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernelSquareElem<<<numBlocks, threadsPerBlock>>>(a, b, N);

  }

}


