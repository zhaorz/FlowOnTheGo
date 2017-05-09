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

#include "cpuDot.h"

using namespace timer;

namespace cu {

  /* Computes <A,B> */
  float cpuDot(
      float* A, float* B, int N) {

    std::cout << "[start] dot" << std::endl;
    auto start_total = now();

    // B = A * B
    auto start_dot = now();
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += A[i] * B[i];
    }
    calc_print_elapsed("cpu sum(A * B)", start_dot);

    calc_print_elapsed("dot total", start_total);

    return sum;

  }

}


