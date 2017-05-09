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

#include "../common/timer.h"

#include "thrustDot.h"

using namespace timer;

namespace cu {

  /* Computes <A,B> */
  float thrustDot(
      float* A, float* B, int N) {

    std::cout << "[start] thrustDot" << std::endl;
    auto start_total = now();

    // transfer to device
    auto start_copy = now();
    thrust::device_vector<float> d_A(A, A + N);
    thrust::device_vector<float> d_B(B, B + N);
    calc_print_elapsed("thrust copy H->D", start_copy);

    // B = A * B
    auto start_mult = now();
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_B.begin(),
        thrust::multiplies<float>());
    calc_print_elapsed("thrust transform A * B", start_mult);

    auto start_reduce = now();
    float sum = thrust::reduce(d_B.begin(), d_B.end(), 0.0f, thrust::plus<float>());
    calc_print_elapsed("thrust reduce(sum)", start_reduce);

    calc_print_elapsed("thrustDot total", start_total);

    return sum;

  }

}


