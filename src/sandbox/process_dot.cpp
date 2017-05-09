// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_filtering_functions.h>

// Local
#include "../common/timer.h"
#include "process.h"

#include "../kernels/thrustDot.h"
#include "../kernels/cublasDot.h"

using namespace timer;

void process(const char* input_file, const char* output_file) {

  std::cout << "Initializing host arrays" << std::endl;

  int N = 64;
  int repeat = 4;

  float* A = new float[N];
  float* B = new float[N];

  for (int i = 0; i < N; i++) {
    // A[i] = (i * 251) % 17;
    // B[i] = ((N-i) * 17) % 29;
    A[i] = i;
    B[i] = i;
  }

  for (int c = 0; c < repeat; c++) {
    float dotProduct = cu::thrustDot(A, B, N);
    std::cout << "thrustDot(A,B) = " << dotProduct << std::endl;
  }

  std::cout << std::endl;

  for (int c = 0; c < repeat; c++) {
    float dotProduct = cu::cublasDot(A, B, N);
    std::cout << "cublasDot(A,B) = " << dotProduct << std::endl;
  }

  delete[] A;
  delete[] B;
}

