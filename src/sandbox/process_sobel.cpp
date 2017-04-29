/**
 * Implements a sobel kernel
 */

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

#include "../kernels/sobel.h"

using namespace timer;

void process(const char* input_file, const char* output_file) {
  cv::Mat I0, I0_f;

  auto start_read = now();

  // Get input
  I0 = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);

  calc_print_elapsed("imread", start_read);

  // Check for invalid input
  if(!I0.data) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    exit(1);
  }

  auto start_convert = now();

  // Convert to float
  I0.convertTo(I0_f, CV_32FC3);

  cv::Mat I1_f = I0_f.clone();

  calc_print_elapsed("convertTo float", start_convert);


  cu::sobel(I0_f, I1_f, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);


  auto start_write = now();

  // Write output
  cv::imwrite(output_file, I1_f);

  calc_print_elapsed("write", start_write);
}

