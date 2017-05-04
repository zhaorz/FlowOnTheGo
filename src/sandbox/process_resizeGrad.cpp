/**
 * Implements a resizeGrad kernel
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

#include "../kernels/resizeGrad.h"

using namespace timer;

void process(const char* input_file, const char* output_file) {
  cv::Mat src, dst, dst_x, dst_y;

  auto start_read = now();

  // Get input
  cv::Mat I = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);

  calc_print_elapsed("imread", start_read);

  // Check for invalid input
  if(!I.data) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    exit(1);
  }

  auto start_convert = now();

  // Convert to float
  I.convertTo(src, CV_32FC3);

  calc_print_elapsed("convertTo float", start_convert);


  auto start_resize = now();

  cu::resizeGrad(src, dst, dst_x, dst_y, .5, .5);

  calc_print_elapsed("resizeGrad", start_resize);


  auto start_write = now();

  // Write output
  cv::imwrite(output_file, dst);
  cv::imwrite("out_dx.png", dst_x);
  cv::imwrite("out_dy.png", dst_y);

  calc_print_elapsed("write", start_write);
}

