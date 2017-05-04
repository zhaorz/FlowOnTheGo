/**
 * Test a pad kernel
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

#include "../kernels/pad.h"

using namespace timer;

void process(const char* input_file, const char* output_file) {
  cv::Mat I0, I0_f, I1_f;

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

  calc_print_elapsed("convertTo float", start_convert);


  auto start_resize = now();

  I1_f.data = (uchar*) cu::pad((float*) I0_f.data, I0_f.size().width, I0_f.size().height,
      20, 50, 100, 200, true);

  calc_print_elapsed("resize", start_resize);


  auto start_write = now();

  // Write output
  cv::imwrite(output_file, I1_f);

  calc_print_elapsed("write", start_write);
}

