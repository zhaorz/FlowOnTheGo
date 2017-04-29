/**
 * Basic tests on rgbMat
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// OpenCV
#include <opencv2/opencv.hpp>

// Local
#include "../common/timer.h"
#include "../common/rgbMat.h"
#include "process.h"

using namespace timer;

void process(const char* input_file, const char* output_file) {
  cv::Mat I0, I1;

  auto start_read = now();

  // Get input
  I0 = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);
  I0.convertTo(I0, CV_32FC3);

  calc_print_elapsed("imread", start_read);

  // Check for invalid input
  if(!I0.data) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    exit(1);
  }

  auto start_convert = now();

  rgbMat rgbI0(I0);

  calc_print_elapsed("Mat -> rgbMat", start_convert);


  auto start_toMat = now();

  I1 = rgbI0.toCVMat();

  calc_print_elapsed("rgbMat -> Mat", start_toMat);


  auto start_write = now();

  // Write output
  cv::imwrite(output_file, I1);

  calc_print_elapsed("write", start_write);
}

