#define CV_CPU_HAS_SUPPORT_SSE2 0

// System
#include <iomanip>
#include <iostream>
#include <chrono>
#include <math.h>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA
#include <cuda.h>

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_color_conversion.h>



#define MAX_FEATURES 100


/**
 * General purpose high resolution timer.
 *
 * Usage:
 *
 *  auto start = now();
 *
 *  DIS->calc(prevGray, gray, flow);
 *
 *  auto dt = now() - start;
 *  auto us = std::chrono::duration_cast<std::chrono::milliseconds>(dt);
 *  double duration = us.count();
 *
 *  std::cout << "time: " << duration << " ms" << std::endl;
 *
 */
static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

void print_usage() {
  std::cout << "Usage: ./sandbox <input_img> <output_img>" << std::endl;
}

int main(int argc, char **argv)
{
  const char* input_file  = argv[1];
  const char* output_file = argv[2];

  if (argc == 3) {
    cv::Mat I0, I1;

    // Get input
    I0 = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);

    // Check for invalid input
    if(!I0.data) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

    I1 = I0.clone();

    // Write output
    cv::imwrite(output_file, I1);

    return 0;
  }

  else {
    print_usage();
  }

  return 0;
}
