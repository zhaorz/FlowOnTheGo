#define CV_CPU_HAS_SUPPORT_SSE2 0

// System
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>
#include <math.h>

// OpenCV
#include <opencv2/opencv.hpp>

// Local
#include "../common/cuda_helper.h"
#include "process.h"


void print_usage() {
  std::cout << "Usage: ./sandbox <input_img> <output_img>" << std::endl;
  std::cout << "       ./sandbox <img1> <img2> ...  <output_prefix>" << std::endl;
}

int main(int argc, char **argv)
{
  if (argc == 3) {
    initializeCuda(argc, argv);

    const char* input_file  = argv[1];
    const char* output_file = argv[2];

    process(input_file, output_file);
  }

  else if (argc > 3) {
    initializeCuda(argc, argv);

    int num_inputs = argc - 2;
    std::string output_prefix(argv[argc-1]);

    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const char* input_file  = argv[1 + input_idx];
      std::string output_file = output_prefix + "_" + std::to_string(input_idx) + ".png";

      std::cout << "[" << input_idx + 1 << "/" << num_inputs << "] Processing "
        << input_file << " => " << output_file << std::endl;

      process(input_file, output_file.c_str());

      std::cout << std::endl;
    }
  }

  else {
    print_usage();
  }

  return 0;
}
