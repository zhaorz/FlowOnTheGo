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
#include <cuda_runtime.h>

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_filtering_functions.h>

// Local
#include "../common/Exceptions.h"
#include "../common/cuda_helper.h"



#define MAX_FEATURES 100

typedef std::chrono::high_resolution_clock Clock;

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
static std::chrono::time_point<Clock> now() {
  return Clock::now();
}

double time_diff(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> end) {
  auto dt = end - start;
  auto us = std::chrono::duration_cast<std::chrono::milliseconds>(dt);
  return us.count();
}

double calc_print_elapsed(const char* name, std::chrono::time_point<Clock> start) {
  double duration = time_diff(start, now());
  std::cout << "[time] " << duration << " (ms) : " << name  << std::endl;
  return duration;
}

void print_usage() {
  std::cout << "Usage: ./sandbox <input_img> <output_img>" << std::endl;
}

int main(int argc, char **argv)
{
  const char* input_file  = argv[1];
  const char* output_file = argv[2];

  initializeCuda(argc, argv);

  if (argc == 3) {
    cv::Mat I0, I0_f;

    // Record only relevant times
    double compute_time = 0.0;

    auto start_read = now();

    // Get input
    I0 = cv::imread(input_file, cv::IMREAD_GRAYSCALE);

    calc_print_elapsed("imread", start_read);

    // Check for invalid input
    if(!I0.data) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }


    auto start_convert = now();

    // Convert to float
    I0.convertTo(I0_f, CV_32F);

    calc_print_elapsed("convertTo float", start_convert);


    cv::Size sz = I0_f.size();

    int width = sz.width;
    int height = sz.height;

    std::cout << "Processing " << width << "x" << height << " image" << std::endl;

    // pSrc pointer to image data
    Npp32f* pHostSrc = (float*) I0_f.data;

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * sizeof(float);
    unsigned int nDstStep = nSrcStep;

    NppiSize oSizeROI = { width, height };


    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp32f* pDeviceSrc, *pDeviceDst;
    checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, width * height * sizeof(float)) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * sizeof(float), cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    auto start_sobel = now();

    NPP_CHECK_NPP(
        nppiFilterSobelHoriz_32f_C1R (pDeviceSrc, nSrcStep, pDeviceDst, nDstStep, oSizeROI) );

    compute_time += calc_print_elapsed("sobel", start_sobel);


    auto start_memcpy_dh = now();

    // Copy result to host, reuse the same pointer
    float pHostDst[width * height * sizeof(float)];
    checkCudaErrors(
        cudaMemcpy(pHostDst, pDeviceDst, width * height * sizeof(float), cudaMemcpyDeviceToHost) );

    calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);


    auto start_mat_init = now();

    // Init output image
    cv::Mat I1(height, width, CV_32F, pHostDst);

    calc_print_elapsed("cv mat init", start_mat_init);


    auto start_write = now();

    // Write output
    cv::imwrite(output_file, I1);

    calc_print_elapsed("write", start_write);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    std::cout << "Done." << std::endl;

    std::cout << "Primary compute time: " << compute_time << " (ms)" << std::endl;

    return 0;
  }

  else {
    print_usage();
  }

  return 0;
}
