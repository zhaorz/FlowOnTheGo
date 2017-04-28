/**
 * Implements a sobel process
 */

// System
#include <iostream>
#include <chrono>
#include <string>

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
// #include "../common/cuda_helper.h"
#include "process.h"

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
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(dt);
  return us.count() / 1000.0;
}

double calc_print_elapsed(const char* name, std::chrono::time_point<Clock> start) {
  double duration = time_diff(start, now());
  std::cout << "[time] " << duration << " (ms) : " << name  << std::endl;
  return duration;
}

void process(const char* input_file, const char* output_file) {
  cv::Mat I0, I0_f;

  // Record only relevant times
  double compute_time = 0.0;

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
  unsigned int pixel_width = 3 * sizeof(float);

  calc_print_elapsed("convertTo float", start_convert);


  cv::Size sz = I0_f.size();

  int width = sz.width;
  int height = sz.height;

  std::cout << "Processing " << width << "x" << height << " image" << std::endl;

  // pSrc pointer to image data
  Npp32f* pHostSrc = (float*) I0_f.data;

  // The width, in bytes, of the image, sometimes referred to as pitch
  unsigned int nSrcStep = width * pixel_width;
  unsigned int nDstStep = nSrcStep;

  NppiSize oSizeROI = { width, height };


  auto start_cuda_malloc = now();

  // Allocate device memory
  Npp32f* pDeviceSrc, *pDeviceDst;
  checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * pixel_width) );
  checkCudaErrors( cudaMalloc((void**) &pDeviceDst, width * height * pixel_width) );

  calc_print_elapsed("cudaMalloc", start_cuda_malloc);


  auto start_memcpy_hd = now();

  // Copy image to device
  checkCudaErrors(
      cudaMemcpy(pDeviceSrc, pHostSrc, width * height * pixel_width, cudaMemcpyHostToDevice) );

  calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


  auto start_sobel = now();

  NPP_CHECK_NPP(
      nppiFilterSobelHoriz_32f_C3R (pDeviceSrc, nSrcStep, pDeviceDst, nDstStep, oSizeROI) );

  compute_time += calc_print_elapsed("sobel", start_sobel);


  auto start_memcpy_dh = now();


  // Copy result to host, reuse the same pointer
  float* pHostDst = (float*) I0_f.data;
  checkCudaErrors(
      cudaMemcpy(pHostDst, pDeviceDst, width * height * pixel_width, cudaMemcpyDeviceToHost) );

  calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);


  auto start_write = now();

  // Write output
  cv::imwrite(output_file, I0_f);

  calc_print_elapsed("write", start_write);

  cudaFree((void*) pDeviceSrc);
  cudaFree((void*) pDeviceDst);

  std::cout << "[complete] Primary compute time: " << compute_time << " (ms)" << std::endl;
}

