/**
 * Implements pyramid construction as a kernel.
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
#include "../common/Exceptions.h"
#include "../common/timer.h"

#include "pyramid.h"

using namespace timer;

namespace cu {

  void constructImgPyramids(
      const cv::Mat& I,
      cv::Mat* Is, cv::Mat* Ixs, cv::Mat* Iys,
      int nLevels) {

    // Timing
    auto start_total = now();
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = channels * sizeof(float);

    // Setup
    Is[0] = I.clone();

    cv::Size sz = I.size();
    int width   = sz.width;
    int height  = sz.height;

    unsigned int nSrcStep = width * elemSize;

    // Gradient params
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
    NppiSize  oSize   = { width, height };
    NppiPoint oOffset = { 0, 0 };
    NppiSize  oROI    = { width, height };

    // Resize params
    int eInterpolation = NPPI_INTER_LINEAR;    // Linear interpolation
    double scaleX = 0.5;
    double scaleY = 0.5;
    double shiftX = 0.0;
    double shiftY = 0.0;

    std::cout << "[start] constructImgPyramids: processing "
      << width << "x" << height << " image" << std::endl;

    // Allocate device memory
    auto start_cuda_malloc = now();
    Npp32f *pDeviceI, *pDeviceIx, *pDeviceIy, *pDeviceTmp;

    checkCudaErrors( cudaMalloc((void**) &pDeviceI,   width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceIx,  width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceIy,  width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceTmp, width * height * elemSize) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);

    // Copy over initial image
    auto start_memcpy_hd = now();

    checkCudaErrors(
        cudaMemcpy(pDeviceI, (float*) Is[0].data, width * height * elemSize, cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy I[0] H->D", start_memcpy_hd);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Apply first gradients to Is[0]
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // dx's
    auto start_dx = now();
    NPP_CHECK_NPP(
        nppiFilterSobelHorizBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIx, nSrcStep, oROI, eBorderType) );
    compute_time += calc_print_elapsed("sobel: Ixs[0]", start_dx);

    auto start_cp_dx = now();
    Ixs[0].create(height, width, CV_32FC3);
    checkCudaErrors(
        cudaMemcpy(Ixs[0].data, pDeviceIx,
          width * height * elemSize, cudaMemcpyDeviceToHost) );
    compute_time += calc_print_elapsed("Ixs[0] cudaMemcpy D->H", start_cp_dx);

    // dy's
    auto start_dy = now();
    NPP_CHECK_NPP(
        nppiFilterSobelVertBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIy, nSrcStep, oROI, eBorderType) );
    compute_time += calc_print_elapsed("sobel: Iys[0]", start_dy);

    auto start_cp_dy = now();
    Iys[0].create(height, width, CV_32FC3);
    checkCudaErrors(
        cudaMemcpy(Iys[0].data, pDeviceIy,
          width * height * elemSize, cudaMemcpyDeviceToHost) );
    compute_time += calc_print_elapsed("Iys[0] cudaMemcpy D->H", start_cp_dy);


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // For every finer level, resize and apply the gradients
    ////////////////////////////////////////////////////////////////////////////////////////////////


    for (int i = 1; i < nLevels; i++) {

      // Get the new size
      NppiRect srcRect = { 0, 0, width, height };
      NppiRect dstRect;
      NPP_CHECK_NPP(
          nppiGetResizeRect (srcRect, &dstRect, scaleX, scaleY, shiftX, shiftY, eInterpolation) );

      std::cout << "constructImgPyramids level " << i << ": "
        << dstRect.width << "x" << dstRect.height << std::endl;

      int nDstStep = dstRect.width * elemSize;

      // Resize I => Tmp
      NPP_CHECK_NPP(
          nppiResizeSqrPixel_32f_C3R (
            pDeviceI, oSize, nSrcStep, srcRect,
            pDeviceTmp, nDstStep, dstRect,
            scaleX, scaleY, shiftX, shiftY, eInterpolation) );

      // Put the resized image back into I
      std::swap(pDeviceI, pDeviceTmp);

      // And update the dimensions
      nSrcStep = nDstStep;
      width = dstRect.width; height = dstRect.height;
      oSize.width = width; oSize.height = height;
      oROI.width  = width; oROI.height  = height;

      // dx's
      auto start_dx = now();
      NPP_CHECK_NPP(
          nppiFilterSobelHorizBorder_32f_C3R (
            pDeviceI,  nSrcStep, oSize, oOffset,
            pDeviceIx, nSrcStep, oROI, eBorderType) );
      compute_time += calc_print_elapsed("sobel: Ixs[i]", start_dx);

      // dy's
      auto start_dy = now();
      NPP_CHECK_NPP(
          nppiFilterSobelVertBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize, oOffset,
            pDeviceIy, nSrcStep, oROI, eBorderType) );
      compute_time += calc_print_elapsed("sobel: Iys[i]", start_dy);

      // Allocate host destinations
      auto start_host_alloc = now();
      Is[i].create(dstRect.height, dstRect.width, CV_32FC3);
      Ixs[i].create(dstRect.height, dstRect.width, CV_32FC3);
      Iys[i].create(dstRect.height, dstRect.width, CV_32FC3);

      std::cout << "Is[" << i << "]: "  << Is[i].size() << " channels: " << Is[i].channels()
        << " type: " << Is[i].type() << std::endl;

      compute_time += calc_print_elapsed("host alloc", start_host_alloc);

      // Copy over data
      auto start_cp = now();
      checkCudaErrors(
          cudaMemcpy(Is[i].data, pDeviceI,
            dstRect.width * dstRect.height * elemSize, cudaMemcpyDeviceToHost) );
      checkCudaErrors(
          cudaMemcpy(Ixs[i].data, pDeviceIx,
            dstRect.width * dstRect.height * elemSize, cudaMemcpyDeviceToHost) );
      checkCudaErrors(
          cudaMemcpy(Iys[i].data, pDeviceIy,
            dstRect.width * dstRect.height * elemSize, cudaMemcpyDeviceToHost) );
      compute_time += calc_print_elapsed("pyramid cudaMemcpy D->H", start_cp);

    }

    calc_print_elapsed("total time", start_total);
    std::cout << "[done] constructImgPyramids: primmary compute time: " << compute_time  << std::endl;
  }

}

