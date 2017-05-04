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
      int padding, int nLevels) {

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

    // Mask params
    const Npp32f pSrcKernel[3] = { 1, 0, -1 };
    Npp32s nMaskSize = 3;
    Npp32s nAnchor   = 1;  // Kernel is centered over pixel

    // Resize params
    int eInterpolation = NPPI_INTER_LINEAR;    // Linear interpolation
    double scaleX = 0.5;
    double scaleY = 0.5;
    double shiftX = 0.0;
    double shiftY = 0.0;

    // Padding params
    int padWidth = 2 * padding + width;
    int padHeight = 2 * padding + height;
    const Npp32f PAD_VAL[3] = { 0.0, 0.0, 0.0 };

    std::cout << "[start] constructImgPyramids: processing "
      << width << "x" << height << " image" << std::endl;

    // Allocate device memory (to account for padding too
    auto start_cuda_malloc = now();
    Npp32f *pDeviceI, *pDeviceIx, *pDeviceIy;
    Npp32f *pDevicePaddedI, *pDevicePaddedIx, *pDevicePaddedIy;
    Npp32f *pDeviceTmp, *pDeviceKernel;

    checkCudaErrors( cudaMalloc((void**) &pDeviceI,  width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceIx, width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceIy, width * height * elemSize) );

    checkCudaErrors( cudaMalloc((void**) &pDevicePaddedI,  padWidth * padHeight * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDevicePaddedIx, padWidth * padHeight * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDevicePaddedIy, padWidth * padHeight * elemSize) );

    checkCudaErrors( cudaMalloc((void**) &pDeviceTmp,    width * height * elemSize)  );
    checkCudaErrors( cudaMalloc((void**) &pDeviceKernel, nMaskSize * sizeof(Npp32f)) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);

    // Copy over initial image and kernel
    auto start_memcpy_hd = now();

    checkCudaErrors(
        cudaMemcpy(pDeviceI, (float*) Is[0].data, width * height * elemSize, cudaMemcpyHostToDevice) );
    checkCudaErrors(
        cudaMemcpy(pDeviceKernel, pSrcKernel, nMaskSize * sizeof(Npp32f), cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy I[0] H->D", start_memcpy_hd);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Apply first gradients to Is[0]
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // dx's
    auto start_dx = now();
    NPP_CHECK_NPP(
        // nppiFilterSobelHorizBorder_32f_C3R (
        //   pDeviceI, nSrcStep, oSize, oOffset,
        //   pDeviceIx, nSrcStep, oROI, eBorderType)

        nppiFilterRowBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIx, nSrcStep, oROI,
          pDeviceKernel, nMaskSize, nAnchor, eBorderType)
        );
    compute_time += calc_print_elapsed("sobel: Ixs[0]", start_dx);

    // dy's
    auto start_dy = now();
    NPP_CHECK_NPP(
        // nppiFilterSobelVertBorder_32f_C3R (
        //   pDeviceI, nSrcStep, oSize, oOffset,
        //   pDeviceIy, nSrcStep, oROI, eBorderType)

        nppiFilterColumnBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIy, nSrcStep, oROI,
          pDeviceKernel, nMaskSize, nAnchor, eBorderType)
        );
    compute_time += calc_print_elapsed("sobel: Iys[0]", start_dy);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Pad Is[0] I, dx, dy
    ////////////////////////////////////////////////////////////////////////////////////////////////


    NppiSize oPadSize = { padWidth, padHeight };
    int nDstStep = oPadSize.width * elemSize;

    // Pad original
    NPP_CHECK_NPP(
        nppiCopyReplicateBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize,
          pDevicePaddedI, nDstStep, oPadSize, padding, padding) );

    // Pad dx, dy
    NPP_CHECK_NPP(
        nppiCopyConstBorder_32f_C3R (
          pDeviceIx, nSrcStep, oSize,
          pDevicePaddedIx, nDstStep, oPadSize, padding, padding, PAD_VAL) );
    NPP_CHECK_NPP(
        nppiCopyConstBorder_32f_C3R (
          pDeviceIy, nSrcStep, oSize,
          pDevicePaddedIy, nDstStep, oPadSize, padding, padding, PAD_VAL) );

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy Is[0] I, dx, dy
    ////////////////////////////////////////////////////////////////////////////////////////////////

    auto start_cp_Is0 = now();
    Is[0].create(oPadSize.height, oPadSize.width, CV_32FC3);
    checkCudaErrors(
        cudaMemcpy(Is[0].data, pDevicePaddedI,
          oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
    compute_time += calc_print_elapsed("Is[0] cudaMemcpy D->H", start_cp_Is0);

    auto start_cp_dx = now();
    Ixs[0].create(oPadSize.height, oPadSize.width, CV_32FC3);
    checkCudaErrors(
        cudaMemcpy(Ixs[0].data, pDevicePaddedIx,
          oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
    compute_time += calc_print_elapsed("Ixs[0] cudaMemcpy D->H", start_cp_dx);

    auto start_cp_dy = now();
    Iys[0].create(oPadSize.height, oPadSize.width, CV_32FC3);
    checkCudaErrors(
        cudaMemcpy(Iys[0].data, pDevicePaddedIy,
          oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
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
          // nppiFilterSobelHorizBorder_32f_C3R (
          //   pDeviceI,  nSrcStep, oSize, oOffset,
          //   pDeviceIx, nSrcStep, oROI, eBorderType)

          nppiFilterRowBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize, oOffset,
            pDeviceIx, nSrcStep, oROI,
            pDeviceKernel, nMaskSize, nAnchor, eBorderType)
          );
      compute_time += calc_print_elapsed("sobel: Ixs[i]", start_dx);

      // dy's
      auto start_dy = now();
      NPP_CHECK_NPP(
          // nppiFilterSobelVertBorder_32f_C3R (
          //   pDeviceI, nSrcStep, oSize, oOffset,
          //   pDeviceIy, nSrcStep, oROI, eBorderType)

          nppiFilterColumnBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize, oOffset,
            pDeviceIy, nSrcStep, oROI,
            pDeviceKernel, nMaskSize, nAnchor, eBorderType)
        );
      compute_time += calc_print_elapsed("sobel: Iys[i]", start_dy);

      //////////////////////////////////////////////////////////////////////////////////////////////
      // Pad I, dx, dy
      //////////////////////////////////////////////////////////////////////////////////////////////

      padWidth  = width + 2 * padding;
      padHeight = height + 2 * padding;
      NppiSize oPadSize = { padWidth, padHeight };
      nDstStep = oPadSize.width * elemSize;

      // Pad original
      NPP_CHECK_NPP(
          nppiCopyReplicateBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize,
            pDevicePaddedI, nDstStep, oPadSize, padding, padding) );

      // Pad dx, dy
      NPP_CHECK_NPP(
          nppiCopyConstBorder_32f_C3R (
            pDeviceIx, nSrcStep, oSize,
            pDevicePaddedIx, nDstStep, oPadSize, padding, padding, PAD_VAL) );
      NPP_CHECK_NPP(
          nppiCopyConstBorder_32f_C3R (
            pDeviceIy, nSrcStep, oSize,
            pDevicePaddedIy, nDstStep, oPadSize, padding, padding, PAD_VAL) );

      // Allocate host destinations
      auto start_host_alloc = now();
      Is[i].create(oPadSize.height, oPadSize.width, CV_32FC3);
      Ixs[i].create(oPadSize.height, oPadSize.width, CV_32FC3);
      Iys[i].create(oPadSize.height, oPadSize.width, CV_32FC3);
      compute_time += calc_print_elapsed("host alloc", start_host_alloc);

      // Copy over data
      auto start_cp = now();
      checkCudaErrors(
          cudaMemcpy(Is[i].data, pDevicePaddedI,
            oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
      checkCudaErrors(
          cudaMemcpy(Ixs[i].data, pDevicePaddedIx,
            oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
      checkCudaErrors(
          cudaMemcpy(Iys[i].data, pDevicePaddedIy,
            oPadSize.width * oPadSize.height * elemSize, cudaMemcpyDeviceToHost) );
      compute_time += calc_print_elapsed("pyramid cudaMemcpy D->H", start_cp);

    }

    // Clean up
    cudaFree(pDeviceI);
    cudaFree(pDeviceIx);
    cudaFree(pDeviceIy);
    cudaFree(pDevicePaddedI);
    cudaFree(pDevicePaddedIx);
    cudaFree(pDevicePaddedIy);
    cudaFree(pDeviceTmp);
    cudaFree(pDeviceKernel);

    calc_print_elapsed("total time", start_total);
    std::cout << "[done] constructImgPyramids: primmary compute time: " << compute_time  << std::endl;
  }

}

