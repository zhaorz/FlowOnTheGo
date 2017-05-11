/**
 * Implements a sobel kernel
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <cstring>

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

#include "sobel.h"

using namespace timer;

namespace cu {

  /**
   * Perform a sobel filter on src and store it in dest.
   * dest must be allocated.
   * Accepts 3-channel 32-bit float matrices.
   *
   * Params:
   *   src     input image.
   *   dst     output image of the same size and the same number of channels as src .
   *   ddepth  output image depth.
   *   dx      order of the derivative x. Only 0, 1 supported.
   *   dy      order of the derivative y. Only 0, 1 supported.
   *   ksize   (unused) size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
   *   scale   (unused) optional scale factor for the computed derivative values
   *   delta   (optional|unused) delta value that is added to the results prior to storing them in dst.
   *   borderType  (unused) pixel extrapolation method, see cv::BorderTypes
   */
  void sobel(
      const cv::Mat& src, cv::Mat& dest, int ddepth, int dx, int dy,
      int ksize, double scale, double delta, int borderType) {

    if (src.type() != CV_32FC3) {
      throw std::invalid_argument("sobel: invalid input matrix type");
    }

    if ( !((dx == 1 && dy == 0) ||
          (dx == 0 && dy == 1)) ) {
      throw std::invalid_argument("sobel: only accepts first order derivatives");
    }

    // Compute time of relevant kernel
    double compute_time = 0.0;
    double total_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = 3 * sizeof(float);

    cv::Size sz = src.size();
    int width   = sz.width;
    int height  = sz.height;

    std::cout << "[start] sobel: processing " << width << "x" << height << " image" << std::endl;

    // pSrc pointer to image data
    Npp32f* pHostSrc = (float*) src.data;

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * elemSize;
    unsigned int nDstStep = nSrcStep;

    NppiSize oSrcSize = { width, height };
    NppiPoint oSrcOffset = { 0, 0 };
    NppiSize oSizeROI = { width, height };

    // For 1D convolution
    const Npp32f pKernel[3] = { 1, 0, -1 };
    Npp32s nMaskSize =  3;
    Npp32s nAnchor   = 1;  // Kernel is centered over pixel
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp32f* pDeviceSrc, *pDeviceDst;

    checkCudaErrors( cudaHostAlloc((void**) &pDeviceSrc, width * height * elemSize, cudaHostAllocMapped) );
    checkCudaErrors( cudaHostAlloc((void**) &pDeviceDst, width * height * elemSize, cudaHostAllocMapped) );

    // For custom row/col kernel
    Npp32f* pDeviceKernel;
    checkCudaErrors( cudaHostAlloc((void**) &pDeviceKernel, nMaskSize * sizeof(Npp32f), cudaHostAllocMapped) );

    calc_print_elapsed("cudaHostAlloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    std::memcpy(pDeviceSrc, pHostSrc, width * height * elemSize);
    std::memcpy(pDeviceKernel, pKernel, nMaskSize * sizeof(Npp32f));

    calc_print_elapsed("std::memcpy D->H", start_memcpy_hd);


    bool useHoriz = (dx == 1);

    auto start_sobel = now();

    NPP_CHECK_NPP(
        (useHoriz)
        // For built in sobel
        // ? nppiFilterPrewittHorizBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)
        // : nppiFilterPrewittVertBorder_32f_C3R  (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)

        // Custom row filter
        ? nppiFilterRowBorder_32f_C3R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI, pDeviceKernel, nMaskSize, nAnchor, eBorderType)

        : nppiFilterColumnBorder_32f_C3R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI, pDeviceKernel, nMaskSize, nAnchor, eBorderType)

        // Sobel with mask
        // ? nppiFilterSobelHorizMaskBorder_32f_C1R (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, NPP_MASK_SIZE_1_X_3, eBorderType)
        // : nppiFilterSobelVertMaskBorder_32f_C1R  (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, NPP_MASK_SIZE_1_X_3, eBorderType)
        );

    compute_time += calc_print_elapsed("sobel", start_sobel);


    auto start_memcpy_dh = now();

    // Copy result to host
    dest.create(height, width, CV_32FC3);
    float* pHostDst = (float*) dest.data;

    std::memcpy(pHostDst, pDeviceDst, width * height * elemSize);

    total_time += calc_print_elapsed("std::memcpy H<-D", start_memcpy_dh);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    // Only for custom row/col filter
    cudaFree((void*) pDeviceKernel);

    std::cout << "[done] sobel" << std::endl;
    std::cout << "  primary compute time: " << compute_time << " (ms)" << std::endl;
    std::cout << "  total compute time:   " << compute_time + total_time << " (ms)" << std::endl;
  }

}

