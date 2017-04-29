/**
 * Implements a sobel kernel
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
      cv::Mat src, cv::Mat dest, int ddepth, int dx, int dy,
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

    // NppiBorderType eBorderType = NPP_BORDER_MIRROR;        ** raises NPP_NOT_SUPPORTED_MODE_ERROR(-9999)
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp32f* pDeviceSrc, *pDeviceDst;
    checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, width * height * elemSize) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * elemSize, cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    bool useHoriz = (dx == 1);

    auto start_sobel = now();

    NPP_CHECK_NPP(
        (useHoriz)
        // ? nppiFilterSobelHoriz_32f_C3R (pDeviceSrc, nSrcStep, pDeviceDst, nDstStep, oSizeROI)
        // : nppiFilterSobelVert_32f_C3R  (pDeviceSrc, nSrcStep, pDeviceDst, nDstStep, oSizeROI)
        ? nppiFilterSobelHorizBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)
        : nppiFilterSobelVertBorder_32f_C3R  (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)
        );

    compute_time += calc_print_elapsed("sobel", start_sobel);


    auto start_memcpy_dh = now();


    // Copy result to host
    float* pHostDst = new float[width * height * channels];

    checkCudaErrors(
        cudaMemcpy(pHostDst, pDeviceDst, width * height * elemSize, cudaMemcpyDeviceToHost) );

    calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);

    cv::Mat dest_wrapper(height, width, CV_32FC3, pHostDst);
    dest_wrapper.copyTo(dest);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    delete[] pHostDst;

    std::cout << "[done] sobel: primary compute time: " << compute_time << " (ms)" << std::endl;
  }

}

