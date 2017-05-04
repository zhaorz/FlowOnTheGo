/**
 * Implements a resize kernel
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

#include "resize.h"

using namespace timer;

namespace cu {

  /**
   * Perform a resize filter on src and store it in dest.
   * Accepts 3-channel 32-bit float matrices.
   *
   * Params:
   *   src  input image.
   *   dst  output image; it has the size dsize (when it is non-zero)
   *        or the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
   *   dsize  output image size; if it equals zero, it is computed as:
   *   fx  scale factor along the horizontal axis
   *   fy  scale factor along the vertical axis
   *   interpolation  interpolation method, see cv::InterpolationFlags
   */
  void resize(
      const cv::Mat& src, cv::Mat& dest, cv::Size dsize,
      double fx, double fy, int interpolation) {

    if (src.type() != CV_32FC3) {
      throw std::invalid_argument("resize: invalid input matrix type");
    }

    // Compute time of relevant kernel
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = 3 * sizeof(float);

    cv::Size sz = src.size();
    int width   = sz.width;
    int height  = sz.height;

    std::cout << "[start] resize: processing " << width << "x" << height << " image" << std::endl;

    // pSrc pointer to image data
    Npp32f* pHostSrc = (float*) src.data;

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * elemSize;

    NppiSize oSrcSize = { width, height };
    NppiRect oSrcROI  = { 0, 0, width, height };

    NppiRect dstRect;

    double scaleX = 0.5;
    double scaleY = 0.5;
    double shiftX = 0.0;
    double shiftY = 0.0;

    int eInterpolation = NPPI_INTER_LINEAR;    // Linear interpolation


    auto start_get_resize_rect = now();

    // Get the destination size
    NPP_CHECK_NPP(
        nppiGetResizeRect (oSrcROI, &dstRect, scaleX, scaleY, shiftX, shiftY, eInterpolation) );

    calc_print_elapsed("get_resize_rect", start_get_resize_rect);

    unsigned int nDstStep = dstRect.width * elemSize;

    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp32f* pDeviceSrc, *pDeviceDst;

    checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, dstRect.width * dstRect.height * elemSize) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * elemSize, cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    auto start_resize = now();

    NPP_CHECK_NPP(

        nppiResizeSqrPixel_32f_C3R (
          pDeviceSrc, oSrcSize, nSrcStep, oSrcROI,
          pDeviceDst, nDstStep, dstRect,
          scaleX, scaleY, shiftX, shiftY, eInterpolation)

        );

    compute_time += calc_print_elapsed("resize", start_resize);


    auto start_memcpy_dh = now();

    // Copy result to host
    float* pHostDst = new float[dstRect.width * dstRect.height * channels];

    checkCudaErrors(
        cudaMemcpy(pHostDst, pDeviceDst,
          dstRect.width * dstRect.height * elemSize, cudaMemcpyDeviceToHost) );

    calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);

    cv::Mat dstWrapper(dstRect.height, dstRect.width, CV_32FC3, pHostDst);

    dstWrapper.copyTo(dest);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    delete[] pHostDst;

    std::cout << "[done] resize: primary compute time: " << compute_time << " (ms)" << std::endl;
  }

}

