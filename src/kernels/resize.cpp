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
      cv::Mat src, cv::Mat dest, cv::Size dsize,
      double fx, double fy, int interpolation) {

    if (src.type() != CV_32FC3) {
      throw std::invalid_argument("resize: invalid input matrix type");
    }

    if ( !((dx == 1 && dy == 0) ||
          (dx == 0 && dy == 1)) ) {
      throw std::invalid_argument("resize: only accepts first order derivatives");
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

    // For custom row/col kernel
    // Npp32f* pDeviceKernel;
    // checkCudaErrors( cudaMalloc((void**) &pDeviceKernel, nMaskSize * sizeof(Npp32f)) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * elemSize, cudaMemcpyHostToDevice) );

    // Copy kernel to device (only for custom row/col filter)
    // checkCudaErrors(
    //     cudaMemcpy(pDeviceKernel, pKernel, nMaskSize * sizeof(Npp32f), cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    bool useHoriz = (dx == 1);

    auto start_resize = now();

    NPP_CHECK_NPP(
        (useHoriz)
        // For built in resize
        ? nppiFilterSobelHorizBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)
        : nppiFilterSobelVertBorder_32f_C3R  (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, eBorderType)

        // Custom row filter
        // ? nppiFilterRowBorder_32f_C3R    (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, pDeviceKernel, nMaskSize, nAnchor, eBorderType)
        // : nppiFilterColumnBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset, pDeviceDst, nDstStep, oSizeROI, pDeviceKernel, nMaskSize, nAnchor, eBorderType)
        );

    compute_time += calc_print_elapsed("resize", start_resize);


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

    // Only for custom row/col filter
    // cudaFree((void*) pDeviceKernel);

    delete[] pHostDst;

    std::cout << "[done] resize: primary compute time: " << compute_time << " (ms)" << std::endl;
  }

}

