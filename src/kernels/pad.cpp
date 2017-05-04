/**
 * Implements a pad kernel
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

#include "pad.h"

using namespace timer;

namespace cu {

  /**
   * Perform border padding with constant (0) or replication on src and store it in dest.
   * Accepts 3-channel 32-bit float matrices.
   *
   * Params:
   *   src          input image.
   *   dst          output image;
   *                it has size of src + padding
   *   top          top padding
   *   bottom       bottom padding
   *   left         left padding
   *   right        right padding
   *   replicate    whether to replicate or constant
   */
  void pad(
      const cv::Mat& src, cv::Mat& dest, int top,
      int bottom, int left, int right, bool replicate) {

    if (src.type() != CV_32FC3) {
      throw std::invalid_argument("pad: invalid input matrix type");
    }

    // Compute time of relevant kernel
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = 3 * sizeof(float);

    cv::Size sz = src.size();
    int width   = sz.width;
    int height  = sz.height;
    int destWidth = left + width + right;
    int destHeight = top + height + bottom;

    std::cout << "[start] pad: processing " << width << "x" << height << " image" << std::endl;

    // pSrc pointer to image data
    Npp32f* pHostSrc = (float*) src.data;

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * elemSize;
    unsigned int nDstStep = destWidth * elemSize;

    NppiSize oSrcSizeROI = { width, height };
    // NppiSize oDestSizeROI = { left + width + right, top + height + bottom};
    NppiSize oDstSizeROI = { destWidth, destHeight };
    const Npp32f padVal[3] = {0.0, 0.0, 0.0};

    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp32f* pDeviceSrc, *pDeviceDst;

    checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, destWidth * destHeight *  elemSize) );
    checkCudaErrors( cudaMemset(pDeviceDst, 0, destWidth * destHeight * elemSize) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * elemSize, cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    auto start_pad = now();

    NPP_CHECK_NPP(

        (replicate)
        ? nppiCopyReplicateBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSizeROI, pDeviceDst,
            nDstStep, oDstSizeROI, top, left)
        : nppiCopyConstBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSizeROI, pDeviceDst,
            nDstStep, oDstSizeROI, top, left, padVal)

        );

    compute_time += calc_print_elapsed("pad", start_pad);


    auto start_memcpy_dh = now();

    // Copy result to host
    dest.create(destHeight, destWidth, CV_32FC3);

    checkCudaErrors(
        cudaMemcpy(dest.data, pDeviceDst,
          destWidth * destHeight * elemSize, cudaMemcpyDeviceToHost) );

    calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    std::cout << "[done] pad: primary compute time: " << compute_time << " (ms)" << std::endl;
  }

}
