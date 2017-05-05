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
   * Perform border padding with constant (0) or replication on src.
   * Accepts 3-channel 32-bit float matrices. Returns pointer to device dest.
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
  Npp32f* pad(
      Npp32f* src,
      int width, int height,
      int top, int bottom, int left, int right, bool replicate) {

    // Compute time of relevant kernel
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = 3 * sizeof(float);

    int destWidth = left + width + right;
    int destHeight = top + height + bottom;

    std::cout << "[start] pad: processing " << width << "x" << height << " image" << std::endl;

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * elemSize;
    unsigned int nDstStep = destWidth * elemSize;

    NppiSize oSrcSizeROI = { width, height };
    // NppiSize oDestSizeROI = { left + width + right, top + height + bottom};
    NppiSize oDstSizeROI = { destWidth, destHeight };
    const Npp32f padVal[3] = {0.0, 0.0, 0.0};

    // Allocate device memory
    auto start_cuda_malloc = now();
    Npp32f* pDeviceDst;
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, destWidth * destHeight *  elemSize) );
    if (!replicate)
      checkCudaErrors( cudaMemset(pDeviceDst, 0, destWidth * destHeight * elemSize) );
    calc_print_elapsed("cudaMalloc", start_cuda_malloc);

    Npp32f* pDeviceSrc = src;

    auto start_pad = now();

    NPP_CHECK_NPP(

        (replicate)
        ? nppiCopyReplicateBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSizeROI, pDeviceDst,
            nDstStep, oDstSizeROI, top, left)
        : nppiCopyConstBorder_32f_C3R (pDeviceSrc, nSrcStep, oSrcSizeROI, pDeviceDst,
            nDstStep, oDstSizeROI, top, left, padVal)

        );

    compute_time += calc_print_elapsed("pad", start_pad);

    std::cout << "[done] pad: primary compute time: " << compute_time << " (ms)" << std::endl;

    return pDeviceDst;
  }

}
