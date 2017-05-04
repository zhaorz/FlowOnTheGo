/**
 * Implements a pad kernel
 */

#ifndef __KERNEL_PAD_H__
#define __KERNEL_PAD_H__

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
#include "../sandbox/process.h"

namespace cu {

  /**
   * Perform border padding with constant (0) or replication on src and store it in dest.
   * Accepts 3-channel 32-bit float matrices.
   *
   * Params:
   *   src        input image.
   *   dst        output image;
   *              it has size of src + padding
   *   top        top padding
   *   bottom     bottom padding
   *   left       left padding
   *   right      right padding
   *   replicate  whether to replicate or constant pad
   */
  void pad(
      const cv::Mat& src, cv::Mat& dest, int top,
      int bottom, int left, int right, bool replicate);

}

#endif // end __KERNEL_PAD_H__
