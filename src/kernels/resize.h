/**
 * Implements a resize kernel
 */

#ifndef __KERNEL_RESIZE_H__
#define __KERNEL_RESIZE_H__

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
      double fx, double fy, int interpolation);

}

#endif // end __KERNEL_RESIZE_H__
