/**
 * Implements a sobel kernel
 */

#ifndef __KERNEL_SOBEL_H__
#define __KERNEL_SOBEL_H__

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
   * Perform a sobel filter on src and store it in dest.
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
      int ksize, double scale, double delta, int borderType);

}

#endif // end __KERNEL_SOBEL_H__
