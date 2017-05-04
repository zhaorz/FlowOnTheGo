/**
 * Implements a resize kernel
 */

#ifndef __KERNEL_RESIZE_GRAD_H__
#define __KERNEL_RESIZE_GRAD_H__

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

  void resizeGrad(
      const cv::Mat& src,
      cv::Mat& dst,
      cv::Mat& dst_x,
      cv::Mat& dst_y,
      double scaleX, double scaleY);

}

#endif // end __KERNEL_RESIZE_GRAD_H__
