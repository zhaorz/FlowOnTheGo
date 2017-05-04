/**
 * Implements pyramid construction as a kernel
 */

#ifndef __KERNEL_PYRAMID_H__
#define __KERNEL_PYRAMID_H__

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
#include "../sandbox/process.h"

namespace cu {

  void constructImgPyramids(
      const cv::Mat& I,
      cv::Mat* Is, cv::Mat* Ixs, cv::Mat* Iys,
      int padding, int nLevels);

}

#endif // end __KERNEL_PYRAMID_H__
