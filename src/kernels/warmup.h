/**
 * Implements a resize kernel
 */

#ifndef __KERNEL_WARMUP_H__
#define __KERNEL_WARMUP_H__

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
   * Warms up the GPU by performing a malloc and memcpy and launching a kernel.
   */
  void warmup();

}

#endif // end __KERNEL_WARMUP_H__
