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
      Npp32f* I, float** Is, float** Ixs, float** Iys,
      Npp32f* pDeviceIx, Npp32f* pDeviceIy, Npp32f* pDeviceTmp,
      Npp32f* pDeviceWew, int width, int height,
      int padding, int nLevels);

}

#endif // end __KERNEL_PYRAMID_H__
