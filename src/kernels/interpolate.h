/**
 * Implements a bilinear interpolation kernel
 */

#ifndef __KERNEL_INTERPOLATE_H__
#define __KERNEL_INTERPOLATE_H__

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Local
#include "../common/Exceptions.h"
#include "../common/timer.h"
#include "../sandbox/process.h"

namespace cu {

  void interpolatePatch(
      float* pDeviceRawDiff, float* pDeviceI, float* weight,
      int width_pad, int starty, int startx, int patchSize);

  void normalizeMean(
      float* pDeviceRawDiff, cublasHandle_t handle, int patchSize);

}

#endif // end __KERNEL_INTERPOLATE_H__
