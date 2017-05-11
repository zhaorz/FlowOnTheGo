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

// Local
#include "../common/Exceptions.h"
#include "../common/timer.h"
#include "../sandbox/process.h"

namespace cu {

  void interpolatePatch(
      float* pDeviceRawDiff, const float* pDeviceI, float* weight,
      int width_pad, int starty, int startx, int patchSize);

  void normalizeMean(
      float* pDeviceRawDiff, int patchSize);

}

#endif // end __KERNEL_INTERPOLATE_H__
