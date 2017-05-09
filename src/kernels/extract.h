/**
 * Implements a patch extraction kernel
 */

#ifndef __KERNEL_EXTRACT_H__
#define __KERNEL_EXTRACT_H__

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

  void extractPatch(
      float* pDevicePatch, float* pDevicePatchX, float* pDevicePatchY,
      const float* I0, const float* I0x, const float* I0y, int patch_offset,
      int patch_size, int width_pad);

}

#endif // end __KERNEL_EXTRACT_H__
