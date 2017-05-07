/**
 * Implements kernels for flow densification
 */

#ifndef __KERNEL_DENSIFY_H__
#define __KERNEL_DENSIFY_H__

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

  void densifyPatch(
      float* pDeviceCostDiff, float* pDeviceFlowOut, float* pDeviceWeights,
      float flowX, float flowY,
      int midpointX, int midpointY,
      int width, int height, bool verbose,
      int patchSize, float minErrVal);

  void normalizeFlow(
      float* pDeviceFlowOut, float* pDeviceWeights, int N);

}

#endif // end __KERNEL_DENSIFY_H__

