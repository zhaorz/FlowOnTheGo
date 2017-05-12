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
#include "../patch.h"

using namespace OFC;

namespace cu {

  void densifyPatch(
      float* pDeviceCostDiff, float* pDeviceFlowOut, float* pDeviceWeights,
      dev_patch_state* states, int ip, 
      int midpointX, int midpointY,
      int width, int height,
      int patchSize, float minErrVal);


  void densifyPatches(
      float** costs, float* flow, float* weights,
      dev_patch_state* states, int n_patches,
      const opt_params* op, const img_params* i_params);


  void normalizeFlow(
      float* pDeviceFlowOut, float* pDeviceWeights, int N);

}

#endif // end __KERNEL_DENSIFY_H__

