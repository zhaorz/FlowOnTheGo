/**
 * Implements kernels for optimization steps
 * Mainly interpolating patch, computing cost error,
 * calculating deltap and computing cost error.
 */

#ifndef __KERNEL_OPTIMIZE_H__
#define __KERNEL_OPTIMIZE_H__

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Local
#include "../common/Exceptions.h"
#include "../common/timer.h"
#include "../sandbox/process.h"
#include "../patch.h"

using namespace OFC;

namespace cu {

  void interpolateAndComputeErr(dev_patch_state* states,
      float** raw_diff, float** costs, float** patches, const float* I1,
      int n_patches, const opt_params* op,
      const img_params* i_params, bool project);


}

#endif // end __KERNEL_OPTIMIZE_H__
