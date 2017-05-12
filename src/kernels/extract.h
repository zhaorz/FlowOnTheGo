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

  void extractPatch(
      float* pDevicePatch, float* pDevicePatchX, float* pDevicePatchY,
      const float* I0, const float* I0x, const float* I0y, int patch_offset,
      int patch_size, int width_pad);

  // void extractPatchesAndHessians(
  //     float** patches, float** patchxs, float** patchys,
  //     const float * I0, const float * I0x, const float * I0y,
  //     float* H00, float* H01, float* H11,
  //     float** tempXX, float** tempXY, float** tempYY,
  //     float* midpointX, float* midpointY, int n_patches,
  //     const opt_params* op, const img_params* i_params);

  void extractPatchesAndHessians(
      float** patches, float** patchxs, float** patchys,
      const float * I0, const float * I0x, const float * I0y,
      float** tempXX, float** tempXY, float** tempYY,
      dev_patch_state* states, int n_patches,
      const opt_params* op, const img_params* i_params);

  void initCoarserOF(const float* flowPrev, dev_patch_state* states,
      int n_patches, const img_params* i_params);


}

#endif // end __KERNEL_EXTRACT_H__
