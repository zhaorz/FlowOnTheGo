/**
 * Implements a patch extraction kernel
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cuda_helper.h"
#include "../common/Exceptions.h"
#include "../common/timer.h"


#include "extract.h"

__global__  void kernelExtractPatch(
    float* pDevicePatch, float* pDevicePatchX, float* pDevicePatchY,
    const float* I0, const float* I0x, const float* I0y, int patch_offset,
    int patchSize, int width_pad) {

  int patchIdx = threadIdx.x + blockIdx.x * patchSize;
  int imgIdx = patch_offset + threadIdx.x + blockIdx.x * width_pad;

  pDevicePatch[3 * patchIdx] = I0[3 * imgIdx];
  pDevicePatchX[3 * patchIdx] = I0x[3 * imgIdx];
  pDevicePatchY[3 * patchIdx] = I0y[3 * imgIdx];

  pDevicePatch[3 * patchIdx + 1] = I0[3 * imgIdx + 1];
  pDevicePatchX[3 * patchIdx + 1] = I0x[3 * imgIdx + 1];
  pDevicePatchY[3 * patchIdx + 1] = I0y[3 * imgIdx + 1];

  pDevicePatch[3 * patchIdx + 2] = I0[3 * imgIdx + 2];
  pDevicePatchX[3 * patchIdx + 2] = I0x[3 * imgIdx + 2];
  pDevicePatchY[3 * patchIdx + 2] = I0y[3 * imgIdx + 2];
}


namespace cu {

  void extractPatch(
      float* pDevicePatch, float* pDevicePatchX, float* pDevicePatchY,
      const float* I0, const float* I0x, const float* I0y, int patch_offset,
      int patchSize, int width_pad) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    kernelExtractPatch<<<nBlocks, nThreadsPerBlock>>>(
        pDevicePatch, pDevicePatchX, pDevicePatchY,
        I0, I0x, I0y, patch_offset,
        patchSize, width_pad);

  }

}
