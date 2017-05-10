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

  int patchIdx = threadIdx.x + blockIdx.x * 3 * patchSize;
  int imgIdx = 3 * patch_offset + threadIdx.x + blockIdx.x * 3 * width_pad;

  pDevicePatch[patchIdx] = I0[imgIdx];
  pDevicePatchX[patchIdx] = I0x[imgIdx];
  pDevicePatchY[patchIdx] = I0y[imgIdx];

}


__global__ void kernelExtractPatches(
    float** patches, float** patchxs, float** patchys,
    const float * I0, const float * I0x, const float * I0y,
    float* midpointX, float* midpointY, int padding,
    int patch_size, int width_pad) {


  int patchId = blockIdx.x;
  int tid = threadIdx.x;
  float* patch = patches[patchId];
  float* patchX = patchxs[patchId];
  float* patchY = patchys[patchId];

  int x = round(midpointX[patchId]) + padding;
  int y = round(midpointY[patchId]) + padding;

  int lb = -patch_size / 2;
  int offset = 3 * ((x + lb) + (y + lb) * width_pad) + tid;

  for (int i = tid, j = offset; i < patch_size * patch_size * 3;
      i += 3 * patch_size, j += 3 * width_pad) {
    patch[i] = I0[j];
    patchX[i] = I0x[j];
    patchY[i] = I0y[j];
  }

}


namespace cu {

  void extractPatch(
      float* pDevicePatch, float* pDevicePatchX, float* pDevicePatchY,
      const float* I0, const float* I0x, const float* I0y, int patch_offset,
      int patchSize, int width_pad) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = 3 * patchSize;

    kernelExtractPatch<<<nBlocks, nThreadsPerBlock>>>(
        pDevicePatch, pDevicePatchX, pDevicePatchY,
        I0, I0x, I0y, patch_offset,
        patchSize, width_pad);

  }


  void extractPatches(float** patches, float** patchxs, float** patchys,
      const float * I0, const float * I0x, const float * I0y,
      float* midpointX, float* midpointY, int n_patches,
      const opt_params* op, const img_params* i_params) {

    int nBlocks = n_patches;
    int nThreadsPerBlock = 3 * op->patch_size;

    kernelExtractPatches<<<nBlocks, nThreadsPerBlock>>>(
        patches, patchxs, patchys,
        I0, I0x, I0y, midpointX, midpointY,
        i_params->padding, op->patch_size, i_params->width_pad);

  }

}
