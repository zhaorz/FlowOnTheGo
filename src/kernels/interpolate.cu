/**
 * Implements a bilinear interpolation kernel
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


#include "interpolate.h"

__global__ void kernelInterpolatePatch(
    float* pDeviceRawDiff, const float* pDeviceI, float* weight,
    int width_pad, int starty, int startx, int patchSize) {

  int x = threadIdx.x + startx;
  int y = blockIdx.x  + starty;
  int patchIdx = threadIdx.x + blockIdx.x * patchSize;

  if (x < startx + patchSize && y < starty + patchSize) {
    const float* img_e = pDeviceI + x * 3;
    const float* img_a = img_e + y * width_pad * 3;
    const float* img_c = img_e + (y - 1) * width_pad * 3;
    const float* img_b = img_a - 3;
    const float* img_d = img_c - 3;

    int diff = x * 3 + y * width_pad * 3;
    pDeviceRawDiff[3 * patchIdx] =
      weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);
    ++img_a; ++img_b; ++img_c; ++img_d;
    pDeviceRawDiff[3 * patchIdx + 1] =
      weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);
    ++img_a; ++img_b; ++img_c; ++img_d;
    pDeviceRawDiff[3 * patchIdx + 2] =
      weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);
  }

}

__global__ void kernelNormalizeMean(
    float* src, float mean, int patch_size) {

  int i = blockIdx.x * patch_size + threadIdx.x;

  src[3 * i]     -= mean;
  src[3 * i + 1] -= mean;
  src[3 * i + 2] -= mean;

}


namespace cu {

  void interpolatePatch(
      float* pDeviceRawDiff, const float* pDeviceI, float* weight,
      int width_pad, int starty, int startx, int patchSize) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    kernelInterpolatePatch<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceRawDiff, pDeviceI, weight,
        width_pad, starty, startx, patchSize);

  }

  void normalizeMean(
      float* src, cublasHandle_t handle, int patchSize) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    float mean;
    CUBLAS_CHECK (
        cublasSasum(handle, patchSize * patchSize * 3, src, 1, &mean) );
    mean = mean / (patchSize * patchSize * 3);

    kernelNormalizeMean<<<nBlocks, nThreadsPerBlock>>>(
        src, mean, patchSize);

  }

}
