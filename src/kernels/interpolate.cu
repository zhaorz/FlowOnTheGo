/**
 * Implements a bilinear interpolation kernel
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "interpolate.h"

__global__ void kernelInterpolatePatch(
    float* pDeviceRawDiff, float* pDeviceI, float* weight,
    int width_pad, int starty, int startx, int patchSize) {

  int x = threadIdx.x + startx;
  int y = blockIdx.x  + starty;
  int patchIdx = threadIdx.x + blockIdx.x * patchSize;

  if (x < startx + patchSize && y < starty + patchSize) {
    float* img_e = pDeviceI + x * 3;
    float* img_a = img_e + y * width_pad * 3;
    float* img_c = img_e + (y - 1) * width_pad * 3;
    float* img_b = img_a - 3;
    float* img_d = img_c - 3;

    int diff = x * 3 + y * width_pad * 3;
    if (diff < 0 || patchIdx >= patchSize * patchSize) {
      printf("DEVICE: uh oh. patchIdx %d, x %d, y %d, startx %d, starty %d\n", patchIdx, x, y, startx, starty);
    }
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
    float* pDeviceRawDiff, float mean, int patch_size) {

  int i = blockIdx.x * patch_size + threadIdx.x;

  pDeviceRawDiff[3 * i]     -= mean;
  pDeviceRawDiff[3 * i + 1] -= mean;
  pDeviceRawDiff[3 * i + 2] -= mean;

}


namespace cu {

  void interpolatePatch(
      float* pDeviceRawDiff, float* pDeviceI, float* weight,
      int width_pad, int starty, int startx, int patchSize) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    kernelInterpolatePatch<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceRawDiff, pDeviceI, weight,
        width_pad, starty, startx, patchSize);

  }

  void normalizeMean(
      float* pDeviceRawDiff, float mean, int patchSize) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    kernelNormalizeMean<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceRawDiff, mean, patchSize);

  }

}
