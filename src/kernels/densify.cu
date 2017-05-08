/**
 * Implements kernels for flow densification
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "densify.h"

__global__ void kernelDensifyPatch(
    float* pDeviceCostDiff, float* pDeviceFlowOut, float* pDeviceWeights,
    float flowX, float flowY,
    int midpointX, int midpointY,
    int width, int height,
    int patchSize, float minErrVal) {

  int lower_bound = -patchSize / 2;

  int x = threadIdx.x + lower_bound;
  int y = blockIdx.x  + lower_bound;

  int xt = x + midpointX;
  int yt = y + midpointY;

  if (xt >= 0 && yt >= 0 && xt < width && yt < height) {

    int i = yt * width + xt;
    int j = blockIdx.x * patchSize + threadIdx.x;

    float absw = (float) (fmaxf(minErrVal, pDeviceCostDiff[3 * j]));
    absw += (float) (fmaxf(minErrVal, pDeviceCostDiff[3 * j + 1]));
    absw += (float) (fmaxf(minErrVal, pDeviceCostDiff[3 * j + 2]));
    absw = 1.0 / absw;

    // Weight contribution RGB
    pDeviceWeights[i] += absw;

    pDeviceFlowOut[2 * i] += flowX * absw;
    pDeviceFlowOut[2 * i + 1] += flowY * absw;
  }

}

__global__ void kernelNormalizeFlow(
    float* pDeviceFlowOut, float* pDeviceWeights, int N) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N && pDeviceWeights[i] > 0) {
    pDeviceFlowOut[2 * i]     /= pDeviceWeights[i];
    pDeviceFlowOut[2 * i + 1] /= pDeviceWeights[i];
  }

}

namespace cu {

  void densifyPatch(
      float* pDeviceCostDiff, float* pDeviceFlowOut, float* pDeviceWeights,
      float flowX, float flowY,
      int midpointX, int midpointY,
      int width, int height,
      int patchSize, float minErrVal) {

    int nBlocks = patchSize;
    int nThreadsPerBlock = patchSize;

    kernelDensifyPatch<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceCostDiff, pDeviceFlowOut, pDeviceWeights,
        flowX, flowY,
        midpointX, midpointY,
        width, height,
        patchSize, minErrVal);
  }

  void normalizeFlow(
      float* pDeviceFlowOut, float* pDeviceWeights, int N) {

    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelNormalizeFlow<<<nBlocks, nThreadsPerBlock>>>(pDeviceFlowOut, pDeviceWeights, N);
  }

}
