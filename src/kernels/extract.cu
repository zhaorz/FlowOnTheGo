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


// __global__ void kernelExtractPatchesAndHessians(
//     float** patches, float** patchxs, float** patchys,
//     const float * I0, const float * I0x, const float * I0y,
//     float* H00, float* H01, float* H11,
//     float** tempXX, float** tempXY, float** tempYY,
//     float* midpointX, float* midpointY, int padding,
//     int patch_size, int width_pad) {
__global__ void kernelExtractPatchesAndHessians(
    float** patches, float** patchxs, float** patchys,
    const float * I0, const float * I0x, const float * I0y,
    float** tempXX, float** tempXY, float** tempYY,
    dev_patch_state* states, int padding,
    int patch_size, int width_pad) {


  int patchId = blockIdx.x;
  int tid = threadIdx.x;
  float* patch = patches[patchId];
  float* patchX = patchxs[patchId];
  float* patchY = patchys[patchId];
  float* XX = tempXX[patchId];
  float* XY = tempXY[patchId];
  float* YY = tempYY[patchId];

  int x = round(states[patchId].midpoint_orgx) + padding;
  int y = round(states[patchId].midpoint_orgy) + padding;

  int lb = -patch_size / 2;
  int offset = 3 * ((x + lb) + (y + lb) * width_pad) + tid;

  for (int i = tid, j = offset; i < patch_size * patch_size * 3;
      i += 3 * patch_size, j += 3 * width_pad) {
    patch[i] = I0[j];
    patchX[i] = I0x[j];
    patchY[i] = I0y[j];
    XX[i] = patchX[i] * patchX[i];
    XY[i] = patchX[i] * patchY[i];
    YY[i] = patchY[i] * patchY[i];
  }

  __syncthreads();

  // Mean normalize
  __shared__ float mean;

  if (tid == 0) {

    mean = 0.0;
    for (int i = 0; i < patch_size * patch_size * 3; i++) {
      mean += patch[i];
    }
    mean /= patch_size * patch_size * 3;

  }

  __syncthreads();

  for (int i = tid; i < patch_size * patch_size * 3;
      i+= 3 * patch_size) {
    patch[i] -= mean;
  }

  // TODO: can this be done in parallel?
  if (tid == 0) {

    float h00 = 0.0, h01 = 0.0, h11 = 0.0;

    for (int i = 0; i < patch_size * patch_size * 3; i++) {
      h00 += XX[i];
      h01 += XY[i];
      h11 += YY[i];
    }

    // If not invertible adjust values
    if (h00 * h11 - h01 * h01 == 0) {
      h00 += 1e-10;
      h11 += 1e-10;
    }

    states[patchId].H00 = h00;
    states[patchId].H01 = h01;
    states[patchId].H11 = h11;

  }


}

// TODO: merge this with above kernel?
__global__ void kernelInitCoarserOF(
    float* flowPrev, dev_patch_state* states, int width) {

  int patchId = blockIdx.x;
  int x = floor(states[patchId].midpoint_orgx / 2);
  int y = floor(states[patchId].midpoint_orgy / 2);
  int i = y * width + x;

  states[patchId].p_orgx = flowPrev[2 * i] * 2;
  states[patchId].p_orgy = flowPrev[2 * i + 1] * 2;
  states[patchId].p_curx = flowPrev[2 * i] * 2;
  states[patchId].p_cury = flowPrev[2 * i + 1] * 2;

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


  // void extractPatchesAndHessians(
  //     float** patches, float** patchxs, float** patchys,
  //     const float * I0, const float * I0x, const float * I0y,
  //     float* H00, float* H01, float* H11,
  //     float** tempXX, float** tempXY, float** tempYY,
  //     float* midpointX, float* midpointY, int n_patches,
  //     const opt_params* op, const img_params* i_params) {
  void extractPatchesAndHessians(
      float** patches, float** patchxs, float** patchys,
      const float * I0, const float * I0x, const float * I0y,
      float** tempXX, float** tempXY, float** tempYY,
      dev_patch_state* states, int n_patches,
      const opt_params* op, const img_params* i_params) {

    int nBlocks = n_patches;
    int nThreadsPerBlock = 3 * op->patch_size;

    kernelExtractPatchesAndHessians<<<nBlocks, nThreadsPerBlock>>>(
        patches, patchxs, patchys,
        I0, I0x, I0y, tempXX, tempXY, tempYY, 
        states, i_params->padding, op->patch_size, i_params->width_pad);

  }


  void initCoarserOF(float* flowPrev, dev_patch_state* states,
      int n_patches, int width) {

    int nBlocks = n_patches;
    int nThreadsPerBlock = 1;

    kernelInitCoarserOF<<<nBlocks, nThreadsPerBlock>>>(flowPrev, states, width);

  }


}
