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

using namespace timer;

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


  __shared__ float h00_0, h01_0, h11_0;
  __shared__ float h00_1, h01_1, h11_1;
  if (tid == 0) {

    h00_0 = 0.0, h01_0 = 0.0, h11_0 = 0.0;

    for (int i = 0; i < patch_size * patch_size * 3 / 2; i++) {
      h00_0 += XX[i];
      h01_0 += XY[i];
      h11_0 += YY[i];
    }

  } else if (tid == 1) {

    h00_1 = 0.0, h01_1 = 0.0, h11_1 = 0.0;

    for (int i = patch_size * patch_size * 3 / 2; i < patch_size * patch_size * 3; i++) {
      h00_1 += XX[i];
      h01_1 += XY[i];
      h11_1 += YY[i];
    }

  }

  __syncthreads();
  if (tid == 0) {
    float h00 = h00_0 + h00_1;
    float h01 = h01_0 + h01_1;
    float h11 = h11_0 + h11_1;

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
    float* flowPrev, dev_patch_state* states,
    int width, int lb, int ub_w, int ub_h) {

  int patchId = blockIdx.x;
  int x = floor(states[patchId].midpoint_orgx / 2);
  int y = floor(states[patchId].midpoint_orgy / 2);
  int i = y * width + x;

  states[patchId].p_orgx = flowPrev[2 * i] * 2;
  states[patchId].p_orgy = flowPrev[2 * i + 1] * 2;
  states[patchId].p_curx = flowPrev[2 * i] * 2;
  states[patchId].p_cury = flowPrev[2 * i + 1] * 2;

  states[patchId].midpoint_curx += states[patchId].p_curx;
  states[patchId].midpoint_cury += states[patchId].p_cury;

  //Check if initial position is already invalid
  if (states[patchId].midpoint_curx < lb
      || states[patchId].midpoint_cury < lb
      || states[patchId].midpoint_curx > ub_w
      || states[patchId].midpoint_cury > ub_h) {

    states[patchId].has_converged = 1;
    states[patchId].has_opt_started = 1;

  } else {

    states[patchId].count = 0; // reset iteration counter
    states[patchId].delta_p_sq_norm = 1e-10;
    states[patchId].delta_p_sq_norm_init = 1e-10;  // set to arbitrary low value, s.t. that loop condition is definitely true on first iteration
    states[patchId].mares = 1e5;          // mean absolute residual
    states[patchId].mares_old = 1e20; // for rate of change, keep mares from last iteration in here. Set high so that loop condition is definitely true on first iteration
    states[patchId].has_converged = 0;

    states[patchId].has_opt_started = 1;
    states[patchId].invalid = false;

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

    cudaDeviceSynchronize();

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

    // cudaDeviceSynchronize();

  }


  void initCoarserOF(float* flowPrev, dev_patch_state* states,
      int n_patches, const img_params* i_params) {

    int nBlocks = n_patches;
    int nThreadsPerBlock = 1;

    kernelInitCoarserOF<<<nBlocks, nThreadsPerBlock>>>(
        flowPrev, states, i_params->width / 2, i_params->l_bound,
        i_params->u_bound_width, i_params->u_bound_height);

    // cudaDeviceSynchronize();

  }


}
