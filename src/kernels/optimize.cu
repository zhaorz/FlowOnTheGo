/**
 * Implements kernels for optimization steps
 * Mainly interpolating patch, computing cost error,
 * calculating deltap and computing cost error.
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


#include "optimize.h"

__device__ void calcProjection(dev_patch_state* states,
    float* patchX, float* patchY, float* tempX, float* tempY,
    float* raw, int patch_size, int out_thresh,
    int lb, int ubw, int ubh) {

  int patchId = blockIdx.x;
  int tid = threadIdx.x;

  for (int i = tid; i < 3 * patch_size * patch_size; i += 3 * patch_size) {
    tempX[i] = patchX[i] * raw[i];
    tempY[i] = patchY[i] * raw[i];
  }

  __syncthreads();

  if (tid == 0) {
    states[patchId].count++;

    float dpx = 0.0;
    float dpy = 0.0;
    for (int i = 0; i < 3 * patch_size * patch_size; i++) {
      dpx += tempX[i];
      dpy += tempY[i];
    }

    float det = states[patchId].H00 * states[patchId].H11
      - states[patchId].H01 * states[patchId].H01;
    states[patchId].delta_px = states[patchId].H11 * dpx - states[patchId].H01 * dpy;
    states[patchId].delta_py = states[patchId].H00 * dpy - states[patchId].H01 * dpx;
    states[patchId].delta_px /= det;
    states[patchId].delta_py /= det;

    // Update flow
    states[patchId].p_curx -= states[patchId].delta_px;
    states[patchId].p_cury -= states[patchId].delta_py;

    // Update midpoint
    states[patchId].midpoint_curx = states[patchId].midpoint_orgx 
      + states[patchId].p_curx;
    states[patchId].midpoint_cury = states[patchId].midpoint_orgy 
      + states[patchId].p_cury;

    // Outlier check
    float norm = sqrt((states[patchId].midpoint_curx - states[patchId].midpoint_orgx) 
        * (states[patchId].midpoint_curx - states[patchId].midpoint_orgx)
        + (states[patchId].midpoint_cury - states[patchId].midpoint_orgy)
        * (states[patchId].midpoint_cury - states[patchId].midpoint_orgy));

    if (norm > out_thresh || states[patchId].midpoint_curx < lb 
        || states[patchId].midpoint_cury < lb
        || states[patchId].midpoint_curx > ubw 
        || states[patchId].midpoint_cury > ubh) {

      states[patchId].p_curx = states[patchId].p_orgx;
      states[patchId].p_cury = states[patchId].p_orgy;

      // Update midpoint
      states[patchId].midpoint_curx = states[patchId].midpoint_orgx 
        + states[patchId].p_curx;
      states[patchId].midpoint_cury = states[patchId].midpoint_orgy 
        + states[patchId].p_cury;

      states[patchId].has_converged = 1;
      states[patchId].has_opt_started = 1;

    }

  }

}


__global__ void kernelInterpolateAndComputeErr(
    dev_patch_state* states, float** raw_diff, float** costs,
    float** patches, float** patchXs, float** patchYs, const float* I1,
    float** tempXX, float** tempYY,
    int n_patches, int padding, int patch_size,
    int width_pad, int gd_iter, float res_thresh, float dp_thresh,
    float dr_thresh, float out_thresh, int lb, int ubw, int ubh) {

  int patchId = blockIdx.x;
  int tid = threadIdx.x;
  float* raw = raw_diff[patchId];
  float* cost = costs[patchId];
  float* patch = patches[patchId];
  float* patchX = patchXs[patchId];
  float* patchY = patchYs[patchId];
  float* tempX = tempXX[patchId];
  float* tempY = tempYY[patchId];
  bool notFirst = false;

  while (!states[patchId].has_converged) {

    if (notFirst) {
      calcProjection(states, patchX, patchY, tempX, tempY, raw, patch_size,
          out_thresh, lb, ubw, ubh);
    }


    // Interpolate the patch

    float pos0, pos1, pos2, pos3, resid0, resid1, w0, w1, w2, w3;

    // Compute the bilinear weight vector, for patch without orientation/scale change
    // weight vector is constant for all pixels
    // TODO: compare performance when shared and only tid 0 does this precomp
    pos0 = ceil(states[patchId].midpoint_curx + .00001f); // ensure rounding up to natural numbers
    pos1 = ceil(states[patchId].midpoint_cury + .00001f);
    pos2 = floor(states[patchId].midpoint_curx);
    pos3 = floor(states[patchId].midpoint_cury);

    resid0 = states[patchId].midpoint_curx - (float)pos2;
    resid1 = states[patchId].midpoint_cury - (float)pos3;
    w0 = resid0 * resid1;
    w1 = (1 - resid0) * resid1;
    w2 = resid0 * (1- resid1);
    w3 = (1 - resid0) * (1 - resid1);

    pos0 += padding;
    pos1 += padding;

    int lb = -patch_size / 2;
    int x = 3 * (pos0 + lb) + tid;
    int starty = pos1 + lb;


    for (int i = tid, j = starty; i < patch_size * patch_size * 3;
        i += 3 * patch_size, j += 3 * width_pad) {

      const float* img_e = I1 + x;
      const float* img_a = img_e + j * width_pad * 3;
      const float* img_c = img_e + (j - 1) * width_pad * 3;
      const float* img_b = img_a - 3;
      const float* img_d = img_c - 3;
      raw[i] = w0 * (*img_a) + w1 * (*img_b) + w2 * (*img_c) + w3 * (*img_d);

    }

    // Compute mean
    __shared__ float mean;

    if (tid == 0) {

      mean = 0.0;
      for (int i = 0; i < patch_size * patch_size * 3; i++) {
        mean += raw[i];
      }
      mean /= patch_size * patch_size * 3;

    }

    __syncthreads();

    // Normalize and compute cost
    for (int i = tid; i < patch_size * patch_size * 3;
        i+= 3 * patch_size) {
      raw[i] -= mean;
      raw[i] -= patch[i];
      cost[i] = raw[i] * raw[i];
    }

    if (tid == 0) {
      float c = 0.0;
      for (int i = 0; i < patch_size * patch_size * 3; i++) {
        c += cost[i];
      }
      states[patchId].cost = c;

      // Check convergence

      // Compute step norm
      states[patchId].delta_p_sq_norm = 
        states[patchId].delta_px * states[patchId].delta_px + 
        states[patchId].delta_py * states[patchId].delta_py;

      if (states[patchId].count == 1)
        states[patchId].delta_p_sq_norm_init = states[patchId].delta_p_sq_norm;

      // Check early termination criterions
      states[patchId].mares_old = states[patchId].mares;
      states[patchId].mares = c / (3 * patch_size * patch_size);

      if (!((states[patchId].count < gd_iter) & (states[patchId].mares > res_thresh)
            & ((states[patchId].count < gd_iter) 
              | (states[patchId].delta_p_sq_norm / states[patchId].delta_p_sq_norm_init >= dp_thresh))
            & ((states[patchId].count < gd_iter) 
              | (states[patchId].mares / states[patchId].mares_old <= dr_thresh)))) {

        states[patchId].has_converged = 1;

      }

    }

    notFirst = true;

  }

}



namespace cu {

  void interpolateAndComputeErr(dev_patch_state* states,
      float** raw_diff, float** costs, float** patches, float** patchXs,
      float** patchYs, float** tempXX, float** tempYY, const float* I1,
      int n_patches, const opt_params* op,
      const img_params* i_params) {

    int nBlocks = n_patches;
    int nThreadsPerBlock = 3 * op->patch_size;

    kernelInterpolateAndComputeErr<<<nBlocks, nThreadsPerBlock>>>(
        states, raw_diff, costs, patches, patchXs, patchYs, I1,
        tempXX, tempYY, n_patches,
        i_params->padding, op->patch_size, i_params->width_pad, 
        op->grad_descent_iter, op->res_thresh, op->dp_thresh,
        op->dr_thresh, op->outlier_thresh, i_params->l_bound, i_params->u_bound_width,
        i_params->u_bound_height);


  }

}
