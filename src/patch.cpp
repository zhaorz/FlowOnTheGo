#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common/cuda_helper.h"
#include "common/Exceptions.h"
#include "common/timer.h"

#include "patch.h"
#include "kernels/interpolate.h"
#include "kernels/extract.h"


using std::cout;
using std::endl;
using std::vector;

namespace OFC {


  PatClass::PatClass(
      const img_params* _i_params,
      const opt_params* _op,
      const int _patch_id)
    :
      i_params(_i_params),
      op(_op),
      patch_id(_patch_id) {

        p_state = new patch_state;

        patch.resize(op->n_vals,1);
        patch_x.resize(op->n_vals,1);
        patch_y.resize(op->n_vals,1);

        /*checkCudaErrors(
            cudaMalloc ((void**) &pDevicePatch, patch.size() * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc ((void**) &pDevicePatchX, patch_x.size() * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc ((void**) &pDevicePatchY, patch_y.size() * sizeof(float)) );*/
        /*checkCudaErrors(
            cudaMalloc ((void**) &pDeviceRawDiff, patch.size() * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc ((void**) &pDeviceCostDiff, patch.size() * sizeof(float)) );*/
        checkCudaErrors(
            cudaMalloc ((void**) &pDeviceWeights, 4 * sizeof(float)) );


        // Timing
        hessianTime = 0;
        projectionTime = 0;
        costTime = 0;
        interpolateTime = 0;
        meanTime = 0;

        hessianCalls = 0;
        projectionCalls = 0;
        costCalls = 0;
        interpolateCalls = 0;
        meanCalls = 0;

      }


  PatClass::~PatClass() {

    /*cudaFree(pDevicePatch);
    cudaFree(pDevicePatchX);
    cudaFree(pDevicePatchY);*/

    // cudaFree(pDeviceRawDiff);
    // cudaFree(pDeviceCostDiff);
    cudaFree(pDeviceWeights);

    delete p_state;

  }

  // void PatClass::InitializePatch(const float * _I0,
  //     const float * _I0x, const float * _I0y, const Eigen::Vector2f _midpoint) {
  void PatClass::InitializePatch(float * _patch,
      float * _patchx, float * _patchy, float H00, float H01, float H11,
      const Eigen::Vector2f _midpoint) {

    // I0 = _I0;
    // I0x = _I0x;
    // I0y = _I0y;

    pDevicePatch = _patch;
    pDevicePatchX = _patchx;
    pDevicePatchY = _patchy;

    midpoint = _midpoint;

    ResetPatchState();

    p_state->hessian(0,0) = H00;
    p_state->hessian(0,1) = H01;
    p_state->hessian(1,0) = p_state->hessian(0,1);
    p_state->hessian(1,1) = H11;

    //ExtractPatch();
    // ComputeHessian(H00, H01, H11);

  }

  void PatClass::ComputeHessian(float H00, float H01, float H11) {

    /*gettimeofday(&tv_start, nullptr);

    CUBLAS_CHECK (
        cublasSdot(op->cublasHandle, patch.size(),
          pDevicePatchX, 1, pDevicePatchX, 1, &(p_state->hessian(0,0))) );
    CUBLAS_CHECK (
        cublasSdot(op->cublasHandle, patch.size(),
          pDevicePatchX, 1, pDevicePatchY, 1, &(p_state->hessian(0,1))) );
    CUBLAS_CHECK (
        cublasSdot(op->cublasHandle, patch.size(),
          pDevicePatchY, 1, pDevicePatchY, 1, &(p_state->hessian(1,1))) );

    p_state->hessian(1,0) = p_state->hessian(0,1);

    gettimeofday(&tv_end, nullptr);

    hessianTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
    hessianCalls++;*/

    p_state->hessian(0,0) = H00;
    p_state->hessian(0,1) = H01;
    p_state->hessian(1,0) = p_state->hessian(0,1);
    p_state->hessian(1,1) = H11;

    // If not invertible adjust values
    if (p_state->hessian.determinant() == 0) {
      p_state->hessian(0,0) += 1e-10;
      p_state->hessian(1,1) += 1e-10;
    }

  }

  void PatClass::SetTargetImage(const float * _I1) {

    pDeviceI = _I1;

    ResetPatchState();

  }

  void PatClass::ResetPatchState() {

    p_state->has_converged = 0;
    p_state->has_opt_started = 0;

    p_state->midpoint_org = midpoint;
    p_state->midpoint_cur = midpoint;

    p_state->p_org.setZero();
    p_state->p_cur.setZero();
    p_state->delta_p.setZero();

    p_state->delta_p_sq_norm = 1e-10;
    p_state->delta_p_sq_norm_init = 1e-10;
    p_state->mares = 1e20;
    p_state->mares_old = 1e20;
    p_state->count = 0;
    p_state->invalid = false;

    p_state->cost = 0.0;
  }

  void PatClass::OptimizeStart(const Eigen::Vector2f p_prev, bool conv) {

    p_state->p_org = p_prev;
    p_state->p_cur = p_prev;

    UpdateMidpoint();

    // save starting location, only needed for outlier check
    p_state->midpoint_org = p_state->midpoint_cur;

    //Check if initial position is already invalid
    if (p_state->midpoint_cur[0] < i_params->l_bound
        || p_state->midpoint_cur[1] < i_params->l_bound
        || p_state->midpoint_cur[0] > i_params->u_bound_width
        || p_state->midpoint_cur[1] > i_params->u_bound_height) {

      p_state->has_converged=1;
      p_state->has_opt_started=1;

    } else {

      p_state->count = 0; // reset iteration counter
      p_state->delta_p_sq_norm = 1e-10;
      p_state->delta_p_sq_norm_init = 1e-10;  // set to arbitrary low value, s.t. that loop condition is definitely true on first iteration
      p_state->mares = 1e5;          // mean absolute residual
      p_state->mares_old = 1e20; // for rate of change, keep mares from last iteration in here. Set high so that loop condition is definitely true on first iteration
      p_state->has_converged=0;

     //  OptimizeComputeErrImg(false, c);
      p_state->has_converged = conv;

      p_state->has_opt_started = 1;
      p_state->invalid = false;

    }

  }

  void PatClass::OptimizeIter() {

    // Do one optimize iteration

    if (!p_state->has_converged) {

      p_state->count++;

      // Projection onto sd_images
      gettimeofday(&tv_start, nullptr);
      CUBLAS_CHECK (
          cublasSdot(op->cublasHandle, patch.size(),
            pDevicePatchX, 1, pDeviceRawDiff, 1, &(p_state->delta_p[0])) );
      CUBLAS_CHECK (
          cublasSdot(op->cublasHandle, patch.size(),
            pDevicePatchY, 1, pDeviceRawDiff, 1, &(p_state->delta_p[1])) );
      gettimeofday(&tv_end, nullptr);
      projectionTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
        (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
      projectionCalls++;


      p_state->delta_p = p_state->hessian.llt().solve(p_state->delta_p); // solve linear system

      p_state->p_cur -= p_state->delta_p; // update flow vector

      // compute patch locations based on new parameter vector
      UpdateMidpoint();

      // check if patch(es) moved too far from starting location, if yes, stop iteration and reset to starting location
      // check if query patch moved more than >padval from starting location -> most likely outlier
      if ((p_state->midpoint_org - p_state->midpoint_cur).norm() > op->outlier_thresh
          || p_state->midpoint_cur[0] < i_params->l_bound
          || p_state->midpoint_cur[1] < i_params->l_bound
          || p_state->midpoint_cur[0] > i_params->u_bound_width
          || p_state->midpoint_cur[1] > i_params->u_bound_height) {

        // Reset because this is an outlier
        p_state->p_cur = p_state->p_org;
        UpdateMidpoint();
        p_state->has_converged=1;
        p_state->has_opt_started=1;

      }

      OptimizeComputeErrImg(true, -0.0);

    }

  }

  inline void PatClass::UpdateMidpoint() {

    p_state->midpoint_cur = midpoint + p_state->p_cur;

  }

  void PatClass::ComputeCostErr() {

    // L2-Norm

    const float alpha = -1.0;

    gettimeofday(&tv_start, nullptr);
    // raw = raw - patch
    CUBLAS_CHECK (
        cublasSaxpy(op->cublasHandle, patch.size(), &alpha,
          pDevicePatch, 1, pDeviceRawDiff, 1) );

    // Element-wise multiplication
    CUBLAS_CHECK (
        cublasSdgmm(op->cublasHandle, CUBLAS_SIDE_RIGHT,
          1, patch.size(), pDeviceRawDiff, 1, pDeviceRawDiff, 1, pDeviceCostDiff, 1) );

    // Sum
    CUBLAS_CHECK (
        cublasSasum(op->cublasHandle, patch.size(),
          pDeviceCostDiff, 1, &(p_state->cost)) );
    gettimeofday(&tv_end, nullptr);

    costTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
    costCalls++;

  }

  void PatClass::OptimizeComputeErrImg(bool interp, float c) {

    if (interp) {
      InterpolatePatch();
      ComputeCostErr();
    } else {
      p_state->cost = c;
    }

    // Compute step norm
    p_state->delta_p_sq_norm = p_state->delta_p.squaredNorm();
    if (p_state->count == 1)
      p_state->delta_p_sq_norm_init = p_state->delta_p_sq_norm;

    // Check early termination criterions
    p_state->mares_old = p_state->mares;
    p_state->mares = p_state->cost / op->n_vals;

    if (!((p_state->count < op->grad_descent_iter) & (p_state->mares > op->res_thresh)
          & ((p_state->count < op->grad_descent_iter) | (p_state->delta_p_sq_norm / p_state->delta_p_sq_norm_init >= op->dp_thresh))
          & ((p_state->count < op->grad_descent_iter) | (p_state->mares / p_state->mares_old <= op->dr_thresh)))) {
      p_state->has_converged = 1;
    }

  }

  // Extract patch on integer position, and gradients, No Bilinear interpolation
  void PatClass::ExtractPatch() {

    int x = round(midpoint[0]) + i_params->padding;
    int y = round(midpoint[1]) + i_params->padding;

    int lb = -op->patch_size / 2;
    int patch_offset = (x + lb) + (y + lb) * i_params->width_pad;

    // Extract patch
    /*cu::extractPatch(pDevicePatch, pDevicePatchX, pDevicePatchY,
        I0, I0x, I0y, patch_offset, op->patch_size, i_params->width_pad);*/

    gettimeofday(&tv_start, nullptr);
    // Mean Normalization
    if (op->use_mean_normalization > 0) {
      cu::normalizeMean(pDevicePatch, op->patch_size);
    }

    gettimeofday(&tv_end, nullptr);
    meanTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
    meanCalls++;



  }

  // Extract patch on float position with bilinear interpolation, no gradients.
  void PatClass::InterpolatePatch() {

    Eigen::Vector2f resid;
    Eigen::Vector4f weight; // bilinear weight vector
    Eigen::Vector4i pos;
    Eigen::Vector2i pos_iter;

    // Compute the bilinear weight vector, for patch without orientation/scale change
    // weight vector is constant for all pixels
    pos[0] = ceil(p_state->midpoint_cur[0] + .00001f); // ensure rounding up to natural numbers
    pos[1] = ceil(p_state->midpoint_cur[1] + .00001f);
    pos[2] = floor(p_state->midpoint_cur[0]);
    pos[3] = floor(p_state->midpoint_cur[1]);

    resid[0] = p_state->midpoint_cur[0] - (float)pos[2];
    resid[1] = p_state->midpoint_cur[1] - (float)pos[3];
    weight[0] = resid[0] * resid[1];
    weight[1] = (1 - resid[0]) * resid[1];
    weight[2] = resid[0] * (1- resid[1]);
    weight[3] = (1 - resid[0]) * (1 - resid[1]);

    pos[0] += i_params->padding;
    pos[1] += i_params->padding;

    int lb = -op->patch_size / 2;
    int startx = pos[0] + lb;
    int starty = pos[1] + lb;

    gettimeofday(&tv_start, nullptr);
    checkCudaErrors(
        cudaMemcpy(pDeviceWeights, weight.data(),
          4 * sizeof(float), cudaMemcpyHostToDevice) );


    cu::interpolatePatch(pDeviceRawDiff, pDeviceI, pDeviceWeights,
        i_params->width_pad, starty, startx, op->patch_size);

    gettimeofday(&tv_end, nullptr);
    interpolateTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
    interpolateCalls++;

    gettimeofday(&tv_start, nullptr);
    // Mean Normalization
    if (op->use_mean_normalization > 0) {
      cu::normalizeMean(pDeviceRawDiff, op->patch_size);
    }

    gettimeofday(&tv_end, nullptr);
    meanTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
    meanCalls++;


  }

}
