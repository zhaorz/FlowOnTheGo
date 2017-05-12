#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>
#include <sys/time.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/cuda_helper.h"
#include "common/timer.h"
#include "kernels/densify.h"
#include "kernels/extract.h"
#include "kernels/optimize.h"

#include <stdio.h>

#include "patch.h"
#include "patchgrid.h"


using std::cout;
using std::endl;
using std::vector;


namespace OFC {

  PatGridClass::PatGridClass(
      const img_params* _i_params,
      const opt_params* _op)
    : i_params(_i_params), op(_op) {

      // Generate grid on current scale
      steps = op->steps;
      n_patches_width = ceil((float) i_params->width /  (float) steps);
      n_patches_height = ceil((float) i_params->height / (float) steps);
      const int offsetw = floor((i_params->width - (n_patches_width - 1) * steps) / 2);
      const int offseth = floor((i_params->height - (n_patches_height - 1) * steps) / 2);

      n_patches = n_patches_width * n_patches_height;
      midpoints_ref.resize(n_patches);

      checkCudaErrors(
          cudaMalloc ((void**) &pDevicePatchStates, n_patches * sizeof(dev_patch_state)) );
      pHostDevicePatchStates = new dev_patch_state[n_patches];

      for (int x = 0; x < n_patches_width; ++x) {
        for (int y = 0; y < n_patches_height; ++y) {

          int i = x * n_patches_height + y;

          midpoints_ref[i][0] = x * steps + offsetw;
          midpoints_ref[i][1] = y * steps + offseth;

        }
      }


      // Aggregate flow
      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceWeights, i_params->width * i_params->height * sizeof(float)) );
      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceFlowOut, i_params->width * i_params->height * 2 * sizeof(float)) );

      // Patches and Hessians
      checkCudaErrors(
          cudaMalloc((void**) &pDevicePatches, n_patches * sizeof(float*)) );
      checkCudaErrors(
          cudaMalloc((void**) &pDevicePatchXs, n_patches * sizeof(float*)) );
      checkCudaErrors(
          cudaMalloc((void**) &pDevicePatchYs, n_patches * sizeof(float*)) );

      checkCudaErrors(
          cudaMalloc((void**) &pDeviceTempXX, n_patches * sizeof(float*)) );
      checkCudaErrors(
          cudaMalloc((void**) &pDeviceTempXY, n_patches * sizeof(float*)) );
      checkCudaErrors(
          cudaMalloc((void**) &pDeviceTempYY, n_patches * sizeof(float*)) );

      checkCudaErrors(
          cudaMalloc((void**) &pDeviceRaws, n_patches * sizeof(float*)) );
      checkCudaErrors(
          cudaMalloc((void**) &pDeviceCosts, n_patches * sizeof(float*)) );

      pHostDevicePatches = new float*[n_patches];
      pHostDevicePatchXs = new float*[n_patches];
      pHostDevicePatchYs = new float*[n_patches];

      pHostDeviceTempXX = new float*[n_patches];
      pHostDeviceTempXY = new float*[n_patches];
      pHostDeviceTempYY = new float*[n_patches];

      pHostDeviceRaws = new float*[n_patches];
      pHostDeviceCosts = new float*[n_patches];

      for (int i = 0; i < n_patches; i++) {

        checkCudaErrors(
            cudaMalloc((void**) &pHostDevicePatches[i], op->n_vals * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc((void**) &pHostDevicePatchXs[i], op->n_vals * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc((void**) &pHostDevicePatchYs[i], op->n_vals * sizeof(float)) );

        checkCudaErrors(
            cudaMalloc((void**) &pHostDeviceTempXX[i], op->n_vals * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc((void**) &pHostDeviceTempXY[i], op->n_vals * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc((void**) &pHostDeviceTempYY[i], op->n_vals * sizeof(float)) );

        checkCudaErrors(
            cudaMalloc((void**) &pHostDeviceRaws[i], op->n_vals * sizeof(float)) );
        checkCudaErrors(
            cudaMalloc((void**) &pHostDeviceCosts[i], op->n_vals * sizeof(float)) );

        pHostDevicePatchStates[i].has_converged = false;
        pHostDevicePatchStates[i].has_opt_started = false;
        pHostDevicePatchStates[i].H00 = 0.0;
        pHostDevicePatchStates[i].H01 = 0.0;
        pHostDevicePatchStates[i].H11 = 0.0;
        pHostDevicePatchStates[i].p_orgx = 0;
        pHostDevicePatchStates[i].p_orgy = 0;
        pHostDevicePatchStates[i].p_curx = 0;
        pHostDevicePatchStates[i].p_cury = 0;
        pHostDevicePatchStates[i].delta_px = 0;
        pHostDevicePatchStates[i].delta_py = 0;
        pHostDevicePatchStates[i].midpoint_curx = midpoints_ref[i][0];
        pHostDevicePatchStates[i].midpoint_cury = midpoints_ref[i][1];
        pHostDevicePatchStates[i].midpoint_orgx = midpoints_ref[i][0];
        pHostDevicePatchStates[i].midpoint_orgy = midpoints_ref[i][1];
        pHostDevicePatchStates[i].delta_p_sq_norm = 1e-10;
        pHostDevicePatchStates[i].delta_p_sq_norm_init = 1e-10;
        pHostDevicePatchStates[i].mares = 1e20;
        pHostDevicePatchStates[i].mares_old = 1e20;
        pHostDevicePatchStates[i].count = 0;
        pHostDevicePatchStates[i].invalid = false;
        pHostDevicePatchStates[i].cost = 0;

      }

      checkCudaErrors( cudaMemcpy(pDevicePatches, pHostDevicePatches,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDevicePatchXs, pHostDevicePatchXs,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDevicePatchYs, pHostDevicePatchYs,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );


      checkCudaErrors( cudaMemcpy(pDeviceTempXX, pHostDeviceTempXX,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDeviceTempXY, pHostDeviceTempXY,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDeviceTempYY, pHostDeviceTempYY,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );


      checkCudaErrors( cudaMemcpy(pDeviceRaws, pHostDeviceRaws,
            n_patches * sizeof(float*), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDeviceCosts, pHostDeviceCosts,
            n_patches * sizeof(float*), cudaMemcpyHostToDevice) );


      checkCudaErrors( cudaMemcpy(pDevicePatchStates, pHostDevicePatchStates,
            n_patches * sizeof(dev_patch_state), cudaMemcpyHostToDevice) );

      // Prev flow
      int flow_size = i_params->width * i_params->height / 2;
      checkCudaErrors(
          cudaMalloc ((void**) &pDevFlowPrev, flow_size * sizeof(float)) );


      // Hessian
      H00 = new float[n_patches];
      H01 = new float[n_patches];
      H11 = new float[n_patches];

      checkCudaErrors( cudaMalloc((void**) &pDeviceH00, n_patches * sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**) &pDeviceH01, n_patches * sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**) &pDeviceH11, n_patches * sizeof(float)) );

      aggregateTime = 0.0;
      meanTime = 0.0;
      extractTime = 0.0;
      optiTime = 0.0;
      coarseTime = 0.0;
    }


  PatGridClass::~PatGridClass() {

    for (int i = 0; i < n_patches; ++i) {

      cudaFree(pHostDevicePatches[i]);
      cudaFree(pHostDevicePatchXs[i]);
      cudaFree(pHostDevicePatchYs[i]);

      cudaFree(pHostDeviceTempXX[i]);
      cudaFree(pHostDeviceTempXY[i]);
      cudaFree(pHostDeviceTempYY[i]);

      cudaFree(pHostDeviceRaws[i]);
      cudaFree(pHostDeviceCosts[i]);

    }

    cudaFree(pDevicePatches);
    cudaFree(pDevicePatchXs);
    cudaFree(pDevicePatchYs);

    cudaFree(pDeviceRaws);
    cudaFree(pDeviceCosts);

    delete pHostDeviceRaws;
    delete pHostDeviceCosts;

    delete pHostDevicePatches;
    delete pHostDevicePatchXs;
    delete pHostDevicePatchYs;

    delete pHostDeviceTempXX;
    delete pHostDeviceTempXY;
    delete pHostDeviceTempYY;

    cudaFree(pDevFlowPrev);

    cudaFree(pDeviceH00);
    cudaFree(pDeviceH01);
    cudaFree(pDeviceH11);

    delete H00;
    delete H01;
    delete H11;

    cudaFree(pDeviceTempXX);
    cudaFree(pDeviceTempXY);
    cudaFree(pDeviceTempYY);

  }


  void PatGridClass::InitializeGrid(const float * _I0, const float * _I0x, const float * _I0y) {

    I0 = _I0;
    I0x = _I0x;
    I0y = _I0y;

    gettimeofday(&tv_start, nullptr);

    cu::extractPatchesAndHessians(pDevicePatches, pDevicePatchXs, pDevicePatchYs,
        I0, I0x, I0y, pDeviceTempXX, pDeviceTempXY, pDeviceTempYY,
        pDevicePatchStates, n_patches, op, i_params);

    gettimeofday(&tv_end, nullptr);
    extractTime = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;

  }


  void PatGridClass::SetTargetImage(const float * _I1) {

    I1 = _I1;

  }


  void PatGridClass::Optimize() {

    gettimeofday(&tv_start, nullptr);

    cu::interpolateAndComputeErr(pDevicePatchStates, pDeviceRaws, pDeviceCosts,
        pDevicePatches, pDevicePatchXs, pDevicePatchYs, pDeviceTempXX, pDeviceTempYY,
        I1, n_patches, op, i_params, true);

    gettimeofday(&tv_end, nullptr);
    optiTime = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
  }


  void PatGridClass::InitializeFromCoarserOF(float * flow_prev) {

    gettimeofday(&tv_start, nullptr);

    cu::initCoarserOF(flow_prev, pDevicePatchStates,
        n_patches, i_params);

    gettimeofday(&tv_end, nullptr);
    coarseTime = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;

  }


  void PatGridClass::AggregateFlowDense(float *flowout) {

    gettimeofday(&tv_start, nullptr);

    // Device mem
    checkCudaErrors(
        cudaMemset (pDeviceWeights, 0.0, i_params->width * i_params->height * sizeof(float)) );
    checkCudaErrors(
        cudaMemset (pDeviceFlowOut, 0.0, i_params->width * i_params->height * 2 * sizeof(float)) );

    cu::densifyPatches(
        pDeviceCosts, pDeviceFlowOut, pDeviceWeights,
        pDevicePatchStates, n_patches, op, i_params);

    gettimeofday(&tv_end, nullptr);
    aggregateTime = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;

    gettimeofday(&tv_start, nullptr);

    // Normalize all pixels
    cu::normalizeFlow(flowout, pDeviceFlowOut, pDeviceWeights, 2 * i_params->width * i_params->height);

    gettimeofday(&tv_end, nullptr);
    meanTime = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;


  }


  void PatGridClass::printTimings() {

    cout << endl;
    cout << "===============Timings (ms)===============" << endl;
    cout << "[extract]      " << extractTime << endl;
    cout << "[coarse]       " << coarseTime << endl;
    cout << "[optiTime]      " << optiTime << endl;
    cout << "[aggregate]    " << aggregateTime << endl;
    cout << "[flow norm]    " << meanTime << endl;
    cout << "==========================================" << endl;

  }

}
