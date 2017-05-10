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
      p_init.resize(n_patches);
      patches.reserve(n_patches);

      midpointX_host = new float[n_patches];
      midpointY_host = new float[n_patches];

      int patch_id = 0;
      for (int x = 0; x < n_patches_width; ++x) {
        for (int y = 0; y < n_patches_height; ++y) {

          int i = x * n_patches_height + y;

          midpoints_ref[i][0] = x * steps + offsetw;
          midpoints_ref[i][1] = y * steps + offseth;
          midpointX_host[i] = x * steps + offsetw;
          midpointY_host[i] = y * steps + offseth;
          p_init[i].setZero();

          patches.push_back(new OFC::PatClass(i_params, op, patch_id));
          patch_id++;

        }
      }

      // Midpoint
      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceMidpointX, n_patches * sizeof(float)) );
      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceMidpointY, n_patches * sizeof(float)) );
      checkCudaErrors( cudaMemcpy(pDeviceMidpointX, midpointX_host,
          n_patches * sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(pDeviceMidpointY, midpointY_host,
          n_patches * sizeof(float), cudaMemcpyHostToDevice) );

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

      pHostDevicePatches = new float*[n_patches];
      pHostDevicePatchXs = new float*[n_patches];
      pHostDevicePatchYs = new float*[n_patches];

      float* pHostDeviceTempXX[n_patches];
      float* pHostDeviceTempXY[n_patches];
      float* pHostDeviceTempYY[n_patches];

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
    }

  PatGridClass::~PatGridClass() {

    for (int i = 0; i < n_patches; ++i) {
      cudaFree(pDevicePatches[i]);
      cudaFree(pDevicePatchXs[i]);
      cudaFree(pDevicePatchYs[i]);

      cudaFree(pDeviceTempXX[i]);
      cudaFree(pDeviceTempXY[i]);
      cudaFree(pDeviceTempYY[i]);

      delete patches[i];
    }

    cudaFree(pDevicePatches);
    cudaFree(pDevicePatchXs);
    cudaFree(pDevicePatchYs);

    delete pHostDevicePatches;
    delete pHostDevicePatchXs;
    delete pHostDevicePatchYs;

    delete midpointX_host;
    delete midpointY_host;
    cudaFree(pDeviceMidpointX);
    cudaFree(pDeviceMidpointY);

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
        I0, I0x, I0y, pDeviceH00, pDeviceH01, pDeviceH11,
        pDeviceTempXX, pDeviceTempXY, pDeviceTempYY,
        pDeviceMidpointX, pDeviceMidpointY, n_patches, op, i_params);

    checkCudaErrors(
        cudaMemcpy(H00, pDeviceH00, n_patches * sizeof(float), cudaMemcpyDeviceToHost) );
    checkCudaErrors(
        cudaMemcpy(H01, pDeviceH01, n_patches * sizeof(float), cudaMemcpyDeviceToHost) );
    checkCudaErrors(
        cudaMemcpy(H11, pDeviceH11, n_patches * sizeof(float), cudaMemcpyDeviceToHost) );

    gettimeofday(&tv_end, nullptr);
    extractTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;

    for (int i = 0; i < n_patches; ++i) {
      patches[i]->InitializePatch(pHostDevicePatches[i],
          pHostDevicePatchXs[i], pHostDevicePatchYs[i],
          H00[i], H01[i], H11[i], midpoints_ref[i]);
      p_init[i].setZero();
    }

  }

  void PatGridClass::SetTargetImage(const float * _I1) {

    I1 = _I1;

    for (int i = 0; i < n_patches; ++i) {
      patches[i]->SetTargetImage(I1);
    }

  }

  void PatGridClass::Optimize() {

// #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_patches; ++i) {
      patches[i]->OptimizeIter(p_init[i]);
    }

  }

  void PatGridClass::InitializeFromCoarserOF(const float * flow_prev) {

    for (int ip = 0; ip < n_patches; ++ip) {

      int x = floor(midpoints_ref[ip][0] / 2);
      int y = floor(midpoints_ref[ip][1] / 2);
      int i = y * (i_params->width / 2) + x;

      p_init[ip](0) = flow_prev[2 * i] * 2;
      p_init[ip](1) = flow_prev[2 * i + 1] * 2;

    }

  }

  void PatGridClass::AggregateFlowDense(float *flowout) {

    bool isValid[n_patches];
    float flowXs[n_patches];
    float flowYs[n_patches];
    float* costs[n_patches];

    for (int i = 0; i < n_patches; i++) {
      isValid[i] = patches[i]->IsValid();
      flowXs[i] = (*(patches[i]->GetCurP()))[0];
      flowYs[i] = (*(patches[i]->GetCurP()))[1];
      costs[i] = patches[i]->GetDeviceCostDiffPtr();
    }

    bool *deviceIsValid;
    float* deviceFlowXs, * deviceFlowYs;
    float** deviceCosts;

    checkCudaErrors(
          cudaMalloc ((void**) &deviceIsValid, n_patches * sizeof(bool)) );
    checkCudaErrors(
          cudaMalloc ((void**) &deviceFlowXs, n_patches * sizeof(float)) );
    checkCudaErrors(
          cudaMalloc ((void**) &deviceFlowYs, n_patches * sizeof(float)) );
    checkCudaErrors(
          cudaMalloc ((void**) &deviceCosts, n_patches * sizeof(float*)) );

    checkCudaErrors( cudaMemcpy(deviceIsValid, isValid,
          n_patches * sizeof(bool), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(deviceFlowXs, flowXs,
          n_patches * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(deviceFlowYs, flowYs,
          n_patches * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(deviceCosts, costs,
          n_patches * sizeof(float*), cudaMemcpyHostToDevice) );


    gettimeofday(&tv_start, nullptr);

    // Device mem
    checkCudaErrors(
        cudaMemset (pDeviceWeights, 0.0, i_params->width * i_params->height * sizeof(float)) );
    checkCudaErrors(
        cudaMemset (pDeviceFlowOut, 0.0, i_params->width * i_params->height * 2 * sizeof(float)) );

    cu::densifyPatches(
        deviceCosts, pDeviceFlowOut, pDeviceWeights,
        deviceFlowXs, deviceFlowYs, deviceIsValid,
        pDeviceMidpointX, pDeviceMidpointY, n_patches,
        op, i_params);
    /*for (int ip = 0; ip < n_patches; ++ip) {
      if (patches[ip]->IsValid()) {

        const Eigen::Vector2f* fl = patches[ip]->GetCurP(); // flow displacement of this patch

        float* pweight = patches[ip]->GetDeviceCostDiffPtr(); // use image error as weight

        cu::densifyPatch(
            pweight, pDeviceFlowOut, pDeviceWeights,
            (*fl)[0], (*fl)[1],
            midpoints_ref[ip][0], midpoints_ref[ip][1],
            i_params->width, i_params->height,
            op->patch_size, op->min_errval);

      }
    }*/

    gettimeofday(&tv_end, nullptr);
    aggregateTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;

    gettimeofday(&tv_start, nullptr);
    // Normalize all pixels
    cu::normalizeFlow(pDeviceFlowOut, pDeviceWeights, i_params->width * i_params->height);

    checkCudaErrors(
        cudaMemcpy(flowout, pDeviceFlowOut,
          i_params->width * i_params->height * 2 * sizeof(float), cudaMemcpyDeviceToHost) );

    gettimeofday(&tv_end, nullptr);
    meanTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
  }

  void PatGridClass::printTimings() {

    double tot_hessianTime = 0,
           tot_projectionTime = 0, tot_costTime = 0,
           tot_interpolateTime = 0, tot_meanTime = 0;
    int tot_hessianCalls = 0,
        tot_projectionCalls = 0, tot_costCalls = 0,
        tot_interpolateCalls = 0, tot_meanCalls = 0;

    for (auto & element : patches) {
      tot_hessianTime += element->hessianTime;
      tot_projectionTime += element->projectionTime;
      tot_costTime += element->costTime;
      tot_interpolateTime += element->interpolateTime;
      tot_meanTime += element->meanTime;

      tot_hessianCalls += element->hessianCalls;
      tot_projectionCalls += element->projectionCalls;
      tot_costCalls += element->costCalls;
      tot_interpolateCalls += element->interpolateCalls;
      tot_meanCalls += element->meanCalls;
    }

    cout << endl;
    cout << "===============Timings (ms)===============" << endl;
    cout << "Avg grad descent iterations:        " << float(tot_costCalls) / float(n_patches) << endl;
    cout << "[hessian]      " << tot_hessianTime;
    cout << "  tot => " << tot_hessianTime / tot_hessianCalls << " avg" << endl;
    cout << "[project]      " << tot_projectionTime;
    cout << "  tot => " << tot_projectionTime / tot_projectionCalls << " avg" << endl;
    cout << "[cost]         " << tot_costTime;
    cout << "  tot => " << tot_costTime / tot_costCalls << " avg" << endl;
    cout << "[interpolate]  " << tot_interpolateTime;
    cout << "  tot => " << tot_interpolateTime / tot_interpolateCalls << " avg" << endl;
    cout << "[mean norm]    " << tot_meanTime;
    cout << "  tot => " << tot_meanTime / tot_meanCalls << " avg" << endl;
    cout << "[extract]      " << extractTime << endl;
    cout << "[aggregate]    " << aggregateTime << endl;
    cout << "[flow norm]    " << meanTime << endl;
    cout << "==========================================" << endl;


  }

}
