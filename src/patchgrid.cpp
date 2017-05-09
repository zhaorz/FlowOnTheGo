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

      int patch_id = 0;
      for (int x = 0; x < n_patches_width; ++x) {
        for (int y = 0; y < n_patches_height; ++y) {

          int i = x * n_patches_height + y;

          midpoints_ref[i][0] = x * steps + offsetw;
          midpoints_ref[i][1] = y * steps + offseth;
          p_init[i].setZero();

          patches.push_back(new OFC::PatClass(i_params, op, patch_id));
          patch_id++;

        }
      }

      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceWeights, i_params->width * i_params->height * sizeof(float)) );
      checkCudaErrors(
          cudaMalloc ((void**) &pDeviceFlowOut, i_params->width * i_params->height * 2 * sizeof(float)) );

      aggregateTime = 0.0;
    }

  PatGridClass::~PatGridClass() {

    for (int i = 0; i < n_patches; ++i)
      delete patches[i];

  }

  void PatGridClass::InitializeGrid(const float * _I0, const float * _I0x, const float * _I0y) {

    I0 = _I0;
    I0x = _I0x;
    I0y = _I0y;

    for (int i = 0; i < n_patches; ++i) {
      patches[i]->InitializePatch(I0, I0x, I0y, midpoints_ref[i]);
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

    gettimeofday(&tv_start, nullptr);
    // Device mem
    checkCudaErrors(
        cudaMemset (pDeviceWeights, 0.0, i_params->width * i_params->height * sizeof(float)) );
    checkCudaErrors(
        cudaMemset (pDeviceFlowOut, 0.0, i_params->width * i_params->height * 2 * sizeof(float)) );

    for (int ip = 0; ip < n_patches; ++ip) {
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
    }

    // Normalize all pixels
    cu::normalizeFlow(pDeviceFlowOut, pDeviceWeights, i_params->width * i_params->height);

    checkCudaErrors(
        cudaMemcpy(flowout, pDeviceFlowOut,
          i_params->width * i_params->height * 2 * sizeof(float), cudaMemcpyDeviceToHost) );

    gettimeofday(&tv_end, nullptr);
    aggregateTime += (tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
      (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f;
  }

  void PatGridClass::printTimings() {

    double tot_extractTime = 0, tot_hessianTime = 0,
           tot_projectionTime = 0, tot_costTime = 0, tot_interpolateTime = 0;
    int tot_extractCalls = 0, tot_hessianCalls = 0,
        tot_projectionCalls = 0, tot_costCalls = 0, tot_interpolateCalls = 0;

    for (auto & element : patches) {
      tot_extractTime += element->extractTime;
      tot_hessianTime += element->hessianTime;
      tot_projectionTime += element->projectionTime;
      tot_costTime += element->costTime;
      tot_interpolateTime += element->interpolateTime;

      tot_extractCalls += element->extractCalls;
      tot_hessianCalls += element->hessianCalls;
      tot_projectionCalls += element->projectionCalls;
      tot_costCalls += element->costCalls;
      tot_interpolateCalls += element->interpolateCalls;
    }

    cout << endl;
    cout << "===============Timings (ms)===============" << endl;
    cout << "[extract]      " << tot_extractTime;
    cout << "  tot => " << tot_extractTime / tot_extractCalls << " avg" << endl;
    cout << "[hessian]      " << tot_hessianTime;
    cout << "  tot => " << tot_hessianTime / tot_hessianCalls << " avg" << endl;
    cout << "[project]      " << tot_projectionTime;
    cout << "  tot => " << tot_projectionTime / tot_projectionCalls << " avg" << endl;
    cout << "[cost]         " << tot_costTime;
    cout << "  tot => " << tot_costTime / tot_costCalls << " avg" << endl;
    cout << "[interpolate]  " << tot_interpolateTime;
    cout << "  tot => " << tot_interpolateTime / tot_interpolateCalls << " avg" << endl;
    cout << "[aggregate]    " << aggregateTime << endl;
    cout << "==========================================" << endl;


  }

}
