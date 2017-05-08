#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/cuda_helper.h"
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

  void PatGridClass::AggregateFlowDense(float *flowout) const {

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

  }

}
