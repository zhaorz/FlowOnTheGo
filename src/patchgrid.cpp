#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

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

    I0_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);
    I0x_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);
    I0y_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);

    I1_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);
    I1x_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);
    I1y_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, i_params->height, i_params->width);

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

  }

  PatGridClass::~PatGridClass() {

    delete I0_eg;
    delete I0x_eg;
    delete I0y_eg;

    delete I1_eg;
    delete I1x_eg;
    delete I1y_eg;

    for (int i = 0; i < n_patches; ++i)
      delete patches[i];

  }

  void PatGridClass::InitializeGrid(const float * _I0, const float * _I0x, const float * _I0y) {

    I0 = _I0;
    I0x = _I0x;
    I0y = _I0y;

    new (I0_eg) Eigen::Map<const Eigen::MatrixXf>(I0, i_params->height, i_params->width);
    new (I0x_eg) Eigen::Map<const Eigen::MatrixXf>(I0x, i_params->height, i_params->width);
    new (I0y_eg) Eigen::Map<const Eigen::MatrixXf>(I0y, i_params->height, i_params->width);

    for (int i = 0; i < n_patches; ++i) {
      patches[i]->InitializePatch(I0_eg, I0x_eg, I0y_eg, midpoints_ref[i]);
      p_init[i].setZero();
    }

  }

  void PatGridClass::SetTargetImage(const float * _I1, const float * _I1x, const float * _I1y) {

    I1 = _I1;
    I1x = _I1x;
    I1y = _I1y;

    new (I1_eg) Eigen::Map<const Eigen::MatrixXf>(I1, i_params->height, i_params->width);
    new (I1x_eg) Eigen::Map<const Eigen::MatrixXf>(I1x, i_params->height, i_params->width);
    new (I1y_eg) Eigen::Map<const Eigen::MatrixXf>(I1y, i_params->height, i_params->width);

    for (int i = 0; i < n_patches; ++i) {
      patches[i]->SetTargetImage(I1_eg, I1x_eg, I1y_eg);
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

    float* weights = new float[i_params->width * i_params->height];

    memset(flowout, 0, sizeof(float) * (2 * i_params->width * i_params->height));
    memset(weights, 0, sizeof(float) * (i_params->width * i_params->height));

    for (int ip = 0; ip < n_patches; ++ip) {

      if (patches[ip]->IsValid()) {

        const Eigen::Vector2f* fl = patches[ip]->GetCurP(); // flow displacement of this patch
        Eigen::Vector2f flnew;

        const float * pweight = patches[ip]->GetCostDiffPtr(); // use image error as weight

        int lower_bound = -op->patch_size / 2;
        int upper_bound = op->patch_size / 2 - 1;

        for (int y = lower_bound; y <= upper_bound; ++y) {
          for (int x = lower_bound; x <= upper_bound; ++x, ++pweight) {

            int yt = (y + midpoints_ref[ip][1]);
            int xt = (x + midpoints_ref[ip][0]);

            if (xt >= 0 && yt >= 0 && xt < i_params->width && yt < i_params->height) {

              int i = yt * i_params->width + xt;

              // Weight contribution
              float absw = 1.0f /  (float)(std::max(op->min_errval, *pweight));

              flnew = (*fl) * absw;
              weights[i] += absw;

              flowout[2 * i] += flnew[0];
              flowout[2 * i + 1] += flnew[1];
            }

          }
        }

      }

    }

    // normalize each pixel by dividing displacement by aggregated weights from all patches
    for (int yi = 0; yi < i_params->height; ++yi) {
      for (int xi = 0; xi < i_params->width; ++xi) {

        int i = yi * i_params->width + xi;
        if (weights[i] > 0) {
          flowout[2 * i] /= weights[i];
          flowout[2 * i + 1] /= weights[i];
        }

      }
    }

    delete[] weights;
  }

}
