#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>

#include "patch.h"

using std::cout;
using std::endl;
using std::vector;

namespace OFC {

  typedef __v4sf v4sf;

  PatClass::PatClass(
      const img_params* _i_params,
      const opt_params* _op,
      const int _patch_id)
    :
      i_params(_i_params),
      op(_op),
      patch_id(_patch_id) {

        p_state = new patch_state;
        InitializeError();

        patch.resize(op->n_vals,1);
        patch_x.resize(op->n_vals,1);
        patch_y.resize(op->n_vals,1);

      }

  void PatClass::InitializeError() {

    p_state->raw_diff.resize(op->n_vals,1);
    p_state->cost_diff.resize(op->n_vals,1);

  }

  PatClass::~PatClass() {

    delete p_state;

  }

  void PatClass::InitializePatch(Eigen::Map<const Eigen::MatrixXf> * _I0,
      Eigen::Map<const Eigen::MatrixXf> * _I0x,
      Eigen::Map<const Eigen::MatrixXf> * _I0y, const Eigen::Vector2f _midpoint) {

    I0 = _I0;
    I0x = _I0x;
    I0y = _I0y;
    midpoint = _midpoint;

    ResetPatchState();
    ExtractPatch();
    ComputeHessian();

  }

  void PatClass::ComputeHessian() {

    p_state->hessian(0,0) = (patch_x.array() * patch_x.array()).sum();
    p_state->hessian(0,1) = (patch_x.array() * patch_y.array()).sum();
    p_state->hessian(1,1) = (patch_y.array() * patch_y.array()).sum();
    p_state->hessian(1,0) = p_state->hessian(0,1);

    // If not invertible adjust values
    if (p_state->hessian.determinant() == 0) {
      p_state->hessian(0,0) += 1e-10;
      p_state->hessian(1,1) += 1e-10;
    }

  }

  void PatClass::SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * _I1,
      Eigen::Map<const Eigen::MatrixXf> * _I1x,
      Eigen::Map<const Eigen::MatrixXf> * _I1y) {

    I1 = _I1;
    I1x = _I1x;
    I1y = _I1y;

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

  }

  void PatClass::OptimizeStart(const Eigen::Vector2f p_prev) {

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
      p_state->raw_diff = patch;
      p_state->has_opt_started=1;

    } else {

      p_state->count = 0; // reset iteration counter
      p_state->delta_p_sq_norm = 1e-10;
      p_state->delta_p_sq_norm_init = 1e-10;  // set to arbitrary low value, s.t. that loop condition is definitely true on first iteration
      p_state->mares = 1e5;          // mean absolute residual
      p_state->mares_old = 1e20; // for rate of change, keep mares from last iteration in here. Set high so that loop condition is definitely true on first iteration
      p_state->has_converged=0;

      OptimizeComputeErrImg();

      p_state->has_opt_started = 1;
      p_state->invalid = false;

    }

  }

  void PatClass::OptimizeIter(const Eigen::Vector2f p_prev) {

    if (!p_state->has_opt_started) {

      ResetPatchState();
      OptimizeStart(p_prev);

    }

    // optimize patch until convergence, or do only one iteration if DIS visualization is used
    while (!p_state->has_converged) {

      p_state->count++;

      // Projection onto sd_images
      p_state->delta_p[0] = (patch_x.array() * p_state->raw_diff.array()).sum();
      p_state->delta_p[1] = (patch_y.array() * p_state->raw_diff.array()).sum();

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

      OptimizeComputeErrImg();

    }

  }

  inline void PatClass::UpdateMidpoint() {

    p_state->midpoint_cur = midpoint + p_state->p_cur;

  }

  void PatClass::ComputeCostErr() {

    v4sf * raw = (v4sf*) p_state->raw_diff.data(),
         * img = (v4sf*) p_state->raw_diff.data(),
         * templ = (v4sf*) patch.data(),
         * cost = (v4sf*) p_state->cost_diff.data();

    switch (op->cost_func) {
      case 1:
        // L1-Norm

        for (int i = op->n_vals / 4; i--; ++raw, ++img, ++templ, ++cost) {
          (*raw) = (*img) - (*templ);   // difference image
          // sign(raw_diff) * sqrt(abs(raw_diff))
          (*raw) = __builtin_ia32_orps(__builtin_ia32_andps(op->negzero, (*raw)),
              __builtin_ia32_sqrtps(__builtin_ia32_andnps(op->negzero, (*raw))));
          (*cost) = __builtin_ia32_andnps(op->negzero,  (*raw));
        }

        break;
      case 2:
        // Pseudo Huber cost function

        for (int i = op->n_vals / 4; i--; ++raw, ++img, ++templ, ++cost) {
          (*raw) = (*img) - (*templ);   // difference image
          // sign(raw_diff) * sqrt( 2*b^2*( sqrt(1+abs(raw_diff)^2/b^2)+1)  ))
          (*raw) = __builtin_ia32_orps(__builtin_ia32_andps(op->negzero, (*raw)),
              __builtin_ia32_sqrtps (
                __builtin_ia32_mulps(
                  __builtin_ia32_sqrtps(op->ones + __builtin_ia32_divps(__builtin_ia32_mulps((*raw),(*raw)),
                      op->norm_outlier_tmpbsq)) - op->ones, op->norm_outlier_tmp2bsq)
                )
              );
          (*cost) = __builtin_ia32_andnps(op->negzero,  (*raw) );
        }

        break;
      case 0:
      default:
        // L2-Norm

        for (int i = op->n_vals / 4; i--; ++raw, ++img, ++templ, ++cost) {
          (*raw) = (*img) - (*templ);  // difference image
          (*cost) = __builtin_ia32_andnps(op->negzero, (*raw));
        }

        break;
    }

  }

  void PatClass::OptimizeComputeErrImg() {

    InterpolatePatch();
    ComputeCostErr();

    // Compute step norm
    p_state->delta_p_sq_norm = p_state->delta_p.squaredNorm();
    if (p_state->count == 1)
      p_state->delta_p_sq_norm_init = p_state->delta_p_sq_norm;

    // Check early termination criterions
    p_state->mares_old = p_state->mares;
    p_state->mares = p_state->cost_diff.lpNorm<1>() / (op->n_vals);

    if (!((p_state->count < op->grad_descent_iter) & (p_state->mares > op->res_thresh)
          & ((p_state->count < op->grad_descent_iter) | (p_state->delta_p_sq_norm / p_state->delta_p_sq_norm_init >= op->dp_thresh))
          & ((p_state->count < op->grad_descent_iter) | (p_state->mares / p_state->mares_old <= op->dr_thresh)))) {
      p_state->has_converged = 1;
    }

  }

  // Extract patch on integer position, and gradients, No Bilinear interpolation
  void PatClass::ExtractPatch() {

    float *patch_f = patch.data();
    float *patch_xf = patch_x.data();
    float *patch_yf = patch_y.data();

    int x, y;
    x = round(midpoint[0]) + i_params->padding;
    y = round(midpoint[1]) + i_params->padding;

    int posxx = 0;

    int lb = -op->patch_size / 2;
    int ub = op->patch_size / 2 - 1;

    for (int j = lb; j <= ub; ++j) {
      for (int i = lb; i <= ub; ++i, ++posxx) {

        // RGB
        int idx = 3 * ((x + i) + (y + j) * i_params->width_pad);

        patch_f[posxx] = I0->data()[idx]; patch_xf[posxx] = I0x->data()[idx]; patch_yf[posxx] = I0y->data()[idx]; ++posxx; ++idx;
        patch_f[posxx] = I0->data()[idx]; patch_xf[posxx] = I0x->data()[idx]; patch_yf[posxx] = I0y->data()[idx]; ++posxx; ++idx;
        patch_f[posxx] = I0->data()[idx]; patch_xf[posxx] = I0x->data()[idx]; patch_yf[posxx] = I0y->data()[idx];

      }
    }

    // Mean Normalization
    if (op->use_mean_normalization > 0)
      patch.array() -= (patch.sum() / op->n_vals);

  }

  // Extract patch on float position with bilinear interpolation, no gradients.
  void PatClass::InterpolatePatch() {

    float *raw = p_state->raw_diff.data();

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

    const float * img_a, * img_b, * img_c, * img_d, *img_e;

    // RGB
    img_e = I1->data() + (pos[0] - op->patch_size / 2) * 3;

    int lb = -op->patch_size / 2;
    int ub = op->patch_size / 2 - 1;

    for (pos_iter[1] = pos[1] + lb; pos_iter[1] <= pos[1] + ub; ++pos_iter[1]) {

      // RGB
      img_a = img_e + pos_iter[1] * i_params->width_pad * 3;
      img_c = img_e + (pos_iter[1] - 1) * i_params->width_pad * 3;
      img_b = img_a - 3;
      img_d = img_c - 3;

      for (pos_iter[0] = pos[0] + lb; pos_iter[0] <= pos[0] + ub; ++pos_iter[0],
          ++raw, ++img_a, ++img_b, ++img_c, ++img_d) {

        // RGB
        (*raw) = weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);
        ++raw; ++img_a; ++img_b; ++img_c; ++img_d;
        (*raw) = weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);
        ++raw; ++img_a; ++img_b; ++img_c; ++img_d;
        (*raw) = weight[0] * (*img_a) + weight[1] * (*img_b) + weight[2] * (*img_c) + weight[3] * (*img_d);

      }

    }

    // Mean Normalization
    if (op->use_mean_normalization > 0)
      p_state->raw_diff.array() -= (p_state->raw_diff.sum() / op->n_vals);

  }

}
