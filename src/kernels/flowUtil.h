#ifndef __KERNEL_FLOW_UTIL_H__
#define __KERNEL_FLOW_UTIL_H__

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// Local
#include "../FDF1.0.1/image.h"
#include "../common/Exceptions.h"
#include "../common/timer.h"
#include "../sandbox/process.h"
#include "../patch.h"

using namespace OFC;

namespace cu {

  void dataTerm(
      image_t *a11, image_t *a12, image_t *a22,
      image_t *b1, image_t *b2, 
      image_t *mask, 
      image_t *wx, image_t *wy,
      image_t *du, image_t *dv, 
      image_t *uu, image_t *vv, 
      color_image_t *Ix,  color_image_t *Iy,  color_image_t *Iz,
      color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy,
      color_image_t *Ixz, color_image_t *Iyz, 
      const float half_delta_over3, const float half_beta, const float half_gamma_over3);

  void subLaplacianHoriz(
      float *src, float *dst, float *weights, int height, int width, int stride);

  void subLaplacianVert(
      float *src, float *dst, float *weights, int height, int stride);

  void sor(
      float *du, float *dv,
      float *a11, float *a12, float *a22,
      float *b1, float *b2,
      float *horiz, float *vert,
      int iterations, float omega,
      int height, int width, int stride);

  void getMeanImageAndDiff(
      float *img1, float *img2, float *avgImg, float *diff,
      int height, int stride);

  void colorImageDerivative(
      float *dst, float *src, float *pDeviceKernel, int height, int width, int stride, bool horiz);

  void imageDerivative(
      float *dst, float *src, float *pDeviceKernel, int height, int width, int stride, bool horiz);

  void smoothnessTerm(
      float *dst_horiz, float *dst_vert, float *smoothness,
      float *ux,  float *uy,  float *vx,  float *vy,
      float quarter_alpha, float epsilon_smooth,
      int height, int width, int stride);

  void flowUpdate(
      float *uu, float *vv, float *wx, float *wy, float *du, float *dv,
      int height, int width, int stride);

}

#endif // end __KERNEL_FLOW_UTIL_H__

