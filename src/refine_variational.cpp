#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/cuda_helper.h"
#include "kernels/flowUtil.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>
#include <arm_neon.h>

#include "refine_variational.h"
#include "common/timer.h"

using std::cout;
using std::endl;
using std::vector;

using namespace timer;

namespace OFC {

  VarRefClass::VarRefClass(const float * _I0, const float * _I1,
      const img_params* _i_params, const opt_params* _op, float *flowout)
    : i_params(_i_params), op(_op) {

      // initialize parameters
      vr.alpha = op->var_ref_alpha;
      vr.beta = 0.0f;  // for matching term, not needed for us
      vr.gamma = op->var_ref_gamma;
      vr.delta = op->var_ref_delta;
      vr.inner_iter = i_params->curr_lvl + 1;
      vr.solve_iter = op->var_ref_iter;
      vr.sor_omega = op->var_ref_sor_weight;

      vr.tmp_quarter_alpha = 0.25f * vr.alpha;
      vr.tmp_half_gamma_over3 = vr.gamma * 0.5f / 3.0f;
      vr.tmp_half_delta_over3 = vr.delta * 0.5f / 3.0f;
      vr.tmp_half_beta = vr.beta * 0.5f;

      auto start_kernels = now();
      float pHostColorDerivativeKernel[5] = { 1.0 / 12.0, 2.0 / 3.0, 0.0, -2.0 / 3.0, 1.0 / 12.0 };
      checkCudaErrors( cudaHostAlloc((void**) &pDeviceColorDerivativeKernel, 5 * sizeof(float), cudaHostAllocMapped) );
      memcpy(pDeviceColorDerivativeKernel, pHostColorDerivativeKernel, 5 * sizeof(float));

      float pHostDerivativeKernel[3] = { 0.5, 0.0, -0.5 };
      checkCudaErrors( cudaHostAlloc((void**) &pDeviceDerivativeKernel, 3 * sizeof(float), cudaHostAllocMapped) );
      memcpy(pDeviceDerivativeKernel, pHostDerivativeKernel, 3 * sizeof(float));
      calc_print_elapsed("derivative kernels", start_kernels);

      // copy flow initialization into FV structs
      static int noparam = 2; // Optical flow

      auto start_flow_sep = now();
      std::vector<image_t*> flow_sep(noparam);

      for (int i = 0; i < noparam; ++i)
        flow_sep[i] = image_new(i_params->width, i_params->height);

      cu::sepFlow(flow_sep, flowout, i_params->height, i_params->width);
      calc_print_elapsed("refine: flow_sep", start_flow_sep);

      auto start_alloc_coeffs = now();
      checkCudaErrors( cudaMalloc((void**) &pDeviceSubLaplacianCoeffs,
            flow_sep[0]->height * flow_sep[0]->stride * sizeof(float)) );
      calc_print_elapsed("laplacian coeffs malloc", start_alloc_coeffs);

      // copy image data into FV structs
      auto start_copy_images = now();
      color_image_t * I0, * I1;
      I0 = color_image_new(i_params->width, i_params->height);
      I1 = color_image_new(i_params->width, i_params->height);

      cu::copyImage(I0, _I0, i_params->width_pad, i_params->padding, i_params->height, i_params->width);
      cu::copyImage(I1, _I1, i_params->width_pad, i_params->padding, i_params->height, i_params->width);

      calc_print_elapsed("copy images", start_copy_images);

      // Call solver
      auto start_solver = now();
      RefLevelOF(flow_sep[0], flow_sep[1], I0, I1);
      calc_print_elapsed("RefLevelOF [total]", start_solver);

      // Copy flow result back
      auto start_copy = now();
      cu::mergeFlow(flow_sep, flowout, i_params->height, i_params->width);
      calc_print_elapsed("refine: copy back", start_copy);

      // free FV structs
      for (int i = 0; i < noparam; ++i )
        image_delete(flow_sep[i]);

      color_image_delete(I0);
      color_image_delete(I1);

    }


  void VarRefClass::copyimage(const float* img, color_image_t * img_t) {

    // remove image padding, start at first valid pixel
    const float * img_st = img + 3 * (i_params->width_pad + 1 ) * (i_params->padding);

    for (int yi = 0; yi < i_params->height; ++yi) {
      for (int xi = 0; xi < i_params->width; ++xi, ++img_st) {

        // RGB
        int i = yi * img_t->stride + xi;
        img_t->c1[i] =  (*img_st); ++img_st;
        img_t->c2[i] =  (*img_st); ++img_st;
        img_t->c3[i] =  (*img_st);

      }

      img_st += 3 * 2 * i_params->padding;

    }

  }


  void VarRefClass::RefLevelOF(image_t *wx, image_t *wy, const color_image_t *im1, const color_image_t *im2) {

    int i_inner_iteration;
    int width  = wx->width;
    int height = wx->height;
    int stride = wx->stride;


    auto start_setup = now();
    image_t *du = image_new(width,height), *dv = image_new(width,height), // the flow increment
            *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
            *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff. from (i,j) to (i+1,j)
            *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
            *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), // system matrix A of Ax=b for each pixel
            *b1 = image_new(width,height), *b2 = image_new(width,height); // system matrix b of Ax=b for each pixel

    color_image_t *w_im2 = color_image_new(width,height), // warped second image
                  *Ix = color_image_new(width,height), *Iy = color_image_new(width,height), *Iz = color_image_new(width,height), // first order derivatives
                  *Ixx = color_image_new(width,height), *Ixy = color_image_new(width,height),
                  *Iyy = color_image_new(width,height), *Ixz = color_image_new(width,height), *Iyz = color_image_new(width,height); // second order derivatives
    calc_print_elapsed("RefLevelOF setup", start_setup);

    // warp second image
    auto start_image_warp = now();
    cu::warpImage(w_im2, mask, im2, wx, wy);
    cudaDeviceSynchronize();
    calc_print_elapsed("RefLevelOF image_warp", start_image_warp);

    // compute derivatives
    auto start_get_derivs = now();
    cu::getDerivatives(im1, w_im2, pDeviceColorDerivativeKernel, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    cudaDeviceSynchronize();
    calc_print_elapsed("RefLevelOF get_derivatives", start_get_derivs);

    // erase du and dv
    auto start_image_erase = now();
    image_erase(du);
    image_erase(dv);
    cudaDeviceSynchronize();
    calc_print_elapsed("RefLevelOF image_erase", start_image_erase);

    // initialize uu and vv
    // TODO: use cudaMemcpy
    memcpy(uu->c1,wx->c1,wx->stride*wx->height*sizeof(float));
    memcpy(vv->c1,wy->c1,wy->stride*wy->height*sizeof(float));
    // inner fixed point iterations
    for(i_inner_iteration = 0 ; i_inner_iteration < vr.inner_iter ; i_inner_iteration++) {
      auto start_iteration = now();
      std::string iterStr = "[" + std::to_string(i_inner_iteration) + "]";

      //  compute robust function and system
      auto start_smooth = now();
      cu::computeSmoothness(smooth_horiz, smooth_vert, uu, vv, pDeviceDerivativeKernel, vr.tmp_quarter_alpha );
      cudaDeviceSynchronize();
      calc_print_elapsed(("RefLevelOF " + iterStr + " smoothness").c_str(), start_smooth);

      auto start_data = now();
      cu::dataTerm(a11, a12, a22, b1, b2,
          mask, wx, wy, du, dv, uu, vv, 
          Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, 
          vr.tmp_half_delta_over3, vr.tmp_half_beta, vr.tmp_half_gamma_over3);
      cudaDeviceSynchronize();
      calc_print_elapsed(("RefLevelOF " + iterStr + " data").c_str(), start_data);

      auto start_lapalcian = now();
      cu::subLaplacian(b1, wx, smooth_horiz, smooth_vert, pDeviceSubLaplacianCoeffs);
      cu::subLaplacian(b2, wy, smooth_horiz, smooth_vert, pDeviceSubLaplacianCoeffs);
      cudaDeviceSynchronize();
      calc_print_elapsed(("RefLevelOF " + iterStr + " laplacian").c_str(), start_lapalcian);

      // solve system
      auto start_sor = now();
      cu::sor(du->c1, dv->c1, a11->c1, a12->c1, a22->c1,
          b1->c1, b2->c1, smooth_horiz->c1, smooth_vert->c1,
          vr.solve_iter, vr.sor_omega, du->height, du->width, du->stride);
      cudaDeviceSynchronize();
      calc_print_elapsed(("RefLevelOF " + iterStr + " sor").c_str(), start_sor);

      // update flow plus flow increment
      auto start_flow_update = now();
      cu::flowUpdate(
          uu->c1, vv->c1, wx->c1, wy->c1, du->c1, dv->c1,
          height, width, stride);
      cudaDeviceSynchronize();
      calc_print_elapsed(("RefLevelOF " + iterStr + " flow update").c_str(), start_flow_update);

      calc_print_elapsed(("RefLevelOF " + iterStr + " [total]").c_str(), start_iteration);

    }
    // add flow increment to current flow
    // TODO: use cudaMemcpy
    auto start_increment_flow = now();
    memcpy(wx->c1,uu->c1,uu->stride*uu->height*sizeof(float));
    memcpy(wy->c1,vv->c1,vv->stride*vv->height*sizeof(float));
    cudaDeviceSynchronize();
    calc_print_elapsed("RefLevelOF increment flow", start_increment_flow);

    // free memory
    auto start_cleanup = now();
    image_delete(du); image_delete(dv);
    image_delete(mask);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete(a11); image_delete(a12); image_delete(a22);
    image_delete(b1); image_delete(b2);

    color_image_delete(w_im2);
    color_image_delete(Ix); color_image_delete(Iy); color_image_delete(Iz);
    color_image_delete(Ixx); color_image_delete(Ixy); color_image_delete(Iyy); color_image_delete(Ixz); color_image_delete(Iyz);
    cudaDeviceSynchronize();
    calc_print_elapsed("RefLevelOF cleanup", start_cleanup);
  }


  VarRefClass::~VarRefClass() {
    cudaFree(pDeviceColorDerivativeKernel); 
    cudaFree(pDeviceDerivativeKernel); 
    cudaFree(pDeviceSubLaplacianCoeffs); 
  }

}
