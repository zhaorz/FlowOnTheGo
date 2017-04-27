#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>

#include "refine_variational.h"

using std::cout;
using std::endl;
using std::vector;


namespace OFC
{

  VarRefClass::VarRefClass(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in,
      const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in,
      const camparam* cpt_in,const camparam* cpo_in,const optparam* op_in, float *flowout)
    : cpt(cpt_in), cpo(cpo_in), op(op_in)
  {

    // initialize parameters
    tvparams.alpha = op->tv_alpha;
    tvparams.beta = 0.0f;  // for matching term, not needed for us
    tvparams.gamma = op->tv_gamma;
    tvparams.delta = op->tv_delta;
    tvparams.n_inner_iteration = (cpt->curr_lv+1);
    tvparams.n_solver_iteration = op->tv_solverit;//5;
    tvparams.sor_omega = op->tv_sor;

    tvparams.tmp_quarter_alpha = 0.25f*tvparams.alpha;
    tvparams.tmp_half_gamma_over3 = tvparams.gamma*0.5f/3.0f;
    tvparams.tmp_half_delta_over3 = tvparams.delta*0.5f/3.0f;
    tvparams.tmp_half_beta = tvparams.beta*0.5f;

    float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
    deriv = convolution_new(2, deriv_filter, 0);
    float deriv_filter_flow[2] = {0.0f, -0.5f};
    deriv_flow = convolution_new(1, deriv_filter_flow, 0);

    // copy flow initialization into FV structs
    static int noparam = 2; // Optical flow

    std::vector<image_t*> flow_sep(noparam);

    for (int i = 0; i < noparam; ++i )
      flow_sep[i] = image_new(cpt->width,cpt->height);

    for (int iy = 0; iy < cpt->height; ++iy)
      for (int ix = 0; ix < cpt->width; ++ix)
      {
        int i  = iy * cpt->width          + ix;
        int is = iy * flow_sep[0]->stride + ix;
        for (int j = 0; j < noparam; ++j)
          flow_sep[j]->c1[is] = flowout[i*noparam + j];
      }

    // copy image data into FV structs
    image_t * im_ao, *im_bo;
    im_ao = image_new(cpt->width,cpt->height);
    im_bo = image_new(cpt->width,cpt->height);

    copyimage(im_ao_in, im_ao);
    copyimage(im_bo_in, im_bo);

    // Call solver
    RefLevelOF(flow_sep[0], flow_sep[1], im_ao, im_bo);

    // Copy flow result back
    for (int iy = 0; iy < cpt->height; ++iy)
      for (int ix = 0; ix < cpt->width; ++ix)
      {
        int i  = iy * cpt->width          + ix;
        int is = iy * flow_sep[0]->stride + ix;
        for (int j = 0; j < noparam; ++j)
          flowout[i*noparam + j] = flow_sep[j]->c1[is];
      }

    // free FV structs
    for (int i = 0; i < noparam; ++i )
      image_delete(flow_sep[i]);

    convolution_delete(deriv);
    convolution_delete(deriv_flow);


    image_delete(im_ao);
    image_delete(im_bo);
  }


  void VarRefClass::copyimage(const float* img, image_t * img_t)
  {
    const float * img_st = img +     (cpt->tmp_w + 1 ) * (cpt->imgpadding); // remove image padding, start at first valid pixel

    for (int yi = 0; yi < cpt->height; ++yi)
    {
      for (int xi = 0; xi < cpt->width; ++xi, ++img_st)
      {
        int i    = yi*img_t->stride+ xi;

        img_t->c1[i] =  (*img_st);
      }
      img_st +=     2 * cpt->imgpadding;
    }
  }


  void VarRefClass::RefLevelOF(image_t *wx, image_t *wy, const image_t *im1, const image_t *im2)
  {
    int i_inner_iteration;
    int width  = wx->width;
    int height = wx->height;
    int stride = wx->stride;


    image_t *du = image_new(width,height), *dv = image_new(width,height), // the flow increment
            *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
            *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff. from (i,j) to (i+1,j)
            *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
            *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), // system matrix A of Ax=b for each pixel
            *b1 = image_new(width,height), *b2 = image_new(width,height); // system matrix b of Ax=b for each pixel

    image_t *w_im2 = image_new(width,height), // warped second image
            *Ix = image_new(width,height), *Iy = image_new(width,height), *Iz = image_new(width,height), // first order derivatives
            *Ixx = image_new(width,height), *Ixy = image_new(width,height), *Iyy = image_new(width,height), *Ixz = image_new(width,height), *Iyz = image_new(width,height); // second order derivatives

    // warp second image
    image_warp(w_im2, mask, im2, wx, wy);
    // compute derivatives
    get_derivatives(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    // erase du and dv
    image_erase(du);
    image_erase(dv);
    // initialize uu and vv
    memcpy(uu->c1,wx->c1,wx->stride*wx->height*sizeof(float));
    memcpy(vv->c1,wy->c1,wy->stride*wy->height*sizeof(float));
    // inner fixed point iterations
    for(i_inner_iteration = 0 ; i_inner_iteration < tvparams.n_inner_iteration ; i_inner_iteration++)
    {
      //  compute robust function and system
      compute_smoothness(smooth_horiz, smooth_vert, uu, vv, deriv_flow, tvparams.tmp_quarter_alpha );
      //compute_data_and_match(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, desc_weight, desc_flow_x, desc_flow_y, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
      compute_data(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
      sub_laplacian(b1, wx, smooth_horiz, smooth_vert);
      sub_laplacian(b2, wy, smooth_horiz, smooth_vert);

      // solve system
#ifdef WITH_OPENMP
      sor_coupled_slow_but_readable(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega); // slower but parallelized
#else
      sor_coupled(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega);
#endif

      // update flow plus flow increment
      int i;
      v4sf *uup = (v4sf*) uu->c1, *vvp = (v4sf*) vv->c1, *wxp = (v4sf*) wx->c1, *wyp = (v4sf*) wy->c1, *dup = (v4sf*) du->c1, *dvp = (v4sf*) dv->c1;
      for( i=0 ; i<height*stride/4 ; i++)
      {
        (*uup) = (*wxp) + (*dup);
        (*vvp) = (*wyp) + (*dvp);
        uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;
      }

    }
    // add flow increment to current flow
    memcpy(wx->c1,uu->c1,uu->stride*uu->height*sizeof(float));
    memcpy(wy->c1,vv->c1,vv->stride*vv->height*sizeof(float));

    // free memory
    image_delete(du); image_delete(dv);
    image_delete(mask);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete(a11); image_delete(a12); image_delete(a22);
    image_delete(b1); image_delete(b2);

    image_delete(w_im2);
    image_delete(Ix); image_delete(Iy); image_delete(Iz);
    image_delete(Ixx); image_delete(Ixy); image_delete(Iyy); image_delete(Ixz); image_delete(Iyz);

  }


  void VarRefClass::RefLevelDE(image_t *wx, const image_t *im1, const image_t *im2)
  {
    int i_inner_iteration;
    int width  = wx->width;
    int height = wx->height;
    int stride = wx->stride;

    image_t *du = image_new(width,height), *wy_dummy = image_new(width,height), // the flow increment
            *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
            *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff. from (i,j) to (i+1,j)
            *uu = image_new(width,height), // flow plus flow increment
            *a11 = image_new(width,height), // system matrix A of Ax=b for each pixel
            *b1 = image_new(width,height); // system matrix b of Ax=b for each pixel

    image_erase(wy_dummy);

    image_t *w_im2 = image_new(width,height), // warped second image
            *Ix = image_new(width,height), *Iy = image_new(width,height), *Iz = image_new(width,height), // first order derivatives
            *Ixx = image_new(width,height), *Ixy = image_new(width,height), *Iyy = image_new(width,height), *Ixz = image_new(width,height), *Iyz = image_new(width,height); // second order derivatives

    // warp second image
    image_warp(w_im2, mask, im2, wx, wy_dummy);
    // compute derivatives
    get_derivatives(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    // erase du and dv
    image_erase(du);

    // initialize uu and vv
    memcpy(uu->c1,wx->c1,wx->stride*wx->height*sizeof(float));

    // inner fixed point iterations
    for(i_inner_iteration = 0 ; i_inner_iteration < tvparams.n_inner_iteration ; i_inner_iteration++)
    {
      //  compute robust function and system
      compute_smoothness(smooth_horiz, smooth_vert, uu, wy_dummy, deriv_flow, tvparams.tmp_quarter_alpha );
      compute_data_DE(a11, b1, mask, wx, du, uu, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
      sub_laplacian(b1, wx, smooth_horiz, smooth_vert);

      // solve system
      sor_coupled_slow_but_readable_DE(du, a11, b1, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega);

      // update flow plus flow increment
      int i;
      v4sf *uup = (v4sf*) uu->c1, *wxp = (v4sf*) wx->c1, *dup = (v4sf*) du->c1;

      if(cpt->camlr==0)  // check if right or left camera, needed to truncate values above/below zero
      {
        for( i=0 ; i<height*stride/4 ; i++)
        {
          (*uup) = __builtin_ia32_minps(   (*wxp) + (*dup)   ,  op->zero);
          uup+=1; wxp+=1; dup+=1;
        }
      }
      else
      {
        for( i=0 ; i<height*stride/4 ; i++)
        {
          (*uup) = __builtin_ia32_maxps(   (*wxp) + (*dup)   ,  op->zero);
          uup+=1; wxp+=1; dup+=1;
        }
      }
    }
    // add flow increment to current flow
    memcpy(wx->c1,uu->c1,uu->stride*uu->height*sizeof(float));

    // free memory
    image_delete(du); image_delete(wy_dummy);
    image_delete(mask);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu);
    image_delete(a11);
    image_delete(b1);

    image_delete(w_im2);
    image_delete(Ix); image_delete(Iy); image_delete(Iz);
    image_delete(Ixx); image_delete(Ixy); image_delete(Iyy); image_delete(Ixz); image_delete(Iyz);
  }


  VarRefClass::~VarRefClass()
  {

  }

}
