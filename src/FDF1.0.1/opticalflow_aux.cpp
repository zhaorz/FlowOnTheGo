#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>

#include "opticalflow_aux.h"
#include "../kernels/flowUtil.h"
#include "../common/timer.h"

#include <arm_neon.h>

using namespace timer;

#if (VECTOR_WIDTH == 4)
typedef float32x4_t v4sf;
#else
typedef float v4sf;
#endif

#define datanorm 0.1f*0.1f//0.01f // square of the normalization factor
#define epsilon_color (0.001f*0.001f)//0.000001f
#define epsilon_grad (0.001f*0.001f)//0.000001f
#define epsilon_desc (0.001f*0.001f)//0.000001f
#define epsilon_smooth (0.001f*0.001f)//0.000001f

/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
void image_warp(color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy)
{
  int i, j, offset, incr_line = mask->stride-mask->width, x, y, x1, x2, y1, y2;
  float xx, yy, dx, dy;
  for(j=0,offset=0 ; j<src->height ; j++)
  {
    for(i=0 ; i<src->width ; i++,offset++)
    {
      xx = i+wx->c1[offset];
      yy = j+wy->c1[offset];
      x = floor(xx);
      y = floor(yy);
      dx = xx-x;
      dy = yy-y;
      mask->c1[offset] = (xx>=0 && xx<=src->width-1 && yy>=0 && yy<=src->height-1);
      x1 = MINMAX_TA(x,src->width);
      x2 = MINMAX_TA(x+1,src->width);
      y1 = MINMAX_TA(y,src->height);
      y2 = MINMAX_TA(y+1,src->height);
      dst->c1[offset] = 
        src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
        src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
        src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
        src->c1[y2*src->stride+x2]*dx*dy;
      dst->c2[offset] = 
        src->c2[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
        src->c2[y1*src->stride+x2]*dx*(1.0f-dy) +
        src->c2[y2*src->stride+x1]*(1.0f-dx)*dy +
        src->c2[y2*src->stride+x2]*dx*dy;
      dst->c3[offset] = 
        src->c3[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
        src->c3[y1*src->stride+x2]*dx*(1.0f-dy) +
        src->c3[y2*src->stride+x1]*(1.0f-dx)*dy +
        src->c3[y2*src->stride+x2]*dx*dy;
    }
    offset += incr_line;
  }
}


/* compute image first and second order spatio-temporal derivatives of a color image */
void get_derivatives(
    const color_image_t *im1, const color_image_t *im2, float *pDeviceKernel,
    color_image_t *dx, color_image_t *dy, color_image_t *dt, 
    color_image_t *dxx, color_image_t *dxy, color_image_t *dyy, color_image_t *dxt, color_image_t *dyt)
{
  // derivatives are computed on the mean of the first image and the warped second image
  color_image_t *tmp_im2 = color_image_new(im2->width,im2->height);    

  int height = im2->height;
  int width = im2->width;
  int stride = im2->stride;

  cu::getMeanImageAndDiff(im1->c1, im2->c1, tmp_im2->c1, dt->c1, im1->height, im1->stride);

  // compute all other derivatives
  cu::colorImageDerivative(dx->c1,  tmp_im2->c1, pDeviceKernel, height, width, stride, true); // horizontal
  cu::colorImageDerivative(dy->c1,  tmp_im2->c1, pDeviceKernel, height, width, stride, false);
  cu::colorImageDerivative(dxx->c1, dx->c1,      pDeviceKernel, height, width, stride, true);
  cu::colorImageDerivative(dxy->c1, dx->c1,      pDeviceKernel, height, width, stride, false);
  cu::colorImageDerivative(dyy->c1, dy->c1,      pDeviceKernel, height, width, stride, false);
  cu::colorImageDerivative(dxt->c1, dt->c1,      pDeviceKernel, height, width, stride, true);
  cu::colorImageDerivative(dyt->c1, dt->c1,      pDeviceKernel, height, width, stride, false);

  // free memory
  color_image_delete(tmp_im2);
}


/* compute the smoothness term */
/* It is represented as two images, the first one for horizontal smoothness, the second for vertical
   in dst_horiz, the pixel i,j represents the smoothness weight between pixel i,j and i,j+1
   in dst_vert, the pixel i,j represents the smoothness weight between pixel i,j and i+1,j */
void compute_smoothness(image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, float *deriv_flow, const float quarter_alpha){
  const int width = uu->width, height = vv->height, stride = uu->stride;
  int j;
  image_t *ux = image_new(width,height), *vx = image_new(width,height), *uy = image_new(width,height), *vy = image_new(width,height), *smoothness = image_new(width,height);

  // compute derivatives [-0.5 0 0.5]
  cu::imageDerivative(ux->c1, uu->c1, deriv_flow, height, width, stride, true);
  cu::imageDerivative(vx->c1, vv->c1, deriv_flow, height, width, stride, true);
  cu::imageDerivative(uy->c1, uu->c1, deriv_flow, height, width, stride, false);
  cu::imageDerivative(vy->c1, vv->c1, deriv_flow, height, width, stride, false);

  // compute smoothness
  float *uxp = (float*) ux->c1, *vxp = (float*) vx->c1, *uyp = (float*) uy->c1, *vyp = (float*) vy->c1, *sp = (float*) smoothness->c1;
  const float qa = quarter_alpha;
  const float epsmooth = epsilon_smooth;

  for(j=0 ; j< height*stride; j++){
    *sp = qa / sqrtf(
        (*uxp)*(*uxp) + (*uyp)*(*uyp) + (*vxp)*(*vxp) + (*vyp)*(*vyp) + epsmooth );

    sp+=1;uxp+=1; uyp+=1; vxp+=1; vyp+=1;
  }

  image_delete(ux); image_delete(uy); image_delete(vx); image_delete(vy); 


  // compute dst_horiz
  float *dsthp = (float*) dst_horiz->c1; sp = (float*) smoothness->c1;
  for(j=0;j<height;j++){
    // create an aligned copy
    float *spf = (float*) sp;

    for(int i = 0; i < stride; i++){
      *dsthp = (*sp) + (*(sp + 1));
      dsthp+=1; sp+=1;
    }

    // Cleanup extra columns
    memset( &dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1));
  }

  // compute dst_vert
  float *dstvp = (float*) dst_vert->c1, *sp_bottom = (float*) (smoothness->c1+stride); sp = (float*) smoothness->c1;
  for(j = 0 ; j < (height - 1) * stride; j++){
    *dstvp = (*sp) + (*sp_bottom);
    dstvp+=1; sp+=1; sp_bottom+=1;
  }

  // Cleanup last row
  memset( &dst_vert->c1[(height-1)*stride], 0, sizeof(float)*stride);
  image_delete(smoothness);
}





/* sub the laplacian (smoothness term) to the right-hand term */
void sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert){

  cu::subLaplacianHoriz(src->c1, dst->c1, weight_horiz->c1, src->height, src->width, src->stride);

  cu::subLaplacianVert(src->c1, dst->c1, weight_vert->c1, src->height, src->stride);
}

/* compute the dataterm // REMOVED MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
void compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3)
{
#if (UNIFIED_MEM)
  return cu::dataTerm(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, half_delta_over3, half_beta, half_gamma_over3);
#endif


#if (VECTOR_WIDTH == 4)
  const v4sf dnorm = {datanorm, datanorm, datanorm, datanorm};
  const v4sf hdover3 = {half_delta_over3, half_delta_over3, half_delta_over3, half_delta_over3};
  const v4sf epscolor = {epsilon_color, epsilon_color, epsilon_color, epsilon_color};
  const v4sf hgover3 = {half_gamma_over3, half_gamma_over3, half_gamma_over3, half_gamma_over3};
  const v4sf epsgrad = {epsilon_grad, epsilon_grad, epsilon_grad, epsilon_grad};
  //const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
  //const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};
#else
  const v4sf dnorm = datanorm;
  const v4sf hdover3 = half_delta_over3;
  const v4sf epscolor = epsilon_color;
  const v4sf hgover3 = half_gamma_over3;
  const v4sf epsgrad = epsilon_grad;
#endif

  v4sf *dup = (v4sf*) du->c1, *dvp = (v4sf*) dv->c1,
       *maskp = (v4sf*) mask->c1,
       *a11p = (v4sf*) a11->c1, *a12p = (v4sf*) a12->c1, *a22p = (v4sf*) a22->c1, 
       *b1p = (v4sf*) b1->c1, *b2p = (v4sf*) b2->c1, 
       *ix1p=(v4sf*)Ix->c1, *iy1p=(v4sf*)Iy->c1, *iz1p=(v4sf*)Iz->c1, *ixx1p=(v4sf*)Ixx->c1, *ixy1p=(v4sf*)Ixy->c1, *iyy1p=(v4sf*)Iyy->c1, *ixz1p=(v4sf*)Ixz->c1, *iyz1p=(v4sf*) Iyz->c1, 
       *ix2p=(v4sf*)Ix->c2, *iy2p=(v4sf*)Iy->c2, *iz2p=(v4sf*)Iz->c2, *ixx2p=(v4sf*)Ixx->c2, *ixy2p=(v4sf*)Ixy->c2, *iyy2p=(v4sf*)Iyy->c2, *ixz2p=(v4sf*)Ixz->c2, *iyz2p=(v4sf*) Iyz->c2, 
       *ix3p=(v4sf*)Ix->c3, *iy3p=(v4sf*)Iy->c3, *iz3p=(v4sf*)Iz->c3, *ixx3p=(v4sf*)Ixx->c3, *ixy3p=(v4sf*)Ixy->c3, *iyy3p=(v4sf*)Iyy->c3, *ixz3p=(v4sf*)Ixz->c3, *iyz3p=(v4sf*) Iyz->c3, 
       *uup = (v4sf*) uu->c1, *vvp = (v4sf*)vv->c1, *wxp = (v4sf*)wx->c1, *wyp = (v4sf*)wy->c1;


  memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
  memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
  memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
  memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
  memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

  int i;
  for(i = 0 ; i<uu->height*uu->stride/VECTOR_WIDTH ; i++){
    v4sf tmp, tmp2, n1, n2;
    v4sf tmp3, tmp4, tmp5, tmp6, n3, n4, n5, n6;
    // dpsi color
    if(half_delta_over3){
      tmp  = *iz1p + (*ix1p)*(*dup) + (*iy1p)*(*dvp);
      n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
      tmp2 = *iz2p + (*ix2p)*(*dup) + (*iy2p)*(*dvp);
      n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
      tmp3 = *iz3p + (*ix3p)*(*dup) + (*iy3p)*(*dvp);
      n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
#if (VECTOR_WIDTH == 4)
      tmp = (*maskp) * hdover3 / vsqrtq_f32(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
#else
      tmp = (*maskp) * hdover3 / sqrtf(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
#endif
      tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
      *a11p += tmp  * (*ix1p) * (*ix1p);
      *a12p += tmp  * (*ix1p) * (*iy1p);
      *a22p += tmp  * (*iy1p) * (*iy1p);
      *b1p -=  tmp  * (*iz1p) * (*ix1p);
      *b2p -=  tmp  * (*iz1p) * (*iy1p);
      *a11p += tmp2 * (*ix2p) * (*ix2p);
      *a12p += tmp2 * (*ix2p) * (*iy2p);
      *a22p += tmp2 * (*iy2p) * (*iy2p);
      *b1p -=  tmp2 * (*iz2p) * (*ix2p);
      *b2p -=  tmp2 * (*iz2p) * (*iy2p);
      *a11p += tmp3 * (*ix3p) * (*ix3p);
      *a12p += tmp3 * (*ix3p) * (*iy3p);
      *a22p += tmp3 * (*iy3p) * (*iy3p);
      *b1p -=  tmp3 * (*iz3p) * (*ix3p);
      *b2p -=  tmp3 * (*iz3p) * (*iy3p);
    }

    // dpsi gradient
    n1 = (*ixx1p) * (*ixx1p) + (*ixy1p) * (*ixy1p) + dnorm;
    n2 = (*iyy1p) * (*iyy1p) + (*ixy1p) * (*ixy1p) + dnorm;
    tmp  = *ixz1p + (*ixx1p) * (*dup) + (*ixy1p) * (*dvp);
    tmp2 = *iyz1p + (*ixy1p) * (*dup) + (*iyy1p) * (*dvp);
    n3 = (*ixx2p) * (*ixx2p) + (*ixy2p) * (*ixy2p) + dnorm;
    n4 = (*iyy2p) * (*iyy2p) + (*ixy2p) * (*ixy2p) + dnorm;
    tmp3 = *ixz2p + (*ixx2p) * (*dup) + (*ixy2p) * (*dvp);
    tmp4 = *iyz2p + (*ixy2p) * (*dup) + (*iyy2p) * (*dvp);
    n5 = (*ixx3p) * (*ixx3p) + (*ixy3p) * (*ixy3p) + dnorm;
    n6 = (*iyy3p) * (*iyy3p) + (*ixy3p) * (*ixy3p) + dnorm;
    tmp5 = *ixz3p + (*ixx3p) * (*dup) + (*ixy3p) * (*dvp);
    tmp6 = *iyz3p + (*ixy3p) * (*dup) + (*iyy3p) * (*dvp);
#if (VECTOR_WIDTH == 4)
    tmp = (*maskp) * hgover3 / vsqrtq_f32(
        tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
#else
    tmp = (*maskp) * hgover3 / sqrtf(
        tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
#endif
    tmp6 = tmp/n6; tmp5 = tmp/n5; tmp4 = tmp/n4; tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;      
    *a11p += tmp *(*ixx1p)*(*ixx1p) + tmp2*(*ixy1p)*(*ixy1p);
    *a12p += tmp *(*ixx1p)*(*ixy1p) + tmp2*(*ixy1p)*(*iyy1p);
    *a22p += tmp2*(*iyy1p)*(*iyy1p) + tmp *(*ixy1p)*(*ixy1p);
    *b1p -=  tmp *(*ixx1p)*(*ixz1p) + tmp2*(*ixy1p)*(*iyz1p);
    *b2p -=  tmp2*(*iyy1p)*(*iyz1p) + tmp *(*ixy1p)*(*ixz1p);
    *a11p += tmp3*(*ixx2p)*(*ixx2p) + tmp4*(*ixy2p)*(*ixy2p);
    *a12p += tmp3*(*ixx2p)*(*ixy2p) + tmp4*(*ixy2p)*(*iyy2p);
    *a22p += tmp4*(*iyy2p)*(*iyy2p) + tmp3*(*ixy2p)*(*ixy2p);
    *b1p -=  tmp3*(*ixx2p)*(*ixz2p) + tmp4*(*ixy2p)*(*iyz2p);
    *b2p -=  tmp4*(*iyy2p)*(*iyz2p) + tmp3*(*ixy2p)*(*ixz2p);
    *a11p += tmp5*(*ixx3p)*(*ixx3p) + tmp6*(*ixy3p)*(*ixy3p);
    *a12p += tmp5*(*ixx3p)*(*ixy3p) + tmp6*(*ixy3p)*(*iyy3p);
    *a22p += tmp6*(*iyy3p)*(*iyy3p) + tmp5*(*ixy3p)*(*ixy3p);
    *b1p -=  tmp5*(*ixx3p)*(*ixz3p) + tmp6*(*ixy3p)*(*iyz3p);
    *b2p -=  tmp6*(*iyy3p)*(*iyz3p) + tmp5*(*ixy3p)*(*ixz3p);  



    dup+=1; dvp+=1; maskp+=1; a11p+=1; a12p+=1; a22p+=1; b1p+=1; b2p+=1; 
    ix1p+=1; iy1p+=1; iz1p+=1; ixx1p+=1; ixy1p+=1; iyy1p+=1; ixz1p+=1; iyz1p+=1;
    ix2p+=1; iy2p+=1; iz2p+=1; ixx2p+=1; ixy2p+=1; iyy2p+=1; ixz2p+=1; iyz2p+=1;
    ix3p+=1; iy3p+=1; iz3p+=1; ixx3p+=1; ixy3p+=1; iyy3p+=1; ixz3p+=1; iyz3p+=1;
    uup+=1;vvp+=1;wxp+=1; wyp+=1;

  }
}

