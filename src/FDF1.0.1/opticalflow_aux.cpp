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

  cu::warpImage(dst, mask, src, wx, wy);


  // int i, j, offset, incr_line = mask->stride-mask->width, x, y, x1, x2, y1, y2;
  // float xx, yy, dx, dy;
  // for(j=0,offset=0 ; j<src->height ; j++)
  // {
  //   for(i=0 ; i<src->width ; i++,offset++)
  //   {
  //     xx = i+wx->c1[offset];
  //     yy = j+wy->c1[offset];
  //     x = floor(xx);
  //     y = floor(yy);
  //     dx = xx-x;
  //     dy = yy-y;
  //     mask->c1[offset] = (xx>=0 && xx<=src->width-1 && yy>=0 && yy<=src->height-1);
  //     x1 = MINMAX_TA(x,src->width);
  //     x2 = MINMAX_TA(x+1,src->width);
  //     y1 = MINMAX_TA(y,src->height);
  //     y2 = MINMAX_TA(y+1,src->height);
  //     dst->c1[offset] = 
  //       src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
  //       src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
  //       src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
  //       src->c1[y2*src->stride+x2]*dx*dy;
  //     dst->c2[offset] = 
  //       src->c2[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
  //       src->c2[y1*src->stride+x2]*dx*(1.0f-dy) +
  //       src->c2[y2*src->stride+x1]*(1.0f-dx)*dy +
  //       src->c2[y2*src->stride+x2]*dx*dy;
  //     dst->c3[offset] = 
  //       src->c3[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
  //       src->c3[y1*src->stride+x2]*dx*(1.0f-dy) +
  //       src->c3[y2*src->stride+x1]*(1.0f-dx)*dy +
  //       src->c3[y2*src->stride+x2]*dx*dy;
  //   }
  //   offset += incr_line;
  // }
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

  cu::smoothnessTerm(
      dst_horiz->c1, dst_vert->c1, smoothness->c1,
      ux->c1, uy->c1, vx->c1, vy->c1,
      quarter_alpha, epsilon_smooth,
      height, width, stride);

  // Cleanup extra columns
  for(j=0;j<height;j++){
    memset(&dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1));
  }
  // Cleanup last row
  memset( &dst_vert->c1[(height-1)*stride], 0, sizeof(float)*stride);

  image_delete(ux); image_delete(uy); image_delete(vx); image_delete(vy); 
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
  return cu::dataTerm(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, half_delta_over3, half_beta, half_gamma_over3);
}

