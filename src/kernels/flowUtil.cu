// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "../common/cuda_helper.h"

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include "../common/timer.h"
#include "../FDF1.0.1/image.h"
#include "flowUtil.h"

using namespace timer;

#define datanorm        0.1f*0.1f      //0.01f // square of the normalization factor
#define epsilon_color  (0.001f*0.001f) //0.000001f
#define epsilon_grad   (0.001f*0.001f) //0.000001f
#define epsilon_desc   (0.001f*0.001f) //0.000001f
#define epsilon_smooth (0.001f*0.001f) //0.000001f

__global__ void kernelDataTerm(
    float *a11c1, float *a12c1, float *a22c1,
    float *b1c1, float *b2c1, 
    float *maskc1, 
    float *wxc1, float *wyc1,
    float *duc1, float *dvc1, 
    float *uuc1, float *vvc1, 
    float *Ixc1,    float *Ixc2,    float *Ixc3,
    float *Iyc1,    float *Iyc2,    float *Iyc3,
    float *Izc1,    float *Izc2,    float *Izc3,
    float *Ixxc1,   float *Ixxc2,   float *Ixxc3,
    float *Ixyc1,   float *Ixyc2,   float *Ixyc3,
    float *Iyyc1,   float *Iyyc2,   float *Iyyc3,
    float *Ixzc1,   float *Ixzc2,   float *Ixzc3,
    float *Iyzc1,   float *Iyzc2,   float *Iyzc3, 
    const float half_delta_over3, const float half_beta, const float half_gamma_over3, int N) {

  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < N) {

    const float dnorm    = datanorm;
    const float hdover3  = half_delta_over3;
    const float epscolor = epsilon_color;
    const float hgover3  = half_gamma_over3;
    const float epsgrad  = epsilon_grad;

    float *dup  = (float*) duc1 + tidx,
          *dvp = (float*) dvc1 + tidx,
          *maskp = (float*) maskc1 + tidx,
          *a11p  = (float*) a11c1 + tidx,
          *a12p = (float*) a12c1 + tidx,
          *a22p = (float*) a22c1 + tidx, 
          *b1p   = (float*) b1c1 + tidx,
          *b2p = (float*) b2c1 + tidx, 
          *ix1p  = (float*) Ixc1 + tidx,
          *iy1p=(float*)Iyc1 + tidx,
          *iz1p=(float*)Izc1 + tidx,
          *ixx1p=(float*)Ixxc1 + tidx,
          *ixy1p=(float*)Ixyc1 + tidx,
          *iyy1p=(float*)Iyyc1 + tidx,
          *ixz1p=(float*)Ixzc1 + tidx,
          *iyz1p=(float*) Iyzc1 + tidx, 
          *ix2p  = (float*) Ixc2 + tidx,
          *iy2p=(float*)Iyc2 + tidx,
          *iz2p=(float*)Izc2 + tidx,
          *ixx2p=(float*)Ixxc2 + tidx,
          *ixy2p=(float*)Ixyc2 + tidx,
          *iyy2p=(float*)Iyyc2 + tidx,
          *ixz2p=(float*)Ixzc2 + tidx,
          *iyz2p=(float*) Iyzc2 + tidx, 
          *ix3p  = (float*) Ixc3 + tidx,
          *iy3p=(float*)Iyc3 + tidx,
          *iz3p=(float*)Izc3 + tidx,
          *ixx3p=(float*)Ixxc3 + tidx,
          *ixy3p=(float*)Ixyc3 + tidx,
          *iyy3p=(float*)Iyyc3 + tidx,
          *ixz3p=(float*)Ixzc3 + tidx,
          *iyz3p=(float*) Iyzc3 + tidx;


    float tmp, tmp2, n1, n2;
    float tmp3, tmp4, tmp5, tmp6, n3, n4, n5, n6;

    // dpsi color
    if(half_delta_over3){
      tmp  = *iz1p + (*ix1p)*(*dup) + (*iy1p)*(*dvp);
      n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
      tmp2 = *iz2p + (*ix2p)*(*dup) + (*iy2p)*(*dvp);
      n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
      tmp3 = *iz3p + (*ix3p)*(*dup) + (*iy3p)*(*dvp);
      n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
      tmp = (*maskp) * hdover3 / sqrtf(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
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
    tmp = (*maskp) * hgover3 / sqrtf(
        tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
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
  }

}

__global__ void kernelSubLaplacianVert(
    float *src, float *nextSrc,
    float *dst, float *nextDst,
    float *weights, int height, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < stride) {
    float *wvp    = weights + tidx,
          *srcp   = src + tidx,
          *srcp_s = nextSrc + tidx,
          *dstp   = dst + tidx,
          *dstp_s = nextDst + tidx;

    for (int j = 0; j < height - 1; j++) {
      float tmp = (*wvp) * ((*srcp_s)-(*srcp));
      *dstp += tmp;
      *dstp_s -= tmp;
      wvp += stride; srcp += stride; srcp_s += stride; dstp += stride; dstp_s += stride;
    }
  }

}

__global__ void kernelSubLaplacianHoriz(
    float *src, float *dst, float *weights, float *coeffs, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int col  = tidx % width;

  const int BLOCK_HEIGHT = 1;

  if (tidx < width) {
    float *pSrc         = src + tidx,
          *pDst         = dst + tidx,
          *pWeight      = weights + tidx,
          *pCoeffCalc   = coeffs + tidx,
          *pCoeffUpdate = pCoeffCalc;

    int nBlocks = (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    int jCalc = 0;
    int jUpdate = 0;

    // Block calculation and update so coeffs fit in cache

    for (int iBlock = 0; iBlock < nBlocks; iBlock++) {

      // Calc coeffs
      for (int j = 0; j < BLOCK_HEIGHT && jCalc < height; j++, jCalc++) {
        // Do not calculate the last column
        if (col != width - 1)
          *pCoeffCalc = (*pWeight) * ( *(pSrc + 1) - *pSrc );

        pSrc += stride; pWeight += stride; pCoeffCalc += stride;
      }

      // Update dst
      for (int j = 0; j < BLOCK_HEIGHT && jUpdate < height; j++, jUpdate++) {
        float update = 0.0;

        if (col != 0)
          update -= *(pCoeffUpdate - 1);
        if (col != width - 1)
          update += *pCoeffUpdate;

        *pDst += update;

        pDst += stride; pCoeffUpdate += stride;
      }
    }
  }
}

__global__ void kernelSubLaplacianHorizFillCoeffs(
    float *src, float *weights, float *coeffs, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int col  = tidx % stride;

  // Do not calculate the last column
  if (tidx < width && col != width - 1) {
    float *pSrc    = src + tidx,
          *pWeight = weights + tidx,
          *pCoeff  = coeffs + tidx;

    for (int j = 0; j < height; j++) {
      *pCoeff = (*pWeight) * ( *(pSrc + 1) - *pSrc );

      pSrc += stride; pWeight += stride; pCoeff += stride;
    }
  }
}

__global__ void kernelSubLaplacianHorizApplyCoeffs(
    float *dst, float *coeffs, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int col  = tidx % stride;

  if (tidx < width) {

    float *pDst   = dst + tidx,
          *pCoeff = coeffs + tidx;

    for (int j = 0; j < height; j++) {
      float update = 0.0;

      if (col != 0)
        update -= *(pCoeff - 1);
      if (col != width - 1)
        update += *pCoeff;

      *pDst += update;

      pDst += stride; pCoeff += stride;
    }
  }

  // if (col < width) {

  //   float *pDst   = dst + tidx,
  //         *pCoeff = coeffs + tidx;

  //   float update = 0.0;

  //   if (col != 0)
  //     update -= *(pCoeff - 1);
  //   if (col != width - 1)
  //     update += *pCoeff;

  //   *pDst += update;
  // }
}

__global__ void kernelSorStep(
    float *du, float *dv,
    float *a11, float *a12, float *a22,
    const float *b1, const float *b2,
    const float *horiz, const float *vert,
    const int iterations, const float omega,
    int height, int width, int stride, bool odd) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = tidx / width;
  int i  = tidx % width;

  bool shouldRun = (odd)
    ? ((i + j) % 2 == 1)
    : ((i + j) % 2 == 0);

  if (tidx < width * height && shouldRun) {

    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2;
    sigma_u = 0.0f;
    sigma_v = 0.0f;
    sum_dpsis = 0.0f;

    int here  = j * stride + i;
    int left  = j * stride + i - 1;
    int right = j * stride + i + 1;
    int up    = (j-1) * stride + i;
    int down  = (j+1) * stride + i;

    if(j>0)
    {
      sigma_u   -= vert[up] * du[up];
      sigma_v   -= vert[up] * dv[up];
      sum_dpsis += vert[up];
    }
    if(i>0)
    {
      sigma_u   -= horiz[left] * du[left];
      sigma_v   -= horiz[left] * dv[left];
      sum_dpsis += horiz[left];
    }
    if(j<height-1)
    {
      sigma_u   -= vert[here] * du[down];
      sigma_v   -= vert[here] * dv[down];
      sum_dpsis += vert[here];
    }
    if(i<width-1)
    {
      sigma_u   -= horiz[here] * du[right];
      sigma_v   -= horiz[here] * dv[right];
      sum_dpsis += horiz[here];
    }

    A11 = a11[here] + sum_dpsis;
    A12 = a12[here];
    A22 = a22[here] + sum_dpsis;

    B1 = b1[here] - sigma_u;
    B2 = b2[here] - sigma_v;

    du[here] = (1.0f-omega) * du[here] + omega/A11 * (B1 - A12 * dv[here]);
    dv[here] = (1.0f-omega) * dv[here] + omega/A22 * (B2 - A12 * du[here]);

  }
}

__global__ void kernelGetMeanImageAndDiff(
    float *img1, float *img2, float *avgImg, float *diff,
    int height, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  // For 3 channels images
  if (tidx < 3 * stride) {
    float
      *pImg1   = img1 + tidx,
      *pImg2   = img2 + tidx,
      *pAvgImg = avgImg + tidx,
      *pDiff   = diff + tidx;

    for (int j = 0; j < height; j++) {
      *pAvgImg = 0.5 * ((*pImg1) + (*pImg2));
      *pDiff   = (*pImg2) - (*pImg1);

      pImg1 += stride; pImg2 += stride; pAvgImg += stride; pDiff += stride;
    }

  }

}


__global__ void kernelFlowMag(
    float *dst,  float *ux,  float *uy,  float *vx,  float *vy,
    float qa, float epsmooth, int height, int width, int stride, int N) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (; i < N; i+= blockDim.x * gridDim.x) {
    dst[i] = qa / sqrt(
        (ux[i])*(ux[i]) + (uy[i])*(uy[i]) + (vx[i])*(vx[i]) + (vy[i])*(vy[i]) + epsmooth );
  }

  // if (tidx < stride) {
  //   float *uxp = ux + tidx,
  //         *uyp = uy + tidx,
  //         *vxp = vx + tidx,
  //         *vyp = vy + tidx,
  //         *sp  = dst + tidx;

  //   for (int j = 0; j < height; j++) {
  //     *sp = qa / sqrtf(
  //         (*uxp)*(*uxp) + (*uyp)*(*uyp) + (*vxp)*(*vxp) + (*vyp)*(*vyp) + epsmooth );

  //     uxp += stride; uyp += stride; vxp += stride; vyp += stride; sp += stride; 
  //   }
  // }
}

__global__ void kernelSmoothnessHorizVert(
    float *dst_horiz, float *dst_vert, float *smoothness, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < height * stride) {

    float *dst_horiz_p = dst_horiz + tidx,
          *dst_vert_p  = dst_vert  + tidx,
          *sp          = smoothness + tidx;

    *dst_horiz_p = *sp + *(sp + 1);
    *dst_vert_p  = *sp + *(sp + stride);
  }
}


__global__ void kernelFlowUpdate(
    float *uu, float *vv, float *wx, float *wy, float *du, float *dv,
    int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < height * stride) {

    float *uup = uu + tidx,
          *vvp = vv + tidx,
          *wxp = wx + tidx,
          *wyp = wy + tidx,
          *dup = du + tidx,
          *dvp = dv + tidx;

    (*uup) = (*wxp) + (*dup);
    (*vvp) = (*wyp) + (*dvp);
  }

}


__global__ void kernelWarpImage(
    float *dst1, float *dst2, float *dst3, float *mask,
    float *src1, float *src2, float *src3,
    float *wx, float *wy,
    int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  int i = tidx % stride;
  int j = tidx / stride;
  int offset = j * stride + i;

  if (i < width && j < height) {
    float xx = i + wx[offset];
    float yy = j + wy[offset];
    int x = floor(xx);
    int y = floor(yy);
    float dx = xx - x;
    float dy = yy - y;

    // Set mask according to bounds
    mask[offset] = (xx >= 0 && xx < width && yy >= 0 && yy < height);

    int x1 = MINMAX_TA(x, width);
    int x2 = MINMAX_TA(x + 1, width);
    int y1 = MINMAX_TA(y, height);
    int y2 = MINMAX_TA(y + 1, height);

    dst1[offset] = 
      src1[y1 * stride + x1] * (1.0f-dx) * (1.0f-dy) +
      src1[y1 * stride + x2] * dx * (1.0f-dy) +
      src1[y2 * stride + x1] * (1.0f-dx) * dy +
      src1[y2 * stride + x2] * dx * dy;
    dst2[offset] = 
      src2[y1 * stride + x1] * (1.0f-dx) * (1.0f-dy) +
      src2[y1 * stride + x2] * dx * (1.0f-dy) +
      src2[y2 * stride + x1] * (1.0f-dx) * dy +
      src2[y2 * stride + x2] * dx * dy;
    dst3[offset] = 
      src3[y1 * stride + x1] * (1.0f-dx) * (1.0f-dy) +
      src3[y1 * stride + x2] * dx * (1.0f-dy) +
      src3[y2 * stride + x1] * (1.0f-dx) * dy +
      src3[y2 * stride + x2] * dx * dy;

  }
}


__global__ void kernelSepFlow(
    float *flowx, float *flowy, float *flowout, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  int ix = tidx % width;
  int iy = tidx / width;

  if (ix < width && iy < height) {
    
    int i = iy * width + ix;
    int is = iy * stride + ix;

    flowx[is] = flowout[2 * i];
    flowy[is] = flowout[2 * i + 1];
  }
}


__global__ void kernelMergeFlow(
    float *flowx, float *flowy, float *flowout, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  int ix = tidx % width;
  int iy = tidx / width;

  if (ix < width && iy < height) {
    
    int i = iy * width + ix;
    int is = iy * stride + ix;

    flowout[2 * i] = flowx[is];
    flowout[2 * i + 1] = flowy[is];
  }
}

__global__ void kernelCopyImage(
    float *dst1, float *dst2, float *dst3, const float *src,
    int width_pad, int padding, int height, int width, int stride) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  const float *pSrcStart = src + 3 * (width_pad + 1) * padding;

  int ix = tidx % width;
  int iy = tidx / width;

  if (ix < width && iy < height) {

    const float *pSrc = pSrcStart + (iy * 3 * width_pad) + 3 * ix;
    int i = iy * stride + ix;

    dst1[i] = *pSrc; pSrc++;
    dst2[i] = *pSrc; pSrc++;
    dst3[i] = *pSrc; pSrc++;
  }
}


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
      const float half_delta_over3, const float half_beta, const float half_gamma_over3) {

    checkCudaErrors( cudaMemset(a11->c1, 0, sizeof(float)*uu->height*uu->stride) );
    checkCudaErrors( cudaMemset(a12->c1, 0, sizeof(float)*uu->height*uu->stride) );
    checkCudaErrors( cudaMemset(a22->c1, 0, sizeof(float)*uu->height*uu->stride) );
    checkCudaErrors( cudaMemset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride) );
    checkCudaErrors( cudaMemset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride) );

    // Set up device pointers
    float *a11c1,
          *a12c1,    *a22c1,
          *b1c1,     *b2c1, 
          *maskc1, 
          *wxc1,     *wyc1,
          *duc1,     *dvc1, 
          *uuc1,     *vvc1, 
          *Ixc1,     *Ixc2,     *Ixc3,
          *Iyc1,     *Iyc2,     *Iyc3,
          *Izc1,     *Izc2,     *Izc3,
          *Ixxc1,    *Ixxc2,    *Ixxc3,
          *Ixyc1,    *Ixyc2,    *Ixyc3,
          *Iyyc1,    *Iyyc2,    *Iyyc3,
          *Ixzc1,    *Ixzc2,    *Ixzc3,
          *Iyzc1,    *Iyzc2,    *Iyzc3;

    a11c1  =  a11->c1;
    a12c1  =  a12->c1;
    a22c1  =  a22->c1;
    b1c1   =  b1->c1;
    b2c1   =  b2->c1;
    maskc1 =  mask->c1;
    wxc1   =  wx->c1;
    wyc1   =  wy->c1;
    duc1   =  du->c1;
    dvc1   =  dv->c1;
    uuc1   =  uu->c1;
    vvc1   =  vv->c1;
    Ixc1   =  Ix->c1;
    Ixc2   =  Ix->c2;
    Ixc3   =  Ix->c3;
    Iyc1   =  Iy->c1;
    Iyc2   =  Iy->c2;
    Iyc3   =  Iy->c3;
    Izc1   =  Iz->c1;
    Izc2   =  Iz->c2;
    Izc3   =  Iz->c3;
    Ixxc1  =  Ixx->c1;
    Ixxc2  =  Ixx->c2;
    Ixxc3  =  Ixx->c3;
    Ixyc1  =  Ixy->c1;
    Ixyc2  =  Ixy->c2;
    Ixyc3  =  Ixy->c3;
    Iyyc1  =  Iyy->c1;
    Iyyc2  =  Iyy->c2;
    Iyyc3  =  Iyy->c3;
    Ixzc1  =  Ixz->c1;
    Ixzc2  =  Ixz->c2;
    Ixzc3  =  Ixz->c3;
    Iyzc1  =  Iyz->c1;
    Iyzc2  =  Iyz->c2;
    Iyzc3  =  Iyz->c3;

    int N = uu->height*uu->stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelDataTerm<<<nBlocks, nThreadsPerBlock>>>(
        a11c1, a12c1, a22c1,
        b1c1, b2c1, 
        maskc1, 
        wxc1, wyc1,
        duc1, dvc1, 
        uuc1, vvc1, 
        Ixc1,    Ixc2,    Ixc3,
        Iyc1,    Iyc2,    Iyc3,
        Izc1,    Izc2,    Izc3,
        Ixxc1,   Ixxc2,   Ixxc3,
        Ixyc1,   Ixyc2,   Ixyc3,
        Iyyc1,   Iyyc2,   Iyyc3,
        Ixzc1,   Ixzc2,   Ixzc3,
        Iyzc1,   Iyzc2,   Iyzc3, 
        half_delta_over3, half_beta, half_gamma_over3, N);

  };

  void subLaplacian(
      image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert, float *coeffs) {

    cu::subLaplacianHoriz(src->c1, dst->c1, weight_horiz->c1, coeffs, src->height, src->width, src->stride);
    cu::subLaplacianVert(src->c1, dst->c1, weight_vert->c1, src->height, src->stride);

  }

  void subLaplacianHoriz(
      float *src, float *dst, float *weights, float *coeffs, int height, int width, int stride) {

    float *pDeviceCoeffs = coeffs;

    float *pDeviceSrc = src,
          *pDeviceDst = dst,
          *pDeviceWeights = weights;

    int N = width;
    // int N = height * stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    auto start_horiz = now();

    kernelSubLaplacianHorizFillCoeffs<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceSrc, pDeviceWeights, pDeviceCoeffs, height, width, stride);

    kernelSubLaplacianHorizApplyCoeffs<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceDst, pDeviceCoeffs, height, width, stride);

    // kernelSubLaplacianHoriz<<<nBlocks, nThreadsPerBlock>>>(
    //     pDeviceSrc, pDeviceDst, pDeviceWeights, pDeviceCoeffs, height, width, stride);
    calc_print_elapsed("laplacian horiz", start_horiz);
  }

  void subLaplacianVert(
      float *src, float *dst, float *weights, int height, int stride) {

    int N = stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    auto start_vert = now();
    kernelSubLaplacianVert<<<nBlocks, nThreadsPerBlock>>>(
        src, src + stride, dst, dst + stride, weights, height, stride);
    calc_print_elapsed("laplacian vert", start_vert);
  }

  void sor(
      float *du, float *dv,
      float *a11, float *a12, float *a22,
      float *b1, float *b2,
      float *horiz, float *vert,
      int iterations, float omega,
      int height, int width, int stride) {

    // Device setup
    float 
      *d_du,
    *d_dv,
    *d_a11,
    *d_a12,
    *d_a22,
    *d_b1,
    *d_b2,
    *d_horiz,
    *d_vert;

    d_du    = du;
    d_dv    = dv;
    d_a11   = a11;
    d_a12   = a12;
    d_a22   = a22;
    d_b1    = b1;
    d_b2    = b2;
    d_horiz = horiz;
    d_vert  = vert;

    int N = width * height;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    for(int iter = 0 ; iter<iterations ; iter++)
    {

      auto start_sor_odd = now();
      kernelSorStep<<<nBlocks, nThreadsPerBlock>>>(
          d_du, d_dv,
          d_a11, d_a12, d_a22,
          d_b1, d_b2,
          d_horiz, d_vert,
          iterations, omega,
          height, width, stride, true);

      cudaDeviceSynchronize();
      calc_print_elapsed("sor step odd", start_sor_odd);

      auto start_sor_even = now();
      kernelSorStep<<<nBlocks, nThreadsPerBlock>>>(
          d_du, d_dv,
          d_a11, d_a12, d_a22,
          d_b1, d_b2,
          d_horiz, d_vert,
          iterations, omega,
          height, width, stride, false);
      cudaDeviceSynchronize();
      calc_print_elapsed("sor step even", start_sor_even);
    }  
  }

  void getMeanImageAndDiff(
      float *img1, float *img2, float *avgImg, float *diff,
      int height, int stride) {

    int N = 3 * stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelGetMeanImageAndDiff<<<nBlocks, nThreadsPerBlock>>>(
        img1, img2, avgImg, diff,
        height, stride);

  }

  void colorImageDerivative(
      float *dst, float *src, float *pDeviceColorDerivativeKernel, 
      int height, int width, int stride, bool horiz) {

    Npp32f *pDeviceSrc = src;
    Npp32f *pDeviceDst = dst;

    size_t elemSize = sizeof(float);
    unsigned int nSrcStep = stride * elemSize;
    unsigned int nDstStep = nSrcStep;

    NppiSize oSrcSize = { width, height };
    NppiPoint oSrcOffset = { 0, 0 };
    NppiSize oSizeROI = { width, height };
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    NPP_CHECK_NPP(
        (horiz)
        ? nppiFilterRowBorder_32f_C1R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI,
          pDeviceColorDerivativeKernel, 5, 2, eBorderType)
        : nppiFilterColumnBorder_32f_C1R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI,
          pDeviceColorDerivativeKernel, 5, 2, eBorderType)
        );
  }

  // Expects filter kernel of the form
  //   { -0.5, 0.0, 0.5 }
  void imageDerivative(
      float *dst, float *src, float *pDeviceDerivativeKernel, 
      int height, int width, int stride, bool horiz) {

    Npp32f *pDeviceSrc = src;
    Npp32f *pDeviceDst = dst;

    size_t elemSize = sizeof(float);
    unsigned int nSrcStep = stride * elemSize;
    unsigned int nDstStep = nSrcStep;

    NppiSize oSrcSize = { width, height };
    NppiPoint oSrcOffset = { 0, 0 };
    NppiSize oSizeROI = { width, height };
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    NPP_CHECK_NPP(
        (horiz)
        ? nppiFilterRowBorder_32f_C1R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI,
          pDeviceDerivativeKernel, 3, 1, eBorderType)
        : nppiFilterColumnBorder_32f_C1R (
          pDeviceSrc, nSrcStep, oSrcSize, oSrcOffset,
          pDeviceDst, nDstStep, oSizeROI,
          pDeviceDerivativeKernel, 3, 1, eBorderType)
        );
  }

  void smoothnessTerm(
      float *dst_horiz, float *dst_vert, float *smoothness,
      float *ux,  float *uy,  float *vx,  float *vy,
      float qa, float epsmooth,
      int height, int width, int stride) {

    int N = height * width;
    int nThreadsPerBlock = 64;
    int nBlocks = 56;

    auto start_mag = now();
    kernelFlowMag<<<nBlocks, nThreadsPerBlock>>> (
        smoothness, ux, uy, vx, vy,
        qa, epsmooth, height, width, stride, N);
    cudaDeviceSynchronize();
    calc_print_elapsed("smoothnessTerm magnitude", start_mag);

    N = height * stride;
    nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    auto start_horizvert = now();
    kernelSmoothnessHorizVert<<< nBlocks, nThreadsPerBlock >>> (
        dst_horiz, dst_vert, smoothness, height, width, stride);
    cudaDeviceSynchronize();
    calc_print_elapsed("smoothnessTerm horiz vert", start_horizvert);

  }


  void flowUpdate(
      float *uu, float *vv, float *wx, float *wy, float *du, float *dv,
      int height, int width, int stride) {

    int N = height * stride;
    int nThreadsPerBlock = 128;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelFlowUpdate<<< nBlocks, nThreadsPerBlock >>> (
        uu, vv, wx, wy, du, dv,
        height, width, stride);

  }

  /*
     Warp an image `src` into `dst` using warp vectors `wx`, `wy`.
     Store `mask[i]` = 0 or 1 if pixel i goes outisde or inside image bounds.
   */
  void warpImage(
      color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy) {

    int N = src->height * src->stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelWarpImage<<< nBlocks, nThreadsPerBlock >>> (
        dst->c1, dst->c2, dst->c3, mask->c1,
        src->c1, src->c2, src->c3,
        wx->c1,  wy->c1,  src->height, src->width, src->stride);
  }

  
  void computeSmoothness(
      image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, float *deriv_flow,
      image_t *ux, image_t *uy, image_t *vx, image_t *vy, image_t *smoothness, 
      const float quarter_alpha) {

    auto start_setup = now();
    const int width = uu->width, height = vv->height, stride = uu->stride;
    calc_print_elapsed("smoothness setup", start_setup);

    // compute derivatives [-0.5 0 0.5]
    auto start_derivs = now();
    cu::imageDerivative(ux->c1, uu->c1, deriv_flow, height, width, stride, true);
    cu::imageDerivative(vx->c1, vv->c1, deriv_flow, height, width, stride, true);
    cu::imageDerivative(uy->c1, uu->c1, deriv_flow, height, width, stride, false);
    cu::imageDerivative(vy->c1, vv->c1, deriv_flow, height, width, stride, false);
    calc_print_elapsed("smoothness derivatives", start_derivs);

    auto start_calc = now();
    cu::smoothnessTerm(
        dst_horiz->c1, dst_vert->c1, smoothness->c1,
        ux->c1, uy->c1, vx->c1, vy->c1,
        quarter_alpha, epsilon_smooth,
        height, width, stride);
    calc_print_elapsed("smoothness term", start_calc);

    // Cleanup extra columns
    auto start_cleanup = now();
    // // Doesn't really affect the output
    // for(int j = 0; j < height; j++){
    //   // memset(&dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1));
    //   checkCudaErrors( cudaMemset(&dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1)) );
    // }
    checkCudaErrors( cudaMemset( &dst_vert->c1[(height-1)*stride], 0, sizeof(float)*stride) );
    calc_print_elapsed("smoothness cleanup", start_cleanup);
  }

  void getDerivatives(
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

  void sepFlow(std::vector<image_t*> flow_sep, float *flowout, int height, int width) {

    int N = width * height;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelSepFlow<<< nBlocks, nThreadsPerBlock >>>(
        flow_sep[0]->c1, flow_sep[1]->c1, flowout, height, width, flow_sep[0]->stride);

  }

  void mergeFlow(std::vector<image_t*> flow_sep, float *flowout, int height, int width) {

    int N = width * height;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelMergeFlow<<< nBlocks, nThreadsPerBlock >>>(
        flow_sep[0]->c1, flow_sep[1]->c1, flowout, height, width, flow_sep[0]->stride);
  }

  void copyImage(color_image_t *dst, const float *src, int width_pad, int padding, int height, int width) {

    int N = width * height;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelCopyImage<<< nBlocks, nThreadsPerBlock >>> (
        dst->c1, dst->c2, dst->c3, src,
        width_pad, padding, height, width, dst->stride);
  }

}
