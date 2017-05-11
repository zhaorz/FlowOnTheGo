// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

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

  const int BLOCK_HEIGHT = 4;

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
  int col  = tidx % width;

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
  int col  = tidx % width;

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

    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
    memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

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

    checkCudaErrors( cudaHostGetDevicePointer(&a11c1,    a11->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&a12c1,    a12->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&a22c1,    a22->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&b1c1,     b1->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&b2c1,     b2->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&maskc1,   mask->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&wxc1,     wx->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&wyc1,     wy->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&duc1,     du->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&dvc1,     dv->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&uuc1,     uu->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&vvc1,     vv->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixc1,     Ix->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixc2,     Ix->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixc3,     Ix->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyc1,     Iy->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyc2,     Iy->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyc3,     Iy->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Izc1,     Iz->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Izc2,     Iz->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Izc3,     Iz->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixxc1,    Ixx->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixxc2,    Ixx->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixxc3,    Ixx->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixyc1,    Ixy->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixyc2,    Ixy->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixyc3,    Ixy->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyyc1,    Iyy->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyyc2,    Iyy->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyyc3,    Iyy->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixzc1,    Ixz->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixzc2,    Ixz->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Ixzc3,    Ixz->c3, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyzc1,    Iyz->c1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyzc2,    Iyz->c2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&Iyzc3,    Iyz->c3, 0) );

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


  void subLaplacianHoriz(
      float *src, float *dst, float *weights, int height, int width, int stride) {

    auto start_coeff_malloc = now();
    float *pDeviceCoeffs;
    checkCudaErrors( cudaMalloc((void**) &pDeviceCoeffs, height * stride * sizeof(float)) );
    calc_print_elapsed("laplacian coeff malloc", start_coeff_malloc);

    // Setup device pointers
    auto start_device_ptr = now();
    float *pDeviceSrc, *pDeviceDst, *pDeviceWeights;
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceSrc, src, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceDst, dst, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceWeights, weights, 0) );
    calc_print_elapsed("laplacian get device ptrs", start_device_ptr);

    int N = width;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    auto start_kernels = now();
    kernelSubLaplacianHoriz<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceSrc, pDeviceDst, pDeviceWeights, pDeviceCoeffs, height, width, stride);
    calc_print_elapsed("laplacian kernels", start_kernels);

    auto start_free = now();
    cudaFree(pDeviceCoeffs);
    calc_print_elapsed("laplacian cudaFree", start_free);
  }

  void subLaplacianVert(
      float *src, float *dst, float *weights, int height, int stride) {

    float *d_src, *d_dst, *d_weights;

    checkCudaErrors( cudaHostGetDevicePointer(&d_src, src, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_dst, dst, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_weights, weights, 0) );

    int N = stride;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    kernelSubLaplacianVert<<<nBlocks, nThreadsPerBlock>>>(
        d_src, d_src + stride, d_dst, d_dst + stride, d_weights, height, stride);

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

    checkCudaErrors( cudaHostGetDevicePointer(&d_du,    du, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_dv,    dv, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_a11,   a11, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_a12,   a12, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_a22,   a22, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_b1,    b1, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_b2,    b2, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_horiz, horiz, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&d_vert,  vert, 0) );

    int N = width * height;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    for(int iter = 0 ; iter<iterations ; iter++)
    {

      kernelSorStep<<<nBlocks, nThreadsPerBlock>>>(
          d_du, d_dv,
          d_a11, d_a12, d_a22,
          d_b1, d_b2,
          d_horiz, d_vert,
          iterations, omega,
          height, width, stride, true);

      cudaDeviceSynchronize();

      kernelSorStep<<<nBlocks, nThreadsPerBlock>>>(
          d_du, d_dv,
          d_a11, d_a12, d_a22,
          d_b1, d_b2,
          d_horiz, d_vert,
          iterations, omega,
          height, width, stride, false);
    }  
  }

}
