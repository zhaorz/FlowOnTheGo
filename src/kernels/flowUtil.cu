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

__global__ void kernelSubLaplacianHorizFillCoeffs(
    float *src, float *weights, float *coeffs, int height, int width) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int col  = tidx % width;

  // Do not calculate the last column
  if (tidx < height * width && col != width - 1) {
    float *pSrc    = src + tidx,
          *pWeight = weights + tidx,
          *pCoeff  = coeffs + tidx;

    *pCoeff = (*pWeight) * ( *(pSrc + 1) - *pSrc );
  }
}

__global__ void kernelSubLaplacianHorizApplyCoeffs(
    float *dst, float *coeffs, int height, int width) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int col  = tidx % width;

  if (tidx < height * width) {

    float *pDst   = dst + tidx,
          *pCoeff = coeffs + tidx;

    float update = 0.0;

    if (col != 0)
      update -= *(pCoeff - 1);
    if (col != width - 1)
      update += *pCoeff;

    *pDst += update;
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
    float *pDeviceSrc, *pDeviceDst, *pDeviceWeights;
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceSrc, src, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceDst, dst, 0) );
    checkCudaErrors( cudaHostGetDevicePointer(&pDeviceWeights, weights, 0) );

    int N = height * width;
    int nThreadsPerBlock = 64;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;

    auto start_kernels = now();
    kernelSubLaplacianHorizFillCoeffs<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceSrc, pDeviceWeights, pDeviceCoeffs, height, width);

    kernelSubLaplacianHorizApplyCoeffs<<<nBlocks, nThreadsPerBlock>>>(
        pDeviceDst, pDeviceCoeffs, height, width);
    calc_print_elapsed("laplacian kernels", start_kernels);

    cudaFree(pDeviceCoeffs);

    // const int offset = stride - width;

    // float *src_ptr = src,
    //       *dst_ptr = dst,
    //       *weight_horiz_ptr = weights;

    // float *coeffs = new float[height * stride];

    // float *coeffs_ptr = coeffs;

    // // Calculate coeffs
    // for(int j = 0; j < height; j++) { // faster than for(j=0;j<src->height;j++)
    //   for(int i = 0; i < width - 1; i++) {
    //     float tmp = (*weight_horiz_ptr)*((*(src_ptr+1))-(*src_ptr));

    //     *coeffs_ptr = tmp;
    //     src_ptr++;
    //     weight_horiz_ptr++;
    //     coeffs_ptr++;
    //   }
    //   src_ptr += offset+1;
    //   weight_horiz_ptr += offset+1;
    //   coeffs_ptr += offset+1;
    // }

    // coeffs_ptr = coeffs;

    // // Apply
    // for(int j = 0; j < height; j++) { // faster than for(j=0;j<src->height;j++)
    //   for(int i = 0; i < width; i++) {
    //     float update = 0.0;

    //     if (i != width - 1)
    //       update += *coeffs_ptr;
    //     if (i != 0)
    //       update -= *(coeffs_ptr-1);

    //     *dst_ptr += update;
    //     dst_ptr++;
    //     coeffs_ptr++;
    //   }
    //   dst_ptr += offset;
    //   coeffs_ptr += offset;
    // }

    // delete[] coeffs;
  }

  // TODO: Non-deterministic, see what's up
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

}
