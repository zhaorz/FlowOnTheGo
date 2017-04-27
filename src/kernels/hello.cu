/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "hello.h"
#include "get_device.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void helloKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

void launchHelloKernel() {

  int devID;
  cudaDeviceProp props;

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(0, (const char **) NULL);

  //Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
      devID, props.name, props.major, props.minor);

  printf("printf() is called. Output:\n\n");

  //Kernel configuration, where a two-dimensional grid and
  //three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2, 2, 2);
  helloKernel<<<dimGrid, dimBlock>>>(10);
  cudaDeviceSynchronize();

}
