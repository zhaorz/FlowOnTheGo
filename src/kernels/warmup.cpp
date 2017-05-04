/**
 * Implements a warmup kernel
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_filtering_functions.h>

// Local
#include "../common/timer.h"

#include "warmup.h"

using namespace timer;

namespace cu {

  /**
   * Warms up the GPU by performing a malloc and memcpy and launching a kernel.
   */
  void warmup() {

    // Compute time of relevant kernel
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 1;
    size_t elemSize = channels * sizeof(short); // 16 bits

    int width = 720;
    int height = 480;
    NppiSize oSizeROI = { width, height };

    std::cout << "[start] warmup: processing " << width << "x" << height << " image" << std::endl;

    // pSrc pointer to image data
    Npp16s* pHostSrc = new short[width * height];
    for (int i = 0; i < width * height; i++) {
      pHostSrc[i] = i;
    }

    // The width, in bytes, of the image, sometimes referred to as pitch
    unsigned int nSrcStep = width * elemSize;
    unsigned int nDstStep = width * sizeof(char); // 8-bit elements

    auto start_cuda_malloc = now();

    // Allocate device memory
    Npp16s* pDeviceSrc;
    Npp8u*  pDeviceDst;

    checkCudaErrors( cudaMalloc((void**) &pDeviceSrc, width * height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceDst, width * height * sizeof(char)) );

    calc_print_elapsed("cudaMalloc", start_cuda_malloc);


    auto start_memcpy_hd = now();

    // Copy image to device
    checkCudaErrors(
        cudaMemcpy(pDeviceSrc, pHostSrc, width * height * elemSize, cudaMemcpyHostToDevice) );

    calc_print_elapsed("cudaMemcpy H->D", start_memcpy_hd);


    auto start_warmup = now();

    NPP_CHECK_NPP(

        nppiConvert_16s8u_C1R (pDeviceSrc, nSrcStep, pDeviceDst, nDstStep, oSizeROI)

        );

    compute_time += calc_print_elapsed("warmup", start_warmup);


    auto start_memcpy_dh = now();

    // Copy result to host
    char* pHostDst = new char[width * height];

    checkCudaErrors(
        cudaMemcpy(pHostDst, pDeviceDst,
          width * height * sizeof(char), cudaMemcpyDeviceToHost) );

    calc_print_elapsed("cudaMemcpy H<-D", start_memcpy_dh);

    cudaFree((void*) pDeviceSrc);
    cudaFree((void*) pDeviceDst);

    delete[] pHostDst;

    std::cout << "[done] warmup: primary compute time: " << compute_time << " (ms)" << std::endl;
  }

}

