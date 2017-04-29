/**
 * RGB (3 channel) 32-bit float image containers.
 */

#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include <opencv2/opencv.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helper.h"

#include "RgbMat.h"


/******************************************************************************/
/* RgbMat                                                                     */
/******************************************************************************/

RgbMat::RgbMat(int _height, int _width) : height(_height), width(_width) {
  data = new float[height * width * channels];
}

RgbMat::RgbMat(cv::Mat cvMat) {
  if (cvMat.type() != CV_32FC3) {
    throw std::invalid_argument("RgbMat: invalid input matrix type");
  }

  if (cvMat.elemSize() != elemSize) {
    throw std::invalid_argument("RgbMat: elem size does not match expected");
  }

  if (cvMat.channels() != channels) {
    throw std::invalid_argument("RgbMat: channels does not match expected");
  }

  if (!cvMat.isContinuous()) {
    throw std::invalid_argument("RgbMat: matrix not continuous");
  }

  cv::Size sz = cvMat.size();
  height = sz.height;
  width  = sz.width;

  data = new float[height * width * channels];

  std::memcpy(data, cvMat.data, height * width * elemSize);
}

RgbMat::~RgbMat() {
  delete[] data;
}

cv::Mat RgbMat::toMat() {
  cv::Mat I(height, width, CV_32FC3);
  std::memcpy(I.data, data, height * width * elemSize);
  return I;
}

/******************************************************************************/
/* GpuRgbMat                                                                  */
/******************************************************************************/

GpuRgbMat::GpuRgbMat(int _height, int _width) : height(_height), width(_width) {
  checkCudaErrors( cudaMalloc((void**) &data, height * width * elemSize) );
}

GpuRgbMat::GpuRgbMat(RgbMat cpuMat) : height(cpuMat.height), width(cpuMat.width) {
  std::cout << "GpuRgbMat allocating " << width << "x" << height << " elemSize = " << elemSize
    << " to " << data << std::endl;
  checkCudaErrors( cudaMalloc((void**) &data, height * width * elemSize) );
  std::cout << "GpuRgbMat allocated to " << data << std::endl;

  upload(cpuMat);
}

GpuRgbMat::~GpuRgbMat() {
  std::cout << "GpuRgbMat destructor called" << std::endl;
  checkCudaErrors( cudaFree(data) );
}

void GpuRgbMat::upload(RgbMat cpuMat) {
  if ((cpuMat.width != width || cpuMat.height != height)) {
    throw std::invalid_argument("GpuRgbMat::upload dimension mismatch");
  }

  std::cout << "uploading " << width << "x" << height << " elemSize = " << elemSize
    << " from " << cpuMat.data << " to " << data << std::endl;

  cudaMemcpy(data, cpuMat.data, width * height * elemSize, cudaMemcpyHostToDevice);

  // checkCudaErrors(
  //     cudaMemcpy(data, cpuMat.data, width * height * elemSize, cudaMemcpyHostToDevice) );
}

void GpuRgbMat::download(RgbMat cpuMat) {
  if ((cpuMat.width != width || cpuMat.height != height)) {
    throw std::invalid_argument("GpuRgbMat::upload dimension mismatch");
  }

  std::cout << "downloading " << width << "x" << height << " elemSize = " << elemSize
    << " from " << data << " to " << cpuMat.data << std::endl;
  checkCudaErrors(
      cudaMemcpy(cpuMat.data, data, width * height * elemSize, cudaMemcpyDeviceToHost) );
}


