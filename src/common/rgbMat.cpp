/**
 * RGB (3 channel) 32-bit float image containers.
 */

#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "rgbMat.h"


rgbMat::rgbMat(int _height, int _width) : height(_height), width(_width) {
  data = new float[height * width * channels];
}


rgbMat::rgbMat(cv::Mat cvMat) {
  if (cvMat.type() != CV_32FC3) {
    throw std::invalid_argument("rgbMat: invalid input matrix type");
  }

  if (cvMat.elemSize() != elemSize) {
    throw std::invalid_argument("rgbMat: elem size does not match expected");
  }

  if (cvMat.channels() != channels) {
    throw std::invalid_argument("rgbMat: channels does not match expected");
  }

  if (!cvMat.isContinuous()) {
    throw std::invalid_argument("rgbMat: matrix not continuous");
  }

  cv::Size sz = cvMat.size();
  height = sz.height;
  width  = sz.width;

  data = new float[height * width * channels];

  std::memcpy(data, cvMat.data, height * width * elemSize);
}

rgbMat::~rgbMat() {
  delete[] data;
}

cv::Mat rgbMat::toCVMat() {
  cv::Mat I(height, width, CV_32FC3);
  std::memcpy(I.data, data, height * width * elemSize);
  return I;
}

