/**
 * RGB (3 channel) 32-bit float image containers.
 */

#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "RgbMat.h"


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

