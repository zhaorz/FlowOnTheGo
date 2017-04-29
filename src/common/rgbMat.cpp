/**
 * RGB (3 channel) 32-bit float image containers.
 */

#include <iostream>
#include <chrono>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "rgbMat.h"


rgbMat::rgbMat(int _height, int _width) : height(_height), width(_width) {
  data = new float[height * width * pixel_width];
}


rgbMat::rgbMat(cv::Mat cvMat) {
  if (cvMat.type() != CV_32FC3) {
    throw std::invalid_argument("rgbMat: invalid input matrix type");
  }

  if (cvMat.elemSize() != pixel_width) {
    throw std::invalid_argument("rgbMat: elem size does not match expected pixel_width");
  }

  cv::Size sz = cvMat.size();
  height = sz.height;
  width  = sz.width;

  size_t n_bytes = height * width * pixel_width;

  data = new float[height * width * pixel_width];

}

