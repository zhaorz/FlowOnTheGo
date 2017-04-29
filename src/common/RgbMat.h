/**
 * RGB (3 channel) 32-bit float image containers.
 */

#ifndef __RGB_MAT_H__
#define __RGB_MAT_H__

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

class RgbMat {

  /**
   * RgbMat represents 2D RGB (3 channel) images as flat arrays of floats.
   */

  public:

    RgbMat(int height, int width);
    RgbMat(cv::Mat cvMat);
    ~RgbMat();

    int height;
    int width;

    cv::Mat toMat();

    const int channels = 3;
    const size_t elemSize = channels * sizeof(float);
    float* data;

};

class GpuRgbMat {

  public:

    GpuRgbMat(int height, int width);
    GpuRgbMat(RgbMat cpuMat);
    ~GpuRgbMat();

    void upload(RgbMat I);
    void download(RgbMat I);

    int height;
    int width;

    const int channels = 3;
    const size_t elemSize = channels * sizeof(float);
    float* data;

};

#endif // end __RGB_MAT_H__


