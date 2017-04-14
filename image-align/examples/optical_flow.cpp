/**
   This file is part of Image Alignment.

   Copyright Christoph Heindl 2015

   Image Alignment is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Image Alignment is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with Image Alignment.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <imagealign/imagealign.h>
IA_DISABLE_PRAGMA_WARN(4190)
IA_DISABLE_PRAGMA_WARN(4244)
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/optflow.hpp>
IA_DISABLE_PRAGMA_WARN_END
IA_DISABLE_PRAGMA_WARN_END
#include <iomanip>
#include <iostream>
#include <chrono>
#include <math.h>

/**
   This example is based on OpenCVs Lucas Kanade Optical Flow example.
   It demonstrates how Image Alignment can be used to perform optical flow.
*/

namespace ia = imagealign;

const int MAX_FEATURES = 100;

template<class Scalar>
cv::Point_<Scalar> toP(const cv::Matx<Scalar, 2, 1> &p) {
  return cv::Point_<Scalar>(p(0), p(1));
}

template<int WarpType, class Scalar>
void drawRectOfTemplate(cv::Mat &img, const ia::Warp<WarpType, Scalar> &w, cv::Size tplSize, cv::Scalar color)
{
  typedef typename ia::WarpTraits<WarpType, Scalar>::PointType PointType;

  PointType c0 = w(PointType(Scalar(0.5), Scalar(0.5)));
  PointType c1 = w(PointType(Scalar(0.5) + tplSize.width, Scalar(0.5)));
  PointType c2 = w(PointType(Scalar(0.5) + tplSize.width, Scalar(0.5) + tplSize.height));
  PointType c3 = w(PointType(Scalar(0.5), Scalar(0.5) + tplSize.height));

  cv::line(img, toP(c0), toP(c1), color, 1, CV_AA);
  cv::line(img, toP(c1), toP(c2), color, 1, CV_AA);
  cv::line(img, toP(c2), toP(c3), color, 1, CV_AA);
  cv::line(img, toP(c3), toP(c0), color, 1, CV_AA);
}

void opticalFlowIA(cv::Mat &prevGray,
                   cv::Mat &gray,
                   std::vector<cv::Point2f> &prevPoints,
                   std::vector<cv::Point2f> &points,
                   std::vector<uchar> &status,
                   std::vector<float> &err)
{
  const int LEVELS = 3;

  // Will be using pure translational motion
  typedef ia::WarpTranslationF WarpType;

  // In conjunction with inverse compositional algorithm
  typedef ia::AlignInverseCompositional< WarpType > AlignType;

  // We will also make use of the face, that we can share gray among all aligners
  ia::ImagePyramid target;
  target.create(gray, LEVELS);

  // Create an aligner for each point
  std::vector<AlignType> aligners(prevPoints.size());

  // Create a warp for each point. Note we use identity transform here.
  std::vector<WarpType> warps(prevPoints.size());

  // Prepare outputs
  points.resize(prevPoints.size());
  status.resize(prevPoints.size());
  err.resize(prevPoints.size());

#pragma omp parallel for
  for (int i = 0; i < (int)aligners.size(); ++i) {

    // The template will be a rectangular region around the point
    const int windowOff = 15;
    const cv::Point2f p = prevPoints[i];

    int l = (int)(p.x - windowOff);
    int t = (int)(p.y - windowOff);
    int r = (int)(p.x + windowOff);
    int b = (int)(p.y + windowOff);

    // Clamp to region
    l = std::min<int>(gray.cols - 1, std::max<int>(0, l));
    t = std::min<int>(gray.rows - 1, std::max<int>(0, t));
    r = std::min<int>(gray.cols - 1, std::max<int>(0, r));
    b = std::min<int>(gray.rows - 1, std::max<int>(0, b));
    cv::Rect roi(l, t, r - l, b - t);

    if (roi.area() < 10) {
      status[i] = 0;
      continue;
    }

    // Move corner to top left
    float offsetX = (float)l - p.x;
    float offsetY = (float)t - p.y;

    // Initialize warp
    ia::WarpTranslationF::Traits::ParamType wp(p.x + offsetX, p.y + offsetY);
    warps[i].setParameters(wp);

    // Initialize aligner
    aligners[i].prepare(prevGray(roi), target, warps[i], LEVELS);
    aligners[i].align(warps[i], 20, 0.03f);

    // Extract result
    wp = warps[i].parameters();
    points[i].x = wp(0) - offsetX;
    points[i].y = wp(1) - offsetY;
    err[i] = aligners[i].lastError();
    status[i] = err[i] < 40*40;
  }

}

void opticalFlowCV(cv::Mat &prevGray,
                   cv::Mat &gray,
                   std::vector<cv::Point2f> &prevPoints,
                   std::vector<cv::Point2f> &points,
                   std::vector<uchar> &status,
                   std::vector<float> &err)
{
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
  cv::Size winSize(31,31);

  calcOpticalFlowPyrLK(prevGray, gray, prevPoints, points, status, err, winSize, 3, termcrit, 0, 0.001);
}

void opticalFlowFB(cv::Mat &prevGray,
                   cv::Mat &gray,
                   cv::Mat &flow)
{
  // cv::Size winSize(31,31);

  double pyr_scale = 0.5;
  int levels = 6;
  int winsize = 15;
  int iterations = 8;
  int poly_n = 5;
  double poly_sigma = 1.5;

  cv::calcOpticalFlowFarneback(prevGray, gray, flow, pyr_scale, levels, winsize,
                               iterations, poly_n, poly_sigma, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
}

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0
        // s = 0, v is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

// http://stackoverflow.com/questions/20064818/how-to-draw-optical-flow-images-from-oclpyrlkopticalflowdense
void FlowToRGB(const cv::Mat & inpFlow,
               cv::Mat &rgbFlow,
               const float & max_size,
               bool use_value)
{
  if(inpFlow.empty()) return;
  if(inpFlow.depth() != CV_32F)
    throw(std::runtime_error("FlowToRGB: error inpFlow wrong data type (has be CV_32FC2)"));
  const float grad2deg = (float)(180/3.141);
  cv::Mat pol(inpFlow.size(), CV_32FC2);

  float mean_val = 0, min_val = 1000, max_val = 0;
  float _dx, _dy;

  for(int r = 0; r < inpFlow.rows; r++)
    {
      for(int c = 0; c < inpFlow.cols; c++)
        {
          float x = inpFlow.at<cv::Point2f>(r,c).x;
          float y = inpFlow.at<cv::Point2f>(r,c).y;

          float mag = sqrt(x * x + y * y);
          float theta = atan2(y, x);
          theta *= grad2deg;

          if (theta < 0) theta += 360;

          if (r == 100 && c == 100) {
            std::cout << "x:     " << x << std::endl;
            std::cout << "y:     " << y << std::endl;
            std::cout << "mag:   " << mag << std::endl;
            std::cout << "theta: " << theta << std::endl;
          }

          mean_val += mag;
          max_val = MAX(max_val, mag);
          min_val = MIN(min_val, mag);
          pol.at<cv::Point2f>(r,c) = cv::Point2f(mag, theta);
        }
    }

  std::cout << "max_val: " << max_val << std::endl;
  std::cout << "min_val: " << min_val << std::endl;

  mean_val /= inpFlow.size().area();
  float scale = max_val - min_val;
  float shift = -min_val;//-mean_val + scale;
  // scale = 255.f/scale;
  if( max_size > 0)
    {
      scale = 255.f/max_size;
      shift = 0;
    }

  for(int r = 0; r < inpFlow.rows; r++)
    {
      for(int c = 0; c < inpFlow.cols; c++)
        {
          cv::Point2f vpol = pol.at<cv::Point2f>(r,c);

          float mag = vpol.x;
          float theta = vpol.y;

          hsv in;

          in.h = theta;
          in.s = (mag + shift) / scale;
          in.v = 1.0;

          // std::cout << "hsv2rgb begin" << std::endl;
          rgb out = hsv2rgb(in);
          // std::cout << "hsv2rgb end" << std::endl;

          cv::Vec3b & color = rgbFlow.at<cv::Vec3b>(r,c);

          color.val[0] = out.r * 255.0;
          color.val[1] = out.g * 255.0;
          color.val[2] = out.b * 255.0;
        }
    }
}

void drawDenseOpticalFlow(cv::Mat &image,
                          cv::Mat &flow)
{
  FlowToRGB(flow, image, 50, true);
}

void drawOpticalFlow(cv::Mat &image,
                     std::vector<cv::Point2f> &prevPoints,
                     std::vector<cv::Point2f> &points,
                     std::vector<uchar> &status)
{
  for (size_t i = 0; i < prevPoints.size(); ++i) {
    if (!status[i])
      continue;

    cv::circle(image, points[i], 3, cv::Scalar(0,255,0), -1, 8);
  }
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

int main(int argc, char **argv)
{

  // Usage:
  //
  //   ./example_optflow img0.png img1.png out.png
  //
  // Computes a flow (in .flo format) from img0 and img2.
  //

  if (argc == 4) {
    cv::Mat img0, img1, gray, prevGray, flow, flowImg;

    img0 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
    img1 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(!img0.data || !img1.data)                              // Check for invalid input
      {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
      }

    // Initialize flowImg
    img0.copyTo(flowImg);

    cv::cvtColor(img0, prevGray, CV_BGR2GRAY);
    cv::cvtColor(img1, gray, CV_BGR2GRAY);

    auto DIS = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);

    // DIS->setFinestScale(0);
    // DIS->setPatchStride(1);

    int finestScale = DIS->getFinestScale();
    int gradientDescentIterations = DIS->getGradientDescentIterations();
    int patchSize = DIS->getPatchSize();
    int patchStride = DIS->getPatchStride();
    int variationalRefinementIterations = DIS->getVariationalRefinementIterations();

    std::cout << "finestScale:                     " << finestScale << std::endl;
    std::cout << "gradientDescentIterations:       " << gradientDescentIterations << std::endl;
    std::cout << "patchSize:                       " << patchSize << std::endl;
    std::cout << "patchStride:                     " << patchStride << std::endl;
    std::cout << "variationalRefinementIterations: " << variationalRefinementIterations << std::endl;
    std::cout << std::endl;

    auto start = now();

    DIS->calc(prevGray, gray, flow);

    auto dt = now() - start;
    auto us = std::chrono::duration_cast<std::chrono::milliseconds>(dt);
    double duration = us.count();

    std::cout << "time: " << duration << " ms" << std::endl;


    drawDenseOpticalFlow(flowImg, flow);

    cv::imwrite(argv[3], flowImg);

    cv::imshow("Optical Flow", flowImg);
    int key = cv::waitKey(0);

    return 0;
  }

  // Usage:
  //
  //   ./example_optflow
  //
  // Uses the webcam

  else {

    // Use the webcam as input

    cv::VideoCapture cap;

    if (!cap.isOpened()) {
      std::cerr << "Failed to open capture device." << std::endl;
      return -1;
    }

    cv::Mat gray, prevGray, flow, image, frame, flowImg;
    std::vector<cv::Point2f> points[2];

    // cv::optflow::DISOpticalFlow DIS = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_FAST);
    auto DIS = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_FAST);

    // bool init = false;
    bool done = false;

    bool init = true;
    while (!done) {

      // Grab frame
      cap >> frame;
      if (frame.empty())
        break;

      frame.copyTo(image);
      cv::cvtColor(image, gray, CV_BGR2GRAY);

      int key;

      if (init) {
        cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
        cv::Size subPixWinSize(10,10), winSize(31,31);

        // Shi-Thomasi corner features.
        cv::goodFeaturesToTrack(gray, points[1], MAX_FEATURES, 0.01, 10, cv::Mat(), 3, 0, 0.04);
        cv::cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1,-1), termcrit);

        // Init flowImg
        image.copyTo(flowImg);

        init = false;

      } else if (!points[0].empty()) {
        if(prevGray.empty())
          gray.copyTo(prevGray);

        std::vector<uchar> status;
        std::vector<float> err;

        // Perform optical flow
        //opticalFlowIA(prevGray, gray, points[0], points[1], status, err);
        //opticalFlowCV(prevGray, gray, points[0], points[1], status, err);
        //opticalFlowFB(prevGray, gray, flow);

        DIS->calc(prevGray, gray, flow);

        // std::cout << flow[0] << std::endl;

        std::cout << "Drawing dense opt flow" << std::endl;

        drawDenseOpticalFlow(flowImg, flow);
        cv::imshow("Optical Flow", flowImg);
        key = cv::waitKey(0);

        // Draw optical flow results
        size_t k = 0;
        for (size_t i = 0; i < points[1].size(); ++i) {
          if (!status[i])
            continue;

          points[1][k++] = points[1][i];
        }
        points[1].resize(k);

      }

      // int key = cv::waitKey(10);

      switch (key) {
      case 'x':
        done = true;
        break;

      case 'r':
        init = true;
        break;
      }

      std::swap(points[1], points[0]);
      cv::swap(prevGray, gray);

    }

  }

  return 0;
}
