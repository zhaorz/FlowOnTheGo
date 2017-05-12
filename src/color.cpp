#define CV_CPU_HAS_SUPPORT_SSE2 0

#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <math.h>

#define MAX_FEATURES 100
int ncols = 0;
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];

void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel()
{
  // relative lengths of color transitions:
  // these are chosen based on perceptual similarity
  // (e.g. one can distinguish more shades between red and yellow 
  //  than between yellow and green)
  int RY = 15;
  int YG = 6;
  int GC = 4;
  int CB = 11;
  int BM = 13;
  int MR = 6;
  ncols = RY + YG + GC + CB + BM + MR;
  //printf("ncols = %d\n", ncols);
  if (ncols > MAXCOLS)
    exit(1);
  int i;
  int k = 0;
  for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
  for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
  for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
  for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
  for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
  for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void computeColor(float fx, float fy, cv::Vec3b &color) {
  if (ncols == 0)
    makecolorwheel();

  float rad = sqrt(fx * fx + fy * fy);
  float a = atan2(-fy, -fx) / M_PI;
  float fk = (a + 1.0) / 2.0 * (ncols-1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % ncols;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = colorwheel[k0][b] / 255.0;
    float col1 = colorwheel[k1][b] / 255.0;
    float col = (1 - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1 - rad * (1 - col); // increase saturation with radius
    else
      col *= .75; // out of range
    color.val[2 - b] = (int)(255.0 * col);
  }
}

void colorFlow(const cv::Mat & inpFlow,
    cv::Mat &rgbFlow,
    const float & maxmotion,
    bool use_value) {

  if(inpFlow.empty()) return;
  if(inpFlow.depth() != CV_32F)
    throw(std::runtime_error("FlowToRGB: error inpFlow wrong data type (has be CV_32FC2)"));

  float maxx = -999, maxy = -999;
  float minx =  999, miny =  999;
  float maxrad = -1;

  for(int r = 0; r < inpFlow.rows; r++) {
    for(int c = 0; c < inpFlow.cols; c++) {
      float fx = inpFlow.at<cv::Point2f>(r,c).x;
      float fy = inpFlow.at<cv::Point2f>(r,c).y;

      maxx = MAX(maxx, fx);
      maxy = MAX(maxy, fy);
      minx = MIN(minx, fx);
      miny = MIN(miny, fy);

      float rad = sqrt(fx * fx + fy * fy);
      maxrad = MAX(maxrad, rad);
    }
  }
  printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
      maxrad, minx, maxx, miny, maxy);

  if (maxmotion > 0) // i.e., specified on commandline
    maxrad = maxmotion;
  if (maxrad == 0) // if flow == 0 everywhere
    maxrad = 1;

  fprintf(stderr, "normalizing by %g\n", maxrad);

  for(int r = 0; r < inpFlow.rows; r++) {
    for(int c = 0; c < inpFlow.cols; c++) {
      float fx = inpFlow.at<cv::Point2f>(r,c).x;
      float fy = inpFlow.at<cv::Point2f>(r,c).y;

      cv::Vec3b &color = rgbFlow.at<cv::Vec3b>(r,c);

      computeColor(fx, fy, color);
    }
  }
}



/**

  Nice hsv -> rgb conversion

http://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both

*/

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
void flowToRGB(const cv::Mat & inpFlow,
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

