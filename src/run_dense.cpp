#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>

#include "oflow.h"
#include "kernels/warmup.h"
#include "kernels/pad.h"

// CUDA
#include <cuda_runtime.h>

using namespace std;
using namespace OFC;

// Save a Depth/OF/SF as .flo file
void SaveFlowFile(cv::Mat& img, const char* filename) {

  cv::Size szt = img.size();
  int width = szt.width, height = szt.height;
  int nc = img.channels();
  float tmp[nc];

  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

  // write the header
  fprintf(stream, "PIEH");
  if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
      (int)fwrite(&height, sizeof(int),   1, stream) != 1)
    cout << "WriteFile: problem writing header" << endl;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      tmp[0] = img.at<cv::Vec2f>(y,x)[0];
      tmp[1] = img.at<cv::Vec2f>(y,x)[1];

      if ((int)fwrite(tmp, sizeof(float), nc, stream) != nc)
        cout << "WriteFile: problem writing data" << endl;
    }
  }

  fclose(stream);

}


// Read a depth/OF/SF as file
void ReadFlowFile(cv::Mat& img, const char* filename) {

  FILE *stream = fopen(filename, "rb");
  if (stream == 0)
    cout << "ReadFile: could not open %s" << endl;

  int width, height;
  float tag;
  int nc = img.channels();
  float tmp[nc];

  if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
      (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
      (int)fread(&height, sizeof(int),   1, stream) != 1)
    cout << "ReadFile: problem reading file %s" << endl;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      if ((int)fread(tmp, sizeof(float), nc, stream) != nc)
        cout << "ReadFile(%s): file is too short" << endl;

      if (nc==1) // depth
        img.at<float>(y,x) = tmp[0];
      else if (nc==2) // Optical Flow
      {
        img.at<cv::Vec2f>(y,x)[0] = tmp[0];
        img.at<cv::Vec2f>(y,x)[1] = tmp[1];
      }
      else if (nc==4) // Scene Flow
      {
        img.at<cv::Vec4f>(y,x)[0] = tmp[0];
        img.at<cv::Vec4f>(y,x)[1] = tmp[1];
        img.at<cv::Vec4f>(y,x)[2] = tmp[2];
        img.at<cv::Vec4f>(y,x)[3] = tmp[3];
      }
    }
  }

  if (fgetc(stream) != EOF)
    cout << "ReadFile(%s): file is too long" << endl;

  fclose(stream);

}


int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize) {

  float scale = (2.0f * (float) imgwidth) / ((float) fratio * (float) patchsize);
  return std::max(0, (int) std::floor(log2(scale)));

}

int main( int argc, char** argv ) {

  // Warmup GPU
  cu::warmup();

  // Timing structures
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);


  // Parse input images
  char *I0_file = argv[1];
  char *I1_file = argv[2];
  char *flow_file = argv[3];

  // CV mats original and float
  cv::Mat I0_mat, I1_mat;
  cv::Mat I0_fmat, I1_fmat;

  // Load images
  I0_mat = cv::imread(I0_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  I1_mat = cv::imread(I1_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  int width_org = I0_mat.size().width;   // unpadded original image size
  int height_org = I0_mat.size().height;  // unpadded original image size

  // Parse rest of parameters
  opt_params op;

  if (argc <= 5) {

    op.use_mean_normalization = true; op.cost_func = 0;
    op.var_ref_alpha = 10.0; op.var_ref_gamma = 10.0; op.var_ref_delta = 5.0;
    op.var_ref_iter = 3; op.var_ref_sor_weight = 1.6;
    op.verbosity = 2; // Default: Plot detailed timings

    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.

    int op_point = 2; // Default operating point
    if (argc == 5)    // Use provided operating point
      op_point = atoi(argv[4]);

    switch (op_point) {

      case 1:
        op.patch_size = 8; op.patch_stride = 0.3;
        op.coarsest_scale = AutoFirstScaleSelect(width_org, fratio, op.patch_size);
        op.finest_scale = std::max(op.coarsest_scale - 2,0); op.grad_descent_iter = 16;
        op.use_var_ref = false;
        break;
      case 3:
        op.patch_size = 12; op.patch_stride = 0.75;
        op.coarsest_scale = AutoFirstScaleSelect(width_org, fratio, op.patch_size);
        op.finest_scale = std::max(op.coarsest_scale - 4,0); op.grad_descent_iter = 16;
        op.use_var_ref = true;
        break;
      case 4:
        op.patch_size = 12; op.patch_stride = 0.75;
        op.coarsest_scale = AutoFirstScaleSelect(width_org, fratio, op.patch_size);
        op.finest_scale = std::max(op.coarsest_scale - 5,0); op.grad_descent_iter = 128;
        op.use_var_ref = true;
        break;
      case 2:
      default:
        op.patch_size = 8; op.patch_stride = 0.4;
        op.coarsest_scale = AutoFirstScaleSelect(width_org, fratio, op.patch_size);
        op.finest_scale = std::max(op.coarsest_scale - 2,0); op.grad_descent_iter = 12;
        op.use_var_ref = true;
        break;

    }
  } else {

    int acnt = 4; // Argument counter
    op.coarsest_scale = atoi(argv[acnt++]);
    op.finest_scale = atoi(argv[acnt++]);
    op.grad_descent_iter = atoi(argv[acnt++]);
    op.patch_size = atoi(argv[acnt++]);
    op.patch_stride = atof(argv[acnt++]);
    op.use_mean_normalization = atoi(argv[acnt++]);
    op.cost_func = atoi(argv[acnt++]);
    op.use_var_ref = atoi(argv[acnt++]);
    op.var_ref_alpha = atof(argv[acnt++]);
    op.var_ref_gamma = atof(argv[acnt++]);
    op.var_ref_delta = atof(argv[acnt++]);
    op.var_ref_iter = atoi(argv[acnt++]);
    op.var_ref_sor_weight = atof(argv[acnt++]);
    op.verbosity = atoi(argv[acnt++]);

  }


  // convert to float
  I0_mat.convertTo(I0_fmat, CV_32F);
  I1_mat.convertTo(I1_fmat, CV_32F);

  // Pad image such that width and height are restless divisible on all scales (except last)
  int padw = 0, padh = 0;
  int max_scale = pow(2, op.coarsest_scale); // enforce restless division by this number on coarsest scale
  int div = width_org % max_scale;
  if (div > 0) padw = max_scale - div;
  div = height_org % max_scale;
  if (div > 0) padh = max_scale - div;

  if (padh > 0 || padw > 0) {

    cu::pad(I0_fmat, I0_fmat, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);
    cu::pad(I1_fmat, I1_fmat, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);

  }

  // Create image paramaters
  img_params iparams;

  // padded image size, ensures divisibility by 2 on all scales (except last)
  iparams.width = I0_fmat.size().width;
  iparams.height = I0_fmat.size().height;
  iparams.padding = op.patch_size;


  // Timing, image loading
  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Image loading     ) (ms): %3g\n", tt);

  }


  // Create Optical Flow object
  OFClass ofc(op);

  // Run main optical flow / depth algorithm
  float scale_fact = pow(2, op.finest_scale);
  cv::Mat flow_mat(iparams.height / scale_fact , iparams.width / scale_fact, CV_32FC2); // Optical Flow

  ofc.calc(I0_fmat, I1_fmat, iparams, nullptr, (float*) flow_mat.data);


  if (op.verbosity > 1) gettimeofday(&start_time, NULL);

  // Resize to original scale, if not run to finest level
  if (op.finest_scale != 0) {

    flow_mat *= scale_fact;
    cv::resize(flow_mat, flow_mat, cv::Size(), scale_fact, scale_fact , cv::INTER_LINEAR);

  }

  // If image was padded, remove padding before saving to file
  flow_mat = flow_mat(cv::Rect((int) floor((float) padw / 2.0f),(int) floor((float) padh / 2.0f),
        width_org, height_org));

  // Save Result Image
  SaveFlowFile(flow_mat, flow_file);

  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);

  }

  return 0;

}
