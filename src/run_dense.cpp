#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "params.h"
#include "oflow.h"
#include "kernels/warmup.h"
#include "kernels/pad.h"
#include "common/timer.h"
#include "common/cuda_helper.h"


using namespace std;
using namespace OFC;
using namespace timer;

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

  initializeCuda(argc, argv);

  // Warmup GPU
  cu::warmup();

  // Timing structures
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);


  // Parse input images
  char *I0_file = argv[1];
  char *I1_file = argv[2];
  char *I2_file = argv[3];
  char *I3_file = argv[4];
  std::string flow_file = argv[5];

  // CV mats original and float
  cv::Mat I0_mat, I1_mat, I2_mat, I3_mat;
  cv::Mat I0_fmat, I1_fmat, I2_fmat, I3_fmat;

  // Load images
  I0_mat = cv::imread(I0_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  I1_mat = cv::imread(I1_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  I2_mat = cv::imread(I2_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  I3_mat = cv::imread(I3_file, CV_LOAD_IMAGE_COLOR);   // Read the file

  int width_org = I0_mat.size().width;   // unpadded original image size
  int height_org = I0_mat.size().height;  // unpadded original image size

  // convert to float
  I0_mat.convertTo(I0_fmat, CV_32F);
  I1_mat.convertTo(I1_fmat, CV_32F);
  I2_mat.convertTo(I2_fmat, CV_32F);
  I3_mat.convertTo(I3_fmat, CV_32F);

  int channels = 3;
  int elemSize = channels * sizeof(Npp32f);

  /* memcpy to cuda */
  Npp32f* I0, *I1, *I2, *I3;
  auto start_cuda_malloc = now();
  checkCudaErrors( cudaMalloc((void**) &I0, width_org * height_org * elemSize) );
  checkCudaErrors( cudaMalloc((void**) &I1, width_org * height_org * elemSize) );
  checkCudaErrors( cudaMalloc((void**) &I2, width_org * height_org * elemSize) );
  checkCudaErrors( cudaMalloc((void**) &I3, width_org * height_org * elemSize) );
  calc_print_elapsed("I0, I1 cudaMalloc", start_cuda_malloc);

  auto start_memcpy_hd = now();
  checkCudaErrors(
      cudaMemcpy(I0, (float*) I0_fmat.data, width_org * height_org * elemSize, cudaMemcpyHostToDevice) );
  checkCudaErrors(
      cudaMemcpy(I1, (float*) I1_fmat.data, width_org * height_org * elemSize, cudaMemcpyHostToDevice) );
  checkCudaErrors(
      cudaMemcpy(I2, (float*) I2_fmat.data, width_org * height_org * elemSize, cudaMemcpyHostToDevice) );
  checkCudaErrors(
      cudaMemcpy(I3, (float*) I3_fmat.data, width_org * height_org * elemSize, cudaMemcpyHostToDevice) );
  calc_print_elapsed("cudaMemcpy I0, I1 H->D", start_memcpy_hd);


  // Parse rest of parameters
  opt_params op;

  if (true) {

    op.use_mean_normalization = true;
    op.var_ref_alpha = 10.0; op.var_ref_gamma = 10.0; op.var_ref_delta = 5.0;
    op.var_ref_iter = 3; op.var_ref_sor_weight = 1.6;
    op.verbosity = 2; // Default: Plot detailed timings

    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.

    int op_point = 2; // Default operating point
    /*if (argc == 5)    // Use provided operating point
      op_point = atoi(argv[4]);*/

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

    int acnt = 6; // Argument counter
    op.coarsest_scale = atoi(argv[acnt++]);
    op.finest_scale = atoi(argv[acnt++]);
    op.grad_descent_iter = atoi(argv[acnt++]);
    op.patch_size = atoi(argv[acnt++]);
    op.patch_stride = atof(argv[acnt++]);
    op.use_mean_normalization = atoi(argv[acnt++]);
    op.use_var_ref = atoi(argv[acnt++]);
    op.var_ref_alpha = atof(argv[acnt++]);
    op.var_ref_gamma = atof(argv[acnt++]);
    op.var_ref_delta = atof(argv[acnt++]);
    op.var_ref_iter = atoi(argv[acnt++]);
    op.var_ref_sor_weight = atof(argv[acnt++]);
    op.verbosity = atoi(argv[acnt++]);

  }



  // Pad image such that width and height are restless divisible on all scales (except last)
  int padw = 0, padh = 0;
  int max_scale = pow(2, op.coarsest_scale); // enforce restless division by this number on coarsest scale
  int div = width_org % max_scale;
  if (div > 0) padw = max_scale - div;
  div = height_org % max_scale;
  if (div > 0) padh = max_scale - div;

  if (padh > 0 || padw > 0) {
    Npp32f* I0Padded = cu::pad(
        I0, width_org, height_org, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);

    Npp32f* I1Padded = cu::pad(
        I1, width_org, height_org, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);

    Npp32f* I2Padded = cu::pad(
        I2, width_org, height_org, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);


    Npp32f* I3Padded = cu::pad(
        I3, width_org, height_org, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), true);

    cudaFree(I0);
    cudaFree(I1);
    cudaFree(I2);
    cudaFree(I3);

    I0 = I0Padded;
    I1 = I1Padded;
    I2 = I2Padded;
    I3 = I3Padded;
  }



  // Create image paramaters
  img_params iparams;

  // padded image size, ensures divisibility by 2 on all scales (except last)
  iparams.width = width_org + padw;
  iparams.height = height_org + padh;
  iparams.padding = op.patch_size;


  // Timing, image loading
  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Image loading     ) (ms): %3g\n", tt);

  }


  // Create Optical Flow object
  OFClass ofc(op, iparams);
  ofc.first(I0, iparams);

  // Run main optical flow / depth algorithm
  float scale_fact = pow(2, op.finest_scale);
  float* outflow1, * outflow2, * outflow3;
  checkCudaErrors(
      cudaMalloc((void**) &(outflow1), 2 * iparams.height / scale_fact
        * iparams.width / scale_fact * sizeof(float)) );
  checkCudaErrors(
      cudaMalloc((void**) &(outflow2), 2 * iparams.height / scale_fact
        * iparams.width / scale_fact * sizeof(float)) );
  checkCudaErrors(
      cudaMalloc((void**) &(outflow3), 2 * iparams.height / scale_fact
        * iparams.width / scale_fact * sizeof(float)) );

  if (op.verbosity > 1) gettimeofday(&start_time, NULL);
  ofc.next(I1, iparams, nullptr, outflow1);
  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;

    printf("[next] %3g\n", tt);
    gettimeofday(&start_time, NULL);
  }
  ofc.next(I2, iparams, outflow1, outflow2);
  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;

    printf("[next] %3g\n", tt);
    gettimeofday(&start_time, NULL);
  }
  ofc.next(I3, iparams, outflow2, outflow3);
  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;

    printf("[next] %3g\n", tt);
    gettimeofday(&start_time, NULL);
  }

  cv::Mat flow_mat1(iparams.height / scale_fact , iparams.width / scale_fact,
      CV_32FC2);
  cv::Mat flow_mat2(iparams.height / scale_fact , iparams.width / scale_fact,
      CV_32FC2);
  cv::Mat flow_mat3(iparams.height / scale_fact , iparams.width / scale_fact,
      CV_32FC2);

  checkCudaErrors(
      cudaMemcpy(flow_mat1.data, outflow1, 2 * iparams.height / scale_fact 
        * iparams.width / scale_fact * sizeof(float), cudaMemcpyDeviceToHost) );
  checkCudaErrors(
      cudaMemcpy(flow_mat2.data, outflow2, 2 * iparams.height / scale_fact 
        * iparams.width / scale_fact * sizeof(float), cudaMemcpyDeviceToHost) );
  checkCudaErrors(
      cudaMemcpy(flow_mat3.data, outflow3, 2 * iparams.height / scale_fact 
        * iparams.width / scale_fact * sizeof(float), cudaMemcpyDeviceToHost) );

  if (op.verbosity > 1) gettimeofday(&start_time, NULL);

  // Resize to original scale, if not run to finest level
  if (op.finest_scale != 0) {

    flow_mat1 *= scale_fact;
    cv::resize(flow_mat1, flow_mat1, cv::Size(), scale_fact, scale_fact , cv::INTER_LINEAR);
    flow_mat2 *= scale_fact;
    cv::resize(flow_mat2, flow_mat2, cv::Size(), scale_fact, scale_fact , cv::INTER_LINEAR);
    flow_mat3 *= scale_fact;
    cv::resize(flow_mat3, flow_mat3, cv::Size(), scale_fact, scale_fact , cv::INTER_LINEAR);

  }
  // If image was padded, remove padding before saving to file
  flow_mat1 = flow_mat1(cv::Rect((int) floor((float) padw / 2.0f),(int) floor((float) padh / 2.0f),
        width_org, height_org));
  flow_mat2 = flow_mat2(cv::Rect((int) floor((float) padw / 2.0f),(int) floor((float) padh / 2.0f),
        width_org, height_org));
  flow_mat3 = flow_mat3(cv::Rect((int) floor((float) padw / 2.0f),(int) floor((float) padh / 2.0f),
        width_org, height_org));

  // Save Result Image
  SaveFlowFile(flow_mat1, (flow_file + "-1.flo").c_str());
  SaveFlowFile(flow_mat2, (flow_file + "-2.flo").c_str());
  SaveFlowFile(flow_mat3, (flow_file + "-3.flo").c_str());

  if (op.verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);

  }

  return 0;

}
