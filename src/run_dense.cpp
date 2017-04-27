
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>

#include "oflow.h"


using namespace std;

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


void ConstructImgPyramide(const cv::Mat & img_fmat, cv::Mat * img_mats, 
    cv::Mat * imgx_mats, cv::Mat * imgy_mats, const float ** imgs, 
    const float ** imgxs, const float ** imgys, 
    const int coarsest_scale, const int finest_scale, 
    const int imgpadding, const int padw, const int padh) {

  // Construct image and gradient pyramides
  for (int i = 0; i <= coarsest_scale; ++i)  {
    // At finest scale: copy directly, for all other: downscale previous scale by .5
    if (i == 0) {
      img_mats[i] = img_fmat.clone();
    } else {
      cv::resize(img_mats[i-1], img_mats[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
    }

    img_mats[i].convertTo(img_mats[i], CV_32FC1);

    // Generate gradients
    cv::Sobel(img_mats[i], imgx_mats[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(img_mats[i], imgy_mats[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
    imgx_mats[i].convertTo(imgx_mats[i], CV_32F);
    imgy_mats[i].convertTo(imgy_mats[i], CV_32F);
  }

  // Pad images
  for (int i = 0; i <= coarsest_scale; ++i) {

    // Replicate padding for images
    copyMakeBorder(img_mats[i], img_mats[i], imgpadding, imgpadding,
        imgpadding, imgpadding, cv::BORDER_REPLICATE);
    imgs[i] = (float*) img_mats[i].data;

    // Zero pad for gradients
    copyMakeBorder(imgx_mats[i], imgx_mats[i], imgpadding, imgpadding,
        imgpadding, imgpadding, cv::BORDER_CONSTANT, 0);
    copyMakeBorder(imgy_mats[i], imgy_mats[i], imgpadding, imgpadding,
        imgpadding, imgpadding, cv::BORDER_CONSTANT, 0);

    imgxs[i] = (float*) imgx_mats[i].data;
    imgys[i] = (float*) imgy_mats[i].data;

  }

}


int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize) {

  float scale = (2.0f * (float) imgwidth) / ((float) fratio * (float) patchsize);
  return std::max(0, (int) std::floor(log2(scale)));

}


int main( int argc, char** argv ) {

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
  I0_mat = cv::imread(I0_file, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
  I1_mat = cv::imread(I1_file, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
  int width_org = I0_mat.size().width;   // unpadded original image size
  int height_org = I0_mat.size().height;  // unpadded original image size

  // Parse rest of parameters
  int coarsest_scale, finest_scale, grad_descent_iter, patch_size, cost_func, var_ref_iter, verbosity;
  bool use_mean_normalization, use_var_ref;
  float patch_stride, var_ref_alpha, var_ref_gamma, var_ref_delta, var_ref_sor_weight;

  if (argc <= 5) {

    use_mean_normalization = true; cost_func = 0;
    var_ref_alpha = 10.0; var_ref_gamma = 10.0; var_ref_delta = 5.0;
    var_ref_iter = 3; var_ref_sor_weight = 1.6;
    verbosity = 2; // Default: Plot detailed timings

    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.

    int op_point = 2; // Default operating point
    if (argc == 5)    // Use provided operating point
      op_point = atoi(argv[4]);

    switch (op_point) {

      case 1:
        patch_size = 8; patch_stride = 0.3;
        coarsest_scale = AutoFirstScaleSelect(width_org, fratio, patch_size);
        finest_scale = std::max(coarsest_scale-2,0); grad_descent_iter = 16;
        use_var_ref = false;
        break;
      case 3:
        patch_size = 12; patch_stride = 0.75;
        coarsest_scale = AutoFirstScaleSelect(width_org, fratio, patch_size);
        finest_scale = std::max(coarsest_scale-4,0); grad_descent_iter = 16;
        use_var_ref = true;
        break;
      case 4:
        patch_size = 12; patch_stride = 0.75;
        coarsest_scale = AutoFirstScaleSelect(width_org, fratio, patch_size);
        finest_scale = std::max(coarsest_scale-5,0); grad_descent_iter = 128;
        use_var_ref = true;
        break;
      case 2:
      default:
        patch_size = 8; patch_stride = 0.4;
        coarsest_scale = AutoFirstScaleSelect(width_org, fratio, patch_size);
        finest_scale = std::max(coarsest_scale-2,0); grad_descent_iter = 12;
        use_var_ref = true;
        break;

    }
  } else {

    int acnt = 4; // Argument counter
    coarsest_scale = atoi(argv[acnt++]);
    finest_scale = atoi(argv[acnt++]);
    grad_descent_iter = atoi(argv[acnt++]);
    patch_size = atoi(argv[acnt++]);
    patch_stride = atof(argv[acnt++]);
    use_mean_normalization = atoi(argv[acnt++]);
    cost_func = atoi(argv[acnt++]);
    use_var_ref = atoi(argv[acnt++]);
    var_ref_alpha = atof(argv[acnt++]);
    var_ref_gamma = atof(argv[acnt++]);
    var_ref_delta = atof(argv[acnt++]);
    var_ref_iter = atoi(argv[acnt++]);
    var_ref_sor_weight = atof(argv[acnt++]);
    verbosity = atoi(argv[acnt++]);

  }


  // Pad image such that width and height are restless divisible on all scales (except last)
  int padw = 0, padh = 0;
  int max_scale = pow(2, coarsest_scale); // enforce restless division by this number on coarsest scale
  int div = width_org % max_scale;
  if (div > 0) padw = max_scale - div;
  div = height_org % max_scale;
  if (div > 0) padh = max_scale - div;

  if (padh > 0 || padw > 0) {

    copyMakeBorder(I0_mat, I0_mat, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), cv::BORDER_REPLICATE);
    copyMakeBorder(I1_mat, I1_mat, floor((float) padh / 2.0f), ceil((float) padh / 2.0f),
        floor((float) padw / 2.0f), ceil((float) padw / 2.0f), cv::BORDER_REPLICATE);

  }

  // padded image size, ensures divisibility by 2 on all scales (except last)
  int width_pad = I0_mat.size().width;
  int height_pad = I0_mat.size().height;

  // Timing, image loading
  if (verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Image loading     ) (ms): %3g\n", tt);
    gettimeofday(&start_time, NULL);

  }

  // convert to float
  I0_mat.convertTo(I0_fmat, CV_32F);
  I1_mat.convertTo(I1_fmat, CV_32F);


  // Generate scale pyramides
  const float* I0s[coarsest_scale+1];
  const float* I1s[coarsest_scale+1];
  const float* I0xs[coarsest_scale+1];
  const float* I0ys[coarsest_scale+1];
  const float* I1xs[coarsest_scale+1];
  const float* I1ys[coarsest_scale+1];

  cv::Mat I0_mats[coarsest_scale+1];
  cv::Mat I1_mats[coarsest_scale+1];
  cv::Mat I0x_mats[coarsest_scale+1];
  cv::Mat I0y_mats[coarsest_scale+1];
  cv::Mat I1x_mats[coarsest_scale+1];
  cv::Mat I1y_mats[coarsest_scale+1];

  ConstructImgPyramide(I0_fmat, I0_mats, I0x_mats, I0y_mats, I0s, I0xs, I0ys,
      coarsest_scale, finest_scale, patch_size, padw, padh);
  ConstructImgPyramide(I1_fmat, I1_mats, I1x_mats, I1y_mats, I1s, I1xs, I1ys,
      coarsest_scale, finest_scale, patch_size, padw, padh);

  // Timing, image gradients and pyramid
  if (verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Pyramide+Gradients) (ms): %3g\n", tt);

  }


  // Run main optical flow / depth algorithm
  float scale_fact = pow(2, finest_scale);
  cv::Mat flow_mat(height_pad / scale_fact , width_pad / scale_fact, CV_32FC2); // Optical Flow

  OFC::OFClass ofc(I0s, I0xs, I0ys,
      I1s, I1xs, I1ys,
      patch_size,  // extra image padding to avoid border violation check
      (float*) flow_mat.data,   // pointer to n-band output float array
      nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
      width_pad, height_pad,
      coarsest_scale, finest_scale, grad_descent_iter, patch_size, patch_stride,
      cost_func, use_mean_normalization,
      use_var_ref, var_ref_alpha, var_ref_gamma, var_ref_delta, var_ref_iter, var_ref_sor_weight,
      verbosity);

  if (verbosity > 1) gettimeofday(&start_time, NULL);


  // Resize to original scale, if not run to finest level
  if (finest_scale != 0) {

    flow_mat *= scale_fact;
    cv::resize(flow_mat, flow_mat, cv::Size(), scale_fact, scale_fact , cv::INTER_LINEAR);

  }

  // If image was padded, remove padding before saving to file
  flow_mat = flow_mat(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));

  // Save Result Image
  SaveFlowFile(flow_mat, flow_file);

  if (verbosity > 1) {

    gettimeofday(&end_time, NULL);
    double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);

  }

  return 0;

}
