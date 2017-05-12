#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp> // needed for verbosity >= 3, DISVISUAL
#include <opencv2/highgui/highgui.hpp> // needed for verbosity >= 3, DISVISUAL
#include <opencv2/imgproc/imgproc.hpp> // needed for verbosity >= 3, DISVISUAL

#include <nppi.h>

#include <sys/time.h>    // timeof day
#include <stdio.h>

#include "oflow.h"

#include "kernels/resize.h"
#include "kernels/pad.h"
#include "kernels/resizeGrad.h"
#include "kernels/sobel.h"
#include "kernels/pyramid.h"
#include "common/timer.h"


using std::cout;
using std::endl;
using std::vector;

using namespace timer;

namespace OFC {

  OFClass::OFClass(opt_params _op, img_params _iparams) {

    struct timeval tv_start_all, tv_end_all, tv_start_all_global, tv_end_all_global;
    if (op.verbosity > 1) gettimeofday(&tv_start_all_global, nullptr);

    // Parse optimization parameters
    op = _op;
    op.outlier_thresh = (float) op.patch_size / 2;
    op.steps = std::max(1, (int) floor(op.patch_size * (1 - op.patch_stride)));
    op.n_vals = 3 * pow(op.patch_size, 2);
    op.n_scales = op.coarsest_scale - op.finest_scale + 1;
    // float norm_outlier2 = pow(op.norm_outlier, 2);
    // op.norm_outlier_tmpbsq = (v4sf) {norm_outlier2, norm_outlier2, norm_outlier2, norm_outlier2};
    // op.norm_outlier_tmp2bsq = __builtin_ia32_mulps(op.norm_outlier_tmpbsq, op.twos);
    // op.norm_outlier_tmp4bsq = __builtin_ia32_mulps(op.norm_outlier_tmpbsq, op.fours);
    op.dp_thresh = 0.05 * 0.05;
    op.dr_thresh = 0.95;
    op.res_thresh = 0.0;

    // Initialize cuBLAS
    cublasStatus_t stat = cublasCreate(&op.cublasHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      exit(-1);
    }

    // Allocate scale pyramides
    I0s = new float*[op.coarsest_scale+1];
    I1s = new float*[op.coarsest_scale+1];
    I0xs = new float*[op.coarsest_scale+1];
    I0ys = new float*[op.coarsest_scale+1];
    I1xs = new float*[op.coarsest_scale+1];
    I1ys = new float*[op.coarsest_scale+1];

    // Create grids on each scale
    if (op.verbosity>1) gettimeofday(&tv_start_all, nullptr);


    int elemSize = 3 * sizeof(float);
    grid.resize(op.n_scales);
    flow.resize(op.n_scales);
    iparams.resize(op.n_scales);
    for (int sl = op.coarsest_scale; sl >= 0; --sl) {

      int i = sl - op.finest_scale;

      float scale_fact = pow(2, -sl); // scaling factor at current scale
      if (i >= 0) {
        iparams[i].scale_fact = scale_fact;
        iparams[i].height = _iparams.height * scale_fact;
        iparams[i].width = _iparams.width * scale_fact;
        iparams[i].padding = _iparams.padding;
        iparams[i].l_bound = -(float) op.patch_size / 2;
        iparams[i].u_bound_width = (float) (iparams[i].width + op.patch_size / 2 - 2);
        iparams[i].u_bound_height = (float) (iparams[i].height + op.patch_size / 2 - 2);
        iparams[i].width_pad = iparams[i].width + 2 * _iparams.padding;
        iparams[i].height_pad = iparams[i].height + 2 * _iparams.padding;
        iparams[i].curr_lvl = sl;

        // flow[i]   = new float[2 * iparams[i].width * iparams[i].height];
        checkCudaErrors(
            cudaHostAlloc((void**) &(flow[i]),
              2 * iparams[i].width * iparams[i].height * sizeof(float), cudaHostAllocMapped) );
        grid[i]   = new OFC::PatGridClass(&(iparams[i]), &op);
      }

      int padWidth = _iparams.width * scale_fact + 2 * _iparams.padding;
      int padHeight = _iparams.height * scale_fact + 2 * _iparams.padding;

      checkCudaErrors( cudaMalloc((void**) &I0s[sl],  padWidth * padHeight * elemSize) );
      checkCudaErrors( cudaMalloc((void**) &I0xs[sl], padWidth * padHeight * elemSize) );
      checkCudaErrors( cudaMalloc((void**) &I0ys[sl], padWidth * padHeight * elemSize) );

      checkCudaErrors( cudaMalloc((void**) &I1s[sl],  padWidth * padHeight * elemSize) );
      checkCudaErrors( cudaMalloc((void**) &I1xs[sl], padWidth * padHeight * elemSize) );
      checkCudaErrors( cudaMalloc((void**) &I1ys[sl], padWidth * padHeight * elemSize) );
    }

    // Timing, Grid memory allocation
    if (op.verbosity>1) {

      gettimeofday(&tv_end_all, nullptr);
      double tt_gridconst = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      printf("TIME (Grid Memo. Alloc. ) (ms): %3g\n", tt_gridconst);

    }

    const Npp32f pSrcKernel[3] = { 1, 0, -1 };
    Npp32s nMaskSize = 3;

    checkCudaErrors( cudaMalloc((void**) &pDeviceIx, _iparams.width * _iparams.height * elemSize) );
    checkCudaErrors( cudaMalloc((void**) &pDeviceIy, _iparams.width * _iparams.height * elemSize) );

    checkCudaErrors( cudaMalloc((void**) &pDeviceTmp, _iparams.width * _iparams.height * elemSize)  );
    checkCudaErrors( cudaMalloc((void**) &pDeviceWew, nMaskSize * sizeof(Npp32f)) );

    checkCudaErrors(
        cudaMemcpy(pDeviceWew, pSrcKernel, nMaskSize * sizeof(Npp32f), cudaMemcpyHostToDevice) );

    // Timing, Setup
    if (op.verbosity>1) {

      gettimeofday(&tv_end_all_global, nullptr);
      double tt = (tv_end_all_global.tv_sec-tv_start_all_global.tv_sec)*1000.0f + (tv_end_all_global.tv_usec-tv_start_all_global.tv_usec)/1000.0f;
      printf("TIME (Setup) (ms): %3g\n", tt);
    }

  }

  OFClass::~OFClass() {

    cublasDestroy(op.cublasHandle);

    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      cudaFree(flow[sl - op.finest_scale]);
      delete grid[sl - op.finest_scale];

    }

    for (int i = 0; i <= op.coarsest_scale; i++) {
      cudaFree(I0s[i]);
      cudaFree(I0xs[i]);
      cudaFree(I0ys[i]);

      cudaFree(I1s[i]);
      cudaFree(I1xs[i]);
      cudaFree(I1ys[i]);
    }

    delete I0s;
    delete I1s;
    delete I0xs;
    delete I0ys;
    delete I1xs;
    delete I1ys;

    cudaFree(pDeviceIx);
    cudaFree(pDeviceIy);
    cudaFree(pDeviceTmp);
    cudaFree(pDeviceWew);
  }


  void OFClass::ConstructImgPyramids(img_params iparams) {

    // Timing structures
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Construct image and gradient pyramides
    /*cu::constructImgPyramids(I0, I0s, I0xs, I0ys,
        pDeviceIx, pDeviceIy, pDeviceTmp, pDeviceWew,
        iparams.width, iparams.height,
        op.patch_size, op.coarsest_scale + 1);*/
    cu::constructImgPyramids(I1, I1s, I1xs, I1ys,
        pDeviceIx, pDeviceIy, pDeviceTmp, pDeviceWew,
        iparams.width, iparams.height,
        op.patch_size, op.finest_scale, op.coarsest_scale);

    // Timing, image gradients and pyramid
    if (op.verbosity > 1) {

      gettimeofday(&end_time, NULL);
      double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
      printf("TIME (Pyramids+Gradients) (ms): %3g\n", tt);

    }

  }


  void OFClass::first(Npp32f* _I1, img_params _iparams) {
    I1 = _I1;
    std::cout << "[Processing first frame] " << _iparams.height << "x" << _iparams.width << std::endl;

    ConstructImgPyramids(_iparams);
    std::cout << "[Done with first frame]" << std::endl;
  }


  void OFClass::next(Npp32f* _I1, img_params _iparams, float * initflow, float * outflow) {

    std::swap(I0, I1);
    std::swap(I0s, I1s);
    std::swap(I0xs, I1xs);
    std::swap(I0ys, I1ys);
    I1 = _I1;

    std::cout << "[Processing next frame] " << _iparams.height << "x" << _iparams.width << std::endl;

    ConstructImgPyramids(_iparams);

    std::cout << "coarsest: " <<  op.coarsest_scale << " finest: " << op.finest_scale << std::endl;

    // Variables for algorithm timings
    struct timeval tv_start_all, tv_end_all, tv_start_all_global, tv_end_all_global;
    if (op.verbosity > 0) gettimeofday(&tv_start_all_global, nullptr);

    // ... per each scale
    double tt_patconstr[op.n_scales], tt_patinit[op.n_scales], tt_patoptim[op.n_scales],
           tt_compflow[op.n_scales], tt_tvopt[op.n_scales], tt_all[op.n_scales];
    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      tt_patconstr[sl - op.finest_scale] = 0;
      tt_patinit[sl - op.finest_scale] = 0;
      tt_patoptim[sl - op.finest_scale] = 0;
      tt_compflow[sl - op.finest_scale] = 0;
      tt_tvopt[sl - op.finest_scale] = 0;
      tt_all[sl - op.finest_scale] = 0;

    }


    // Main loop; Operate over scales, coarse-to-fine
    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      int ii = sl - op.finest_scale;
      if (op.verbosity > 1) gettimeofday(&tv_start_all, nullptr);


      // Initialize grid (Step 1 in Algorithm 1 of paper)
      grid[ii]->InitializeGrid(I0s[sl], I0xs[sl], I0ys[sl]);
      grid[ii]->SetTargetImage(I1s[sl]);

      // Timing, Grid construction
      if (op.verbosity > 1) {

        gettimeofday(&tv_end_all, nullptr);
        tt_patconstr[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        tt_all[ii] += tt_patconstr[ii];
        gettimeofday(&tv_start_all, nullptr);

      }


      // Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)
      if (sl < op.coarsest_scale) {
        // initialize from flow at previous coarser scale
        grid[ii]->InitializeFromCoarserOF(flow[ii+1]);
      } else if (sl == op.coarsest_scale && initflow != nullptr) {
        // initialization given input flow
        grid[ii]->InitializeFromCoarserOF(initflow);
      }

      // Timing, Grid initialization
      if (op.verbosity > 1) {

        gettimeofday(&tv_end_all, nullptr);
        tt_patinit[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        tt_all[ii] += tt_patinit[ii];
        gettimeofday(&tv_start_all, nullptr);

      }


      // Dense Inverse Search. (Step 3 in Algorithm 1 of paper)
      // Parallel over all patches
      grid[ii]->Optimize();

      // Timing, DIS
      if (op.verbosity>1) {

        gettimeofday(&tv_end_all, nullptr);
        tt_patoptim[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        tt_all[ii] += tt_patoptim[ii];
        gettimeofday(&tv_start_all, nullptr);

      }


      // Densification. (Step 4 in Algorithm 1 of paper)
      float *out_ptr = flow[ii];
      if (sl == op.finest_scale)
        out_ptr = outflow;

      grid[ii]->AggregateFlowDense(out_ptr);


      // Timing, Densification
      if (op.verbosity > 1) {

        gettimeofday(&tv_end_all, nullptr);
        tt_compflow[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        tt_all[ii] += tt_compflow[ii];
        gettimeofday(&tv_start_all, nullptr);

      }


      // Variational refinement, (Step 5 in Algorithm 1 of paper)
      if (op.use_var_ref) {
      // if (false) {
        // float* I0H, * I1H;
        // int elemSize = 3 * sizeof(float);
        // int size = iparams[ii].width_pad * iparams[ii].height_pad * elemSize;
        // I0H = (float*) malloc(size);
        // I1H = (float*) malloc(size);

        // auto start = now();
        // checkCudaErrors(
        //     cudaMemcpy(I0H, I0s[sl], size, cudaMemcpyDeviceToHost) );
        // checkCudaErrors(
        //     cudaMemcpy(I1H, I1s[sl], size, cudaMemcpyDeviceToHost) );
        // calc_print_elapsed("pre var-ref memcpy", start);

        // OFC::VarRefClass var_ref(I0H, I1H, &(iparams[ii]), &op, out_ptr);

        cudaDeviceSynchronize();

        OFC::VarRefClass var_ref(I0s[sl], I1s[sl], &(iparams[ii]), &op, out_ptr);

        cudaDeviceSynchronize();

        // delete I0H;
        // delete I1H;

      }

      // Timing, Variational Refinement
      if (op.verbosity > 1)
      {

        gettimeofday(&tv_end_all, nullptr);
        tt_tvopt[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        tt_all[ii] += tt_tvopt[ii];
        printf("TIME (Sc: %i, #p:%6i, pconst, pinit, poptim, cflow, tvopt, total): %8.2f %8.2f %8.2f %8.2f %8.2f -> %8.2f ms.\n", sl, grid[ii]->GetNumPatches(), tt_patconstr[ii], tt_patinit[ii], tt_patoptim[ii], tt_compflow[ii], tt_tvopt[ii], tt_all[ii]);

      }

    }

    // Timing, total algorithm run-time
    if (op.verbosity > 0) {

      gettimeofday(&tv_end_all_global, nullptr);
      double tt = (tv_end_all_global.tv_sec-tv_start_all_global.tv_sec)*1000.0f + (tv_end_all_global.tv_usec-tv_start_all_global.tv_usec)/1000.0f;
      printf("TIME (O.Flow Run-Time   ) (ms): %3g\n", tt);

    }

    // Detailed timing reports
    if (op.verbosity > 1) {
      for (auto &g : grid) {
        g->printTimings();
      }
    }

    std::cout << "[Done with frame]" << std::endl;

  }

}
