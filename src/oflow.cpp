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

#include <sys/time.h>    // timeof day
#include <stdio.h>

#include "oflow.h"
#include "patchgrid.h"
#include "refine_variational.h"

#include "kernels/resize.h"
#include "kernels/pad.h"
#include "kernels/resizeGrad.h"
#include "kernels/sobel.h"
#include "kernels/pyramid.h"
#include "common/RgbMat.h"
#include "common/timer.h"


using std::cout;
using std::endl;
using std::vector;

using namespace timer;

namespace OFC {

  OFClass::OFClass(opt_params _op) {

    // Parse optimization parameters
    op = _op;
    op.outlier_thresh = (float) op.patch_size / 2;
    op.steps = std::max(1, (int) floor(op.patch_size * (1 - op.patch_stride)));
    op.n_vals = 3 * pow(op.patch_size, 2);
    op.n_scales = op.coarsest_scale - op.finest_scale + 1;
    float norm_outlier2 = pow(op.norm_outlier, 2);
    op.norm_outlier_tmpbsq = (v4sf) {norm_outlier2, norm_outlier2, norm_outlier2, norm_outlier2};
    op.norm_outlier_tmp2bsq = __builtin_ia32_mulps(op.norm_outlier_tmpbsq, op.twos);
    op.norm_outlier_tmp4bsq = __builtin_ia32_mulps(op.norm_outlier_tmpbsq, op.fours);
    op.dp_thresh = 0.05 * 0.05;
    op.dr_thresh = 0.95;
    op.res_thresh = 0.0;

    // Allocate scale pyramides
    I0s = new float*[op.coarsest_scale+1];
    I1s = new float*[op.coarsest_scale+1];
    I0xs = new float*[op.coarsest_scale+1];
    I0ys = new float*[op.coarsest_scale+1];
    I1xs = new float*[op.coarsest_scale+1];
    I1ys = new float*[op.coarsest_scale+1];

    I0_mats = new cv::Mat[op.coarsest_scale+1];
    I1_mats = new cv::Mat[op.coarsest_scale+1];
    I0x_mats = new cv::Mat[op.coarsest_scale+1];
    I0y_mats = new cv::Mat[op.coarsest_scale+1];
    I1x_mats = new cv::Mat[op.coarsest_scale+1];
    I1y_mats = new cv::Mat[op.coarsest_scale+1];

  }



  void OFClass::ConstructImgPyramids() {

    // Timing structures
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Construct image and gradient pyramides
    cu::constructImgPyramids(I0, I0_mats, I0x_mats, I0y_mats, op.patch_size, op.coarsest_scale + 1);
    cu::constructImgPyramids(I1, I1_mats, I1x_mats, I1y_mats, op.patch_size, op.coarsest_scale + 1);

    auto start_pad = now();

    // Pad images
    for (int i = 0; i <= op.coarsest_scale; ++i) {

      // Replicate padding for images
      // cu::pad(I0_mats[i], I0_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, true);
      // cu::pad(I1_mats[i], I1_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, true);
      I0s[i] = (float*) I0_mats[i].data;
      I1s[i] = (float*) I1_mats[i].data;

      // Zero pad for gradients
      // cu::pad(I0x_mats[i], I0x_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, false);
      // cu::pad(I0y_mats[i], I0y_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, false);
      // cu::pad(I1x_mats[i], I1x_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, false);
      // cu::pad(I1y_mats[i], I1y_mats[i], op.patch_size, op.patch_size,
      //     op.patch_size, op.patch_size, false);

      I0xs[i] = (float*) I0x_mats[i].data;
      I0ys[i] = (float*) I0y_mats[i].data;
      I1xs[i] = (float*) I1x_mats[i].data;
      I1ys[i] = (float*) I1y_mats[i].data;

    }

    calc_print_elapsed("pad images", start_pad);

    // Timing, image gradients and pyramid
    if (op.verbosity > 1) {

      gettimeofday(&end_time, NULL);
      double tt = (end_time.tv_sec-start_time.tv_sec)*1000.0f + (end_time.tv_usec-start_time.tv_usec)/1000.0f;
      printf("TIME (Pyramids+Gradients) (ms): %3g\n", tt);

    }

  }



  void OFClass::calc(cv::Mat _I0, cv::Mat _I1, img_params _iparams, const float * initflow, float * outflow) {

    I0 = _I0;
    I1 = _I1;

    std::cout << "I0 " << I0.size() << " channels: " << I0.channels()
      << " type: " << I0.type() << std::endl;

    printf("Constructing pyramids\n");
    ConstructImgPyramids();

    if (op.verbosity > 1) cout << ", cflow " << endl;

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

    if (op.verbosity>1) gettimeofday(&tv_start_all, nullptr);


    // Create grids on each scale
    vector<OFC::PatGridClass*> grid(op.n_scales);
    vector<float*> flow(op.n_scales);
    iparams.resize(op.n_scales);
    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      int i = sl - op.finest_scale;

      float scale_fact = pow(2, -sl); // scaling factor at current scale
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

      flow[i]   = new float[2 * iparams[i].width * iparams[i].height];
      grid[i]   = new OFC::PatGridClass(&(iparams[i]), &op);

    }


    // Timing, Grid memory allocation
    if (op.verbosity>1) {

      gettimeofday(&tv_end_all, nullptr);
      double tt_gridconst = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      printf("TIME (Grid Memo. Alloc. ) (ms): %3g\n", tt_gridconst);

    }


    // Main loop; Operate over scales, coarse-to-fine
    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      int ii = sl - op.finest_scale;
      if (op.verbosity > 1) gettimeofday(&tv_start_all, nullptr);


      // Initialize grid (Step 1 in Algorithm 1 of paper)
      grid[ii]->InitializeGrid(I0s[sl], I0xs[sl], I0ys[sl]);
      grid[ii]->SetTargetImage(I1s[sl], I1xs[sl], I1ys[sl]);

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

        OFC::VarRefClass var_ref(I0s[sl], I1s[sl], &(iparams[ii]), &op, out_ptr);

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

    // Clean up
    for (int sl = op.coarsest_scale; sl >= op.finest_scale; --sl) {

      delete[] flow[sl - op.finest_scale];
      delete grid[sl - op.finest_scale];

    }


    // Timing, total algorithm run-time
    if (op.verbosity > 0) {

      gettimeofday(&tv_end_all_global, nullptr);
      double tt = (tv_end_all_global.tv_sec-tv_start_all_global.tv_sec)*1000.0f + (tv_end_all_global.tv_usec-tv_start_all_global.tv_usec)/1000.0f;
      printf("TIME (O.Flow Run-Time   ) (ms): %3g\n", tt);

    }

  }

}
