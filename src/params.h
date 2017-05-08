// Holds all of the paramaters structures

#ifndef PARAMS_HEADER
#define PARAMS_HEADER

namespace OFC {

  // typedef __v4sf v4sf;


  typedef struct {
    int width;      // image width, does not include '2 * padding', but includes original padding to ensure integer divisible image width and height
    int height;     // image height, does not include '2 * padding', but includes original padding to ensure integer divisible image width and height
    int padding;    // image padding in pixels at all sides, images padded with replicated border, gradients padded with zero, ADD THIS ONLY WHEN ADDRESSING THE IMAGE OR GRADIENT
    float l_bound;   // lower bound for valid image region, pre-compute for image padding to avoid border check
    float u_bound_width;  // upper width bound for valid image region, pre-compute for image padding to avoid border check
    float u_bound_height;  // upper height bound for valid image region, pre-compute for image padding to avoid border check
    int width_pad;      // width + 2 * padding
    int height_pad;      // height + 2 * padding
    float scale_fact;   // scaling factor at current scale
    int curr_lvl;    // current level
  } img_params ;


  typedef struct {
    // Explicitly set parameters:
    int coarsest_scale;
    int finest_scale;
    int patch_size;
    float patch_stride;
    bool use_mean_normalization;
    // Termination
    int grad_descent_iter;
    float dp_thresh;
    float dr_thresh;
    float res_thresh;
    // Verbosity, 0: plot nothing, 1: final internal timing 2: complete iteration timing, (UNCOMMENTED -> 3: Display flow scales, 4: Display flow scale iterations)
    int verbosity;
    bool use_var_ref;
    int var_ref_iter;
    float var_ref_alpha;
    float var_ref_gamma;
    float var_ref_delta;
    float var_ref_sor_weight;         // Successive-over-relaxation weight

    // Automatically set parameters / fixed parameters
    float outlier_thresh;          // displacement threshold (in px) before a patch is flagged as outlier
    int steps;                    // horizontal and vertical distance (in px) between patch centers
    int n_vals;                   // number of points in patch (=p_samp_s*p_samp_s)
    int n_scales;                 // total number of scales
    float min_errval = 2.0f;       // 1/max(this, error) for pixel averaging weight
    float norm_outlier = 5.0f;     // norm error threshold for huber norm

    // cuBLAS
    cublasHandle_t cublasHandle;

    // Helper variables
    // v4sf zero     = (v4sf) {0.0f, 0.0f, 0.0f, 0.0f};
    // v4sf negzero  = (v4sf) {-0.0f, -0.0f, -0.0f, -0.0f};
    // v4sf half     = (v4sf) {0.5f, 0.5f, 0.5f, 0.5f};
    // v4sf ones     = (v4sf) {1.0f, 1.0f, 1.0f, 1.0f};
    // v4sf twos     = (v4sf) {2.0f, 2.0f, 2.0f, 2.0f};
    // v4sf fours    = (v4sf) {4.0f, 4.0f, 4.0f, 4.0f};
    // v4sf norm_outlier_tmpbsq;
    // v4sf norm_outlier_tmp2bsq;
    // v4sf norm_outlier_tmp4bsq;
  } opt_params;

}

#endif /* PARAMS_HEADER */
