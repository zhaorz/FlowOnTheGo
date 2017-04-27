// Class implements main flow computation loop over all scales

#ifndef OFC_HEADER
#define OFC_HEADER

using std::cout;
using std::endl;

namespace OFC {

  typedef __v4sf v4sf;


  typedef struct {
    int width;                // image width, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
    int height;               // image height, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
    int imgpadding;           // image padding in pixels at all sides, images padded with replicated border, gradients padded with zero, ADD THIS ONLY WHEN ADDRESSING THE IMAGE OR GRADIENT
    float tmp_lb;             // lower bound for valid image region, pre-compute for image padding to avoid border check
    float tmp_ubw;            // upper width bound for valid image region, pre-compute for image padding to avoid border check
    float tmp_ubh;            // upper height bound for valid image region, pre-compute for image padding to avoid border check
    int tmp_w;                // width + 2*imgpadding
    int tmp_h;                // height + 2*imgpadding
    float sc_fct;             // scaling factor at current scale
    int curr_lv;              // current level
    int camlr;                // 0: left camera, 1: right camera, used only for depth, to restrict sideways patch motion
  } camparam ;

  typedef struct {
    // Explicitly set parameters:
    int coarsest_scale;
    int finest_scale;
    int patch_size;
    int grad_descent_iter;
    bool use_mean_normalization;
    // Verbosity, 0: plot nothing, 1: final internal timing 2: complete iteration timing, (UNCOMMENTED -> 3: Display flow scales, 4: Display flow scale iterations)
    int verbosity;
    // Cost function: 0: L2-Norm, 1: L1-Norm, 2: PseudoHuber-Norm
    int cost_func;
    bool use_var_ref;
    int var_ref_iter;
    float var_ref_alpha;
    float var_ref_gamma;
    float var_ref_delta;
    float var_ref_sor_weight;         // Successive-over-relaxation weight

    // Automatically set parameters / fixed parameters
    float patch_stride;
    float outlierthresh;          // displacement threshold (in px) before a patch is flagged as outlier
    int steps;                    // horizontal and vertical distance (in px) between patch centers
    int novals;                   // number of points in patch (=p_samp_s*p_samp_s)
    int noscales;                 // total number of scales
    float minerrval = 2.0f;       // 1/max(this, error) for pixel averaging weight
    float normoutlier = 5.0f;     // norm error threshold for huber norm

    // Helper variables
    v4sf zero     = (v4sf) {0.0f, 0.0f, 0.0f, 0.0f};
    v4sf negzero  = (v4sf) {-0.0f, -0.0f, -0.0f, -0.0f};
    v4sf half     = (v4sf) {0.5f, 0.5f, 0.5f, 0.5f};
    v4sf ones     = (v4sf) {1.0f, 1.0f, 1.0f, 1.0f};
    v4sf twos     = (v4sf) {2.0f, 2.0f, 2.0f, 2.0f};
    v4sf fours    = (v4sf) {4.0f, 4.0f, 4.0f, 4.0f};
    v4sf normoutlier_tmpbsq;
    v4sf normoutlier_tmp2bsq;
    v4sf normoutlier_tmp4bsq;

  } optparam;



  class OFClass {

    public:
      OFClass(const float ** im_ao_in, const float ** im_ao_dx_in, const float ** im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients.
          // E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
          // im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
          // IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0,
          const float ** im_bo_in, const float ** im_bo_dx_in, const float ** im_bo_dy_in,
          const int imgpadding_in,
          float * outflow,          // Output-flow:         has to be of size to fit the last  computed OF scale [width / 2^(last scale)   , height / 2^(last scale)]   , 1 channel depth / 2 for OF
          const float * initflow,   // Initialization-flow: has to be of size to fit the first computed OF scale [width / 2^(first scale+1), height / 2^(first scale+1)], 1 channel depth / 2 for OF
          const int width_in, const int height_in,
          const int sc_f_in, const int sc_l_in,
          const int grad_descent_iter_in,
          const int padval_in,
          const float patch_stride_in,
          const int cost_func_in,
          const bool patnorm_in,
          const bool use_var_ref_in,
          const float var_ref_alpha_in,
          const float var_ref_gamma_in,
          const float var_ref_delta_in,
          const int var_ref_iter,
          const float var_ref_sor_in,
          const int verbosity_in);

    private:

      // needed for verbosity >= 3, DISVISUAL
      //void DisplayDrawPatchBoundary(cv::Mat img, const Eigen::Vector2f pt, const float sc);

      const float ** im_ao, ** im_ao_dx, ** im_ao_dy;
      const float ** im_bo, ** im_bo_dx, ** im_bo_dy;

      optparam op;                    // Struct for pptimization parameters
      std::vector<camparam> cpl, cpr; // Struct (for each scale) for camera/image parameter
  };


}

#endif /* OFC_HEADER */
