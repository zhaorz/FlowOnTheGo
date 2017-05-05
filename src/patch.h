// Class implements step (3.) in Algorithm 1 of the paper:
// It finds the displacement of one patch from reference/template image to the closest-matching patch in target image via gradient descent.

#ifndef PAT_HEADER
#define PAT_HEADER

#include <cublas_v2.h>

#include "oflow.h" // For camera intrinsic and opt. parameter struct

namespace OFC {

  typedef struct {
    bool has_converged;
    bool has_opt_started;

    // reference/template patch
    Eigen::Matrix<float, Eigen::Dynamic, 1> raw_diff; // image error to reference image
    Eigen::Matrix<float, Eigen::Dynamic, 1> cost_diff; // absolute error image

    float cost; // absolute error

    Eigen::Matrix<float, 2, 2> hessian; // Hessian for optimization
    Eigen::Vector2f p_org, p_cur, delta_p; // point position, displacement to starting position, iteration update

    // start positions, current point position, patch norm
    Eigen::Matrix<float,1,1> norm;
    Eigen::Vector2f midpoint_cur;
    Eigen::Vector2f midpoint_org;

    float delta_p_sq_norm = 1e-10;
    float delta_p_sq_norm_init = 1e-10;
    float mares = 1e20; // mares: Mean Absolute RESidual
    float mares_old = 1e20;
    int count = 0;
    bool invalid = false;
  } patch_state;


  class PatClass {

    public:
      PatClass(const img_params* _i_params,
          const opt_params* _op,
          const int _patch_id);

      ~PatClass();

      void InitializePatch(Eigen::Map<const Eigen::MatrixXf> * _I0, Eigen::Map<const Eigen::MatrixXf> * _I0x, Eigen::Map<const Eigen::MatrixXf> * _I0y, const Eigen::Vector2f _midpoint);
      void SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * _I1, Eigen::Map<const Eigen::MatrixXf> * _I1x, Eigen::Map<const Eigen::MatrixXf> * _I1y);

      void OptimizeIter(const Eigen::Vector2f p_prev);

      inline const bool IsConverged() const { return p_state->has_converged; }
      inline const bool HasOptStarted() const { return p_state->has_opt_started; }
      inline const Eigen::Vector2f GetTargMidpoint() const { return p_state->midpoint_cur; }
      inline const bool IsValid() const { return !p_state->invalid; }
      inline const float * GetCostDiffPtr() const { return (float*) p_state->cost_diff.data(); }

      inline const Eigen::Vector2f* GetCurP() const { return &(p_state->p_cur); }
      inline const Eigen::Vector2f* GetOrgP() const { return &(p_state->p_org); }

    private:

      cublasHandle_t handle;
      cublasStatus_t stat;

      void OptimizeStart(const Eigen::Vector2f p_prev);

      void OptimizeComputeErrImg();
      void UpdateMidpoint();
      void ResetPatchState();
      void ComputeHessian();
      void InitializeError();
      void ComputeCostErr();

      // Extract patch on integer position, and gradients, No Bilinear interpolation
      void ExtractPatch();
      // Extract patch on float position with bilinear interpolation, no gradients.
      void InterpolatePatch();

      Eigen::Vector2f midpoint; // reference point location
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch;
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch_x;
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch_y;

      float* pDevicePatch;
      float* pDevicePatchX;
      float* pDevicePatchY;

      float* pDeviceRawDiff;
      float* pDeviceCostDiff;

      Eigen::Map<const Eigen::MatrixXf> * I0, * I0x, * I0y;
      Eigen::Map<const Eigen::MatrixXf> * I1, * I1x, * I1y;

      const img_params* i_params;
      const opt_params* op;
      const int patch_id;

      patch_state * p_state = nullptr; // current patch state

  };

}

#endif /* PAT_HEADER */
