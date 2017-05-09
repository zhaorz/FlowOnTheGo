// Class implements step (3.) in Algorithm 1 of the paper:
// It finds the displacement of one patch from reference/template image to the closest-matching patch in target image via gradient descent.

#ifndef PAT_HEADER
#define PAT_HEADER

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include "params.h" // For camera intrinsic and opt. parameter struct

namespace OFC {

  typedef struct {
    bool has_converged;
    bool has_opt_started;

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

    float cost = 0.0;
  } patch_state;


  class PatClass {

    public:
      PatClass(const img_params* _i_params,
          const opt_params* _op,
          const int _patch_id);

      ~PatClass();

      void InitializePatch(const float * _I0, const float * _I0x,
          const float * _I0y, const Eigen::Vector2f _midpoint);
      void SetTargetImage(const float * _I1);

      void OptimizeIter(const Eigen::Vector2f p_prev);

      inline const bool IsConverged() const { return p_state->has_converged; }
      inline const bool HasOptStarted() const { return p_state->has_opt_started; }
      inline const Eigen::Vector2f GetTargMidpoint() const { return p_state->midpoint_cur; }
      inline const bool IsValid() const { return !p_state->invalid; }
      inline float * GetDeviceCostDiffPtr() const { return (float*) pDeviceCostDiff; }


      inline const Eigen::Vector2f* GetCurP() const { return &(p_state->p_cur); }
      inline const Eigen::Vector2f* GetOrgP() const { return &(p_state->p_org); }
      inline const int GetPatchId() const { return patch_id; }

      struct timeval tv_start, tv_end;
      double extractTime, hessianTime, projectionTime, costTime, interpolateTime;

    private:

      void OptimizeStart(const Eigen::Vector2f p_prev);

      void OptimizeComputeErrImg();
      void UpdateMidpoint();
      void ResetPatchState();
      void ComputeHessian();
      void ComputeCostErr();

      // Extract patch on integer position, and gradients, No Bilinear interpolation
      void ExtractPatch();
      // Extract patch on float position with bilinear interpolation, no gradients.
      void InterpolatePatch();


      const float* pDeviceI;

      float* pDevicePatch;
      float* pDevicePatchX;
      float* pDevicePatchY;

      float* pDeviceRawDiff;
      float* pDeviceCostDiff;
      float* pDeviceWeights;


      Eigen::Vector2f midpoint; // reference point location
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch;
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch_x;
      Eigen::Matrix<float, Eigen::Dynamic, 1> patch_y;

      const float * I0, * I0x, * I0y;

      const img_params* i_params;
      const opt_params* op;
      const int patch_id;

      patch_state * p_state = nullptr; // current patch state

  };

}

#endif /* PAT_HEADER */
