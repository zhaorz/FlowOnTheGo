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

    float H00, H01, H11;
    float p_orgx, p_orgy;
    float p_curx, p_cury;
    float delta_px, delta_py;

    // start positions, current point position, patch norm
    float midpoint_curx, midpoint_cury;
    float midpoint_orgx, midpoint_orgy;

    float delta_p_sq_norm = 1e-10;
    float delta_p_sq_norm_init = 1e-10;
    float mares = 1e20; // mares: Mean Absolute RESidual
    float mares_old = 1e20;
    int count = 0;
    bool invalid = false;

    float cost = 0.0;
  } dev_patch_state;

}

#endif /* PAT_HEADER */
