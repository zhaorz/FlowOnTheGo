#ifndef VARREF_HEADER
#define VARREF_HEADER

#include "FDF1.0.1/image.h"
#include "FDF1.0.1/opticalflow_aux.h"
#include "FDF1.0.1/solver.h"

#include "oflow.h"

namespace OFC {

  typedef __v4sf v4sf;

  typedef struct {

    float alpha;             // smoothness weight
    float beta;              // matching weight
    float gamma;             // gradient constancy assumption weight
    float delta;             // color constancy assumption weight
    int inner_iter;
    int solve_iter;          // number of solver iterations
    float sor_omega;         // omega parameter of sor method

    float tmp_quarter_alpha;
    float tmp_half_gamma_over3;
    float tmp_half_delta_over3;
    float tmp_half_beta;

  } VR_params;


  class VarRefClass {

    public:
      VarRefClass(const float * _I0, const float * _I1,
          const img_params* _i_params, const opt_params* _op, float *flowout);
      ~VarRefClass();

    private:

      convolution_t *deriv, *deriv_flow;

      void copyimage(const float* img, color_image_t * img_t);
      void RefLevelOF(image_t *wx, image_t *wy, const color_image_t *im1, const color_image_t *im2);

      VR_params vr;
      const img_params* i_params;
      const opt_params* op;

  };

}

#endif /* VARREF_HEADER */
