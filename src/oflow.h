// Class implements main flow computation loop over all scales

#ifndef OFC_HEADER
#define OFC_HEADER

#include <nppi.h>
#include <cublas_v2.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "params.h"
#include "patchgrid.h"
#include "refine_variational.h"

using std::cout;
using std::endl;

namespace OFC {

  class OFClass {

    public:
      OFClass(opt_params _op, img_params _i_params);
      ~OFClass();

      void first(Npp32f* _I1, img_params _iparams);
      void next(Npp32f* _I1, img_params _iparams, float * initflow, float * outflow);

    private:
      void ConstructImgPyramids(img_params iparams);

      Npp32f* I0, * I1;

      float ** I0s, ** I0xs, ** I0ys;
      float ** I1s, ** I1xs, ** I1ys;

      opt_params op;                     // Struct for optimization parameters
      std::vector<img_params> iparams;    // Struct (for each scale) for image parameter

      std::vector<PatGridClass*> grid;
      std::vector<float*> flow;

      // Temp images to speedup pyramid generation
      Npp32f* pDeviceIx, *pDeviceIy, *pDeviceTmp, *pDeviceWew;
  };

}

#endif /* OFC_HEADER */
