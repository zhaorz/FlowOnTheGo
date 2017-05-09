#ifndef __KERNEL_THRUST_DOT_H__
#define __KERNEL_THRUST_DOT_H__

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// Local
#include "../common/Exceptions.h"
#include "../common/timer.h"
#include "../sandbox/process.h"

namespace cu {

  /* Computes <A,B> */
  float thrustDot(
      float* A, float* B, int N);

}

#endif // end __KERNEL_THRUST_DOT_H__

