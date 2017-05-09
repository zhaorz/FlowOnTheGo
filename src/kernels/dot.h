#ifndef __KERNEL_CUSTOM_DOT_H__
#define __KERNEL_CUSTOM_DOT_H__

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
  float dot(
      float* A, float* B, int N);

}

#endif // end __KERNEL_CUSTOM_DOT_H__

