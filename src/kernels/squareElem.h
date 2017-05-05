/**
 * Implements a resize kernel
 */

#ifndef __KERNEL_SQUARE_ELEM_H__
#define __KERNEL_SQUARE_ELEM_H__

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

  void squareElem(
      float* a, float* b, int N);

}

#endif // end __KERNEL_SQUARE_ELEM_H__

