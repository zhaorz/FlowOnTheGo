/**
 * Implements pyramid construction as a kernel.
 */

// System
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// NVIDIA Perf Primitives
#include <nppi.h>
#include <nppi_filtering_functions.h>

// Local
#include "../common/Exceptions.h"
#include "../common/timer.h"

#include "pyramid.h"

using namespace timer;

namespace cu {

  void constructImgPyramids(
      Npp32f* src, float** Is, float** Ixs, float** Iys,
      Npp32f* pDeviceIx, Npp32f* pDeviceIy, Npp32f* pDeviceTmp,
      Npp32f* pDeviceWew, int width, int height,
      int padding, int nLevels) {

    // Timing
    auto start_total = now();
    double compute_time = 0.0;

    // CV_32FC3 is made up of RGB floats
    int channels = 3;
    size_t elemSize = channels * sizeof(float);

    unsigned int nSrcStep = width * elemSize;

    // Gradient params
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
    NppiSize  oSize   = { width, height };
    NppiPoint oOffset = { 0, 0 };
    NppiSize  oROI    = { width, height };

    // Mask params
    Npp32s nMaskSize = 3;
    Npp32s nAnchor   = 1;  // Kernel is centered over pixel

    // Resize params
    int eInterpolation = NPPI_INTER_LINEAR;    // Linear interpolation
    double scaleX = 0.5;
    double scaleY = 0.5;
    double shiftX = 0.0;
    double shiftY = 0.0;

    // Padding params
    int padWidth = 2 * padding + width;
    int padHeight = 2 * padding + height;
    const Npp32f PAD_VAL[3] = { 0.0, 0.0, 0.0 };

    std::cout << "[start] constructImgPyramids: processing "
      << width << "x" << height << " image" << std::endl;

    Npp32f* pDeviceI = src;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Apply first gradients to Is[0]
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // dx's
    auto start_dx = now();
    NPP_CHECK_NPP(
        // nppiFilterSobelHorizBorder_32f_C3R (
        //   pDeviceI, nSrcStep, oSize, oOffset,
        //   pDeviceIx, nSrcStep, oROI, eBorderType)

        nppiFilterRowBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIx, nSrcStep, oROI,
          pDeviceWew, nMaskSize, nAnchor, eBorderType)
        );
    cudaDeviceSynchronize();
    compute_time += calc_print_elapsed("sobel: Ixs[0]", start_dx);

    // dy's
    auto start_dy = now();
    NPP_CHECK_NPP(
        // nppiFilterSobelVertBorder_32f_C3R (
        //   pDeviceI, nSrcStep, oSize, oOffset,
        //   pDeviceIy, nSrcStep, oROI, eBorderType)

        nppiFilterColumnBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize, oOffset,
          pDeviceIy, nSrcStep, oROI,
          pDeviceWew, nMaskSize, nAnchor, eBorderType)
        );
    cudaDeviceSynchronize();
    compute_time += calc_print_elapsed("sobel: Iys[0]", start_dy);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Pad Is[0] I, dx, dy
    ////////////////////////////////////////////////////////////////////////////////////////////////


    NppiSize oPadSize = { padWidth, padHeight };
    int nDstStep = oPadSize.width * elemSize;

    // Pad original
    NPP_CHECK_NPP(
        nppiCopyReplicateBorder_32f_C3R (
          pDeviceI, nSrcStep, oSize,
          Is[0], nDstStep, oPadSize, padding, padding) );

    // Pad dx, dy
    NPP_CHECK_NPP(
        nppiCopyConstBorder_32f_C3R (
          pDeviceIx, nSrcStep, oSize,
          Ixs[0], nDstStep, oPadSize, padding, padding, PAD_VAL) );
    NPP_CHECK_NPP(
        nppiCopyConstBorder_32f_C3R (
          pDeviceIy, nSrcStep, oSize,
          Iys[0], nDstStep, oPadSize, padding, padding, PAD_VAL) );
    cudaDeviceSynchronize();


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // For every finer level, resize and apply the gradients
    ////////////////////////////////////////////////////////////////////////////////////////////////


    for (int i = 1; i < nLevels; i++) {

      // Get the new size
      NppiRect srcRect = { 0, 0, width, height };
      NppiRect dstRect;
      NPP_CHECK_NPP(
          nppiGetResizeRect (srcRect, &dstRect, scaleX, scaleY, shiftX, shiftY, eInterpolation) );

      std::cout << "constructImgPyramids level " << i << ": "
        << dstRect.width << "x" << dstRect.height << std::endl;

      int nDstStep = dstRect.width * elemSize;

      // Resize I => Tmp
      NPP_CHECK_NPP(
          nppiResizeSqrPixel_32f_C3R (
            pDeviceI, oSize, nSrcStep, srcRect,
            pDeviceTmp, nDstStep, dstRect,
            scaleX, scaleY, shiftX, shiftY, eInterpolation) );

      // Put the resized image back into I
      std::swap(pDeviceI, pDeviceTmp);

      // And update the dimensions
      nSrcStep = nDstStep;
      width = dstRect.width; height = dstRect.height;
      oSize.width = width; oSize.height = height;
      oROI.width  = width; oROI.height  = height;

      // dx's
      auto start_dx = now();
      NPP_CHECK_NPP(
          // nppiFilterSobelHorizBorder_32f_C3R (
          //   pDeviceI,  nSrcStep, oSize, oOffset,
          //   pDeviceIx, nSrcStep, oROI, eBorderType)

          nppiFilterRowBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize, oOffset,
            pDeviceIx, nSrcStep, oROI,
            pDeviceWew, nMaskSize, nAnchor, eBorderType)
          );
      cudaDeviceSynchronize();
      compute_time += calc_print_elapsed("sobel: Ixs[i]", start_dx);

      // dy's
      auto start_dy = now();
      NPP_CHECK_NPP(
          // nppiFilterSobelVertBorder_32f_C3R (
          //   pDeviceI, nSrcStep, oSize, oOffset,
          //   pDeviceIy, nSrcStep, oROI, eBorderType)

          nppiFilterColumnBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize, oOffset,
            pDeviceIy, nSrcStep, oROI,
            pDeviceWew, nMaskSize, nAnchor, eBorderType)
          );
      cudaDeviceSynchronize();
      compute_time += calc_print_elapsed("sobel: Iys[i]", start_dy);

      //////////////////////////////////////////////////////////////////////////////////////////////
      // Pad I, dx, dy
      //////////////////////////////////////////////////////////////////////////////////////////////

      padWidth  = width + 2 * padding;
      padHeight = height + 2 * padding;
      oPadSize = { padWidth, padHeight };
      nDstStep = oPadSize.width * elemSize;

      // Pad original
      NPP_CHECK_NPP(
          nppiCopyReplicateBorder_32f_C3R (
            pDeviceI, nSrcStep, oSize,
            Is[i], nDstStep, oPadSize, padding, padding) );

      // Pad dx, dy
      NPP_CHECK_NPP(
          nppiCopyConstBorder_32f_C3R (
            pDeviceIx, nSrcStep, oSize,
            Ixs[i], nDstStep, oPadSize, padding, padding, PAD_VAL) );
      NPP_CHECK_NPP(
          nppiCopyConstBorder_32f_C3R (
            pDeviceIy, nSrcStep, oSize,
            Iys[i], nDstStep, oPadSize, padding, padding, PAD_VAL) );
      cudaDeviceSynchronize();

    }

    calc_print_elapsed("total time", start_total);
    std::cout << "[done] constructImgPyramids: primmary compute time: " << compute_time  << std::endl;
  }

}
