/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_UTIL_NPP_EXCEPTIONS_H
#define NV_UTIL_NPP_EXCEPTIONS_H


#include <string>
#include <sstream>
#include <iostream>

#include "cuda_helper.h"

static const char *_cudaGetErrorEnum(NppStatus eStatusNPP)
{
  switch (eStatusNPP)
  {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";

    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";

    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";

    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";

    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";

    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";

    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";

    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";

    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";

    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";

    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";

    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";

    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";

    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";

    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

    case NPP_CORRUPTED_DATA_ERROR:
      return "NPP_CORRUPTED_DATA_ERROR";

    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";

    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";

    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";

    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";

    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";

    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";

    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";

    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";

    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";

    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";

    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";

    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";

    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";

    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";

    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";

    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";

    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERR";

    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";

    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";

    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";

    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";

    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";

    case NPP_ERROR:
      return "NPP_ERROR";

    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";

    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";

    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";

    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";

    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";

    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";

    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";

    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";

    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";

    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";

    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";

    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";

    default:
      return "<unknown>";
  }

}


/// All npp related C++ classes are put into the npp namespace.
namespace npp
{

  /// Exception base class.
  ///     This exception base class will be used for everything C++ throught
  /// the NPP project.
  ///     The exception contains a string message, as well as data fields for a string
  /// containing the name of the file as well as the line number where the exception was thrown.
  ///     The easiest way of throwing exceptions and providing filename and line number is
  /// to use one of the ASSERT macros defined for that purpose.
  class Exception
  {
    public:
      /// Constructor.
      /// \param rMessage A message with information as to why the exception was thrown.
      /// \param rFileName The name of the file where the exception was thrown.
      /// \param nLineNumber Line number in the file where the exception was thrown.
      explicit
        Exception(const std::string &rMessage = "", const std::string &rFileName = "", unsigned int nLineNumber = 0)
        : sMessage_(rMessage), sFileName_(rFileName), nLineNumber_(nLineNumber)
        { };

      Exception(const Exception &rException)
        : sMessage_(rException.sMessage_), sFileName_(rException.sFileName_), nLineNumber_(rException.nLineNumber_)
      { };

      virtual
        ~Exception()
        { };

      /// Get the exception's message.
      const
        std::string &
        message()
        const
        {
          return sMessage_;
        }

      /// Get the exception's file info.
      const
        std::string &
        fileName()
        const
        {
          return sFileName_;
        }

      /// Get the exceptions's line info.
      unsigned int
        lineNumber()
        const
        {
          return nLineNumber_;
        }


      /// Create a clone of this exception.
      ///      This creates a new Exception object on the heap. It is
      /// the responsibility of the user of this function to free this memory
      /// (delete x).
      virtual
        Exception *
        clone()
        const
        {
          return new Exception(*this);
        }

      /// Create a single string with all the exceptions information.
      ///     The virtual toString() method is used by the operator<<()
      /// so that all exceptions derived from this base-class can print
      /// their full information correctly even if a reference to their
      /// exact type is not had at the time of printing (i.e. the basic
      /// operator<<() is used).
      virtual
        std::string
        toString()
        const
        {
          std::ostringstream oOutputString;
          oOutputString << fileName() << ":" << lineNumber() << ": " << message();
          return oOutputString.str();
        }

    private:
      std::string sMessage_;      ///< Message regarding the cause of the exception.
      std::string sFileName_;     ///< Name of the file where the exception was thrown.
      unsigned int nLineNumber_;  ///< Line number in the file where the exception was thrown
  };

  /// Output stream inserter for Exception.
  /// \param rOutputStream The stream the exception information is written to.
  /// \param rException The exception that's being written.
  /// \return Reference to the output stream being used.
  inline std::ostream &
    operator << (std::ostream &rOutputStream, const Exception &rException)
    {
      rOutputStream << rException.toString();
      return rOutputStream;
    }

  /// Basic assert macro.
  ///     This macro should be used to enforce any kind of pre or post conditions.
  /// Unlike the C-runtime assert macro, this macro does not abort execution, but throws
  /// a C++ exception. The exception is automatically filled with information about the failing
  /// condition, the filename and line number where the exception was thrown.
  /// \note The macro is written in such a way that omitting a semicolon after its usage
  ///     causes a compiler error. The correct way to invoke this macro is:
  /// NPP_ASSERT(n < MAX);
#define NPP_ASSERT(C) do {if (!(C)) throw npp::Exception(#C " assertion faild!", __FILE__, __LINE__);} while(false)

  // ASSERT macro.
  //  Same functionality as the basic assert macro with the added ability to pass
  //  a message M. M should be a string literal.
  //  Note: Never use code inside ASSERT() that causes a side-effect ASSERT macros may get compiled
  //      out in release mode.
#define NPP_ASSERT_MSG(C, M) do {if (!(C)) throw npp::Exception(#C " assertion faild! Message: " M, __FILE__, __LINE__);} while(false)

#ifdef _DEBUG
  /// Basic debug assert macro.
  ///     This macro is identical in every respect to NPP_ASSERT(C) but it does get compiled to a
  /// no-op in release builds. It is therefor of utmost importance to not put statements into
  /// this macro that cause side effects required for correct program execution.
#define NPP_DEBUG_ASSERT(C) do {if (!(C)) throw npp::Exception(#C " debug assertion faild!", __FILE__, __LINE__);} while(false)
#else
#define NPP_DEBUG_ASSERT(C)
#endif

  /// ASSERT for null-pointer test.
  /// It is safe to put code with side effects into this macro. Also: This macro never
  /// gets compiled to a no-op because resource allocation may fail based on external causes not under
  /// control of a software developer.
#define NPP_ASSERT_NOT_NULL(P) do {if ((P) == 0) throw npp::Exception(#P " not null assertion faild!", __FILE__, __LINE__);} while(false)

  /// Macro for flagging methods as not implemented.
  /// The macro throws an exception with a message that an implementation was missing
#define NPP_NOT_IMPLEMENTED() do {throw npp::Exception("Implementation missing!", __FILE__, __LINE__);} while(false)

  /// Macro for checking error return code of CUDA (runtime) calls.
  /// This macro never gets disabled.
#define NPP_CHECK_CUDA(S) do {cudaError_t eCUDAResult; \
  eCUDAResult = S; \
  if (eCUDAResult != cudaSuccess) std::cout << "NPP_CHECK_CUDA - eCUDAResult = " << eCUDAResult << std::endl; \
  NPP_ASSERT(eCUDAResult == cudaSuccess);} while (false)

  /// Macro for checking error return code for NPP calls.
#define NPP_CHECK_NPP(S) do {NppStatus eStatusNPP; \
  eStatusNPP = S; \
  if (eStatusNPP != NPP_SUCCESS) std::cout << "NPP_CHECK_NPP - eStatusNPP = " << _cudaGetErrorEnum(eStatusNPP) << "("<< eStatusNPP << ")" << std::endl; \
  NPP_ASSERT(eStatusNPP == NPP_SUCCESS);} while (false)

  /// Macro for checking error return codes from cuFFT calls.
#define NPP_CHECK_CUFFT(S) do {cufftResult eCUFFTResult; \
  eCUFFTResult = S; \
  if (eCUFFTResult != NPP_SUCCESS) std::cout << "NPP_CHECK_CUFFT - eCUFFTResult = " << eCUFFTResult << std::endl; \
  NPP_ASSERT(eCUFFTResult == CUFFT_SUCCESS);} while (false)

} // npp namespace

#endif // NV_UTIL_NPP_EXCEPTIONS_H
