 /* Copyright 2009-2012 NVIDIA Corporation.  All rights reserved. 
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to NVIDIA intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and 
  * conditions of a form of NVIDIA software license agreement by and 
  * between NVIDIA and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of NVIDIA is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
  * OF THESE LICENSED DELIVERABLES. 
  * 
  * U.S. Government End Users.  These Licensed Deliverables are a 
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
  * 1995), consisting of "commercial computer software" and "commercial 
  * computer software documentation" as such terms are used in 48 
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
  * U.S. Government End Users acquire the Licensed Deliverables with 
  * only those rights set forth herein. 
  * 
  * Any use of the Licensed Deliverables in individual and commercial 
  * software must include, in the user documentation and internal 
  * comments to the code, the above Disclaimer and U.S. Government End 
  * Users Notice. 
  */ 
#ifndef NV_NPPI_STATISTICS_FUNCTIONS_H
#define NV_NPPI_STATISTICS_FUNCTIONS_H
 
/**
 * \file nppi_statistics_functions.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_statistics_functions Statistics Functions
 *  @ingroup nppi
 *
 * Routines computing statistical image information.
 *
 *
 */
///@{

/** @defgroup image_sum Sum
 */
///@{

/** @name Sum
 *  Sum functions compute the sum of all the pixel values in an image. If the image contains multiple 
 *  channels, the sums will be calculated for each channel separately. Functions also require 
 *  scratch buffer during the computation. For details, please refer \ref general_scratch_buffer.
 *  The nppiSumGetBuffer_X_X functions compute the size of the scratch buffer. It is the user's
 *  responsibility to allocate the sufficient GPU memory based on the size and pass the memory pointer
 *  to the sum functions.
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiSum_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * Device scratch buffer size (in bytes) for nppiSum_8u64s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u64s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);


/** 
 * Device scratch buffer size (in bytes) for nppiSum_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_8u64s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u64s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiSum_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiSumGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);
 
/**
 * 1-channel 8-bit unsigned char image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pSum Pointer to the computed sum.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pSum);

/**
 * 1-channel 8-bit unsigned char image sum with 64-bit long long result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_8u64s_C1R to determine the minium number of bytes required.
 * \param pSum Pointer to the computed sum.
 */
NppStatus 
nppiSum_8u64s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64s * pSum);

/**
 * 1-channel 16-bit unsigned short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pSum Pointer to the computed sum.
 */
NppStatus 
nppiSum_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pSum);

/**
 * 1-channel 16-bit signed short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pSum Pointer to the computed sum.
 */
NppStatus 
nppiSum_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pSum);

/**
 * 1-channel 32-bit floating point image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pSum Pointer to the computed sum.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pSum);

/**
 * 3-channel 8-bit unsigned char image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 */
NppStatus 
nppiSum_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 3-channel 16-bit unsigned short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 */
NppStatus 
nppiSum_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 3-channel 16-bit signed short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 */NppStatus 
nppiSum_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 3-channel 32-bit floating point image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 */
NppStatus 
nppiSum_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 4-channel 8-bit unsigned char image sum with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel (alpha channel is not computed).
*/
NppStatus 
nppiSum_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 4-channel 16-bit unsigned short image sum with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel (alpha channel is not computed).
*/
NppStatus 
nppiSum_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 4-channel 16-bit signed short image sum with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiSumGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel (alpha channel is not computed).
*/
NppStatus 
nppiSum_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 4-channel 32-bit floating point image sum with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel (alpha channel is not computed).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[3]);

/**
 * 4-channel 8-bit unsigned char image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[4]);

/**
 * 4-channel 8-bit unsigned char image sum with 64-bit long long result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_8u64s_C4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u64s_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64s aSum[4]);

/**
 * 4-channel 16-bit unsigned short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[4]);

/**
 * 4-channel 16-bit signed short image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[4]);

/**
 * 4-channel 32-bit floating point image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiSumGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aSum Array that contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[4]);

///@}

///@} image_sum

/** @defgroup image_min Minimum
 */
///@{

/** @name Min
 *  These min routines find the minimal pixel value of an image. If the image has multiple channels,
 *  the functions find the minimum for each channel separately. The scratch buffer is also required
 *  by the functions.
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiMin_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMin_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image min. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pMin);

/**
 * 1-channel 16-bit unsigned short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u * pMin);

/**
 * 1-channel 16-bit signed short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s * pMin);

/**
 * 1-channel 32-bit floating point image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f * pMin);

/**
 * 3-channel 8-bit unsigned char image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[3]);

/**
 * 3-channel 16-bit unsigned short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[3]);

/**
 * 3-channel 16-bit signed short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[3]);

/**
 * 3-channel 32-bit floating point image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[3]);

/**
 * 4-channel 8-bit unsigned char image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[4]);

/**
 * 4-channel 16-bit unsigned short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus 
nppiMin_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[4]);

/**
 * 4-channel 16-bit signed short integer image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus 
nppiMin_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[4]);

/**
 * 4-channel 32-bit floating point image min.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus 
nppiMin_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[4]);

/**
 * 4-channel 8-bit unsigned char image min (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[3]);

/**
 * 4-channel 16-bit unsigned short integer image min (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[3]);

/**
 * 4-channel 16-bit signed short integer image min (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[3]);

/**
 * 4-channel 32-bit floating point image min (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMin_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[3]);

///@}

/** @name MinIndx
 *  The functions find the minimal value and its indices (X and Y coordinates) of an image. If the image contains
 *  multiple channels, the function will find the values and the indices for each channel separately. 
 *  If there are several minima in the selected region of interest, the function returns the top leftmost position. 
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiMinIndx_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinIndx_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinIndxGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image min with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image min value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image min value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pMin, int * pIndexX, int * pIndexY);

/**
 * 1-channel 16-bit unsigned short integer image min with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image min value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image min value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u * pMin, int * pIndexX, int * pIndexY);

/**
 * 1-channel 16-bit signed short integer image min with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image min value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image min value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s * pMin, int * pIndexX, int * pIndexY);

/**
 * 1-channel 32-bit floating point image min with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image min value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image min value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f * pMin, int * pIndexX, int * pIndexY);

/**
 * 3-channel 8-bit unsigned char image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 16-bit unsigned short integer image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 16-bit signed short integer image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 32-bit floating point image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 8-bit unsigned char image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 16-bit unsigned short integer image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 16-bit signed short integer image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 32-bit floating point image min values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 8-bit unsigned char image min values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for the three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for the three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 16-bit unsigned short integer image min values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for the three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for the three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 16-bit signed short integer image min values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for the three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for the three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMin[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 32-bit floating point image min values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMinIndxGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aMin Device-memory array receiving the minimum result, three elements for the three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image min values, three elements for the three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image min values, three elements for the three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinIndx_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMin[3], int aIndexX[3], int aIndexY[3]);

///@}

///@} image_min

/** @defgroup image_max Maximum
 */
///@{

/** @name Max
 *  These max routines find the maximal pixel value of an image. If the image has multiple channels,
 *  the functions find the maximum for each channel separately. The scratch buffer is also required
 *  by the functions.
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiMax_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMaxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMax_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pMax);

/**
 * 1-channel 16-bit unsigned short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u * pMax);

/**
 * 1-channel 16-bit signed short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s * pMax);

/**
 * 1-channel 32-bit floating point image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f * pMax);

/**
 * 3-channel 8-bit unsigned char image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[3]);

/**
 * 3-channel 16-bit unsigned short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[3]);

/**
 * 3-channel 16-bit signed short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[3]);

/**
 * 3-channel 32-bit floating point image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[3]);

/**
 * 4-channel 8-bit unsigned char image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[4]);

/**
 * 4-channel 16-bit unsigned short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[4]);

/**
 * 4-channel 16-bit signed short integer image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[4]);

/**
 * 4-channel 32-bit floating point image max.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum results, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[4]);

/**
 * 4-channel 8-bit unsigned char image max (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[3]);

/**
 * 4-channel 16-bit unsigned short integer image max (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[3]);

/**
 * 4-channel 16-bit signed short integer image max (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[3]);

/**
 * 4-channel 32-bit floating point image max (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMax_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[3]);

///@}

/** @name MaxIndx
 *  The functions find the max value and its indices (X and Y coordinates) of an image. If the image contains multiple
 *  channels, the functions finds the values and their indices for each channel separately.
 *  If there are several maxima in the selected region of interest, the function returns the top leftmost position.
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiMaxIndx_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMaxIndxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMaxIndx_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMaxIndxGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image max value with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image max value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image max value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u * pMax, int * pIndexX, int * pIndexY);

/**
 * 1-channel 16-bit unsigned short integer image max value with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image max value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image max value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u * pMax, int * pIndexX, int * pIndexY);

/**
 * 1-channel 16-bit signed short integer image max value with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image max value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image max value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s * pMax, int * pIndexX, int * pIndexY);

/**
 * 1-channel 32-bit floating point image max value with its X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pIndexX Device-memory pointer to the X coordinate of the image max value.
 * \param pIndexY Device-memory pointer to the Y coordinate of the image max value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f * pMax, int * pIndexX, int * pIndexY);

/**
 * 3-channel 8-bit unsigned char image max values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 16-bit unsigned short integer image max values with their X and Y coordinates.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 16-bit signed short integer image max values with their X and Y coordinates.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 3-channel 32-bit floating point image max values with their X and Y coordinates.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus 
nppiMaxIndx_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 8-bit unsigned char image max valueswith their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 16-bit unsigned short integer image max values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus 
nppiMaxIndx_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 16-bit signed short integer image max values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 32-bit floating point image max values with their X and Y coordinates.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, four elements for four channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[4], int aIndexX[4], int aIndexY[4]);

/**
 * 4-channel 8-bit unsigned char image max values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for the first three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp8u aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 16-bit unsigned short integer image max values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for the first three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16u aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 16-bit signed short integer image max values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for the first three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp16s aMax[3], int aIndexX[3], int aIndexY[3]);

/**
 * 4-channel 32-bit floating point image max values with their X and Y coordinates (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMaxIndxGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param aIndexX Device-memory array to the X coordinates of the image max values, three elements for the first three channels.
 * \param aIndexY Device-memory array to the Y coordinates of the image max values, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMaxIndx_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp32f aMax[3], int aIndexX[3], int aIndexY[3]);

///@}

///@} image_max

/** @defgroup image_min_max Minimum_Maximum
 */
///@{

/** @name MinMax
 *  The functions find the minimum and maximum values of an image. If the image contains multiple channles,
 *  the function find the values for each channel separately. The functions also require the device scratch buffer.
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus
nppiMinMaxGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinManx_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus
nppiMinMaxGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. 
 *        Use \ref nppiMinMaxGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pMin, Npp8u * pMax, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16u_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u * pMin, Npp16u * pMax, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit signed short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16s_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16s * pMin, Npp16s * pMax, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating point image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_32f_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f * pMin, Npp32f * pMax, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned image minimum and maximum  values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_8u_C3R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u aMin[3], Npp8u aMax[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16u_C3R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u aMin[3], Npp16u aMax[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit signed short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16s_C3R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16s aMin[3], Npp16s aMax[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating point image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for three channels..
 * \param aMax Device-memory array receiving the maximum result, three elements for three channels..
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_32f_C3R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f aMin[3], Npp32f aMax[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned image minimum and maximum values (alpha channel is not calculated).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_8u_AC4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u aMin[3], Npp8u aMax[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image minimum and maximum values (alpha channel is not calculated).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16u_AC4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u aMin[3], Npp16u aMax[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image minimum and maximum values (alpha channel is not calculated).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16s_AC4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16s aMin[3], Npp16s aMax[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating point image minimum and maximum values (alpha channel is not calculated).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, three elements for the first three channels.
 * \param aMax Device-memory array receiving the maximum result, three elements for the first three channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_32f_AC4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f aMin[3], Npp32f aMax[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_8u_C4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u aMin[4], Npp8u aMax[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16u_C4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u aMin[4], Npp16u aMax[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_16s_C4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16s aMin[4], Npp16s aMax[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating point image minimum and maximum values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-memory array receiving the minimum result, four elements for four channels.
 * \param aMax Device-memory array receiving the maximum result, four elements for four channels.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferHostSize_32f_C4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f aMin[4], Npp32f aMax[4], Npp8u * pDeviceBuffer);

///@}

/** @name MinMaxIndx
 *  MinMax value and their indices (X and Y coordinates) of images.
 *  If there are several minima and maxima in the selected region of interest, the function returns the top leftmost position.
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8u_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8u_C3CR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8s_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8s_C3CR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_16u_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_16u_C3CR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_32f_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_32f_C3CR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMinMaxIndx_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMinMaxIndxGetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image minimum and maximum values with their indices.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8u_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit signed char image minimum and maximum values with their indices.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8s_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8s_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image minimum and maximum values with their indices.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_16u_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating point image minimum and maximum values with their indices.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_32f_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_EVEN_STEP_ERROR if an invalid floating-point image is specified.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit unsigned char image minimum and maximum values with their indices, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8u_C1MR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit signed char image minimum and maximum values with their indices, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8s_C1MR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image minimum and maximum values with their indices, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_16u_C1MR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes. 
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating point image minimum and maximum values with their indices, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_32f_C1MR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_EVEN_STEP_ERROR if an invalid floating-point image is specified.
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image minimum and maximum values with their indices, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8u_C3CR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8u_C3CR(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit signed char image minimum and maximum values with their indices, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8s_C3CR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8s_C3CR(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image minimum and maximum values with their indices, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_16u_C3CR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_16u_C3CR(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating point image minimum and maximum values with their indices, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_32f_C3CR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_32f_C3CR(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image minimum and maximum values with their indices, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8u_C3CMR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pMinValue, Npp8u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit signed char image minimum and maximum values with their indices, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_8s_C3CMR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8s * pMinValue, Npp8s * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image minimum and maximum values with their indices, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_16u_C3CMR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp16u * pMinValue, Npp16u * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating point image minimum and maximum values with their indices, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pMinValue Device-memory pointer receiving the minimum value.
 * \param pMaxValue Device-memory pointer receiving the maximum value.
 * \param pMinIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the minimum value.
 * \param pMaxIndex Device-memory pointer receiving the indicies (X and Y coordinates) of the maximum value.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxIndxGetBufferHostSize_32f_C3CMR to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 * If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0},
 * pMinValue = 0, pMaxValue = 0.
 * If any of pMinValue, pMaxValue, pMinIndex, or pMaxIndex is not needed, zero pointer must be passed correspondingly.
 */
NppStatus 
nppiMinMaxIndx_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp32f * pMinValue, Npp32f * pMaxValue, NppiPoint * pMinIndex, NppiPoint * pMaxIndex, Npp8u * pDeviceBuffer);

///@}

///@} image_min_max

/** @defgroup image_mean Mean
 */
///@{

/** @name Mean
 *  The functions compute the mean value of all the pixel values in an image. All the mean results are stored in a 64-bit
 *  double precision format. If the image contains multiple channels,the functions calculate the mean for 
 *  each channel separately. The mean functions require additional scratch buffer for computations.
 */
///@{

/**
 * Device scratch buffer size (in bytes) for nppiMean_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMeanGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * Device scratch buffer size (in bytes) for nppiMean_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMeanGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * Device scratch buffer size (in bytes) for nppiMean_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
*/
NppStatus 
nppiMeanGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);
 
/** 
 * Device scratch buffer size (in bytes) for nppiMean_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanGetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 1-channel 8-bit unsigned char image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 16-bit unsigned short integer image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 16-bit signed short integer image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16s_C1R to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 32-bit floating point image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param pMean Device-memory pointer receiving the mean result.
 *        Use \ref nppiMeanGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus 
nppiMean_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 3-channel 8-bit unsigned char image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_C3R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 3-channel 16-bit unsigned short image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_C3R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 3-channel 16-bit signed short image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16s_C3R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for three channels.
 */
NppStatus 
nppiMean_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 3-channel 32-bit floating point image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_32f_C3R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus 
nppiMean_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 4-channel 8-bit unsigned char image mean with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_AC4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 4-channel 16-bit unsigned short image mean with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_AC4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 4-channel 16-bit signed short image mean with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16s_AC4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 4-channel 32-bit floating point image mean with 64-bit double precision result.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_32f_AC4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, three elements for the first three channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus 
nppiMean_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[3]);

/**
 * 4-channel 8-bit unsigned char image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_C4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[4]);

/**
 * 4-channel 16-bit unsigned short image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_C4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[4]);

/**
 * 4-channel 16-bit signed short image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16s_C4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMean_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[4]);

/**
 * 4-channel 32-bit floating point image mean with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_32f_C4R to determine the minium number of bytes required.
 * \param aMean Array that contains computed mean, four elements for four channels.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus 
nppiMean_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aMean[4]);

/**
 * 1-channel 8-bit unsigned char image mean with 64-bit double precision result, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_C1MR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus 
nppiMean_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 8-bit signed char image mean with 64-bit double precision result, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8s_C1MR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus 
nppiMean_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 16-bit unsigned short integer image mean with 64-bit double precision result, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_C1MR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus 
nppiMean_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 1-channel 32-bit floating point image mean with 64-bit double precision result, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_32f_C1MR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus 
nppiMean_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 3-channel 8-bit unsigned char image mean with 64-bit double precision result, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8u_C3CMR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus 
nppiMean_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 3-channel 8-bit signed char image mean with 64-bit double precision result, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_8s_C3CMR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus 
nppiMean_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 3-channel 16-bit unsigned short integer image mean with 64-bit double precision result, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_16u_C3CMR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus 
nppiMean_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean);

/**
 * 3-channel 32-bit floating point image mean with 64-bit double precision result, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanGetBufferHostSize_32f_C3CMR to determine the minium number of bytes required.
 * \param pMean Device-memory pointer receiving the mean result.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus 
nppiMean_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean);

///@}

///@} image_mean

/** @defgroup image_mean_and_standard_deviation Mean And Standard Deviation
 */
///@{

/** @name Mean and Standard Deviation
 *  The routines compute the mean and standard deviation of image pixel values and store them in a 64-bit double precision format.
 *  The functions require the additional device memroy for the computations.
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8s_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8u_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8u_C3CR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8s_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8s_C3CR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_16u_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_16u_C3CR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_32f_C3CR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_32f_C3CR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiMean_StdDev_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDevGetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/**
 * 1-channel 8-bit unsigned char image mean and standard deviation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8u_C1R to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 8-bit signed char image mean and standard deviation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8s_C1R to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_8s_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 16-bit unsigned short int image mean and standard deviation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_16u_C1R to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 32-bit floating-point image mean and standard deviation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_32f_C1R to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus nppiMean_StdDev_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 8-bit unsigned char image mean and standard deviation, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8u_C1MR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 8-bit signed char image mean and standard deviation, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8s_C1MR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 16-bit unsigned short int image mean and standard deviation, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_16u_C1MR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 1-channel 32-bit floating-point image mean and standard deviation, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_32f_C1MR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified.
 */
NppStatus nppiMean_StdDev_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 8-bit unsigned char image mean and standard deviation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8u_C3CR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_8u_C3CR(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 8-bit signed char image mean and standard deviation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8s_C3CR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_8s_C3CR(const Npp8s * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 16-bit unsigned short int image mean and standard deviation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_16u_C3CR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_16u_C3CR(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 32-bit floating-point image mean and standard deviation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_32f_C3CR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_32f_C3CR(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 8-bit unsigned char image mean and standard deviation, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8u_C3CMR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 8-bit signed char image mean and standard deviation, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_8s_C3CMR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 16-bit unsigned short int image mean and standard deviation, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_16u_C3CMR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

/**
 * 3-channel 32-bit floating-point image mean and standard deviation, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 *        Use \ref nppiMeanStdDevGetBufferHostSize_32f_C3CMR to determine the minium number of bytes required.
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiMean_StdDev_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

///@}

///@} image_mean_and_standard_deviation

/** @defgroup image_infinity_norm Infinity Norm
 */
///@{

/** @name Infinity Norm
 *  These functions compute the infinity norm of an image. The infinity norm is defined as the largest pixel value of
 *  the image. If the image contains multiple channles, the functions will compute the norm for each channel separately.
 *  The functions require the addition device scratch buffer for the computations.
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32s_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormInf_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormInfGetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/**
 * 1-channel 8-bit unsigned char image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit signed short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16s_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit signed int image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32s_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32s_C1R(const Npp32s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating-point image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit unsigned char image infinity norm, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit signed char image infinity norm, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8s_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image infinity norm, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating-point image infinity norm, \ref masked_operation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit signed short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16s_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image infinity norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image infinity norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image infinity norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16s_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image infinity norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16s_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image infinity norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_Inf_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image infinity norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_Inf_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit signed char image infinity norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_8s_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_Inf_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image infinity norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_16u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_Inf_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image infinity norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormInfGetBufferHostSize_32f_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if an invalid 
 * floating-point image is specified, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_Inf_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

///@}

///@} image_infinity_norm

/** @defgroup image_L1_norm L1 Norm
 */
///@{

/** @name L1 Norm
 *  These functions compute the L1 norm of an image. The L1 norm is defined as the sum of all the absolute pixle values
 *  in the image. If the image contains multiple channles, the functions will compute the norm for each channel separately.
 *  The functions require the addition device scratch buffer for the computations.
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL1_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL1GetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/**
 * 1-channel 8-bit unsigned char image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit signed short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16s_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating-point image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit unsigned char image L1 norm, \ref masked_operation
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 8-bit signed char image L1 norm, \ref masked_operation
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8s_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image L1 norm, \ref masked_operation
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating-point image L1 norm, \ref masked_operation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit signed short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16s_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image L1 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image L1 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image L1 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16s_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image L1 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16s_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image L1 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L1_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image L1 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L1_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit signed char image L1 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_8s_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L1_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image L1 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_16u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L1_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image L1 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL1GetBufferHostSize_32f_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if the step of 
 * the source image cannot be divided by 4, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L1_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

///@}

///@} image_L1_norm

/** @defgroup image_L2_norm L2 Norm
 */
///@{

/** @name L2 Norm
 *  These functions compute the L2 norm of an image. The L2 norm is defined as the sum of all the square pixle values
 *  in the image. If the image contains multiple channles, the functions will compute the norm for each channel separately.
 *  The functions require the addition device scratch buffer for the computations.
 */
///@{


/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16s_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8s_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8s_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_C1MR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_C1MR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16s_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16s_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_C3R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16s_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16s_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_AC4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_AC4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16s_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16s_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_C4R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_8s_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_8s_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_16u_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_16u_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppiNormL2_32f_C3CMR.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiNormL2GetBufferHostSize_32f_C3CMR(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);


/**
 * 1-channel 8-bit unsigned char image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit unsigned short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16u_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 16-bit signed short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16s_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 1-channel 32-bit floating-point image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_32f_C1R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * \ref masked_operation 1-channel 8-bit unsigned char image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8u_C1MR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * \ref masked_operation 1-channel 8-bit signed char image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8s_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8s_C1MR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * \ref masked_operation 1-channel 16-bit unsigned short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16u_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16u_C1MR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * \ref masked_operation 1-channel 32-bit floating-point image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_32f_C1MR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_EVEN_STEP_ERROR if the step 
 * of the source image cannot be divided by 4. 
 */
NppStatus nppiNorm_L2_32f_C1MR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16u_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit signed short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16s_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_32f_C3R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image L2 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image L2 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16u_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image L2 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16s_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image L2 norm (alpha channel is not computed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of three channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_32f_AC4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[3], Npp8u * pDeviceBuffer);

/**
 * 4-channel 8-bit unsigned char image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_8u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit unsigned short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16u_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 16-bit signed short image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_16s_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 4-channel 32-bit floating-point image L2 norm.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aNorm Array that contains the norm values of four channels.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiNormL2GetBufferHostSize_32f_C4R to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiNorm_L2_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp64f aNorm[4], Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit unsigned char image L2 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiNormL2GetBufferHostSize_8u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L2_8u_C3CMR(const Npp8u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 8-bit signed char image L2 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiNormL2GetBufferHostSize_8s_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L2_8s_C3CMR(const Npp8s * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 16-bit unsigned short image L2 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppiNormL2GetBufferHostSize_16u_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L2_16u_C3CMR(const Npp16u * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

/**
 * 3-channel 32-bit floating-point image L2 norm, \ref masked_operation, \ref channel_of_interest.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nCOI \ref channel_of_interest_number.
 * \param pNorm Pointer to the norm value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 * Use \ref nppiNormL2GetBufferHostSize_32f_C3CMR to compute the required size (in bytes).
 * \return \ref image_data_error_codes, \ref roi_error_codes, NPP_NOT_EVEN_STEP_ERROR if the step 
 * of the source image cannot be divided by 4, or NPP_COI_ERROR if an invalid channel of interest is specified.
 */
NppStatus nppiNorm_L2_32f_C3CMR(const Npp32f * pSrc, int nSrcStep, const Npp8u * pMask, int nMaskStep, NppiSize oSizeROI, int nCOI, Npp64f * pNorm, Npp8u * pDeviceBuffer);

///@}

///@} image_L2_norm

/** @defgroup image_norm_diff Norm Diff
 */
///@{

/** @name NormDiff
 *  Norm of pixel differences between two images.
 */
///@{

/**
 * 8-bit unsigned L1 norm of pixel differences.
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrcStep1 \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrcStep2 \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pRetVal Contains computed L1-norm of differences. This is a host pointer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiNormDiff_L1_8u_C1R(const Npp8u * pSrc1, int nSrcStep1, 
                                 const Npp8u * pSrc2, int nSrcStep2, 
                                 NppiSize oSizeROI, Npp64f * pRetVal);

/**
 * 8-bit unsigned L2 norm of pixel differences.
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrcStep1 \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrcStep2 \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pRetVal Contains computed L1-norm of differences. This is a host pointer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiNormDiff_L2_8u_C1R(const Npp8u * pSrc1, int nSrcStep1, 
                                 const Npp8u * pSrc2, int nSrcStep2, 
                                 NppiSize oSizeROI, Npp64f * pRetVal);

/**
 * 8-bit unsigned Infinity Norm of pixel differences.
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrcStep1 \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrcStep2 \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param *pRetVal Contains computed L1-norm of differences. This is a host pointer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiNormDiff_Inf_8u_C1R(const Npp8u * pSrc1, int nSrcStep1, 
                                  const Npp8u * pSrc2, int nSrcStep2, 
                                  NppiSize oSizeROI, Npp64f * pRetVal);
                                  
///@}

///@} image_norm_diff

/** @defgroup image_integral Integral and Rectangular Standard Deviation
 */
///@{
                                  
/** @name Integral
 */
///@{

/**
 * SqrIntegral Transforms an image to integral and integral of pixel squares
 * representation. This function assumes that the integral and integral of squares 
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pSqr \ref destination_image_pointer.
 * \param nSqrStep \ref destination_image_line_step.
 * \param oSrcROI \ref roi_specification.
 * \param val The value to add to pDst image pixels
 * \param valSqr The value to add to pSqr image pixels
 * \param integralImageNewHeight Extended height of output surfaces (needed by
 *        transpose in primitive)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrIntegral_8u32s32f_C1R(Npp8u  * pSrc, int nSrcStep, 
                             Npp32s * pDst, int nDstStep, 
                             Npp32f * pSqr, int nSqrStep,
                             NppiSize oSrcROI, Npp32s val, Npp32f valSqr, Npp32s integralImageNewHeight);

/**
 * RectStdDev Computes the standard deviation of integral images
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSqr \ref destination_image_pointer.
 * \param nSqrStep \ref destination_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rect rectangular window
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRectStdDev_32s32f_C1R(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, 
                                Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect rect);

///@} Integral group

///@} image_integral

/** @defgroup image_histogram Histogram
 */
///@{

/** @name Histogram
 */
///@{

/**
 * Compute levels with even distribution.
 *
 * \param hpLevels A host pointer to array which receives the levels being
 *        computed. The array needs to be of size nLevels.
 * \param nLevels The number of levels being computed. nLevels must be at least
 *        2, otherwise an NPP_HISTO_NUMBER_OF_LEVELS_ERROR error is returned.
 * \param nLowerLevel Lower boundary value of the lowest level.
 * \param nUpperLevel Upper boundary value of the greatest level.
 * \return Error code.
*/
NppStatus
nppiEvenLevelsHost_32s(Npp32s * hpLevels, int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel);

/**
 * Scratch-buffer size for nppiHistogramEven_8u_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels ,int * hpBufferSize);

/**
 * 8-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param nLevels Number of levels.
 * \param nLowerLevel Lower boundary of lowest level bin.
 * \param nUpperLevel Upper boundary of highest level bin.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_8u_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                         int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_8u_C3R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_8u_C3R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 3 channel 8-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_8u_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                         int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_8u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_8u_C4R(NppiSize oSizeROI, int nLevels[4] ,int * hpBufferSize);

/**
 * 4 channel 8-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_8u_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, 
                               Npp32s * pHist[4], 
                         int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_8u_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_8u_AC4R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 4 channel (alpha as the last channel) 8-bit unsigned histogram with evenly distributed bins.
 * Alpha channel is ignored during histogram computation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_8u_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                Npp32s * pHist[3], 
                          int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16u_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16u_C1R(NppiSize oSizeROI, int nLevels ,int * hpBufferSize);

/**
 * 16-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param nLevels Number of levels.
 * \param nLowerLevel Lower boundary of lowest level bin.
 * \param nUpperLevel Upper boundary of highest level bin.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16u_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                          int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16u_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16u_C3R(NppiSize oSizeROI, int nLevels[3] , int * hpBufferSize);

/**
 * 3 channel 16-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16u_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                 Npp32s * pHist[3], 
                           int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16u_C4R(NppiSize oSizeROI, int nLevels[4] ,int * hpBufferSize);

/**
 * 4 channel 16-bit unsigned histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16u_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                Npp32s * pHist[4], 
                          int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16u_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16u_AC4R(NppiSize oSizeROI, int nLevels[3] , int * hpBufferSize);

/**
 * 4 channel (alpha as the last channel) 16-bit unsigned histogram with evenly distributed bins.
 * Alpha channel is ignored during histogram computation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16u_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                 Npp32s * pHist[3], 
                           int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16s_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16s_C1R(NppiSize oSizeROI, int nLevels ,int * hpBufferSize);

/**
 * 16-bit signed histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param nLevels Number of levels.
 * \param nLowerLevel Lower boundary of lowest level bin.
 * \param nUpperLevel Upper boundary of highest level bin.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16s_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                         int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16s_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16s_C3R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 3 channel 16-bit signed histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16s_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                 Npp32s * pHist[3], 
                           int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16s_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16s_C4R(NppiSize oSizeROI, int nLevels[4] ,int * hpBufferSize);

/**
 * 4 channel 16-bit signed histogram with evenly distributed bins.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16s_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                Npp32s * pHist[4], 
                          int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramEven_16s_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramEvenGetBufferSize_16s_AC4R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 4 channel (alpha as the last channel) 16-bit signed histogram with evenly distributed bins.
 * Alpha channel is ignored during histogram computation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving computed histograms per color channel. 
 *      Array pointed by pHist[i] be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param nLowerLevel Array containing lower-level of lowest bin per color channel.
 * \param nUpperLevel Array containing upper-level of highest bin per color channel.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramEvenGetBufferSize_16s_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramEven_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, 
                                 Npp32s * pHist[3], 
                           int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_8u_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_8u_C1R(NppiSize oSizeROI, int nLevels ,int * hpBufferSize);


/**
 * 8-bit unsigned histogram with bins determined by pLevels array.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_8u_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                          const Npp32s * pLevels, int nLevels, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_8u_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_8u_C3R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 3 channel 8-bit unsigned histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_8u_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                           const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_8u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_8u_C4R(NppiSize oSizeROI, int nLevels[4] ,int * hpBufferSize);

/**
 * 4 channel 8-bit unsigned histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_8u_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[4], 
                          const Npp32s * pLevels[4], int nLevels[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_8u_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_8u_AC4R(NppiSize oSizeROI, int nLevels[3] ,int * hpBufferSize);

/**
 * 4 channel (alpha as a last channel) 8-bit unsigned histogram with bins determined by pLevels.
 * Alpha channel is ignored during the histograms computations.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_8u_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                           const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16u_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16u_C1R(NppiSize oSizeROI, int nLevels ,int * hpBufferSize);

/**
 * 16-bit unsigned histogram with bins determined by pLevels array.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16u_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                           const Npp32s * pLevels, int nLevels, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16u_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16u_C3R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 3 channel 16-bit unsigned histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16u_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16u_C4R(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize);

/**
 * 4 channel 16-bit unsigned histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16u_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[4], 
                           const Npp32s * pLevels[4], int nLevels[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16u_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16u_AC4R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 4 channel (alpha as a last channel) 16-bit unsigned histogram with bins determined by pLevels.
 * Alpha channel is ignored during the histograms computations.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16u_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16s_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16s_C1R(NppiSize oSizeROI, int nLevels, int * hpBufferSize);

/**
 * 16-bit signed histogram with bins determined by pLevels array.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                           const Npp32s * pLevels, int nLevels, Npp8u * pBuffer);


/**
 * Scratch-buffer size for nppiHistogramRange_16s_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16s_C3R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 3 channel 16-bit signed histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16s_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16s_C4R(NppiSize oSizeROI, int nLevels[4] ,int * hpBufferSize);

/**
 * 4 channel 16-bit signed histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16s_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[4], 
                           const Npp32s * pLevels[4], int nLevels[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_16s_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_16s_AC4R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 4 channel (alpha as a last channel) 16-bit signed histogram with bins determined by pLevels.
 * Alpha channel is ignored during the histograms computations.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_16_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32s * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_32f_C1R.
 * 
 * \param oSizeROI \ref roi_specification.
 * \param nLevels Number of levels in the histogram.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_32f_C1R(NppiSize oSizeROI, int nLevels, int * hpBufferSize);

/**
 * 32-bit float histogram with bins determined by pLevels array.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram. 
 *      The array must be of size nLevels-1. 
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_32f_C1R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist, 
                           const Npp32f * pLevels, int nLevels, Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_32f_C3R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_32f_C3R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 3 channel 32-bit float histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_32f_C3R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32f * pLevels[3], int nLevels[3], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_32f_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_32f_C4R(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize);

/**
 * 4 channel 32-bit float histogram with bins determined by pLevels.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_32f_C4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[4], 
                           const Npp32f * pLevels[4], int nLevels[4], Npp8u * pBuffer);

/**
 * Scratch-buffer size for nppiHistogramRange_32f_AC4R.
 * 
 * \param oSizeROI ROI size.
 * \param nLevels Array containing number of levels per color channel.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiHistogramRangeGetBufferSize_32f_AC4R(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize);

/**
 * 4 channel (alpha as a last channel) 32-bit float histogram with bins determined by pLevels.
 * Alpha channel is ignored during the histograms computations.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Array of pointers which are receiving the computed histograms per color channel. 
 *      Array pointed by pHist[i] must be of size nLevels[i]-1.
 * \param nLevels Array containing number of levels per color channel. 
 * \param pLevels Array containing pointers to level-arrays per color channel.
        Array pointed by pLevel[i] must be of size nLevels[i].
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_32f_AC4R) 
 *      scratch buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramRange_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist[3], 
                            const Npp32f * pLevels[3], int nLevels[3], Npp8u * pBuffer);

///@} Histogram group

///@} image_histogram

///@} image_statistics_functions

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_STATISTICS_FUNCTIONS_H
