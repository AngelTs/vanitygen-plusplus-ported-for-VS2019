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
#ifndef NV_NPPI_FILTERING_FUNCTIONS_H
#define NV_NPPI_FILTERING_FUNCTIONS_H
 
/**
 * \file nppi_filtering_functions.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_filtering_functions Filtering Functions
 *  @ingroup nppi
 *
 * Linear and non-linear image filtering functions.
 *
 *
 */
///@{

/** @defgroup image_1D_linear_filter 1D Linear Filter
 */
///@{

/** @name 1D Linear Filter
 *  1D mask Linear Convolution Filter, with rescaling, for 8 bit images.
 */
///@{

/**
 * 8-bit unsigned 1D (column) image convolution.
 * 
 * Apply convolution filter with user specified 1D column of weights.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring column pixel
 * values in the source image defined by nKernelDim and nAnchorY, divided by
 * nDivisor. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference w.r.t the
 *                 source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 4 channel 8-bit unsigned 1D (column) image convolution.
 * 
 * Apply convolution filter with user specified 1D column of weights.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring column pixel
 * values in the source image defined by nKernelDim and nAnchorY, divided by
 * nDivisor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference w.r.t the
 *                 source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned 1D (row) image convolution.
 *
 * Apply general linear Row convolution filter, with rescaling, in a 1D mask
 * region around each source pixel for 1-channel 8 bit/pixel images.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring row pixel values
 * in the source image defined by iKernelDim and iAnchorX, divided by iDivisor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.  
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference w.r.t the
 *                 source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 4 channel 8-bit unsigned 1D (row) image convolution.
 *
 * Apply general linear Row convolution filter, with rescaling, in a 1D mask
 * region around each source pixel for 1-channel 8 bit/pixel images.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring row pixel values
 * in the source image defined by iKernelDim and iAnchorX, divided by iDivisor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.  
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference w.r.t the
 *                 source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRow_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

///@}

///@} image_1D_linear_filter

/** @defgroup image_1D_window_sum 1D Window Sum
 */
///@{

/** @name 1D Window Sum
 *  1D mask Window Sum for 8 bit images.
 */
///@{

/**
 * 8-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference w.r.t the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_8u32f_C1R(const Npp8u * pSrc, Npp32s nSrcStep, 
                                              Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                        Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference w.r.t the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_8u32f_C1R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                 Npp32f * pDst, Npp32s nDstStep, 
                           NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);
///@}

///@} image_1D_window_sum

/** @defgroup image_convolution Convolution
 */
///@{

/** @name Convolution (2D Masks)
 * General purpose 2D convolution filters.
 */
///@{

/**
 * 8-bit unsigned convolution filter.
 * 
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
                   
/**
 * 4 channel 8-bit unsigned convolution filter.
 * 
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \param nDivisor The factor by which the convolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

///@}

///@} image_convolution

/** @defgroup image_2D_fixed_linear_filters 2D Fixed Linear Filters
 */
///@{

/** @name 2D Linear Fixed Filters
 *  2D linear fixed filters for 8 bit images.
 */
///@{

/**
 * 8-bit unsigned box filter.
 *
 * Computes the average pixel values of the pixels under a rectangular mask.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference w.r.t
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 4 channel 8-bit unsigned box filter.
 *
 * Computes the average pixel values of the pixels under a rectangular mask.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference w.r.t
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

///@}

///@} image_2D_fixed_linear_filters

/** @defgroup image_rank_filters Rank Filters
 */
///@{

/** @name Image Rank Filters
 *  Min, Median, and Max image filters.
 */
///@{

/**
 * 8-bit unsigned maximum filter.
 *
 * Result pixel value is the maximum of pixel values under the rectangular
 * mask region.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 4 channel 8-bit unsigned maximum filter.
 *
 * Result pixel value is the maximum of pixel values under the rectangular
 * mask region.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 8-bit unsigned minimum filter.
 *
 * Result pixel value is the minimum of pixel values under the rectangular
 * mask region.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 4 channel 8-bit unsigned minimum filter.
 *
 * Result pixel value is the minimum of pixel values under the rectangular
 * mask region.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

///@}

///@} image_rank_filters

///@} image_filtering_functions

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_FILTERING_FUNCTIONS_H
