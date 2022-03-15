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
#ifndef NV_NPPI_MORPHOLIGICAL_OPERATIONS_H
#define NV_NPPI_MORPHOLOGICAL_OPERATIONS_H
 
/**
 * \file nppi_morphological_operations.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_morphological_operations Morphological Operations
 *  @ingroup nppi
 *
 * Morphological image operations.
 *
 *
 */
///@{

/** @defgroup image_dilation_and_erosion Dilation And Erosion
 */
///@{

/**
 * 8-bit unsigned image dilation.
 * 
 * Dilation computes the output pixel as the maximum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero to not 
 * participate in the maximum search.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDilate_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 4 channel 8-bit unsigned image dilation.
 * 
 * Dilation computes the output pixel as the maximum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero to not 
 * participate in the maximum search.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiDilate_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);


/**
 * 8-bit unsigned image erosion.
 * 
 * Erosion computes the output pixel as the minimum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero to not 
 * participate in the maximum search.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiErode_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * 4 channel 8-bit unsigned image erosion.
 * 
 * Erosion computes the output pixel as the minimum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero to not 
 * participate in the maximum search.
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiErode_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                 const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

///@} image_dilation_and_erosion

///@} image_morphological_operations

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_MORPHOLOGICAL_OPERATIONS_H
