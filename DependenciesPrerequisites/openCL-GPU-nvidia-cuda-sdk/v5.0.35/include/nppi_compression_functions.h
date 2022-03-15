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
#ifndef NV_NPPI_COMPRESSION_FUNCTIONS_H
#define NV_NPPI_COMPRESSION_FUNCTIONS_H
 
/**
 * \file nppi_compression_functions.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_compression Compression
 *  @ingroup nppi
 *
 * Image compression primitives.
 *
 * The JPEG standard defines a flow of level shift, DCT and quantization for
 * forward JPEG transform and inverse level shift, IDCT and de-quantization
 * for inverse JPEG transform. This group has the functions for both forward
 * and inverse functions. 
 */
///@{

/** @defgroup image_quantization Quantization Functions
 */
///@{

/**
 * Apply quality factor to raw 8-bit quantization table.
 *
 * This is effectively and in-place method that modifies a given raw
 * quantization table based on a quality factor.
 * Note that this method is a host method and that the pointer to the
 * raw quantization table is a host pointer.
 *
 * \param hpQuantRawTable Raw quantization table.
 * \param nQualityFactor Quality factor for the table. Range is [1:100].
 * \return Error code:
 *      #NPP_NULL_POINTER_ERROR is returned if hpQuantRawTable is 0.
 */
NppStatus 
nppiQuantFwdRawTableInit_JPEG_8u(Npp8u * hpQuantRawTable, int nQualityFactor);

/**
 * Initializes a quantization table for nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
 *    The method creates a 16-bit version of the raw table and converts the 
 * data order from zigzag layout to original row-order layout since raw
 * quantization tables are typically stored in zigzag format.
 *
 * This method is a host method. It consumes and produces host data. I.e. the pointers
 * passed to this function must be host pointers. The resulting table needs to be
 * transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R()
 * function.
 *
 * \param hpQuantRawTable Host pointer to raw quantization table as returned by 
 *      nppiQuantFwdRawTableInit_JPEG_8u(). The raw quantization table is assumed to be in
 *      zigzag order.
 * \param hpQuantFwdRawTable Forward quantization table for use with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
 * \return Error code:
 *      #NPP_NULL_POINTER_ERROR pQuantRawTable is 0.
 */
NppStatus 
nppiQuantFwdTableInit_JPEG_8u16u(const Npp8u * hpQuantRawTable, Npp16u * hpQuantFwdRawTable);

/**
 * Initializes a quantization table for nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R().
 *      The nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R() method uses a quantization table
 * in a 16-bit format allowing for faster processing. In addition it converts the 
 * data order from zigzag layout to original row-order layout. Typically raw
 * quantization tables are stored in zigzag format.
 *
 * This method is a host method and consumes and produces host data. I.e. the pointers
 * passed to this function must be host pointers. The resulting table needs to be
 * transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R()
 * function.
 *
 * \param hpQuantRawTable Raw quantization table.
 * \param hpQuantFwdRawTable Inverse quantization table.
 * \return #NPP_NULL_POINTER_ERROR pQuantRawTable or pQuantFwdRawTable is0.
 */
NppStatus 
nppiQuantInvTableInit_JPEG_8u16u(const Npp8u * hpQuantRawTable, Npp16u * hpQuantFwdRawTable);


/**
 * Forward DCT, quantization and level shift part of the JPEG encoding.
 * Input is expected in 8x8 macro blocks and output is expected to be in 64x1
 * macro blocks.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pQuantFwdTable Forward quantization tables for JPEG encoding created
 *          using nppiQuantInvTableInit_JPEG_8u16u().
 * \param oSizeROI \ref roi_specification.
 * \return Error codes:
 *         - #NPP_SIZE_ERROR For negative input height/width or not a multiple of
 *           8 width/height.
 *         - #NPP_STEP_ERROR If input image width is not multiple of 8 or does not
 *           match ROI.
 *         - #NPP_NULL_POINTER_ERROR If the destination pointer is 0.
 */
NppStatus 
nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(const Npp8u  * pSrc, int nSrcStep, 
                                          Npp16s * pDst, int nDstStep, 
                                    const Npp16u * pQuantFwdTable, NppiSize oSizeROI);

/**
 * Inverse DCT, de-quantization and level shift part of the JPEG decoding.
 * Input is expected in 64x1 macro blocks and output is expected to be in 8x8
 * macro blocks.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pQuantInvTable Inverse quantization tables for JPEG decoding created
 *           using nppiQuantInvTableInit_JPEG_8u16u().
 * \param oSizeROI \ref roi_specification.
 * \return Error codes:
 *         - #NPP_SIZE_ERROR For negative input height/width or not a multiple of
 *           8 width/height.
 *         - #NPP_STEP_ERROR If input image width is not multiple of 8 or does not
 *           match ROI.
 *         - #NPP_NULL_POINTER_ERROR If the destination pointer is 0.
 */
NppStatus 
nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(const Npp16s * pSrc, int nSrcStep, 
                                          Npp8u  * pDst, int nDstStep, 
                                    const Npp16u * pQuantInvTable, NppiSize oSizeROI);
                                    
///@} image_quantization

///@} image_compression


#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_COMPRESSION_FUNCTIONS_H
