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
#ifndef NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
#define NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
 
/**
 * \file nppi_data_exchange_and_initialization.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_data_exchange_and_initialization Data Exchange and Initialization
 *  @ingroup nppi
 *
 * Primitives for initializtion, copying and converting image data.
 *
 */
///@{

/** 
 * @defgroup image_set Set
 */
///@{

/** @name Image-Memory Set
 * Set methods for images of various types. Images are passed to these primitives via a pointer
 * to the image data (first pixel in the ROI) and a step-width, i.e. the number of bytes between
 * successive lines. The size of the area to be set (region-of-interest, ROI) is specified via
 * a NppiSize struct. 
 * In addition to the image data and ROI, all methods have a parameter to specify the value being
 * set. In case of single channel images this is a single value, in case of multi-channel, an 
 * array of values is passed. 
 */
///@{

/** 
 * 8-bit image set.
 * \param nValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8s_C1R(Npp8s nValue, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit two-channel image set.
 * \param aValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8s_C2R(Npp8s aValue[2], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit three-channel image set.
 * \param aValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8s_C3R(Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit four-channel image set.
 * \param aValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8s_C4R(Npp8s aValue[4], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit four-channel image set ignoring alpha channel.
 * \param aValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8s_AC4R(Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit unsigned image set.
 * \param nValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_C1R( Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 8-bit unsigned image set. 
 * The 8-bit mask image affects setting of the respective pixels in the destination image.
 * If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
 * destination pixel is set to specified value.
 * \param nValue The pixel value to be set.
 * \param pDst Pointer \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_C1MR(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 8-bit unsigned image set.
 * \param aValues Four-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_C4R(const Npp8u aValues[4], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 8-bit unsigned image set.
 * \param aValues Four-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_C4MR(const Npp8u aValues[4], Npp8u* pDst, int nDstStep, NppiSize oSizeROI,
                const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_AC4R(const Npp8u aValues[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_AC4MR(const Npp8u aValues[3], Npp8u * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 8-bit unsigned image set affecting only single channel.
 * For RGBA images, this method allows setting of a single of the four (RGBA) values 
 * without changing the contents of the other three channels. The channel is selected
 * via the pDst pointer. The pointer needs to point to the actual first value to be set,
 * e.g. in order to set the R-channel (first channel), one would pass pDst unmodified, since
 * its value actually points to the r channel. If one wanted to modify the B channel (second
 * channel), one would pass pDst + 2 to the function.
 * \param nValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_8u_C4CR(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit unsigned image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C1R(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 16-bit unsigned image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C1MR( Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * 2 channel 16-bit unsigned image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C2R(const Npp16u aValues[2], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C4R(const Npp16u aValues[4], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 16-bit unsigned image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C4MR(const Npp16u aValues[4], Npp16u * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_AC4R(const Npp16u aValues[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_AC4MR(const Npp16u aValues[3], Npp16u * pDst, int nDstStep, 
                  NppiSize oSizeROI,
                  const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 16-bit unsigned image set affecting only single channel.
 * For RGBA images, this method allows setting of a single of the four (RGBA) values 
 * without changing the contents of the other three channels. The channel is selected
 * via the pDst pointer. The pointer needs to point to the actual first value to be set,
 * e.g. in order to set the R-channel (first channel), one would pass pDst unmodified, since
 * its value actually points to the r channel. If one wanted to modify the B channel (second
 * channel), one would pass pDst + 2 to the function.
 * \param nValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16u_C4CR(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C1R(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 16-bit image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C1MR(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * 2 channel 16-bit image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C2R(const Npp16s aValues[2], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C4R(const Npp16s aValues[4], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 16-bit image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C4MR(const Npp16s aValues[4], Npp16s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * 4 channel 16-bit image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_AC4R(const Npp16s aValues[3], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_AC4MR(const Npp16s aValues[3], Npp16s * pDst, int nDstStep, 
                  NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 16-bit unsigned image set affecting only single channel.
 * For RGBA images, this method allows setting of a single of the four (RGBA) values 
 * without changing the contents of the other three channels. The channel is selected
 * via the pDst pointer. The pointer needs to point to the actual first value to be set,
 * e.g. in order to set the R-channel (first channel), one would pass pDst unmodified, since
 * its value actually points to the r channel. If one wanted to modify the B channel (second
 * channel), one would pass pDst + 2 to the function.
 * \param nValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16s_C4CR(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer image set.
 * \param oValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16sc_C1R(Npp16sc oValue, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer two-channel image set.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16sc_C2R(Npp16sc aValue[2], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer three-channel image set.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16sc_C3R(Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set ignoring alpha.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16sc_AC4R(Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_16sc_C4R(Npp16sc aValue[4], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_C1R(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 32-bit image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_C1MR(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 32-bit image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_C4R( const Npp32s aValues[4], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 32-bit image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_C4MR(const Npp32s aValues[4], Npp32s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * 4 channel 16-bit image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_AC4R(const Npp32s aValues[3], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_AC4MR(const Npp32s aValues[3], Npp32s * pDst, int nDstStep, 
                  NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 32-bit unsigned image set affecting only single channel.
 * For RGBA images, this method allows setting of a single of the four (RGBA) values 
 * without changing the contents of the other three channels. The channel is selected
 * via the pDst pointer. The pointer needs to point to the actual first value to be set,
 * e.g. in order to set the R-channel (first channel), one would pass pDst unmodified, since
 * its value actually points to the r channel. If one wanted to modify the B channel (second
 * channel), one would pass pDst + 2 to the function.
 * \param nValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32s_C4CR(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit complex integer image set.
 * \param oValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32sc_C1R(Npp32sc oValue, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two channel 32-bit complex integer image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32sc_C2R(Npp32sc aValue[2], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit complex integer image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32sc_C3R(Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit complex integer image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32sc_C4R(Npp32sc aValue[4], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex integer four-channel image set ignoring alpha.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32sc_AC4R(Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * 32-bit floating point image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_C1R(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 32-bit floating point image set.
 * \param nValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_C1MR(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 32-bit floating point image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_C4R(const Npp32f aValues[4], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 32-bit floating point image set.
 * \param aValues New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_C4MR(const Npp32f aValues[4], Npp32f * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_AC4R(const Npp32f aValues[3], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * For RGBA images, this method allows setting of the RGB values without changing the contents
 * of the alpha-channel (fourth channel).
 * \param aValues Three-channel array containing the pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the mask image. This is a single channel 8-bit unsigned int image.
 * \param nMaskStep Number of bytes between line starts of successive lines in the mask image.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_AC4MR(const Npp32f aValues[3], Npp32f * pDst, int nDstStep, 
                  NppiSize oSizeROI,
                  const Npp8u * pMask, int nMaskStep);

/** 
 * 4 channel 32-bit floating point image set affecting only single channel.
 * For RGBA images, this method allows setting of a single of the four (RGBA) values 
 * without changing the contents of the other three channels. The channel is selected
 * via the pDst pointer. The pointer needs to point to the actual first value to be set,
 * e.g. in order to set the R-channel (first channel), one would pass pDst unmodified, since
 * its value actually points to the r channel. If one wanted to modify the B channel (second
 * channel), one would pass pDst + 2 to the function.
 * \param nValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32f_C4CR(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * Single channel 32-bit complex image set.
 * \param oValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32fc_C1R(Npp32fc oValue, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two channel 32-bit complex image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32fc_C2R(Npp32fc aValue[2], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit complex image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32fc_C3R(Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit complex image set.
 * \param aValue The pixel-value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32fc_C4R(Npp32fc aValue[4], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex four-channel image set ignoring alpha.
 * \param aValue New pixel value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSet_32fc_AC4R(Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

///@}

///@} image_set

/** 
 * @defgroup image_copy Copy
 */
///@{

/** @name Image-Memory Copy
 * Copy methods for images of various types. In addition to routines for
 * copying pixels of identical layout from one image to another, there are
 * copy routines for select channels as well as packed-planar conversions:
 * - Select channel to multi-channel copy. E.g. given a three-channel source
 *      and destination image one may copy the second channel of the source
 *      to the third channel of the destination.
 * - Single channel to multi-channel copy. E.g. given a single-channel source
 *      and a four-channel destination, one may copy the contents of the single-
 *      channel source to the second channel of the destination.
 * - Select channel to single-channel copy. E.g. given a three-channel source
 *      and a single-channel destination one may copy the third channel of the 
 *      source to the destination.
 * - Multi-channel to planar copy. These copy operations split a multi-channel
 *      image into a set of single-channel images.
 * - Planar image to multi-channel copy. These copy routines combine separate
 *      color-planes (single channel images) into a single multi-channel image.
 * 
 */
///@{

/** 
 * 8-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8s_C1R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 8-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8s_C2R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 8-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8s_C3R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8s_C4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy, ignoring alpha channel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy, not affecting Alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16sc_C1R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 16-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16sc_C2R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16sc_C3R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16sc_C4R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16sc_AC4R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * 32-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy, not affecting Alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32sc_C1R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 32-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32sc_C2R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32sc_C3R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32sc_C4R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32sc_AC4R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * 32-bit floating point image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy, not affecting Alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * 32-bit floating-point complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32fc_C1R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 32-bit floating-point complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32fc_C2R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit floating-point complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32fc_C3R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32fc_C4R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32fc_AC4R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * \ref masked_operation 8-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_C1MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 8-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_C3MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_C4MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_8u_AC4MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_C1MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_C3MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_C4MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16u_AC4MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_C1MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_C3MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_C4MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_16s_AC4MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_C1MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_C3MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_C4MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32s_AC4MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit float image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_C1MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit float image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_C3MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_C4MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy, ignoring alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiCopy_32f_AC4MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);


/** 
 * Select-channel 8-bit unsigned image copy for three-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C3CR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 8-bit unsigned image copy for four-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C4CR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 16-bit signed image copy for three-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C3CR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 16-bit signed image copy for four-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C4CR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 16-bit unsigned image copy for three-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C3CR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 16-bit unsigned image copy for four-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C4CR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 32-bit signed image copy for three-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C3CR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 32-bit signed image copy for four-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C4CR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 32-bit float image copy for three-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C3CR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Select-channel 32-bit float image copy for four-channel images.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C4CR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 8-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C3C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 8-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C4C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C3C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C4C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C3C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C4C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C3C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C4C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit float image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit float image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C4C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * Single-channel to three-channel 8-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C1C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 8-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C1C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C1C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C1C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C1C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit unsigned image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C1C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C1C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit signed image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C1C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit float image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C1C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit float image copy.
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C1C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * Three-channel 8-bit unsigned packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_C4P4R(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C3P3R(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_C4P4R(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C3P3R(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_C4P4R(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C3P3R(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_C4P4R(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit float packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C3P3R(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit float packed to planar image copy.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst Planar \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_C4P4R(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 8-bit unsigned planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_P3C3R(const Npp8u * const aSrc[3], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_8u_P4C4R(const Npp8u * const aSrc[4], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_P3C3R(const Npp16u * const aSrc[3], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16u_P4C4R(const Npp16u * const aSrc[4], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_P3C3R(const Npp16s * const aSrc[3], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_16s_P4C4R(const Npp16s * const aSrc[4], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_P3C3R(const Npp32s * const aSrc[3], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32s_P4C4R(const Npp32s * const aSrc[4], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit float planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_P3C3R(const Npp32f * const aSrc[3], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit float planar to packed image copy.
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiCopy_32f_P4C4R(const Npp32f * const aSrc[4], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

///@}

///@} image_copy

/** 
 * @defgroup image_convert Convert
 */
///@{

/** 
 * @name Bit-Depth Conversion
 * Convert bit-depth up and down.
 *
 * The integer conversion methods do not involve any scaling. Conversions that reduce bit-depth saturate
 * values exceeding the reduced range to the range's maximum/minimum value.
 * When converting from floating-point values to integer values, a rounding mode can be specified. After rounding
 * to integer values the values get saturated to the destination data type's range.
 */
///@{


/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16u_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16u_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16u_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16u_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u16s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32f_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32f_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32f_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32f_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32s_C1R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32s_C3R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32s_C4R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32f_C1R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32f_C3R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32f_C4R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32f_AC4R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32s_C1R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32s_C3R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32s_C4R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32s_AC4R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32f_C1R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32f_C3R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32f_C4R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32f_AC4R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32s_C1R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32s_C3R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32s_C4R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32f_C3R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32f_C4R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32f_AC4R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s8u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s16u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s16s_C1R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8s32u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s16u_C1Rs(const Npp16s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32u_C1Rs(const Npp16s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32u_C1R(const Npp16u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s32u_C1Rs(const Npp32s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit signed to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s32f_C1R(const Npp32s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit unsigned to 32-bit floating-point conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32u32f_C1R(const Npp32u * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);



/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u8u_C1R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u8u_C3R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u8u_C4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u8u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          

/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s8u_C1R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s8u_C3R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s8u_C4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiConvert_16s8u_AC4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8u_C1R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8u_C3R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8u_C4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiConvert_32s8u_AC4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
      
/** 
 * Single channel 32-bit signed to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8s_C1R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8s_C3R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32s8s_C4R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiConvert_32s8s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);
          

NppStatus 
nppiConvert_8u8s_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_16u8s_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_16s8s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_16u16s_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32u8u_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32u8s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32u16u_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32u16s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32u32s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32s16u_C1RSfs(const Npp32s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppiConvert_32s16s_C1RSfs(const Npp32s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8u_C1R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8u_C3R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8u_C4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8u_AC4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8s_C1R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8s_C3R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8s_C4R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8s_AC4R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16u_C1R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16u_C3R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16u_C4R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16u_AC4R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16s_C1R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16s_C3R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16s_C4R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16s_AC4R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);


/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f8s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f16s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit unsigned conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f32u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit signed conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_32f32s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

///@}

///@} image_convert

/** 
 * @defgroup image_copy_constant_border Copy Constant Border
 */
///@{

/** @name Copy Const Border
 * Methods for copying images and padding borders with a constant, user-specifiable color.
 */
///@{

/** 
 * 8-bit unsigned image copy width constant border color.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and constant border color (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The height of the border at the bottom of the
 *      destination ROI is implicitly defined by the size of the source ROI: 
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param nValue The pixel value to be set for border pixels.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     Npp8u nValue);

/**
 * 4channel 8-bit unsigned image copy with constant border color.
 * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region-of-interest.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size of the destination region-of-interest.
 * \param nTopBorderHeight Height of top border.
 * \param nLeftBorderWidth Width of left border.
 * \param aValue Vector of the RGBA values of the border pixels to be set.
  * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus nppiCopyConstBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Npp8u aValue[4]);
                                       
/**
 * 4 channel 8-bit unsigned image copy with constant border color.
 * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region-of-interest.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size of the destination region-of-interest.
 * \param nTopBorderHeight Height of top border.
 * \param nLeftBorderWidth Width of left border.
 * \param aValue Vector of the RGB values of the border pixels. Because this method does not
 *      affect the destination image's alpha channel, only three components of the border color
 *      are needed.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCopyConstBorder_8u_AC4R(const Npp8u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp8u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp8u aValue[3]);

/** 32-bit image copy with constant border color.
 * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region-of-interest.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size of the destination region-of-interest.
 * \param nTopBorderHeight Height of top border.
 * \param nLeftBorderWidth Width of left border.
 * \param nValue Border luminance value.
 * \return \ref image_data_error_codes, \ref roi_error_codes
*/
NppStatus nppiCopyConstBorder_32s_C1R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp32s nValue);

///@}

///@} image_copy_constant_border

/** 
 * @defgroup image_transpose_and_swap_channels Transpose And Swap Channels
 */
///@{

/** @name Image Transpose
 * Methods for transposing images of various types. Like matrix transpose, image transpose is a mirror along the image's
 * diagonal (upper-left to lower-right corner).
 */
///@{

/**
 * 8-bit image transpose.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst Pointer to the destination ROI.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 *
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiTranspose_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI);
///@}


/** @name Image Color Channel Swap
 * Methods for exchanging the color channels of an image. The methods support arbitrary permutations of the original
 * channels, including replication.
 */
///@{

/**
 * 4 channel 8-bit unsigned swap channels, in-place.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

///@}

///@} image_transpose_and_swap_channels

///@} image_data_exchange_and_initialization

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
