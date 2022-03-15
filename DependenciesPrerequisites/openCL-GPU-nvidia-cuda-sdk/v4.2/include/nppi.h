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
#ifndef NV_NPPI_H
#define NV_NPPI_H
 
/**
 * \file nppi.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup nppi NPP Image Processing
 * @{
 */


typedef enum {
    NPPI_OP_ALPHA_OVER,
    NPPI_OP_ALPHA_IN,
    NPPI_OP_ALPHA_OUT,
    NPPI_OP_ALPHA_ATOP,
    NPPI_OP_ALPHA_XOR,
    NPPI_OP_ALPHA_PLUS,
    NPPI_OP_ALPHA_OVER_PREMUL,
    NPPI_OP_ALPHA_IN_PREMUL,
    NPPI_OP_ALPHA_OUT_PREMUL,
    NPPI_OP_ALPHA_ATOP_PREMUL,
    NPPI_OP_ALPHA_XOR_PREMUL,
    NPPI_OP_ALPHA_PLUS_PREMUL,
    NPPI_OP_ALPHA_PREMUL
} NppiAlphaOp;


/** @defgroup image_memory_management Memory Management
 *
 * Routines for allocating and deallocating pitched image storage. These methods
 * are provided for convenience. They allocate memory that may contain additional
 * padding bytes at the end of each line of pixels. Though padding is not necessary
 * for any of the NPP image-processing primitives to work correctly, its absense may
 * cause sever performance degradation compared to properly padded images.
 *
 *
 */
///@{

/** @name Image-Memory Allocation
 * ImageAllocator methods for 2D arrays of data. The allocators have width and height parameters
 * to specify the size of the image data being allocated. They return a pointer to the
 * newly created memory and return the numbers of bytes between successive lines. 
 *
 * If the memory allocation failed due to lack of free device memory or device memory fragmentation
 * the routine returns 0.
 *
 * All allocators return memory with line strides that are 
 * beneficial for performance. It is not mandatory to use these allocators. Any valid CUDA device-memory
 * pointers can be used by the NPP primitives and there are no restrictions on line strides.
 */
///@{

/**
 * 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp8u  * 
nppiMalloc_8u_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp8u  * 
nppiMalloc_8u_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp8u  * 
nppiMalloc_8u_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 8-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp8u  * 
nppiMalloc_8u_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16u * 
nppiMalloc_16u_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16u * 
nppiMalloc_16u_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16u * 
nppiMalloc_16u_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 16-bit unsigned image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16u * 
nppiMalloc_16u_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16s * 
nppiMalloc_16s_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16s * 
nppiMalloc_16s_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 16-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16s * 
nppiMalloc_16s_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 1 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16sc * 
nppiMalloc_16sc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16sc * 
nppiMalloc_16sc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16sc * 
nppiMalloc_16sc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 16-bit signed complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp16sc * 
nppiMalloc_16sc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32s * 
nppiMalloc_32s_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32s * 
nppiMalloc_32s_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 32-bit signed image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32s * 
nppiMalloc_32s_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32sc * 
nppiMalloc_32sc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32sc * 
nppiMalloc_32sc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32sc * 
nppiMalloc_32sc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 32-bit integer complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32sc * 
nppiMalloc_32sc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);


/**
 * 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32f * 
nppiMalloc_32f_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32f * 
nppiMalloc_32f_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32f * 
nppiMalloc_32f_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 32-bit floating point image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32f * 
nppiMalloc_32f_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32fc * 
nppiMalloc_32fc_C1(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 2 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32fc * 
nppiMalloc_32fc_C2(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 3 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32fc * 
nppiMalloc_32fc_C3(int nWidthPixels, int nHeightPixels, int * pStepBytes);

/**
 * 4 channel 32-bit float complex image memory allocator.
 * \param nWidthPixels Image width.
 * \param nHeightPixels Image height.
 * \param pStepBytes \ref line_step.
 * \return Pointer to new image data.
 */
Npp32fc * 
nppiMalloc_32fc_C4(int nWidthPixels, int nHeightPixels, int * pStepBytes);

///@} Malloc


/**
 * Free method for any 2D allocated memory.
 * This method should be used to free memory allocated with any of the nppiMalloc_<modifier> methods.
 * \param pData A pointer to memory allocated using nppiMalloc_<modifier>.
 */
void 
nppiFree(void * pData);

///@} Image-Memory Management


/** 
 * @defgroup image_data_exchange_and_initialization Data-Exchange and Initialization
 *
 * Primitives for initializtion, copying and converting image data.
 *
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
 * 8-bit unsigned to 16-bit unsigned conversion.
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
 * 16-bit unsigned to 8-bit unsigned conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 8-bit unsigned to 16-bit unsigned  conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 16-bit unsigned to 8-bit unsigned conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 8-bit unsigned to 16-bit signed conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 16-bit signed to 8-bit unsigned conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 8-bit unsigned to 16-bit signed conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 16-bit signed to 8-bit unsignedconversion, not affecting Alpha.
 * For detailed documentation see nppiConvert_8u16u_C1R().
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
 * 4 channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 4 channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 16-bit singedto 32-bit floating point conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 32-bit floating point to 16-bit conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 8-bit unsigned to 32-bit floating point conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit unsigned to 32-bit floating point conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit floating point to 16-bit unsigned conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 32-bit floating point to 8-bit unsigned conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
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
 * 16-bit unsigned to 32-bit signed conversion.
 * For detailed documentation see nppiConverte_8u16u_C1R().
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16u32s_C1R(const Npp16u * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit to 32-bit  conversion.
 * For detailed documentation see nppiConvert_8u16u_C1R().
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiConvert_16s32s_C1R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

///@}




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

///@} image_data_exchange_and_initialization


/** @defgroup image_arithmetic Arithmetic and Logical Operations
 */
///@{

/** @name AddC
 * Adds a constant value to each pixel of an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel..
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C3IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel..
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_AC4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C3IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_AC4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C3IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_AC4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C3IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_AC4IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C3IRSfs(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C3IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_AC4IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image add constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C3IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_AC4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C3IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_AC4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of AddC
///@}

/** @name MulC
 * Multiplies each pixel of an image by a constant value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C3IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_AC4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C3IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_AC4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C3IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_AC4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C3IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_AC4IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C3IRSfs(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C3IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_AC4IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image multiply by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C3IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_AC4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C3IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_AC4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of MulC
///@}

/** @name MulCScale
 * Multiplies each pixel of an image by a constant value then scales the result by the maximum value for the data bit width.
 */
///@{

/** 
 * One 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C3IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_AC4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C3IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_AC4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of MulCScale
///@}

/** @name SubC
 * Subtracts a constant value from each pixel of an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C3IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_AC4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C3IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_AC4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C3IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_AC4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C3IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_AC4IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C3IRSfs(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C3IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_AC4IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image subtract constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C3IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_AC4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C3IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_AC4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of SubC
///@}

/** @name DivC
 * Divides each pixel of an image by a constant value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C3IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_AC4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C4IRSfs(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C3IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_AC4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C4IRSfs(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C3IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_AC4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C4IRSfs(const Npp16s * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C3IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pConstants, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_AC4IRSfs(const Npp16sc * pConstants, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C3IRSfs(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C3IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pConstants, 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_AC4IRSfs(const Npp32sc * pConstants, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image divided by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C3IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_AC4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pConstants, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C4IR(const Npp32f * pConstants, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C3IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_AC4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pConstants, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C4IR(const Npp32fc * pConstants, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of DivC
///@}

/** @name AbsDiffC
 * Determines absolute difference between each pixel of an image and a constant value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp8u nConstant);

/** 
 * One 16-bit unsigned short channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp32u nConstant);

/** 
 * One 32-bit floating point channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, Npp32f nConstant);

// end of AbsDiffC
///@}

/** @name Add Image
 * Pixel by pixel addition of two images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image add.
 * Add the pixel values of corresponding pixels in the ROI and write them to the output image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiAdd_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);                        

/** 
 * One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Add Image
///@}

/** @name Add Square Image
 * Pixel by pixel addition of squared pixels from source image to floating point pixel values of destination image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_8u32f_C1IMR(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_8u32f_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                               Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_16u32f_C1IMR(const Npp16u * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                 Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_16u32f_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                                Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_32f_C1IMR(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                              Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                             Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Add Square Image
///@}

/** @name Add Product Image
 * Pixel by pixel addition of product of pixels from two source images to floating point pixel values of destination image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_8u32f_C1IMR(const Npp8u * pSrc1,  int nSrc1Step,    const Npp8u * pSrc2, int nSrc2Step,
                           const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,    int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_8u32f_C1IR(const Npp8u * pSrc1,    int nSrc1Step,   const Npp8u * pSrc2, int nSrc2Step,
                                Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_16u32f_C1IMR(const Npp16u * pSrc1, int nSrc1Step,    const Npp16u * pSrc2, int nSrc2Step,
						    const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_16u32f_C1IR(const Npp16u * pSrc1,    int nSrc1Step,   const Npp16u * pSrc2, int nSrc2Step,
                                 Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_32f_C1IMR(const Npp32f * pSrc1, int nSrc1Step,    const Npp32f * pSrc2, int nSrc2Step,
						 const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_32f_C1IR(const Npp32f * pSrc1,    int nSrc1Step,   const Npp32f * pSrc2, int nSrc2Step,
                              Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Add Product Image
///@}

/** @name Add Weighted Image
 * Pixel by pixel addition of alpha weighted pixel values from a source image to floating point pixel values of destination image.
 */
///@{

/** 
 * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_8u32f_C1IMR(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                  Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha);

/** 
 * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_8u32f_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                                 Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_16u32f_C1IMR(const Npp16u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                   Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha);

/** 
 * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_16u32f_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                                  Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 32-bit floating point channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_32f_C1IMR(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 32-bit floating point channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                               Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

// end of Add Weighted Image
///@}

/** @name Mul Image
 * Pixel by pixel multiply of two images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead.
 * 1 channel 32-bit image multiplication.
 * Multiply corresponding pixels in ROI. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMul_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI); 

/** 
 * One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Mul Image
///@}

/** @name MulScale Image
 * Pixel by pixel multiplies each pixel of two images then scales the result by the maximum value for the data bit width.
 */
///@{

/** 
 * One 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                            Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                             Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of MulScale Image
///@}

/** @name Sub Image
 * Pixel by pixel subtraction of two images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image subtraction.
 * Subtract pSrc1's pixels from corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSub_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Sub Image
///@}

/** @name Div Image
 * Pixel by pixel division of two images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image division.
 * Divide pixels in pSrc2 by pSrc1's pixels. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiDiv_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Div Image
///@}

/** @name Div_Round Image
 * Pixel by pixel division of two images using result rounding modes.
 */
///@{

/** 
 * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 8-bit unsigned char channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                                Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 16-bit unsigned short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                 Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 16-bit signed short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                 Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

// end of Div_Round Image
///@}

/** @name Abs Image
 * Absolute value of each pixel value in an image.
 */
///@{

/** 
 * One 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C1IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C3IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image absolute value with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image absolute value with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_AC4IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C4IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image absolute value with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image absolute value with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);


// end of Abs Image
///@}

/** @name AbsDiff Image
 * Pixel by pixel absolute difference between two images.
 */
///@{

/** 
 * One 8-bit unsigned char channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

// end of AbsDiff Image
///@}

/** @name Sqr Image
 * Square each pixel in an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_AC4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_AC4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_AC4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_AC4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_AC4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_AC4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image squared with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image squared with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Sqr Image
///@}

/** @name Sqrt Image
 * Pixel by pixel square root of each pixel in an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_AC4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_AC4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_AC4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_AC4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_AC4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_AC4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image square root with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image square root with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Sqrt Image
///@}

/** @name Ln Image
 * Pixel by pixel natural logarithm of each pixel in an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Ln Image
///@}

/** @name Exp Image
 * Exponential value of each pixel in an image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image exponential.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image exponential.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image exponential.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image exponential.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Exp Image
///@}

/** @name AndC Image
 * Pixel by pixel logical and of an image with a constant.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C3IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                       Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical and with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_AC4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C3IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                        Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical and with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_AC4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C3IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                        Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical and with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_AC4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical and with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of AndC Image
///@}

/** @name OrC Image
 * Pixel by pixel logical or of an image with a constant.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C3IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_AC4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C3IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_AC4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C3IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_AC4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of OrC Image
///@}

/** @name XorC Image
 * Pixel by pixel logical exclusive or of an image with a constant.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C3IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                       Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_AC4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pConstants, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C4IR(const Npp8u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C3IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                        Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_AC4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pConstants, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C4IR(const Npp16u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C3IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                        Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_AC4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pConstants, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C4IR(const Npp32s * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of XorC Image
///@}

/** @name RShiftC Image
 * Pixel by pixel right shift of an image by a constant value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C1IR(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C3IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image right shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_AC4IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C4IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C1R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit signed char channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C1IR(const Npp32u nConstant, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C3R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit signed char channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C3IR(const Npp32u * pConstants, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_AC4R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel in place image right shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_AC4IR(const Npp32u * pConstants, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C4R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C4IR(const Npp32u * pConstants, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C1IR(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C3IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image right shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_AC4IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C4IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C1R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C1IR(const Npp32u nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C3R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C3IR(const Npp32u * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_AC4R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image right shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_AC4IR(const Npp32u * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C4R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C4IR(const Npp32u * pConstants, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C1IR(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C3IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image right shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_AC4IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image right shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C4IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of RShiftC Image
///@}

/** @name LShiftC Image
 * Pixel by pixel left shift of an image by a constant value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C1IR(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C3IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image left shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_AC4IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C4IR(const Npp32u * pConstants, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C1IR(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C3IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image left shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_AC4IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C4IR(const Npp32u * pConstants, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C1IR(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C3IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image left shift by constant with unmodified alpha.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_AC4IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u * pConstants, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image left shift by constant.
 * \param pConstants pointer to a list of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C4IR(const Npp32u * pConstants, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of LShiftC Image
///@}

/** @name And Image
 * Pixel by pixel logical and of images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                       Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                        Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                        Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of And Image
///@}

/** @name Or Image
 * Pixel by pixel logical or of images.
 */
///@{
 
/** 
 * One 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Or Image
///@}

/** @name Xor Image
 * Pixel by pixel logical exclusive or of images.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                       Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                        Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                        Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Xor Image
///@}

/** @name Not Image
 * Pixel by pixel logical not of image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C1IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C3IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical not with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical not with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_AC4IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C4IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

// end of Not Image
///@}

/** @name AlphaCompC Image
 * Composite two images using constant alpha values.
 */
///@{

/** 
 * One 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2,
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Three 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition with alpha using constant source alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 8-bit signed char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8s_C1R(const Npp8s * pSrc1, int nSrc1Step, Npp8s nAlpha1, const Npp8s * pSrc2, int nSrc2Step, Npp8s nAlpha2,
                            Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2,
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Three 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition with alpha using constant source alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit signed short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16s_C1R(const Npp16s * pSrc1, int nSrc1Step, Npp16s nAlpha1, const Npp16s * pSrc2, int nSrc2Step, Npp16s nAlpha2,
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit unsigned integer channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32u_C1R(const Npp32u * pSrc1, int nSrc1Step, Npp32u nAlpha1, const Npp32u * pSrc2, int nSrc2Step, Npp32u nAlpha2,
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit signed integer channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, Npp32s nAlpha1, const Npp32s * pSrc2, int nSrc2Step, Npp32s nAlpha2,
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit floating point channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0.0 - 1.0).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0.0 - 1.0).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, Npp32f nAlpha1, const Npp32f * pSrc2, int nSrc2Step, Npp32f nAlpha2,
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

// end of AlphaCompC Image
///@}

/** @name AlphaPremulC Image
 * Premultiplies pixels of an image using a constant alpha value.
 */
///@{

/** 
 * One 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C1IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C3IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C4IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image premultiplication with alpha using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication with alpha using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_AC4IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C1IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C3IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C4IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication with alpha using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication with alpha using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_AC4IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of AlphaPremulC Image
///@}

/** @name AlphaComp Image
 * Composite two images using alpha opacity values contained in each image.
 */
///@{

/** 
 * One 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8u_AC1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 8-bit signed char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8s_AC1R(const Npp8s * pSrc1, int nSrc1Step, const Npp8s * pSrc2, int nSrc2Step, 
                            Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16u_AC1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit signed short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16s_AC1R(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32u_AC1R(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32u_AC4R(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32s_AC1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32f_AC1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

// end of AlphaComp Image
///@}

/** @name AlphaPremul Image
 * Premultiplies image pixels by image alpha opacity values.
 */
///@{

/** 
 * Four 8-bit unsigned char channel image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

// end of AlphaPremul Image
///@}

// end of Image Arithmetic
///@}


/** 
 * @defgroup image_threshold_and_compare_operations Threshold and Compare Operations
 *
 * Methods for pixel-wise threshold and compare operations.
 */
///@{

/** 
 * @name Threshold
 *
 * Threshold pixels.
 */
///@{

/** 
 * 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                Npp32f nThreshold, NppCmpOp eComparisonOperation); 


/** 
 * 4 channel 8-bit unsigned image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                      Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u aThresholds[3], NppCmpOp eComparisonOperation);
///@}

/** @name Image Compare Methods
 * Compare the pixels of two images and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 */
///@{

/** 
 * 4 channel 8-bit unsigned image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_C4R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step,
                              const Npp8u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_32f_C1R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u  * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);
///@}

///@} image_threshold_and_compare_operations
 
/** @defgroup image_statistics_functions Statistics Functions
 *
 * Routines computing statistical image information.
 *
 *
 */
///@{

/** @name Mean_StdDev
 *  Computes the mean and standard deviation of image pixel values
 */
///@{

/** 
 * Device scratch buffer size (in bytes) for mean and standard deviation of image.
 * This primitive provides the correct buffer size for nppiMean_StdDev_8u_C1R.
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return NPP_SUCCESS
 */
NppStatus 
nppiMeanStdDev8uC1RGetBufferHostSize(NppiSize oSizeROI, int * hpBufferSize/* host pointer */);

/**
 * 8-bit unsigned mean standard deviation.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param pMean Contains computed mean.
 * \param pStdDev Contains computed standard deviation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMean_StdDev_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pMean, Npp64f * pStdDev );

///@}

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
/** @name Sum
 *  Sum of 8 bit images.
 */
///@{

/** 
 * Scratch-buffer size for nppiSum_8u_C1R.
 * 
 * \param oSizeROI ROI size.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiReductionGetBufferHostSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/** 
 * Scratch-buffer size for nppiSum_8u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus 
nppiReductionGetBufferHostSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);
 
/**
 * 8-bit unsigned image sum with 64-bit long long result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param *pSum Contains computed sum.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u64s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64s * pSum);

/**
 * 8-bit unsigned image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param *pSum Contains computed sum.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f * pSum);

/**
 * 4 channel 8-bit unsigned image sum with 64-bit long long result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param aSum Array contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u64s_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64s aSum[4]);

/**
 * 4 channel 8-bit unsigned image sum with 64-bit double precision result.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer
 * \param aSum Array contains computed sum for each channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSum_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pDeviceBuffer, Npp64f aSum[4]);

///@}

/** @name MinMax
 *  Minimum and maximum of 8-bit images.
 */
///@{

/** 
 * Scratch-buffer size for nppiMinManx_8u_C1R.
 * 
 * \param oSizeROI ROI size.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus
nppiMinMaxGetBufferSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 8-bit unsigned pixel minimum and maximum.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMin Device-memory pointer receiving the minimum result.
 * \param pMax Device-memory pointer receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferSize_8u_C1R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMinMax_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u * pMin, Npp8u * pMax, Npp8u * pDeviceBuffer);

/** 
 * Scratch-buffer size for nppiMinManx_8u_C4R.
 * 
 * \param oSizeROI ROI size.
 * \param hpBufferSize Host pointer where required buffer size is returned.
 * \return Error Code.
 */
NppStatus
nppiMinMaxGetBufferSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize /* host pointer */);

/**
 * 4 channel 8-bit unsigned pixel minimum and maximum.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aMin Device-pointer (array) receiving the minimum result.
 * \param aMax Device-pointer (array) receiving the maximum result.
 * \param pDeviceBuffer Buffer to a scratch memory. Use \ref nppiMinMaxGetBufferSize_8u_C4R to determine
 *          the minium number of bytes required.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * \note Unlike nppiMinMax_8u_C1R, this primitive returns its results as device
 *       pointers.
 */
NppStatus 
nppiMinMax_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u aMin[4], Npp8u aMax[4], Npp8u * pDeviceBuffer);
///@}



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

///@} image_statistics_functions


/** @defgroup image_filtering_functions Filtering Functions
 *
 * Linear and non-linear image filtering functions.
 *
 *
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


///@} image_filtering_functions


/** @defgroup image_morphological_operations Morphological Operations
 *
 * Morphological image operations.
 *
 *
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


///@} image_morphological_operations


/** @defgroup image_linear_transforms Image Linear Transforms
 *
 * Linear image transformations.
 *
 *
 */
///@{



/**
 * 32-bit floating point complex to 32-bit floating point magnitude.
 * 
 * Converts complex-number pixel image to single channel image computing
 * the result pixels as the magnitude of the complex values.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiMagnitude_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep,
                                Npp32f  * pDst, int nDstStep,
                          NppiSize oSizeROI);
                                          
/**
 * 32-bit floating point complex to 32-bit floating point squared magnitude.
 * 
 * Converts complex-number pixel image to single channel image computing
 * the result pixels as the squared magnitude of the complex values.
 * 
 * The squared magnitude is an itermediate result of magnitude computation and
 * can thus be computed faster than actual magnitude. If magnitudes are required
 * for sorting/comparing only, using this function instead of nppiMagnitude_32fc32f_C1R
 * can be a worthwhile performance optimization.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiMagnitudeSqr_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep,
                                   Npp32f  * pDst, int nDstStep,
                             NppiSize oSizeROI);
                                          
///@} image_linear_transforms



/**
 * @defgroup image_compression Compression
 *
 * Image compression primitives.
 *
 * The JPEG standard defines a flow of level shift, DCT and quantization for
 * forward JPEG transform and inverse level shift, IDCT and de-quantization
 * for inverse JPEG transform. This group has the functions for both forward
 * and inverse functions. 
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
                                    
///@} image_compression

/** @defgroup image_geometric_transforms Geometric Transforms
 *
 * Routines manipulating an image's geometry.
 *
 */
///@{

/** @name Resize
 *  Resizes 8 bit images. Handles C1 and C4 images.
 */
///@{

/**
 * 8-bit unsigned image resize.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstROISize Size in pixels of the destination image
 * \param xFactor Factors by which x dimension is changed 
 * \param yFactor Factors by which y dimension is changed 
 * \param eInterpolation The type of eInterpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
 *           srcROIRect has no intersection with the source image.
 *         - #NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - #NPP_RESIZE_FACTOR_ERROR Indicates an error condition if either xFactor or
 *           yFactor is less than or equal to zero.
 *         - #NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 */
NppStatus 
nppiResize_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, 
                        NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiSize dstROISize,
                        double xFactor, double yFactor, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image resize.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstROISize Size in pixels of the destination image
 * \param xFactor Factors by which x dimension is changed 
 * \param yFactor Factors by which y dimension is changed 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
 *           srcROIRect has no intersection with the source image.
 *         - #NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - #NPP_RESIZE_FACTOR_ERROR Indicates an error condition if either xFactor or
 *           yFactor is less than or equal to zero.
 *         - #NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 */
NppStatus nppiResize_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, 
                              NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiSize dstROISize,
                              double xFactor, double yFactor, int eInterpolation);

///@}

/** @name Rotate
 *  Rotates an image around the origin (0,0) and then shifts it.
 *
 */
///@{

/**
 * Compute shape of rotated image.
 * 
 * \param oSrcROI Region-of-interest of the source image.
 * \param aQuad Array of 2D points. These points are the locations of the corners
 *      of the rotated ROI. 
 * \param nAngle The rotation nAngle.
 * \param nShiftX Post-rotation shift in x-direction
 * \param nShiftY Post-rotation shift in y-direction
 * \return \ref roi_error_codes.
 */
NppStatus
nppiGetRotateQuad(NppiRect oSrcROI, double aQuad[4][2], double nAngle, double nShiftX, double nShiftY);

/**
 * Compute bounding-box of rotated image.
 * \param oSrcROI Region-of-interest of the source image.
 * \param aBoundingBox Two 2D points representing the bounding-box of the rotated image. All four points
 *      from nppiGetRotateQuad are contained inside the axis-aligned rectangle spanned by the the two
 *      points of this bounding box.
 * \param nAngle The rotation angle.
 * \param nShiftX Post-rotation shift in x-direction.
 * \param nShiftY Post-rotation shift in y-direction.
 *
 * \return \ref roi_error_codes.
 */
NppStatus 
nppiGetRotateBound(NppiRect oSrcROI, double aBoundingBox[2][2], double nAngle, double nShiftX, double nShiftY);

/**
 * 8-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRotate_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 8-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image rotate ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 16-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 16-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image rotate ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                          Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                    double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 32-bit float image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 32-bit float image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 32-bit float image rotate.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 32-bit float image rotate ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 */
NppStatus 
nppiRotate_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                          Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                    double nAngle, double nShiftX, double nShiftY, int eInterpolation);
///@}


/** @name Mirror
 *  Mirrors images horizontally, vertically and diagonally.
 */
///@{

/**
 * 8-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                  NppiSize oROI, NppiAxis flip);
                              
/**
 * 3 channel 8-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                  NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                  NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned image mirror not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 16-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);
                              
/**
 * 3 channel 16-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned image mirror not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, 
                    NppiSize oROI, NppiAxis flip);

/**
 * 32-bit image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);
                              
/**
 * 3 channel 32-bit image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit image mirror not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, 
                    NppiSize oROI, NppiAxis flip);


/**
 * 32-bit float image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);
                              
/**
 * 3 channel 32-bit float image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float image mirror.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                   NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float image mirror not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep Distance in bytes between starts of consecutive lines of the
 *        destination image.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_MIRROR_FLIP_ERR if flip has an illegal value.
 */
NppStatus 
nppiMirror_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, 
                    NppiSize oROI, NppiAxis flip);

///@}

/** @name Affine warping, affine transform calculation
 * Affine warping of an image is the transform of image pixel positions,
 * defined by the following formulas:
 * \f[
 * X_{new} = C_{00} * x + C_{01} * y + C_{02} \qquad
 * Y_{new} = C_{10} * x + C_{11} * y + C_{12} \qquad
 * C = \left[ \matrix{C_{00} & C_{01} & C_{02} \cr C_{10} & C_{11} & C_{12} } \right]
 * \f]
 * That is, any pixel with coordinates 
 * \f$(X_{new},Y_{new})\f$ in the transformed image is sourced from 
 * coordinates \f$(x,y)\f$ in the original image. The mapping \f$C\f$ is completely
 * specified by 6 values
 * \f$C_{ij}, i=\overline{0,1}, j=\overline{0,2}\f$.
 * The transform maps parallel lines to parallel
 * lines and preserves ratios of distances of points to lines.
 * Implementation specific properties are
 * discussed in each function's documentation.
 */
///@{


/**
 * Calculates affine transform coefficients given source rectangular ROI and
 * its destination quadrangle projection
 *
 * \param srcRoi Source ROI
 * \param quad Destination quadrangle
 * \param coeffs Affine transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_AFFINE_QUAD_INCORRECT_WARNING Indicates a warning when quad
 *           does not conform to the transform properties. Fourth vertex is
 *           ignored, internally computed coordinates are used instead
 */
NppStatus 
nppiGetAffineTransform(NppiRect srcRoi, const double quad[4][2], double coeffs[2][3]);


/**
 * Calculates affine transform projection of given source rectangular ROI
 *
 * \param srcRoi Source ROI
 * \param quad Destination quadrangle
 * \param coeffs Affine transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetAffineQuad(NppiRect srcRoi, double quad[4][2], const double coeffs[2][3]);


/**
 * Calculates bounding box of the affine transform projection of the given
 * source rectangular ROI
 *
 * \param srcRoi Source ROI
 * \param bound Bounding box of the transformed source ROI
 * \param coeffs Affine transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetAffineBound(NppiRect srcRoi, double bound[2][2], const double coeffs[2][3]);


/**
 * Affine transform of an image (8bit unsigned integer, single channel). This
 * function operates using given transform coefficients that can be obtained
 * by using nppiGetAffineTransform function or set explicitly. The function
 * operates on source and destination regions of interest. The affine warp
 * function transforms the source image pixel coordinates \f$(x,y)\f$ according
 * to the following formulas:
 * \f[
 * X_{new} = C_{00} * x + C_{01} * y + C_{02} \qquad
 * Y_{new} = C_{10} * x + C_{11} * y + C_{12} \qquad
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with 
 * destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated memory. In case
 * when the conditions above are not satisfied, the function may decrease in
 * speed slightly and will return NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           interpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffine_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                       int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpAffine_8u_C1R
 */
NppStatus 
nppiWarpAffine_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                       int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpAffine_8u_C1R
 */
NppStatus 
nppiWarpAffine_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                       int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpAffine_8u_C1R
 */
NppStatus 
nppiWarpAffine_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpAffine_8u_C1R
 */
NppStatus 
nppiWarpAffine_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[3], 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                       int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpAffine_8u_C1R
 */
NppStatus 
nppiWarpAffine_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[4], 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                       int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, single channel).
 * This function operates using given transform coefficients that can be
 * obtained by using nppiGetAffineTransform function or set explicitly. Thus
 * there is no need to invert coefficients in your application before calling
 * WarpAffineBack. The function operates on source and destination regions of
 * interest.
 * The affine warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * C_{00} * X_{new} + C_{01} * Y_{new} + C_{02} = x  \qquad
 * C_{10} * X_{new} + C_{11} * Y_{new} + C_{12} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but doesn't perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes aligned. This is
 * always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected image are separated from
 * the ROI by at least 63 bytes from each side. However, this requires the
 * whole ROI to be part of allocated memory. In case when the conditions above
 * are not satisfied, the function may decrease in speed slightly and will
 * return NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           interpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineBack_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                    int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                    int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpAffineBack_8u_C1R
 */
NppStatus 
nppiWarpAffineBack_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                           int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpAffineBack_8u_C1R
 */
NppStatus 
nppiWarpAffineBack_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                           int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpAffineBack_8u_C1R
 */
NppStatus 
nppiWarpAffineBack_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpAffineBack_8u_C1R
 */
NppStatus 
nppiWarpAffineBack_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[3], 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                           int eInterpolation);


/**
 * Inverse affine transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpAffineBack_8u_C1R
 */
NppStatus 
nppiWarpAffineBack_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[4], 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                           int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, single channel). This
 * function performs affine warping of a the specified quadrangle in the
 * source image to the specified quadrangle in the destination image. The
 * function nppiWarpAffineQuad uses the same formulas for pixel mapping as in
 * nppiWarpAffine function. The transform coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * eInterpolation method and written to the destination ROI.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant, but does not
 * perform memory access checks and requires the destination ROI to be 64 bytes
 * aligned. Hence any destination ROI is chunked into 3 vertical stripes: the
 * first and the third are processed by accurate kernels and the central one is
 * processed by the fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes aligned. This is
 * always true if the values <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected image are separated from
 * the ROI by at least 63 bytes from each side. However, this requires the
 * whole ROI to be part of allocated memory. In case when the conditions above
 * are not satisfied, the function may decrease in speed slightly and will
 * return NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 *         ignored, internally computed coordinates are used instead
 */
NppStatus 
nppiWarpAffineQuad_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpAffineQuad_8u_C1R
 */
NppStatus 
nppiWarpAffineQuad_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpAffineQuad_8u_C1R
 */
NppStatus 
nppiWarpAffineQuad_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpAffineQuad_8u_C1R
 */
NppStatus 
nppiWarpAffineQuad_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpAffineQuad_8u_C1R
 */
NppStatus 
nppiWarpAffineQuad_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst[3], int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpAffineQuad_8u_C1R
 */
NppStatus 
nppiWarpAffineQuad_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst[4], int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, single channel). This
 * function operates using given transform coefficients that can be obtained by
 * using nppiGetAffineTransform function or set explicitly. The function
 * operates on source and destination regions of interest. The affine warp
 * function transforms the source image pixel coordinates \f$(x,y)\f$
 * according to the following formulas:
 * \f[
 * X_{new} = C_{00} * x + C_{01} * y + C_{02} \qquad
 * Y_{new} = C_{10} * x + C_{11} * y + C_{12} \qquad
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant, but does not
 * perform memory access checks and requires the destination ROI to be 64 bytes
 * aligned. Hence any destination ROI is chunked into 3 vertical stripes: the
 * first and the third are processed by accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes aligned. This is
 * always true if the values <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt>  are multiples of
 * 64. Another rule of thumb is to specify destination ROI in such way that
 * left and right sides of the projected image are separated from the ROI by at
 * least 63 bytes from each side. However, this requires the whole ROI to be
 * part of allocated memory. In case when the conditions above are not
 * satisfied, the function may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffine_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, three channels)
 * \see nppiWarpAffine_16u_C1R
 */
NppStatus 
nppiWarpAffine_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four channels)
 * \see nppiWarpAffine_16u_C1R
 */
NppStatus 
nppiWarpAffine_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four channels RGBA)
 * \see nppiWarpAffine_16u_C1R
 */
NppStatus 
nppiWarpAffine_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                         int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, three planes)
 * \see nppiWarpAffine_16u_C1R
 */
NppStatus 
nppiWarpAffine_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[3], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four planes)
 * \see nppiWarpAffine_16u_C1R
 */
NppStatus 
nppiWarpAffine_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[4], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, single 
 * channel). This function operates using given transform coefficients that 
 * can be obtained by using nppiGetAffineTransform function or set explicitly.
 * Thus there is no need to invert coefficients in your application 
 * before calling WarpAffineBack. The function operates on source and
 * destination regions of interest.
 * The affine warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * C_{00} * X_{new} + C_{01} * Y_{new} + C_{02} = x  \qquad
 * C_{10} * X_{new} + C_{11} * Y_{new} + C_{12} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but doesn't perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function may
 * decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineBack_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, three channels)
 * \see nppiWarpAffineBack_16u_C1R
 */
NppStatus 
nppiWarpAffineBack_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, four channels)
 * \see nppiWarpAffineBack_16u_C1R
 */
NppStatus 
nppiWarpAffineBack_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, four channels
 * RGBA)
 * \see nppiWarpAffineBack_16u_C1R
 */
NppStatus 
nppiWarpAffineBack_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                             int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                             int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, three planes)
 * \see nppiWarpAffineBack_16u_C1R
 */
NppStatus 
nppiWarpAffineBack_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[3], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (16bit unsigned integer, four planes)
 * \see nppiWarpAffineBack_16u_C1R
 */
NppStatus 
nppiWarpAffineBack_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[4], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, single channel). This
 * function performs affine warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpAffineQuad uses the same
 * formulas for pixel mapping as in nppiWarpAffine function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but doesn't perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in 
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineQuad_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, three channels). 
 * \see nppiWarpAffineQuad_16u_C1R
 */
NppStatus 
nppiWarpAffineQuad_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four channels). 
 * \see nppiWarpAffineQuad_16u_C1R
 */
NppStatus 
nppiWarpAffineQuad_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four channels RGBA). 
 * \see nppiWarpAffineQuad_16u_C1R
 */
NppStatus 
nppiWarpAffineQuad_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, three planes). 
 * \see nppiWarpAffineQuad_16u_C1R
 */
NppStatus 
nppiWarpAffineQuad_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (16bit unsigned integer, four planes). 
 * \see nppiWarpAffineQuad_16u_C1R
 */
NppStatus 
nppiWarpAffineQuad_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, single channel). This function
 * operates using given transform coefficients that 
 * can be obtained by using nppiGetAffineTransform function or set explicitly.
 * The function operates on source and destination regions 
 * of interest. The affine warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = C_{00} * x + C_{01} * y + C_{02} \qquad
 * Y_{new} = C_{10} * x + C_{11} * y + C_{12} \qquad
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffine_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (32bit float, three channels).
 * \see nppiWarpAffine_32f_C1R
 */
NppStatus 
nppiWarpAffine_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (32bit float, four channels).
 * \see nppiWarpAffine_32f_C1R
 */
NppStatus 
nppiWarpAffine_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpAffine_32f_C1R
 */
NppStatus 
nppiWarpAffine_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                         int eInterpolation);


/**
 * Affine transform of an image (32bit float, three planes).
 * \see nppiWarpAffine_32f_C1R
 */
NppStatus 
nppiWarpAffine_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[3], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Affine transform of an image (32bit float, four planes).
 * \see nppiWarpAffine_32f_C1R
 */
NppStatus 
nppiWarpAffine_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[4], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                        int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, single channel). This
 * function operates using given transform coefficients that 
 * can be obtained by using nppiGetAffineTransform function or set explicitly.
 * Thus there is no need to invert coefficients in your application 
 * before calling WarpAffineBack. The function operates on source and
 * destination regions of interest.
 * The affine warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * C_{00} * X_{new} + C_{01} * Y_{new} + C_{02} = x  \qquad
 * C_{10} * X_{new} + C_{11} * Y_{new} + C_{12} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineBack_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                     int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                     int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, three channels).
 * \see nppiWarpAffineBack_32f_C1R
 */
NppStatus 
nppiWarpAffineBack_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                     int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                     int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, four channels).
 * \see nppiWarpAffineBack_32f_C1R
 */
NppStatus 
nppiWarpAffineBack_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                     int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                     int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpAffineBack_32f_C1R
 */
NppStatus 
nppiWarpAffineBack_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                      int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                      int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, three planes).
 * \see nppiWarpAffineBack_32f_C1R
 */
NppStatus 
nppiWarpAffineBack_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[3], 
                                     int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                     int eInterpolation);


/**
 * Inverse affine transform of an image (32bit float, four planes).
 * \see nppiWarpAffineBack_32f_C1R
 */
NppStatus 
nppiWarpAffineBack_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[4], 
                                     int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                     int eInterpolation);


/**
 * Affine transform of an image (32bit float, single channel). This function
 * performs affine warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpAffineQuad uses the same
 * formulas for pixel mapping as in nppiWarpAffine function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineQuad_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, three channels).
 * \see nppiWarpAffineQuad_32f_C1R
 */
NppStatus 
nppiWarpAffineQuad_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, four channels).
 * \see nppiWarpAffineQuad_32f_C1R
 */
NppStatus 
nppiWarpAffineQuad_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpAffineQuad_32f_C1R
 */
NppStatus 
nppiWarpAffineQuad_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, three planes).
 * \see nppiWarpAffineQuad_32f_C1R
 */
NppStatus 
nppiWarpAffineQuad_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit float, four planes).
 * \see nppiWarpAffineQuad_32f_C1R
 */
NppStatus 
nppiWarpAffineQuad_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, single channel). This
 * function operates using given transform coefficients that 
 * can be obtained by using nppiGetAffineTransform function or set explicitly.
 * The function operates on source and destination regions 
 * of interest. The affine warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = C_{00} * x + C_{01} * y + C_{02} \qquad
 * Y_{new} = C_{10} * x + C_{11} * y + C_{12} \qquad
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffine_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, three channels).
 * \see nppiWarpAffine_32s_C1R
 */
NppStatus 
nppiWarpAffine_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int interpolation);


/**
 * Affine transform of an image (32bit signed integer, four channels).
 * \see nppiWarpAffine_32s_C1R
 */
NppStatus 
nppiWarpAffine_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpAffine_32s_C1R
 */
NppStatus 
nppiWarpAffine_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                  int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                  int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, three planes).
 * \see nppiWarpAffine_32s_C1R
 */
NppStatus 
nppiWarpAffine_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[3], 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, four planes).
 * \see nppiWarpAffine_32s_C1R
 */
NppStatus 
nppiWarpAffine_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[4], 
                                 int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                 int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, single
 * channel). This function operates using given transform coefficients that 
 * can be obtained by using nppiGetAffineTransform function or set explicitly.
 * Thus there is no need to invert coefficients in your application 
 * before calling WarpAffineBack. The function operates on source and
 * destination regions of interest.
 * The affine warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * C_{00} * X_{new} + C_{01} * Y_{new} + C_{02} = x  \qquad
 * C_{10} * X_{new} + C_{11} * Y_{new} + C_{12} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetAffineQuad and nppiGetAffineBound can help with
 * destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Affine transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineBack_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, three channels).
 * \see nppiWarpAffineBack_32s_C1R
 */
NppStatus 
nppiWarpAffineBack_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, four channels).
 * \see nppiWarpAffineBack_32s_C1R
 */
NppStatus 
nppiWarpAffineBack_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpAffineBack_32s_C1R
 */
NppStatus 
nppiWarpAffineBack_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                             int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                             int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, three planes).
 * \see nppiWarpAffineBack_32s_C1R
 */
NppStatus 
nppiWarpAffineBack_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[3], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Inverse affine transform of an image (32bit signed integer, four planes).
 * \see nppiWarpAffineBack_32s_C1R
 */
NppStatus 
nppiWarpAffineBack_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[4], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[2][3], 
                                            int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, single channel). This
 * function performs affine warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpAffineQuad uses the same
 * formulas for pixel mapping as in nppiWarpAffine function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 *           misalignment
 */
NppStatus 
nppiWarpAffineQuad_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, three channels).
 * \see nppiWarpAffineQuad_32s_C1R
 */
NppStatus 
nppiWarpAffineQuad_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, four channels).
 * \see nppiWarpAffineQuad_32s_C1R
 */
NppStatus 
nppiWarpAffineQuad_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpAffineQuad_32s_C1R
 */
NppStatus 
nppiWarpAffineQuad_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, three planes).
 * \see nppiWarpAffineQuad_32s_C1R
 */
NppStatus 
nppiWarpAffineQuad_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Affine transform of an image (32bit signed integer, four planes).
 * \see nppiWarpAffineQuad_32s_C1R
 */
NppStatus 
nppiWarpAffineQuad_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);
///@}


/** @name Perspective warping, perspective transform calculation
 *  Perspective warping of an image is the transform of image pixel positions,
 * defined by the following formulas:
 * \f[
 * X_{new} = \frac{C_{00} * x + C_{01} * y + C_{02}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * Y_{new} = \frac{C_{10} * x + C_{11} * y + C_{12}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * C = \left[ \matrix{C_{00} & C_{01} & C_{02} \cr C_{10} & C_{11} & C_{12} \cr C_{20} & C_{21} & C_{22} } \right]
 * \f]
 * That is, any pixel of the transformed image with coordinates 
 * \f$(X_{new},Y_{new})\f$ has a preimage with 
 * coordinates \f$(x,y)\f$. The mapping \f$C\f$ is fully defined by 8 values
 * \f$C_{ij}, (i,j)=\overline{0,2}\f$, except of \f$C_{22}\f$, which is a
 * normalizer.
 * The transform has a property of mapping any convex quadrangle to a convex
 * quadrangle, which is used in a group of functions nppiWarpPerspectiveQuad.
 * The NPPI implementation of perspective transform has some issues which are
 * discussed in each function's documentation.
 */
///@{


/**
 * Calculates perspective transform coefficients given source rectangular ROI
 * and its destination quadrangle projection
 *
 * \param srcRoi Source ROI
 * \param quad Destination quadrangle
 * \param coeffs Perspective transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveTransform(NppiRect srcRoi, const double quad[4][2], double coeffs[3][3]);


/**
 * Calculates perspective transform projection of given source rectangular
 * ROI
 *
 * \param srcRoi Source ROI
 * \param quad Destination quadrangle
 * \param coeffs Perspective transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveQuad(NppiRect srcRoi, double quad[4][2], const double coeffs[3][3]);


/**
 * Calculates bounding box of the perspective transform projection of the
 * given source rectangular ROI
 *
 * \param srcRoi Source ROI
 * \param bound Bounding box of the transformed source ROI
 * \param coeffs Perspective transform coefficients
 * \return Error codes:
 *         - #NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveBound(NppiRect srcRoi, double bound[2][2], const double coeffs[3][3]);


/**
 * Perspective transform of an image (8bit unsigned integer, single channel).
 * This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. The function operates on source and destination regions 
 * of interest. The perspective warp function transforms the source image
 * pixel coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = \frac{C_{00} * x + C_{01} * y + C_{02}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * Y_{new} = \frac{C_{10} * x + C_{11} * y + C_{12}}{C_{20} * x + C_{21} * y + C_{22}}
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspective_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                       int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpPerspective_8u_C1R
 */
NppStatus 
nppiWarpPerspective_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                       int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpPerspective_8u_C1R
 */
NppStatus 
nppiWarpPerspective_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                       int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpPerspective_8u_C1R
 */
NppStatus 
nppiWarpPerspective_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpPerspective_8u_C1R
 */
NppStatus 
nppiWarpPerspective_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[3], 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                       int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpPerspective_8u_C1R
 */
NppStatus 
nppiWarpPerspective_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[4], 
                                       int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                       int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, single
 * channel). This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. Thus there is no need to invert coefficients in your application 
 * before calling WarpPerspectiveBack. The function operates on source and
 * destination regions of interest.
 * The perspective warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * \frac{C_{00} * X_{new} + C_{01} * Y_{new} + C_{02}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = x  \qquad
 * \frac{C_{10} * X_{new} + C_{11} * Y_{new} + C_{12}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI 
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI
 * in such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                         int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpPerspectiveBack_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                           int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpPerspectiveBack_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                           int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpPerspectiveBack_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpPerspectiveBack_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[3], 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                           int eInterpolation);


/**
 * Inverse perspective transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpPerspectiveBack_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp8u* pDst[4], 
                                           int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                           int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, single channel).
 * This function performs perspective warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpPerspectiveQuad uses the same
 * formulas for pixel mapping as in nppiWarpPerspective function. The
 * transform coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection
 * of destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI
 * in such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C1R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, three channels).
 * \see nppiWarpPerspectiveQuad_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C3R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four channels).
 * \see nppiWarpPerspectiveQuad_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four channels RGBA).
 * \see nppiWarpPerspectiveQuad_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_AC4R(const Npp8u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp8u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, three planes).
 * \see nppiWarpPerspectiveQuad_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_P3R(const Npp8u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst[3], int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (8bit unsigned integer, four planes).
 * \see nppiWarpPerspectiveQuad_8u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_P4R(const Npp8u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                           const double srcQuad[4][2], Npp8u* pDst[4], int nDstStep, NppiRect dstRoi, 
                                           const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, single channel).
 * This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. The function operates on source and destination regions 
 * of interest. The perspective warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = \frac{C_{00} * x + C_{01} * y + C_{02}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * Y_{new} = \frac{C_{10} * x + C_{11} * y + C_{12}}{C_{20} * x + C_{21} * y + C_{22}}
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The
 * fast method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs  Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspective_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, three channels)
 * \see nppiWarpPerspective_16u_C1R
 */
NppStatus 
nppiWarpPerspective_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four channels)
 * \see nppiWarpPerspective_16u_C1R
 */
NppStatus 
nppiWarpPerspective_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four channels RGBA)
 * \see nppiWarpPerspective_16u_C1R
 */
NppStatus 
nppiWarpPerspective_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                         int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, three planes)
 * \see nppiWarpPerspective_16u_C1R
 */
NppStatus 
nppiWarpPerspective_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[3], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four planes)
 * \see nppiWarpPerspective_16u_C1R
 */
NppStatus 
nppiWarpPerspective_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[4], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, single
 * channel). This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. Thus there is no need to invert coefficients in your application 
 * before calling WarpPerspectiveBack. The function operates on source and
 * destination regions of interest.
 * The perspective warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * \frac{C_{00} * X_{new} + C_{01} * Y_{new} + C_{02}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = x  \qquad
 * \frac{C_{10} * X_{new} + C_{11} * Y_{new} + C_{12}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI
 * to be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, three channels)
 * \see nppiWarpPerspectiveBack_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, four channels)
 * \see nppiWarpPerspectiveBack_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, four channels RGBA)
 * \see nppiWarpPerspectiveBack_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst, 
                                             int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                             int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, three planes)
 * \see nppiWarpPerspectiveBack_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[3], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (16bit unsigned integer, four planes)
 * \see nppiWarpPerspectiveBack_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp16u* pDst[4], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, single channel).
 * This function performs perspective warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpPerspectiveQuad uses the same
 * formulas for pixel mapping as in nppiWarpPerspective function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * NPPI specific recommendation: 
 * The function operates using 2 types of kernels: fast and accurate. The fast
 * method is about 4 times faster than its accurate variant,
 * but does not perform memory access checks and requires the destination ROI to
 * be 64 bytes aligned. Hence any destination ROI is 
 * chunked into 3 vertical stripes: the first and the third are processed by
 * accurate kernels and the central one is processed by the
 * fast one.
 * In order to get the maximum available speed of execution, the projection of
 * destination ROI onto image addresses must be 64 bytes
 * aligned. This is always true if the values 
 * <tt>(int)((void *)(pDst + dstRoi.x))</tt> and 
 * <tt>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</tt> 
 * are multiples of 64. Another rule of thumb is to specify destination ROI in
 * such way that left and right sides of the projected 
 * image are separated from the ROI by at least 63 bytes from each side.
 * However, this requires the whole ROI to be part of allocated 
 * memory. In case when the conditions above are not satisfied, the function
 * may decrease in speed slightly and will return 
 * NPP_MISALIGNED_DST_ROI_WARNING warning.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C1R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, three channels). 
 * \see nppiWarpPerspectiveQuad_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C3R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four channels). 
 * \see nppiWarpPerspectiveQuad_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four channels RGBA). 
 * \see nppiWarpPerspectiveQuad_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_AC4R(const Npp16u* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp16u* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, three planes). 
 * \see nppiWarpPerspectiveQuad_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_P3R(const Npp16u* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (16bit unsigned integer, four planes). 
 * \see nppiWarpPerspectiveQuad_16u_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_P4R(const Npp16u* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp16u* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, single channel). This
 * function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. The function operates on source and destination regions 
 * of interest. The perspective warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = \frac{C_{00} * x + C_{01} * y + C_{02}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * Y_{new} = \frac{C_{10} * x + C_{11} * y + C_{12}}{C_{20} * x + C_{21} * y + C_{22}}
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspective_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit float, three channels).
 * \see nppiWarpPerspective_32f_C1R
 */
NppStatus 
nppiWarpPerspective_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four channels).
 * \see nppiWarpPerspective_32f_C1R
 */
NppStatus 
nppiWarpPerspective_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpPerspective_32f_C1R
 */
NppStatus 
nppiWarpPerspective_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                         int eInterpolation);


/**
 * Perspective transform of an image (32bit float, three planes).
 * \see nppiWarpPerspective_32f_C1R
 */
NppStatus 
nppiWarpPerspective_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[3], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four planes).
 * \see nppiWarpPerspective_32f_C1R
 */
NppStatus 
nppiWarpPerspective_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[4], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, single channel).
 * This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. Thus there is no need to invert coefficients in your application 
 * before calling WarpPerspectiveBack. The function operates on source and
 * destination regions of interest.
 * The perspective warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * \frac{C_{00} * X_{new} + C_{01} * Y_{new} + C_{02}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = x  \qquad
 * \frac{C_{10} * X_{new} + C_{11} * Y_{new} + C_{12}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, three channels).
 * \see nppiWarpPerspectiveBack_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, four channels).
 * \see nppiWarpPerspectiveBack_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpPerspectiveBack_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst, 
                                             int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                             int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, three planes).
 * \see nppiWarpPerspectiveBack_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[3], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit float, four planes).
 * \see nppiWarpPerspectiveBack_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32f* pDst[4], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Perspective transform of an image (32bit float, single channel).
 * This function performs perspective warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpPerspectiveQuad uses the same
 * formulas for pixel mapping as in nppiWarpPerspective function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C1R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, three channels).
 * \see nppiWarpPerspectiveQuad_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C3R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four channels).
 * \see nppiWarpPerspectiveQuad_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four channels RGBA).
 * \see nppiWarpPerspectiveQuad_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_AC4R(const Npp32f* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp32f* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, three planes).
 * \see nppiWarpPerspectiveQuad_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_P3R(const Npp32f* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit float, four planes).
 * \see nppiWarpPerspectiveQuad_32f_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_P4R(const Npp32f* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32f* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, single channel).
 * This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. The function operates on source and destination regions 
 * of interest. The perspective warp function transforms the source image pixel
 * coordinates \f$(x,y)\f$ according to the following formulas:
 * \f[
 * X_{new} = \frac{C_{00} * x + C_{01} * y + C_{02}}{C_{20} * x + C_{21} * y + C_{22}} \qquad
 * Y_{new} = \frac{C_{10} * x + C_{11} * y + C_{12}}{C_{20} * x + C_{21} * y + C_{22}}
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs  Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspective_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, three channels).
 * \see nppiWarpPerspective_32s_C1R
 */
NppStatus 
nppiWarpPerspective_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four channels).
 * \see nppiWarpPerspective_32s_C1R
 */
NppStatus 
nppiWarpPerspective_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpPerspective_32s_C1R
 */
NppStatus 
nppiWarpPerspective_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                         int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                         int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, three planes).
 * \see nppiWarpPerspective_32s_C1R
 */
NppStatus 
nppiWarpPerspective_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[3], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four planes).
 * \see nppiWarpPerspective_32s_C1R
 */
NppStatus 
nppiWarpPerspective_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[4], 
                                        int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                        int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, single
 * channel). This function operates using given transform coefficients that 
 * can be obtained by using nppiGetPerspectiveTransform function or set
 * explicitly. Thus there is no need to invert coefficients in your
 * application 
 * before calling WarpPerspectiveBack. The function operates on source
 * and destination regions of interest.
 * The perspective warp function transforms the source image pixel coordinates
 * \f$(x,y)\f$ according to the following formulas:
 * \f[
 * \frac{C_{00} * X_{new} + C_{01} * Y_{new} + C_{02}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = x  \qquad
 * \frac{C_{10} * X_{new} + C_{11} * Y_{new} + C_{12}}{C_{20} * X_{new} + C_{21} * Y_{new} + C_{22}} = y
 * \f]
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
 * with destination ROI specification.
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param coeffs Perspective transform coefficients
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, three channels).
 * \see nppiWarpPerspectiveBack_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, four channels).
 * \see nppiWarpPerspectiveBack_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpPerspectiveBack_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst, 
                                             int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                             int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, three planes).
 * \see nppiWarpPerspectiveBack_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[3], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Inverse perspective transform of an image (32bit signed integer, four planes).
 * \see nppiWarpPerspectiveBack_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveBack_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, Npp32s* pDst[4], 
                                            int nDstStep, NppiRect dstRoi, const double coeffs[3][3], 
                                            int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, single channel).
 * This function performs perspective warping of a the specified
 * quadrangle in the source image to the specified quadrangle in the
 * destination image. The function nppiWarpPerspectiveQuad uses the same
 * formulas for pixel mapping as in nppiWarpPerspective function. The transform
 * coefficients are computed internally.
 * The transformed part of the source image is resampled using the specified
 * interpolation method and written to the destination ROI.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels
 * \param nSrcStep \ref source_image_line_step.
 * \param srcRoi Source ROI
 * \param srcQuad Source quadrangle
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param dstRoi Destination ROI
 * \param dstQuad Destination quadrangle
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *         - #NPP_RECT_ERROR Indicates an error condition if width or height of
 *           the intersection of the srcRoi and source image is less than or
 *           equal to 1
 *         - #NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *           srcRoi has no intersection with the source image
 *         - #NPP_INTERPOLATION_ERROR Indicates an error condition if
 *           eInterpolation has an illegal value
 *         - #NPP_COEFF_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - #NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *           operation is performed if the transformed source ROI has no
 *           intersection with the destination ROI
 *         - #NPP_MISALIGNED_DST_ROI_WARNING Indicates a warning that the speed
 *           of primitive execution was reduced due to destination ROI
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C1R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, three channels).
 * \see nppiWarpPerspectiveQuad_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C3R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four channels).
 * \see nppiWarpPerspectiveQuad_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four channels RGBA).
 * \see nppiWarpPerspectiveQuad_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_AC4R(const Npp32s* pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                             const double srcQuad[4][2], Npp32s* pDst, int nDstStep, NppiRect dstRoi, 
                                             const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, three planes).
 * \see nppiWarpPerspectiveQuad_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_P3R(const Npp32s* pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst[3], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);


/**
 * Perspective transform of an image (32bit signed integer, four planes).
 * \see nppiWarpPerspectiveQuad_32s_C1R
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_P4R(const Npp32s* pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect srcRoi, 
                                            const double srcQuad[4][2], Npp32s* pDst[4], int nDstStep, NppiRect dstRoi, 
                                            const double dstQuad[4][2], int eInterpolation);

///@}


///@} image_geometry_transforms



/** @defgroup image_color_conversion Color Conversion
 *
 * Image color space and sample conversion operations.
 *
 */
///@{

/** @name RGBToYCbCr 
 *  RGB to YCbCr color conversion.
 */
///@{
/**
 * 3 channel 8-bit unsigned packed RGB to packed YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRGBToYCbCr_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned RGB to 2 channel chroma packed YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRGBToYCbCr422_8u_C3C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRGBToYCbCr420_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u ** pDst, int nDstStep[3], NppiSize oSizeROI);

/**
 * 3 channel planar 8-bit unsigned RGB to YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRGBToYCbCr_8u_P3R(const Npp8u * const * pSrc, int nSrcStep, Npp8u ** pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned RGB to YCbCr color conversion, ignoring Alpha.
 * 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRGBToYCbCr_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

///@}


/** @name YCbCrToRGB 
 *  YCbCr to RGB color conversion.
 */
///@{

/**
 * 3 channel 8-bit unsigned packed YCbCr to RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCrToRGB_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCrToRGB_8u_P3R(const Npp8u * const * pSrc, int nSrcStep, Npp8u ** pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed YCbCr to RGB color conversion, not affecting Alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCrToRGB_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned YCbCr422 to 3 channel packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr422ToRGB_8u_C2C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI); 

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr420ToRGB_8u_P3C3R(const Npp8u * const * pSrc, int nSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

///@}


/** @name Sample Pattern Conversion. 
 */
///@{

/**
 * 3 channel 8-bit unsigned planar YCbCr:422 to YCbCr:420 resampling.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr422ToYCbCr420_8u_P3R(const Npp8u * const * pSrc, int nSrcStep[3], Npp8u ** pDst, int nDstStep[3], NppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr:422 to YCbCr:411 resampling.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr422ToYCbCr411_8u_P3R(const Npp8u * const * pSrc, int nSrcStep[3], Npp8u ** pDst, int nDstStep[3], NppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr:420 to YCbCr:422 resampling.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr420ToYCbCr422_8u_P3R(const Npp8u * const * pSrc, int nSrcStep[3], Npp8u ** pDst, int nDstStep[3], NppiSize oSizeROI);


/**
 * 3 channel 8-bit unsigned planar YCbCr:420 to YCbCr:411 resampling.
 *
 * \param pSrc Array of pointers to the source image planes.
 * \param aSrcStep Array with distances in bytes between starts of consecutive
 *        lines of the source image planes.
 * \param pDstY \ref destination_image_pointer. Y-channel.
 * \param nDstYStep \ref destination_image_line_step. Y-channel.
 * \param pDstCbCr \ref destination_image_pointer. CbCr image.
 * \param nDstCbCrStep \ref destination_image_line_step. CbCr image.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiYCbCr420ToYCbCr411_8u_P3P2R(const Npp8u * const * pSrc, int aSrcStep[3], 
                                          Npp8u * pDstY, int nDstYStep, Npp8u * pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI);

///@}


/** @name Color Processing
 *  Color manipuliation functions.
 */
///@{

/**
 * 3 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point pixel values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param twist The color twist matrix with floating-point pixel values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiColorTwist32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                         NppiSize oSizeROI, const Npp32f twist[3][4]);

/**
 * 3 channel planar 8-bit unsigned color twist.
 *
 * An input color twist matrix with floating-point pixel values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param twist The color twist matrix with floating-point pixel values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiColorTwist32f_8u_P3R(const Npp8u * const * pSrc, int nSrcStep, Npp8u ** pDst, int nDstStep, 
                         NppiSize oSizeROI, const Npp32f twist[3][4]);

/**
 * 4 channel 8-bit unsigned color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point pixel values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param twist The color twist matrix with floating-point pixel values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiColorTwist32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, 
                          NppiSize oSizeROI, const Npp32f twist[3][4]);

/**
 * 8-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through linear
 * interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values
 * \param pLevels Pointer to an array of user defined INPUT values
 * \param nLevels Number of user defined input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - #NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than
 *         2.
 */
NppStatus 
nppiLUT_Linear_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                const Npp32s * pValues, const Npp32s * pLevels, int nLevels);

/**
 * 3 channel 8-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear
 * interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Double pointer to an [3] of arrays of user defined OUTPUT
 *        values per CHANNEL
 * \param pLevels Double pointer to an [3] of arrays of user defined INPUT
 *        values per CHANNEL
 * \param nLevels A [3] array of user defined input/output mapping points
 *        (levels) per CHANNEL
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - #NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than
 *         2.
 */
NppStatus 
nppiLUT_Linear_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32s * pValues[3], const Npp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through linear
 * interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Double pointer to an [4] of arrays of user defined OUTPUT
 *        values per CHANNEL
 * \param pLevels Double pointer to an [4] of arrays of user defined INPUT
 *        values per CHANNEL
 * \param nLevels A [4] array of user defined input/output mapping points
 *        (levels) per CHANNEL
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - #NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than
 *         2.
 */
NppStatus 
nppiLUT_Linear_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                 const Npp32s * pValues[4], const Npp32s * pLevels[4], int nLevels[4]);

///@}

///@} image_color_conversion


/** @defgroup image_labeling_and_segmentation Labeling and Segmentation
 *
 * Pixel labeling and image segementation operations.
 *
 */
///@{

#if defined (__cplusplus)
struct NppiGraphcutState;
#else
typedef struct NppiGraphcutState NppiGraphcutState;
#endif

/**
 * Calculates the size of the temporary buffer for graph-cut with 4 neighborhood labeling.
 *
 * \see nppiGraphcutInitAlloc(), nppiGraphcut_32s8u().
 * 
 * \param oSize Graph size.
 * \param pBufSize Pointer to variable that returns the size of the
 *        temporary buffer. 
 *
 * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
 *         or a warning
 * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *         has zero or negative value
 * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
 *         pointer is NULL
 */
NppStatus nppiGraphcutGetSize(NppiSize oSize, int * pBufSize);

/**
 * Calculates the size of the temporary buffer for graph-cut with 8 neighborhood labeling.
 *
 * \see nppiGraphcut8InitAlloc(), nppiGraphcut8_32s8u().
 * 
 * \param oSize Graph size.
 * \param pBufSize Pointer to variable that returns the size of the
 *        temporary buffer. 
 *
 * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
 *         or a warning
 * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *         has zero or negative value
 * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
 *         pointer is NULL
 */
NppStatus nppiGraphcut8GetSize(NppiSize oSize, int * pBufSize);

/**
 * Initializes graph-cut state structure and allocates additional resources for graph-cut with 8 neighborhood labeling.
 *
 * \see nppiGraphcut_32s8u(), nppiGraphcutGetSize().
 * 
 * \param oSize Graph size
 * \param ppState Pointer to pointer to graph-cut state structure. 
 * \param pDeviceMem pDeviceMem to the sufficient amount of device memory. The CUDA runtime or NPP memory allocators
 *      must be used to allocate this memory. The minimum amount of device memory required
 *      to run graph-cut on a for a specific image size is computed by nppiGraphcutGetSize().
 *
 * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
 *         or a warning
 * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *         has zero or negative value
 * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
 *         pointer is NULL
 */
NppStatus nppiGraphcutInitAlloc(NppiSize oSize, NppiGraphcutState** ppState, Npp8u* pDeviceMem);

/**
 * Allocates and initializes the graph-cut state structure and additional resources for graph-cut with 8 neighborhood labeling.
 *
 * \see nppiGraphcut8_32s8u(), nppiGraphcut8GetSize().
 * 
 * \param oSize Graph size
 * \param ppState Pointer to pointer to graph-cut state structure. 
 * \param pDeviceMem to the sufficient amount of device memory. The CUDA runtime or NPP memory allocators
 *      must be used to allocate this memory. The minimum amount of device memory required
 *      to run graph-cut on a for a specific image size is computed by nppiGraphcut8GetSize().
 *
 * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
 *         or a warning
 * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *         has zero or negative value
 * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
 *         pointer is NULL
 */
NppStatus nppiGraphcut8InitAlloc(NppiSize oSize, NppiGraphcutState ** ppState, Npp8u * pDeviceMem);

/**
 * Frees the additional resources of the graph-cut state structure.
 *
 * \see nppiGraphcutInitAlloc
 * \see nppiGraphcut8InitAlloc
 * 
 * \param pState Pointer to graph-cut state structure. 
 *
 * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
 *         or a warning
 * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *         has zero or negative value
 * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pState
 *         pointer is NULL
 */
NppStatus nppiGraphcutFree(NppiGraphcutState * pState);

/**
 * Graphcut of a flow network (32bit signed integer edge capacities). The
 * function computes the minimal cut (graphcut) of a 2D regular 4-connected
 * graph. 
 * The inputs are the capacities of the horizontal (in transposed form),
 * vertical and terminal (source and sink) edges. The capacities to source and
 * sink 
 * are stored as capacity differences in the terminals array 
 * ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
 * edge capacities 
 * for boundary edges that would connect to nodes outside the specified domain
 * are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
 * computed labeling may be wrong!
 * The computed binary labeling is encoded as unsigned 8bit values (0 / 255).
 *
 * \see nppiGraphcutInitAlloc(), nppiGraphcutFree(), nppiGraphcutGetSize().
 *
 * \param pTerminals Pointer to differences of terminal edge capacities
 *        (terminal(x) = source(x) - sink(x))
 * \param pLeftTransposed Pointer to transposed left edge capacities
 *        (left(0,*) must be 0)
 * \param pRightTransposed Pointer to transposed right edge capacities
 *        (right(width-1,*) must be 0)
 * \param pTop Pointer to top edge capacities (top(*,0) must be 0)
 * \param pBottom Pointer to bottom edge capacities (bottom(*,height-1)
 *        must be 0)
 * \param nStep Step in bytes between any pair of sequential rows of edge
 *        capacities
 * \param nTransposedStep Step in bytes between any pair of sequential
 *        rows of tranposed edge capacities
 * \param size Graph size
 * \param pLabel Pointer to destination label image 
 * \param nLabelStep Step in bytes between any pair of sequential rows of
 *        label image
 * \param pState Pointer to graph-cut state structure. This structure must be
 *          initialized allocated and initialized using nppiGraphcutInitAlloc().
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGraphcut_32s8u(Npp32s * pTerminals, Npp32s * pLeftTransposed, Npp32s * pRightTransposed, Npp32s * pTop, Npp32s * pBottom, int nStep, int nTransposedStep, NppiSize size, Npp8u * pLabel, int nLabelStep, NppiGraphcutState *pState);


/**
 * Graphcut of a flow network (32bit signed integer edge capacities). The
 * function computes the minimal cut (graphcut) of a 2D regular 8-connected
 * graph. 
 * The inputs are the capacities of the horizontal (in transposed form),
 * vertical and terminal (source and sink) edges. The capacities to source and
 * sink 
 * are stored as capacity differences in the terminals array 
 * ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
 * edge capacities 
 * for boundary edges that would connect to nodes outside the specified domain
 * are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
 * computed labeling may be wrong!
 * The computed binary labeling is encoded as unsigned 8bit values (0 / 255).
 *
 * \see nppiGraphcut8InitAlloc(), nppiGraphcutFree(), nppiGraphcut8GetSize().
 *
 * \param pTerminals Pointer to differences of terminal edge capacities
 *        (terminal(x) = source(x) - sink(x))
 * \param pLeftTransposed Pointer to transposed left edge capacities
 *        (left(0,*) must be 0)
 * \param pRightTransposed Pointer to transposed right edge capacities
 *        (right(width-1,*) must be 0)
 * \param pTop Pointer to top edge capacities (top(*,0) must be 0)
 * \param pTopLeft Pointer to top left edge capacities (topleft(*,0) 
 *		  & topleft(0,*) must be 0)
 * \param pTopRight Pointer to top right edge capacities (topright(*,0)
 *        & topright(width-1,*) must be 0)
 * \param pBottom Pointer to bottom edge capacities (bottom(*,height-1)
 *        must be 0)
 * \param pBottomLeft Pointer to bottom left edge capacities 
 *        (bottomleft(*,height-1) && bottomleft(0,*) must be 0)
 * \param pBottomRight Pointer to bottom right edge capacities 
 *        (bottomright(*,height-1) && bottomright(width-1,*) must be 0)
 * \param nStep Step in bytes between any pair of sequential rows of edge
 *        capacities
 * \param nTransposedStep Step in bytes between any pair of sequential
 *        rows of tranposed edge capacities
 * \param size Graph size
 * \param pLabel Pointer to destination label image 
 * \param nLabelStep Step in bytes between any pair of sequential rows of
 *        label image
 * \param pState Pointer to graph-cut state structure. This structure must be
 *          initialized allocated and initialized using nppiGraphcut8InitAlloc().
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGraphcut8_32s8u(Npp32s * pTerminals, Npp32s * pLeftTransposed, Npp32s * pRightTransposed, Npp32s * pTop, Npp32s * pTopLeft, Npp32s * pTopRight, Npp32s * pBottom, Npp32s * pBottomLeft, Npp32s * pBottomRight, int nStep, int nTransposedStep, NppiSize size, Npp8u * pLabel, int nLabelStep, NppiGraphcutState *pState);


///@} image_labeling_and_segmentation


// end of Image Processing module
///@}
 
#ifdef __cplusplus
} // extern "C"
#endif

#endif // NV_NPPI_H
