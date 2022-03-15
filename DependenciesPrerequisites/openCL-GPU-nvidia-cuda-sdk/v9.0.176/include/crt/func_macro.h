/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#if !defined(__FUNC_MACRO_H__)
#define __FUNC_MACRO_H__

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#error -- incorrect inclusion of a cudart header file

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

#if defined(__GNUC__)

#define __func__(decl) \
        inline decl

#define __device_func__(decl) \
        static __attribute__((__unused__)) decl

#elif defined(_WIN32)

#define __func__(decl) \
        static inline decl

#define __device_func__(decl) \
        static decl

#endif /* __GNUC__ */

#endif /* __FUNC_MACRO_H__ */
