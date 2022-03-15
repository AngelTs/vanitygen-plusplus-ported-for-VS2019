/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2011, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#if !defined(__STORAGE_CLASS_H__)
#define __STORAGE_CLASS_H__

#if !defined(__var_used__)

#define __var_used__

#endif /* __var_used__ */

#if !defined(__loc_sc__)

#define __loc_sc__(loc, size, sc) \
        __storage##_##sc##size##loc loc

#endif /* !__loc_sc__ */

#if !defined(__storage___device__)
#define __storage___device__ static __var_used__
#endif /* __storage___device__ */

#if !defined(__storage_extern__device__)
#define __storage_extern__device__ static __var_used__
#endif /* __storage_extern__device__ */

#if !defined(__storage_auto__device__)
#define __storage_auto__device__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_auto__device__ */

#if !defined(__storage_static__device__)
#define __storage_static__device__ static __var_used__
#endif /* __storage_static__device__ */

#if !defined(__storage___constant__)
#define __storage___constant__ static __var_used__
#endif /* __storage___constant__ */

#if !defined(__storage_extern__constant__)
#define __storage_extern__constant__ static __var_used__
#endif /* __storage_extern__constant__ */

#if !defined(__storage_auto__constant__)
#define __storage_auto__constant__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_auto__constant__ */

#if !defined(__storage_static__constant__)
#define __storage_static__constant__ static __var_used__
#endif /* __storage_static__constant__ */

#if !defined(__storage___shared__)
#define __storage___shared__ static __var_used__
#endif /* __storage___shared__ */

#if !defined(__storage_extern__shared__)
#define __storage_extern__shared__ static __var_used__
#endif /* __storage_extern__shared__ */

#if !defined(__storage_auto__shared__)
#define __storage_auto__shared__ static
#endif /* __storage_auto__shared__ */

#if !defined(__storage_static__shared__)
#define __storage_static__shared__ static __var_used__
#endif /* __storage_static__shared__ */

#if !defined(__storage__unsized__shared__)
#define __storage__unsized__shared__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage__unsized__shared__ */

#if !defined(__storage_extern_unsized__shared__)
#define __storage_extern_unsized__shared__ static __var_used__
#endif /* __storage_extern_unsized__shared__ */

#if !defined(__storage_auto_unsized__shared__)
#define __storage_auto_unsized__shared__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_auto_unsized__shared__ */

#if !defined(__storage_static_unsized__shared__)
#define __storage_static_unsized__shared__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_static_unsized__shared__ */

#if !defined(__storage___text__)
#define __storage___text__ static __var_used__
#endif /* __storage___text__ */

#if !defined(__storage_extern__text__)
#define __storage_extern__text__ static __var_used__
#endif /* __storage_extern__text__ */

#if !defined(__storage_auto__text__)
#define __storage_auto__text__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_auto__text__ */

#if !defined(__storage_static__text__)
#define __storage_static__text__ static __var_used__
#endif /* __storage_static__text__ */

#if !defined(__storage___surf__)
#define __storage___surf__ static __var_used__
#endif /* __storage___surf__ */

#if !defined(__storage_extern__surf__)
#define __storage_extern__surf__ static __var_used__
#endif /* __storage_extern__surf__ */

#if !defined(__storage_auto__surf__)
#define __storage_auto__surf__ @@@ COMPILER @@@ ERROR @@@
#endif /* __storage_auto__surf__ */

#if !defined(__storage_static__surf__)
#define __storage_static__surf__ static __var_used__
#endif /* __storage_static__surf__ */

#endif /* !__STORAGE_CLASS_H__ */
