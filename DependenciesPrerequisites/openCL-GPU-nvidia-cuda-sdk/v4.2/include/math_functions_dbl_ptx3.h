/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

#if !defined(__MATH_FUNCTIONS_DBL_PTX3_H__)
#define __MATH_FUNCTIONS_DBL_PTX3_H__

/* True double precision implementations, since native double support */

#if defined(__CUDABE__)

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

static __forceinline__ double rint(double a)
{
  return __builtin_round(a);
}

static __forceinline__ long int lrint(double a)
{
#if defined(__LP64__)
  return (long int)__double2ll_rn(a);
#else /* __LP64__ */
  return (long int)__double2int_rn(a);
#endif /* __LP64__ */
}

static __forceinline__ long long int llrint(double a)
{
  return __double2ll_rn(a);
}

static __forceinline__ double nearbyint(double a)
{
  return __builtin_round(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

static __forceinline__ int __signbitd(double a)
{
  return (int)((unsigned int)__double2hiint(a) >> 31);
}

static __forceinline__ int __isfinited(double a)
{
  return fabs(a) < CUDART_INF;
}

static __forceinline__ int __isinfd(double a)
{
  return fabs(a) == CUDART_INF;
}

static __forceinline__ int __isnand(double a)
{
  return !(fabs(a) <= CUDART_INF);
}

#if defined(__APPLE__)

static __forceinline__ int __signbitl(/* we do not support long double yet, hence double */double a)
{
  return __signbitd((double)a);
}

static __forceinline__ int __isfinite(/* we do not support long double yet, hence double */double a)
{
  return __isfinited((double)a);
}

static __forceinline__ int __isinf(/* we do not support long double yet, hence double */double a)
{
  return __isinfd((double)a);
}

static __forceinline__ int __isnan(/* we do not support long double yet, hence double */double a)
{
  return __isnand((double)a);
}

#else /* __APPLE__ */

static __forceinline__ int __signbit(double a)
{
  return __signbitd(a);
}

static __forceinline__ int __signbitl(/* we do not support long double yet, hence double */double a)
{
  return __signbit((double)a);
}

static __forceinline__ int __finite(double a)
{
  return __isfinited(a);
}

static __forceinline__ int __finitel(/* we do not support long double yet, hence double */double a)
{
  return __finite((double)a);
}

static __forceinline__ int __isinf(double a)
{
  return __isinfd(a);
}

static __forceinline__ int __isinfl(/* we do not support long double yet, hence double */double a)
{
  return __isinf((double)a);
}

static __forceinline__ int __isnan(double a)
{
  return __isnand(a);
}

static __forceinline__ int __isnanl(/* we do not support long double yet, hence double */double a)
{
  return __isnan((double)a);
}

#endif /* __APPLE__ */

static __forceinline__ double copysign(double a, double b)
{
  int alo, ahi, bhi;

  bhi = __double2hiint(b);
  alo = __double2loint(a);
  ahi = __double2hiint(a);
  ahi = (bhi & 0x80000000) | (ahi & ~0x80000000);
  return __hiloint2double(ahi, alo);
}

/* like copysign, but requires that argument a is postive */
static __forceinline__ double __internal_copysign_pos(double a, double b)
{
  int alo, ahi, bhi;

  bhi = __double2hiint(b);
  alo = __double2loint(a);
  ahi = __double2hiint(a);
  ahi = (bhi & 0x80000000) | ahi;
  return __hiloint2double(ahi, alo);
}

static __forceinline__ double __internal_fast_rcp(double a) 
{ 
  double e, y; 
  float x; 
  asm ("cvt.f32.f64.rn     %0,%1;" : "=f"(x) : "d"(a)); 
  asm ("rcp.approx.f32.ftz %0,%1;" : "=f"(x) : "f"(x)); 
  asm ("cvt.f64.f32        %0,%1;" : "=d"(y) : "f"(x)); 
  e = __fma_rn (-a, y, 1.0); 
  e = __fma_rn ( e, e,   e); 
  y = __fma_rn ( e, y,   y); 
  return y; 
} 

static __forceinline__ double __internal_fast_rsqrt(double a) 
{
  double x, e, t;
  float f;
  asm ("cvt.f32.f64.rn       %0, %1;" : "=f"(f) : "d"(a));
  asm ("rsqrt.approx.f32.ftz %0, %1;" : "=f"(f) : "f"(f));
  asm ("cvt.f64.f32          %0, %1;" : "=d"(x) : "f"(f));
  t = __dmul_rn (x, -x);
  e = __fma_rn (t, a, 1.0);
  t = __dmul_rn (0.75, e);
  t = __fma_rn (t, e, e);
  t = __hiloint2double (__double2hiint(t) - 0x00100000, __double2loint(t));
  t = __fma_rn (t, x, x);
  return t;
}

/* 1152 bits of 2/PI for Payne-Hanek style argument reduction. */
static __constant__ unsigned long long int __cudart_i2opi_d [] = {
  0x6bfb5fb11f8d5d08ULL,
  0x3d0739f78a5292eaULL,
  0x7527bac7ebe5f17bULL,
  0x4f463f669e5fea2dULL,
  0x6d367ecf27cb09b7ULL,
  0xef2f118b5a0a6d1fULL,
  0x1ff897ffde05980fULL,
  0x9c845f8bbdf9283bULL,
  0x3991d639835339f4ULL,
  0xe99c7026b45f7e41ULL,
  0xe88235f52ebb4484ULL,
  0xfe1deb1cb129a73eULL,
  0x06492eea09d1921cULL,
  0xb7246e3a424dd2e0ULL,
  0xfe5163abdebbc561ULL,
  0xdb6295993c439041ULL,
  0xfc2757d1f534ddc0ULL,
  0xa2f9836e4e441529ULL,
};

/* Payne-Hanek style argument reduction. */
static
#if __CUDA_ARCH__ >= 200
__noinline__
#else
__forceinline__
#endif
double __internal_trig_reduction_slowpathd(double a, int *quadrant)
{
  unsigned long long int ia;
  unsigned long long int s;
  unsigned long long int result[5];
  unsigned long long int phi, plo;
  unsigned long long int hi, lo;
  unsigned int e;
  int idx;
  int q;
  ia = __double_as_longlong(a);
  s = ia & 0x8000000000000000ULL;
  e = (unsigned int)(((ia >> 52) & 0x7ff) - 1024);
  ia = (ia << 11) | 0x8000000000000000ULL;
  /* compute x * 2/pi */
  idx = 16 - (e >> 6);
  hi = 0;
#pragma unroll 1
  for (q = (idx-1); q < min(18,idx+3); q++) {
    plo = __cudart_i2opi_d[q] * ia;
    phi = __umul64hi (__cudart_i2opi_d[q], ia);
    lo = hi + plo;
    hi = phi + (lo < plo);
    result[q-(idx-1)] = lo;
  }
  result[q-(idx-1)] = hi;
  e = e & 63;
  /* shift result such that hi:lo<127:126> are the least significant
     integer bits, and hi:lo<125:0> are the fractional bits of the result
  */
  hi = result[3];
  lo = result[2];
  if (e) {
    q = 64 - e;
    hi = (hi << e) | (lo >> q);
    lo = (lo << e) | (result[1] >> q);
  }
  q = (int)(hi >> 62);
  /* fraction */
  hi = (hi << 2) | (lo >> 62);
  lo = (lo << 2);
  e = hi >> 63; /* fraction >= 0.5 */
  q += e;
  if (s) q = -q;
  *quadrant = q;
  if (e) {
    unsigned long long int t;
    hi = ~hi;
    lo = -(long long int)lo;
    t = (lo == 0ULL);
    hi += t;
    s = s ^ 0x8000000000000000ULL;
  }
  /* normalize fraction */
  e = __clzll(hi);
  if (e) {
    hi = (hi << e) | (lo >> (64 - e)); 
  }
  lo = hi * 0xC90FDAA22168C235ULL;
  hi = __umul64hi (hi, 0xC90FDAA22168C235ULL);
  if ((long long int)hi > 0) {
    hi = (hi << 1) | (lo >> 63);
    e++;
  }
  ia = s | ((((unsigned long long int)(1022 - e)) << 52) + 
            ((((hi + 1) >> 10) + 1) >> 1));
  return __longlong_as_double(ia);
}

static __forceinline__ double __internal_trig_reduction_kerneld(double a, int *quadrant)
{
  double j, t;
  int q;
  /* NOTE: for an input of -0, this returns -0 */
  q = __double2int_rn (a * CUDART_2_OVER_PI);
  j = (double)q;
  /* Constants from S. Boldo, M. Daumas, R.-C. Li: "Formally Verified Argument
   * Reduction with a Fused-Multiply-Add", retrieved from the internet at
   * http://arxiv.org/PS_cache/arxiv/pdf/0708/0708.3722v1.pdf
   */
  t = __fma_rn (-j, 1.5707963267948966e+000, a);
  t = __fma_rn (-j, 6.1232339957367574e-017, t);
  t = __fma_rn (-j, 8.4784276603688985e-032, t);
  if (fabs(a) > CUDART_TRIG_PLOSS) {
    t = __internal_trig_reduction_slowpathd (a, &q);
  }
  *quadrant = q;
  return t;
}

/* approximate sine on -pi/4...+pi/4 */
static __forceinline__ double __internal_sin_kerneld(double x)
{
  double x2, z;
  x2 = x * x;
  z =                   1.5896230157221844E-010;
  z = __fma_rn (z, x2, -2.5050747762850355E-008);
  z = __fma_rn (z, x2,  2.7557313621385676E-006);
  z = __fma_rn (z, x2, -1.9841269829589539E-004);
  z = __fma_rn (z, x2,  8.3333333333221182E-003);
  z = __fma_rn (z, x2, -1.6666666666666630E-001);
  z  = z * x2;
  z  = __fma_rn (z, x, x);
  return z;
}

/* approximate cosine on -pi/4...+pi/4 */
static __forceinline__ double __internal_cos_kerneld(double x)
{
  double x2, z;
  x2 = x * x;
  z  =                  -1.136788825395985E-011;   
  z  = __fma_rn (z, x2,  2.087588480545065E-009);
  z  = __fma_rn (z, x2, -2.755731555403950E-007);
  z  = __fma_rn (z, x2,  2.480158729365970E-005);
  z  = __fma_rn (z, x2, -1.388888888888074E-003);
  z  = __fma_rn (z, x2,  4.166666666666664E-002);
  z  = __fma_rn (z, x2, -5.000000000000000E-001);
  z  = __fma_rn (z, x2,  1.000000000000000E+000);
  return z;
}

/* approximate tangent on -pi/4...+pi/4 */
static __forceinline__ double __internal_tan_kerneld(double x, int i)
{
  double x2, z, q;
  x2 = x * x;
  z =                   9.8006287203286300E-006;
  z = __fma_rn (z, x2, -2.4279526494179897E-005);
  z = __fma_rn (z, x2,  4.8644173130937162E-005);
  z = __fma_rn (z, x2, -2.5640012693782273E-005);
  z = __fma_rn (z, x2,  6.7223984330880073E-005);
  z = __fma_rn (z, x2,  8.3559287318211639E-005);
  z = __fma_rn (z, x2,  2.4375039850848564E-004);
  z = __fma_rn (z, x2,  5.8886487754856672E-004);
  z = __fma_rn (z, x2,  1.4560454844672040E-003);
  z = __fma_rn (z, x2,  3.5921008885857180E-003);
  z = __fma_rn (z, x2,  8.8632379218613715E-003);
  z = __fma_rn (z, x2,  2.1869488399337889E-002);
  z = __fma_rn (z, x2,  5.3968253972902704E-002);
  z = __fma_rn (z, x2,  1.3333333333325342E-001);
  z = __fma_rn (z, x2,  3.3333333333333381E-001);
  z = z * x2;
  q = __fma_rn (z, x, x);
  if (i) {
    double s = q - x; 
    double w = __fma_rn (z, x, -s); // tail of q
    z = __internal_fast_rcp (q);
    z = -z;
    s = __fma_rn (q, z, 1.0);
    q = __fma_rn (__fma_rn (z, w, s), z, z);
  }           
  return q;
}

/* approximates exp(a)-1 on [-log(1.5),log(1.5)] accurate to 1 ulp */
static __forceinline__ double __internal_expm1_kernel (double a)
{
  double t;
  t =                 2.08842685477913050E-009;
  t = __fma_rn (t, a, 2.51366409033551950E-008);
  t = __fma_rn (t, a, 2.75574612072447230E-007);
  t = __fma_rn (t, a, 2.75571539284473460E-006);
  t = __fma_rn (t, a, 2.48015869443077950E-005);
  t = __fma_rn (t, a, 1.98412699878799470E-004);
  t = __fma_rn (t, a, 1.38888888892029890E-003);
  t = __fma_rn (t, a, 8.33333333327662860E-003);
  t = __fma_rn (t, a, 4.16666666666656370E-002);
  t = __fma_rn (t, a, 1.66666666666667380E-001);
  t = __fma_rn (t, a, 5.00000000000000000E-001);
  t = t * a;
  t = __fma_rn (t, a, a);
  return t;
}

/* approximate 2*atanh(0.5*a) on [-0.25,0.25] */
static __forceinline__ double __internal_atanh_kernel (double a_1, double a_2)
{
  double a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;
  t =                  7.597322383488143E-002/65536.0;
  t = __fma_rn (t, a2, 6.457518383364042E-002/16384.0);          
  t = __fma_rn (t, a2, 7.705685707267146E-002/4096.0);
  t = __fma_rn (t, a2, 9.090417561104036E-002/1024.0);
  t = __fma_rn (t, a2, 1.111112158368149E-001/256.0);
  t = __fma_rn (t, a2, 1.428571416261528E-001/64.0);
  t = __fma_rn (t, a2, 2.000000000069858E-001/16.0);
  t = __fma_rn (t, a2, 3.333333333333198E-001/4.0);
  t = t * a2;
  t = __fma_rn (t, a, a_2);
  t = t + a_1;
  return t;
}

static __forceinline__ double __internal_fast_log(double a)
{
  int hi, lo;
  double m, u2, t, f, g, u, e;

  /* x = m * 2^e. log(a) = log(m) + e*log(2) */
  hi = __double2hiint(a);
  lo = __double2loint(a);
  e = (double)((hi >> 20) & 0x7fe) - 1022;
  m = __hiloint2double ((hi & 0x801fffff) + 0x3fe00000, lo);
  /* log((1+m)/(1-m)) = 2*atanh(m). log(m) = 2*atanh ((m-1)/(m+1)) */
  f = m - 1.0;
  g = m + 1.0;
  g = __internal_fast_rcp(g);
  u = f * g;
  t = __fma_rn (-2.0, u, f);
  t = __fma_rn (-u, f, t);
  u = __fma_rn (t, g, u);
  /* compute atanh(u) = atanh ((m-1)/(m+1)) */
  u2 = u * u;
  t =                  8.5048800515742276E-002;
  t = __fma_rn (t, u2, 4.1724849126860759E-002);
  t = __fma_rn (t, u2, 6.0524726220470643E-002);
  t = __fma_rn (t, u2, 6.6505606704187079E-002);
  t = __fma_rn (t, u2, 7.6932741976622004E-002);
  t = __fma_rn (t, u2, 9.0908722394788727E-002);
  t = __fma_rn (t, u2, 1.1111111976824838E-001);
  t = __fma_rn (t, u2, 1.4285714274058975E-001);
  t = __fma_rn (t, u2, 2.0000000000077559E-001);
  t = __fma_rn (t, u2, 3.3333333333333154E-001);
  t = t * u2;
  t = __fma_rn (t, u, u);
  t = t + t;
  /* log(a) = log(m) + ln2 * e */
  t = __fma_rn (e, 6.93147180559945290e-001, t);
  return t;
}

static __forceinline__ double __internal_exp2i_kernel(int b)
{
  return __hiloint2double((b + 1023) << 20, 0);
}

static __forceinline__ double __internal_half(double a)
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi - 0x00100000, ilo);
}

static __forceinline__ double __internal_twice(double a)
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi + 0x00100000, ilo);
}

static __forceinline__ double sin(double a)
{
  double z;
  int i;
  if (__isinfd(a) || (a == CUDART_ZERO)) {
    return __dmul_rn(a, CUDART_ZERO);
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

static __forceinline__ double sinpi(double a)
{
#if defined(__CUDANVVM__)
  double z;
#else /* __CUDANVVM__ */
  volatile double z;
#endif /* __CUDANVVM__ */
  long long l;
  int i;
 
  if (__isinfd(a)) {
    a = __dmul_rn(a, CUDART_ZERO); /* return NaN */
  }
  if (a == trunc(a)) {
    a = __longlong_as_double(__double_as_longlong(a)&0x8000000000000000ULL);
  }
  l = __double2ll_rn (a * 2.0f);
  i = (int)l;
  z = (double)l;
  z = __fma_rn (-z, 0.5, a);
  z = __fma_rn (z, CUDART_PI_HI, z * CUDART_PI_LO);
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  if (a == CUDART_ZERO) {
    z = a;
  }
  return z;
}

static __forceinline__ double cospi(double a)
{
#if defined(__CUDANVVM__)
  double z;
#else /* __CUDANVVM__ */
  volatile double z;
#endif /* __CUDANVVM__ */
  long long l;
  int i;
 
  if (fabs(a) > CUDART_TWO_TO_53) {
    a = __dmul_rn (a, CUDART_ZERO);
  }
  l = __double2ll_rn (a * 2.0f);
  i = (int)l;
  z = (double)l;
  z = __fma_rn (-z, 0.5, a);
  z = __fma_rn (z, CUDART_PI_HI, z * CUDART_PI_LO);
  i++;
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  if (z == CUDART_ZERO) {
    z = fabs (z);
  }
  return z;
}

static __forceinline__ double cos(double a)
{
  double z;
  int i;
  if (__isinfd(a)) {
    return CUDART_NAN;
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  i++;
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

/* Compute cos(x + offset) with phase offset computed after argument
   reduction for higher accuracy.  Needed for Bessel approximation 
   functions.
*/
static __forceinline__ double __cos_offset(double a, double offset)
{
  double z;
  int i;

  z = __internal_trig_reduction_kerneld(a, &i);
  a = z + offset + (i & 3) * CUDART_PIO2;
  return cos(a);
}


static __forceinline__ void sincos(double a, double *sptr, double *cptr)
{
#if defined(__CUDANVVM__)
  double s, c;
#else /* __CUDANVVM__ */
  volatile double s, c;
#endif /* __CUDANVVM__ */
  double t;
  int i;
  
  if (__isinfd(a)) {
    a = __dmul_rn (a, CUDART_ZERO); /* return NaN */
  }
  a = __internal_trig_reduction_kerneld(a, &i);
  c = __internal_cos_kerneld(a);
  s = __internal_sin_kerneld(a);
  t = s;
  if (i & 1) {
    s = c;
    c = t;
  }
  if (i & 2) {
    s = -s;
  }
  i++;
  if (i & 2) {
    c = -c;
  }
  if (a == CUDART_ZERO) { /* preserve negative zero */
    s = a;
  }
  *sptr = s;
  *cptr = c;
}

static __forceinline__ double tan(double a)
{
  double z;
  int i;
  if (__isinfd(a)) {
    return __dadd_rn (a, -a); /* return NaN */
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  z = __internal_tan_kerneld(z, i & 1);
  return z;
}

static __forceinline__ double log(double a)
{
  double m, f, g, u, v, tmp, q, ulo, log_lo, log_hi;
  int ihi, ilo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);

  if ((a > CUDART_ZERO) && (a < CUDART_INF)) {
    int e = -1023;
    /* normalize denormals */
    if ((unsigned)ihi < (unsigned)0x00100000) {
      a = a * CUDART_TWO_TO_54;
      e -= 54;
      ihi = __double2hiint(a);
      ilo = __double2loint(a);
    }
    /* a = m * 2^e. m <= sqrt(2): log2(a) = log2(m) + e.
     * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
     */
    e += (ihi >> 20);
    ihi = (ihi & 0x800fffff) | 0x3ff00000;
    m = __hiloint2double (ihi, ilo);
    if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
      m = __internal_half(m);
      e = e + 1;
    }
    /* log((1+m)/(1-m)) = 2*atanh(m). log(m) = 2*atanh ((m-1)/(m+1)) */
    f = m - 1.0;
    g = m + 1.0;
    g = __internal_fast_rcp(g);
    u = f * g;
    u = u + u;  
    /* u = 2.0 * (m - 1.0) / (m + 1.0) */
    v = u * u;
    q =                 6.7261411553826339E-2/65536.0;
    q = __fma_rn (q, v, 6.6133829643643394E-2/16384.0);
    q = __fma_rn (q, v, 7.6940931149150890E-2/4096.0);
    q = __fma_rn (q, v, 9.0908745692137444E-2/1024.0);
    q = __fma_rn (q, v, 1.1111111499059706E-1/256.0);
    q = __fma_rn (q, v, 1.4285714283305975E-1/64.0);
    q = __fma_rn (q, v, 2.0000000000007223E-1/16.0);
    q = __fma_rn (q, v, 3.3333333333333326E-1/4.0);
    tmp = 2.0 * (f - u);
    tmp = __fma_rn (-u, f, tmp); // tmp = remainder of division
    ulo = g * tmp;               // less significant quotient bits
    /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
    q = q * v;
    /* log_hi + log_lo = log(m) to more than double precision */ 
    log_hi = u;
    log_lo = __fma_rn (q, u, ulo);
    /* log_hi + log_lo = log(m)+e*log(2)=log(a) to more than double precision*/
    q   = __fma_rn ( e, CUDART_LN2_HI, log_hi);
    tmp = __fma_rn (-e, CUDART_LN2_HI, q);
    tmp = tmp - log_hi;
    log_hi = q;
    log_lo = log_lo - tmp;
    log_lo = __fma_rn (e, CUDART_LN2_LO, log_lo);
    return log_hi + log_lo;
  } else {
    if (__isnand(a)) {
      return a + a;
    }
    /* log(0) = -INF */
    if (a == 0) {
      return -CUDART_INF;
    }
    /* log(INF) = INF */
    if (a == CUDART_INF) {
      return a;
    }
    /* log(x) is undefined for x < 0.0, return INDEFINITE */
    return CUDART_NAN;
  }
}

/* Requires |x.y| > |y.y|. 8 DP operations */
static __forceinline__ double2 __internal_ddadd_xgty (double2 x, double2 y)
{
  double2 z;
  double r, s, e;
  r = x.y + y.y;
  e = x.y - r;
  s = ((e + y.y) + y.x) + x.x;
  z.y = e = r + s;
  z.x = (r - e) + s;
  return z;
}

/* Take full advantage of FMA. Only 7 DP operations */
static __forceinline__ double2 __internal_ddmul (double2 x, double2 y)
{
  double e;
  double2 t, z;
  t.y = __dmul_rn (x.y, y.y);       /* prevent FMA-merging */
  t.x = __fma_rn (x.y, y.y, -t.y);
  t.x = __fma_rn (x.y, y.x, t.x);
  t.x = __fma_rn (x.x, y.y, t.x);
  z.y = e = t.y + t.x;
  z.x = (t.y - e) + t.x;
  return z;
}

static __forceinline__ double2 __internal_log_ext_prec(double a)
{
  double2 res;
  double2 qq, cc, uu, tt;
  double f, g, u, v, q, ulo, tmp, m;
  int ilo, ihi, expo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  expo = (ihi >> 20) & 0x7ff;
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= CUDART_TWO_TO_54;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }  
  expo -= 1023;
  /* log(a) = log(m*2^expo) = 
     log(m) + log(2)*expo, if m < sqrt(2), 
     log(m*0.5) + log(2)*(expo+1), if m >= sqrt(2)
  */
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  m = __hiloint2double (ihi, ilo);
  if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
    m = __internal_half(m);
    expo = expo + 1;
  }
  /* compute log(m) with extended precision using an algorithm derived from 
   * P.T.P. Tang, "Table Driven Implementation of the Logarithm Function", 
   * TOMS, Vol. 16., No. 4, December 1990, pp. 378-400. A modified polynomial 
   * approximation to atanh(x) on the interval [-0.1716, 0.1716] is utilized.
   */
  f = m - 1.0;
  g = m + 1.0;
  g = __internal_fast_rcp(g);
  u = f * g;
  u = u + u;  
  /* u = 2.0 * (m - 1.0) / (m + 1.0) */
  v = u * u;
  q =                 6.6253631649203309E-2/65536.0;
  q = __fma_rn (q, v, 6.6250935587260612E-2/16384.0);
  q = __fma_rn (q, v, 7.6935437806732829E-2/4096.0);
  q = __fma_rn (q, v, 9.0908878711093280E-2/1024.0);
  q = __fma_rn (q, v, 1.1111111322892790E-1/256.0);
  q = __fma_rn (q, v, 1.4285714284546502E-1/64.0);
  q = __fma_rn (q, v, 2.0000000000003113E-1/16.0);
  q = q * v;
  /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
  tmp = 2.0 * (f - u);
  tmp = __fma_rn (-u, f, tmp); // tmp = remainder of division
  ulo = g * tmp;               // less significand quotient bits
  /* switch to double-double at this point */
  qq.y = q;
  qq.x = 0.0;
  uu.y = u;
  uu.x = ulo;
  cc.y =  3.3333333333333331E-1/4.0;
  cc.x = -9.8201492846582465E-18/4.0;
  qq = __internal_ddadd_xgty (cc, qq);
  /* compute log(m) in double-double format */
  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  uu = __internal_ddadd_xgty (uu, qq);
  u   = uu.y;
  ulo = uu.x;
  /* log(2)*expo in double-double format */
  tt.y = __dmul_rn(expo, 6.9314718055966296e-01); /* multiplication is exact */
  tt.x = __dmul_rn(expo, 2.8235290563031577e-13);
  /* log(a) = log(m) + log(2)*expo;  if expo != 0, |log(2)*expo| > |log(m)| */
  res = __internal_ddadd_xgty (tt, uu);
  return res;
}

static __forceinline__ double log2(double a)
{
  double t;
  t = log(a);
  return __fma_rn (t, CUDART_L2E_HI, t * CUDART_L2E_LO);
}

static __forceinline__ double log10(double a)
{
  double t;
  t = log(a);
  return __fma_rn (t, CUDART_LGE_HI, t * CUDART_LGE_LO);
}

static __forceinline__ double log1p(double a)
{
  double t;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x3fe55555) || ((int)i < (int)0xbfd99999)) {
    /* Compute log2(a+1) = 2*atanh(a/(a+2)) */
    t = a + 2.0;
    t = a / t;
    t = -a * t;
    t = __internal_atanh_kernel(a, t);
    return t;
  }
  return log (a + CUDART_ONE);
}

/* approximate exp() on [sqrt(0.5), sqrt(2)] */
static __forceinline__ double __internal_exp_poly(double a)
{
  double t;

  t =                 2.5052097064908941E-008;
  t = __fma_rn (t, a, 2.7626262793835868E-007);
  t = __fma_rn (t, a, 2.7557414788000726E-006);
  t = __fma_rn (t, a, 2.4801504602132958E-005);
  t = __fma_rn (t, a, 1.9841269707468915E-004);
  t = __fma_rn (t, a, 1.3888888932258898E-003);
  t = __fma_rn (t, a, 8.3333333333978320E-003);
  t = __fma_rn (t, a, 4.1666666666573905E-002);
  t = __fma_rn (t, a, 1.6666666666666563E-001);
  t = __fma_rn (t, a, 5.0000000000000056E-001);
  t = __fma_rn (t, a, 1.0000000000000000E+000);
  t = __fma_rn (t, a, 1.0000000000000000E+000);
  return t;
}

/* compute a * 2^i */
static __forceinline__ double __internal_exp_scale(double a, int i)
{
  double z;
  int j, k;

  k = (i << 20) + (1023 << 20);
  if (abs(i) < 1021) {
    z = __hiloint2double (k, 0);
    z = z * a;
  } else {
    j = 0x40000000;
    if (i < 0) {
      k += (55 << 20);
      j -= (55 << 20);
    }
    k = k - (1 << 20);
    z = __hiloint2double (j, 0); /* 2^-54 if a is denormal, 2.0 otherwise */
    a = a * z;
    z = __hiloint2double (k, 0);
    z = a * z;
  }
  return z;
}

static __forceinline__ double __internal_exp_kernel(double a, int scale)
{ 
  double t, z;
  int i;

  t = rint (a * CUDART_L2E);
  i = (int)t;
  z = __fma_rn (t, -CUDART_LN2_HI, a);
  z = __fma_rn (t, -CUDART_LN2_LO, z);
  t = __internal_exp_poly (z);
  z = __internal_exp_scale (t, i + scale); 
  return z;
}   

static __forceinline__ double __internal_old_exp_kernel(double a, int scale)
{ 
  double t, z;
  int i, j, k;

  t = rint (a * CUDART_L2E);
  i = (int)t;
  z = __fma_rn (t, -CUDART_LN2_HI, a);
  z = __fma_rn (t, -CUDART_LN2_LO, z);
  t = __internal_expm1_kernel (z);
  k = ((i + scale) << 20) + (1023 << 20);
  if (abs(i) < 1021) {
    z = __hiloint2double (k, 0);
    z = __fma_rn (t, z, z);
  } else {
    j = 0x40000000;
    if (i < 0) {
      k += (55 << 20);
      j -= (55 << 20);
    }
    k = k - (1 << 20);
    z = __hiloint2double (j, 0); /* 2^-54 if a is denormal, 2.0 otherwise */
    t = __fma_rn (t, z, z);
    z = __hiloint2double (k, 0);
    z = t * z;
  }
  return z;
}   

static __forceinline__ double exp(double a)
{
  double t;
  int i;
  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40862e43) || ((int)i < (int)0xC0874911)) {
    t = __internal_exp_kernel(a, 0);
    return t;
  }
  t = (i < 0) ? CUDART_ZERO : CUDART_INF;
  if (__isnand(a)) {
    t = a + a;
  }
  return t;
}

static __forceinline__ double exp2(double a)
{
  double t, z;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40900000) || ((int)i < (int)0xc090cc00)) {
    t = rint (a);
    z = a - t;
    i = (int)t;
    /* 2^z = exp(log(2)*z) */
    z = __fma_rn (z, CUDART_LN2_HI, z * CUDART_LN2_LO);
    t = __internal_exp_poly (z);
    z = __internal_exp_scale (t, i); 
    return z;
  } 
  t = (i < 0) ? CUDART_ZERO : CUDART_INF;
  if (__isnand(a)) {
    t = a + a;
  }
  return t;
}

static __forceinline__ double exp10(double a)
{
  double z;
  double t;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40734414) || ((int)i < (int)0xc07439b8)) {
    t = rint (a * CUDART_L2T);
    i = (int)t;
    z = __fma_rn (t, -CUDART_LG2_HI, a);
    z = __fma_rn (t, -CUDART_LG2_LO, z);
    /* 2^z = exp(log(10)*z) */
    z = __fma_rn (z, CUDART_LNT_HI, z * CUDART_LNT_LO);
    t = __internal_exp_poly (z);
    z = __internal_exp_scale (t, i); 
    return z;
  } 
  t = (i < 0) ? CUDART_ZERO : CUDART_INF;
  if (__isnand(a)) {
    t = a + a;
  }
  return t;
}

static __forceinline__ double __internal_expm1_scaled(double a, int scale)
{ 
  double t, z, u;
  int i, j, k;
  k = __double2hiint(a);
  t = rint (a * CUDART_L2E);
  i = (int)t + scale;
  z = __fma_rn (t, -CUDART_LN2_HI, a);
  z = __fma_rn (t, -CUDART_LN2_LO, z);
  k = k + k;
  if ((unsigned)k < (unsigned)0x7fb3e647) {
    z = a;
    i = 0;
  }
  t = __internal_expm1_kernel(z);
  j = i;
  if (i == 1024) j--;
  u = __internal_exp2i_kernel(j);
  a = __hiloint2double(0x3ff00000 + (scale << 20), 0);
  a = u - a;
  t = __fma_rn (t, u, a);
  if (i == 1024) t = t + t;
  if (k == 0) t = z;              /* preserve -0 */
  return t;
}   

static __forceinline__ double expm1(double a)
{
  double t;
  int k;

  k = __double2hiint(a);
  if (((unsigned)k < (unsigned)0x40862e43) || ((int)k < (int)0xc04a8000)) {
    return __internal_expm1_scaled(a, 0);
  }
  t = (k < 0) ? -CUDART_ONE : CUDART_INF;
  if (__isnand(a)) {
    t = a + a;
  }
  return t;
}

static __forceinline__ double cosh(double a)
{
  double t, z;
  int i;

  z = fabs(a);
  i = __double2hiint(z);
  if ((unsigned)i < (unsigned)0x408633cf) {
    z = __internal_exp_kernel(z, -2);
    t = __internal_fast_rcp (z);
    z = __fma_rn(2.0, z, 0.125 * t);
    return z;
  } else {
    if (z > 0.0) a = CUDART_INF_F;
    return a + a;
  }
}

static __forceinline__ double sinh(double a)
{
  double s, z;
  s = a;
  a = fabs(a);
  if (__double2hiint(a) < 0x3ff00000) { /* risk of catastrophic cancellation */
    double a2 = a * a;
    /* approximate sinh(x) on [0,1] with a polynomial */
    z =                  1.632386098183803E-010;
    z = __fma_rn (z, a2, 2.504854501385687E-008);
    z = __fma_rn (z, a2, 2.755734274788706E-006);
    z = __fma_rn (z, a2, 1.984126976294102E-004);
    z = __fma_rn (z, a2, 8.333333333452911E-003);
    z = __fma_rn (z, a2, 1.666666666666606E-001);
    z = z * a2;
    z = __fma_rn (z, a, a);
  } else {
    z = __internal_expm1_scaled (a, -1);
    z = z + z / __fma_rn (2.0, z, 1.0);
    if (a >= CUDART_LN2_X_1025) {
      z = CUDART_INF;     /* overflow -> infinity */
    }
  } 
  z = __internal_copysign_pos(z, s);
  return z;
}

static __forceinline__ double tanh(double a)
{
  double t;
  t = fabs(a);
  if (t >= 0.55) {
    double s;
    s = __internal_fast_rcp (__internal_old_exp_kernel (2.0 * t, 0) + 1.0);
    s = __fma_rn (2.0, -s, 1.0);
    if (t > 350.0) {
      s = 1.0;       /* overflow -> 1.0 */
    }
    a = __internal_copysign_pos(s, a);
  } else {
    double a2;
    a2 = a * a;
    t =                   5.102147717274194E-005;
    t = __fma_rn (t, a2, -2.103023983278533E-004);
    t = __fma_rn (t, a2,  5.791370145050539E-004);
    t = __fma_rn (t, a2, -1.453216755611004E-003);
    t = __fma_rn (t, a2,  3.591719696944118E-003);
    t = __fma_rn (t, a2, -8.863194503940334E-003);
    t = __fma_rn (t, a2,  2.186948597477980E-002);
    t = __fma_rn (t, a2, -5.396825387607743E-002);
    t = __fma_rn (t, a2,  1.333333333316870E-001);
    t = __fma_rn (t, a2, -3.333333333333232E-001);
    t = t * a2;
    t = __fma_rn (t, a, a);
    a = __internal_copysign_pos(t, a);
  }
  return a;
}

static __forceinline__ double __internal_atan_kernel(double a)
{
  double t, a2;
  a2 = a * a;
  t =                  -2.0258553044438358E-005 ;
  t = __fma_rn (t, a2,  2.2302240345758510E-004);
  t = __fma_rn (t, a2, -1.1640717779930576E-003);
  t = __fma_rn (t, a2,  3.8559749383629918E-003);
  t = __fma_rn (t, a2, -9.1845592187165485E-003);
  t = __fma_rn (t, a2,  1.6978035834597331E-002);
  t = __fma_rn (t, a2, -2.5826796814495994E-002);
  t = __fma_rn (t, a2,  3.4067811082715123E-002);
  t = __fma_rn (t, a2, -4.0926382420509971E-002);
  t = __fma_rn (t, a2,  4.6739496199157994E-002);
  t = __fma_rn (t, a2, -5.2392330054601317E-002);
  t = __fma_rn (t, a2,  5.8773077721790849E-002);
  t = __fma_rn (t, a2, -6.6658603633512573E-002);
  t = __fma_rn (t, a2,  7.6922129305867837E-002);
  t = __fma_rn (t, a2, -9.0909012354005225E-002);
  t = __fma_rn (t, a2,  1.1111110678749424E-001);
  t = __fma_rn (t, a2, -1.4285714271334815E-001);
  t = __fma_rn (t, a2,  1.9999999999755019E-001);
  t = __fma_rn (t, a2, -3.3333333333331860E-001);
  t = t * a2;
  t = __fma_rn (t, a, a);
  return t;
}

static __forceinline__ double atan2(double a, double b)
{
  double t0, t1, t3;
  if (__isnand(a) || __isnand(b)) {
    return a + b;
  }
  t3 = fabs(b);
  t1 = fabs(a);
  if (t3 == 0.0 && t1 == 0.0) {
    t3 = (__double2hiint(b) < 0) ? CUDART_PI : 0;
  } else if (__isinfd(t3) && __isinfd(t1)) {
    t3 = (__double2hiint(b) < 0) ? CUDART_3PIO4 : CUDART_PIO4;
  } else {
    t0 = fmax (t1, t3);
    t1 = fmin (t1, t3);
    t3 = t1 / t0;
    t3 = __internal_atan_kernel(t3);
    /* Map result according to octant. */
    if (fabs(a) > fabs(b)) t3 = CUDART_PIO2 - t3;
    if (b < 0.0)           t3 = CUDART_PI - t3;
  }
  t3 = __internal_copysign_pos(t3, a);
  return t3;
}

static __forceinline__ double atan(double a)
{
  double t0, t1;
  /* reduce argument to first octant */
  t0 = fabs(a);
  t1 = t0;
  if (t0 > 1.0) {
    t1 = __internal_fast_rcp (t1);
    if (t0 == CUDART_INF) t1 = 0.0;
  }
  /* approximate atan(r) in first octant */
  t1 = __internal_atan_kernel(t1);
  /* map result according to octant. */
  if (t0 > 1.0) {
    t1 = CUDART_PIO2 - t1;
  }
  return __internal_copysign_pos(t1, a);
}

/* b should be the square of a */
static __forceinline__ double __internal_asin_kernel(double a, double b)
{
  double r;
  r =                  6.259798167646803E-002;
  r = __fma_rn (r, b, -7.620591484676952E-002);
  r = __fma_rn (r, b,  6.686894879337643E-002);
  r = __fma_rn (r, b, -1.787828218369301E-002); 
  r = __fma_rn (r, b,  1.745227928732326E-002);
  r = __fma_rn (r, b,  1.000422754245580E-002);
  r = __fma_rn (r, b,  1.418108777515123E-002);
  r = __fma_rn (r, b,  1.733194598980628E-002);
  r = __fma_rn (r, b,  2.237350511593569E-002);
  r = __fma_rn (r, b,  3.038188875134962E-002);
  r = __fma_rn (r, b,  4.464285849810986E-002);
  r = __fma_rn (r, b,  7.499999998342270E-002);
  r = __fma_rn (r, b,  1.666666666667375E-001);
  r = r * b;
  return r;
}

static __forceinline__ double asin(double a)
{
  double fa, t0, t1;
  int ihi, ahi;
  ahi = __double2hiint(a);
  fa  = fabs(a);
  ihi = __double2hiint(fa);
  if (ihi < 0x3fe26666) {
    t1 = fa * fa;
    t1 = __internal_asin_kernel (fa, t1);
    t1 = __fma_rn (t1, fa, fa);
    t1 = __internal_copysign_pos(t1, a);
  } else {
    t1 = __fma_rn (-0.5, fa, 0.5);
    t0 = sqrt (t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = -2.0 * t0;
    t1 = __fma_rn (t0, t1, CUDART_PIO2_LO);
    t0 = t0 + CUDART_PIO4_HI;
    t1 = t0 + t1;
    t1 = t1 + CUDART_PIO4_HI;
    if (ahi < 0x3ff00000) {
      t1 = __internal_copysign_pos(t1, a);
    }
  }
  return t1;
}

static __forceinline__ double acos(double a)
{
  double t0, t1;
  int ihi, ahi;

  ahi = __double2hiint(a);
  t0 = fabs (a);
  ihi = __double2hiint(t0);
  if (ihi < 0x3fe26666) {  
    t1 = t0 * t0;
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    if (ahi < 0) {
      t0 = __dadd_rn (t0, +CUDART_PIO2_LO);
      t0 = __dadd_rn (CUDART_PIO2_HI, +t0);
    } else {
      t0 = __dadd_rn (t0, -CUDART_PIO2_LO);
      t0 = __dadd_rn (CUDART_PIO2_HI, -t0);
    }
  } else {
    t1 = __fma_rn (-0.5, t0, 0.5);
    t0 = sqrt (t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    t0 = 2.0 * t0;
    if (ahi < 0) {    
      t0 = __dadd_rn (t0, -CUDART_PI_LO);
      t0 = __dadd_rn (CUDART_PI_HI, -t0);
    }
  } 
  return t0;
}

static __forceinline__ double acosh(double a)
{
  double t;
  t = a - 1.0;
  if (fabs(t) > CUDART_TWO_TO_52) {
    /* for large a, acosh = log(2*a) */
    return CUDART_LN2 + log(a);
  } else {
    t = t + sqrt(__fma_rn(a, t, t));
    return log1p(t);
  }  
}

static __forceinline__ double asinh(double a)
{
  double fa, t;
  fa = fabs(a);
  if (__double2hiint(fa) >= 0x5ff00000) { /* prevent intermediate underflow */
    t = CUDART_LN2 + log(fa);
  } else {
    t = fa * fa;
    t = log1p (fa + t / (1.0 + sqrt(1.0 + t)));
  }
  return __internal_copysign_pos(t, a);  
}

static __forceinline__ double atanh(double a)
{
  double fa, t;
  fa = fabs(a);
  t = (2.0 * fa) / (1.0 - fa);
  t = 0.5 * log1p(t);
  if (__double2hiint(a) < 0) {
    t = -t;
  }
  return t;
}

static __forceinline__ double hypot(double a, double b)
{
  double v, w, t, fa, fb;

  fa = fabs(a);
  fb = fabs(b);
  v = fmax(fa, fb);
  w = fmin(fa, fb);
  t = w / v;
  t = __fma_rn (t, t, 1.0);
  t = v * sqrt(t);
  if (v == 0.0) {
    t = v + w;         /* fixup for zero divide */
  }
  if ((!(fa <= CUDART_INF)) || (!(fb <= CUDART_INF))) {
    t = a + b;         /* fixup for NaNs */
  }
  if (v == CUDART_INF) {
    t = v + w;         /* fixup for infinities */
  }
  return t;
}

static __forceinline__ double cbrt(double a)
{
  float s;
  double t, r;
  int ilo, ihi, expo, nexpo, denorm;
  if ((a == 0.0) || !(__isfinited(a))) {
    return a + a;
  } 
  t = fabs(a);
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  denorm = 0;
  if (expo == 0) {
    /* denormal */
    t = t * CUDART_TWO_TO_54;
    denorm = 18;
    ilo = __double2loint(t);
    ihi = __double2hiint(t);
    expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  }
  /* scale into float range */
  nexpo = __float2int_rn(CUDART_THIRD_F * (float)(expo - 1022));
  ihi -= (3 * nexpo) << 20;
  r = __hiloint2double(ihi, ilo);
  /* initial approximation */
  s = (float)r;
  t = exp2f(-CUDART_THIRD_F * __log2f(s));    /* approximate invcbrt */
  t = __fma_rn(__fma_rn(t*t,-r*t,1.0), CUDART_THIRD*t, t);/* refine invcbrt */
  t = r * t * t;                                        /* approximate cbrt */
  t = __fma_rn(t - (r / (t * t)), -CUDART_THIRD, t);         /* refine cbrt */
  /* scale result back into double range */
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  ihi += (nexpo - denorm) << 20;
  t = __hiloint2double(ihi, ilo);
  if (__double2hiint(a) < 0) {
    t = -t;
  }
  return t;
}

static __forceinline__ double rcbrt(double a)
{
  float s;
  double t, r;
  int ilo, ihi, expo, nexpo, denorm;
  if ((a == 0.0) || !(__isfinited(a))) {
    return 1.0 / a;
  } 
  t = fabs(a);
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  denorm = 0;
  if (expo == 0) {
    /* denormal */
    t = t * CUDART_TWO_TO_54;
    denorm = 18;
    ilo = __double2loint(t);
    ihi = __double2hiint(t);
    expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  }
  /* scale into float range */
  nexpo = __float2int_rn(CUDART_THIRD_F * (float)(expo - 1022));
  ihi -= (3 * nexpo) << 20;
  r = __hiloint2double(ihi, ilo);
  /* initial approximation */
  s = (float)r;
  t = exp2f(-CUDART_THIRD_F * __log2f(s));    /* approximate invcbrt */
  t = __fma_rn(__fma_rn(t*t,-r*t,1.0), CUDART_THIRD*t, t);/* refine invcbrt */
  t = __fma_rn(__fma_rn(t*t,-r*t,1.0), CUDART_THIRD*t, t);/* refine invcbrt */
  /* scale result back into double range */
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  ihi += (-(nexpo - denorm)) << 20;
  t = __hiloint2double(ihi, ilo);
  if (__double2hiint(a) < 0) {
    t = -t;
  }
  return t;
}

static __forceinline__ double __internal_accurate_pow(double a, double b)
{
  double2 loga;
  double2 prod;
  double t_hi, t_lo;
  double tmp;
  double e;

  /* compute log(a) in double-double format*/
  loga = __internal_log_ext_prec(a);

  /* prevent overflow during extended precision multiply */
  if (fabs(b) > 1e304) b *= 1.220703125e-4;
  /* compute b * log(a) in double-double format */
  t_hi = __dmul_rn (loga.y, b);   /* prevent FMA-merging */
  t_lo = __fma_rn (loga.y, b, -t_hi);
  t_lo = __fma_rn (loga.x, b, t_lo);
  prod.y = e = t_hi + t_lo;
  prod.x = (t_hi - e) + t_lo;

  /* compute pow(a,b) = exp(b*log(a)) */
  tmp = exp(prod.y);
  /* prevent -INF + INF = NaN */
  if (!__isinfd(tmp)) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    tmp = __fma_rn (tmp, prod.x, tmp);
  }
  return tmp;
}

static __forceinline__ double pow(double a, double b)
{
  int bIsOddInteger;
  double t;

  if (a == 1.0 || b == 0.0) {
    return 1.0;
  } 
  if (__isnand(a) || __isnand(b)) {
    return a + b;
  }
  if (a == CUDART_INF) {
    return (__double2hiint(b) < 0) ?  CUDART_ZERO : CUDART_INF;
  }
  if (__isinfd(b)) {
    if (a == -1.0) {
      return 1.0;
    }
    t = fabs(a) > 1.0 ? CUDART_INF : CUDART_ZERO;
    if (b < CUDART_ZERO) {
      t = 1.0 / t;
    }
    return t;
  }
  bIsOddInteger = fabs(b - (2.0f * trunc(0.5 * b))) == 1.0;
  if (a == CUDART_ZERO) {
    t = bIsOddInteger ? a : CUDART_ZERO;
    if (b < CUDART_ZERO) {
      t = 1.0 / t;
    }
    return t;
  } 
  if (a == -CUDART_INF) {
    t = (b < CUDART_ZERO) ? -1.0/a : -a;
    if (bIsOddInteger) {
      t = __longlong_as_double(__double_as_longlong(t)^0x8000000000000000ULL);
    }
    return t;
  } 
  if ((a < CUDART_ZERO) && (b != trunc(b))) {
    return CUDART_NAN;
  } 
  t = fabs(a);
  t = __internal_accurate_pow(t, b);
  if ((a < CUDART_ZERO) && bIsOddInteger) {
    t = __longlong_as_double(__double_as_longlong(t) ^ 0x8000000000000000ULL); 
  }
  return t;
}

static __forceinline__ double j0(double a)
{
  double t, r, x;
  r = 0.0;
  t = fabs(a);
  if (t <= 3.962451833991041709e0) {
    x = ((t - 2.404825557695772886e0) - -1.176691651530894036e-16);
    r = -4.668055296522885552e-16;
    r = __fma_rn(r, x, -6.433200527258554127e-15);
    r = __fma_rn(r, x, 1.125154785441239563e-13);
    r = __fma_rn(r, x, 1.639521934089839047e-12);
    r = __fma_rn(r, x, -2.534199601670673987e-11);
    r = __fma_rn(r, x, -3.166660834754117150e-10);
    r = __fma_rn(r, x, 4.326570922239416813e-9);
    r = __fma_rn(r, x, 4.470057037570427580e-8);
    r = __fma_rn(r, x, -5.304914441394479122e-7);
    r = __fma_rn(r, x, -4.338826303234108986e-6);
    r = __fma_rn(r, x, 4.372919273219640746e-5);
    r = __fma_rn(r, x, 2.643770367619977359e-4);
    r = __fma_rn(r, x, -2.194200359017061189e-3);
    r = __fma_rn(r, x, -8.657669593307546971e-3);
    r = __fma_rn(r, x, 5.660177443794636720e-2);
    r = __fma_rn(r, x, 1.079387017549203048e-1);
    r = __fma_rn(r, x, -5.191474972894667417e-1);
    r *= x;
  } else if (t <= 7.086903011598661433e0) {
    x = ((t - 5.520078110286310569e0) - 8.088597146146722332e-17);
    r = 3.981548125960367572e-16;
    r = __fma_rn(r, x, 5.384425646000319613e-15);
    r = __fma_rn(r, x, -1.208169028319422770e-13);
    r = __fma_rn(r, x, -1.379791615846302261e-12);
    r = __fma_rn(r, x, 2.745222536512400531e-11);
    r = __fma_rn(r, x, 2.592191169087820231e-10);
    r = __fma_rn(r, x, -4.683395694900245463e-9);
    r = __fma_rn(r, x, -3.511535752914609294e-8);
    r = __fma_rn(r, x, 5.716490702257101151e-7);
    r = __fma_rn(r, x, 3.199786905053059080e-6);
    r = __fma_rn(r, x, -4.652109073941537520e-5);
    r = __fma_rn(r, x, -1.751857289934499263e-4);
    r = __fma_rn(r, x, 2.257440229032805189e-3);
    r = __fma_rn(r, x, 4.631042145907517116e-3);
    r = __fma_rn(r, x, -5.298855286760461442e-2);
    r = __fma_rn(r, x, -3.082065142559364118e-2);
    r = __fma_rn(r, x, 3.402648065583681602e-1);
    r *= x;
  } else if (t <= 1.022263117596264692e1) {
    x = ((t - 8.653727912911012510e0) - -2.928126073207789799e-16);
    r = -4.124304662099804879e-16;
    r = __fma_rn(r, x, -4.596716020545263225e-15);
    r = __fma_rn(r, x, 1.243104269818899322e-13);
    r = __fma_rn(r, x, 1.149516171925282771e-12);
    r = __fma_rn(r, x, -2.806255120718408997e-11);
    r = __fma_rn(r, x, -2.086671689271728758e-10);
    r = __fma_rn(r, x, 4.736806709085623724e-9);
    r = __fma_rn(r, x, 2.694156819104033891e-8);
    r = __fma_rn(r, x, -5.679379510457043302e-7);
    r = __fma_rn(r, x, -2.288391007218622664e-6);
    r = __fma_rn(r, x, 4.482303544494819955e-5);
    r = __fma_rn(r, x, 1.124348678929902644e-4);
    r = __fma_rn(r, x, -2.060335155125843105e-3);
    r = __fma_rn(r, x, -2.509302227210569083e-3);
    r = __fma_rn(r, x, 4.403377496341183417e-2);
    r = __fma_rn(r, x, 1.568412496095387618e-2);
    r = __fma_rn(r, x, -2.714522999283819349e-1);
    r *= x;
  } else if (!__isinfd(t)) {
    double y = __internal_fast_rcp(t);
    double y2 = y * y;
    double f, arg;
    f = -1.749518042413318545e4;
    f = __fma_rn(f, y2, 1.609818826277744392e3);
    f = __fma_rn(f, y2, -9.327297929346906358e1);
    f = __fma_rn(f, y2, 5.754657357710742716e0);
    f = __fma_rn(f, y2, -5.424139391385890407e-1);
    f = __fma_rn(f, y2, 1.035143619926359032e-1);
    f = __fma_rn(f, y2, -6.249999788858900951e-2);
    f = __fma_rn(f, y2, 9.999999999984622301e-1);
    arg = -2.885116220349355482e6;
    arg = __fma_rn(arg, y2, 2.523286424277686747e5);
    arg = __fma_rn(arg, y2, -1.210196952664123455e4);
    arg = __fma_rn(arg, y2, 4.916296687065029687e2);
    arg = __fma_rn(arg, y2, -2.323271029624128303e1);
    arg = __fma_rn(arg, y2, 1.637144946408570334e0);
    arg = __fma_rn(arg, y2, -2.095680312729443495e-1);
    arg = __fma_rn(arg, y2, 6.510416335987831427e-2);
    arg = __fma_rn(arg, y2, -1.249999999978858578e-1);
    arg = __fma_rn(arg, y, t);
    r = rsqrt(t) * CUDART_SQRT_2OPI * f * __cos_offset(arg, -7.8539816339744831e-1);
  } else {
  /* Input is infinite. */
    r = 0.0;
  }
  return r;
}

static __forceinline__ double j1(double a)
{
  double t, r, x;
  r = 0.0;
  t = fabs(a);
  if (t <= 2.415852985103756012e0) {
    x = ((t - 0.000000000000000000e-1) - 0.000000000000000000e-1);
    r = 8.018399195792647872e-15;
    r = __fma_rn(r, x, -2.118695440834766210e-13);
    r = __fma_rn(r, x, 2.986477477755093929e-13);
    r = __fma_rn(r, x, 3.264658690505054749e-11);
    r = __fma_rn(r, x, 2.365918244990000764e-12);
    r = __fma_rn(r, x, -5.655535980321211576e-9);
    r = __fma_rn(r, x, 5.337726421910612559e-12);
    r = __fma_rn(r, x, 6.781633105423295953e-7);
    r = __fma_rn(r, x, 3.551463066921223471e-12);
    r = __fma_rn(r, x, -5.425347399642436942e-5);
    r = __fma_rn(r, x, 6.141520947159623346e-13);
    r = __fma_rn(r, x, 2.604166666526797937e-3);
    r = __fma_rn(r, x, 1.929721653824376829e-14);
    r = __fma_rn(r, x, -6.250000000000140166e-2);
    r = __fma_rn(r, x, 4.018089105880317857e-17);
    r = __fma_rn(r, x, 5.000000000000000000e-1);
    r *= x;
  } else if (t <= 5.423646320011565535e0) {
    x = ((t - 3.831705970207512468e0) - -1.526918409008806686e-16);
    r = -5.512780891825248469e-15;
    r = __fma_rn(r, x, 1.208228522598007249e-13);
    r = __fma_rn(r, x, 1.250828223475420523e-12);
    r = __fma_rn(r, x, -2.797792344085172005e-11);
    r = __fma_rn(r, x, -2.362345221426392649e-10);
    r = __fma_rn(r, x, 4.735362223346154893e-9);
    r = __fma_rn(r, x, 3.248288715654640665e-8);
    r = __fma_rn(r, x, -5.727805561466869718e-7);
    r = __fma_rn(r, x, -3.036863401211637746e-6);
    r = __fma_rn(r, x, 4.620870128840665444e-5);
    r = __fma_rn(r, x, 1.746642907294104828e-4);
    r = __fma_rn(r, x, -2.233125339145115504e-3);
    r = __fma_rn(r, x, -5.179719245640395341e-3);
    r = __fma_rn(r, x, 5.341044413272456881e-2);
    r = __fma_rn(r, x, 5.255614585697734181e-2);
    r = __fma_rn(r, x, -4.027593957025529803e-1);
    r *= x;
  } else if (t <= 8.594527402439170415e0) {
    x = ((t - 7.015586669815618848e0) - -9.414165653410388908e-17);
    r = 4.423133061281035160e-15;
    r = __fma_rn(r, x, -1.201320120922480112e-13);
    r = __fma_rn(r, x, -1.120851060072903875e-12);
    r = __fma_rn(r, x, 2.798783538427610697e-11);
    r = __fma_rn(r, x, 2.065329706440647244e-10);
    r = __fma_rn(r, x, -4.720444222309518119e-9);
    r = __fma_rn(r, x, -2.727342515669842039e-8);
    r = __fma_rn(r, x, 5.665269543584226731e-7);
    r = __fma_rn(r, x, 2.401580794492155375e-6);
    r = __fma_rn(r, x, -4.499147527210508836e-5);
    r = __fma_rn(r, x, -1.255079095508101735e-4);
    r = __fma_rn(r, x, 2.105587143238240189e-3);
    r = __fma_rn(r, x, 3.130291726048001991e-3);
    r = __fma_rn(r, x, -4.697047894974023391e-2);
    r = __fma_rn(r, x, -2.138921280934158106e-2);
    r = __fma_rn(r, x, 3.001157525261325398e-1);
    r *= x;
  } else if (!__isinfd(t)) {
    double y = __internal_fast_rcp(t);
    double y2 = y * y;
    double f, arg;
    f = 1.485383005325836814e4;
    f = __fma_rn(f, y2, -1.648096811830575007e3);
    f = __fma_rn(f, y2, 1.101438783774615899e2);
    f = __fma_rn(f, y2, -7.551889723469123794e0);
    f = __fma_rn(f, y2, 8.042591538676234775e-1);
    f = __fma_rn(f, y2, -1.933557706160460576e-1);
    f = __fma_rn(f, y2, 1.874999929278536315e-1);
    f = __fma_rn(f, y2, 1.000000000005957013e0);
    arg = -6.214794014836998139e7;
    arg = __fma_rn(arg, y2, 6.865585630355566740e6);
    arg = __fma_rn(arg, y2, -3.832405418387809768e5);
    arg = __fma_rn(arg, y2, 1.571235974698157042e4);
    arg = __fma_rn(arg, y2, -6.181902458868638632e2);
    arg = __fma_rn(arg, y2, 3.039697998508859911e1);
    arg = __fma_rn(arg, y2, -2.368515193214345782e0);
    arg = __fma_rn(arg, y2, 3.708961732933458433e-1);
    arg = __fma_rn(arg, y2, -1.640624965735098806e-1);
    arg = __fma_rn(arg, y2, 3.749999999976813547e-1);
    arg = __fma_rn(arg, y, t);
    r = rsqrt(t) * CUDART_SQRT_2OPI * f * __cos_offset(arg, -2.3561944901923449e0); 
  } else {
  /* Input is infinite. */
    r = 0.0;
  }
  if (a < 0.0) {
    r = -r;
  }
  if (t < 1e-30) {
    r = a * 0.5;
  }
  return r;
}

static __forceinline__ double y0(double a)
{
  double t, r, x;
  r = 0.0;
  t = fabs(a);
  if (t <= 7.967884831395837253e-1) {
    x = t * t;
    r = 5.374806887266719984e-17;
    r = __fma_rn(r, x, -1.690851667879507126e-14);
    r = __fma_rn(r, x, 4.136256698488524230e-12);
    r = __fma_rn(r, x, -7.675202391864751950e-10);
    r = __fma_rn(r, x, 1.032530269160133847e-7);
    r = __fma_rn(r, x, -9.450377743948014966e-6);
    r = __fma_rn(r, x, 5.345180760328465709e-4);
    r = __fma_rn(r, x, -1.584294153256949819e-2);
    r = __fma_rn(r, x, 1.707584669151278045e-1);
    r *= (x - 4.322145581245422363e-1) - -1.259433890510308629e-9;
    r += CUDART_2_OVER_PI * log(t) * j0(t);
  } else if (t <= 2.025627692797012713e0) {
    x = ((t - 8.935769662791674950e-1) - 2.659623153972038487e-17);
    r = -3.316256912072560202e-5;
    r = __fma_rn(r, x, 4.428203736344834521e-4);
    r = __fma_rn(r, x, -2.789856306341642004e-3);
    r = __fma_rn(r, x, 1.105846367024121250e-2);
    r = __fma_rn(r, x, -3.107223394960596102e-2);
    r = __fma_rn(r, x, 6.626287772780777019e-2);
    r = __fma_rn(r, x, -1.125221809100773462e-1);
    r = __fma_rn(r, x, 1.584073414576677719e-1);
    r = __fma_rn(r, x, -1.922273494240156200e-1);
    r = __fma_rn(r, x, 2.093393446684197468e-1);
    r = __fma_rn(r, x, -2.129333765401472400e-1);
    r = __fma_rn(r, x, 2.093702358334368907e-1);
    r = __fma_rn(r, x, -2.037455528835861451e-1);
    r = __fma_rn(r, x, 1.986558106005199553e-1);
    r = __fma_rn(r, x, -1.950678188917356060e-1);
    r = __fma_rn(r, x, 1.933768292594399973e-1);
    r = __fma_rn(r, x, -1.939501240454329922e-1);
    r = __fma_rn(r, x, 1.973356651370720138e-1);
    r = __fma_rn(r, x, -2.048771973714162697e-1);
    r = __fma_rn(r, x, 2.189484270119261000e-1);
    r = __fma_rn(r, x, -2.261217135462367245e-1);
    r = __fma_rn(r, x, 2.205528284817022400e-1);
    r = __fma_rn(r, x, -4.920789342629753871e-1);
    r = __fma_rn(r, x, 8.794208024971947868e-1);
    r *= x;
  } else if (t <= 5.521864739808315283e0) {
    x = ((t - 3.957678419314857976e0) - -1.076434069756270603e-16);
    r = -1.494114173821677059e-15;
    r = __fma_rn(r, x, -1.013791156119442377e-15);
    r = __fma_rn(r, x, 1.577311216240784649e-14);
    r = __fma_rn(r, x, 3.461700831703228390e-14);
    r = __fma_rn(r, x, -1.390049111128330285e-13);
    r = __fma_rn(r, x, -2.651585913591809710e-14);
    r = __fma_rn(r, x, -2.563422432591884445e-13);
    r = __fma_rn(r, x, 3.152125074327968061e-12);
    r = __fma_rn(r, x, -1.135177389965644664e-11);
    r = __fma_rn(r, x, 4.326378313976470202e-11);
    r = __fma_rn(r, x, -1.850879474448778845e-10);
    r = __fma_rn(r, x, 7.689088938316559034e-10);
    r = __fma_rn(r, x, -3.657694558233732877e-9);
    r = __fma_rn(r, x, 1.892629263079880039e-8);
    r = __fma_rn(r, x, -2.185282420222553349e-8);
    r = __fma_rn(r, x, -2.934871156586473999e-7);
    r = __fma_rn(r, x, -4.893369556967850888e-6);
    r = __fma_rn(r, x, 5.092291346093084947e-5);
    r = __fma_rn(r, x, 1.952694025023884918e-4);
    r = __fma_rn(r, x, -2.183518873989655565e-3);
    r = __fma_rn(r, x, -6.852566677116652717e-3);
    r = __fma_rn(r, x, 5.852382210516620525e-2);
    r = __fma_rn(r, x, 5.085590959215843115e-2);
    r = __fma_rn(r, x, -4.025426717750241745e-1);
    r *= x;
  } else if (t <= 8.654198051899094858e0) {
    x = ((t - 7.086051060301772786e0) - -8.835285723085408128e-17);
    r = 3.951031695740590034e-15;
    r = __fma_rn(r, x, -1.110810503294961990e-13);
    r = __fma_rn(r, x, -1.310829469053465703e-12);
    r = __fma_rn(r, x, 2.824621267525193929e-11);
    r = __fma_rn(r, x, 2.302923649674420956e-10);
    r = __fma_rn(r, x, -4.717174021172401832e-9);
    r = __fma_rn(r, x, -3.098470041689314533e-8);
    r = __fma_rn(r, x, 5.749349008560620678e-7);
    r = __fma_rn(r, x, 2.701363791846417715e-6);
    r = __fma_rn(r, x, -4.595140667075523833e-5);
    r = __fma_rn(r, x, -1.406025977407872123e-4);
    r = __fma_rn(r, x, 2.175984016431612746e-3);
    r = __fma_rn(r, x, 3.318348268895694383e-3);
    r = __fma_rn(r, x, -4.802407007625847379e-2);
    r = __fma_rn(r, x, -2.117523655676954025e-2);
    r = __fma_rn(r, x, 3.000976149104751523e-1);
    r *= x;
  } else if (!__isinfd(t)) {
    double y = __internal_fast_rcp(t);
    double y2 = y * y;
    double f, arg;
    f = -1.121823763318965797e4;
    f = __fma_rn(f, y2, 1.277353533221286625e3);
    f = __fma_rn(f, y2, -8.579416608392857313e1);
    f = __fma_rn(f, y2, 5.662125060937317933e0);
    f = __fma_rn(f, y2, -5.417345171533867187e-1);
    f = __fma_rn(f, y2, 1.035114040728313117e-1);
    f = __fma_rn(f, y2, -6.249999082419847168e-2);
    f = __fma_rn(f, y2, 9.999999999913266047e-1);
    arg = 5.562900148486682495e7;
    arg = __fma_rn(arg, y2, -6.039326416769045405e6);
    arg = __fma_rn(arg, y2, 3.303804467797655961e5);
    arg = __fma_rn(arg, y2, -1.320780106166394580e4);
    arg = __fma_rn(arg, y2, 5.015151566589033791e2);
    arg = __fma_rn(arg, y2, -2.329056718317451669e1);
    arg = __fma_rn(arg, y2, 1.637366947135598716e0);
    arg = __fma_rn(arg, y2, -2.095685710525915790e-1);
    arg = __fma_rn(arg, y2, 6.510416411708590256e-2);
    arg = __fma_rn(arg, y2, -1.249999999983588544e-1);
    arg = __fma_rn(arg, y, t);
    r = rsqrt(t) * CUDART_SQRT_2OPI * f * __cos_offset(arg, -2.356194490192344929e0);
  } else {
    /* Input is infinite. */
    r = 0.0;
  }
  if (a < 0.0) {
    r = CUDART_NAN;
  }
  return r;
}

static __forceinline__ double y1(double a)
{
  double t, r, x;
  r = 0.0;
  t = fabs(a);
  if (t < 1e-308) {
    /* Denormalized inputs need care to avoid overflow on 1/t */
    r = -CUDART_2_OVER_PI / t;
  } else if (t <= 1.298570663015508497e0) {
    x = t * t;
    r = 2.599016977114429789e-13;
    r = __fma_rn(r, x, -5.646936040707309767e-11);
    r = __fma_rn(r, x, 8.931867331036295581e-9);
    r = __fma_rn(r, x, -9.926740542145188722e-7);
    r = __fma_rn(r, x, 7.164268749708438400e-5);
    r = __fma_rn(r, x, -2.955305336079382290e-3);
    r = __fma_rn(r, x, 5.434868816051021539e-2);
    r = __fma_rn(r, x, -1.960570906462389407e-1);
    r *= t;
    r += CUDART_2_OVER_PI * (log(t) * j1(t) - 1.0 / t);
  } else if (t <= 3.213411183412576033e0) {
    x = ((t - 2.197141326031017083e0) - -4.825983587645496567e-17);
    r = -3.204918540045980739e-9;
    r = __fma_rn(r, x, 1.126985362938592444e-8);
    r = __fma_rn(r, x, -9.725182107962382221e-9);
    r = __fma_rn(r, x, 1.083612003186428926e-9);
    r = __fma_rn(r, x, -3.318806432859500986e-8);
    r = __fma_rn(r, x, 1.152009920780307640e-7);
    r = __fma_rn(r, x, -2.165762322547769634e-7);
    r = __fma_rn(r, x, 4.248883280005704350e-7);
    r = __fma_rn(r, x, -9.597291015128258274e-7);
    r = __fma_rn(r, x, 2.143651955073189370e-6);
    r = __fma_rn(r, x, -4.688317848511307222e-6);
    r = __fma_rn(r, x, 1.026066296099274397e-5);
    r = __fma_rn(r, x, -2.248872084380127776e-5);
    r = __fma_rn(r, x, 4.924499594496305443e-5);
    r = __fma_rn(r, x, -1.077609598179235436e-4);
    r = __fma_rn(r, x, 2.358698833633901006e-4);
    r = __fma_rn(r, x, -5.096012361553002188e-4);
    r = __fma_rn(r, x, 1.066853008500809634e-3);
    r = __fma_rn(r, x, -2.595241693183597629e-3);
    r = __fma_rn(r, x, 7.422553332334889779e-3);
    r = __fma_rn(r, x, -4.797811669942416563e-3);
    r = __fma_rn(r, x, -3.285739740527982705e-2);
    r = __fma_rn(r, x, -1.185145457490981991e-1);
    r = __fma_rn(r, x, 5.207864124022675290e-1);
    r *= x;
  } else if (t <= 7.012843454562652030e0) {
    x = ((t - 5.429681040794134717e0) - 4.162514026670377007e-16);
    r = 3.641000824697897087e-16;
    r = __fma_rn(r, x, 6.273399595774693961e-16);
    r = __fma_rn(r, x, -1.656717829265264444e-15);
    r = __fma_rn(r, x, -1.793477656341538960e-14);
    r = __fma_rn(r, x, 4.410546816390020042e-14);
    r = __fma_rn(r, x, -1.387851333205382620e-13);
    r = __fma_rn(r, x, 1.170075916815038820e-12);
    r = __fma_rn(r, x, -4.612886656846937267e-12);
    r = __fma_rn(r, x, 2.222126653072601592e-12);
    r = __fma_rn(r, x, -3.852562731318657049e-10);
    r = __fma_rn(r, x, 5.598172933325135304e-9);
    r = __fma_rn(r, x, 2.550481704211604017e-8);
    r = __fma_rn(r, x, -5.464422265470442015e-7);
    r = __fma_rn(r, x, -2.863862325810848798e-6);
    r = __fma_rn(r, x, 4.645867915733586050e-5);
    r = __fma_rn(r, x, 1.466208928172848137e-4);
    r = __fma_rn(r, x, -2.165998751115648553e-3);
    r = __fma_rn(r, x, -4.160115934377754676e-3);
    r = __fma_rn(r, x, 5.094793974342303605e-2);
    r = __fma_rn(r, x, 3.133867744408601330e-2);
    r = __fma_rn(r, x, -3.403180455234405821e-1);
    r *= x;
  } else if (t <= 9.172580349585524928e0) {
    x = ((t - 8.596005868331168642e0) - 2.841583834006366401e-16);
    r = 2.305446091542135639e-16;
    r = __fma_rn(r, x, -1.372616651279859895e-13);
    r = __fma_rn(r, x, -1.067085198258553687e-12);
    r = __fma_rn(r, x, 2.797080742350623921e-11);
    r = __fma_rn(r, x, 1.883663311130206595e-10);
    r = __fma_rn(r, x, -4.684316504597157100e-9);
    r = __fma_rn(r, x, -2.441923258474869187e-8);
    r = __fma_rn(r, x, 5.586530988420728856e-7);
    r = __fma_rn(r, x, 2.081926450587367740e-6);
    r = __fma_rn(r, x, -4.380739676566903498e-5);
    r = __fma_rn(r, x, -1.042014850604930338e-4);
    r = __fma_rn(r, x, 2.011492014389694005e-3);
    r = __fma_rn(r, x, 2.417956732829416259e-3);
    r = __fma_rn(r, x, -4.340642670740071929e-2);
    r = __fma_rn(r, x, -1.578988436429690570e-2);
    r = __fma_rn(r, x, 2.714598773115335373e-1);
    r *= x;
  } else if (!__isinfd(t)) {
    double y = __internal_fast_rcp(t);
    double y2 = y * y;
    double f, arg;
    f = 1.765479925082250655e4;
    f = __fma_rn(f, y2, -1.801727125254790963e3);
    f = __fma_rn(f, y2, 1.136675500338510290e2);
    f = __fma_rn(f, y2, -7.595622833654403827e0);
    f = __fma_rn(f, y2, 8.045758488114477247e-1);
    f = __fma_rn(f, y2, -1.933571068757167499e-1);
    f = __fma_rn(f, y2, 1.874999959666924232e-1);
    f = __fma_rn(f, y2, 1.000000000003085088e0);
    arg = -8.471357607824940103e7;
    arg = __fma_rn(arg, y2, 8.464204863822212443e6);
    arg = __fma_rn(arg, y2, -4.326694608144371887e5);
    arg = __fma_rn(arg, y2, 1.658700399613585250e4);
    arg = __fma_rn(arg, y2, -6.279420695465894369e2);
    arg = __fma_rn(arg, y2, 3.046796375066591622e1);
    arg = __fma_rn(arg, y2, -2.368852258237428732e0);
    arg = __fma_rn(arg, y2, 3.708971794716567350e-1);
    arg = __fma_rn(arg, y2, -1.640624982860321990e-1);
    arg = __fma_rn(arg, y2, 3.749999999989471755e-1);
    arg = __fma_rn(arg, y, t);
    r = rsqrt(t) * CUDART_SQRT_2OPI * f * __cos_offset(arg, -3.926990816987241548e0);
  } else {
    r = 0.0;
  }
  if (a <= 0.0) {
    if (a == 0.0) {
      r = -CUDART_INF;
    } else {
      r = CUDART_NAN;
    }
  }
  return r;
}

/* Bessel functions of the second kind of integer
 * order are calculated using the forward recurrence:
 * Y(n+1, x) = 2n Y(n, x) / x - Y(n-1, x)
 */
static __forceinline__ double yn(int n, double a)
{
  double yip1; // is Y(i+1, a)
  double yi = y1(a); // is Y(i, a)
  double yim1 = y0(a); // is Y(i-1, a)
  double two_over_a;
  int i;
  if(n == 0) {
    return y0(a);
  }
  if(n == 1) {
    return y1(a);
  }
  if(n < 0) {
    return CUDART_NAN;
  }
  if(!(a >= 0.0)) {
    // also catches NAN input
    return CUDART_NAN;
  }
  if (fabs(a) < 1e-308) {
    /* Denormalized inputs need care to avoid overflow on 1/a */
    return -CUDART_2_OVER_PI / a;
  }
  two_over_a = 2.0 / a;
  for(i = 1; i < n; i++) {
    // Use forward recurrence, 6.8.4 from Hart et al.
    yip1 = __fma_rn(i * two_over_a,  yi, -yim1);
    yim1 = yi;
    yi = yip1;
  }
  if(__isnand(yip1)) {
    // We got overflow in forward recurrence
    return -CUDART_INF;
  }
  return yip1;
}

/* Bessel functions of the first kind of integer
 * order are calculated directly using the forward recurrence:
 * J(n+1, x) = 2n J(n, x) / x - J(n-1, x)
 * for large x.  For x small relative to n, Miller's algorithm is used 
 * as described in: F. W. J. Olver, "Error Analysis of Miller's Recurrence
 * Algorithm", Math. of Computation, Vol. 18, No. 85, 1964.
 */
static __forceinline__ double jn(int n, double a)
{
  double jip1 = 0.0; // is J(i+1, a)
  double ji = 1.0; // is J(i, a)
  double jim1; // is J(i-1, a)
  double lambda = 0.0;
  double sum = 0.0;
  int i;
  if(n == 0) {
    return y0(a);
  }
  if(n == 1) {
    return y1(a);
  }
  if(n < 0) {
    return CUDART_NAN;
  }
  if(fabs(a) > (double)n - 1.0) {
    // Use forward recurrence, numerically stable for large x
    double two_over_a = 2.0 / a;
    double ji = j1(a); // is J(i, a)
    double jim1 = j0(a); // is J(i-1, a)
    for(i = 1; i < n; i++) {
      jip1 = __fma_rn(i * two_over_a, ji, -jim1);
      jim1 = ji;
      ji = jip1;
    }
    return jip1;
  } else {
    /* Limit m based on comments from Press et al. "Numerical Recipes
       in C", 2nd ed., p. 234--235, 1992. 
    */
    double two_over_a = 2.0 / a;
    int m = n + (int)sqrt(n * 60);
    m = (m >> 1) << 1;
    for(i = m; i >= 1; --i) {
      // Use backward recurrence
      jim1 = __fma_rn(i * two_over_a, ji, -jip1);
      jip1 = ji;
      // Rescale to avoid intermediate overflow
      if(fabsf(jim1) > 1e15) {
        jim1 *= 1e-15;
        jip1 *= 1e-15;
        lambda *= 1e-15;
        sum *= 1e-15;
      }
      ji = jim1;
      if(i - 1 == n) {
        lambda = ji;
      }
      if(i & 1) {
        sum += 2.0 * ji;
      }
    }
    sum -= ji;
    return lambda / sum;
  }
}


static __forceinline__ double erf(double a)
{
  double t, r, q;

  t = fabs(a);
  if (t >= 1.0) {
    r =                 -1.28836351230756500E-019;
    r = __fma_rn (r, t,  1.30597472161093370E-017);
    r = __fma_rn (r, t, -6.33924401259620500E-016);
    r = __fma_rn (r, t,  1.96231865908940140E-014);
    r = __fma_rn (r, t, -4.35272243559990750E-013);
    r = __fma_rn (r, t,  7.37083927929352150E-012);
    r = __fma_rn (r, t, -9.91402142550461630E-011);
    r = __fma_rn (r, t,  1.08817017167760820E-009);
    r = __fma_rn (r, t, -9.93918713097634620E-009);
    r = __fma_rn (r, t,  7.66739923255145500E-008);
    r = __fma_rn (r, t, -5.05440278302806720E-007);
    r = __fma_rn (r, t,  2.87474157099000620E-006);
    r = __fma_rn (r, t, -1.42246725399722510E-005);
    r = __fma_rn (r, t,  6.16994555079419460E-005);
    r = __fma_rn (r, t, -2.36305221938908790E-004);
    r = __fma_rn (r, t,  8.05032844055371070E-004);
    r = __fma_rn (r, t, -2.45833366629108140E-003);
    r = __fma_rn (r, t,  6.78340988296706120E-003);
    r = __fma_rn (r, t, -1.70509103597554640E-002);
    r = __fma_rn (r, t,  3.93322852515666300E-002);
    r = __fma_rn (r, t, -8.37271292613764040E-002);
    r = __fma_rn (r, t,  1.64870423707623280E-001);
    r = __fma_rn (r, t, -2.99729521787681470E-001);
    r = __fma_rn (r, t,  4.99394435612628580E-001);
    r = __fma_rn (r, t, -7.52014596480123030E-001);
    r = __fma_rn (r, t,  9.99933138314926250E-001);
    r = __fma_rn (r, t, -1.12836725321102670E+000);
    r = __fma_rn (r, t,  9.99998988715182450E-001);
    q = __internal_exp_kernel(-t * t, 0);
    r = __fma_rn (r, -q, 1.0);
    if (t >= 6.5) {
      r = 1.0;
    }    
    a = __internal_copysign_pos(r, a);
  } else {
    q = a * a;
    r =                 -7.77946848895991420E-010;
    r = __fma_rn (r, q,  1.37109803980285950E-008);
    r = __fma_rn (r, q, -1.62063137584932240E-007);
    r = __fma_rn (r, q,  1.64471315712790040E-006);
    r = __fma_rn (r, q, -1.49247123020098620E-005);
    r = __fma_rn (r, q,  1.20552935769006260E-004);
    r = __fma_rn (r, q, -8.54832592931448980E-004);
    r = __fma_rn (r, q,  5.22397760611847340E-003);
    r = __fma_rn (r, q, -2.68661706431114690E-002);
    r = __fma_rn (r, q,  1.12837916709441850E-001);
    r = __fma_rn (r, q, -3.76126389031835210E-001);
    r = __fma_rn (r, q,  1.12837916709551260E+000);
    a = r * a;
  }
  return a;
}

/*
 * This erfinv implementation is derived with minor modifications from:
 * Mike Giles, Approximating the erfinv function, GPU Gems 4, volume 2
 * Retrieved from http://www.gpucomputing.net/?q=node/1828 on 8/15/2010
 */
static __forceinline__ double erfinv(double a)
{
  if (fabs(a) < 1.0) {
    double t, r;
    t = __fma_rn (a, -a, 1.0);
    t = - __internal_fast_log (t);
    if (t < 6.25) {
      t = t - 3.125;
      r =                 -3.6444120640178197e-21;
      r = __fma_rn (r, t, -1.6850591381820166e-19);
      r = __fma_rn (r, t,  1.2858480715256400e-18);
      r = __fma_rn (r, t,  1.1157877678025181e-17);
      r = __fma_rn (r, t, -1.3331716628546209e-16);
      r = __fma_rn (r, t,  2.0972767875968562e-17);
      r = __fma_rn (r, t,  6.6376381343583238e-15);
      r = __fma_rn (r, t, -4.0545662729752069e-14);
      r = __fma_rn (r, t, -8.1519341976054722e-14);
      r = __fma_rn (r, t,  2.6335093153082323e-12);
      r = __fma_rn (r, t, -1.2975133253453532e-11);
      r = __fma_rn (r, t, -5.4154120542946279e-11);
      r = __fma_rn (r, t,  1.0512122733215323e-09);
      r = __fma_rn (r, t, -4.1126339803469837e-09);
      r = __fma_rn (r, t, -2.9070369957882005e-08);
      r = __fma_rn (r, t,  4.2347877827932404e-07);
      r = __fma_rn (r, t, -1.3654692000834679e-06);
      r = __fma_rn (r, t, -1.3882523362786469e-05);
      r = __fma_rn (r, t,  1.8673420803405714e-04);
      r = __fma_rn (r, t, -7.4070253416626698e-04);
      r = __fma_rn (r, t, -6.0336708714301491e-03);
      r = __fma_rn (r, t,  2.4015818242558962e-01);
      r = __fma_rn (r, t,  1.6536545626831027e+00);
    } else {
      t = sqrt(t);
      if (t < 4.0) {
        t = t - 3.25;
        r =                  2.2137376921775787e-09;
        r = __fma_rn (r, t,  9.0756561938885391e-08);
        r = __fma_rn (r, t, -2.7517406297064545e-07);
        r = __fma_rn (r, t,  1.8239629214389228e-08);
        r = __fma_rn (r, t,  1.5027403968909828e-06);
        r = __fma_rn (r, t, -4.0138675269815460e-06);
        r = __fma_rn (r, t,  2.9234449089955446e-06);
        r = __fma_rn (r, t,  1.2475304481671779e-05);
        r = __fma_rn (r, t, -4.7318229009055734e-05);
        r = __fma_rn (r, t,  6.8284851459573175e-05);
        r = __fma_rn (r, t,  2.4031110387097894e-05);
        r = __fma_rn (r, t, -3.5503752036284748e-04);
        r = __fma_rn (r, t,  9.5328937973738050e-04);
        r = __fma_rn (r, t, -1.6882755560235047e-03);
        r = __fma_rn (r, t,  2.4914420961078508e-03);
        r = __fma_rn (r, t, -3.7512085075692412e-03);
        r = __fma_rn (r, t,  5.3709145535900636e-03);
        r = __fma_rn (r, t,  1.0052589676941592e+00);
        r = __fma_rn (r, t,  3.0838856104922208e+00);
      } else {
        t = t - 5.0;
        r =                 -2.7109920616438573e-11;
        r = __fma_rn (r, t, -2.5556418169965252e-10);
        r = __fma_rn (r, t,  1.5076572693500548e-09);
        r = __fma_rn (r, t, -3.7894654401267370e-09);
        r = __fma_rn (r, t,  7.6157012080783394e-09);
        r = __fma_rn (r, t, -1.4960026627149240e-08);
        r = __fma_rn (r, t,  2.9147953450901081e-08);
        r = __fma_rn (r, t, -6.7711997758452339e-08);
        r = __fma_rn (r, t,  2.2900482228026655e-07);
        r = __fma_rn (r, t, -9.9298272942317003e-07);
        r = __fma_rn (r, t,  4.5260625972231537e-06);
        r = __fma_rn (r, t, -1.9681778105531671e-05);
        r = __fma_rn (r, t,  7.5995277030017761e-05);
        r = __fma_rn (r, t, -2.1503011930044477e-04);
        r = __fma_rn (r, t, -1.3871931833623122e-04);
        r = __fma_rn (r, t,  1.0103004648645344e+00);
        r = __fma_rn (r, t,  4.8499064014085844e+00);
      }
    }
    return r * a;
  } else {
    if (__isnand(a)) {
      return a + a;
    }
    if (fabs(a) == 1.0) {
      return a * CUDART_INF;
    }
    return CUDART_NAN;
  }
}

static __forceinline__ double erfcinv(double a)
{
  double t;
  if (a <= CUDART_ZERO) {
    t = CUDART_NAN;
    if (a == CUDART_ZERO) {
      t = (1.0 - a) * CUDART_INF;
    }
  } 
  else if (a >= 0.0625) {
    t = erfinv (1.0 - a);
  }
  else if (a >= 1e-100) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
    */
    double p, q;
    t = __internal_fast_log (a);
    t = __internal_fast_rsqrt (-t);
    p =                 2.7834010353747001060e-3;
    p = __fma_rn (p, t, 8.6030097526280260580e-1);
    p = __fma_rn (p, t, 2.1371214997265515515e+0);
    p = __fma_rn (p, t, 3.1598519601132090206e+0);
    p = __fma_rn (p, t, 3.5780402569085996758e+0);
    p = __fma_rn (p, t, 1.5335297523989890804e+0);
    p = __fma_rn (p, t, 3.4839207139657522572e-1);
    p = __fma_rn (p, t, 5.3644861147153648366e-2);
    p = __fma_rn (p, t, 4.3836709877126095665e-3);
    p = __fma_rn (p, t, 1.3858518113496718808e-4);
    p = __fma_rn (p, t, 1.1738352509991666680e-6);
    q =              t+ 2.2859981272422905412e+0;
    q = __fma_rn (q, t, 4.3859045256449554654e+0);
    q = __fma_rn (q, t, 4.6632960348736635331e+0);
    q = __fma_rn (q, t, 3.9846608184671757296e+0);
    q = __fma_rn (q, t, 1.6068377709719017609e+0);
    q = __fma_rn (q, t, 3.5609087305900265560e-1);
    q = __fma_rn (q, t, 5.3963550303200816744e-2);
    q = __fma_rn (q, t, 4.3873424022706935023e-3);
    q = __fma_rn (q, t, 1.3858762165532246059e-4);
    q = __fma_rn (q, t, 1.1738313872397777529e-6);
    t = p / (q * t);
  }
  else {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 82
    */
    double p, q;
    t = log (a);
    t = rsqrt (-t);
    p =                 6.9952990607058154858e-1;
    p = __fma_rn (p, t, 1.9507620287580568829e+0);
    p = __fma_rn (p, t, 8.2810030904462690216e-1);
    p = __fma_rn (p, t, 1.1279046353630280005e-1);
    p = __fma_rn (p, t, 6.0537914739162189689e-3);
    p = __fma_rn (p, t, 1.3714329569665128933e-4);
    p = __fma_rn (p, t, 1.2964481560643197452e-6);
    p = __fma_rn (p, t, 4.6156006321345332510e-9);
    p = __fma_rn (p, t, 4.5344689563209398450e-12);
    q =              t+ 1.5771922386662040546e+0;
    q = __fma_rn (q, t, 2.1238242087454993542e+0);
    q = __fma_rn (q, t, 8.4001814918178042919e-1);
    q = __fma_rn (q, t, 1.1311889334355782065e-1);
    q = __fma_rn (q, t, 6.0574830550097140404e-3);
    q = __fma_rn (q, t, 1.3715891988350205065e-4);
    q = __fma_rn (q, t, 1.2964671850944981713e-6);
    q = __fma_rn (q, t, 4.6156017600933592558e-9);
    q = __fma_rn (q, t, 4.5344687377088206783e-12);
    t = p / (q * t);
  }
  return t;
}

static __forceinline__ double __internal_erfcx_kernel (double a)
{
  /*  
   * The implementation of erfcx() is based on the algorithm in: M. M. Shepherd
   * and J. G. Laframboise, "Chebyshev Approximation of (1 + 2x)exp(x^2)erfc x
   * in 0 <= x < INF", Mathematics of Computation, Vol. 36, No. 153, January
   * 1981, pp. 249-253. For the core approximation, the input domain [0,INF] is
   * transformed via (x-k) / (x+k) where k is a precision-dependent constant.  
   * Here, we choose k = 4.0, so input domain [0,27.3] is transformed to the   
   * core approximation domain [-1,0.744409].   
   */  
  double t1, t2, t3, t4;  
  /* (1+2*x)*exp(x*x)*erfc(x) */ 
  /* t2 = (x-4.0)/(x+4.0), transforming [0,INF] to [-1,+1] */
  t1 = a - 4.0; 
  t2 = a + 4.0; 
  t2 = __internal_fast_rcp(t2);
  t3 = t1 * t2;
  t4 = __dadd_rn (t3, 1.0);         /* prevent FMA-merging */
  t1 = __fma_rn (-4.0, t4, a); 
  t1 = __fma_rn (-t3, a, t1); 
  t2 = __fma_rn (t2, t1, t3); 
  /* approximate on [-1, 0.744409] */   
  t1 =                   -3.5602694826817400E-010; 
  t1 = __fma_rn (t1, t2, -9.7239122591447274E-009); 
  t1 = __fma_rn (t1, t2, -8.9350224851649119E-009); 
  t1 = __fma_rn (t1, t2,  1.0404430921625484E-007); 
  t1 = __fma_rn (t1, t2,  5.8806698585341259E-008); 
  t1 = __fma_rn (t1, t2, -8.2147414929116908E-007); 
  t1 = __fma_rn (t1, t2,  3.0956409853306241E-007); 
  t1 = __fma_rn (t1, t2,  5.7087871844325649E-006); 
  t1 = __fma_rn (t1, t2, -1.1231787437600085E-005); 
  t1 = __fma_rn (t1, t2, -2.4399558857200190E-005); 
  t1 = __fma_rn (t1, t2,  1.5062557169571788E-004); 
  t1 = __fma_rn (t1, t2, -1.9925637684786154E-004); 
  t1 = __fma_rn (t1, t2, -7.5777429182785833E-004); 
  t1 = __fma_rn (t1, t2,  5.0319698792599572E-003); 
  t1 = __fma_rn (t1, t2, -1.6197733895953217E-002); 
  t1 = __fma_rn (t1, t2,  3.7167515553018733E-002); 
  t1 = __fma_rn (t1, t2, -6.6330365827532434E-002); 
  t1 = __fma_rn (t1, t2,  9.3732834997115544E-002); 
  t1 = __fma_rn (t1, t2, -1.0103906603555676E-001); 
  t1 = __fma_rn (t1, t2,  6.8097054254735140E-002); 
  t1 = __fma_rn (t1, t2,  1.5379652102605428E-002); 
  t1 = __fma_rn (t1, t2, -1.3962111684056291E-001); 
  t1 = __fma_rn (t1, t2,  1.2329951186255526E+000); 
  /* (1+2*x)*exp(x*x)*erfc(x) / (1+2*x) = exp(x*x)*erfc(x) */
  t2 = __fma_rn (2.0, a, 1.0);  
  t2 = __internal_fast_rcp(t2);
  t3 = t1 * t2; 
  t4 = __fma_rn (a, -2.0*t3, t1); 
  t4 = __dadd_rn (t4, -t3);         /* prevent FMA-merging */
  t1 = __fma_rn (t4, t2, t3); 
  return t1;
}

static __forceinline__ double erfc(double a)  
{  
  double x, t1, t2, t3;  

  if (__isnand(a)) return a + a;
  x = fabs(a); 
  t1 = __internal_erfcx_kernel (x);
  /* exp(-x*x) * exp(x*x)*erfc(x) = erfc(x) */  
  t2 = -x * x;  
  t3 = __internal_exp_kernel (t2, 0);  
  t2 = __fma_rn (-x, x, -t2);  
  t2 = __fma_rn (t3, t2, t3);  
  t1 = t1 * t2;  
  if (x > 27.3) t1 = 0.0;  
  return (__double2hiint(a) < 0) ? (2.0 - t1) : t1; 
}

static __forceinline__ double erfcx(double a)  
{
  double x, t1, t2, t3;
  x = fabs(a); 
  if ((unsigned)__double2hiint(x) < (unsigned)0x40400000) {
    t1 = __internal_erfcx_kernel(x);
  } else {
    /* asymptotic expansion for large aguments */
    t2 = 1.0 / x;
    t3 = t2 * t2;
    t1 =                   -29.53125;
    t1 = __fma_rn (t1, t3, +6.5625);
    t1 = __fma_rn (t1, t3, -1.875);
    t1 = __fma_rn (t1, t3, +0.75);
    t1 = __fma_rn (t1, t3, -0.5);
    t1 = __fma_rn (t1, t3, +1.0);
    t2 = t2 * 5.6418958354775628e-001;
    t1 = t1 * t2;
  }
  if (__double2hiint(a) < 0) {
    /* erfcx(x) = 2*exp(x^2) - erfcx(|x|) */
    t2 = x * x;
    t3 = __fma_rn (x, x, -t2);
    t2 = exp (t2);
    t2 = t2 + t2;
    t3 = __fma_rn (t2, t3, t2);
    t1 = t3 - t1;
    if (t2 == CUDART_INF) t1 = t2;
  }
  return t1;
}

/* approximate 1.0/(a*gamma(a)) on [-0.5,0.5] */
static __forceinline__ double __internal_tgamma_kernel(double a)
{
  double t;
  t =                 -4.42689340712524750E-010;
  t = __fma_rn (t, a, -2.02665918466589540E-007);
  t = __fma_rn (t, a,  1.13812117211195270E-006);
  t = __fma_rn (t, a, -1.25077348166307480E-006);
  t = __fma_rn (t, a, -2.01365017404087710E-005);
  t = __fma_rn (t, a,  1.28050126073544860E-004);
  t = __fma_rn (t, a, -2.15241408115274180E-004);
  t = __fma_rn (t, a, -1.16516754597046040E-003);
  t = __fma_rn (t, a,  7.21894322484663810E-003);
  t = __fma_rn (t, a, -9.62197153268626320E-003);
  t = __fma_rn (t, a, -4.21977345547223940E-002);
  t = __fma_rn (t, a,  1.66538611382503560E-001);
  t = __fma_rn (t, a, -4.20026350341054440E-002);
  t = __fma_rn (t, a, -6.55878071520257120E-001);
  t = __fma_rn (t, a,  5.77215664901532870E-001);
  t = __fma_rn (t, a,  1.00000000000000000E+000);
  return t;
}

/* Stirling approximation for gamma(a), a > 20 */
static __forceinline__ double __internal_stirling_poly(double a)
{
  double x = __internal_fast_rcp(a);
  double z = 0.0;
  z = __fma_rn (z, x,  8.3949872067208726e-004);
  z = __fma_rn (z, x, -5.1717909082605919e-005);
  z = __fma_rn (z, x, -5.9216643735369393e-004);
  z = __fma_rn (z, x,  6.9728137583658571e-005);
  z = __fma_rn (z, x,  7.8403922172006662e-004);
  z = __fma_rn (z, x, -2.2947209362139917e-004);
  z = __fma_rn (z, x, -2.6813271604938273e-003);
  z = __fma_rn (z, x,  3.4722222222222220e-003);
  z = __fma_rn (z, x,  8.3333333333333329e-002);
  z = __fma_rn (z, x,  1.0000000000000000e+000);
  return z;
}

static __forceinline__ double __internal_tgamma_stirling(double a)
{
  if (a < 1.7162437695630274e+002) {
    double t_hi, t_lo, e;

    double2 loga, prod;
    double z = __internal_stirling_poly (a);
    double b = a - 0.5;

    /* compute log(a) in double-double format*/
    loga = __internal_log_ext_prec(a);

    /* compute (a - 0.5) * log(a) in double-double format */
    t_hi = __dmul_rn (loga.y, b);  /* prevent FMA-merging */
    t_lo = __fma_rn (loga.y, b, -t_hi);
    t_lo = __fma_rn (loga.x, b, t_lo);
    prod.y = e = t_hi + t_lo;
    prod.x = (t_hi - e) + t_lo;

    /* compute (a - 0.5) * log(a) - a in double-double format */
    loga.y = -a;
    loga.x = 0.0;
    prod = __internal_ddadd_xgty (prod, loga);

    /* compute pow(a,b) = exp(b*log(a)) */
    a = exp(prod.y);
    /* prevent -INF + INF = NaN */
    if (!__isinfd(a)) {
      /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
       * exp(prod.y) + prod.x * exp(prod.y) 
       */
      a = __fma_rn (a, prod.x, a);
    }
    a = __fma_rn (a, CUDART_SQRT_2PI_HI, a * CUDART_SQRT_2PI_LO);
    return a * z;
  } else {
    return CUDART_INF;
  }
}

static __forceinline__ double tgamma(double a)
{
  double s, xx, x = a;
  if (__isnand(a)) {
    return a + a;
  }
  if (fabs(x) < 15.0) {
     /* Based on: Kraemer, W.: "Berechnung der Gammafunktion G(x) fuer reelle 
      * Punkt- und Intervallargumente". Zeitschrift fuer angewandte Mathematik 
      * und Mechanik, Vol. 70 (1990), No. 6, pp. 581-584
      */
    if (x >= 0.0) {
      s = 1.0;
      xx = x;
      while (xx > 1.5) {
        s = __fma_rn(s, xx, -s);
        xx = xx - 1.0;
      }
      if (x >= 0.5) {
        xx = xx - 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      if (x < 0.5) {
        xx = xx * x;
      }
      s = s / xx;
    } else {
      xx = x;
      s = xx;
      if (x == trunc(x)) {
        return CUDART_NAN;
      }
      while (xx < -0.5) {
        s = __fma_rn (s, xx, s);
        xx = xx + 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      s = s * xx;
      s = 1.0 / s;
    }
    return s;
  } else {
    if (x >= 0.0) {
      return __internal_tgamma_stirling (x);
    } else {
      double t;
      int quot;
      if (x == trunc(x)) {
        return CUDART_NAN;
      }
      if (x < -185.0) {
        int negative;
        x = floor(x);
        negative = ((x - (2.0 * floor(0.5 * x))) == 1.0);
        return negative ? CUDART_NEG_ZERO : CUDART_ZERO;
      }
      /* compute sin(pi*x) accurately */
      xx = rint (__internal_twice(x));
      quot = (int)xx;
      xx = __fma_rn (-0.5, xx, x);
      xx = xx * CUDART_PI;
      if (quot & 1) {
        xx = __internal_cos_kerneld (xx);
      } else {
        xx = __internal_sin_kerneld (xx);
      }
      if (quot & 2) {
        xx = -xx;
      }
      s = __internal_exp_kernel (x, 0);
      x = fabs (x);
      t = x - 0.5;
      if (x > 140.0) t = __internal_half(t);
      t = __internal_accurate_pow (x, t);
      if (x > 140.0) s = s * t;
      s = s * __internal_stirling_poly (x);
      s = s * x;
      s = s * xx;
      s = 1.0 / s;
      s = __fma_rn (s, CUDART_SQRT_PIO2_HI, CUDART_SQRT_PIO2_LO * s);
      s = s / t;
      return s;
    }
  }
}

static __forceinline__ double __internal_lgamma_pos(double a)
{
  double sum;
  double s, t;

  if (a == CUDART_INF) {
    return a;
  }
  if (a >= 3.0) {
    if (a >= 8.0) {
      /* Stirling approximation; coefficients from Hart et al, "Computer 
       * Approximations", Wiley 1968. Approximation 5404. 
       */
      s = __internal_fast_rcp(a);
      t = s * s;
      sum =                   -0.1633436431e-2;
      sum = __fma_rn (sum, t,  0.83645878922e-3);
      sum = __fma_rn (sum, t, -0.5951896861197e-3);
      sum = __fma_rn (sum, t,  0.793650576493454e-3);
      sum = __fma_rn (sum, t, -0.277777777735865004e-2);
      sum = __fma_rn (sum, t,  0.833333333333331018375e-1);
      sum = __fma_rn (sum, s,  0.918938533204672);
      s = __internal_half(log (a));
      t = a - 0.5;
      sum = __fma_rn(s, t, sum);
      t = __fma_rn (s, t, - a);
      t = t + sum;
      return t;
    } else {
      a = a - 3.0;
      s =                 -4.02412642744125560E+003;
      s = __fma_rn (s, a, -2.97693796998962000E+005);
      s = __fma_rn (s, a, -6.38367087682528790E+006);
      s = __fma_rn (s, a, -5.57807214576539320E+007);
      s = __fma_rn (s, a, -2.24585140671479230E+008);
      s = __fma_rn (s, a, -4.70690608529125090E+008);
      s = __fma_rn (s, a, -7.62587065363263010E+008);
      s = __fma_rn (s, a, -9.71405112477113250E+008);
      t =              a  -1.02277248359873170E+003;
      t = __fma_rn (t, a, -1.34815350617954480E+005);
      t = __fma_rn (t, a, -4.64321188814343610E+006);
      t = __fma_rn (t, a, -6.48011106025542540E+007);
      t = __fma_rn (t, a, -4.19763847787431360E+008);
      t = __fma_rn (t, a, -1.25629926018000720E+009);
      t = __fma_rn (t, a, -1.40144133846491690E+009);
      t = s / t;
      t = t + a;
      return t;
    }
  } else if (a >= 1.5) {
    a = a - 2.0;
    t =                  9.84839283076310610E-009;
    t = __fma_rn (t, a, -6.69743850483466500E-008);
    t = __fma_rn (t, a,  2.16565148880011450E-007);
    t = __fma_rn (t, a, -4.86170275781575260E-007);
    t = __fma_rn (t, a,  9.77962097401114400E-007);
    t = __fma_rn (t, a, -2.03041287574791810E-006);
    t = __fma_rn (t, a,  4.36119725805364580E-006);
    t = __fma_rn (t, a, -9.43829310866446590E-006);
    t = __fma_rn (t, a,  2.05106878496644220E-005);
    t = __fma_rn (t, a, -4.49271383742108440E-005);
    t = __fma_rn (t, a,  9.94570466342226000E-005);
    t = __fma_rn (t, a, -2.23154589559238440E-004);
    t = __fma_rn (t, a,  5.09669559149637430E-004);
    t = __fma_rn (t, a, -1.19275392649162300E-003);
    t = __fma_rn (t, a,  2.89051032936815490E-003);
    t = __fma_rn (t, a, -7.38555102806811700E-003);
    t = __fma_rn (t, a,  2.05808084278121250E-002);
    t = __fma_rn (t, a, -6.73523010532073720E-002);
    t = __fma_rn (t, a,  3.22467033424113040E-001);
    t = __fma_rn (t, a,  4.22784335098467190E-001);
    t = t * a;
    return t;
  } else if (a >= 0.7) {
    a = 1.0 - a;
    t =                 1.17786911519331130E-002;  
    t = __fma_rn (t, a, 3.89046747413522300E-002);
    t = __fma_rn (t, a, 5.90045711362049900E-002);
    t = __fma_rn (t, a, 6.02143305254344420E-002);
    t = __fma_rn (t, a, 5.61652708964839180E-002);
    t = __fma_rn (t, a, 5.75052755193461370E-002);
    t = __fma_rn (t, a, 6.21061973447320710E-002);
    t = __fma_rn (t, a, 6.67614724532521880E-002);
    t = __fma_rn (t, a, 7.14856037245421020E-002);
    t = __fma_rn (t, a, 7.69311251313347100E-002);
    t = __fma_rn (t, a, 8.33503129714946310E-002);
    t = __fma_rn (t, a, 9.09538288991182800E-002);
    t = __fma_rn (t, a, 1.00099591546322310E-001);
    t = __fma_rn (t, a, 1.11334278141734510E-001);
    t = __fma_rn (t, a, 1.25509666613462880E-001);
    t = __fma_rn (t, a, 1.44049896457704160E-001);
    t = __fma_rn (t, a, 1.69557177031481600E-001);
    t = __fma_rn (t, a, 2.07385551032182120E-001);
    t = __fma_rn (t, a, 2.70580808427600350E-001);
    t = __fma_rn (t, a, 4.00685634386517050E-001);
    t = __fma_rn (t, a, 8.22467033424113540E-001);
    t = __fma_rn (t, a, 5.77215664901532870E-001);
    t = t * a;
    return t;
  } else {
    t=                  -9.04051686831357990E-008;
    t = __fma_rn (t, a,  7.06814224969349250E-007);
    t = __fma_rn (t, a, -3.80702154637902830E-007);
    t = __fma_rn (t, a, -2.12880892189316100E-005);
    t = __fma_rn (t, a,  1.29108470307156190E-004);
    t = __fma_rn (t, a, -2.15932815215386580E-004);
    t = __fma_rn (t, a, -1.16484324388538480E-003);
    t = __fma_rn (t, a,  7.21883433044470670E-003);
    t = __fma_rn (t, a, -9.62194579514229560E-003);
    t = __fma_rn (t, a, -4.21977386992884450E-002);
    t = __fma_rn (t, a,  1.66538611813682460E-001);
    t = __fma_rn (t, a, -4.20026350606819980E-002);
    t = __fma_rn (t, a, -6.55878071519427450E-001);
    t = __fma_rn (t, a,  5.77215664901523870E-001);
    t = t * a;
    t = __fma_rn (t, a, a);
    return -log (t);
  }
}

static __forceinline__ double lgamma(double a)
{
  double t;
  double i;
  long long int quot;
  if (__isnand(a)) {
    return a + a;
  }
  t = __internal_lgamma_pos(fabs(a));
  if (a >= 0.0) return t;
  a = fabs(a);
  i = trunc(a);       
  if (a == i) return CUDART_INF; /* a is an integer: return infinity */
  if (a < 1e-19) return -log(a);
  i = rint (2.0 * a);
  quot = (long long int)i;
  i = __fma_rn (-0.5, i, a);
  i = i * CUDART_PI;
  if (quot & 1) {
    i = __internal_cos_kerneld(i);
  } else {
    i = __internal_sin_kerneld(i);
  }
  i = fabs(i);
  t = log(CUDART_PI / (i * a)) - t;
  return t;
}

static __forceinline__ double ldexp(double a, int b)
{
  double fa = fabs (a);
  if ((fa == CUDART_ZERO) || (fa == CUDART_INF) || (!(fa <= CUDART_INF))) {
    return a + a;
  }
  if (b == 0) {
    return a;
  }
  if (b >  2200) b =  2200;
  if (b < -2200) b = -2200;
  if (abs (b) < 1022) {
    return a * __internal_exp2i_kernel(b);
  }
  if (abs (b) < 2044) {
    int bhalf = b / 2;
    return a * __internal_exp2i_kernel (bhalf) * 
           __internal_exp2i_kernel (b - bhalf);
  } else {
    int bquarter = b / 4;
    double t = __internal_exp2i_kernel(bquarter);
    return a * t * t * t *__internal_exp2i_kernel (b - 3 * bquarter);
  }
}

static __forceinline__ double scalbn(double a, int b)
{
  /* On binary systems, ldexp(x,exp) is equivalent to scalbn(x,exp) */
  return ldexp(a, b);
}

static __forceinline__ double scalbln(double a, long int b)
{
#if defined(__LP64__)
  /* clamp to integer range prior to conversion */
  if (b < -2147483648L) b = -2147483648L;
  if (b >  2147483647L) b =  2147483647L;
#endif /* __LP64__ */
  return scalbn(a, (int)b);
}

static __forceinline__ double frexp(double a, int *b)
{
  double fa = fabs(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < CUDART_TWO_TO_M1022) {
    a *= CUDART_TWO_TO_54;
    denorm = 54;
  } else {
    denorm = 0;
  }
  expo = (__double2hiint(a) >> 20) & 0x7ff;
  if ((fa == 0.0) || (expo == 0x7ff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 1022;
    a = __longlong_as_double((__double_as_longlong(a) & 0x800fffffffffffffULL)|
                              0x3fe0000000000000ULL);
  }
  *b = expo;
  return a;  
}

static __forceinline__ double modf(double a, double *b)
{
  double t;
  if (__isfinited(a)) {
    t = trunc(a);
    *b = t;
    t = a - t;
    return __internal_copysign_pos(t, a);
  } else if (__isinfd(a)) {
    t = 0.0;
    *b = a;
    return __internal_copysign_pos(t, a);
  } else {
    *b = a + a; 
    return a + a;
  }  
}

static __forceinline__ double fmod(double a, double b)
{
  double orig_a = a;
  double orig_b = b;
  a = fabs(a);
  b = fabs(b);
  if (!((a <= CUDART_INF) && (b <= CUDART_INF))) {
      return orig_a + orig_b;
  }
  if (a == CUDART_INF || b == 0.0) {
    return CUDART_NAN;
  } else if (a >= b) {
    int bhi = __double2hiint(b);
    int blo = __double2loint(b);
    int ahi = __double2hiint(a);
    double scaled_b = 0.0;
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5;
    }
    return __internal_copysign_pos(a, orig_a);
  } else {
    return orig_a;
  }
}

static __forceinline__ double remainder(double a, double b)
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot0 = 0;  /* quotient bit 0 */
  int bhi;
  int blo;
  int ahi;
  if (__isnand(a) || __isnand(b)) {
    return a + b;
  }
  orig_a = a;
  a = fabs(a);
  b = fabs(b);
  if (a == CUDART_INF || b == 0.0) {
    return CUDART_NAN;
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5;
    }
  }
  /* round quotient to nearest even */
  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  return a;
}

static __forceinline__ double remquo(double a, double b, int *c)
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot = 0;  /* trailing quotient bits  */
  unsigned int sign;
  int bhi;
  int blo;
  int ahi;
  if (__isnand(a) || __isnand(b)) {
    *c = quot;
    return a + b;
  }
  orig_a = a;
  sign = 0 - ((__double2hiint(a) ^ __double2hiint(b)) < 0);
  a = fabs(a);
  b = fabs(b);
  if (a == CUDART_INF || b == 0.0) {
    *c = quot;
    return CUDART_NAN;
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5;
    }
  }
  /* round quotient to nearest even */
  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  quot = quot & CUDART_REMQUO_MASK_F;
  quot = quot ^ sign;
  quot = quot - sign;
  *c = quot;
  return a;
}

static __forceinline__ double nextafter(double a, double b)
{
  unsigned long long int ia;
  unsigned long long int ib;
  ia = __double_as_longlong(a);
  ib = __double_as_longlong(b);
  if (__isnand(a) || __isnand(b)) return a + b; /* NaN */
  if (((ia | ib) << 1) == 0ULL) return b;
  if ((ia + ia) == 0ULL) {
    return __internal_copysign_pos(CUDART_MIN_DENORM, b);   /* crossover */
  }
  if ((a < b) && (a < 0.0)) ia--;
  if ((a < b) && (a > 0.0)) ia++;
  if ((a > b) && (a < 0.0)) ia++;
  if ((a > b) && (a > 0.0)) ia--;
  a = __longlong_as_double(ia);
  return a;
}

static __forceinline__ double nan(const char *tagp)
{
  unsigned long long int i;

  i = __internal_nan_kernel (tagp);
  i = (i & 0x000fffffffffffffULL) | 0x7ff8000000000000ULL;
  return __longlong_as_double(i);
}

static __forceinline__ double round(double a)
{
  double fa = fabs(a);
  if (fa >= CUDART_TWO_TO_52) {
    return a;
  } else {      
    double u;
    u = trunc(fa + 0.5);
    if (fa < 0.5) u = 0;
    u = __internal_copysign_pos(u, a);
    return u;
  }
}

static __forceinline__ long long int llround(double a)
{
  return (long long int)round(a);
}

static __forceinline__ long int lround(double a)
{
#if defined(__LP64__)
  return (long int)llround(a);
#else /* __LP64__ */
  return (long int)round(a);
#endif /* __LP64__ */
}

static __forceinline__ double fdim(double a, double b)
{
  double t;
  t = a - b;    /* default also takes care of NaNs */
  if (a <= b) {
    t = 0.0;
  }
  return t;
}

static __forceinline__ int ilogb(double a)
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__isnand(a)) return -__cuda_INT_MAX-1;
  if (__isinfd(a)) return __cuda_INT_MAX;
  if (a == 0.0) return -__cuda_INT_MAX-1;
  a = fabs(a);
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= CUDART_TWO_TO_M1022) {
    return ((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    return -1011 - __clzll(i);
  }
}

static __forceinline__ double logb(double a)
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__isnand(a)) return a + a;
  a = fabs(a);
  if (a == CUDART_INF) return a;
  if (a == 0.0) return -CUDART_INF;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= CUDART_TWO_TO_M1022) {
    return (double)((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    int expo = -1011 - __clzll(i);
    return (double)expo;
  }
}

static __forceinline__ double fma(double a, double b, double c)
{
  return __fma_rn(a, b, c);
}

#endif /* __CUDABE__ */

#endif /* __MATH_FUNCTIONS_DBL_PTX3_H__ */
