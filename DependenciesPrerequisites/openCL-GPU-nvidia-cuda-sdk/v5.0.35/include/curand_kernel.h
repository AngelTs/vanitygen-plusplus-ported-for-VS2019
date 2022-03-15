
 /* Copyright 2010-2012 NVIDIA Corporation.  All rights reserved.
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


#if !defined(CURAND_KERNEL_H_)
#define CURAND_KERNEL_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#include "curand.h"
#include "curandDiscrete.h"
#include "curand_precalc.h"
#include "curand_mrg32k3a.h"
#include "curand_mtgp32_kernel.h"
#include <math.h>

#define MAX_XOR_N (5)
#define SKIPAHEAD_BLOCKSIZE (4)
#define SKIPAHEAD_MASK ((1<<SKIPAHEAD_BLOCKSIZE)-1)
#define CURAND_2POW32_DOUBLE (4294967296.)
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_DOUBLE (2.3283064365386963e-10) 
#define CURAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define CURAND_2PI (6.2831855f)
#define CURAND_2POW53_INV_2PI_DOUBLE (1.1102230246251565e-16 * 6.2831853071795860)
#define CURAND_PI_DOUBLE  (3.1415926535897932)
#define CURAND_2PI_DOUBLE (6.2831853071795860)
#define CURAND_SQRT2 (-1.4142135f)
#define CURAND_SQRT2_DOUBLE (-1.4142135623730951)

#if !defined(QUALIFIERS)
#define QUALIFIERS static inline __device__
#endif

/* Test RNG */
/* This generator uses the formula:
   x_n = x_(n-1) + 1 mod 2^32
   x_0 = (unsigned int)seed * 3
   Subsequences are spaced 31337 steps apart.
*/
struct curandStateTest {
    unsigned int v;
};

/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateTest curandStateTest_t;
/** \endcond */

/* XORSHIFT FAMILY RNGs */
/* These generators are a family proposed by Marsaglia.  They keep state
   in 32 bit chunks, then use repeated shift and xor operations to scramble
   the bits.  The following generators are a combination of a simple Weyl
   generator with an N variable XORSHIFT generator.
*/

/* XORSHIFT RNG */
/* This generator uses the xorwow formula of
www.jstatsoft.org/v08/i14/paper page 5
Has period 2^192 - 2^32.
*/
/**
 * CURAND XORWOW state 
 */
struct curandStateXORWOW;

/*
 * Implementation details not in reference documentation */
struct curandStateXORWOW {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND XORWOW state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandStateXORWOW_t;

#define EXTRA_FLAG_NORMAL         0x00000001
#define EXTRA_FLAG_LOG_NORMAL     0x00000002
/** \endcond */

/* Combined Multiple Recursive Generators */
/* These generators are a family proposed by L'Ecuyer.  They keep state
   in sets of doubles, then use repeated modular arithmetic multiply operations 
   to scramble the bits in each set, and combine the result.
*/

/* MRG32k3a RNG */
/* This generator uses the MRG32k3A formula of
http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/streams4.pdf
Has period 2^191.
*/

/* moduli for the recursions */
/** \cond UNHIDE_DEFINES */
#define MRG32K3A_MOD1 4294967087.
#define MRG32K3A_MOD2 4294944443.

/* Constants used in generation */

#define MRG32K3A_A12  1403580.
#define MRG32K3A_A13N 810728.
#define MRG32K3A_A21  527612.
#define MRG32K3A_A23N 1370589.
#define MRG32K3A_NORM 2.328306549295728e-10
//
// #define MRG32K3A_BITS_NORM ((double)((POW32_DOUBLE-1.0)/MOD1))
//  above constant, used verbatim, rounds differently on some host systems.
#define MRG32K3A_BITS_NORM 1.000000048662


/* Constants for address manipulation */

#define MRG32K3A_SKIPUNITS_DOUBLES   (sizeof(struct sMRG32k3aSkipUnits)/sizeof(double))
#define MRG32K3A_SKIPSUBSEQ_DOUBLES  (sizeof(struct sMRG32k3aSkipSubSeq)/sizeof(double))
#define MRG32K3A_SKIPSEQ_DOUBLES     (sizeof(struct sMRG32k3aSkipSeq)/sizeof(double))
/** \endcond */




/**
 * CURAND MRG32K3A state 
 */
struct curandStateMRG32k3a;

/* Implementation details not in reference documentation */
struct curandStateMRG32k3a {
    double s1[3];
    double s2[3];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND MRG32K3A state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;
/** \endcond */

/* SOBOL QRNG */
/**
 * CURAND Sobol32 state 
 */
struct curandStateSobol32;

/* Implementation details not in reference documentation */
struct curandStateSobol32 {
    unsigned int i, x;
    unsigned int direction_vectors[32];
};

/**
 * CURAND Sobol32 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol32 curandStateSobol32_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol32 state 
 */
struct curandStateScrambledSobol32;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};

/**
 * CURAND Scrambled Sobol32 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;
/** \endcond */

/**
 * CURAND Sobol64 state 
 */
struct curandStateSobol64;

/* Implementation details not in reference documentation */
struct curandStateSobol64 {
    unsigned long long i, x;
    unsigned long long direction_vectors[64];
};

/**
 * CURAND Sobol64 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol64 curandStateSobol64_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol64 state 
 */
struct curandStateScrambledSobol64;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};

/**
 * CURAND Scrambled Sobol64 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;
/** \endcond */

/**
 * Default RNG
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;
/** \endcond */

/****************************************************************************/
/* Utility functions needed by RNGs */
/****************************************************************************/
/** \cond UNHIDE_UTILITIES */
/* 
   multiply vector by matrix, store in result
   matrix is n x n, measured in 32 bit units
   matrix is stored in row major order
   vector and result cannot be same pointer
*/
QUALIFIERS void __curand_matvec(unsigned int *vector, unsigned int *matrix, 
                                unsigned int *result, int n)
{
    for(int i = 0; i < n; i++) {
        result[i] = 0;
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < 32; j++) {
            if(vector[i] & (1 << j)) {
                for(int k = 0; k < n; k++) {
                    result[k] ^= matrix[n * (i * 32 + j) + k];
                }
            }
        }
    }
}


/* generate identity matrix */
QUALIFIERS void __curand_matidentity(unsigned int *matrix, int n)
{
    int r;
    for(int i = 0; i < n * 32; i++) {
        for(int j = 0; j < n; j++) {
            r = i & 31;
            if(i / 32 == j) {
                matrix[i * n + j] = (1 << r);
            } else {
                matrix[i * n + j] = 0;
            }
        }
    }
}

/* multiply matrixA by matrixB, store back in matrixA
   matrixA and matrixB must not be same matrix */
QUALIFIERS void __curand_matmat(unsigned int *matrixA, unsigned int *matrixB, int n)
{
    unsigned int result[MAX_XOR_N];
    for(int i = 0; i < n * 32; i++) {
        __curand_matvec(matrixA + i * n, matrixB, result, n);
        for(int j = 0; j < n; j++) {
            matrixA[i * n + j] = result[j];
        }
    }
}

/* copy vectorA to vector */
QUALIFIERS void __curand_veccopy(unsigned int *vector, unsigned int *vectorA, int n)
{
    for(int i = 0; i < n; i++) {
        vector[i] = vectorA[i];
    }
}

/* copy matrixA to matrix */
QUALIFIERS void __curand_matcopy(unsigned int *matrix, unsigned int *matrixA, int n)
{
    for(int i = 0; i < n * n * 32; i++) {
        matrix[i] = matrixA[i];
    }
}

/* compute matrixA to power p, store result in matrix */
QUALIFIERS void __curand_matpow(unsigned int *matrix, unsigned int *matrixA, 
                                unsigned long long p, int n)
{
    unsigned int matrixR[MAX_XOR_N * MAX_XOR_N * 32];
    unsigned int matrixS[MAX_XOR_N * MAX_XOR_N * 32];
    __curand_matidentity(matrix, n);
    __curand_matcopy(matrixR, matrixA, n);
    while(p) {
        if(p & 1) {
            __curand_matmat(matrix, matrixR, n);
        }
        __curand_matcopy(matrixS, matrixR, n);
        __curand_matmat(matrixR, matrixS, n);
        p >>= 1;
    }
}

/* Convert unsigned int to float, use no intrinsics */
QUALIFIERS float __curand_uint32AsFloat (unsigned int i)
{
    union {
        float f;
        unsigned int i;
    } xx;
    xx.i = i;
    return xx.f;
}

/* Convert two unsigned ints to double, use no intrinsics */
QUALIFIERS double __curand_hilouint32AsDouble (unsigned int hi, unsigned int lo)
{
    union {
        double f;
        unsigned int hi;
        unsigned int lo;
    } xx;
    xx.hi = hi;
    xx.lo = lo;
    return xx.f;
}

/* Convert unsigned int to float, as efficiently as possible */
QUALIFIERS float __curand_uint32_as_float(unsigned int x)
{
#if __CUDA_ARCH__ > 0
    return __int_as_float(x);
#elif !defined(__CUDA_ARCH__)
    return __curand_uint32AsFloat(x);
#endif
}

/*
QUALIFIERS double __curand_hilouint32_as_double(unsigned int hi, unsigned int lo)
{
#if __CUDA_ARCH__ > 0
    return __hiloint2double(hi, lo);
#elif !defined(__CUDA_ARCH__)
    return hilouint32AsDouble(hi, lo);
#endif
}
*/

/****************************************************************************/
/* Utility functions needed by MRG32k3a RNG                                 */
/* Matrix operations modulo some integer less than 2**32, done in           */
/* double precision floating point, with care not to overflow 53 bits       */
/****************************************************************************/

/* return i mod m.                                                          */
/* assumes i and m are integers represented accurately in doubles           */

QUALIFIERS double curand_MRGmod(double i, double m)
{
    double quo;
    double rem;
    quo = floor(i/m);
    rem = i - (quo*m);
    if (rem < 0.0) rem += m;
    return rem;    
}

/* Multiplication modulo m. Inputs i and j less than 2**32                  */
/* Ensure intermediate results do not exceed 2**53                          */

QUALIFIERS double curand_MRGmodMul(double i, double j, double m)
{
    double tempHi;
    double tempLo;
    
    tempHi = floor(i/131072.0);
    tempLo = i - (tempHi*131072.0);
    tempLo = curand_MRGmod( curand_MRGmod( (tempHi * j), m) * 131072.0 + curand_MRGmod(tempLo * j, m),m);

    if (tempLo < 0.0) tempLo += m;
    return tempLo;
}

/* multiply 3 by 3 matrices of doubles, modulo m                            */

QUALIFIERS void curand_MRGmatMul3x3(double i1[][3],double i2[][3],double o[][3],double m)
{
    int i,j;
    double temp[3][3];
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            temp[i][j] = ( curand_MRGmodMul(i1[i][0], i2[0][j], m) + 
                           curand_MRGmodMul(i1[i][1], i2[1][j], m) + 
                           curand_MRGmodMul(i1[i][2], i2[2][j], m));
            temp[i][j] = curand_MRGmod( temp[i][j], m );
        }
    }
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            o[i][j] = temp[i][j];
        }
    }
}

/* multiply 3 by 3 matrix times 3 by 1 vector of doubles, modulo m          */

QUALIFIERS void curand_MRGmatVecMul3x3( double i[][3], double v[], double m)
{  
    int k;
    double t[3];
    for (k = 0; k < 3; k++) {
        t[k] = ( curand_MRGmodMul(i[k][0], v[0], m) + 
                 curand_MRGmodMul(i[k][1], v[1], m) + 
                 curand_MRGmodMul(i[k][2], v[2], m) );
        t[k] = curand_MRGmod( t[k], m );
    } 
    for (k = 0; k < 3; k++) {
        v[k] = t[k];
    }

}

/* raise a 3 by 3 matrix of doubles to a 64 bit integer power pow, modulo m */
/* input is index zero of an array of 3 by 3 matrices m,                    */
/* each m = m[0]**(2**index)                                                */

QUALIFIERS void curand_MRGmatPow3x3( double in[][3][3], double o[][3], double m, unsigned long long pow )
{
    int i,j;
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            o[i][j] = 0;
            if ( i == j ) o[i][j] = 1;
        }
    }
    i = 0;
    curand_MRGmatVecMul3x3(o,o[0],m);
    while (pow) {
        if ( pow & 1ll ) {
             curand_MRGmatMul3x3(in[i], o, o, m);
        }
        i++;
        pow >>= 1;
    }
}

/* raise a 3 by 3 matrix of doubles to the power                            */
/* 2 to the power (pow modulo 191), modulo m                                */

QUALIFIERS void curnand_MRGmatPow2Pow3x3( double in[][3], double o[][3], double m, unsigned long pow )
{
    double temp[3][3];
    int i,j;
    pow = pow % 191;
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            temp[i][j] = in[i][j];
        }
    }
    while (pow) {
        curand_MRGmatMul3x3(temp, temp, temp, m);
        pow--;
    }
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            o[i][j] = temp[i][j];
        }
    }
}

/** \endcond */

/****************************************************************************/
/* Kernel implementations of RNGs                                           */
/****************************************************************************/

/* Test RNG */

QUALIFIERS void curand_init(unsigned long long seed, 
                                            unsigned long long subsequence, 
                                            unsigned long long offset, 
                                            curandStateTest_t *state)
{
    state->v = (unsigned int)(seed * 3) + (unsigned int)(subsequence * 31337) + \
                     (unsigned int)offset;
}

QUALIFIERS unsigned int curand(curandStateTest_t *state)
{
    unsigned int r = state->v++;
    return r;
}

QUALIFIERS void skipahead(unsigned long long n, curandStateTest_t *state)
{
    state->v += (unsigned int)n;
}

/* XORWOW RNG */

template <typename T, int n>
QUALIFIERS void __curand_generate_skipahead_matrix_xor(unsigned int matrix[])
{
    T state;
    // Generate matrix that advances one step
    // matrix has n * n * 32 32-bit elements
    // solve for matrix by stepping single bit states
    for(int i = 0; i < 32 * n; i++) {
        state.d = 0;
        for(int j = 0; j < n; j++) {
            state.v[j] = 0;
        }
        state.v[i / 32] = (1 << (i & 31));
        curand(&state);
        for(int j = 0; j < n; j++) {
            matrix[i * n + j] = state.v[j];
        }
    }
}

template <typename T, int n>
QUALIFIERS void _skipahead_scratch(unsigned long long x, T *state, unsigned int *scratch)
{
    // unsigned int matrix[n * n * 32];
    unsigned int *matrix = scratch;
    // unsigned int matrixA[n * n * 32];
    unsigned int *matrixA = scratch + (n * n * 32);
    // unsigned int vector[n];
    unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
    // unsigned int result[n];
    unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
    unsigned long long p = x;
    for(int i = 0; i < n; i++) {
        vector[i] = state->v[i];
    }
    int matrix_num = 0;
    while(p && (matrix_num < PRECALC_NUM_MATRICES - 1)) {
        for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec(vector, precalc_xorwow_offset_matrix[matrix_num], result, n);
#else
            __curand_matvec(vector, precalc_xorwow_offset_matrix_host[matrix_num], result, n);
#endif
            __curand_veccopy(vector, result, n);
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    if(p) {
#ifdef __CUDA_ARCH__
        __curand_matcopy(matrix, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
        __curand_matcopy(matrix, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
    }
    while(p) {
        for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
            __curand_matvec(vector, matrixA, result, n);
            __curand_veccopy(vector, result, n);
        }
        p >>= SKIPAHEAD_BLOCKSIZE;
        if(p) {
            for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
                __curand_matmat(matrix, matrixA, n);
                __curand_matcopy(matrixA, matrix, n);
            }
        }
    }
    for(int i = 0; i < n; i++) {
        state->v[i] = vector[i];
    }
    state->d += 362437 * (unsigned int)x;
}

template <typename T, int n>
QUALIFIERS void _skipahead_sequence_scratch(unsigned long long x, T *state, unsigned int *scratch)
{
    // unsigned int matrix[n * n * 32];
    unsigned int *matrix = scratch;
    // unsigned int matrixA[n * n * 32];
    unsigned int *matrixA = scratch + (n * n * 32);
    // unsigned int vector[n];
    unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
    // unsigned int result[n];
    unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
    unsigned long long p = x;
    for(int i = 0; i < n; i++) {
        vector[i] = state->v[i];
    }
    int matrix_num = 0;
    while(p && matrix_num < PRECALC_NUM_MATRICES - 1) {
        for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec(vector, precalc_xorwow_matrix[matrix_num], result, n);
#else
            __curand_matvec(vector, precalc_xorwow_matrix_host[matrix_num], result, n);
#endif
            __curand_veccopy(vector, result, n);
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    if(p) {
#ifdef __CUDA_ARCH__
        __curand_matcopy(matrix, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
        __curand_matcopy(matrix, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
    }
    while(p) {
        for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
            __curand_matvec(vector, matrixA, result, n);
            __curand_veccopy(vector, result, n);
        }
        p >>= SKIPAHEAD_BLOCKSIZE;
        if(p) {
            for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
                __curand_matmat(matrix, matrixA, n);
                __curand_matcopy(matrixA, matrix, n);
            }
        }
    }
    for(int i = 0; i < n; i++) {
        state->v[i] = vector[i];
    }
    /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

/**
 * \brief Update XORWOW state to skip \p n elements.
 *
 * Update the XORWOW state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead(unsigned long long n, curandStateXORWOW_t *state)
{
    unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
    _skipahead_scratch<curandStateXORWOW_t, 5>(n, state, (unsigned int *)scratch);
}

/**
 * \brief Update XORWOW state to skip ahead \p n subsequences.
 *
 * Update the XORWOW state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \f$ 2^{67} \f$ elements long, so this means the function will skip ahead
 * \f$ 2^{67} \cdot n\f$ elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_sequence(unsigned long long n, curandStateXORWOW_t *state)
{
    unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
    _skipahead_sequence_scratch<curandStateXORWOW_t, 5>(n, state, (unsigned int *)scratch);
}

QUALIFIERS void _curand_init_scratch(unsigned long long seed, 
                                     unsigned long long subsequence, 
                                     unsigned long long offset, 
                                     curandStateXORWOW_t *state,
                                     unsigned int *scratch)
{
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;
    _skipahead_sequence_scratch<curandStateXORWOW_t, 5>(subsequence, state, scratch);
    _skipahead_scratch<curandStateXORWOW_t, 5>(offset, state, scratch);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
}

/**
 * \brief Initialize XORWOW state.
 *
 * Initialize XORWOW state in \p state with the given \p seed, \p subsequence,
 * and \p offset.
 *
 * All input values of \p seed, \p subsequence, and \p offset are legal.  Large
 * values for \p subsequence and \p offset require more computation and so will
 * take more time to complete.
 *
 * A value of 0 for \p seed sets the state to the values of the original
 * published version of the \p xorwow algorithm.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(unsigned long long seed, 
                            unsigned long long subsequence, 
                            unsigned long long offset, 
                            curandStateXORWOW_t *state)
{
    unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
    _curand_init_scratch(seed, subsequence, offset, state, (unsigned int*)scratch);
}
/**
 * \brief Return 32-bits of pseudorandomness from an XORWOW generator.
 *
 * Return 32-bits of pseudorandomness from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
QUALIFIERS unsigned int curand(curandStateXORWOW_t *state)
{
    unsigned int t;
    t = (state->v[0] ^ (state->v[0] >> 2));
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] <<4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}

/* MRG32k3a RNG */

/* Base generator for MRG32k3a                                              */
/* note that the parameters have been selected such that intermediate       */
/* results stay within 53 bits                                              */


#if __CUDA_ARCH__ > 0
/*  nj's implementation */
QUALIFIERS double curand_MRG32k3a (curandStateMRG32k3a_t *state)
{
    const double m1 = 4294967087.;
    const double m2 = 4294944443.;
    const double a12  = 1403580.;
    const double a13n = 810728.;
    const double a21  = 527612.;
    const double a23n = 1370589.;

    const double rh1 =  2.3283065498378290e-010;  /* (1.0 / m1)__hi */
    const double rl1 = -1.7354913086174288e-026;  /* (1.0 / m1)__lo */
    const double rh2 =  2.3283188252407387e-010;  /* (1.0 / m2)__hi */
    const double rl2 =  2.4081018096503646e-026;  /* (1.0 / m2)__lo */

    double q, p1, p2;
    p1 = a12 * state->s1[1] - a13n * state->s1[0];
    q = trunc (fma (p1, rh1, p1 * rl1));
    p1 -= q * m1;  
    if (p1 < 0.0) p1 += m1;
    state->s1[0] = state->s1[1];   state->s1[1] = state->s1[2];   state->s1[2] = p1;
    p2 = a21 * state->s2[2] - a23n * state->s2[0];
    q = trunc (fma (p2, rh2, p2 * rl2));
    p2 -= q * m2;  
    if (p2 < 0.0) p2 += m2;
    state->s2[0] = state->s2[1];   state->s2[1] = state->s2[2];   state->s2[2] = p2;
    if (p1 <= p2) return (p1 - p2 + m1);
    else return (p1 - p2);
}
/* end nj's implementation */
#else
QUALIFIERS double curand_MRG32k3a(curandStateMRG32k3a_t *state)
{
    double p1,p2,r;
    p1 = (MRG32K3A_A12 * state->s1[1]) - (MRG32K3A_A13N * state->s1[0]);
    p1 = curand_MRGmod(p1, MRG32K3A_MOD1);
    if (p1 < 0.0) p1 += MRG32K3A_MOD1;
    state->s1[0] = state->s1[1]; 
    state->s1[1] = state->s1[2]; 
    state->s1[2] = p1;
    p2 = (MRG32K3A_A21 * state->s2[2]) - (MRG32K3A_A23N * state->s2[0]);
    p2 = curand_MRGmod(p2, MRG32K3A_MOD2);
    if (p2 < 0) p2 += MRG32K3A_MOD2;
    state->s2[0] = state->s2[1]; 
    state->s2[1] = state->s2[2]; 
    state->s2[2] = p2;
    r = p1 - p2;
    if (r <= 0) r += MRG32K3A_MOD1;
    return r;
}
#endif


/**
 * \brief Update MRG32k3a state to skip \p n elements.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead(unsigned long long n, curandStateMRG32k3a_t *state)
{
    double t[3][3];
#ifdef __CUDA_ARCH__
    curand_MRGmatPow3x3( mrg32k3aM1, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(mrg32k3aM2, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else
    curand_MRGmatPow3x3( mrg32k3aM1Host, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(mrg32k3aM2Host, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif
}

/**
 * \brief Update MRG32k3a state to skip ahead \p n subsequences.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \f$ 2^{76} \f$ elements long, so this means the function will skip ahead
 * \f$ 2^{76} \cdot n\f$ elements.
 *
 * Valid values of \p n are 0 to \f$ 2^51 \f$.  Note \p n will be masked to 51 bits
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update 
 */
QUALIFIERS void skipahead_subsequence(unsigned long long n, curandStateMRG32k3a_t *state)
{
    double t[3][3];
#ifdef __CUDA_ARCH__
    curand_MRGmatPow3x3( mrg32k3aM1SubSeq, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3( mrg32k3aM2SubSeq, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else    
    curand_MRGmatPow3x3( mrg32k3aM1SubSeqHost, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3( mrg32k3aM2SubSeqHost, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif    
}

/**
 * \brief Update MRG32k3a state to skip ahead \p n sequences.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n sequences.  Each
 * sequence is \f$ 2^{127} \f$ elements long, so this means the function will skip ahead
 * \f$ 2^{127} \cdot n\f$ elements. 
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of sequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_sequence(unsigned long long n, curandStateMRG32k3a_t *state)
{
    double t[3][3];
#ifdef __CUDA_ARCH__    
    curand_MRGmatPow3x3( mrg32k3aM1Seq, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(  mrg32k3aM2Seq, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else
    curand_MRGmatPow3x3( mrg32k3aM1SeqHost, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(  mrg32k3aM2SeqHost, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif    
}


/**
 * \brief Initialize MRG32k3a state.
 *
 * Initialize MRG32k3a state in \p state with the given \p seed, \p subsequence,
 * and \p offset.
 *
 * All input values of \p seed, \p subsequence, and \p offset are legal. 
 * \p subsequence will be truncated to 51 bits to avoid running into the next sequence
 *
 * A value of 0 for \p seed sets the state to the values of the original
 * published version of the \p MRG32k3a algorithm.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(unsigned long long seed, 
                            unsigned long long subsequence, 
                            unsigned long long offset, 
                            curandStateMRG32k3a_t *state)
{
    int i;
    for ( i=0; i<3; i++ ) {
        state->s1[i] = 12345.;
        state->s2[i] = 12345.;
    }
    if (seed != 0ull) {
        unsigned int x1 = ((unsigned int)seed) ^ 0x55555555UL;
        unsigned int x2 = (unsigned int)((seed >> 32) ^ 0xAAAAAAAAUL);
        state->s1[0] = curand_MRGmodMul(x1, state->s1[0], MRG32K3A_MOD1);
        state->s1[1] = curand_MRGmodMul(x2, state->s1[1], MRG32K3A_MOD1);
        state->s1[2] = curand_MRGmodMul(x1, state->s1[2], MRG32K3A_MOD1);
        state->s2[0] = curand_MRGmodMul(x2, state->s2[0], MRG32K3A_MOD2);
        state->s2[1] = curand_MRGmodMul(x1, state->s2[1], MRG32K3A_MOD2);
        state->s2[2] = curand_MRGmodMul(x2, state->s2[2], MRG32K3A_MOD2);
    } 
    skipahead_subsequence( subsequence, state );
    skipahead( offset, state );
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
}

/**
 * \brief Update Sobol32 state to skip \p n elements.
 *
 * Update the Sobol32 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
QUALIFIERS void skipahead(unsigned int n, T state)
{
    unsigned int i_gray;
    state->i += n;
    /* Convert state->i to gray code */
    i_gray = state->i ^ (state->i >> 1);
    for(unsigned int k = 0; k < 32; k++) {
        if(i_gray & (1 << k)) {
            state->x ^= state->direction_vectors[k];
        }
    }
    return;
}

/**
 * \brief Update Sobol64 state to skip \p n elements.
 *
 * Update the Sobol64 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
QUALIFIERS void skipahead(unsigned long long n, T state)
{
    unsigned long long i_gray;
    state->i += n;
    /* Convert state->i to gray code */
    i_gray = state->i ^ (state->i >> 1);
    for(unsigned k = 0; k < 64; k++) {
        if(i_gray & (1ULL << k)) {
            state->x ^= state->direction_vectors[k];
        }
    }
    return;
}

/**
 * \brief Initialize Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and 
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 * direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,                                            
                                            unsigned int offset, 
                                            curandStateSobol32_t *state)
{
    state->i = 0;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = 0;
    skipahead<curandStateSobol32_t *>(offset, state);
}
/**
 * \brief Initialize Scrambled Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and 
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,
                                            unsigned int scramble_c,
                                            unsigned int offset, 
                                            curandStateScrambledSobol32_t *state)
{
    state->i = 0;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = scramble_c;
    skipahead<curandStateScrambledSobol32_t *>(offset, state);
}

template<typename XT>
QUALIFIERS int __curand_find_trailing_zero(XT x)
{
#if __CUDA_ARCH__ > 0
    unsigned long long z = x;
    int y = __ffsll(~z) | 0x40;
    return (y - 1) & 0x3F;
#else
    unsigned long long z = x;
    int i = 1;
    while(z & 1) {
        i ++;
        z >>= 1;
    }
    return i - 1;
#endif
}
/**
 * \brief Initialize Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and 
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors64_t direction_vectors,
                                            unsigned long long offset, 
                                            curandStateSobol64_t *state)
{
    state->i = 0;
    for(int i = 0; i < 64; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = 0;
    skipahead<curandStateSobol64_t *>(offset, state);
}

template<typename PT>
QUALIFIERS void _skipahead_stride(int n_log2, PT state)
{
    /* Moving from i to i+2^n_log2 element in gray code is flipping two bits */
    unsigned int shifted_i = state->i >> n_log2;
    state->x ^= state->direction_vectors[n_log2 - 1];
    state->x ^= state->direction_vectors[
        __curand_find_trailing_zero(shifted_i) + n_log2];
    state->i += 1 << n_log2;

}
/**
 * \brief Initialize Scrambled Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and 
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors64_t direction_vectors,
                                            unsigned long long scramble_c,
                                            unsigned long long offset, 
                                            curandStateScrambledSobol64_t *state)
{
    state->i = 0;
    for(int i = 0; i < 64; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = scramble_c;
    skipahead<curandStateScrambledSobol64_t *>(offset, state);
}

/**
 * \brief Return 32-bits of quasirandomness from a Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

QUALIFIERS unsigned int curand(curandStateSobol32_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned int res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 32-bits of quasirandomness from a scrambled Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

QUALIFIERS unsigned int curand(curandStateScrambledSobol32_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned int res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

QUALIFIERS unsigned long long curand(curandStateSobol64_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned long long res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a scrambled Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

QUALIFIERS unsigned long long curand(curandStateScrambledSobol64_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned long long res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}


/******************************************************/

QUALIFIERS float _curand_uniform(unsigned int x)
{
    return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
}

QUALIFIERS float _curand_uniform(unsigned long long x)
{
    unsigned int t;
    t = (unsigned int)(x >> 32); 
    return t * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
}

QUALIFIERS double _curand_uniform_double(unsigned int x)
{
    return x * CURAND_2POW32_INV_DOUBLE + (CURAND_2POW32_INV_DOUBLE/2.0);
}

QUALIFIERS double _curand_uniform_double(unsigned long long x)
{
    return (x >> 11) * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
}

QUALIFIERS double _curand_uniform_double_hq(unsigned int x, unsigned int y)
{
    unsigned long long z = (unsigned long long)x ^ 
        ((unsigned long long)y << (53 - 32));
    return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
}

QUALIFIERS float curand_uniform(curandStateTest_t *state)
{
    return _curand_uniform(curand(state));
}

QUALIFIERS double curand_uniform_double(curandStateTest_t *state)
{
    return _curand_uniform_double(curand(state));
}

/**
 * \brief Return a uniformly distributed float from an XORWOW generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses one call.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateXORWOW_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from an XORWOW generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses exactly two calls.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateXORWOW_t *state)
{
    unsigned int x, y;
    x = curand(state);
    y = curand(state);
    return _curand_uniform_double_hq(x, y);
}

/**
 * \brief Return a uniformly distributed float from an MRG32k3a generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation returns up to 23 bits of mantissa, with the minimum 
 * return value \f$ 2^{-32} \f$ 
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateMRG32k3a_t *state)
{
    return ((float)(curand_MRG32k3a(state)*MRG32K3A_NORM));
}

/**
 * \brief Return a uniformly distributed double from an MRG32k3a generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned. 
 *
 * Note the implementation returns at most 32 random bits of mantissa as 
 * outlined in the seminal paper by L'Ecuyer.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateMRG32k3a_t *state)
{
    return curand_MRG32k3a(state)*MRG32K3A_NORM;
}

/**
 * \brief Return 32-bits of pseudorandomness from an MRG32k3a generator.
 *
 * Return 32-bits of pseudorandomness from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
QUALIFIERS unsigned int curand(curandStateMRG32k3a_t *state)
{
    double dRet;
    dRet = (double)curand_MRG32k3a(state)*(double)MRG32K3A_BITS_NORM;
    return (unsigned int)dRet;  
}

/**
 * \brief Return a uniformly distributed float from a MTGP32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateMtgp32_t *state)
{
    return _curand_uniform(curand(state));
}
/**
 * \brief Return a uniformly distributed double from a MTGP32 generator.
 *
 * Return a uniformly distributed double between \p 0.0f and \p 1.0f 
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0f and \p 1.0f
 */
QUALIFIERS double curand_uniform_double(curandStateMtgp32_t *state)
{
    return _curand_uniform_double(curand(state));
}

/**
 * \brief Return a uniformly distributed float from a Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateSobol32_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateSobol32_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateScrambledSobol32_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateScrambledSobol32_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateSobol64_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateSobol64_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateScrambledSobol64_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateScrambledSobol64_t *state)
{
    return _curand_uniform_double(curand(state));
}

QUALIFIERS float2 _curand_box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
    float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
#if __CUDA_ARCH__ > 0
    float s = sqrtf(-2.0f * logf(u));
    __sincosf(v, &result.x, &result.y);
#else
    float s = sqrtf(-2.0f * logf(u));
    result.x = sinf(v);
    result.y = cosf(v);
#endif
    result.x *= s;
    result.y *= s; 
    return result;
}

QUALIFIERS float2 curand_box_muller_mrg(curandStateMRG32k3a_t * state)
{        
    float x, y;
    x = curand_uniform(state);
    y = curand_uniform(state) * CURAND_2PI;
    float2 result;
#if __CUDA_ARCH__ > 0
    float s = sqrtf(-2.0f * logf(x));
    __sincosf(y, &result.x, &result.y);
#else
    float s = sqrtf(-2.0f * logf(x));
    result.x = sinf(y);
    result.y = cosf(y);
#endif
    result.x *= s;
    result.y *= s;
    return result;
}

QUALIFIERS double2 
_curand_box_muller_double(unsigned int x0, unsigned int x1, 
                          unsigned int y0, unsigned int y1)
{
    double2 result;
    unsigned long long zx = (unsigned long long)x0 ^ 
        ((unsigned long long)x1 << (53 - 32));
    double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
    unsigned long long zy = (unsigned long long)y0 ^ 
        ((unsigned long long)y1 << (53 - 32));
    double v = zy * (CURAND_2POW53_INV_DOUBLE*2.0) + CURAND_2POW53_INV_DOUBLE;
    double s = sqrt(-2.0 * log(u));
    
#if __CUDA_ARCH__ > 0
    sincospi(v, &result.x, &result.y);
#else 
    result.x = sin(v*CURAND_PI_DOUBLE);
    result.y = cos(v*CURAND_PI_DOUBLE);
#endif    
    result.x *= s;
    result.y *= s;

    return result;
}

QUALIFIERS double2 
curand_box_muller_mrg_double(curandStateMRG32k3a_t * state) 
{
    double x, y;
    double2 result;    
    x = curand_uniform_double(state);
    y = curand_uniform_double(state) * 2.0;

    double s = sqrt(-2.0 * log(x));
#if __CUDA_ARCH__ > 0
    sincospi(y, &result.x, &result.y);
#else
    result.x = sin(y*CURAND_PI_DOUBLE);
    result.y = cos(y*CURAND_PI_DOUBLE);
#endif   
    result.x *= s;
    result.y *= s;
    return result;
}

template <typename R>
QUALIFIERS float2 curand_box_muller(R *state)
{
    float2 result;
    unsigned int x = curand(state);
    unsigned int y = curand(state);
    result = _curand_box_muller(x, y);
    return result;
}

template <typename R>
QUALIFIERS double2 curand_box_muller_double(R *state)
{
    double2 result;
    unsigned int x0 = curand(state);
    unsigned int x1 = curand(state);
    unsigned int y0 = curand(state);
    unsigned int y1 = curand(state);
    result = _curand_box_muller_double(x0, x1, y0, y1);
    return result;
}

QUALIFIERS float _curand_normal_icdf(unsigned int x)
{
#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
    float s = CURAND_SQRT2;
    // Mirror to avoid loss of precision
    if(x > 0x80000000UL) {
        x = 0xffffffffUL - x;
        s = -s;
    }
    float p = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcinvf(2.0f * p);
#else
    return 0.0f;
#endif
}

QUALIFIERS float _curand_normal_icdf(unsigned long long x)
{
#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
    unsigned int t = (unsigned int)(x >> 32);
    float s = CURAND_SQRT2;
    // Mirror to avoid loss of precision
    if(t > 0x80000000UL) {
        t = 0xffffffffUL - t;
        s = -s;
    }
    float p = t * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcinvf(2.0f * p);
#else
    return 0.0f;
#endif
}

QUALIFIERS double _curand_normal_icdf_double(unsigned int x)
{
#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
    double s = CURAND_SQRT2_DOUBLE;
    // Mirror to avoid loss of precision
    if(x > 0x80000000UL) {
        x = 0xffffffffUL - x;
        s = -s;
    }
    double p = x * CURAND_2POW32_INV_DOUBLE + (CURAND_2POW32_INV_DOUBLE/2.0);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcinv(2.0 * p);
#else
    return 0.0;
#endif
}

QUALIFIERS double _curand_normal_icdf_double(unsigned long long x)
{
#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
    double s = CURAND_SQRT2_DOUBLE;
    x >>= 11;
    // Mirror to avoid loss of precision
    if(x > 0x10000000000000UL) {
        x = 0x1fffffffffffffUL - x;
        s = -s;
    }
    double p = x * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcinv(2.0 * p);
#else
    return 0.0;
#endif
}
 

/**
 * \brief Return a normally distributed float from an XORWOW generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::curand_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateXORWOW_t *state)
{
    if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        unsigned int x, y;
        x = curand(state);
        y = curand(state);
        float2 v = _curand_box_muller(x, y);
        state->boxmuller_extra = v.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return a normally distributed float from an MRG32k3a generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::curand_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateMRG32k3a_t *state)
{
    if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        float2 v = curand_box_muller_mrg(state);
        state->boxmuller_extra = v.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an XORWOW generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float2 curand_normal2(curandStateXORWOW_t *state)
{
    return curand_box_muller(state);
}

/**
 * \brief Return two normally distributed floats from an MRG32k3a generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float2 curand_normal2(curandStateMRG32k3a_t *state)
{
    return curand_box_muller_mrg(state);
}

/**
 * \brief Return a normally distributed float from a MTGP32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateMtgp32_t *state)
{
    return _curand_normal_icdf(curand(state));
}
/**
 * \brief Return a normally distributed float from a Sobol32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateSobol32_t *state)
{
    return _curand_normal_icdf(curand(state));
}

/**
 * \brief Return a normally distributed float from a scrambled Sobol32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateScrambledSobol32_t *state)
{
    return _curand_normal_icdf(curand(state));
}

/**
 * \brief Return a normally distributed float from a Sobol64 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateSobol64_t *state)
{
    return _curand_normal_icdf(curand(state));
}

/**
 * \brief Return a normally distributed float from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float curand_normal(curandStateScrambledSobol64_t *state)
{
    return _curand_normal_icdf(curand(state));
}

/**
 * \brief Return a normally distributed double from an XORWOW generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::curand_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateXORWOW_t *state)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
        unsigned int x0, x1, y0, y1;
        x0 = curand(state);
        x1 = curand(state);
        y0 = curand(state);
        y1 = curand(state);
        double2 v = _curand_box_muller_double(x0, x1, y0, y1);
        state->boxmuller_extra_double = v.y;
        state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}
/**
 * \brief Return a normally distributed double from an MRG32k3a generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::curand_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateMRG32k3a_t *state)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
        double2 v = curand_box_muller_mrg_double(state);
        state->boxmuller_extra_double = v.y;
        state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return two normally distributed doubles from an XORWOW generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double2 where each element is from a
 * distribution with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double2 curand_normal2_double(curandStateXORWOW_t *state)
{
    return curand_box_muller_double(state);
}

/**
 * \brief Return two normally distributed doubles from an MRG32k3a generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and
 * standard deviation \p 1.0 from the MRG32k3a generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double2 where each element is from a
 * distribution with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double2 curand_normal2_double(curandStateMRG32k3a_t *state)
{
    return curand_box_muller_mrg_double(state);
}

/**
 * \brief Return a normally distributed double from an MTGP32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateMtgp32_t *state)
{
    return _curand_normal_icdf_double(curand(state));
}

/**
 * \brief Return a normally distributed double from an Sobol32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateSobol32_t *state)
{
    return _curand_normal_icdf_double(curand(state));
}

/**
 * \brief Return a normally distributed double from a scrambled Sobol32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateScrambledSobol32_t *state)
{
    return _curand_normal_icdf_double(curand(state));
}

/**
 * \brief Return a normally distributed double from a Sobol64 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateSobol64_t *state)
{
    return _curand_normal_icdf_double(curand(state));
}

/**
 * \brief Return a normally distributed double from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double curand_normal_double(curandStateScrambledSobol64_t *state)
{
    return _curand_normal_icdf_double(curand(state));
}

// begin log_normal

/**
 * \brief Return a log-normally distributed float from an XORWOW generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state, 
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::curand_log_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state  - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateXORWOW_t *state, float mean, float stddev)
{
    if(state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        unsigned int x, y;
        x = curand(state);
        y = curand(state);
        float2 v = _curand_box_muller(x, y);
        state->boxmuller_extra = exp(mean + (stddev * v.y));
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an XORWOW generator.
 *
 * Return two log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float2 curand_log_normal2(curandStateXORWOW_t *state, float mean, float stddev)
{
    float2 v = curand_box_muller(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed float from an MRG32k3a generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state, 
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::curand_log_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state  - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateMRG32k3a_t *state, float mean, float stddev)
{
    if(state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        float2 v = curand_box_muller_mrg(state);
        state->boxmuller_extra = exp(mean + (stddev * v.y));
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an MRG32k3a generator.
 *
 * Return two log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float2 curand_log_normal2(curandStateMRG32k3a_t *state, float mean, float stddev)
{
    float2 v = curand_box_muller_mrg(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed float from an MTGP32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateMtgp32_t *state, float mean, float stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf(curand(state))));
}

/**
 * \brief Return a log-normally distributed float from a Sobol32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateSobol32_t *state, float mean, float stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf(curand(state))));
}
/**
 * \brief Return a log-normally distributed float from a scrambled Sobol32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateScrambledSobol32_t *state, float mean, float stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf(curand(state))));
}

/**
 * \brief Return a log-normally distributed float from a Sobol64 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results, then converts to log-normal
 * distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateSobol64_t *state, float mean, float stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf(curand(state))));
}

/**
 * \brief Return a log-normally distributed float from a scrambled Sobol64 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results, then converts to log-normal
 * distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float curand_log_normal(curandStateScrambledSobol64_t *state, float mean, float stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf(curand(state))));
}

/**
 * \brief Return a log-normally distributed double from an XORWOW generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::curand_log_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateXORWOW_t *state, double mean, double stddev)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_LOG_NORMAL) {
        unsigned int x0, x1, y0, y1;
        x0 = curand(state);
        x1 = curand(state);
        y0 = curand(state);
        y1 = curand(state);
        double2 v = _curand_box_muller_double(x0, x1, y0, y1);
        state->boxmuller_extra_double = exp(mean + (stddev * v.y));
        state->boxmuller_flag_double = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return two log-normally distributed doubles from an XORWOW generator.
 *
 * Return two log-normally distributed doubles derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, and transforms them to log-normal distribution,.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double2 curand_log_normal2_double(curandStateXORWOW_t *state, double mean, double stddev)
{
    double2 v = curand_box_muller_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed double from an MRG32k3a generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::curand_log_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateMRG32k3a_t *state, double mean, double stddev)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_LOG_NORMAL) {
        double2 v = curand_box_muller_mrg_double(state);
        state->boxmuller_extra_double = exp(mean + (stddev * v.y));
        state->boxmuller_flag_double = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return two log-normally distributed doubles from an MRG32k3a generator.
 *
 * Return two log-normally distributed doubles derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, and transforms them to log-normal distribution,.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double2 curand_log_normal2_double(curandStateMRG32k3a_t *state, double mean, double stddev)
{
    double2 v = curand_box_muller_mrg_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed double from an MTGP32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateMtgp32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf_double(curand(state))));
}

/**
 * \brief Return a log-normally distributed double from a Sobol32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateSobol32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf_double(curand(state))));
}

/**
 * \brief Return a log-normally distributed double from a scrambled Sobol32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateScrambledSobol32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf_double(curand(state))));
}

/**
 * \brief Return a log-normally distributed double from a Sobol64 generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateSobol64_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf_double(curand(state))));
}

/**
 * \brief Return a log-normally distributed double from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the inverse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double curand_log_normal_double(curandStateScrambledSobol64_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _curand_normal_icdf_double(curand(state))));
}

// begin poisson


#define CR_CUDART_PI               3.1415926535897931e+0
#define CR_CUDART_TWO_TO_52        4503599627370496.0

QUALIFIERS int __cr_isnan(double a)
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 > 0xffe0000000000000ull;
}


QUALIFIERS int __cr_isinf(double a)
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 == 0xffe0000000000000ull;
}


QUALIFIERS double __cr_copysign(double a, double b)
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvta, cvtb;

  cvta.d = a;
  cvtb.d = b;
  cvta.l = (cvta.l & 0x7fffffffffffffffULL) | (cvtb.l & 0x8000000000000000ULL);
  return cvta.d;
}

QUALIFIERS double __cr_rint(double a)
{
  double fa = fabs(a);
  double u = CR_CUDART_TWO_TO_52 + fa;
  if (fa >= CR_CUDART_TWO_TO_52) {
    u = a;
  } else {
    u = u - CR_CUDART_TWO_TO_52;
    u = __cr_copysign (u, a);
  }
  return u;  
}

QUALIFIERS float __cr_rsqrt(float a) 
{
#ifdef __CUDA_ARCH__
    asm ("rsqrt.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
#else
    a = 1.0 / sqrt (a);
#endif
    return a;
}

QUALIFIERS float __cr_exp (float a)
{
#ifdef __CUDA_ARCH__
    a = a * 1.4426950408889634074;
    asm ("ex2.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
#else
    a = exp (a);
#endif
    return a;
}

QUALIFIERS float __cr_log (float a)
{
#ifdef __CUDA_ARCH__
    asm ("lg2.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
    a = a * 0.69314718055994530942;
#else
    a = log (a);
#endif
    return a;
}

QUALIFIERS float __cr_rcp (float a)
{
#ifdef __CUDA_ARCH__
    asm ("rcp.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
#else
    a = 1.0 / a;
#endif
    return a;
}

/* Computes regularized gamma function:  gammainc(a,x)/gamma(a) */
QUALIFIERS float __cr_pgammainc (float a, float x)
{
    float t, alpha, beta;

    /* First level parametrization constants */
    float ma1 = 1.43248035075540910,
          ma2 = 0.12400979329415655,
          ma3 = 0.00025361074907033,
          mb1 = 0.21096734870196546,
          mb2 = 1.97381164089999420,
          mb3 = 0.94201734077887530;

    /* Second level parametrization constants (depends only on a) */

    alpha = __cr_rsqrt (a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = __cr_rsqrt (a - mb2);
    beta = mb1 * beta + mb3;

    /* Final approximation (depends on a and x) */

    t = a - x;
    t = alpha * t - beta;
    t = 1.0 + __cr_exp (t);
    t = t * t;
    t = __cr_rcp (t);

    /* Negative a,x or a,x=NAN requires special handling */
    //t = !(x > 0 && a >= 0) ? 0.0 : t;

    return t;
}

/* Computes inverse of pgammainc */
QUALIFIERS float __cr_pgammaincinv (float a, float y)
{
    float t, alpha, beta;

    /* First level parametrization constants */

    float ma1 = 1.43248035075540910,
          ma2 = 0.12400979329415655,
          ma3 = 0.00025361074907033,
          mb1 = 0.21096734870196546,
          mb2 = 1.97381164089999420,
          mb3 = 0.94201734077887530;

    /* Second level parametrization constants (depends only on a) */

    alpha = __cr_rsqrt (a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = __cr_rsqrt (a - mb2);
    beta = mb1 * beta + mb3;

    /* Final approximation (depends on a and y) */

    t = __cr_rsqrt (y) - 1.0;
    t = __cr_log (t);
    t = beta + t;
    t = - t * __cr_rcp (alpha) + a;
    /* Negative a,x or a,x=NAN requires special handling */
    //t = !(y > 0 && a >= 0) ? 0.0 : t;
    return t;
}

static __constant__ double __cr_lgamma_table [] = { 0.000000000000000000e-1,
    0.000000000000000000e-1,
    6.931471805599453094e-1,
    1.791759469228055001e0,
    3.178053830347945620e0,
    4.787491742782045994e0,
    6.579251212010100995e0,
    8.525161361065414300e0,
    1.060460290274525023e1};


QUALIFIERS double __cr_lgamma_integer(int a)
{
    double s;
    double t;
    double fa = fabs((float)a);
    double sum;

    if (a > 8) {
        /* Stirling approximation; coefficients from Hart et al, "Computer 
         * Approximations", Wiley 1968. Approximation 5404. 
         */
        s = 1.0 / fa;
        t = s * s;
        sum =          -0.1633436431e-2;
        sum = sum * t + 0.83645878922e-3;
        sum = sum * t - 0.5951896861197e-3;
        sum = sum * t + 0.793650576493454e-3;
        sum = sum * t - 0.277777777735865004e-2;
        sum = sum * t + 0.833333333333331018375e-1;
        sum = sum * s + 0.918938533204672;
        s = 0.5 * log (fa);
        t = fa - 0.5;
        s = s * t;
        t = s - fa;
        s = s + sum;
        t = t + s;
        return t;
    } else {
#ifdef __CUDA_ARCH__
        return __cr_lgamma_table [(int) fa-1];
#else
        switch(a) {
            case 1: return 0.000000000000000000e-1;
            case 2: return 0.000000000000000000e-1;
            case 3: return 6.931471805599453094e-1;
            case 4: return 1.791759469228055001e0;
            case 5: return 3.178053830347945620e0;
            case 6: return 4.787491742782045994e0;
            case 7: return 6.579251212010100995e0;
            case 8: return 8.525161361065414300e0;
            default: return 1.060460290274525023e1;
        }
#endif
    }
}

QUALIFIERS double __cr_lgamma(double a)
{
  double s;
  double t;
  double i;
  double fa;
  double sum;
  long long int quot;

  if (__cr_isnan(a) || __cr_isinf(a)) {
    return a * a;
  }
  fa = fabs(a);
  if (fa >= 3.0) {
    if (fa >= 8.0) {
      /* Stirling approximation; coefficients from Hart et al, "Computer 
       * Approximations", Wiley 1968. Approximation 5404. 
       */
      s = 1.0 / fa;
      t = s * s;
      sum =          -0.1633436431e-2;
      sum = sum * t + 0.83645878922e-3;
      sum = sum * t - 0.5951896861197e-3;
      sum = sum * t + 0.793650576493454e-3;
      sum = sum * t - 0.277777777735865004e-2;
      sum = sum * t + 0.833333333333331018375e-1;
      sum = sum * s + 0.918938533204672;
      s = 0.5 * log (fa);
      t = fa - 0.5;
      s = s * t;
      t = s - fa;
      s = s + sum;
      t = t + s;
    } else {
      i = fa - 3.0;
      s =        -4.02412642744125560E+003;
      s = s * i - 2.97693796998962000E+005;
      s = s * i - 6.38367087682528790E+006;
      s = s * i - 5.57807214576539320E+007;
      s = s * i - 2.24585140671479230E+008;
      s = s * i - 4.70690608529125090E+008;
      s = s * i - 7.62587065363263010E+008;
      s = s * i - 9.71405112477113250E+008;
      t =     i - 1.02277248359873170E+003;
      t = t * i - 1.34815350617954480E+005;
      t = t * i - 4.64321188814343610E+006;
      t = t * i - 6.48011106025542540E+007;
      t = t * i - 4.19763847787431360E+008;
      t = t * i - 1.25629926018000720E+009;
      t = t * i - 1.40144133846491690E+009;
      t = s / t;
      t = t + i;
    }
  } else if (fa >= 1.5) {
    i = fa - 2.0;
    t =         9.84839283076310610E-009;
    t = t * i - 6.69743850483466500E-008;
    t = t * i + 2.16565148880011450E-007;
    t = t * i - 4.86170275781575260E-007;
    t = t * i + 9.77962097401114400E-007;
    t = t * i - 2.03041287574791810E-006;
    t = t * i + 4.36119725805364580E-006;
    t = t * i - 9.43829310866446590E-006;
    t = t * i + 2.05106878496644220E-005;
    t = t * i - 4.49271383742108440E-005;
    t = t * i + 9.94570466342226000E-005;
    t = t * i - 2.23154589559238440E-004;
    t = t * i + 5.09669559149637430E-004;
    t = t * i - 1.19275392649162300E-003;
    t = t * i + 2.89051032936815490E-003;
    t = t * i - 7.38555102806811700E-003;
    t = t * i + 2.05808084278121250E-002;
    t = t * i - 6.73523010532073720E-002;
    t = t * i + 3.22467033424113040E-001;
    t = t * i + 4.22784335098467190E-001;
    t = t * i;
  } else if (fa >= 0.7) {
    i = 1.0 - fa;
    t =         1.17786911519331130E-002;  
    t = t * i + 3.89046747413522300E-002;
    t = t * i + 5.90045711362049900E-002;
    t = t * i + 6.02143305254344420E-002;
    t = t * i + 5.61652708964839180E-002;
    t = t * i + 5.75052755193461370E-002;
    t = t * i + 6.21061973447320710E-002;
    t = t * i + 6.67614724532521880E-002;
    t = t * i + 7.14856037245421020E-002;
    t = t * i + 7.69311251313347100E-002;
    t = t * i + 8.33503129714946310E-002;
    t = t * i + 9.09538288991182800E-002;
    t = t * i + 1.00099591546322310E-001;
    t = t * i + 1.11334278141734510E-001;
    t = t * i + 1.25509666613462880E-001;
    t = t * i + 1.44049896457704160E-001;
    t = t * i + 1.69557177031481600E-001;
    t = t * i + 2.07385551032182120E-001;
    t = t * i + 2.70580808427600350E-001;
    t = t * i + 4.00685634386517050E-001;
    t = t * i + 8.22467033424113540E-001;
    t = t * i + 5.77215664901532870E-001;
    t = t * i;
  } else {
    t =         -9.04051686831357990E-008;
    t = t * fa + 7.06814224969349250E-007;
    t = t * fa - 3.80702154637902830E-007;
    t = t * fa - 2.12880892189316100E-005;
    t = t * fa + 1.29108470307156190E-004;
    t = t * fa - 2.15932815215386580E-004;
    t = t * fa - 1.16484324388538480E-003;
    t = t * fa + 7.21883433044470670E-003;
    t = t * fa - 9.62194579514229560E-003;
    t = t * fa - 4.21977386992884450E-002;
    t = t * fa + 1.66538611813682460E-001;
    t = t * fa - 4.20026350606819980E-002;
    t = t * fa - 6.55878071519427450E-001;
    t = t * fa + 5.77215664901523870E-001;
    t = t * fa;
    t = t * fa + fa;
    t = -log (t);
  }
  if (a >= 0.0) return t;
  if (fa < 1e-19) return -log(fa);
  
  i = floor(fa);  

  if (fa == i) return 1.0 / (fa - i); /* a is an integer: return infinity */

  i = __cr_rint (2.0 * fa);
  quot = (long long int)i;
  i = fa - 0.5 * i;
  i = i * CR_CUDART_PI;
  if (quot & 1) {
    i = cos(i);
  } else {
    i = sin(i);
  }
  i = fabs(i);
  t = log(CR_CUDART_PI / (i * fa)) - t;
  return t;
}

#define KNUTH_FLOAT_CONST 60.0
template <typename T>
// Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming, Volume 2
QUALIFIERS unsigned int curand_poisson_knuth(T *state, float lambda)
{
  unsigned int k = 0;
  float p=exp(lambda);
  do{
      k++;
      p *= curand_uniform(state);
  }while (p > 1.0);
  return k-1;
}

template <typename T>
// Marsaglia, Tsang, Wang Journal of Statistical Software, square histogram.
QUALIFIERS unsigned int _curand_M2_double(T x, curandDistributionM2Shift_t distributionM2)
{
    double u = _curand_uniform_double(x);
    int j = (int) floor(distributionM2->length*u);  
//    int result;
    if (u < distributionM2->histogram->V[j]) return distributionM2->shift + j;
    return distributionM2->shift + distributionM2->histogram->K[j];
}


template <typename STATE>
QUALIFIERS unsigned int curand_M2_double(STATE *state, curandDistributionM2Shift_t distributionM2)
{
    return _curand_M2_double(curand(state), distributionM2);
}

template <typename T>
QUALIFIERS unsigned int _curand_binary_search_double(T x, curandDistributionShift_t distribution)
{
    double u = _curand_uniform_double(x);
    int min = 0;
    int max = distribution->length-1;
    do{
        int mid = (max + min)/2;
        if (u <= distribution->probability[mid]){
            max = mid;
        }else{
            min = mid+1;
        }
    }while (min < max);
    return distribution->shift + min;
}

template <typename STATE>
QUALIFIERS unsigned int curand_binary_search_double(STATE *state, curandDistributionShift_t distribution)
{
    return _curand_binary_search_double(curand(state), distribution);
}


#define MAGIC_DOUBLE_CONST 500.0
template <typename T>
//George S. Fishman Discrete-event simulation: modeling, programming, and analysis
QUALIFIERS unsigned int _curand_poisson_ITR_double(T x, double lambda)
{
  double L,p = 1.0;
  double q = 1.0;
  unsigned int k = 0;
  int pow=0;
  double u = _curand_uniform_double(x);
  do{
      if (lambda > (double)(pow+MAGIC_DOUBLE_CONST)){
          L = exp(-MAGIC_DOUBLE_CONST);
      }else{
          L = exp((double)(pow - lambda));
      }
      p *= L;
      q *= L;
      pow += (int) MAGIC_DOUBLE_CONST;
      while (u > q){
        k++;
        p *= ((double)lambda / (double) k);
        q += p;
      }
  }while((double)pow < lambda);
  return k;
}

template <typename T>
/* Rejection Method for Poisson distribution based on gammainc approximation */
QUALIFIERS unsigned int curand_poisson_gammainc(T state, float lambda){
    float y, x, t, z,v;
    float logl = __cr_log (lambda);
    while (true) {
        y = curand_uniform (state);
        x = __cr_pgammaincinv (lambda, y);
        x = floor (x);
        z = curand_uniform (state);
        v = (__cr_pgammainc (lambda, x + 1.0) - __cr_pgammainc (lambda, x)) * 1.25;
        z = z*v;
        t = __cr_exp (-lambda + x * logl - __cr_lgamma_integer ((int)(1.0 + x)));
        if ((z < t) && (v>=1e-20))
            break;
    }
    return (unsigned int)x;
}

// Note below that the round to nearest integer, where needed,is done in line with code that
// assumes the range of values is < 2**32

template <typename T>
QUALIFIERS unsigned int _curand_poisson(T x, double lambda)
{
    if (lambda < 1000)
        return _curand_poisson_ITR_double(x, lambda);
    return (unsigned int)((sqrt(lambda) * _curand_normal_icdf_double(x)) + lambda + 0.5); //Round to nearest
}

template <typename T>
QUALIFIERS unsigned int _curand_poisson_from_normal(T x, double lambda)
{
    return (unsigned int)((sqrt(lambda) * _curand_normal_icdf(x)) + lambda + 0.5); //Round to nearest
}

template <typename STATE>
QUALIFIERS unsigned int curand_poisson_from_normal(STATE state, double lambda)
{
    return (unsigned int)((sqrt(lambda) * curand_normal(state)) + lambda + 0.5); //Round to nearest
}

template <typename T>
QUALIFIERS unsigned int _curand_discrete(T x, curandDiscreteDistribution_t discrete_distribution){
    if (discrete_distribution->method == CURAND_M2){
        return _curand_M2_double(x, discrete_distribution->M2);
    }
    return (unsigned int)((discrete_distribution->stddev * _curand_normal_icdf_double(x)) + discrete_distribution->mean + 0.5);
}


template <typename STATE>
QUALIFIERS unsigned int curand__discrete(STATE state, curandDiscreteDistribution_t discrete_distribution){
    if (discrete_distribution->method == CURAND_M2){
        return curand_M2_double(state, discrete_distribution->M2);
    }
    return (unsigned int)((discrete_distribution->stddev * curand_normal_double(state)) + discrete_distribution->mean + 0.5); //Round to nearest
}


/**
 * \brief Return a Poisson-distributed unsigned int from a XORWOW generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the XORWOW generator in \p state,
 * increment the  position of the generator by a variable amount, depending 
 * on the algorithm used.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateXORWOW_t *state, double lambda)
{
    if (lambda < 64)
        return curand_poisson_knuth(state, lambda);
    if (lambda > 4000)
        return (unsigned int)((sqrt(lambda) * curand_normal_double(state)) + lambda + 0.5); //Round to nearest
    return curand_poisson_gammainc(state, lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a MRG32k3A generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the MRG32k3a generator in \p state,
 * increment the position of the generator by a variable amount, depending 
 * on the algorithm used.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateMRG32k3a_t *state, double lambda)
{
    if (lambda < 64)
        return curand_poisson_knuth(state, lambda);
    if (lambda > 4000)
        return (unsigned int)((sqrt(lambda) * curand_normal_double(state)) + lambda + 0.5); //Round to nearest
    return curand_poisson_gammainc(state, lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a MTGP32 generator.
 *
 * Return a single int from a Poisson
 * distribution with lambda \p lambda from the MTGP32 generator in \p state,
 * increment the position of the generator by one.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateMtgp32_t *state, double lambda)
{
    return _curand_poisson(curand(state), lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a Sobol32 generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the Sobol32 generator in \p state,
 * increment the position of the generator by one.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */

QUALIFIERS unsigned int curand_poisson(curandStateSobol32_t *state, double lambda)
{
    return _curand_poisson(curand(state), lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a scrambled Sobol32 generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the scrambled Sobol32 generator in \p state,
 * increment the position of the generator by one.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateScrambledSobol32_t *state, double lambda)
{
    return _curand_poisson(curand(state), lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a Sobol64 generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateSobol64_t *state, double lambda)
{
    return _curand_poisson(curand(state), lambda);
}

/**
 * \brief Return a Poisson-distributed unsigned int from a scrambled Sobol64 generator.
 *
 * Return a single unsigned int from a Poisson
 * distribution with lambda \p lambda from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param lambda - Lambda of the Poisson distribution
 *
 * \return Poisson-distributed unsigned int with lambda \p lambda
 */
QUALIFIERS unsigned int curand_poisson(curandStateScrambledSobol64_t *state, double lambda)
{
    return _curand_poisson(curand(state), lambda);
}

/*
 * \brief Return a discrete distributed unsigned int from a XORWOW generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateXORWOW_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a MRG32k3a generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateMRG32k3a_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a MTGP32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the MTGP32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateMtgp32_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a Sobol32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateSobol32_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a scrambled Sobol32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateScrambledSobol32_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a Sobol64 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateSobol64_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a scrambled Sobol64 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int curand_discrete(curandStateScrambledSobol64_t *state, curandDiscreteDistribution_t discrete_distribution)
{
    return curand__discrete(state, discrete_distribution);
}

__device__ static unsigned int *__get_precalculated_matrix(int n)
{
    if(n == 0) {
        return precalc_xorwow_matrix[n];
    }
    if(n == 2) {
        return precalc_xorwow_offset_matrix[n];
    }
    return precalc_xorwow_matrix[n];
}

__host__ static unsigned int *__get_precalculated_matrix_host(int n)
{
    if(n == 1) {
        return precalc_xorwow_matrix_host[n];
    }
    if(n == 3) {
        return precalc_xorwow_offset_matrix_host[n];
    }
    return precalc_xorwow_matrix_host[n];
}

__device__ static double *__get_mrg32k3a_matrix(int n)
{
    if(n == 0) {
        return mrg32k3aM1[n][0];
    }
    if(n == 2) {
        return mrg32k3aM2[n][0];
    }
    if(n == 4) {
        return mrg32k3aM1SubSeq[n][0];
    }
    if(n == 6) {
        return mrg32k3aM2SubSeq[n][0];
    }
    if(n == 8) {
        return mrg32k3aM1Seq[n][0];
    }
    if(n == 10) {
        return mrg32k3aM2Seq[n][0];
    }
    return mrg32k3aM1[n][0];
}

__host__ static double *__get_mrg32k3a_matrix_host(int n)
{
    if(n == 1) {
        return mrg32k3aM1Host[n][0];
    }
    if(n == 3) {
        return mrg32k3aM2Host[n][0];
    }
    if(n == 5) {
        return mrg32k3aM1SubSeqHost[n][0];
    }
    if(n == 7) {
        return mrg32k3aM2SubSeqHost[n][0];
    }
    if(n == 9) {
        return mrg32k3aM1SeqHost[n][0];
    }
    if(n == 11) {
        return mrg32k3aM2SeqHost[n][0];
    }
    return mrg32k3aM1Host[n][0];
}

__host__ static double *__get__cr_lgamma_table_host(void) {
    return __cr_lgamma_table;
}

/** @} */

#endif // !defined(CURAND_KERNEL_H_)
