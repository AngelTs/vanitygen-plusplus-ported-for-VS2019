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
 
/*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines) 
 * on top of the CUDA runtime. 
 */

#if !defined(CUBLAS_H_)
#define CUBLAS_H_

#include <cuda_runtime.h>

#ifndef CUBLASAPI
#ifdef _WIN32
#define CUBLASAPI __stdcall
#else
#define CUBLASAPI 
#endif
#endif

#include "driver_types.h"
#include "cuComplex.h"   /* import complex data type */

#include "cublas_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* CUBLAS data types */
#define cublasStatus cublasStatus_t

cublasStatus CUBLASAPI cublasInit (void);
cublasStatus CUBLASAPI cublasShutdown (void);
cublasStatus CUBLASAPI cublasGetError (void);

cublasStatus CUBLASAPI cublasGetVersion(int *version);
cublasStatus CUBLASAPI cublasAlloc (int n, int elemSize, void **devicePtr);

cublasStatus CUBLASAPI cublasFree (void *devicePtr);


cublasStatus CUBLASAPI cublasSetKernelStream (cudaStream_t stream);



/* ---------------- CUBLAS BLAS1 functions ---------------- */
/* NRM2 */
float CUBLASAPI cublasSnrm2 (int n, const float *x, int incx);
double CUBLASAPI cublasDnrm2 (int n, const double *x, int incx);
float CUBLASAPI cublasScnrm2 (int n, const cuComplex *x, int incx);
double CUBLASAPI cublasDznrm2 (int n, const cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* DOT */
float CUBLASAPI cublasSdot (int n, const float *x, int incx, const float *y, 
                            int incy);
double CUBLASAPI cublasDdot (int n, const double *x, int incx, const double *y, 
                            int incy);
cuComplex CUBLASAPI cublasCdotu (int n, const cuComplex *x, int incx, const cuComplex *y, 
                            int incy);
cuComplex CUBLASAPI cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y, 
                            int incy);
cuDoubleComplex CUBLASAPI cublasZdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, 
                            int incy);
cuDoubleComplex CUBLASAPI cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, 
                            int incy);
/*------------------------------------------------------------------------*/
/* SCAL */
void CUBLASAPI cublasSscal (int n, float alpha, float *x, int incx);
void CUBLASAPI cublasDscal (int n, double alpha, double *x, int incx);
void CUBLASAPI cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx);
void CUBLASAPI cublasZscal (int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx);

void CUBLASAPI cublasCsscal (int n, float alpha, cuComplex *x, int incx);
void CUBLASAPI cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* AXPY */
void CUBLASAPI cublasSaxpy (int n, float alpha, const float *x, int incx, 
                            float *y, int incy);
void CUBLASAPI cublasDaxpy (int n, double alpha, const double *x, 
                            int incx, double *y, int incy);
void CUBLASAPI cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, 
                            int incx, cuComplex *y, int incy);
void CUBLASAPI cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x, 
                            int incx, cuDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/
/* COPY */
void CUBLASAPI cublasScopy (int n, const float *x, int incx, float *y, 
                            int incy);
void CUBLASAPI cublasDcopy (int n, const double *x, int incx, double *y, 
                            int incy);
void CUBLASAPI cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y,
                            int incy);
void CUBLASAPI cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y,
                            int incy);
/*------------------------------------------------------------------------*/
/* SWAP */
void CUBLASAPI cublasSswap (int n, float *x, int incx, float *y, int incy);
void CUBLASAPI cublasDswap (int n, double *x, int incx, double *y, int incy);
void CUBLASAPI cublasCswap (int n, cuComplex *x, int incx, cuComplex *y, int incy);
void CUBLASAPI cublasZswap (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);           
/*------------------------------------------------------------------------*/
/* AMAX */
int CUBLASAPI cublasIsamax (int n, const float *x, int incx);
int CUBLASAPI cublasIdamax (int n, const double *x, int incx);
int CUBLASAPI cublasIcamax (int n, const cuComplex *x, int incx);
int CUBLASAPI cublasIzamax (int n, const cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* AMIN */
int CUBLASAPI cublasIsamin (int n, const float *x, int incx);
int CUBLASAPI cublasIdamin (int n, const double *x, int incx);

int CUBLASAPI cublasIcamin (int n, const cuComplex *x, int incx);
int CUBLASAPI cublasIzamin (int n, const cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* ASUM */
float CUBLASAPI cublasSasum (int n, const float *x, int incx);
double CUBLASAPI cublasDasum (int n, const double *x, int incx);
float CUBLASAPI cublasScasum (int n, const cuComplex *x, int incx);
double CUBLASAPI cublasDzasum (int n, const cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* ROT */
void CUBLASAPI cublasSrot (int n, float *x, int incx, float *y, int incy, 
                           float sc, float ss);
void CUBLASAPI cublasDrot (int n, double *x, int incx, double *y, int incy, 
                           double sc, double ss);
void CUBLASAPI cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, 
                           int incy, float c, cuComplex s);
void CUBLASAPI cublasZrot (int n, cuDoubleComplex *x, int incx, 
                           cuDoubleComplex *y, int incy, double sc, 
                           cuDoubleComplex cs);
void CUBLASAPI cublasCsrot (int n, cuComplex *x, int incx, cuComplex *y,
                            int incy, float c, float s);
void CUBLASAPI cublasZdrot (int n, cuDoubleComplex *x, int incx, 
                            cuDoubleComplex *y, int incy, double c, double s);
/*------------------------------------------------------------------------*/
/* ROTG */
void CUBLASAPI cublasSrotg (float *sa, float *sb, float *sc, float *ss);
void CUBLASAPI cublasDrotg (double *sa, double *sb, double *sc, double *ss);
void CUBLASAPI cublasCrotg (cuComplex *ca, cuComplex cb, float *sc,
                                     cuComplex *cs);                                     
void CUBLASAPI cublasZrotg (cuDoubleComplex *ca, cuDoubleComplex cb, double *sc,
                            cuDoubleComplex *cs);                                                               
/*------------------------------------------------------------------------*/
/* ROTM */
void CUBLASAPI cublasSrotm(int n, float *x, int incx, float *y, int incy, 
                           const float* sparam);
void CUBLASAPI cublasDrotm(int n, double *x, int incx, double *y, int incy, 
                           const double* sparam);
/*------------------------------------------------------------------------*/
/* ROTMG */
void CUBLASAPI cublasSrotmg (float *sd1, float *sd2, float *sx1, 
                             const float *sy1, float* sparam);
void CUBLASAPI cublasDrotmg (double *sd1, double *sd2, double *sx1, 
                             const double *sy1, double* sparam);
                           
/* --------------- CUBLAS BLAS2 functions  ---------------- */
/* GEMV */
void CUBLASAPI cublasSgemv (char trans, int m, int n, float alpha,
                            const float *A, int lda, const float *x, int incx,
                            float beta, float *y, int incy);
void CUBLASAPI cublasDgemv (char trans, int m, int n, double alpha,
                            const double *A, int lda, const double *x, int incx,
                            double beta, double *y, int incy);
void CUBLASAPI cublasCgemv (char trans, int m, int n, cuComplex alpha,
                            const cuComplex *A, int lda, const cuComplex *x, int incx,
                            cuComplex beta, cuComplex *y, int incy);
void CUBLASAPI cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha,
                            const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                            cuDoubleComplex beta, cuDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/
/* GBMV */
void CUBLASAPI cublasSgbmv (char trans, int m, int n, int kl, int ku, 
                            float alpha, const float *A, int lda, 
                            const float *x, int incx, float beta, float *y, 
                            int incy);
void CUBLASAPI cublasDgbmv (char trans, int m, int n, int kl, int ku, 
                            double alpha, const double *A, int lda, 
                            const double *x, int incx, double beta, double *y, 
                            int incy);
void CUBLASAPI cublasCgbmv (char trans, int m, int n, int kl, int ku, 
                            cuComplex alpha, const cuComplex *A, int lda, 
                            const cuComplex *x, int incx, cuComplex beta, cuComplex *y, 
                            int incy);
void CUBLASAPI cublasZgbmv (char trans, int m, int n, int kl, int ku, 
                            cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, 
                            const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, 
                            int incy);                  
/*------------------------------------------------------------------------*/
/* TRMV */
void CUBLASAPI cublasStrmv (char uplo, char trans, char diag, int n, 
                            const float *A, int lda, float *x, int incx);
void CUBLASAPI cublasDtrmv (char uplo, char trans, char diag, int n, 
                            const double *A, int lda, double *x, int incx);
void CUBLASAPI cublasCtrmv (char uplo, char trans, char diag, int n, 
                            const cuComplex *A, int lda, cuComplex *x, int incx);
void CUBLASAPI cublasZtrmv (char uplo, char trans, char diag, int n, 
                            const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TBMV */
void CUBLASAPI cublasStbmv (char uplo, char trans, char diag, int n, int k, 
                            const float *A, int lda, float *x, int incx);
void CUBLASAPI cublasDtbmv (char uplo, char trans, char diag, int n, int k, 
                            const double *A, int lda, double *x, int incx);
void CUBLASAPI cublasCtbmv (char uplo, char trans, char diag, int n, int k, 
                            const cuComplex *A, int lda, cuComplex *x, int incx);
void CUBLASAPI cublasZtbmv (char uplo, char trans, char diag, int n, int k, 
                            const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TPMV */                                                    
void CUBLASAPI cublasStpmv(char uplo, char trans, char diag, int n, const float *AP, float *x, int incx);

void CUBLASAPI cublasDtpmv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);

void CUBLASAPI cublasCtpmv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx);
                                         
void CUBLASAPI cublasZtpmv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TRSV */
void CUBLASAPI cublasStrsv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx);

void CUBLASAPI cublasDtrsv(char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx);

void CUBLASAPI cublasCtrsv(char uplo, char trans, char diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);

void CUBLASAPI cublasZtrsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda, 
                                         cuDoubleComplex *x, int incx);       
/*------------------------------------------------------------------------*/
/* TPSV */
void CUBLASAPI cublasStpsv(char uplo, char trans, char diag, int n, const float *AP, 
                          float *x, int incx);
                                                                                                            
void CUBLASAPI cublasDtpsv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);

void CUBLASAPI cublasCtpsv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx);

void CUBLASAPI cublasZtpsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, 
                                         cuDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/                                         
/* TBSV */                                         
void CUBLASAPI cublasStbsv(char uplo, char trans, 
                                         char diag, int n, int k, const float *A, 
                                         int lda, float *x, int incx);

void CUBLASAPI cublasDtbsv(char uplo, char trans, 
                                         char diag, int n, int k, const double *A, 
                                         int lda, double *x, int incx);
void CUBLASAPI cublasCtbsv(char uplo, char trans, 
                                         char diag, int n, int k, const cuComplex *A, 
                                         int lda, cuComplex *x, int incx);      
                                         
void CUBLASAPI cublasZtbsv(char uplo, char trans, 
                                         char diag, int n, int k, const cuDoubleComplex *A, 
                                         int lda, cuDoubleComplex *x, int incx);  
/*------------------------------------------------------------------------*/                                         
/* SYMV/HEMV */
void CUBLASAPI cublasSsymv (char uplo, int n, float alpha, const float *A,
                            int lda, const float *x, int incx, float beta, 
                            float *y, int incy);
void CUBLASAPI cublasDsymv (char uplo, int n, double alpha, const double *A,
                            int lda, const double *x, int incx, double beta, 
                            double *y, int incy);
void CUBLASAPI cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A,
                            int lda, const cuComplex *x, int incx, cuComplex beta, 
                            cuComplex *y, int incy);
void CUBLASAPI cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A,
                            int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, 
                            cuDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/       
/* SBMV/HBMV */
void CUBLASAPI cublasSsbmv (char uplo, int n, int k, float alpha, 
                            const float *A, int lda, const float *x, int incx, 
                            float beta, float *y, int incy);
void CUBLASAPI cublasDsbmv (char uplo, int n, int k, double alpha, 
                            const double *A, int lda, const double *x, int incx, 
                            double beta, double *y, int incy);
void CUBLASAPI cublasChbmv (char uplo, int n, int k, cuComplex alpha, 
                            const cuComplex *A, int lda, const cuComplex *x, int incx, 
                            cuComplex beta, cuComplex *y, int incy);
void CUBLASAPI cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha, 
                            const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, 
                            cuDoubleComplex beta, cuDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/       
/* SPMV/HPMV */
void CUBLASAPI cublasSspmv(char uplo, int n, float alpha,
                                     const float *AP, const float *x,
                                     int incx, float beta, float *y, int incy);
void CUBLASAPI cublasDspmv(char uplo, int n, double alpha,
                                     const double *AP, const double *x,
                                     int incx, double beta, double *y, int incy);
void CUBLASAPI cublasChpmv(char uplo, int n, cuComplex alpha,
                                     const cuComplex *AP, const cuComplex *x,
                                     int incx, cuComplex beta, cuComplex *y, int incy);
void CUBLASAPI cublasZhpmv(char uplo, int n, cuDoubleComplex alpha,
                                     const cuDoubleComplex *AP, const cuDoubleComplex *x,
                                     int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy);

/*------------------------------------------------------------------------*/       
/* GER */
void CUBLASAPI cublasSger (int m, int n, float alpha, const float *x, int incx,
                           const float *y, int incy, float *A, int lda);
void CUBLASAPI cublasDger (int m, int n, double alpha, const double *x, int incx,
                           const double *y, int incy, double *A, int lda);

void CUBLASAPI cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x,
                            int incx, const cuComplex *y, int incy,
                            cuComplex *A, int lda);
void CUBLASAPI cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x,
                            int incx, const cuComplex *y, int incy,
                            cuComplex *A, int lda);
void CUBLASAPI cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                            int incx, const cuDoubleComplex *y, int incy,
                            cuDoubleComplex *A, int lda);
void CUBLASAPI cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                            int incx, const cuDoubleComplex *y, int incy,
                            cuDoubleComplex *A, int lda);
/*------------------------------------------------------------------------*/       
/* SYR/HER */
void CUBLASAPI cublasSsyr (char uplo, int n, float alpha, const float *x,
                           int incx, float *A, int lda);
void CUBLASAPI cublasDsyr (char uplo, int n, double alpha, const double *x,
                           int incx, double *A, int lda);

void CUBLASAPI cublasCher (char uplo, int n, float alpha, 
                           const cuComplex *x, int incx, cuComplex *A, int lda);
void CUBLASAPI cublasZher (char uplo, int n, double alpha, 
                           const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);

/*------------------------------------------------------------------------*/       
/* SPR/HPR */
void CUBLASAPI cublasSspr (char uplo, int n, float alpha, const float *x,
                           int incx, float *AP);
void CUBLASAPI cublasDspr (char uplo, int n, double alpha, const double *x,
                           int incx, double *AP);
void CUBLASAPI cublasChpr (char uplo, int n, float alpha, const cuComplex *x,
                           int incx, cuComplex *AP);
void CUBLASAPI cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x,
                           int incx, cuDoubleComplex *AP);
/*------------------------------------------------------------------------*/       
/* SYR2/HER2 */
void CUBLASAPI cublasSsyr2 (char uplo, int n, float alpha, const float *x, 
                            int incx, const float *y, int incy, float *A, 
                            int lda);
void CUBLASAPI cublasDsyr2 (char uplo, int n, double alpha, const double *x, 
                            int incx, const double *y, int incy, double *A, 
                            int lda);
void CUBLASAPI cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x, 
                            int incx, const cuComplex *y, int incy, cuComplex *A, 
                            int lda);
void CUBLASAPI cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, 
                            int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, 
                            int lda);

/*------------------------------------------------------------------------*/       
/* SPR2/HPR2 */
void CUBLASAPI cublasSspr2 (char uplo, int n, float alpha, const float *x, 
                            int incx, const float *y, int incy, float *AP);
void CUBLASAPI cublasDspr2 (char uplo, int n, double alpha,
                            const double *x, int incx, const double *y,
                            int incy, double *AP);
void CUBLASAPI cublasChpr2 (char uplo, int n, cuComplex alpha,
                            const cuComplex *x, int incx, const cuComplex *y,
                            int incy, cuComplex *AP);
void CUBLASAPI cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha,
                            const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                            int incy, cuDoubleComplex *AP);
/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */
void CUBLASAPI cublasSgemm (char transa, char transb, int m, int n, int k, 
                            float alpha, const float *A, int lda, 
                            const float *B, int ldb, float beta, float *C, 
                            int ldc);
void CUBLASAPI cublasDgemm (char transa, char transb, int m, int n, int k,
                            double alpha, const double *A, int lda, 
                            const double *B, int ldb, double beta, double *C, 
                            int ldc);              
void CUBLASAPI cublasCgemm (char transa, char transb, int m, int n, int k, 
                            cuComplex alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, cuComplex beta,
                            cuComplex *C, int ldc);
void CUBLASAPI cublasZgemm (char transa, char transb, int m, int n,
                            int k, cuDoubleComplex alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            cuDoubleComplex beta, cuDoubleComplex *C,
                            int ldc);                   
/* -------------------------------------------------------*/
/* SYRK */
void CUBLASAPI cublasSsyrk (char uplo, char trans, int n, int k, float alpha, 
                            const float *A, int lda, float beta, float *C, 
                            int ldc);
void CUBLASAPI cublasDsyrk (char uplo, char trans, int n, int k,
                            double alpha, const double *A, int lda,
                            double beta, double *C, int ldc);

void CUBLASAPI cublasCsyrk (char uplo, char trans, int n, int k,
                            cuComplex alpha, const cuComplex *A, int lda,
                            cuComplex beta, cuComplex *C, int ldc);
void CUBLASAPI cublasZsyrk (char uplo, char trans, int n, int k,
                            cuDoubleComplex alpha,
                            const cuDoubleComplex *A, int lda,
                            cuDoubleComplex beta,
                            cuDoubleComplex *C, int ldc);
/* ------------------------------------------------------- */
/* HERK */
void CUBLASAPI cublasCherk (char uplo, char trans, int n, int k,
          float alpha, const cuComplex *A, int lda,
          float beta, cuComplex *C, int ldc);
void CUBLASAPI cublasZherk (char uplo, char trans, int n, int k,
                            double alpha,
                            const cuDoubleComplex *A, int lda,
                            double beta,
                            cuDoubleComplex *C, int ldc);
/* ------------------------------------------------------- */
/* SYR2K */
void CUBLASAPI cublasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
           const float *A, int lda, const float *B, int ldb, 
           float beta, float *C, int ldc);

void CUBLASAPI cublasDsyr2k (char uplo, char trans, int n, int k,
           double alpha, const double *A, int lda,
           const double *B, int ldb, double beta,
           double *C, int ldc);
void CUBLASAPI cublasCsyr2k (char uplo, char trans, int n, int k,
           cuComplex alpha, const cuComplex *A, int lda,
           const cuComplex *B, int ldb, cuComplex beta,
           cuComplex *C, int ldc);

void CUBLASAPI cublasZsyr2k (char uplo, char trans, int n, int k,
                             cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                             cuDoubleComplex *C, int ldc);                             
/* ------------------------------------------------------- */
/* HER2K */
void CUBLASAPI cublasCher2k (char uplo, char trans, int n, int k,
           cuComplex alpha, const cuComplex *A, int lda,
           const cuComplex *B, int ldb, float beta,
           cuComplex *C, int ldc);

void CUBLASAPI cublasZher2k (char uplo, char trans, int n, int k,
                             cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb, double beta,
                             cuDoubleComplex *C, int ldc); 

/*------------------------------------------------------------------------*/       
/* SYMM*/
void CUBLASAPI cublasSsymm (char side, char uplo, int m, int n, float alpha, 
          const float *A, int lda, const float *B, int ldb,
          float beta, float *C, int ldc);
void CUBLASAPI cublasDsymm (char side, char uplo, int m, int n, double alpha, 
          const double *A, int lda, const double *B, int ldb,
          double beta, double *C, int ldc);
          
void CUBLASAPI cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha, 
          const cuComplex *A, int lda, const cuComplex *B, int ldb,
          cuComplex beta, cuComplex *C, int ldc);
          
void CUBLASAPI cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha, 
                            const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                            cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
/*------------------------------------------------------------------------*/       
/* HEMM*/
void CUBLASAPI cublasChemm (char side, char uplo, int m, int n,
                            cuComplex alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, cuComplex beta,
                            cuComplex *C, int ldc);
void CUBLASAPI cublasZhemm (char side, char uplo, int m, int n,
                            cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                            cuDoubleComplex *C, int ldc);  

/*------------------------------------------------------------------------*/       
/* TRSM*/
void CUBLASAPI cublasStrsm (char side, char uplo, char transa, char diag,
                            int m, int n, float alpha, const float *A, int lda,
                            float *B, int ldb);

void CUBLASAPI cublasDtrsm (char side, char uplo, char transa,
                            char diag, int m, int n, double alpha,
                            const double *A, int lda, double *B,
                            int ldb);

void CUBLASAPI cublasCtrsm (char side, char uplo, char transa, char diag,
                            int m, int n, cuComplex alpha, const cuComplex *A,
                            int lda, cuComplex *B, int ldb);

void CUBLASAPI cublasZtrsm (char side, char uplo, char transa,
                            char diag, int m, int n, cuDoubleComplex alpha,
                            const cuDoubleComplex *A, int lda,
                            cuDoubleComplex *B, int ldb);                                                        
/*------------------------------------------------------------------------*/       
/* TRMM*/
void CUBLASAPI cublasStrmm (char side, char uplo, char transa, char diag,
                            int m, int n, float alpha, const float *A, int lda,
                            float *B, int ldb);
void CUBLASAPI cublasDtrmm (char side, char uplo, char transa,
                            char diag, int m, int n, double alpha,
                            const double *A, int lda, double *B,
                            int ldb);
void CUBLASAPI cublasCtrmm (char side, char uplo, char transa, char diag,
                            int m, int n, cuComplex alpha, const cuComplex *A,
                            int lda, cuComplex *B, int ldb);
void CUBLASAPI cublasZtrmm (char side, char uplo, char transa,
                            char diag, int m, int n, cuDoubleComplex alpha,
                            const cuDoubleComplex *A, int lda, cuDoubleComplex *B,
                            int ldb);                                                                            

                                                                                                                                                                                                 
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(CUBLAS_H_) */
