/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori, Tilmann Matthaei, Ke-Long Zhang.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#ifndef BLAS_VECTORIZED_H
#define BLAS_VECTORIZED_H

// BLAS naming convention: LDA = leading dimension of A

#ifdef SSE

#include "sse_blas_vectorized.h"

#define simd_cgem_inverse sse_cgem_inverse

#ifdef AVX512

#include "vectorized_blas_avx512.h"
#define simd_cgemv  avx512_cgemv
#define simd_cgenmv avx512_cgenmv
// #define simd_cgem_inverse avx512_cgem_inverse

#elif defined(AVX2)

#include "vectorized_blas_avx.h"
#define simd_cgemv  avx_cgemv
#define simd_cgenmv avx_cgenmv
// #define simd_cgem_inverse avx_cgem_inverse

#elif defined(SSE)
#define simd_cgemv        sse_cgemv
#define simd_cgenmv       sse_cgenmv
// #define simd_cgem_inverse sse_cgem_inverse

#endif

#define PRINT_MACRO_HELPER(x) #x
#define PRINT_MACRO(x)        #x "=" PRINT_MACRO_HELPER(x)
//#pragma message(PRINT_MACRO(simd_cgemv))
//#pragma message(PRINT_MACRO(simd_cgenmv))
//#pragma message(PRINT_MACRO(simd_cgem_inverse))

// C=A*B+C
static inline void cgemv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
#ifdef __MIC__
    switch (lda) {
    case 16:
        return cgemv_lda16(N, A, B, C);
    case 32:
        return cgemv_lda32(N, A, B, C);
    case 48:
        return cgemv_lda48(N, A, B, C);
    case 64:
        return cgemv_lda64(N, A, B, C);
    default:
        fprintf(stderr, "FATAL: cgemv called with invalid leading dimension for A\n");
    }
#else
    simd_cgemv(N, A, lda, B, C);
#endif
}

// C=-A*B+C
static inline void cgenmv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
#ifdef __MIC__
    switch (lda) {
    case 16:
        return cgenmv_lda16(N, A, B, C);
    case 32:
        return cgenmv_lda32(N, A, B, C);
    case 48:
        return cgenmv_lda48(N, A, B, C);
    case 64:
        return cgenmv_lda64(N, A, B, C);
    default:
        fprintf(stderr, "FATAL: cgenmv called with invalid leading dimension for A\n");
    }
#else
    simd_cgenmv(N, A, lda, B, C);
#endif
}


static inline void cgem_inverse(const int N, OPERATOR_TYPE_float *A_inverse, OPERATOR_TYPE_float *A, int lda)
{
#ifdef __MIC__
    switch (lda) {
    case 16:
        return cgem_inverse_lda16(N, A_inverse, A);
    case 32:
        return cgem_inverse_lda32(N, A_inverse, A);
    case 48:
        return cgem_inverse_lda48(N, A_inverse, A);
    case 64:
        return cgem_inverse_lda64(N, A_inverse, A);
    default:
        fprintf(stderr, "FATAL: cgem_inverse called with invalid leading dimension for A\n");
    }
#else
    simd_cgem_inverse(N, A_inverse, A, lda);
#endif
}

#endif 
#endif // BLAS_VECTORIZED_H
