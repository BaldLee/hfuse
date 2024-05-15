#ifndef __GEMM_CUBLAS_CUH__
#define __GEMM_CUBLAS_CUH__

#include <cublas_v2.h>
#include <cuda_runtime.h>

void gemm_cublas(float* input_a, float* input_b, float* input_c, const int M,
                 const int N, const int K);

float bench_gemm_cublas(float* input_a, float* input_b, float* input_c,
                        const int M, const int N, const int K, const int loop);

#endif  // __GEMM_CUBLAS_CUH__