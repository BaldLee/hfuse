#ifndef __GEMM_H__
#define __GEMM_H__

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_common.h"

/*
 * PTX ISA doc for mma
 * (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
 */
__global__ void gemm(const float* h_a, const float* h_b, /* input */
                     const float* h_c,                   /* output */
                     const int M, const int N, const int K);

void gemm(const float* h_a, const float* h_b, /* input */
          const float* h_c,                   /* output */
          const int M, const int N, const int K);

float benchmark_gemm(const float* h_a, const float* h_b, /* input */
                     const float* h_c,                   /* output */
                     const int M, const int N, const int K, const int loop);

#endif  // __GEMM_H__