#ifndef __GEMM_CUBLAS_CUH__
#define __GEMM_CUBLAS_CUH__

void gemm_cublas(float* input_a, float* input_b, float* input_c, const int M,
                 const int N, const int K);

#endif  // __GEMM_CUBLAS_CUH__