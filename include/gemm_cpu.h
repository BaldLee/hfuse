#ifndef __GEMM_CPU_H__
#define __GEMM_CPU_H__

void gemm_cpu(float* input_a, float* input_b, float* input_c, const int M,
              const int N, const int K);

#endif  // __GEMM_CPU_H__