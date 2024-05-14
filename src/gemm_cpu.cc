#include "../include/gemm_cpu.h"

void gemm_cpu(float* input_a, float* input_b, float* input_c, const int M,
              const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                input_c[i * N + j] += input_a[i * K + k] * input_b[k * N + j];
            }
        }
    }
}