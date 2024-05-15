#include "../include/gemm.h"

#define SHMEM_STRIDE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 8
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32
#define lgd 4

/*
 * This gemm is currently designed for fix tiling size (2048*2048*2048)
 * TRUE gemm has alpha and beta, they are 1.0 and 1.0 here.
 */
__global__ void gemm(const float* h_a, const float* h_b, /* input */
                     const float* h_c,                   /* output */
                     const int M, const int N, const int K) {
    __shared__ float smem[BLOCK_K * BLOCK_M + BLOCK_N * BLOCK_K];
    float* smem_a = &smem[0];
    float* smem_b = &smem[BLOCK_K * BLOCK_M];

    int warp_num = blockDim.x / warpSize;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int start_m = blockIdx.x * BLOCK_M;
    int start_n = blockIdx.y * BLOCK_N;

    for (int i_k = 0; i_k < M / BLOCK_M; i_k++) {
        int start_k = i_k * BLOCK_K;

        __syncthreads();
    }
}

float benchmark_gemm(const float* h_a, const float* h_b, /* input */
                     const float* h_c,                   /* output */
                     const int M, const int N, const int K, const int loop) {
    return 0.0;
}