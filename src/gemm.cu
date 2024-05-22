#include "../include/gemm.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32
#define lgd 4

/*
 * This gemm is currently designed for fix tiling size
 * TRUE gemm has alpha and beta, they are 1.0 and 1.0 here.
 * Using tensor core: mma fp16 m16n8k16
 * A: row-major, B: col-major
 */
__global__ void gemm_kernel(const half* a, const half* b, /* input */
                            half* c,                      /* output */
                            const int M, const int N, const int K) {
    __shared__ half A_smem[MMA_M * MMA_K];
    __shared__ half B_smem[MMA_N * MMA_K];
    __shared__ half C_smem[MMA_M * MMA_N];

    const size_t laneid = threadIdx.x % warpSize;
    const size_t block_row = blockIdx.y * MMA_M;
    const size_t block_col = blockIdx.x * MMA_N;

    if (laneid < MMA_M) {
        INT4WRAP(C_smem[OFFSET(laneid, 0, MMA_N)]) =
            INT4WRAP(c[OFFSET(block_row + laneid, block_col, N)]);
    }
    uint32_t RC[2] = {0, 0};
    RC[0] = UINT32WRAP(C_smem[OFFSET(laneid / 4, (laneid % 4) * 2, MMA_N)]);
    RC[1] = UINT32WRAP(C_smem[OFFSET(laneid / 4 + 8, (laneid % 4) * 2, MMA_N)]);

    const size_t K_tiles_num = div_ceil(K, MMA_K);
    for (size_t i = 0; i < K_tiles_num; i++) {
        const size_t k_stride = i * MMA_K;
        // TODO(baldlee): replace it with cp.async
        INT4WRAP(A_smem[OFFSET(laneid / 2, (laneid % 2) * 8, MMA_K)]) =
            INT4WRAP(a[OFFSET(block_row + laneid / 2,
                              k_stride + (laneid % 2) * 8, K)]);
        if (laneid < MMA_N * 2) {
            INT4WRAP(B_smem[OFFSET(laneid / 2, (laneid % 2) * 8, MMA_K)]) =
                INT4WRAP(b[OFFSET(block_col + laneid / 2,
                                  k_stride + (laneid % 2) * 8, K)]);
        }
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
            &A_smem[OFFSET(laneid % 16, (laneid / 16) * 8, MMA_K)]);
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
            &B_smem[OFFSET(laneid % 8, ((laneid / 8) % 2) * 8, MMA_K)]);

        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);
        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                  RC[1]);

        __syncthreads();
    }  // K_tiles_num loop

    UINT32WRAP(C_smem[OFFSET(laneid / 4, (laneid % 4) * 2, MMA_N)]) = RC[0];
    UINT32WRAP(C_smem[OFFSET(laneid / 4 + 8, (laneid % 4) * 2, MMA_N)]) = RC[1];

    __syncthreads();

    if (laneid < MMA_M) {
        INT4WRAP(c[OFFSET(block_row + laneid, block_col, N)]) =
            INT4WRAP(C_smem[OFFSET(laneid, 0, MMA_N)]);
    }
}

void gemm(const half* h_a, const half* h_b, /* input */
          half* h_c,                        /* output */
          const int M, const int N, const int K) {
    half *d_a, *d_b, *d_c;
    cudaMalloc(reinterpret_cast<void**>(&d_a), M * K * sizeof(half));
    cudaMalloc(reinterpret_cast<void**>(&d_b), N * K * sizeof(half));
    cudaMalloc(reinterpret_cast<void**>(&d_c), M * N * sizeof(half));

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, M * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(N / MMA_N, M / MMA_M);
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

float benchmark_gemm(const half* a, const half* b, /* input */
                     const half* c,                /* output */
                     const int M, const int N, const int K, const int loop) {
    return 0.0;
}