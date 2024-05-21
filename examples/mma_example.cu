#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define LDMATRIX_X1(R, addr)                                              \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R)                                                \
                 : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                            \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)         \
    asm volatile(                                                           \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, " \
        "%3, %4, %5}, {%6, %7}, {%8, %9};\n"                                \
        : "=r"(RD0), "=r"(RD1)                                              \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1),       \
          "r"(RC0), "r"(RC1))

#define INT4PTR(ptr) reinterpret_cast<int4 *>(var)

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__global__ void mmaNaiveKernel(const half *__restrict__ A,
                               const half *__restrict__ B, half *__restrict__ C,
                               size_t M, size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        /*
         * The movement of A tile from global to shared memory.
         * Shape of A_smem is MMA_M * MMA_K, i.e. 16 * 16
         * Every thread is responsable for 8 elements (128bit) which can be
         * moved in one single instruction.
         * BlockDim = 32, so one warp for one block.
         * There is the location map for every thread in A_smem.
         *  __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
         * |00|__|__|__|__|__|__|__|01|__|__|__|__|__|__|__|
         * |02|__|__|__|__|__|__|__|03|__|__|__|__|__|__|__|
         * |04|__|__|__|__|__|__|__|05|__|__|__|__|__|__|__|
         * |06|__|__|__|__|__|__|__|07|__|__|__|__|__|__|__|
         * |08|__|__|__|__|__|__|__|09|__|__|__|__|__|__|__|
         * |10|__|__|__|__|__|__|__|11|__|__|__|__|__|__|__|
         * |12|__|__|__|__|__|__|__|13|__|__|__|__|__|__|__|
         * |14|__|__|__|__|__|__|__|15|__|__|__|__|__|__|__|
         * |16|__|__|__|__|__|__|__|17|__|__|__|__|__|__|__|
         * |18|__|__|__|__|__|__|__|19|__|__|__|__|__|__|__|
         * |20|__|__|__|__|__|__|__|21|__|__|__|__|__|__|__|
         * |22|__|__|__|__|__|__|__|23|__|__|__|__|__|__|__|
         * |24|__|__|__|__|__|__|__|25|__|__|__|__|__|__|__|
         * |26|__|__|__|__|__|__|__|27|__|__|__|__|__|__|__|
         * |28|__|__|__|__|__|__|__|29|__|__|__|__|__|__|__|
         * |30|__|__|__|__|__|__|__|31|__|__|__|__|__|__|__|
         * Given laneid, row_index is laneid / 2, col_index is (laneid % 2) * 8
         */
        ((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2)[0] =
            ((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) +
             lane_id % 2)[0];

        /*
         * The movement of B tile from global to shared memory.
         * Shape of B_tile is MMA_K * MMA_N, i.e. 16 * 8.
         * B_smem is col-major for mma instruction, so the shape of B_smem is
         * MMA_N * MMA_K, i.e. 8 * 16.
         * Only first 16 threads in warp need to do the movement.
         * Thereis the location map for every thread in B_smem.
         *  __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
         * |00|__|__|__|__|__|__|__|01|__|__|__|__|__|__|__|
         * |02|__|__|__|__|__|__|__|03|__|__|__|__|__|__|__|
         * |04|__|__|__|__|__|__|__|05|__|__|__|__|__|__|__|
         * |06|__|__|__|__|__|__|__|07|__|__|__|__|__|__|__|
         * |08|__|__|__|__|__|__|__|09|__|__|__|__|__|__|__|
         * |10|__|__|__|__|__|__|__|11|__|__|__|__|__|__|__|
         * |12|__|__|__|__|__|__|__|13|__|__|__|__|__|__|__|
         * |14|__|__|__|__|__|__|__|15|__|__|__|__|__|__|__|
         * Given laneid, row_index is laneid / 2, col_index is (laneid % 2) * 8
         */
        if (lane_id < MMA_N * 2) {
            ((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2)[0] =
                ((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
                 lane_id % 2)[0];
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        /*
         * The location map of A_smem_lane_addr for every thread.
         * ldmatrix.sync.aligned.x4 will execute all 32 threads
         *  __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
         * |00|__|__|__|__|__|__|__|16|__|__|__|__|__|__|__|
         * |01|__|__|__|__|__|__|__|17|__|__|__|__|__|__|__|
         * |02|__|__|__|__|__|__|__|18|__|__|__|__|__|__|__|
         * |03|__|__|__|__|__|__|__|19|__|__|__|__|__|__|__|
         * |04|__|__|__|__|__|__|__|20|__|__|__|__|__|__|__|
         * |05|__|__|__|__|__|__|__|21|__|__|__|__|__|__|__|
         * |06|__|__|__|__|__|__|__|22|__|__|__|__|__|__|__|
         * |07|__|__|__|__|__|__|__|23|__|__|__|__|__|__|__|
         * |08|__|__|__|__|__|__|__|24|__|__|__|__|__|__|__|
         * |09|__|__|__|__|__|__|__|25|__|__|__|__|__|__|__|
         * |10|__|__|__|__|__|__|__|26|__|__|__|__|__|__|__|
         * |11|__|__|__|__|__|__|__|27|__|__|__|__|__|__|__|
         * |12|__|__|__|__|__|__|__|28|__|__|__|__|__|__|__|
         * |13|__|__|__|__|__|__|__|29|__|__|__|__|__|__|__|
         * |14|__|__|__|__|__|__|__|30|__|__|__|__|__|__|__|
         * |15|__|__|__|__|__|__|__|31|__|__|__|__|__|__|__|
         */
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        /*
         * The location map of B_smem_lane_addr for every thread.
         * ldmatrix.sync.aligned.x2 will execute first 16 threads
         *  __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
         * |00|__|__|__|__|__|__|__|08|__|__|__|__|__|__|__|
         * |01|__|__|__|__|__|__|__|09|__|__|__|__|__|__|__|
         * |02|__|__|__|__|__|__|__|10|__|__|__|__|__|__|__|
         * |03|__|__|__|__|__|__|__|11|__|__|__|__|__|__|__|
         * |04|__|__|__|__|__|__|__|12|__|__|__|__|__|__|__|
         * |05|__|__|__|__|__|__|__|13|__|__|__|__|__|__|__|
         * |06|__|__|__|__|__|__|__|14|__|__|__|__|__|__|__|
         * |07|__|__|__|__|__|__|__|15|__|__|__|__|__|__|__|
         */
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
            &B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
#if 0
        printf("lane_id: %lu, row: %lu, col: %lu, mem_addr: %u\n", lane_id,
               lane_id % 8, ((lane_id / 8) % 2) * 8, B_smem_lane_addr);
#endif
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                  RC[1]);

        __syncthreads();
    }  // End of K_tiles loop

    ((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4)[0] = RC[0];
    ((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4)[0] = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        ((int4 *)(&C[(warp_row + lane_id) * N + warp_col]))[0] =
            ((int4 *)(&C_smem[lane_id][0]))[0];
    }
}

void gemm_cpu(half *input_a, half *input_b, half *input_c, const int M,
              const int N, const int K) {
    /* A is row-major and B is col-major */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                input_c[i * N + j] = static_cast<half>(
                    static_cast<float>(input_c[i * N + j]) +
                    static_cast<float>(input_a[i * K + k]) *
                        static_cast<float>(input_b[j * K + k]));
            }
        }
    }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
    half *a, *b, *c, *c_cpu;
    const int M = 32, N = 32, K = 32;
    a = reinterpret_cast<half *>(malloc(M * K * sizeof(half)));
    b = reinterpret_cast<half *>(malloc(K * N * sizeof(half)));
    c = reinterpret_cast<half *>(malloc(M * N * sizeof(half)));
    c_cpu = reinterpret_cast<half *>(malloc(M * N * sizeof(half)));

    half *d_a, *d_b, *d_c;
    cudaMalloc(reinterpret_cast<void **>(&d_a), M * K * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&d_b), K * N * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&d_c), M * N * sizeof(half));

    for (int i = 0; i < M * K; i++) {
        a[i] = (i - M * K / 2) * 0.0002f;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = (i - M * K / 2) * 0.0002f;
    }
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
        c_cpu[i] = 0.0f;
    }

    cudaMemcpy(d_a, a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, M * N * sizeof(half), cudaMemcpyHostToDevice);

    mmaNaive(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    gemm_cpu(a, b, c_cpu, M, N, K);

    printf("\nmma res:\n");
    for (int i = 0; i < M * N; i++) {
        printf("%.4f ", static_cast<float>(c[i]));
    }

    printf("\ncpu res:\n");
    for (int i = 0; i < M * N; i++) {
        printf("%.4f ", static_cast<float>(c_cpu[i]));
    }

    printf("\ndiff:\n");
    for (int i = 0; i < M * N; i++) {
        printf("%.4f ",
               static_cast<float>(c[i]) - static_cast<float>(c_cpu[i]));
    }

    free(a);
    free(b);
    free(c);
    free(c_cpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}