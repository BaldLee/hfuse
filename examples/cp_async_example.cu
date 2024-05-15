#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CP_ASYNC_CA_16(dst, src)                                        \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" \
                 :                                                      \
                 : "l"(dst), "l"(src))

#define CP_ASYNC_CG_16(dst, src, Bytes)                                 \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
                 :                                                      \
                 : "l"(dst), "l"(src))

#define BLOCK_DIM 32

__global__ void cp_async_example(float* in, float* out, const int N) {
    __shared__ float smem[BLOCK_DIM * 4];
    const int start_idx = blockIdx.x * (N / gridDim.x);
    for (int i = 0; i < N / gridDim.x; i += blockDim.x * 4) {
        auto async_dst = __cvta_generic_to_shared(smem + threadIdx.x * 4);
        auto async_src = in + start_idx + i + threadIdx.x * 4;
        CP_ASYNC_CA_16(async_dst, async_src);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        ((float4*)&out[start_idx + i + threadIdx.x * 4])[0] =
            ((float4*)(&smem[threadIdx.x * 4]))[0];
    }
}

int main() {
    const int N = 1024;
    float *h_in, *h_out, *d_in, *d_out;
    h_in = reinterpret_cast<float*>(malloc(N * sizeof(float)));
    h_out = reinterpret_cast<float*>(malloc(N * sizeof(float)));
    cudaMalloc(reinterpret_cast<void**>(&d_in), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_out), N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim((N + block_dim.x * 4 - 1) / (block_dim.x * 4));

    cp_async_example<<<grid_dim, block_dim>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%f ", h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}