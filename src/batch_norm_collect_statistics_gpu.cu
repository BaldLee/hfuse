#include <cuda_runtime.h>

#include "../include/batch_norm_collect_statistics_gpu.cuh"
#include "../include/from_pytorch.cuh"

__global__ void batch_norm_collect_statistics_kernel(
    const float* __restrict__ input, int height, int width, int depth,
    float epsilon, float* __restrict__ save_mean,
    float* __restrict__ save_transformed_var) {
    __shared__ int shared_n[2 * 2 * C10_WARP_SIZE +
                            C10_WARP_SIZE];  // Shared memory for storing
                                             // intermediate results
    __shared__ float
        shared_avg_var[2 * C10_WARP_SIZE];  // Storage for averages and
                                            // variances in shared memory

    int plane = blockIdx.x;
    int N = height * depth;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Compute the mean and variance across (batch, x/y/z)
    float avg = 0;
    float var_n = 0;
    int n = 0;
    for (int batch = threadIdx.y; batch < height; batch += blockDim.y) {
        for (int x = threadIdx.x; x < depth; x += blockDim.x) {
            float v = input[batch * width * depth + plane * depth + x];
            float d1 = v - avg;
            n++;
            avg += d1 / n;
            var_n += d1 * (v - avg);
        }
    }

    // Parallel reduction in warp
    for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    // Write each warp's result into shared memory
    __syncthreads();
    if (tid % C10_WARP_SIZE == 0) {
        shared_n[tid / C10_WARP_SIZE] = n;
        shared_avg_var[tid / C10_WARP_SIZE * 2] = avg;
        shared_avg_var[tid / C10_WARP_SIZE * 2 + 1] = var_n;
    }
    __syncthreads();

    // Final reduction from shared memory to a single number per block
    if (tid < C10_WARP_SIZE) {
        n = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE ? shared_n[tid] : 0);
        avg = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE
                   ? shared_avg_var[2 * tid]
                   : 0);
        var_n = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE
                     ? shared_avg_var[2 * tid + 1]
                     : 0);
    }
    for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    // Save the mean and variance, using transformations as needed
    if (tid == 0) {
        if (save_mean != NULL) {
            save_mean[plane] = avg;
        }
        if (save_transformed_var != NULL) {
            save_transformed_var[plane] =
                var_n / N + epsilon;  // Assuming the transformation is to add
                                      // epsilon (modify as needed)
        }
    }
}

void batch_norm_collect_statistics_gpu(const float* h_input, int height,
                                       int width, int depth, float epsilon,
                                       float* h_mean,
                                       float* h_transformed_var) {
    // Allocate memory on device
    float *d_input, *d_mean, *d_transformed_var;
    const int total_elements = height * width * depth;
    cudaMalloc(&d_input, total_elements * sizeof(float));
    cudaMalloc(&d_mean, width * sizeof(float));
    cudaMalloc(&d_transformed_var, width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, total_elements * sizeof(float),
               cudaMemcpyHostToDevice);

    // Kernel dimensions
    dim3 blocks(width);    // One block per channel
    dim3 threads(16, 16);  // 256 threads per block

    // Launch the kernel
    batch_norm_collect_statistics_kernel<<<blocks, threads>>>(
        d_input, height, width, depth, epsilon, d_mean, d_transformed_var);

    // Copy results back to host
    cudaMemcpy(h_mean, d_mean, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transformed_var, d_transformed_var, width * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_transformed_var);
}