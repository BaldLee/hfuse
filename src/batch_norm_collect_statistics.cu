#include "../include/batch_norm_collect_statistics.h"
#include "../include/from_pytorch.h"

/* This kernel comes from pytorch/aten/src/ATen/native/cuda/Normalization.cuh
 * (pytorch/pytorch commit d59f1da6)
 * The PART A,B,C is devided according to the paper: Automatic Horizontal Fusion
 * for GPU Kernels (https://dblp.org/rec/conf/cgo/LiZPL22)
 */
__global__ void batch_norm_collect_statistics_kernel(
    const float* __restrict__ input, int height, int width, int depth,
    float epsilon, float* __restrict__ save_mean,
    float* __restrict__ save_transformed_var) {
    __shared__ int
        shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];  // Shared memory for storing
                                                  // intermediate results
    __shared__ float
        shared_avg_var[2 * WARP_SIZE];  // Storage for averages and
                                        // variances in shared memory

    int plane = blockIdx.x;
    int N = height * depth;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // PART A: Compute the mean and varience across (batch, x)
    // It uses shuffles to partially aggregate the results
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
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    // PART B: Write partially aggegated results to shared mem
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        shared_n[tid / WARP_SIZE] = n;
        shared_avg_var[tid / WARP_SIZE * 2] = avg;
        shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
    }
    __syncthreads();

    // PART C: Another round of suffles to finalize the results
    if (tid < WARP_SIZE) {
        n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
        avg =
            (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid]
                                                       : 0);
        var_n = (tid < blockDim.x * blockDim.y / WARP_SIZE
                     ? shared_avg_var[2 * tid + 1]
                     : 0);
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) +
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

void batch_norm_collect_statistics(const float* h_input, int height, int width,
                                   int depth, float epsilon, float* h_mean,
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
    dim3 threads(16, 64);  // 1024 threads per block

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

float benchmark_batch_norm_collect_statistics(const float* h_input, int height,
                                              int width, int depth,
                                              float epsilon, float* h_mean,
                                              float* h_transformed_var,
                                              const int loop) {
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
    dim3 threads(16, 64);  // 1024 threads per block

    // Warm up
    for (int i = 0; i < 5; i++) {
        batch_norm_collect_statistics_kernel<<<blocks, threads>>>(
            d_input, height, width, depth, epsilon, d_mean, d_transformed_var);
    }

    float msec = 0.0;
    float total = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < loop; i++) {
        cudaEventRecord(start);
        batch_norm_collect_statistics_kernel<<<blocks, threads>>>(
            d_input, height, width, depth, epsilon, d_mean, d_transformed_var);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msec, start, stop);
        total += msec;
    }

    // Copy results back to host
    cudaMemcpy(h_mean, d_mean, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transformed_var, d_transformed_var, width * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_transformed_var);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total / loop;
}