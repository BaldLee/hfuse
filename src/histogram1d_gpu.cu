#include <cuda_runtime.h>

#include "../include/histogram1d_gpu.cuh"

/* This kernel comes from pytorch/aten/src/ATen/native/cuda/SummaryOps.cu
 * (pytorch/pytorch commit d59f1da6)
 *
 * The PART A,B,C is devided according to the paper: Automatic Horizontal Fusion
 * for GPU Kernels (https://dblp.org/rec/conf/cgo/LiZPL22)
 *
 * We assume the "getop" is "getDummyOp", which always returns 1. In origin
 * code, "getop" is used to get the weight from weight tensor c. The tensor info
 * of c is passed by the anonymous function "getWeightsOp". Refer to
 * CUDA_tensor_histogram in pytorch/aten/src/ATen/native/cuda/SummaryOps.cu for
 * details.
 */
__global__ void histogram1D_kernel(float* a,       /* output */
                                   const float* b, /* input */
                                   int nbins, float minvalue, float maxvalue,
                                   int totalElements) {
    extern __shared__ float smem[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // PARTA:Initialize shared memory counters
    for (int idx = threadIdx.x; idx < nbins; idx += blockDim.x) {
        smem[idx] = 0;
    }
    __syncthreads();

    // PART B: Go over the input b to increment shared counters
    while (i < totalElements) {
        float bVal = b[i];
        if (bVal >= minvalue && bVal <= maxvalue) {
            int bin = static_cast<int>((bVal - minvalue) /
                                       (maxvalue - minvalue) * nbins);
            atomicAdd(&smem[bin], 1);
        }
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // PART C: Increment the output a with the shared counters
    for (int idx = threadIdx.x; idx < nbins; idx += blockDim.x) {
        atomicAdd(&a[idx], smem[idx]);
    }
}

void histogram1D_gpu(float* h_a,       /* output */
                     const float* h_b, /* input */
                     int nbins, float minvalue, float maxvalue,
                     int totalElements) {
    size_t size = totalElements * sizeof(float);

    // Allocate memory
    float *d_a, *d_b;
    cudaMalloc(&d_a, nbins * sizeof(float));
    cudaMalloc(&d_b, size);

    // Copy data to device
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemset(d_a, 0, nbins * sizeof(float));

    // Configure kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    histogram1D_kernel<<<blocksPerGrid, threadsPerBlock,
                         nbins * sizeof(float)>>>(d_a, d_b, nbins, minvalue,
                                                  maxvalue, totalElements);

    cudaMemcpy(h_a, d_a, nbins * sizeof(float), cudaMemcpyDeviceToHost);
}