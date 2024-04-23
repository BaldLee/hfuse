#include <cuda_runtime.h>

#include "../include/histogram1d_gpu.cuh"

/*
  Modified kernel for computing the histogram of the input using shared memory.
 */
__global__ void histogram1D_kernel(float* a,       /* output */
                                   const float* b, /* input */
                                   int nbins, float minvalue, float maxvalue,
                                   int totalElements) {
    extern __shared__ float smem[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory
    for (int idx = threadIdx.x; idx < nbins; idx += blockDim.x) {
        smem[idx] = 0;
    }
    __syncthreads();

    // Populate the shared memory histogram
    while (i < totalElements) {
        float bVal = b[i];
        if (bVal >= minvalue && bVal <= maxvalue) {
            int bin = static_cast<int>((bVal - minvalue) /
                                       (maxvalue - minvalue) * nbins);
            atomicAdd(&smem[bin],
                      1);  // Simple increment since we are just counting
        }
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Copy from shared memory to global memory
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