#include "../include/bncs_and_hist.cuh"

float benchmark_bncs_and_hist(const float* h_input, int height, int width,
                              int depth, float epsilon, float* h_mean,
                              float* h_transformed_var, float* h_a, /* output */
                              const float* h_b,                     /* input */
                              int nbins, float minvalue, float maxvalue,
                              int k2_totalElements, const int loop) {
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Init for bncs
    float *d_input, *d_mean, *d_transformed_var;
    const int total_elements = height * width * depth;
    cudaMalloc(&d_input, total_elements * sizeof(float));
    cudaMalloc(&d_mean, width * sizeof(float));
    cudaMalloc(&d_transformed_var, width * sizeof(float));
    cudaMemcpy(d_input, h_input, total_elements * sizeof(float),
               cudaMemcpyHostToDevice);

    // Init for hist
    size_t k2_size = k2_totalElements * sizeof(float);
    float *d_a, *d_b;
    cudaMalloc(&d_a, nbins * sizeof(float));
    cudaMalloc(&d_b, k2_size);
    cudaMemcpy(d_b, h_b, k2_size, cudaMemcpyHostToDevice);
    cudaMemset(d_a, 0, nbins * sizeof(float));

    // Configure bncs kernel
    dim3 blocks(width);    // One block per channel
    dim3 threads(16, 64);  // 1024 threads per block

    // Configure hist kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = 128;

    // Warm up
    for (int i = 0; i < 5; i++) {
        batch_norm_collect_statistics_kernel<<<blocks, threads, 0, stream0>>>(
            d_input, height, width, depth, epsilon, d_mean, d_transformed_var);
        histogram1D_kernel<<<blocksPerGrid, threadsPerBlock,
                             nbins * sizeof(float), stream1>>>(
            d_a, d_b, nbins, minvalue, maxvalue, k2_totalElements);
        cudaDeviceSynchronize();
    }

    float msec = 0.0;
    float total = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < loop; i++) {
        cudaEventRecord(start);
        batch_norm_collect_statistics_kernel<<<blocks, threads, 0, stream0>>>(
            d_input, height, width, depth, epsilon, d_mean, d_transformed_var);
        histogram1D_kernel<<<blocksPerGrid, threadsPerBlock,
                             nbins * sizeof(float), stream1>>>(
            d_a, d_b, nbins, minvalue, maxvalue, k2_totalElements);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msec, start, stop);
        total += msec;
    }

    // Copy results back to host
    cudaMemcpy(h_mean, d_mean, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transformed_var, d_transformed_var, width * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_a, nbins * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_transformed_var);
    cudaFree(d_a);
    cudaFree(d_b);
    return total / loop;
}