#ifndef __BATCH_NORM_COLLECT_STATISTICS_H__
#define __BATCH_NORM_COLLECT_STATISTICS_H__

#include <cuda_runtime.h>
#include <stdio.h>

void batch_norm_collect_statistics(const float* h_input, int height, int width,
                                   int depth, float epsilon, float* h_mean,
                                   float* h_transformed_var);

float benchmark_batch_norm_collect_statistics(const float* h_input, int height,
                                              int width, int depth,
                                              float epsilon, float* h_mean,
                                              float* h_transformed_var,
                                              const int loop);

__global__ void batch_norm_collect_statistics_kernel(
    const float* __restrict__ input, int height, int width, int depth,
    float epsilon, float* __restrict__ save_mean,
    float* __restrict__ save_transformed_var);

#endif  // __BATCH_NORM_COLLECT_STATISTICS_H__