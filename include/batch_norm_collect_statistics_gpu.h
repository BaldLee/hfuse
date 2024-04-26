#ifndef __BATCH_NORM_COLLECT_STATISTICS_GPU_H__
#define __BATCH_NORM_COLLECT_STATISTICS_GPU_H__

#include <stdio.h>

void batch_norm_collect_statistics_gpu(const float* h_input, int height,
                                       int width, int depth, float epsilon,
                                       float* h_mean, float* h_transformed_var);

float benchmark_batch_norm_collect_statistics_gpu(
    const float* h_input, int height, int width, int depth, float epsilon,
    float* h_mean, float* h_transformed_var, const int loop);

#endif  // __BATCH_NORM_COLLECT_STATISTICS_GPU_H__