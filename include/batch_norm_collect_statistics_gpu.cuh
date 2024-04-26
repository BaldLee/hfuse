#ifndef __BATCH_NORM_COLLECT_STATISTICS_GPU_H__
#define __BATCH_NORM_COLLECT_STATISTICS_GPU_H__

#include "batch_norm_collect_statistics_gpu.h"

__global__ void batch_norm_collect_statistics_kernel(
    const float* __restrict__ input, int height, int width, int depth,
    float epsilon, float* __restrict__ save_mean,
    float* __restrict__ save_transformed_var);

#endif  // __BATCH_NORM_COLLECT_STATISTICS_GPU_H__