#ifndef __BATCH_NORM_COLLECT_STATISTICS_GPU_H__
#define __BATCH_NORM_COLLECT_STATISTICS_GPU_H__

void batch_norm_collect_statistics_gpu(const float* h_input, int height,
                                       int width, int depth, float epsilon,
                                       float* h_mean, float* h_transformed_var);

#endif  // __BATCH_NORM_COLLECT_STATISTICS_GPU_H__