#ifndef __BATCH_NORM_COLLECT_STATISTICS_CPU_H__
#define __BATCH_NORM_COLLECT_STATISTICS_CPU_H__

#include <math.h>

void batch_norm_collect_statistics_cpu(const float* input, int height,
                                       int width, int depth, float epsilon,
                                       float* save_mean,
                                       float* save_transformed_var);

#endif  // __BATCH_NORM_COLLECT_STATISTICS_CPU_H__