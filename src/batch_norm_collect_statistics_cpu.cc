#include "../include/batch_norm_collect_statistics_cpu.h"

void batch_norm_collect_statistics_cpu(const float* input, int height,
                                       int width, int depth, float epsilon,
                                       float* save_mean,
                                       float* save_transformed_var) {
    for (int plane = 0; plane < width; plane++) {
        float sum = 0;
        float sum_sq = 0;
        int N = height * depth;

        for (int batch = 0; batch < height; batch++) {
            for (int x = 0; x < depth; x++) {
                float v = input[batch * width * depth + plane * depth + x];
                sum += v;
                sum_sq += v * v;
            }
        }

        float mean = sum / N;
        float variance =
            (sum_sq / N) - (mean * mean);  // Var(X) = E(X^2) - E(X)^2
        save_mean[plane] = mean;
        save_transformed_var[plane] = variance + epsilon;
    }
}