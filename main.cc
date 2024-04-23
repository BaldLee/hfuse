#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/batch_norm_collect_statistics_cpu.h"
#include "include/batch_norm_collect_statistics_gpu.cuh"

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Define the dimensions of the data
    const int height = 128;  // Number of batches
    const int width = 128;   // Number of channels (planes in the kernel)
    const int depth = 256;   // Spatial dimension (combined x/y/z)

    // Total elements in the input array
    const int total_elements = height * width * depth;
    float epsilon = 0.00001f;

    // Allocate memory for input and output on host
    float* h_input = (float*)malloc(total_elements * sizeof(float));
    float* h_mean = (float*)malloc(width * sizeof(float));
    float* h_transformed_var = (float*)malloc(width * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < total_elements; ++i) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    // Test for cpu
    batch_norm_collect_statistics_cpu(h_input, height, width, depth, epsilon,
                                      h_mean, h_transformed_var);

    // Print CPU results
    printf("CPU Mean values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Mean[%d]: %f\n", i, h_mean[i]);
    }

    printf("CPU Transformed Variance values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Transformed Variance[%d]: %f\n", i, h_transformed_var[i]);
    }

    // Test for gpu
    batch_norm_collect_statistics_gpu(h_input, height, width, depth, epsilon,
                                      h_mean, h_transformed_var);

    // Print results
    printf("GPU Mean values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Mean[%d]: %f\n", i, h_mean[i]);
    }

    printf("GPU Transformed Variance values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Transformed Variance[%d]: %f\n", i, h_transformed_var[i]);
    }

    // Free host memory
    free(h_input);
    free(h_mean);
    free(h_transformed_var);

    return 0;
}
