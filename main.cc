#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/batch_norm_collect_statistics_cpu.h"
#include "include/batch_norm_collect_statistics_gpu.cuh"
#include "include/hfused_kernel.cuh"
#include "include/histogram1d_cpu.h"
#include "include/histogram1d_gpu.cuh"

void do_batch_norm_collect_statistics() {
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
}

void do_histogram1d() {
    srand(time(NULL));  // Seed for random number generation
    int nbins = 256;
    float minvalue = 0.0f;
    float maxvalue = 100.0f;
    int totalElements = 1024 * 1024;  // 1M elements

    float* h_a = (float*)malloc(nbins * sizeof(float));
    float* h_b = (float*)malloc(totalElements * sizeof(float));

    // Initialize input data
    for (int i = 0; i < totalElements; ++i) {
        h_b[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / maxvalue));
    }

    // Test for cpu
    histogram1D_cpu(h_a, h_b, nbins, minvalue, maxvalue, totalElements);
    printf("CPU results:\n");
    for (int i = 0; i < nbins; i++) {
        printf("a[%d]: %f\n", i, h_a[i]);
    }

    // Test for gpu
    histogram1D_gpu(h_a, h_b, nbins, minvalue, maxvalue, totalElements);
    printf("GPU results:\n");
    for (int i = 0; i < nbins; i++) {
        printf("a[%d]: %f\n", i, h_a[i]);
    }

    free(h_a);
    free(h_b);
}

void do_hfused() {
    // Initialize random seed
    srand(time(NULL));

    // Init for batch_norm_collect_statistics
    const int height = 128;  // Number of batches
    const int width = 128;   // Number of channels (planes in the kernel)
    const int depth = 256;   // Spatial dimension (combined x/y/z)
    const int k1_total_elements = height * width * depth;
    float epsilon = 0.00001f;
    float* h_input = (float*)malloc(k1_total_elements * sizeof(float));
    float* h_mean = (float*)malloc(width * sizeof(float));
    float* h_transformed_var = (float*)malloc(width * sizeof(float));

    // Init for histogram1d
    int nbins = 256;
    float minvalue = 0.0f;
    float maxvalue = 100.0f;
    int k2_totalElements = 1024 * 1024;  // 1M elements
    float* h_a = (float*)malloc(nbins * sizeof(float));
    float* h_b = (float*)malloc(k2_totalElements * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < k1_total_elements; ++i) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k2_totalElements; ++i) {
        h_b[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / maxvalue));
    }

    // Test for CPU
    batch_norm_collect_statistics_cpu(h_input, height, width, depth, epsilon,
                                      h_mean, h_transformed_var);
    printf("BNCS CPU Mean values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Mean[%d]: %f\n", i, h_mean[i]);
    }
    printf("BNCS CPU Transformed Variance values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Transformed Variance[%d]: %f\n", i, h_transformed_var[i]);
    }
    histogram1D_cpu(h_a, h_b, nbins, minvalue, maxvalue, k2_totalElements);
    printf("histogram1D CPU results:\n");
    for (int i = 0; i < nbins; i++) {
        printf("a[%d]: %f\n", i, h_a[i]);
    }

    // Test for hfuse
    hfused(h_input, height, width, depth, epsilon, h_mean, h_transformed_var,
           h_a, h_b, nbins, minvalue, maxvalue, k2_totalElements);
    printf("BNCS GPU Mean values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Mean[%d]: %f\n", i, h_mean[i]);
    }
    printf("BNCS GPU Transformed Variance values:\n");
    for (int i = 0; i < width; ++i) {
        printf("Transformed Variance[%d]: %f\n", i, h_transformed_var[i]);
    }
    printf("histogram1D GPU results:\n");
    for (int i = 0; i < nbins; i++) {
        printf("a[%d]: %f\n", i, h_a[i]);
    }

    free(h_input);
    free(h_mean);
    free(h_transformed_var);
    free(h_a);
    free(h_b);
}

int main() {
    // printf("batch_norm_collect_statistics\n");
    // do_batch_norm_collect_statistics();
    // printf("histogram1d\n");
    // do_histogram1d();
    do_hfused();
    return 0;
}
