#include <stdlib.h>
#include <time.h>

#include "include/bncs_and_hist.h"
#include "include/hfused.h"

/* We will fix:
 *   input size of bncs and hist
 * We will tune:
 *   girdDim, which is width in bncs
 * Best performance tunned: 8388608 262144 128 (I forget the device, V100 maybe)
 */
void tunning_hfuse(const int k1_total_elements, const int k2_total_elements,
                   const int grid_dim) {
    srand(time(NULL));

    // Init for batch_norm_collect_statistics
    int width = grid_dim;
    const int depth = 256;  // Spatial dimension (combined x/y/z)
    int height = k1_total_elements / depth / width;
    float epsilon = 0.00001f;
    float* h_input = (float*)malloc(k1_total_elements * sizeof(float));

    // Init for histogram1d
    int nbins = 256;
    float minvalue = 0.0f;
    float maxvalue = 100.0f;
    float* h_a = (float*)malloc(nbins * sizeof(float));
    float* h_b = (float*)malloc(k2_total_elements * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < k1_total_elements; ++i) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k2_total_elements; ++i) {
        h_b[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / maxvalue));
    }

    float* h_mean = (float*)malloc(width * sizeof(float));
    float* h_transformed_var = (float*)malloc(width * sizeof(float));

    hfused(h_input, height, width, depth, epsilon, h_mean, h_transformed_var,
           h_a, h_b, nbins, minvalue, maxvalue, k2_total_elements);

    free(h_input);
    free(h_mean);
    free(h_transformed_var);
    free(h_a);
    free(h_b);
    return;
}

/* We will fix:
 *   input size of bncs and hist
 * We will tune:
 *   girdDim, which is width in bncs
 */
void tunning_bncs_and_hist(const int k1_total_elements, const int k2_total_elements,
                   const int grid_dim) {
    srand(time(NULL));

    // Init for batch_norm_collect_statistics
    int width = grid_dim;
    const int depth = 256;  // Spatial dimension (combined x/y/z)
    int height = k1_total_elements / depth / width;
    float epsilon = 0.00001f;
    float* h_input = (float*)malloc(k1_total_elements * sizeof(float));

    // Init for histogram1d
    int nbins = 256;
    float minvalue = 0.0f;
    float maxvalue = 100.0f;
    float* h_a = (float*)malloc(nbins * sizeof(float));
    float* h_b = (float*)malloc(k2_total_elements * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < k1_total_elements; ++i) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k2_total_elements; ++i) {
        h_b[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / maxvalue));
    }

    float* h_mean = (float*)malloc(width * sizeof(float));
    float* h_transformed_var = (float*)malloc(width * sizeof(float));

    bncs_and_hist(h_input, height, width, depth, epsilon, h_mean, h_transformed_var,
           h_a, h_b, nbins, minvalue, maxvalue, k2_total_elements);

    free(h_input);
    free(h_mean);
    free(h_transformed_var);
    free(h_a);
    free(h_b);
    return;
}

int main(int argc, char* argv[]) {
    tunning_hfuse(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    return 0;
}
