#ifndef __BNCS_AND_HIST_H__
#define __BNCS_AND_HIST_H__

float benchmark_bncs_and_hist(const float* h_input, int height, int width,
                              int depth, float epsilon, float* h_mean,
                              float* h_transformed_var, float* h_a, /* output */
                              const float* h_b,                     /* input */
                              int nbins, float minvalue, float maxvalue,
                              int k2_totalElements, const int loop);
#endif  // __BNCS_AND_HIST_H__