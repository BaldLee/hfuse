#ifndef __HFUSED_H__
#define __HFUSED_H__

void hfused(const float* k1_input, int height, int width, int depth,
            float epsilon, float* h_mean,
            float* h_transformed_var /* params for kernel1*/, float* k2_output,
            const float* k2_input, int nbins, float minvalue, float maxvalue,
            int total_elements /* params for kernel2*/);

float benchmark_hfused(const float* k1_input, int height, int width, int depth,
                       float epsilon, float* h_mean,
                       float* h_transformed_var /* params for kernel1*/,
                       float* k2_output, const float* k2_input, int nbins,
                       float minvalue, float maxvalue,
                       int total_elements, /* params for kernel2*/
                       const int loop);

#endif  // __HFUSED_H__