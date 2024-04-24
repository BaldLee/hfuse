#ifndef __HFUSED_KERNEL_CUH__
#define __HFUSED_KERNEL_CUH__

void hfused_kernel(const float* k1_input, int height, int width, int depth,
                   float epsilon, float* h_mean,
                   float* h_transformed_var /* params for kernel1*/,
                   float* k2_output, const float* k2_input, int nbins,
                   float minvalue, float maxvalue,
                   int totalElements /* params for kernel2*/
);

#endif  // __HFUSED_KERNEL_CUH__