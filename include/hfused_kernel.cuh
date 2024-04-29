#ifndef __HFUSED_KERNEL_CUH__
#define __HFUSED_KERNEL_CUH__

#include <cuda_runtime.h>

#include "from_pytorch.cuh"
#include "hfused.h"

__global__ void hfused_kernel_kernel7_1(
    const float* k1_input, int height, int width, int depth, float epsilon,
    float* out_mean, float* out_transformed_var /* params for kernel1*/,
    float* k2_output, const float* k2_input, int nbins, float minvalue,
    float maxvalue, int total_elements /* params for kernel2*/);

#endif  // __HFUSED_KERNEL_CUH__