#ifndef __HISTOGRAM1D_CUH__
#define __HISTOGRAM1D_CUH__

#include <cuda_runtime.h>

#include "histogram1d.h"

__global__ void histogram1D_kernel(float* a,       /* output */
                                   const float* b, /* input */
                                   int nbins, float minvalue, float maxvalue,
                                   int total_elements);

#endif  // __HISTOGRAM1D_CUH__