#ifndef __HISTOGRAM1D_CUH__
#define __HISTOGRAM1D_CUH__

void histogram1D_gpu(float* h_a,       /* output */
                     const float* h_b, /* input */
                     int nbins, float minvalue, float maxvalue,
                     int totalElements);

float benchmark_histogram1D_gpu(float* h_a,       /* output */
                                const float* h_b, /* input */
                                int nbins, float minvalue, float maxvalue,
                                int totalElements, const int loop);

#endif  // __HISTOGRAM1D_CUH__