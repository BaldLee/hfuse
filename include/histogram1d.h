#ifndef __HISTOGRAM1D_H__
#define __HISTOGRAM1D_H__

void histogram1D(float* h_a,       /* output */
                 const float* h_b, /* input */
                 int nbins, float minvalue, float maxvalue, int total_elements);

float benchmark_histogram1D(float* h_a,       /* output */
                            const float* h_b, /* input */
                            int nbins, float minvalue, float maxvalue,
                            int total_elements, const int loop);

#endif  // __HISTOGRAM1D_H__