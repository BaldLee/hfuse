#ifndef __HISTOGRAM1D_CPU_H__
#define __HISTOGRAM1D_CPU_H__

void histogram1D_cpu(float* h_a,       /* output */
                     const float* h_b, /* input */
                     int nbins, float minvalue, float maxvalue,
                     int total_elements);

#endif  // __HISTOGRAM1D_CPU_H__