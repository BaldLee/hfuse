#include "../include/histogram1d_cpu.h"

void histogram1D_cpu(float* h_a,       /* output */
                     const float* h_b, /* input */
                     int nbins, float minvalue, float maxvalue,
                     int totalElements) {
    for (int i = 0; i < totalElements; ++i) {
        if (h_b[i] >= minvalue && h_b[i] <= maxvalue) {
            int bin = static_cast<int>((h_b[i] - minvalue) /
                                       (maxvalue - minvalue) * nbins);
            h_a[bin]++;
        }
    }
}