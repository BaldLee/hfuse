#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define SMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, \
                  RC2, RC3)                                                   \
    asm volatile(                                                             \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                  \
        "{%0, %1, %2, %3},"                                                   \
        "{%4, %5, %6, %7},"                                                   \
        "{%8, %9},"                                                           \
        "{%10, %11, %12, %13};\n"                                             \
        : "=l"(RD0), "=l"(RD1), "=l"(RD2), "=l"(RD3)                          \
        : "l"(RA0), "l"(RA1), "l"(RA2), "l"(RA3), "l"(RB0), "l"(RB0),         \
          "l"(RC0), "l"(RC1), "l"(RC2), "l"(RC3))

int main() { return 0; }