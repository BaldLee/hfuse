#ifndef __CUDA_COMMON_CUH__
#define __CUDA_COMMON_CUH__

#include <cuda_fp16.h>
#include <cuda_runtime.h>

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define OFFSET(row, col, ld) (((row) * (ld)) + (col))

#define INT4WRAP(val) ((int4*)(&(val)))[0]

#define UINT32WRAP(val) ((uint32_t*)(&(val)))[0]

#define LDMATRIX_X1(R, addr)                                              \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R)                                                \
                 : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                            \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
    (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes)                                       \
    asm volatile(                                                          \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes)                                       \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), \
                 "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes)                                       \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), \
                 "l"(src), "n"(Bytes))
#endif

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)         \
    asm volatile(                                                           \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, " \
        "%3, %4, %5}, {%6, %7}, {%8, %9};\n"                                \
        : "=r"(RD0), "=r"(RD1)                                              \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1),       \
          "r"(RC0), "r"(RC1))

#endif  // __CUDA_COMMON_CUH__