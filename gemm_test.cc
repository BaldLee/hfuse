#include "include/gemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/gemm_cpu.h"
#include "include/gemm_cublas.h"

void print_matrix(const float *matrix, const int M, const int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            printf("%.4f  ", matrix[m * M + n]);
        }
        printf("\n");
    }
}

template <typename T>
void print_matrix_json(const T *matrix, const int M, const int N,
                       FILE *fout = stdout) {
    // We assert M > 0 and N > 0
    if (M <= 0 || N <= 0) {
        return;
    }
    fprintf(fout, "[");
    fprintf(fout, "[%.4f", static_cast<float>(matrix[0]));
    for (int n = 1; n < N; n++) {
        fprintf(fout, ",%.4f", static_cast<float>(matrix[n]));
    }
    fprintf(fout, "]");
    for (int m = 1; m < M; m++) {
        fprintf(fout, ",[%.4f", static_cast<float>(matrix[m * M]));
        for (int n = 1; n < N; n++) {
            fprintf(fout, ",%.4f", static_cast<float>(matrix[m * M + n]));
        }
        fprintf(fout, "]");
    }
    fprintf(fout, "]\n");
}

void gemm_accu_test() {
    srand(time(NULL));
    float *a, *b, *c0, *c1;
    half *ha, *hb, *hc;

    const int M = 128, N = 128, K = 128;
    a = (float *)malloc(M * K * sizeof(float));
    b = (float *)malloc(K * N * sizeof(float));
    c0 = (float *)malloc(M * N * sizeof(float));
    c1 = (float *)malloc(M * N * sizeof(float));
    ha = (half *)malloc(M * K * sizeof(half));
    hb = (half *)malloc(K * N * sizeof(half));
    hc = (half *)malloc(M * N * sizeof(half));

    /*
     * Initialize inputs
     * Matrix a is row-major and matrix b is col-major
     */
    for (int i = 0; i < M * K; i++) {
        a[i] = static_cast<float>(rand() - RAND_MAX / 2) /
               static_cast<float>(RAND_MAX);
        ha[i] = static_cast<half>(a[i]);
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = static_cast<float>(rand() - RAND_MAX / 2) /
               static_cast<float>(RAND_MAX);
        hb[i] = static_cast<half>(b[i]);
    }
    for (int i = 0; i < M * N; i++) {
        c0[i] = 0.0;
        c1[i] = 0.0;
        hc[i] = 0.0;
    }

#if 0
    print_matrix(a, M, K);
    print_matrix(b, K, N);
#endif

    gemm_cpu(a, b, c0, M, N, K);
    gemm_cublas(a, b, c1, M, N, K);
    gemm(ha, hb, hc, M, N, K);

    FILE *cpu_out = fopen("cpu.json", "w+");
    FILE *cublas_out = fopen("cublas.json", "w+");
    FILE *gemm_out = fopen("gemm.json", "w+");
    print_matrix_json(c0, M, N, cpu_out);
    print_matrix_json(c1, M, N, cublas_out);
    print_matrix_json(hc, M, N, gemm_out);
    fclose(cpu_out);
    fclose(cublas_out);
    fclose(gemm_out);

    free(a);
    free(b);
    free(c0);
    free(c1);
    free(ha);
    free(hb);
    free(hc);
}

void bench_gemm() {
    srand(time(NULL));
    float *a, *b, *c0, *c1;
    float *diff;
    const int M = 2048, N = 2048, K = 2048;
    a = (float *)malloc(M * K * sizeof(float));
    b = (float *)malloc(K * N * sizeof(float));
    c0 = (float *)malloc(M * N * sizeof(float));
    c1 = (float *)malloc(M * N * sizeof(float));
    diff = (float *)malloc(M * N * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < M * K; i++) {
        a[i] = static_cast<float>(rand() - RAND_MAX / 2) /
               static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = static_cast<float>(rand() - RAND_MAX / 2) /
               static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < M * N; i++) {
        c0[i] = 0.0;
        c1[i] = 0.0;
    }

    // Benchmark
    const int loop = 1000;
    auto cublas_time = bench_gemm_cublas(a, b, c0, M, N, K, loop);

    printf("cublas time: %.4fms\n", cublas_time);
}

int main() {
#if 0
    gemm_accu_test()
    bench_gemm();
#endif
    gemm_accu_test();
}