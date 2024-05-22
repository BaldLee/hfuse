#include "../include/gemm_cublas.h"

/*
 * This function refers to NVIDIA's example
 * (https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu)
 */
void gemm_cublas(float* input_a, float* input_b, float* input_c, const int M,
                 const int N, const int K) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;
    cublasCreate(&cublasH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);

    const float alpha = 1.0;
    const float beta = 1.0;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * M * N);

    cudaMemcpyAsync(d_a, input_a, sizeof(float) * M * K, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_b, input_b, sizeof(float) * K * N, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_c, input_c, sizeof(float) * M * N, cudaMemcpyHostToDevice,
                    stream);

    // /*
    //  * Transposing problem in cuBLAS:
    //  * All matrices are considered in column-major in cuBLAS.
    //  * We want to do row-major gemm, so we need to do transpose.
    //  * We can do transpose for A and B with cublas api: transa and transb,
    //  * but transc is not offered.
    //  * How to handle it:
    //  * C = A * B => C^T = B^T * A^T
    //  * What we want is C^T, so we can switch the order of A and B.
    //  * And the A and B is row-major, they are A^T and B^T in cublas sight, so
    //  * transa and transb are all CUBLAS_OP_N (no need for transposing).
    //  * In one sentence: if we want do "C(row-major) = A(row-major) *
    //  * B(row-major)", do "C = B * A" in cublas.
    //  */
    // cublasOperation_t transa = CUBLAS_OP_N;
    // cublasOperation_t transb = CUBLAS_OP_N;
    // cublasSgemm(cublasH, transa, transb, M, N, K, &alpha, d_b, ldb, d_a, lda,
    //             &beta, d_c, ldc);

    /*
     * Now Matrix a is row-major and matrix b is col-major
     * We want C^T, and C^T = B^T * A^T => B * A^T (recall matrix b is
     * col-major)
     */
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasSgemm(cublasH, transa, transb, M, N, K, &alpha, d_b, ldb, d_a, lda,
                &beta, d_c, ldc);

    cudaMemcpyAsync(input_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost,
                    stream);

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

float bench_gemm_cublas(float* input_a, float* input_b, float* input_c,
                        const int M, const int N, const int K, const int loop) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;
    cublasCreate(&cublasH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);

    const float alpha = 1.0;
    const float beta = 1.0;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * M * N);

    cudaMemcpyAsync(d_a, input_a, sizeof(float) * M * K, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_b, input_b, sizeof(float) * K * N, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_c, input_c, sizeof(float) * M * N, cudaMemcpyHostToDevice,
                    stream);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Warm up
    for (int i = 0; i < 5; i++) {
        cublasSgemm(cublasH, transa, transb, M, N, K, &alpha, d_b, ldb, d_a,
                    lda, &beta, d_c, ldc);
    }

    float msec = 0.0;
    float total = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < loop; i++) {
        cudaEventRecord(start);
        cublasSgemm(cublasH, transa, transb, M, N, K, &alpha, d_b, ldb, d_a,
                    lda, &beta, d_c, ldc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msec, start, stop);
        total += msec;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total / loop;
}