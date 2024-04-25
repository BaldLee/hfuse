#include <cuda_runtime.h>

#include "../include/from_pytorch.cuh"
#include "../include/hfused_kernel.cuh"

// Refernce code from the paper
#if 0
void fused_kernel(...){
  //Prologue of the fused kernel
  int global_tid = threadIdx.x + threadIdx.y * blockDim.x +
                   threadIdx.z * blockDim.x * blockDim.y;
  int threadIdx_x , threadIdx_y , threadIdx_z;
  int blockDim_x , blockDim_y , blockDim_z; 
  if (global_tid < 896) { 
    blockDim_x = 896 / 16; 
    blockDim_y = 16; 
    blockDim_z = 1; 
    threadIdx_x = global_tid % blockDim_x;
    threadIdx_y = global_tid / blockDim_x % blockDim_y;
    threadIdx_z = 1;
  } else {
    blockDim_x = 128;
    blockDim_y = 1;
    blockDim_z = 1;
    threadIdx_x = (global_tid - 896) % blockDim_x;
    threadIdx_y = 1;
    threadIdx_z = 1;
  }
  //Variable decls for batch_norm_collect_statistics()
  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];
  ...
  //Variable decls for kernelHistogram1D()
  extern__shared__ unsigned char my_smem[];
  output_t* smem;
  if(!(global_tid < 896)) goto K1_end;
  //batch_norm_collect_statistics() PART A
  ...
  //A PTX assembly to only sync 896 threads.
  asm("bar.sync 1, 896;");
  //batch_norm_collect_statistics() PART B
  ...
  asm("bar.sync 1, 896;");
  //batch_norm_collect_statistics() PART C
  ...
K1_end:
  if (global_tid < 896) goto K2_end;
  //kernelHistogram1D() PART A
  smem = reinterpret_cast<output_t*>(my_smem);
  for (int i = threadIdx_x; i < a.sizes[0]; i += blockDim_x) {
    smem[i] = 0;
  }
  //A PTX assembly to only sync 128 threads.
  asm("bar.sync 2, 128;");
  //kernelHistogram1D() PART B
  ...
  asm("bar.sync 2, 128;");
  //kernelHistogram1D() PART C
  ...
K2_end:
}
#endif

/* It is terrible to hard code the blockDim
 * But the paper did it first, so.
 * The blockDim is 1024, it is devided into 896 and 128.
 * 896 is a 56 * 16 block for batch_norm_collect_statistics()
 * 128 is a 128 * 1 block for histogram1D()
 *
 */
__global__ void hfused_kernel_kernel7_1(
    const float* k1_input, int height, int width, int depth, float epsilon,
    float* out_mean, float* out_transformed_var /* params for kernel1*/,
    float* k2_output, const float* k2_input, int nbins, float minvalue,
    float maxvalue, int totalElements /* params for kernel2*/) {
    // Prologue of the fused kernel
    int global_tid = threadIdx.x + threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;
    int threadIdx_x, threadIdx_y, threadIdx_z, blockDim_x, blockDim_y,
        blockDim_z;

    if (global_tid < 896) {
        blockDim_x = 896 / 16;  // origin: (16,16), now: (56,16)
        blockDim_y = 16;
        blockDim_z = 1;
        threadIdx_x = global_tid % blockDim_x;
        threadIdx_y = global_tid / blockDim_x % blockDim_y;
        threadIdx_z = 1;
    } else {
        blockDim_x = 128;
        blockDim_y = 1;
        blockDim_z = 1;
        threadIdx_x = (global_tid - 896) % blockDim_x;
        threadIdx_y = 1;
        threadIdx_z = 1;
    }
    // Variable decls for batch_norm_collect_statistics()
    extern __shared__ float smem[];
    // __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];
    // __shared__ float shared_avg_var[2 * WARP_SIZE];
    int* shared_n = (int*)smem;
    float* shared_avg_var = (float*)&shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];
    int plane = blockIdx.x;
    int N = height * depth;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
    float avg = 0;
    float var_n = 0;
    int n = 0;
    // Variable decls for kernelHistogram1D()

    if (!(global_tid < 896)) goto K1_end;

    // PART A: Compute the mean and varience across (batch, x)
    // It uses shuffles to partially aggregate the results

    for (int batch = threadIdx_y; batch < height; batch += blockDim_y) {
        for (int x = threadIdx_x; x < depth; x += blockDim_x) {
            float v = k1_input[batch * width * depth + plane * depth + x];
            float d1 = v - avg;
            n++;
            avg += d1 / n;
            var_n += d1 * (v - avg);
        }
    }

    // Parallel reduction in warp
    for (int i = 0; i < getMSB(WARP_SIZE); i++) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    asm("bar.sync 1, 896;");

    // PART B: Write partially aggegated results to shared mem
    if (tid % WARP_SIZE == 0) {
        shared_n[tid / WARP_SIZE] = n;
        shared_avg_var[tid / WARP_SIZE * 2] = avg;
        shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
    }

    asm("bar.sync 1, 896;");

    // PART C: Another round of suffles to finalize the results
    if (tid < WARP_SIZE) {
        n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
        avg =
            (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid]
                                                       : 0);
        var_n = (tid < blockDim.x * blockDim.y / WARP_SIZE
                     ? shared_avg_var[2 * tid + 1]
                     : 0);
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        float o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        float factor = 1.0 / fmaxf(1.0, n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    // Save the mean and variance, using transformations as needed
    if (tid == 0) {
        if (out_mean != NULL) {
            out_mean[plane] = avg;
        }
        if (out_transformed_var != NULL) {
            out_transformed_var[plane] =
                var_n / N + epsilon;  // Assuming the transformation is to
                                      // add epsilon (modify as needed)
        }
    }

K1_end:
    if (global_tid < 896) goto K2_end;
    // PARTA:Initialize shared memory counters
    for (int idx = threadIdx.x; idx < nbins; idx += blockDim.x) {
        smem[idx] = 0;
    }

    asm("bar.sync 2, 128");

    // PART B: Go over the input b to increment shared counters
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < totalElements;
         i += blockDim.x * gridDim.x) {
        float bVal = k2_input[i];
        if (bVal >= minvalue && bVal <= maxvalue) {
            int bin = static_cast<int>((bVal - minvalue) /
                                       (maxvalue - minvalue) * nbins);
            if (bin == nbins) {
                bin -= 1;
            }
            atomicAdd(&smem[bin], 1);
        }
    }

    asm("bar.sync 2, 128");

    // PART C: Increment the output a with the shared counters
    for (int idx = threadIdx.x; idx < nbins; idx += blockDim.x) {
        atomicAdd(&k2_output[idx], smem[idx]);
    }
K2_end:
    return;
}

void hfused(const float* k1_input, int height, int width, int depth,
            float epsilon, float* h_mean,
            float* h_transformed_var /* params for kernel1*/, float* k2_output,
            const float* k2_input, int nbins, float minvalue, float maxvalue,
            int totalElements /* params for kernel2*/

) {
    // Allocation for batch_norm_collect_statistics
    float *d_k1_input, *d_k1_out_mean, *d_k1_out_transformed_var;
    const int total_elements = height * width * depth;
    cudaMalloc(&d_k1_input, total_elements * sizeof(float));
    cudaMalloc(&d_k1_out_mean, width * sizeof(float));
    cudaMalloc(&d_k1_out_transformed_var, width * sizeof(float));

    // Allocation for histogram1D
    size_t k2_size = totalElements * sizeof(float);
    float *d_k2_output, *d_k2_input;
    cudaMalloc(&d_k2_output, nbins * sizeof(float));
    cudaMalloc(&d_k2_input, k2_size);

    // Copy data from host to device
    cudaMemcpy(d_k1_input, k1_input, total_elements * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k2_input, d_k2_input, k2_size, cudaMemcpyHostToDevice);
    cudaMemset(d_k2_output, 0, nbins * sizeof(float));

    /* Two kernel use different size of shared memory
     * We allocate one dynamic shared memory and they will share it. */
    size_t shmem_size = max((2 * 2 * WARP_SIZE + WARP_SIZE) * sizeof(int) +
                                (2 * WARP_SIZE) * sizeof(float),
                            nbins * sizeof(float));

    int gridDim = 128;
    int blockDim = 1024;
    hfused_kernel_kernel7_1<<<gridDim, blockDim, shmem_size>>>(
        d_k1_input, height, width, depth, epsilon, d_k1_out_mean,
        d_k1_out_transformed_var, d_k2_output, d_k2_input, nbins, minvalue,
        maxvalue, totalElements);

    // Copy data from device to host
    cudaMemcpy(h_mean, d_k1_out_mean, width * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_transformed_var, d_k1_out_transformed_var,
               width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k2_output, d_k2_output, nbins * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_k1_input);
    cudaFree(d_k1_out_mean);
    cudaFree(d_k1_out_transformed_var);
    cudaFree(d_k2_output);
    cudaFree(d_k2_input);
}