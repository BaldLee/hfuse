#include <cuda_runtime.h>

#include "../include/hfused_kernel.cuh"

/*
void fused_kernel(...){
  //Prologue of the fused kernel
  intglobal_tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *
BlockDim.x * blockDim.y; int threadIdx_x , threadIdx_y , threadIdx_z; int
blockDim_x , blockDim_y , blockDim_z; if (global_tid < 896) { blockDim_x = 896 /
16; blockDim_y = 16; blockDim_z = 1; threadIdx_x = global_tid % blockDim_x;
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
*/

__global__ void hfused_kernel_kernel(
    const float* k1_input, int height, int width, int depth, float epsilon,
    float* h_mean, float* h_transformed_var /* params for kernel1*/,
    float* k2_output, const float* k2_input, int nbins, float minvalue,
    float maxvalue, int totalElements /* params for kernel2*/) {
      
    }

void hfused_kernel(const float* k1_input, int height, int width, int depth,
                   float epsilon, float* h_mean,
                   float* h_transformed_var /* params for kernel1*/,
                   float* k2_output, const float* k2_input, int nbins,
                   float minvalue, float maxvalue,
                   int totalElements /* params for kernel2*/
) {}