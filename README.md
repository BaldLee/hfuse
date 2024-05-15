# README

## How to make
1. Set `CUDA_HOME` to the path of the location of cuda. (`/usr/local/cuda` usually)
   - `export CUDA_HOME=/usr/local/cuda`
2. Just `make`

## How to plot
1. Install matplotlib
2. python python/plot_script.py -f out.json

## Research Notes
1. We fused bncs(batch_norm_collect_statistics) and hist(histogram1d). They are all IO bound kernels and the evaluation results are similar to the paper.
   1. Further TODO: go deep into these kernels by NSIGHT COMPUTE with Garfee.
2. Fuse IO bound kernel and compute bound kernel.
   1. Gemm, damn classic compute bound kernel. (It is IO bound, too)
   2. Now I have to implement gemm by myself first. There are a lot of mess, check it later.

### GEMM implement
1. I have done it on V100 before, now I want to rebuild it on A100. Ampere has brought a lot of new features.
2. Tiling is hard, but I don't want to note them here.
3. Kernels can be imporved by PTX. There are some details. Refer ptx doc for anything.
   1. `mma.sync.aligned`. The mma instrcution.
   2. `cp.async.ca`. Copy data from global mem to shared mem asynchronously.
      1. A simple example locates in `examples/cp_async_example.cu`.
   3. `ldmatrix.sync.aligned`. Load matrices from shared mem for mma instruction. It is in warp level.