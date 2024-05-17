CUDA_ROOT_DIR = ${CUDA_HOME}

NVCC = $(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS = -lineinfo -arch=sm_80
NVCC_LIBS = 

CUDA_LIB_DIR = -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR = -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS = -lcudart -lcublas

all: cp_async mma ldmatrix

cp_async: cp_async_example.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

mma: mma_example.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

ldmatrix: ldmatrix.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

clean:
	$(RM) *.out