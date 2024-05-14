CUDA_ROOT_DIR = ${CUDA_HOME}

NVCC = $(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS = -lineinfo
NVCC_LIBS = 

CUDA_LIB_DIR = -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR = -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS = -lcudart -lcublas

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

MAIN_OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/batch_norm_collect_statistics.o $(OBJ_DIR)/batch_norm_collect_statistics_cpu.o $(OBJ_DIR)/histogram1d.o $(OBJ_DIR)/histogram1d_cpu.o $(OBJ_DIR)/hfused_kernel.o $(OBJ_DIR)/bncs_and_hist.o
TUNNING_OBJS = $(OBJ_DIR)/tunning.o  $(OBJ_DIR)/hfused_kernel.o $(OBJ_DIR)/bncs_and_hist.o $(OBJ_DIR)/batch_norm_collect_statistics.o $(OBJ_DIR)/histogram1d.o
GEMM_TEST_OBJS = $(OBJ_DIR)/gemm_test.o $(OBJ_DIR)/gemm_cpu.o $(OBJ_DIR)/gemm_cublas.o
all: main tunning gemm_test

main : $(MAIN_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJS) -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

tunning : $(TUNNING_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(TUNNING_OBJS) -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

gemm_test : $(GEMM_TEST_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GEMM_TEST_OBJS) -o $@.out $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/%.o : %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	$(RM) bin/* *.o *.out