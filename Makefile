CUDA_ROOT_DIR = /usr/local/cuda-11.7

NVCC = nvcc
NVCC_FLAGS = 
NVCC_LIBS = 

CUDA_LIB_DIR = -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR = -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS = -lcudart

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

EXE = main
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/batch_norm_collect_statistics_gpu.o $(OBJ_DIR)/batch_norm_collect_statistics_cpu.o $(OBJ_DIR)/histogram1d_gpu.o $(OBJ_DIR)/histogram1d_cpu.o 

$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/%.o : %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	$(RM) bin/* *.o $(EXE)