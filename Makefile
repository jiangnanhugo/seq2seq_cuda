GPU=1
CUDNN=1
MKL=1
GFLAGS=1

CUDA_DIR := /beluga-scratch/apps/cuda/cuda-8.0
CUDA_INCLUDE = "$(CUDA_DIR)/include"
CUDA_LIB = "$(CUDA_DIR)/lib64"
MKL_DIR := /net/kihara/home/wang3702/Workspace/intel_mkl/mkl
MKL_INCLUDE = "$(MKL_DIR)/include"
MKL_LIB = "$(MKL_DIR)/lib/intel64"
GFLAGS_DIR:=/net/kihara/home/wang3702/Workspace/gflags-2.2.1/build
GFLAGS_LIB = "$(GFLAGS_DIR)/lib"
GFLAGS_INCLUDE ="$(GFLAGS_DIR)/include"


OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./tools
SLIB=libseq2seq.so
ALIB=libseq2seq.a
EXEC=seq2seq
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=/beluga-scratch/apps/cuda/cuda-8.0/bin/nvcc
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread
NVCC_LDFLAGS= -lm
COMMON=-I"include/"
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -std=c++11 -g

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
endif
ifeq ($(MKL), 1)
COMMON+= -I$(MKL_INCLUDE)
LDFLAGS+= -L$(MKL_LIB) -lmkl_rt -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
NVCC_LDFLAGS+= -L$(MKL_LIB) -lmkl_rt -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
endif

ifeq ($(GFLAGS), 1)
COMMON+= -I$(GFLAGS_INCLUDE)
LDFLAGS+= -L$(GFLAGS_LIB) -lgflags
NVCC_LDFLAGS+= -L$(GFLAGS_LIB) -lgflags
endif

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(GPU), 1)
COMMON+= -DGPU -I$(CUDA_INCLUDE)
CFLAGS+= -DGPU -use_fast_math
NVCC_CFLAGS= -ccbin $(CPP) -g -std=c++11 -use_fast_math -G -Xcompiler -fPIC
LDFLAGS+= -L$(CUDA_LIB) -lcuda -lcudart -lcublas -lcurand
NVCC_LDFLAGS+= -L$(CUDA_LIB) -lcuda -lcudart -lcublas -lcurand
endif
ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
CFLAGS+= -DCUDNN
NVCC_CFLAGS+= -DCUDNN
NVCC_LDFLAGS+= -lcudnn
endif

OBJ=common.o embd.o softmax.o blob.o loss.o data_reader.o cudnn_util.o gpu_common.o fc.o activation.o rnn.o optimizer.o model.o attention_decoder.o
EXECOBJA=common.o embd.o softmax.o blob.o loss.o data_reader.o cudnn_util.o gpu_common.o fc.o activation.o rnn.o optimizer.o model.o attention_decoder.o seq2seq.o 

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard include/*.h)

all:obj $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(NVCC) $(COMMON) $(ARCH) $(NVCC_CFLAGS) $^ -o $@ $(NVCC_LDFLAGS) $(ALIB)
# $(EXEC): $(EXECOBJ) $(ALIB)
# 	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_CFLAGS) $(ARCH) $(COMMON)  -c $< -o $@ $(NVCC_LDFLAGS)

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*
