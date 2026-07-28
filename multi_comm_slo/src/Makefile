# Makefile for libmulti_comm.so
# Uses NCCL 2.30.7 with native trafficClass support

NCCL_HOME ?= /home/why/LongLiu_rebuild/nccl-master
NCCL_LIB ?= /lib/x86_64-linux-gnu
CUDA_HOME ?= /usr/local/cuda

CC = gcc
CFLAGS = -fPIC -O2 -Wall -I$(NCCL_HOME)/build/include -I$(CUDA_HOME)/include -I.
LDFLAGS = -Wl,--no-as-needed -shared -L$(NCCL_LIB) -lnccl -L$(CUDA_HOME)/lib64 -lcudart

TARGET = ../build/libmulti_comm.so
SRC = multi_comm.c

all: $(TARGET)

$(TARGET): $(SRC) multi_comm.h
	@mkdir -p ../build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
