# Makefile for libmulti_comm.so
# Uses NCCL 2.30.7 with native trafficClass support

# System NCCL: /usr/include/nccl.h (2.29.7 header, has trafficClass) +
# libnccl.so.2.30.7. Local nccl-master source build no longer exists.
# NOTE: gcc searches /usr/local/include before /usr/include, and the
# /usr/local/include/nccl.h is old (v21700, no trafficClass), so we vendor
# the correct header in ./include/ and put it first in the include path.
NCCL_HOME ?= /usr
NCCL_LIB ?= /usr/lib/x86_64-linux-gnu
CUDA_HOME ?= /usr/local/cuda

CC = gcc
CFLAGS = -fPIC -O2 -Wall -Iinclude -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I.
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
