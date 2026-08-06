#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp130s0f0np0
export NCCL_DSCP_ADAPTER_ENABLED=1
export NCCL_DSCP_UPDATE_IB_QP=1
export NCCL_DSCP_SLO_THRESHOLD=$3
export JOB_ID=$1
export RANK=1
export MASTER_PORT=$2
export ITERS=200
export MASTER_ADDR=192.10.10.226
export DDP_MODEL=$4
nohup python3 /home/why/LongLiu_rebuild/testbed/experiment/ddp_model.py > /tmp/worker${1}_ll.log 2>&1 &
echo "Worker $1 port $2 ci=$3 model=$4 PID=$!"
