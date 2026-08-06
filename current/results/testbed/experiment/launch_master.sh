#!/bin/bash
export CUDA_VISIBLE_DEVICES=$2
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export JOB_ID=$1
export RANK=0
export MASTER_PORT=$3
export ALLREDUCE_MB=200
export ITERS=200
export MASTER_ADDR=192.10.10.226
export NCCL_DSCP_ADAPTER_ENABLED=1
export NCCL_DSCP_UPDATE_IB_QP=1
export NCCL_DSCP_SLO_THRESHOLD=$4
nohup python3 /home/why/LongLiu_rebuild/testbed/experiment/concurrent_ddp.py > /tmp/master${1}_ll.log 2>&1 &
echo "Master $1 launched on GPU$2, port $3, ci=$4, PID=$!"
