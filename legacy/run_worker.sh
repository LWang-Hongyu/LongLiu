#!/bin/bash
# Run on guolab-10 (10.1)
set -e

EXPERIMENT=
MASTER_A=29500
MASTER_B=29501
SCRIPT_DIR=/home/why/LongLiu_rebuild/testbed

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
export NCCL_IB_HCA=mlx5_0

run_job_a() {
    local ll_enabled=
    local ll_ci=
    echo [10.1 JOB A] LL= ci=
    NCCL_IB_HCA=mlx5_0 LONGLIU_ENABLED= LONGLIU_C_I=     CUDA_VISIBLE_DEVICES=0     python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1         --master_addr=10.157.197.107 --master_port=         /job_a.py 2>&1 | tee /tmp/worker_a_.log &
    WA=
}

run_job_b() {
    local ll_enabled=
    local ll_ci=
    echo [10.1 JOB B] LL= ci=
    NCCL_IB_HCA=mlx5_0 LONGLIU_ENABLED= LONGLIU_C_I=     CUDA_VISIBLE_DEVICES=0     python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1         --master_addr=10.157.197.107 --master_port=         /job_b.py 2>&1 | tee /tmp/worker_b_.log &
    WB=
}

echo === 10.1 WORKER:  ===

case  in
    baseline)
        run_job_a 0 1.5; run_job_b 0 1.5
        wait ; wait 
        echo === 10.1 WORKER DONE ===
        ;;
    tight)
        run_job_a 1 1.2; run_job_b 1 2.0
        wait ; wait 
        echo === 10.1 WORKER DONE ===
        ;;
    swap)
        run_job_a 1 2.0; run_job_b 1 1.2
        wait ; wait 
        echo === 10.1 WORKER DONE ===
        ;;
    *)
        echo Usage: bash {baseline|tight|swap}
        exit 1
        ;;
esac

echo === RESULT ===
grep RESULT /tmp/worker_a_.log /tmp/worker_b_.log 2>/dev/null
