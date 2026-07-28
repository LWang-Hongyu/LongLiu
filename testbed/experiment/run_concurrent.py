#!/usr/bin/env python3
"""
LongLiu Concurrent Experiment Orchestrator.
Runs 3 scenarios: baseline, tight/loose, swapped.
"""
import subprocess, time, os, sys

REMOTE = "10.157.197.26"
REMOTE_USER = "why"
MASTER_ADDR = "192.10.10.226"
N_MB = 200
ITERS = 200
BASE_DIR = "/home/why/LongLiu_rebuild/testbed/experiment"

def ssh_cmd(host, cmd):
    return f"ssh {host} '{cmd}'"

def run_scenario(label, job1_ci, job2_ci, longliu_enabled):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {label}")
    print(f"  Job1 ci={job1_ci}  Job2 ci={job2_ci}  LongLiu={longliu_enabled}")
    print(f"{'='*60}\n")

    # Clean up old result files
    subprocess.run(f"rm -f /tmp/concurrent_job*.txt", shell=True, capture_output=True)
    subprocess.run(ssh_cmd(REMOTE, "rm -f /tmp/concurrent_job*.txt"), shell=True, capture_output=True)

    env_base = f"ALLREDUCE_MB={N_MB} ITERS={ITERS} MASTER_ADDR={MASTER_ADDR}"
    ll_vars = f"LONGLIU_ENABLED={longliu_enabled} LONGLIU_C_I="

    # Start workers on 10.1 (both on GPU 0, independent CUDA contexts)
    worker1 = (
        f"ssh -f {REMOTE} 'cd {BASE_DIR} && "
        f"CUDA_VISIBLE_DEVICES=0 JOB_ID=1 RANK=1 MASTER_PORT=29500 "
        f"{env_base} {ll_vars}{job1_ci} "
        f"python3 concurrent_ddp.py > /tmp/worker1_ll.log 2>&1'"
    )
    worker2 = (
        f"ssh -f {REMOTE} 'cd {BASE_DIR} && "
        f"CUDA_VISIBLE_DEVICES=0 JOB_ID=2 RANK=1 MASTER_PORT=29501 "
        f"{env_base} {ll_vars}{job2_ci} "
        f"python3 concurrent_ddp.py > /tmp/worker2_ll.log 2>&1'"
    )

    subprocess.run(worker1, shell=True)
    subprocess.run(worker2, shell=True)
    print(f"  Workers launched, waiting 5s for NCCL init...")
    time.sleep(5)

    # Start masters on 226
    master1 = (
        f"cd {BASE_DIR} && "
        f"CUDA_VISIBLE_DEVICES=0 JOB_ID=1 RANK=0 MASTER_PORT=29500 "
        f"{env_base} {ll_vars}{job1_ci} "
        f"python3 concurrent_ddp.py > /tmp/master1_ll.log 2>&1"
    )
    master2 = (
        f"cd {BASE_DIR} && "
        f"CUDA_VISIBLE_DEVICES=1 JOB_ID=2 RANK=0 MASTER_PORT=29501 "
        f"{env_base} {ll_vars}{job2_ci} "
        f"python3 concurrent_ddp.py > /tmp/master2_ll.log 2>&1"
    )

    p1 = subprocess.Popen(master1, shell=True)
    p2 = subprocess.Popen(master2, shell=True)

    # Wait with timeout
    start = time.time()
    timeout = 600
    while time.time() - start < timeout:
        if p1.poll() is not None and p2.poll() is not None:
            break
        time.sleep(2)

    if p1.poll() is None:
        print(f"  WARNING: Job 1 master timed out, killing...", flush=True)
        p1.kill()
    if p2.poll() is None:
        print(f"  WARNING: Job 2 master timed out, killing...", flush=True)
        p2.kill()

    time.sleep(2)  # let workers finish

    # Collect results
    print(f"\n--- Results for {label} ---")
    for fname in ["/tmp/concurrent_job1_rank0.txt", "/tmp/concurrent_job2_rank0.txt"]:
        try:
            with open(fname) as f:
                print(f.readline().strip())
        except:
            print(f"  MISSING: {fname}")

    # Collect worker results from 10.1
    for fname in ["concurrent_job1_rank1.txt", "concurrent_job2_rank1.txt"]:
        result = subprocess.run(
            f"cat /tmp/{fname} 2>/dev/null || {ssh_cmd(REMOTE, f'cat /tmp/{fname} 2>/dev/null || echo MISSING')}",
            shell=True, capture_output=True, text=True
        )
        line = result.stdout.strip().split('\n')[0] if result.stdout else "MISSING"
        print(f"  Worker: {line}")

    print()

if __name__ == "__main__":
    scenarios = [
        ("Baseline (LongLiu OFF)",          "1.5", "1.5", "0"),
        ("LongLiu Tight (J1=1.2, J2=3.0)",  "1.2", "3.0", "1"),
        ("LongLiu Swap (J1=3.0, J2=1.2)",   "3.0", "1.2", "1"),
    ]

    for label, j1_ci, j2_ci, ll_en in scenarios:
        run_scenario(label, j1_ci, j2_ci, ll_en)

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
