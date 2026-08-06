#!/usr/bin/env python3
"""
TC Sweep - reliable version.
Uses management interface for TCP bootstrap, IB for data.
SSH master runs in background via subprocess with stdin closed.
"""
import subprocess, sys, os, time

MASTER_HOST = "10.157.197.107"
TC_VALUES = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
BASE_PORT = 29700

def kill_all():
    subprocess.run("pkill -9 -f dscp_tc_sweep 2>/dev/null", shell=True)
    # Kill remote via mgmt ip
    subprocess.run(
        f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no {MASTER_HOST} "
        f"'pkill -9 -f dscp_tc_sweep 2>/dev/null' 2>/dev/null",
        shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_one(tc, port):
    """Run one TC value test. Uses mgmt IP for bootstrap."""
    
    # Start master on 226 via SSH in background
    # Key: close stdin so SSH doesn't block
    master_cmd = (
        f"source ~/.bashrc && cd /home/why/LongLiu_rebuild/testbed && "
        f"NCCL_DEBUG=WARN NCCL_IB_TC={tc} "
        f"NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 "
        f"NCCL_SOCKET_IFNAME=enp "
        f"MASTER_ADDR=192.10.10.226 MASTER_PORT={port} "
        f"WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 "
        f"LD_PRELOAD=/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 "
        f"python3 dscp_tc_sweep.py"
    )
    
    master_proc = subprocess.Popen(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
         MASTER_HOST, master_cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Give master time to start listening
    time.sleep(6)
    
    # Check if master is still alive
    if master_proc.poll() is not None:
        # Master already died - read its output
        master_out = master_proc.stdout.read().decode('utf-8', errors='replace')
        return tc, False, f"master died early: {master_out[:200]}", ""
    
    # Start worker locally
    env = os.environ.copy()
    env.update({
        'NCCL_IB_TC': str(tc),
        'NCCL_IB_HCA': 'mlx5_0',
        'NCCL_IB_GID_INDEX': '3',
        'NCCL_SOCKET_IFNAME': 'enp',
        'NCCL_DEBUG': 'INFO',
        'MASTER_ADDR': '192.10.10.226',
        'MASTER_PORT': str(port),
        'WORLD_SIZE': '2',
        'RANK': '1',
        'LOCAL_RANK': '0',
        'LD_PRELOAD': '/home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2',
    })
    
    worker_proc = subprocess.Popen(
        [sys.executable, '/home/why/LongLiu_rebuild/testbed/dscp_tc_sweep.py'],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    try:
        worker_out, _ = worker_proc.communicate(timeout=120)
        worker_ok = worker_proc.returncode == 0
    except subprocess.TimeoutExpired:
        worker_proc.kill()
        worker_proc.communicate()
        worker_ok = False
        worker_out = b"WORKER TIMEOUT"
    
    try:
        master_out, _ = master_proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        master_proc.kill()
        master_proc.communicate()
        master_out = b"MASTER TIMEOUT"
    
    worker_text = worker_out.decode('utf-8', errors='replace')
    
    # Extract NCCL_IB_TC log
    tc_line = ""
    for line in worker_text.split('\n'):
        if 'NCCL_IB_TC set by environment' in line:
            tc_line = line.strip()
            break
    
    success = worker_ok and ('TC_SWEEP_RESULT' in worker_text)
    
    if not success:
        err = ""
        for line in worker_text.split('\n'):
            if 'Error' in line or 'WARN socketWait' in line:
                err = line.strip()[:150]
                break
    
    return tc, success, tc_line, worker_text


if __name__ == "__main__":
    kill_all()
    time.sleep(2)
    
    print("=" * 60)
    print("DSCP TC Sweep (mgmt bootstrap, IB data)")
    print(f"Values: {TC_VALUES}")
    print("=" * 60)
    
    results = []
    for i, tc in enumerate(TC_VALUES):
        port = BASE_PORT + i
        print(f"[{i+1:2d}/{len(TC_VALUES)}] TC={tc:2d} ... ", end='', flush=True)
        
        tc_val, ok, tc_line, _ = run_one(tc, port)
        
        if ok:
            print(f"PASS  {tc_line}")
        else:
            print(f"FAIL  {tc_line}")
        
        results.append((tc_val, ok, tc_line))
        kill_all()
        time.sleep(2)
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    for tc, ok, tc_line in results:
        print(f"  TC={tc:2d}  {'PASS' if ok else 'FAIL'}")
    print(f"\nPassed: {passed}/{len(results)}")
