#!/usr/bin/env python3
"""RDMA 严格优先级实验 — 用 --tos 直接设置 DSCP（rdma_cm 模式）"""
import subprocess, time, os, sys, threading

NODE_226 = "192.10.10.226"
DEV = "mlx5_0"
OUTDIR = "/tmp/ib_results"
os.makedirs(OUTDIR, exist_ok=True)

def ssh(cmd, timeout=10):
    """Run SSH command, return (rc, stdout, stderr)"""
    try:
        r = subprocess.run(
            ["ssh", NODE_226, cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"

def start_server(port, mode="sl", sl=0, tos=None):
    """Start ib_write_bw server on 226"""
    cmd = f"nohup ib_write_bw -R --port={port} -d {DEV} --report_gbits"
    if mode == "tos" and tos is not None:
        cmd += f" --tos={tos}"
    else:
        cmd += f" --sl={sl}"
    cmd += " -D 30"
    cmd += f" > /tmp/ib_srv_{port}.log 2>&1 &"
    rc, out, err = ssh(cmd)
    return rc

def run_client(port, label, mode="sl", sl=0, tos=None, duration=8, outfile=None):
    """Run ib_write_bw client, return results dict"""
    cmd = ["ib_write_bw", "-R", f"--port={port}", NODE_226, "-d", DEV, "--report_gbits"]
    if mode == "tos" and tos is not None:
        cmd += [f"--tos={tos}"]
    else:
        cmd += [f"--sl={sl}"]
    cmd += ["-D", str(duration)]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration+10)
        # Parse: " 65536    NNNNNN   0.00   XX.XX  YYY"
        for line in r.stdout.split('\n'):
            if 'BW average' in line or (line.strip() and line.strip().split()[0] == '65536'):
                parts = line.strip().split()
                if len(parts) >= 4:
                    bw = parts[-2]
                    if outfile:
                        with open(outfile, 'w') as f:
                            f.write(line + '\n')
                    return {"label": label, "port": port, "bw": bw, "raw": line.strip()}
    except Exception as e:
        return {"label": label, "port": port, "error": str(e)}
    return {"label": label, "port": port, "error": "no data"}

def run_experiment_dual(port_a, label_a, port_b, label_b, mode_a="sl", sl_a=0, tos_a=None, mode_b="sl", sl_b=0, tos_b=None, duration=8):
    """Run dual QP experiment with concurrent clients"""
    print(f"\n=== Dual QP: {label_a} vs {label_b} ===")
    
    # Start servers
    if mode_a == "tos":
        start_server(port_a, mode="tos", tos=tos_a)
    else:
        start_server(port_a, mode="sl", sl=sl_a)
    
    if mode_b == "tos":
        start_server(port_b, mode="tos", tos=tos_b)
    else:
        start_server(port_b, mode="sl", sl=sl_b)
    
    time.sleep(4)
    
    # Verify servers are running
    rc, out, _ = ssh("ps aux | grep 'ib_write_bw -R' | grep -v grep | wc -l")
    n_servers = int(out.strip() or 0)
    print(f"  Servers running: {n_servers}")
    if n_servers < 2:
        print("  WARNING: Not all servers started!")
        rc, out, _ = ssh("ps aux | grep 'ib_write_bw -R' | grep -v grep")
        print(f"  Running: {out}")
    
    # Start clients concurrently
    out_a = os.path.join(OUTDIR, f"dual_{label_a}.txt")
    out_b = os.path.join(OUTDIR, f"dual_{label_b}.txt")
    
    t1 = threading.Thread(target=lambda: run_client(port_a, label_a, mode_a, sl_a, tos_a, duration, out_a))
    t2 = threading.Thread(target=lambda: run_client(port_b, label_b, mode_b, sl_b, tos_b, duration, out_b))
    
    t1.start()
    time.sleep(0.5)  # stagger slightly to ensure truly concurrent
    t2.start()
    
    t1.join()
    t2.join()
    
    # Print results
    for outfile, lbl in [(out_a, label_a), (out_b, label_b)]:
        if os.path.exists(outfile):
            with open(outfile) as f:
                print(f"  {lbl}: {f.read().strip()}")

def run_single(port, label, mode="sl", sl=0, tos=None, duration=8):
    """Single client baseline"""
    print(f"\n=== Single QP: {label} ===")
    if mode == "tos":
        start_server(port, mode="tos", tos=tos)
    else:
        start_server(port, mode="sl", sl=sl)
    time.sleep(3)
    
    outfile = os.path.join(OUTDIR, f"single_{label}.txt")
    result = run_client(port, label, mode, sl, tos, duration, outfile)
    if result and "raw" in result:
        print(f"  {label}: {result['raw']}")

# ===== MAIN =====
print("=" * 60)
print("RDMA 严格优先级实验 — 10.1 (50G) → 226 (100G)")
print(f"HCA QoS: tsa=strict, DSCP trust, rdma_cm mode")
print("=" * 60)

# Cleanup
ssh("pkill -9 ib_write_bw")

# 实验1: 单 QP 基线 (tos=0)
run_single(22001, "tos0_baseline", mode="tos", tos=0)
time.sleep(2)

# 实验2: 双 QP 同 tos=0（公平基线）
run_experiment_dual(
    22010, "tos0_A", 22011, "tos0_B",
    mode_a="tos", tos_a=0,
    mode_b="tos", tos_b=0
)
time.sleep(2)

# 实验3: 双 QP tos=0(low) vs tos=56(high, TC7)
run_experiment_dual(
    22020, "tos0_low", 22021, "tos56_high",
    mode_a="tos", tos_a=0,
    mode_b="tos", tos_b=56
)
time.sleep(2)

# 实验4: 双 QP tos=0(low) vs tos=48(CNP, TC6)
run_experiment_dual(
    22030, "tos0_low", 22031, "tos48_TC6",
    mode_a="tos", tos_a=0,
    mode_b="tos", tos_b=48
)
time.sleep(2)

# 实验5: 双 QP tos=0 vs tos=24 (TC3)
run_experiment_dual(
    22040, "tos0_low", 22041, "tos24_TC3",
    mode_a="tos", tos_a=0,
    mode_b="tos", tos_b=24
)

# Cleanup
ssh("pkill -9 ib_write_bw")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
