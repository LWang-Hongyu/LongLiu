#!/usr/bin/env python3
"""
LongLiu Concurrent DDP Experiment - Script to run as either Job A or Job B.

Env vars:
  JOB_ID: 1 or 2 (identifies the job)
  RANK: 0=master(226) or 1=worker(10.1)
  MASTER_PORT: 29500 for J1, 29501 for J2
  MASTER_ADDR: 192.10.10.226
  LONGLIU_ENABLED: 0 or 1
  LONGLIU_C_I: 1.2~3.0 (relaxation coefficient)
  ALLREDUCE_MB: message size in MB (default 200)
  ITERS: number of iterations (default 200)
"""
import os, time, torch, torch.distributed as dist
import ctypes

job_id = os.environ['JOB_ID']
rank = int(os.environ['RANK'])
master_port = os.environ['MASTER_PORT']
master_addr = os.environ.get('MASTER_ADDR', '192.10.10.226')
n_mb = int(os.environ.get('ALLREDUCE_MB', '200'))
iters = int(os.environ.get('ITERS', '200'))
ll_enabled = os.environ.get('LONGLIU_ENABLED', '0')
ll_ci = os.environ.get('LONGLIU_C_I', '1.5')

os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = str(rank)

ll_start = ll_end = None
if ll_enabled == '1':
    try:
        libc = ctypes.CDLL('libnccl.so.2')
        ll_start = libc.ncclLongLiuIterStart
        ll_end   = libc.ncclLongLiuIterEnd
        print(f'[JOB {job_id}] LongLiu ctypes loaded', flush=True)
    except Exception as e:
        print(f'[JOB {job_id}] LongLiu ctypes FAILED: {e}', flush=True)

dist.init_process_group('nccl')
torch.cuda.set_device(0)

n = (n_mb * 1024 * 1024) // 4
data = torch.randn(n, device='cuda')

for _ in range(3):
    dist.all_reduce(data)
torch.cuda.synchronize()

times = []
for i in range(iters):
    if ll_start: ll_start()
    t0 = time.monotonic()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000
    if ll_end: ll_end()
    times.append(elapsed)
    if i % 20 == 0:
        print(f'[JOB {job_id} RANK {rank} iter {i}] {elapsed:.1f}ms', flush=True)

avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
p99 = sorted(times)[int(len(times) * 0.99)]
result_line = f'RESULT | JOB {job_id} RANK {rank} | LL={ll_enabled} ci={ll_ci} | avg={avg:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms n={len(times)}'
print(f'\n{result_line}', flush=True)

result_file = f'/tmp/concurrent_job{job_id}_rank{rank}.txt'
with open(result_file, 'w') as f:
    f.write(result_line + '\n')
    for t in times:
        f.write(f'{t:.3f}\n')

dist.destroy_process_group()
