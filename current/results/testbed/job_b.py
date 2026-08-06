import os, time, torch, torch.distributed as dist, ctypes, torchvision

try:
    libc = ctypes.CDLL('libnccl.so.2')
    ll_start = libc.ncclLongLiuIterStart
    ll_end = libc.ncclLongLiuIterEnd
    longliu = True
except Exception:
    ll_start = None; ll_end = None; longliu = False

WARMUP = 10; ITERS = 210

dist.init_process_group('nccl')
rank = dist.get_rank()
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
print(f"[JOB B rank={rank}] longliu={longliu}", flush=True)

model = torchvision.models.resnet50(weights=None).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
x = torch.randn(32, 3, 224, 224, device='cuda')

times = []
for i in range(WARMUP + ITERS):
    if ll_start: ll_start()
    t0 = time.monotonic()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000
    if ll_end: ll_end()
    if i >= WARMUP:
        times.append(elapsed)
    if rank == 0 and i % 50 == 0:
        print(f"[JOB B] iter={i} {elapsed:.1f}ms", flush=True)

avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
print(f"\n[JOB B RESULT] avg={avg:.1f}ms p95={p95:.1f}ms n={len(times)}", flush=True)
dist.destroy_process_group()
