import os, time, torch, torch.distributed as dist, ctypes

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
print(f"[JOB A rank={rank}] longliu={longliu}", flush=True)

# Medium model (~134M params, ~2GB with optimizer)
class MidModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 4096)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x.sum()

model = MidModel().cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
x = torch.randn(64, 4096, device='cuda')

times = []
for i in range(WARMUP + ITERS):
    if ll_start: ll_start()
    t0 = time.monotonic()
    loss = model(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000
    if ll_end: ll_end()
    if i >= WARMUP:
        times.append(elapsed)
    if rank == 0 and i % 50 == 0:
        print(f"[JOB A] iter={i} {elapsed:.1f}ms", flush=True)

avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
print(f"\n[JOB A RESULT] avg={avg:.1f}ms p95={p95:.1f}ms n={len(times)}", flush=True)
dist.destroy_process_group()
