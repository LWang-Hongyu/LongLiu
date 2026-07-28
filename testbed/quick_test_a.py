import os, time, torch, torch.distributed as dist, ctypes

try:
    libc = ctypes.CDLL('libnccl.so.2')
    ll_start = libc.ncclLongLiuIterStart
    ll_end = libc.ncclLongLiuIterEnd
    longliu = True
except Exception:
    ll_start = None
    ll_end = None
    longliu = False

from transformers import BertForPreTraining, BertConfig

WARMUP = 5
ITERS = 20

dist.init_process_group('nccl')
rank = dist.get_rank()
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)

print(f'[JOB A rank={rank}] longliu={longliu}', flush=True)

config = BertConfig()
model = BertForPreTraining(config).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

x = torch.randint(0, 30522, (2, 128), device='cuda')
mask = torch.ones(2, 128, device='cuda', dtype=torch.long)
y = torch.randint(0, 30522, (2, 128), device='cuda')

times = []
for i in range(WARMUP + ITERS):
    if ll_start:
        ll_start()
    t0 = time.monotonic()
    loss = model(x, attention_mask=mask, labels=y).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000
    if ll_end:
        ll_end()
    if i >= WARMUP:
        times.append(elapsed)
    if rank == 0:
        print(f'[JOB A] iter={i} {elapsed:.1f}ms', flush=True)

if times:
    avg = sum(times) / len(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f'[JOB A RESULT] avg={avg:.1f}ms p95={p95:.1f}ms n={len(times)}', flush=True)
dist.destroy_process_group()
