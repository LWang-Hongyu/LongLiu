#!/usr/bin/env python3
"""LongLiu DSCP DDP training with real models (MidModel / ResNet-50)."""
import os, time, torch, torch.distributed as dist

JOB_ID = os.environ.get("JOB_ID", "0")
RANK = int(os.environ.get("RANK", "0"))
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
MASTER_ADDR = os.environ.get("MASTER_ADDR", "192.10.10.226")
MODEL = os.environ.get("DDP_MODEL", "mid")  # mid or resnet
WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "200"))

os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = MASTER_PORT
os.environ["WORLD_SIZE"] = "2"

# Remove old LongLiu env vars (DSCP handles scheduling now)
# NCCL_DSCP_* env vars are set by launch scripts

dist.init_process_group("nccl")
torch.cuda.set_device(0)
dist.all_reduce(torch.randn(1, device=chr(99)+chr(117)+chr(100)+chr(97))); torch.cuda.synchronize()
rank = dist.get_rank()

if MODEL == "mid":
    class MidModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4096, 4096)
            self.fc2 = torch.nn.Linear(4096, 4096)
            self.fc3 = torch.nn.Linear(4096, 4096)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x.sum()
    model = MidModel().cuda()
    x = torch.randn(64, 4096, device="cuda")
    label = "MidModel(134M)"
elif MODEL == "resnet":
    import torchvision
    model = torchvision.models.resnet50(weights=None).cuda()
    x = torch.randn(32, 3, 224, 224, device="cuda")
    label = "ResNet-50(25M)"
else:
    # Fallback: pure AllReduce benchmark
    model = None
    n = (200 * 1024 * 1024) // 4
    data = torch.randn(n, device="cuda")
    label = "AllReduce(200MB)"

if model is not None:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print(f"[JOB {JOB_ID} rank={rank}] model={label} DSCP enabled", flush=True)

times = []
for i in range(WARMUP + ITERS):
    t0 = time.monotonic()
    if model is not None:
        loss = model(x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        dist.all_reduce(data)
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000
    if i >= WARMUP:
        times.append(elapsed)
        if i % 20 == 0:
            print(f"[JOB {JOB_ID} rank={rank}] iter {i} {elapsed:.1f}ms", flush=True)

avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
result = f"RESULT | JOB {JOB_ID} RANK {rank} | {label} | avg={avg:.1f}ms p95={p95:.1f}ms n={len(times)}"
print(f"\n{result}", flush=True)

rf = f"/tmp/concurrent_job{JOB_ID}_rank{rank}.txt"
with open(rf, "w") as f:
    f.write(result + "\n")
    for t in times:
        f.write(f"{t:.3f}\n")

dist.destroy_process_group()
