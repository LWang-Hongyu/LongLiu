#!/usr/bin/env python3
"""
P4 GPT Training — Real distributed training over P4 switch with DSCP priorities.

Model: GPT-2 style transformer (~78M params, ~312MB gradients)
Modes:
  solo    — Single job, standard NCCL (baseline)
  fair    — Both jobs, standard NCCL (no prioritization)
  longliu — Both jobs, MultiCommWrapper (DSCP priority scheduling)

Usage:
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.10.10.110 MASTER_PORT=29500 \
    python3 p4_train_gpt.py --mode longliu --job 1
"""

import os, sys, time, csv, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# GPT-2 style model
# ============================================================
class GPTConfig:
    def __init__(self, vocab_size=50257, n_embd=768, n_layer=6,
                 n_head=12, n_positions=512, d_ff=3072, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions
        self.d_ff = d_ff
        self.dropout = dropout


def tiny_config():
    """Smaller GPT config for dual-job GPU memory constrained scenarios."""
    return GPTConfig(
        vocab_size=50257,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_positions=256,
        d_ff=2048,
        dropout=0.1,
    )

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.c_proj(attn))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Bandwidth correction: for Ring AllReduce with N ranks,
# per-direction wire bandwidth = size_Gb * (n-1)/n / time.
# ============================================================
def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur


# ============================================================
# Gradient allreduce helpers
# ============================================================
def flatten_gradients(model):
    """Flatten all parameter gradients into a single 1D tensor."""
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        return None
    return torch.cat(grads)

def unflatten_gradients(model, flat_grad):
    """Copy flattened gradients back into model parameters."""
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        numel = p.grad.numel()
        p.grad.data.copy_(flat_grad[offset:offset+numel].view(p.grad.shape))
        offset += numel


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['solo', 'fair', 'longliu'])
    parser.add_argument('--job', type=int, required=True, choices=[1, 2],
                        help='Job ID (1=strict SLO, 2=loose SLO)')
    parser.add_argument('--no-warmup', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--tiny', action='store_true',
                        help='Use tiny GPT config (512D, 6L, 256seq) for GPU memory savings')
    args = parser.parse_args()

    MODE = args.mode
    JOB_ID = args.job
    BATCH_SIZE = args.batch_size
    SEQ_LEN = args.seq_len

    # Job config
    if JOB_ID == 1:
        SLO_C_I = 1.5
        NUM_ITERS = 300
        ITERS_PER_WINDOW = 20
        NUM_WINDOWS = NUM_ITERS // ITERS_PER_WINDOW  # 15
    else:
        SLO_C_I = 2.5
        NUM_ITERS = 200
        ITERS_PER_WINDOW = 20
        NUM_WINDOWS = NUM_ITERS // ITERS_PER_WINDOW  # 10

    _adapter_enabled = (MODE == 'longliu')

    # Model
    if args.tiny:
        config = tiny_config()
        SEQ_LEN = config.n_positions  # override seq_len to match positional embeddings
        print(f"[Job{JOB_ID}] Using TINY config: {config.n_embd}D, {config.n_layer}L, "
              f"seq={SEQ_LEN}, batch={BATCH_SIZE}")
    else:
        config = GPTConfig(n_positions=SEQ_LEN)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = GPT(config).to(device)

    # Per-iteration gradient size
    param_bytes = sum(p.numel() * 4 for p in model.parameters())
    BYTES_PER_ITER = param_bytes  # gradient allreduce size

    if torch.cuda.current_device() == 0:
        print(f"[Job{JOB_ID}] MODE={MODE}, Model=GPT({config.n_layer}L,{config.n_embd}D), "
              f"Params={param_bytes/1e6:.0f}M, GradSize={BYTES_PER_ITER/1e6:.0f}MB")
        print(f"[Job{JOB_ID}] {NUM_ITERS} iters, {ITERS_PER_WINDOW}/window, "
              f"{NUM_WINDOWS} windows, SLO c_i={SLO_C_I}, "
              f"batch={BATCH_SIZE}, seq={SEQ_LEN}")

    # ============================================================
    # Distributed setup + MultiComm (longliu mode)
    # ============================================================
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    _mc_wrapper = None
    _slo_scheduler = None

    if _adapter_enabled:
        master_addr = os.environ.get('MASTER_ADDR', '192.10.10.110')
        port = int(os.environ.get('MULTI_COMM_PORT',
                    os.environ.get('MASTER_PORT', '29500')))

        from slo_scheduler import MultiCommWrapper, SLOScheduler
        _slo_scheduler = SLOScheduler(
            slo_threshold=SLO_C_I,
        )
        _mc_wrapper = MultiCommWrapper(
            _slo_scheduler, rank, world_size, '0',
            master_addr, port)

        def window_start(window):
            if _mc_wrapper is not None:
                _mc_wrapper.window_start(window)

        def window_end(window):
            if _mc_wrapper is not None:
                _mc_wrapper.window_end(window, data_size=BYTES_PER_ITER)

        def allreduce(tensor):
            if _mc_wrapper is not None:
                # datatype 必须传 ncclFloat32(=7)。传 0(=ncclInt8) 会让 NCCL
                # 只同步 count×1B（梯度 float32 的 1/4）且按 int8 求和 → 训练数值无效。
                # 2026-08-10 修正：此前 7/16 的 longliu 运行即受此 bug 影响
                # （solo 76ms vs longliu 28ms 的 3-4 倍差距即 1/4 传输量的证据）。
                _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                                       tensor.numel(), 7, 0, 0)
            else:
                torch.distributed.all_reduce(tensor)
    else:
        torch.distributed.init_process_group('nccl')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        def window_start(window):
            pass

        def window_end(window):
            pass

        def allreduce(tensor):
            torch.distributed.all_reduce(tensor)

    # ============================================================
    # Optimizer
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ============================================================
    # Training loop
    # ============================================================
    t_total_start = time.perf_counter()
    results = []

    for window in range(NUM_WINDOWS):
        window_start(window)

        for i in range(ITERS_PER_WINDOW):
            global_iter = window * ITERS_PER_WINDOW + i

            # Generate random training data
            input_ids = torch.randint(0, config.vocab_size,
                                      (BATCH_SIZE, SEQ_LEN), device=device)
            labels = torch.randint(1, config.vocab_size,
                                   (BATCH_SIZE, SEQ_LEN), device=device)

            # Forward
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, config.vocab_size),
                           labels.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # AllReduce gradients
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            flat_grad = flatten_gradients(model)
            if flat_grad is not None:
                allreduce(flat_grad)
                unflatten_gradients(model, flat_grad)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            comm_dur = t1 - t0

            bw_gbps = bus_bw_gbps(BYTES_PER_ITER, comm_dur, world_size) \
                      if comm_dur > 0 else 0.0

            # Optimizer step
            optimizer.step()

            # Phase determination
            if MODE == 'solo':
                phase = 'solo'
            elif JOB_ID == 2:
                phase = 'contested'
            elif global_iter < 100:
                phase = 'solo_rampup'
            else:
                phase = 'contested'

            if rank == 0:
                print(f"[Job{JOB_ID}] iter {global_iter:2d} (window {window}, "
                      f"iter {i}): comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps, loss={loss.item():.4f} [{phase}]")

            results.append({
                'iter': global_iter,
                'window': window,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
                'phase': phase,
                # loss 用于验证训练数值有效性：datatype 修复后（ncclFloat32）
                # loss 应随迭代稳定下降；若为 int8 则梯度 1/4 且求和错误，loss 不降。
                'loss': round(loss.item(), 4),
            })

        window_end(window)

    t_total_end = time.perf_counter()

    # ============================================================
    # Save results
    # ============================================================
    if rank == 0:
        csv_path = f'/tmp/p4_train_JOB{JOB_ID}_{MODE}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'window', 'comm_dur_s',
                             'bw_gbps', 'phase', 'loss'])
            for r in results:
                writer.writerow([r['iter'], r['window'],
                                 r['comm_dur_s'], r['bw_gbps'],
                                 r['phase'], r['loss']])
        print(f"[Job{JOB_ID}] Results saved to {csv_path}")

        # Summary
        contested = [r for r in results if r['phase'] == 'contested']
        solo_phase = [r for r in results if r['phase'] == 'solo_rampup']

        if solo_phase:
            avg_solo = sum(r['comm_dur_s'] for r in solo_phase) / len(solo_phase)
            avg_bw = sum(r['bw_gbps'] for r in solo_phase) / len(solo_phase)
            print(f"[Job{JOB_ID}] Solo ({len(solo_phase)} iters): "
                  f"avg_comm={avg_solo*1000:.1f}ms, avg_bw={avg_bw:.2f} Gbps")

        if contested:
            avg_cont = sum(r['comm_dur_s'] for r in contested) / len(contested)
            avg_bw = sum(r['bw_gbps'] for r in contested) / len(contested)
            print(f"[Job{JOB_ID}] Contested ({len(contested)} iters): "
                  f"avg_comm={avg_cont*1000:.1f}ms, avg_bw={avg_bw:.2f} Gbps")

        print(f"[Job{JOB_ID}] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.destroy()
    elif not _adapter_enabled:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
