"""feas_boundary_v3 场景设计辅助工具。

不运行仿真，只计算 attain 表和约束检查。
输出设计稿供用户批准。
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms

OVERHEAD = 1.3
OVERLAP = 0.85
HOST_BW = 100e9  # 100 Gbps


def bits_per_iter(model: str, dp: int) -> float:
    """计算 bits_per_iter（与 synthetic.py 的 _compute_mb_per_iter 同源）。"""
    params = MODEL_PARAMS[model]
    bpp = 2 if params.get("fp16", True) else 4
    bytes_per_iter = 2 * params["params"] * bpp / max(dp, 1)
    mb_per_iter = bytes_per_iter / (1024 * 1024)
    return mb_per_iter * 8 * 1024 * 1024


def comm_solo_ms(model: str, dp: int) -> float:
    return bits_per_iter(model, dp) / HOST_BW * 1000.0


def attain_bw_gbps(model: str, dp: int, ci: float, comp_ms: float | None = None) -> float:
    """锚点语义 attain_bw (Gbps)."""
    bits = bits_per_iter(model, dp)
    wire_bits = bits * OVERHEAD
    comm = comm_solo_ms(model, dp)
    if comp_ms is None:
        comp_ms = get_comp_ms(model, default=50.0)
    comm_budget = ci * comm * OVERHEAD
    target = max(comp_ms, comm_budget) + (1 - OVERLAP) * min(comp_ms, comm_budget)
    eff = target - comp_ms
    if eff <= 0:
        return float('inf')
    return wire_bits / (eff * 1e-3) / 1e9


def main():
    # 候选模型
    models = ["LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16", 
              "BERT-Large-fp16", "ViT-Large", "ViT-Base", "BERT-Base"]
    
    print("=== 模型 attain 速查表 (anchor semantics) ===")
    print(f"{'model':<20} {'dp':>3} {'ci':>5} {'comp_ms':>8} {'bits':>12} {'comm_solo':>10} {'attain(G)':>10}")
    for m in models:
        for dp in [1, 2, 4, 8]:
            for ci in [1.3, 1.5, 2.0, 2.5, 3.0]:
                try:
                    comp = get_comp_ms(m, default=50.0)
                    bits = bits_per_iter(m, dp) / 1e6  # Mbits
                    comm = comm_solo_ms(m, dp)
                    att = attain_bw_gbps(m, dp, ci, comp)
                    if att < 500:  # reasonable range
                        print(f"{m:<20} {dp:>3} {ci:>5.1f} {comp:>8.1f} {bits:>12.1f}M {comm:>10.1f}ms {att:>10.1f}G")
                except Exception as e:
                    pass

    print()
    print("="*60)
    print("设计约束：")
    print("  Σattain_P ∈ [600, 640] Gbps")
    print("  Σattain_S ∈ [240, 280] Gbps")
    print("  C* = ΣP + 0.5·ΣS ∈ [730, 780] G")
    print("  13-15 jobs, premium/standard 混层，全跨 pod")
    print("="*60)
    
    # 尝试几个设计
    designs = [
        {
            "name": "Design A: 8P+5S",
            "premium": [
                ("LLaMA-2-13B", 8, 1.5, 4),
                ("LLaMA-2-7B", 8, 1.5, 2),
                ("BERT-Large-fp16", 4, 2.0, 1),
                ("T5-11B-fp16", 8, 2.0, 1),
            ],
            "standard": [
                ("LLaMA-2-13B", 8, 3.0, 2),
                ("BERT-Large-fp16", 4, 3.0, 2),
                ("ViT-Base", 2, 3.0, 1),
            ],
        },
        {
            "name": "Design B: 9P+5S",
            "premium": [
                ("LLaMA-2-13B", 8, 1.5, 3),
                ("LLaMA-2-7B", 8, 1.5, 3),
                ("BERT-Large-fp16", 4, 1.5, 1),
                ("T5-11B-fp16", 8, 2.0, 1),
                ("BERT-Large-fp16", 2, 2.0, 1),
            ],
            "standard": [
                ("LLaMA-2-13B", 8, 3.0, 2),
                ("BERT-Large-fp16", 4, 3.0, 2),
                ("ViT-Base", 2, 3.0, 1),
            ],
        },
    ]
    
    for d in designs:
        print(f"\n--- {d['name']} ---")
        sp = 0
        ss = 0
        nj = 0
        for tier_name, tier_list in [("P", d["premium"]), ("S", d["standard"])]:
            for m, dp, ci, count in tier_list:
                att = attain_bw_gbps(m, dp, ci)
                total = att * count
                print(f"  [{tier_name}] {m}(dp={dp},ci={ci}) ×{count}: {att:.1f}G each = {total:.1f}G")
                if tier_name == "P":
                    sp += total
                else:
                    ss += total
                nj += count
        cs = sp + 0.5 * ss
        ratio = cs / 800
        print(f"  ΣP={sp:.1f}G, ΣS={ss:.1f}G, C*={cs:.1f}G ({ratio:.1%}×800G), jobs={nj}")
        print(f"  ΣP in [600,640]? {'✓' if 600 <= sp <= 640 else '✗'}")
        print(f"  ΣS in [240,280]? {'✓' if 240 <= ss <= 280 else '✗'}")
        print(f"  C* in [730,780]?  {'✓' if 730 <= cs <= 780 else '✗'}")


if __name__ == "__main__":
    main()
