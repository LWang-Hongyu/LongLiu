"""
128 节点扩展 workload：Table 4 的分层配置。

保持与 Table 3 相同的比例：
- 大模型（tight ci=1.5）：50% → 64/128
- 中模型（medium ci=2.0）：33% → 43/128
- 小模型（loose ci=3.0）：17% → 21/128
"""

from typing import List, Tuple

# 128 hosts 的分层 workload：128 个 job
TABLE4_TIERED_WORKLOAD_128: List[Tuple[str, int, float]] = [
    # 大模型：64 个，ci=1.5
    # LLaMA-2-13B (8 GPUs): 16 个
    ("LLaMA-2-13B", 8, 1.5) for _ in range(16)
] + [
    ("LLaMA-2-7B", 4, 1.5) for _ in range(24)
] + [
    ("T5-11B-fp16", 8, 1.5) for _ in range(8)
] + [
    ("LLaMA-2-7B", 8, 1.5) for _ in range(16)
] + [
    # 中模型：43 个，ci=2.0
    ("BERT-Large-fp16", 2, 2.0) for _ in range(10)
] + [
    ("BERT-Large-fp16", 4, 2.0) for _ in range(8)
] + [
    ("ViT-Large", 8, 2.0) for _ in range(10)
] + [
    ("ViT-Base", 2, 2.0) for _ in range(10)
] + [
    ("BERT-Base", 1, 2.0) for _ in range(5)
] + [
    # 小模型：21 个，ci=3.0
    ("ResNet-18", 1, 3.0) for _ in range(10)
] + [
    ("ResNet-50-fp16", 2, 3.0) for _ in range(8)
] + [
    ("MobileNet", 1, 3.0) for _ in range(3)
]

# 展开列表（上面是生成器表达式，需要展平）
TABLE4_TIERED_WORKLOAD_128 = [
    # 大模型：64 个，ci=1.5
    ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5),
    ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5),
    ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5),
    ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5), ("LLaMA-2-13B", 8, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5), ("LLaMA-2-7B", 4, 1.5),
    ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5),
    ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5), ("T5-11B-fp16", 8, 1.5),
    ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5),
    ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5),
    ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5),
    ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5), ("LLaMA-2-7B", 8, 1.5),
    
    # 中模型：43 个，ci=2.0
    ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0),
    ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0),
    ("BERT-Large-fp16", 2, 2.0), ("BERT-Large-fp16", 2, 2.0),
    ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0),
    ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0), ("BERT-Large-fp16", 4, 2.0),
    ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0),
    ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0),
    ("ViT-Large", 8, 2.0), ("ViT-Large", 8, 2.0),
    ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0),
    ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0),
    ("ViT-Base", 2, 2.0), ("ViT-Base", 2, 2.0),
    ("BERT-Base", 1, 2.0), ("BERT-Base", 1, 2.0), ("BERT-Base", 1, 2.0), ("BERT-Base", 1, 2.0), ("BERT-Base", 1, 2.0),
    
    # 小模型：21 个，ci=3.0
    ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0),
    ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0),
    ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0),
    ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0),
    ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0), ("ResNet-50-fp16", 2, 3.0),
    ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0), ("ResNet-18", 1, 3.0),
]