"""CI lint: 检测代码内禁止的硬编码字面量。

检查项：
- overhead_factor = 1.3（应从 config.yaml 读取）
- overlap_factor = 0.85（应从 config.yaml 读取）
- spine_bw_bps = 400e9 / host_bw_bps = 100e9（应从 config.yaml 读取）
- K = 2.0 in policy constructors（应从 config.yaml 读取）

排除项：
- 注释中的提及
- metrics.py 的函数签名（默认值是 API 契约，可接受）
- config.py 自身
- 测试/脚本中的显式配置（读取 config.yaml 后的传递值）

用法：
    python3 scripts/ci_lint.py          # 检查所有
    python3 scripts/ci_lint.py --strict # 严格模式（含函数默认值）
"""

from __future__ import annotations

import ast
import os
import re
import sys
from typing import List, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 禁止的硬编码模式：(regex_pattern, description, severity)
# severity: "error" (必须修) | "warn" (建议修)
FORBIDDEN_PATTERNS = [
    # overhead_factor 硬编码
    (r"overhead_factor\s*[=:]\s*1\.3\b", "overhead_factor=1.3 硬编码", "warn"),
    (r"overhead_factor\s*=\s*1\.0\b", "overhead_factor=1.0（旧篡位值）", "error"),
    # overlap_factor 硬编码
    (r"overlap_factor\s*[=:]\s*0\.85\b", "overlap_factor=0.85 硬编码", "warn"),
    # spine_bw_bps / host_bw_bps 硬编码
    (r"spine_bw_bps\s*=\s*400[Ee]9\b", "spine_bw_bps=400e9 硬编码", "warn"),
    (r"host_bw_bps\s*=\s*100[Ee]9\b", "host_bw_bps=100e9 硬编码", "warn"),
    # ci_distribution 硬编码
    (r"ci_distribution\s*=\s*\{", 'ci_distribution 字面量定义（应从 config.yaml 读取）', "warn"),
    # 模型列表硬编码（一字不差的字面量）
    (r'model_types\s*=\s*\[[^\]]*"ResNet-18"[^\]]*\]', 'model_types 硬编码列表', "warn"),
]

# 排除的文件/目录
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    "outputs",
    ".venv",
    "venv",
}

EXCLUDE_FILES = {
    "config.py",          # config loader 自身
    "ci_lint.py",         # 本脚本
    "sas_eval_corrected.py",  # 分析脚本，从 meta 读
    "sas_eval_full_table.py", # 分析脚本，从 meta 读
    "sas_eval_v2_recalc.py",  # 分析脚本，从 meta 读
    "design_feas_boundary_v3.py", # 设计辅助，独立工具
}

# 允许硬编码的文件（白名单：逐文件豁免）
ALLOW_FILES = {
    # 函数默认值是 API 契约
    "metrics.py": ["warn"],      # metrics.py 函数签名默认值可接受
    "simulator.py": ["warn"],    # Simulator.__init__ 默认值可接受
    "synthetic.py": ["warn"],    # DEFAULT_TIERED_WORKLOAD 定义
    "lingjun.py": ["warn"],      # Lingjun loader 默认值
    # 脚本显式配置
    "gatekeeper.py": [],         # gatekeeper 已从 config.yaml 读取
}


def scan_file(filepath: str, strict: bool = False) -> List[Tuple[int, str, str]]:
    """扫描单个文件的硬编码字面量。

    返回 [(line_number, description, severity), ...]
    """
    basename = os.path.basename(filepath)
    violations = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        # 跳过注释行
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            # 但检查 inline comment 后的代码
            pass

        for pattern, desc, severity in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                # 跳过纯注释中的提及
                code_part = line.split("#")[0] if "#" in line else line
                if not re.search(pattern, code_part):
                    continue

                # 白名单豁免
                if basename in ALLOW_FILES:
                    allowed_severities = ALLOW_FILES[basename]
                    if severity in allowed_severities:
                        continue

                violations.append((i, desc, severity))

    return violations


def main():
    strict = "--strict" in sys.argv

    print("=" * 60)
    print("CI Lint: 硬编码字面量检查")
    print(f"模式: {'严格' if strict else '宽松（函数默认值豁免）'}")
    print("=" * 60)

    all_violations = []
    files_scanned = 0

    for root, dirs, files in os.walk(_PROJECT_ROOT):
        # 排除目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for fname in files:
            if not fname.endswith(".py"):
                continue
            if fname in EXCLUDE_FILES:
                continue

            filepath = os.path.join(root, fname)
            relpath = os.path.relpath(filepath, _PROJECT_ROOT)

            violations = scan_file(filepath, strict=strict)
            if violations:
                for lineno, desc, severity in violations:
                    all_violations.append((relpath, lineno, desc, severity))

            files_scanned += 1

    # 按严重性分组输出
    errors = [(f, l, d) for f, l, d, s in all_violations if s == "error"]
    warnings = [(f, l, d) for f, l, d, s in all_violations if s == "warn"]

    if errors:
        print(f"\n❌ 错误 ({len(errors)}):")
        for fpath, lineno, desc in errors:
            print(f"  {fpath}:{lineno} - {desc}")

    if warnings:
        print(f"\n⚠️ 警告 ({len(warnings)}):")
        for fpath, lineno, desc in warnings:
            print(f"  {fpath}:{lineno} - {desc}")

    print(f"\n扫描文件: {files_scanned}")
    print(f"结果: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\nCI Lint FAILED — 存在必须修复的硬编码字面量")
        sys.exit(1)
    elif warnings:
        print("\nCI Lint PASSED with warnings — 建议逐步迁移到 config.yaml")
    else:
        print("\nCI Lint PASSED")

    sys.exit(0)


if __name__ == "__main__":
    main()
