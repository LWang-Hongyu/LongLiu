#!/usr/bin/env python3
"""
fig6_data.csv 提取脚本（参考版）

从 round{1,2}_console.log 的 Results Summary 段提取 epoch 级 slowdown 数据。
用法：python3 _extract_fig6.py [--dir .]

输出：fig6_data.csv（覆盖已有文件）
依赖：无外部依赖，仅 Python 3 stdlib

注：此脚本为**参考版本**，完整数据已在 fig6_data.csv 中。
绘图由仿真侧统一执行。
"""
import csv, os, re, sys

def parse_console(path):
    """Parse console.log → list of epoch dicts."""
    with open(path) as f:
        text = f.read()
    
    records = []
    # Split at === <scheduler> ===
    parts = re.split(r'^=== (\w+) ===', text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        sched = parts[i]
        body = parts[i+1]
        jparts = re.split(r'^--- Job (\w) ---', body, flags=re.MULTILINE)
        for j in range(1, len(jparts), 2):
            job = jparts[j]
            csv_text = jparts[j+1].strip()
            lines = csv_text.split('\n')
            header = lines[0].split(',')
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith('='):
                    continue
                vals = line.split(',')
                if len(vals) == len(header):
                    row = dict(zip(header, vals))
                    row['scheduler'] = sched
                    row['job'] = job
                    records.append(row)
    return records

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='.', help='Directory with console.log files')
    args = parser.parse_args()
    
    rounds = [
        ('orig_r1', 'LL→CX', os.path.join(args.dir, 'round1_console.log')),
        ('orig_r2', 'CX→LL', os.path.join(args.dir, 'round2_console.log')),
    ]
    
    all_rows = []
    for rid, order, path in rounds:
        if not os.path.exists(path):
            print(f"Skip {path}: not found", file=sys.stderr)
            continue
        for ep in parse_console(path):
            ep['round_id'] = rid
            ep['order'] = order
            all_rows.append(ep)
    
    out = os.path.join(args.dir, 'fig6_data_raw.csv')
    fieldnames = ['round_id','order','scheduler','job','epoch','phase',
                  'payload_mb','c_i','sleep_us','avg_comm_s','avg_bw_gbps',
                  'pi','priority','dscp','slowdown','t_target_ms']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f"Written {len(all_rows)} rows → {out}")

if __name__ == '__main__':
    main()
