#!/usr/bin/env python3
"""
分析第一个epoch字节数大的原因
"""

import json
import sys

def analyze_epoch_bytes(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if 'epochs' not in data or len(data['epochs']) < 2:
        print("Error: Need at least 2 epochs to compare")
        return
    
    print("=" * 80)
    print("第一个Epoch字节数分析")
    print("=" * 80)
    print()
    
    # 分析前几个epoch
    for epoch_idx in range(min(5, len(data['epochs']))):
        epoch_data = data['epochs'][epoch_idx]
        iterations = epoch_data.get('iterations', [])
        
        if not iterations:
            continue
        
        total_bytes = sum(iter['total_bytes'] for iter in iterations)
        avg_bytes_per_iter = total_bytes / len(iterations)
        
        print(f"Epoch {epoch_idx}:")
        print(f"  迭代数: {len(iterations)}")
        print(f"  总字节数: {total_bytes/(1024**3):.6f} GB")
        print(f"  平均每迭代: {avg_bytes_per_iter/(1024*1024):.6f} MB")
        
        # 分析第一个迭代
        if iterations:
            first_iter = iterations[0]
            print(f"  第一个迭代字节数: {first_iter['total_bytes']/(1024*1024):.6f} MB")
            print(f"  第一个迭代操作数: {first_iter['num_ops']}")
            
            # 统计操作类型
            ops_by_type = {}
            for op in first_iter.get('operations', []):
                op_type = op['func']
                if op_type not in ops_by_type:
                    ops_by_type[op_type] = {'count': 0, 'bytes': 0}
                ops_by_type[op_type]['count'] += 1
                ops_by_type[op_type]['bytes'] += op['bytes']
            
            print("  第一个迭代的操作类型:")
            for op_type, stats in sorted(ops_by_type.items()):
                print(f"    {op_type}: {stats['count']}次, {stats['bytes']/(1024*1024):.6f} MB")
        
        print()
    
    # 详细分析第一个epoch的第一个迭代
    print("=" * 80)
    print("第一个Epoch的第一个迭代详细分析")
    print("=" * 80)
    print()
    
    first_epoch = data['epochs'][0]
    first_iterations = first_epoch.get('iterations', [])
    
    if first_iterations:
        first_iter = first_iterations[0]
        print(f"Iteration {first_iter['iteration']}:")
        print(f"  总字节数: {first_iter['total_bytes']/(1024*1024):.6f} MB")
        print(f"  开始时间: {first_iter['start_time']:.9f}")
        print(f"  结束时间: {first_iter['end_time']:.9f}")
        print(f"  持续时间: {first_iter['duration']:.6f} 秒")
        print()
        print("  所有操作详情:")
        for i, op in enumerate(first_iter.get('operations', []), 1):
            print(f"    操作 {i}:")
            print(f"      类型: {op['func']}")
            print(f"      字节数: {op['bytes']} bytes ({op['bytes']/(1024*1024):.6f} MB)")
            print(f"      开始时间: {op['start_time']:.9f}")
            print(f"      结束时间: {op['end_time']:.9f}")
            print(f"      持续时间: {op['duration']:.6f} 秒")
    
    # 对比第二个epoch的第一个迭代
    print()
    print("=" * 80)
    print("第二个Epoch的第一个迭代对比")
    print("=" * 80)
    print()
    
    if len(data['epochs']) >= 2:
        second_epoch = data['epochs'][1]
        second_iterations = second_epoch.get('iterations', [])
        
        if second_iterations:
            second_iter = second_iterations[0]
            print(f"Iteration {second_iter['iteration']}:")
            print(f"  总字节数: {second_iter['total_bytes']/(1024*1024):.6f} MB")
            print(f"  操作数: {second_iter['num_ops']}")
            print()
            print("  所有操作详情:")
            for i, op in enumerate(second_iter.get('operations', []), 1):
                print(f"    操作 {i}:")
                print(f"      类型: {op['func']}")
                print(f"      字节数: {op['bytes']} bytes ({op['bytes']/(1024*1024):.6f} MB)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_first_epoch.py <stats_file.json>")
        sys.exit(1)
    
    analyze_epoch_bytes(sys.argv[1])
