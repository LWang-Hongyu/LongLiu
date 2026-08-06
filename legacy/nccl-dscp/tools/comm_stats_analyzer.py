#!/usr/bin/env python3
"""
NCCL Communication Statistics Analyzer

This tool reads and analyzes NCCL communication statistics from JSON files
exported by the NCCL communication statistics module.

Usage:
    python comm_stats_analyzer.py <stats_file.json>
    python comm_stats_analyzer.py --dir <directory>
"""

# ----------longliu8 add----------

import json
import sys
import argparse
import os
import glob
from typing import Dict, List, Any, Optional

# 尝试导入dynamic_dscp_adapter以使用其DSCP映射方法
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../trainDistCode'))
    from dynamic_dscp_adapter import DynamicDSCPAdapter
    HAS_DSCP_ADAPTER = True
except ImportError:
    HAS_DSCP_ADAPTER = False
    DynamicDSCPAdapter = None

def priority_to_dscp(priority: float, priority_history: List[float] = None) -> int:
    """
    根据优先级值确定DSCP值（7级优先级，支持动态映射）
    直接使用DynamicDSCPAdapter的方法，避免重复实现
    
    Args:
        priority: 优先级值（ui），ui越大表示越紧急
        priority_history: 优先级历史记录（用于动态映射），如果为None则使用默认映射
        
    Returns:
        DSCP值，值越大表示优先级越高
    """
    if HAS_DSCP_ADAPTER:
        # 使用DynamicDSCPAdapter的方法
        adapter = DynamicDSCPAdapter()
        
        # 如果有历史数据，更新适配器的历史记录
        if priority_history and len(priority_history) >= 2:
            for p in priority_history:
                adapter.update_priority_history(p)
        
        return adapter.priority_to_dscp(priority)
    else:
        # 回退到简单的默认映射（如果无法导入DynamicDSCPAdapter）
        if priority >= 1.6:
            return 46  # EF (最高)
        elif priority >= 1.4:
            return 34  # AF41 (高)
        elif priority >= 1.2:
            return 36  # AF42 (中高)
        elif priority >= 1.0:
            return 26  # AF31 (中等)
        elif priority >= 0.8:
            return 28  # AF32 (中低)
        elif priority >= 0.6:
            return 18  # AF21 (低)
        else:
            return 0   # BE (最低)

def load_stats(filename: str) -> Dict[str, Any]:
    """Load statistics from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def find_stats_files(directory: str) -> List[str]:
    """Find all stats JSON files in directory (excluding per-epoch files)."""
    pattern = os.path.join(directory, "comm_stats_rank*.json")
    files = glob.glob(pattern)
    # Exclude per-epoch files (e.g., comm_stats_rank0_epoch0.json)
    files = [f for f in files if '_epoch' not in os.path.basename(f)]
    return sorted(files)

def get_rank_from_filename(filename: str) -> int:
    """Extract rank number from filename."""
    basename = os.path.basename(filename)
    # Extract rank number from comm_stats_rank0.json
    try:
        rank_str = basename.split('rank')[1].split('.')[0]
        return int(rank_str)
    except (IndexError, ValueError):
        return -1

def calculate_ideal_bandwidth(all_operations: List[Dict[str, Any]], percentile: float = 95.0) -> float:
    """
    Calculate ideal bandwidth from all operations.
    Uses high percentile to filter out outliers and get theoretical maximum.
    
    The ideal bandwidth represents the link's maximum capacity under ideal conditions,
    calculated as the bandwidth during pure communication time (excluding compute wait time).
    
    Args:
        all_operations: List of all operation dictionaries with 'bytes' and 'duration'
        percentile: Percentile to use for ideal bandwidth (default 95.0 for top 5%)
    
    Returns:
        Ideal bandwidth in Gbps
    """
    if not all_operations:
        return 0.0
    
    # Calculate total bytes and total communication time
    total_bytes = 0
    total_comm_time = 0
    bandwidths = []
    
    for op in all_operations:
        op_bytes = op.get('bytes', 0)
        op_duration = op.get('duration', 0)
        
        if op_bytes > 0:
            total_bytes += op_bytes
            if op_duration > 0:
                total_comm_time += op_duration
                # Only consider operations with reasonable size (>= 1KB) to avoid noise
                if op_bytes >= 1024:
                    op_bandwidth = (op_bytes * 8.0 / 1e9) / op_duration
                    bandwidths.append(op_bandwidth)
    
    if not bandwidths or total_comm_time == 0:
        return 0.0
    
    # Method 1: Calculate average bandwidth during communication time (more realistic)
    avg_bandwidth_comm_time = (total_bytes * 8.0 / 1e9) / total_comm_time
    
    # Method 2: Use high percentile of individual operation bandwidths (theoretical peak)
    bandwidths.sort()
    percentile_idx = int(len(bandwidths) * (percentile / 100.0))
    percentile_idx = min(percentile_idx, len(bandwidths) - 1)
    percentile_bandwidth = bandwidths[percentile_idx]
    
    # Use the higher of the two as ideal bandwidth
    # This represents the link's capacity when fully utilized for communication
    ideal_bandwidth = max(avg_bandwidth_comm_time, percentile_bandwidth)
    
    return ideal_bandwidth

def analyze_epoch(epoch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single epoch."""
    iterations = epoch_data.get('iterations', [])
    
    if not iterations:
        return None
    
    # Calculate total bytes for this epoch
    total_bytes = sum(iter['total_bytes'] for iter in iterations)
    
    # Get start and end timestamps
    start_time = min(iter['start_time'] for iter in iterations)
    end_time = max(iter['end_time'] for iter in iterations)
    
    # Calculate total duration (wall-clock time)
    total_duration = end_time - start_time
    
    # Calculate total communication time (sum of all operation durations)
    comm_duration = sum(iter['duration'] for iter in iterations)
    
    # Estimate compute time (total time - communication time)
    # Note: This is an approximation as there may be overlap or gaps
    compute_duration = max(0, total_duration - comm_duration)
    
    # Calculate average bandwidth (Gbps) - only consider communication time, not compute wait time
    # Bandwidth = (total_bytes * 8 bits/byte) / comm_duration (seconds) / 1e9 (Gbps)
    avg_bandwidth = (total_bytes * 8.0 / 1e9) / comm_duration if comm_duration > 0 else 0.0
    
    # Collect all operations for ideal bandwidth calculation
    all_ops = []
    for iter_data in iterations:
        for op in iter_data.get('operations', []):
            all_ops.append(op)
    
    # Calculate ideal bandwidth (95th percentile, representing link capacity during pure communication)
    ideal_bandwidth = calculate_ideal_bandwidth(all_ops, percentile=95.0)
    
    return {
        'epoch': epoch_data['epoch'],
        'total_bytes': total_bytes,
        'total_bytes_gb': total_bytes / (1024 ** 3),
        'start_time': start_time,
        'end_time': end_time,
        'total_duration': total_duration,  # Wall-clock time
        'comm_duration': comm_duration,    # Communication time
        'compute_duration': compute_duration,  # Estimated compute time
        'avg_bandwidth_gbps': avg_bandwidth,  # Average bandwidth in Gbps
        'ideal_bandwidth_gbps': ideal_bandwidth,  # Ideal bandwidth in Gbps (99th percentile)
        'num_iterations': len(iterations)
    }

def calculate_epoch_priority(epoch_analyses: List[Dict[str, Any]], slo_threshold: float = 1.2, ideal_bandwidth: float = 0.0) -> List[Dict[str, Any]]:
    """
    Calculate priority for each epoch.
    
    Priority formula: ui = ai / ei
    - ai: Actual time used for first i epochs = current_epoch_end_time - first_epoch_start_time
    - ei: Expected completion time for first i epochs = ci * ideal_time_for_i_epochs
    - ci: SLO threshold (default 1.2)
    - ideal_time_for_i_epochs = i * (epoch1_compute_time + epoch1_bytes / ideal_bandwidth)
    
    Note: Uses epoch index 1 (second epoch) compute time and ideal_bandwidth (calculated from first 2 epochs).
    """
    if len(epoch_analyses) < 2:
        # Need at least 2 epochs to calculate priority
        return epoch_analyses
    
    # Get first epoch start time (first iteration's start time)
    first_epoch_start_time = epoch_analyses[0]['start_time']
    
    # Get second epoch (index 1) as reference for ideal time calculation
    epoch1 = epoch_analyses[1]  # Second epoch (index 1)
    
    # Calculate ideal time for one epoch
    # ideal_time = compute_time + bytes / bandwidth
    # bandwidth = bytes * 8 / duration / 1e9 (Gbps)
    # So: bytes / bandwidth = bytes / (bytes * 8 / duration / 1e9) = duration * 1e9 / 8
    # But we can use: bytes / bandwidth = total_bytes * 8 / bandwidth_gbps / 1e9
    # Actually, bandwidth = (bytes * 8) / duration / 1e9, so bytes / bandwidth = duration * 1e9 / 8
    # Wait, let me recalculate: if bandwidth = (bytes * 8) / time / 1e9, then time = (bytes * 8) / bandwidth / 1e9
    # So: ideal_comm_time = bytes / bandwidth = (bytes * 8) / bandwidth_gbps / 1e9
    
    epoch1_compute_time = epoch1['compute_duration']
    epoch1_bytes = epoch1['total_bytes']
    
    # Calculate ideal communication time for one epoch using ideal_bandwidth
    # If ideal_bandwidth is provided, use it; otherwise fallback to epoch1's average bandwidth
    if ideal_bandwidth > 0:
        ideal_comm_time_per_epoch = (epoch1_bytes * 8.0) / (ideal_bandwidth * 1e9)
    else:
        # Fallback to epoch1's average bandwidth if ideal_bandwidth not provided
        epoch1_bandwidth_gbps = epoch1['avg_bandwidth_gbps']
        if epoch1_bandwidth_gbps > 0:
            ideal_comm_time_per_epoch = (epoch1_bytes * 8.0) / (epoch1_bandwidth_gbps * 1e9)
        else:
            ideal_comm_time_per_epoch = epoch1['comm_duration']
    
    # Ideal time for one epoch = compute time + communication time
    ideal_time_per_epoch = epoch1_compute_time + ideal_comm_time_per_epoch
    
    # Calculate priority for each epoch
    for i, analysis in enumerate(epoch_analyses):
        epoch_index = i + 1  # i-th epoch (1-indexed)
        
        # ai: Actual time used for first i epochs
        # = current epoch end time - first epoch start time
        ai = analysis['end_time'] - first_epoch_start_time
        
        # ei: Expected completion time for first i epochs
        # = ci * i * ideal_time_per_epoch
        ei = slo_threshold * epoch_index * ideal_time_per_epoch
        
        # Calculate priority
        if ei > 0:
            priority = ai / ei
        else:
            priority = 0.0
        
        # Add priority to analysis
        analysis['priority'] = priority
        analysis['actual_time_ai'] = ai
        analysis['expected_time_ei'] = ei
        analysis['ideal_time_per_epoch'] = ideal_time_per_epoch
    
    return epoch_analyses

def print_summary(stats: Dict[str, Any], filename: Optional[str] = None):
    """Print epoch-level summary statistics."""
    if 'epochs' not in stats:
        print("Error: This file does not contain epoch data.")
        print("Please use a file with epoch structure (comm_stats_rank*.json)")
        return
    
    total_epochs = stats.get('total_epochs', 0)
    
    print("=" * 100)
    print("NCCL Communication Statistics - Epoch Summary")
    print("=" * 100)
    if filename:
        rank = get_rank_from_filename(filename)
        print(f"Rank: {rank}")
    print(f"Total Epochs: {total_epochs}")
    print()
    
    if total_epochs == 0:
        print("No epochs recorded.")
        return
    
    # Analyze all epochs
    epoch_analyses = []
    for epoch_data in stats['epochs']:
        analysis = analyze_epoch(epoch_data)
        if analysis:
            epoch_analyses.append(analysis)
    
    if not epoch_analyses:
        print("No valid epoch data found.")
        return
    
    # Calculate ideal bandwidth using only first two epochs
    ideal_bandwidth = 0.0
    if len(stats['epochs']) >= 2:
        # Collect operations only from first two epochs
        all_operations_first_two_epochs = []
        for epoch_idx in range(2):  # Only first two epochs (index 0 and 1)
            epoch_data = stats['epochs'][epoch_idx]
            for iter_data in epoch_data.get('iterations', []):
                for op in iter_data.get('operations', []):
                    all_operations_first_two_epochs.append(op)
        
        # Calculate ideal bandwidth from first two epochs only
        ideal_bandwidth = calculate_ideal_bandwidth(all_operations_first_two_epochs, percentile=95.0)
    
    # Calculate priority for each epoch (requires at least 2 epochs)
    # Pass ideal_bandwidth to priority calculation
    if len(epoch_analyses) >= 2:
        epoch_analyses = calculate_epoch_priority(epoch_analyses, slo_threshold=1.2, ideal_bandwidth=ideal_bandwidth)
        
        # Calculate DSCP for each epoch based on priority
        # Collect priority history for dynamic mapping
        all_priorities_for_dscp_mapping = [a.get('priority', 0.0) for a in epoch_analyses if 'priority' in a]
        
        for analysis in epoch_analyses:
            if 'priority' in analysis:
                priority = analysis['priority']
                # Calculate DSCP using priority_to_dscp function
                # Pass priority history for dynamic mapping if available
                dscp = priority_to_dscp(priority, priority_history=all_priorities_for_dscp_mapping if len(all_priorities_for_dscp_mapping) >= 2 else None)
                analysis['dscp'] = dscp
    
    # Print epoch statistics table
    if len(epoch_analyses) >= 2 and 'priority' in epoch_analyses[0]:
        print("Epoch Statistics (with Priority and DSCP):")
        print(f"{'Epoch':<8} {'Total Bytes (GB)':<18} {'Total Time (s)':<18} {'Comm Time (s)':<18} {'Compute Time (s)':<18} {'Avg Bandwidth (Gbps)':<20} {'Priority (ui)':<15} {'DSCP':<8} {'Comm %':<10}")
        print("-" * 155)
        
        # DSCP名称映射
        dscp_names = {
            46: "EF", 34: "AF41", 36: "AF42",
            26: "AF31", 28: "AF32", 18: "AF21", 0: "BE"
        }
        
        for analysis in epoch_analyses:
            comm_percent = (analysis['comm_duration'] / analysis['total_duration'] * 100) if analysis['total_duration'] > 0 else 0
            priority = analysis.get('priority', 0.0)
            dscp = analysis.get('dscp', 0)
            dscp_name = dscp_names.get(dscp, f"{dscp}")
            print(f"{analysis['epoch']:<8} "
                  f"{analysis['total_bytes_gb']:<18.6f} "
                  f"{analysis['total_duration']:<18.6f} "
                  f"{analysis['comm_duration']:<18.6f} "
                  f"{analysis['compute_duration']:<18.6f} "
                  f"{analysis['avg_bandwidth_gbps']:<20.2f} "
                  f"{priority:<15.4f} "
                  f"{dscp_name:<8} "
                  f"{comm_percent:<10.2f}%")
    else:
        print("Epoch Statistics:")
        print(f"{'Epoch':<8} {'Total Bytes (GB)':<18} {'Total Time (s)':<18} {'Comm Time (s)':<18} {'Compute Time (s)':<18} {'Avg Bandwidth (Gbps)':<20} {'Comm %':<10}")
        print("-" * 130)
        
        for analysis in epoch_analyses:
            comm_percent = (analysis['comm_duration'] / analysis['total_duration'] * 100) if analysis['total_duration'] > 0 else 0
            print(f"{analysis['epoch']:<8} "
                  f"{analysis['total_bytes_gb']:<18.6f} "
                  f"{analysis['total_duration']:<18.6f} "
                  f"{analysis['comm_duration']:<18.6f} "
                  f"{analysis['compute_duration']:<18.6f} "
                  f"{analysis['avg_bandwidth_gbps']:<20.2f} "
                  f"{comm_percent:<10.2f}%")
    
    print()
    
    # Print ideal bandwidth information
    if ideal_bandwidth > 0:
        print("Ideal Bandwidth Analysis:")
        print("=" * 100)
        print(f"Link Ideal Maximum Bandwidth: {ideal_bandwidth:.2f} Gbps")
        print(f"  (Calculated from first two epochs only)")
        print(f"  (Calculated as the maximum of:")
        print(f"   - Average bandwidth during pure communication time")
        print(f"   - 95th percentile of individual operation bandwidths)")
        print(f"  This represents the link's capacity when fully utilized for communication,")
        print(f"  excluding compute wait time and measurement outliers.")
        print()
    
    # Print priority analysis if available
    if len(epoch_analyses) >= 2 and 'priority' in epoch_analyses[0]:
        print("Priority Analysis:")
        print("=" * 100)
        epoch1 = epoch_analyses[1]  # Second epoch as reference
        print(f"SLO Threshold (ci): 1.2")
        print(f"Reference Epoch (Epoch 1):")
        print(f"  Compute Time: {epoch1['compute_duration']:.6f} seconds")
        print(f"  Total Bytes: {epoch1['total_bytes_gb']:.6f} GB")
        # Display ideal bandwidth if available, otherwise show epoch1's average bandwidth
        if ideal_bandwidth > 0:
            print(f"  Ideal Maximum Bandwidth: {ideal_bandwidth:.2f} Gbps")
        else:
            print(f"  Average Bandwidth (during communication): {epoch1['avg_bandwidth_gbps']:.2f} Gbps")
        ideal_time = epoch_analyses[0].get('ideal_time_per_epoch', 0)
        print(f"  Ideal Time per Epoch: {ideal_time:.6f} seconds")
        print()
        print("Priority Interpretation:")
        print("  ui < 1.0: Ahead of schedule (faster than expected)")
        print("  ui = 1.0: On schedule (meets SLO)")
        print("  ui > 1.0: Behind schedule (slower than expected)")
        print()
    
    # Overall statistics
    total_bytes_all = sum(a['total_bytes'] for a in epoch_analyses)
    total_duration_all = sum(a['total_duration'] for a in epoch_analyses)
    total_comm_duration_all = sum(a['comm_duration'] for a in epoch_analyses)
    total_compute_duration_all = sum(a['compute_duration'] for a in epoch_analyses)
    
    print("Overall Statistics:")
    print(f"  Total Communication: {total_bytes_all / (1024**3):.6f} GB")
    print(f"  Total Duration: {total_duration_all:.6f} seconds ({total_duration_all/60:.2f} minutes)")
    print(f"  Total Communication Time: {total_comm_duration_all:.6f} seconds ({total_comm_duration_all/60:.2f} minutes)")
    print(f"  Total Compute Time (estimated): {total_compute_duration_all:.6f} seconds ({total_compute_duration_all/60:.2f} minutes)")
    if total_comm_duration_all > 0:
        # Average bandwidth calculated only during communication time (excluding compute wait time)
        avg_bandwidth = (total_bytes_all * 8 / 1e9) / total_comm_duration_all
        print(f"  Average Bandwidth (during communication): {avg_bandwidth:.2f} Gbps")
    if total_duration_all > 0:
        comm_percent = (total_comm_duration_all / total_duration_all * 100)
        print(f"  Communication Time Ratio: {comm_percent:.2f}%")
    if ideal_bandwidth > 0:
        print(f"  Ideal Maximum Bandwidth (from first 2 epochs): {ideal_bandwidth:.2f} Gbps")
    print()
    
def print_multi_file_summary(files: List[str]):
    """Print epoch-level summary statistics for multiple files."""
    print("=" * 100)
    print("Multi-File Summary (All Ranks) - Epoch Level")
    print("=" * 100)
    print()
    
    all_stats = []
    for filename in files:
        try:
            stats = load_stats(filename)
            if 'epochs' not in stats:
                print(f"Warning: {filename} does not contain epoch data, skipping.", file=sys.stderr)
                continue
            rank = get_rank_from_filename(filename)
            stats['_filename'] = filename
            stats['_rank'] = rank
            all_stats.append(stats)
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}", file=sys.stderr)
            continue
    
    if not all_stats:
        print("No valid stats files found.")
        return
    
    # Print per-rank epoch statistics
    print("Per-Rank Epoch Statistics:")
    print(f"{'Rank':<8} {'Epoch':<8} {'Total Bytes (GB)':<18} {'Total Time (s)':<18} {'Comm Time (s)':<18} {'Compute Time (s)':<18} {'Avg Bandwidth (Gbps)':<20}")
    print("-" * 130)
    
    for stats in all_stats:
        rank = stats.get('_rank', -1)
        for epoch_data in stats.get('epochs', []):
            analysis = analyze_epoch(epoch_data)
            if analysis:
                print(f"{rank:<8} "
                      f"{analysis['epoch']:<8} "
                      f"{analysis['total_bytes_gb']:<18.6f} "
                      f"{analysis['total_duration']:<18.6f} "
                      f"{analysis['comm_duration']:<18.6f} "
                      f"{analysis['compute_duration']:<18.6f} "
                      f"{analysis['avg_bandwidth_gbps']:<20.2f}")
    
    print()

    # Overall statistics
    total_bytes_all = 0
    total_duration_all = 0
    total_comm_duration_all = 0
    total_compute_duration_all = 0
    
    for stats in all_stats:
        for epoch_data in stats.get('epochs', []):
            analysis = analyze_epoch(epoch_data)
            if analysis:
                total_bytes_all += analysis['total_bytes']
                total_duration_all += analysis['total_duration']
                total_comm_duration_all += analysis['comm_duration']
                total_compute_duration_all += analysis['compute_duration']
    
    print("Aggregate Statistics (All Ranks):")
    print(f"  Total Ranks: {len(all_stats)}")
    print(f"  Total Communication: {total_bytes_all / (1024**3):.6f} GB")
    print(f"  Total Duration: {total_duration_all:.6f} seconds ({total_duration_all/60:.2f} minutes)")
    print(f"  Total Communication Time: {total_comm_duration_all:.6f} seconds ({total_comm_duration_all/60:.2f} minutes)")
    print(f"  Total Compute Time (estimated): {total_compute_duration_all:.6f} seconds ({total_compute_duration_all/60:.2f} minutes)")
    if total_duration_all > 0:
        avg_bandwidth_all = (total_bytes_all * 8 / 1e9) / total_duration_all
        comm_percent = (total_comm_duration_all / total_duration_all * 100)
        print(f"  Average Bandwidth: {avg_bandwidth_all:.2f} Gbps")
        print(f"  Communication Time Ratio: {comm_percent:.2f}%")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze NCCL communication statistics (Epoch Level)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python comm_stats_analyzer.py comm_stats_rank0.json
  
  # Analyze all files in a directory
  python comm_stats_analyzer.py --dir /path/to/staticsJson
        """
    )
    parser.add_argument(
        'stats_file',
        type=str,
        nargs='?',
        help='Path to the statistics JSON file'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing stats JSON files (will analyze all rank files)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dir:
            # Analyze directory
            files = find_stats_files(args.dir)
            if not files:
                print(f"Error: No stats files found in '{args.dir}'", file=sys.stderr)
                sys.exit(1)
            
            # Print multi-file summary
            print_multi_file_summary(files)
        elif args.stats_file:
            # Analyze single file
            stats = load_stats(args.stats_file)
            print_summary(stats, args.stats_file)
        else:
            parser.print_help()
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

# ----------longliu8 add----------