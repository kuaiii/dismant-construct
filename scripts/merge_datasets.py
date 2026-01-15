#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集合并脚本

用法:
    python scripts/merge_datasets.py \
        --input_dirs data/fine_tuning/dismantle data/fine_tuning/construct \
        --output_file data/fine_tuning/combined/train.json
"""

import argparse
import json
from pathlib import Path
from typing import List
from collections import Counter


def merge_datasets(input_dirs: List[str], output_file: str, split_ratio: float = 0.9):
    """
    合并多个数据集
    
    Args:
        input_dirs: 输入目录列表（每个目录应包含 train.json 和 eval.json）
        output_file: 输出文件路径
        split_ratio: 训练集比例（用于重新划分）
    """
    all_train_samples = []
    all_eval_samples = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        train_file = input_path / "train.json"
        eval_file = input_path / "eval.json"
        
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                all_train_samples.extend(samples)
            print(f"Loaded {len(samples)} training samples from {input_dir}")
        
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                all_eval_samples.extend(samples)
            print(f"Loaded {len(samples)} eval samples from {input_dir}")
    
    # 合并并重新划分
    all_samples = all_train_samples + all_eval_samples
    
    # 打乱
    import random
    random.shuffle(all_samples)
    
    # 重新划分
    split_idx = int(len(all_samples) * split_ratio)
    merged_train = all_samples[:split_idx]
    merged_eval = all_samples[split_idx:]
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_output = output_path.parent / "train.json"
    eval_output = output_path.parent / "eval.json"
    
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(merged_train, f, ensure_ascii=False, indent=2)
    
    with open(eval_output, 'w', encoding='utf-8') as f:
        json.dump(merged_eval, f, ensure_ascii=False, indent=2)
    
    print(f"\nMerged {len(merged_train)} training samples")
    print(f"Merged {len(merged_eval)} eval samples")
    print(f"Saved to {train_output} and {eval_output}")
    
    # 打印统计信息
    task_types = [s['meta'].get('task', 'unknown') for s in merged_train]
    print(f"\nTask distribution: {Counter(task_types)}")


def analyze_dataset(data_file: str):
    """分析数据集统计信息"""
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {data_file}")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    
    # 任务类型分布
    task_types = [s['meta'].get('task', 'unknown') for s in samples]
    print(f"\nTask type distribution:")
    for task, count in Counter(task_types).items():
        print(f"  {task}: {count} ({count/len(samples)*100:.1f}%)")
    
    # 数据源分布
    data_sources = [s['meta'].get('data_source', 'unknown') for s in samples]
    print(f"\nData source distribution:")
    for source, count in Counter(data_sources).items():
        print(f"  {source}: {count} ({count/len(samples)*100:.1f}%)")
    
    # 节点数分布
    node_counts = [s['meta'].get('num_nodes', 0) for s in samples]
    if node_counts:
        print(f"\nNode count statistics:")
        print(f"  Min: {min(node_counts)}")
        print(f"  Max: {max(node_counts)}")
        print(f"  Mean: {sum(node_counts) / len(node_counts):.2f}")
        print(f"  Median: {sorted(node_counts)[len(node_counts)//2]}")
    
    # auxiliary_labels 统计
    label_values = []
    for s in samples:
        if 'auxiliary_labels' in s:
            label_values.extend(s['auxiliary_labels'].values())
    
    if label_values:
        print(f"\nLabel value statistics:")
        print(f"  Min: {min(label_values):.4f}")
        print(f"  Max: {max(label_values):.4f}")
        print(f"  Mean: {sum(label_values) / len(label_values):.4f}")


def main():
    parser = argparse.ArgumentParser(description="合并数据集")
    
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                        help="输入目录列表（每个目录应包含 train.json 和 eval.json）")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出文件路径（实际会生成 train.json 和 eval.json）")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="训练集比例（默认：0.9）")
    parser.add_argument("--analyze", action="store_true",
                        help="分析合并后的数据集")
    
    args = parser.parse_args()
    
    # 合并数据集
    merge_datasets(args.input_dirs, args.output_file, args.split_ratio)
    
    # 分析数据集
    if args.analyze:
        output_path = Path(args.output_file)
        train_file = output_path.parent / "train.json"
        if train_file.exists():
            analyze_dataset(str(train_file))


if __name__ == "__main__":
    main()
