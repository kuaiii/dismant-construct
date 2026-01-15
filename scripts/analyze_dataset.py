#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集分析脚本

用法:
    python scripts/analyze_dataset.py data/fine_tuning/dismantle/train.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_dataset(data_file: str):
    """分析数据集统计信息"""
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {data_file}")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    
    if len(samples) == 0:
        print("Warning: Empty dataset!")
        return
    
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
    
    # 操作预算分布
    budgets = []
    for s in samples:
        budget_step = s['meta'].get('budget_step', '1/10')
        if '/' in budget_step:
            total_steps = int(budget_step.split('/')[1])
            budgets.append(total_steps)
    
    if budgets:
        print(f"\nBudget statistics:")
        print(f"  Min: {min(budgets)}")
        print(f"  Max: {max(budgets)}")
        print(f"  Mean: {sum(budgets) / len(budgets):.2f}")
    
    # auxiliary_labels 统计
    label_values = []
    label_counts = []
    for s in samples:
        if 'auxiliary_labels' in s:
            labels = s['auxiliary_labels']
            label_values.extend(labels.values())
            label_counts.append(len(labels))
    
    if label_values:
        print(f"\nLabel value statistics:")
        print(f"  Min: {min(label_values):.4f}")
        print(f"  Max: {max(label_values):.4f}")
        print(f"  Mean: {sum(label_values) / len(label_values):.4f}")
        print(f"  Std: {(sum((x - sum(label_values)/len(label_values))**2 for x in label_values) / len(label_values))**0.5:.4f}")
    
    if label_counts:
        print(f"\nCandidate count per sample:")
        print(f"  Min: {min(label_counts)}")
        print(f"  Max: {max(label_counts)}")
        print(f"  Mean: {sum(label_counts) / len(label_counts):.2f}")
    
    # 检查数据完整性
    print(f"\nData quality check:")
    missing_labels = sum(1 for s in samples if 'auxiliary_labels' not in s or not s['auxiliary_labels'])
    missing_conversations = sum(1 for s in samples if 'conversations' not in s or len(s['conversations']) < 3)
    
    print(f"  Samples with missing labels: {missing_labels}")
    print(f"  Samples with incomplete conversations: {missing_conversations}")
    
    if missing_labels == 0 and missing_conversations == 0:
        print("  ✓ All samples are complete")


def main():
    parser = argparse.ArgumentParser(description="分析数据集")
    
    parser.add_argument("data_file", type=str, help="数据文件路径（train.json 或 eval.json）")
    
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        return
    
    analyze_dataset(str(data_file))


if __name__ == "__main__":
    main()
