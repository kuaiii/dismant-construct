#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证脚本
用于快速验证整个 Dismant-Construct 统一框架的流程

流程:
1. 数据生成（小规模）
2. 模型训练（1 epoch）
3. 评估验证

用法:
    # 快速验证（约 5-10 分钟）
    python scripts/quick_validate.py

    # 跳过训练，仅验证数据和评估
    python scripts/quick_validate.py --skip_training

    # 使用现有检查点进行评估
    python scripts/quick_validate.py --checkpoint outputs/xxx/checkpoints/best
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def step1_generate_data(output_dir: str, num_graphs: int = 5, budget: int = 3):
    """步骤1: 生成训练数据"""
    print_section("步骤1: 生成训练数据")
    
    from scripts.generate_data import (
        generate_single_graph_data,
        load_graph_from_file,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # 生成小规模 BA 图数据
    for i in range(num_graphs):
        num_nodes = np.random.randint(30, 60)
        task_type = "dismantle" if i % 2 == 0 else "construct"
        
        print(f"  [{i+1}/{num_graphs}] 生成 BA 图 (n={num_nodes}), task={task_type}")
        
        samples = generate_single_graph_data(
            graph=None,
            graph_type="ba",
            num_nodes=num_nodes,
            task_type=task_type,
            budget=budget,
            graph_idx=i,
            data_source="generate",
            semantic_type="network"
        )
        all_samples.extend(samples)
        print(f"       生成 {len(samples)} 个样本")
    
    print(f"\n  总样本数: {len(all_samples)}")
    
    # 划分训练集和验证集
    np.random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    # 保存数据
    train_path = output_path / "train.json"
    eval_path = output_path / "eval.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
    
    print(f"  训练集: {len(train_samples)} 样本 -> {train_path}")
    print(f"  验证集: {len(eval_samples)} 样本 -> {eval_path}")
    
    # 显示样本示例
    print("\n  样本示例:")
    sample = all_samples[0]
    print(f"    ID: {sample['id']}")
    print(f"    Task: {sample['meta']['task']}")
    print(f"    Sign: {sample['meta'].get('sign', 'N/A')}")
    print(f"    Labels: {list(sample['auxiliary_labels'].keys())[:3]}...")
    
    return str(train_path), str(eval_path)


def step2_quick_train(
    train_data: str,
    eval_data: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 1,
):
    """步骤2: 快速训练"""
    print_section("步骤2: 快速训练")
    
    import subprocess
    
    cmd = [
        "python", "scripts/train.py",
        "--train_data", train_data,
        "--eval_data", eval_data,
        "--output_dir", output_dir,
        "--phase", "1",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    print(f"  命令: {' '.join(cmd)}")
    print(f"\n  开始训练...\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            capture_output=False,
            timeout=600,  # 10 分钟超时
        )
        
        if result.returncode != 0:
            print(f"\n  训练过程返回非零状态码: {result.returncode}")
            return None
        
        # 查找检查点
        checkpoint_dir = Path(output_dir) / "resilience_llm" / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.iterdir())
            if checkpoints:
                best_checkpoint = checkpoint_dir / "best"
                if best_checkpoint.exists():
                    print(f"\n  检查点已保存: {best_checkpoint}")
                    return str(best_checkpoint)
                return str(checkpoints[0])
        
        print(f"\n  未找到检查点")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"\n  训练超时")
        return None
    except Exception as e:
        print(f"\n  训练失败: {e}")
        return None


def step3_evaluate(graph_path: str = None, checkpoint: str = None):
    """步骤3: 评估验证"""
    print_section("步骤3: 评估验证")
    
    from src.evaluation import UnifiedEvaluator
    from src.attack import HighestDegreeAttack, RandomAttack
    
    # 如果没有提供图，生成一个测试图
    if graph_path is None:
        print("  生成测试图...")
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        graph_name = "test_ba_50"
    else:
        print(f"  加载图: {graph_path}")
        from scripts.unified_evaluate import load_graph
        G = load_graph(graph_path)
        graph_name = Path(graph_path).stem
    
    print(f"  图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    evaluator = UnifiedEvaluator()
    budget = int(G.number_of_nodes() * 0.3)
    edge_budget = int(G.number_of_edges() * 0.1)
    
    # ==================== Dismant 评估 ====================
    print("\n  [Dismant 评估]")
    
    # HDA
    hda = HighestDegreeAttack(recalculate=True)
    hda_result = hda.attack(G, budget, graph_name=graph_name)
    r_res_hda = float(np.trapezoid(hda_result.lcc_values, hda_result.removal_fractions))
    print(f"    HDA R_res: {r_res_hda:.4f}")
    
    # Random
    random_attack = RandomAttack(seed=42)
    random_results = random_attack.attack_multiple_runs(G, budget, num_runs=5, graph_name=graph_name)
    avg_random = random_results["average_result"]
    r_res_random = float(np.trapezoid(avg_random.lcc_values, avg_random.removal_fractions))
    print(f"    Random R_res: {r_res_random:.4f}")
    
    # ==================== Construct 评估 ====================
    print("\n  [Construct 评估]")
    
    # 原始图韧性
    original_eval = evaluator.evaluate_construct(
        original_graph=G,
        reconstructed_graph=G,
        added_edges=[],
        method_name="Original",
        graph_name=graph_name,
    )
    print(f"    Original R_tar: {original_eval.r_original_tar:.4f}")
    print(f"    Original R_ran: {original_eval.r_original_ran:.4f}")
    
    # 随机构造
    random_g, random_edges = evaluator._random_construct(G, edge_budget)
    random_construct = evaluator.evaluate_construct(
        original_graph=G,
        reconstructed_graph=random_g,
        added_edges=random_edges,
        method_name="RandomConstruct",
        graph_name=graph_name,
    )
    print(f"    RandomConstruct R_tar: {random_construct.r_tar:.4f} (Δ={random_construct.r_improvement:+.2%})")
    
    # 度数构造
    degree_g, degree_edges = evaluator._degree_based_construct(G, edge_budget)
    degree_construct = evaluator.evaluate_construct(
        original_graph=G,
        reconstructed_graph=degree_g,
        added_edges=degree_edges,
        method_name="DegreeConstruct",
        graph_name=graph_name,
    )
    print(f"    DegreeConstruct R_tar: {degree_construct.r_tar:.4f} (Δ={degree_construct.r_improvement:+.2%})")
    
    # ==================== 汇总 ====================
    print("\n" + "="*60)
    print(" 评估结果汇总")
    print("="*60)
    print("\n  [Dismant] R_res (越小越好):")
    print(f"    {'HDA':<20} {r_res_hda:.4f}")
    print(f"    {'Random':<20} {r_res_random:.4f}")
    
    print("\n  [Construct] R_tar (越大越好):")
    print(f"    {'Original':<20} {original_eval.r_original_tar:.4f}")
    print(f"    {'RandomConstruct':<20} {random_construct.r_tar:.4f}")
    print(f"    {'DegreeConstruct':<20} {degree_construct.r_tar:.4f}")
    
    return {
        "dismant": {
            "HDA": r_res_hda,
            "Random": r_res_random,
        },
        "construct": {
            "Original": original_eval.r_original_tar,
            "RandomConstruct": random_construct.r_tar,
            "DegreeConstruct": degree_construct.r_tar,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="快速验证脚本")
    
    parser.add_argument("--skip_training", action="store_true",
                        help="跳过训练步骤")
    parser.add_argument("--skip_data", action="store_true",
                        help="跳过数据生成步骤")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="使用现有检查点")
    parser.add_argument("--graph", type=str, default=None,
                        help="用于评估的图文件")
    parser.add_argument("--output_dir", type=str, default="outputs/quick_validate",
                        help="输出目录")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" 快速验证脚本 - Dismant & Construct 统一框架")
    print("="*60)
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 输出目录: {args.output_dir}")
    print("="*60)
    
    start_time = time.time()
    
    # 步骤1: 数据生成
    train_data, eval_data = None, None
    if not args.skip_data:
        data_dir = Path(args.output_dir) / "data"
        train_data, eval_data = step1_generate_data(
            output_dir=str(data_dir),
            num_graphs=5,
            budget=3,
        )
    
    # 步骤2: 训练
    checkpoint = args.checkpoint
    if not args.skip_training and not checkpoint and train_data:
        checkpoint = step2_quick_train(
            train_data=train_data,
            eval_data=eval_data,
            output_dir=args.output_dir,
            epochs=1,
            batch_size=1,
        )
    
    # 步骤3: 评估
    results = step3_evaluate(
        graph_path=args.graph,
        checkpoint=checkpoint,
    )
    
    # 完成
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(" 验证完成!")
    print("="*60)
    print(f" 总耗时: {elapsed:.1f} 秒")
    print(f" 检查点: {checkpoint or 'N/A'}")
    print("="*60)
    
    # 保存结果
    results_file = Path(args.output_dir) / "validation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "checkpoint": checkpoint,
            "results": results,
        }, f, indent=2)
    print(f"\n结果已保存: {results_file}")


if __name__ == "__main__":
    main()
