#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评估脚本
支持 Dismant 和 Construct 两种任务的端到端评估

用法:
    # 1. 仅评估 Dismant 基线（HDA vs Random）
    python scripts/unified_evaluate.py \
        --task dismant \
        --graph data/raw_graphs/true/Colt.gml \
        --output_dir results/dismant

    # 2. 使用 LLM 进行 Dismant 评估（需要训练好的模型）
    python scripts/unified_evaluate.py \
        --task dismant \
        --graph data/raw_graphs/true/Colt.gml \
        --checkpoint outputs/model/resilience_llm/checkpoints/best \
        --output_dir results/dismant

    # 3. 评估 Construct 基线
    python scripts/unified_evaluate.py \
        --task construct \
        --graph data/raw_graphs/true/Colt.gml \
        --edge_budget 10 \
        --output_dir results/construct

    # 4. 完整评估（Dismant + Construct）
    python scripts/unified_evaluate.py \
        --task both \
        --graph data/raw_graphs/true/Colt.gml \
        --output_dir results/full

    # 5. 批量评估多个图
    python scripts/unified_evaluate.py \
        --task both \
        --graph_dir data/raw_graphs/true \
        --output_dir results/batch
"""

import argparse
import sys
from pathlib import Path
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.evaluation import UnifiedEvaluator, EvaluationResult
from src.attack import HighestDegreeAttack, RandomAttack


def load_graph(graph_path: str) -> nx.Graph:
    """加载网络图"""
    path = Path(graph_path)
    
    if path.suffix == '.gml':
        try:
            G = nx.read_gml(path, label='id')
        except Exception:
            G = _load_gml_robust(path)
    elif path.suffix == '.graphml':
        G = nx.read_graphml(path)
    elif path.suffix == '.edgelist':
        G = nx.read_edgelist(path)
    else:
        raise ValueError(f"不支持的图格式: {path.suffix}")
    
    if G.is_directed():
        G = G.to_undirected()
    
    # 保留最大连通分量
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G


def _load_gml_robust(filepath: Path) -> nx.Graph:
    """稳健地加载 GML 文件"""
    g = nx.Graph()
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_node = False
    in_edge = False
    node_id = None
    node_label = None
    edge_u = None
    edge_v = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("node"):
            in_node = True
            node_id = None
            node_label = None
            continue
        if line.startswith("edge"):
            in_edge = True
            edge_u = None
            edge_v = None
            continue

        if line == "]":
            if in_node and node_id is not None:
                g.add_node(node_id, label=node_label)
            if in_edge and edge_u is not None and edge_v is not None:
                g.add_edge(edge_u, edge_v)
            in_node = False
            in_edge = False
            continue

        if in_node:
            if line.startswith("id "):
                try:
                    node_id = int(line.split()[1])
                except Exception:
                    pass
            elif line.startswith('label "'):
                if line.endswith('"') and len(line) >= 8:
                    node_label = line[len('label "'): -1]
        elif in_edge:
            if line.startswith("source "):
                try:
                    edge_u = int(line.split()[1])
                except Exception:
                    pass
            elif line.startswith("target "):
                try:
                    edge_v = int(line.split()[1])
                except Exception:
                    pass

    return g


def plot_dismant_comparison(
    results: Dict[str, Dict],
    output_path: str,
    graph_name: str,
    graph_info: Dict,
):
    """绘制 Dismant 结果对比图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'HDA': '#e74c3c', 'Random': '#3498db', 'LLM': '#2ecc71'}
    markers = {'HDA': 'o', 'Random': 's', 'LLM': '^'}
    
    for name, data in results.items():
        color = colors.get(name, '#9b59b6')
        marker = markers.get(name, 'D')
        
        r_res = data.get('r_res', 0)
        lcc = data.get('lcc_curve', [])
        fracs = data.get('removal_fractions', [])
        
        if lcc and fracs:
            label = f"{name} (R_res={r_res:.4f})"
            ax.plot(
                fracs, lcc,
                color=color, marker=marker, markersize=4,
                markevery=max(1, len(fracs) // 20),
                linewidth=2, label=label, alpha=0.8
            )
    
    # 绘制崩溃阈值线
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, 
               label='Collapse Threshold (LCC=20%)', alpha=0.7)
    
    ax.set_xlabel('Fraction of Removed Nodes (q)', fontsize=12)
    ax.set_ylabel('Largest Connected Component Ratio (LCC)', fontsize=12)
    ax.set_title(
        f"Network Dismantling: {graph_name}\n"
        f"(N={graph_info['nodes']}, E={graph_info['edges']})",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图像已保存: {output_path}")


def plot_construct_comparison(
    results: Dict[str, Dict],
    output_path: str,
    graph_name: str,
    graph_info: Dict,
):
    """绘制 Construct 结果对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'Original': '#95a5a6',
        'LLM': '#2ecc71',
        'RandomConstruct': '#3498db',
        'DegreeConstruct': '#9b59b6',
    }
    
    # 左图：HDA 攻击对比
    ax1 = axes[0]
    for name, data in results.items():
        color = colors.get(name, '#e74c3c')
        r_tar = data.get('r_tar', data.get('r_original_tar', 0))
        lcc = data.get('hda_lcc_curve', [])
        fracs = data.get('hda_removal_fractions', [])
        
        if lcc and fracs:
            label = f"{name} (R_tar={r_tar:.4f})"
            ax1.plot(fracs, lcc, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax1.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Fraction of Removed Nodes', fontsize=12)
    ax1.set_ylabel('LCC Ratio', fontsize=12)
    ax1.set_title('HDA Attack (R_tar)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 右图：Random 攻击对比
    ax2 = axes[1]
    for name, data in results.items():
        color = colors.get(name, '#e74c3c')
        r_ran = data.get('r_ran', data.get('r_original_ran', 0))
        lcc = data.get('random_lcc_curve', [])
        fracs = data.get('random_removal_fractions', [])
        
        if lcc and fracs:
            label = f"{name} (R_ran={r_ran:.4f})"
            ax2.plot(fracs, lcc, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Fraction of Removed Nodes', fontsize=12)
    ax2.set_ylabel('LCC Ratio', fontsize=12)
    ax2.set_title('Random Attack (R_ran)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle(
        f"Network Reconstruction Evaluation: {graph_name}\n"
        f"(N={graph_info['nodes']}, E={graph_info['edges']})",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图像已保存: {output_path}")


def evaluate_single_graph(
    graph_path: str,
    task: str,
    output_dir: str,
    budget: Optional[int] = None,
    edge_budget: Optional[int] = None,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """评估单个图"""
    
    graph_name = Path(graph_path).stem
    print(f"\n{'='*60}")
    print(f"评估图: {graph_name}")
    print(f"{'='*60}")
    
    # 加载图
    G = load_graph(graph_path)
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    
    graph_info = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
    }
    
    # 设置预算
    budget = budget or int(G.number_of_nodes() * 0.3)
    edge_budget = edge_budget or int(G.number_of_edges() * 0.1)
    
    # 创建输出目录
    exp_dir = Path(output_dir) / graph_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "graph_name": graph_name,
        "graph_path": str(graph_path),
        "graph_info": graph_info,
        "timestamp": datetime.now().isoformat(),
    }
    
    evaluator = UnifiedEvaluator()
    
    # ==================== Dismant 评估 ====================
    if task in ["dismant", "both"]:
        print(f"\n  [Dismant] 攻击预算: {budget}")
        
        dismant_results = {}
        
        # HDA 基线
        print("  运行 HDA 攻击...")
        hda = HighestDegreeAttack(recalculate=True)
        hda_result = hda.attack(G, budget, graph_name=graph_name)
        dismant_results["HDA"] = {
            "r_res": float(np.trapezoid(hda_result.lcc_values, hda_result.removal_fractions)),
            "lcc_curve": hda_result.lcc_values,
            "removal_fractions": hda_result.removal_fractions,
            "collapse_fraction": hda_result.find_collapse_point(0.2),
        }
        print(f"    HDA R_res: {dismant_results['HDA']['r_res']:.4f}")
        
        # Random 基线
        print("  运行 Random 攻击...")
        random_attack = RandomAttack(seed=42)
        random_results = random_attack.attack_multiple_runs(G, budget, num_runs=10, graph_name=graph_name)
        avg_random = random_results["average_result"]
        dismant_results["Random"] = {
            "r_res": float(np.trapezoid(avg_random.lcc_values, avg_random.removal_fractions)),
            "lcc_curve": avg_random.lcc_values,
            "removal_fractions": avg_random.removal_fractions,
            "collapse_fraction": avg_random.collapse_fraction,
            "std_r_res": random_results["std_r_res"],
        }
        print(f"    Random R_res: {dismant_results['Random']['r_res']:.4f} (±{random_results['std_r_res']:.4f})")
        
        # LLM 评估（如果提供了 checkpoint）
        if checkpoint:
            print("  运行 LLM 攻击...")
            try:
                from src.attack import LLMAttack
                llm_attack = LLMAttack(checkpoint_path=checkpoint, device=device)
                llm_result = llm_attack.attack(G, budget, graph_name=graph_name)
                dismant_results["LLM"] = {
                    "r_res": float(np.trapezoid(llm_result.lcc_values, llm_result.removal_fractions)),
                    "lcc_curve": llm_result.lcc_values,
                    "removal_fractions": llm_result.removal_fractions,
                    "collapse_fraction": llm_result.find_collapse_point(0.2),
                }
                print(f"    LLM R_res: {dismant_results['LLM']['r_res']:.4f}")
            except Exception as e:
                print(f"    LLM 攻击失败: {e}")
        
        results["dismant"] = dismant_results
        
        # 绘制 Dismant 对比图
        plot_dismant_comparison(
            dismant_results,
            str(exp_dir / "dismant_comparison.png"),
            graph_name,
            graph_info,
        )
    
    # ==================== Construct 评估 ====================
    if task in ["construct", "both"]:
        print(f"\n  [Construct] 边预算: {edge_budget}")
        
        construct_results = {}
        
        # 首先评估原始图
        print("  评估原始图韧性...")
        original_eval = evaluator.evaluate_construct(
            original_graph=G,
            reconstructed_graph=G,  # 原始图自己
            added_edges=[],
            method_name="Original",
            graph_name=graph_name,
        )
        construct_results["Original"] = {
            "r_tar": original_eval.r_original_tar,
            "r_ran": original_eval.r_original_ran,
            "hda_lcc_curve": original_eval.hda_lcc_curve,
            "random_lcc_curve": original_eval.random_lcc_curve,
            "hda_removal_fractions": original_eval.hda_removal_fractions,
            "random_removal_fractions": original_eval.random_removal_fractions,
        }
        print(f"    Original R_tar: {original_eval.r_original_tar:.4f}")
        print(f"    Original R_ran: {original_eval.r_original_ran:.4f}")
        
        # 随机构造基线
        print("  运行 RandomConstruct...")
        random_g, random_edges = evaluator._random_construct(G, edge_budget)
        random_construct = evaluator.evaluate_construct(
            original_graph=G,
            reconstructed_graph=random_g,
            added_edges=random_edges,
            method_name="RandomConstruct",
            graph_name=graph_name,
        )
        construct_results["RandomConstruct"] = {
            "r_tar": random_construct.r_tar,
            "r_ran": random_construct.r_ran,
            "r_improvement": random_construct.r_improvement,
            "hda_lcc_curve": random_construct.hda_lcc_curve,
            "random_lcc_curve": random_construct.random_lcc_curve,
            "hda_removal_fractions": random_construct.hda_removal_fractions,
            "random_removal_fractions": random_construct.random_removal_fractions,
        }
        print(f"    RandomConstruct R_tar: {random_construct.r_tar:.4f} (Δ={random_construct.r_improvement:+.2%})")
        
        # 度数构造基线
        print("  运行 DegreeConstruct...")
        degree_g, degree_edges = evaluator._degree_based_construct(G, edge_budget)
        degree_construct = evaluator.evaluate_construct(
            original_graph=G,
            reconstructed_graph=degree_g,
            added_edges=degree_edges,
            method_name="DegreeConstruct",
            graph_name=graph_name,
        )
        construct_results["DegreeConstruct"] = {
            "r_tar": degree_construct.r_tar,
            "r_ran": degree_construct.r_ran,
            "r_improvement": degree_construct.r_improvement,
            "hda_lcc_curve": degree_construct.hda_lcc_curve,
            "random_lcc_curve": degree_construct.random_lcc_curve,
            "hda_removal_fractions": degree_construct.hda_removal_fractions,
            "random_removal_fractions": degree_construct.random_removal_fractions,
        }
        print(f"    DegreeConstruct R_tar: {degree_construct.r_tar:.4f} (Δ={degree_construct.r_improvement:+.2%})")
        
        results["construct"] = construct_results
        
        # 绘制 Construct 对比图
        plot_construct_comparison(
            construct_results,
            str(exp_dir / "construct_comparison.png"),
            graph_name,
            graph_info,
        )
    
    # 保存结果
    results_file = exp_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {results_file}")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("评估结果汇总")
    print(f"{'='*60}")
    
    if "dismant" in results:
        print("\n[Dismant] R_res (越小表示拆解效果越好):")
        for name, data in results["dismant"].items():
            r_res = data["r_res"]
            collapse = data.get("collapse_fraction")
            collapse_str = f"{collapse:.2%}" if collapse else "N/A"
            print(f"  {name:<20} R_res={r_res:.4f}  崩溃点={collapse_str}")
    
    if "construct" in results:
        print("\n[Construct] R_tar / R_ran (越大表示韧性越好):")
        for name, data in results["construct"].items():
            r_tar = data.get("r_tar", data.get("r_original_tar", 0))
            r_ran = data.get("r_ran", data.get("r_original_ran", 0))
            improvement = data.get("r_improvement", 0)
            print(f"  {name:<20} R_tar={r_tar:.4f}  R_ran={r_ran:.4f}  Δ={improvement:+.2%}")
    
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="统一评估脚本 - Dismant & Construct")
    
    # 任务类型
    parser.add_argument("--task", type=str, default="both",
                        choices=["dismant", "construct", "both"],
                        help="任务类型: dismant, construct, both")
    
    # 输入
    parser.add_argument("--graph", type=str, default=None,
                        help="单个图文件路径")
    parser.add_argument("--graph_dir", type=str, default=None,
                        help="图文件目录（批量评估）")
    
    # 预算
    parser.add_argument("--budget", type=int, default=None,
                        help="Dismant 攻击预算（默认: 节点数的 30%%）")
    parser.add_argument("--edge_budget", type=int, default=None,
                        help="Construct 边预算（默认: 边数的 10%%）")
    
    # 模型
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LLM 模型检查点路径（可选）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备: cuda / cpu")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="results/unified",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.graph and not args.graph_dir:
        parser.error("必须指定 --graph 或 --graph_dir")
    
    print("=" * 60)
    print("统一评估框架 - Dismant & Construct")
    print("=" * 60)
    print(f"任务类型: {args.task}")
    print(f"输出目录: {args.output_dir}")
    
    all_results = []
    
    if args.graph:
        # 单图评估
        result = evaluate_single_graph(
            graph_path=args.graph,
            task=args.task,
            output_dir=args.output_dir,
            budget=args.budget,
            edge_budget=args.edge_budget,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        all_results.append(result)
    
    elif args.graph_dir:
        # 批量评估
        graph_dir = Path(args.graph_dir)
        graph_files = list(graph_dir.glob("*.gml")) + list(graph_dir.glob("*.graphml"))
        
        print(f"\n找到 {len(graph_files)} 个图文件")
        
        for gf in graph_files:
            try:
                result = evaluate_single_graph(
                    graph_path=str(gf),
                    task=args.task,
                    output_dir=args.output_dir,
                    budget=args.budget,
                    edge_budget=args.edge_budget,
                    checkpoint=args.checkpoint,
                    device=args.device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n评估 {gf.name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 保存汇总结果
    if len(all_results) > 1:
        summary_file = Path(args.output_dir) / "batch_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n批量汇总已保存: {summary_file}")
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
