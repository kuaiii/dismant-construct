#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
攻击算法评估与可视化脚本

功能：
1. 加载网络图数据
2. 运行多种攻击算法 (HDA, Random, LLM)
3. 计算评估指标 (R_res, 崩溃点)
4. 生成可视化对比图

用法:
    python scripts/evaluate_attacks.py \
        --graph data/raw_graphs/true/Colt.gml \
        --budget 50 \
        --output_dir results

    python scripts/evaluate_attacks.py \
        --graph data/raw_graphs/syn/BA_100.gml \
        --budget 30 \
        --algorithms hda random \
        --output_dir results

    python scripts/evaluate_attacks.py \
        --graph data/raw_graphs/true/Colt.gml \
        --budget 50 \
        --algorithms hda random llm \
        --checkpoint outputs/mixed_model/checkpoints/best \
        --output_dir results
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.attack import HighestDegreeAttack, RandomAttack, LLMAttack, AttackResult


def load_graph(graph_path: str) -> nx.Graph:
    """
    加载网络图
    
    对于 GML 文件，使用稳健的加载方式：
    - 使用 id 作为节点标识符（避免 label 为 None 导致的重复标签错误）
    - 处理缺失或重复的标签
    """
    path = Path(graph_path)
    
    if path.suffix == '.gml':
        # 尝试使用 id 作为标签（更稳健）
        try:
            G = nx.read_gml(path, label='id')
        except Exception:
            # 如果失败，使用自定义解析器（处理重复 None 标签的情况）
            G = _load_gml_robust(path)
    elif path.suffix == '.graphml':
        G = nx.read_graphml(path)
    elif path.suffix == '.edgelist':
        G = nx.read_edgelist(path)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
    else:
        raise ValueError(f"不支持的图格式: {path.suffix}")
    
    # 转换为无向图（如果需要）
    if G.is_directed():
        G = G.to_undirected()
    
    return G


def _load_gml_robust(filepath: Path) -> nx.Graph:
    """
    稳健地加载 GML 文件，处理重复标签问题
    
    使用 id 作为节点标识符，而不是 label
    """
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
                # 使用 id 作为节点标识符，label 作为属性
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
                # label "xxx"
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


def run_attack_algorithm(
    algorithm_name: str,
    graph: nx.Graph,
    budget: int,
    dataset_name: str,
    graph_name: str,
    collapse_threshold: float = 0.2,
    random_runs: int = 10,
    random_seed: int = 42,
    checkpoint_path: str = None,
    config_path: str = "configs/default.yaml",
    device: str = "cuda",
) -> AttackResult:
    """
    运行指定的攻击算法
    
    Args:
        algorithm_name: 算法名称 ('hda', 'random', 'llm')
        graph: 网络图
        budget: 攻击预算
        dataset_name: 数据集名称
        graph_name: 图名称
        collapse_threshold: 崩溃阈值
        random_runs: 随机攻击运行次数
        random_seed: 随机种子
        checkpoint_path: LLM 模型检查点路径（仅用于 llm 算法）
        config_path: 配置文件路径（仅用于 llm 算法）
        device: 设备 (cuda/cpu)（仅用于 llm 算法）
    
    Returns:
        AttackResult: 攻击结果
    """
    if algorithm_name.lower() == 'hda':
        attacker = HighestDegreeAttack(recalculate=True)
        result = attacker.attack(
            graph=graph,
            budget=budget,
            dataset_name=dataset_name,
            graph_name=graph_name,
            collapse_threshold=collapse_threshold,
        )
        return result
    
    elif algorithm_name.lower() == 'random':
        attacker = RandomAttack(seed=random_seed)
        # 多次运行取平均
        multi_result = attacker.attack_multiple_runs(
            graph=graph,
            budget=budget,
            num_runs=random_runs,
            dataset_name=dataset_name,
            graph_name=graph_name,
            collapse_threshold=collapse_threshold,
        )
        return multi_result["average_result"]
    
    elif algorithm_name.lower() == 'llm':
        if checkpoint_path is None:
            raise ValueError("LLM 算法需要提供 --checkpoint 参数")
        attacker = LLMAttack(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )
        result = attacker.attack(
            graph=graph,
            budget=budget,
            dataset_name=dataset_name,
            graph_name=graph_name,
            collapse_threshold=collapse_threshold,
        )
        return result
    
    else:
        raise ValueError(f"未知的算法: {algorithm_name}")


def compute_r_res_correct(removal_fractions: List[float], lcc_values: List[float]) -> float:
    """
    正确计算 R_res (LCC 曲线下面积)
    
    R_res = ∫₀^(q_max) LCC(q) dq
    
    使用梯形积分法计算
    """
    if len(removal_fractions) < 2 or len(lcc_values) < 2:
        return 0.0
    
    # 确保长度一致
    n = min(len(removal_fractions), len(lcc_values))
    x = np.array(removal_fractions[:n])
    y = np.array(lcc_values[:n])
    
    # 梯形积分
    r_res = np.trapezoid(y, x)
    return float(r_res)


def find_collapse_intersection(
    removal_fractions: List[float],
    lcc_values: List[float],
    threshold: float = 0.2
) -> Optional[tuple]:
    """
    找到 LCC 曲线与崩溃阈值线的交点
    
    Returns:
        (x, y) 交点坐标，或 None
    """
    for i in range(1, len(lcc_values)):
        if lcc_values[i] < threshold and lcc_values[i-1] >= threshold:
            # 线性插值找精确交点
            x1, y1 = removal_fractions[i-1], lcc_values[i-1]
            x2, y2 = removal_fractions[i], lcc_values[i]
            
            if y1 != y2:
                t = (y1 - threshold) / (y1 - y2)
                x_intersect = x1 + t * (x2 - x1)
                return (x_intersect, threshold)
            else:
                return (x1, threshold)
    
    return None


def plot_attack_comparison(
    results: List[AttackResult],
    output_path: str,
    collapse_threshold: float = 0.2,
    title: str = "Network Dismantling Comparison"
):
    """
    绘制攻击算法对比图
    
    Args:
        results: 攻击结果列表
        output_path: 输出路径
        collapse_threshold: 崩溃阈值
        title: 图表标题
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 颜色映射
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 存储交点信息
    intersections = []
    
    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 重新计算正确的 R_res
        r_res = compute_r_res_correct(result.removal_fractions, result.lcc_values)
        
        # 绘制 LCC 曲线
        label = f"{result.algorithm_name} (R_res={r_res:.4f})"
        ax.plot(
            result.removal_fractions,
            result.lcc_values,
            color=color,
            marker=marker,
            markersize=4,
            markevery=max(1, len(result.removal_fractions) // 20),
            linewidth=2,
            label=label,
            alpha=0.8,
        )
        
        # 找到与崩溃线的交点
        intersection = find_collapse_intersection(
            result.removal_fractions,
            result.lcc_values,
            collapse_threshold
        )
        
        if intersection:
            intersections.append({
                'name': result.algorithm_name,
                'point': intersection,
                'color': color,
            })
    
    # 绘制崩溃阈值线 (LCC = 20%)
    ax.axhline(
        y=collapse_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Collapse Threshold (LCC={collapse_threshold*100:.0f}%)',
        alpha=0.7,
    )
    
    # 标注交点
    for inter in intersections:
        x, y = inter['point']
        ax.scatter([x], [y], color=inter['color'], s=150, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(
            f"{inter['name']}\n({x:.2%})",
            xy=(x, y),
            xytext=(x + 0.03, y + 0.05),
            fontsize=9,
            ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
        )
    
    # 设置坐标轴
    ax.set_xlabel('Fraction of Removed Nodes (q)', fontsize=12)
    ax.set_ylabel('Largest Connected Component Ratio (LCC)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置范围
    ax.set_xlim(-0.02, max(r.removal_fractions[-1] for r in results) + 0.05)
    ax.set_ylim(-0.02, 1.05)
    
    # 添加网格
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图像已保存: {output_path}")
    
    return intersections


def save_experiment_data(
    results: List[AttackResult],
    output_dir: str,
    collapse_threshold: float = 0.2
):
    """
    保存实验数据到文件
    
    Args:
        results: 攻击结果列表
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存每个算法的详细结果
    for result in results:
        # 重新计算正确的 R_res
        result.r_res = compute_r_res_correct(result.removal_fractions, result.lcc_values)
        result.collapse_fraction = result.find_collapse_point(collapse_threshold)
        
        result_file = output_path / f"{result.algorithm_name}_result.json"
        result.save(result_file)
        print(f"✅ 结果已保存: {result_file}")
    
    # 保存汇总对比表
    summary = {
        "timestamp": datetime.now().isoformat(),
        "collapse_threshold": collapse_threshold,
        "algorithms": [],
    }
    
    for result in results:
        algo_summary = {
            "name": result.algorithm_name,
            "r_res": result.r_res,
            "collapse_fraction": result.collapse_fraction,
            "initial_nodes": result.initial_nodes,
            "initial_edges": result.initial_edges,
            "budget": result.budget,
            "nodes_removed": len(result.attack_sequence),
        }
        summary["algorithms"].append(algo_summary)
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 汇总已保存: {summary_file}")
    
    # 保存为 CSV 格式（便于分析）
    csv_file = output_path / "comparison.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Algorithm,R_res,Collapse_Fraction,Initial_Nodes,Initial_Edges,Budget\n")
        for result in results:
            collapse = result.collapse_fraction if result.collapse_fraction else "N/A"
            f.write(f"{result.algorithm_name},{result.r_res:.6f},{collapse},{result.initial_nodes},{result.initial_edges},{result.budget}\n")
    print(f"✅ CSV 已保存: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="攻击算法评估与可视化")
    
    parser.add_argument("--graph", type=str, required=True, help="图文件路径")
    parser.add_argument("--budget", type=int, default=None, help="攻击预算 (默认: 节点数的 30%%)")
    parser.add_argument("--algorithms", nargs='+', default=['hda', 'random'], 
                        help="要评估的算法列表 (hda, random, llm)")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--dataset_name", type=str, default=None, help="数据集名称")
    parser.add_argument("--experiment_id", type=str, default=None, help="实验 ID")
    parser.add_argument("--collapse_threshold", type=float, default=0.2, help="崩溃阈值 (默认 0.2)")
    parser.add_argument("--random_runs", type=int, default=10, help="随机攻击运行次数")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--checkpoint", type=str, default=None, help="LLM 模型检查点路径（用于 llm 算法）")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径（用于 llm 算法）")
    parser.add_argument("--device", type=str, default="cuda", help="设备 cuda/cpu（用于 llm 算法）")
    
    args = parser.parse_args()
    
    # 解析参数
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"❌ 图文件不存在: {graph_path}")
        return
    
    # 提取图名称和数据集名称
    graph_name = graph_path.stem
    dataset_name = args.dataset_name or graph_path.parent.name
    
    # 生成实验 ID
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建输出目录
    # experiment_base_dir = results/{dataset_name}/{attack_mode}/{experiment_id}
    attack_mode = "_".join(args.algorithms)
    experiment_dir = os.path.join(args.output_dir, dataset_name, attack_mode, experiment_id)
    
    print("=" * 60)
    print("攻击算法评估")
    print("=" * 60)
    print(f"图文件: {graph_path}")
    print(f"数据集: {dataset_name}")
    print(f"图名称: {graph_name}")
    print(f"算法: {args.algorithms}")
    print(f"输出目录: {experiment_dir}")
    print("=" * 60)
    
    # 加载图
    print("\n正在加载图...")
    G = load_graph(str(graph_path))
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    
    # 设置攻击预算
    budget = args.budget or int(G.number_of_nodes() * 0.3)
    print(f"攻击预算: {budget}")
    
    # 运行攻击算法
    results = []
    
    for algo_name in args.algorithms:
        print(f"\n运行算法: {algo_name}...")
        try:
            result = run_attack_algorithm(
                algorithm_name=algo_name,
                graph=G,
                budget=budget,
                dataset_name=dataset_name,
                graph_name=graph_name,
                collapse_threshold=args.collapse_threshold,
                random_runs=args.random_runs,
                random_seed=args.random_seed,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                device=args.device,
            )
            results.append(result)
            
            # 计算正确的 R_res
            r_res = compute_r_res_correct(result.removal_fractions, result.lcc_values)
            collapse = result.find_collapse_point(args.collapse_threshold)
            
            print(f"  R_res: {r_res:.4f}")
            print(f"  崩溃点: {collapse:.2%}" if collapse else "  崩溃点: 未崩溃")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\n❌ 没有成功运行的算法")
        return
    
    # 保存数据
    print("\n保存实验数据...")
    save_experiment_data(results, experiment_dir, args.collapse_threshold)
    
    # 绘制对比图
    print("\n生成可视化图像...")
    plot_path = os.path.join(experiment_dir, "comparison_plot.png")
    title = f"Network Dismantling: {graph_name}\n(N={G.number_of_nodes()}, E={G.number_of_edges()}, Budget={budget})"
    intersections = plot_attack_comparison(
        results=results,
        output_path=plot_path,
        collapse_threshold=args.collapse_threshold,
        title=title,
    )
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'算法':<25} {'R_res':<12} {'崩溃点':<12}")
    print("-" * 60)
    for result in results:
        r_res = compute_r_res_correct(result.removal_fractions, result.lcc_values)
        collapse = result.find_collapse_point(args.collapse_threshold)
        collapse_str = f"{collapse:.2%}" if collapse else "N/A"
        print(f"{result.algorithm_name:<25} {r_res:<12.4f} {collapse_str:<12}")
    print("=" * 60)
    print(f"\n所有结果已保存到: {experiment_dir}")


if __name__ == "__main__":
    main()
