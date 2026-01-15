#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据生成脚本
从网络模拟中生成训练数据。

用法:
    # 从合成网络数据生成
    python scripts/generate_data.py --data_source syn --num_graphs 100 --output_dir data/fine_tuning
    
    # 从真实网络数据生成
    python scripts/generate_data.py --data_source true --num_graphs 50 --output_dir data/fine_tuning
    
    # 混合使用（合成+真实+生成）
    python scripts/generate_data.py --data_source all --num_graphs 200 --output_dir data/fine_tuning
    
    # 仅生成 BA/ER 图（原有方式）
    python scripts/generate_data.py --data_source generate --graph_type ba --num_graphs 100
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx


def generate_node_semantics(num_nodes: int, graph_type: str = "generic", node_ids: Optional[List] = None) -> Dict:
    """
    生成节点语义描述
    
    Args:
        num_nodes: 节点数量
        graph_type: 图类型，用于生成相关语义
        node_ids: 节点ID列表（如果提供，使用这些ID；否则使用 0 到 num_nodes-1）
    
    Returns:
        节点语义字典 {node_id: semantic_description}
    """
    # 预定义语义模板
    server_roles = [
        "核心调度服务器，负责全网同步",
        "边缘路由器，负责本地流量",
        "备用电源接口",
        "数据存储节点",
        "网关节点，连接外部网络",
        "负载均衡器",
        "监控服务器",
        "日志收集器",
        "认证服务器",
        "缓存服务器"
    ]
    
    infra_roles = [
        "主变电站",
        "配电站",
        "输电塔",
        "用户接入点",
        "储能设备",
        "发电站",
        "调度中心",
        "备用线路节点"
    ]
    
    social_roles = [
        "意见领袖",
        "社区管理员",
        "普通用户",
        "活跃贡献者",
        "信息中转节点"
    ]
    
    if graph_type == "network":
        roles = server_roles
    elif graph_type == "infra":
        roles = infra_roles
    elif graph_type == "social":
        roles = social_roles
    else:
        roles = server_roles + infra_roles
    
    semantics = {}
    if node_ids is None:
        node_ids = list(range(num_nodes))
    
    for node_id in node_ids:
        role = random.choice(roles)
        # 添加一些随机性
        importance = random.choice(["关键", "重要", "普通", "辅助"])
        semantics[node_id] = f"{importance}{role}"
    
    return semantics


def load_graph_from_file(filepath: Path) -> Optional[nx.Graph]:
    """
    从文件加载图
    
    Args:
        filepath: 图文件路径
    
    Returns:
        NetworkX 图对象，如果加载失败返回 None
    """
    try:
        if filepath.suffix == '.gml':
            # GML 文件可能使用不同的标签，尝试几种方式
            try:
                graph = nx.read_gml(filepath, label='id')
            except Exception:
                try:
                    graph = nx.read_gml(filepath)
                except Exception:
                    # 最后尝试使用 label=None
                    graph = nx.read_gml(filepath, label=None)
        elif filepath.suffix == '.graphml':
            graph = nx.read_graphml(filepath)
        else:
            print(f"  Warning: Unsupported file format: {filepath.suffix}")
            return None
        
        # 转换为无向图（如果是有向图）
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # 只保留最大连通分量
        if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()
        
        return graph
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def get_graph_files(data_source: str, raw_graphs_dir: Path) -> List[Tuple[Path, str]]:
    """
    获取图文件列表
    
    Args:
        data_source: 数据源类型 ("syn", "true", "all")
        raw_graphs_dir: 原始图数据目录
    
    Returns:
        List[(filepath, source_type)]: 图文件路径和来源类型的列表
    """
    graph_files = []
    
    if data_source in ["syn", "all"]:
        syn_dir = raw_graphs_dir / "syn"
        if syn_dir.exists():
            syn_files = list(syn_dir.glob("*.gml"))
            graph_files.extend([(f, "syn") for f in syn_files])
    
    if data_source in ["true", "all"]:
        true_dir = raw_graphs_dir / "true"
        if true_dir.exists():
            true_gml = list(true_dir.glob("*.gml"))
            true_graphml = list(true_dir.glob("*.graphml"))
            graph_files.extend([(f, "true") for f in true_gml + true_graphml])
    
    return graph_files


def generate_single_graph_data(
    graph: Optional[nx.Graph] = None,
    graph_type: Optional[str] = None,
    num_nodes: Optional[int] = None,
    task_type: str = "dismantle",
    budget: int = 10,
    graph_idx: int = 0,
    graph_file: Optional[Path] = None,
    data_source: str = "generate",
    semantic_type: str = "generic"
) -> List[Dict]:
    """
    为单个图生成训练数据
    
    Args:
        graph: NetworkX 图对象（如果提供则直接使用）
        graph_type: 图类型 ("ba", "er")，仅在 graph=None 时使用
        num_nodes: 节点数，仅在 graph=None 时使用
        task_type: 任务类型 ("dismantle", "construct")
        budget: 操作预算
        graph_idx: 图索引
        graph_file: 图文件路径（用于记录）
        data_source: 数据来源 ("generate", "syn", "true", "all")
        semantic_type: 语义类型
    
    Returns:
        样本列表
    """
    from src.env.simulator import NetworkEnvironment, TaskType
    from src.env.metrics import ResilienceMetrics
    from src.data.ocg_builder import OCGExtractor
    
    # 如果没有提供图，则生成图
    if graph is None:
        if graph_type == "ba":
            m = max(2, num_nodes // 20) if num_nodes else 3
            graph = nx.barabasi_albert_graph(num_nodes, m)
        elif graph_type == "er":
            p = 3.0 / num_nodes if num_nodes else 0.05
            graph = nx.erdos_renyi_graph(num_nodes, p)
            # 确保连通
            while not nx.is_connected(graph):
                graph = nx.erdos_renyi_graph(num_nodes, p * 1.2)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    # 确保图是连通的（只保留最大连通分量）
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()
    
    actual_num_nodes = graph.number_of_nodes()
    
    # 生成语义（基于实际节点数和节点ID）
    node_ids = list(graph.nodes())
    node_semantics = generate_node_semantics(actual_num_nodes, semantic_type, node_ids=node_ids)
    
    # 创建环境
    task = TaskType.DISMANTLE if task_type == "dismantle" else TaskType.CONSTRUCT
    env = NetworkEnvironment(
        graph=graph,
        task_type=task,
        budget=min(budget, actual_num_nodes // 2),  # 预算不能超过节点数的一半
        spectral_top_k=min(10, actual_num_nodes // 5),
        node_semantics=node_semantics
    )
    
    # 初始化工具
    metrics = ResilienceMetrics()
    extractor = OCGExtractor(language="zh")
    
    samples = []
    
    # 根据任务类型选择不同的生成逻辑
    if task_type == "dismantle":
        samples = _generate_dismantle_data(
            env, metrics, extractor, budget, graph_idx, graph_file,
            data_source, semantic_type, node_semantics, actual_num_nodes
        )
    else:  # construct
        samples = _generate_construct_data(
            env, metrics, extractor, budget, graph_idx, graph_file,
            data_source, semantic_type, node_semantics, actual_num_nodes
        )
    
    return samples


def _generate_dismantle_data(
    env, metrics, extractor, budget, graph_idx, graph_file,
    data_source, semantic_type, node_semantics, actual_num_nodes
) -> List[Dict]:
    """生成 dismantle 任务的训练数据"""
    samples = []
    
    # 模拟操作序列
    for step in range(budget):
        if env.graph.number_of_nodes() < 3:
            break
        
        # 获取候选节点 (这里用简单的度数排序模拟谱梯度剪枝)
        nodes = list(env.graph.nodes())
        degrees = dict(env.graph.degree())
        sorted_nodes = sorted(nodes, key=lambda x: degrees[x], reverse=True)
        candidates = sorted_nodes[:min(5, len(sorted_nodes))]
        
        if not candidates:
            break
        
        # 计算真实影响分数 (auxiliary_labels)
        impact_scores = metrics.batch_compute_impact_scores(env.graph, candidates)
        
        # 构建操作和标签
        auxiliary_labels = {}
        operations = []
        for idx, node in enumerate(candidates):
            op_id = f"op_{idx+1:02d}"
            # 归一化分数到 [0, 1]
            score = impact_scores.get(node, 0)
            if score < 0:
                score = 0
            auxiliary_labels[op_id] = round(score, 4)
            operations.append({"op_id": op_id, "target": node})
        
        # 获取正确排序
        ground_truth = sorted(
            auxiliary_labels.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ground_truth_ranking = [op_id for op_id, _ in ground_truth]
        
        # 生成推理过程
        reasoning = generate_reasoning_trace_dismantle(
            candidates, 
            auxiliary_labels, 
            ground_truth_ranking,
            env.graph,
            node_semantics
        )
        
        # 提取 OCG 并构建样本
        ocg_data = extractor.extract_ocg(
            graph=env.graph,
            candidate_nodes=candidates,
            task_type="dismantle",
            current_step=step + 1,
            total_steps=budget,
            node_semantics=node_semantics
        )
        
        sample = extractor.build_conversation_data(
            ocg_data=ocg_data,
            ground_truth_ranking=ground_truth_ranking,
            auxiliary_labels=auxiliary_labels,
            reasoning_trace=reasoning
        )
        
        # 更新样本 ID 和元数据
        graph_type_str = data_source if data_source != "generate" else "ba"
        sample["id"] = f"train_dismantle_{data_source}_{graph_idx:04d}_{step:02d}"
        sample["meta"]["graph_type"] = graph_type_str
        sample["meta"]["num_nodes"] = actual_num_nodes
        sample["meta"]["graph_idx"] = graph_idx
        sample["meta"]["data_source"] = data_source
        sample["meta"]["sign"] = -1  # 符号函数：dismantle 是减小韧性
        if graph_file:
            sample["meta"]["graph_file"] = str(graph_file.name)
        
        samples.append(sample)
        
        # 执行最佳操作
        if ground_truth_ranking:
            best_op = ground_truth_ranking[0]
            best_node = operations[int(best_op.split("_")[1]) - 1]["target"]
            if best_node in env.graph:
                env.graph.remove_node(best_node)
        
        env.current_step += 1
    
    return samples


def _generate_construct_data(
    env, metrics, extractor, budget, graph_idx, graph_file,
    data_source, semantic_type, node_semantics, actual_num_nodes
) -> List[Dict]:
    """生成 construct 任务的训练数据（添加边来增强韧性）"""
    samples = []
    
    # 模拟操作序列
    for step in range(budget):
        if env.graph.number_of_nodes() < 3:
            break
        
        # 获取候选边（使用环境的剪枝方法）
        candidate_edges = env.prune_candidates(candidate_type="edge", top_k=5)
        
        if not candidate_edges:
            break
        
        # 计算添加边的增益分数
        edge_gains = metrics.batch_compute_edge_gains(env.graph, candidate_edges)
        
        # 构建操作和标签
        auxiliary_labels = {}
        operations = []
        for idx, edge in enumerate(candidate_edges):
            op_id = f"op_{idx+1:02d}"
            score = edge_gains.get(edge, 0)
            if score < 0:
                score = 0
            auxiliary_labels[op_id] = round(score, 4)
            operations.append({"op_id": op_id, "target": edge})
        
        # 获取正确排序
        ground_truth = sorted(
            auxiliary_labels.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ground_truth_ranking = [op_id for op_id, _ in ground_truth]
        
        # 生成推理过程
        reasoning = generate_reasoning_trace_construct(
            candidate_edges, 
            auxiliary_labels, 
            ground_truth_ranking,
            env.graph,
            node_semantics
        )
        
        # 构建自定义的 OCG 数据（因为 construct 是边操作）
        sample = build_construct_conversation_data(
            graph=env.graph,
            candidate_edges=candidate_edges,
            current_step=step + 1,
            total_steps=budget,
            node_semantics=node_semantics,
            ground_truth_ranking=ground_truth_ranking,
            auxiliary_labels=auxiliary_labels,
            reasoning_trace=reasoning
        )
        
        # 更新样本 ID 和元数据
        graph_type_str = data_source if data_source != "generate" else "ba"
        sample["id"] = f"train_construct_{data_source}_{graph_idx:04d}_{step:02d}"
        sample["meta"]["graph_type"] = graph_type_str
        sample["meta"]["num_nodes"] = actual_num_nodes
        sample["meta"]["graph_idx"] = graph_idx
        sample["meta"]["data_source"] = data_source
        sample["meta"]["sign"] = +1  # 符号函数：construct 是增大韧性
        if graph_file:
            sample["meta"]["graph_file"] = str(graph_file.name)
        
        samples.append(sample)
        
        # 执行最佳操作（添加边）
        if ground_truth_ranking and operations:
            best_op = ground_truth_ranking[0]
            best_edge = operations[int(best_op.split("_")[1]) - 1]["target"]
            u, v = best_edge
            if u in env.graph and v in env.graph:
                env.graph.add_edge(u, v)
        
        env.current_step += 1
    
    return samples


def build_construct_conversation_data(
    graph: nx.Graph,
    candidate_edges: List[Tuple],
    current_step: int,
    total_steps: int,
    node_semantics: Dict,
    ground_truth_ranking: List[str],
    auxiliary_labels: Dict[str, float],
    reasoning_trace: str
) -> Dict:
    """为 construct 任务构建对话数据"""
    import json
    
    # 系统提示（强调这是统一的韧性优化任务）
    system_prompt = (
        "你是一个网络韧性优化专家。你的目标是通过分析局部子图结构（OCG）"
        "和节点语义，选择能最显著改变网络韧性积分 R_res 的操作。"
        "本次任务是构造任务（σ=+1）：选择添加后能最大化提升网络韧性的边。"
    )
    
    # 构建用户提示
    user_prompt = f"【当前状态】\n步骤：{current_step} / {total_steps}\n"
    user_prompt += "目标：最大化韧性 (Construct, σ=+1)\n\n"
    user_prompt += "【候选边信息】\n以下是候选边及其端点的语义摘要：\n\n"
    
    for idx, edge in enumerate(candidate_edges, 1):
        u, v = edge
        sem_u = node_semantics.get(u, f"节点{u}")
        sem_v = node_semantics.get(v, f"节点{v}")
        deg_u = graph.degree(u) if u in graph else 0
        deg_v = graph.degree(v) if v in graph else 0
        
        user_prompt += f"{idx}. 边 [{u} — {v}]:\n"
        user_prompt += f"   - 端点1 [{u}]: {sem_u}，度数 {deg_u}\n"
        user_prompt += f"   - 端点2 [{v}]: {sem_v}，度数 {deg_v}\n"
        user_prompt += f"   - 连接意义：添加此边可增强两节点间的连通性\n\n"
    
    user_prompt += "【候选操作列表】\n"
    for idx, edge in enumerate(candidate_edges, 1):
        u, v = edge
        user_prompt += f"- [op_{idx:02d}]: 添加边 ({u}, {v})\n"
    
    user_prompt += "\n请分析上述选项，并按推荐优先级排序（增益最大的优先）。"
    
    # 生成 Assistant 回复
    response = {
        "reasoning_trace": reasoning_trace,
        "ranked_list": ground_truth_ranking,
        "best_action": ground_truth_ranking[0] if ground_truth_ranking else ""
    }
    assistant_content = f"```json\n{json.dumps(response, ensure_ascii=False, indent=2)}\n```"
    
    return {
        "id": f"train_construct_{current_step:03d}",
        "meta": {
            "task": "construct",
            "budget_step": f"{current_step}/{total_steps}"
        },
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "user", "value": user_prompt},
            {"from": "assistant", "value": assistant_content}
        ],
        "auxiliary_labels": auxiliary_labels
    }


def generate_reasoning_trace(
    candidates: List,
    auxiliary_labels: Dict[str, float],
    ground_truth_ranking: List[str],
    graph,
    node_semantics: Dict
) -> str:
    """生成推理过程文本（兼容旧代码，默认调用 dismantle）"""
    return generate_reasoning_trace_dismantle(
        candidates, auxiliary_labels, ground_truth_ranking, graph, node_semantics
    )


def generate_reasoning_trace_dismantle(
    candidates: List,
    auxiliary_labels: Dict[str, float],
    ground_truth_ranking: List[str],
    graph,
    node_semantics: Dict
) -> str:
    """生成 dismantle 任务的推理过程文本"""
    import networkx as nx
    
    reasoning_parts = []
    
    for rank, op_id in enumerate(ground_truth_ranking[:3], 1):
        idx = int(op_id.split("_")[1]) - 1
        if idx >= len(candidates):
            continue
        
        node = candidates[idx]
        score = auxiliary_labels[op_id]
        degree = graph.degree(node)
        semantic = node_semantics.get(node, "未知")
        
        # 检查是否是割点
        is_articulation = node in set(nx.articulation_points(graph)) if nx.is_connected(graph) else False
        
        reason = f"{rank}. 分析 [{op_id}] (移除节点 {node}): "
        reason += f"该节点是'{semantic}'，"
        reason += f"度数为 {degree}"
        if degree > len(candidates):
            reason += " (高度数)"
        
        if is_articulation:
            reason += "，是网络的割点，移除会导致网络分裂"
        
        if score > 0.5:
            reason += f"。预计破坏分数 {score:.2f}，破坏力较大。"
        elif score > 0.2:
            reason += f"。预计破坏分数 {score:.2f}，有一定破坏力。"
        else:
            reason += f"。预计破坏分数 {score:.2f}，影响有限。"
        
        reasoning_parts.append(reason)
    
    return "\n".join(reasoning_parts)


def generate_reasoning_trace_construct(
    candidate_edges: List[Tuple],
    auxiliary_labels: Dict[str, float],
    ground_truth_ranking: List[str],
    graph,
    node_semantics: Dict
) -> str:
    """生成 construct 任务的推理过程文本（添加边）"""
    import networkx as nx
    
    reasoning_parts = []
    
    for rank, op_id in enumerate(ground_truth_ranking[:3], 1):
        idx = int(op_id.split("_")[1]) - 1
        if idx >= len(candidate_edges):
            continue
        
        edge = candidate_edges[idx]
        u, v = edge
        score = auxiliary_labels[op_id]
        
        # 获取端点信息
        deg_u = graph.degree(u) if u in graph else 0
        deg_v = graph.degree(v) if v in graph else 0
        sem_u = node_semantics.get(u, f"节点{u}")
        sem_v = node_semantics.get(v, f"节点{v}")
        
        reason = f"{rank}. 分析 [{op_id}] (添加边 {u}-{v}): "
        reason += f"连接'{sem_u}'(度{deg_u}) 与 '{sem_v}'(度{deg_v})"
        
        # 分析连接的意义
        if deg_u < 3 or deg_v < 3:
            reason += "，可增强低度数节点的冗余连接"
        elif deg_u > 5 and deg_v > 5:
            reason += "，连接两个核心节点，增强骨干韧性"
        else:
            reason += "，平衡网络结构"
        
        if score > 0.1:
            reason += f"。预计增益分数 {score:.4f}，增益显著。"
        elif score > 0.01:
            reason += f"。预计增益分数 {score:.4f}，有一定增益。"
        else:
            reason += f"。预计增益分数 {score:.4f}，增益较小。"
        
        reasoning_parts.append(reason)
    
    return "\n".join(reasoning_parts)


def main():
    parser = argparse.ArgumentParser(description="生成网络韧性优化训练数据")
    
    # 快速测试模式
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式：使用少量数据快速跑通流程")
    
    parser.add_argument("--data_source", type=str, default="generate", 
                        choices=["generate", "syn", "true", "all"],
                        help="数据来源: generate(生成BA/ER), syn(合成网络), true(真实网络), all(混合)")
    parser.add_argument("--raw_graphs_dir", type=str, default="data/raw_graphs",
                        help="原始图数据目录")
    parser.add_argument("--num_graphs", type=int, default=100, 
                        help="使用的图数量（对于文件数据，会随机采样）")
    parser.add_argument("--graph_type", type=str, default="ba", 
                        choices=["ba", "er", "mixed"],
                        help="图类型（仅在 data_source=generate 时有效）")
    parser.add_argument("--min_nodes", type=int, default=50, 
                        help="最小节点数（仅在 data_source=generate 时有效）")
    parser.add_argument("--max_nodes", type=int, default=200, 
                        help="最大节点数（仅在 data_source=generate 时有效）")
    parser.add_argument("--task_type", type=str, default="dismantle", 
                        choices=["dismantle", "construct", "both"], 
                        help="任务类型")
    parser.add_argument("--budget", type=int, default=10, 
                        help="每个图的操作预算")
    parser.add_argument("--output_dir", type=str, default="data/fine_tuning", 
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split_ratio", type=float, default=0.9, 
                        help="训练集比例")
    parser.add_argument("--min_graph_size", type=int, default=20,
                        help="最小图大小（节点数），小于此大小的图会被跳过")
    
    args = parser.parse_args()
    
    # 快速测试模式：覆盖参数为小规模
    if args.quick_test:
        print("=" * 60)
        print("🚀 快速测试模式：使用少量数据快速跑通流程")
        print("=" * 60)
        args.num_graphs = 5  # 只用 5 个图
        args.budget = 3  # 每个图只生成 3 步数据
        args.min_nodes = 30
        args.max_nodes = 50
        args.task_type = "both"  # 混合生成 dismantle 和 construct
        args.output_dir = "data/fine_tuning/quick_test"
        args.data_source = "generate"  # 直接生成图
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_graphs_dir = Path(args.raw_graphs_dir)
    
    print(f"Generating data with config:")
    print(f"  - Data source: {args.data_source}")
    print(f"  - Num graphs: {args.num_graphs}")
    print(f"  - Task type: {args.task_type}")
    print(f"  - Budget: {args.budget}")
    print(f"  - Output dir: {output_dir}")
    if args.data_source == "generate":
        print(f"  - Graph type: {args.graph_type}")
        print(f"  - Nodes range: [{args.min_nodes}, {args.max_nodes}]")
    
    all_samples = []
    
    # 确定任务类型
    if args.task_type == "both":
        tasks = ["dismantle", "construct"]
    else:
        tasks = [args.task_type]
    
    # 准备图列表
    graphs_to_process = []
    
    if args.data_source == "generate":
        # 生成图模式
        if args.graph_type == "mixed":
            graph_types = ["ba", "er"]
        else:
            graph_types = [args.graph_type]
        
        for i in range(args.num_graphs):
            num_nodes = random.randint(args.min_nodes, args.max_nodes)
            graph_type = random.choice(graph_types)
            graphs_to_process.append(("generate", None, graph_type, num_nodes, None))
    else:
        # 从文件加载模式
        graph_files = get_graph_files(args.data_source, raw_graphs_dir)
        
        if not graph_files:
            print(f"Error: No graph files found in {raw_graphs_dir}")
            print(f"Please check that {args.data_source} directory exists and contains .gml or .graphml files")
            return
        
        print(f"Found {len(graph_files)} graph files")
        
        # 随机采样
        if len(graph_files) > args.num_graphs:
            graph_files = random.sample(graph_files, args.num_graphs)
        
        for graph_file, source_type in graph_files:
            graphs_to_process.append((source_type, graph_file, None, None, graph_file))
    
    # 处理所有图
    total_graphs = len(graphs_to_process)
    successful = 0
    failed = 0
    
    for i, (data_source, graph_file, graph_type, num_nodes, file_path) in enumerate(graphs_to_process):
        task_type = random.choice(tasks)
        semantic_type = random.choice(["network", "infra", "generic"])
        
        try:
            # 加载或生成图
            if data_source == "generate":
                graph = None
                print(f"[{i+1}/{total_graphs}] Generating {graph_type} graph with {num_nodes} nodes, task={task_type}")
            else:
                print(f"[{i+1}/{total_graphs}] Loading graph from {graph_file.name}, task={task_type}")
                graph = load_graph_from_file(graph_file)
                if graph is None:
                    failed += 1
                    continue
                
                actual_nodes = graph.number_of_nodes()
                if actual_nodes < args.min_graph_size:
                    print(f"  Skipped: graph too small ({actual_nodes} nodes < {args.min_graph_size})")
                    failed += 1
                    continue
            
            # 生成训练数据
            samples = generate_single_graph_data(
                graph=graph,
                graph_type=graph_type,
                num_nodes=num_nodes,
                task_type=task_type,
                budget=args.budget,
                graph_idx=i,
                graph_file=graph_file if data_source != "generate" else None,
                data_source=data_source,
                semantic_type=semantic_type
            )
            
            all_samples.extend(samples)
            successful += 1
            print(f"  Generated {len(samples)} samples")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\nProcessing completed: {successful} successful, {failed} failed")
    
    print(f"\nTotal samples generated: {len(all_samples)}")
    
    # 划分训练集和验证集
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * args.split_ratio)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    # 保存数据
    train_path = output_dir / "train.json"
    eval_path = output_dir / "eval.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(train_samples)} training samples to {train_path}")
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(eval_samples)} evaluation samples to {eval_path}")
    
    # 保存配置
    config = {
        "data_source": args.data_source,
        "num_graphs": args.num_graphs,
        "task_type": args.task_type,
        "budget": args.budget,
        "seed": args.seed,
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "successful_graphs": successful,
        "failed_graphs": failed
    }
    
    if args.data_source == "generate":
        config["graph_type"] = args.graph_type
        config["min_nodes"] = args.min_nodes
        config["max_nodes"] = args.max_nodes
    else:
        config["raw_graphs_dir"] = str(args.raw_graphs_dir)
        config["min_graph_size"] = args.min_graph_size
    
    config_path = output_dir / "data_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nData generation completed!")
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
