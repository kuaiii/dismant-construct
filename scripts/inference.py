#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型推理脚本 - 在实际网络上测试模型

用法:
    python scripts/inference.py --checkpoint outputs/mixed_model/checkpoints/best --graph data/raw_graphs/syn/graph_001.gml
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import networkx as nx
from typing import List, Dict, Union, Tuple

from src.model.fusion_llm import ResilienceLLM, ModelConfig
from src.env.simulator import NetworkEnvironment, TaskType
from src.env.metrics import ResilienceMetrics
from src.data.ocg_builder import OCGExtractor


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def predict_action(
    model: ResilienceLLM,
    env: NetworkEnvironment,
    ocg_extractor: OCGExtractor,
    device: str = "cuda"
) -> Union[int, Tuple[int, int]]:
    """
    使用模型预测下一个操作
    
    Args:
        model: 训练好的模型
        env: 网络环境
        ocg_extractor: OCG 提取器
        device: 设备
    
    Returns:
        - 对于 dismantle: 选择的节点 ID
        - 对于 construct: (源节点, 目标节点) 元组
    """
    # 根据任务类型获取候选
    if env.task_type == TaskType.CONSTRUCT:
        # Construct: 获取边候选
        edge_candidates = env.prune_candidates(candidate_type="edge")
        if not edge_candidates:
            return None
        
        # 将边候选转换为节点候选（用于 OCG 提取）
        # 取所有边候选中的唯一节点
        candidate_nodes = list(set([u for u, v in edge_candidates] + [v for u, v in edge_candidates]))
        candidate_nodes = candidate_nodes[:env.spectral_top_k]  # 限制数量
    else:
        # Dismantle: 获取节点候选
        candidate_nodes = env.prune_candidates(candidate_type="node")
        if not candidate_nodes:
            return None
        edge_candidates = None
    
    candidates = candidate_nodes
    
    # 提取 OCG 并构建 prompt
    ocg_data = ocg_extractor.extract_ocg(
        graph=env.graph,
        candidate_nodes=candidates,
        task_type=env.task_type.value,
        current_step=env.current_step + 1,
        total_steps=env.budget
    )
    
    # 构建输入文本
    # 推理阶段只需要 user_prompt（build_conversation_data 是训练数据构造接口）
    input_text = ocg_data.user_prompt
    
    # Tokenize
    inputs = model.tokenizer(
        input_text,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 获取候选操作位置索引（简化：使用序列末尾）
    num_candidates = len(candidates)
    seq_len = input_ids.shape[1]
    candidate_indices = torch.tensor(
        [[seq_len - num_candidates + j for j in range(num_candidates)]],
        device=device,
        dtype=torch.long
    )
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_indices=candidate_indices,
            return_scores=True
        )
        
        if "scores" in outputs and outputs["scores"] is not None:
            scores = outputs["scores"][0]  # [num_candidates]
            # 选择分数最高的候选
            best_idx = torch.argmax(scores).item()
            selected_node = candidates[best_idx]
            
            # 对于 construct 任务，需要选择目标节点
            if env.task_type == TaskType.CONSTRUCT:
                # 从边候选中找到包含该节点的最佳边
                if edge_candidates:
                    # 优先选择包含选中节点且分数高的边
                    candidate_edges_with_node = [(u, v) for u, v in edge_candidates 
                                                  if u == selected_node or v == selected_node]
                    if candidate_edges_with_node:
                        # 选择第一个可用的边（可以改进为基于某种评分）
                        u, v = candidate_edges_with_node[0]
                        if u == selected_node:
                            return (u, v)
                        else:
                            return (v, u)
                
                # 如果没有找到合适的边，使用启发式方法选择目标节点
                remaining_nodes = [n for n in env.graph.nodes() 
                                  if n != selected_node and not env.graph.has_edge(selected_node, n)]
                if remaining_nodes:
                    # 选择度数最高的节点作为目标（增加连接性）
                    degrees = dict(env.graph.degree())
                    target = max(remaining_nodes, key=lambda n: degrees.get(n, 0))
                    return (selected_node, target)
                return None
            else:
                # Dismantle: 直接返回节点
                return selected_node
        else:
            # 如果没有 scores，使用启发式方法
            import random
            if env.task_type == TaskType.CONSTRUCT:
                selected_node = random.choice(candidates)
                remaining_nodes = [n for n in env.graph.nodes() 
                                  if n != selected_node and not env.graph.has_edge(selected_node, n)]
                if remaining_nodes:
                    degrees = dict(env.graph.degree())
                    target = max(remaining_nodes, key=lambda n: degrees.get(n, 0))
                    return (selected_node, target)
                return None
            else:
                return random.choice(candidates)


def run_inference(
    checkpoint_path: str,
    graph_path: str,
    task_type: str = "dismantle",
    budget: int = 10,
    config_path: str = "configs/default.yaml",
    device: str = "cuda"
):
    """
    在单个图上运行推理
    
    Args:
        checkpoint_path: 检查点路径
        graph_path: 图文件路径
        task_type: 任务类型 (dismantle/construct)
        budget: 操作预算
        config_path: 配置文件路径
        device: 设备
    """
    print("=" * 60)
    print("模型推理测试")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建模型
    print("\n正在加载模型...")
    model_config = ModelConfig(
        llm_model_name=config['model']['llm']['model_name'],
        use_lora=config['model']['lora']['enabled'],
        lora_r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['alpha'],
        lora_dropout=config['model']['lora']['dropout'],
        use_geometric_encoder=config['model']['geometric_encoder']['enabled'],
        d_model=config['model']['fusion']['d_model']
    )
    
    model = ResilienceLLM(model_config)
    model.initialize(device=device)
    
    # 加载检查点
    checkpoint_path_obj = Path(checkpoint_path)
    
    # 如果路径不存在，尝试智能查找
    if not checkpoint_path_obj.exists():
        # 如果路径以 "best" 结尾，尝试在父目录查找 epoch 目录
        if checkpoint_path_obj.name == "best" or checkpoint_path_obj.name.endswith("best"):
            parent_dir = checkpoint_path_obj.parent
            if parent_dir.exists():
                print(f"⚠️  路径 {checkpoint_path} 不存在，尝试在父目录 {parent_dir} 查找检查点...")
                checkpoint_path_obj = parent_dir
        else:
            raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    if checkpoint_path_obj.is_dir():
        # 查找所有 epoch 目录
        epoch_dirs = sorted(checkpoint_path_obj.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.startswith("epoch_") else 0, reverse=True)
        if epoch_dirs:
            # 使用最新的 epoch
            latest_epoch_dir = epoch_dirs[0]
            checkpoint_file = latest_epoch_dir / "model.pt"
            if checkpoint_file.exists():
                print(f"找到检查点: {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"在 {latest_epoch_dir} 中未找到 model.pt")
        else:
            checkpoint_files = list(checkpoint_path_obj.glob("*.pt")) + list(checkpoint_path_obj.glob("*.pth"))
            if checkpoint_files:
                checkpoint_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                print(f"加载检查点: {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"在 {checkpoint_path_obj} 中未找到模型文件")
    elif checkpoint_path_obj.is_file():
        print(f"加载检查点: {checkpoint_path_obj}")
        checkpoint = torch.load(checkpoint_path_obj, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    model.eval()
    print("模型加载完成")
    
    # 加载图
    print(f"\n正在加载图: {graph_path}")
    graph = NetworkEnvironment.load_graph(graph_path)
    print(f"图节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    
    # 创建环境
    task = TaskType.DISMANTLE if task_type.lower() == "dismantle" else TaskType.CONSTRUCT
    env = NetworkEnvironment(
        graph=graph,
        task_type=task,
        budget=budget
    )
    
    # 创建 OCG 提取器
    ocg_extractor = OCGExtractor(
        language=config.get('ocg', {}).get('language', 'zh')
    )
    
    # 创建指标计算器
    metrics = ResilienceMetrics()
    
    # 记录初始状态
    initial_lcc = metrics.compute_lcc_ratio(env.graph)
    # R_res 需要 LCC 曲线；推理过程中用当前已观测到的曲线近似
    lcc_curve = [initial_lcc]
    initial_r_res = metrics.compute_r_res(lcc_curve)
    
    print(f"\n初始状态:")
    print(f"  LCC 比例: {initial_lcc:.4f}")
    print(f"  R_res: {initial_r_res:.4f}")
    
    # 执行推理
    print(f"\n开始推理 (预算: {budget} 步)...")
    actions_taken = []
    
    for step in range(budget):
        # 预测下一个操作
        action = predict_action(model, env, ocg_extractor, device)
        
        if action is None:
            print(f"步骤 {step + 1}: 没有可用候选，提前结束")
            break
        
        # 执行操作
        if task == TaskType.DISMANTLE:
            selected_node = action
            env.remove_node(selected_node)
            action_desc = f"移除节点 {selected_node}"
            actions_taken.append(selected_node)
        else:
            # Construct: action 是 (源节点, 目标节点) 元组
            u, v = action
            if u in env.graph and v in env.graph and not env.graph.has_edge(u, v):
                env.add_edge(u, v)
                action_desc = f"添加边 ({u}, {v})"
                actions_taken.append((u, v))
            else:
                print(f"步骤 {step + 1}: 边 ({u}, {v}) 已存在或节点不存在，跳过")
                continue
        
        # 计算当前状态
        current_lcc = metrics.compute_lcc_ratio(env.graph)
        lcc_curve.append(current_lcc)
        current_r_res = metrics.compute_r_res(lcc_curve)
        
        print(f"步骤 {step + 1}: {action_desc}")
        print(f"  LCC: {current_lcc:.4f}, R_res: {current_r_res:.4f}")
    
    # 最终结果
    final_lcc = metrics.compute_lcc_ratio(env.graph)
    # 确保最终状态也计入曲线（如果循环提前 break 且未 append）
    if not lcc_curve or lcc_curve[-1] != final_lcc:
        lcc_curve.append(final_lcc)
    final_r_res = metrics.compute_r_res(lcc_curve)
    
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    print(f"执行的操作数: {len(actions_taken)}")
    print(f"操作序列: {actions_taken}")
    print(f"\n初始 -> 最终:")
    print(f"  LCC: {initial_lcc:.4f} -> {final_lcc:.4f} (变化: {final_lcc - initial_lcc:.4f})")
    print(f"  R_res: {initial_r_res:.4f} -> {final_r_res:.4f} (变化: {final_r_res - initial_r_res:.4f})")
    
    if task == TaskType.DISMANTLE:
        print(f"\n拆解效果: R_res 降低了 {initial_r_res - final_r_res:.4f}")
    else:
        print(f"\n构造效果: R_res 提高了 {final_r_res - initial_r_res:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行推理")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--graph", type=str, required=True, help="图文件路径")
    parser.add_argument("--task", type=str, default="dismantle", choices=["dismantle", "construct"], help="任务类型")
    parser.add_argument("--budget", type=int, default=10, help="操作预算")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        task_type=args.task,
        budget=args.budget,
        config_path=args.config,
        device=args.device
    )


if __name__ == "__main__":
    main()
