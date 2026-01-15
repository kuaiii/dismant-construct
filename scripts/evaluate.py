#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本

用法:
    python scripts/evaluate.py --checkpoint outputs/mixed_model/checkpoints/best
    python scripts/evaluate.py --checkpoint outputs/mixed_model/checkpoints/best --eval_data data/fine_tuning/combined/eval.json
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from src.model.fusion_llm import ResilienceLLM, ModelConfig
from src.data.dataset import create_dataloader
from src.model.loss import ListMLELoss


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_ranking_metrics(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, float]:
    """
    计算排序指标
    
    Args:
        scores: 预测分数 [batch_size, num_candidates]
        labels: 真实标签 [batch_size, num_candidates]
        mask: 有效候选掩码 [batch_size, num_candidates]
    
    Returns:
        指标字典
    """
    batch_size = scores.shape[0]
    
    # 确保数据在 CPU 上并转换为 float32（避免 Half 精度问题）
    scores = scores.detach().cpu().float()
    labels = labels.detach().cpu().float()
    if mask is not None:
        mask = mask.detach().cpu()
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
        labels = labels.masked_fill(~mask.bool(), 0.0)
    
    # NDCG@K
    def ndcg_at_k(y_true, y_pred, k=5):
        """计算 NDCG@K"""
        # 获取 top-k 预测
        k_actual = min(k, len(y_pred))
        _, top_k_indices = torch.topk(y_pred, k_actual, dim=-1)
        # 确保索引是 long 类型
        top_k_indices = top_k_indices.long()
        top_k_true = y_true[top_k_indices]
        
        # DCG - 将张量转换为 numpy 进行计算
        top_k_true_np = top_k_true.numpy()
        dcg = 0.0
        for i, rel in enumerate(top_k_true_np):
            dcg += float(rel) / np.log2(i + 2)
        
        # IDCG (理想情况)
        ideal_sorted = torch.sort(y_true, descending=True)[0]
        ideal_sorted_np = ideal_sorted[:k].numpy()
        idcg = 0.0
        for i, rel in enumerate(ideal_sorted_np):
            idcg += float(rel) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    def mrr(y_true, y_pred):
        """计算 MRR"""
        sorted_indices = torch.argsort(y_pred, descending=True).long()
        for rank, idx in enumerate(sorted_indices.tolist(), 1):
            if y_true[idx].item() > 0:
                return 1.0 / rank
        return 0.0
    
    # 计算指标
    all_ndcg_5 = []
    all_ndcg_10 = []
    all_mrr = []
    all_top1_acc = []
    
    for i in range(batch_size):
        y_true = labels[i]
        y_pred = scores[i]
        
        # 过滤无效候选
        if mask is None:
            valid_mask = (y_true > 0).bool()  # 确保是布尔类型
        else:
            valid_mask = mask[i].bool() if mask[i].dtype != torch.bool else mask[i]  # 确保是布尔类型
        
        if valid_mask.sum() == 0:
            continue
        
        # 使用布尔索引
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # NDCG
        ndcg5 = ndcg_at_k(y_true_valid, y_pred_valid, k=5)
        ndcg10 = ndcg_at_k(y_true_valid, y_pred_valid, k=10)
        all_ndcg_5.append(ndcg5)
        all_ndcg_10.append(ndcg10)
        
        # MRR
        mrr_score = mrr(y_true_valid, y_pred_valid)
        all_mrr.append(mrr_score)
        
        # Top-1 Accuracy
        top1_idx = torch.argmax(y_pred_valid).long()
        top1_acc = 1.0 if y_true_valid[top1_idx].item() == y_true_valid.max().item() else 0.0
        all_top1_acc.append(top1_acc)
    
    return {
        "ndcg@5": np.mean(all_ndcg_5) if all_ndcg_5 else 0.0,
        "ndcg@10": np.mean(all_ndcg_10) if all_ndcg_10 else 0.0,
        "mrr": np.mean(all_mrr) if all_mrr else 0.0,
        "top1_accuracy": np.mean(all_top1_acc) if all_top1_acc else 0.0,
        "num_samples": len(all_ndcg_5)
    }


def evaluate_model(
    checkpoint_path: str,
    eval_data_path: str,
    config_path: str = "configs/default.yaml",
    device: str = "cuda"
):
    """
    评估模型
    
    Args:
        checkpoint_path: 检查点路径
        eval_data_path: 评估数据路径
        config_path: 配置文件路径
        device: 设备
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建模型配置
    model_config = ModelConfig(
        llm_model_name=config['model']['llm']['model_name'],
        use_lora=config['model']['lora']['enabled'],
        lora_r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['alpha'],
        lora_dropout=config['model']['lora']['dropout'],
        use_geometric_encoder=config['model']['geometric_encoder']['enabled'],
        d_model=config['model']['fusion']['d_model']
    )
    
    # 创建模型
    print("\n正在加载模型...")
    model = ResilienceLLM(model_config)
    model.initialize(device=device)
    
    # 加载检查点
    checkpoint_path_obj = Path(checkpoint_path)
    
    # 尝试多种路径格式
    possible_paths = [
        checkpoint_path_obj,  # 直接路径
        checkpoint_path_obj / "model.pt",  # 检查点目录下的 model.pt
        checkpoint_path_obj.parent / "checkpoints" / checkpoint_path_obj.name / "model.pt",  # outputs/xxx/checkpoints/best/model.pt
        checkpoint_path_obj.parent.parent / "checkpoints" / checkpoint_path_obj.name / "model.pt",  # 更深一层
    ]
    
    # 如果是目录，尝试查找所有可能的检查点文件
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
            # 直接查找 .pt 文件
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
                # 尝试加载 adapter
                adapter_path = checkpoint_path_obj / "adapter_model.bin"
                if adapter_path.exists():
                    print(f"加载 LoRA 适配器: {adapter_path}")
                    # LoRA 适配器会自动加载
                else:
                    raise FileNotFoundError(f"在 {checkpoint_path} 中未找到模型文件。请检查路径是否正确。")
    elif checkpoint_path_obj.is_file():
        # 直接是文件
        print(f"加载检查点: {checkpoint_path_obj}")
        checkpoint = torch.load(checkpoint_path_obj, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        # 尝试查找可能的路径
        found = False
        for path in possible_paths:
            if path.exists() and path.is_file():
                print(f"找到检查点: {path}")
                checkpoint = torch.load(path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                found = True
                break
        
        if not found:
            # 列出可能的路径帮助用户
            print(f"\n❌ 错误: 检查点路径不存在: {checkpoint_path}")
            print("\n请尝试以下路径之一:")
            print(f"  1. outputs/resilience_llm/checkpoints/epoch_3/model.pt")
            print(f"  2. outputs/mixed_model/checkpoints/epoch_3/model.pt")
            print(f"  3. 或指定具体的检查点文件路径")
            raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    model.eval()
    print("模型加载完成")
    
    # 加载评估数据
    print(f"\n正在加载评估数据: {eval_data_path}")
    eval_loader = create_dataloader(
        data_path=eval_data_path,
        tokenizer=model.tokenizer,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        max_length=config['data']['loading']['max_length']
    )
    print(f"评估样本数: {len(eval_loader.dataset)}")
    
    # 评估
    print("\n开始评估...")
    all_metrics = []
    total_loss = 0.0
    num_batches = 0
    
    loss_fn = ListMLELoss()
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="评估中"):
            input_ids = batch.get("input_ids")
            if input_ids is None:
                continue
            
            input_ids = input_ids.to(device)
            attention_mask = batch["attention_mask"].to(device)
            auxiliary_labels = batch["auxiliary_labels"].to(device)
            candidate_mask = batch.get("candidate_mask")
            if candidate_mask is not None:
                candidate_mask = candidate_mask.to(device)
            
            # 获取候选操作位置索引（直接实现，不依赖训练器）
            batch_size, seq_len = input_ids.shape
            
            # 从 batch 中获取候选数量
            auxiliary_labels_tensor = batch.get("auxiliary_labels")
            if auxiliary_labels_tensor is None:
                continue
            
            num_candidates = auxiliary_labels_tensor.shape[1] if isinstance(auxiliary_labels_tensor, torch.Tensor) else 0
            
            if num_candidates == 0:
                continue
            
            # 简化方法：使用序列末尾的 token 位置
            candidate_indices = []
            attention_mask_batch = batch.get("attention_mask", None)
            
            for i in range(batch_size):
                if attention_mask_batch is not None:
                    # 使用 attention_mask 找到最后一个有效位置
                    valid_length = attention_mask_batch[i].sum().item()
                    # 使用最后几个位置作为候选操作的表示
                    positions = []
                    for j in range(num_candidates):
                        # 从末尾往前取位置
                        pos = max(0, valid_length - num_candidates + j)
                        positions.append(pos)
                    candidate_indices.append(positions)
                else:
                    # 如果没有 attention_mask，使用序列末尾
                    positions = [max(0, seq_len - num_candidates + j) for j in range(num_candidates)]
                    candidate_indices.append(positions)
            
            candidate_indices = torch.tensor(candidate_indices, device=input_ids.device, dtype=torch.long)
            
            # 模型前向传播（使用 no_grad 确保在评估模式下）
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=False):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_indices=candidate_indices,
                    return_scores=True
                )
            
            # 计算损失
            if "scores" in outputs and outputs["scores"] is not None:
                scores = outputs["scores"]
                # 确保 auxiliary_labels 与 scores 在同一设备上
                auxiliary_labels_batch = auxiliary_labels.to(scores.device)
                candidate_mask_batch = candidate_mask.to(scores.device) if candidate_mask is not None else None
                
                loss = loss_fn(scores, auxiliary_labels_batch, mask=candidate_mask_batch)
                total_loss += loss.item()
                num_batches += 1
                
                # 计算排序指标
                metrics = compute_ranking_metrics(scores, auxiliary_labels_batch, candidate_mask_batch)
                all_metrics.append(metrics)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"平均损失: {avg_loss:.4f}")
    
    if all_metrics:
        avg_ndcg5 = np.mean([m["ndcg@5"] for m in all_metrics])
        avg_ndcg10 = np.mean([m["ndcg@10"] for m in all_metrics])
        avg_mrr = np.mean([m["mrr"] for m in all_metrics])
        avg_top1 = np.mean([m["top1_accuracy"] for m in all_metrics])
        total_samples = sum([m["num_samples"] for m in all_metrics])
        
        print(f"\n排序指标:")
        print(f"  NDCG@5:  {avg_ndcg5:.4f}")
        print(f"  NDCG@10: {avg_ndcg10:.4f}")
        print(f"  MRR:     {avg_mrr:.4f}")
        print(f"  Top-1 准确率: {avg_top1:.4f}")
        print(f"\n评估样本数: {total_samples}")
    else:
        print("⚠️ 警告: 没有有效的评估样本")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径（目录或文件）")
    parser.add_argument("--eval_data", type=str, default=None, help="评估数据路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 默认评估数据路径
    if args.eval_data is None:
        # 尝试从检查点目录推断
        checkpoint_dir = Path(args.checkpoint).parent.parent if Path(args.checkpoint).is_file() else Path(args.checkpoint).parent
        eval_data_path = checkpoint_dir.parent / "data" / "fine_tuning" / "combined" / "eval.json"
        if not eval_data_path.exists():
            eval_data_path = Path("data/fine_tuning/combined/eval.json")
    else:
        eval_data_path = args.eval_data
    
    if not Path(eval_data_path).exists():
        print(f"❌ 错误: 评估数据文件不存在: {eval_data_path}")
        print("请使用 --eval_data 指定评估数据路径")
        return
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        eval_data_path=str(eval_data_path),
        config_path=args.config,
        device=args.device
    )


if __name__ == "__main__":
    main()
