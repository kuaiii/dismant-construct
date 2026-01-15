# -*- coding: utf-8 -*-
"""
ResilienceDataset: 网络韧性优化数据集
负责加载和处理微调数据集。

核心功能：
1. 加载 JSON 格式的对话数据
2. 处理 auxiliary_labels 用于 ListMLE 训练
3. 支持多种数据格式 (LLaMA-Factory, Alpaca, etc.)
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class DataSample:
    """数据样本数据类"""
    sample_id: str
    task_type: str
    current_step: int
    total_steps: int
    system_prompt: str
    user_prompt: str
    assistant_response: str
    auxiliary_labels: Dict[str, float]
    operation_ids: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def get_sorted_operations(self) -> List[str]:
        """根据 auxiliary_labels 返回排序后的操作 ID"""
        sorted_ops = sorted(
            self.auxiliary_labels.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [op_id for op_id, _ in sorted_ops]
    
    def get_label_tensor(self, operation_order: Optional[List[str]] = None) -> torch.Tensor:
        """
        获取标签张量
        
        Args:
            operation_order: 操作 ID 顺序，None 使用 operation_ids
        
        Returns:
            标签张量 [num_operations]
        """
        order = operation_order or self.operation_ids
        labels = [self.auxiliary_labels.get(op_id, 0.0) for op_id in order]
        return torch.tensor(labels, dtype=torch.float32)


class ResilienceDataset(Dataset):
    """
    网络韧性优化数据集
    
    加载符合微调格式的 JSON 数据，支持：
    1. 对话格式 (conversations)
    2. auxiliary_labels 处理
    3. 数据增强和过滤
    
    数据格式示例：
    {
        "id": "train_dismantle_001",
        "meta": {"task": "dismantle", "budget_step": "1/10"},
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "user", "value": "..."},
            {"from": "assistant", "value": "..."}
        ],
        "auxiliary_labels": {"op_01": 0.95, "op_02": 0.15, ...}
    }
    
    Attributes:
        data_path: 数据文件/目录路径
        samples: 加载的数据样本列表
        tokenizer: 分词器 (可选)
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        tokenizer: Any = None,
        max_length: int = 2048,
        task_filter: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_tokenization: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSON 文件路径或包含 JSON 文件的目录
            tokenizer: HuggingFace 分词器
            max_length: 最大 token 长度
            task_filter: 任务类型过滤 ("dismantle", "construct", None)
            transform: 数据变换函数
            cache_tokenization: 是否缓存分词结果
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_filter = task_filter
        self.transform = transform
        self.cache_tokenization = cache_tokenization
        
        # 加载数据
        self.samples: List[DataSample] = []
        self._load_data()
        
        # 分词缓存
        self._tokenization_cache: Dict[str, Dict] = {}
    
    def _load_data(self) -> None:
        """加载数据文件"""
        paths = self._resolve_paths(self.data_path)
        
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理单个样本或样本列表
            if isinstance(data, list):
                for item in data:
                    sample = self._parse_sample(item)
                    if sample and self._filter_sample(sample):
                        self.samples.append(sample)
            else:
                sample = self._parse_sample(data)
                if sample and self._filter_sample(sample):
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {len(paths)} files")
    
    def _resolve_paths(self, data_path: Union[str, Path, List[str]]) -> List[Path]:
        """解析数据路径"""
        if isinstance(data_path, list):
            return [Path(p) for p in data_path]
        
        path = Path(data_path)
        if path.is_file():
            return [path]
        elif path.is_dir():
            return list(path.glob("*.json"))
        else:
            raise ValueError(f"Invalid data path: {data_path}")
    
    def _parse_sample(self, item: Dict) -> Optional[DataSample]:
        """解析单个数据样本"""
        try:
            # 提取对话内容
            conversations = item.get("conversations", [])
            system_prompt = ""
            user_prompt = ""
            assistant_response = ""
            
            for conv in conversations:
                role = conv.get("from", "")
                value = conv.get("value", "")
                
                if role == "system":
                    system_prompt = value
                elif role == "user":
                    user_prompt = value
                elif role == "assistant":
                    assistant_response = value
            
            # 提取 auxiliary_labels
            auxiliary_labels = item.get("auxiliary_labels", {})
            operation_ids = list(auxiliary_labels.keys())
            
            # 提取元数据
            meta = item.get("meta", {})
            task_type = meta.get("task", "unknown")
            
            # 解析步骤信息
            budget_step = meta.get("budget_step", "1/10")
            if "/" in budget_step:
                current_step, total_steps = map(int, budget_step.split("/"))
            else:
                current_step, total_steps = 1, 10
            
            return DataSample(
                sample_id=item.get("id", f"sample_{len(self.samples)}"),
                task_type=task_type,
                current_step=current_step,
                total_steps=total_steps,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                auxiliary_labels=auxiliary_labels,
                operation_ids=operation_ids,
                meta=meta
            )
        except Exception as e:
            print(f"Error parsing sample: {e}")
            return None
    
    def _filter_sample(self, sample: DataSample) -> bool:
        """过滤样本"""
        if self.task_filter is not None:
            return sample.task_type == self.task_filter
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据样本
        
        Returns:
            Dict:
                - "sample_id": 样本 ID
                - "input_text": 输入文本 (system + user)
                - "target_text": 目标文本 (assistant)
                - "auxiliary_labels": 标签张量 [num_candidates]
                - "operation_ids": 操作 ID 列表
                - "input_ids": Token IDs (如果有 tokenizer)
                - "attention_mask": 注意力掩码 (如果有 tokenizer)
                - "labels": 目标 token IDs (如果有 tokenizer)
        """
        sample = self.samples[idx]
        
        # 构建输入文本
        input_text = self._build_input_text(sample)
        target_text = sample.assistant_response
        
        result = {
            "sample_id": sample.sample_id,
            "input_text": input_text,
            "target_text": target_text,
            "auxiliary_labels": sample.get_label_tensor(),
            "operation_ids": sample.operation_ids,
            "task_type": sample.task_type,
            "meta": sample.meta
        }
        
        # 分词 (如果有 tokenizer)
        if self.tokenizer is not None:
            tokenized = self._tokenize(sample.sample_id, input_text, target_text)
            result.update(tokenized)
        
        # 应用变换
        if self.transform is not None:
            result = self.transform(result)
        
        return result
    
    def _build_input_text(self, sample: DataSample) -> str:
        """构建输入文本"""
        parts = []
        if sample.system_prompt:
            parts.append(f"[System]\n{sample.system_prompt}")
        if sample.user_prompt:
            parts.append(f"[User]\n{sample.user_prompt}")
        return "\n\n".join(parts)
    
    def _tokenize(
        self, 
        sample_id: str,
        input_text: str, 
        target_text: str
    ) -> Dict[str, torch.Tensor]:
        """分词处理"""
        # 检查缓存
        if self.cache_tokenization and sample_id in self._tokenization_cache:
            return self._tokenization_cache[sample_id]
        
        # 完整文本
        full_text = f"{input_text}\n\n[Assistant]\n{target_text}"
        
        # 分词
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }
        
        # 构建 labels (用于 Causal LM)
        # 只在 assistant 部分计算损失
        input_encodings = self.tokenizer(
            f"{input_text}\n\n[Assistant]\n",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_length = input_encodings["input_ids"].shape[1]
        
        labels = result["input_ids"].clone()
        labels[:input_length] = -100  # 忽略输入部分
        result["labels"] = labels
        
        # 缓存
        if self.cache_tokenization:
            self._tokenization_cache[sample_id] = result
        
        return result
    
    def get_sample_by_id(self, sample_id: str) -> Optional[DataSample]:
        """根据 ID 获取样本"""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        task_counts = {}
        label_stats = []
        
        for sample in self.samples:
            task_counts[sample.task_type] = task_counts.get(sample.task_type, 0) + 1
            if sample.auxiliary_labels:
                label_stats.extend(sample.auxiliary_labels.values())
        
        return {
            "total_samples": len(self.samples),
            "task_distribution": task_counts,
            "label_mean": np.mean(label_stats) if label_stats else 0,
            "label_std": np.std(label_stats) if label_stats else 0,
            "label_min": min(label_stats) if label_stats else 0,
            "label_max": max(label_stats) if label_stats else 0,
        }


class ResilienceDataCollator:
    """
    数据整理器
    
    将多个样本整理成批次，处理：
    1. 动态填充
    2. auxiliary_labels 对齐
    3. 候选掩码生成
    """
    
    def __init__(
        self,
        tokenizer: Any = None,
        max_candidates: int = 10,
        pad_to_multiple_of: Optional[int] = 8
    ):
        """
        初始化整理器
        
        Args:
            tokenizer: 分词器
            max_candidates: 最大候选数量
            pad_to_multiple_of: 填充到的倍数
        """
        self.tokenizer = tokenizer
        self.max_candidates = max_candidates
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 样本列表
        
        Returns:
            整理后的批次字典
        """
        batch_size = len(batch)
        
        # 收集字段
        sample_ids = [item["sample_id"] for item in batch]
        task_types = [item["task_type"] for item in batch]
        
        # 处理 auxiliary_labels
        max_candidates = max(len(item["operation_ids"]) for item in batch)
        max_candidates = min(max_candidates, self.max_candidates)
        
        labels_list = []
        masks_list = []
        op_ids_list = []
        
        for item in batch:
            labels = item["auxiliary_labels"]
            num_candidates = len(labels)
            
            # 填充
            if num_candidates < max_candidates:
                padding = torch.zeros(max_candidates - num_candidates)
                labels = torch.cat([labels, padding])
                mask = torch.cat([
                    torch.ones(num_candidates),
                    torch.zeros(max_candidates - num_candidates)
                ])
            else:
                labels = labels[:max_candidates]
                mask = torch.ones(max_candidates)
            
            labels_list.append(labels)
            masks_list.append(mask)
            
            # 操作 ID
            ops = item["operation_ids"][:max_candidates]
            ops = ops + ["<pad>"] * (max_candidates - len(ops))
            op_ids_list.append(ops)
        
        result = {
            "sample_ids": sample_ids,
            "task_types": task_types,
            "auxiliary_labels": torch.stack(labels_list),
            "candidate_mask": torch.stack(masks_list),
            "operation_ids": op_ids_list,
        }
        
        # 处理 tokenized 字段
        if "input_ids" in batch[0]:
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            
            result["input_ids"] = input_ids
            result["attention_mask"] = attention_mask
            
            if "labels" in batch[0]:
                result["labels"] = torch.stack([item["labels"] for item in batch])
        
        return result


# ==================== 数据生成工具 ====================

class DataGenerator:
    """
    训练数据生成器
    
    从 NetworkEnvironment 模拟中生成训练数据。
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/fine_tuning",
        num_samples_per_graph: int = 10,
        language: str = "zh"
    ):
        """
        初始化生成器
        
        Args:
            output_dir: 输出目录
            num_samples_per_graph: 每个图生成的样本数
            language: 输出语言
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples_per_graph = num_samples_per_graph
        self.language = language
    
    def generate_from_environment(
        self,
        env,  # NetworkEnvironment
        node_semantics: Optional[Dict] = None,
        sample_id_prefix: str = "train"
    ) -> List[Dict]:
        """
        从环境生成训练样本
        
        Args:
            env: NetworkEnvironment 实例
            node_semantics: 节点语义字典
            sample_id_prefix: 样本 ID 前缀
        
        Returns:
            生成的样本列表
        """
        from .ocg_builder import OCGExtractor
        from ..env.metrics import ResilienceMetrics
        
        extractor = OCGExtractor(language=self.language)
        metrics = ResilienceMetrics()
        
        samples = []
        env.reset()
        
        for step in range(env.budget):
            # 获取候选节点
            candidates = env.prune_candidates()
            if not candidates:
                break
            
            # 计算 auxiliary_labels
            impact_scores = metrics.batch_compute_impact_scores(env.graph, candidates)
            
            # 构建操作映射
            auxiliary_labels = {}
            operations = []
            for idx, node in enumerate(candidates):
                op_id = f"op_{idx+1:02d}"
                auxiliary_labels[op_id] = impact_scores[node]
                operations.append({
                    "op_id": op_id,
                    "target": node
                })
            
            # 获取正确排序
            ground_truth = sorted(
                auxiliary_labels.items(),
                key=lambda x: x[1],
                reverse=True
            )
            ground_truth_ranking = [op_id for op_id, _ in ground_truth]
            
            # 提取 OCG 并构建样本
            ocg_data = extractor.extract_ocg(
                graph=env.graph,
                candidate_nodes=candidates,
                task_type=env.task_type.value,
                current_step=step + 1,
                total_steps=env.budget,
                node_semantics=node_semantics
            )
            
            sample = extractor.build_conversation_data(
                ocg_data=ocg_data,
                ground_truth_ranking=ground_truth_ranking,
                auxiliary_labels=auxiliary_labels
            )
            sample["id"] = f"{sample_id_prefix}_{step:03d}"
            
            samples.append(sample)
            
            # 执行最佳操作
            best_op = ground_truth_ranking[0]
            best_node = operations[int(best_op.split("_")[1]) - 1]["target"]
            env.graph.remove_node(best_node)
            env.current_step += 1
        
        return samples
    
    def save_samples(
        self,
        samples: List[Dict],
        filename: str = "training_data.json"
    ) -> Path:
        """保存样本到文件"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(samples)} samples to {filepath}")
        return filepath


# ==================== 便捷函数 ====================

def create_dataloader(
    data_path: Union[str, Path],
    tokenizer: Any = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = None,  # None = 自动检测
    **kwargs
) -> DataLoader:
    """
    便捷函数：创建数据加载器
    
    Args:
        data_path: 数据路径
        tokenizer: 分词器
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数 (None=自动, Windows默认0, Linux默认4)
        **kwargs: 传递给 Dataset 的其他参数
    
    Returns:
        DataLoader 实例
    """
    import os
    
    dataset = ResilienceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        **kwargs
    )
    
    collator = ResilienceDataCollator(tokenizer=tokenizer)
    
    # 自动检测 num_workers
    if num_workers is None:
        # Windows 上多进程有问题，使用 0
        if os.name == 'nt':
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if num_workers == 0 else False  # Windows 下 pin_memory 也可能有问题
    )


def load_json_data(filepath: Union[str, Path]) -> List[Dict]:
    """便捷函数：加载 JSON 数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]
