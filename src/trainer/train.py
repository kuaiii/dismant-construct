# -*- coding: utf-8 -*-
"""
ResilienceTrainer: 模型训练器
负责完整的训练流程，包括 LoRA 微调和 ListMLE 排序学习。

核心功能：
1. 两阶段训练支持 (Phase 1: LLM only, Phase 2: Joint)
2. ListMLE 排序损失训练
3. 训练状态管理和检查点
4. 评估指标计算
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    output_dir: str = "outputs"
    experiment_name: str = "resilience_llm"
    seed: int = 42
    
    # 训练超参数
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 排序损失配置
    ranking_loss_type: str = "listmle"  # "listmle", "listnet", "combined"
    ranking_loss_weight: float = 1.0
    lm_loss_weight: float = 0.5
    
    # 训练阶段
    phase: int = 1  # 1: LLM only, 2: Joint
    freeze_llm_in_phase2: bool = False
    
    # 评估配置
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # 设备配置
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False
    
    # 其他
    resume_from_checkpoint: Optional[str] = None
    max_samples: Optional[int] = None  # 用于调试


@dataclass
class TrainingState:
    """训练状态"""
    global_step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    train_loss_history: List[float] = field(default_factory=list)
    eval_metrics_history: List[Dict] = field(default_factory=list)


class ResilienceTrainer:
    """
    网络韧性优化模型训练器
    
    支持两阶段训练：
    - Phase 1: 仅训练 LLM (LoRA 参数)，使用标准 LM 损失 + ListMLE
    - Phase 2: 联合训练 LLM + GNN + Fusion，使用 ListMLE 排序损失
    
    训练流程：
    1. 加载数据和模型
    2. 配置优化器和调度器
    3. 训练循环
    4. 评估和保存检查点
    
    Attributes:
        config: 训练配置
        model: 模型实例
        train_dataloader: 训练数据加载器
        eval_dataloader: 评估数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        state: 训练状态
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练模型
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        
        self.state = TrainingState()
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        # 设置输出目录
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._setup_training()
    
    def _setup_logging(self) -> None:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_training(self) -> None:
        """设置训练组件"""
        # 设置设备
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # 设置损失函数
        from ..model.loss import create_ranking_loss
        self.loss_fn = create_ranking_loss(
            loss_type=self.config.ranking_loss_type
        )
        
        # 设置优化器
        self._setup_optimizer()
        
        # 设置学习率调度器
        self._setup_scheduler()
        
        # 混合精度训练配置
        # 注意：由于自定义排序损失和 scoring_head 的复杂性，AMP + GradScaler 组合
        # 容易产生 FP16 梯度问题，因此默认禁用 GradScaler，仅使用 autocast 加速计算
        self.scaler = None
        self.use_amp = False  # 是否使用 autocast
        self.use_scaler = False  # 是否使用 GradScaler
        
        if self.config.fp16 or self.config.bf16:
            # 检查模型参数的数据类型
            param_dtypes = set()
            for name, param in self.model.named_parameters():
                param_dtypes.add(param.dtype)
            
            print(f"📊 模型参数类型: {param_dtypes}")
            
            # 无论模型是什么精度，都只使用 autocast 而不使用 GradScaler
            # 这样可以获得计算加速，同时避免梯度精度问题
            print("✅  启用 autocast 混合精度计算（不使用 GradScaler 以避免梯度问题）")
            self.use_amp = True
            self.use_scaler = False
            self.scaler = None
        else:
            print("ℹ️  使用 FP32 全精度训练")
    
    def _setup_optimizer(self) -> None:
        """设置优化器"""
        # 收集需要优化的参数
        if self.config.phase == 1:
            # Phase 1: 仅优化 LoRA 参数
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Phase 2: 优化所有参数或冻结 LLM
            if self.config.freeze_llm_in_phase2:
                # 冻结 LLM，只优化 GNN 和 Fusion
                params = []
                for name, param in self.model.named_parameters():
                    if "llm" not in name.lower() or "lora" in name.lower():
                        params.append(param)
            else:
                params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 检查是否有可训练参数
        if len(params) == 0:
            error_msg = (
                "No trainable parameters found!\n"
                "This usually means the model has not been initialized properly.\n"
                "Please ensure:\n"
                "1. model.initialize(device) has been called\n"
                "2. _load_llm() and _apply_lora() methods are implemented\n"
                "3. Check src/model/fusion_llm.py and docs/training_setup_guide.md"
            )
            raise ValueError(error_msg)
        
        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _setup_scheduler(self) -> None:
        """设置学习率调度器"""
        num_training_steps = (
            len(self.train_dataloader) * self.config.num_epochs 
            // self.config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        # Warmup + Cosine Annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=1e-7
        )
        
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps]
        )
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            训练结果字典
        """
        self.logger.info(f"Starting training - Phase {self.config.phase}")
        self.logger.info(f"Config: {self.config}")
        
        # 恢复检查点
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # 训练循环
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch()
            
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Loss: {epoch_loss:.4f}")
            
            # 评估
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self.state.eval_metrics_history.append(eval_metrics)
                
                # 保存最佳模型
                if eval_metrics.get("ndcg", 0) > self.state.best_metric:
                    self.state.best_metric = eval_metrics["ndcg"]
                    self._save_checkpoint("best")
            
            # 保存检查点
            self._save_checkpoint(f"epoch_{epoch + 1}")
        
        self.logger.info("Training completed!")
        return {
            "final_loss": self.state.train_loss_history[-1] if self.state.train_loss_history else 0,
            "best_metric": self.state.best_metric,
            "total_steps": self.state.global_step
        }
    
    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.state.epoch + 1}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss = self._training_step(batch)
            
            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
            
            # 确保损失是 FP32，避免 FP16 梯度问题
            if loss.dtype != torch.float32:
                loss = loss.float()
            
            # NaN/Inf 检查：跳过无效的批次
            loss_value = loss.item()
            if not (loss_value == loss_value) or loss_value == float('inf') or loss_value == float('-inf'):
                # loss_value != loss_value 是检测 NaN 的技巧
                self.logger.warning(f"跳过批次 {batch_idx}：损失为 NaN 或 Inf")
                self.optimizer.zero_grad()  # 清除可能的无效梯度
                continue
            
            # 反向传播
            if self.use_scaler and self.scaler is not None:
                # FP32 模型 + AMP + GradScaler：使用 scaled backward
                self.scaler.scale(loss).backward()
            else:
                # 无 GradScaler：直接反向传播
                loss.backward()
            
            total_loss += loss_value * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # 梯度更新
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_scaler and self.scaler is not None:
                    # FP32 + AMP 的梯度更新流程
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 原生 FP16 或纯 FP32：直接更新
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
                
                # 日志
                if self.state.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })
                
                # 中间评估
                if (self.eval_dataloader is not None and 
                    self.state.global_step % self.config.eval_steps == 0):
                    eval_metrics = self.evaluate()
                    self.model.train()
                
                # 保存检查点
                if self.state.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.state.global_step}")
        
        epoch_loss = total_loss / max(num_batches, 1)
        self.state.train_loss_history.append(epoch_loss)
        
        return epoch_loss
    
    def _training_step(self, batch: Dict) -> torch.Tensor:
        """
        单步训练
        
        Args:
            batch: 批次数据
        
        Returns:
            loss: 损失值
        """
        # 移动数据到设备
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        auxiliary_labels = batch["auxiliary_labels"].to(self.device)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(self.device)
        
        # 混合精度（只在 use_amp=True 时启用）
        with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.use_amp):
            # 模型前向传播
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # 获取候选操作位置索引（从 prompt 中提取或使用简化方法）
                candidate_indices = self._extract_candidate_indices(batch, input_ids)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_indices=candidate_indices,
                    return_scores=True
                )
                
                # LM 损失
                if labels is not None and self.config.lm_loss_weight > 0:
                    labels = labels.to(self.device)
                    lm_loss = self._compute_lm_loss(outputs["logits"], labels)
                else:
                    lm_loss = 0.0
                
                # 排序损失
                if "scores" in outputs and outputs["scores"] is not None:
                    ranking_loss = self.loss_fn(
                        outputs["scores"],
                        auxiliary_labels,
                        mask=candidate_mask
                    )
                else:
                    # 如果没有 scores，使用 hidden states 计算
                    hidden_states = outputs.get("hidden_states")
                    ranking_loss = self._compute_ranking_loss_from_hidden_states(
                        hidden_states,
                        outputs["logits"],
                        auxiliary_labels,
                        candidate_mask,
                        attention_mask
                    )
            else:
                # 如果没有 input_ids，说明数据加载有问题
                raise ValueError(
                    "input_ids 为空！请确保数据加载器传入了 tokenizer。"
                    "检查 scripts/train.py 中 create_dataloader 是否传入了 tokenizer 参数。"
                )
            
            # 总损失
            total_loss = (
                self.config.lm_loss_weight * lm_loss +
                self.config.ranking_loss_weight * ranking_loss
            )
            
            # NaN 检查和处理
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                # 记录警告但不中断训练
                self.logger.warning(f"检测到 NaN/Inf 损失，跳过此批次 (lm_loss={lm_loss}, ranking_loss={ranking_loss})")
                # 返回一个小的有效损失值，避免梯度更新出问题
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss
    
    def _compute_lm_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算语言模型损失"""
        # 移位 logits 和 labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 检查是否有有效的 labels（非 -100）
        valid_mask = (shift_labels != -100)
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            # 没有有效的 labels，返回 0 损失
            self.logger.debug("LM 损失计算：没有有效的 labels，跳过")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 检查 logits 是否包含 NaN/Inf
        if torch.isnan(shift_logits).any() or torch.isinf(shift_logits).any():
            self.logger.warning("LM 损失计算：logits 包含 NaN/Inf")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 最终 NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"LM 损失计算结果为 NaN/Inf (有效样本数: {num_valid})")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        return loss
    
    def _extract_candidate_indices(
        self, 
        batch: Dict, 
        input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        从 batch 中提取候选操作在序列中的位置索引
        
        简化实现：使用序列末尾的 token 位置作为候选操作的表示位置
        更精确的实现需要从 prompt 中解析操作描述的位置
        
        Args:
            batch: 批次数据
            input_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            candidate_indices: [batch_size, num_candidates] 或 None
        """
        batch_size, seq_len = input_ids.shape
        
        # 从 batch 中获取候选数量
        auxiliary_labels = batch.get("auxiliary_labels")
        if auxiliary_labels is None:
            return None
        
        num_candidates = auxiliary_labels.shape[1] if isinstance(auxiliary_labels, torch.Tensor) else 0
        
        if num_candidates == 0:
            return None
        
        # 简化方法：使用序列末尾的 token 位置
        # 实际应用中，应该从 prompt 中解析操作描述的位置
        # 这里使用最后一个有效 token 的位置（考虑 padding）
        candidate_indices = []
        attention_mask = batch.get("attention_mask", None)
        
        for i in range(batch_size):
            if attention_mask is not None:
                # 使用 attention_mask 找到最后一个有效位置
                valid_length = attention_mask[i].sum().item()
                # 使用最后几个位置作为候选操作的表示
                # 简化：每个候选操作使用一个位置
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
        
        return torch.tensor(candidate_indices, device=input_ids.device, dtype=torch.long)
    
    def _compute_ranking_loss_from_hidden_states(
        self,
        hidden_states: Optional[torch.Tensor],
        logits: torch.Tensor,
        auxiliary_labels: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        从 hidden states 或 logits 计算排序损失（当模型没有直接输出 scores 时）
        
        简化方法：使用序列的平均池化表示来计算分数
        """
        batch_size, seq_len = logits.shape[:2]
        num_candidates = auxiliary_labels.shape[1]
        
        # 如果有 hidden_states，使用它；否则使用 logits
        if hidden_states is not None:
            # 使用 hidden states 的平均池化
            if attention_mask is not None:
                # 加权平均（只考虑有效 token）
                # 确保 attention_mask 的数据类型与 hidden_states 一致
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [batch_size, seq_len, 1]
                masked_hidden = hidden_states * mask_expanded
                # 确保除法操作的数据类型一致
                sum_mask = attention_mask.sum(dim=1, keepdim=True).to(hidden_states.dtype).unsqueeze(-1)
                sum_mask = sum_mask.clamp(min=1.0)  # 防止除以0
                pooled = masked_hidden.sum(dim=1) / sum_mask.squeeze(-1)
            else:
                pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            
            # 投影到统一维度
            pooled = self.model._project_to_d_model(pooled.unsqueeze(1))  # [batch_size, 1, d_model]
            pooled = pooled.squeeze(1)  # [batch_size, d_model]
        else:
            # 使用 logits 的最后一个位置（简化方法）
            if attention_mask is not None:
                valid_lengths = attention_mask.sum(dim=1).long()  # [batch_size], 确保是 long 类型
                last_logits = []
                for i in range(batch_size):
                    last_pos = max(0, valid_lengths[i].item() - 1)
                    last_logits.append(logits[i, last_pos, :])
                pooled = torch.stack(last_logits)  # [batch_size, vocab_size]
            else:
                pooled = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 投影到统一维度
            if pooled.shape[1] != self.model.config.d_model:
                pooled = self.model._project_to_d_model(pooled.unsqueeze(1))
                pooled = pooled.squeeze(1)
        
        # 扩展到候选数量
        pooled = pooled.unsqueeze(1).expand(-1, num_candidates, -1)  # [batch_size, num_candidates, d_model]
        
        # 确保 scoring_head 与输入数据类型一致
        if self.model.scoring_head[0].weight.dtype != pooled.dtype:
            self.model.scoring_head = self.model.scoring_head.to(pooled.dtype)
        if self.model.scoring_head[0].weight.device != pooled.device:
            self.model.scoring_head = self.model.scoring_head.to(pooled.device)
        
        # 通过 scoring_head 计算分数
        scores = self.model.scoring_head(pooled).squeeze(-1)  # [batch_size, num_candidates]
        
        # 计算排序损失
        ranking_loss = self.loss_fn(
            scores,
            auxiliary_labels,
            mask=candidate_mask
        )
        
        return ranking_loss
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                auxiliary_labels = batch["auxiliary_labels"].to(self.device).float()
                candidate_mask = batch["candidate_mask"].to(self.device).float()
                
                # 获取模型预测
                input_ids = batch.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    # 提取候选索引
                    candidate_indices = self._extract_candidate_indices(batch, input_ids)
                    
                    # 禁用混合精度以避免数据类型问题
                    with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=False):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            candidate_indices=candidate_indices,
                            return_scores=True
                        )
                    
                    if "scores" in outputs and outputs["scores"] is not None:
                        scores = outputs["scores"]
                    else:
                        continue
                else:
                    continue
                
                # 计算损失
                loss = self.loss_fn(scores, auxiliary_labels, mask=candidate_mask)
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签（确保转换为 float 再移到 CPU）
                all_scores.append(scores.detach().cpu().float())
                all_labels.append(auxiliary_labels.detach().cpu().float())
        
        # 计算指标
        from ..model.loss import RankingMetrics
        
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "ndcg": 0.0,
            "mrr": 0.0,
            "precision_at_1": 0.0
        }
        
        # 计算每个样本的指标
        ndcg_list = []
        mrr_list = []
        p1_list = []
        
        for i in range(all_scores.shape[0]):
            ndcg_list.append(RankingMetrics.ndcg(all_scores[i], all_labels[i]))
            mrr_list.append(RankingMetrics.mrr(all_scores[i], all_labels[i]))
            p1_list.append(RankingMetrics.precision_at_k(all_scores[i], all_labels[i], k=1))
        
        metrics["ndcg"] = sum(ndcg_list) / len(ndcg_list)
        metrics["mrr"] = sum(mrr_list) / len(mrr_list)
        metrics["precision_at_1"] = sum(p1_list) / len(p1_list)
        
        self.logger.info(f"Evaluation - Loss: {metrics['loss']:.4f}, "
                        f"NDCG: {metrics['ndcg']:.4f}, "
                        f"MRR: {metrics['mrr']:.4f}, "
                        f"P@1: {metrics['precision_at_1']:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, name: str) -> None:
        """保存检查点"""
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # 保存优化器状态
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None
        }, checkpoint_dir / "optimizer.pt")
        
        # 保存训练状态
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "train_loss_history": self.state.train_loss_history,
            "eval_metrics_history": self.state.eval_metrics_history
        }
        with open(checkpoint_dir / "state.json", 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_path)
        
        # 加载模型
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 加载优化器状态
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            opt_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(opt_state["optimizer"])
            self.scheduler.load_state_dict(opt_state["scheduler"])
            if self.scaler and opt_state.get("scaler"):
                self.scaler.load_state_dict(opt_state["scaler"])
        
        # 加载训练状态
        state_path = checkpoint_dir / "state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state_dict = json.load(f)
            self.state.global_step = state_dict["global_step"]
            self.state.epoch = state_dict["epoch"]
            self.state.best_metric = state_dict["best_metric"]
            self.state.train_loss_history = state_dict["train_loss_history"]
            self.state.eval_metrics_history = state_dict["eval_metrics_history"]
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_dir}")


# ==================== 便捷函数 ====================

def train_resilience_model(
    model: nn.Module,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：训练韧性优化模型
    
    Args:
        model: 模型实例
        train_data_path: 训练数据路径
        eval_data_path: 评估数据路径
        output_dir: 输出目录
        **kwargs: 训练配置参数
    
    Returns:
        训练结果
    """
    from ..data.dataset import create_dataloader
    
    # 创建数据加载器
    train_loader = create_dataloader(train_data_path, batch_size=kwargs.get("batch_size", 4))
    eval_loader = None
    if eval_data_path:
        eval_loader = create_dataloader(eval_data_path, batch_size=kwargs.get("batch_size", 4), shuffle=False)
    
    # 创建配置
    config = TrainingConfig(output_dir=output_dir, **kwargs)
    
    # 创建训练器
    trainer = ResilienceTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=config
    )
    
    # 训练
    return trainer.train()
