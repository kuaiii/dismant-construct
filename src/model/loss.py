# -*- coding: utf-8 -*-
"""
排序损失函数模块
实现 ListMLE 和 ListNet 等排序学习损失函数。

核心功能：
1. ListMLELoss: 基于 Plackett-Luce 模型的列表级损失
2. ListNetLoss: 基于 KL 散度的列表级损失
3. 支持 auxiliary_labels 作为真实排序依据

参考文献：
- ListMLE: Xia et al., "Listwise Approach to Learning to Rank", ICML 2008
- ListNet: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach", ICML 2007
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListMLELoss(nn.Module):
    """
    ListMLE (Listwise Maximum Likelihood Estimation) 损失函数
    
    基于 Plackett-Luce 排序概率模型。
    
    数学原理：
    给定预测分数 s = [s_1, s_2, ..., s_n] 和真实排序 π = [π_1, π_2, ..., π_n]，
    ListMLE 损失为：
    
    L = -log P(π|s) = -Σᵢ log(exp(s_{π_i}) / Σⱼ≥ᵢ exp(s_{π_j}))
    
    其中 π_i 表示真实排序中第 i 位的元素。
    
    特点：
    - 列表级损失，直接优化排序
    - 基于似然估计，理论基础扎实
    - 对真实排序敏感，适合有明确排序标签的场景
    
    与 auxiliary_labels 的关系：
    auxiliary_labels 提供每个操作的真实影响分数，用于生成真实排序 π。
    例如：{"op_01": 0.95, "op_02": 0.15, "op_03": 0.40}
    → 真实排序 π = [0, 2, 1] (op_01 > op_03 > op_02)
    
    Attributes:
        eps: 数值稳定性常数
        temperature: 温度参数，控制分布平滑度
    """
    
    def __init__(
        self, 
        eps: float = 1e-10, 
        temperature: float = 1.0,
        reduction: str = "mean"
    ):
        """
        初始化 ListMLE 损失
        
        Args:
            eps: 防止 log(0) 的小常数
            temperature: 温度参数，>1 使分布更平滑
            reduction: 损失归约方式 ("mean", "sum", "none")
        """
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        scores: torch.Tensor,
        auxiliary_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 ListMLE 损失
        
        Args:
            scores: 模型预测分数 [batch_size, num_candidates]
            auxiliary_labels: 真实影响分数 [batch_size, num_candidates]
                             分数越高表示优先级越高
            mask: 有效候选掩码 [batch_size, num_candidates]
                  1 表示有效，0 表示填充
        
        Returns:
            loss: 标量损失值
        
        Example:
            >>> loss_fn = ListMLELoss()
            >>> scores = torch.tensor([[0.5, 0.3, 0.8]])  # 模型预测
            >>> labels = torch.tensor([[0.95, 0.15, 0.40]])  # auxiliary_labels
            >>> loss = loss_fn(scores, labels)
        """
        batch_size, num_candidates = scores.shape
        device = scores.device
        
        # 确保输入为 float 类型，避免 Half 精度问题
        scores = scores.float()
        auxiliary_labels = auxiliary_labels.float()
        
        # 检查输入是否包含 NaN 或 Inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # 用 0 替换异常值
            scores = torch.where(torch.isnan(scores) | torch.isinf(scores), 
                                torch.zeros_like(scores), scores)
        
        if torch.isnan(auxiliary_labels).any() or torch.isinf(auxiliary_labels).any():
            auxiliary_labels = torch.where(torch.isnan(auxiliary_labels) | torch.isinf(auxiliary_labels),
                                          torch.zeros_like(auxiliary_labels), auxiliary_labels)
        
        # 为 auxiliary_labels 添加微小扰动，避免完全相同的值导致排序不稳定
        # 这对于所有 label 都相同的情况特别重要
        noise = torch.randn_like(auxiliary_labels) * 1e-8
        auxiliary_labels = auxiliary_labels + noise
        
        # 1. 根据 auxiliary_labels 获取真实排序
        # 按分数降序排序，得到排序索引
        true_ranking = torch.argsort(auxiliary_labels, dim=1, descending=True).long()  # 确保是 long 类型
        # true_ranking[i, j] = 第 i 个样本中排第 j 位的候选索引
        
        # 2. 应用温度缩放
        scores = scores / self.temperature
        
        # 限制 scores 范围，防止 exp 溢出
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        
        # 3. 按真实排序重排预测分数
        # 将 scores 按 true_ranking 的顺序排列
        sorted_scores = torch.gather(scores, dim=1, index=true_ranking)
        # sorted_scores[i, j] = 第 i 个样本中真实排名第 j 的预测分数
        
        # 4. 计算 ListMLE 损失
        # L = -Σᵢ log(exp(s_{π_i}) / Σⱼ≥ᵢ exp(s_{π_j}))
        # 等价于: L = -Σᵢ (s_{π_i} - log(Σⱼ≥ᵢ exp(s_{π_j})))
        
        # 计算从后往前的 cumsum(exp(s))，用于计算分母
        # 先翻转，累加，再翻转回来
        max_scores = sorted_scores.max(dim=1, keepdim=True)[0]  # 数值稳定性
        exp_scores = torch.exp(sorted_scores - max_scores)
        
        # 从后往前累加: cumsum_from_back[i] = Σⱼ≥ᵢ exp(s_j)
        exp_scores_flipped = torch.flip(exp_scores, dims=[1])
        cumsum_flipped = torch.cumsum(exp_scores_flipped, dim=1)
        cumsum_from_back = torch.flip(cumsum_flipped, dims=[1])
        
        # log(Σⱼ≥ᵢ exp(s_{π_j})) = log(cumsum_from_back) + max_scores
        log_cumsum = torch.log(cumsum_from_back + self.eps) + max_scores
        
        # ListMLE = -Σᵢ (s_{π_i} - log_cumsum_i)
        per_position_loss = log_cumsum - sorted_scores
        
        # 5. 应用掩码 (如果有)
        if mask is not None:
            # 确保 mask 是 float 类型用于数学运算
            mask = mask.float()
            # 按真实排序重排掩码（true_ranking 已经是 long 类型）
            sorted_mask = torch.gather(mask, dim=1, index=true_ranking)
            per_position_loss = per_position_loss * sorted_mask
            # 计算有效长度
            valid_counts = sorted_mask.sum(dim=1).clamp(min=1)
            loss_per_sample = per_position_loss.sum(dim=1) / valid_counts
        else:
            loss_per_sample = per_position_loss.mean(dim=1)
        
        # 6. 最终 NaN 检查，如果仍有 NaN 则返回 0
        if torch.isnan(loss_per_sample).any():
            loss_per_sample = torch.where(torch.isnan(loss_per_sample),
                                         torch.zeros_like(loss_per_sample),
                                         loss_per_sample)
        
        # 7. 归约
        if self.reduction == "mean":
            result = loss_per_sample.mean()
        elif self.reduction == "sum":
            result = loss_per_sample.sum()
        else:  # "none"
            result = loss_per_sample
        
        # 最终安全检查
        if isinstance(result, torch.Tensor) and result.numel() == 1:
            if torch.isnan(result) or torch.isinf(result):
                return torch.tensor(0.0, device=device, dtype=result.dtype, requires_grad=True)
        
        return result
    
    def compute_from_dict_labels(
        self,
        scores: torch.Tensor,
        auxiliary_labels_dict: Dict[str, float],
        candidate_order: List[str]
    ) -> torch.Tensor:
        """
        便捷方法：从字典格式的 auxiliary_labels 计算损失
        
        Args:
            scores: 模型预测分数 [1, num_candidates] 或 [num_candidates]
            auxiliary_labels_dict: {"op_01": 0.95, "op_02": 0.15, ...}
            candidate_order: 候选操作顺序 ["op_01", "op_02", ...]
        
        Returns:
            loss: 损失值
        """
        # 转换为张量
        labels = [auxiliary_labels_dict[op_id] for op_id in candidate_order]
        labels_tensor = torch.tensor(labels, device=scores.device).unsqueeze(0)
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        return self.forward(scores, labels_tensor)


class ListNetLoss(nn.Module):
    """
    ListNet 损失函数
    
    基于 KL 散度比较预测分布和真实分布。
    
    数学原理：
    L = KL(P_true || P_pred) = Σᵢ P_true(i) * log(P_true(i) / P_pred(i))
    
    其中：
    - P_pred(i) = exp(s_i) / Σⱼ exp(s_j)  (Softmax)
    - P_true(i) = exp(y_i) / Σⱼ exp(y_j)  (基于 auxiliary_labels)
    
    特点：
    - 比 ListMLE 更平滑，因为使用概率分布而非硬排序
    - 对分数差异敏感，而非仅关注排序
    
    Attributes:
        eps: 数值稳定性常数
        temperature: 温度参数
    """
    
    def __init__(
        self,
        eps: float = 1e-10,
        temperature: float = 1.0,
        label_temperature: float = 1.0,
        reduction: str = "mean"
    ):
        """
        初始化 ListNet 损失
        
        Args:
            eps: 数值稳定性常数
            temperature: 预测分数的温度
            label_temperature: 标签分数的温度
            reduction: 归约方式
        """
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.label_temperature = label_temperature
        self.reduction = reduction
    
    def forward(
        self,
        scores: torch.Tensor,
        auxiliary_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 ListNet 损失 (KL 散度)
        
        Args:
            scores: 模型预测分数 [batch_size, num_candidates]
            auxiliary_labels: 真实影响分数 [batch_size, num_candidates]
            mask: 有效候选掩码
        
        Returns:
            loss: KL 散度损失
        """
        # 确保输入为 float 类型，避免 Half 精度问题
        scores = scores.float()
        auxiliary_labels = auxiliary_labels.float()
        
        # 1. 计算预测分布 (Softmax)
        pred_probs = F.softmax(scores / self.temperature, dim=-1)
        
        # 2. 计算真实分布 (基于 auxiliary_labels)
        true_probs = F.softmax(auxiliary_labels / self.label_temperature, dim=-1)
        
        # 3. 应用掩码
        if mask is not None:
            # 确保 mask 是 float 类型
            mask = mask.float()
            # 将无效位置的概率设为 0
            pred_probs = pred_probs * mask
            true_probs = true_probs * mask
            # 重新归一化
            pred_probs = pred_probs / (pred_probs.sum(dim=-1, keepdim=True) + self.eps)
            true_probs = true_probs / (true_probs.sum(dim=-1, keepdim=True) + self.eps)
        
        # 4. 计算 KL 散度
        # KL(P || Q) = Σ P * log(P / Q) = Σ P * (log P - log Q)
        log_pred = torch.log(pred_probs + self.eps)
        log_true = torch.log(true_probs + self.eps)
        
        kl_div = true_probs * (log_true - log_pred)
        loss_per_sample = kl_div.sum(dim=-1)
        
        # 5. 归约
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class CombinedRankingLoss(nn.Module):
    """
    组合排序损失
    
    结合 ListMLE 和其他损失函数，支持多目标优化。
    
    L_total = α * L_ListMLE + β * L_ListNet + γ * L_margin
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        margin: float = 1.0,
        temperature: float = 1.0
    ):
        """
        初始化组合损失
        
        Args:
            alpha: ListMLE 权重
            beta: ListNet 权重
            gamma: Margin Loss 权重
            margin: 边际损失的边距
            temperature: 温度参数
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.listmle = ListMLELoss(temperature=temperature)
        self.listnet = ListNetLoss(temperature=temperature) if beta > 0 else None
        self.margin = margin
    
    def forward(
        self,
        scores: torch.Tensor,
        auxiliary_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Returns:
            Dict:
                - "total": 总损失
                - "listmle": ListMLE 损失
                - "listnet": ListNet 损失 (如果启用)
                - "margin": Margin 损失 (如果启用)
        """
        losses = {}
        
        # ListMLE
        if self.alpha > 0:
            losses["listmle"] = self.listmle(scores, auxiliary_labels, mask)
        
        # ListNet
        if self.beta > 0 and self.listnet is not None:
            losses["listnet"] = self.listnet(scores, auxiliary_labels, mask)
        
        # Margin Loss (Pairwise)
        if self.gamma > 0:
            losses["margin"] = self._compute_margin_loss(scores, auxiliary_labels, mask)
        
        # 总损失
        total = 0.0
        if "listmle" in losses:
            total = total + self.alpha * losses["listmle"]
        if "listnet" in losses:
            total = total + self.beta * losses["listnet"]
        if "margin" in losses:
            total = total + self.gamma * losses["margin"]
        
        losses["total"] = total
        
        return losses
    
    def _compute_margin_loss(
        self,
        scores: torch.Tensor,
        auxiliary_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算 Pairwise Margin Loss"""
        # 确保输入为 float 类型
        scores = scores.float()
        auxiliary_labels = auxiliary_labels.float()
        
        batch_size, num_candidates = scores.shape
        
        # 扩展维度进行成对比较
        scores_i = scores.unsqueeze(2)  # [B, N, 1]
        scores_j = scores.unsqueeze(1)  # [B, 1, N]
        labels_i = auxiliary_labels.unsqueeze(2)
        labels_j = auxiliary_labels.unsqueeze(1)
        
        # 计算标签差异的符号 (谁更大)
        label_diff = torch.sign(labels_i - labels_j)  # [B, N, N]
        
        # Margin loss: max(0, margin - sign(y_i - y_j) * (s_i - s_j))
        score_diff = scores_i - scores_j
        margin_loss = F.relu(self.margin - label_diff * score_diff)
        
        # 只考虑上三角 (避免重复)
        triu_mask = torch.triu(torch.ones(num_candidates, num_candidates, device=scores.device, dtype=scores.dtype), diagonal=1)
        margin_loss = margin_loss * triu_mask.unsqueeze(0)
        
        # 应用掩码
        if mask is not None:
            # 确保 mask 是 float 类型
            mask = mask.float()
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) * triu_mask.unsqueeze(0)
            margin_loss = margin_loss * pair_mask
            valid_pairs = pair_mask.sum(dim=[1, 2]).clamp(min=1)
            return (margin_loss.sum(dim=[1, 2]) / valid_pairs).mean()
        
        num_pairs = num_candidates * (num_candidates - 1) / 2
        return margin_loss.sum(dim=[1, 2]).mean() / max(num_pairs, 1)


# ==================== 评估指标 ====================

class RankingMetrics:
    """
    排序评估指标
    
    包含 NDCG、MRR、Precision@K 等指标。
    """
    
    @staticmethod
    def ndcg(
        scores: torch.Tensor,
        labels: torch.Tensor,
        k: Optional[int] = None
    ) -> float:
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            scores: 预测分数 [num_candidates]
            labels: 真实分数 [num_candidates]
            k: Top-K，None 表示全部
        
        Returns:
            NDCG 分数
        """
        # 确保在 CPU 上进行计算，并转换为 float
        scores = scores.detach().cpu().float()
        labels = labels.detach().cpu().float()
        
        n = scores.shape[0]
        k = k or n
        k = min(k, n)  # 确保 k 不超过样本数
        
        # 按预测分数排序（确保索引是 long 类型）
        pred_ranking = torch.argsort(scores, descending=True)[:k].long()
        
        # 按真实分数排序 (理想排序)
        ideal_ranking = torch.argsort(labels, descending=True)[:k].long()
        
        # DCG
        gains = labels[pred_ranking]
        discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float))
        dcg = (gains / discounts).sum()
        
        # Ideal DCG
        ideal_gains = labels[ideal_ranking]
        idcg = (ideal_gains / discounts).sum()
        
        return (dcg / (idcg + 1e-10)).item()
    
    @staticmethod
    def mrr(
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        
        Args:
            scores: 预测分数
            labels: 真实分数
        
        Returns:
            MRR 分数
        """
        # 确保在 CPU 上进行计算，并转换为 float
        scores = scores.detach().cpu().float()
        labels = labels.detach().cpu().float()
        
        # 找到真实最佳候选
        best_idx = torch.argmax(labels).long()
        
        # 在预测排序中的位置（确保索引是 long 类型）
        pred_ranking = torch.argsort(scores, descending=True).long()
        rank = (pred_ranking == best_idx).nonzero(as_tuple=True)[0].item() + 1
        
        return 1.0 / rank
    
    @staticmethod
    def precision_at_k(
        scores: torch.Tensor,
        labels: torch.Tensor,
        k: int = 1
    ) -> float:
        """
        计算 Precision@K
        
        Args:
            scores: 预测分数
            labels: 真实分数
            k: Top-K
        
        Returns:
            Precision@K
        """
        # 确保在 CPU 上进行计算，并转换为 float
        scores = scores.detach().cpu().float()
        labels = labels.detach().cpu().float()
        
        # 确保 k 不超过样本数
        k = min(k, scores.shape[0])
        
        pred_top_k = set(torch.argsort(scores, descending=True)[:k].long().numpy())
        true_top_k = set(torch.argsort(labels, descending=True)[:k].long().numpy())
        
        return len(pred_top_k & true_top_k) / k
    
    @staticmethod
    def kendall_tau(
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        计算 Kendall's Tau 相关系数
        
        Args:
            scores: 预测分数
            labels: 真实分数
        
        Returns:
            Kendall's Tau [-1, 1]
        """
        from scipy.stats import kendalltau
        
        # 确保在 CPU 上进行计算，并转换为 float
        scores_np = scores.detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().float().numpy()
        
        tau, _ = kendalltau(scores_np, labels_np)
        return tau if tau == tau else 0.0  # 处理 NaN 的情况


# ==================== 便捷函数 ====================

def create_ranking_loss(
    loss_type: str = "listmle",
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建排序损失函数
    
    Args:
        loss_type: 损失类型 ("listmle", "listnet", "combined")
        **kwargs: 损失函数参数
    
    Returns:
        损失函数模块
    """
    if loss_type == "listmle":
        return ListMLELoss(**kwargs)
    elif loss_type == "listnet":
        return ListNetLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedRankingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
