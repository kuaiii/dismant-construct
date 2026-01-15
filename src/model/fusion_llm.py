# -*- coding: utf-8 -*-
"""
ResilienceLLM: 网络韧性优化大语言模型
结合 LLM 的语义理解能力和可选的 GNN 结构编码能力。

核心组件：
1. ResilienceLLM: 主模型类，支持 LoRA 微调
2. GeometricEncoder: 图结构编码器 (可选)
3. FusionModule: 语义-结构特征融合模块
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置常量
D_GEO_RAW = 512     # GNN 原始输出维度
D_LLM_RAW = 4096    # Llama-3 原始输出维度
D_MODEL = 1024      # 统一投影后的对齐维度


@dataclass
class ModelConfig:
    """模型配置数据类"""
    # LLM 配置
    llm_model_name: str = "meta-llama/Meta-Llama-3-8B"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None  # None 表示使用默认值
    
    # 几何编码器配置
    use_geometric_encoder: bool = False
    geo_input_dim: int = 64
    geo_hidden_dim: int = D_GEO_RAW
    geo_num_layers: int = 3
    geo_encoder_type: str = "gin"  # "gin", "gat", "transformer"
    
    # 融合配置
    d_model: int = D_MODEL
    fusion_type: str = "gated"  # "gated", "attention", "concat"
    num_attention_heads: int = 8
    
    # 输出配置
    num_candidates: int = 10  # 最大候选数
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class GeometricEncoder(nn.Module):
    """
    图结构编码器
    
    使用 GNN (GIN/GAT/GraphTransformer) 对图结构进行编码，
    提取节点的拓扑特征表示。
    
    这是一个可选模块，当 use_geometric_encoder=True 时启用。
    
    Attributes:
        encoder_type: GNN 类型
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_layers: GNN 层数
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = D_GEO_RAW,
        output_dim: int = D_MODEL,
        num_layers: int = 3,
        encoder_type: str = "gin"
    ):
        """
        初始化几何编码器
        
        Args:
            input_dim: 输入节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (对齐到 d_model)
            num_layers: GNN 层数
            encoder_type: 编码器类型 ("gin", "gat", "transformer")
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 构建 GNN 层
        self.encoder = self._build_encoder()
        
        # 投影层：将 GNN 维度映射到统一维度 d_model
        self.projector = nn.Linear(hidden_dim, output_dim)
        
        # LayerNorm
        self.norm = nn.LayerNorm(output_dim)
    
    def _build_encoder(self) -> nn.Module:
        """
        构建 GNN 编码器
        
        Returns:
            GNN 模块
        
        Note:
            实际实现需要 torch_geometric
        """
        # TODO: 实现具体的 GNN 编码器
        # 这里返回占位符，实际使用时需要 PyG
        return nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes] (用于批处理多个图)
        
        Returns:
            H_geo: 几何嵌入 [num_nodes, d_model]
        """
        # 1. GNN 编码
        raw_geo_emb = self.encoder(x)  # [num_nodes, hidden_dim]
        
        # 2. 投影对齐
        H_geo = self.projector(raw_geo_emb)  # [num_nodes, d_model]
        
        # 3. LayerNorm
        H_geo = self.norm(H_geo)
        
        return H_geo


class GatedFusionModule(nn.Module):
    """
    门控融合模块
    
    将语义特征 (LLM) 和几何特征 (GNN) 进行动态融合。
    
    融合策略：
    Z_fused = λ_geo * H_geo + λ_sem * H_sem
    其中 λ = softmax(MLP(H_prompt))
    """
    
    def __init__(self, d_model: int = D_MODEL, num_heads: int = 8):
        """
        初始化融合模块
        
        Args:
            d_model: 特征维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 门控网络
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # 输出两个权重
        )
        
        # Cross-Attention (可选增强)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        H_geo: torch.Tensor, 
        H_sem: torch.Tensor, 
        H_prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            H_geo: 几何嵌入 [batch_size, num_nodes, d_model]
            H_sem: 语义嵌入 [batch_size, num_nodes, d_model]
            H_prompt: 提示嵌入 [batch_size, 1, d_model] (用于动态门控)
        
        Returns:
            Z_fused: 融合特征 [batch_size, num_nodes, d_model]
        """
        # 如果没有几何特征，直接返回语义特征
        if H_geo is None:
            return H_sem
        
        # 1. 计算门控权重
        if H_prompt is not None:
            # 使用 prompt 动态计算权重
            gates = torch.softmax(self.gate_fc(H_prompt.mean(dim=1)), dim=-1)
            # gates: [batch_size, 2]
            lambda_geo = gates[:, 0:1].unsqueeze(-1)  # [batch_size, 1, 1]
            lambda_sem = gates[:, 1:2].unsqueeze(-1)
        else:
            # 默认等权重
            lambda_geo = 0.5
            lambda_sem = 0.5
        
        # 2. 加权融合
        H_combined = lambda_geo * H_geo + lambda_sem * H_sem
        
        # 3. 残差连接和归一化
        Z_fused = self.norm(self.output_proj(H_combined) + H_sem)
        
        return Z_fused


class ResilienceLLM(nn.Module):
    """
    网络韧性优化大语言模型
    
    核心模型类，结合 LLM 语义理解和可选的 GNN 结构编码。
    支持通过 LoRA 进行高效微调。
    
    架构：
    1. LLM Backbone (with LoRA): 处理 OCG Prompt，提取语义特征
    2. GeometricEncoder (可选): 编码图结构特征
    3. FusionModule (可选): 融合语义和结构特征
    4. ScoringHead: 输出候选操作的排序分数
    
    训练模式：
    - Phase 1: 仅训练 LLM (LoRA)
    - Phase 2: 联合训练 LLM + GNN + Fusion
    
    Attributes:
        config: 模型配置
        llm: LLM 主干网络
        geo_encoder: 几何编码器 (可选)
        fusion: 融合模块 (可选)
        scoring_head: 分数预测头
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化 ResilienceLLM
        
        Args:
            config: 模型配置，None 使用默认配置
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # 初始化组件
        self.llm = None  # 延迟加载
        self.tokenizer = None
        self.geo_encoder = None
        self.fusion = None
        self.scoring_head = None
        
        self._initialized = False
    
    def initialize(self, device: str = "cuda"):
        """
        延迟初始化模型组件
        
        Args:
            device: 设备 ("cuda", "cpu")
        
        Note:
            由于 LLM 较大，采用延迟加载策略
        """
        if self._initialized:
            return
        
        # 检查并设置 HuggingFace 镜像（如果未设置）
        import os
        if 'HF_ENDPOINT' not in os.environ:
            # 尝试使用镜像站点（中国用户友好）
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print("ℹ️  使用 HuggingFace 镜像站点: https://hf-mirror.com")
            print("   如需更改，请设置环境变量: HF_ENDPOINT")
        
        # 1. 加载 LLM
        self._load_llm(device)
        
        # 2. 应用 LoRA (如果启用)
        if self.config.use_lora:
            self._apply_lora()
        
        # 3. 初始化几何编码器 (如果启用)
        if self.config.use_geometric_encoder:
            # 获取 LLM 的数据类型（LLM 已加载）
            model_dtype = next(self.llm.parameters()).dtype
            
            self.geo_encoder = GeometricEncoder(
                input_dim=self.config.geo_input_dim,
                hidden_dim=self.config.geo_hidden_dim,
                output_dim=self.config.d_model,
                num_layers=self.config.geo_num_layers,
                encoder_type=self.config.geo_encoder_type
            ).to(device).to(model_dtype)  # 确保使用与 LLM 相同的数据类型
            
            # 初始化融合模块
            # 获取 LLM 的数据类型（LLM 已加载）
            model_dtype = next(self.llm.parameters()).dtype
            
            self.fusion = GatedFusionModule(
                d_model=self.config.d_model,
                num_heads=self.config.num_attention_heads
            ).to(device).to(model_dtype)  # 确保使用与 LLM 相同的数据类型
        
        # 4. 初始化分数预测头
        # 获取 LLM 的数据类型（LLM 已加载）
        model_dtype = next(self.llm.parameters()).dtype
        
        self.scoring_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.d_model // 2, 1)
        ).to(device).to(model_dtype)  # 确保使用与 LLM 相同的数据类型
        
        self._initialized = True
    
    def _load_llm(self, device: str) -> None:
        """
        加载预训练 LLM
        
        Args:
            device: 设备
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"正在加载模型: {self.config.llm_model_name}")
        print(f"设备: {device}")
        
        # 内存优化配置
        # 注意：不指定 dtype/torch_dtype，让模型以默认精度加载
        # 混合精度训练由 autocast 处理，模型参数需保持 FP32
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            # 不使用 device_map="auto"，避免自动转换为 FP16
            load_kwargs["device_map"] = {"": 0}  # 直接放到 GPU 0
        else:
            load_kwargs["device_map"] = None
        
        # 加载模型
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            **load_kwargs
        )
        
        # 确保模型参数是 FP32（混合精度训练需要 FP32 参数）
        if next(self.llm.parameters()).dtype != torch.float32:
            print(f"⚠️  模型加载为 {next(self.llm.parameters()).dtype}，转换为 FP32...")
            self.llm = self.llm.float()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name,
            trust_remote_code=True
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 移动到设备（如果使用 CPU）
        if device == "cpu":
            self.llm = self.llm.to(device)
        
        print("模型加载完成!")
    
    def _apply_lora(self) -> None:
        """
        应用 LoRA 适配器
        
        使用 PEFT 库的 LoRA 对 LLM 进行参数高效微调。
        """
        from peft import LoraConfig, get_peft_model, TaskType
        
        print("正在应用 LoRA 适配器...")
        
        # 确定目标模块（根据模型架构调整）
        model_name_lower = self.config.llm_model_name.lower()
        if "qwen" in model_name_lower:
            # Qwen 模型的模块名称
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
            # LLaMA 模型的模块名称
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "chatglm" in model_name_lower:
            # ChatGLM 模型的模块名称
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            # 默认使用配置中的模块
            target_modules = self.config.lora_target_modules
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        print("LoRA 适配器应用完成!")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
        graph_data: Optional[Any] = None,  # PyG Data 对象
        return_scores: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            candidate_indices: 候选操作在序列中的位置索引 [batch_size, num_candidates]
            graph_data: PyG 图数据 (可选，用于几何编码)
            return_scores: 是否返回排序分数
        
        Returns:
            Dict:
                - "logits": LLM 输出 logits [batch_size, seq_len, vocab_size]
                - "scores": 候选操作分数 [batch_size, num_candidates] (如果 return_scores=True)
                - "hidden_states": 隐藏状态 (可选)
        """
        # 1. LLM 前向传播
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = llm_outputs.logits
        hidden_states = llm_outputs.hidden_states[-1]  # 最后一层隐藏状态
        
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states  # 保存 hidden states 供训练使用
        }
        
        # 2. 提取候选操作的语义嵌入
        if return_scores:
            if candidate_indices is not None:
                # 从隐藏状态中提取候选位置的表示
                batch_size, num_candidates = candidate_indices.shape
                
                # 收集候选位置的嵌入
                H_sem = self._gather_candidate_embeddings(hidden_states, candidate_indices)
                # H_sem: [batch_size, num_candidates, hidden_dim]
                
                # 投影到统一维度
                H_sem = self._project_to_d_model(H_sem)
                
                # 3. 几何编码和融合 (如果启用)
                if self.geo_encoder is not None and graph_data is not None:
                    H_geo = self.geo_encoder(
                        graph_data.x, 
                        graph_data.edge_index,
                        graph_data.batch if hasattr(graph_data, 'batch') else None
                    )
                    # 选取候选节点的几何嵌入
                    H_geo = self._gather_geometric_embeddings(H_geo, graph_data, candidate_indices)
                    
                    # 融合
                    H_fused = self.fusion(H_geo, H_sem)
                else:
                    H_fused = H_sem
                
                # 4. 计算排序分数
                # 确保 scoring_head 与输入数据类型一致
                if self.scoring_head[0].weight.dtype != H_fused.dtype:
                    self.scoring_head = self.scoring_head.to(H_fused.dtype)
                if self.scoring_head[0].weight.device != H_fused.device:
                    self.scoring_head = self.scoring_head.to(H_fused.device)
                    
                scores = self.scoring_head(H_fused).squeeze(-1)  # [batch_size, num_candidates]
                outputs["scores"] = scores
            else:
                # 如果没有 candidate_indices，返回 None
                # 让训练代码使用 hidden_states 计算分数
                outputs["scores"] = None
        
        return outputs
    
    def _gather_candidate_embeddings(
        self, 
        hidden_states: torch.Tensor,
        candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        从隐藏状态中收集候选位置的嵌入
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            candidate_indices: [batch_size, num_candidates]
        
        Returns:
            embeddings: [batch_size, num_candidates, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_candidates = candidate_indices.shape[1]
        
        # 确保索引是 long 类型（用于 gather 操作）
        candidate_indices = candidate_indices.long()
        
        # 扩展索引维度以便 gather
        indices = candidate_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        
        # 收集嵌入
        embeddings = torch.gather(hidden_states, dim=1, index=indices)
        
        return embeddings
    
    def _gather_geometric_embeddings(
        self,
        H_geo: torch.Tensor,
        graph_data: Any,
        candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        收集候选节点的几何嵌入
        
        Args:
            H_geo: 全图节点嵌入 [total_nodes, d_model]
            graph_data: PyG 数据
            candidate_indices: 候选索引
        
        Returns:
            [batch_size, num_candidates, d_model]
        """
        # TODO: 实现几何嵌入收集
        raise NotImplementedError("_gather_geometric_embeddings")
    
    def _project_to_d_model(self, embeddings: torch.Tensor) -> torch.Tensor:
        """投影到统一维度"""
        # 如果维度已经匹配，直接返回
        if embeddings.shape[-1] == self.config.d_model:
            return embeddings
        
        # 否则使用线性投影
        if not hasattr(self, 'semantic_projector') or self.semantic_projector is None:
            self.semantic_projector = nn.Linear(
                embeddings.shape[-1], 
                self.config.d_model
            ).to(embeddings.device).to(embeddings.dtype)  # 确保数据类型匹配
        
        # 确保投影层的设备和数据类型与输入匹配
        if self.semantic_projector.weight.device != embeddings.device:
            self.semantic_projector = self.semantic_projector.to(embeddings.device)
        if self.semantic_projector.weight.dtype != embeddings.dtype:
            self.semantic_projector = self.semantic_projector.to(embeddings.dtype)
        
        return self.semantic_projector(embeddings)
    
    def get_ranking_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        获取候选操作的排序分数
        
        用于推理时获取排序分数。
        
        Args:
            input_ids: Token IDs
            attention_mask: 注意力掩码
            candidate_indices: 候选操作位置索引
        
        Returns:
            scores: [batch_size, num_candidates]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_indices=candidate_indices,
            return_scores=True
        )
        return outputs["scores"]
    
    def rank_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_ids: List[List[str]]
    ) -> List[List[str]]:
        """
        对候选操作进行排序
        
        Args:
            input_ids: Token IDs
            attention_mask: 注意力掩码
            candidate_indices: 候选位置索引
            candidate_ids: 候选操作 ID 列表 [["op_01", "op_02", ...], ...]
        
        Returns:
            ranked_ids: 排序后的操作 ID 列表
        """
        scores = self.get_ranking_scores(input_ids, attention_mask, candidate_indices)
        
        ranked_results = []
        for batch_idx in range(scores.shape[0]):
            batch_scores = scores[batch_idx]  # [num_candidates]
            batch_ids = candidate_ids[batch_idx]
            
            # 按分数降序排序
            sorted_indices = torch.argsort(batch_scores, descending=True)
            ranked = [batch_ids[i] for i in sorted_indices.cpu().numpy()]
            ranked_results.append(ranked)
        
        return ranked_results
    
    def save_pretrained(self, save_path: str) -> None:
        """保存模型"""
        # TODO: 实现保存逻辑
        raise NotImplementedError("save_pretrained")
    
    def load_pretrained(self, load_path: str) -> None:
        """加载模型"""
        # TODO: 实现加载逻辑
        raise NotImplementedError("load_pretrained")
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_trainable_parameters(self) -> None:
        """打印可训练参数信息"""
        trainable = self.get_trainable_parameters()
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ==================== 工厂函数 ====================

def create_resilience_llm(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    use_lora: bool = True,
    use_geometric: bool = False,
    lora_r: int = 8,
    device: str = "cuda"
) -> ResilienceLLM:
    """
    工厂函数：创建 ResilienceLLM 实例
    
    Args:
        model_name: LLM 模型名称
        use_lora: 是否使用 LoRA
        use_geometric: 是否使用几何编码器
        lora_r: LoRA rank
        device: 设备
    
    Returns:
        ResilienceLLM 实例
    """
    config = ModelConfig(
        llm_model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        use_geometric_encoder=use_geometric
    )
    
    model = ResilienceLLM(config)
    # model.initialize(device)  # 按需初始化
    
    return model
