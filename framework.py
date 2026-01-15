# 维度配置
D_GEO_RAW = 512    # GNN 原始输出维度
D_LLM_RAW = 4096   # Llama-3 原始输出维度
D_MODEL = 1024     # 统一投影后的对齐维度 (d_model)
N_NODES = "N"      # 动态节点数量

# B. 几何塔 (Geometric Tower)
import torch
import torch.nn as nn
from torch_geometric.data import Data

class GeometricTower(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 使用 Graph Transformer 或 GIN
        self.encoder = GraphTransformer(in_channels=input_dim, out_channels=hidden_dim)
        # 投影层：将 GNN 维度映射到统一维度 d_model
        self.projector = nn.Linear(hidden_dim, D_MODEL)

    def forward(self, graph_data: Data) -> torch.Tensor:
        """
        Input:
            graph_data: PyG Data 对象
                - x: 节点初始特征 [N, input_dim] (可以是简单的 One-hot 或 Degree feature)
                - edge_index: 邻接关系 [2, E]
        Output:
            H_geo: 对齐后的几何嵌入 [N, D_MODEL]
        """
        # 1. 提取拓扑特征
        raw_geo_emb = self.encoder(graph_data.x, graph_data.edge_index) 
        
        # 2. 投影对齐
        H_geo = self.projector(raw_geo_emb) 
        
        return H_geo  # [N, 1024]
    

# C. 语义塔 (Semantic Tower)
class SemanticTower(nn.Module):
    def __init__(self, llm_model_name, use_lora=True):
        super().__init__()
        # 加载预训练 LLM (如 'meta-llama/Meta-Llama-3-8B')
        self.llm = AutoModel.from_pretrained(llm_model_name)
        if use_lora:
            self.llm = apply_lora(self.llm) # 伪代码：应用 LoRA
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # 投影层：将 LLM 维度映射到统一维度 d_model
        self.projector = nn.Linear(D_LLM_RAW, D_MODEL)

    def forward(self, node_texts: list[str], prompt_text: str = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            node_texts: 节点文本列表，长度为 N
            prompt_text: 任务指令 (例如 "Find the most vulnerable node")
        Output:
            H_sem: 节点语义嵌入 [N, D_MODEL]
            H_prompt: 提示嵌入 (用于 Query) [1, D_MODEL]
        """
        # 1. Tokenize 节点文本 (Batch processing)
        inputs = self.tokenizer(node_texts, return_tensors="pt", padding=True, truncation=True)
        
        # 2. 获取 LLM 最后一层 hidden state (取 [CLS] 或 平均池化)
        outputs = self.llm(**inputs)
        raw_sem_emb = outputs.last_hidden_state.mean(dim=1) # [N, 4096]
        
        # 3. 处理 Prompt (如果有)
        H_prompt = None
        if prompt_text:
            p_inputs = self.tokenizer(prompt_text, return_tensors="pt")
            p_out = self.llm(**p_inputs)
            raw_prompt_emb = p_out.last_hidden_state[:,-1,:] # 取最后一个 Token 代表句意
            H_prompt = self.projector(raw_prompt_emb) # [1, 1024]

        # 4. 投影对齐
        H_sem = self.projector(raw_sem_emb) # [N, 1024]
        
        return H_sem, H_prompt


# D. 跨模态融合层 (Fusion Layer)
class GatedFusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        self.gate_fc = nn.Linear(d_model, 2) # 生成两个标量权重 lambda_geo, lambda_sem

    def forward(self, H_geo: torch.Tensor, H_sem: torch.Tensor, H_prompt: torch.Tensor) -> torch.Tensor:
        """
        Input:
            H_geo: [N, d_model]
            H_sem: [N, d_model]
            H_prompt: [1, d_model] (作为 Query)
        Output:
            Z_fused: 融合后的节点表示 [N, d_model]
        """
        # 1. 拼接 Key 和 Value
        # 这里我们将 Geo 和 Sem 视为信息的不同来源
        # 简单策略：直接加权求和，或者拼接后做 Attention
        
        # 策略A：利用 Prompt 计算动态门控权重 (Dynamic Gating)
        # 门控系数取决于任务 Prompt
        gates = torch.softmax(self.gate_fc(H_prompt), dim=-1) # [1, 2]
        lambda_geo = gates[0, 0]
        lambda_sem = gates[0, 1]
        
        # 2. 基础融合
        H_combined = lambda_geo * H_geo + lambda_sem * H_sem
        
        # 3. (可选) 进一步利用 Prompt 对节点进行 Attention 加权
        # Query=H_prompt, Key=H_combined, Value=H_combined
        # 注意：这里维度需要调整为 [Seq_len, Batch, Dim] 适配 torch.nn.MHA
        # 这里的 Attention 作用是：根据 Prompt 筛选哪些节点特征更重要
        # 但在节点分类/排序任务中，通常是对每个节点独立增强
        
        Z_fused = H_combined # 简化版，实际可用更复杂的 Cross-Attn
        
        return Z_fused


# E. 主模型封装 (SIG-FM Main Wrapper)
class SIG_FM(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_tower = GeometricTower(...)
        self.sem_tower = SemanticTower(...) # 设为 eval 模式或只训练 LoRA
        self.fusion = GatedFusionLayer(D_MODEL)
        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 输出标量分数
        )

    def forward(self, graph_data, node_texts, task_prompt):
        # 1. 获取双塔特征
        H_geo = self.geo_tower(graph_data)
        H_sem, H_prompt = self.sem_tower(node_texts, task_prompt)
        
        # 2. 融合 (Phase 2 训练重点)
        Z = self.fusion(H_geo, H_sem, H_prompt)
        
        # 3. 预测干预分数 (Predict Intervention Score)
        logits = self.decoder(Z) # [N, 1]
        
        # 4. 转换为概率分布 (用于采样或排序)
        probs = torch.softmax(logits, dim=0) 
        
        return probs


