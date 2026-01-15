# 🔗 Neural-Symbolic Network Resilience Optimization Framework

神经符号网络韧性优化框架 - 结合大语言模型 (LLM) 和图神经网络 (GNN) 的网络韧性优化系统。

## 📋 项目概述

本框架实现了一个神经符号系统，用于网络韧性优化任务（拆解/构造）。核心思想是：

1. **操作中心图 (OCG)**: 提取候选节点周围的局部子图结构和语义信息
2. **谱梯度剪枝**: 将候选空间从 O(N²) 降低到 O(N)
3. **ListMLE 排序学习**: 使用 `auxiliary_labels` 进行排序损失优化
4. **LoRA 微调**: 参数高效地微调大语言模型

## 🗂️ 项目结构

```
project_root/
├── data/
│   ├── raw_graphs/           # BA, ER 原始图数据
│   └── fine_tuning/          # 生成的 JSON 微调数据
│       └── sample_data.json  # 示例数据
├── src/
│   ├── env/
│   │   ├── simulator.py      # NetworkEnvironment (图状态管理、谱梯度剪枝)
│   │   └── metrics.py        # R_res 韧性积分计算
│   ├── data/
│   │   ├── ocg_builder.py    # OCG 提取和 Prompt 生成
│   │   └── dataset.py        # PyTorch Dataset 和 DataLoader
│   ├── model/
│   │   ├── fusion_llm.py     # ResilienceLLM 主模型架构
│   │   └── loss.py           # ListMLELoss 排序损失函数
│   └── trainer/
│       └── train.py          # 训练循环和评估
├── scripts/
│   ├── generate_data.py      # 数据生成脚本
│   └── train.py              # 训练启动脚本
├── configs/
│   └── default.yaml          # 默认配置文件
├── requirements.txt          # 依赖项
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 最小安装（测试用）
pip install transformers peft

# 完整安装（推荐）
pip install -r requirements.txt
```

**注意**：详细安装说明请参考 [安装指南](INSTALL.md)

### 2. 生成训练数据

**从合成/真实网络数据生成：**

```bash
# 从合成网络数据生成（syn 目录）
python scripts/generate_data.py \
    --data_source syn \
    --num_graphs 100 \
    --output_dir data/fine_tuning

# 从真实网络数据生成（true 目录）
python scripts/generate_data.py \
    --data_source true \
    --num_graphs 50 \
    --output_dir data/fine_tuning

# 混合使用（syn + true）
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --output_dir data/fine_tuning
```

**生成 BA/ER 图（原有方式）：**

```bash
python scripts/generate_data.py \
    --data_source generate \
    --graph_type ba \
    --num_graphs 100 \
    --min_nodes 50 \
    --max_nodes 200 \
    --output_dir data/fine_tuning
```

详细说明请参考 [数据生成指南](docs/data_generation_guide.md)

### 3. 下载模型（首次运行）

```bash
# 方法 1: 使用下载脚本（推荐）
python scripts/download_model.py

# 方法 2: 使用测试脚本（会自动下载）
python scripts/test_model_loading.py
```

**注意**: 模型会自动下载到 HuggingFace 缓存目录（约 3GB）

### 4. 启动训练

```bash
# 基础训练
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/mixed_model \
    --phase 1 \
    --epochs 3

# 小规模测试（推荐先运行）
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/test_run \
    --phase 1 \
    --epochs 1 \
    --batch_size 1
```

**详细说明**: 请参考 [开始训练指南](docs/training_start_guide.md)

### 5. 模型推理

```bash
# Dismantle 任务推理（移除节点以降低韧性）
python scripts/inference.py \
    --checkpoint outputs/dismantle_model/checkpoints/best \
    --graph data/raw_graphs/true/Colt.gml \
    --task dismantle \
    --budget 10

# Construct 任务推理（添加边以提高韧性）
python scripts/inference.py \
    --checkpoint outputs/construct_model/checkpoints/best \
    --graph data/raw_graphs/true/Colt.gml \
    --task construct \
    --budget 10
```

> 📖 **详细指南**: 
> - Dismantle 任务: 参考 `docs/workflow_guide.md`
> - Construct 任务: 参考 `docs/construct_experiment_guide.md`

## 📊 核心模块说明

### NetworkEnvironment (simulator.py)

网络环境模拟器，负责：
- 维护图状态 G_t
- 执行 **谱梯度剪枝** (Spectral Gradient Pruning)
- 计算候选操作的影响分数

```python
from src.env.simulator import NetworkEnvironment, create_environment

# 创建环境
env = create_environment(
    graph_type="ba",
    num_nodes=100,
    task="dismantle",
    budget=10,
    spectral_top_k=50
)

# 获取候选节点 (谱梯度剪枝后)
candidates = env.prune_candidates()

# 执行操作
reward, done = env.execute_operation(operation)
```

**谱梯度计算原理**:
- 计算图拉普拉斯矩阵 L = D - A
- 求解 Fiedler 向量 (第二小特征值对应的特征向量)
- 节点 i 的谱梯度 ≈ |v₂[i]|² × d_i

### OCGExtractor (ocg_builder.py)

操作中心图提取器，负责：
- 提取候选节点的 k-hop 子图
- 计算结构特征（度数、聚类系数、割点等）
- 融合语义信息生成 Prompt

```python
from src.data.ocg_builder import OCGExtractor

extractor = OCGExtractor(hop_distance=1, language="zh")

# 提取 OCG
ocg_data = extractor.extract_ocg(
    graph=env.graph,
    candidate_nodes=candidates,
    task_type="dismantle",
    current_step=1,
    total_steps=10,
    node_semantics=semantics
)

# 构建训练样本
sample = extractor.build_conversation_data(
    ocg_data=ocg_data,
    ground_truth_ranking=["op_01", "op_03", "op_02"],
    auxiliary_labels={"op_01": 0.95, "op_02": 0.15, "op_03": 0.40}
)
```

### ListMLELoss (loss.py)

基于 Plackett-Luce 模型的排序损失函数：

```python
from src.model.loss import ListMLELoss

loss_fn = ListMLELoss(temperature=1.0)

# scores: 模型预测分数 [batch_size, num_candidates]
# auxiliary_labels: 真实影响分数 [batch_size, num_candidates]
loss = loss_fn(scores, auxiliary_labels, mask=candidate_mask)
```

**ListMLE 数学原理**:
```
L = -log P(π|s) = -Σᵢ log(exp(s_{π_i}) / Σⱼ≥ᵢ exp(s_{π_j}))
```

其中 π 是根据 `auxiliary_labels` 得到的真实排序。

### ResilienceLLM (fusion_llm.py)

主模型架构，支持：
- LoRA 微调 LLM
- 可选的几何编码器 (GNN)
- 门控融合模块

```python
from src.model.fusion_llm import ResilienceLLM, ModelConfig

config = ModelConfig(
    llm_model_name="meta-llama/Meta-Llama-3-8B",
    use_lora=True,
    lora_r=8,
    use_geometric_encoder=False
)

model = ResilienceLLM(config)
model.initialize(device="cuda")

# 获取排序分数
scores = model.get_ranking_scores(input_ids, attention_mask, candidate_indices)
```

## 📁 数据格式

训练数据采用对话格式，兼容 LLaMA-Factory 等微调框架：

```json
{
  "id": "train_dismantle_001",
  "meta": {
    "task": "dismantle",
    "budget_step": "1/10"
  },
  "conversations": [
    {"from": "system", "value": "系统提示..."},
    {"from": "user", "value": "OCG 描述和候选列表..."},
    {"from": "assistant", "value": "推理和排序结果..."}
  ],
  "auxiliary_labels": {
    "op_01": 0.95,
    "op_02": 0.15,
    "op_03": 0.40
  }
}
```

**关键字段**:
- `auxiliary_labels`: 用于 ListMLE 计算的真实影响分数
- `conversations`: 标准对话格式，用于 LLM 微调

## 🔄 数据流

```
Raw Graph (BA/ER)
      ↓
NetworkEnvironment (谱梯度剪枝)
      ↓
OCGExtractor (提取 OCG)
      ↓
JSON Data (conversations + auxiliary_labels)
      ↓
ResilienceDataset (加载和预处理)
      ↓
ResilienceLLM (模型前向传播)
      ↓
ListMLELoss (排序损失计算)
      ↓
Model Update (LoRA 参数更新)
```

## ⚙️ 配置说明

主要配置项 (`configs/default.yaml`):

```yaml
model:
  llm:
    model_name: "meta-llama/Meta-Llama-3-8B"
  lora:
    enabled: true
    r: 8
    alpha: 32

training:
  loss:
    ranking_type: "listmle"  # listmle, listnet, combined
    ranking_weight: 1.0
    lm_weight: 0.5

environment:
  spectral_pruning:
    enabled: true
    top_k: 50
```

## 📈 评估指标

- **NDCG** (Normalized Discounted Cumulative Gain)
- **MRR** (Mean Reciprocal Rank)
- **Precision@K**
- **Kendall's Tau** (排序相关性)

## 🔧 关键点检查

1. ✅ **ListMLE**: `loss.py` 中实现了基于 `auxiliary_labels` 的排序损失
2. ✅ **谱梯度剪枝**: `simulator.py` 中预留了 `compute_spectral_gradient` 和 `prune_candidates` 接口
3. ✅ **OCG 构建**: `ocg_builder.py` 中实现了图状态到 Prompt 文本的转换

## 📝 TODO

- [ ] 实现完整的谱梯度计算 (稀疏特征值求解)
- [ ] 添加更多 GNN 编码器类型 (GAT, GraphTransformer)
- [ ] 支持分布式训练
- [ ] 添加推理服务 API

## 📄 License

MIT License

## 🤝 Contributing

欢迎提交 Issue 和 Pull Request！
