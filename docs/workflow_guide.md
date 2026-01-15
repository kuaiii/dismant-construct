# 完整工作流程指南

本指南提供从数据生成到模型训练的完整流程。

## 📋 目录

1. [数据生成阶段](#数据生成阶段)
2. [数据准备和验证](#数据准备和验证)
3. [模型训练阶段](#模型训练阶段)
4. [模型评估阶段](#模型评估阶段)
5. [常见问题](#常见问题)

---

## 数据生成阶段

### 步骤 1: 生成 Dismantle（拆解）数据集

**任务目标**: 最小化网络韧性，选择破坏性最大的节点进行移除。

```bash
# 从合成网络生成 dismantle 数据
python scripts/generate_data.py \
    --data_source syn \
    --num_graphs 100 \
    --task_type dismantle \
    --budget 10 \
    --output_dir data/fine_tuning/dismantle_syn

# 从真实网络生成 dismantle 数据
python scripts/generate_data.py \
    --data_source true \
    --num_graphs 50 \
    --task_type dismantle \
    --budget 10 \
    --output_dir data/fine_tuning/dismantle_true

# 混合数据源
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --task_type dismantle \
    --budget 10 \
    --output_dir data/fine_tuning/dismantle_all
```

### 步骤 2: 生成 Construct（构造）数据集

**任务目标**: 最大化网络韧性，选择增益最大的边进行添加。

```bash
# 从合成网络生成 construct 数据
python scripts/generate_data.py \
    --data_source syn \
    --num_graphs 100 \
    --task_type construct \
    --budget 10 \
    --output_dir data/fine_tuning/construct_syn

# 从真实网络生成 construct 数据
python scripts/generate_data.py \
    --data_source true \
    --num_graphs 50 \
    --task_type construct \
    --budget 10 \
    --output_dir data/fine_tuning/construct_true

# 混合数据源
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --task_type construct \
    --budget 10 \
    --output_dir data/fine_tuning/construct_all
```

### 步骤 3: 生成混合任务数据集（可选）

如果需要同时训练两种任务，可以生成混合数据集：

```bash
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --task_type both \
    --budget 10 \
    --output_dir data/fine_tuning/mixed_all
```

**注意**: `--task_type both` 会在同一批次中随机生成 dismantle 和 construct 两种任务的数据。

---

## 数据准备和验证

### 步骤 4: 合并数据集（可选）

如果需要将多个数据集合并：

```python
# 合并脚本示例 (scripts/merge_datasets.py)
import json
from pathlib import Path

def merge_datasets(input_dirs, output_file):
    """合并多个数据集"""
    all_samples = []
    
    for input_dir in input_dirs:
        train_file = Path(input_dir) / "train.json"
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                all_samples.extend(samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"合并了 {len(all_samples)} 个样本到 {output_file}")

# 使用示例
merge_datasets(
    input_dirs=[
        "data/fine_tuning/dismantle_syn",
        "data/fine_tuning/construct_syn"
    ],
    output_file="data/fine_tuning/combined_train.json"
)
```

### 步骤 5: 数据统计和验证

检查生成的数据：

```python
# 数据统计脚本示例
import json
from pathlib import Path
from collections import Counter

def analyze_dataset(data_file):
    """分析数据集统计信息"""
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"总样本数: {len(samples)}")
    
    # 任务类型分布
    task_types = [s['meta'].get('task', 'unknown') for s in samples]
    print(f"任务类型分布: {Counter(task_types)}")
    
    # 数据源分布
    data_sources = [s['meta'].get('data_source', 'unknown') for s in samples]
    print(f"数据源分布: {Counter(data_sources)}")
    
    # 节点数分布
    node_counts = [s['meta'].get('num_nodes', 0) for s in samples]
    print(f"节点数范围: [{min(node_counts)}, {max(node_counts)}]")
    print(f"平均节点数: {sum(node_counts) / len(node_counts):.2f}")
    
    # auxiliary_labels 统计
    label_values = []
    for s in samples:
        if 'auxiliary_labels' in s:
            label_values.extend(s['auxiliary_labels'].values())
    
    if label_values:
        print(f"标签值范围: [{min(label_values):.4f}, {max(label_values):.4f}]")
        print(f"平均标签值: {sum(label_values) / len(label_values):.4f}")

# 使用示例
analyze_dataset("data/fine_tuning/dismantle_syn/train.json")
```

---

## 模型训练阶段

### 步骤 6: 准备训练配置

编辑 `configs/default.yaml` 或创建新的配置文件：

```yaml
# configs/train_config.yaml
data:
  fine_tuning_dir: "data/fine_tuning/dismantle_syn"  # 或 construct_syn, mixed_all 等

model:
  llm:
    model_name: "meta-llama/Meta-Llama-3-8B"  # 根据实际情况修改
  
  lora:
    enabled: true
    r: 8
    alpha: 32

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  ranking_loss_type: "listmle"
  phase: 1  # Phase 1: LLM only, Phase 2: Joint training
```

### 步骤 7: Phase 1 训练（LLM LoRA 微调）

**目标**: 仅训练 LLM 的 LoRA 参数，学习语义理解和排序。

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --train_data data/fine_tuning/dismantle_syn/train.json \
    --eval_data data/fine_tuning/dismantle_syn/eval.json \
    --phase 1 \
    --epochs 3 \
    --output_dir outputs/dismantle_phase1
```

**训练检查点**:
- 检查 `outputs/dismantle_phase1/training.log` 查看训练日志
- 检查 `outputs/dismantle_phase1/checkpoints/` 目录保存的模型

### 步骤 8: Phase 2 训练（联合训练，可选）

**目标**: 如果需要使用 GNN 编码器，进行联合训练。

```bash
# 修改配置启用几何编码器
# configs/train_config.yaml:
#   model.geometric_encoder.enabled: true
#   training.phase: 2

python scripts/train.py \
    --config configs/train_config.yaml \
    --train_data data/fine_tuning/dismantle_syn/train.json \
    --eval_data data/fine_tuning/dismantle_syn/eval.json \
    --phase 2 \
    --epochs 2 \
    --output_dir outputs/dismantle_phase2 \
    --resume outputs/dismantle_phase1/checkpoints/best
```

---

## 模型评估阶段

### 步骤 9: 评估模型性能

```python
# 评估脚本示例
from src.model.fusion_llm import ResilienceLLM, ModelConfig
from src.data.dataset import create_dataloader
from src.model.loss import RankingMetrics
import torch

# 加载模型
config = ModelConfig(...)
model = ResilienceLLM(config)
model.load_pretrained("outputs/dismantle_phase1/checkpoints/best")

# 加载评估数据
eval_loader = create_dataloader(
    data_path="data/fine_tuning/dismantle_syn/eval.json",
    batch_size=8,
    shuffle=False
)

# 评估指标
model.eval()
all_ndcg = []
all_mrr = []

with torch.no_grad():
    for batch in eval_loader:
        scores = model.get_ranking_scores(
            batch["input_ids"],
            batch["attention_mask"],
            batch["candidate_indices"]
        )
        
        labels = batch["auxiliary_labels"]
        for i in range(scores.shape[0]):
            ndcg = RankingMetrics.ndcg(scores[i], labels[i])
            mrr = RankingMetrics.mrr(scores[i], labels[i])
            all_ndcg.append(ndcg)
            all_mrr.append(mrr)

print(f"Average NDCG: {sum(all_ndcg) / len(all_ndcg):.4f}")
print(f"Average MRR: {sum(all_mrr) / len(all_mrr):.4f}")
```

### 步骤 10: 在真实网络上测试

```python
# 测试脚本示例
from src.env.simulator import NetworkEnvironment
from src.env.metrics import ResilienceMetrics

# 加载测试图
test_graph = NetworkEnvironment.load_graph("path/to/test/graph.gml", format="gml")

# 创建环境
env = NetworkEnvironment(
    graph=test_graph,
    task_type=TaskType.DISMANTLE,
    budget=10
)

# 使用模型进行预测
# ... 实现推理循环
```

---

## 推荐的工作流程

### 方案 A: 分别训练 Dismantle 和 Construct

```bash
# 1. 生成数据
python scripts/generate_data.py --data_source syn --task_type dismantle --num_graphs 100 --output_dir data/fine_tuning/dismantle
python scripts/generate_data.py --data_source syn --task_type construct --num_graphs 100 --output_dir data/fine_tuning/construct

# 2. 训练 Dismantle 模型
python scripts/train.py --train_data data/fine_tuning/dismantle/train.json --output_dir outputs/dismantle_model

# 3. 训练 Construct 模型
python scripts/train.py --train_data data/fine_tuning/construct/train.json --output_dir outputs/construct_model
```

### 方案 B: 混合训练（单模型处理两种任务）

```bash
# 1. 生成混合数据
python scripts/generate_data.py --data_source all --task_type both --num_graphs 200 --output_dir data/fine_tuning/mixed

# 2. 训练混合模型
python scripts/train.py --train_data data/fine_tuning/mixed/train.json --output_dir outputs/mixed_model
```

---

## 常见问题

### Q1: Dismantle 和 Construct 任务的区别是什么？

- **Dismantle（拆解）**: 目标是**最小化**网络韧性，选择**移除**破坏性最大的节点
- **Construct（构造）**: 目标是**最大化**网络韧性，选择**添加**增益最大的边

### Q2: 应该生成多少数据？

- **小型实验**: 100-200 个图，每个图 10 步 → 1000-2000 个样本
- **完整训练**: 500-1000 个图，每个图 10-20 步 → 5000-20000 个样本
- **大规模训练**: 1000+ 个图 → 20000+ 个样本

### Q3: 如何平衡 Dismantle 和 Construct 数据？

如果需要混合训练，建议：
- 50% Dismantle + 50% Construct
- 或根据实际应用场景调整比例

### Q4: 训练需要多长时间？

- **Phase 1 (LoRA)**: 
  - 1000 样本: ~1-2 小时（单 GPU）
  - 10000 样本: ~10-20 小时
- **Phase 2 (Joint)**: 
  - 通常比 Phase 1 慢 2-3 倍

### Q5: 如何选择 budget 参数？

- **小图 (<100 节点)**: budget = 5-10
- **中图 (100-500 节点)**: budget = 10-20
- **大图 (>500 节点)**: budget = 20-50

注意: budget 不应超过节点数的一半。

### Q6: 数据生成失败怎么办？

检查：
1. 图文件格式是否正确（.gml 或 .graphml）
2. 图是否连通（脚本会自动提取最大连通分量）
3. 图大小是否满足 `--min_graph_size` 要求
4. 查看错误日志定位问题

---

## 下一步建议

1. **数据质量检查**: 生成数据后，使用统计脚本验证数据质量
2. **小规模实验**: 先用少量数据（50-100 个图）测试完整流程
3. **超参数调优**: 根据验证集性能调整学习率、batch size 等
4. **模型对比**: 比较不同配置（LoRA rank、loss type 等）的效果
5. **真实场景测试**: 在真实网络数据上评估模型泛化能力
