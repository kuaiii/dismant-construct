# 评估完成后的实验建议

基于您的评估结果（NDCG@5: 0.9013, MRR: 0.9132, Top-1: 0.8632），以下是继续实验的建议。

## 📊 当前评估结果分析

您的模型表现**优秀**：
- ✅ NDCG@5 > 0.9: 排序质量非常高
- ✅ MRR > 0.9: 能准确识别最优操作
- ✅ Top-1 准确率 > 0.86: 大部分情况下能选择最佳操作

## 🎯 下一步实验建议

### 1. 在实际网络上进行推理测试（推荐优先）

**目标**: 验证模型在真实场景中的表现

```powershell
# 测试拆解任务
python scripts/inference.py `
    --checkpoint outputs/resilience_llm/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task dismantle `
    --budget 10

# 测试构造任务（如果支持）
python scripts/inference.py `
    --checkpoint outputs/resilience_llm/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task construct `
    --budget 10
```

**观察指标**:
- R_res 的变化量（拆解任务应该降低，构造任务应该提高）
- LCC 比例的变化
- 操作序列的合理性

### 2. 批量测试多个图

创建批量测试脚本，评估模型在不同图上的表现：

```powershell
# 创建批量测试脚本
python -c "
import glob
from pathlib import Path
import subprocess

checkpoint = 'outputs/resilience_llm/checkpoints/best'
test_graphs = glob.glob('data/raw_graphs/syn/*.gml')[:20]  # 测试前20个图

results = []
for graph_path in test_graphs:
    print(f'\n测试图: {graph_path}')
    # 运行推理并收集结果
    # ...
"
```

### 3. 分析错误案例

**目标**: 找出模型失败的情况，改进训练数据或模型

```python
# 创建错误分析脚本 scripts/analyze_errors.py
import json
import torch
from scripts.evaluate import evaluate_model

# 评估并保存详细结果
results = evaluate_model(...)

# 找出 Top-1 预测错误的样本
error_samples = []
for i, sample in enumerate(eval_loader):
    if predictions[i] != ground_truth[i]:
        error_samples.append({
            'sample_id': sample['sample_id'],
            'predicted': predictions[i],
            'ground_truth': ground_truth[i],
            'scores': scores[i].tolist()
        })

# 保存错误案例
with open('error_analysis.json', 'w') as f:
    json.dump(error_samples, f, indent=2)
```

### 4. 继续训练以提升性能（可选）

如果希望进一步提升性能，可以：

**选项 A: 增加训练轮数**
```powershell
python scripts/train.py `
    --train_data data/fine_tuning/combined/train.json `
    --eval_data data/fine_tuning/combined/eval.json `
    --output_dir outputs/resilience_llm_v2 `
    --phase 1 `
    --epochs 5 `
    --resume outputs/resilience_llm/checkpoints/best
```

**选项 B: 调整超参数**
```powershell
# 尝试不同的学习率
python scripts/train.py `
    --train_data data/fine_tuning/combined/train.json `
    --eval_data data/fine_tuning/combined/eval.json `
    --output_dir outputs/resilience_llm_lr_tune `
    --phase 1 `
    --epochs 3 `
    --lr 1e-5  # 或 3e-5, 5e-5
```

**选项 C: 增加训练数据**
```powershell
# 生成更多训练数据
python scripts/generate_data.py `
    --data_source all `
    --num_graphs 500 `
    --output_dir data/fine_tuning/expanded

# 合并数据集
python scripts/merge_datasets.py `
    --input data/fine_tuning/combined/train.json `
    --input data/fine_tuning/expanded/train.json `
    --output data/fine_tuning/merged/train.json
```

### 5. 消融实验（Ablation Study）

**目标**: 理解各个组件的作用

**实验 1: LoRA rank 的影响**
```powershell
# 测试不同的 LoRA rank
for r in 4 8 16 32; do
    python scripts/train.py `
        --train_data data/fine_tuning/combined/train.json `
        --eval_data data/fine_tuning/combined/eval.json `
        --output_dir outputs/ablation_lora_r_$r `
        --lora_r $r `
        --epochs 3
done
```

**实验 2: 损失函数的影响**
```powershell
# 测试不同的损失函数
python scripts/train.py --ranking_loss_type listmle ...
python scripts/train.py --ranking_loss_type listnet ...
python scripts/train.py --ranking_loss_type combined ...
```

**实验 3: 谱梯度剪枝的影响**
```powershell
# 测试不同的 top_k 值
for k in 20 50 100 200; do
    # 修改配置文件中的 spectral_top_k
    python scripts/generate_data.py --spectral_top_k $k ...
done
```

### 6. 跨数据集泛化测试

**目标**: 验证模型在不同类型图上的泛化能力

```powershell
# 在真实网络上测试（如果训练数据是合成图）
python scripts/inference.py `
    --checkpoint outputs/resilience_llm/checkpoints/best `
    --graph data/raw_graphs/true/real_network_001.gml `
    --task dismantle

# 在不同规模的图上测试
for size in small medium large; do
    python scripts/inference.py `
        --checkpoint outputs/resilience_llm/checkpoints/best `
        --graph data/raw_graphs/${size}/graph_001.gml `
        --task dismantle
done
```

### 7. 对比实验

**目标**: 与基线方法对比

```python
# 创建对比脚本 scripts/compare_baselines.py
from src.env.simulator import NetworkEnvironment
from src.env.metrics import ResilienceMetrics

# 基线方法 1: 随机选择
def random_baseline(env, budget):
    # ...

# 基线方法 2: 度数中心性
def degree_baseline(env, budget):
    # ...

# 基线方法 3: 您的模型
def model_baseline(env, budget, model):
    # ...

# 对比结果
results = {
    'random': [],
    'degree': [],
    'model': []
}
```

### 8. 可视化分析

**目标**: 可视化模型决策过程

```python
# 创建可视化脚本 scripts/visualize_decisions.py
import matplotlib.pyplot as plt
import networkx as nx

def visualize_attack_sequence(graph, attack_sequence, save_path):
    """可视化攻击序列对网络的影响"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for step, node in enumerate(attack_sequence[:10]):
        # 绘制当前图状态
        # ...
    
    plt.savefig(save_path)
```

## 📈 实验记录建议

建议创建一个实验日志，记录每次实验的结果：

```markdown
# 实验日志

## 实验 1: 基础训练
- 日期: 2026-01-14
- 配置: Phase 1, LoRA r=8, epochs=3
- 结果: NDCG@5=0.9013, MRR=0.9132, Top-1=0.8632
- 备注: 表现优秀

## 实验 2: 推理测试
- 日期: 2026-01-14
- 测试图: 20 个合成图
- 结果: 平均 R_res 降低 0.XX
- 备注: 需要进一步分析
```

## 🎯 优先级建议

基于您当前的优秀评估结果，建议按以下优先级进行：

1. **高优先级**:
   - ✅ 在实际网络上进行推理测试（验证泛化能力）
   - ✅ 批量测试多个图（评估稳定性）

2. **中优先级**:
   - 分析错误案例（找出改进方向）
   - 跨数据集泛化测试（验证鲁棒性）

3. **低优先级**:
   - 消融实验（理解模型组件）
   - 继续训练（如果性能已满足需求，可能不需要）

## 📝 快速开始命令

```powershell
# 1. 推理测试（单图）
python scripts/inference.py `
    --checkpoint outputs/resilience_llm/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task dismantle

# 2. 重新评估（确认结果）
python scripts/evaluate.py `
    --checkpoint outputs/resilience_llm/checkpoints/best `
    --eval_data data/fine_tuning/combined/eval.json

# 3. 查看训练日志
Get-Content outputs/resilience_llm/training.log -Tail 50
```

## 🔍 问题诊断

如果推理结果不理想，检查：

1. **模型是否正确加载**
   ```python
   # 检查模型参数
   print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
   ```

2. **数据格式是否匹配**
   ```python
   # 检查数据格式
   sample = eval_loader.dataset[0]
   print(sample.keys())
   ```

3. **OCG 提取是否正常**
   ```python
   # 检查 OCG 提取
   ocg_data = extractor.extract_ocg(...)
   print(f"OCG 节点数: {len(ocg_data['nodes'])}")
   ```

---

**祝实验顺利！** 🚀
