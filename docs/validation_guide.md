# 模型验证指南

训练完成后，使用以下步骤验证模型性能。

## 📋 验证步骤概览

1. **评估模型** - 在评估集上计算排序指标
2. **推理测试** - 在实际网络上测试模型
3. **分析结果** - 检查模型性能和改进方向

---

## 步骤 1: 评估模型性能

### 基础评估

```powershell
# 使用训练时保存的最佳检查点
python scripts/evaluate.py `
    --checkpoint outputs/mixed_model/checkpoints/best `
    --eval_data data/fine_tuning/combined/eval.json
```

### 评估指标说明

- **NDCG@5 / NDCG@10**: 归一化折损累积增益，衡量排序质量
- **MRR**: 平均倒数排名，衡量第一个相关结果的位置
- **Top-1 准确率**: 最高分预测是否是最优选择

### 预期结果

- NDCG@5 > 0.5: 模型表现良好
- NDCG@5 > 0.7: 模型表现优秀
- MRR > 0.6: 模型能较好地识别最优操作

---

## 步骤 2: 在实际网络上测试

### 单图推理测试

```powershell
# 测试拆解任务
python scripts/inference.py `
    --checkpoint outputs/mixed_model/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task dismantle `
    --budget 10

# 测试构造任务
python scripts/inference.py `
    --checkpoint outputs/mixed_model/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task construct `
    --budget 10
```

### 批量测试（创建脚本）

创建 `scripts/batch_inference.py` 来测试多个图：

```python
import glob
from pathlib import Path

checkpoint = "outputs/mixed_model/checkpoints/best"
test_graphs = glob.glob("data/raw_graphs/syn/*.gml")[:10]  # 测试前10个图

for graph_path in test_graphs:
    print(f"\n测试图: {graph_path}")
    # 调用 inference.py
    # ...
```

---

## 步骤 3: 检查训练结果

### 查看训练日志

```powershell
# 查看训练日志
cat outputs/mixed_model/training.log

# 或使用 PowerShell
Get-Content outputs/mixed_model/training.log -Tail 50
```

### 检查检查点

```powershell
# 列出所有检查点
ls outputs/mixed_model/checkpoints/

# 检查最佳模型
ls outputs/mixed_model/checkpoints/best/
```

### 分析训练曲线

如果保存了训练历史，可以绘制损失曲线：

```python
import json
import matplotlib.pyplot as plt

# 加载训练状态
with open("outputs/mixed_model/checkpoints/state.json") as f:
    state = json.load(f)

# 绘制损失曲线
plt.plot(state["train_loss_history"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
```

---

## 步骤 4: 诊断问题

### 问题 1: 损失为 NaN

**可能原因**:
1. 学习率过大导致梯度爆炸
2. 数据中有异常值
3. 损失计算有问题

**解决方法**:
1. 检查训练日志，找到损失变为 NaN 的步骤
2. 减小学习率（例如从 2e-5 到 1e-5）
3. 添加梯度裁剪（已在配置中设置 `max_grad_norm: 1.0`）
4. 检查数据是否有异常

```powershell
# 重新训练，使用更小的学习率
python scripts/train.py `
    --train_data data/fine_tuning/combined/train.json `
    --eval_data data/fine_tuning/combined/eval.json `
    --output_dir outputs/mixed_model_v2 `
    --phase 1 `
    --epochs 3 `
    --lr 1e-5  # 减小学习率
```

### 问题 2: 评估指标为 0

**可能原因**:
1. 模型未正确加载
2. 数据格式不匹配
3. 候选操作提取有问题

**解决方法**:
1. 检查模型是否正确加载
2. 验证数据格式
3. 检查评估脚本的输出

### 问题 3: 推理结果不合理

**可能原因**:
1. 模型未充分训练
2. 任务类型不匹配
3. OCG 提取有问题

**解决方法**:
1. 增加训练轮数
2. 检查任务类型是否正确
3. 验证 OCG 提取逻辑

---

## 快速验证命令

### 一键验证（推荐）

```powershell
# 1. 评估模型
python scripts/evaluate.py --checkpoint outputs/mixed_model/checkpoints/best

# 2. 测试推理
python scripts/inference.py `
    --checkpoint outputs/mixed_model/checkpoints/best `
    --graph data/raw_graphs/syn/graph_001.gml `
    --task dismantle
```

---

## 下一步

验证完成后，可以：

1. **继续训练**: 如果性能不够好，增加训练轮数或调整超参数
2. **Phase 2 训练**: 如果配置了 GNN，进行联合训练
3. **实际应用**: 在真实网络数据上测试
4. **模型优化**: 根据验证结果调整模型架构

---

## 参考

- [训练设置指南](training_setup_guide.md)
- [工作流程指南](workflow_guide.md)
- [内存优化指南](memory_optimization.md)
