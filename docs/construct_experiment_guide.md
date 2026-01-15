# Construct 网络实验指南

本指南介绍如何开展 **Construct（构造）** 任务的实验，即通过添加边来最大化网络韧性。

## 📋 目录

1. [任务概述](#任务概述)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型推理](#模型推理)
5. [结果分析](#结果分析)
6. [常见问题](#常见问题)

---

## 任务概述

### Construct 任务目标

**Construct（构造）任务**的目标是：通过添加边来**最大化网络韧性积分 R_res**。

- **操作类型**: 添加边 `(u, v)`
- **优化目标**: 最大化 R_res（韧性面积积分）
- **约束条件**: 
  - 不能添加已存在的边
  - 预算限制（最多添加 `budget` 条边）

### 与 Dismantle 的区别

| 特性 | Dismantle（拆解） | Construct（构造） |
|------|------------------|------------------|
| 操作 | 移除节点 | 添加边 |
| 目标 | 最小化 R_res | 最大化 R_res |
| 候选空间 | O(N) 节点 | O(N²) 边对 |
| 剪枝策略 | 谱梯度节点排序 | 谱梯度边排序 |

---

## 数据准备

### 步骤 1: 生成 Construct 训练数据

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

# 混合数据源（推荐）
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --task_type construct \
    --budget 10 \
    --output_dir data/fine_tuning/construct_all
```

### 步骤 2: 合并训练和评估数据

```bash
# 合并 construct 数据
python scripts/combine_data.py \
    --input_dirs data/fine_tuning/construct_syn data/fine_tuning/construct_true \
    --output_dir data/fine_tuning/construct_combined \
    --train_ratio 0.8
```

### 步骤 3: 验证数据格式

```bash
# 检查数据统计
python scripts/analyze_dataset.py data/fine_tuning/construct_combined/train.json

# 查看样本示例
python scripts/analyze_dataset.py data/fine_tuning/construct_combined/train.json --show_samples 3
```

**预期输出示例**:
```
任务类型: construct
样本数: 1500
平均候选数: 5.2
操作类型分布:
  - add_edge: 100%
```

---

## 模型训练

### 选项 1: 仅训练 Construct 任务

```bash
python scripts/train.py \
    --train_data data/fine_tuning/construct_combined/train.json \
    --eval_data data/fine_tuning/construct_combined/eval.json \
    --output_dir outputs/construct_model \
    --phase 1 \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-5
```

### 选项 2: 混合任务训练（推荐）

同时训练 Dismantle 和 Construct 任务，提高模型泛化能力：

```bash
# 先合并两种任务的数据
python scripts/combine_data.py \
    --input_dirs data/fine_tuning/dismantle_combined data/fine_tuning/construct_combined \
    --output_dir data/fine_tuning/mixed_tasks \
    --train_ratio 0.8

# 训练混合任务模型
python scripts/train.py \
    --train_data data/fine_tuning/mixed_tasks/train.json \
    --eval_data data/fine_tuning/mixed_tasks/eval.json \
    --output_dir outputs/mixed_model \
    --phase 1 \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-5
```

### 训练监控

训练过程中会输出：
- **Loss**: 排序损失（Ranking Loss）
- **Accuracy**: Top-1 准确率（选择最佳操作的比例）
- **NDCG@K**: 归一化折损累积增益

---

## 模型推理

### 基础推理命令

```bash
python scripts/inference.py \
    --checkpoint outputs/construct_model/checkpoints/best \
    --graph data/raw_graphs/true/Colt.gml \
    --task construct \
    --budget 10
```

### 参数说明

- `--checkpoint`: 检查点路径（可以是 `best` 目录，会自动查找最新 epoch）
- `--graph`: 图文件路径（支持 `.gml`, `.graphml`, `.edgelist` 格式）
- `--task`: 任务类型，必须是 `construct`
- `--budget`: 操作预算（最多添加的边数）
- `--config`: 配置文件路径（可选，默认 `configs/default.yaml`）
- `--device`: 设备（`cuda` 或 `cpu`）

### 推理输出示例

```
============================================================
模型推理测试
============================================================

正在加载模型...
找到检查点: outputs/construct_model/checkpoints/epoch_3/model.pt
模型加载完成

正在加载图: data/raw_graphs/true/Colt.gml
图节点数: 153, 边数: 177

初始状态:
  LCC 比例: 1.0000
  R_res: 1.0000

开始推理 (预算: 10 步)...
步骤 1: 添加边 (5, 12)
  LCC: 1.0000, R_res: 1.0000
步骤 2: 添加边 (8, 15)
  LCC: 1.0000, R_res: 1.0000
...

============================================================
推理结果
============================================================
执行的操作数: 10
操作序列: [(5, 12), (8, 15), ...]

初始 -> 最终:
  LCC: 1.0000 -> 1.0000 (变化: 0.0000)
  R_res: 1.0000 -> 1.0000 (变化: 0.0000)

构造效果: R_res 提高了 0.0000
============================================================
```

### 批量实验

对多个图进行批量推理：

```bash
# 创建批量推理脚本
for graph in data/raw_graphs/true/*.gml; do
    echo "Processing $graph..."
    python scripts/inference.py \
        --checkpoint outputs/construct_model/checkpoints/best \
        --graph "$graph" \
        --task construct \
        --budget 10 \
        --output results/construct/$(basename $graph .gml).json
done
```

---

## 结果分析

### 关键指标

1. **R_res 提升量**: `ΔR_res = R_res_final - R_res_initial`
   - 正值表示韧性提升
   - 值越大，构造效果越好

2. **LCC 变化**: 最大连通分量比例
   - Construct 任务中，LCC 通常保持为 1.0（图已连通）
   - 如果 LCC 提升，说明连接了原本分离的组件

3. **边添加效率**: 每条边对 R_res 的平均贡献
   - `效率 = ΔR_res / 添加边数`

### 可视化分析

```python
# 示例：分析构造效果
import json
import matplotlib.pyplot as plt

results = []
for result_file in glob("results/construct/*.json"):
    with open(result_file) as f:
        data = json.load(f)
        results.append({
            'graph': data['graph_name'],
            'r_res_initial': data['initial_r_res'],
            'r_res_final': data['final_r_res'],
            'delta': data['final_r_res'] - data['initial_r_res']
        })

# 绘制 R_res 提升分布
deltas = [r['delta'] for r in results]
plt.hist(deltas, bins=20)
plt.xlabel('ΔR_res')
plt.ylabel('频数')
plt.title('Construct 任务 R_res 提升分布')
plt.show()
```

---

## 常见问题

### Q1: Construct 任务中 R_res 没有提升？

**可能原因**:
1. 图已经高度连通（LCC = 1.0），添加边的影响有限
2. 预算太小，不足以产生显著变化
3. 模型未充分训练

**解决方案**:
- 增加预算（`--budget 20` 或更多）
- 在稀疏图上测试（边数/节点数 < 2）
- 检查训练数据质量

### Q2: 如何选择最佳预算？

**建议**:
- **稀疏图**（边数/节点数 < 1.5）: `budget = 节点数 * 0.1 ~ 0.2`
- **中等密度**（1.5 ~ 2.5）: `budget = 节点数 * 0.05 ~ 0.1`
- **密集图**（> 2.5）: `budget = 节点数 * 0.02 ~ 0.05`

### Q3: Construct 和 Dismantle 可以共享模型吗？

**可以**，但需要：
1. 使用混合任务数据训练
2. 确保训练数据中两种任务比例均衡
3. 推理时明确指定 `--task construct` 或 `--task dismantle`

### Q4: 如何评估构造效果？

**评估方法**:
1. **与基线对比**: 随机添加边 vs 模型选择
2. **与最优解对比**: 使用贪心算法找到近似最优解
3. **消融实验**: 测试不同候选剪枝策略的效果

---

## 进阶实验

### 实验 1: 不同预算下的效果

```bash
for budget in 5 10 15 20; do
    python scripts/inference.py \
        --checkpoint outputs/construct_model/checkpoints/best \
        --graph data/raw_graphs/true/Colt.gml \
        --task construct \
        --budget $budget \
        > results/budget_${budget}.log
done
```

### 实验 2: 不同图类型的对比

```bash
# 合成图 vs 真实图
for graph_type in syn true; do
    for graph in data/raw_graphs/${graph_type}/*.gml; do
        python scripts/inference.py \
            --checkpoint outputs/construct_model/checkpoints/best \
            --graph "$graph" \
            --task construct \
            --budget 10
    done
done
```

### 实验 3: 模型对比

```bash
# 对比不同检查点
for epoch in 1 2 3; do
    python scripts/inference.py \
        --checkpoint outputs/construct_model/checkpoints/epoch_${epoch} \
        --graph data/raw_graphs/true/Colt.gml \
        --task construct \
        --budget 10
done
```

---

## 总结

Construct 任务的关键点：

1. ✅ **数据准备**: 确保训练数据包含足够的 construct 样本
2. ✅ **模型训练**: 可以使用混合任务训练提高泛化能力
3. ✅ **推理设置**: 明确指定 `--task construct`，合理设置预算
4. ✅ **结果分析**: 关注 R_res 提升量，而非绝对值

**下一步**: 尝试在不同类型的网络上进行实验，分析模型在不同场景下的表现！
