# 统一框架指南 - Dismant & Construct

本文档描述了 Dismant（拆解）和 Construct（构造）两种任务的统一框架设计。

## 1. 框架概述

### 1.1 任务定义

| 任务 | 目标 | 操作 | 评价指标 | 符号函数 σ |
|------|------|------|----------|------------|
| **Dismant** | 最快瓦解网络 | 移除节点 | LCC, R_res (越小越好) | -1 |
| **Construct** | 最大化韧性 | 添加边 | R_tar, R_ran (越大越好) | +1 |

### 1.2 核心指标

1. **LCC (Largest Connected Component)**: 最大连通分量占初始节点数的比例
2. **R_res (Resilience Integral)**: 韧性积分，LCC 曲线下的面积
   - 计算方式: `R_res = ∫₀¹ LCC(q) dq`
   - Dismant: 越小表示拆解效果越好
   - Construct: 越大表示网络韧性越好

3. **R_tar (Targeted Attack Resilience)**: 目标攻击（HDA）下的韧性
4. **R_ran (Random Attack Resilience)**: 随机攻击下的韧性（多次平均）

## 2. 统一数据格式

### 2.1 训练样本格式

```json
{
  "id": "train_dismantle_syn_0001_00",
  "meta": {
    "task": "dismantle",        // 或 "construct"
    "budget_step": "1/10",
    "sign": -1,                 // dismantle=-1, construct=+1
    "graph_type": "syn",
    "num_nodes": 100,
    "data_source": "syn"
  },
  "conversations": [
    {"from": "system", "value": "系统提示..."},
    {"from": "user", "value": "OCG 描述..."},
    {"from": "assistant", "value": "推理和排序结果..."}
  ],
  "auxiliary_labels": {
    "op_01": 0.95,
    "op_02": 0.40,
    "op_03": 0.15
  }
}
```

### 2.2 关键字段说明

- `sign`: 符号函数，用于统一优化目标
  - Dismant: `sign = -1`，目标是最小化 R_res
  - Construct: `sign = +1`，目标是最大化 R_res

- `auxiliary_labels`: 用于 ListMLE 排序学习的真实影响分数
  - Dismant: 节点移除后的 LCC 下降量
  - Construct: 边添加后的 LCC/连通性增益

## 3. 评估流程

### 3.1 Dismant 评估

```
输入: 初始图 G, 攻击序列 [n1, n2, ..., nk]
输出: R_res, LCC 曲线, 崩溃点

流程:
1. 按顺序移除节点
2. 每步记录 LCC 比例
3. 计算韧性积分 R_res = ∫ LCC(q) dq
4. 找到崩溃点 (LCC < 20%)
```

### 3.2 Construct 评估

```
输入: 原始图 G, 重构图 G', 添加的边列表
输出: R_tar, R_ran, 韧性提升比例

流程:
1. 对重构图 G' 执行 HDA 攻击 -> 计算 R_tar
2. 对重构图 G' 执行 Random 攻击 (多次) -> 计算 R_ran
3. 对比原始图的 R_tar_original, R_ran_original
4. 计算韧性提升: Δ = (R_tar - R_tar_original) / R_tar_original
```

## 4. 使用方法

### 4.1 快速验证

```bash
# 快速跑通整个流程
python scripts/quick_validate.py

# 跳过训练，仅验证评估逻辑
python scripts/quick_validate.py --skip_training
```

### 4.2 数据生成

```bash
# 生成混合数据（dismantle + construct）
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --task_type both \
    --output_dir data/fine_tuning
```

### 4.3 统一评估

```bash
# 评估 Dismant 任务
python scripts/unified_evaluate.py \
    --task dismant \
    --graph data/raw_graphs/true/Colt.gml \
    --output_dir results/dismant

# 评估 Construct 任务
python scripts/unified_evaluate.py \
    --task construct \
    --graph data/raw_graphs/true/Colt.gml \
    --edge_budget 10 \
    --output_dir results/construct

# 完整评估（Dismant + Construct）
python scripts/unified_evaluate.py \
    --task both \
    --graph data/raw_graphs/true/Colt.gml \
    --output_dir results/full
```

### 4.4 批量评估

```bash
# 批量评估目录下所有图
python scripts/unified_evaluate.py \
    --task both \
    --graph_dir data/raw_graphs/true \
    --output_dir results/batch
```

## 5. API 使用

### 5.1 评估 API

```python
from src.evaluation import UnifiedEvaluator, evaluate_dismant, evaluate_construct

# 创建评估器
evaluator = UnifiedEvaluator(
    collapse_threshold=0.2,
    random_runs=10,
    random_seed=42,
)

# Dismant 评估
dismant_result = evaluator.evaluate_dismant(
    graph=G,
    attack_sequence=nodes_to_remove,
    method_name="LLM",
)
print(f"R_res: {dismant_result.r_res}")
print(f"崩溃点: {dismant_result.collapse_fraction}")

# Construct 评估
construct_result = evaluator.evaluate_construct(
    original_graph=G_original,
    reconstructed_graph=G_reconstructed,
    added_edges=edges_added,
    method_name="LLM",
)
print(f"R_tar: {construct_result.r_tar}")
print(f"R_ran: {construct_result.r_ran}")
print(f"提升: {construct_result.r_improvement:.2%}")
```

### 5.2 与基线对比

```python
# Dismant 与基线对比
result = evaluator.evaluate_dismant_with_baselines(
    graph=G,
    attack_sequence=my_sequence,
    method_name="MyMethod",
    include_hda=True,
    include_random=True,
)

print("My Method:", result.dismant_result.r_res)
print("HDA Baseline:", result.baseline_results["HDA"]["r_res"])
print("Random Baseline:", result.baseline_results["Random"]["r_res"])

# Construct 与基线对比
result = evaluator.evaluate_construct_with_baselines(
    original_graph=G,
    reconstructed_graph=G_new,
    added_edges=edges,
    include_random_construct=True,
    include_degree_construct=True,
)
```

## 6. 输出结构

### 6.1 Dismant 输出

```
results/dismant/GraphName/20260115_120000/
├── dismant_comparison.png    # LCC 曲线对比图
└── results.json              # 详细结果数据
```

### 6.2 Construct 输出

```
results/construct/GraphName/20260115_120000/
├── construct_comparison.png  # HDA/Random 攻击对比图
└── results.json              # 详细结果数据
```

### 6.3 结果 JSON 格式

```json
{
  "graph_name": "Colt",
  "graph_info": {"nodes": 153, "edges": 177},
  "timestamp": "2026-01-15T12:00:00",
  "dismant": {
    "HDA": {"r_res": 0.2345, "collapse_fraction": 0.15},
    "Random": {"r_res": 0.4567, "collapse_fraction": 0.35}
  },
  "construct": {
    "Original": {"r_tar": 0.3456, "r_ran": 0.5678},
    "RandomConstruct": {"r_tar": 0.3789, "r_improvement": 0.096},
    "DegreeConstruct": {"r_tar": 0.3901, "r_improvement": 0.129}
  }
}
```

## 7. 框架扩展

### 7.1 添加新的攻击策略

继承 `BaseAttack` 类：

```python
from src.attack.base import BaseAttack

class MyAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="MyAttack")
    
    def select_node(self, graph, **kwargs):
        # 实现节点选择逻辑
        nodes = list(graph.nodes())
        return nodes[0] if nodes else None
```

### 7.2 添加新的构造策略

在 `UnifiedEvaluator` 中添加方法：

```python
def _my_construct_strategy(self, graph, budget):
    g = graph.copy()
    added = []
    # 实现边添加逻辑
    return g, added
```

## 8. 常见问题

### Q1: Dismant 和 Construct 的优化目标如何统一？

使用符号函数 σ：
- 统一损失: `L = σ * ranking_loss`
- Dismant (σ=-1): 最小化 R_res
- Construct (σ=+1): 最大化 R_res

### Q2: 如何解释 R_res 指标？

- R_res ∈ [0, 1]
- Dismant: R_res 越小，网络被瓦解得越快
- Construct: R_res 越大，网络韧性越好

### Q3: Construct 为什么要用 HDA 和 Random 两种攻击评估？

- HDA (R_tar): 测试网络对目标攻击的抵抗力
- Random (R_ran): 测试网络对随机故障的容错能力
- 好的重构应该同时提升两者
