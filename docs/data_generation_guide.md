# 数据生成指南

本指南说明如何从不同的数据源生成训练数据。

> **完整工作流程请参考**: [工作流程指南](workflow_guide.md)

## 数据源选项

### 1. 生成 BA/ER 图（原有方式）

```bash
python scripts/generate_data.py \
    --data_source generate \
    --graph_type ba \
    --num_graphs 100 \
    --min_nodes 50 \
    --max_nodes 200 \
    --output_dir data/fine_tuning
```

### 2. 从合成网络数据生成（syn 目录）

```bash
python scripts/generate_data.py \
    --data_source syn \
    --num_graphs 100 \
    --raw_graphs_dir data/raw_graphs \
    --output_dir data/fine_tuning \
    --budget 10
```

这会从 `data/raw_graphs/syn/` 目录随机选择 100 个 .gml 文件生成训练数据。

### 3. 从真实网络数据生成（true 目录）

```bash
python scripts/generate_data.py \
    --data_source true \
    --num_graphs 50 \
    --raw_graphs_dir data/raw_graphs \
    --output_dir data/fine_tuning \
    --budget 10
```

这会从 `data/raw_graphs/true/` 目录随机选择 50 个文件（.gml 或 .graphml）生成训练数据。

### 4. 混合使用（syn + true）

```bash
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --raw_graphs_dir data/raw_graphs \
    --output_dir data/fine_tuning \
    --budget 10
```

这会混合使用 `syn/` 和 `true/` 目录中的图文件。

## 参数说明

### 必需参数

- `--data_source`: 数据来源
  - `generate`: 生成 BA/ER 图
  - `syn`: 使用合成网络数据
  - `true`: 使用真实网络数据
  - `all`: 混合使用 syn 和 true

### 可选参数

- `--raw_graphs_dir`: 原始图数据目录（默认：`data/raw_graphs`）
- `--num_graphs`: 使用的图数量（默认：100）
- `--budget`: 每个图的操作预算（默认：10）
- `--task_type`: 任务类型（`dismantle`, `construct`, `both`，默认：`dismantle`）
- `--output_dir`: 输出目录（默认：`data/fine_tuning`）
- `--seed`: 随机种子（默认：42）
- `--split_ratio`: 训练集比例（默认：0.9）
- `--min_graph_size`: 最小图大小，小于此大小的图会被跳过（默认：20）

### 仅在 `data_source=generate` 时有效

- `--graph_type`: 图类型（`ba`, `er`, `mixed`，默认：`ba`）
- `--min_nodes`: 最小节点数（默认：50）
- `--max_nodes`: 最大节点数（默认：200）

## 输出

脚本会在 `output_dir` 目录下生成：

- `train.json`: 训练集数据
- `eval.json`: 验证集数据
- `data_config.json`: 数据生成配置信息

每个训练样本包含：

- `id`: 样本 ID
- `meta`: 元数据（任务类型、图类型、节点数等）
- `conversations`: 对话格式的数据
- `auxiliary_labels`: 用于 ListMLE 训练的标签

## 示例

### 从合成网络生成 100 个样本

```bash
python scripts/generate_data.py \
    --data_source syn \
    --num_graphs 100 \
    --budget 10 \
    --task_type dismantle \
    --output_dir data/fine_tuning/syn_dismantle
```

### 从真实网络生成 50 个样本（混合任务）

```bash
python scripts/generate_data.py \
    --data_source true \
    --num_graphs 50 \
    --budget 15 \
    --task_type both \
    --output_dir data/fine_tuning/true_mixed
```

### 混合数据源生成 200 个样本

```bash
python scripts/generate_data.py \
    --data_source all \
    --num_graphs 200 \
    --budget 10 \
    --min_graph_size 30 \
    --output_dir data/fine_tuning/mixed_all
```

## 注意事项

1. **图文件格式**：支持 `.gml` 和 `.graphml` 格式
2. **图预处理**：
   - 自动转换为无向图
   - 只保留最大连通分量
   - 跳过节点数小于 `min_graph_size` 的图
3. **预算设置**：`budget` 不能超过图节点数的一半
4. **内存使用**：处理大量图时注意内存占用
5. **文件路径**：确保 `raw_graphs_dir` 目录存在且包含对应子目录
