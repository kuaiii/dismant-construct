# 开始训练完整指南

本指南提供从模型下载到开始训练的完整步骤。

## 📋 前置检查清单

- [x] 数据集已生成（train.json 和 eval.json）
- [x] 环境已配置（transformers, peft 已安装）
- [ ] 模型已下载（下一步）
- [ ] 训练脚本已准备好

## 🚀 完整训练流程

### 步骤 1: 下载模型（首次运行）

模型会在首次使用时自动下载，但建议先手动下载以避免训练中断：

```bash
# 方法 1: 使用下载脚本（推荐）
python scripts/download_model.py

# 方法 2: 使用测试脚本（会自动下载）
python scripts/test_model_loading.py
```

**注意**：
- 模型会下载到 HuggingFace 缓存目录（约 3GB）
- 下载时间取决于网络速度（通常 5-30 分钟）
- 如果网络有问题，参考 `docs/network_troubleshooting.md`

### 步骤 2: 验证模型加载

确保模型可以正常加载：

```bash
python scripts/test_model_loading.py
```

如果看到 "✅ 所有测试通过！"，说明模型已准备好。

### 步骤 3: 检查数据

确认训练数据已准备好：

```bash
# 检查数据文件
python scripts/analyze_dataset.py data/fine_tuning/combined/train.json
```

### 步骤 4: 开始训练

#### 基础训练命令

```bash
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/mixed_model \
    --phase 1 \
    --epochs 3
```

#### 完整参数示例

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/mixed_model \
    --phase 1 \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-5
```

#### 小规模测试（推荐先运行）

```bash
# 只训练 1 个 epoch，小 batch size
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/test_run \
    --phase 1 \
    --epochs 1 \
    --batch_size 1
```

### 步骤 5: 监控训练

训练过程中会显示：
- 当前 epoch 和步数
- 损失值
- 学习率
- 评估指标（NDCG, MRR 等）

日志文件保存在：`outputs/mixed_model/training.log`

### 步骤 6: 检查训练结果

训练完成后，检查点保存在：
```
outputs/mixed_model/
├── checkpoints/
│   ├── best/          # 最佳模型
│   ├── epoch_1/        # 每个 epoch 的检查点
│   └── step_500/       # 定期保存的检查点
└── training.log        # 训练日志
```

## ⚙️ 训练参数说明

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--phase` | 训练阶段 | 1 (LLM only) 或 2 (Joint) |
| `--epochs` | 训练轮数 | 3-5 |
| `--batch_size` | 批大小 | 2-4 (RTX 3060 12GB) |
| `--lr` | 学习率 | 2e-5 (默认) |
| `--fp16` | 混合精度 | True (节省显存) |

### 显存优化

如果遇到 CUDA OOM 错误：

```bash
# 减小 batch size
--batch_size 1

# 增加梯度累积
# 在 configs/default.yaml 中设置:
# training.gradient_accumulation_steps: 8
```

## 🔍 常见问题

### Q1: 训练很慢怎么办？

**A**: 
- 减小 batch size
- 使用更小的模型
- 减少训练数据量（测试用）

### Q2: 显存不足

**A**:
```bash
# 使用最小配置
--batch_size 1
# 在配置文件中增加 gradient_accumulation_steps
```

### Q3: 如何恢复训练？

**A**:
```bash
python scripts/train.py \
    --resume outputs/mixed_model/checkpoints/step_500 \
    --train_data data/fine_tuning/combined/train.json \
    ...
```

### Q4: 训练中断了怎么办？

**A**: 检查点会自动保存，使用 `--resume` 参数继续训练。

## 📊 训练监控

### 查看训练日志

```bash
# Windows PowerShell
Get-Content outputs/mixed_model/training.log -Tail 50

# Linux/Mac
tail -f outputs/mixed_model/training.log
```

### 检查 GPU 使用情况

```bash
# Windows (需要 nvidia-smi)
nvidia-smi -l 1

# 或在 Python 中
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## ✅ 训练成功标志

训练成功时，你应该看到：

1. ✅ 模型加载成功
2. ✅ 数据加载成功
3. ✅ 训练循环正常运行
4. ✅ 损失值逐渐下降
5. ✅ 评估指标（NDCG, MRR）逐渐提升
6. ✅ 检查点正常保存

## 🎯 下一步

训练完成后：
1. 评估模型性能
2. 在测试集上验证
3. 调整超参数（如果需要）
4. 进行 Phase 2 训练（如果使用 GNN）
