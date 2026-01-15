# GPU 内存优化指南

## 问题描述

在 RTX 3060 12GB 上训练时可能遇到 CUDA 内存不足的错误。

## 解决方案

### 1. 立即解决方案

#### 步骤 1: 清理 GPU 内存

```powershell
# 运行清理脚本
python scripts/clear_gpu_memory.py
```

或者手动清理：

```powershell
# 检查 GPU 使用情况
nvidia-smi

# 如果有其他进程占用 GPU，先关闭它们
# 然后运行训练脚本
```

#### 步骤 2: 使用优化后的配置

配置文件 `configs/default.yaml` 已经优化：
- `batch_size: 1` (从 4 减小到 1)
- `gradient_accumulation_steps: 8` (从 4 增加到 8，保持有效批大小)
- `max_length: 1024` (从 2048 减小到 1024)

#### 步骤 3: 设置环境变量

```powershell
# 设置 PyTorch 内存分配策略
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 然后运行训练
python scripts/train.py --train_data data/fine_tuning/combined/train.json --eval_data data/fine_tuning/combined/eval.json --output_dir outputs/mixed_model --phase 1 --epochs 3
```

### 2. 进一步优化

如果仍然内存不足，可以尝试：

#### 选项 A: 更小的批大小和更长的序列

```yaml
# configs/default.yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16  # 增加累积步数

data:
  loading:
    max_length: 512  # 进一步减小序列长度
```

#### 选项 B: 使用 CPU Offloading

修改 `src/model/fusion_llm.py` 中的 `_load_llm` 方法，添加 CPU offloading：

```python
load_kwargs.update({
    "device_map": "auto",
    "max_memory": {0: "8GB", "cpu": "30GB"},  # 部分层放到 CPU
})
```

#### 选项 C: 使用 8-bit 量化

安装 `bitsandbytes`:

```powershell
pip install bitsandbytes
```

然后在 `_load_llm` 中添加：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

load_kwargs["quantization_config"] = quantization_config
```

### 3. 监控内存使用

在训练过程中监控 GPU 内存：

```powershell
# 在另一个终端窗口运行
watch -n 1 nvidia-smi
```

### 4. 常见问题

#### Q: 为什么显示 "25.84 GiB is allocated" 但 GPU 只有 12GB？

A: 这可能是显示错误，或者有多个进程/模型实例占用内存。运行 `nvidia-smi` 检查实际使用情况。

#### Q: 如何确认内存优化是否生效？

A: 在训练开始时查看输出：
```
GPU 内存使用: X.XX GB
```

如果超过 10GB，需要进一步优化。

#### Q: 训练速度太慢怎么办？

A: 这是内存优化的权衡。可以：
1. 使用更小的模型（如 TinyLlama）
2. 减小 `max_length`
3. 使用梯度检查点（需要修改模型代码）

### 5. 推荐的配置组合

#### 配置 1: 最小内存使用（推荐用于 12GB GPU）

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  fp16: true

data:
  loading:
    max_length: 512
```

#### 配置 2: 平衡配置

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  fp16: true

data:
  loading:
    max_length: 1024
```

### 6. 训练命令示例

```powershell
# 设置环境变量
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 清理内存
python scripts/clear_gpu_memory.py

# 开始训练
python scripts/train.py `
    --train_data data/fine_tuning/combined/train.json `
    --eval_data data/fine_tuning/combined/eval.json `
    --output_dir outputs/mixed_model `
    --phase 1 `
    --epochs 3 `
    --batch_size 1
```
