# 训练设置指南

本指南说明如何设置训练环境并开始训练。

## ⚠️ 重要提示

当前代码框架中的 LLM 加载部分（`src/model/fusion_llm.py` 中的 `_load_llm` 和 `_apply_lora` 方法）是**占位符实现**，需要根据你的实际需求进行实现。

## 🔧 前置要求

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备 LLM 模型

你需要选择一个预训练的 LLM 模型，例如：
- `meta-llama/Meta-Llama-3-8B`
- `Qwen/Qwen2-7B`
- `THUDM/chatglm3-6b`
- `mistralai/Mistral-7B-v0.1`

**注意**：需要 HuggingFace 账号和访问权限才能下载某些模型。

### 3. 实现 LLM 加载

编辑 `src/model/fusion_llm.py`，实现 `_load_llm` 和 `_apply_lora` 方法：

```python
def _load_llm(self, device: str) -> None:
    """加载预训练 LLM"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 加载模型和分词器
    self.llm = AutoModelForCausalLM.from_pretrained(
        self.config.llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True  # 如果需要
    )
    
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.config.llm_model_name,
        trust_remote_code=True
    )
    
    # 设置 pad token
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

def _apply_lora(self) -> None:
    """应用 LoRA 适配器"""
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        r=self.config.lora_r,
        lora_alpha=self.config.lora_alpha,
        lora_dropout=self.config.lora_dropout,
        target_modules=self.config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    self.llm = get_peft_model(self.llm, lora_config)
    self.llm.print_trainable_parameters()
```

## 🚀 开始训练

### 步骤 1: 确保数据已生成

```bash
# 检查数据文件
ls data/fine_tuning/combined/train.json
ls data/fine_tuning/combined/eval.json
```

### 步骤 2: 运行训练

```bash
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/mixed_model \
    --phase 1 \
    --epochs 3
```

## 🔍 常见错误和解决方案

### 错误 1: `optimizer got an empty parameter list`

**原因**: 模型没有可训练参数，通常是因为：
1. 模型没有调用 `initialize()`
2. `_load_llm` 方法未实现或抛出异常

**解决方案**:
1. 确保在训练脚本中调用了 `model.initialize(device)`
2. 实现 `_load_llm` 和 `_apply_lora` 方法
3. 检查模型是否正确加载

### 错误 2: `CUDA out of memory`

**原因**: GPU 内存不足

**解决方案**:
1. 减小 batch size: `--batch_size 2` 或 `--batch_size 1`
2. 使用梯度累积: 增加 `gradient_accumulation_steps`
3. 使用更小的模型或使用量化

### 错误 3: `Model not found` 或 `401 Unauthorized`

**原因**: 无法访问 HuggingFace 模型

**解决方案**:
1. 登录 HuggingFace: `huggingface-cli login`
2. 或者手动下载模型到本地，修改模型路径
3. 检查网络连接和访问权限

## 📝 最小化示例

如果你想快速测试代码框架（不使用真实的 LLM），可以创建一个简化版本：

```python
# 在 src/model/fusion_llm.py 中
def _load_llm(self, device: str) -> None:
    """最小化测试版本 - 仅用于代码框架测试"""
    import torch.nn as nn
    
    # 创建一个简单的占位符模型（仅用于测试）
    class DummyLLM(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=4096):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True)
                for _ in range(2)
            ])
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return type('Output', (), {
                'logits': nn.Linear(x.shape[-1], vocab_size)(x),
                'last_hidden_state': x
            })()
    
    self.llm = DummyLLM()
    self.tokenizer = None  # 需要实现一个简单的 tokenizer
    
    print("⚠️ Warning: Using dummy LLM for testing only!")
```

**注意**: 这只是一个占位符，不能用于实际训练。真实训练需要加载预训练的 LLM。
