# 快速测试指南（RTX 3060 12GB）

本指南帮助你在 RTX 3060 12GB 显卡上快速设置和测试模型。

## 📋 推荐模型

对于 RTX 3060 12GB，推荐以下小模型（按推荐顺序）：

1. **Qwen2.5-1.5B-Instruct** (推荐 ⭐)
   - 模型：`Qwen/Qwen2.5-1.5B-Instruct`
   - 参数量：1.5B
   - 显存需求：~3GB (FP16)
   - 支持中文，性能优秀

2. **TinyLlama-1.1B**
   - 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - 参数量：1.1B
   - 显存需求：~2.5GB (FP16)
   - 轻量级，适合测试

3. **ChatGLM3-6B** (如果内存足够)
   - 模型：`THUDM/chatglm3-6b`
   - 参数量：6B
   - 显存需求：~12GB (FP16)，需要使用量化
   - 支持中文

## 🔧 安装步骤

### 步骤 1: 安装依赖

```bash
# 安装 transformers 和 peft
pip install transformers>=4.35.0 peft>=0.6.0

# 或者安装所有依赖
pip install -r requirements.txt
```

### 步骤 2: 登录 HuggingFace（可选但推荐）

```bash
# 安装 huggingface-hub
pip install huggingface-hub

# 登录（需要 HuggingFace 账号）
huggingface-cli login

# 或者设置 token
# export HF_TOKEN=your_token_here
```

**注意**：
- 某些模型（如 LLaMA）需要申请访问权限
- Qwen 和 TinyLlama 通常不需要特殊权限
- 如果没有账号，可以注册：https://huggingface.co/join

## 📝 实现 LLM 加载代码

编辑 `src/model/fusion_llm.py`，找到 `_load_llm` 和 `_apply_lora` 方法，替换为以下代码：

### 方法 1: 使用 Qwen2.5-1.5B（推荐）

```python
def _load_llm(self, device: str) -> None:
    """加载预训练 LLM"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"正在加载模型: {self.config.llm_model_name}")
    print(f"设备: {device}")
    
    # 设置数据类型
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 加载模型
    self.llm = AutoModelForCausalLM.from_pretrained(
        self.config.llm_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载分词器
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.config.llm_model_name,
        trust_remote_code=True
    )
    
    # 设置 pad token
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    # 移动到设备（如果使用 CPU）
    if device == "cpu":
        self.llm = self.llm.to(device)
    
    print("模型加载完成!")

def _apply_lora(self) -> None:
    """应用 LoRA 适配器"""
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("正在应用 LoRA 适配器...")
    
    # 确定目标模块（根据模型架构调整）
    model_name_lower = self.config.llm_model_name.lower()
    if "qwen" in model_name_lower:
        # Qwen 模型的模块名称
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
        # LLaMA 模型的模块名称
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "chatglm" in model_name_lower:
        # ChatGLM 模型的模块名称
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        # 默认使用配置中的模块
        target_modules = self.config.lora_target_modules
    
    lora_config = LoraConfig(
        r=self.config.lora_r,
        lora_alpha=self.config.lora_alpha,
        lora_dropout=self.config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    self.llm = get_peft_model(self.llm, lora_config)
    self.llm.print_trainable_parameters()
    print("LoRA 适配器应用完成!")
```

## ⚙️ 更新配置文件

编辑 `configs/default.yaml`，将模型名称改为：

```yaml
model:
  llm:
    model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # 推荐用于测试
    # 或者使用: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## 🚀 测试步骤

### 步骤 1: 测试模型加载

创建一个测试脚本 `scripts/test_model_loading.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试模型加载"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.fusion_llm import ResilienceLLM, ModelConfig

def test_model_loading():
    config = ModelConfig(
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        use_lora=True,
        lora_r=8
    )
    
    print("创建模型...")
    model = ResilienceLLM(config)
    
    print("初始化模型...")
    try:
        model.initialize(device="cuda")
        print("✅ 模型加载成功!")
        
        # 打印可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
```

运行测试：

```bash
python scripts/test_model_loading.py
```

### 步骤 2: 开始训练

如果模型加载成功，可以开始训练：

```bash
python scripts/train.py \
    --train_data data/fine_tuning/combined/train.json \
    --eval_data data/fine_tuning/combined/eval.json \
    --output_dir outputs/test_model \
    --phase 1 \
    --epochs 1 \
    --batch_size 2  # 小 batch size 适合测试
```

## 🔍 常见问题

### Q1: CUDA out of memory

**解决方案**:
1. 减小 batch size: `--batch_size 1`
2. 使用更小的模型（如 TinyLlama）
3. 使用量化（需要 bitsandbytes）：
   ```python
   from transformers import BitsAndBytesConfig
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   ```

### Q2: 模型下载慢

**解决方案**:
1. 使用镜像站点（中国用户）
2. 设置环境变量：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 或者手动下载到本地

### Q3: 找不到模型

**解决方案**:
1. 检查模型名称是否正确
2. 确认 HuggingFace 访问权限
3. 检查网络连接
4. 尝试使用其他模型（如 TinyLlama）

## 📊 显存使用估算

对于 RTX 3060 12GB：

| 模型 | 参数量 | FP16 显存 | 推荐 batch size |
|------|--------|-----------|----------------|
| Qwen2.5-1.5B | 1.5B | ~3GB | 4-8 |
| TinyLlama-1.1B | 1.1B | ~2.5GB | 4-8 |
| ChatGLM3-6B (4bit) | 6B | ~4GB | 2-4 |

## ✅ 验证清单

- [ ] 安装 transformers 和 peft
- [ ] 登录 HuggingFace（如果需要）
- [ ] 实现 `_load_llm` 方法
- [ ] 实现 `_apply_lora` 方法
- [ ] 更新配置文件中的模型名称
- [ ] 测试模型加载成功
- [ ] 检查可训练参数数量
- [ ] 开始小规模训练测试
