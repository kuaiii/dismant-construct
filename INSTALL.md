# 安装指南

## 快速安装（测试用）

### 1. 安装核心依赖

```bash
# 安装 transformers 和 peft（必需）
pip install transformers>=4.35.0 peft>=0.6.0

# 如果需要加速下载，可以安装 huggingface-hub
pip install huggingface-hub
```

### 2. 安装完整依赖（推荐）

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import transformers; import peft; print('✅ 安装成功')"
```

## HuggingFace 设置（可选但推荐）

### 注册账号

1. 访问 https://huggingface.co/join
2. 注册账号并验证邮箱

### 登录（用于下载需要权限的模型）

```bash
# 安装 huggingface-hub（如果还没安装）
pip install huggingface-hub

# 登录
huggingface-cli login

# 输入你的 token（在 https://huggingface.co/settings/tokens 获取）
```

**注意**：
- 对于 Qwen 和 TinyLlama 等开源模型，通常不需要登录
- 对于 LLaMA 等模型，需要申请访问权限

## 测试模型加载

```bash
# 测试默认模型（Qwen2.5-1.5B）
python scripts/test_model_loading.py

# 或测试其他模型
python scripts/test_model_loading.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 常见问题

### Q: pip 安装慢怎么办？

**A**: 使用国内镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers peft
```

### Q: CUDA 相关错误？

**A**: 确保 PyTorch 支持 CUDA：

```bash
# 检查 CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

如果不支持，需要重新安装 PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q: 模型下载慢（中国用户）？

**A**: 使用镜像站点：

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```
