# 网络问题排查指南

## 问题：ProxyError 或无法连接到 HuggingFace

这是常见的网络连接问题，通常出现在中国用户或企业网络中。

## 🔧 解决方案

### 方案 1: 使用 HuggingFace 镜像站点（推荐）

#### 方法 A: 设置环境变量

**Windows PowerShell:**
```powershell
# 临时设置（当前会话）
$env:HF_ENDPOINT="https://hf-mirror.com"

# 永久设置（用户级别）
[System.Environment]::SetEnvironmentVariable('HF_ENDPOINT', 'https://hf-mirror.com', 'User')
```

**Windows CMD:**
```cmd
# 临时设置
set HF_ENDPOINT=https://hf-mirror.com

# 永久设置
setx HF_ENDPOINT "https://hf-mirror.com"
```

**Linux/Mac:**
```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

#### 方法 B: 在代码中设置

创建 `scripts/setup_hf_mirror.py`：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

在运行任何脚本前先运行：
```bash
python scripts/setup_hf_mirror.py
```

### 方案 2: 禁用代理

如果你在代理环境中但不需要代理：

**Windows PowerShell:**
```powershell
# 清除代理设置
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:http_proxy=""
$env:https_proxy=""
```

**在代码中禁用代理:**
```python
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
```

### 方案 3: 配置正确的代理

如果需要使用代理：

```python
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'
```

### 方案 4: 使用离线下载

如果网络完全无法访问，可以：

1. **使用其他设备下载模型**：
   - 在有网络的设备上运行模型下载
   - 将模型文件夹复制到本地
   
2. **手动下载**：
   - 访问 https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct
   - 手动下载所需文件
   - 将模型放在 `./models/Qwen2.5-1.5B-Instruct/` 目录

3. **使用本地路径**：
   ```python
   model_name = "./models/Qwen2.5-1.5B-Instruct"  # 本地路径
   ```

### 方案 5: 禁用 SSL 验证（不推荐，仅测试用）

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## 🚀 快速修复脚本

创建并运行以下脚本来自动设置：

```bash
# Windows PowerShell
python scripts/fix_network.py

# 或手动设置
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/test_model_loading.py
```

## ✅ 验证连接

测试是否能连接到镜像站点：

```python
import requests
try:
    response = requests.get("https://hf-mirror.com", timeout=5)
    print(f"✅ 镜像站点连接成功: {response.status_code}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
```

## 📝 常见镜像站点

- **hf-mirror.com** (推荐，中国用户)
- **hf.co** (官方站点)
- **huggingface.co** (官方站点)

## 🔍 调试步骤

1. **检查环境变量**:
   ```python
   import os
   print("HF_ENDPOINT:", os.environ.get('HF_ENDPOINT'))
   print("HTTP_PROXY:", os.environ.get('HTTP_PROXY'))
   print("HTTPS_PROXY:", os.environ.get('HTTPS_PROXY'))
   ```

2. **测试网络连接**:
   ```python
   import requests
   try:
       r = requests.get("https://hf-mirror.com", timeout=10)
       print("✅ 可以访问镜像站点")
   except Exception as e:
       print(f"❌ 无法访问: {e}")
   ```

3. **查看详细错误**:
   在代码中添加更详细的错误信息输出。
