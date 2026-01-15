# 低内存训练脚本（适用于 RTX 3060 12GB）
# PowerShell 版本

# 设置环境变量
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 清理 GPU 内存
Write-Host "清理 GPU 内存..." -ForegroundColor Yellow
python scripts/clear_gpu_memory.py

# 开始训练（使用最小内存配置）
Write-Host "`n开始训练（低内存模式）..." -ForegroundColor Green
python scripts/train.py `
    --train_data data/fine_tuning/combined/train.json `
    --eval_data data/fine_tuning/combined/eval.json `
    --output_dir outputs/mixed_model `
    --phase 1 `
    --epochs 3 `
    --batch_size 1
