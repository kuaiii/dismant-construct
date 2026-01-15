#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载脚本

自动下载 HuggingFace 模型到本地缓存。

用法:
    python scripts/download_model.py
    python scripts/download_model.py --model_name Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

# 设置镜像站点（如果未设置）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("ℹ️  已设置 HuggingFace 镜像站点: https://hf-mirror.com")


def download_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """下载模型到本地缓存"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 缺少依赖库")
        print("请安装: pip install transformers huggingface-hub")
        return False
    
    print("=" * 60)
    print("模型下载工具")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"镜像站点: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print()
    
    try:
        print("步骤 1/2: 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ 分词器下载完成")
        print()
        
        print("步骤 2/2: 下载模型权重...")
        print("⚠️  这可能需要几分钟到几十分钟，取决于网络速度")
        print("   模型大小约 3GB (Qwen2.5-1.5B)")
        print()
        
        # 下载模型（只下载配置，不加载到内存）
        snapshot_download(
            repo_id=model_name,
            local_files_only=False,
            resume_download=True
        )
        
        print()
        print("✅ 模型下载完成！")
        print()
        print("模型已保存到 HuggingFace 缓存目录:")
        print(f"  Windows: C:\\Users\\{os.getenv('USERNAME')}\\.cache\\huggingface\\hub")
        print(f"  Linux/Mac: ~/.cache/huggingface/hub")
        print()
        print("现在可以运行训练脚本了:")
        print("  python scripts/train.py --train_data data/fine_tuning/combined/train.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 模型名称错误")
        print("3. 访问权限问题")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 尝试使用镜像站点: set HF_ENDPOINT=https://hf-mirror.com")
        print("3. 运行网络测试: python scripts/test_network.py")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="要下载的模型名称"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model_name)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
