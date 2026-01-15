#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络问题修复脚本

自动设置 HuggingFace 镜像和环境变量，解决网络连接问题。

用法:
    python scripts/fix_network.py
"""

import os
import sys


def setup_hf_mirror():
    """设置 HuggingFace 镜像站点"""
    # 设置镜像站点（中国用户推荐）
    mirror_url = "https://hf-mirror.com"
    os.environ['HF_ENDPOINT'] = mirror_url
    
    # 清除可能冲突的代理设置（如果需要）
    # os.environ.pop('HTTP_PROXY', None)
    # os.environ.pop('HTTPS_PROXY', None)
    
    print("=" * 60)
    print("网络环境配置")
    print("=" * 60)
    print(f"✅ HF_ENDPOINT = {mirror_url}")
    print()


def test_connection():
    """测试网络连接"""
    import requests
    
    mirrors = [
        ("hf-mirror.com (推荐)", "https://hf-mirror.com"),
        ("huggingface.co (官方)", "https://huggingface.co"),
    ]
    
    print("测试网络连接...")
    print()
    
    for name, url in mirrors:
        try:
            response = requests.get(url, timeout=5)
            status = "✅ 可访问" if response.status_code == 200 else f"⚠️ 状态码: {response.status_code}"
            print(f"{name}: {status}")
        except Exception as e:
            print(f"{name}: ❌ 无法访问 - {str(e)[:50]}")
    print()


def show_current_settings():
    """显示当前环境变量设置"""
    print("当前环境变量设置:")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(未设置)')}")
    print(f"  HTTP_PROXY: {os.environ.get('HTTP_PROXY', '(未设置)')}")
    print(f"  HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '(未设置)')}")
    print()


def main():
    print("\n网络问题修复工具")
    print("=" * 60)
    print()
    
    # 显示当前设置
    show_current_settings()
    
    # 设置镜像
    setup_hf_mirror()
    
    # 测试连接
    try:
        test_connection()
    except ImportError:
        print("⚠️  requests 未安装，跳过连接测试")
        print("   安装: pip install requests")
    
    print("=" * 60)
    print("配置完成！")
    print()
    print("现在可以运行测试脚本:")
    print("  python scripts/test_model_loading.py")
    print()
    print("注意: 这些设置仅在当前 Python 进程中有效。")
    print("如果关闭终端后需要重新设置，请:")
    print("  1. 运行此脚本，或")
    print("  2. 手动设置环境变量")
    print()


if __name__ == "__main__":
    main()
