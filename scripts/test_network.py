#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络连接测试脚本

测试是否能访问 HuggingFace 及其镜像站点。

用法:
    python scripts/test_network.py
"""

import os
import sys
from pathlib import Path

# 尝试导入 requests
try:
    import requests
except ImportError:
    print("❌ requests 未安装")
    print("请安装: pip install requests")
    sys.exit(1)


def test_url(name, url, timeout=5):
    """测试 URL 是否可访问"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name}: 可访问 (状态码: {response.status_code})")
            return True
        else:
            print(f"⚠️  {name}: 返回状态码 {response.status_code}")
            return False
    except requests.exceptions.ProxyError as e:
        print(f"❌ {name}: 代理错误 - {str(e)[:100]}")
        return False
    except requests.exceptions.SSLError as e:
        print(f"❌ {name}: SSL 错误 - {str(e)[:100]}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {name}: 连接超时")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ {name}: 连接错误 - {str(e)[:100]}")
        return False
    except Exception as e:
        print(f"❌ {name}: 未知错误 - {str(e)[:100]}")
        return False


def main():
    print("=" * 60)
    print("HuggingFace 网络连接测试")
    print("=" * 60)
    print()
    
    # 显示当前环境变量
    print("当前环境变量:")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(未设置)')}")
    print(f"  HTTP_PROXY: {os.environ.get('HTTP_PROXY', '(未设置)')}")
    print(f"  HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '(未设置)')}")
    print()
    
    # 测试各个站点
    print("测试网络连接:")
    print("-" * 60)
    
    test_sites = [
        ("HuggingFace 镜像 (hf-mirror.com)", "https://hf-mirror.com"),
        ("HuggingFace 官方 (huggingface.co)", "https://huggingface.co"),
        ("模型页面示例", "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct"),
    ]
    
    results = []
    for name, url in test_sites:
        result = test_url(name, url)
        results.append(result)
        print()
    
    # 总结
    print("=" * 60)
    if any(results):
        print("✅ 至少有一个站点可以访问")
        if results[0]:
            print("   推荐使用镜像站点: https://hf-mirror.com")
            print("   设置方法: set HF_ENDPOINT=https://hf-mirror.com")
        elif results[1]:
            print("   可以使用官方站点")
    else:
        print("❌ 所有站点都无法访问")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 检查代理设置（如需要）")
        print("3. 尝试禁用代理: set HTTP_PROXY= && set HTTPS_PROXY=")
        print("4. 使用离线下载方式")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
