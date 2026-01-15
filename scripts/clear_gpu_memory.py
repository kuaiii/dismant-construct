#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理 GPU 内存脚本

用法:
    python scripts/clear_gpu_memory.py
"""

import torch
import gc

def clear_gpu_memory():
    """清理 GPU 内存"""
    print("正在清理 GPU 内存...")
    
    # 清理 Python 垃圾回收
    gc.collect()
    
    # 清理 PyTorch 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 显示当前 GPU 内存使用情况
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
    
    print("GPU 内存清理完成!")

if __name__ == "__main__":
    clear_gpu_memory()
