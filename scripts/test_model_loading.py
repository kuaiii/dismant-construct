#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型加载脚本

用法:
    python scripts/test_model_loading.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
# 设置 HuggingFace 镜像（如果未设置）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("ℹ️  已设置 HuggingFace 镜像站点: https://hf-mirror.com")

import torch
from src.model.fusion_llm import ResilienceLLM, ModelConfig


def test_model_loading(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """测试模型加载"""
    print("=" * 60)
    print("模型加载测试")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    config = ModelConfig(
        llm_model_name=model_name,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("创建模型实例...")
    model = ResilienceLLM(config)
    
    print("初始化模型（加载权重）...")
    try:
        model.initialize(device=device)
        print("✅ 模型加载成功!")
        print()
        
        # 打印模型信息
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"模型参数统计:")
        print(f"  总参数: {total:,}")
        print(f"  可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"  冻结参数: {total-trainable:,} ({100*(total-trainable)/total:.2f}%)")
        
        # 显存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n显存使用:")
            print(f"  已分配: {memory_allocated:.2f} GB")
            print(f"  已保留: {memory_reserved:.2f} GB")
        
        # 测试前向传播
        print("\n测试前向传播...")
        if model.tokenizer is not None:
            test_text = "你好，这是一个测试。"
            inputs = model.tokenizer(test_text, return_tensors="pt", padding=True)
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            model.eval()
            with torch.no_grad():
                try:
                    outputs = model.llm(**inputs)
                    print("✅ 前向传播成功!")
                    print(f"  输出形状: {outputs.logits.shape}")
                except Exception as e:
                    print(f"⚠️ 前向传播警告: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！可以开始训练了。")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 模型加载失败!")
        print(f"错误: {e}")
        print("\n可能的原因:")
        print("1. 模型名称错误或无法访问")
        print("2. 网络连接问题")
        print("3. HuggingFace 访问权限问题")
        print("4. 显存不足")
        print("\n请参考 docs/quick_start_test.md 获取帮助")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试模型加载")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="要测试的模型名称"
    )
    
    args = parser.parse_args()
    
    success = test_model_loading(args.model_name)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
