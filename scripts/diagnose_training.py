#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练问题诊断脚本
用于排查 NaN/Inf 问题的根源
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np


def check_data_quality(data_path: str):
    """检查数据质量"""
    print("=" * 60)
    print("1. 检查数据质量")
    print("=" * 60)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"样本数量: {len(data)}")
    
    issues = []
    for i, sample in enumerate(data):
        aux_labels = sample.get("auxiliary_labels", {})
        
        # 检查 auxiliary_labels
        for op_id, value in aux_labels.items():
            if value != value:  # NaN check
                issues.append(f"样本 {i} ({sample.get('id', 'unknown')}): {op_id} 值为 NaN")
            if abs(value) == float('inf'):
                issues.append(f"样本 {i} ({sample.get('id', 'unknown')}): {op_id} 值为 Inf")
            if abs(value) > 100:
                issues.append(f"样本 {i} ({sample.get('id', 'unknown')}): {op_id} 值异常大: {value}")
        
        # 检查文本长度
        convs = sample.get("conversations", [])
        total_len = sum(len(c.get("value", "")) for c in convs)
        if total_len > 10000:
            issues.append(f"样本 {i}: 文本总长度过长: {total_len}")
    
    if issues:
        print(f"\n⚠️ 发现 {len(issues)} 个潜在问题:")
        for issue in issues[:20]:  # 只显示前 20 个
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues) - 20} 个问题")
    else:
        print("✅ 数据质量检查通过，未发现异常值")
    
    # 统计 auxiliary_labels 分布
    all_values = []
    for sample in data:
        all_values.extend(sample.get("auxiliary_labels", {}).values())
    
    if all_values:
        print(f"\nauxiliary_labels 统计:")
        print(f"  最小值: {min(all_values):.6f}")
        print(f"  最大值: {max(all_values):.6f}")
        print(f"  均值: {np.mean(all_values):.6f}")
        print(f"  标准差: {np.std(all_values):.6f}")
        print(f"  零值比例: {sum(1 for v in all_values if v == 0) / len(all_values) * 100:.1f}%")
    
    return len(issues) == 0


def check_model_forward(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """检查模型前向传播"""
    print("\n" + "=" * 60)
    print("2. 检查模型前向传播")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试不同精度
    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
        print(f"\n测试 {dtype_name} 精度:")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto"
            )
            model.eval()
            
            # 简单测试
            test_text = "Hello, how are you?"
            inputs = tokenizer(test_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            
            if has_nan or has_inf:
                print(f"  ❌ {dtype_name} 输出包含 NaN/Inf!")
            else:
                print(f"  ✅ {dtype_name} 输出正常")
                print(f"     logits 范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ {dtype_name} 测试失败: {e}")
    
    return True


def check_training_step():
    """检查单步训练"""
    print("\n" + "=" * 60)
    print("3. 检查单步训练 (禁用 FP16)")
    print("=" * 60)
    
    import yaml
    from src.model.fusion_llm import ResilienceLLM, ModelConfig
    from src.data.dataset import create_dataloader
    
    # 加载配置
    with open("configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model_config = ModelConfig(
        llm_model_name=config['model']['llm']['model_name'],
        use_lora=config['model']['lora']['enabled'],
        lora_r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['alpha'],
        lora_dropout=config['model']['lora']['dropout'],
    )
    
    print("初始化模型...")
    model = ResilienceLLM(model_config)
    model.initialize(device="cuda")
    model.train()
    
    # 加载数据
    print("加载数据...")
    train_loader = create_dataloader(
        data_path="data/fine_tuning/combined/train.json",
        tokenizer=model.tokenizer,
        batch_size=1,
        shuffle=False,
        max_length=config['data']['loading']['max_length']
    )
    
    # 测试几个 batch
    print("\n测试前 10 个 batch:")
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        
        # 禁用 AMP，使用纯 FP32
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_scores=True
            )
        
        logits = outputs.get("logits")
        if logits is not None:
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            
            if has_nan or has_inf:
                print(f"  Batch {i}: ❌ logits 包含 NaN/Inf")
                print(f"    样本 ID: {batch.get('sample_ids', ['unknown'])[0]}")
                print(f"    输入长度: {attention_mask.sum().item()}")
            else:
                print(f"  Batch {i}: ✅ 正常, logits 范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        else:
            print(f"  Batch {i}: logits 为 None")
    
    return True


def suggest_fixes():
    """建议修复方案"""
    print("\n" + "=" * 60)
    print("4. 建议的修复方案")
    print("=" * 60)
    
    print("""
根据诊断结果，建议尝试以下修复方案（按优先级排序）:

【方案 1】禁用 FP16 混合精度训练
  修改 configs/default.yaml:
  training:
    fp16: false
    bf16: false

【方案 2】降低学习率
  python scripts/train.py --lr 1e-5

【方案 3】使用 BF16（如果 GPU 支持）
  training:
    fp16: false
    bf16: true

【方案 4】减小梯度裁剪阈值
  optimizer:
    max_grad_norm: 0.5

【方案 5】检查并修复数据集
  - 移除包含异常值的样本
  - 归一化 auxiliary_labels
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="训练问题诊断")
    parser.add_argument("--data", type=str, default="data/fine_tuning/combined/train.json")
    parser.add_argument("--skip-model", action="store_true", help="跳过模型测试")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练测试")
    args = parser.parse_args()
    
    print("🔍 开始诊断训练问题...\n")
    
    # 1. 检查数据
    data_ok = check_data_quality(args.data)
    
    # 2. 检查模型
    if not args.skip_model:
        check_model_forward()
    
    # 3. 检查训练
    if not args.skip_training:
        try:
            check_training_step()
        except Exception as e:
            print(f"训练测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 建议修复
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("诊断完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
