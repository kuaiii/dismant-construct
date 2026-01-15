#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本：一键运行数据生成 -> 训练 -> 验证

用法:
    python scripts/quick_test_pipeline.py

功能:
    1. 生成少量混合数据（dismantle + construct）
    2. 使用小模型快速训练 1 个 epoch
    3. 验证模型输出格式
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd, description):
    """运行命令并打印状态"""
    print("\n" + "=" * 60)
    print(f"🔄 {description}")
    print("=" * 60)
    print(f"命令: {cmd}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Windows 需要特殊处理
    if os.name == 'nt':
        result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    else:
        result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} 完成 (耗时: {elapsed:.1f}s)")
    else:
        print(f"\n❌ {description} 失败 (返回码: {result.returncode})")
        return False
    
    return True


def check_data_generated(data_dir):
    """检查数据是否正确生成"""
    train_path = data_dir / "train.json"
    eval_path = data_dir / "eval.json"
    
    if not train_path.exists():
        print(f"❌ 训练数据未找到: {train_path}")
        return False
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"\n📊 数据统计:")
    print(f"  - 训练样本数: {len(train_data)}")
    
    # 统计任务类型
    dismantle_count = sum(1 for s in train_data if s['meta']['task'] == 'dismantle')
    construct_count = sum(1 for s in train_data if s['meta']['task'] == 'construct')
    print(f"  - Dismantle 样本: {dismantle_count}")
    print(f"  - Construct 样本: {construct_count}")
    
    # 检查符号函数标记
    has_sign = all('sign' in s['meta'] for s in train_data)
    if has_sign:
        print(f"  - 符号函数标记: ✅ 已添加")
    else:
        print(f"  - 符号函数标记: ⚠️ 部分样本缺失")
    
    if eval_path.exists():
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        print(f"  - 验证样本数: {len(eval_data)}")
    
    return True


def validate_sample_format(data_dir):
    """验证样本格式"""
    train_path = data_dir / "train.json"
    
    if not train_path.exists():
        return False
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print("\n📋 样本格式验证:")
    
    # 检查第一个样本
    sample = train_data[0]
    
    # 必需字段
    required_fields = ['id', 'meta', 'conversations', 'auxiliary_labels']
    for field in required_fields:
        if field in sample:
            print(f"  - {field}: ✅")
        else:
            print(f"  - {field}: ❌ 缺失")
            return False
    
    # 检查 meta 字段
    meta_fields = ['task', 'budget_step', 'sign']
    for field in meta_fields:
        if field in sample['meta']:
            print(f"  - meta.{field}: ✅")
        else:
            print(f"  - meta.{field}: ⚠️ 缺失")
    
    # 检查对话格式
    print(f"  - conversations 长度: {len(sample['conversations'])}")
    
    # 显示一个样本的内容片段
    print("\n📝 样本示例 (前200字符):")
    user_content = sample['conversations'][1]['value'][:200]
    print(f"  {user_content}...")
    
    return True


def main():
    print("=" * 60)
    print("🚀 网络韧性优化框架 - 快速测试流水线")
    print("=" * 60)
    print("\n本脚本将执行以下步骤:")
    print("  1. 生成少量混合训练数据 (dismantle + construct)")
    print("  2. 训练模型 (1 epoch)")
    print("  3. 验证训练结果")
    print()
    
    # ========== 步骤 1: 生成数据 ==========
    data_cmd = "python scripts/generate_data.py --quick_test"
    if not run_command(data_cmd, "步骤 1/3: 生成混合训练数据"):
        print("\n⚠️ 数据生成失败，请检查错误信息")
        return
    
    # 验证数据
    data_dir = PROJECT_ROOT / "data" / "fine_tuning" / "quick_test"
    if not check_data_generated(data_dir):
        print("\n⚠️ 数据验证失败")
        return
    
    if not validate_sample_format(data_dir):
        print("\n⚠️ 样本格式验证失败")
        return
    
    # ========== 步骤 2: 训练模型 ==========
    train_cmd = (
        "python scripts/train.py "
        f"--train_data {data_dir / 'train.json'} "
        f"--eval_data {data_dir / 'eval.json'} "
        f"--output_dir outputs/quick_test "
        "--epochs 1 "
        "--batch_size 1"
    )
    
    print("\n" + "=" * 60)
    print("📦 步骤 2/3: 训练模型")
    print("=" * 60)
    print("\n⚠️ 注意: 如果这是首次运行，需要下载模型权重，可能需要几分钟")
    print("如果下载失败，可以手动运行: python scripts/download_model.py")
    print()
    
    if not run_command(train_cmd, "步骤 2/3: 训练模型 (1 epoch)"):
        print("\n⚠️ 训练失败，请检查:")
        print("  1. 模型是否已下载")
        print("  2. GPU 内存是否足够")
        print("  3. 依赖是否安装完整")
        print("\n可以尝试使用 CPU 模式:")
        print("  修改 configs/default.yaml 中的 device: cpu")
        return
    
    # ========== 步骤 3: 验证结果 ==========
    print("\n" + "=" * 60)
    print("✅ 步骤 3/3: 验证训练结果")
    print("=" * 60)
    
    output_dir = PROJECT_ROOT / "outputs" / "quick_test"
    if output_dir.exists():
        print(f"\n模型输出目录: {output_dir}")
        
        # 列出生成的文件
        files = list(output_dir.rglob("*"))
        print(f"生成的文件数: {len(files)}")
        
        # 检查日志
        log_files = [f for f in files if f.suffix == '.log']
        if log_files:
            print(f"\n日志文件: {log_files[0]}")
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("🎉 快速测试流水线完成!")
    print("=" * 60)
    print("\n下一步建议:")
    print("  1. 查看训练日志了解模型学习情况")
    print("  2. 使用完整数据集进行正式训练:")
    print("     python scripts/generate_data.py --task_type both --num_graphs 50")
    print("  3. 运行推理测试:")
    print("     python scripts/inference.py --checkpoint outputs/quick_test/checkpoints/best")


if __name__ == "__main__":
    main()
