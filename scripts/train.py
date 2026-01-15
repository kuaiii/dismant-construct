#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练脚本

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --phase 1 --epochs 3
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import gc


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="训练网络韧性优化模型")
    
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--train_data", type=str, default=None, help="训练数据路径")
    parser.add_argument("--eval_data", type=str, default=None, help="评估数据路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--phase", type=int, default=None, help="训练阶段 (1 或 2)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.train_data:
        config['data']['fine_tuning_dir'] = args.train_data
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.phase:
        config['training']['phase'] = args.phase
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['learning_rate'] = args.lr
    if args.seed:
        config['seed'] = args.seed
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("=" * 60)
    print("网络韧性优化模型训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"训练阶段: Phase {config['training']['phase']}")
    print(f"训练轮数: {config['training']['num_epochs']}")
    print(f"批大小: {config['training']['batch_size']}")
    print(f"学习率: {config['training']['optimizer']['learning_rate']}")
    print("=" * 60)
    
    # 导入模块
    from src.model.fusion_llm import ResilienceLLM, ModelConfig
    from src.data.dataset import create_dataloader
    from src.trainer.train import ResilienceTrainer, TrainingConfig
    
    # 数据路径
    train_data_path = args.train_data or str(Path(config['data']['fine_tuning_dir']) / "train.json")
    eval_data_path = args.eval_data or str(Path(config['data']['fine_tuning_dir']) / "eval.json")
    
    # 检查数据文件
    if not Path(train_data_path).exists():
        print(f"错误: 训练数据文件不存在: {train_data_path}")
        print("请先运行 python scripts/generate_data.py 生成数据")
        return
    
    print(f"训练数据: {train_data_path}")
    print(f"评估数据: {eval_data_path}")
    
    # 创建模型配置
    model_config = ModelConfig(
        llm_model_name=config['model']['llm']['model_name'],
        use_lora=config['model']['lora']['enabled'],
        lora_r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['alpha'],
        lora_dropout=config['model']['lora']['dropout'],
        use_geometric_encoder=config['model']['geometric_encoder']['enabled'],
        d_model=config['model']['fusion']['d_model']
    )
    
    # 创建模型
    print("\n正在创建模型...")
    model = ResilienceLLM(model_config)
    
    # 初始化模型（需要实现 _load_llm 和 _apply_lora 方法）
    try:
        device = config['training']['device']
        print(f"正在初始化模型 (device: {device})...")
        print("注意: 如果遇到错误，请检查 src/model/fusion_llm.py 中的 _load_llm() 实现")
        model.initialize(device=device)
        print("模型初始化完成")
    except NotImplementedError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请参考 docs/training_setup_guide.md 实现 LLM 加载功能")
        print("需要实现以下方法:")
        print("  1. src/model/fusion_llm.py -> _load_llm()")
        print("  2. src/model/fusion_llm.py -> _apply_lora()")
        return
    except Exception as e:
        print(f"\n❌ 模型初始化失败: {e}")
        print("\n可能的原因:")
        print("  1. 模型权重未找到（需要 HuggingFace 访问权限）")
        print("  2. CUDA/GPU 配置问题")
        print("  3. 依赖库未安装完整")
        print("\n请参考 docs/training_setup_guide.md 获取帮助")
        import traceback
        traceback.print_exc()
        return
    
    # 创建数据加载器（需要传入 tokenizer）
    print("\n正在加载数据...")
    
    # 获取 tokenizer（模型初始化后才有）
    if model.tokenizer is None:
        print("❌ 错误: 模型 tokenizer 未初始化")
        print("请确保模型已正确初始化")
        return
    
    tokenizer = model.tokenizer
    max_length = config['data']['loading']['max_length']
    
    train_loader = create_dataloader(
        data_path=train_data_path,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        max_length=max_length
    )
    
    eval_loader = None
    if Path(eval_data_path).exists():
        eval_loader = create_dataloader(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            max_length=max_length
        )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    if eval_loader:
        print(f"评估样本数: {len(eval_loader.dataset)}")
    
    # 创建训练配置
    training_config = TrainingConfig(
        output_dir=config['training']['output_dir'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        warmup_ratio=config['training']['optimizer']['warmup_ratio'],
        max_grad_norm=config['training']['optimizer']['max_grad_norm'],
        ranking_loss_type=config['training']['loss']['ranking_type'],
        ranking_loss_weight=config['training']['loss']['ranking_weight'],
        lm_loss_weight=config['training']['loss']['lm_weight'],
        phase=config['training']['phase'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        device=config['training']['device'],
        fp16=config['training']['fp16'],
        resume_from_checkpoint=args.resume
    )
    
    # 创建训练器
    print("\n正在初始化训练器...")
    trainer = ResilienceTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print()
    
    try:
        results = trainer.train()
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"最终损失: {results['final_loss']:.4f}")
        print(f"最佳指标: {results['best_metric']:.4f}")
        print(f"总步数: {results['total_steps']}")
        print(f"\n模型保存在: {training_config.output_dir}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("检查点已保存，可以使用 --resume 参数继续训练")
    except Exception as e:
        print(f"\n\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("1. 模型是否正确加载")
        print("2. 数据格式是否正确")
        print("3. 显存是否足够")
        raise


if __name__ == "__main__":
    main()
