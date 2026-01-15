#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找所有可用的检查点

用法:
    python scripts/find_checkpoints.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def find_checkpoints(output_dir: str = "outputs"):
    """查找所有检查点"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"输出目录不存在: {output_dir}")
        return
    
    print("=" * 60)
    print("查找检查点")
    print("=" * 60)
    
    checkpoints = []
    
    # 查找所有 checkpoints 目录
    for checkpoint_dir in output_path.rglob("checkpoints"):
        for epoch_dir in checkpoint_dir.iterdir():
            if epoch_dir.is_dir():
                model_file = epoch_dir / "model.pt"
                if model_file.exists():
                    checkpoints.append({
                        "path": str(model_file),
                        "epoch": epoch_dir.name,
                        "parent": str(checkpoint_dir.parent)
                    })
    
    if checkpoints:
        print(f"\n找到 {len(checkpoints)} 个检查点:\n")
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"{i}. {ckpt['path']}")
            print(f"   目录: {ckpt['parent']}")
            print(f"   Epoch: {ckpt['epoch']}")
            print()
        
        # 推荐使用最新的
        latest = max(checkpoints, key=lambda x: int(x['epoch'].split('_')[1]) if x['epoch'].startswith('epoch_') else 0)
        print("=" * 60)
        print("推荐使用最新的检查点:")
        print(f"  {latest['path']}")
        print("\n评估命令示例:")
        print(f"  python scripts/evaluate.py --checkpoint {latest['parent']}/checkpoints/{latest['epoch']}")
        print("=" * 60)
    else:
        print("\n未找到任何检查点文件")
        print(f"请检查 {output_dir} 目录下是否有训练保存的模型")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="查找所有检查点")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()
    
    find_checkpoints(args.output_dir)
