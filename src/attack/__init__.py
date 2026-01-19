# -*- coding: utf-8 -*-
"""
Attack Algorithms Module
网络拆解攻击算法模块

提供多种网络拆解策略：
- HighestDegreeAttack: 高度数攻击
- RandomAttack: 随机攻击
- LLMAttack: 基于 LLM 的智能攻击 (本项目方法)
"""

from .base import BaseAttack, AttackResult
from .highest_degree import HighestDegreeAttack
from .random_attack import RandomAttack
from .llm_attack import LLMAttack

__all__ = [
    "BaseAttack",
    "AttackResult", 
    "HighestDegreeAttack",
    "RandomAttack",
    "LLMAttack",
]
