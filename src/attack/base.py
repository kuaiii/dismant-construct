# -*- coding: utf-8 -*-
"""
BaseAttack: 网络拆解攻击基类

定义攻击算法的标准接口和通用功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import networkx as nx
from pathlib import Path
import json
import os


@dataclass
class AttackResult:
    """攻击结果数据类"""
    # 基本信息
    algorithm_name: str
    dataset_name: str
    graph_name: str
    
    # 攻击序列
    attack_sequence: List[Any]  # 被移除的节点列表
    
    # 指标数据
    removal_fractions: List[float]  # 移除节点比例序列 [0, 1/n, 2/n, ..., budget/n]
    lcc_values: List[float]  # LCC 比例序列
    r_res: float  # 韧性积分 (曲线下面积)
    
    # 崩溃点信息
    collapse_threshold: float = 0.2  # 崩溃阈值 (LCC < 20%)
    collapse_fraction: Optional[float] = None  # 到达崩溃点时的移除比例
    
    # 元信息
    initial_nodes: int = 0
    initial_edges: int = 0
    budget: int = 0
    
    # 额外数据
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def compute_r_res(self) -> float:
        """
        计算 R_res (韧性积分 - LCC曲线下面积)
        
        R_res = ∫₀¹ LCC(q) dq
        使用梯形积分法
        """
        if len(self.lcc_values) < 2:
            return 0.0
        
        # 使用梯形积分计算曲线下面积
        # x 轴是 removal_fractions, y 轴是 lcc_values
        r_res = np.trapezoid(self.lcc_values, self.removal_fractions)
        return float(r_res)
    
    def find_collapse_point(self, threshold: float = 0.2) -> Optional[float]:
        """
        找到网络崩溃点 (LCC 首次低于阈值的位置)
        
        Args:
            threshold: 崩溃阈值 (默认 0.2 即 20%)
        
        Returns:
            移除比例，如果未崩溃则返回 None
        """
        for i, lcc in enumerate(self.lcc_values):
            if lcc < threshold:
                # 线性插值找到精确的崩溃点
                if i > 0:
                    lcc_prev = self.lcc_values[i - 1]
                    frac_prev = self.removal_fractions[i - 1]
                    frac_curr = self.removal_fractions[i]
                    
                    # 线性插值: 找到 LCC = threshold 的位置
                    if lcc_prev != lcc:
                        t = (lcc_prev - threshold) / (lcc_prev - lcc)
                        collapse_frac = frac_prev + t * (frac_curr - frac_prev)
                        return collapse_frac
                return self.removal_fractions[i]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "algorithm_name": self.algorithm_name,
            "dataset_name": self.dataset_name,
            "graph_name": self.graph_name,
            "attack_sequence": [str(n) for n in self.attack_sequence],
            "removal_fractions": self.removal_fractions,
            "lcc_values": self.lcc_values,
            "r_res": self.r_res,
            "collapse_threshold": self.collapse_threshold,
            "collapse_fraction": self.collapse_fraction,
            "initial_nodes": self.initial_nodes,
            "initial_edges": self.initial_edges,
            "budget": self.budget,
            "extra_metrics": self.extra_metrics,
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存结果到 JSON 文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AttackResult":
        """从 JSON 文件加载结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            algorithm_name=data["algorithm_name"],
            dataset_name=data["dataset_name"],
            graph_name=data["graph_name"],
            attack_sequence=data["attack_sequence"],
            removal_fractions=data["removal_fractions"],
            lcc_values=data["lcc_values"],
            r_res=data["r_res"],
            collapse_threshold=data.get("collapse_threshold", 0.2),
            collapse_fraction=data.get("collapse_fraction"),
            initial_nodes=data.get("initial_nodes", 0),
            initial_edges=data.get("initial_edges", 0),
            budget=data.get("budget", 0),
            extra_metrics=data.get("extra_metrics", {}),
        )


class BaseAttack(ABC):
    """
    网络拆解攻击基类
    
    所有攻击算法都需要继承此类并实现 select_node 方法。
    """
    
    def __init__(self, name: str = "BaseAttack"):
        """
        初始化攻击算法
        
        Args:
            name: 算法名称
        """
        self.name = name
    
    @abstractmethod
    def select_node(self, graph: nx.Graph, **kwargs) -> Optional[Any]:
        """
        选择下一个要移除的节点
        
        Args:
            graph: 当前网络图
            **kwargs: 额外参数
        
        Returns:
            选中的节点 ID，如果没有可选节点则返回 None
        """
        pass
    
    def attack(
        self,
        graph: nx.Graph,
        budget: int,
        dataset_name: str = "unknown",
        graph_name: str = "unknown",
        collapse_threshold: float = 0.2,
        **kwargs
    ) -> AttackResult:
        """
        执行攻击并记录结果
        
        Args:
            graph: 初始网络图
            budget: 攻击预算 (最多移除的节点数)
            dataset_name: 数据集名称
            graph_name: 图名称
            collapse_threshold: 崩溃阈值
            **kwargs: 传递给 select_node 的额外参数
        
        Returns:
            AttackResult: 攻击结果
        """
        # 复制图以避免修改原图
        g = graph.copy()
        initial_nodes = g.number_of_nodes()
        initial_edges = g.number_of_edges()
        
        # 初始化记录
        attack_sequence = []
        removal_fractions = [0.0]  # 初始移除比例为 0
        lcc_values = [self._compute_lcc_ratio(g, initial_nodes)]  # 初始 LCC
        
        # 执行攻击
        for step in range(budget):
            if g.number_of_nodes() == 0:
                break
            
            # 选择节点
            node = self.select_node(g, **kwargs)
            if node is None:
                break
            
            # 移除节点
            g.remove_node(node)
            attack_sequence.append(node)
            
            # 记录指标
            removal_frac = len(attack_sequence) / initial_nodes
            removal_fractions.append(removal_frac)
            
            lcc_ratio = self._compute_lcc_ratio(g, initial_nodes)
            lcc_values.append(lcc_ratio)
        
        # 创建结果对象
        result = AttackResult(
            algorithm_name=self.name,
            dataset_name=dataset_name,
            graph_name=graph_name,
            attack_sequence=attack_sequence,
            removal_fractions=removal_fractions,
            lcc_values=lcc_values,
            r_res=0.0,  # 稍后计算
            collapse_threshold=collapse_threshold,
            initial_nodes=initial_nodes,
            initial_edges=initial_edges,
            budget=budget,
        )
        
        # 计算 R_res 和崩溃点
        result.r_res = result.compute_r_res()
        result.collapse_fraction = result.find_collapse_point(collapse_threshold)
        
        return result
    
    def _compute_lcc_ratio(self, graph: nx.Graph, initial_nodes: int) -> float:
        """
        计算 LCC 比例 (相对于初始节点数)
        
        Args:
            graph: 当前图
            initial_nodes: 初始节点数
        
        Returns:
            LCC 比例
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        connected_components = list(nx.connected_components(graph))
        if not connected_components:
            return 0.0
        
        largest_cc_size = max(len(cc) for cc in connected_components)
        return largest_cc_size / initial_nodes
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
