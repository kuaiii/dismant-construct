# -*- coding: utf-8 -*-
"""
Highest Degree Attack (HDA)
高度数攻击算法

策略：每次移除当前网络中度数最高的节点。
这是一种经典的目标攻击策略，对无标度网络特别有效。

参考文献:
- Albert, R., Jeong, H., & Barabási, A. L. (2000). 
  "Error and attack tolerance of complex networks." Nature, 406(6794), 378-382.
"""

from typing import Optional, Any, List
import networkx as nx
from .base import BaseAttack


class HighestDegreeAttack(BaseAttack):
    """
    高度数攻击 (Highest Degree Attack, HDA)
    
    攻击策略：
    每一步选择当前网络中度数最高的节点进行移除。
    
    特点：
    - 对无标度网络 (BA 模型) 非常有效
    - 攻击枢纽节点 (hub nodes) 可以快速瓦解网络
    - 时间复杂度: O(n) 每步选择
    
    变体：
    - recalculate=True: 每步重新计算度数 (自适应)
    - recalculate=False: 使用初始度数排序 (一次性计算)
    """
    
    def __init__(self, recalculate: bool = True):
        """
        初始化高度数攻击算法
        
        Args:
            recalculate: 是否每步重新计算度数
                - True: 自适应攻击，每步选择当前最高度数节点
                - False: 静态攻击，按初始度数排序
        """
        super().__init__(name="HighestDegreeAttack" if recalculate else "HighestDegreeAttack_Static")
        self.recalculate = recalculate
        self._initial_ranking = None
    
    def select_node(self, graph: nx.Graph, **kwargs) -> Optional[Any]:
        """
        选择度数最高的节点
        
        Args:
            graph: 当前网络图
        
        Returns:
            度数最高的节点 ID
        """
        if graph.number_of_nodes() == 0:
            return None
        
        if self.recalculate:
            # 自适应攻击：每次选择当前度数最高的节点
            degrees = dict(graph.degree())
            if not degrees:
                return None
            
            # 找到度数最高的节点
            max_node = max(degrees.keys(), key=lambda n: degrees[n])
            return max_node
        else:
            # 静态攻击：使用预计算的排序
            if self._initial_ranking is None:
                return None
            
            # 返回排序中第一个仍在图中的节点
            for node in self._initial_ranking:
                if node in graph:
                    return node
            return None
    
    def attack(
        self,
        graph: nx.Graph,
        budget: int,
        dataset_name: str = "unknown",
        graph_name: str = "unknown",
        collapse_threshold: float = 0.2,
        **kwargs
    ):
        """
        执行高度数攻击
        
        如果使用静态模式，先计算初始度数排序。
        """
        if not self.recalculate:
            # 预计算初始度数排序
            degrees = dict(graph.degree())
            self._initial_ranking = sorted(
                degrees.keys(), 
                key=lambda n: degrees[n], 
                reverse=True
            )
        
        return super().attack(
            graph=graph,
            budget=budget,
            dataset_name=dataset_name,
            graph_name=graph_name,
            collapse_threshold=collapse_threshold,
            **kwargs
        )


class AdaptiveHighestDegreeAttack(HighestDegreeAttack):
    """
    自适应高度数攻击 (Adaptive HDA)
    
    每一步重新计算度数，选择当前最高度数节点。
    """
    
    def __init__(self):
        super().__init__(recalculate=True)
        self.name = "AdaptiveHDA"


class InitialDegreeAttack(HighestDegreeAttack):
    """
    初始度数攻击 (Initial Degree Attack)
    
    按初始度数排序，不重新计算。
    """
    
    def __init__(self):
        super().__init__(recalculate=False)
        self.name = "InitialDegreeAttack"
