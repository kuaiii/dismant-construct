# -*- coding: utf-8 -*-
"""
Random Attack
随机攻击算法

策略：每次随机选择一个节点进行移除。
这是最简单的攻击基线，模拟随机故障。

参考文献:
- Albert, R., Jeong, H., & Barabási, A. L. (2000). 
  "Error and attack tolerance of complex networks." Nature, 406(6794), 378-382.
"""

from typing import Optional, Any, List
import random
import networkx as nx
from .base import BaseAttack


class RandomAttack(BaseAttack):
    """
    随机攻击 (Random Attack)
    
    攻击策略：
    每一步随机选择一个节点进行移除。
    
    特点：
    - 模拟随机故障
    - 对无标度网络的破坏效果较弱
    - 作为攻击算法的基线对比
    - 时间复杂度: O(1) 每步选择
    
    参数：
    - seed: 随机种子，用于复现结果
    - num_runs: 运行次数（用于取平均）
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机攻击算法
        
        Args:
            seed: 随机种子，None 表示不设置
        """
        super().__init__(name="RandomAttack")
        self.seed = seed
        self._rng = random.Random(seed)
    
    def select_node(self, graph: nx.Graph, **kwargs) -> Optional[Any]:
        """
        随机选择一个节点
        
        Args:
            graph: 当前网络图
        
        Returns:
            随机选择的节点 ID
        """
        if graph.number_of_nodes() == 0:
            return None
        
        nodes = list(graph.nodes())
        return self._rng.choice(nodes)
    
    def set_seed(self, seed: int) -> None:
        """
        设置随机种子
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        self._rng = random.Random(seed)
    
    def attack_multiple_runs(
        self,
        graph: nx.Graph,
        budget: int,
        num_runs: int = 10,
        dataset_name: str = "unknown",
        graph_name: str = "unknown",
        collapse_threshold: float = 0.2,
        **kwargs
    ) -> dict:
        """
        多次运行随机攻击并计算平均结果
        
        Args:
            graph: 初始网络图
            budget: 攻击预算
            num_runs: 运行次数
            dataset_name: 数据集名称
            graph_name: 图名称
            collapse_threshold: 崩溃阈值
        
        Returns:
            dict: 包含平均结果和所有运行结果的字典
        """
        import numpy as np
        
        all_results = []
        all_lcc_curves = []
        all_r_res = []
        all_collapse_fractions = []
        
        for run_idx in range(num_runs):
            # 设置不同的种子
            self.set_seed(self.seed + run_idx if self.seed else run_idx)
            
            # 执行攻击
            result = self.attack(
                graph=graph,
                budget=budget,
                dataset_name=dataset_name,
                graph_name=f"{graph_name}_run{run_idx}",
                collapse_threshold=collapse_threshold,
                **kwargs
            )
            
            all_results.append(result)
            all_lcc_curves.append(result.lcc_values)
            all_r_res.append(result.r_res)
            if result.collapse_fraction is not None:
                all_collapse_fractions.append(result.collapse_fraction)
        
        # 计算平均 LCC 曲线
        # 确保所有曲线长度一致
        max_len = max(len(curve) for curve in all_lcc_curves)
        padded_curves = []
        for curve in all_lcc_curves:
            if len(curve) < max_len:
                # 用最后一个值填充
                padded = curve + [curve[-1]] * (max_len - len(curve))
            else:
                padded = curve
            padded_curves.append(padded)
        
        avg_lcc_curve = np.mean(padded_curves, axis=0).tolist()
        std_lcc_curve = np.std(padded_curves, axis=0).tolist()
        
        # 创建平均结果
        avg_result = all_results[0]  # 使用第一个结果作为模板
        avg_result.algorithm_name = f"RandomAttack_avg{num_runs}"
        avg_result.graph_name = graph_name
        avg_result.lcc_values = avg_lcc_curve
        avg_result.r_res = np.mean(all_r_res)
        avg_result.collapse_fraction = np.mean(all_collapse_fractions) if all_collapse_fractions else None
        
        return {
            "average_result": avg_result,
            "all_results": all_results,
            "avg_r_res": np.mean(all_r_res),
            "std_r_res": np.std(all_r_res),
            "avg_collapse_fraction": np.mean(all_collapse_fractions) if all_collapse_fractions else None,
            "std_lcc_curve": std_lcc_curve,
        }
