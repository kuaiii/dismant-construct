# -*- coding: utf-8 -*-
"""
UnifiedEvaluator: 统一评估框架
支持 Dismant（拆解）和 Construct（构造）两种任务的统一评估。

核心指标：
- Dismant: LCC 曲线、R_res（韧性积分 - 越小越好表示拆解效果越好）
- Construct: 重构后的 R_tar（目标攻击韧性）、R_ran（随机攻击韧性）

使用方法：
    evaluator = UnifiedEvaluator()
    
    # Dismant 评估
    result = evaluator.evaluate_dismant(
        graph=G,
        attack_sequence=nodes_to_remove,
        method_name="LLM"
    )
    
    # Construct 评估
    result = evaluator.evaluate_construct(
        original_graph=G_original,
        reconstructed_graph=G_reconstructed,
        budget=20,
        method_name="LLM"
    )
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import networkx as nx

# 导入攻击模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.attack import HighestDegreeAttack, RandomAttack, AttackResult


@dataclass
class DismantResult:
    """拆解任务评估结果"""
    method_name: str
    graph_name: str
    
    # 核心指标
    r_res: float  # 韧性积分（越小表示拆解效果越好）
    lcc_curve: List[float]  # LCC 变化曲线
    removal_fractions: List[float]  # 移除比例序列
    
    # 攻击序列
    attack_sequence: List[Any]
    
    # 崩溃信息
    collapse_fraction: Optional[float] = None  # 崩溃点（LCC < 20%）
    collapse_threshold: float = 0.2
    
    # 元信息
    initial_nodes: int = 0
    initial_edges: int = 0
    budget: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "method_name": self.method_name,
            "graph_name": self.graph_name,
            "r_res": self.r_res,
            "lcc_curve": self.lcc_curve,
            "removal_fractions": self.removal_fractions,
            "attack_sequence": [str(n) for n in self.attack_sequence],
            "collapse_fraction": self.collapse_fraction,
            "collapse_threshold": self.collapse_threshold,
            "initial_nodes": self.initial_nodes,
            "initial_edges": self.initial_edges,
            "budget": self.budget,
        }


@dataclass
class ConstructResult:
    """构造任务评估结果"""
    method_name: str
    graph_name: str
    
    # 核心指标
    r_tar: float  # 目标攻击（HDA）下的韧性积分
    r_ran: float  # 随机攻击下的韧性积分（多次平均）
    r_improvement: float  # 相对原始图的韧性提升比例
    
    # 详细曲线
    hda_lcc_curve: List[float]
    random_lcc_curve: List[float]
    hda_removal_fractions: List[float]
    random_removal_fractions: List[float]
    
    # 添加的边
    added_edges: List[Tuple[Any, Any]] = field(default_factory=list)
    
    # 原始图韧性（用于对比）
    r_original_tar: float = 0.0  # 原始图的 HDA 韧性
    r_original_ran: float = 0.0  # 原始图的随机韧性
    
    # 元信息
    initial_nodes: int = 0
    initial_edges: int = 0
    final_edges: int = 0
    budget: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "method_name": self.method_name,
            "graph_name": self.graph_name,
            "r_tar": self.r_tar,
            "r_ran": self.r_ran,
            "r_improvement": self.r_improvement,
            "r_original_tar": self.r_original_tar,
            "r_original_ran": self.r_original_ran,
            "hda_lcc_curve": self.hda_lcc_curve,
            "random_lcc_curve": self.random_lcc_curve,
            "hda_removal_fractions": self.hda_removal_fractions,
            "random_removal_fractions": self.random_removal_fractions,
            "added_edges": [(str(u), str(v)) for u, v in self.added_edges],
            "initial_nodes": self.initial_nodes,
            "initial_edges": self.initial_edges,
            "final_edges": self.final_edges,
            "budget": self.budget,
        }


@dataclass
class EvaluationResult:
    """统一评估结果（可包含 dismant 或 construct 或两者）"""
    task_type: str  # "dismant", "construct", "both"
    graph_name: str
    
    # 任务结果
    dismant_result: Optional[DismantResult] = None
    construct_result: Optional[ConstructResult] = None
    
    # 对比基线结果
    baseline_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            "task_type": self.task_type,
            "graph_name": self.graph_name,
        }
        if self.dismant_result:
            result["dismant"] = self.dismant_result.to_dict()
        if self.construct_result:
            result["construct"] = self.construct_result.to_dict()
        if self.baseline_results:
            result["baselines"] = self.baseline_results
        return result
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存结果到 JSON 文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class UnifiedEvaluator:
    """
    统一评估器
    
    支持两种任务模式：
    1. Dismant（拆解）: 评估攻击序列的效果，计算 R_res 和 LCC 曲线
    2. Construct（构造）: 评估重构网络的韧性，计算 R_tar 和 R_ran
    """
    
    def __init__(
        self,
        collapse_threshold: float = 0.2,
        random_runs: int = 10,
        random_seed: int = 42,
    ):
        """
        初始化评估器
        
        Args:
            collapse_threshold: 网络崩溃阈值（LCC 低于此值视为崩溃）
            random_runs: 随机攻击运行次数（用于取平均）
            random_seed: 随机种子
        """
        self.collapse_threshold = collapse_threshold
        self.random_runs = random_runs
        self.random_seed = random_seed
    
    # ==================== Dismant 评估 ====================
    
    def evaluate_dismant(
        self,
        graph: nx.Graph,
        attack_sequence: List[Any],
        method_name: str = "LLM",
        graph_name: str = "unknown",
    ) -> DismantResult:
        """
        评估拆解任务
        
        给定一个攻击序列，计算其对网络的破坏效果。
        
        Args:
            graph: 初始网络图
            attack_sequence: 要移除的节点序列
            method_name: 方法名称
            graph_name: 图名称
        
        Returns:
            DismantResult: 拆解评估结果
        """
        g = graph.copy()
        initial_nodes = g.number_of_nodes()
        initial_edges = g.number_of_edges()
        
        # 记录 LCC 曲线
        removal_fractions = [0.0]
        lcc_curve = [self._compute_lcc_ratio(g, initial_nodes)]
        
        # 执行攻击序列
        for node in attack_sequence:
            if node in g:
                g.remove_node(node)
            
            removal_frac = (initial_nodes - g.number_of_nodes()) / initial_nodes
            removal_fractions.append(removal_frac)
            lcc_curve.append(self._compute_lcc_ratio(g, initial_nodes))
        
        # 计算 R_res
        r_res = self._compute_r_res(removal_fractions, lcc_curve)
        
        # 找崩溃点
        collapse_fraction = self._find_collapse_point(lcc_curve, removal_fractions)
        
        return DismantResult(
            method_name=method_name,
            graph_name=graph_name,
            r_res=r_res,
            lcc_curve=lcc_curve,
            removal_fractions=removal_fractions,
            attack_sequence=attack_sequence,
            collapse_fraction=collapse_fraction,
            collapse_threshold=self.collapse_threshold,
            initial_nodes=initial_nodes,
            initial_edges=initial_edges,
            budget=len(attack_sequence),
        )
    
    def evaluate_dismant_with_baselines(
        self,
        graph: nx.Graph,
        attack_sequence: Optional[List[Any]] = None,
        method_name: str = "LLM",
        graph_name: str = "unknown",
        budget: Optional[int] = None,
        include_hda: bool = True,
        include_random: bool = True,
    ) -> EvaluationResult:
        """
        评估拆解任务并与基线对比
        
        Args:
            graph: 初始网络图
            attack_sequence: LLM/自定义方法的攻击序列（可选）
            method_name: 方法名称
            graph_name: 图名称
            budget: 攻击预算（默认为节点数的 30%）
            include_hda: 是否包含 HDA 基线
            include_random: 是否包含随机攻击基线
        
        Returns:
            EvaluationResult: 包含主方法和基线的评估结果
        """
        budget = budget or int(graph.number_of_nodes() * 0.3)
        
        # 主方法评估
        main_result = None
        if attack_sequence:
            main_result = self.evaluate_dismant(
                graph=graph,
                attack_sequence=attack_sequence[:budget],
                method_name=method_name,
                graph_name=graph_name,
            )
        
        # 基线评估
        baselines = {}
        
        if include_hda:
            hda = HighestDegreeAttack(recalculate=True)
            hda_result = hda.attack(
                graph=graph,
                budget=budget,
                dataset_name="evaluation",
                graph_name=graph_name,
                collapse_threshold=self.collapse_threshold,
            )
            baselines["HDA"] = {
                "r_res": self._compute_r_res(hda_result.removal_fractions, hda_result.lcc_values),
                "collapse_fraction": hda_result.find_collapse_point(self.collapse_threshold),
                "lcc_curve": hda_result.lcc_values,
                "removal_fractions": hda_result.removal_fractions,
            }
        
        if include_random:
            random_attack = RandomAttack(seed=self.random_seed)
            random_results = random_attack.attack_multiple_runs(
                graph=graph,
                budget=budget,
                num_runs=self.random_runs,
                dataset_name="evaluation",
                graph_name=graph_name,
                collapse_threshold=self.collapse_threshold,
            )
            avg_result = random_results["average_result"]
            baselines["Random"] = {
                "r_res": self._compute_r_res(avg_result.removal_fractions, avg_result.lcc_values),
                "collapse_fraction": avg_result.collapse_fraction,
                "lcc_curve": avg_result.lcc_values,
                "removal_fractions": avg_result.removal_fractions,
                "std_r_res": random_results["std_r_res"],
            }
        
        return EvaluationResult(
            task_type="dismant",
            graph_name=graph_name,
            dismant_result=main_result,
            baseline_results=baselines,
        )
    
    # ==================== Construct 评估 ====================
    
    def evaluate_construct(
        self,
        original_graph: nx.Graph,
        reconstructed_graph: nx.Graph,
        added_edges: List[Tuple[Any, Any]],
        method_name: str = "LLM",
        graph_name: str = "unknown",
        attack_budget: Optional[int] = None,
    ) -> ConstructResult:
        """
        评估构造任务
        
        对重构后的网络分别用 HDA 和 Random 攻击，计算 R_tar 和 R_ran。
        
        Args:
            original_graph: 原始网络图
            reconstructed_graph: 重构后的网络图
            added_edges: 添加的边列表
            method_name: 方法名称
            graph_name: 图名称
            attack_budget: 攻击预算（默认为节点数的 30%）
        
        Returns:
            ConstructResult: 构造评估结果
        """
        attack_budget = attack_budget or int(reconstructed_graph.number_of_nodes() * 0.3)
        
        # 对重构图进行 HDA 攻击
        hda = HighestDegreeAttack(recalculate=True)
        hda_result = hda.attack(
            graph=reconstructed_graph,
            budget=attack_budget,
            dataset_name="construct_eval",
            graph_name=graph_name,
            collapse_threshold=self.collapse_threshold,
        )
        r_tar = self._compute_r_res(hda_result.removal_fractions, hda_result.lcc_values)
        
        # 对重构图进行随机攻击
        random_attack = RandomAttack(seed=self.random_seed)
        random_results = random_attack.attack_multiple_runs(
            graph=reconstructed_graph,
            budget=attack_budget,
            num_runs=self.random_runs,
            dataset_name="construct_eval",
            graph_name=graph_name,
            collapse_threshold=self.collapse_threshold,
        )
        avg_random = random_results["average_result"]
        r_ran = self._compute_r_res(avg_random.removal_fractions, avg_random.lcc_values)
        
        # 计算原始图的韧性（用于对比）
        hda_original = hda.attack(
            graph=original_graph,
            budget=attack_budget,
            dataset_name="original",
            graph_name=graph_name,
        )
        r_original_tar = self._compute_r_res(
            hda_original.removal_fractions, hda_original.lcc_values
        )
        
        random_original = random_attack.attack_multiple_runs(
            graph=original_graph,
            budget=attack_budget,
            num_runs=self.random_runs,
            dataset_name="original",
            graph_name=graph_name,
        )
        r_original_ran = self._compute_r_res(
            random_original["average_result"].removal_fractions,
            random_original["average_result"].lcc_values,
        )
        
        # 计算韧性提升
        # 使用 HDA 韧性作为主要指标（更有挑战性）
        if r_original_tar > 0:
            r_improvement = (r_tar - r_original_tar) / r_original_tar
        else:
            r_improvement = 0.0
        
        return ConstructResult(
            method_name=method_name,
            graph_name=graph_name,
            r_tar=r_tar,
            r_ran=r_ran,
            r_improvement=r_improvement,
            hda_lcc_curve=hda_result.lcc_values,
            random_lcc_curve=avg_random.lcc_values,
            hda_removal_fractions=hda_result.removal_fractions,
            random_removal_fractions=avg_random.removal_fractions,
            added_edges=added_edges,
            r_original_tar=r_original_tar,
            r_original_ran=r_original_ran,
            initial_nodes=original_graph.number_of_nodes(),
            initial_edges=original_graph.number_of_edges(),
            final_edges=reconstructed_graph.number_of_edges(),
            budget=len(added_edges),
        )
    
    def evaluate_construct_with_baselines(
        self,
        original_graph: nx.Graph,
        reconstructed_graph: Optional[nx.Graph] = None,
        added_edges: Optional[List[Tuple[Any, Any]]] = None,
        method_name: str = "LLM",
        graph_name: str = "unknown",
        edge_budget: Optional[int] = None,
        attack_budget: Optional[int] = None,
        include_random_construct: bool = True,
        include_degree_construct: bool = True,
    ) -> EvaluationResult:
        """
        评估构造任务并与基线对比
        
        基线策略：
        1. Random Construct: 随机选择边添加
        2. Degree-based Construct: 连接低度数节点
        
        Args:
            original_graph: 原始网络图
            reconstructed_graph: LLM/自定义方法重构的图（可选）
            added_edges: 添加的边列表
            method_name: 方法名称
            graph_name: 图名称
            edge_budget: 添加边的预算
            attack_budget: 评估时的攻击预算
            include_random_construct: 是否包含随机构造基线
            include_degree_construct: 是否包含度数基线
        
        Returns:
            EvaluationResult: 包含主方法和基线的评估结果
        """
        edge_budget = edge_budget or int(original_graph.number_of_edges() * 0.1)
        attack_budget = attack_budget or int(original_graph.number_of_nodes() * 0.3)
        
        # 主方法评估
        main_result = None
        if reconstructed_graph and added_edges:
            main_result = self.evaluate_construct(
                original_graph=original_graph,
                reconstructed_graph=reconstructed_graph,
                added_edges=added_edges,
                method_name=method_name,
                graph_name=graph_name,
                attack_budget=attack_budget,
            )
        
        # 基线评估
        baselines = {}
        
        if include_random_construct:
            # 随机添加边
            random_graph, random_edges = self._random_construct(
                original_graph, edge_budget
            )
            random_result = self.evaluate_construct(
                original_graph=original_graph,
                reconstructed_graph=random_graph,
                added_edges=random_edges,
                method_name="RandomConstruct",
                graph_name=graph_name,
                attack_budget=attack_budget,
            )
            baselines["RandomConstruct"] = random_result.to_dict()
        
        if include_degree_construct:
            # 度数引导添加边
            degree_graph, degree_edges = self._degree_based_construct(
                original_graph, edge_budget
            )
            degree_result = self.evaluate_construct(
                original_graph=original_graph,
                reconstructed_graph=degree_graph,
                added_edges=degree_edges,
                method_name="DegreeConstruct",
                graph_name=graph_name,
                attack_budget=attack_budget,
            )
            baselines["DegreeConstruct"] = degree_result.to_dict()
        
        return EvaluationResult(
            task_type="construct",
            graph_name=graph_name,
            construct_result=main_result,
            baseline_results=baselines,
        )
    
    # ==================== 辅助方法 ====================
    
    def _compute_lcc_ratio(self, graph: nx.Graph, initial_nodes: int) -> float:
        """计算 LCC 比例"""
        if graph.number_of_nodes() == 0:
            return 0.0
        ccs = list(nx.connected_components(graph))
        if not ccs:
            return 0.0
        largest = max(len(cc) for cc in ccs)
        return largest / initial_nodes
    
    def _compute_r_res(
        self, removal_fractions: List[float], lcc_values: List[float]
    ) -> float:
        """计算 R_res（韧性积分）"""
        if len(removal_fractions) < 2 or len(lcc_values) < 2:
            return 0.0
        n = min(len(removal_fractions), len(lcc_values))
        x = np.array(removal_fractions[:n])
        y = np.array(lcc_values[:n])
        return float(np.trapezoid(y, x))
    
    def _find_collapse_point(
        self, lcc_values: List[float], removal_fractions: List[float]
    ) -> Optional[float]:
        """找到崩溃点"""
        for i in range(1, len(lcc_values)):
            if lcc_values[i] < self.collapse_threshold:
                if lcc_values[i-1] >= self.collapse_threshold:
                    # 线性插值
                    y1, y2 = lcc_values[i-1], lcc_values[i]
                    x1, x2 = removal_fractions[i-1], removal_fractions[i]
                    if y1 != y2:
                        t = (y1 - self.collapse_threshold) / (y1 - y2)
                        return x1 + t * (x2 - x1)
                return removal_fractions[i]
        return None
    
    def _random_construct(
        self, graph: nx.Graph, budget: int
    ) -> Tuple[nx.Graph, List[Tuple]]:
        """随机添加边"""
        import random
        random.seed(self.random_seed)
        
        g = graph.copy()
        nodes = list(g.nodes())
        added = []
        attempts = 0
        max_attempts = budget * 100
        
        while len(added) < budget and attempts < max_attempts:
            u, v = random.sample(nodes, 2)
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                added.append((u, v))
            attempts += 1
        
        return g, added
    
    def _degree_based_construct(
        self, graph: nx.Graph, budget: int
    ) -> Tuple[nx.Graph, List[Tuple]]:
        """度数引导添加边（优先连接低度数节点）"""
        g = graph.copy()
        added = []
        
        for _ in range(budget):
            # 按度数排序，优先选择低度数节点
            degrees = dict(g.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n])
            
            # 尝试在低度数节点之间添加边
            found = False
            for i in range(min(20, len(sorted_nodes))):
                for j in range(i + 1, min(20, len(sorted_nodes))):
                    u, v = sorted_nodes[i], sorted_nodes[j]
                    if not g.has_edge(u, v):
                        g.add_edge(u, v)
                        added.append((u, v))
                        found = True
                        break
                if found:
                    break
            
            if not found:
                break
        
        return g, added


# ==================== 便捷函数 ====================

def evaluate_dismant(
    graph: nx.Graph,
    attack_sequence: List[Any],
    method_name: str = "LLM",
    graph_name: str = "unknown",
    include_baselines: bool = True,
) -> Union[DismantResult, EvaluationResult]:
    """便捷函数：评估拆解任务"""
    evaluator = UnifiedEvaluator()
    if include_baselines:
        return evaluator.evaluate_dismant_with_baselines(
            graph=graph,
            attack_sequence=attack_sequence,
            method_name=method_name,
            graph_name=graph_name,
        )
    else:
        return evaluator.evaluate_dismant(
            graph=graph,
            attack_sequence=attack_sequence,
            method_name=method_name,
            graph_name=graph_name,
        )


def evaluate_construct(
    original_graph: nx.Graph,
    reconstructed_graph: nx.Graph,
    added_edges: List[Tuple[Any, Any]],
    method_name: str = "LLM",
    graph_name: str = "unknown",
    include_baselines: bool = True,
) -> Union[ConstructResult, EvaluationResult]:
    """便捷函数：评估构造任务"""
    evaluator = UnifiedEvaluator()
    if include_baselines:
        return evaluator.evaluate_construct_with_baselines(
            original_graph=original_graph,
            reconstructed_graph=reconstructed_graph,
            added_edges=added_edges,
            method_name=method_name,
            graph_name=graph_name,
        )
    else:
        return evaluator.evaluate_construct(
            original_graph=original_graph,
            reconstructed_graph=reconstructed_graph,
            added_edges=added_edges,
            method_name=method_name,
            graph_name=graph_name,
        )
