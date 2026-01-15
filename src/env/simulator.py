# -*- coding: utf-8 -*-
"""
NetworkEnvironment: 网络环境模拟器
负责维护图状态 G_t、执行 O(N²) -> O(N) 的谱梯度剪枝、管理候选操作。

核心功能：
1. 图状态管理 (Graph State Management)
2. 谱梯度计算与候选剪枝 (Spectral Gradient Pruning)
3. 操作执行与状态转移 (Action Execution)
"""

from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod


class TaskType(Enum):
    """任务类型枚举"""
    DISMANTLE = "dismantle"  # 拆解任务：最小化韧性
    CONSTRUCT = "construct"   # 构造任务：最大化韧性


@dataclass
class NodeInfo:
    """节点信息数据类"""
    node_id: Union[int, str]
    degree: int
    clustering_coeff: float
    betweenness: float
    semantic_desc: str = ""  # 语义描述
    spectral_score: float = 0.0  # 谱梯度分数
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "degree": self.degree,
            "clustering_coeff": self.clustering_coeff,
            "betweenness": self.betweenness,
            "semantic_desc": self.semantic_desc,
            "spectral_score": self.spectral_score
        }


@dataclass
class Operation:
    """操作数据类"""
    op_id: str
    op_type: str  # "remove_node", "add_edge", "remove_edge"
    target: Union[int, str, Tuple]  # 目标节点或边
    predicted_impact: float = 0.0  # 预测影响分数
    
    def to_dict(self) -> Dict:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "target": self.target,
            "predicted_impact": self.predicted_impact
        }


@dataclass
class GraphState:
    """图状态快照"""
    step: int
    num_nodes: int
    num_edges: int
    largest_cc_size: int  # 最大连通分量大小
    resilience_score: float  # 当前韧性分数
    removed_nodes: Set = field(default_factory=set)
    added_edges: Set = field(default_factory=set)


class NetworkEnvironment:
    """
    网络环境模拟器
    
    负责：
    1. 维护和更新图状态 G_t
    2. 执行谱梯度剪枝，将候选空间从 O(N²) 降至 O(N)
    3. 计算并返回韧性指标
    4. 执行操作并更新图状态
    
    Attributes:
        graph (nx.Graph): 当前网络图
        task_type (TaskType): 任务类型
        budget (int): 操作预算
        current_step (int): 当前步骤
        history (List[GraphState]): 状态历史
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        task_type: TaskType = TaskType.DISMANTLE,
        budget: int = 10,
        spectral_top_k: int = 50,
        node_semantics: Optional[Dict[Union[int, str], str]] = None
    ):
        """
        初始化网络环境
        
        Args:
            graph: NetworkX 图对象
            task_type: 任务类型 (拆解/构造)
            budget: 操作预算 (总步数)
            spectral_top_k: 谱梯度剪枝后保留的 Top-K 候选数
            node_semantics: 节点语义描述字典 {node_id: "语义描述"}
        """
        self.graph = graph.copy()
        self.initial_graph = graph.copy()
        self.task_type = task_type
        self.budget = budget
        self.spectral_top_k = spectral_top_k
        self.node_semantics = node_semantics or {}
        
        self.current_step = 0
        self.history: List[GraphState] = []
        
        # 缓存计算结果
        self._laplacian_cache = None
        self._fiedler_cache = None
        self._betweenness_cache = None
        
        # 记录初始状态
        self._record_state()
    
    def reset(self) -> 'NetworkEnvironment':
        """
        重置环境到初始状态
        
        Returns:
            self: 重置后的环境实例
        """
        self.graph = self.initial_graph.copy()
        self.current_step = 0
        self.history = []
        self._invalidate_cache()
        self._record_state()
        return self
    
    # ==================== 谱梯度剪枝接口 ====================
    
    def compute_spectral_gradient(self) -> Dict[Union[int, str], float]:
        """
        计算所有节点的谱梯度分数
        
        基于 Fiedler 向量 (代数连通度对应的特征向量) 计算节点移除
        对网络连通性的影响。这是将候选空间从 O(N²) 降至 O(N) 的关键。
        
        数学原理：
        - 计算图拉普拉斯矩阵 L = D - A
        - 求解第二小特征值 λ₂ (Fiedler value) 及其特征向量 v₂
        - 节点 i 的谱梯度 ≈ |v₂[i]|² × d_i (度数加权)
        
        Returns:
            Dict[node_id, spectral_score]: 节点谱梯度分数字典
        
        Complexity:
            O(N) 计算复杂度 (使用稀疏矩阵特征值求解)
        """
        # TODO: 实现谱梯度计算
        # 1. 获取或计算 Fiedler 向量
        # 2. 计算每个节点的谱梯度分数
        # 3. 返回分数字典
        raise NotImplementedError("compute_spectral_gradient")
    
    def _compute_fiedler_vector(self) -> np.ndarray:
        """
        计算 Fiedler 向量 (第二小特征值对应的特征向量)
        
        Returns:
            np.ndarray: Fiedler 向量，shape = [num_nodes]
        """
        # TODO: 使用 scipy.sparse.linalg.eigsh 高效计算
        raise NotImplementedError("_compute_fiedler_vector")
    
    def _compute_laplacian_matrix(self) -> np.ndarray:
        """
        计算图的拉普拉斯矩阵 L = D - A
        
        使用稀疏矩阵格式以支持大规模图。
        
        Returns:
            稀疏拉普拉斯矩阵
        """
        # TODO: 实现稀疏拉普拉斯矩阵计算
        raise NotImplementedError("_compute_laplacian_matrix")
    
    def prune_candidates(
        self, 
        candidate_type: str = "node",
        top_k: Optional[int] = None
    ) -> List[Union[int, str, Tuple]]:
        """
        基于谱梯度剪枝候选集
        
        将全量候选 O(N²) 或 O(N) 剪枝为 Top-K 高影响力候选。
        
        Args:
            candidate_type: 候选类型
                - "node": 节点移除候选 (用于 dismantle)
                - "edge": 边添加候选 (用于 construct)
        
        Returns:
            剪枝后的候选列表，按谱梯度分数降序排列
        
        Note:
            对于 edge 类型，会使用近似算法避免 O(N²) 遍历
        """
        # 轻量可用实现（避免推理流程因 TODO 中断）：
        # - node: 按度数降序取 Top-K
        # - edge: 在高阶节点之间提议若干条不存在的边
        k = top_k or self.spectral_top_k
        if candidate_type == "node":
            nodes = list(self.graph.nodes())
            if not nodes:
                return []
            nodes_sorted = sorted(nodes, key=lambda n: self.graph.degree(n), reverse=True)
            return nodes_sorted[: min(k, len(nodes_sorted))]

        if candidate_type == "edge":
            nodes = list(self.graph.nodes())
            if len(nodes) < 2:
                return []
            nodes_sorted = sorted(nodes, key=lambda n: self.graph.degree(n), reverse=True)
            pool = nodes_sorted[: min(max(k, 50), len(nodes_sorted))]
            candidates: List[Tuple] = []
            seen = set()
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    u, v = pool[i], pool[j]
                    if self.graph.has_edge(u, v):
                        continue
                    key = (u, v) if u <= v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((u, v))
                    if len(candidates) >= k:
                        return candidates
            return candidates

        raise ValueError(f"Unsupported candidate_type: {candidate_type}")

    # ==================== 推理脚本需要的简化操作接口 ====================

    def remove_node(self, node_id: Union[int, str]) -> None:
        """移除节点（推理脚本使用）"""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self.current_step += 1
            self._invalidate_cache()
            self._record_state()

    def add_edge(self, u: Union[int, str], v: Union[int, str]) -> None:
        """添加边（推理脚本使用）"""
        if u in self.graph and v in self.graph and u != v:
            self.graph.add_edge(u, v)
            self.current_step += 1
            self._invalidate_cache()
            self._record_state()
    
    # ==================== 操作执行接口 ====================
    
    def get_candidate_operations(self) -> List[Operation]:
        """
        获取当前步骤的候选操作列表
        
        根据任务类型和谱梯度剪枝结果生成候选操作。
        
        Returns:
            List[Operation]: 候选操作列表
        """
        # TODO: 生成候选操作
        raise NotImplementedError("get_candidate_operations")
    
    def execute_operation(self, operation: Operation) -> Tuple[float, bool]:
        """
        执行操作并更新图状态
        
        Args:
            operation: 要执行的操作
        
        Returns:
            Tuple[reward, done]:
                - reward: 韧性变化量 ΔR_res
                - done: 是否达到终止条件
        """
        # TODO: 实现操作执行
        # 1. 根据操作类型修改图
        # 2. 计算韧性变化
        # 3. 更新状态历史
        # 4. 检查终止条件
        raise NotImplementedError("execute_operation")
    
    def step(self, action_idx: int) -> Tuple[GraphState, float, bool, Dict]:
        """
        执行一步操作 (强化学习风格接口)
        
        Args:
            action_idx: 候选操作列表中的索引
        
        Returns:
            Tuple[state, reward, done, info]:
                - state: 新状态
                - reward: 奖励 (韧性变化)
                - done: 是否结束
                - info: 额外信息
        """
        # TODO: 实现 step 逻辑
        raise NotImplementedError("step")
    
    # ==================== 状态查询接口 ====================
    
    def get_current_state(self) -> GraphState:
        """获取当前图状态"""
        return self.history[-1] if self.history else self._create_state_snapshot()
    
    def get_node_info(self, node_id: Union[int, str]) -> NodeInfo:
        """
        获取节点详细信息
        
        Args:
            node_id: 节点标识符
        
        Returns:
            NodeInfo: 节点信息数据类
        """
        # TODO: 实现节点信息提取
        raise NotImplementedError("get_node_info")
    
    def get_candidate_nodes_info(self, top_k: Optional[int] = None) -> List[NodeInfo]:
        """
        获取候选节点的详细信息 (用于 OCG 构建)
        
        Args:
            top_k: 返回 Top-K 候选，默认使用 spectral_top_k
        
        Returns:
            List[NodeInfo]: 候选节点信息列表
        """
        # TODO: 实现候选节点信息提取
        raise NotImplementedError("get_candidate_nodes_info")
    
    def get_subgraph(
        self, 
        center_node: Union[int, str], 
        hops: int = 1
    ) -> nx.Graph:
        """
        提取以指定节点为中心的 k-hop 子图
        
        Args:
            center_node: 中心节点
            hops: 跳数
        
        Returns:
            子图
        """
        # TODO: 实现子图提取
        raise NotImplementedError("get_subgraph")
    
    # ==================== 内部方法 ====================
    
    def _record_state(self) -> None:
        """记录当前状态到历史"""
        state = self._create_state_snapshot()
        self.history.append(state)
    
    def _create_state_snapshot(self) -> GraphState:
        """创建当前状态快照"""
        from .metrics import ResilienceMetrics
        
        largest_cc = max(nx.connected_components(self.graph), key=len) \
            if self.graph.number_of_nodes() > 0 else set()
        
        return GraphState(
            step=self.current_step,
            num_nodes=self.graph.number_of_nodes(),
            num_edges=self.graph.number_of_edges(),
            largest_cc_size=len(largest_cc),
            resilience_score=0.0,  # TODO: 计算实际韧性分数
            removed_nodes=set(),
            added_edges=set()
        )
    
    def _invalidate_cache(self) -> None:
        """使缓存失效"""
        self._laplacian_cache = None
        self._fiedler_cache = None
        self._betweenness_cache = None
    
    # ==================== 图生成工具 ====================
    
    @staticmethod
    def generate_ba_graph(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
        """
        生成 Barabási-Albert 无标度网络
        
        Args:
            n: 节点数
            m: 每个新节点连接的边数
            seed: 随机种子
        
        Returns:
            BA 图
        """
        return nx.barabasi_albert_graph(n, m, seed=seed)
    
    @staticmethod
    def generate_er_graph(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
        """
        生成 Erdős-Rényi 随机网络
        
        Args:
            n: 节点数
            p: 边存在概率
            seed: 随机种子
        
        Returns:
            ER 图
        """
        return nx.erdos_renyi_graph(n, p, seed=seed)
    
    @staticmethod
    def load_graph(filepath: str, format: str = "auto") -> nx.Graph:
        """
        从文件加载图
        
        Args:
            filepath: 文件路径
            format: 文件格式 ("auto", "edgelist", "adjlist", "gml", "graphml")
        
        Returns:
            加载的图
        """
        # 自动根据后缀推断格式（推理脚本常直接传 .gml/.graphml 路径）
        if format is None or str(format).lower() == "auto":
            suffix = Path(filepath).suffix.lower().lstrip(".")
            if suffix in {"gml", "graphml", "adjlist", "edgelist"}:
                format = suffix
            else:
                # 未识别后缀时，回退到 edgelist（兼容旧行为）
                format = "edgelist"

        def _load_topology_zoo_gml(fp: str) -> nx.Graph:
            """
            更稳健地加载 Topology Zoo 风格的 .gml：
            - 节点使用 id 作为 key（避免 label 缺失导致的 'None' 重复）
            - 去重重复边（Topology Zoo 文件里可能出现相同 source/target 多次）
            """
            g = nx.Graph()
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            in_node = False
            in_edge = False
            node_id = None
            node_label = None
            edge_u = None
            edge_v = None

            for raw in lines:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("node"):
                    in_node = True
                    node_id = None
                    node_label = None
                    continue
                if line.startswith("edge"):
                    in_edge = True
                    edge_u = None
                    edge_v = None
                    continue

                if line == "]":
                    if in_node and node_id is not None:
                        g.add_node(node_id, label=node_label)
                    if in_edge and edge_u is not None and edge_v is not None:
                        # 无向图：重复边直接 add_edge 即可（Graph 会自动去重）
                        g.add_edge(edge_u, edge_v)
                    in_node = False
                    in_edge = False
                    continue

                if in_node:
                    if line.startswith("id "):
                        try:
                            node_id = int(line.split()[1])
                        except Exception:
                            pass
                    elif line.startswith('label "'):
                        # label "xxx"
                        if line.endswith('"') and len(line) >= 8:
                            node_label = line[len('label "'): -1]
                elif in_edge:
                    if line.startswith("source "):
                        try:
                            edge_u = int(line.split()[1])
                        except Exception:
                            pass
                    elif line.startswith("target "):
                        try:
                            edge_v = int(line.split()[1])
                        except Exception:
                            pass

            return g

        loaders = {
            "edgelist": nx.read_edgelist,
            "adjlist": nx.read_adjlist,
            # Topology Zoo 的 .gml 可能出现缺失 label/重复边，用自定义解析更稳健
            "gml": _load_topology_zoo_gml,
            "graphml": nx.read_graphml,
        }
        if format not in loaders:
            raise ValueError(f"Unsupported format: {format}")
        return loaders[format](filepath)


# ==================== 便捷函数 ====================

def create_environment(
    graph_type: str = "ba",
    num_nodes: int = 100,
    task: str = "dismantle",
    **kwargs
) -> NetworkEnvironment:
    """
    便捷函数：创建网络环境
    
    Args:
        graph_type: 图类型 ("ba", "er", "custom")
        num_nodes: 节点数量
        task: 任务类型 ("dismantle", "construct")
        **kwargs: 其他参数
    
    Returns:
        NetworkEnvironment 实例
    """
    if graph_type == "ba":
        m = kwargs.get("m", 3)
        graph = NetworkEnvironment.generate_ba_graph(num_nodes, m)
    elif graph_type == "er":
        p = kwargs.get("p", 0.05)
        graph = NetworkEnvironment.generate_er_graph(num_nodes, p)
    elif graph_type == "custom":
        graph = kwargs.get("graph")
        if graph is None:
            raise ValueError("Must provide 'graph' for custom type")
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    task_type = TaskType.DISMANTLE if task == "dismantle" else TaskType.CONSTRUCT
    
    return NetworkEnvironment(
        graph=graph,
        task_type=task_type,
        budget=kwargs.get("budget", 10),
        spectral_top_k=kwargs.get("spectral_top_k", 50),
        node_semantics=kwargs.get("node_semantics")
    )
