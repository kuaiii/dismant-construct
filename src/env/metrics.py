# -*- coding: utf-8 -*-
"""
ResilienceMetrics: 网络韧性指标计算
负责计算 R_res (韧性积分) 及相关韧性指标。

核心指标：
1. R_res: 韧性面积积分 (Resilience Area Integral)
2. LCC Curve: 最大连通分量变化曲线
3. 代数连通度 (Algebraic Connectivity)
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class ResilienceResult:
    """韧性计算结果数据类"""
    r_res: float  # 韧性积分值
    lcc_curve: List[float]  # 最大连通分量比例曲线
    steps: List[int]  # 步骤序列
    algebraic_connectivity: float  # 代数连通度
    effective_resistance: float  # 有效电阻 (可选指标)


class ResilienceMetrics:
    """
    网络韧性指标计算器
    
    核心指标 R_res 定义：
    R_res = ∫₀¹ σ(q) dq
    
    其中：
    - q = 移除节点比例 (0 到 1)
    - σ(q) = |LCC(q)| / N，即最大连通分量占总节点数的比例
    - R_res ∈ [0, 1]，值越大表示网络越有韧性
    
    计算方法：
    使用梯形积分法近似计算曲线下面积
    """
    
    def __init__(self, normalize: bool = True):
        """
        初始化韧性计算器
        
        Args:
            normalize: 是否归一化结果到 [0, 1]
        """
        self.normalize = normalize
    
    def compute_r_res(
        self, 
        lcc_curve: List[float], 
        num_steps: Optional[int] = None
    ) -> float:
        """
        计算韧性积分 R_res
        
        使用梯形积分法计算 LCC 曲线下的面积。
        
        Args:
            lcc_curve: 最大连通分量比例序列 [σ(0), σ(1/n), ..., σ(1)]
            num_steps: 总步骤数，用于归一化
        
        Returns:
            float: 韧性积分值 R_res
        
        Example:
            >>> metrics = ResilienceMetrics()
            >>> lcc_curve = [1.0, 0.8, 0.5, 0.3, 0.1, 0.0]
            >>> r_res = metrics.compute_r_res(lcc_curve)
        """
        if len(lcc_curve) < 2:
            return 0.0
        
        # 使用梯形积分法
        # R_res = Σ (σ(i) + σ(i+1)) / 2 × Δq
        curve = np.array(lcc_curve)
        n_points = len(curve)
        
        # 步长 Δq = 1 / (n_points - 1)
        delta_q = 1.0 / (n_points - 1) if n_points > 1 else 1.0
        
        # 梯形积分
        r_res = np.trapz(curve, dx=delta_q)
        
        if self.normalize:
            # 归一化：最大可能面积为 1.0
            r_res = min(r_res, 1.0)
        
        return float(r_res)
    
    def compute_lcc_ratio(self, graph: nx.Graph) -> float:
        """
        计算当前图的 LCC 比例
        
        Args:
            graph: NetworkX 图
        
        Returns:
            float: |LCC| / N
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        connected_components = list(nx.connected_components(graph))
        if not connected_components:
            return 0.0
        
        largest_cc_size = max(len(cc) for cc in connected_components)
        return largest_cc_size / graph.number_of_nodes()
    
    def compute_algebraic_connectivity(self, graph: nx.Graph) -> float:
        """
        计算代数连通度 (Fiedler Value)
        
        代数连通度 λ₂ 是拉普拉斯矩阵的第二小特征值，
        反映网络的整体连通性强度。
        
        Args:
            graph: NetworkX 图
        
        Returns:
            float: 代数连通度 λ₂
        """
        if graph.number_of_nodes() < 2:
            return 0.0
        
        if not nx.is_connected(graph):
            return 0.0
        
        # 使用 NetworkX 内置函数
        try:
            return nx.algebraic_connectivity(graph)
        except Exception:
            return 0.0
    
    def compute_effective_resistance(self, graph: nx.Graph) -> float:
        """
        计算总有效电阻 (Total Effective Resistance)
        
        有效电阻反映网络的全局连通冗余度。
        
        Args:
            graph: NetworkX 图
        
        Returns:
            float: 总有效电阻
        """
        # TODO: 实现有效电阻计算
        # R_eff = N × Σ(1/λ_i)，其中 λ_i 是非零特征值
        raise NotImplementedError("compute_effective_resistance")
    
    def simulate_attack_sequence(
        self, 
        graph: nx.Graph, 
        attack_sequence: List[Union[int, str]],
        metric: str = "lcc"
    ) -> Tuple[List[float], float]:
        """
        模拟攻击序列并计算韧性曲线
        
        按给定顺序逐步移除节点，记录 LCC 变化曲线。
        
        Args:
            graph: 初始网络图
            attack_sequence: 节点移除顺序
            metric: 记录的指标 ("lcc", "algebraic_connectivity")
        
        Returns:
            Tuple[curve, r_res]:
                - curve: 指标变化曲线
                - r_res: 韧性积分值
        """
        g = graph.copy()
        n_initial = g.number_of_nodes()
        
        curve = []
        
        # 初始状态
        if metric == "lcc":
            curve.append(self.compute_lcc_ratio(g))
        elif metric == "algebraic_connectivity":
            curve.append(self.compute_algebraic_connectivity(g))
        
        # 逐步移除节点
        for node in attack_sequence:
            if node in g:
                g.remove_node(node)
            
            if metric == "lcc":
                # 重新计算 LCC 比例（相对于原始节点数）
                if g.number_of_nodes() == 0:
                    ratio = 0.0
                else:
                    ccs = list(nx.connected_components(g))
                    largest_cc_size = max(len(cc) for cc in ccs) if ccs else 0
                    ratio = largest_cc_size / n_initial
                curve.append(ratio)
            elif metric == "algebraic_connectivity":
                curve.append(self.compute_algebraic_connectivity(g))
        
        r_res = self.compute_r_res(curve)
        
        return curve, r_res
    
    def compute_impact_score(
        self, 
        graph: nx.Graph, 
        target_node: Union[int, str]
    ) -> float:
        """
        计算移除单个节点的影响分数
        
        用于生成训练数据的 auxiliary_labels。
        
        Args:
            graph: 当前网络图
            target_node: 目标节点
        
        Returns:
            float: 影响分数 (0-1)，值越大表示影响越大
        
        Note:
            影响分数计算考虑：
            1. LCC 变化量
            2. 代数连通度变化量
            3. 节点中心性
        """
        if target_node not in graph:
            return 0.0
        
        # 移除前的状态
        lcc_before = self.compute_lcc_ratio(graph)
        
        # 模拟移除
        g_temp = graph.copy()
        g_temp.remove_node(target_node)
        
        # 移除后的状态
        if g_temp.number_of_nodes() == 0:
            lcc_after = 0.0
        else:
            ccs = list(nx.connected_components(g_temp))
            largest_cc = max(len(cc) for cc in ccs) if ccs else 0
            lcc_after = largest_cc / graph.number_of_nodes()
        
        # 影响分数 = LCC 下降比例
        impact = lcc_before - lcc_after
        
        # 归一化到 [0, 1]
        return max(0.0, min(1.0, impact))
    
    def batch_compute_impact_scores(
        self, 
        graph: nx.Graph, 
        candidate_nodes: List[Union[int, str]]
    ) -> Dict[Union[int, str], float]:
        """
        批量计算候选节点的影响分数
        
        Args:
            graph: 当前网络图
            candidate_nodes: 候选节点列表
        
        Returns:
            Dict[node_id, impact_score]: 影响分数字典
        """
        scores = {}
        for node in candidate_nodes:
            scores[node] = self.compute_impact_score(graph, node)
        return scores
    
    def compute_edge_gain(
        self, 
        graph: nx.Graph, 
        edge: Tuple[Union[int, str], Union[int, str]]
    ) -> float:
        """
        计算添加单条边的韧性增益（用于 construct 任务）
        
        增益计算逻辑：
        1. 如果边连接了两个不同的连通分量，增益很大（合并分量）
        2. 如果边在同一连通分量内，增益较小（增加冗余路径）
        
        Args:
            graph: 当前网络图
            edge: 要添加的边 (u, v)
        
        Returns:
            float: 增益分数 (0-1)，值越大表示增益越大
        """
        u, v = edge
        if u not in graph or v not in graph:
            return 0.0
        if graph.has_edge(u, v):
            return 0.0  # 边已存在
        
        # 获取添加边前的状态
        n_initial = graph.number_of_nodes()
        lcc_before = self.compute_lcc_ratio(graph)
        
        # 模拟添加边
        g_temp = graph.copy()
        g_temp.add_edge(u, v)
        
        # 计算添加边后的 LCC 比例
        lcc_after = self.compute_lcc_ratio(g_temp)
        
        # 增益 = LCC 增长比例
        gain = lcc_after - lcc_before
        
        # 如果 LCC 没变化，考虑代数连通度的变化（增加冗余路径）
        if abs(gain) < 1e-6:
            # 通过度数和聚类系数估算增益
            # 连接低聚类系数区域的边有更高增益
            try:
                cc_u = nx.clustering(graph, u)
                cc_v = nx.clustering(graph, v)
                # 聚类系数越低，添加边的增益越大
                gain = (2.0 - cc_u - cc_v) / (2.0 * n_initial) * 0.1
            except Exception:
                gain = 0.001
        
        # 归一化到 [0, 1]
        return max(0.0, min(1.0, gain))
    
    def batch_compute_edge_gains(
        self, 
        graph: nx.Graph, 
        candidate_edges: List[Tuple[Union[int, str], Union[int, str]]]
    ) -> Dict[Tuple, float]:
        """
        批量计算候选边的增益分数（用于 construct 任务）
        
        Args:
            graph: 当前网络图
            candidate_edges: 候选边列表 [(u1, v1), (u2, v2), ...]
        
        Returns:
            Dict[edge, gain_score]: 边增益分数字典
        """
        scores = {}
        for edge in candidate_edges:
            scores[edge] = self.compute_edge_gain(graph, edge)
        return scores
    
    def evaluate_full_attack(
        self, 
        graph: nx.Graph, 
        attack_sequence: List[Union[int, str]]
    ) -> ResilienceResult:
        """
        完整评估攻击序列的韧性影响
        
        Args:
            graph: 初始网络图
            attack_sequence: 完整攻击序列
        
        Returns:
            ResilienceResult: 完整韧性评估结果
        """
        lcc_curve, r_res = self.simulate_attack_sequence(
            graph, attack_sequence, metric="lcc"
        )
        
        alg_conn = self.compute_algebraic_connectivity(graph)
        
        return ResilienceResult(
            r_res=r_res,
            lcc_curve=lcc_curve,
            steps=list(range(len(lcc_curve))),
            algebraic_connectivity=alg_conn,
            effective_resistance=0.0  # TODO: 实现
        )


# ==================== 便捷函数 ====================

def compute_r_res(graph: nx.Graph, attack_sequence: List) -> float:
    """便捷函数：计算韧性积分"""
    metrics = ResilienceMetrics()
    _, r_res = metrics.simulate_attack_sequence(graph, attack_sequence)
    return r_res


def compute_node_impact(graph: nx.Graph, node) -> float:
    """便捷函数：计算单节点影响分数"""
    metrics = ResilienceMetrics()
    return metrics.compute_impact_score(graph, node)


def rank_nodes_by_impact(
    graph: nx.Graph, 
    candidates: List
) -> List[Tuple[any, float]]:
    """
    便捷函数：按影响分数排序候选节点
    
    Returns:
        List[(node_id, score)]: 按分数降序排列的列表
    """
    metrics = ResilienceMetrics()
    scores = metrics.batch_compute_impact_scores(graph, candidates)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def calculate_csa(G, centers, alpha=0.2):
    """
    计算控制供给可用性 (Control Supply Availability, CSA)
    CSA = 1/|V_S| * sum(1 - e^(-alpha * k_i))
    
    支持 GPU 加速计算
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        alpha (float): 衰减因子，默认 0.2
        
    Returns:
        float: CSA 值
    """
    if G.number_of_nodes() == 0:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 1.0
    
    # 检查是否使用 GPU 加速
    use_gpu = is_gpu_available() and TORCH_AVAILABLE and len(switch_nodes) > 100
    
    # 预计算连通分量及其包含的控制器数量
    components = list(nx.connected_components(G))
    node_to_k = {}
    
    for comp in components:
        # 计算该分量内的控制器数量
        num_centers_in_comp = len(comp.intersection(centers_set))
        for node in comp:
            node_to_k[node] = num_centers_in_comp
    
    if use_gpu:
        # GPU 加速版本
        k_values = np.array([node_to_k.get(i, 0) for i in switch_nodes], dtype=np.float32)
        k_tensor = torch.tensor(k_values, device=DEVICE)
        
        # 计算 1 - exp(-alpha * k_i)
        csa_tensor = 1 - torch.exp(-alpha * k_tensor)
        total_csa = torch.sum(csa_tensor).item()
    else:
        # CPU 版本
        total_csa = 0.0
        for i in switch_nodes:
            k_i = node_to_k.get(i, 0)
            term = 1 - math.exp(-alpha * k_i)
            total_csa += term
        
    return total_csa / len(switch_nodes)

def calculate_control_entropy(G, centers):
    """
    计算控制熵 (Control Entropy, H_C)
    H_C = - sum(p_j * ln(p_j))
    p_j = A_j / sum(A_m)
    A_j: 控制器 j 的服务域大小 (连通的交换节点数)
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        
    Returns:
        float: H_C 值
    """
    if not centers:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 0.0

    components = list(nx.connected_components(G))
    A_values = []
    
    
    A_map = {} # center -> A_j
    
    for comp in components:
        # 计算该分量内的交换节点数量
        switches_in_comp = [n for n in comp if n not in centers_set]
        size_sw = len(switches_in_comp)
        
        # 找出该分量内的所有控制器
        centers_in_comp = [n for n in comp if n in centers_set]
        
        for c in centers_in_comp:
            A_map[c] = size_sw
    
    A_list = [A_map.get(c, 0) for c in centers]
    sum_A = sum(A_list)
    
    if sum_A == 0:
        return 0.0
        
    H_C = 0.0
    for A_j in A_list:
        if A_j > 0:
            p_j = A_j / sum_A
            H_C += p_j * math.log(p_j)
            
    return -H_C

def calculate_wcp(G, centers, beta=1.0):
    """
    计算加权控制势能 (Weighted Control Potential, WCP)
    WCP = 1/|V_S| * sum_i ( sum_j ( 1 / (d_ij)^beta ) )
    
    支持 GPU 加速计算
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        beta (float): 距离衰减因子
        
    Returns:
        float: WCP 值
    """
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    num_switches = len(switch_nodes)
    
    if num_switches == 0:
        return 0.0
    
    # 检查是否使用 GPU 加速
    use_gpu = is_gpu_available() and TORCH_AVAILABLE and num_switches > 50
    
    if use_gpu:
        # GPU 加速版本
        nodes_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        switch_indices = [node_to_idx[n] for n in switch_nodes if n in node_to_idx]
        
        # 预计算从所有控制器到所有节点的距离
        num_centers = len(centers)
        num_nodes = len(nodes_list)
        
        # 距离矩阵: centers x nodes
        dist_matrix = np.full((num_centers, num_nodes), np.inf, dtype=np.float32)
        
        for i, center in enumerate(centers):
            try:
                lengths = nx.single_source_shortest_path_length(G, center)
                for target, dist in lengths.items():
                    if target in node_to_idx:
                        dist_matrix[i, node_to_idx[target]] = dist
            except Exception:
                continue
        
        # 转移到 GPU
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
        
        # 提取交换节点的距离
        switch_dists = dist_tensor[:, switch_indices]  # centers x switches
        
        # 计算 1/dist^beta，处理 inf 和 0
        valid_mask = (switch_dists > 0) & (switch_dists < float('inf'))
        inv_dists = torch.where(valid_mask, 1.0 / (switch_dists ** beta), torch.zeros_like(switch_dists))
        
        # 对每个交换节点求和所有控制器的贡献
        switch_potentials = torch.sum(inv_dists, dim=0)  # shape: (switches,)
        
        # 计算总 WCP
        total_wcp = torch.sum(switch_potentials).item()
        
    else:
        node_potentials = {n: 0.0 for n in switch_nodes}
        
        is_weighted = nx.is_weighted(G)
        
        for center in centers:
            try:
                if is_weighted:
                    lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
                else:
                    lengths = nx.single_source_shortest_path_length(G, center)
            except Exception:
                continue
                
            for target, dist in lengths.items():
                if target in node_potentials:
                    if dist > 0:
                        val = 1.0 / (dist ** beta)
                        node_potentials[target] += val
                    
        total_wcp = sum(node_potentials.values())
    return total_wcp / num_switches