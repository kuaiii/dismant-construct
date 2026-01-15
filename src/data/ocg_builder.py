# -*- coding: utf-8 -*-
"""
OCGExtractor: 操作中心图 (Operation-Centric Graph) 提取器
负责提取候选节点周围的局部子图结构，并转化为 LLM 可理解的 Prompt 文本。

核心功能：
1. 提取以候选节点为中心的 k-hop 子图
2. 计算节点结构特征 (度数、聚类系数、中心性等)
3. 融合节点语义信息
4. 生成结构化的 Prompt 文本
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import networkx as nx
import numpy as np


@dataclass
class NodeFeature:
    """节点特征数据类"""
    node_id: Union[int, str]
    degree: int
    degree_level: str  # "high", "medium", "low"
    clustering_coeff: float
    betweenness_centrality: float
    neighbors: List[Union[int, str]]
    neighbor_descriptions: List[str]
    semantic_description: str = ""
    is_articulation_point: bool = False  # 是否是割点
    community_role: str = ""  # 社区角色描述


@dataclass
class OCGData:
    """操作中心图数据类"""
    task_type: str  # "dismantle" or "construct"
    current_step: int
    total_steps: int
    candidate_features: List[NodeFeature]
    operations: List[Dict]
    system_prompt: str
    user_prompt: str
    graph_summary: Dict = field(default_factory=dict)


class OCGExtractor:
    """
    操作中心图提取器
    
    将图结构和候选操作转化为 LLM 可理解的 Prompt 格式。
    
    OCG 核心思想：
    - 只关注候选节点及其 k-hop 邻域 (局部信息)
    - 结合结构特征 (拓扑) 和语义特征 (文本描述)
    - 生成符合对话微调格式的 Prompt
    
    Attributes:
        hop_distance: 提取子图的跳数
        max_neighbors_display: 显示的最大邻居数
        degree_thresholds: 度数等级阈值
    """
    
    def __init__(
        self,
        hop_distance: int = 1,
        max_neighbors_display: int = 5,
        degree_thresholds: Tuple[int, int] = (3, 8),
        language: str = "zh"
    ):
        """
        初始化 OCG 提取器
        
        Args:
            hop_distance: 提取子图的跳数 (默认 1-hop)
            max_neighbors_display: Prompt 中显示的最大邻居数
            degree_thresholds: (low_threshold, high_threshold)
                - degree < low_threshold: "低"
                - degree >= high_threshold: "高"
                - otherwise: "中"
            language: 输出语言 ("zh" 中文, "en" 英文)
        """
        self.hop_distance = hop_distance
        self.max_neighbors_display = max_neighbors_display
        self.degree_thresholds = degree_thresholds
        self.language = language
        
        # Prompt 模板
        self._init_prompt_templates()
    
    def _init_prompt_templates(self) -> None:
        """初始化 Prompt 模板"""
        if self.language == "zh":
            # 系统提示模板现在支持符号函数 σ
            self.system_template_dismantle = (
                "你是一个网络韧性优化专家。你的目标是通过分析局部子图结构（OCG）"
                "和节点语义，选择能最显著改变网络韧性积分 R_res 的操作。"
                "本次任务是拆解任务（σ=-1）：选择移除后能最大化降低网络韧性的节点。"
            )
            self.system_template_construct = (
                "你是一个网络韧性优化专家。你的目标是通过分析局部子图结构（OCG）"
                "和节点语义，选择能最显著改变网络韧性积分 R_res 的操作。"
                "本次任务是构造任务（σ=+1）：选择添加后能最大化提升网络韧性的边。"
            )
            # 保留旧模板用于兼容
            self.system_template = (
                "你是一个网络韧性优化专家。你的目标是通过分析局部子图结构（OCG）"
                "和节点语义，选择能最显著改变网络韧性积分 R_res 的操作。"
                "对于拆解任务（σ=-1），选择破坏性最大的节点；"
                "对于构造任务（σ=+1），选择增益最大的连接。"
            )
            self.user_template_header = (
                "【当前状态】\n"
                "步骤：{current_step} / {total_steps}\n"
                "目标：{objective}\n\n"
                "【局部拓扑与语义 (OCG)】\n"
                "以下是候选节点及其 {hop}-hop 邻居的语义摘要：\n\n"
            )
            self.node_template = (
                "{index}. 节点 [{node_id}]:\n"
                "   - 语义：{semantic}\n"
                "   - 结构：度数 {degree} ({degree_level})，连接了 {neighbors}。\n"
                "   - 关键性：{criticality}\n\n"
            )
            self.operations_header = "【候选操作列表】\n"
            self.operation_template = "- [{op_id}]: {op_description}\n"
            self.user_template_footer = "\n请分析上述选项，并按推荐优先级排序。"
        else:  # English
            self.system_template_dismantle = (
                "You are a network resilience optimization expert. "
                "Your goal is to analyze the local subgraph structure (OCG) "
                "and node semantics to select operations that most significantly "
                "affect the network resilience integral R_res. "
                "This is a DISMANTLE task (σ=-1): choose the node whose removal "
                "maximizes resilience reduction."
            )
            self.system_template_construct = (
                "You are a network resilience optimization expert. "
                "Your goal is to analyze the local subgraph structure (OCG) "
                "and node semantics to select operations that most significantly "
                "affect the network resilience integral R_res. "
                "This is a CONSTRUCT task (σ=+1): choose the edge whose addition "
                "maximizes resilience improvement."
            )
            self.system_template = (
                "You are a network resilience optimization expert. "
                "Your goal is to analyze the local subgraph structure (OCG) "
                "and node semantics to select operations that most significantly "
                "affect the network resilience integral R_res. "
                "For dismantling tasks (σ=-1), choose the most destructive node; "
                "for construction tasks (σ=+1), choose the most beneficial connection."
            )
            self.user_template_header = (
                "[Current State]\n"
                "Step: {current_step} / {total_steps}\n"
                "Objective: {objective}\n\n"
                "[Local Topology & Semantics (OCG)]\n"
                "The following are semantic summaries of candidate nodes "
                "and their {hop}-hop neighbors:\n\n"
            )
            self.node_template = (
                "{index}. Node [{node_id}]:\n"
                "   - Semantics: {semantic}\n"
                "   - Structure: degree {degree} ({degree_level}), "
                "connected to {neighbors}.\n"
                "   - Criticality: {criticality}\n\n"
            )
            self.operations_header = "[Candidate Operations]\n"
            self.operation_template = "- [{op_id}]: {op_description}\n"
            self.user_template_footer = "\nPlease analyze the options above and rank by recommendation priority."
    
    def extract_node_features(
        self, 
        graph: nx.Graph, 
        node_id: Union[int, str],
        node_semantics: Optional[Dict[Union[int, str], str]] = None
    ) -> NodeFeature:
        """
        提取单个节点的完整特征
        
        Args:
            graph: NetworkX 图
            node_id: 目标节点 ID
            node_semantics: 节点语义描述字典
        
        Returns:
            NodeFeature: 节点特征数据
        """
        if node_id not in graph:
            raise ValueError(f"Node {node_id} not in graph")
        
        # 基础结构特征
        degree = graph.degree(node_id)
        clustering = nx.clustering(graph, node_id)
        
        # 度数等级
        low_thresh, high_thresh = self.degree_thresholds
        if degree < low_thresh:
            degree_level = "低" if self.language == "zh" else "low"
        elif degree >= high_thresh:
            degree_level = "高" if self.language == "zh" else "high"
        else:
            degree_level = "中" if self.language == "zh" else "medium"
        
        # 邻居信息
        neighbors = list(graph.neighbors(node_id))[:self.max_neighbors_display]
        neighbor_descriptions = []
        if node_semantics:
            for n in neighbors:
                desc = node_semantics.get(n, f"节点{n}" if self.language == "zh" else f"node{n}")
                neighbor_descriptions.append(desc)
        
        # 中介中心性 (可能较慢，大图中考虑采样)
        try:
            betweenness = nx.betweenness_centrality(graph).get(node_id, 0.0)
        except Exception:
            betweenness = 0.0
        
        # 检查是否是割点 (关节点)
        articulation_points = set(nx.articulation_points(graph)) \
            if nx.is_connected(graph) else set()
        is_articulation = node_id in articulation_points
        
        # 语义描述
        semantic = ""
        if node_semantics:
            semantic = node_semantics.get(node_id, "")
        
        return NodeFeature(
            node_id=node_id,
            degree=degree,
            degree_level=degree_level,
            clustering_coeff=clustering,
            betweenness_centrality=betweenness,
            neighbors=neighbors,
            neighbor_descriptions=neighbor_descriptions,
            semantic_description=semantic,
            is_articulation_point=is_articulation,
            community_role=""  # TODO: 社区检测
        )
    
    def extract_ocg(
        self,
        graph: nx.Graph,
        candidate_nodes: List[Union[int, str]],
        task_type: str,
        current_step: int,
        total_steps: int,
        node_semantics: Optional[Dict[Union[int, str], str]] = None
    ) -> OCGData:
        """
        提取操作中心图数据
        
        核心方法：将图状态和候选节点转化为 OCG 数据结构。
        
        Args:
            graph: 当前网络图
            candidate_nodes: 候选节点列表 (经谱梯度剪枝后)
            task_type: 任务类型 ("dismantle" / "construct")
            current_step: 当前步骤
            total_steps: 总步骤数
            node_semantics: 节点语义描述字典
        
        Returns:
            OCGData: 操作中心图数据
        """
        # 提取候选节点特征
        candidate_features = []
        for node in candidate_nodes:
            if node in graph:
                feature = self.extract_node_features(graph, node, node_semantics)
                candidate_features.append(feature)
        
        # 生成操作列表
        operations = []
        for idx, node in enumerate(candidate_nodes):
            op_id = f"op_{idx+1:02d}"
            if task_type == "dismantle":
                op_desc = f"移除 {node}" if self.language == "zh" else f"Remove {node}"
            else:
                op_desc = f"添加连接到 {node}" if self.language == "zh" else f"Add edge to {node}"
            operations.append({
                "op_id": op_id,
                "op_type": "remove_node" if task_type == "dismantle" else "add_edge",
                "target": node,
                "description": op_desc
            })
        
        # 生成 Prompts（根据任务类型选择系统提示）
        if task_type == "dismantle":
            system_prompt = self.system_template_dismantle
        elif task_type == "construct":
            system_prompt = self.system_template_construct
        else:
            system_prompt = self.system_template
        
        user_prompt = self._build_user_prompt(
            candidate_features, operations, task_type, 
            current_step, total_steps
        )
        
        # 图摘要
        graph_summary = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "num_candidates": len(candidate_nodes)
        }
        
        return OCGData(
            task_type=task_type,
            current_step=current_step,
            total_steps=total_steps,
            candidate_features=candidate_features,
            operations=operations,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            graph_summary=graph_summary
        )
    
    def _build_user_prompt(
        self,
        candidate_features: List[NodeFeature],
        operations: List[Dict],
        task_type: str,
        current_step: int,
        total_steps: int
    ) -> str:
        """构建用户 Prompt 文本"""
        # 目标描述（包含符号函数）
        if self.language == "zh":
            objective = "最小化韧性 (Dismantle, σ=-1)" if task_type == "dismantle" \
                else "最大化韧性 (Construct, σ=+1)"
        else:
            objective = "Minimize resilience (Dismantle, σ=-1)" if task_type == "dismantle" \
                else "Maximize resilience (Construct, σ=+1)"
        
        # Header
        prompt = self.user_template_header.format(
            current_step=current_step,
            total_steps=total_steps,
            objective=objective,
            hop=self.hop_distance
        )
        
        # 节点描述
        for idx, feat in enumerate(candidate_features, 1):
            # 邻居描述
            if feat.neighbor_descriptions:
                neighbors_str = ", ".join(
                    f"{n} ({d})" for n, d in 
                    zip(feat.neighbors[:3], feat.neighbor_descriptions[:3])
                )
            else:
                neighbors_str = ", ".join(str(n) for n in feat.neighbors[:3])
            
            # 关键性描述
            criticality_parts = []
            if feat.is_articulation_point:
                if self.language == "zh":
                    criticality_parts.append("位于网络的割点位置")
                else:
                    criticality_parts.append("located at articulation point")
            
            if feat.clustering_coeff < 0.2:
                if self.language == "zh":
                    criticality_parts.append("邻居之间连接稀疏")
                else:
                    criticality_parts.append("neighbors sparsely connected")
            elif feat.clustering_coeff > 0.6:
                if self.language == "zh":
                    criticality_parts.append("邻居之间存在多条替代路径（聚类系数高）")
                else:
                    criticality_parts.append("multiple alternate paths exist (high clustering)")
            
            criticality = "；".join(criticality_parts) if criticality_parts else \
                ("普通节点" if self.language == "zh" else "regular node")
            
            prompt += self.node_template.format(
                index=idx,
                node_id=feat.node_id,
                semantic=feat.semantic_description or 
                    ("未知" if self.language == "zh" else "unknown"),
                degree=feat.degree,
                degree_level=feat.degree_level,
                neighbors=neighbors_str,
                criticality=criticality
            )
        
        # 操作列表
        prompt += self.operations_header
        for op in operations:
            prompt += self.operation_template.format(
                op_id=op["op_id"],
                op_description=op["description"]
            )
        
        # Footer
        prompt += self.user_template_footer
        
        return prompt
    
    def build_conversation_data(
        self,
        ocg_data: OCGData,
        ground_truth_ranking: List[str],
        auxiliary_labels: Dict[str, float],
        reasoning_trace: str = ""
    ) -> Dict[str, Any]:
        """
        构建符合微调格式的对话数据
        
        生成符合 LLaMA-Factory 等框架要求的 JSON 格式数据。
        
        Args:
            ocg_data: OCG 数据
            ground_truth_ranking: 正确的操作排序 ["op_01", "op_03", "op_02"]
            auxiliary_labels: 辅助标签 {"op_01": 0.95, "op_02": 0.15, ...}
            reasoning_trace: 推理过程文本
        
        Returns:
            Dict: 符合微调格式的对话数据
        """
        # 生成 Assistant 回复
        response = {
            "reasoning_trace": reasoning_trace,
            "ranked_list": ground_truth_ranking,
            "best_action": ground_truth_ranking[0] if ground_truth_ranking else ""
        }
        
        assistant_content = f"```json\n{json.dumps(response, ensure_ascii=False, indent=2)}\n```"
        
        return {
            "id": f"train_{ocg_data.task_type}_{ocg_data.current_step:03d}",
            "meta": {
                "task": ocg_data.task_type,
                "budget_step": f"{ocg_data.current_step}/{ocg_data.total_steps}"
            },
            "conversations": [
                {
                    "from": "system",
                    "value": ocg_data.system_prompt
                },
                {
                    "from": "user",
                    "value": ocg_data.user_prompt
                },
                {
                    "from": "assistant",
                    "value": assistant_content
                }
            ],
            "auxiliary_labels": auxiliary_labels
        }
    
    def generate_reasoning_trace(
        self,
        candidate_features: List[NodeFeature],
        ground_truth_ranking: List[str],
        auxiliary_labels: Dict[str, float]
    ) -> str:
        """
        生成推理过程文本 (用于 Chain-of-Thought)
        
        Args:
            candidate_features: 候选节点特征列表
            ground_truth_ranking: 正确排序
            auxiliary_labels: 辅助标签
        
        Returns:
            str: 推理过程文本
        """
        # TODO: 实现更智能的推理过程生成
        # 可以基于 auxiliary_labels 的分数差异生成解释
        raise NotImplementedError("generate_reasoning_trace")


# ==================== 便捷函数 ====================

def extract_ocg_from_env(
    env,  # NetworkEnvironment
    candidate_nodes: List,
    node_semantics: Optional[Dict] = None
) -> OCGData:
    """
    便捷函数：从 NetworkEnvironment 提取 OCG
    
    Args:
        env: NetworkEnvironment 实例
        candidate_nodes: 候选节点列表
        node_semantics: 节点语义字典
    
    Returns:
        OCGData
    """
    extractor = OCGExtractor()
    return extractor.extract_ocg(
        graph=env.graph,
        candidate_nodes=candidate_nodes,
        task_type=env.task_type.value,
        current_step=env.current_step,
        total_steps=env.budget,
        node_semantics=node_semantics or env.node_semantics
    )


def build_training_sample(
    env,
    candidate_nodes: List,
    ground_truth_ranking: List[str],
    auxiliary_labels: Dict[str, float],
    node_semantics: Optional[Dict] = None
) -> Dict:
    """
    便捷函数：构建单个训练样本
    
    Args:
        env: NetworkEnvironment 实例
        candidate_nodes: 候选节点
        ground_truth_ranking: 正确排序
        auxiliary_labels: 辅助标签
        node_semantics: 节点语义
    
    Returns:
        Dict: 训练样本
    """
    extractor = OCGExtractor()
    ocg_data = extractor.extract_ocg(
        graph=env.graph,
        candidate_nodes=candidate_nodes,
        task_type=env.task_type.value,
        current_step=env.current_step,
        total_steps=env.budget,
        node_semantics=node_semantics or env.node_semantics
    )
    
    return extractor.build_conversation_data(
        ocg_data=ocg_data,
        ground_truth_ranking=ground_truth_ranking,
        auxiliary_labels=auxiliary_labels
    )
