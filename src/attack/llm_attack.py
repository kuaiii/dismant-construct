# -*- coding: utf-8 -*-
"""
LLM Attack
基于 LLM 的智能攻击算法

策略：使用训练好的 LLM 模型来选择下一个要移除的节点。
模型基于 OCG (Operation Candidate Graph) 和网络结构特征进行决策。

这是本项目提出的方法，结合了图神经网络和语言模型的优势。
"""

from typing import Optional, Any
import torch
import networkx as nx
import yaml
from pathlib import Path

from .base import BaseAttack
from src.model.fusion_llm import ResilienceLLM, ModelConfig
from src.env.simulator import NetworkEnvironment, TaskType
from src.data.ocg_builder import OCGExtractor


class LLMAttack(BaseAttack):
    """
    基于 LLM 的智能攻击 (LLM-based Attack)
    
    攻击策略：
    使用训练好的 LLM 模型，基于 OCG 和网络结构特征，
    智能选择下一个要移除的节点。
    
    特点：
    - 结合图神经网络和语言模型
    - 考虑网络拓扑和语义信息
    - 比传统启发式方法更智能
    - 需要加载训练好的模型
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device: str = "cuda",
        name: str = "LLMAttack"
    ):
        """
        初始化 LLM 攻击算法
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
            device: 设备 (cuda/cpu)
            name: 算法名称
        """
        super().__init__(name=name)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        
        # 延迟加载模型（在第一次使用时加载）
        self._model = None
        self._ocg_extractor = None
        self._env = None
        self._config = None
    
    def _load_model(self):
        """延迟加载模型和相关组件"""
        if self._model is not None:
            return
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 创建模型
        model_config = ModelConfig(
            llm_model_name=self._config['model']['llm']['model_name'],
            use_lora=self._config['model']['lora']['enabled'],
            lora_r=self._config['model']['lora']['r'],
            lora_alpha=self._config['model']['lora']['alpha'],
            lora_dropout=self._config['model']['lora']['dropout'],
            use_geometric_encoder=self._config['model']['geometric_encoder']['enabled'],
            d_model=self._config['model']['fusion']['d_model']
        )
        
        self._model = ResilienceLLM(model_config)
        self._model.initialize(device=self.device)
        
        # 加载检查点
        checkpoint_path_obj = Path(self.checkpoint_path)
        
        # 智能查找检查点
        if not checkpoint_path_obj.exists():
            if checkpoint_path_obj.name == "best" or checkpoint_path_obj.name.endswith("best"):
                parent_dir = checkpoint_path_obj.parent
                if parent_dir.exists():
                    checkpoint_path_obj = parent_dir
        
        if checkpoint_path_obj.is_dir():
            # 查找所有 epoch 目录
            epoch_dirs = sorted(
                checkpoint_path_obj.glob("epoch_*"),
                key=lambda p: int(p.name.split("_")[1]) if p.name.startswith("epoch_") else 0,
                reverse=True
            )
            if epoch_dirs:
                latest_epoch_dir = epoch_dirs[0]
                checkpoint_file = latest_epoch_dir / "model.pt"
                if checkpoint_file.exists():
                    checkpoint = torch.load(checkpoint_file, map_location=self.device)
                else:
                    raise FileNotFoundError(f"在 {latest_epoch_dir} 中未找到 model.pt")
            else:
                checkpoint_files = list(checkpoint_path_obj.glob("*.pt")) + list(checkpoint_path_obj.glob("*.pth"))
                if checkpoint_files:
                    checkpoint_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    checkpoint = torch.load(checkpoint_file, map_location=self.device)
                else:
                    raise FileNotFoundError(f"在 {checkpoint_path_obj} 中未找到模型文件")
        elif checkpoint_path_obj.is_file():
            checkpoint = torch.load(checkpoint_path_obj, map_location=self.device)
        else:
            raise FileNotFoundError(f"检查点路径不存在: {self.checkpoint_path}")
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self._model.load_state_dict(checkpoint, strict=False)
        
        self._model.eval()
        
        # 创建 OCG 提取器
        self._ocg_extractor = OCGExtractor(
            language=self._config.get('ocg', {}).get('language', 'zh')
        )
    
    def select_node(
        self,
        graph: nx.Graph,
        current_step: int = 0,
        total_steps: int = 10,
        **kwargs
    ) -> Optional[Any]:
        """
        使用 LLM 模型选择下一个要移除的节点
        
        Args:
            graph: 当前网络图
            current_step: 当前步骤
            total_steps: 总步骤数
            **kwargs: 额外参数
        
        Returns:
            选中的节点 ID，如果没有可选节点则返回 None
        """
        # 延迟加载模型
        self._load_model()
        
        # 创建临时环境（用于获取候选节点）
        if self._env is None or self._env.graph is not graph:
            self._env = NetworkEnvironment(
                graph=graph,
                task_type=TaskType.DISMANTLE,
                budget=total_steps
            )
            self._env.current_step = current_step
        else:
            self._env.graph = graph
            self._env.current_step = current_step
        
        # 获取候选节点（使用谱梯度剪枝）
        candidate_nodes = self._env.prune_candidates(candidate_type="node")
        if not candidate_nodes:
            return None
        
        # 提取 OCG 并构建 prompt
        ocg_data = self._ocg_extractor.extract_ocg(
            graph=graph,
            candidate_nodes=candidate_nodes,
            task_type="dismantle",
            current_step=current_step + 1,
            total_steps=total_steps
        )
        
        # 构建输入文本
        input_text = ocg_data.user_prompt
        
        # Tokenize
        inputs = self._model.tokenizer(
            input_text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 获取候选操作位置索引
        num_candidates = len(candidate_nodes)
        seq_len = input_ids.shape[1]
        candidate_indices = torch.tensor(
            [[seq_len - num_candidates + j for j in range(num_candidates)]],
            device=self.device,
            dtype=torch.long
        )
        
        # 模型推理
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                candidate_indices=candidate_indices,
                return_scores=True
            )
            
            if "scores" in outputs and outputs["scores"] is not None:
                scores = outputs["scores"][0]  # [num_candidates]
                # 选择分数最高的候选
                best_idx = torch.argmax(scores).item()
                selected_node = candidate_nodes[best_idx]
                return selected_node
            else:
                # 如果没有 scores，回退到随机选择
                import random
                return random.choice(candidate_nodes)
    
    def attack(
        self,
        graph: nx.Graph,
        budget: int,
        dataset_name: str = "unknown",
        graph_name: str = "unknown",
        collapse_threshold: float = 0.2,
        **kwargs
    ) -> "AttackResult":
        """
        执行攻击并记录结果（重写以传递步骤信息）
        
        Args:
            graph: 初始网络图
            budget: 攻击预算
            dataset_name: 数据集名称
            graph_name: 图名称
            collapse_threshold: 崩溃阈值
            **kwargs: 传递给 select_node 的额外参数
        
        Returns:
            AttackResult: 攻击结果
        """
        from .base import AttackResult
        
        # 复制图以避免修改原图
        g = graph.copy()
        initial_nodes = g.number_of_nodes()
        initial_edges = g.number_of_edges()
        
        # 初始化记录
        attack_sequence = []
        removal_fractions = [0.0]
        lcc_values = [self._compute_lcc_ratio(g, initial_nodes)]
        
        # 执行攻击
        for step in range(budget):
            if g.number_of_nodes() == 0:
                break
            
            # 选择节点（传递步骤信息）
            node = self.select_node(g, current_step=step, total_steps=budget, **kwargs)
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
            r_res=0.0,
            collapse_threshold=collapse_threshold,
            initial_nodes=initial_nodes,
            initial_edges=initial_edges,
            budget=budget,
        )
        
        # 计算 R_res 和崩溃点
        result.r_res = result.compute_r_res()
        result.collapse_fraction = result.find_collapse_point(collapse_threshold)
        
        return result
