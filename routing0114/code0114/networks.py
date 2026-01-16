# networks.py
"""
神经网络模块：

1) ActorCritic: 单 Critic 的 Actor-Critic 网络（用于 baseline PPO）
2) MultiHeadActorCritic: 多 Critic 的 Actor-Critic 网络（用于 MultiCritic + Lagrangian PPO）
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Apply orthogonal initialization to a linear layer with configurable gain.
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    简单 MLP Actor-Critic（单 Critic）：
    - 输入: obs [batch, obs_dim]
    - 输出: policy logits [batch, act_dim], value [batch, 1]
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_action(self, obs: torch.Tensor):
        """
        obs: [obs_dim] 或 [1, obs_dim]
        返回: action(int), log_prob(tensor), value(tensor)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        用于 PPO update 时计算：
        - 新 policy 下的 log_prob
        - entropy
        - value
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)


class MultiHeadActorCritic(nn.Module):
    """
    多头 Actor-Critic 网络（用于 MultiCritic + Lagrangian PPO）：

    结构：
    - 共享 backbone: 若干线性层 + Tanh 激活
    - policy head: 输出动作 logits，支持 action_mask
    - reward value head: v_reward(features) -> [batch, 1]
    - cost value heads: 用 nn.ModuleDict，对每个 cost_name 生成一个线性头

    Args:
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐藏层维度
        cost_names: cost 名称列表，例如 ["energy", "load"]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        cost_names: List[str],
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.cost_names = list(cost_names)

        # 完全解耦的backbone
        self.actor_backbone = self._build_backbone(obs_dim, hidden_dim)
        self.reward_backbone = self._build_backbone(obs_dim, hidden_dim)
        self.cost_backbone = self._build_backbone(obs_dim, hidden_dim)

        # policy head
        self.policy_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # reward value head
        self.v_reward_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # cost value heads (用 ModuleDict 管理)
        self.v_cost_heads = nn.ModuleDict({
            name: layer_init(nn.Linear(hidden_dim, 1), std=1.0) for name in self.cost_names
        })

    @staticmethod
    def _build_backbone(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。

        Args:
            obs: [batch, obs_dim] 观测
            action_mask: [batch, act_dim] 或 [act_dim] 可选的动作掩码，
                         True/1 表示合法动作，False/0 表示非法动作

        Returns:
            logits: [batch, act_dim] 动作 logits（已应用 mask）
            v_reward: [batch] reward value
            v_costs: Dict[str, Tensor[batch]] 各 cost 的 value
        """
        actor_features = self.actor_backbone(obs)
        reward_features = self.reward_backbone(obs)
        cost_features = self.cost_backbone(obs)

        # policy logits
        logits = self.policy_head(actor_features)

        # 应用 action_mask：对非法动作填 -1e9
        if action_mask is not None:
            mask = action_mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [1, act_dim]
            # 扩展到 batch 维度
            if mask.shape[0] == 1 and logits.shape[0] > 1:
                mask = mask.expand(logits.shape[0], -1)
            logits = logits.masked_fill(~mask, -1e9)

        # reward value
        v_reward = self.v_reward_head(reward_features).squeeze(-1)  # [batch]

        # cost values
        v_costs = {
            name: head(cost_features).squeeze(-1)
            for name, head in self.v_cost_heads.items()
        }

        return logits, v_reward, v_costs

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float, Dict[str, float]]:
        """
        采样单步动作。

        Args:
            obs: [obs_dim] 或 [1, obs_dim] 观测
            action_mask: [act_dim] 可选的动作掩码

        Returns:
            action: 采样的动作 (int)
            log_prob: 动作的 log 概率 (float)
            entropy: 策略熵 (float)
            v_reward: reward value (float)
            v_costs: Dict[str, float] 各 cost 的 value
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        logits, v_reward, v_costs = self.forward(obs, action_mask)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (
            action.item(),
            log_prob.item(),
            entropy.item(),
            v_reward.item(),
            {name: v.item() for name, v in v_costs.items()},
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        批量评估给定 obs/actions 的 log_prob、熵与 value（用于 PPO 更新）。

        Args:
            obs: [batch, obs_dim] 观测
            actions: [batch] 动作
            action_mask: [batch, act_dim] 可选的动作掩码

        Returns:
            log_probs: [batch] 动作的 log 概率
            entropy: [batch] 策略熵
            v_reward: [batch] reward value
            v_costs: Dict[str, Tensor[batch]] 各 cost 的 value
        """
        logits, v_reward, v_costs = self.forward(obs, action_mask)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, v_reward, v_costs
