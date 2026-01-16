"""
ppo_scalar.py

Scalar PPO Agent for Ablation Study (V5 Baseline)
- Uses single value head (ActorCritic from networks.py)
- No Lagrangian multipliers
- Optimizes scalar reward: R - α*C_E - β*C_L
- Simple rollout buffer without cost tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import ActorCritic


@dataclass
class ScalarPPOConfig:
    """Scalar PPO 配置类"""
    
    # 环境维度
    obs_dim: int
    act_dim: int
    hidden_dim: int = 128
    
    # PPO 超参
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    
    # 优化器超参
    lr: float = 3e-4
    batch_size: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    
    device: str = "cpu"


class ScalarPPO:
    """
    标量 PPO Agent（用于消融实验）
    
    核心逻辑：
    - 使用单个 value head 评估标量化的 reward
    - 不使用 Lagrangian，不显式建模 cost
    - 优化目标：最大化 R - α*C_E - β*C_L（权重由外部传入）
    """
    
    def __init__(self, config: ScalarPPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # 初始化网络（单 value head）
        self.network = ActorCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Rollout buffer
        self.buffer_size = 0
        self.buffer_obs: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_rewards: List[float] = []  # 标量化后的 reward
        self.buffer_dones: List[bool] = []
        self.buffer_log_probs: List[float] = []
        self.buffer_values: List[float] = []
        self.buffer_action_masks: List[np.ndarray] = []
    
    # ==================== Rollout Buffer ====================
    
    def _reset_buffer(self):
        """重置 rollout buffer"""
        self.buffer_size = 0
        self.buffer_obs.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_dones.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()
        self.buffer_action_masks.clear()
    
    def collect_rollout(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,  # 已经是标量化的 reward
        done: bool,
        log_prob: float,
        action_mask: np.ndarray,
        value: float,
    ):
        """收集单步交互数据"""
        self.buffer_obs.append(obs.copy())
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)
        self.buffer_action_masks.append(action_mask.copy())
        self.buffer_size += 1
    
    # ==================== GAE Computation ====================
    
    def compute_gae(
        self,
        next_obs: np.ndarray,
        next_done: bool,
        action_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        Returns:
            advantages: [buffer_size] 优势函数
            returns: [buffer_size] 目标 return
        """
        # 获取 next_value
        with torch.no_grad():
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            # forward() 不需要 mask，只计算 logits 和 value
            _, next_value = self.network(obs_tensor)
            next_value = next_value.item()
            if next_done:
                next_value = 0.0
        
        # 转换为 numpy
        rewards = np.array(self.buffer_rewards, dtype=np.float32)
        values = np.array(self.buffer_values, dtype=np.float32)
        dones = np.array(self.buffer_dones, dtype=bool)
        
        # GAE 计算
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0.0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value_t = next_value
                next_not_done = 1.0 - float(next_done)
            else:
                next_value_t = values[t + 1]
                next_not_done = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.cfg.gamma * next_value_t * next_not_done - values[t]
            advantages[t] = last_gae_lam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_not_done * last_gae_lam
        
        returns = advantages + values
        
        # 确保返回的形状正确
        assert advantages.shape == (self.buffer_size,), f"advantages shape mismatch: {advantages.shape} vs ({self.buffer_size},)"
        assert returns.shape == (self.buffer_size,), f"returns shape mismatch: {returns.shape} vs ({self.buffer_size},)"
        
        return advantages, returns
    
    # ==================== Policy Update ====================
    
    def update(self, advantages: np.ndarray, returns: np.ndarray):
        """
        PPO 策略更新
        
        Args:
            advantages: [buffer_size] 优势函数
            returns: [buffer_size] 目标 return
            
        Returns:
            metrics: Dict 包含 policy_loss, value_loss, entropy 等
        """
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为 tensor
        obs_tensor = torch.tensor(np.array(self.buffer_obs), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(self.buffer_actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(np.array(self.buffer_action_masks), dtype=torch.float32, device=self.device)
        
        # 调试：确保所有 tensor 的第一维度一致
        assert obs_tensor.shape[0] == self.buffer_size, f"obs_tensor.shape[0]={obs_tensor.shape[0]} != buffer_size={self.buffer_size}"
        assert actions_tensor.shape[0] == self.buffer_size, f"actions_tensor mismatch"
        assert old_log_probs.shape[0] == self.buffer_size, f"old_log_probs mismatch"
        assert advantages_tensor.shape[0] == self.buffer_size, f"advantages_tensor mismatch"
        assert returns_tensor.shape[0] == self.buffer_size, f"returns_tensor.shape={returns_tensor.shape}, expected [{self.buffer_size}]"
        assert action_masks.shape[0] == self.buffer_size, f"action_masks mismatch"
        
        # Mini-batch SGD
        batch_size = self.buffer_size
        minibatch_size = min(self.cfg.minibatch_size, batch_size)
        indices = np.arange(batch_size)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_steps = 0
        
        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            continue_training = True
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_masks = action_masks[mb_indices]
                
                # 前向传播
                mb_new_log_probs, mb_entropy, mb_values = self.network.evaluate_actions(
                    mb_obs, mb_actions, mb_masks
                )
                
                # Policy loss (PPO clip)
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                clip_ratio = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                policy_loss = -torch.min(ratio * mb_advantages, clip_ratio * mb_advantages).mean()
                
                # Value loss
                value_loss = F.mse_loss(mb_values, mb_returns)
                
                # Entropy bonus
                entropy = mb_entropy.mean()
                
                # Total loss
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.ent_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_steps += 1
                
                # KL 散度早停
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - mb_new_log_probs).mean().item()
                if approx_kl > self.cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL {approx_kl:.4f}")
                    continue_training = False
                    break
            
            if not continue_training:
                break
        
        # 清空 buffer
        self._reset_buffer()
        
        # 返回指标
        metrics = {
            "policy_loss": total_policy_loss / max(1, update_steps),
            "value_loss": total_value_loss / max(1, update_steps),
            "entropy": total_entropy / max(1, update_steps),
            "approx_kl": approx_kl,
        }
        
        return metrics
    
    # ==================== Action Selection ====================
    
    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            action: int
            log_prob: float
            value: float
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        action, log_prob, entropy, value = self.network.get_action(obs_tensor, mask_tensor)
        
        # 转换 tensor 为 float
        return action, log_prob.item(), value.item()
    
    # ==================== Save/Load ====================
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
