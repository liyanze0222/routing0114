"""
ppo_multi_agent.py

多 Critic + 拉格朗日约束 PPO 实现。

========== 算法概述 ==========

目标是求解 CMDP（Constrained MDP）：
    max_π J_r(π)  s.t.  J_{c_k}(π) ≤ b_k

使用拉格朗日松弛：
    L(π, λ) = J_r(π) - Σ_k λ_k (J_{c_k}(π) - b_k)

λ 更新（在线梯度上升）：
    λ_k ← max(0, λ_k + λ_lr * (avg_cost_k - b_k))

========== 两种模式 ==========

1) 固定 λ 模式（惩罚法）：update_lambdas=False
   - λ 由 initial_lambdas 设定，训练过程中不改动
   - 仍然用 r_eff = r - Σ λ_k c_k 做 PPO 更新
   - 用于扫描固定 λ 做 trade-off 曲线

2) Lagrange 模式（自适应 λ）：update_lambdas=True
   - λ 初始值来自 initial_lambdas（一般为 0）
   - 随训练自动根据 (avg_cost_k - budget_k) 更新
   - 希望 avg_cost_k 收敛到 budget_k 附近
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import MultiHeadActorCritic


@dataclass
class MultiCriticPPOConfig:
    """MultiCritic PPO 配置类。"""

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
    batch_size: int = 2048       # 每次采样的总步数
    minibatch_size: int = 256    # 每个 minibatch 的大小
    update_epochs: int = 10      # 每个 batch 更新多少轮

    # ========== 约束相关 ==========
    # cost_budgets: 各 cost 的预算（每步平均成本上限）
    cost_budgets: Dict[str, float] = field(default_factory=lambda: {
        "energy": 1.2,
        "load": 0.08,
    })

    # lambda_lr: 拉格朗日乘子更新步长（默认，当 lambda_lrs 未指定时使用）
    lambda_lr: float = 0.01

    # lambda_lrs: 每个 cost 独立的 λ 更新步长
    #   若为 None，则所有 cost 使用 lambda_lr
    #   否则使用 lambda_lrs[name]，缺失的 key 回退到 lambda_lr
    lambda_lrs: Optional[Dict[str, float]] = None

    # update_lambdas: 是否在训练过程中更新 λ
    #   True  -> Lagrange 模式（自适应 λ）
    #   False -> 固定 λ 模式（惩罚法）
    update_lambdas: bool = True

    # initial_lambdas: λ 的初始值
    #   若为 None，则所有 λ_k 初始化为 0
    initial_lambdas: Optional[Dict[str, float]] = None

    # risk_factor: 风险敏感成本 = mean + risk_factor * std
    #   0.0 表示仅使用平均成本（保持旧行为）
    #   >0 时增加成本波动对 λ 更新的影响
    risk_factor: float = 0.0

    # cost_value_coef: cost value loss 在总 value loss 中的权重系数
    #   1.0 -> reward 和 cost 的 value loss 等权
    #   0.0 -> 完全忽略 cost value loss
    cost_value_coef: float = 1.0

    lambda_gap_mode: str = "absolute"

    # ========== 新增：dual(λ) 更新模式 ==========
    # dual_update_mode:
    #   - standard: 旧行为（保持向后兼容）
    #   - hysteresis: gap EMA + deadband + 滞回
    #   - decorrelated: gap 相关性消耦
    #   - both: decorrelated 后再 hysteresis
    dual_update_mode: str = "standard"
    dual_gap_ema_beta: float = 0.10
    dual_deadband: float = 0.02
    dual_lr_down_scale: float = 0.20
    dual_corr_ema_beta: float = 0.05

    # ========== 对偶更新稳定化开关（默认关闭，保持旧行为） ==========
    # D1: gap 的 EMA 平滑（降噪）
    #   lambda_gap_ema_beta=0.0 表示不启用 EMA（默认，保持旧行为）
    #   lambda_gap_ema_beta>0 时用 EMA 平滑后的 gap 更新 λ，推荐 0.05~0.1
    lambda_gap_ema_beta: float = 0.0

    # D2: λ 更新频率（two-timescale）
    #   lambda_update_freq=1 表示每个 iter 都更新 λ（默认，保持旧行为）
    #   lambda_update_freq=5 表示每 5 个 iter 更新一次 λ
    lambda_update_freq: int = 1

    # D3: dead-zone（死区）
    #   lambda_deadzone=0.0 表示不启用死区（默认，保持旧行为）
    #   若 |gap| < lambda_deadzone，视为 0，不更新 λ
    lambda_deadzone: float = 0.0

    # D4: λ 上限（可选）
    #   lambda_max=None 表示不限制（默认，保持旧行为）
    #   设置为正数时，λ 被 clamp 到 [0, lambda_max]
    lambda_max: Optional[float] = None

    # ========== D5-D6: 非对称学习率（Asymmetric LR） ==========
    # 当 gap > 0 时使用 lambda_lr_up，gap < 0 时使用 lambda_lr_down
    # 若为 None，则使用 lambda_lr 或 lambda_lrs 中的值
    # 这允许更激进地增加 λ（惩罚约束违反）同时更保守地减少 λ
    lambda_lr_up: Optional[float] = None    # gap > 0 时的学习率
    lambda_lr_down: Optional[float] = None  # gap < 0 时的学习率

    # ========== D7: per-cost deadzone ==========
    # 允许为每个 cost 设置不同的 deadzone
    # 若为 None，则所有 cost 使用 lambda_deadzone
    lambda_deadzones: Optional[Dict[str, float]] = None

    # ========== D8: 非对称 deadzone（Asymmetric Deadzone） ==========
    # 当 gap > 0 时（约束被违反）使用 lambda_deadzone_up
    # 当 gap < 0 时（约束已满足）使用 lambda_deadzone_down
    # 若为 None，则使用 lambda_deadzone 或 lambda_deadzones 中的值
    # 这允许对约束违反更敏感（小 deadzone），同时对约束满足更宽容（大 deadzone）
    lambda_deadzone_up: Optional[float] = None    # gap > 0 时的 deadzone
    lambda_deadzone_down: Optional[float] = None  # gap < 0 时的 deadzone

    # ========== E: Shared Cost Critic 模式 ==========
    # cost_critic_mode: 控制 cost value head 的结构
    #   "separate"（默认）: 每个 cost 有独立的 value head
    #   "shared": 所有 cost 共享一个 value head（输出维度 = len(cost_names)）
    cost_critic_mode: str = "separate"

    # value_head_mode: reward+cost value head 结构
    #   "standard"（默认）：reward 独立 head，cost 根据 cost_critic_mode
    #   "shared_all": 单头同时输出 reward+所有 cost（忽略 cost_critic_mode）
    value_head_mode: str = "standard"

    device: str = "cpu"


class MultiCriticPPO:
    """
    多 Critic + 拉格朗日约束 PPO Agent。

    核心逻辑：
    - 使用 MultiHeadActorCritic 网络（一个 reward value head + 多个 cost value head）
    - 对 reward 和每个 cost 分别计算 GAE
    - 策略优势使用拉格朗日加权：adv_eff = adv_reward - Σ λ_k * adv_cost_k
    - λ 更新：λ_k ← max(0, λ_k + λ_lr * (avg_cost_k - budget_k))
    """

    def __init__(self, config: MultiCriticPPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # cost 名称由 cost_budgets 的 key 决定
        self.cost_names: List[str] = list(config.cost_budgets.keys())

        # 初始化网络
        self.network = MultiHeadActorCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dim=config.hidden_dim,
            cost_names=self.cost_names,
            cost_critic_mode=config.cost_critic_mode,
            value_head_mode=config.value_head_mode,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        # 初始化拉格朗日乘子 λ
        # 若 initial_lambdas 为 None，则全部初始化为 0
        # 否则使用传入的值（缺失的 key 用 0 填充）
        self.lambdas: Dict[str, float] = {}
        for name in self.cost_names:
            if config.initial_lambdas is not None:
                self.lambdas[name] = config.initial_lambdas.get(name, 0.0)
            else:
                self.lambdas[name] = 0.0

        # EMA gap 状态（用于 D1 开关）
        self.ema_gaps: Dict[str, float] = {name: 0.0 for name in self.cost_names}

        # Dual 更新的 gap EMA（用于 hysteresis/decorrelated）
        self.dual_gap_ema: Dict[str, float] = {name: 0.0 for name in self.cost_names}

        # 相关性估计的 EMA 状态（仅 energy-load 对）
        self.dual_corr_state = {
            "mean": {name: 0.0 for name in self.cost_names},
            "var": {name: 0.0 for name in self.cost_names},
            "cov": 0.0,
        }

        # 迭代计数器（用于 D2 开关）
        self._iter_count: int = 0

        # 初始化 rollout buffer
        self._reset_buffer()
        
        # [新增] Safety Gym 对齐：全局累计计数器
        self.global_cumulative_costs = {name: 0.0 for name in self.cost_names}
        self.global_total_steps = 0

    # ==================== Rollout Buffer ====================

    def _reset_buffer(self):
        """重置 rollout buffer。"""
        self.rollout_buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "action_masks": [],
            "v_rewards": [],
            "v_costs": {name: [] for name in self.cost_names},
            "costs": {name: [] for name in self.cost_names},
        }

    def collect_rollout(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        action_mask: np.ndarray,
        v_reward: float,
        v_costs: Dict[str, float],
        costs_dict: Dict[str, float],
    ):
        """
        收集单步 rollout 数据。

        Args:
            obs: 观测
            action: 动作
            reward: 奖励
            done: 是否结束（terminated or truncated）
            log_prob: 动作的 log 概率
            action_mask: 动作掩码
            v_reward: reward value 估计
            v_costs: 各 cost 的 value 估计
            costs_dict: 来自环境的 info['cost_components']
        """
        self.rollout_buffer["obs"].append(np.array(obs, copy=True))
        self.rollout_buffer["actions"].append(int(action))
        self.rollout_buffer["rewards"].append(float(reward))
        self.rollout_buffer["dones"].append(float(done))
        self.rollout_buffer["log_probs"].append(float(log_prob))
        self.rollout_buffer["action_masks"].append(np.array(action_mask, copy=True))
        self.rollout_buffer["v_rewards"].append(float(v_reward))

        for name in self.cost_names:
            self.rollout_buffer["v_costs"][name].append(
                float(v_costs.get(name, 0.0))
            )
            self.rollout_buffer["costs"][name].append(
                float(costs_dict.get(name, 0.0))
            )

    # ==================== 动作选择 ====================

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float, Dict[str, float]]:
        """
        从当前策略采样单步动作。

        Args:
            obs: np.ndarray, shape [obs_dim]
            action_mask: np.ndarray, shape [act_dim], True/1=合法动作

        Returns:
            action: 采样的动作 (int)
            log_prob: 动作的 log 概率 (float)
            v_reward: reward value (float)
            v_costs: Dict[str, float] 各 cost 的 value
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)

        action, log_prob, entropy, v_reward, v_costs = self.network.get_action(
            obs_t, action_mask=mask_t
        )

        return action, log_prob, v_reward, v_costs

    # ==================== GAE 计算 ====================

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单个信号（reward 或 cost）的 GAE 和 returns。

        Args:
            rewards: 奖励/成本序列
            values: value 估计序列
            dones: done 标志序列

        Returns:
            advantages: GAE 优势
            returns: 目标 returns
        """
        rewards_arr = np.array(rewards, dtype=np.float32)
        values_arr = np.array(list(values) + [0.0], dtype=np.float32)  # V(s_T+1)=0
        dones_arr = np.array(dones, dtype=np.float32)

        T = len(rewards_arr)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones_arr[t]
            delta = (
                rewards_arr[t]
                + self.cfg.gamma * values_arr[t + 1] * nonterminal
                - values_arr[t]
            )
            advantages[t] = last_gae = (
                delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_gae
            )

        returns = advantages + values_arr[:-1]
        return advantages, returns

    @staticmethod
    def _normalize_advantage(adv: np.ndarray) -> np.ndarray:
        """标准化优势。"""
        adv = adv.astype(np.float32)
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    # ==================== PPO 更新 ====================

    def update(self) -> Dict[str, float]:
        """
        使用当前 rollout buffer 进行 PPO 更新。

        Returns:
            metrics: 训练指标字典
        """
        buffer = self.rollout_buffer
        
        # --- [新增 1] Safety Gym Cost Rate 更新 ---
        batch_steps = len(buffer["obs"])
        self.global_total_steps += batch_steps
        
        for name in self.cost_names:
            batch_cost_sum = np.sum(buffer["costs"][name])
            self.global_cumulative_costs[name] += batch_cost_sum
            
        rho_metrics = {}
        for name in self.cost_names:
            rho = self.global_cumulative_costs[name] / max(1, self.global_total_steps)
            rho_metrics[f"rho_{name}"] = rho

        # 转换为 tensor
        obs = torch.as_tensor(
            np.array(buffer["obs"], dtype=np.float32), device=self.device
        )
        actions = torch.as_tensor(
            np.array(buffer["actions"], dtype=np.int64), device=self.device
        )
        old_log_probs = torch.as_tensor(
            np.array(buffer["log_probs"], dtype=np.float32), device=self.device
        )
        action_masks = torch.as_tensor(
            np.array(buffer["action_masks"], dtype=np.float32), device=self.device
        )

        # ========== 1. 计算 reward 的 GAE ==========
        adv_reward, ret_reward = self._compute_gae(
            buffer["rewards"],
            buffer["v_rewards"],
            buffer["dones"],
        )

        # ========== 2. 计算每个 cost 的 GAE ==========
        adv_costs: Dict[str, np.ndarray] = {}
        ret_costs: Dict[str, np.ndarray] = {}
        for name in self.cost_names:
            adv, ret = self._compute_gae(
                buffer["costs"][name],
                buffer["v_costs"][name],
                buffer["dones"],
            )
            adv_costs[name] = adv
            ret_costs[name] = ret

        # ========== 3. 构造有效优势 adv_eff ==========
        # adv_eff = adv_reward - Σ λ_k * adv_cost_k
        # 注：融合后再进行标准化，避免破坏 GAE 的相对尺度
        adv_eff = adv_reward.copy()
        for name in self.cost_names:
            adv_eff -= self.lambdas[name] * adv_costs[name]
        adv_eff = self._normalize_advantage(adv_eff)

        # 转换为 tensor
        adv_eff_t = torch.as_tensor(adv_eff, dtype=torch.float32, device=self.device)
        ret_reward_t = torch.as_tensor(ret_reward, dtype=torch.float32, device=self.device)
        ret_costs_t = {
            name: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for name, arr in ret_costs.items()
        }

        # ========== 4. PPO 更新循环 ==========
        batch_size = obs.shape[0]
        indices = np.arange(batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_steps = 0
        
        # --- [新增 2] PPO 诊断列表初始化 ---
        clip_fracs = []
        approx_kls = []

        continue_training = True
        for epoch in range(self.cfg.update_epochs):
            if not continue_training:
                break
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_action_masks = action_masks[mb_idx]
                mb_adv_eff = adv_eff_t[mb_idx]
                mb_ret_reward = ret_reward_t[mb_idx]
                mb_ret_costs = {name: ret_costs_t[name][mb_idx] for name in self.cost_names}

                # 前向传播
                new_log_probs, entropy, v_reward, v_costs = self.network.evaluate_actions(
                    mb_obs, mb_actions, action_mask=mb_action_masks
                )
                
                # [新增 3] 在计算 loss 前记录诊断信息
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

                    ratio = torch.exp(log_ratio)
                    clipped = (ratio > 1.0 + self.cfg.clip_coef) | (ratio < 1.0 - self.cfg.clip_coef)
                    clip_fracs.append(torch.as_tensor(clipped, dtype=torch.float32).mean().item())

                # ===== Policy Loss (PPO-Clip) =====
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv_eff
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                ) * mb_adv_eff
                policy_loss = -torch.min(surr1, surr2).mean()

                # ===== Value Loss =====
                # reward value loss
                value_loss_reward = 0.5 * F.mse_loss(v_reward, mb_ret_reward)

                # cost value losses
                value_loss_costs = []
                for name in self.cost_names:
                    loss_cost = 0.5 * F.mse_loss(v_costs[name], mb_ret_costs[name])
                    value_loss_costs.append(loss_cost)

                # 总 value loss = reward_loss + cost_value_coef * Σ cost_losses
                if self.cfg.cost_value_coef > 0.0 and len(value_loss_costs) > 0:
                    value_loss = value_loss_reward + self.cfg.cost_value_coef * sum(value_loss_costs)
                else:
                    value_loss = value_loss_reward

                # ===== 总损失 =====
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.ent_coef * entropy.mean()
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_steps += 1
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                if approx_kl > self.cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL {approx_kl:.4f}")
                    continue_training = False
                    break

            if not continue_training:
                break

        # ========== 5. 计算 avg_cost、gap 并更新 λ ==========
        avg_costs: Dict[str, float] = {}
        gaps: Dict[str, float] = {}
        ratio_gaps: Dict[str, float] = {}
        std_costs: Dict[str, float] = {}
        risk_costs: Dict[str, float] = {}
        risk_gaps: Dict[str, float] = {}
        risk_ratio_gaps: Dict[str, float] = {}

        for name in self.cost_names:
            cost_values = np.array(buffer["costs"][name], dtype=np.float32)
            avg_costs[name] = float(cost_values.mean()) if len(cost_values) > 0 else 0.0
            std_val = float(cost_values.std()) if len(cost_values) > 0 else 0.0
            std_costs[name] = std_val
            budget = self.cfg.cost_budgets.get(name, 0.0)
            gaps[name] = avg_costs[name] - budget
            denom = budget if budget > 0 else 1.0
            ratio_gaps[name] = (avg_costs[name] / (denom + 1e-8)) - 1.0

            risk_cost = avg_costs[name] + self.cfg.risk_factor * std_val
            risk_costs[name] = risk_cost
            risk_gaps[name] = risk_cost - budget
            risk_ratio_gaps[name] = (risk_cost / (denom + 1e-8)) - 1.0

        # 更新迭代计数器
        self._iter_count += 1

        # λ 更新（仅当 update_lambdas=True 时）
        # 兼容旧逻辑：dual_update_mode=standard 完全复用原路径
        gap_used_for_update: Dict[str, float] = {name: 0.0 for name in self.cost_names}
        gap_ema_for_log: Dict[str, float] = {name: risk_ratio_gaps.get(name, 0.0) for name in self.cost_names}
        corr_gap_for_log = 0.0

        if self.cfg.update_lambdas:
            # D2: 检查是否到达更新频率
            should_update_lambda = (self._iter_count % self.cfg.lambda_update_freq == 0)

            # 预备：选择用于 dual 更新的原始 gap（ratio 或 absolute）
            gap_for_mode = risk_ratio_gaps if self.cfg.lambda_gap_mode == "ratio" else risk_gaps

            # ====== dual 模式：hysteresis / decorrelated / both ======
            if self.cfg.dual_update_mode != "standard":
                # 1) gap EMA（若 beta<=0 则退化为原 gap）
                beta_gap = self.cfg.dual_gap_ema_beta
                dual_gbar: Dict[str, float] = {}
                for name in self.cost_names:
                    if beta_gap > 0:
                        self.dual_gap_ema[name] = (1 - beta_gap) * self.dual_gap_ema[name] + beta_gap * gap_for_mode[name]
                        dual_gbar[name] = self.dual_gap_ema[name]
                    else:
                        dual_gbar[name] = gap_for_mode[name]
                    gap_ema_for_log[name] = dual_gbar[name]

                # 2) 相关性消耦（仅 energy-load 对）
                corr_val = 0.0
                if self.cfg.dual_update_mode in ("decorrelated", "both") and {"energy", "load"}.issubset(set(self.cost_names)):
                    b = self.cfg.dual_corr_ema_beta
                    gE = dual_gbar["energy"]
                    gL = dual_gbar["load"]

                    meanE = (1 - b) * self.dual_corr_state["mean"]["energy"] + b * gE
                    meanL = (1 - b) * self.dual_corr_state["mean"]["load"] + b * gL
                    varE = (1 - b) * self.dual_corr_state["var"]["energy"] + b * (gE - meanE) ** 2
                    varL = (1 - b) * self.dual_corr_state["var"]["load"] + b * (gL - meanL) ** 2
                    covEL = (1 - b) * self.dual_corr_state["cov"] + b * (gE - meanE) * (gL - meanL)

                    # 保存状态
                    self.dual_corr_state["mean"]["energy"] = meanE
                    self.dual_corr_state["mean"]["load"] = meanL
                    self.dual_corr_state["var"]["energy"] = varE
                    self.dual_corr_state["var"]["load"] = varL
                    self.dual_corr_state["cov"] = covEL

                    denom = varE * varL
                    if denom < 1e-8:
                        corr_val = 0.0
                    else:
                        corr_val = float(np.clip(covEL / np.sqrt(denom + 1e-8), -0.95, 0.95))
                corr_gap_for_log = corr_val

                # decorrelate（若需要）
                dual_source = dual_gbar
                if self.cfg.dual_update_mode in ("decorrelated", "both") and {"energy", "load"}.issubset(set(self.cost_names)):
                    gtilde_energy = dual_gbar["energy"] - corr_val * dual_gbar["load"]
                    gtilde_load = dual_gbar["load"] - corr_val * dual_gbar["energy"]
                    dual_source = dual_gbar.copy()
                    dual_source["energy"] = gtilde_energy
                    dual_source["load"] = gtilde_load

                # hysteresis 更新（hysteresis 或 both）
                if should_update_lambda:
                    for name in self.cost_names:
                        # 获取基础 lr（保持与旧逻辑一致的优先级）
                        if self.cfg.lambda_lrs is not None:
                            base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                        else:
                            base_lr = self.cfg.lambda_lr

                        # hysteresis 分支
                        if self.cfg.dual_update_mode in ("hysteresis", "both"):
                            g_val = dual_source.get(name, dual_gbar[name])
                            if abs(g_val) <= self.cfg.dual_deadband:
                                gap_used_for_update[name] = 0.0
                                continue

                            lr_up = self.cfg.lambda_lr_up if self.cfg.lambda_lr_up is not None else base_lr
                            step_lr = lr_up if g_val > 0 else lr_up * self.cfg.dual_lr_down_scale
                            new_lambda = max(0.0, self.lambdas[name] + step_lr * g_val)
                            gap_used_for_update[name] = g_val
                        else:
                            # decorrelated-only 分支
                            g_val = dual_source.get(name, dual_gbar[name])
                            lr = base_lr
                            new_lambda = max(0.0, self.lambdas[name] + lr * g_val)
                            gap_used_for_update[name] = g_val

                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas[name] = new_lambda
                else:
                    # 未到更新频率，日志仍保留 gbar 与 corr
                    gap_used_for_update.update({name: 0.0 for name in self.cost_names})

            # ====== standard 模式（原逻辑保持不变） ======
            else:
                if should_update_lambda:
                    for name in self.cost_names:
                        budget = self.cfg.cost_budgets.get(name, 0.0)
                        raw_gap = avg_costs[name] - budget
                        gap_for_update = (
                            ratio_gaps[name]
                            if self.cfg.lambda_gap_mode == "ratio"
                            else raw_gap
                        )
                        # D1: EMA 平滑（lambda_gap_ema_beta > 0 时启用）
                        if self.cfg.lambda_gap_ema_beta > 0:
                            beta = self.cfg.lambda_gap_ema_beta
                            self.ema_gaps[name] = (1 - beta) * self.ema_gaps[name] + beta * gap_for_update
                            effective_gap = self.ema_gaps[name]
                        else:
                            effective_gap = gap_for_update

                        gap_ema_for_log[name] = effective_gap

                        # D3/D7/D8: dead-zone（支持 per-cost、全局、非对称）
                        # 基础 deadzone：优先 per-cost，回退到全局
                        deadzone = self.cfg.lambda_deadzone
                        if self.cfg.lambda_deadzones is not None:
                            deadzone = self.cfg.lambda_deadzones.get(name, self.cfg.lambda_deadzone)

                        # D8: 非对称 deadzone（覆盖基础 deadzone）
                        # gap > 0 时使用 lambda_deadzone_up，gap < 0 时使用 lambda_deadzone_down
                        if effective_gap > 0 and self.cfg.lambda_deadzone_up is not None:
                            deadzone = self.cfg.lambda_deadzone_up
                        elif effective_gap < 0 and self.cfg.lambda_deadzone_down is not None:
                            deadzone = self.cfg.lambda_deadzone_down

                        if deadzone > 0 and abs(effective_gap) < deadzone:
                            effective_gap = 0.0

                        # 获取该 cost 的 lr：优先 lambda_lrs[name]，回退到 lambda_lr
                        if self.cfg.lambda_lrs is not None:
                            base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                        else:
                            base_lr = self.cfg.lambda_lr

                        # D5-D6: 非对称学习率（Asymmetric LR）
                        # gap > 0: 使用 lambda_lr_up（若设置）
                        # gap < 0: 使用 lambda_lr_down（若设置）
                        if effective_gap > 0 and self.cfg.lambda_lr_up is not None:
                            lr = self.cfg.lambda_lr_up
                        elif effective_gap < 0 and self.cfg.lambda_lr_down is not None:
                            lr = self.cfg.lambda_lr_down
                        else:
                            lr = base_lr

                        # λ_k ← max(0, λ_k + lr * effective_gap)
                        new_lambda = max(0.0, self.lambdas[name] + lr * effective_gap)

                        # D4: lambda_max clamp（若设置）
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)

                        self.lambdas[name] = new_lambda
                        gap_used_for_update[name] = effective_gap
                else:
                    # 未到更新频率：记录 EMA 或原 gap 用于日志
                    for name in self.cost_names:
                        gap_ema_for_log[name] = (
                            self.ema_gaps[name] if self.cfg.lambda_gap_ema_beta > 0 else gap_for_mode[name]
                        )
                        gap_used_for_update[name] = 0.0

        # ========== 6. 构造返回的 metrics ==========
        denom = max(update_steps, 1)
        metrics: Dict[str, float] = {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

        # 添加每个 cost 的 avg、λ、gap、KKT 残差、EMA gap
        # 注：kkt 使用更新后的 lambda 乘以原始 gap
        for name in self.cost_names:
            metrics[f"avg_cost_{name}"] = avg_costs[name]
            metrics[f"lambda_{name}"] = self.lambdas[name]
            metrics[f"gap_{name}"] = gaps[name]
            metrics[f"gap_ratio_{name}"] = ratio_gaps[name]
            metrics[f"std_cost_{name}"] = std_costs[name]
            metrics[f"risk_cost_{name}"] = risk_costs[name]
            metrics[f"risk_gap_{name}"] = risk_gaps[name]
            metrics[f"risk_gap_ratio_{name}"] = risk_ratio_gaps[name]
            # KKT 残差 = 更新后的 λ × 原始 gap（用于监控互补松弛条件）
            metrics[f"kkt_{name}"] = self.lambdas[name] * gaps[name]
            metrics[f"risk_kkt_{name}"] = self.lambdas[name] * risk_gaps[name]
            if self.cfg.lambda_gap_ema_beta > 0:
                metrics[f"ema_gap_{name}"] = self.ema_gaps[name]

        # Dual 更新相关的诊断（统一记录 ratio gap 及实际使用的 gap）
        if {"energy", "load"}.issubset(set(self.cost_names)):
            metrics["gap_energy_raw"] = ratio_gaps.get("energy", 0.0)
            metrics["gap_load_raw"] = ratio_gaps.get("load", 0.0)
            metrics["gap_energy_ema"] = gap_ema_for_log.get("energy", ratio_gaps.get("energy", 0.0))
            metrics["gap_load_ema"] = gap_ema_for_log.get("load", ratio_gaps.get("load", 0.0))
            metrics["gap_energy_used"] = gap_used_for_update.get("energy", 0.0)
            metrics["gap_load_used"] = gap_used_for_update.get("load", 0.0)
            metrics["corr_gap_ema"] = corr_gap_for_log
            metrics["risk_gap_energy_raw"] = risk_ratio_gaps.get("energy", 0.0)
            metrics["risk_gap_load_raw"] = risk_ratio_gaps.get("load", 0.0)

        metrics["lambda_gap_mode"] = self.cfg.lambda_gap_mode
        
        # [新增 4] 写入新指标
        metrics.update(rho_metrics)
        metrics["approx_kl"] = np.mean(approx_kls) if approx_kls else 0.0
        metrics["clip_frac"] = np.mean(clip_fracs) if clip_fracs else 0.0

        # 清空 buffer
        self._reset_buffer()

        return metrics

    # ==================== 模型保存/加载 ====================

    def save(self, path: str):
        """保存模型和 λ。"""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lambdas": self.lambdas,
            "ema_gaps": self.ema_gaps,
            "iter_count": self._iter_count,
        }, path)

    def load(self, path: str):
        """加载模型和 λ。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lambdas = checkpoint["lambdas"]
        # 向后兼容：旧 checkpoint 可能没有这些字段
        if "ema_gaps" in checkpoint:
            self.ema_gaps = checkpoint["ema_gaps"]
        if "iter_count" in checkpoint:
            self._iter_count = checkpoint["iter_count"]


# ==================== 兼容性别名 ====================
# 保持与旧代码的兼容
MultiCriticPPOAgent = MultiCriticPPO
