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
import os

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

    # 诊断：是否记录 actor loss 分解日志（默认关闭以避免额外开销）
    log_actor_decomp: bool = False

    # 诊断：actor 梯度分解与 cosine（A2，默认关闭）
    log_actor_grad_decomp: bool = False
    grad_decomp_interval: int = 100

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
    dual_precond_eps: float = 0.05
    dual_precond_clip: float = 2.0
    dual_precond_strength: float = 0.3
    dual_precond_use_ema_stats: bool = True

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
    #   "aggregated": two-critic (V_reward, V_cost_total)，不再有独立 cost heads
    cost_critic_mode: str = "separate"

    # value_head_mode: reward+cost value head 结构
    #   "standard"（默认）：reward 独立 head，cost 根据 cost_critic_mode
    #   "shared_all": 单头同时输出 reward+所有 cost（忽略 cost_critic_mode）
    value_head_mode: str = "standard"

    # ========== Aggregated Cost 模式参数（仅 cost_critic_mode="aggregated" 时使用） ==========
    # 权重：total_cost = wE * (energy_cost / energy_budget) + wL * (load_cost / load_budget)
    agg_cost_w_energy: float = 1.0
    agg_cost_w_load: float = 1.0
    # 是否按预算归一化（True: 相对预算消耗; False: 直接加权和）
    agg_cost_normalize_by_budget: bool = True

    device: str = "cpu"


class MultiCriticPPO:
    """
    多 Critic + 拉格朗日约束 PPO Agent。

    核心逻辑：
    - 使用 MultiHeadActorCritic 网络（一个 reward value head + 多个 cost value head）
    - 对 reward 和每个 cost 分别计算 GAE
    - 策略优势：reward 与 cost surrogate 分离（hinge on positive cost adv）；CMDP PPO-Lagrangian split surrogate
    - λ 更新：λ_k ← max(0, λ_k + λ_lr * (avg_cost_k - budget_k))
    """

    def __init__(self, config: MultiCriticPPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # cost 名称由 cost_budgets 的 key 决定
        self.cost_names: List[str] = list(config.cost_budgets.keys())
        
        # aggregated 模式标记
        self.is_aggregated_mode = (config.cost_critic_mode == "aggregated")

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
        # aggregated 模式：仅使用 lambda_total（存储在 lambda_energy）
        # separate 模式：每个 cost 独立的 λ
        self.lambdas: Dict[str, float] = {}
        if self.is_aggregated_mode:
            # aggregated 模式：仅使用 "energy" key 存储 lambda_total
            if config.initial_lambdas is not None:
                self.lambdas["energy"] = config.initial_lambdas.get("energy", 0.0)
            else:
                self.lambdas["energy"] = 0.0
            # load lambda 设为 0（兼容性，不使用）
            self.lambdas["load"] = 0.0
        else:
            # separate 模式：每个 cost 独立初始化
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

        # [新增] Precond clip 统计：记录最近 N 次是否被 clip
        self.precond_clip_history = {
            "energy": [],
            "load": [],
        }
        self.precond_clip_window = 200  # 统计窗口大小
        # dual 预条件化 clip 统计（energy, load 命中标记的元组）
        self.dual_clip_hist: List[Tuple[float, float]] = []

    def _policy_parameters(self) -> List[torch.nn.Parameter]:
        """返回影响 policy logits 的参数集合（actor backbone + policy head）。

        如果未来改为共享 backbone，请确保这里至少包含用于产生动作 logits 的参数。"""
        return list(self.network.actor_backbone.parameters()) + list(self.network.policy_head.parameters())

    # ==================== Rollout Buffer ====================

    def _reset_buffer(self):
        """重置 rollout buffer。"""
        if self.is_aggregated_mode:
            # aggregated 模式：使用 v_cost_total 和 cost_total
            self.rollout_buffer = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "log_probs": [],
                "action_masks": [],
                "v_rewards": [],
                "v_cost_total": [],
                "cost_total": [],
                # 保留原始 costs 用于日志记录
                "costs": {name: [] for name in self.cost_names},
            }
        else:
            # separate 模式：每个 cost 独立
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
            v_costs: 各 cost 的 value 估计（aggregated 模式下包含 "total" key）
            costs_dict: 来自环境的 info['cost_components']
        """
        self.rollout_buffer["obs"].append(np.array(obs, copy=True))
        self.rollout_buffer["actions"].append(int(action))
        self.rollout_buffer["rewards"].append(float(reward))
        self.rollout_buffer["dones"].append(float(done))
        self.rollout_buffer["log_probs"].append(float(log_prob))
        self.rollout_buffer["action_masks"].append(np.array(action_mask, copy=True))
        self.rollout_buffer["v_rewards"].append(float(v_reward))

        # 保留原始 costs 用于日志
        for name in self.cost_names:
            self.rollout_buffer["costs"][name].append(
                float(costs_dict.get(name, 0.0))
            )

        if self.is_aggregated_mode:
            # aggregated 模式：计算并收集 total_cost
            energy_cost = float(costs_dict.get("energy", 0.0))
            load_cost = float(costs_dict.get("load", 0.0))
            
            if self.cfg.agg_cost_normalize_by_budget:
                # 按预算归一化
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                total_cost = (
                    self.cfg.agg_cost_w_energy * (energy_cost / max(energy_budget, 1e-8)) +
                    self.cfg.agg_cost_w_load * (load_cost / max(load_budget, 1e-8))
                )
            else:
                # 直接加权和
                total_cost = (
                    self.cfg.agg_cost_w_energy * energy_cost +
                    self.cfg.agg_cost_w_load * load_cost
                )
            
            self.rollout_buffer["cost_total"].append(float(total_cost))
            self.rollout_buffer["v_cost_total"].append(float(v_costs.get("total", 0.0)))
        else:
            # separate 模式：分别收集各 cost 的 value
            for name in self.cost_names:
                self.rollout_buffer["v_costs"][name].append(
                    float(v_costs.get(name, 0.0))
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

        # ========== 2. 根据模式计算 cost 的 GAE ==========
        if self.is_aggregated_mode:
            # aggregated 模式：计算 total_cost 的 GAE
            adv_cost_total, ret_cost_total = self._compute_gae(
                buffer["cost_total"],
                buffer["v_cost_total"],
                buffer["dones"],
            )
            # 兼容性：设置空的 adv_costs 和 ret_costs
            adv_costs: Dict[str, np.ndarray] = {}
            ret_costs: Dict[str, np.ndarray] = {}
        else:
            # separate 模式：计算每个 cost 的 GAE
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

        actor_params = self._policy_parameters()

        def _flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
            if not params:
                return torch.tensor([], dtype=torch.float32)
            return torch.cat([p.detach().cpu().reshape(-1) for p in params])

        with torch.no_grad():
            actor_param_before = _flatten_params(actor_params)
            actor_param_norm_l2 = (
                torch.linalg.norm(actor_param_before).item() if actor_param_before.numel() > 0 else 0.0
            )

        # ========== 3. 分离 reward / cost 优势，用于分裂 surrogate ==========
        adv_reward_norm = self._normalize_advantage(adv_reward)
        adv_reward_t = torch.as_tensor(adv_reward_norm, dtype=torch.float32, device=self.device)

        if self.is_aggregated_mode:
            adv_energy_raw = adv_cost_total
            adv_load_raw = np.zeros_like(adv_reward, dtype=np.float32)
        else:
            adv_energy_raw = adv_costs.get("energy", np.zeros_like(adv_reward, dtype=np.float32))
            adv_load_raw = adv_costs.get("load", np.zeros_like(adv_reward, dtype=np.float32))

        adv_energy_t = torch.as_tensor(adv_energy_raw, dtype=torch.float32, device=self.device)
        adv_load_t = torch.as_tensor(adv_load_raw, dtype=torch.float32, device=self.device)

        update_lambdas = bool(getattr(self.cfg, "update_lambdas", False))

        # 统一记录有效开关
        self.cfg.update_lambdas = update_lambdas
        has_cost_adv = (adv_energy_t.numel() > 0) or (adv_load_t.numel() > 0)

        # penalty 生效条件：自适应 λ 或已有 λ>0
        use_penalty_terms = update_lambdas or any(v > 0 for v in self.lambdas.values())
        use_cost_terms = use_penalty_terms and has_cost_adv
        use_lagrange = update_lambdas  # 兼容旧日志字段

        # Advantage diagnostics aligned with the split surrogate.
        adv_penalty_metrics: Dict[str, float] = {}
        lambdaA_metrics: Dict[str, float] = {}
        with torch.no_grad():
            lambda_energy = self.lambdas.get("energy", 0.0)
            lambda_load = self.lambdas.get("load", 0.0)

            adv_penalty = lambda_energy * adv_energy_t + lambda_load * adv_load_t

            adv_reward_abs_mean = adv_reward_t.abs().mean().item()
            adv_penalty_abs_mean = adv_penalty.abs().mean().item()
            adv_penalty_metrics = {
                "adv_reward_abs_mean": adv_reward_abs_mean,
                "adv_penalty_abs_mean": adv_penalty_abs_mean,
                "adv_penalty_to_reward_ratio": adv_penalty_abs_mean / (adv_reward_abs_mean + 1e-8),
                "adv_reward_mean": adv_reward_t.mean().item(),
                "adv_penalty_mean": adv_penalty.mean().item(),
            }

            lambdaA_metrics = {
                "lambdaA_energy_abs_mean": (lambda_energy * adv_energy_t).abs().mean().item(),
                "lambdaA_load_abs_mean": (lambda_load * adv_load_t).abs().mean().item(),
                "lambdaA_total_abs_mean": adv_penalty.abs().mean().item(),
            }

        # 梯度分解（A2）开关：默认关闭，仅当满足间隔时计算
        actor_params = self._policy_parameters()
        should_log_grad_decomp = (
            bool(self.cfg.log_actor_grad_decomp)
            and ((self._iter_count + 1) % max(1, int(self.cfg.grad_decomp_interval)) == 0)
            and len(actor_params) > 0
        )
        grad_decomp_metrics: Dict[str, float] = {}

        def _flatten_grad_list(grads: List[Optional[torch.Tensor]]) -> torch.Tensor:
            flat_parts: List[torch.Tensor] = []
            for grad, param in zip(grads, actor_params):
                if grad is None:
                    flat_parts.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat_parts.append(grad.reshape(-1))
            if not flat_parts:
                return torch.tensor(0.0, device=self.device)
            return torch.cat(flat_parts)

        actor_decomp_enabled = bool(self.cfg.log_actor_decomp)
        actor_decomp_tensors: Dict[str, torch.Tensor] = {
            "adv_r": adv_reward_t,
            "adv_energy": adv_energy_t,
            "adv_load": adv_load_t,
        }
        actor_decomp_stats: Dict[str, float] = {}
        total_pg_r_like = 0.0
        total_pg_p_like = 0.0
        total_pg_c_e_like = 0.0
        total_pg_c_l_like = 0.0
        decomp_update_steps = 0

        if actor_decomp_enabled:
            lambda_energy = self.lambdas.get("energy", 0.0)
            lambda_load = self.lambdas.get("load", 0.0)
            with torch.no_grad():
                penalty_t = lambda_energy * actor_decomp_tensors["adv_energy"] + lambda_load * actor_decomp_tensors["adv_load"]
                actor_decomp_stats = {
                    "adv_r_mean": actor_decomp_tensors["adv_r"].mean().item(),
                    "adv_r_std": actor_decomp_tensors["adv_r"].std().item(),
                    "adv_energy_mean": actor_decomp_tensors["adv_energy"].mean().item(),
                    "adv_energy_std": actor_decomp_tensors["adv_energy"].std().item(),
                    "adv_load_mean": actor_decomp_tensors["adv_load"].mean().item(),
                    "adv_load_std": actor_decomp_tensors["adv_load"].std().item(),
                    "lambda_energy": lambda_energy,
                    "lambda_load": lambda_load,
                    "penalty_mean": penalty_t.mean().item(),
                    "penalty_abs_mean": penalty_t.abs().mean().item(),
                }

        # 转换为 tensor
        ret_reward_t = torch.as_tensor(ret_reward, dtype=torch.float32, device=self.device)
        
        if self.is_aggregated_mode:
            ret_cost_total_t = torch.as_tensor(ret_cost_total, dtype=torch.float32, device=self.device)
            ret_costs_t = {}
        else:
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
                mb_adv_r = adv_reward_t[mb_idx]
                mb_adv_energy = adv_energy_t[mb_idx]
                mb_adv_load = adv_load_t[mb_idx]
                mb_ret_reward = ret_reward_t[mb_idx]

                if not use_cost_terms:
                    mb_adv_energy = torch.zeros_like(mb_adv_energy)
                    mb_adv_load = torch.zeros_like(mb_adv_load)
                
                # aggregated 模式：使用 ret_cost_total_t；separate 模式：使用 ret_costs_t
                if self.is_aggregated_mode:
                    mb_ret_cost_total = ret_cost_total_t[mb_idx]
                    mb_ret_costs = {}
                else:
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
                ratio_clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )

                # 梯度分解与 cosine（A2）：仅在开启且满足间隔时计算一次
                if should_log_grad_decomp and not grad_decomp_metrics:
                    lambda_energy_cur = self.lambdas.get("energy", 0.0)
                    lambda_load_cur = self.lambdas.get("load", 0.0)

                    loss_r = -torch.min(ratio * mb_adv_r, ratio_clipped * mb_adv_r).mean()
                    penalty_mb = lambda_energy_cur * mb_adv_energy + lambda_load_cur * mb_adv_load
                    loss_c = torch.min(ratio * penalty_mb, ratio_clipped * penalty_mb).mean()
                    loss_total = loss_r + loss_c

                    grads_r = torch.autograd.grad(
                        loss_r, actor_params, retain_graph=True, allow_unused=True
                    )
                    grads_c = torch.autograd.grad(
                        loss_c, actor_params, retain_graph=True, allow_unused=True
                    )
                    grads_t = torch.autograd.grad(
                        loss_total, actor_params, retain_graph=True, allow_unused=True
                    )

                    g_r_flat = _flatten_grad_list(grads_r)
                    g_c_flat = _flatten_grad_list(grads_c)
                    g_t_flat = _flatten_grad_list(grads_t)

                    g_r_norm = torch.linalg.norm(g_r_flat).item()
                    g_c_norm = torch.linalg.norm(g_c_flat).item()
                    g_t_norm = torch.linalg.norm(g_t_flat).item()

                    cos_total_r = 0.0
                    cos_total_c = 0.0
                    if g_r_norm > 0 and g_t_norm > 0:
                        cos_total_r = float(torch.dot(g_t_flat, g_r_flat).item() / (g_t_norm * g_r_norm + 1e-12))
                    if g_c_norm > 0 and g_t_norm > 0:
                        cos_total_c = float(torch.dot(g_t_flat, g_c_flat).item() / (g_t_norm * g_c_norm + 1e-12))

                    grad_decomp_metrics = {
                        "g_r_norm": g_r_norm,
                        "g_c_norm": g_c_norm,
                        "g_t_norm": g_t_norm,
                        "g_c_over_r": g_c_norm / (g_r_norm + 1e-12),
                        "cos_total_r": cos_total_r,
                        "cos_total_c": cos_total_c,
                    }

                # ===== Split surrogate for CMDP PPO-Lagrangian =====
                obj_r = torch.min(ratio * mb_adv_r, ratio_clipped * mb_adv_r)

                obj_c_e_mean = torch.tensor(0.0, device=self.device)
                obj_c_l_mean = torch.tensor(0.0, device=self.device)
                if use_cost_terms:
                    if mb_adv_energy.numel() > 0:
                        obj_c_e = torch.min(ratio * mb_adv_energy, ratio_clipped * mb_adv_energy)
                        obj_c_e_mean = obj_c_e.mean()
                    if mb_adv_load.numel() > 0:
                        obj_c_l = torch.min(ratio * mb_adv_load, ratio_clipped * mb_adv_load)
                        obj_c_l_mean = obj_c_l.mean()

                lambda_energy_cur = self.lambdas.get("energy", 0.0)
                lambda_load_cur = self.lambdas.get("load", 0.0)
                pg_loss_r = -obj_r.mean()
                pg_loss_c_e = lambda_energy_cur * obj_c_e_mean
                pg_loss_c_l = lambda_load_cur * obj_c_l_mean
                actor_loss = pg_loss_r + pg_loss_c_e + pg_loss_c_l - self.cfg.ent_coef * entropy.mean()

                # Metrics accumulation (always on for clarity)
                total_pg_r_like += pg_loss_r.item()
                total_pg_p_like += (pg_loss_c_e + pg_loss_c_l).item()
                total_pg_c_e_like += obj_c_e_mean.item()
                total_pg_c_l_like += obj_c_l_mean.item()
                decomp_update_steps += 1

                # ===== Value Loss =====
                # reward value loss
                value_loss_reward = 0.5 * F.mse_loss(v_reward, mb_ret_reward)

                # cost value losses
                if self.is_aggregated_mode:
                    # aggregated 模式：仅计算 v_cost_total 的 loss
                    mb_ret_cost_total = ret_cost_total_t[mb_idx]
                    value_loss_cost_total = 0.5 * F.mse_loss(v_costs["total"], mb_ret_cost_total)
                    if self.cfg.cost_value_coef > 0.0:
                        value_loss = value_loss_reward + self.cfg.cost_value_coef * value_loss_cost_total
                    else:
                        value_loss = value_loss_reward
                else:
                    # separate 模式：计算每个 cost 的 value loss
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
                # actor_loss already包含 entropy term
                loss = actor_loss + self.cfg.value_coef * value_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += actor_loss.item()
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

        with torch.no_grad():
            actor_param_after = _flatten_params(actor_params)
            delta_vec = actor_param_after - actor_param_before
            actor_param_delta_l2 = (
                torch.linalg.norm(delta_vec).item() if delta_vec.numel() > 0 else 0.0
            )
        actor_param_delta_ratio = actor_param_delta_l2 / (actor_param_norm_l2 + 1e-8)

        # ========== 5. 计算 avg_cost、gap 并更新 λ ==========
        avg_costs: Dict[str, float] = {}
        gaps: Dict[str, float] = {}
        ratio_gaps: Dict[str, float] = {}
        std_costs: Dict[str, float] = {}
        risk_costs: Dict[str, float] = {}
        risk_gaps: Dict[str, float] = {}
        risk_ratio_gaps: Dict[str, float] = {}

        if self.is_aggregated_mode:
            # aggregated 模式：计算 total_cost 的统计信息
            total_cost_values = np.array(buffer["cost_total"], dtype=np.float32)
            avg_cost_total = float(total_cost_values.mean()) if len(total_cost_values) > 0 else 0.0
            std_cost_total = float(total_cost_values.std()) if len(total_cost_values) > 0 else 0.0
            
            # 预算归一化后的目标（加权 sum of normalized costs，权重和可不为 1）
            if self.cfg.agg_cost_normalize_by_budget:
                wsum = self.cfg.agg_cost_w_energy + self.cfg.agg_cost_w_load
                total_budget = wsum if wsum > 1e-8 else 1.0
            else:
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                total_budget = self.cfg.agg_cost_w_energy * energy_budget + self.cfg.agg_cost_w_load * load_budget
            
            gap_total = avg_cost_total - total_budget
            ratio_gap_total = (avg_cost_total / max(total_budget, 1e-8)) - 1.0
            
            risk_cost_total = avg_cost_total + self.cfg.risk_factor * std_cost_total
            risk_gap_total = risk_cost_total - total_budget
            risk_ratio_gap_total = (risk_cost_total / max(total_budget, 1e-8)) - 1.0
            
            # 仍然计算原始 cost 的统计（用于日志）
            for name in self.cost_names:
                cost_values = np.array(buffer["costs"][name], dtype=np.float32)
                avg_costs[name] = float(cost_values.mean()) if len(cost_values) > 0 else 0.0
                std_val = float(cost_values.std()) if len(cost_values) > 0 else 0.0
                std_costs[name] = std_val
                budget = self.cfg.cost_budgets.get(name, 0.0)
                gaps[name] = avg_costs[name] - budget
                denom = budget if budget > 0 else 1.0
                ratio_gaps[name] = (avg_costs[name] / (denom + 1e-8)) - 1.0
                risk_costs[name] = avg_costs[name] + self.cfg.risk_factor * std_val
                risk_gaps[name] = risk_costs[name] - budget
                risk_ratio_gaps[name] = (risk_costs[name] / (denom + 1e-8)) - 1.0
        else:
            # separate 模式：计算每个 cost 的统计
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
        # aggregated 模式：仅更新 lambda_total（存储在 lambda["energy"]）
        # separate 模式：更新每个 cost 的 lambda
        gap_used_for_update: Dict[str, float] = {name: 0.0 for name in self.cost_names}
        gap_ema_for_log: Dict[str, float] = {}
        
        if self.is_aggregated_mode:
            gap_ema_for_log["total"] = risk_ratio_gap_total if self.cfg.lambda_gap_mode == "ratio" else risk_gap_total
        else:
            gap_ema_for_log = {name: risk_ratio_gaps.get(name, 0.0) for name in self.cost_names}
        
        corr_gap_for_log = 0.0
        precond_diag: Dict[str, float] = {}
        decor_diag: Dict[str, float] = {}

        if update_lambdas:
            # D2: 检查是否到达更新频率
            should_update_lambda = (self._iter_count % self.cfg.lambda_update_freq == 0)
            if self.is_aggregated_mode:
                # ========== aggregated 模式的 dual update（单标量 gap_total） ==========
                gap_total_used = risk_ratio_gap_total if self.cfg.lambda_gap_mode == "ratio" else risk_gap_total

                # base lr（复用 energy 通道）
                if self.cfg.lambda_lrs is not None:
                    base_lr = self.cfg.lambda_lrs.get("energy", self.cfg.lambda_lr)
                else:
                    base_lr = self.cfg.lambda_lr

                dual_mode = self.cfg.dual_update_mode
                # 聚合模式下 decorrelated/precond 无法区分维度，退化到 hysteresis（或 standard）
                if dual_mode in ("decorrelated", "precond", "preconditioned"):
                    dual_mode = "hysteresis"

                # hysteresis / both（单变量）
                if dual_mode in ("hysteresis", "both"):
                    g_raw = float(gap_total_used)
                    beta_gap = self.cfg.dual_gap_ema_beta
                    if beta_gap > 0:
                        self.dual_gap_ema["energy"] = (1 - beta_gap) * self.dual_gap_ema["energy"] + beta_gap * g_raw
                        g_ema = self.dual_gap_ema["energy"]
                    else:
                        g_ema = g_raw
                    gap_ema_for_log["total"] = g_ema

                    g_used = g_raw if g_raw > 0 else g_ema
                    if abs(g_used) <= self.cfg.dual_deadband:
                        g_used = 0.0

                    if should_update_lambda and g_used != 0.0:
                        lr_up = self.cfg.lambda_lr_up if self.cfg.lambda_lr_up is not None else base_lr
                        step_lr = lr_up if g_used > 0 else lr_up * self.cfg.dual_lr_down_scale
                        new_lambda = self.lambdas["energy"] + step_lr * g_used
                        new_lambda = max(0.0, new_lambda)
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas["energy"] = new_lambda
                        gap_used_for_update["energy"] = g_used
                    else:
                        gap_used_for_update["energy"] = 0.0

                    gap_used_for_update["load"] = 0.0
                    self.lambdas["load"] = 0.0

                else:
                    # standard 聚合：保持原行为（projected ascent，可选 EMA + deadzone）
                    effective_gap = gap_total_used
                    if self.cfg.lambda_gap_ema_beta > 0:
                        beta = self.cfg.lambda_gap_ema_beta
                        self.ema_gaps["energy"] = (1 - beta) * self.ema_gaps.get("energy", 0.0) + beta * effective_gap
                        effective_gap = self.ema_gaps["energy"]
                    gap_ema_for_log["total"] = effective_gap

                    deadzone = self.cfg.lambda_deadzone
                    if deadzone > 0 and abs(effective_gap) < deadzone:
                        effective_gap = 0.0

                    if should_update_lambda:
                        new_lambda = max(0.0, self.lambdas["energy"] + base_lr * effective_gap)
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas["energy"] = new_lambda
                        gap_used_for_update["energy"] = effective_gap
                    else:
                        gap_used_for_update["energy"] = 0.0

                    gap_used_for_update["load"] = 0.0
                    self.lambdas["load"] = 0.0

            else:
                # ========== separate 模式的 dual update（保持原逻辑） ==========
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
                    varE = varL = covEL = 0.0
                    precond_diag: Dict[str, float] = {}
                    has_e_l = {"energy", "load"}.issubset(set(self.cost_names))
                    gE = dual_gbar.get("energy", 0.0)
                    gL = dual_gbar.get("load", 0.0)

                    if has_e_l and self.cfg.dual_update_mode in ("decorrelated", "both", "precond", "preconditioned"):
                        b = self.cfg.dual_corr_ema_beta
                        use_ema_stats = self.cfg.dual_precond_use_ema_stats or self.cfg.dual_update_mode in ("decorrelated", "both")

                        meanE = self.dual_corr_state["mean"]["energy"]
                        meanL = self.dual_corr_state["mean"]["load"]
                        if use_ema_stats:
                            meanE = (1 - b) * meanE + b * gE
                            meanL = (1 - b) * meanL + b * gL
                            varE = (1 - b) * self.dual_corr_state["var"]["energy"] + b * (gE - meanE) ** 2
                            varL = (1 - b) * self.dual_corr_state["var"]["load"] + b * (gL - meanL) ** 2
                            covEL = (1 - b) * self.dual_corr_state["cov"] + b * (gE - meanE) * (gL - meanL)

                            self.dual_corr_state["mean"]["energy"] = meanE
                            self.dual_corr_state["mean"]["load"] = meanL
                            self.dual_corr_state["var"]["energy"] = varE
                            self.dual_corr_state["var"]["load"] = varL
                            self.dual_corr_state["cov"] = covEL
                        else:
                            varE = gE * gE
                            varL = gL * gL
                            covEL = gE * gL

                        denom = varE * varL
                        if self.cfg.dual_update_mode in ("decorrelated", "both"):
                            if denom < 1e-8:
                                corr_val = 0.0
                            else:
                                corr_val = float(np.clip(covEL / np.sqrt(denom + 1e-8), -0.95, 0.95))
                    corr_gap_for_log = corr_val

                    dual_source = dual_gbar

                    # 预条件化（仅 energy-load 对）- 2x2 协方差预条件化（C^{-1} g），带稳健 fallback
                    if has_e_l and self.cfg.dual_update_mode in ("precond", "preconditioned"):
                        eps = float(self.cfg.dual_precond_eps)
                        clip_val = float(self.cfg.dual_precond_clip)
                        strength = float(self.cfg.dual_precond_strength)

                        gE = float(dual_gbar["energy"])
                        gL = float(dual_gbar["load"])
                        a = float(varE)   # Var(E)
                        b = float(varL)   # Var(L)
                        c = float(covEL)  # Cov(E,L)

                        fallback = 0
                        det = float("nan")
                        cond_approx = 1.0

                        # 默认：不变
                        g_pre_energy = gE
                        g_pre_load = gL

                        # 数值检查
                        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                            fallback = 1
                        else:
                            a_eps = a + eps
                            b_eps = b + eps
                            det = a_eps * b_eps - c * c

                            if (not np.isfinite(det)) or det <= 1e-12:
                                # 退化：使用对角预条件化
                                fallback = 1
                                inv_a = 1.0 / max(a_eps, 1e-12)
                                inv_b = 1.0 / max(b_eps, 1e-12)
                                g_pre_energy = inv_a * gE
                                g_pre_load = inv_b * gL
                            else:
                                # 2x2 逆： (1/det) * [[b, -c],[-c, a]]
                                inv00 = b_eps / det
                                inv01 = -c / det
                                inv11 = a_eps / det

                                g_pre_energy = inv00 * gE + inv01 * gL
                                g_pre_load = inv01 * gE + inv11 * gL

                                # condition number 近似（用于诊断）
                                tr = a_eps + b_eps
                                disc = (a_eps - b_eps) ** 2 + 4.0 * c * c
                                sqrt_disc = float(np.sqrt(max(disc, 0.0)))
                                eig_max = 0.5 * (tr + sqrt_disc)
                                eig_min = max(0.5 * (tr - sqrt_disc), 1e-12)
                                cond_approx = float(eig_max / eig_min)

                        # strength 混合：g_used = (1-s)*g + s*g_pre
                        g_used_energy_raw = (1.0 - strength) * gE + strength * g_pre_energy
                        g_used_load_raw = (1.0 - strength) * gL + strength * g_pre_load

                        # clip
                        g_used_energy = float(np.clip(g_used_energy_raw, -clip_val, clip_val))
                        g_used_load = float(np.clip(g_used_load_raw, -clip_val, clip_val))

                        # 记录诊断指标
                        clip_hit_energy = 1.0 if abs(g_used_energy) >= clip_val - 1e-12 else 0.0
                        clip_hit_load = 1.0 if abs(g_used_load) >= clip_val - 1e-12 else 0.0
                        self.dual_clip_hist.append((clip_hit_energy, clip_hit_load))
                        if len(self.dual_clip_hist) > 200:
                            self.dual_clip_hist.pop(0)
                        clip_rate_energy = float(np.mean([x[0] for x in self.dual_clip_hist])) if self.dual_clip_hist else 0.0
                        clip_rate_load = float(np.mean([x[1] for x in self.dual_clip_hist])) if self.dual_clip_hist else 0.0

                        # 预条件化后的 g 用于 hysteresis / 更新
                        dual_source = dual_gbar.copy()
                        dual_source["energy"] = g_used_energy
                        dual_source["load"] = g_used_load

                        # 诊断：缩放比例（相对于原 gbar）
                        shrink_E = (g_used_energy / (gE + 1e-8)) if abs(gE) > 1e-12 else 0.0
                        shrink_L = (g_used_load / (gL + 1e-8)) if abs(gL) > 1e-12 else 0.0

                        precond_diag = {
                            "dual_precond_var_energy": float(a),
                            "dual_precond_var_load": float(b),
                            "dual_precond_cov_el": float(c),
                            "dual_precond_det": float(det),
                            "dual_precond_cond_approx": float(cond_approx),
                            "dual_precond_gpre_energy": float(g_pre_energy),
                            "dual_precond_gpre_load": float(g_pre_load),
                            "dual_precond_clip_energy": float(clip_hit_energy),
                            "dual_precond_clip_load": float(clip_hit_load),
                            "dual_precond_clip_rate_energy": float(clip_rate_energy),
                            "dual_precond_clip_rate_load": float(clip_rate_load),
                            "dual_precond_shrink_energy": float(shrink_E),
                            "dual_precond_shrink_load": float(shrink_L),
                            "dual_precond_fallback": float(fallback),
                        }

                    # decorrelate（若需要）: 用回归系数 cov/var 做残差化（比 corr*gap 更稳健）
                    if self.cfg.dual_update_mode in ("decorrelated", "both") and has_e_l:
                        reg_eps = 1e-8
                        beta_e_on_l = (covEL / (varL + reg_eps)) if varL > 1e-12 else 0.0
                        beta_l_on_e = (covEL / (varE + reg_eps)) if varE > 1e-12 else 0.0
                        beta_clip = 5.0
                        beta_e_on_l = float(np.clip(beta_e_on_l, -beta_clip, beta_clip))
                        beta_l_on_e = float(np.clip(beta_l_on_e, -beta_clip, beta_clip))

                        gtilde_energy = dual_gbar["energy"] - beta_e_on_l * dual_gbar["load"]
                        gtilde_load = dual_gbar["load"] - beta_l_on_e * dual_gbar["energy"]

                        dual_source = dual_gbar.copy()
                        dual_source["energy"] = gtilde_energy
                        dual_source["load"] = gtilde_load

                        decor_diag = {
                            "dual_decor_beta_e_on_l": float(beta_e_on_l),
                            "dual_decor_beta_l_on_e": float(beta_l_on_e),
                        }

                    # hysteresis 更新（hysteresis 或 both）
                    if should_update_lambda:
                        for name in self.cost_names:
                            # 改法2：per-cost lr，优先 lambda_lr_energy/lambda_lr_load
                            if name == "energy":
                                base_lr = getattr(self.cfg, "lambda_lr_energy", None)
                            else:
                                base_lr = getattr(self.cfg, "lambda_lr_load", None)
                            
                            if base_lr is None:
                                if self.cfg.lambda_lrs is not None:
                                    base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                                else:
                                    base_lr = self.cfg.lambda_lr

                            # hysteresis 分支
                            if self.cfg.dual_update_mode in ("hysteresis", "both"):
                                # 改法1：正向用 raw gap，负向用 EMA
                                g_raw = float(gap_for_mode.get(name, 0.0))
                                g_ema = dual_source.get(name, dual_gbar[name])
                                
                                if g_raw > 0:
                                    # 超预算时：直接用 raw gap，不经过 deadband
                                    g_val = g_raw
                                else:
                                    # 可行/富余时：用 EMA + deadband 慢下降
                                    g_val = g_ema
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

                # ====== standard 模式：projected dual ascent（支持 D1-D8，对齐 args）======
                else:
                    # 1) 可选：gap EMA（D1）。注意：这里的 EMA 只用于对偶更新与日志，
                    # 不改变 cost 的统计口径（avg_cost 仍是 batch 平均）。
                    beta = float(getattr(self.cfg, "lambda_gap_ema_beta", 0.0) or 0.0)
                    use_gap_ema = beta > 0.0

                    # 先决定“用于更新/日志的 gap”（EMA 或 raw）
                    for name in self.cost_names:
                        g_raw = float(gap_for_mode.get(name, 0.0))
                        if use_gap_ema:
                            self.ema_gaps[name] = (1.0 - beta) * float(self.ema_gaps.get(name, 0.0)) + beta * g_raw
                            g_for_update = float(self.ema_gaps[name])
                        else:
                            g_for_update = g_raw
                        gap_ema_for_log[name] = float(g_for_update)

                    # 2) D2：两时间尺度（lambda_update_freq）
                    if not should_update_lambda:
                        for name in self.cost_names:
                            gap_used_for_update[name] = 0.0
                    else:
                        for name in self.cost_names:
                            g_signed = float(gap_ema_for_log.get(name, 0.0))

                            # D3 + D7 + D8：deadzone（支持 per-cost 与非对称 deadzone）
                            dz = float(getattr(self.cfg, "lambda_deadzone", 0.0) or 0.0)
                            dz_map = getattr(self.cfg, "lambda_deadzones", None)
                            if isinstance(dz_map, dict):
                                dz = float(dz_map.get(name, dz))

                            dz_up = getattr(self.cfg, "lambda_deadzone_up", None)
                            dz_down = getattr(self.cfg, "lambda_deadzone_down", None)
                            if g_signed > 0.0 and dz_up is not None:
                                dz = float(dz_up)
                            elif g_signed < 0.0 and dz_down is not None:
                                dz = float(dz_down)

                            if dz > 0.0 and abs(g_signed) < dz:
                                gap_used_for_update[name] = 0.0
                                continue

                            # per-cost base lr：优先 lambda_lr_energy / lambda_lr_load，再回退到 lambda_lrs / lambda_lr
                            if name == "energy":
                                base_lr = getattr(self.cfg, "lambda_lr_energy", None)
                            else:
                                base_lr = getattr(self.cfg, "lambda_lr_load", None)

                            if base_lr is None:
                                if self.cfg.lambda_lrs is not None:
                                    base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                                else:
                                    base_lr = self.cfg.lambda_lr
                            base_lr = float(base_lr)

                            # D5-D6：非对称学习率（up / down）
                            lr_up = float(self.cfg.lambda_lr_up) if self.cfg.lambda_lr_up is not None else base_lr
                            lr_down = float(self.cfg.lambda_lr_down) if self.cfg.lambda_lr_down is not None else base_lr
                            if g_signed > 0.0:
                                lr = lr_up
                            elif g_signed < 0.0:
                                lr = lr_down
                            else:
                                lr = base_lr

                            # projected ascent + D4：lambda_max
                            new_lambda = max(0.0, float(self.lambdas.get(name, 0.0)) + lr * g_signed)
                            if self.cfg.lambda_max is not None:
                                new_lambda = min(new_lambda, float(self.cfg.lambda_max))

                            self.lambdas[name] = float(new_lambda)
                            gap_used_for_update[name] = float(g_signed)

        # ========== 6. 构造返回的 metrics ==========
        denom = max(update_steps, 1)
        metrics: Dict[str, float] = {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

        metrics.update({
            "use_lagrange_effective": bool(update_lambdas),  # 兼容旧字段
            "update_lambdas_effective": bool(update_lambdas),
            "use_penalty_terms_effective": bool(use_penalty_terms),
            "use_cost_terms_effective": bool(use_cost_terms),
            "has_cost_adv_effective": bool(has_cost_adv),
        })

        if grad_decomp_metrics:
            metrics.update({k: float(v) for k, v in grad_decomp_metrics.items()})

        pg_loss_r_like = total_pg_r_like / max(1, decomp_update_steps)
        pg_loss_p_like = total_pg_p_like / max(1, decomp_update_steps)
        pg_loss_c_e_like = total_pg_c_e_like / max(1, decomp_update_steps)
        pg_loss_c_l_like = total_pg_c_l_like / max(1, decomp_update_steps)
        metrics.update({
            "pg_loss_total": metrics["policy_loss"],
            "pg_loss_r_like": pg_loss_r_like,
            "pg_loss_p_like": pg_loss_p_like,
            "pg_loss_c_e_like": pg_loss_c_e_like,
            "pg_loss_c_l_like": pg_loss_c_l_like,
        })

        if actor_decomp_enabled:
            actor_decomp_stats.update({
                "pg_loss_total": metrics["policy_loss"],
                "pg_loss_r_like": pg_loss_r_like,
                "pg_loss_p_like": pg_loss_p_like,
            })
            metrics.update(actor_decomp_stats)

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

        # aggregated 模式额外添加 total_cost 相关指标
        if self.is_aggregated_mode:
            metrics["cost_total_mean"] = avg_cost_total
            metrics["cost_total_std"] = std_cost_total
            if self.cfg.agg_cost_normalize_by_budget:
                wsum = self.cfg.agg_cost_w_energy + self.cfg.agg_cost_w_load
                metrics["cost_total_budget"] = wsum if wsum > 1e-8 else 1.0
            else:
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                metrics["cost_total_budget"] = self.cfg.agg_cost_w_energy * energy_budget + self.cfg.agg_cost_w_load * load_budget
            metrics["gap_total"] = gap_total
            metrics["gap_ratio_total"] = ratio_gap_total
            metrics["lambda_total"] = self.lambdas["energy"]  # lambda_total 存储在 lambda_energy
            metrics["kkt_total"] = self.lambdas["energy"] * gap_total
            if self.cfg.lambda_gap_ema_beta > 0:
                metrics["ema_gap_total"] = gap_ema_for_log.get("total", 0.0)

        # Dual 更新相关的诊断（改法3：gap 日志命名去歧义）
        if {"energy", "load"}.issubset(set(self.cost_names)) and not self.is_aggregated_mode:
            # 明确 absolute vs ratio
            metrics["gap_abs_energy"] = gaps.get("energy", 0.0)
            metrics["gap_abs_load"] = gaps.get("load", 0.0)
            metrics["gap_ratio_energy"] = ratio_gaps.get("energy", 0.0)
            metrics["gap_ratio_load"] = ratio_gaps.get("load", 0.0)
            
            # EMA（用于滞回或去相关）
            metrics["gap_abs_energy_ema"] = gap_ema_for_log.get("energy", ratio_gaps.get("energy", 0.0))
            metrics["gap_abs_load_ema"] = gap_ema_for_log.get("load", ratio_gaps.get("load", 0.0))
            
            # 实际用于更新 λ 的 gap
            metrics["gap_abs_energy_used"] = gap_used_for_update.get("energy", 0.0)
            metrics["gap_abs_load_used"] = gap_used_for_update.get("load", 0.0)
            
            # 相关性
            metrics["corr_gap_ema"] = corr_gap_for_log
            
            # 风险敏感（ratio）
            metrics["risk_gap_ratio_energy"] = risk_ratio_gaps.get("energy", 0.0)
            metrics["risk_gap_ratio_load"] = risk_ratio_gaps.get("load", 0.0)
            
            # 预条件化诊断
            if precond_diag:
                metrics.update(precond_diag)

            # 去相关/消耦诊断
            if decor_diag:
                metrics.update(decor_diag)

        metrics["lambda_gap_mode"] = self.cfg.lambda_gap_mode
        
        # [新增 4] 写入新指标
        # rho_*: Safety Gym 风格的全局累计成本率 (cumulative_cost / total_steps)
        metrics.update(rho_metrics)
        # approx_kl: PPO policy 更新的 KL 散度近似（判断 policy 是否还在动）
        metrics["approx_kl"] = np.mean(approx_kls) if approx_kls else 0.0
        # clip_frac: PPO ratio 被 clip 的比例（更新强度辅助）
        metrics["clip_frac"] = np.mean(clip_fracs) if clip_fracs else 0.0
        # actor 参数变化尺度（E2）
        metrics["actor_param_delta_l2"] = actor_param_delta_l2
        metrics["actor_param_norm_l2"] = actor_param_norm_l2
        metrics["actor_param_delta_ratio"] = actor_param_delta_ratio
        # adv_*: advantage 统计（诊断1：cost 是否真的在推动 policy）
        #   - adv_reward_abs_mean: |adv_reward| 的均值
        #   - adv_penalty_abs_mean: |penalty_adv_total| 的均值
        #   - adv_penalty_to_reward_ratio: penalty 在优势层的占比
        metrics.update(adv_penalty_metrics)
        # lambdaA_*: 分约束的拉格朗日项贡献（诊断 cost 作用强度）
        #   - lambdaA_energy_abs_mean: |lambda_energy * adv_energy| 的均值
        #   - lambdaA_load_abs_mean: |lambda_load * adv_load| 的均值
        #   - lambdaA_total_abs_mean: 总惩罚项的绝对值均值
        metrics.update(lambdaA_metrics)

        if actor_decomp_enabled:
            print(
                "[actor-decomp] iter=%d | lambda_load=%.4f | penalty_abs_mean=%.4f | pg_loss_r_like=%.4f | pg_loss_p_like=%.4f | pg_loss_total=%.4f"
                % (
                    self._iter_count,
                    actor_decomp_stats.get("lambda_load", 0.0),
                    actor_decomp_stats.get("penalty_abs_mean", 0.0),
                    actor_decomp_stats.get("pg_loss_r_like", 0.0),
                    actor_decomp_stats.get("pg_loss_p_like", 0.0),
                    metrics.get("policy_loss", 0.0),
                )
            )

        # [Sanity Check] 第一次 update 时打印关键诊断字段（验证日志完整性）
        if self._iter_count == 1:
            diag_keys = [
                "approx_kl", "clip_frac", "entropy",
                "adv_reward_abs_mean", "adv_penalty_abs_mean", "adv_penalty_to_reward_ratio",
                "lambdaA_energy_abs_mean", "lambdaA_load_abs_mean", "lambdaA_total_abs_mean",
            ]
            missing = [k for k in diag_keys if k not in metrics]
            if missing:
                print(f"[WARNING] Missing diagnostic keys in metrics: {missing}")
            else:
                print(f"[OK] All diagnostic keys present in metrics (iter={self._iter_count})")

        # 清空 buffer
        self._reset_buffer()

        return metrics

    def dual_update_only(self, costs_window: Dict[str, List[float]]) -> Dict[str, float]:
        """仅执行对偶(λ)更新，不触碰策略/价值网络参数。

        Args:
            costs_window: 过去窗口内每步的成本序列，包含 "energy" 和 "load"。

        Returns:
            metrics: 仅包含成本、gap、lambda 相关的诊断指标。
        """
        if costs_window is None or not all(k in costs_window for k in ("energy", "load")):
            raise ValueError("costs_window must contain 'energy' and 'load'")

        # 将输入转换为数组（允许额外 key，但只使用已知 cost_names）
        cost_arrays: Dict[str, np.ndarray] = {}
        for name in self.cost_names:
            vals = costs_window.get(name, [])
            cost_arrays[name] = np.array(list(vals), dtype=np.float32)

        avg_costs: Dict[str, float] = {}
        gaps: Dict[str, float] = {}
        ratio_gaps: Dict[str, float] = {}
        std_costs: Dict[str, float] = {}
        risk_costs: Dict[str, float] = {}
        risk_gaps: Dict[str, float] = {}
        risk_ratio_gaps: Dict[str, float] = {}

        # ========== 计算成本统计量（与 update() 第 5 步保持口径一致） ==========
        if self.is_aggregated_mode:
            energy_arr = cost_arrays.get("energy", np.array([], dtype=np.float32))
            load_arr = cost_arrays.get("load", np.array([], dtype=np.float32))
            if energy_arr.shape[0] != load_arr.shape[0]:
                raise ValueError("energy and load sequences must have the same length for aggregated mode")

            if self.cfg.agg_cost_normalize_by_budget:
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                total_cost_vals = (
                    self.cfg.agg_cost_w_energy * (energy_arr / np.maximum(energy_budget, 1e-8))
                    + self.cfg.agg_cost_w_load * (load_arr / np.maximum(load_budget, 1e-8))
                )
            else:
                total_cost_vals = (
                    self.cfg.agg_cost_w_energy * energy_arr
                    + self.cfg.agg_cost_w_load * load_arr
                )

            avg_cost_total = float(total_cost_vals.mean()) if total_cost_vals.size > 0 else 0.0
            std_cost_total = float(total_cost_vals.std()) if total_cost_vals.size > 0 else 0.0

            if self.cfg.agg_cost_normalize_by_budget:
                wsum = self.cfg.agg_cost_w_energy + self.cfg.agg_cost_w_load
                total_budget = wsum if wsum > 1e-8 else 1.0
            else:
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                total_budget = self.cfg.agg_cost_w_energy * energy_budget + self.cfg.agg_cost_w_load * load_budget

            gap_total = avg_cost_total - total_budget
            ratio_gap_total = (avg_cost_total / max(total_budget, 1e-8)) - 1.0

            risk_cost_total = avg_cost_total + self.cfg.risk_factor * std_cost_total
            risk_gap_total = risk_cost_total - total_budget
            risk_ratio_gap_total = (risk_cost_total / max(total_budget, 1e-8)) - 1.0

            for name in self.cost_names:
                arr = cost_arrays.get(name, np.array([], dtype=np.float32))
                avg_val = float(arr.mean()) if arr.size > 0 else 0.0
                std_val = float(arr.std()) if arr.size > 0 else 0.0
                avg_costs[name] = avg_val
                std_costs[name] = std_val
                budget = self.cfg.cost_budgets.get(name, 0.0)
                gaps[name] = avg_val - budget
                denom = budget if budget > 0 else 1.0
                ratio_gaps[name] = (avg_val / (denom + 1e-8)) - 1.0
                risk_cost = avg_val + self.cfg.risk_factor * std_val
                risk_costs[name] = risk_cost
                risk_gaps[name] = risk_cost - budget
                risk_ratio_gaps[name] = (risk_cost / (denom + 1e-8)) - 1.0
        else:
            for name in self.cost_names:
                arr = cost_arrays.get(name, np.array([], dtype=np.float32))
                avg_val = float(arr.mean()) if arr.size > 0 else 0.0
                std_val = float(arr.std()) if arr.size > 0 else 0.0
                avg_costs[name] = avg_val
                std_costs[name] = std_val
                budget = self.cfg.cost_budgets.get(name, 0.0)
                gaps[name] = avg_val - budget
                denom = budget if budget > 0 else 1.0
                ratio_gaps[name] = (avg_val / (denom + 1e-8)) - 1.0
                risk_cost = avg_val + self.cfg.risk_factor * std_val
                risk_costs[name] = risk_cost
                risk_gaps[name] = risk_cost - budget
                risk_ratio_gaps[name] = (risk_cost / (denom + 1e-8)) - 1.0

        # ========== Dual 更新（复刻 update() 的第 5 步逻辑） ==========
        update_lambdas = bool(getattr(self.cfg, "update_lambdas", False))
        self.cfg.update_lambdas = update_lambdas

        self._iter_count += 1

        gap_used_for_update: Dict[str, float] = {name: 0.0 for name in self.cost_names}
        gap_ema_for_log: Dict[str, float] = {}
        corr_gap_for_log = 0.0
        precond_diag: Dict[str, float] = {}
        decor_diag: Dict[str, float] = {}

        if self.is_aggregated_mode:
            if self.cfg.lambda_gap_mode == "ratio":
                gap_total_used = risk_ratio_gap_total
                gap_ema_for_log["total"] = risk_ratio_gap_total
            else:
                gap_total_used = risk_gap_total
                gap_ema_for_log["total"] = risk_gap_total

            if update_lambdas:
                should_update_lambda = (self._iter_count % self.cfg.lambda_update_freq == 0)
                if self.cfg.lambda_lrs is not None:
                    base_lr = self.cfg.lambda_lrs.get("energy", self.cfg.lambda_lr)
                else:
                    base_lr = self.cfg.lambda_lr

                dual_mode = self.cfg.dual_update_mode
                if dual_mode in ("decorrelated", "precond", "preconditioned"):
                    dual_mode = "hysteresis"

                if dual_mode in ("hysteresis", "both"):
                    g_raw = float(gap_total_used)
                    beta_gap = self.cfg.dual_gap_ema_beta
                    if beta_gap > 0:
                        self.dual_gap_ema["energy"] = (1 - beta_gap) * self.dual_gap_ema["energy"] + beta_gap * g_raw
                        g_ema = self.dual_gap_ema["energy"]
                    else:
                        g_ema = g_raw
                    gap_ema_for_log["total"] = g_ema

                    g_used = g_raw if g_raw > 0 else g_ema
                    if abs(g_used) <= self.cfg.dual_deadband:
                        g_used = 0.0

                    if should_update_lambda and g_used != 0.0:
                        lr_up = self.cfg.lambda_lr_up if self.cfg.lambda_lr_up is not None else base_lr
                        step_lr = lr_up if g_used > 0 else lr_up * self.cfg.dual_lr_down_scale
                        new_lambda = self.lambdas["energy"] + step_lr * g_used
                        new_lambda = max(0.0, new_lambda)
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas["energy"] = new_lambda
                        gap_used_for_update["energy"] = g_used
                    else:
                        gap_used_for_update["energy"] = 0.0

                    gap_used_for_update["load"] = 0.0
                    self.lambdas["load"] = 0.0
                else:
                    effective_gap = gap_total_used
                    if self.cfg.lambda_gap_ema_beta > 0:
                        beta = self.cfg.lambda_gap_ema_beta
                        self.ema_gaps["energy"] = (1 - beta) * self.ema_gaps.get("energy", 0.0) + beta * effective_gap
                        effective_gap = self.ema_gaps["energy"]
                    gap_ema_for_log["total"] = effective_gap

                    deadzone = self.cfg.lambda_deadzone
                    if deadzone > 0 and abs(effective_gap) < deadzone:
                        effective_gap = 0.0

                    if should_update_lambda:
                        new_lambda = max(0.0, self.lambdas["energy"] + base_lr * effective_gap)
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas["energy"] = new_lambda
                        gap_used_for_update["energy"] = effective_gap
                    else:
                        gap_used_for_update["energy"] = 0.0

                    gap_used_for_update["load"] = 0.0
                    self.lambdas["load"] = 0.0
        else:
            gap_for_mode = risk_ratio_gaps if self.cfg.lambda_gap_mode == "ratio" else risk_gaps

            if update_lambdas and self.cfg.dual_update_mode != "standard":
                beta_gap = self.cfg.dual_gap_ema_beta
                dual_gbar: Dict[str, float] = {}
                for name in self.cost_names:
                    if beta_gap > 0:
                        self.dual_gap_ema[name] = (1 - beta_gap) * self.dual_gap_ema[name] + beta_gap * gap_for_mode[name]
                        dual_gbar[name] = self.dual_gap_ema[name]
                    else:
                        dual_gbar[name] = gap_for_mode[name]
                    gap_ema_for_log[name] = dual_gbar[name]

                has_e_l = {"energy", "load"}.issubset(set(self.cost_names))
                corr_gap_for_log = 0.0
                varE = varL = covEL = 0.0
                gE = dual_gbar.get("energy", 0.0)
                gL = dual_gbar.get("load", 0.0)
                dual_source = dual_gbar

                if has_e_l and self.cfg.dual_update_mode in ("decorrelated", "both", "precond", "preconditioned"):
                    b = self.cfg.dual_corr_ema_beta
                    use_ema_stats = self.cfg.dual_precond_use_ema_stats or self.cfg.dual_update_mode in ("decorrelated", "both")

                    meanE = self.dual_corr_state["mean"]["energy"]
                    meanL = self.dual_corr_state["mean"]["load"]
                    if use_ema_stats:
                        meanE = (1 - b) * meanE + b * gE
                        meanL = (1 - b) * meanL + b * gL
                        varE = (1 - b) * self.dual_corr_state["var"]["energy"] + b * (gE - meanE) ** 2
                        varL = (1 - b) * self.dual_corr_state["var"]["load"] + b * (gL - meanL) ** 2
                        covEL = (1 - b) * self.dual_corr_state["cov"] + b * (gE - meanE) * (gL - meanL)

                        self.dual_corr_state["mean"]["energy"] = meanE
                        self.dual_corr_state["mean"]["load"] = meanL
                        self.dual_corr_state["var"]["energy"] = varE
                        self.dual_corr_state["var"]["load"] = varL
                        self.dual_corr_state["cov"] = covEL
                    else:
                        varE = gE * gE
                        varL = gL * gL
                        covEL = gE * gL

                    denom = varE * varL
                    if self.cfg.dual_update_mode in ("decorrelated", "both"):
                        if denom < 1e-8:
                            corr_gap_for_log = 0.0
                        else:
                            corr_gap_for_log = float(np.clip(covEL / np.sqrt(denom + 1e-8), -0.95, 0.95))

                if has_e_l and self.cfg.dual_update_mode in ("precond", "preconditioned"):
                    eps = float(self.cfg.dual_precond_eps)
                    clip_val = float(self.cfg.dual_precond_clip)
                    strength = float(self.cfg.dual_precond_strength)

                    gE = float(dual_gbar.get("energy", 0.0))
                    gL = float(dual_gbar.get("load", 0.0))
                    a = float(varE)
                    b_val = float(varL)
                    c = float(covEL)

                    fallback = 0
                    det = float("nan")
                    cond_approx = 1.0

                    g_pre_energy = gE
                    g_pre_load = gL

                    if not (np.isfinite(a) and np.isfinite(b_val) and np.isfinite(c)):
                        fallback = 1
                    else:
                        a_eps = a + eps
                        b_eps = b_val + eps
                        det = a_eps * b_eps - c * c

                        if (not np.isfinite(det)) or det <= 1e-12:
                            fallback = 1
                            inv_a = 1.0 / max(a_eps, 1e-12)
                            inv_b = 1.0 / max(b_eps, 1e-12)
                            g_pre_energy = inv_a * gE
                            g_pre_load = inv_b * gL
                        else:
                            inv00 = b_eps / det
                            inv01 = -c / det
                            inv11 = a_eps / det

                            g_pre_energy = inv00 * gE + inv01 * gL
                            g_pre_load = inv01 * gE + inv11 * gL

                            tr = a_eps + b_eps
                            disc = (a_eps - b_eps) ** 2 + 4.0 * c * c
                            sqrt_disc = float(np.sqrt(max(disc, 0.0)))
                            eig_max = 0.5 * (tr + sqrt_disc)
                            eig_min = max(0.5 * (tr - sqrt_disc), 1e-12)
                            cond_approx = float(eig_max / eig_min)

                    g_used_energy_raw = (1.0 - strength) * gE + strength * g_pre_energy
                    g_used_load_raw = (1.0 - strength) * gL + strength * g_pre_load

                    g_used_energy = float(np.clip(g_used_energy_raw, -clip_val, clip_val))
                    g_used_load = float(np.clip(g_used_load_raw, -clip_val, clip_val))

                    clip_hit_energy = 1.0 if abs(g_used_energy) >= clip_val - 1e-12 else 0.0
                    clip_hit_load = 1.0 if abs(g_used_load) >= clip_val - 1e-12 else 0.0
                    self.dual_clip_hist.append((clip_hit_energy, clip_hit_load))
                    if len(self.dual_clip_hist) > 200:
                        self.dual_clip_hist.pop(0)
                    clip_rate_energy = float(np.mean([x[0] for x in self.dual_clip_hist])) if self.dual_clip_hist else 0.0
                    clip_rate_load = float(np.mean([x[1] for x in self.dual_clip_hist])) if self.dual_clip_hist else 0.0

                    dual_source = dual_gbar.copy()
                    dual_source["energy"] = g_used_energy
                    dual_source["load"] = g_used_load

                    shrink_E = (g_used_energy / (gE + 1e-8)) if abs(gE) > 1e-12 else 0.0
                    shrink_L = (g_used_load / (gL + 1e-8)) if abs(gL) > 1e-12 else 0.0

                    precond_diag = {
                        "dual_precond_var_energy": float(a),
                        "dual_precond_var_load": float(b_val),
                        "dual_precond_cov_el": float(c),
                        "dual_precond_det": float(det),
                        "dual_precond_cond_approx": float(cond_approx),
                        "dual_precond_gpre_energy": float(g_pre_energy),
                        "dual_precond_gpre_load": float(g_pre_load),
                        "dual_precond_clip_energy": float(clip_hit_energy),
                        "dual_precond_clip_load": float(clip_hit_load),
                        "dual_precond_clip_rate_energy": float(clip_rate_energy),
                        "dual_precond_clip_rate_load": float(clip_rate_load),
                        "dual_precond_shrink_energy": float(shrink_E),
                        "dual_precond_shrink_load": float(shrink_L),
                        "dual_precond_fallback": float(fallback),
                    }

                if self.cfg.dual_update_mode in ("decorrelated", "both") and has_e_l:
                    reg_eps = 1e-8
                    beta_e_on_l = (covEL / (varL + reg_eps)) if varL > 1e-12 else 0.0
                    beta_l_on_e = (covEL / (varE + reg_eps)) if varE > 1e-12 else 0.0
                    beta_clip = 5.0
                    beta_e_on_l = float(np.clip(beta_e_on_l, -beta_clip, beta_clip))
                    beta_l_on_e = float(np.clip(beta_l_on_e, -beta_clip, beta_clip))

                    gtilde_energy = dual_gbar["energy"] - beta_e_on_l * dual_gbar["load"]
                    gtilde_load = dual_gbar["load"] - beta_l_on_e * dual_gbar["energy"]

                    dual_source = dual_gbar.copy()
                    dual_source["energy"] = gtilde_energy
                    dual_source["load"] = gtilde_load

                    decor_diag = {
                        "dual_decor_beta_e_on_l": float(beta_e_on_l),
                        "dual_decor_beta_l_on_e": float(beta_l_on_e),
                    }

                should_update_lambda = (self._iter_count % self.cfg.lambda_update_freq == 0)
                if should_update_lambda:
                    for name in self.cost_names:
                        if name == "energy":
                            base_lr = getattr(self.cfg, "lambda_lr_energy", None)
                        else:
                            base_lr = getattr(self.cfg, "lambda_lr_load", None)
                        if base_lr is None:
                            if self.cfg.lambda_lrs is not None:
                                base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                            else:
                                base_lr = self.cfg.lambda_lr

                        if self.cfg.dual_update_mode in ("hysteresis", "both"):
                            g_raw = float(gap_for_mode.get(name, 0.0))
                            g_ema = dual_source.get(name, dual_gbar[name])

                            if g_raw > 0:
                                g_val = g_raw
                            else:
                                g_val = g_ema
                                if abs(g_val) <= self.cfg.dual_deadband:
                                    gap_used_for_update[name] = 0.0
                                    continue

                            lr_up = self.cfg.lambda_lr_up if self.cfg.lambda_lr_up is not None else base_lr
                            step_lr = lr_up if g_val > 0 else lr_up * self.cfg.dual_lr_down_scale
                            new_lambda = max(0.0, self.lambdas[name] + step_lr * g_val)
                            gap_used_for_update[name] = g_val
                        else:
                            g_val = dual_source.get(name, dual_gbar[name])
                            lr = base_lr
                            new_lambda = max(0.0, self.lambdas[name] + lr * g_val)
                            gap_used_for_update[name] = g_val

                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, self.cfg.lambda_max)
                        self.lambdas[name] = new_lambda
                else:
                    gap_used_for_update.update({name: 0.0 for name in self.cost_names})

            elif update_lambdas:
                beta = float(getattr(self.cfg, "lambda_gap_ema_beta", 0.0) or 0.0)
                use_gap_ema = beta > 0.0

                for name in self.cost_names:
                    g_raw = float(gap_for_mode.get(name, 0.0))
                    if use_gap_ema:
                        self.ema_gaps[name] = (1.0 - beta) * float(self.ema_gaps.get(name, 0.0)) + beta * g_raw
                        g_for_update = float(self.ema_gaps[name])
                    else:
                        g_for_update = g_raw
                    gap_ema_for_log[name] = float(g_for_update)

                if self._iter_count % self.cfg.lambda_update_freq == 0:
                    for name in self.cost_names:
                        g_signed = float(gap_ema_for_log.get(name, 0.0))

                        dz = float(getattr(self.cfg, "lambda_deadzone", 0.0) or 0.0)
                        dz_map = getattr(self.cfg, "lambda_deadzones", None)
                        if isinstance(dz_map, dict):
                            dz = float(dz_map.get(name, dz))

                        dz_up = getattr(self.cfg, "lambda_deadzone_up", None)
                        dz_down = getattr(self.cfg, "lambda_deadzone_down", None)
                        if g_signed > 0.0 and dz_up is not None:
                            dz = float(dz_up)
                        elif g_signed < 0.0 and dz_down is not None:
                            dz = float(dz_down)

                        if dz > 0.0 and abs(g_signed) < dz:
                            gap_used_for_update[name] = 0.0
                            continue

                        if name == "energy":
                            base_lr = getattr(self.cfg, "lambda_lr_energy", None)
                        else:
                            base_lr = getattr(self.cfg, "lambda_lr_load", None)
                        if base_lr is None:
                            if self.cfg.lambda_lrs is not None:
                                base_lr = self.cfg.lambda_lrs.get(name, self.cfg.lambda_lr)
                            else:
                                base_lr = self.cfg.lambda_lr
                        base_lr = float(base_lr)

                        lr_up = float(self.cfg.lambda_lr_up) if self.cfg.lambda_lr_up is not None else base_lr
                        lr_down = float(self.cfg.lambda_lr_down) if self.cfg.lambda_lr_down is not None else base_lr
                        if g_signed > 0.0:
                            lr = lr_up
                        elif g_signed < 0.0:
                            lr = lr_down
                        else:
                            lr = base_lr

                        new_lambda = max(0.0, float(self.lambdas.get(name, 0.0)) + lr * g_signed)
                        if self.cfg.lambda_max is not None:
                            new_lambda = min(new_lambda, float(self.cfg.lambda_max))

                        self.lambdas[name] = float(new_lambda)
                        gap_used_for_update[name] = float(g_signed)
                else:
                    gap_used_for_update.update({name: 0.0 for name in self.cost_names})

        # ========== 构造返回的 metrics ==========
        metrics: Dict[str, float] = {}

        for name in self.cost_names:
            metrics[f"avg_cost_{name}"] = avg_costs.get(name, 0.0)
            metrics[f"std_cost_{name}"] = std_costs.get(name, 0.0)
            metrics[f"gap_{name}"] = gaps.get(name, 0.0)
            metrics[f"gap_ratio_{name}"] = ratio_gaps.get(name, 0.0)
            metrics[f"risk_cost_{name}"] = risk_costs.get(name, 0.0)
            metrics[f"risk_gap_{name}"] = risk_gaps.get(name, 0.0)
            metrics[f"risk_gap_ratio_{name}"] = risk_ratio_gaps.get(name, 0.0)
            metrics[f"lambda_{name}"] = self.lambdas.get(name, 0.0)
            metrics[f"kkt_{name}"] = self.lambdas.get(name, 0.0) * gaps.get(name, 0.0)
            metrics[f"risk_kkt_{name}"] = self.lambdas.get(name, 0.0) * risk_gaps.get(name, 0.0)
            if self.cfg.lambda_gap_ema_beta > 0:
                metrics[f"ema_gap_{name}"] = self.ema_gaps.get(name, 0.0)
            metrics[f"gap_used_{name}"] = gap_used_for_update.get(name, 0.0)

        if self.is_aggregated_mode:
            metrics["cost_total_mean"] = avg_cost_total
            metrics["cost_total_std"] = std_cost_total
            if self.cfg.agg_cost_normalize_by_budget:
                wsum = self.cfg.agg_cost_w_energy + self.cfg.agg_cost_w_load
                metrics["cost_total_budget"] = wsum if wsum > 1e-8 else 1.0
            else:
                energy_budget = self.cfg.cost_budgets.get("energy", 1.0)
                load_budget = self.cfg.cost_budgets.get("load", 1.0)
                metrics["cost_total_budget"] = self.cfg.agg_cost_w_energy * energy_budget + self.cfg.agg_cost_w_load * load_budget
            metrics["gap_total"] = gap_total
            metrics["gap_ratio_total"] = ratio_gap_total
            metrics["risk_gap_total"] = risk_gap_total
            metrics["risk_gap_ratio_total"] = risk_ratio_gap_total
            metrics["lambda_total"] = self.lambdas.get("energy", 0.0)
            metrics["kkt_total"] = self.lambdas.get("energy", 0.0) * gap_total
            metrics["risk_kkt_total"] = self.lambdas.get("energy", 0.0) * risk_gap_total
            if self.cfg.lambda_gap_ema_beta > 0:
                metrics["ema_gap_total"] = gap_ema_for_log.get("total", 0.0)

        if {"energy", "load"}.issubset(set(self.cost_names)) and not self.is_aggregated_mode:
            metrics["gap_abs_energy"] = gaps.get("energy", 0.0)
            metrics["gap_abs_load"] = gaps.get("load", 0.0)
            metrics["gap_ratio_energy"] = ratio_gaps.get("energy", 0.0)
            metrics["gap_ratio_load"] = ratio_gaps.get("load", 0.0)
            metrics["gap_abs_energy_ema"] = gap_ema_for_log.get("energy", ratio_gaps.get("energy", 0.0))
            metrics["gap_abs_load_ema"] = gap_ema_for_log.get("load", ratio_gaps.get("load", 0.0))
            metrics["gap_abs_energy_used"] = gap_used_for_update.get("energy", 0.0)
            metrics["gap_abs_load_used"] = gap_used_for_update.get("load", 0.0)
            metrics["corr_gap_ema"] = corr_gap_for_log
            metrics["risk_gap_ratio_energy"] = risk_ratio_gaps.get("energy", 0.0)
            metrics["risk_gap_ratio_load"] = risk_ratio_gaps.get("load", 0.0)

            if precond_diag:
                metrics.update(precond_diag)
            if decor_diag:
                metrics.update(decor_diag)

        metrics["lambda_gap_mode"] = self.cfg.lambda_gap_mode
        metrics["update_lambdas_effective"] = bool(update_lambdas)

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


# ========== Minimal smoke test for aggregated + hysteresis dual update ==========
def _smoke_test_aggregated_hysteresis():
    """Run a tiny update to ensure aggregated + hysteresis dual path does not crash."""
    cfg = MultiCriticPPOConfig(
        obs_dim=4,
        act_dim=2,
        hidden_dim=8,
        batch_size=4,
        minibatch_size=2,
        update_epochs=1,
        cost_critic_mode="aggregated",
        agg_cost_w_energy=1.0,
        agg_cost_w_load=1.0,
        agg_cost_normalize_by_budget=True,
        cost_budgets={"energy": 0.5, "load": 0.5},
        dual_update_mode="hysteresis",
        dual_gap_ema_beta=0.1,
        dual_deadband=0.0,
        lambda_gap_mode="absolute",
        update_lambdas=True,
    )
    agent = MultiCriticPPO(cfg)
    steps = cfg.batch_size
    agent.rollout_buffer = {
        "obs": [np.zeros(cfg.obs_dim, dtype=np.float32) for _ in range(steps)],
        "actions": [0 for _ in range(steps)],
        "log_probs": [0.0 for _ in range(steps)],
        "action_masks": [np.ones(cfg.act_dim, dtype=np.float32) for _ in range(steps)],
        "rewards": [1.0 for _ in range(steps)],
        "v_rewards": [0.0 for _ in range(steps)],
        "dones": [0.0 for _ in range(steps)],
        "cost_total": [0.6 for _ in range(steps)],
        "v_cost_total": [0.0 for _ in range(steps)],
        "costs": {
            "energy": [0.6 for _ in range(steps)],
            "load": [0.6 for _ in range(steps)],
        },
        "v_costs": {
            "energy": [0.0 for _ in range(steps)],
            "load": [0.0 for _ in range(steps)],
        },
    }
    agent.update()
    print("[SMOKE] aggregated+hysteresis update completed without error")


if __name__ == "__main__":
    if os.environ.get("RUN_PPO_MULTI_AGENT_SMOKE", "0") == "1":
        _smoke_test_aggregated_hysteresis()
    # 兼容性：允许直接运行进行简单测试
    print("This module is intended to be imported and used by training scripts.")


# ==================== 兼容性别名 ====================
# 保持与旧代码的兼容
MultiCriticPPOAgent = MultiCriticPPO
