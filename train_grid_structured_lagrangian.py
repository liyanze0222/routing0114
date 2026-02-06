"""
train_grid_structured_lagrangian.py

Grid 环境下的 MultiCritic + Lagrangian PPO 训练入口脚本。

========== 环境结构 ==========

GridRoutingEnv（标准 N x N 全连通网格）
    ↓
GridCostWrapper（提供 energy/load/invalid 三种 cost）
    ↓
GridHardWrapper（根据位置给出 action_mask，仅屏蔽越界动作）

========== 两种训练模式 ==========

1) 固定 λ 模式（惩罚法）：--use_lagrange False
   - λ 由 --initial_lambda_energy 和 --initial_lambda_load 设定
   - 训练过程中不更新 λ
   - 用于扫描固定 λ 做 trade-off 曲线

2) Lagrange 模式（自适应 λ）：--use_lagrange True
   - λ 初始为 0（或由 initial_lambda_* 指定）
   - 随训练自动根据 (avg_cost_k - budget_k) 更新
   - 希望 avg_cost_k 收敛到 budget_k 附近

========== 示例命令 ==========

# 固定 λ 模式
python train_grid_structured_lagrangian.py --use_lagrange False \\
    --initial_lambda_energy 0.5 --initial_lambda_load 1.0

# Lagrange 模式
python train_grid_structured_lagrangian.py --use_lagrange True \\
    --lambda_lr 0.01 --energy_budget 1.2 --load_budget 0.08
"""

import argparse
import copy
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from ppo_multi_agent import MultiCriticPPOConfig, MultiCriticPPO
from utils import set_seed, MetricsLogger, plot_training_curves, plot_safety_gym_curves, make_output_dir, save_grid_route_viz, augment_obs_with_context
# from check_cost_units import run_cost_unit_check  # Optional module


class CurriculumScheduler:
    """
    课程学习调度器：线性衰减 load_budget 从 start → end
    
    Args:
        start_budget: 起始预算（宽松约束）
        end_budget: 终止预算（目标约束）
        decay_iters: 衰减迭代次数
    """
    def __init__(self, start_budget: float, end_budget: float, decay_iters: int):
        self.start_budget = start_budget
        self.end_budget = end_budget
        self.decay_iters = decay_iters
    
    def get_budget(self, iteration: int) -> float:
        """获取当前迭代的预算值（线性衰减）"""
        if iteration >= self.decay_iters:
            return self.end_budget
        # 线性插值: start + (end - start) * (iter / decay_iters)
        ratio = iteration / self.decay_iters
        return self.start_budget + (self.end_budget - self.start_budget) * ratio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Critic PPO + Lagrangian on standard Grid"
    )

    # ========== 基础设置 ==========
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--total_iters", type=int, default=300,
        help="总训练迭代次数（每次迭代采样 batch_size 步）"
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="每次采样的步数")
    parser.add_argument("--minibatch_size", type=int, default=256, help="minibatch 大小")
    parser.add_argument("--update_epochs", type=int, default=10, help="每个 batch 更新轮数")

    # ========== 环境设置 ==========
    parser.add_argument("--grid_size", type=int, default=8, help="网格大小 N（N x N）")
    parser.add_argument(
        "--step_penalty", type=float, default=-1.0,
        help="每一步的基础惩罚（默认 -1.0）"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="episode 最大步数，<0 表示 4*N*N"
    )
    parser.add_argument(
        "--success_reward", type=float, default=10.0,
        help="到达终点时的奖励（默认 10.0）"
    )
    parser.add_argument(
        "--energy_high_density", type=float, default=0.2,
        help="高能耗区域在地图中的比例（0~1）"
    )
    parser.add_argument(
        "--congestion_density", type=float, default=0.3,
        help="拥塞区域的面积比例（0~1），控制拥塞块/点的覆盖面积，强度固定为 Uniform(0,1)"
    )
    parser.add_argument(
        "--congestion_pattern", type=str, default="random",
        choices=["random", "block"],
        help=(
            "cell_congestion 模式下的拥塞图生成模式：\n"
            "  - random（默认）: 随机分布的拥塞点\n"
            "  - block: 在地图上生成一个连续的拥塞块（绕行有意义）"
        )
    )
    parser.add_argument(
        "--randomize_maps_each_reset", type=lambda x: x.lower() == "true", default=True,
        help="是否在每次 reset 时重采样能耗/拥塞地图（训练默认 True，固定地图可设为 False）"
    )
    parser.add_argument(
        "--load_threshold", type=float, default=0.6,
        help="load cost 的软阈值 τ（默认 0.6）; load = max(0, (raw - τ) / (1 - τ))"
    )
    parser.add_argument(
        "--start_goal_mode", type=str, default="random",
        choices=["random", "rect"],
        help="起终点采样模式：random=旧行为，rect=按矩形区域采样"
    )
    parser.add_argument(
        "--start_rect", type=str, default=None,
        help="当 start_goal_mode=rect 时的起点矩形，格式 x0,x1,y0,y1（含端点）"
    )
    parser.add_argument(
        "--goal_rect", type=str, default=None,
        help="当 start_goal_mode=rect 时的终点矩形，格式 x0,x1,y0,y1（含端点）"
    )

    # ========== 观测增强 ==========
    parser.add_argument(
        "--include_congestion_obs", type=lambda x: x.lower() == "true", default=False,
        help="是否在观测中包含 congestion patch"
    )
    parser.add_argument(
        "--congestion_patch_radius", type=int, default=1,
        help="congestion patch 半径"
    )
    parser.add_argument(
        "--include_energy_obs", type=lambda x: x.lower() == "true", default=False,
        help="是否在观测中包含 energy patch"
    )
    parser.add_argument(
        "--energy_patch_radius", type=int, default=1,
        help="energy patch 半径"
    )
    parser.add_argument(
        "--obs_rms", type=lambda x: x.lower() == "true", default=False,
        help="是否对全局观测做 RunningMeanStd 归一化 (GridObsNormWrapper)"
    )

    # ========== Seed Replay / Curriculum 设置 ==========
    parser.add_argument(
        "--train_seed_list", type=str, default=None,
        help="训练时 seed 列表文件路径（txt，每行一个整数 seed）"
    )
    parser.add_argument(
        "--train_seed_mix_prob", type=float, default=0.0,
        help="从 seed_list 采样的概率（0~1，默认 0；例如 0.5 表示 50%% episode 从 seed_list 抽样）"
    )

    # ========== 约束模式 ==========
    parser.add_argument(
        "--use_lagrange", type=lambda x: x.lower() == "true", default=True,
        help="是否使用 Lagrange 模式（True=自适应 λ，False=固定 λ）"
    )
    parser.add_argument("--lambda_lr", type=float, default=0.01, help="λ 更新步长（默认）")
    parser.add_argument(
        "--lambda_lr_energy", type=float, default=None,
        help="energy 的 λ 更新步长（可选，覆盖 lambda_lr）"
    )
    parser.add_argument(
        "--lambda_lr_load", type=float, default=None,
        help="load 的 λ 更新步长（可选，覆盖 lambda_lr）"
    )
    parser.add_argument(
        "--lambda_gap_mode", type=str, default="absolute",
        choices=["absolute", "ratio"],
        help="λ 更新时 gap 的尺度：absolute=原始差值，ratio=相对比例"
    )
    parser.add_argument(
        "--risk_factor", type=float, default=0.0,
        help="风险敏感成本 = mean + risk_factor * std；0 表示关闭风险加成"
    )

    # ========== Dual(λ) 更新模式（向后兼容：默认 standard） ==========
    parser.add_argument(
        "--dual_update_mode", type=str, default="standard",
        choices=["standard", "hysteresis", "decorrelated", "both", "precond", "preconditioned"],
        help="dual 更新模式：standard=旧行为；hysteresis=gap EMA+死区+滞回；decorrelated=相关性消耦；both=先消耦再滞回；precond=协方差预条件化"
    )
    parser.add_argument(
        "--dual_gap_ema_beta", type=float, default=0.10,
        help="dual gap EMA 平滑系数（推荐 0.05~0.2）"
    )
    parser.add_argument(
        "--dual_deadband", type=float, default=0.02,
        help="dual hysteresis 模式的死区阈值（ratio gap）"
    )
    parser.add_argument(
        "--dual_lr_down_scale", type=float, default=0.20,
        help="dual hysteresis 模式下降速度 = 上升速度 * scale"
    )
    parser.add_argument(
        "--dual_corr_ema_beta", type=float, default=0.05,
        help="dual decorrelated 模式相关性估计的 EMA 系数"
    )
    parser.add_argument("--dual_precond_eps", type=float, default=0.05, help="预条件化对角稳定项 eps")
    parser.add_argument("--dual_precond_clip", type=float, default=2.0, help="预条件化后的 g 分量 clip")
    parser.add_argument("--dual_precond_strength", type=float, default=0.3, help="预条件化插值强度 0~1")
    parser.add_argument(
        "--dual_precond_use_ema_stats",
        type=lambda x: x.lower() == "true",
        default=True,
        help="是否使用 EMA 统计 (dual_corr_state) 构建协方差"
    )

    # ========== 约束预算 ==========
    parser.add_argument(
        "--energy_budget", type=float, default=0.10,
        help="energy 成本预算（每步平均能耗上限，energy ∈ [0,1]）"
    )
    parser.add_argument(
        "--load_budget", type=float, default=0.08,
        help="load 成本预算（负载超限事件率上限）"
    )

    # ========== 课程学习（Curriculum Learning） ==========
    parser.add_argument(
        "--use_curriculum", type=lambda x: x.lower() == "true", default=False,
        help="是否启用课程学习（渐进式降低 load_budget）"
    )
    parser.add_argument(
        "--curriculum_start_load_budget", type=float, default=2.0,
        help="课程学习起始 load_budget（宽松约束）"
    )
    parser.add_argument(
        "--curriculum_end_load_budget", type=float, default=0.9,
        help="课程学习终止 load_budget（目标约束）"
    )
    parser.add_argument(
        "--curriculum_iters", type=int, default=600,
        help="课程学习衰减迭代次数（之后保持 end_budget）"
    )

    # ========== 初始 λ 值 ==========
    parser.add_argument(
        "--initial_lambda_energy", type=float, default=0.0,
        help="energy 的初始 λ 值"
    )
    parser.add_argument(
        "--initial_lambda_load", type=float, default=0.0,
        help="load 的初始 λ 值"
    )

    # ========== 对偶更新稳定化开关（默认关闭，保持旧行为） ==========
    parser.add_argument(
        "--lambda_gap_ema_beta", type=float, default=0.0,
        help="D1: gap EMA 系数，0=不启用（默认），>0 启用 EMA 平滑（推荐 0.05~0.1）"
    )
    parser.add_argument(
        "--lambda_update_freq", type=int, default=1,
        help="D2: λ 更新频率，1=每 iter 更新（默认），5=每 5 iter 更新一次"
    )
    parser.add_argument(
        "--lambda_deadzone", type=float, default=0.0,
        help="D3: dead-zone 阈值，0=不启用（默认），若 |gap| < 阈值则不更新 λ（推荐 0.001~0.003）"
    )
    parser.add_argument(
        "--lambda_max", type=float, default=None,
        help="D4: λ 上限，None=不限制（默认），设置为正数时 λ 被 clamp 到 [0, lambda_max]"
    )
    parser.add_argument(
        "--lambda_lr_up", type=float, default=None,
        help="D5: gap > 0 时的 λ 学习率（非对称 LR，用于约束违反时更激进地增加 λ）"
    )
    parser.add_argument(
        "--lambda_lr_down", type=float, default=None,
        help="D6: gap < 0 时的 λ 学习率（非对称 LR，用于约束满足时更保守地减少 λ）"
    )
    parser.add_argument(
        "--lambda_deadzone_energy", type=float, default=None,
        help="D7: energy 的 per-cost deadzone（覆盖 --lambda_deadzone）"
    )
    parser.add_argument(
        "--lambda_deadzone_load", type=float, default=None,
        help="D7: load 的 per-cost deadzone（覆盖 --lambda_deadzone）"
    )
    parser.add_argument(
        "--lambda_deadzone_up", type=float, default=None,
        help="D8: gap > 0 时的非对称 deadzone（约束违反时使用，推荐 0 或很小值）"
    )
    parser.add_argument(
        "--lambda_deadzone_down", type=float, default=None,
        help="D8: gap < 0 时的非对称 deadzone（约束满足时使用，推荐较大值如 0.03）"
    )

    # ========== E: Shared Cost Critic ==========
    parser.add_argument(
        "--cost_critic_mode", type=str, default="separate",
        choices=["separate", "shared", "aggregated"],
        help="E: Cost critic 模式 - 'separate'（默认，独立 value head）、'shared'（共享 value head）或 'aggregated'（two-critic: V_reward + V_cost_total）"
    )
    parser.add_argument(
        "--value_head_mode", type=str, default="standard",
        choices=["standard", "shared_all"],
        help="Value head 模式: standard=默认分头; shared_all=单头输出 reward+所有 cost"
    )
    
    # ========== Aggregated Cost 模式参数（仅 cost_critic_mode='aggregated' 时使用） ==========
    parser.add_argument(
        "--agg_cost_w_energy", type=float, default=1.0,
        help="Aggregated 模式：energy cost 的权重（默认 1.0）"
    )
    parser.add_argument(
        "--agg_cost_w_load", type=float, default=1.0,
        help="Aggregated 模式：load cost 的权重（默认 1.0）"
    )
    parser.add_argument(
        "--agg_cost_normalize_by_budget", type=lambda x: x.lower() == "true", default=True,
        help="Aggregated 模式：是否按预算归一化 total_cost（True=相对预算消耗，False=直接加权和）"
    )

    # ========== B/C: Best Feasible Checkpoint & Early Stop ==========
    parser.add_argument(
        "--enable_best_checkpoint", type=lambda x: x.lower() == "true", default=True,
        help="B: 启用最优可行点存档（默认 True）"
    )
    parser.add_argument(
        "--best_checkpoint_success_thresh", type=float, default=0.95,
        help="B: Best feasible 的成功率门槛（防止早期假可行污染）"
    )
    parser.add_argument(
        "--best_window_fsr", type=int, default=50,
        help="B: Best FSR 使用的窗口大小（最近 W 个 episode，可行率 tie-breaker 对齐）"
    )
    parser.add_argument(
        "--train_buffer_episodes", type=int, default=100,
        help="训练期用于统计指标的 episode ring-buffer 长度（默认 100）。注意 best_window_fsr 必须 <= 该值。"
    )
    parser.add_argument(
        "--best_window_tail", type=int, default=50,
        help="B: Tail-score 使用的最近成功 episode 数（默认 50）"
    )
    parser.add_argument(
        "--tail_percentile", type=float, default=95.0,
        help="B: Tail-score 百分位（默认 95，计算成功 episode per-step cost 的 p95）"
    )
    parser.add_argument(
        "--enable_early_stop", type=lambda x: x.lower() == "true", default=False,
        help="C: 启用早停（当约束稳定时停止训练，默认 False）"
    )
    parser.add_argument(
        "--early_stop_window", type=int, default=50,
        help="C: 早停判断的窗口大小（默认 50）"
    )
    parser.add_argument(
        "--early_stop_gap_std_threshold", type=float, default=0.005,
        help="C: 早停阈值，当所有 gap 的标准差 < 阈值时触发（默认 0.005）"
    )
    parser.add_argument(
        "--early_stop_min_iter", type=int, default=100,
        help="C: 早停最小迭代次数（默认 100）"
    )

    # ========== PPO 超参 ==========
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE λ")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip 系数")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="熵正则系数")
    parser.add_argument("--value_coef", type=float, default=0.5, help="value loss 系数")
    parser.add_argument("--cost_value_coef", type=float, default=1.0, help="cost value loss 权重")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪")
    parser.add_argument(
        "--log_actor_decomp", type=lambda x: x.lower() == "true", default=False,
        help="是否记录 actor loss 分解日志（默认 False）"
    )
    parser.add_argument(
        "--log_actor_grad_decomp", type=lambda x: x.lower() == "true", default=False,
        help="是否记录 actor 梯度分解（A2，默认 False）"
    )
    parser.add_argument(
        "--grad_decomp_interval", type=int, default=100,
        help="梯度分解记录间隔（按 policy update 计数，默认 100）"
    )

    # ========== 输出设置 ==========
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="基础输出目录"
    )
    parser.add_argument(
        "--run_tag", type=str, default=None,
        help="运行标签，若提供则输出目录变为 {output_dir}_{run_tag}"
    )
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")

    # ========== 调试 / 量纲核查 ==========
    parser.add_argument(
        "--unit_check", action="store_true", default=False,
        help="开启量纲一致性核查（仅打印/导出，不影响训练）"
    )
    parser.add_argument(
        "--unit_check_episodes", type=int, default=1,
        help="unit_check 模式下随机采样的 episode 数"
    )

    # ========== 可视化设置 ==========
    parser.add_argument("--vis_interval", type=int, default=0, help="每隔多少 iter 保存一次网格可视化(0=关闭)")
    parser.add_argument("--vis_start_iter", type=int, default=0, help="从第几次迭代开始保存可视化")
    parser.add_argument("--vis_seed", type=int, default=12345, help="可视化 eval 的 seed（固定场景）")
    parser.add_argument("--vis_episodes", type=int, default=1, help="每次保存可视化跑多少条 episode（不同 seed=vis_seed+ep）")
    parser.add_argument("--vis_deterministic", action="store_true", help="可视化用 greedy(argmax) 动作（不加则按策略采样）")

    # ========== 策略条件化参数 ==========
    parser.add_argument(
        "--policy_condition_on_lambda", type=lambda x: x.lower() == "true", default=False,
        help="是否将拉格朗日乘子 lambda 添加到策略观测中（默认 False）"
    )
    parser.add_argument(
        "--policy_condition_on_budget", type=lambda x: x.lower() == "true", default=False,
        help="是否将预算信息添加到策略观测中（默认 False）"
    )
    parser.add_argument(
        "--lambda_obs_clip", type=float, default=None,
        help="lambda 观测特征的裁剪上限（None 表示不裁剪）"
    )

    return parser.parse_args()


def _parse_rect(rect_str: str | None, name: str, grid_size: int):
    """Parse rect string "x0,x1,y0,y1" into a validated tuple within [0, grid_size-1]."""
    if rect_str is None:
        return None
    parts = [p.strip() for p in rect_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"{name} must be 'x0,x1,y0,y1', got: {rect_str}")
    try:
        x0, x1, y0, y1 = map(int, parts)
    except ValueError as e:
        raise ValueError(f"{name} must contain four integers: {rect_str}") from e
    if not (0 <= x0 <= x1 < grid_size and 0 <= y0 <= y1 < grid_size):
        raise ValueError(
            f"{name} out of bounds for grid_size={grid_size}: {(x0, x1, y0, y1)}"
        )
    return (x0, x1, y0, y1)


def _build_env(args):
    """构建环境的辅助函数"""
    max_steps = None if args.max_steps < 0 else args.max_steps
    base_env = GridRoutingEnv(
        grid_size=args.grid_size,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        max_steps=max_steps,
        start_goal_mode=args.start_goal_mode,
        start_rect=args.start_rect,
        goal_rect=args.goal_rect,
    )
    env = GridCostWrapper(
        base_env,
        energy_high_density=args.energy_high_density,
        congestion_density=args.congestion_density,
        congestion_pattern=args.congestion_pattern,
        load_threshold=args.load_threshold,
        randomize_maps_each_reset=args.randomize_maps_each_reset,
    )
    env = GridHardWrapper(env)
    if args.include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=args.congestion_patch_radius)
    if args.include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=args.energy_patch_radius)
    if args.obs_rms:
        env = GridObsNormWrapper(env)
        print("✅ GridObsNormWrapper (Dynamic RunningMeanStd) Attached!")
    return env


def _unwrap_chain(e):
    """遍历环境包装链"""
    seen = set()
    cur = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _find_base_env(e):
    """找到基础 GridRoutingEnv"""
    for x in _unwrap_chain(e):
        if hasattr(x, "agent_row") and hasattr(x, "goal_row"):
            return x
    return None


def _find_cost_wrapper(e):
    """找到 GridCostWrapper"""
    for x in _unwrap_chain(e):
        if isinstance(x, GridCostWrapper):
            return x
    return None


def _clone_to_cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a state_dict and move tensors to CPU for safe serialization."""
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _find_obs_stats(e):
    cur = e
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "obs_rms"):
            return cur.obs_rms
        cur = getattr(cur, "env", None)
    return None


def get_checkpoint_state(agent, env, config, meta=None):
    if hasattr(agent, "actor"):
        model_sd = agent.actor.state_dict()
    elif hasattr(agent, "network"):
        model_sd = _clone_to_cpu_state_dict(agent.network.state_dict())
    else:
        raise AttributeError("Agent has no actor/network state_dict for checkpointing")

    critic_sd = agent.critic.state_dict() if hasattr(agent, "critic") else None

    opt_sd = None
    if hasattr(agent, "optimizer"):
        try:
            opt_sd = copy.deepcopy(agent.optimizer.state_dict())
        except Exception:
            opt_sd = None

    state = {
        "model_state_dict": model_sd,
        "critic_state_dict": critic_sd,
        "optimizer_state_dict": opt_sd,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
        "obs_stats": _find_obs_stats(env),
    }
    if hasattr(agent, "network"):
        state["network_state_dict"] = _clone_to_cpu_state_dict(agent.network.state_dict())
    if meta:
        state.update(meta)
    return state


def _clone_to_cpu_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a flat dict and move any tensors to CPU."""
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def rollout_for_viz(agent, env, seed: int, deterministic: bool):
    """执行一次完整的 episode 用于可视化"""
    obs, info = env.reset(seed=seed)
    action_mask = info.get("action_mask", None)

    base = _find_base_env(env)
    costw = _find_cost_wrapper(env)

    traj = [(int(base.agent_row), int(base.agent_col))]

    done = False
    # 保险起见，最多走 max_steps
    max_steps = env.unwrapped.max_steps if hasattr(env.unwrapped, "max_steps") else 512

    agent.network.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if deterministic:
                # 需要从 agent 或环境获取预算和 lambda 信息
                # 从 agent.cfg.cost_budgets 和 agent.lambdas 获取
                obs_in = augment_obs_with_context(
                    obs,
                    agent.cfg.cost_budgets.get("energy", 0.0),
                    agent.cfg.cost_budgets.get("load", 0.0),
                    agent.lambdas,
                    include_budget=getattr(agent.cfg, "policy_condition_on_budget", False),
                    include_lambda=getattr(agent.cfg, "policy_condition_on_lambda", False),
                    lambda_clip=getattr(agent.cfg, "lambda_obs_clip", None)
                )
                obs_t = torch.tensor(obs_in, dtype=torch.float32, device=agent.device).unsqueeze(0)
                if action_mask is not None:
                    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
                else:
                    mask_t = None
                logits, _, _ = agent.network.forward(obs_t, mask_t)
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                # stochastic 模式也需要增强观测
                obs_in = augment_obs_with_context(
                    obs,
                    agent.cfg.cost_budgets.get("energy", 0.0),
                    agent.cfg.cost_budgets.get("load", 0.0),
                    agent.lambdas,
                    include_budget=getattr(agent.cfg, "policy_condition_on_budget", False),
                    include_lambda=getattr(agent.cfg, "policy_condition_on_lambda", False),
                    lambda_clip=getattr(agent.cfg, "lambda_obs_clip", None)
                )
                action, _, _, _, _ = agent.select_action(obs_in, action_mask=action_mask)

            obs, _, terminated, truncated, info = env.step(action)
            action_mask = info.get("action_mask", None)
            traj.append((int(base.agent_row), int(base.agent_col)))

            done = terminated or truncated
            if done:
                break

    congestion_map = getattr(costw, "_congestion_map", None)
    energy_map = getattr(costw, "_energy_map", None)
    start = traj[0]
    goal = (int(base.goal_row), int(base.goal_col))
    return congestion_map, energy_map, traj, start, goal


def main():
    args = parse_args()

    # 解析并验证起终点矩形（仅在 rect 模式下启用）
    args.start_rect = _parse_rect(args.start_rect, "start_rect", args.grid_size)
    args.goal_rect = _parse_rect(args.goal_rect, "goal_rect", args.grid_size)
    if args.start_goal_mode == "rect":
        if args.start_rect is None or args.goal_rect is None:
            raise ValueError("start_goal_mode=rect requires both --start_rect and --goal_rect")
        # 防止单点完全重合导致无法采样不同起终点
        if (
            args.start_rect == args.goal_rect
            and (args.start_rect[0] == args.start_rect[1])
            and (args.start_rect[2] == args.start_rect[3])
        ):
            raise ValueError("start_rect and goal_rect collapse to the same single cell; cannot ensure start!=goal")

    # 口径保护：best_window_fsr 不应大于 ring-buffer 长度
    if args.train_buffer_episodes > 0 and args.best_window_fsr > args.train_buffer_episodes:
        raise ValueError(
            f"--best_window_fsr ({args.best_window_fsr}) must be <= --train_buffer_episodes ({args.train_buffer_episodes})"
        )

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录（支持 run_tag 和自动递增）
    output_dir = make_output_dir(args.output_dir, args.run_tag)
    # 更新 args 以便后续保存 config.json 时记录最终目录
    args.final_output_dir = output_dir

    # 可选：训练前执行量纲一致性核查（不会修改训练逻辑）
    if args.unit_check:
        print("[UnitCheck] Running cost unit consistency check before training...")
        try:
            from check_cost_units import run_cost_unit_check
            summary, _ = run_cost_unit_check(
                vars(args),
                episodes=args.unit_check_episodes,
                seed=args.seed,
                csv_path=os.path.join(output_dir, "unit_check_steps.csv"),
                quiet=True,
            )
            summary_path = os.path.join(output_dir, "unit_check_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[UnitCheck] Summary saved to {summary_path}")
        except ImportError:
            print("[UnitCheck] check_cost_units module not found, skipping unit check.")
        except Exception as e:
            print(f"[UnitCheck] Failed to run unit check: {e}")
        # 重新设定随机种子，避免对后续训练产生影响
        set_seed(args.seed)

    # ========== 构建环境 ==========
    env = _build_env(args)
    
    # ========== 构建独立的 eval_env 用于可视化 ==========
    eval_env = _build_env(args)

    base_obs_dim = env.observation_space.shape[0]
    context_dim = (2 if args.policy_condition_on_budget else 0) + (2 if args.policy_condition_on_lambda else 0)
    obs_dim = base_obs_dim + context_dim
    act_dim = env.action_space.n

    # ========== 加载训练 seed 列表（用于 seed replay / curriculum） ==========
    train_seed_list = None
    if args.train_seed_list is not None:
        seed_list_path = args.train_seed_list
        if not os.path.isabs(seed_list_path):
            seed_list_path = os.path.join(os.getcwd(), seed_list_path)
        if os.path.exists(seed_list_path):
            train_seed_list = []
            with open(seed_list_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        train_seed_list.append(int(line))
            print(f"Loaded train_seed_list: {len(train_seed_list)} seeds from {seed_list_path}")
            print(f"  First 10: {train_seed_list[:10]}")
        else:
            print(f"[WARNING] train_seed_list file not found: {seed_list_path}")

    train_seed_mix_prob = args.train_seed_mix_prob
    train_rng = np.random.default_rng(args.seed + 12345)  # 独立 RNG 用于 seed replay 采样

    # ========== 设备选择 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 打印配置 ==========
    print("=" * 70)
    print("Multi-Critic PPO + Lagrangian Training")
    print("=" * 70)
    print(f"Device: {device}")
    max_steps_log = "auto(4*N*N)" if args.max_steps < 0 else str(args.max_steps)
    print(
        f"Grid: {args.grid_size}x{args.grid_size} | "
        f"step_penalty={args.step_penalty}, success_reward={args.success_reward}, "
        f"max_steps={max_steps_log}"
    )
    print(
        f"Energy map: binary 0/1, "
        f"density={args.energy_high_density}"
    )
    print(
        f"Load map: pattern={args.congestion_pattern}, "
        f"density={args.congestion_density}, threshold={args.load_threshold}"
    )
    print(f"Start/goal mode: {args.start_goal_mode}")
    if args.start_goal_mode == "rect":
        print(f"  start_rect={args.start_rect}, goal_rect={args.goal_rect}")
    cong_dim = (2 * args.congestion_patch_radius + 1) ** 2
    energy_dim = (2 * args.energy_patch_radius + 1) ** 2
    print("Observation: global coordinates + optional patches")
    print(
        f"  Congestion obs: {args.include_congestion_obs} "
        f"(radius={args.congestion_patch_radius}, dim={cong_dim})"
    )
    print(
        f"  Energy obs: {args.include_energy_obs} "
        f"(radius={args.energy_patch_radius}, "
        f"dim={energy_dim})"
    )
    print(f"Observation dim: {obs_dim} (base={base_obs_dim}, context={context_dim})")
    if args.policy_condition_on_budget or args.policy_condition_on_lambda:
        print(f"  Policy conditioning: budget={args.policy_condition_on_budget}, lambda={args.policy_condition_on_lambda}")
        if args.lambda_obs_clip is not None:
            print(f"  Lambda obs clip: {args.lambda_obs_clip}")
    print(f"Mode: {'Lagrange (adaptive lambda)' if args.use_lagrange else 'Fixed lambda (penalty)'}")
    print(f"Budgets: energy={args.energy_budget:.3f}, load={args.load_budget:.3f}")
    print(f"Initial lambda: energy={args.initial_lambda_energy:.3f}, "
          f"load={args.initial_lambda_load:.3f}")
    if args.risk_factor > 0:
        print(f"Risk-sensitive dual update enabled: risk_factor={args.risk_factor}")
    if args.use_lagrange:
        lr_energy = args.lambda_lr_energy if args.lambda_lr_energy is not None else args.lambda_lr
        lr_load = args.lambda_lr_load if args.lambda_lr_load is not None else args.lambda_lr
        print(f"Lambda LR: default={args.lambda_lr}, energy={lr_energy}, load={lr_load}")
        # 打印稳定化开关
        has_stabilization = (
            args.lambda_gap_ema_beta > 0 or
            args.lambda_update_freq > 1 or
            args.lambda_deadzone > 0 or
            args.lambda_max is not None or
            args.lambda_deadzone_up is not None or
            args.lambda_deadzone_down is not None
        )
        if has_stabilization:
            print("-" * 70)
            print("Dual update stabilization:")
            if args.lambda_gap_ema_beta > 0:
                print(f"  D1 (EMA): beta={args.lambda_gap_ema_beta}")
            if args.lambda_update_freq > 1:
                print(f"  D2 (freq): lambda updates every {args.lambda_update_freq} iters")
            if args.lambda_deadzone > 0:
                print(f"  D3 (deadzone): |gap| < {args.lambda_deadzone} -> skip update")
            if args.lambda_max is not None:
                print(f"  D4 (clamp): lambda <= {args.lambda_max}")
            if args.lambda_lr_up is not None or args.lambda_lr_down is not None:
                print(f"  D5-D6 (asymmetric LR): up={args.lambda_lr_up}, down={args.lambda_lr_down}")
            if args.lambda_deadzone_energy is not None or args.lambda_deadzone_load is not None:
                print(f"  D7 (per-cost deadzone): energy={args.lambda_deadzone_energy}, load={args.lambda_deadzone_load}")
            if args.lambda_deadzone_up is not None or args.lambda_deadzone_down is not None:
                print(f"  D8 (asymmetric deadzone): up={args.lambda_deadzone_up}, down={args.lambda_deadzone_down}")
    if args.cost_critic_mode == "shared":
        print(f"Cost critic mode: SHARED (all costs share one value head)")
    if args.value_head_mode == "shared_all":
        print(f"Value head mode: SHARED_ALL (single head for reward+all costs)")
    if args.enable_best_checkpoint:
        print(f"Best feasible checkpoint: ENABLED")
    if args.enable_early_stop:
        print(f"Early stop: ENABLED (window={args.early_stop_window}, threshold={args.early_stop_gap_std_threshold}, min_iter={args.early_stop_min_iter})")
    if train_seed_list is not None and train_seed_mix_prob > 0:
        print(f"Seed replay: ENABLED (mix_prob={train_seed_mix_prob}, num_seeds={len(train_seed_list)})")
    print("=" * 70)

    # ========== 构建 lambda_lrs ==========
    # 如果指定了 per-cost lr，则组装成 dict
    lambda_lrs = None
    if args.lambda_lr_energy is not None or args.lambda_lr_load is not None:
        lambda_lrs = {}
        if args.lambda_lr_energy is not None:
            lambda_lrs["energy"] = args.lambda_lr_energy
        if args.lambda_lr_load is not None:
            lambda_lrs["load"] = args.lambda_lr_load

    # ========== 构建 lambda_deadzones ==========
    lambda_deadzones = None
    if args.lambda_deadzone_energy is not None or args.lambda_deadzone_load is not None:
        lambda_deadzones = {}
        if args.lambda_deadzone_energy is not None:
            lambda_deadzones["energy"] = args.lambda_deadzone_energy
        if args.lambda_deadzone_load is not None:
            lambda_deadzones["load"] = args.lambda_deadzone_load

    # ========== 构建配置 ==========
    config = MultiCriticPPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        cost_budgets={
            "energy": args.energy_budget,
            "load": args.load_budget,
        },
        lambda_lr=args.lambda_lr,
        lambda_lrs=lambda_lrs,
        update_lambdas=args.use_lagrange,
        initial_lambdas={
            "energy": args.initial_lambda_energy,
            "load": args.initial_lambda_load,
        },
        cost_value_coef=args.cost_value_coef,
        # 对偶更新稳定化开关
        lambda_gap_ema_beta=args.lambda_gap_ema_beta,
        lambda_update_freq=args.lambda_update_freq,
        lambda_deadzone=args.lambda_deadzone,
        lambda_max=args.lambda_max,
        # 非对称学习率
        lambda_lr_up=args.lambda_lr_up,
        lambda_lr_down=args.lambda_lr_down,
        # per-cost deadzone
        lambda_deadzones=lambda_deadzones,
        # 非对称 deadzone
        lambda_deadzone_up=args.lambda_deadzone_up,
        lambda_deadzone_down=args.lambda_deadzone_down,
        lambda_gap_mode=args.lambda_gap_mode,
        risk_factor=args.risk_factor,
        # dual 更新模式
        dual_update_mode=args.dual_update_mode,
        dual_gap_ema_beta=args.dual_gap_ema_beta,
        dual_deadband=args.dual_deadband,
        dual_lr_down_scale=args.dual_lr_down_scale,
        dual_corr_ema_beta=args.dual_corr_ema_beta,
        dual_precond_eps=args.dual_precond_eps,
        dual_precond_clip=args.dual_precond_clip,
        dual_precond_strength=args.dual_precond_strength,
        dual_precond_use_ema_stats=args.dual_precond_use_ema_stats,
        # cost critic 模式
        cost_critic_mode=args.cost_critic_mode,
        value_head_mode=args.value_head_mode,
        # aggregated cost 模式参数
        agg_cost_w_energy=args.agg_cost_w_energy,
        agg_cost_w_load=args.agg_cost_w_load,
        agg_cost_normalize_by_budget=args.agg_cost_normalize_by_budget,
        log_actor_decomp=args.log_actor_decomp,
        log_actor_grad_decomp=args.log_actor_grad_decomp,
        grad_decomp_interval=args.grad_decomp_interval,
        device=device,
    )

    # ========== 创建 Agent ==========
    agent = MultiCriticPPO(config)
    
    # ========== 将策略条件化参数存储到 agent.cfg 以便后续使用 ==========
    agent.cfg.policy_condition_on_budget = args.policy_condition_on_budget
    agent.cfg.policy_condition_on_lambda = args.policy_condition_on_lambda
    agent.cfg.lambda_obs_clip = args.lambda_obs_clip

    # ========== 课程学习调度器 ==========
    curriculum_scheduler = None
    if args.use_curriculum:
        curriculum_scheduler = CurriculumScheduler(
            start_budget=args.curriculum_start_load_budget,
            end_budget=args.curriculum_end_load_budget,
            decay_iters=args.curriculum_iters
        )
        print("=" * 70)
        print("Curriculum Learning: ENABLED")
        print(f"  Load Budget: {args.curriculum_start_load_budget:.3f} → {args.curriculum_end_load_budget:.3f}")
        print(f"  Decay Iterations: {args.curriculum_iters}")
        print(f"  Initial load_budget will be overridden by curriculum")
        print("=" * 70)

    # ========== 初始化 Logger ==========
    logger = MetricsLogger()

    # ========== 训练状态 ==========
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []
    # [新增] 用于 Safety Gym 风格评估的 episode 信息 buffer
    train_infos_buffer: List[Dict[str, Any]] = []

    # ========== Seed Replay 采样辅助函数 ==========
    def sample_reset_seed():
        """根据 seed replay 配置决定是否从 seed list 采样"""
        if train_seed_list is not None and train_seed_mix_prob > 0:
            if train_rng.random() < train_seed_mix_prob:
                return int(train_rng.choice(train_seed_list))
        return None  # None 表示使用环境默认随机 seed

    # 初始 reset（也遵循 seed replay 规则）
    init_seed = sample_reset_seed()
    if init_seed is not None:
        obs, info = env.reset(seed=init_seed)
    else:
        obs, info = env.reset(seed=args.seed)
    action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
    current_map_fingerprint = info.get("map_fingerprint")

    # ========== 记录环境统计信息 ==========
    env_stats = {}

    # 记录能耗与拥塞分布（用于论文和调参）
    energy_mean = info.get("energy_mean", None)
    energy_high_ratio = info.get("energy_high_ratio", None)
    if energy_mean is not None:
        env_stats["energy_mean"] = energy_mean
    if energy_high_ratio is not None:
        env_stats["energy_high_ratio"] = energy_high_ratio

    actual_congestion_ratio = info.get("congestion_ratio", None)
    actual_congestion_mean = info.get("congestion_mean", None)
    if actual_congestion_ratio is not None:
        env_stats["actual_congestion_ratio"] = actual_congestion_ratio
        env_stats["actual_congestion_mean"] = actual_congestion_mean
        env_stats["configured_congestion_density"] = args.congestion_density
        # 打印 warning 如果实际比例与配置差异较大
        if abs(actual_congestion_ratio - args.congestion_density) > 0.05:
            print("=" * 70)
            print("[WARNING] Actual congestion ratio differs from configured density!")
            print(f"   Configured density: {args.congestion_density:.2%}")
            print(f"   Actual ratio:       {actual_congestion_ratio:.2%}")
            if args.congestion_pattern == "block":
                print("   (This is expected for 'block' pattern due to ceil rounding)")
            print("=" * 70)
            print()

    ep_ret, ep_len = 0.0, 0
    # [新增] 用于跟踪 episode 累计成本
    ep_cost_energy, ep_cost_load = 0.0, 0.0
    start_time = time.time()

    # ========== B: Best Feasible Checkpoint 状态 ==========
    best_feasible_return = -float("inf")
    best_feasible_iter = None
    best_feasible_state = None  # 保存 agent 状态的快照
    # ========== B2: Best FSR / Tail Checkpoint 状态 ==========
    best_fsr_value = -float("inf")
    best_fsr_iter = None
    best_fsr_return = -float("inf")
    best_fsr_state = None

    best_tail_value = float("inf")
    best_tail_iter = None
    best_tail_return = -float("inf")
    best_tail_fsr = -float("inf")
    best_tail_state = None
    best_tail_energy_p95 = None
    best_tail_load_p95 = None

    # ========== C: Early Stop 状态 ==========
    gap_history: Dict[str, List[float]] = {"energy": [], "load": []}
    early_stopped = False
    early_stop_iter = None

    # ========== 训练循环 ==========
    for iteration in range(1, args.total_iters + 1):
        map_fingerprints_iter: List[str] = []
        if current_map_fingerprint is not None:
            map_fingerprints_iter.append(current_map_fingerprint)
        # ========== 课程学习：动态更新 load_budget ==========
        if curriculum_scheduler is not None:
            current_load_budget = curriculum_scheduler.get_budget(iteration)
            agent.cfg.cost_budgets['load'] = current_load_budget
            # 也更新 args.load_budget 以便后续日志记录
            args.load_budget = current_load_budget
        
        # 收集 rollout
        for _ in range(config.batch_size):
            # 增强观测（添加预算和 lambda 信息）
            obs_in = augment_obs_with_context(
                obs, 
                args.energy_budget, 
                args.load_budget, 
                agent.lambdas,
                include_budget=args.policy_condition_on_budget,
                include_lambda=args.policy_condition_on_lambda,
                lambda_clip=args.lambda_obs_clip
            )
            
            # 选择动作
            action, log_prob, v_reward, v_costs = agent.select_action(
                obs_in, action_mask=action_mask
            )

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 获取 cost
            cost_components = info.get(
                "cost_components", {"energy": 0.0, "load": 0.0}
            )

            # 存储 transition
            agent.collect_rollout(
                obs=obs_in,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                action_mask=action_mask,
                v_reward=v_reward,
                v_costs=v_costs,
                costs_dict=cost_components,
            )

            ep_ret += reward
            ep_len += 1
            # [新增] 累计 episode 成本
            ep_cost_energy += cost_components.get("energy", 0.0)
            ep_cost_load += cost_components.get("load", 0.0)
            obs = next_obs
            action_mask = info.get("action_mask", action_mask)

            if done:
                # [修复 Bug A] 添加到统计列表
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)
                episode_successes.append(bool(terminated))  # terminated=True 表示成功到达终点
                
                # [新增] 保存 episode 信息到 buffer（用于 Safety Gym 风格评估）
                ep_info = {
                    'episode_return': ep_ret,
                    'episode_length': ep_len,
                    'success': terminated,
                    'episode_cost_energy': ep_cost_energy,
                    'episode_cost_load': ep_cost_load,
                }
                train_infos_buffer.append(ep_info)
                # 保留最近 N 个 episodes（用于 metrics 统计）
                max_buf_eps = int(getattr(args, "train_buffer_episodes", 100))
                if max_buf_eps > 0 and len(train_infos_buffer) > max_buf_eps:
                    train_infos_buffer = train_infos_buffer[-max_buf_eps:]

                ep_ret, ep_len = 0.0, 0
                ep_cost_energy, ep_cost_load = 0.0, 0.0
                # Seed replay：根据 mix_prob 决定是否从 seed_list 采样
                reset_seed = sample_reset_seed()
                if reset_seed is not None:
                    obs, info = env.reset(seed=reset_seed)
                else:
                    obs, info = env.reset()
                action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
                current_map_fingerprint = info.get("map_fingerprint", current_map_fingerprint)
                if current_map_fingerprint is not None:
                    map_fingerprints_iter.append(current_map_fingerprint)

        # PPO 更新
        metrics = agent.update()

        env_map_changes = len(set(map_fingerprints_iter)) if map_fingerprints_iter else 0
        last_map_fingerprint = map_fingerprints_iter[-1] if map_fingerprints_iter else None

        # 计算统计量
        recent_n = min(50, len(episode_returns))
        if recent_n > 0:
            avg_return = float(np.mean(episode_returns[-recent_n:]))
            avg_length = float(np.mean(episode_lengths[-recent_n:]))
            success_rate = float(np.mean(episode_successes[-recent_n:]))
        else:
            avg_return, avg_length, success_rate = 0.0, 0.0, 0.0

        # 记录 metrics（包含新增的 gap 和 KKT 残差）
        log_entry = {
            "iteration": iteration,
            "avg_return": avg_return,
            "avg_length": avg_length,
            "success_rate": success_rate,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "avg_cost_energy": metrics["avg_cost_energy"],
            "avg_cost_load": metrics["avg_cost_load"],
            "lambda_energy": metrics["lambda_energy"],
            "lambda_load": metrics["lambda_load"],
            "budget_energy": args.energy_budget,
            "budget_load": args.load_budget,
            "env_map_changes": env_map_changes,
            # 新增：gap 和 KKT 残差
            "gap_energy": metrics.get("gap_energy", metrics["avg_cost_energy"] - args.energy_budget),
            "gap_load": metrics.get("gap_load", metrics["avg_cost_load"] - args.load_budget),
            "kkt_energy": metrics.get("kkt_energy", metrics["lambda_energy"] * (metrics["avg_cost_energy"] - args.energy_budget)),
            "kkt_load": metrics.get("kkt_load", metrics["lambda_load"] * (metrics["avg_cost_load"] - args.load_budget)),
        }

        if last_map_fingerprint is not None:
            log_entry["env_map_fingerprint"] = last_map_fingerprint

        if args.log_actor_decomp:
            actor_keys = [
                "adv_r_mean", "adv_r_std",
                "adv_energy_mean", "adv_energy_std",
                "adv_load_mean", "adv_load_std",
                "lambda_energy", "lambda_load",
                "penalty_mean", "penalty_abs_mean",
                "adv_eff_mean", "adv_eff_std",
                "pg_loss_total", "pg_loss_r_like", "pg_loss_p_like",
            ]
            for key in actor_keys:
                if key in metrics:
                    log_entry[key] = metrics[key]

        # A2 梯度分解日志（仅当启用时返回）
        for key in [
            "g_r_norm", "g_c_norm", "g_t_norm",
            "g_c_over_r", "cos_total_r", "cos_total_c",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        # 风险敏感指标（若开启）
        for key in [
            "std_cost_energy",
            "std_cost_load",
            "risk_cost_energy",
            "risk_cost_load",
            "risk_gap_energy",
            "risk_gap_load",
            "risk_gap_ratio_energy",
            "risk_gap_ratio_load",
            "risk_kkt_energy",
            "risk_kkt_load",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        if "gap_ratio_energy" in metrics:
            log_entry["gap_ratio_energy"] = metrics["gap_ratio_energy"]
        if "gap_ratio_load" in metrics:
            log_entry["gap_ratio_load"] = metrics["gap_ratio_load"]
        if "lambda_gap_mode" in metrics:
            log_entry["lambda_gap_mode"] = metrics["lambda_gap_mode"]
        # dual 更新诊断
        for key in [
            "gap_energy_raw",
            "gap_load_raw",
            "gap_energy_ema",
            "gap_load_ema",
            "gap_energy_used",
            "gap_load_used",
            "corr_gap_ema",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        # 新增：absolute gap 口径（与 PPO 日志对齐），若存在则补充
        for key in [
            "gap_abs_energy",
            "gap_abs_load",
            "gap_abs_energy_ema",
            "gap_abs_load_ema",
            "gap_abs_energy_used",
            "gap_abs_load_used",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        for key in metrics:
            if key.startswith("dual_precond_"):
                log_entry[key] = metrics[key]
        # 添加 EMA gap（如果存在）
        if "ema_gap_energy" in metrics:
            log_entry["ema_gap_energy"] = metrics["ema_gap_energy"]
        if "ema_gap_load" in metrics:
            log_entry["ema_gap_load"] = metrics["ema_gap_load"]
        # aggregated 模式的额外指标（若存在则写入）
        for key in [
            "cost_total_mean",
            "cost_total_budget",
            "gap_total",
            "gap_ratio_total",
            "lambda_total",
            "kkt_total",
            "ema_gap_total",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        
        # [新增] 添加诊断指标到 log_entry（在 logger.log 之前）
        # 1. Safety Gym 风格累计成本率：rho_* = cumulative_cost / total_steps
        for key in metrics:
            if key.startswith("rho_"):
                log_entry[key] = metrics[key]
        
        # 2. PPO 更新诊断（判断 policy 是否还在动）
        if "approx_kl" in metrics:
            log_entry["approx_kl"] = metrics["approx_kl"]
        if "clip_frac" in metrics:
            log_entry["clip_frac"] = metrics["clip_frac"]
        for key in [
            "actor_param_delta_l2",
            "actor_param_norm_l2",
            "actor_param_delta_ratio",
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]

        # 3. Advantage penalty diagnostics（核心诊断：cost 是否真的在推动 policy）
        #    所有统计量基于"actor 实际使用的 normalized advantage"
        for key in [
            "adv_reward_abs_mean",        # |adv_reward| 均值
            "adv_penalty_abs_mean",       # |penalty_adv_total| 均值
            "adv_penalty_to_reward_ratio",# penalty 占 reward 的比例
            "adv_reward_mean",            # adv_reward 均值（带符号）
            "adv_penalty_mean",           # penalty_adv 均值（带符号）
            "lambdaA_energy_abs_mean",    # |lambda_energy * adv_energy| 均值
            "lambdaA_load_abs_mean",      # |lambda_load * adv_load| 均值
            "lambdaA_total_abs_mean",     # 总惩罚项绝对值均值
        ]:
            if key in metrics:
                log_entry[key] = metrics[key]
        
        # 3. 计算 Safety Gym episode 指标
        log_entry["avg_energy_success"] = 0.0
        log_entry["avg_load_success"] = 0.0
        log_entry["feasible_success_rate"] = 0.0
        log_entry["feasible_success_rate_buffer"] = 0.0
        log_entry["feasible_given_success"] = 0.0
        log_entry["feasible_success_count"] = 0
        log_entry["num_success_episodes"] = 0
        log_entry["num_episodes_buffer"] = len(train_infos_buffer)
        if len(train_infos_buffer) > 0:
            success_episodes = [info for info in train_infos_buffer if info.get('success', False)]
            
            # 计算 Success-Only Costs (per-step mean)
            if len(success_episodes) > 0:
                avg_energy_success = np.mean([ep['episode_cost_energy'] / max(1, ep['episode_length']) for ep in success_episodes])
                avg_load_success = np.mean([ep['episode_cost_load'] / max(1, ep['episode_length']) for ep in success_episodes])
            else:
                avg_energy_success = 0.0
                avg_load_success = 0.0
            
            # 计算可行率（两个版本）
            feasible_success_count = 0
            for ep in success_episodes:
                ep_len = max(1, ep['episode_length'])
                e_mean = ep['episode_cost_energy'] / ep_len
                l_mean = ep['episode_cost_load'] / ep_len
                if e_mean <= args.energy_budget and l_mean <= args.load_budget:
                    feasible_success_count += 1
            
            feasible_success_rate = feasible_success_count / len(train_infos_buffer) if train_infos_buffer else 0.0
            feasible_given_success = feasible_success_count / max(1, len(success_episodes))
            
            # 写入 log_entry
            log_entry["avg_energy_success"] = avg_energy_success
            log_entry["avg_load_success"] = avg_load_success
            log_entry["feasible_success_rate"] = feasible_success_rate
            log_entry["feasible_success_rate_buffer"] = feasible_success_rate
            log_entry["feasible_given_success"] = feasible_given_success
            log_entry["feasible_success_count"] = feasible_success_count
            log_entry["num_success_episodes"] = len(success_episodes)
            log_entry["num_episodes_buffer"] = len(train_infos_buffer)

        # ========== B2: Best FSR / Tail 计算与存档 ==========
        # 口径保护：best_window_fsr 不应大于 buffer 长度
        max_buf_eps = int(getattr(args, "train_buffer_episodes", 100))
        if max_buf_eps > 0 and args.best_window_fsr > max_buf_eps:
            raise ValueError(f"--best_window_fsr ({args.best_window_fsr}) must be <= --train_buffer_episodes ({max_buf_eps})")

        window_fsr = args.best_window_fsr
        episodes_window = train_infos_buffer if window_fsr <= 0 else train_infos_buffer[-window_fsr:]
        has_window_data = len(episodes_window) > 0
        feasible_success_count_w = 0
        success_count_w = 0
        for ep in episodes_window:
            if ep.get('success', False):
                success_count_w += 1
                ep_len = max(1, ep['episode_length'])
                e_mean = ep['episode_cost_energy'] / ep_len
                l_mean = ep['episode_cost_load'] / ep_len
                if e_mean <= args.energy_budget and l_mean <= args.load_budget:
                    feasible_success_count_w += 1
        feasible_success_rate_window = feasible_success_count_w / max(1, len(episodes_window))
        window_returns = [ep['episode_return'] for ep in episodes_window]
        avg_return_window = float(np.mean(window_returns)) if window_returns else 0.0

        # 写入 window 口径指标，避免 metrics vs meta 混淆
        log_entry["best_window_fsr"] = int(window_fsr)
        log_entry["feasible_success_rate_window"] = float(feasible_success_rate_window)
        log_entry["avg_return_window"] = float(avg_return_window)

        tail_k = args.best_window_tail
        success_tail_eps = [ep for ep in train_infos_buffer if ep.get('success', False)]
        if tail_k > 0:
            success_tail_eps = success_tail_eps[-tail_k:]
        energy_rates = [ep['episode_cost_energy'] / max(1, ep['episode_length']) for ep in success_tail_eps]
        load_rates = [ep['episode_cost_load'] / max(1, ep['episode_length']) for ep in success_tail_eps]
        min_tail_samples = max(10, int(tail_k / 5) if tail_k > 0 else 10)
        energy_p95 = None
        load_p95 = None
        tail_score = None
        if len(energy_rates) >= min_tail_samples and len(load_rates) >= min_tail_samples:
            energy_p95 = float(np.percentile(energy_rates, args.tail_percentile))
            load_p95 = float(np.percentile(load_rates, args.tail_percentile))
            if args.energy_budget > 0 and args.load_budget > 0:
                tail_score = max(energy_p95 / args.energy_budget, load_p95 / args.load_budget)

        if args.enable_best_checkpoint and has_window_data:
            fsr_improved = (
                feasible_success_rate_window > best_fsr_value + 1e-6
                or (
                    abs(feasible_success_rate_window - best_fsr_value) < 1e-6
                    and avg_return_window > best_fsr_return + 1e-6
                )
            )
            if fsr_improved:
                best_fsr_value = feasible_success_rate_window
                best_fsr_iter = iteration
                best_fsr_return = avg_return_window
                best_fsr_state = get_checkpoint_state(
                    agent,
                    env,
                    args,
                    meta={
                        "lambdas": _clone_to_cpu_dict(agent.lambdas),
                        "ema_gaps": _clone_to_cpu_dict(agent.ema_gaps),
                        "iter_count": agent._iter_count,
                        "best_return": best_fsr_return,
                        "best_iter": best_fsr_iter,
                        "gap_energy": log_entry["gap_energy"],
                        "gap_load": log_entry["gap_load"],
                        "best_fsr_value": best_fsr_value,
                        "best_tail_value": best_tail_value,
                        "energy_p95": energy_p95,
                        "load_p95": load_p95,
                        "tail_score": tail_score,
                    },
                )

            tail_score_value = tail_score if tail_score is not None else float("inf")
            tail_improved = (
                tail_score_value < best_tail_value - 1e-6
                or (
                    abs(tail_score_value - best_tail_value) < 1e-6
                    and feasible_success_rate_window > best_tail_fsr + 1e-6
                )
                or (
                    abs(tail_score_value - best_tail_value) < 1e-6
                    and abs(feasible_success_rate_window - best_tail_fsr) < 1e-6
                    and avg_return_window > best_tail_return + 1e-6
                )
            )
            if tail_improved:
                best_tail_value = tail_score_value
                best_tail_iter = iteration
                best_tail_return = avg_return_window
                best_tail_fsr = feasible_success_rate_window
                best_tail_energy_p95 = energy_p95
                best_tail_load_p95 = load_p95
                best_tail_state = get_checkpoint_state(
                    agent,
                    env,
                    args,
                    meta={
                        "lambdas": _clone_to_cpu_dict(agent.lambdas),
                        "ema_gaps": _clone_to_cpu_dict(agent.ema_gaps),
                        "iter_count": agent._iter_count,
                        "best_return": best_tail_return,
                        "best_iter": best_tail_iter,
                        "gap_energy": log_entry["gap_energy"],
                        "gap_load": log_entry["gap_load"],
                        "best_fsr_value": best_fsr_value,
                        "best_tail_value": best_tail_value,
                        "energy_p95": energy_p95,
                        "load_p95": load_p95,
                        "tail_score": tail_score,
                    },
                )

        log_entry["energy_p95_last50"] = energy_p95
        log_entry["load_p95_last50"] = load_p95
        log_entry["tail_score_last50"] = tail_score
        log_entry["best_fsr_iter"] = best_fsr_iter
        log_entry["best_tail_iter"] = best_tail_iter

        logger.log(log_entry)

        # ========== B: Best Feasible Checkpoint 逻辑 ==========
        if args.enable_best_checkpoint:
            gap_energy = log_entry["gap_energy"]
            gap_load = log_entry["gap_load"]
            # 可行条件：两个约束都满足（gap <= 0）+ success_rate >= thresh
            is_feasible = (gap_energy <= 0) and (gap_load <= 0)
            is_reliable = (success_rate >= args.best_checkpoint_success_thresh)  # 防止低 success 的假可行
            if is_feasible and is_reliable and avg_return > best_feasible_return:
                best_feasible_return = avg_return
                best_feasible_iter = iteration
                # 保存 agent 状态的快照（深拷贝）
                best_feasible_state = get_checkpoint_state(
                    agent,
                    env,
                    args,
                    meta={
                        "lambdas": _clone_to_cpu_dict(agent.lambdas),
                        "ema_gaps": _clone_to_cpu_dict(agent.ema_gaps),
                        "iter_count": agent._iter_count,
                        "best_return": best_feasible_return,
                        "best_iter": best_feasible_iter,
                        "gap_energy": gap_energy,
                        "gap_load": gap_load,
                    },
                )

        # ========== C: Early Stop 逻辑 ==========
        if args.enable_early_stop and not early_stopped:
            gap_history["energy"].append(log_entry["gap_energy"])
            gap_history["load"].append(log_entry["gap_load"])

            # 只保留最近 window 个值
            window = args.early_stop_window
            if len(gap_history["energy"]) > window:
                gap_history["energy"] = gap_history["energy"][-window:]
                gap_history["load"] = gap_history["load"][-window:]

            # 检查是否可以早停
            if iteration >= args.early_stop_min_iter and len(gap_history["energy"]) >= window:
                std_energy = float(np.std(gap_history["energy"]))
                std_load = float(np.std(gap_history["load"]))
                mean_energy = float(np.mean(gap_history["energy"]))
                mean_load = float(np.mean(gap_history["load"]))
                threshold = args.early_stop_gap_std_threshold

                # 早停条件：
                # 1. gap 稳定（std < threshold）
                # 2. gap 接近可行或已可行（mean < 0.02，允许 2% 的小幅违反）
                is_stable = std_energy < threshold and std_load < threshold
                is_near_feasible = mean_energy < 0.02 and mean_load < 0.02

                if is_stable and is_near_feasible:
                    early_stopped = True
                    early_stop_iter = iteration
                    print("\n" + "=" * 70)
                    print(f"*** EARLY STOP at iteration {iteration} ***")
                    print(f"Gap std: energy={std_energy:.6f}, load={std_load:.6f} (threshold={threshold})")
                    print(f"Gap mean: energy={mean_energy:.6f}, load={mean_load:.6f}")
                    print("=" * 70 + "\n")

        # 打印日志
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            total_steps = iteration * config.batch_size
            
            # === [新增] Safety Gym 风格评估 ===
            
            # 1. 筛选成功 Episode
            success_episodes = [info for info in train_infos_buffer if info.get('success', False)]
            
            # 2. 计算 Success-Only Costs (避免失败样本拉低均值)
            # 注意：这里统一转成 per-step mean，与 budget 保持一致
            if len(success_episodes) > 0:
                avg_energy_success = np.mean([ep['episode_cost_energy'] / max(1, ep['episode_length']) for ep in success_episodes])
                avg_load_success = np.mean([ep['episode_cost_load'] / max(1, ep['episode_length']) for ep in success_episodes])
            else:
                avg_energy_success = 0.0
                avg_load_success = 0.0

            # 3. 计算 "真可行" (Feasible & Success) 比例
            # 严格判断：成功 且 满足双约束
            feasible_success_count = 0
            for ep in success_episodes:
                ep_len = max(1, ep['episode_length'])
                e_mean = ep['episode_cost_energy'] / ep_len
                l_mean = ep['episode_cost_load'] / ep_len
                
                if e_mean <= args.energy_budget and l_mean <= args.load_budget:
                    feasible_success_count += 1
            
            # 计算两个版本的可行率（建议1）
            feasible_success_rate = feasible_success_count / len(train_infos_buffer) if train_infos_buffer else 0.0
            feasible_given_success = feasible_success_count / max(1, len(success_episodes))

            # 4. 写入到 log_entry (供 JSON 保存)
            # 注意：此时 log_entry 已在上面创建，这里需要更新它
            log_entry_sg_metrics = {
                "avg_energy_success": avg_energy_success,
                "avg_load_success": avg_load_success,
                "feasible_success_rate": feasible_success_rate,
                "feasible_given_success": feasible_given_success,
                "feasible_success_count": feasible_success_count,
                "num_success_episodes": len(success_episodes),
                "num_episodes_buffer": len(train_infos_buffer),
            }
            log_entry.update(log_entry_sg_metrics)

            print("=" * 70)
            print(f"Iteration {iteration}/{args.total_iters} | "
                  f"Steps: {total_steps:,} | Time: {elapsed:.1f}s")
            print("-" * 70)
            print(f"Return: {avg_return:.2f} | Length: {avg_length:.1f} | "
                  f"Success: {success_rate:.2%}")
            print("-" * 70)
            print(f"Cost (energy): {metrics['avg_cost_energy']:.4f} "
                  f"(budget: {args.energy_budget:.4f}, "
                  f"gap: {metrics['avg_cost_energy'] - args.energy_budget:+.4f})")
            print(f"Cost (load):   {metrics['avg_cost_load']:.4f} "
                  f"(budget: {args.load_budget:.4f}, "
                  f"gap: {metrics['avg_cost_load'] - args.load_budget:+.4f})")
            if args.risk_factor > 0 and "risk_cost_energy" in metrics:
                print(f"Risk cost (energy): {metrics['risk_cost_energy']:.4f} "
                    f"(gap: {metrics['risk_gap_energy']:+.4f})")
            if args.risk_factor > 0 and "risk_cost_load" in metrics:
                print(f"Risk cost (load):   {metrics['risk_cost_load']:.4f} "
                    f"(gap: {metrics['risk_gap_load']:+.4f})")
            print("-" * 70)
            # [新增] Safety Gym 风格指标打印（从 log_entry 读取，避免重复 update）
            print(f"Success-only costs: Energy={log_entry['avg_energy_success']:.4f}, Load={log_entry['avg_load_success']:.4f}")
            print(f"Feasible & Success Rate: {log_entry['feasible_success_rate']:.2%} ({log_entry['feasible_success_count']}/{log_entry['num_episodes_buffer']} eps)")
            print(f"Feasible given Success: {log_entry['feasible_given_success']:.2%} ({log_entry['feasible_success_count']}/{log_entry['num_success_episodes']} success eps)")
            print("-" * 70)
            print(f"Lambda (energy): {metrics['lambda_energy']:.4f} | "
                  f"Lambda (load): {metrics['lambda_load']:.4f}")
            print("-" * 70)
            print(f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f}")
            print("=" * 70)
            print()

        # ========== 可视化保存 ==========
        if args.vis_interval > 0 and iteration >= args.vis_start_iter and (iteration % args.vis_interval == 0):
            vis_dir = os.path.join(output_dir, "viz")
            os.makedirs(vis_dir, exist_ok=True)

            for ep in range(args.vis_episodes):
                seed = args.vis_seed + ep
                cong, en, traj, start, goal = rollout_for_viz(
                    agent, eval_env, seed=seed, deterministic=args.vis_deterministic
                )
                save_path = os.path.join(vis_dir, f"iter_{iteration:04d}_seed_{seed}.png")
                save_grid_route_viz(
                    cong, en, traj, start, goal,
                    save_path=save_path,
                    title=f"iter={iteration} seed={seed} len={len(traj)-1}"
                )
                print(f"[viz] saved: {save_path}")

        # ========== C: Early Stop break ==========
        if early_stopped:
            break

    # ========== 训练结束 ==========
    total_time = time.time() - start_time
    final_iter = early_stop_iter if early_stopped else args.total_iters
    print("\n" + "=" * 70)
    print("Training Finished!")
    if early_stopped:
        print(f"*** Early stopped at iteration {early_stop_iter} ***")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Final iteration: {final_iter}")
    print(f"Mode: {'Lagrange' if args.use_lagrange else 'Fixed lambda'}")
    print(f"Final lambda: energy={agent.lambdas['energy']:.4f}, "
          f"load={agent.lambdas['load']:.4f}")
    if best_feasible_iter is not None:
        print(f"Best feasible: iter={best_feasible_iter}, return={best_feasible_return:.2f}")
    if best_fsr_iter is not None:
        print(f"Best FSR: iter={best_fsr_iter}, fsr={best_fsr_value:.4f}, return={best_fsr_return:.2f}")
    if best_tail_iter is not None:
        tail_energy_disp = best_tail_energy_p95 if best_tail_energy_p95 is not None else float('nan')
        tail_load_disp = best_tail_load_p95 if best_tail_load_p95 is not None else float('nan')
        print(
            f"Best tail: iter={best_tail_iter}, tail_score={best_tail_value:.4f}, "
            f"fsr={best_tail_fsr:.4f}, return={best_tail_return:.2f}, "
            f"p95_energy={tail_energy_disp:.4f}, p95_load={tail_load_disp:.4f}"
        )
    print("=" * 70)

    # ========== 保存结果 ==========
    # 保存 metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    logger.save(metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    # 保存配置（包含环境统计信息和训练结果元信息）
    config_dict = vars(args).copy()
    config_dict["env_stats"] = env_stats  # 添加环境统计信息
    config_dict["training_results"] = {
        "final_iter": final_iter,
        "early_stopped": early_stopped,
        "early_stop_iter": early_stop_iter,
        "best_feasible_iter": best_feasible_iter,
        "best_feasible_return": best_feasible_return if best_feasible_iter else None,
        "best_fsr_iter": best_fsr_iter,
        "best_fsr_value": best_fsr_value if best_fsr_iter else None,
        "best_fsr_return": best_fsr_return if best_fsr_iter else None,
        "best_tail_iter": best_tail_iter,
        "best_tail_value": best_tail_value if best_tail_iter else None,
        "best_tail_return": best_tail_return if best_tail_iter else None,
        "best_tail_energy_p95": best_tail_energy_p95 if best_tail_iter else None,
        "best_tail_load_p95": best_tail_load_p95 if best_tail_iter else None,
        "best_tail_fsr": best_tail_fsr if best_tail_iter else None,
    }
    # 记录 seed replay 的实际配置
    config_dict["seed_replay"] = {
        "train_seed_list_path": seed_list_path if train_seed_list is not None else None,
        "train_seed_mix_prob": train_seed_mix_prob,
        "num_seeds_loaded": len(train_seed_list) if train_seed_list is not None else 0,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config_path}")

    # 绘制训练曲线（传入 best_feasible_iter 用于画垂直线）
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(logger.get_data(), plot_path, best_iter=best_feasible_iter)
    print(f"Training curves saved to: {plot_path}")
    
    # 绘制 Safety Gym 风格曲线
    safety_gym_path = os.path.join(output_dir, "safety_gym_curves.png")
    plot_safety_gym_curves(logger.get_data(), safety_gym_path)

    # 保存模型（最终模型，含 optimizer 与 obs 归一化统计）
    if args.save_model:
        checkpoint = get_checkpoint_state(agent, env, args)
        ckpt_path = os.path.join(output_dir, "checkpoint_final.pt")
        torch.save(checkpoint, ckpt_path)
        print("✅ Checkpoint (Model + Obs Stats) saved.")

    # 保存 Best Feasible Checkpoint
    if args.enable_best_checkpoint and best_feasible_state is not None:
        best_model_path = os.path.join(output_dir, "best_feasible.pt")
        torch.save(best_feasible_state, best_model_path)
        print(f"Best feasible model saved to: {best_model_path}")
        print(f"  -> iter={best_feasible_iter}, return={best_feasible_return:.2f}, "
              f"gap_energy={best_feasible_state['gap_energy']:.4f}, "
              f"gap_load={best_feasible_state['gap_load']:.4f}")

        # 同时保存 metadata JSON（方便查看）
        best_meta_path = os.path.join(output_dir, "best_feasible_meta.json")
        with open(best_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_feasible_iter": best_feasible_iter,
                "best_feasible_return": best_feasible_return,
                "energy_budget": args.energy_budget,
                "load_budget": args.load_budget,
                "gap_energy": best_feasible_state['gap_energy'],
                "gap_load": best_feasible_state['gap_load'],
            }, f, indent=2, ensure_ascii=False)
        print(f"Best feasible metadata saved to: {best_meta_path}")

    # 保存 Best FSR Checkpoint
    if args.enable_best_checkpoint and best_fsr_state is not None:
        best_fsr_path = os.path.join(output_dir, "best_fsr.pt")
        torch.save(best_fsr_state, best_fsr_path)
        print(f"Best FSR model saved to: {best_fsr_path}")
        print(f"  -> iter={best_fsr_iter}, fsr={best_fsr_value:.4f}, return={best_fsr_return:.2f}")

        best_fsr_meta_path = os.path.join(output_dir, "best_fsr_meta.json")
        with open(best_fsr_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "iter": best_fsr_iter,
                "feasible_success_rate": best_fsr_value,
                "avg_return": best_fsr_return,
                "window": args.best_window_fsr,
                "tail_window": args.best_window_tail,
                "tail_percentile": args.tail_percentile,
                "energy_p95": best_fsr_state.get("energy_p95"),
                "load_p95": best_fsr_state.get("load_p95"),
                "tail_score": best_fsr_state.get("tail_score"),
                "energy_budget": args.energy_budget,
                "load_budget": args.load_budget,
            }, f, indent=2, ensure_ascii=False)
        print(f"Best FSR metadata saved to: {best_fsr_meta_path}")

    # 保存 Best Tail Checkpoint
    if args.enable_best_checkpoint and best_tail_state is not None:
        best_tail_path = os.path.join(output_dir, "best_tail.pt")
        torch.save(best_tail_state, best_tail_path)
        print(f"Best tail model saved to: {best_tail_path}")
        print(
            f"  -> iter={best_tail_iter}, tail_score={best_tail_value:.4f}, "
            f"fsr={best_tail_fsr:.4f}, return={best_tail_return:.2f}"
        )

        best_tail_meta_path = os.path.join(output_dir, "best_tail_meta.json")
        with open(best_tail_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "iter": best_tail_iter,
                "tail_score": best_tail_value,
                "energy_p95": best_tail_energy_p95,
                "load_p95": best_tail_load_p95,
                "feasible_success_rate": best_tail_fsr,
                "avg_return": best_tail_return,
                "window": args.best_window_tail,
                "tail_percentile": args.tail_percentile,
                "energy_budget": args.energy_budget,
                "load_budget": args.load_budget,
            }, f, indent=2, ensure_ascii=False)
        print(f"Best tail metadata saved to: {best_tail_meta_path}")


if __name__ == "__main__":
    main()


# ========= 静态结构 + 全局观测（标准实验设置） =========
# python train_grid_structured_lagrangian.py \
#     --use_lagrange True \
#     --lambda_lr 0.01 \
#     --energy_budget 1.2 \
#     --load_budget 0.08 \
#     --grid_size 8 \
#     --output_dir outputs_global_standard
