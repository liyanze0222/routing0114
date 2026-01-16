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
from ppo_multi_agent import MultiCriticPPOConfig, MultiCriticPPO
from utils import set_seed, MetricsLogger, plot_training_curves, make_output_dir, save_grid_route_viz


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
        "--energy_high_cost", type=float, default=3.0,
        help="高功率干扰区的能耗（默认 3.0）"
    )
    parser.add_argument(
        "--energy_high_density", type=float, default=0.2,
        help="高能耗区域在地图中的比例（0~1）"
    )
    parser.add_argument(
        "--congestion_density", type=float, default=0.3,
        help="拥塞图平均拥堵程度（0~1）"
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
        "--energy_obs_normalize", type=lambda x: x.lower() == "true", default=True,
        help="energy patch 是否归一化到 [0,1]"
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

    # ========== 约束预算 ==========
    parser.add_argument(
        "--energy_budget", type=float, default=1.5,
        help="energy 成本预算（每步平均能耗上限）"
    )
    parser.add_argument(
        "--load_budget", type=float, default=0.08,
        help="load 成本预算（负载超限事件率上限）"
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
        choices=["separate", "shared"],
        help="E: Cost critic 模式 - 'separate'（默认，独立 value head）或 'shared'（共享 value head）"
    )

    # ========== B/C: Best Feasible Checkpoint & Early Stop ==========
    parser.add_argument(
        "--enable_best_checkpoint", type=lambda x: x.lower() == "true", default=True,
        help="B: 启用最优可行点存档（默认 True）"
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

    # ========== 可视化设置 ==========
    parser.add_argument("--vis_interval", type=int, default=0, help="每隔多少 iter 保存一次网格可视化(0=关闭)")
    parser.add_argument("--vis_start_iter", type=int, default=0, help="从第几次迭代开始保存可视化")
    parser.add_argument("--vis_seed", type=int, default=12345, help="可视化 eval 的 seed（固定场景）")
    parser.add_argument("--vis_episodes", type=int, default=1, help="每次保存可视化跑多少条 episode（不同 seed=vis_seed+ep）")
    parser.add_argument("--vis_deterministic", action="store_true", help="可视化用 greedy(argmax) 动作（不加则按策略采样）")

    return parser.parse_args()


def _build_env(args):
    """构建环境的辅助函数"""
    max_steps = None if args.max_steps < 0 else args.max_steps
    base_env = GridRoutingEnv(
        grid_size=args.grid_size,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        max_steps=max_steps,
    )
    env = GridCostWrapper(
        base_env,
        energy_base=1.0,
        energy_high_cost=args.energy_high_cost,
        energy_high_density=args.energy_high_density,
        congestion_density=args.congestion_density,
        congestion_pattern=args.congestion_pattern,
    )
    env = GridHardWrapper(env)
    if args.include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=args.congestion_patch_radius)
    if args.include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=args.energy_patch_radius, normalize=args.energy_obs_normalize)
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
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                if action_mask is not None:
                    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
                else:
                    mask_t = None
                logits, _, _ = agent.network.forward(obs_t, mask_t)
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action, _, _, _, _ = agent.select_action(obs, action_mask=action_mask)

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

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录（支持 run_tag 和自动递增）
    output_dir = make_output_dir(args.output_dir, args.run_tag)
    # 更新 args 以便后续保存 config.json 时记录最终目录
    args.final_output_dir = output_dir

    # ========== 构建环境 ==========
    env = _build_env(args)
    
    # ========== 构建独立的 eval_env 用于可视化 ==========
    eval_env = _build_env(args)

    obs_dim = env.observation_space.shape[0]
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
        f"Energy map: base=1.0, high_cost={args.energy_high_cost}, "
        f"density={args.energy_high_density}"
    )
    print(
        f"Load map: pattern={args.congestion_pattern}, "
        f"density={args.congestion_density}, value_range=[0, 1]"
    )
    cong_dim = (2 * args.congestion_patch_radius + 1) ** 2
    energy_dim = (2 * args.energy_patch_radius + 1) ** 2
    print("Observation: global coordinates + optional patches")
    print(
        f"  Congestion obs: {args.include_congestion_obs} "
        f"(radius={args.congestion_patch_radius}, dim={cong_dim})"
    )
    print(
        f"  Energy obs: {args.include_energy_obs} "
        f"(radius={args.energy_patch_radius}, normalize={args.energy_obs_normalize}, "
        f"dim={energy_dim})"
    )
    print(f"Observation dim: {obs_dim}")
    print(f"Mode: {'Lagrange (adaptive lambda)' if args.use_lagrange else 'Fixed lambda (penalty)'}")
    print(f"Budgets: energy={args.energy_budget:.3f}, load={args.load_budget:.3f}")
    print(f"Initial lambda: energy={args.initial_lambda_energy:.3f}, "
          f"load={args.initial_lambda_load:.3f}")
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
        # cost critic 模式
        cost_critic_mode=args.cost_critic_mode,
        device=device,
    )

    # ========== 创建 Agent ==========
    agent = MultiCriticPPO(config)

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

    # ========== C: Early Stop 状态 ==========
    gap_history: Dict[str, List[float]] = {"energy": [], "load": []}
    early_stopped = False
    early_stop_iter = None

    # ========== 训练循环 ==========
    for iteration in range(1, args.total_iters + 1):
        # 收集 rollout
        for _ in range(config.batch_size):
            # 选择动作
            action, log_prob, v_reward, v_costs = agent.select_action(
                obs, action_mask=action_mask
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
                obs=obs,
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
                
                # [新增] 保存 episode 信息到 buffer（用于 Safety Gym 风格评估）
                ep_info = {
                    'episode_return': ep_ret,
                    'episode_length': ep_len,
                    'success': terminated,
                    'episode_cost_energy': ep_cost_energy,
                    'episode_cost_load': ep_cost_load,
                }
                train_infos_buffer.append(ep_info)
                # 保留最近 100 个 episodes
                if len(train_infos_buffer) > 100:
                    train_infos_buffer = train_infos_buffer[-100:]

                ep_ret, ep_len = 0.0, 0
                ep_cost_energy, ep_cost_load = 0.0, 0.0
                # Seed replay：根据 mix_prob 决定是否从 seed_list 采样
                reset_seed = sample_reset_seed()
                if reset_seed is not None:
                    obs, info = env.reset(seed=reset_seed)
                else:
                    obs, info = env.reset()
                action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))

        # PPO 更新
        metrics = agent.update()

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
            # 新增：gap 和 KKT 残差
            "gap_energy": metrics.get("gap_energy", metrics["avg_cost_energy"] - args.energy_budget),
            "gap_load": metrics.get("gap_load", metrics["avg_cost_load"] - args.load_budget),
            "kkt_energy": metrics.get("kkt_energy", metrics["lambda_energy"] * (metrics["avg_cost_energy"] - args.energy_budget)),
            "kkt_load": metrics.get("kkt_load", metrics["lambda_load"] * (metrics["avg_cost_load"] - args.load_budget)),
        }
        if "gap_ratio_energy" in metrics:
            log_entry["gap_ratio_energy"] = metrics["gap_ratio_energy"]
        if "gap_ratio_load" in metrics:
            log_entry["gap_ratio_load"] = metrics["gap_ratio_load"]
        if "lambda_gap_mode" in metrics:
            log_entry["lambda_gap_mode"] = metrics["lambda_gap_mode"]
        # 添加 EMA gap（如果存在）
        if "ema_gap_energy" in metrics:
            log_entry["ema_gap_energy"] = metrics["ema_gap_energy"]
        if "ema_gap_load" in metrics:
            log_entry["ema_gap_load"] = metrics["ema_gap_load"]
        logger.log(log_entry)

        # ========== B: Best Feasible Checkpoint 逻辑 ==========
        if args.enable_best_checkpoint:
            gap_energy = log_entry["gap_energy"]
            gap_load = log_entry["gap_load"]
            # 可行条件：两个约束都满足（gap <= 0）
            is_feasible = (gap_energy <= 0) and (gap_load <= 0)
            if is_feasible and avg_return > best_feasible_return:
                best_feasible_return = avg_return
                best_feasible_iter = iteration
                # 保存 agent 状态的快照（深拷贝）
                import copy
                best_feasible_state = {
                    "network_state_dict": copy.deepcopy(agent.network.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(agent.optimizer.state_dict()),
                    "lambdas": copy.deepcopy(agent.lambdas),
                    "ema_gaps": copy.deepcopy(agent.ema_gaps),
                    "iter_count": agent._iter_count,
                    "best_return": best_feasible_return,
                    "best_iter": best_feasible_iter,
                    "gap_energy": gap_energy,
                    "gap_load": gap_load,
                }

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
            
            feasible_success_rate = feasible_success_count / len(train_infos_buffer) if train_infos_buffer else 0.0

            # 4. 写入到 log_entry (供 JSON 保存)
            # 注意：此时 log_entry 已在上面创建，这里需要更新它
            log_entry.update({
                "avg_energy_success": avg_energy_success,
                "avg_load_success": avg_load_success,
                "feasible_success_rate": feasible_success_rate,
                "feasible_success_count": feasible_success_count,
                "num_episodes_buffer": len(train_infos_buffer),
            })

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
            print("-" * 70)
            # [新增] Safety Gym 风格指标打印
            print(f"Success-only costs: Energy={avg_energy_success:.4f}, Load={avg_load_success:.4f}")
            print(f"Feasible & Success Rate: {feasible_success_rate:.2%} ({feasible_success_count}/{len(train_infos_buffer)})")
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

    # 保存模型（最终模型）
    if args.save_model:
        model_path = os.path.join(output_dir, "model.pt")
        agent.save(model_path)
        print(f"Model saved to: {model_path}")

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
