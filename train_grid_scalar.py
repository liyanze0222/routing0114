"""
train_grid_scalar.py

Training script for Scalar PPO (V5 Baseline for Ablation Study)

Key Difference from Multi-Head PPO:
- Scalarizes reward BEFORE passing to agent: scalar_reward = R - α*C_E - β*C_L
- Uses single value head (no separate cost critics)
- No Lagrangian updates

Usage:
python train_grid_scalar.py --energy_weight 0.5 --load_weight 2.0 --total_iters 800
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
from grid_obs_norm_wrapper import GridObsNormWrapper
from ppo_scalar import ScalarPPOConfig, ScalarPPO
from utils import set_seed, MetricsLogger, plot_training_curves, make_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scalar PPO (V5 Baseline) - Weighted Sum of Costs"
    )
    
    # ========== 基础设置 ==========
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--total_iters", type=int, default=800, help="总训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=2048, help="每次采样的步数")
    parser.add_argument("--minibatch_size", type=int, default=256, help="minibatch 大小")
    parser.add_argument("--update_epochs", type=int, default=10, help="每个 batch 更新轮数")
    
    # ========== 环境设置 ==========
    parser.add_argument("--grid_size", type=int, default=8, help="网格大小")
    parser.add_argument("--step_penalty", type=float, default=-1.0, help="每步基础惩罚")
    parser.add_argument("--max_steps", type=int, default=256, help="episode 最大步数")
    parser.add_argument("--success_reward", type=float, default=20.0, help="成功奖励")
    
    parser.add_argument("--energy_high_density", type=float, default=0.2, help="高能耗区域密度")
    parser.add_argument("--congestion_density", type=float, default=0.3, help="拥塞密度")
    parser.add_argument(
        "--congestion_pattern", type=str, default="random",
        choices=["random", "block"], help="拥塞模式"
    )
    parser.add_argument("--load_threshold", type=float, default=0.6, help="load cost 的软阈值 τ")
    parser.add_argument(
        "--randomize_maps_each_reset", type=lambda x: x.lower() == "true", default=True,
        help="是否在每次 reset 时重采样能耗/拥塞地图（训练默认 True）"
    )
    
    # ========== 观测增强 ==========
    parser.add_argument(
        "--include_congestion_obs", type=lambda x: x.lower() == "true", default=False,
        help="是否包含拥塞观测"
    )
    parser.add_argument("--congestion_patch_radius", type=int, default=2, help="拥塞 patch 半径")
    parser.add_argument(
        "--include_energy_obs", type=lambda x: x.lower() == "true", default=False,
        help="是否包含能耗观测"
    )
    parser.add_argument("--energy_patch_radius", type=int, default=1, help="能耗 patch 半径")
    parser.add_argument(
        "--obs_rms", type=lambda x: x.lower() == "true", default=False,
        help="是否对全局观测做 RunningMeanStd 归一化"
    )
    
    # ========== Cost Weights (关键参数) ==========
    parser.add_argument(
        "--energy_weight", type=float, default=0.0,
        help="Energy cost 权重 α (scalar_reward = R - α*C_E - β*C_L)"
    )
    parser.add_argument(
        "--load_weight", type=float, default=0.0,
        help="Load cost 权重 β (scalar_reward = R - α*C_E - β*C_L)"
    )

    # ========== 起终点设置（与结构化脚本对齐） ==========
    parser.add_argument(
        "--start_goal_mode", type=str, default="random", choices=["random", "rect"],
        help="起终点采样模式"
    )
    parser.add_argument("--start_rect", type=str, default=None, help="start_goal_mode=rect 时的起点矩形 r0,r1,c0,c1")
    parser.add_argument("--goal_rect", type=str, default=None, help="start_goal_mode=rect 时的终点矩形 r0,r1,c0,c1")

    # ========== 成本预算（与 Multi-Head 口径对齐） ==========
    parser.add_argument(
        "--energy_budget", type=float, default=0.10,
        help="energy 成本预算（per-step 平均能耗上限）"
    )
    parser.add_argument(
        "--load_budget", type=float, default=0.08,
        help="load 成本预算（per-step 平均负载上限）"
    )
    
    # ========== PPO 超参 ==========
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE λ")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip 系数")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="熵正则系数")
    parser.add_argument("--value_coef", type=float, default=0.5, help="value loss 系数")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪")
    
    # ========== 输出与日志 ==========
    parser.add_argument("--output_dir", type=str, default="scalar_ppo_output", help="输出目录")
    parser.add_argument("--run_tag", type=str, default=None, help="运行标签")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")

    # ========== 最优模型存档（简化版） ==========
    parser.add_argument(
        "--enable_best_checkpoint", type=lambda x: x.lower() == "true", default=True,
        help="是否启用最优可行/尾部风险模型存档"
    )
    parser.add_argument(
        "--best_checkpoint_success_thresh", type=float, default=0.95,
        help="Best checkpoint 生效所需的成功率下限"
    )
    parser.add_argument("--best_window_fsr", type=int, default=50, help="Best FSR 使用的窗口大小")
    parser.add_argument("--best_window_tail", type=int, default=50, help="Tail-score 使用的窗口大小")
    parser.add_argument("--tail_percentile", type=float, default=95.0, help="Tail-score 百分位")
    
    return parser.parse_args()


def _build_env(args):
    """构建环境"""
    max_steps = args.max_steps if args.max_steps > 0 else 4 * args.grid_size * args.grid_size
    
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
        env = GridEnergyObsWrapper(
            env, patch_radius=args.energy_patch_radius
        )
    if args.obs_rms:
        env = GridObsNormWrapper(env)
    return env


def _get_checkpoint_state(agent, meta=None):
    """构造与 agent.save 对齐的 checkpoint 结构，并可附带元信息。"""
    state = {
        "network_state_dict": agent.network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict() if hasattr(agent, "optimizer") else None,
    }
    if meta is not None:
        state["meta"] = meta
    return state


def _parse_rect(rect_str: str | None, name: str, grid_size: int):
    """Parse rect string "r0,r1,c0,c1" into a validated tuple within grid bounds."""
    if rect_str is None:
        return None
    parts = [p.strip() for p in rect_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"{name} must be 'r0,r1,c0,c1', got: {rect_str}")
    try:
        r0, r1, c0, c1 = map(int, parts)
    except ValueError as e:
        raise ValueError(f"{name} must contain four integers: {rect_str}") from e
    if not (0 <= r0 <= r1 < grid_size and 0 <= c0 <= c1 < grid_size):
        raise ValueError(
            f"{name} out of bounds for grid_size={grid_size}: {(r0, r1, c0, c1)}"
        )
    return (r0, r1, c0, c1)


def main():
    args = parse_args()

    # 解析 rect 参数（若提供）
    try:
        args.start_rect = _parse_rect(args.start_rect, "start_rect", args.grid_size)
        args.goal_rect = _parse_rect(args.goal_rect, "goal_rect", args.grid_size)
    except ValueError as e:
        raise SystemExit(str(e))
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = make_output_dir(args.output_dir, args.run_tag)
    print(f"Output directory: {output_dir}")
    
    # 保存配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # 构建环境
    env = _build_env(args)
    
    # 获取观测/动作维度
    obs_sample, info = env.reset(seed=args.seed)
    obs_dim = obs_sample.shape[0] if hasattr(obs_sample, 'shape') else len(obs_sample)
    act_dim = 4
    action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
    
    print(f"\n========== Environment Info ==========")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Grid size: {args.grid_size}")
    print(f"Max steps: {args.max_steps if args.max_steps > 0 else 4 * args.grid_size ** 2}")
    print(f"Energy weight α: {args.energy_weight}")
    print(f"Load weight β: {args.load_weight}")
    print(f"Load threshold: {args.load_threshold}")
    print("=" * 40 + "\n")
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # 构建配置
    config = ScalarPPOConfig(
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
        device=device,
    )
    
    # 创建 Agent
    agent = ScalarPPO(config)
    
    # 初始化 Logger
    logger = MetricsLogger()
    
    # 训练状态
    episode_returns: List[float] = []
    episode_scalar_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []
    
    # Episode 统计（用于计算 avg_cost）
    episode_energy_costs: List[float] = []
    episode_load_costs: List[float] = []

    # 训练信息缓冲（用于窗口指标和最优 checkpoint 判定）
    train_infos_buffer: List[Dict[str, Any]] = []

    # 最优 checkpoint 追踪
    best_fsr_value = -float("inf")
    best_fsr_iter = None
    best_fsr_return = -float("inf")
    best_tail_value = float("inf")
    best_tail_iter = None
    best_tail_return = -float("inf")
    best_tail_fsr = -float("inf")
    
    has_budget = (
        args.energy_budget is not None and args.load_budget is not None
        and args.energy_budget > 0 and args.load_budget > 0
    )
    
    # 初始 reset
    obs, info = env.reset(seed=args.seed)
    action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
    current_map_fingerprint = info.get("map_fingerprint")
    
    # 训练循环
    ep_ret = 0.0
    ep_scalar_ret = 0.0
    ep_len = 0
    ep_energy = 0.0
    ep_load = 0.0
    
    start_time = time.time()
    
    print("=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    for iteration in range(1, args.total_iters + 1):
        map_fingerprints_iter: List[str] = []
        if current_map_fingerprint is not None:
            map_fingerprints_iter.append(current_map_fingerprint)
        # Rollout
        for _ in range(config.batch_size):
            # 选择动作
            action, log_prob, value = agent.get_action(obs, action_mask)
            
            # 环境交互
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 提取 cost components
            cost_components = info.get("cost_components", {})
            energy_cost = cost_components.get("energy", 0.0)
            load_cost = cost_components.get("load", 0.0)
            
            # *** 关键：Scalarize Reward ***
            scalar_reward = env_reward - args.energy_weight * energy_cost - args.load_weight * load_cost
            ep_scalar_ret += scalar_reward
            
            # 收集数据（使用标量化的 reward）
            agent.collect_rollout(
                obs=obs,
                action=action,
                reward=scalar_reward,
                done=done,
                log_prob=log_prob,
                action_mask=action_mask,
                value=value,
            )
            
            # 更新 episode 统计
            ep_ret += env_reward  # 记录原始 reward
            ep_len += 1
            ep_energy += energy_cost
            ep_load += load_cost
            
            # 更新状态
            obs = next_obs
            action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
            
            if done:
                # Episode 结束
                episode_returns.append(ep_ret)
                episode_scalar_returns.append(ep_scalar_ret)
                episode_lengths.append(ep_len)
                episode_successes.append(terminated)
                episode_energy_costs.append(ep_energy)
                episode_load_costs.append(ep_load)

                train_infos_buffer.append({
                    "episode_return": ep_ret,
                    "episode_scalar_return": ep_scalar_ret,
                    "episode_length": ep_len,
                    "episode_cost_energy": ep_energy,
                    "episode_cost_load": ep_load,
                    "success": bool(terminated),
                })
                
                # Reset
                obs, info = env.reset()
                action_mask = info.get("action_mask", np.ones(act_dim, dtype=bool))
                current_map_fingerprint = info.get("map_fingerprint", current_map_fingerprint)
                if current_map_fingerprint is not None:
                    map_fingerprints_iter.append(current_map_fingerprint)
                ep_ret = 0.0
                ep_scalar_ret = 0.0
                ep_len = 0
                ep_energy = 0.0
                ep_load = 0.0
        
        env_map_changes = len(set(map_fingerprints_iter)) if map_fingerprints_iter else 0
        last_map_fingerprint = map_fingerprints_iter[-1] if map_fingerprints_iter else None

        # 计算 GAE
        advantages, returns = agent.compute_gae(obs, False, action_mask)
        
        # 更新策略
        metrics = agent.update(advantages, returns)
        
        # 日志记录
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            total_steps = iteration * config.batch_size
            
            # 计算统计量
            avg_return = np.mean(episode_returns[-100:]) if episode_returns else 0.0
            avg_scalar_return = np.mean(episode_scalar_returns[-100:]) if episode_scalar_returns else 0.0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
            success_rate = np.mean(episode_successes[-100:]) if episode_successes else 0.0
            
            # *** 关键：计算 per-step mean cost（与 Multi-Head 口径一致）***
            avg_cost_energy = 0.0
            avg_cost_load = 0.0
            if len(episode_energy_costs) > 0 and len(episode_lengths) > 0:
                # 取最近 100 个 episode
                recent_energy = episode_energy_costs[-100:]
                recent_load = episode_load_costs[-100:]
                recent_lengths = episode_lengths[-100:]
                
                # Per-step mean
                avg_cost_energy = np.mean([e / max(1, l) for e, l in zip(recent_energy, recent_lengths)])
                avg_cost_load = np.mean([ld / max(1, l) for ld, l in zip(recent_load, recent_lengths)])

            gap_energy = avg_cost_energy - args.energy_budget if has_budget else None
            gap_load = avg_cost_load - args.load_budget if has_budget else None
            
            # 构造日志
            log_entry = {
                "iteration": iteration,
                "total_steps": total_steps,
                "elapsed_time": elapsed,
                "avg_return": float(avg_return),
                "avg_scalar_return": float(avg_scalar_return),
                "avg_length": float(avg_length),
                "success_rate": float(success_rate),
                "avg_cost_energy": float(avg_cost_energy),
                "avg_cost_load": float(avg_cost_load),
                "budget_energy": args.energy_budget,
                "budget_load": args.load_budget,
                "gap_energy": float(gap_energy) if gap_energy is not None else None,
                "gap_load": float(gap_load) if gap_load is not None else None,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "approx_kl": metrics.get("approx_kl", 0.0),
                "energy_weight": args.energy_weight,
                "load_weight": args.load_weight,
                "env_map_changes": env_map_changes,
            }

            if last_map_fingerprint is not None:
                log_entry["env_map_fingerprint"] = last_map_fingerprint

            for key in [
                "actor_param_delta_l2",
                "actor_param_norm_l2",
                "actor_param_delta_ratio",
            ]:
                if key in metrics:
                    log_entry[key] = metrics[key]

            # ========== 最优 checkpoint 判定（简化版） ==========
            energy_pxx = None
            load_pxx = None
            tail_score = None
            feasible_success_rate_window = None

            if args.enable_best_checkpoint and has_budget and train_infos_buffer:
                # 最近窗口的可行成功率（FSR）
                window_fsr = args.best_window_fsr
                episodes_window = train_infos_buffer if window_fsr <= 0 else train_infos_buffer[-window_fsr:]
                feasible_success_count = 0
                for ep in episodes_window:
                    if not ep.get("success", False):
                        continue
                    ep_len = max(1, ep.get("episode_length", 1))
                    e_mean = ep.get("episode_cost_energy", 0.0) / ep_len
                    l_mean = ep.get("episode_cost_load", 0.0) / ep_len
                    if e_mean <= args.energy_budget and l_mean <= args.load_budget:
                        feasible_success_count += 1
                feasible_success_rate_window = feasible_success_count / max(1, len(episodes_window))
                window_returns = [ep.get("episode_return", 0.0) for ep in episodes_window]
                avg_return_window = float(np.mean(window_returns)) if window_returns else 0.0

                # 尾部风险指标（成功 episode 的 per-step cost 百分位）
                tail_k = args.best_window_tail
                success_eps = [ep for ep in train_infos_buffer if ep.get("success", False)]
                if tail_k > 0:
                    success_eps = success_eps[-tail_k:]
                energy_rates = [ep.get("episode_cost_energy", 0.0) / max(1, ep.get("episode_length", 1)) for ep in success_eps]
                load_rates = [ep.get("episode_cost_load", 0.0) / max(1, ep.get("episode_length", 1)) for ep in success_eps]
                if energy_rates:
                    energy_pxx = float(np.percentile(energy_rates, args.tail_percentile))
                if load_rates:
                    load_pxx = float(np.percentile(load_rates, args.tail_percentile))
                if energy_pxx is not None and load_pxx is not None:
                    tail_score = max(energy_pxx / args.energy_budget, load_pxx / args.load_budget)

                success_reliable = success_rate >= args.best_checkpoint_success_thresh

                # Best FSR
                fsr_improved = (
                    feasible_success_rate_window is not None
                    and (
                        feasible_success_rate_window > best_fsr_value + 1e-6
                        or (
                            abs(feasible_success_rate_window - best_fsr_value) < 1e-6
                            and avg_return_window > best_fsr_return + 1e-6
                        )
                    )
                )
                if fsr_improved and success_reliable:
                    best_fsr_value = feasible_success_rate_window
                    best_fsr_iter = iteration
                    best_fsr_return = avg_return_window
                    torch.save(
                        _get_checkpoint_state(
                            agent,
                            meta={
                                "best_iter": best_fsr_iter,
                                "best_fsr": best_fsr_value,
                                "avg_return_window": best_fsr_return,
                                "feasible_success_rate_window": feasible_success_rate_window,
                                "tail_score": tail_score,
                            },
                        ),
                        os.path.join(output_dir, "best_fsr.pt"),
                    )
                    print(f"[Best FSR] Updated at iter {iteration}: fsr={best_fsr_value:.4f}, return={best_fsr_return:.3f}")

                # Best tail
                if tail_score is not None:
                    tail_improved = (
                        tail_score < best_tail_value - 1e-6
                        or (
                            abs(tail_score - best_tail_value) < 1e-6
                            and feasible_success_rate_window is not None
                            and feasible_success_rate_window > best_tail_fsr + 1e-6
                        )
                        or (
                            abs(tail_score - best_tail_value) < 1e-6
                            and feasible_success_rate_window is not None
                            and abs(feasible_success_rate_window - best_tail_fsr) < 1e-6
                            and avg_return_window > best_tail_return + 1e-6
                        )
                    )
                    if tail_improved and success_reliable:
                        best_tail_value = tail_score
                        best_tail_iter = iteration
                        best_tail_return = avg_return_window
                        best_tail_fsr = feasible_success_rate_window if feasible_success_rate_window is not None else best_tail_fsr
                        fsr_display = feasible_success_rate_window if feasible_success_rate_window is not None else 0.0
                        torch.save(
                            _get_checkpoint_state(
                                agent,
                                meta={
                                    "best_iter": best_tail_iter,
                                    "best_tail_score": best_tail_value,
                                    "avg_return_window": best_tail_return,
                                    "feasible_success_rate_window": feasible_success_rate_window,
                                    "energy_percentile": energy_pxx,
                                    "load_percentile": load_pxx,
                                },
                            ),
                            os.path.join(output_dir, "best_tail.pt"),
                        )
                        print(
                            f"[Best Tail] Updated at iter {iteration}: tail_score={best_tail_value:.4f}, fsr={fsr_display:.4f}"
                        )

            log_entry["energy_percentile"] = energy_pxx
            log_entry["load_percentile"] = load_pxx
            log_entry["tail_score"] = tail_score
            log_entry["feasible_success_rate_window"] = feasible_success_rate_window
            log_entry["best_fsr_iter"] = best_fsr_iter
            log_entry["best_tail_iter"] = best_tail_iter
            
            logger.log(log_entry)
            
            # 打印
            print("=" * 70)
            print(f"Iteration {iteration}/{args.total_iters} | Steps: {total_steps} | Time: {elapsed:.1f}s")
            print(f"Avg Return: {avg_return:.2f} | Avg Length: {avg_length:.1f} | Success Rate: {success_rate:.2%}")
            if gap_energy is not None and gap_load is not None:
                print(
                    f"Cost (energy): {avg_cost_energy:.4f} | Cost (load): {avg_cost_load:.4f} | "
                    f"Gap: energy {gap_energy:+.4f}, load {gap_load:+.4f}"
                )
                print(
                    f"Budget (per-step): energy {args.energy_budget:.4f} | load {args.load_budget:.4f}"
                )
            else:
                print(f"Cost (energy): {avg_cost_energy:.4f} | Cost (load): {avg_cost_load:.4f}")
            print(f"Weights: α={args.energy_weight:.2f}, β={args.load_weight:.2f}")
            print("-" * 70)
            print(f"Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
            print("=" * 70 + "\n")
    
    # 保存日志
    logger.save(os.path.join(output_dir, "metrics.json"))
    print(f"Metrics saved to: {os.path.join(output_dir, 'metrics.json')}")
    
    # 绘制曲线
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_kwargs = {"best_iter": best_fsr_iter} if best_fsr_iter is not None else {}
    plot_training_curves(logger.get_data(), plot_path, **plot_kwargs)
    print(f"Training curves saved to: {plot_path}")
    
    # 保存模型
    if args.save_model:
        model_path = os.path.join(output_dir, "model.pt")
        agent.save(model_path)
        print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
