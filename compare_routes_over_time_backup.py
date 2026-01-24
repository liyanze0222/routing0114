"""
compare_routes_over_time.py

对比同一个 run_dir 中不同时期 checkpoint 在相同起终点下的路径选择。

用法示例：
python compare_routes_over_time.py \\
    --run_dir outputs/four_group_ablation_20260121/A_multi_critic_adaptive_seed0_v2 \\
    --seed 0 \\
    --episodes 3 \\
    --deterministic True
"""

import os
# 解决 Windows + Anaconda 下 OpenMP 库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# 复用 visualize_rollout.py 的函数
from visualize_rollout import (
    load_config_from_dir,
    inject_obs_stats,
    load_agent,
    select_action,
    _get_action_mask,
    find_cost_wrapper,
)

# 需要手动实现 make_env_from_config 以确保使用完整配置
from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper


def step_env(env, action):
    """
    兼容 gym / gymnasium 的 step() 返回签名。
    
    Returns:
        obs, reward, done, info
    """
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        info.setdefault("terminated", done)
        info.setdefault("truncated", False)
    else:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        info.setdefault("terminated", terminated)
        info.setdefault("truncated", truncated)
    return obs, reward, done, info


def make_env_from_config(cfg: dict, seed: int):
    """
    根据完整配置构建环境（强制读取所有训练参数）。
    
    Args:
        cfg: 从 config.json 加载的配置字典
        seed: 环境种子
    
    Returns:
        构建好的环境
    """
    # 基础参数
    grid_size = cfg.get("grid_size", 8)
    step_penalty = cfg.get("step_penalty", -1.0)
    success_reward = cfg.get("success_reward", 20.0)
    max_steps = cfg.get("max_steps", 256)
    
    # 起终点配置（必须与训练一致）
    start_goal_mode = cfg.get("start_goal_mode", "random")
    start_rect = cfg.get("start_rect", None)
    goal_rect = cfg.get("goal_rect", None)
    
    # Cost 配置
    congestion_pattern = cfg.get("congestion_pattern", "block")
    congestion_density = cfg.get("congestion_density", 0.40)
    energy_high_cost = cfg.get("energy_high_cost", 3.0)
    energy_high_density = cfg.get("energy_high_density", 0.20)
    load_cost_scale = cfg.get("load_cost_scale", 1.0)
    
    # 观测配置
    include_congestion_obs = cfg.get("include_congestion_obs", True)
    congestion_patch_radius = cfg.get("congestion_patch_radius", 2)
    include_energy_obs = cfg.get("include_energy_obs", True)
    energy_patch_radius = cfg.get("energy_patch_radius", 2)
    energy_obs_normalize = cfg.get("energy_obs_normalize", True)
    obs_rms = cfg.get("obs_rms", False)
    
    # 构建环境
    env = GridRoutingEnv(
        grid_size=grid_size,
        step_penalty=step_penalty,
        success_reward=success_reward,
        max_steps=max_steps,
    )
    
    # 设置起终点模式
    if start_goal_mode == "rect" and start_rect is not None and goal_rect is not None:
        # 解析 rect 配置（格式："r1,r2,c1,c2"）
        if isinstance(start_rect, str):
            start_rect = [int(x) for x in start_rect.split(",")]
        if isinstance(goal_rect, str):
            goal_rect = [int(x) for x in goal_rect.split(",")]
        
        env.start_goal_mode = "rect"
        env.start_rect = tuple(start_rect)
        env.goal_rect = tuple(goal_rect)
    else:
        env.start_goal_mode = start_goal_mode
    
    env = GridCostWrapper(
        env,
        congestion_pattern=congestion_pattern,
        congestion_density=congestion_density,
        energy_high_cost=energy_high_cost,
        energy_high_density=energy_high_density,
        load_cost_scale=load_cost_scale,
    )
    
    # Hard wrapper 在 obs wrapper 之前
    env = GridHardWrapper(env)
    
    if include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
    
    if include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius, normalize=energy_obs_normalize)
    
    if obs_rms:
        env = GridObsNormWrapper(env)
    
    env.reset(seed=seed)
    return env


def find_checkpoints(run_dir: str, ckpt_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    查找 run_dir 中存在的 checkpoint 文件。
    
    Args:
        run_dir: 运行目录路径
        ckpt_names: 要查找的 checkpoint 文件名列表（默认：best_fsr.pt, best_feasible.pt, best_tail.pt）
    
    Returns:
        存在的 checkpoint 字典 {name: path}
    """
    if ckpt_names is None:
        ckpt_names = ["best_fsr.pt", "best_feasible.pt", "best_tail.pt", "checkpoint_final.pt"]
    
    found = {}
    for name in ckpt_names:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            found[name] = path
        else:
            print(f"⚠️  Checkpoint not found: {name}")
    
    if not found:
        print(f"❌ No checkpoints found in {run_dir}")
    
    return found


def rollout_single_episode(
    env,
    agent,
    seed: int,
    deterministic: bool,
    device: str,
    max_steps_cfg: int,
) -> Dict:
    """
    执行单次 rollout，返回轨迹和统计信息。
    
    Returns:
        {
            "traj": [(r, c), ...],
            "start": (r, c),
            "goal": (r, c),
            "success": bool,
            "reached_goal": bool,
            "terminated": bool,
            "truncated": bool,
            "episode_length": int,
            "total_energy": float,
            "total_load": float,
            "mean_energy": float,
            "mean_load": float,
        }
    """
    obs, info = env.reset(seed=seed)
    
    # 提取起终点
    start_rc = info.get("start") or info.get("start_pos")
    goal_rc = info.get("goal") or info.get("goal_pos")
    if start_rc is None:
        start_rc = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    if goal_rc is None:
        goal_rc = (env.unwrapped.goal_row, env.unwrapped.goal_col)
    
    traj = [(env.unwrapped.agent_row, env.unwrapped.agent_col)]
    total_energy = 0.0
    total_load = 0.0
    step_count = 0
    
    done = False
    
    while not done:
        mask = _get_action_mask(env)
        action = select_action(agent, obs, mask, deterministic, device)
        obs, reward, done, info = step_env(env, action)
        
        # 记录位置
        traj.append((env.unwrapped.agent_row, env.unwrapped.agent_col))
        
        # 累计 cost
        cost_components = info.get("cost_components", {})
        total_energy += cost_components.get("energy", 0.0)
        total_load += cost_components.get("load", 0.0)
        step_count += 1
    
    # 提取终止信息
    success = info.get("success", False)
    reached_goal = info.get("reached_goal", success)
    terminated = info.get("terminated", False)
    truncated = info.get("truncated", False)
    episode_length = len(traj) - 1  # 不含起点
    
    # Sanity check: 检测可能的 done 处理问题
    if episode_length >= max_steps_cfg and not success:
        print(f"  ⚠️  WARNING: Episode hit max_steps ({max_steps_cfg}) without success!")
        print(f"      Check: 1) done handling, 2) start/goal mismatch, 3) policy quality")
    
    return {
        "traj": traj,
        "start": start_rc,
        "goal": goal_rc,
        "success": success,
        "reached_goal": reached_goal,
        "terminated": terminated,
        "truncated": truncated,
        "episode_length": episode_length,
        "total_energy": total_energy,
        "total_load": total_load,
        "mean_energy": total_energy / max(1, step_count),
        "mean_load": total_load / max(1, step_count),
    }


def plot_single_checkpoint_routes(
    congestion_map: np.ndarray,
    energy_map: np.ndarray,
    rollout_results: List[Dict],
    ckpt_name: str,
    save_path: str,
):
    """
    为单个 checkpoint 的多条轨迹绘制对比图（2子图：左拥堵右能耗）。
    
    Args:
        congestion_map: 拥堵热力图
        energy_map: 能耗热力图
        rollout_results: rollout 结果列表
        ckpt_name: checkpoint 名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 提取共同的起终点（假设所有 episode 相同）
    if rollout_results:
        start = rollout_results[0]["start"]
        goal = rollout_results[0]["goal"]
    else:
        start = goal = (0, 0)
    
    for ax, grid, name in [
        (axes[0], congestion_map, "Load / Congestion"),
        (axes[1], energy_map, "Energy"),
    ]:
        im = ax.imshow(grid, origin="upper", cmap="YlOrRd", alpha=0.7)
        ax.set_title(name)
        
        # 绘制所有轨迹
        colors = plt.cm.tab10(np.linspace(0, 1, len(rollout_results)))
        for i, result in enumerate(rollout_results):
            traj = result["traj"]
            xs = [c for r, c in traj]
            ys = [r for r, c in traj]
            
            label = (f"Ep{i+1}: len={result['episode_length']}, "
                     f"{'✓' if result['success'] else '✗'}, "
                     f"E={result['mean_energy']:.2f}, L={result['mean_load']:.2f}")
            
            # 添加终止状态标记
            status_mark = ""
            if result["truncated"]:
                status_mark = " (TRUNC)"
            elif not result["reached_goal"]:
                status_mark = " (NO-GOAL)"
            
            label += status_mark
            ax.plot(xs, ys, linewidth=2, alpha=0.8, color=colors[i], label=label)
        
        # 标记起终点
        ax.scatter([start[1]], [start[0]], marker="s", s=100, color="green", 
                   edgecolors="black", linewidths=2, label="Start", zorder=10)
        ax.scatter([goal[1]], [goal[0]], marker="*", s=200, color="gold",
                   edgecolors="black", linewidths=2, label="Goal", zorder=10)
        
        # 美化
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"Checkpoint: {ckpt_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def plot_comparison_grid(
    congestion_map: np.ndarray,
    energy_map: np.ndarray,
    all_results: Dict[str, List[Dict]],
    save_path: str,
):
    """
    生成总览对比图：每行一个 checkpoint，每行2子图（左拥堵右能耗）。
    
    Args:
        congestion_map: 拥堵热力图
        energy_map: 能耗热力图
        all_results: {ckpt_name: [rollout_result, ...], ...}
        save_path: 保存路径
    """
    n_ckpts = len(all_results)
    if n_ckpts == 0:
        print("⚠️  No results to plot.")
        return
    
    fig, axes = plt.subplots(n_ckpts, 2, figsize=(14, 6 * n_ckpts))
    
    # 确保 axes 是二维数组
    if n_ckpts == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (ckpt_name, rollout_results) in enumerate(all_results.items()):
        if not rollout_results:
            continue
        
        # 提取起终点
        start = rollout_results[0]["start"]
        goal = rollout_results[0]["goal"]
        
        for col_idx, (grid, name) in enumerate([
            (congestion_map, "Load / Congestion"),
            (energy_map, "Energy"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(grid, origin="upper", cmap="YlOrRd", alpha=0.7)
            
            # 绘制所有轨迹
            colors = plt.cm.tab10(np.linspace(0, 1, len(rollout_results)))
            for i, result in enumerate(rollout_results):
                traj = result["traj"]
                xs = [c for r, c in traj]
                ys = [r for r, c in traj]
                
                status_mark = ""
                if result["truncated"]:
                    status_mark = " (TRUNC)"
                elif not result["reached_goal"]:
                    status_mark = " (NO-GOAL)"
                
                label = (f"Ep{i+1}: len={result['episode_length']}"
                         f"{status_mark}")
                ax.plot(xs, ys, linewidth=2, alpha=0.8, color=colors[i], label=label)
            
            # 标记起终点
            ax.scatter([start[1]], [start[0]], marker="s", s=100, color="green",
                       edgecolors="black", linewidths=2, zorder=10)
            ax.scatter([goal[1]], [goal[0]], marker="*", s=200, color="gold",
                       edgecolors="black", linewidths=2, zorder=10)
            
            # 美化
            title = f"{ckpt_name} - {name}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            ax.grid(True, linewidth=0.5, alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle("Route Comparison Across Checkpoints", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def compute_visitmap_stats(
    env,
    agent,
    seed_start: int,
    n_episodes: int,
    deterministic: bool,
    device: str,
    max_steps_cfg: int,
    grid_size: int,
    cell_of_interest: Tuple[int, int],
) -> Dict:
    """
    统计多个 episode 的访问频率。
    
    Returns:
        {
            "visit_count": np.ndarray[grid_size, grid_size],
            "visit_prob": np.ndarray[grid_size, grid_size],
            "hit_rate_cell": float,
            "success_rate": float,
            "avg_len_success": float,
            "num_success": int,
            "num_episodes": int,
        }
    """
    visit_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    hit_count = 0
    success_count = 0
    total_len_success = 0
    no_goal_count = 0
    
    for ep in range(n_episodes):
        result = rollout_single_episode(
            env, agent, seed_start + ep, deterministic, device, max_steps_cfg
        )
        
        # 统计访问
        visited_cells = set()
        for r, c in result["traj"]:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                visit_count[r, c] += 1
                visited_cells.add((r, c))
        
        # 检查是否访问过 cell_of_interest
        if cell_of_interest in visited_cells:
            hit_count += 1
        
        # 统计成功率
        if result["success"]:
            success_count += 1
            total_len_success += result["episode_length"]
        
        # 检查 NO-GOAL 情况
        if not result["reached_goal"]:
            no_goal_count += 1
    
    # Sanity check: 所有 episode 都没到达 goal
    if no_goal_count == n_episodes:
        raise RuntimeError(
            f"❌ CRITICAL: All {n_episodes} episodes failed to reach goal!\n"
            f"   This indicates:\n"
            f"   1) Goal detection broken (check info['success'] or goal_pos logic)\n"
            f"   2) Config mismatch (start_rect/goal_rect不一致)\n"
            f"   3) Policy completely broken\n"
            f"   Check config.json and environment construction."
        )
    
    # 计算统计量
    visit_prob = visit_count.astype(np.float32) / max(1, n_episodes)
    hit_rate_cell = hit_count / max(1, n_episodes)
    success_rate = success_count / max(1, n_episodes)
    avg_len_success = total_len_success / max(1, success_count) if success_count > 0 else 0.0
    
    return {
        "visit_count": visit_count,
        "visit_prob": visit_prob,
        "hit_rate_cell": hit_rate_cell,
        "success_rate": success_rate,
        "avg_len_success": avg_len_success,
        "num_success": success_count,
        "num_episodes": n_episodes,
    }


def plot_visitmap_single(
    visit_prob: np.ndarray,
    ckpt_name: str,
    cell_of_interest: Tuple[int, int],
    hit_rate: float,
    success_rate: float,
    avg_len: float,
    save_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    绘制单个 checkpoint 的访问频率热图。
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(visit_prob, origin="upper", cmap="hot", vmin=vmin, vmax=vmax)
    
    # 标注 cell_of_interest
    ax.scatter([cell_of_interest[1]], [cell_of_interest[0]], 
               marker="x", s=200, color="cyan", linewidths=3, label=f"Cell {cell_of_interest}")
    
    # 标题
    title = (f"{ckpt_name}\n"
             f"Hit Rate@{cell_of_interest}: {hit_rate:.2%} | "
             f"Success: {success_rate:.2%} | Avg Len: {avg_len:.1f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    # 美化
    ax.set_xticks(range(visit_prob.shape[1]))
    ax.set_yticks(range(visit_prob.shape[0]))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Visit Probability")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def plot_visitmap_diff(
    visit_prob_a: np.ndarray,
    visit_prob_b: np.ndarray,
    ckpt_name_a: str,
    ckpt_name_b: str,
    cell_of_interest: Tuple[int, int],
    save_path: str,
):
    """
    绘制两个 checkpoint 的访问概率差分图。
    """
    diff = visit_prob_a - visit_prob_b
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 使用发散色标
    vmax = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    
    # 标注 cell_of_interest
    ax.scatter([cell_of_interest[1]], [cell_of_interest[0]], 
               marker="x", s=200, color="yellow", linewidths=3, label=f"Cell {cell_of_interest}")
    
    # 标题
    title = f"Visit Prob Diff: {ckpt_name_a} - {ckpt_name_b}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    # 美化
    ax.set_xticks(range(diff.shape[1]))
    ax.set_yticks(range(diff.shape[0]))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Visit Prob")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def run_visitmap_analysis(args, deterministic: bool, cell_of_interest: Tuple[int, int]):
    """
    运行访问频率分析模式。
    """
    # 查找可用 checkpoint
    checkpoints = find_checkpoints(args.run_dir, args.checkpoints)
    if not checkpoints:
        print("❌ No valid checkpoints found. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"VISITMAP ANALYSIS MODE")
    print(f"Run Directory: {args.run_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): {list(checkpoints.keys())}")
    print(f"Episodes: {args.episodes} | Deterministic: {deterministic}")
    print(f"Cell of Interest: {cell_of_interest}")
    print(f"{'='*70}\n")
    
    # 加载配置
    cfg_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"❌ config.json not found in {args.run_dir}")
        return
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    max_steps_cfg = cfg.get("max_steps", 256)
    grid_size = cfg.get("grid_size", 8)
    
    print(f"\nConfig: grid_size={grid_size}, "
          f"max_steps={max_steps_cfg}, "
          f"start_goal_mode={cfg.get('start_goal_mode', 'random')}")
    print(f"Start rect: {cfg.get('start_rect', 'N/A')}, Goal rect: {cfg.get('goal_rect', 'N/A')}\n")
    
    # 创建环境（用于获取地图）
    env = make_env_from_config(cfg, args.seed)
    cost_wrapper = find_cost_wrapper(env)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 为每个 checkpoint 统计访问频率
    all_stats = {}
    
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n--- Processing: {ckpt_name} ---")
        
        # 加载 checkpoint 和 agent
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        agent, _ = load_agent(ckpt_path, obs_dim, act_dim, args.device, checkpoint)
        
        # 重新创建环境并注入 obs_stats
        env = make_env_from_config(cfg, args.seed)
        inject_obs_stats(env, checkpoint, cfg)
        
        # 统计访问频率
        stats = compute_visitmap_stats(
            env, agent, args.seed, args.episodes, deterministic,
            args.device, max_steps_cfg, grid_size, cell_of_interest
        )
        
        all_stats[ckpt_name] = stats
        
        # 打印统计信息
        print(f"  Success Rate: {stats['success_rate']:.2%} ({stats['num_success']}/{stats['num_episodes']})")
        print(f"  Avg Length (Success): {stats['avg_len_success']:.1f}")
        print(f"  Hit Rate @ {cell_of_interest}: {stats['hit_rate_cell']:.2%}")
        
        # 保存单个 checkpoint 的热图
        single_save_path = os.path.join(
            args.run_dir,
            f"{ckpt_name.replace('.pt', '')}_visitmap.png"
        )
        
        # 统一 colormap 范围（第一次遍历确定全局 vmax）
        vmax = max(stats['visit_prob'].max() for stats in all_stats.values())
        
        plot_visitmap_single(
            stats['visit_prob'],
            ckpt_name,
            cell_of_interest,
            stats['hit_rate_cell'],
            stats['success_rate'],
            stats['avg_len_success'],
            single_save_path,
            vmin=0.0,
            vmax=vmax,
        )
    
    # 重新绘制所有热图（使用统一的 vmax）
    vmax_global = max(stats['visit_prob'].max() for stats in all_stats.values())
    for ckpt_name, stats in all_stats.items():
        single_save_path = os.path.join(
            args.run_dir,
            f"{ckpt_name.replace('.pt', '')}_visitmap.png"
        )
        plot_visitmap_single(
            stats['visit_prob'],
            ckpt_name,
            cell_of_interest,
            stats['hit_rate_cell'],
            stats['success_rate'],
            stats['avg_len_success'],
            single_save_path,
            vmin=0.0,
            vmax=vmax_global,
        )
    
    # 生成差分图（如果有 best_tail 和 best_fsr）
    if "best_tail.pt" in all_stats and "best_fsr.pt" in all_stats:
        diff_save_path = os.path.join(args.run_dir, "visitmap_diff_tail_vs_fsr.png")
        plot_visitmap_diff(
            all_stats["best_tail.pt"]["visit_prob"],
            all_stats["best_fsr.pt"]["visit_prob"],
            "best_tail.pt",
            "best_fsr.pt",
            cell_of_interest,
            diff_save_path,
        )
    
    print(f"\n{'='*70}")
    print(f"✅ Visitmap analysis complete!")
    print(f"   Output directory: {args.run_dir}")
    print(f"{'='*70}\n")


def run_route_comparison(args, deterministic: bool):
    """
    运行路径对比模式（原始功能）。
    """
    # 查找可用 checkpoint
    checkpoints = find_checkpoints(args.run_dir, args.checkpoints)
    if not checkpoints:
        print("❌ No valid checkpoints found. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"Run Directory: {args.run_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): {list(checkpoints.keys())}")
    print(f"Seed: {args.seed} | Episodes: {args.episodes} | Deterministic: {deterministic}")
    print(f"{'='*70}\n")
    
    # 加载配置
    cfg_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"❌ config.json not found in {args.run_dir}")
        return
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    # 提取 max_steps 用于 sanity check
    max_steps_cfg = cfg.get("max_steps", 256)
    
    print(f"\nConfig: grid_size={cfg.get('grid_size', 8)}, "
          f"max_steps={max_steps_cfg}, "
          f"start_goal_mode={cfg.get('start_goal_mode', 'random')}")
    print(f"Start rect: {cfg.get('start_rect', 'N/A')}, Goal rect: {cfg.get('goal_rect', 'N/A')}\n")
    
    # 创建环境（用于获取地图和起终点）
    env = make_env_from_config(cfg, args.seed)
    cost_wrapper = find_cost_wrapper(env)
    if cost_wrapper is None:
        print("❌ GridCostWrapper not found in environment.")
        return
    
    congestion_map = cost_wrapper._congestion_map
    energy_map = cost_wrapper._energy_map
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 为每个 checkpoint 执行 rollout
    all_results = {}
    
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n--- Processing: {ckpt_name} ---")
        
        # 加载 checkpoint 和 agent
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        agent, _ = load_agent(ckpt_path, obs_dim, act_dim, args.device, checkpoint)
        
        # 重新创建环境并注入 obs_stats
        env = make_env_from_config(cfg, args.seed)
        inject_obs_stats(env, checkpoint, cfg)
        
        # 执行多次 rollout
        rollout_results = []
        for ep in range(args.episodes):
            result = rollout_single_episode(
                env, agent, args.seed + ep, deterministic, args.device, max_steps_cfg
            )
            rollout_results.append(result)
            
            success_mark = "✓" if result["success"] else "✗"
            term_mark = ""
            if result["truncated"]:
                term_mark = " [TRUNCATED]"
            elif not result["reached_goal"]:
                term_mark = " [NO-GOAL]"
            
            print(f"  Episode {ep+1}: {success_mark} "
                  f"len={result['episode_length']}, "
                  f"mean_energy={result['mean_energy']:.3f}, "
                  f"mean_load={result['mean_load']:.3f}{term_mark}")
        
        # Sanity check: 所有 episode 都失败且长度 == max_steps
        all_failed_max = all(
            not r["success"] and r["episode_length"] >= max_steps_cfg
            for r in rollout_results
        )
        if all_failed_max:
            print(f"\n  ❌ CRITICAL: All episodes hit max_steps without success!")
            print(f"     This suggests: 1) done signal not working, 2) start/goal mismatch, or 3) policy broken")
            print(f"     Check config.json and environment construction.\n")
        
        all_results[ckpt_name] = rollout_results
        
        # 为单个 checkpoint 保存独立图
        single_save_path = os.path.join(
            args.run_dir,
            f"{ckpt_name.replace('.pt', '')}_routes.png"
        )
        plot_single_checkpoint_routes(
            congestion_map,
            energy_map,
            rollout_results,
            ckpt_name,
            single_save_path,
        )
    
    # 生成总览对比图
    compare_save_path = os.path.join(args.run_dir, args.out_name)
    plot_comparison_grid(congestion_map, energy_map, all_results, compare_save_path)
    
    print(f"\n{'='*70}")
    print(f"✅ All done! Output saved to:")
    print(f"   - Comparison: {compare_save_path}")
    for ckpt_name in checkpoints.keys():
        single_path = os.path.join(
            args.run_dir,
            f"{ckpt_name.replace('.pt', '')}_routes.png"
        )
        print(f"   - {ckpt_name}: {single_path}")
    print(f"{'='*70}\n")


def main():
    """主函数：解析参数并分发到对应模式。"""
    parser = argparse.ArgumentParser(
        description="Compare routes from different checkpoints under same start/goal"
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory containing checkpoints and config.json")
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=["best_fsr.pt", "best_feasible.pt", "best_tail.pt"],
                        help="Checkpoint filenames to compare")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for environment reset (ensures same start/goal)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to rollout per checkpoint")
    parser.add_argument("--deterministic", type=str, default="True",
                        help="Use deterministic policy (True/False)")
    parser.add_argument("--out_name", type=str, default="route_compare.png",
                        help="Output filename for comparison plot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for torch inference")
    parser.add_argument("--mode", type=str, default="routes", choices=["routes", "visitmap"],
                        help="Visualization mode: routes (trajectory) or visitmap (visit frequency)")
    parser.add_argument("--cell_of_interest", type=str, default="2,2",
                        help="Cell to track for hit rate analysis (format: r,c)")
    
    args = parser.parse_args()
    
    # 转换 deterministic 参数
    deterministic = args.deterministic.lower() in ("true", "1", "yes")
    
    # 解析 cell_of_interest
    cell_of_interest = tuple(int(x) for x in args.cell_of_interest.split(","))
    
    if args.mode == "visitmap":
        run_visitmap_analysis(args, deterministic, cell_of_interest)
    else:
        run_route_comparison(args, deterministic)


if __name__ == "__main__":
    main()
