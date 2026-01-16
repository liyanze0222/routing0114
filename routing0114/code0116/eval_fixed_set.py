import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os
from typing import Dict, List, Tuple

# 引入你的环境和网络定义
from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from networks import MultiHeadActorCritic
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper

def get_oracle_cost(env, weight_key='load'):
    """使用 Dijkstra 计算给定环境状态下的最优代价"""
    # 构建图
    grid_size = env.unwrapped.grid_size
    G = nx.grid_2d_graph(grid_size, grid_size)
    start = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    goal = (env.unwrapped.goal_row, env.unwrapped.goal_col)
    
    # 寻找 CostWrapper 获取 map 和 scale
    cost_wrapper = None
    curr = env
    while hasattr(curr, 'env'):
        if isinstance(curr, GridCostWrapper):
            cost_wrapper = curr
            break
        curr = curr.env
    if cost_wrapper is None: 
        cost_wrapper = env.unwrapped
    
    # 获取 load_cost_scale（关键修复）
    load_scale = getattr(cost_wrapper, 'load_cost_scale', 1.0)

    for u, v in G.edges():
        r, c = v
        if weight_key == 'load':
            cost = cost_wrapper._congestion_map[r, c] * load_scale  # 应用缩放
        elif weight_key == 'energy':
            cost = cost_wrapper._energy_map[r, c]
        elif weight_key == 'steps':
            cost = 1.0
        else:
            raise ValueError("Unknown weight key")
        G[u][v]['weight'] = cost
        
    try:
        path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        total_cost = 0.0
        for i in range(1, len(path)):
            pos = path[i]
            if weight_key == 'load':
                total_cost += cost_wrapper._congestion_map[pos] * load_scale  # 应用缩放
            elif weight_key == 'energy':
                total_cost += cost_wrapper._energy_map[pos]
            elif weight_key == 'steps':
                total_cost += 1.0
        return total_cost, len(path)-1
    except nx.NetworkXNoPath:
        return float('inf'), 0

def load_config_from_dir(ckpt_path: str) -> Dict:
    """从 checkpoint 目录加载训练配置"""
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[INFO] Loaded config from {config_path}")
        return config
    else:
        print(f"[WARNING] Config not found at {config_path}, using default values")
        return {}

def evaluate_fixed_set(
    model_path: str,
    num_episodes: int = 100,
    seed_start: int = 0,
    device: str = "cpu",
    deterministic: bool = True,
    out_csv: str = None,
    # 环境参数（可从 config 覆盖）
    grid_size: int = 8,
    step_penalty: float = -1.0,
    success_reward: float = 20.0,
    max_steps: int = 256,
    congestion_pattern: str = "block",
    congestion_density: float = 0.40,
    energy_high_cost: float = 3.0,
    energy_high_density: float = 0.20,
    patch_radius: int = 2,
):
    # 尝试从 checkpoint 目录加载配置
    config = load_config_from_dir(model_path)
    
    # 从 config 中读取环境参数（如果存在）
    grid_size = config.get('grid_size', grid_size)
    step_penalty = config.get('step_penalty', step_penalty)
    success_reward = config.get('success_reward', success_reward)
    max_steps = config.get('max_steps', max_steps)
    congestion_pattern = config.get('congestion_pattern', congestion_pattern)
    congestion_density = config.get('congestion_density', congestion_density)
    energy_high_cost = config.get('energy_high_cost', energy_high_cost)
    energy_high_density = config.get('energy_high_density', energy_high_density)
    load_cost_scale = config.get('load_cost_scale', 1.0)  # 关键：读取缩放参数
    
    # 观测配置（关键：必须与训练时一致）
    include_congestion_obs = config.get('include_congestion_obs', True)
    congestion_patch_radius = config.get('congestion_patch_radius', patch_radius)
    include_energy_obs = config.get('include_energy_obs', True)
    energy_patch_radius = config.get('energy_patch_radius', patch_radius)
    energy_obs_normalize = config.get('energy_obs_normalize', True)
    
    print("\n========== Evaluation Environment Config ==========")
    print(f"Grid Size: {grid_size}, Max Steps: {max_steps}")
    print(f"Congestion: {congestion_pattern}, density={congestion_density}")
    print(f"Energy: high_cost={energy_high_cost}, density={energy_high_density}")
    print(f"Load Cost Scale: {load_cost_scale}x (CRITICAL: must match training!)")
    print(f"Observation: Congestion={include_congestion_obs} (r={congestion_patch_radius}), "
          f"Energy={include_energy_obs} (r={energy_patch_radius}, norm={energy_obs_normalize})")
    print("=" * 50 + "\n")
    
    # 1. 配置与训练一致的环境
    def make_env(seed):
        env = GridRoutingEnv(
            grid_size=grid_size,
            step_penalty=step_penalty,
            success_reward=success_reward,
            max_steps=max_steps
        )
        env = GridCostWrapper(
            env,
            congestion_pattern=congestion_pattern,
            congestion_density=congestion_density,
            energy_high_cost=energy_high_cost,
            energy_high_density=energy_high_density,
            load_cost_scale=load_cost_scale  # 传递缩放参数
        )
        # 🔧 修正：Hard wrapper 必须在 obs wrappers 之前（与训练一致）
        env = GridHardWrapper(env)
        # 根据训练配置有条件地添加观测 wrapper
        if include_congestion_obs:
            env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
        if include_energy_obs:
            env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius, normalize=energy_obs_normalize)
        env.reset(seed=seed)
        return env

    # 2. 加载模型（支持 Multi-Head 和 Scalar 两种架构）
    temp_env = make_env(0)
    obs_sample, _ = temp_env.reset(seed=0)
    obs_dim = obs_sample.shape[0] if hasattr(obs_sample, 'shape') else len(obs_sample)
    temp_env.close()
    
    # 🔧 检测网络类型：从 checkpoint 中判断是 Multi-Head 还是 Scalar
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["network_state_dict"]
    
    # 判断依据：Multi-Head 有 "cost_value_heads" 或 "cost_critics"，Scalar 没有
    is_multi_head = any("cost_value_heads" in key or "cost_critics" in key for key in state_dict.keys())
    
    if is_multi_head:
        print("[INFO] Detected Multi-Head network (Lagrangian PPO)")
        from networks import MultiHeadActorCritic
        agent = MultiHeadActorCritic(
            obs_dim=obs_dim,
            act_dim=4,
            hidden_dim=128,
            cost_names=["energy", "load"]
        ).to(device)
    else:
        print("[INFO] Detected Single-Head network (Scalar PPO - V5 Baseline)")
        from networks import ActorCritic
        agent = ActorCritic(
            obs_dim=obs_dim,
            act_dim=4,
            hidden_dim=128
        ).to(device)
    
    agent.load_state_dict(state_dict)
    agent.eval()

    results = []

    print(f"Evaluating on fixed set (Seeds {seed_start}-{seed_start+num_episodes-1})...")

    for i in tqdm(range(num_episodes)):
        seed = seed_start + i
        env = make_env(seed)
        obs, _ = env.reset(seed=seed)
        
        # Oracle 计算
        oracle_min_load_sum, _ = get_oracle_cost(env, 'load')
        oracle_min_energy_sum, _ = get_oracle_cost(env, 'energy')
        oracle_shortest_len, _ = get_oracle_cost(env, 'steps')
        
        # Agent 运行
        done = False
        total_reward = 0
        ep_energy = 0
        ep_load = 0
        ep_len = 0
        success = False
        
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                mask = env.get_action_mask()
                mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                
                # 🔧 根据网络类型调用不同的接口
                if is_multi_head:
                    # Multi-Head: 返回 (action, log_prob, v_reward, v_costs, entropy)
                    action, _, _, _, _ = agent.get_action(obs_t, action_mask=mask_t)
                else:
                    # Scalar: 返回 (action, log_prob, entropy, value)
                    action, _, _, _ = agent.get_action(obs_t, action_mask=mask_t)
            
            obs, reward, done, truncated, info = env.step(action)
            
            cost_components = info.get('cost_components', {})
            step_energy = cost_components.get('energy', 0.0)
            step_load = cost_components.get('load', 0.0)
            
            ep_energy += step_energy
            ep_load += step_load
            total_reward += reward
            ep_len += 1
            
            if done:
                success = done and not truncated
                break
            if truncated:
                success = False
                break
        
        env.close()
        
        results.append({
            "seed": seed,
            "success": success,
            "ep_len": ep_len,
            "agent_energy_sum": ep_energy,
            "agent_load_sum": ep_load,
            "agent_energy_mean": ep_energy / max(1, ep_len),
            "agent_load_mean": ep_load / max(1, ep_len),
            "oracle_min_load_sum": oracle_min_load_sum,
            "oracle_min_energy_sum": oracle_min_energy_sum,
            "oracle_shortest_len": oracle_shortest_len
        })

    df = pd.DataFrame(results)
    
    # 打印最终报告
    print("\n========== Fixed Set Evaluation Report ==========")
    print(f"Success Rate: {df['success'].mean():.2%}")
    print(f"Avg Length: {df['ep_len'].mean():.2f} (Oracle Shortest: {df['oracle_shortest_len'].mean():.2f})")
    print("-" * 30)
    print("Energy (Episode Sum):")
    print(f"  Agent:  {df['agent_energy_sum'].mean():.4f}")
    print(f"  Oracle (Min-Energy Policy): {df['oracle_min_energy_sum'].mean():.4f}")
    print("-" * 30)
    print("Load (Episode Sum):")
    print(f"  Agent:  {df['agent_load_sum'].mean():.4f}")
    print(f"  Oracle (Min-Load Policy): {df['oracle_min_load_sum'].mean():.4f}")
    print("-" * 30)
    print("Metrics Alignment Check:")
    print(f"  Agent Load (Mean per step): {df['agent_load_mean'].mean():.4f}")
    
    # 保存 CSV
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\n[INFO] Results saved to {out_csv}")
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained agent on a fixed set of environments"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to checkpoint (.pt file, e.g., best_feasible.pt)"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=100,
        help="Number of evaluation episodes (seeds)"
    )
    parser.add_argument(
        "--seed_start", type=int, default=0,
        help="Starting seed for evaluation"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic policy (greedy) instead of stochastic"
    )
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="Output CSV file path for results"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu or cuda)"
    )
    
    # 环境参数（可选，默认从 config.json 读取）
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--step_penalty", type=float, default=-1.0)
    parser.add_argument("--success_reward", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--congestion_pattern", type=str, default="block")
    parser.add_argument("--congestion_density", type=float, default=0.40)
    parser.add_argument("--energy_high_cost", type=float, default=3.0)
    parser.add_argument("--energy_high_density", type=float, default=0.20)
    parser.add_argument("--patch_radius", type=int, default=2)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    evaluate_fixed_set(
        model_path=args.ckpt_path,
        num_episodes=args.num_seeds,
        seed_start=args.seed_start,
        device=args.device,
        deterministic=args.deterministic,
        out_csv=args.out_csv,
        grid_size=args.grid_size,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        max_steps=args.max_steps,
        congestion_pattern=args.congestion_pattern,
        congestion_density=args.congestion_density,
        energy_high_cost=args.energy_high_cost,
        energy_high_density=args.energy_high_density,
        patch_radius=args.patch_radius,
    )
