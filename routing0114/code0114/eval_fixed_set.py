import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

# 引入你的环境和网络定义
from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from networks import MultiHeadActorCritic
from grid_local_obs_wrapper import GridLocalObsWrapper # 如果用了这个
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
# 注意：请确保 import 路径正确，可能需要 sys.path.append

def get_oracle_cost(env, weight_key='load'):
    """使用 Dijkstra 计算给定环境状态下的最优代价"""
    # 构建图
    grid_size = env.unwrapped.grid_size
    G = nx.grid_2d_graph(grid_size, grid_size)
    # 修正：使用 agent_row, agent_col 而不是 agent_pos
    start = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    goal = (env.unwrapped.goal_row, env.unwrapped.goal_col)
    
    # 填充权重
    # 注意：GridCostWrapper 将 map 存储在 env.energy_map / env.congestion_map
    # 这里需要根据你的 Wrapper 层次结构获取 map
    # 假设 env 是最外层 Wrapper
    base_env = env.unwrapped
    cost_wrapper = None
    
    # 寻找 CostWrapper 获取 map
    curr = env
    while hasattr(curr, 'env'):
        if isinstance(curr, GridCostWrapper):
            cost_wrapper = curr
            break
        curr = curr.env
    if cost_wrapper is None: 
        # 如果找不到，尝试直接从 base_env 获取（如果你的实现不同）
        cost_wrapper = base_env 

    for u, v in G.edges():
        r, c = v
        if weight_key == 'load':
            cost = cost_wrapper._congestion_map[r, c]  # 修正：使用 _congestion_map
        elif weight_key == 'energy':
            cost = cost_wrapper._energy_map[r, c]      # 修正：使用 _energy_map
        elif weight_key == 'steps':
            cost = 1.0
        else:
            raise ValueError("Unknown weight key")
        G[u][v]['weight'] = cost
        
    try:
        path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        # 计算路径总 Cost (排除 start，包含 end)
        total_cost = 0.0
        for i in range(1, len(path)):
            pos = path[i]
            if weight_key == 'load':
                total_cost += cost_wrapper._congestion_map[pos]  # 修正
            elif weight_key == 'energy':
                total_cost += cost_wrapper._energy_map[pos]      # 修正
            elif weight_key == 'steps':
                total_cost += 1.0
        return total_cost, len(path)-1
    except nx.NetworkXNoPath:
        return float('inf'), 0

def evaluate_fixed_set(
    model_path: str,
    num_episodes: int = 100,
    seed_start: int = 0,
    device: str = "cpu"
):
    # 1. 配置与训练一致的环境
    # 请务必检查这里的参数与 config.json 完全一致
    def make_env(seed):
        env = GridRoutingEnv(
            grid_size=8,
            step_penalty=-1.0,
            success_reward=20.0,
            max_steps=256
        )
        env = GridCostWrapper(
            env,
            congestion_pattern="block",
            congestion_density=0.40,
            energy_high_cost=3.0,
            energy_high_density=0.20
        )
        # 添加训练时使用的观察包装器（关键！）
        env = GridCongestionObsWrapper(
            env,
            patch_radius=2  # 从 config.json 中获取
        )
        env = GridEnergyObsWrapper(
            env,
            patch_radius=1,  # 从 config.json 中获取
            normalize=True   # 从 config.json 中获取
        )
        env = GridHardWrapper(env)
        env.reset(seed=seed) # 关键：固定 Seed
        return env

    # 2. 加载模型
    # 创建临时环境以获取正确的 obs_dim
    temp_env = make_env(0)
    obs_sample, _ = temp_env.reset(seed=0)
    obs_dim = obs_sample.shape[0] if hasattr(obs_sample, 'shape') else len(obs_sample)
    temp_env.close()
    
    agent = MultiHeadActorCritic(
        obs_dim=obs_dim,  # 从实际环境中获取（应该是 192）
        act_dim=4,
        hidden_dim=128,
        cost_names=["energy", "load"]
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint["network_state_dict"])
    agent.eval()

    results = []

    print(f"Evaluating on fixed set (Seeds {seed_start}-{seed_start+num_episodes-1})...")

    for i in tqdm(range(num_episodes)):
        seed = seed_start + i
        env = make_env(seed)
        obs, _ = env.reset(seed=seed)
        
        # --- Oracle 计算 (基准) ---
        oracle_min_load_sum, _ = get_oracle_cost(env, 'load')
        oracle_min_energy_sum, _ = get_oracle_cost(env, 'energy')
        oracle_shortest_len, _ = get_oracle_cost(env, 'steps')
        
        # --- Agent 运行 ---
        done = False
        total_reward = 0
        ep_energy = 0
        ep_load = 0
        ep_len = 0
        success = False
        
        while not done:
            with torch.no_grad():
                # 注意：obs 处理需与 update 中一致
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                # 获取 mask（修正方法名）
                mask = env.get_action_mask()
                mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                
                action, _, _, _, _ = agent.get_action(obs_t, action_mask=mask_t)
            
            obs, reward, done, truncated, info = env.step(action)
            
            # 累积统计：从 info["cost_components"] 中获取
            cost_components = info.get('cost_components', {})
            step_energy = cost_components.get('energy', 0.0)
            step_load = cost_components.get('load', 0.0)
            
            ep_energy += step_energy
            ep_load += step_load
            total_reward += reward
            ep_len += 1
            
            if done:
                # 成功是通过 terminated 判断的（terminated=True 表示到达目标）
                success = done and not truncated
                break
            if truncated:
                success = False
                break
        
        # 记录数据
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
    
    # --- 打印最终报告 ---
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
    
    return df

if __name__ == "__main__":
    # 使用 best_feasible.pt
    evaluate_fixed_set("final_victory_run/Final_Feasible_E1.35_L0.26_R20/best_feasible.pt")