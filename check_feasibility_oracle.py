"""
check_feasibility_oracle.py

Oracle 可行性检查脚本：计算理论上的最优 cost 下界。

通过在随机环境实例上运行 Dijkstra 算法，计算：
1. 最短路径策略 (Step-optimal)
2. 最小能耗策略 (Energy-optimal)  
3. 最小负载策略 (Load-optimal)

这些值代表了在当前环境配置下可达到的性能下界（Oracle Floor）。
"""

import argparse
import numpy as np
import networkx as nx
from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper


def get_oracle_floor(
    num_samples=500,
    grid_size=8,
    energy_high_cost=3.0,
    energy_high_density=0.2,
    congestion_density=0.3,
    congestion_pattern="random",
    load_cost_scale=1.0,  # 🔧 新增：缩放参数
):
    """
    计算 Oracle 可行性下界。
    
    Args:
        num_samples: 采样的环境实例数量
        grid_size: 网格大小
        energy_high_cost: 高能耗区域的成本
        energy_high_density: 高能耗区域的密度
        congestion_density: 拥塞区域的密度
        congestion_pattern: 拥塞图模式 ("random" 或 "block")
    
    Returns:
        sp_avg: 最短路策略的 (Energy, Load) 平均值
        me_avg: 最小能耗策略的 (Energy, Load) 平均值
        ml_avg: 最小负载策略的 (Energy, Load) 平均值
    """
    print(f"\n{'='*70}")
    print(f"Oracle Feasibility Check")
    print(f"{'='*70}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Samples: {num_samples}")
    print(f"Energy Config: base=1.0, high={energy_high_cost}, density={energy_high_density:.2%}")
    print(f"Congestion Config: pattern={congestion_pattern}, density={congestion_density:.2%}")
    print(f"Load Cost Scale: {load_cost_scale}x (CRITICAL for x5 training)")  # 🔧 显示缩放
    print(f"{'='*70}\n")
    
    # 累计器
    shortest_path_costs = []  # 走最短路时的 (E, L)
    min_energy_costs = []     # 走 Energy 最优路径时的 (E, L)
    min_load_costs = []       # 走 Load 最优路径时的 (E, L)
    
    # 创建环境
    base_env = GridRoutingEnv(
        grid_size=grid_size,
        max_steps=256,
    )
    
    # 包装以获得 Cost Map 生成逻辑
    env = GridCostWrapper(
        base_env,
        energy_base=1.0,
        energy_high_cost=energy_high_cost,
        energy_high_density=energy_high_density,
        congestion_density=congestion_density,
        congestion_pattern=congestion_pattern,
        load_cost_scale=load_cost_scale,  # 🔧 传递缩放参数
    )
    
    print("Computing optimal paths...")
    for seed in range(num_samples):
        if (seed + 1) % 100 == 0:
            print(f"  Progress: {seed + 1}/{num_samples}")
        
        # 重置环境获取新的 start/goal 和 cost maps
        obs, info = env.reset(seed=seed)
        start = (env.unwrapped.agent_row, env.unwrapped.agent_col)
        goal = (env.unwrapped.goal_row, env.unwrapped.goal_col)
        
        # 构建 NetworkX 图
        G = nx.grid_2d_graph(grid_size, grid_size)
        
        # 填充边权重
        for u, v in G.edges():
            # v 是目标节点 (r, c)
            r, c = v
            # 从 wrapper 获取 map 值
            e_cost = float(env._energy_map[r, c]) * load_cost_scale  # 🔧 应用缩放
            l_cost = float(env._congestion_map[r, c])
            
            # 设置三种边权：step=1, energy, load
            G[u][v]['step'] = 1.0
            G[u][v]['energy'] = e_cost
            G[u][v]['load'] = l_cost
        
        # 1. 最短路基线 (Step 最优)
        try:
            path = nx.shortest_path(G, source=start, target=goal, weight='step')
            # 计算该路径的 costs (排除 start 点，包含 end 点)
            e_sum, l_sum = 0.0, 0.0
            for i in range(1, len(path)):
                pos = path[i]
                e_sum += float(env._energy_map[pos])
                l_sum += float(env._congestion_map[pos]) * load_cost_scale  # 🔧 应用缩放
            # 转换为 per-step mean
            path_len = len(path) - 1  # 不计 start 点
            if path_len > 0:
                shortest_path_costs.append((e_sum / path_len, l_sum / path_len))
        except nx.NetworkXNoPath:
            print(f"  Warning: No path found for seed {seed} (should not happen in grid)")
            continue

        # 2. Energy 最优基线
        try:
            path = nx.shortest_path(G, source=start, target=goal, weight='energy')
            e_sum, l_sum = 0.0, 0.0
            for i in range(1, len(path)):
                pos = path[i]
                e_sum += float(env._energy_map[pos])
                l_sum += float(env._congestion_map[pos]) * load_cost_scale  # 🔧 应用缩放
            path_len = len(path) - 1
            if path_len > 0:
                min_energy_costs.append((e_sum / path_len, l_sum / path_len))
        except nx.NetworkXNoPath:
            continue

        # 3. Load 最优基线
        try:
            path = nx.shortest_path(G, source=start, target=goal, weight='load')
            e_sum, l_sum = 0.0, 0.0
            for i in range(1, len(path)):
                pos = path[i]
                e_sum += float(env._energy_map[pos])
                l_sum += float(env._congestion_map[pos]) * load_cost_scale  # 🔧 应用缩放
            path_len = len(path) - 1
            if path_len > 0:
                min_load_costs.append((e_sum / path_len, l_sum / path_len))
        except nx.NetworkXNoPath:
            continue

    print(f"  Completed: {num_samples}/{num_samples}\n")
    
    # 统计结果
    sp_avg = np.mean(shortest_path_costs, axis=0) if shortest_path_costs else (0, 0)
    me_avg = np.mean(min_energy_costs, axis=0) if min_energy_costs else (0, 0)
    ml_avg = np.mean(min_load_costs, axis=0) if min_load_costs else (0, 0)
    
    sp_std = np.std(shortest_path_costs, axis=0) if shortest_path_costs else (0, 0)
    me_std = np.std(min_energy_costs, axis=0) if min_energy_costs else (0, 0)
    ml_std = np.std(min_load_costs, axis=0) if min_load_costs else (0, 0)
    
    # 打印结果
    print(f"{'='*70}")
    print("Oracle Results (Per-step Mean Cost)")
    print(f"{'='*70}\n")
    
    print("1. Shortest Path Policy (Minimize Steps, Ignore Costs)")
    print(f"   Energy: {sp_avg[0]:.4f} ± {sp_std[0]:.4f}")
    print(f"   Load:   {sp_avg[1]:.4f} ± {sp_std[1]:.4f}")
    print(f"   → Baseline for comparison (what untrained agent might achieve)\n")
    
    print("2. Min-Energy Policy (Sacrifice everything for low Energy)")
    print(f"   Energy: {me_avg[0]:.4f} ± {me_std[0]:.4f}  ← ENERGY FLOOR")
    print(f"   Load:   {me_avg[1]:.4f} ± {me_std[1]:.4f}")
    print(f"   → Theoretical minimum Energy cost achievable\n")
    
    print("3. Min-Load Policy (Sacrifice everything for low Load)")
    print(f"   Energy: {ml_avg[0]:.4f} ± {ml_std[0]:.4f}")
    print(f"   Load:   {ml_avg[1]:.4f} ± {ml_std[1]:.4f}  ← LOAD FLOOR")
    print(f"   → Theoretical minimum Load cost achievable\n")
    
    print(f"{'='*70}")
    print("Decision Guide:")
    print(f"{'='*70}\n")
    
    # 提供建议
    energy_floor = me_avg[0]
    load_floor = ml_avg[1]
    
    print(f"Energy Budget Analysis:")
    print(f"  Floor (Min-Energy):     {energy_floor:.4f}")
    print(f"  Baseline (Shortest):    {sp_avg[0]:.4f}")
    print(f"  Suggested Safe Budget:  {energy_floor + 0.05:.4f} (Floor + 5%)")
    
    print(f"\nLoad Budget Analysis:")
    print(f"  Floor (Min-Load):       {load_floor:.4f}")
    print(f"  Baseline (Shortest):    {sp_avg[1]:.4f}")
    print(f"  Suggested Safe Budget:  {load_floor + 0.02:.4f} (Floor + 2%)")
    
    print(f"\n{'='*70}")
    print("Trade-off Analysis:")
    print(f"{'='*70}\n")
    
    # 分析 Pareto 前沿
    energy_gap = ml_avg[0] - me_avg[0]  # Min-Load 策略比 Min-Energy 策略多消耗的能量
    load_gap = me_avg[1] - ml_avg[1]    # Min-Energy 策略比 Min-Load 策略多承受的负载
    
    print(f"When optimizing for Energy (vs Load):")
    print(f"  Energy saved: {-energy_gap:.4f} ({-energy_gap/ml_avg[0]*100:.1f}%)")
    print(f"  Load penalty: +{load_gap:.4f} ({load_gap/ml_avg[1]*100:.1f}%)")
    
    print(f"\nWhen optimizing for Load (vs Energy):")
    print(f"  Load saved:   {-load_gap:.4f} ({-load_gap/me_avg[1]*100:.1f}%)")
    print(f"  Energy penalty: +{energy_gap:.4f} ({energy_gap/me_avg[0]*100:.1f}%)")
    
    print(f"\n{'='*70}")
    
    return sp_avg, me_avg, ml_avg


def main():
    parser = argparse.ArgumentParser(description="Oracle Feasibility Check")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of environment instances to sample")
    parser.add_argument("--grid_size", type=int, default=8, help="Grid size (N x N)")
    parser.add_argument("--energy_high_cost", type=float, default=3.0, help="High energy cost value")
    parser.add_argument("--energy_high_density", type=float, default=0.2, help="Density of high energy regions")
    parser.add_argument("--congestion_density", type=float, default=0.3, help="Congestion density")
    parser.add_argument("--congestion_pattern", type=str, default="random", choices=["random", "block"], 
                        help="Congestion pattern")
    parser.add_argument("--load_cost_scale", type=float, default=1.0,  # 🔧 新增参数
                        help="Load cost scaling factor (use 5.0 for x5 training)")
    
    args = parser.parse_args()
    
    get_oracle_floor(
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        energy_high_cost=args.energy_high_cost,
        energy_high_density=args.energy_high_density,
        congestion_density=args.congestion_density,
        congestion_pattern=args.congestion_pattern,
        load_cost_scale=args.load_cost_scale,  # 🔧 传递参数
    )


if __name__ == "__main__":
    main()
