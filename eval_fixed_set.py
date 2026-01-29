import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

# 引入你的环境和网络定义
from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from networks import MultiHeadActorCritic
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper


def _get_action_mask(env):
    """Safely retrieve action_mask from the first wrapper that provides get_action_mask()."""
    cur = env
    while cur is not None:
        if hasattr(cur, "get_action_mask"):
            try:
                return cur.get_action_mask()
            except Exception:
                return None
        cur = getattr(cur, "env", None)
    return None


def _str2bool(val: str) -> bool:
    return str(val).lower() in {"1", "true", "yes"}


def _parse_rect(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return tuple(int(x) for x in val)
    if isinstance(val, str):
        try:
            parts = [int(p) for p in val.split(",")]
            if len(parts) == 4:
                return tuple(parts)
        except Exception:
            return None
    return None


def find_cost_wrapper(env):
    """Traverse wrapper stack to locate GridCostWrapper."""
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GridCostWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None

def get_oracle_cost(env, weight_key='load'):
    """使用 Dijkstra 计算给定环境状态下的最优代价"""
    # 构建图
    grid_size = env.unwrapped.grid_size
    G = nx.grid_2d_graph(grid_size, grid_size)
    start = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    goal = (env.unwrapped.goal_row, env.unwrapped.goal_col)
    
    # 寻找 CostWrapper 获取 map
    cost_wrapper = None
    curr = env
    while hasattr(curr, 'env'):
        if isinstance(curr, GridCostWrapper):
            cost_wrapper = curr
            break
        curr = curr.env
    if cost_wrapper is None: 
        cost_wrapper = env.unwrapped
    
    # 获取 load_threshold（用于 soft-threshold 公式）
    load_threshold = getattr(cost_wrapper, 'load_threshold', 0.6)

    for u, v in G.edges():
        r, c = v
        if weight_key == 'load':
            # 应用同样的 soft-threshold 公式
            raw = cost_wrapper._congestion_map[r, c]
            if load_threshold < 1.0:
                cost = max(0.0, (raw - load_threshold) / (1.0 - load_threshold))
            else:
                cost = 0.0
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
                raw = cost_wrapper._congestion_map[pos]
                if load_threshold < 1.0:
                    total_cost += max(0.0, (raw - load_threshold) / (1.0 - load_threshold))
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


def inject_obs_stats(env, checkpoint, config: Dict):
    need_norm = config.get("obs_rms", False)
    stats = checkpoint.get("obs_stats")

    target = None
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GridObsNormWrapper):
            target = cur
            break
        cur = getattr(cur, "env", None)

    if need_norm and target is None:
        raise RuntimeError("❌ Config开启 obs_rms，但环境中未找到 GridObsNormWrapper")
    if stats is not None and target is None:
        raise RuntimeError("❌ Checkpoint 含 obs_stats，但环境未挂载 GridObsNormWrapper")
    if need_norm and stats is None:
        raise RuntimeError("❌ Config开启 obs_rms，但 checkpoint 中缺少 obs_stats")

    if target is not None and stats is not None:
        current_dim = env.observation_space.shape[0]
        stats_dim = stats.mean.shape[0]
        if current_dim != stats_dim:
            raise RuntimeError(
                f"❌ Obs Dimension Mismatch! Env: {current_dim}, Checkpoint: {stats_dim}. 请检查 Config 是否与模型匹配！"
            )
        target.obs_rms = stats
        target.eval()
        if hasattr(target, "training"):
            target.training = False
        if hasattr(target, "norm_reward"):
            target.norm_reward = False
        print("✅ Obs Stats injected & Frozen (Eval Mode).")


def _select_action(
    agent,
    obs_t: torch.Tensor,
    mask_t: Optional[torch.Tensor],
    is_multi_head: bool,
    deterministic: bool,
):
    """Unified action selector that respects deterministic flag and action mask."""
    if deterministic:
        if is_multi_head:
            logits, _, _ = agent.forward(obs_t, action_mask=mask_t)
        else:
            logits, _ = agent.forward(obs_t)
            if mask_t is not None:
                logits = logits.masked_fill(mask_t == 0, float("-inf"))
        return int(torch.argmax(logits, dim=-1).item())

    if is_multi_head:
        action, _, _, _, _ = agent.get_action(obs_t, action_mask=mask_t)
    else:
        action, _, _, _ = agent.get_action(obs_t, action_mask=mask_t)
    return int(action)

def evaluate_fixed_set(
    model_path: str,
    num_episodes: int = 100,
    seed_start: int = 0,
    device: str = "cpu",
    deterministic: bool = False,
    out_csv: str = None,
    # 环境参数（可从 config 覆盖）
    grid_size: int = 8,
    step_penalty: float = -1.0,
    success_reward: float = 20.0,
    max_steps: int = 256,
    congestion_pattern: str = "block",
    congestion_density: float = 0.40,
    energy_high_density: float = 0.20,
    patch_radius: int = 2,
    start_goal_mode: Optional[str] = None,
    start_rect: Optional[Tuple[int, int, int, int]] = None,
    goal_rect: Optional[Tuple[int, int, int, int]] = None,
    record_trajectory: bool = False,
    out_npz: Optional[str] = None,
    save_trajectory_json: Optional[str] = None,
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
    energy_high_density = config.get('energy_high_density', energy_high_density)
    load_threshold = config.get('load_threshold', 0.6)
    start_goal_mode = config.get('start_goal_mode', start_goal_mode or "random")
    start_rect = _parse_rect(config.get('start_rect', start_rect))
    goal_rect = _parse_rect(config.get('goal_rect', goal_rect))
    energy_budget = config.get('energy_budget')
    load_budget = config.get('load_budget')
    if energy_budget is None or load_budget is None:
        print("[WARN] energy_budget/load_budget not found in config; feasible will fall back to success.")
    
    # 观测配置（关键：必须与训练时一致）
    include_congestion_obs = config.get('include_congestion_obs', True)
    congestion_patch_radius = config.get('congestion_patch_radius', patch_radius)
    include_energy_obs = config.get('include_energy_obs', True)
    energy_patch_radius = config.get('energy_patch_radius', patch_radius)
    obs_rms = config.get('obs_rms', False)
    
    print("\n========== Evaluation Environment Config ==========")
    print(f"Grid Size: {grid_size}, Max Steps: {max_steps}")
    print(f"Congestion: {congestion_pattern}, density={congestion_density}")
    print(f"Energy: density={energy_high_density}")
    print(f"Load threshold: {load_threshold}")
    print(f"Start/Goal: mode={start_goal_mode}, start_rect={start_rect}, goal_rect={goal_rect}")
    print(f"Observation: Congestion={include_congestion_obs} (r={congestion_patch_radius}), "
          f"Energy={include_energy_obs} (r={energy_patch_radius})")
    print("=" * 50 + "\n")

    # 评估保持固定地图（默认不在 reset 重采样）
    randomize_maps_each_reset = False
    
    # 1. 配置与训练一致的环境
    def make_env(seed):
        env = GridRoutingEnv(
            grid_size=grid_size,
            step_penalty=step_penalty,
            success_reward=success_reward,
            max_steps=max_steps,
            start_goal_mode=start_goal_mode,
            start_rect=start_rect,
            goal_rect=goal_rect,
        )
        env = GridCostWrapper(
            env,
            congestion_pattern=congestion_pattern,
            congestion_density=congestion_density,
            energy_high_density=energy_high_density,
            load_threshold=load_threshold,
            randomize_maps_each_reset=randomize_maps_each_reset,
        )
        # 🔧 修正：Hard wrapper 必须在 obs wrappers 之前（与训练一致）
        env = GridHardWrapper(env)
        # 根据训练配置有条件地添加观测 wrapper
        if include_congestion_obs:
            env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
        if include_energy_obs:
            env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius)
        if obs_rms:
            env = GridObsNormWrapper(env)
        env.reset(seed=seed)
        return env

    # 2. 加载模型（支持 Multi-Head 和 Scalar 两种架构）
    temp_env = make_env(0)
    obs_sample, _ = temp_env.reset(seed=0)
    obs_dim = obs_sample.shape[0] if hasattr(obs_sample, 'shape') else len(obs_sample)
    act_dim = temp_env.action_space.n
    temp_env.close()
    
    # 🔧 检测网络类型：从 checkpoint 中判断是 Multi-Head 还是 Scalar
    # PyTorch 2.6 defaulted torch.load to weights_only=True; allow full objects for trusted checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("network_state_dict", checkpoint.get("model_state_dict", checkpoint))
    
    # 判断依据：Multi-Head 有 v_cost_heads / actor_backbone / reward_backbone 等前缀
    is_multi_head = any(
        ("v_cost_heads" in key)
        or ("cost_value_heads" in key)
        or ("cost_critics" in key)
        or ("actor_backbone" in key)
        or ("reward_backbone" in key)
        or ("cost_backbone" in key)
        for key in state_dict.keys()
    )
    
    if is_multi_head:
        print("[INFO] Detected Multi-Head network (Lagrangian PPO)")
        from networks import MultiHeadActorCritic
        agent = MultiHeadActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config.get('hidden_dim', 128),
            cost_names=["energy", "load"]
        ).to(device)
    else:
        print("[INFO] Detected Single-Head network (Scalar PPO - V5 Baseline)")
        from networks import ActorCritic
        agent = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config.get('hidden_dim', 128)
        ).to(device)
    
    agent.load_state_dict(state_dict)
    agent.eval()

    results = []

    record_outputs = record_trajectory or (out_npz is not None) or (save_trajectory_json is not None)
    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int64) if record_outputs else None
    visit_energy_counts = np.zeros((grid_size, grid_size), dtype=np.int64) if record_outputs else None
    visit_load_counts = np.zeros((grid_size, grid_size), dtype=np.int64) if record_outputs else None
    trajectory_records: Dict[int, List[Tuple[int, int]]] = {}

    print(f"Evaluating on fixed set (Seeds {seed_start}-{seed_start+num_episodes-1})...")

    for i in tqdm(range(num_episodes)):
        seed = seed_start + i
        env = make_env(seed)
        inject_obs_stats(env, checkpoint, config)
        obs, _ = env.reset(seed=seed)

        cost_wrapper = find_cost_wrapper(env)
        energy_map = getattr(cost_wrapper, "_energy_map", None) if cost_wrapper is not None else None
        congestion_raw = getattr(cost_wrapper, "_congestion_map", None) if cost_wrapper is not None else None
        load_threshold_env = getattr(cost_wrapper, "load_threshold", load_threshold)
        traj: List[Tuple[int, int]] = []
        if record_outputs:
            r0, c0 = env.unwrapped.agent_row, env.unwrapped.agent_col
            traj.append((r0, c0))
            if 0 <= r0 < grid_size and 0 <= c0 < grid_size:
                visit_counts[r0, c0] += 1
                if energy_map is not None and energy_map[r0, c0] == 1:
                    visit_energy_counts[r0, c0] += 1
                load_hit = False
                if congestion_raw is not None:
                    load_hit = congestion_raw[r0, c0] > load_threshold_env
                if load_hit:
                    visit_load_counts[r0, c0] += 1
        
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
        feasible = False
        
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                mask = _get_action_mask(env)
                mask_t = None
                if mask is not None:
                    mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                
                action = _select_action(
                    agent=agent,
                    obs_t=obs_t,
                    mask_t=mask_t,
                    is_multi_head=is_multi_head,
                    deterministic=deterministic,
                )
            
            obs, reward, done, truncated, info = env.step(action)
            
            cost_components = info.get('cost_components', {})
            step_energy = cost_components.get('energy', 0.0)
            step_load = cost_components.get('load', 0.0)
            
            ep_energy += step_energy
            ep_load += step_load
            total_reward += reward
            ep_len += 1
            
            if record_outputs:
                pos_r, pos_c = env.unwrapped.agent_row, env.unwrapped.agent_col
                traj.append((pos_r, pos_c))
                if 0 <= pos_r < grid_size and 0 <= pos_c < grid_size:
                    visit_counts[pos_r, pos_c] += 1
                    if energy_map is not None and energy_map[pos_r, pos_c] == 1:
                        visit_energy_counts[pos_r, pos_c] += 1
                    load_hit = False
                    if congestion_raw is not None:
                        load_hit = congestion_raw[pos_r, pos_c] > load_threshold_env
                    if not load_hit and step_load > 0:
                        load_hit = True
                    if load_hit:
                        visit_load_counts[pos_r, pos_c] += 1

            if done:
                success = done and not truncated
            if truncated:
                success = False
            if done or truncated:
                break
        
        env.close()

        if record_outputs:
            trajectory_records[seed] = traj

        feasible = success
        if energy_budget is not None:
            feasible = feasible and (ep_energy / max(1, ep_len) <= energy_budget)
        if load_budget is not None:
            feasible = feasible and (ep_load / max(1, ep_len) <= load_budget)
        
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
            "oracle_shortest_len": oracle_shortest_len,
            "detour": ep_len - oracle_shortest_len,
            "feasible": feasible,
        })

    df = pd.DataFrame(results)
    
    # 打印最终报告
    print("\n========== Fixed Set Evaluation Report ==========")
    print(f"Success Rate: {df['success'].mean():.2%}")
    print(f"Avg Length: {df['ep_len'].mean():.2f} (Oracle Shortest: {df['oracle_shortest_len'].mean():.2f})")
    if 'feasible' in df.columns:
        print(f"Feasible Rate: {df['feasible'].mean():.2%}")
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

    if record_outputs:
        total_visits = visit_counts.sum()
        visit_prob = (visit_counts / total_visits) if total_visits > 0 else np.zeros_like(visit_counts, dtype=np.float64)
        if out_npz:
            np.savez(
                out_npz,
                visit_counts=visit_counts,
                visit_prob=visit_prob,
                visit_energy_counts=visit_energy_counts,
                visit_load_counts=visit_load_counts,
                grid_size=grid_size,
            )
            print(f"[INFO] Visit heatmap saved to {out_npz}")
        if save_trajectory_json:
            serializable = [
                {"seed": k, "traj": v}
                for k, v in sorted(trajectory_records.items())
            ]
            with open(save_trajectory_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Trajectories saved to {save_trajectory_json}")
    
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
        "--deterministic", type=_str2bool, nargs="?", const=True, default=False,
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
    parser.add_argument("--energy_high_density", type=float, default=0.20)
    parser.add_argument("--patch_radius", type=int, default=2)
    parser.add_argument("--record_trajectory", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--save_trajectory_json", type=str, default=None)
    parser.add_argument("--start_goal_mode", type=str, default=None)
    parser.add_argument("--start_rect", type=str, default=None)
    parser.add_argument("--goal_rect", type=str, default=None)
    
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
        energy_high_density=args.energy_high_density,
        patch_radius=args.patch_radius,
        start_goal_mode=args.start_goal_mode,
        start_rect=_parse_rect(args.start_rect),
        goal_rect=_parse_rect(args.goal_rect),
        record_trajectory=args.record_trajectory,
        out_npz=args.out_npz,
        save_trajectory_json=args.save_trajectory_json,
    )
