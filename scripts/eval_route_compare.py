"""Route comparison between Lagrange vs reward-only policies on identical eval tasks.

Outputs:
- summary.csv: per-episode metrics for both policies
- summary.json: aggregated statistics & deltas
- plots: cost/gap histograms, feasible curves
- case figures: side-by-side trajectories & per-step costs (actual + expected)

Notes:
- Environment construction mirrors training wrappers; eval_fixed_set keeps maps/start-goal deterministic.
- Expected 1-step cost is computed via cost maps without mutating env (pure transition replica).
- KL evidence uses KL(pi_lagrange || pi_reward) evaluated on lagrange states.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from torch.distributions import Categorical

from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from networks import ActorCritic, MultiHeadActorCritic


# ------------------------- Helpers -------------------------


def load_config_from_dir(ckpt_path: str) -> Dict[str, Any]:
    cfg_path = Path(ckpt_path).parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def inject_obs_stats(env, checkpoint, cfg: Dict[str, Any]):
    need_norm = cfg.get("obs_rms", False)
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
        raise RuntimeError("Config开启 obs_rms，但环境中未找到 GridObsNormWrapper")
    if stats is not None and target is None:
        raise RuntimeError("Checkpoint 含 obs_stats，但环境未挂载 GridObsNormWrapper")
    if need_norm and stats is None:
        raise RuntimeError("Config开启 obs_rms，但 checkpoint 中缺少 obs_stats")

    if target is not None and stats is not None:
        current_dim = env.observation_space.shape[0]
        stats_dim = stats.mean.shape[0]
        if current_dim != stats_dim:
            raise RuntimeError(
                f"Obs dim mismatch: env {current_dim}, stats {stats_dim}. 请检查 config/ckpt 是否匹配"
            )
        target.obs_rms = stats
        target.eval()
        if hasattr(target, "training"):
            target.training = False
        if hasattr(target, "norm_reward"):
            target.norm_reward = False


def find_cost_wrapper(env) -> Optional[GridCostWrapper]:
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, GridCostWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


def find_base_grid(env) -> Optional[GridRoutingEnv]:
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, GridRoutingEnv):
            return cur
        cur = getattr(cur, "env", None)
    return None


def get_action_mask(env) -> Optional[np.ndarray]:
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "get_action_mask"):
            try:
                return cur.get_action_mask()
            except Exception:
                return None
        cur = getattr(cur, "env", None)
    return None


def make_env_from_config(cfg: Dict[str, Any], seed: int, eval_fixed_set: bool = True):
    grid_size = cfg.get("grid_size", 8)
    step_penalty = cfg.get("step_penalty", -1.0)
    success_reward = cfg.get("success_reward", 20.0)
    max_steps = cfg.get("max_steps", 256)
    congestion_pattern = cfg.get("congestion_pattern", "block")
    congestion_density = cfg.get("congestion_density", 0.40)
    energy_high_density = cfg.get("energy_high_density", 0.20)
    load_threshold = cfg.get("load_threshold", 0.6)

    include_congestion_obs = cfg.get("include_congestion_obs", True)
    congestion_patch_radius = cfg.get("congestion_patch_radius", 2)
    include_energy_obs = cfg.get("include_energy_obs", True)
    energy_patch_radius = cfg.get("energy_patch_radius", 2)
    obs_rms = cfg.get("obs_rms", False)

    randomize_maps_each_reset = not eval_fixed_set
    env = GridRoutingEnv(
        grid_size=grid_size,
        step_penalty=step_penalty,
        success_reward=success_reward,
        max_steps=max_steps,
    )
    env = GridCostWrapper(
        env,
        congestion_pattern=congestion_pattern,
        congestion_density=congestion_density,
        energy_high_density=energy_high_density,
        load_threshold=load_threshold,
        randomize_maps_each_reset=randomize_maps_each_reset,
        keep_maps_across_episodes=eval_fixed_set,
    )
    env = GridHardWrapper(env)
    if include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
    if include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius)
    if obs_rms:
        env = GridObsNormWrapper(env)

    env.reset(seed=seed)
    return env


def load_agent(ckpt_path: str, obs_dim: int, act_dim: int, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("network_state_dict", checkpoint.get("model_state_dict", checkpoint))
    cfg = load_config_from_dir(ckpt_path)

    is_multi_head = any(
        ("v_cost_heads" in k)
        or ("cost_value_heads" in k)
        or ("cost_critics" in k)
        or ("actor_backbone" in k)
        or ("reward_backbone" in k)
        or ("cost_backbone" in k)
        for k in state_dict.keys()
    )

    hidden_dim = cfg.get("hidden_dim", 128)
    if is_multi_head:
        agent = MultiHeadActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            cost_names=["energy", "load"],
            cost_critic_mode=cfg.get("cost_critic_mode", "separate"),
            value_head_mode=cfg.get("value_head_mode", "standard"),
        ).to(device)
    else:
        agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)

    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, is_multi_head, checkpoint, cfg


def apply_action(base_env: GridRoutingEnv, action: int) -> Tuple[int, int, bool]:
    r, c = base_env.agent_row, base_env.agent_col
    nr, nc = r, c
    if action == 0:
        nr -= 1
    elif action == 1:
        nr += 1
    elif action == 2:
        nc -= 1
    elif action == 3:
        nc += 1

    if 0 <= nr < base_env.grid_size and 0 <= nc < base_env.grid_size:
        invalid = False
    else:
        nr, nc = r, c
        invalid = True
    return nr, nc, invalid


def expected_costs(cost_wrapper: GridCostWrapper, base_env: GridRoutingEnv, probs: np.ndarray) -> Tuple[float, float]:
    if cost_wrapper is None or base_env is None:
        return float("nan"), float("nan")

    energy_map = getattr(cost_wrapper, "_energy_map", None)
    load_map = getattr(cost_wrapper, "_congestion_map", None)
    load_scale = getattr(cost_wrapper, "load_cost_scale", 1.0)
    if energy_map is None or load_map is None:
        return float("nan"), float("nan")

    expected_energy = 0.0
    expected_load = 0.0
    for a in range(cost_wrapper.action_space.n):
        nr, nc, invalid = apply_action(base_env, a)
        if invalid:
            e_cost = 0.0
        else:
            e_cost = float(energy_map[nr, nc])
        l_cost = float(load_map[nr, nc]) * load_scale
        p = float(probs[a]) if a < len(probs) else 0.0
        expected_energy += p * e_cost
        expected_load += p * l_cost
    return expected_energy, expected_load


def logits_and_probs(agent, obs_t: torch.Tensor, mask_t: Optional[torch.Tensor], is_multi_head: bool):
    if is_multi_head:
        logits, _, _ = agent.forward(obs_t, action_mask=mask_t)
    else:
        logits, _ = agent.forward(obs_t)
        if mask_t is not None:
            logits = logits.masked_fill(mask_t == 0, float("-inf"))
    dist = Categorical(logits=logits)
    probs = dist.probs.squeeze(0)
    entropy = dist.entropy().squeeze(0)
    return logits.squeeze(0), probs, entropy


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    success: bool
    feasible: bool
    total_reward: float
    total_energy: float
    total_load: float
    gap_energy: float
    gap_load: float
    ep_len: int
    map_hash: str
    start: Tuple[int, int]
    goal: Tuple[int, int]
    traj: List[Tuple[int, int]]
    actions: List[int]
    rewards: List[float]
    step_energy: List[float]
    step_load: List[float]
    expected_energy: List[float]
    expected_load: List[float]
    kl_curve: List[float]
    action_entropy: List[float]
    energy_map: Optional[np.ndarray]
    load_map: Optional[np.ndarray]


def run_episode(
    agent,
    other_agent,
    env,
    other_env,
    device: str,
    is_multi_head: bool,
    other_is_multi_head: bool,
    budgets: Tuple[Optional[float], Optional[float]],
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> EpisodeResult:
    cost_wrapper = find_cost_wrapper(env)
    base_env = find_base_grid(env)
    other_mask_source = env  # use same mask for KL evaluation

    if seed is not None:
        obs, info = env.reset(seed=seed)
        other_env.reset(seed=seed)
    else:
        obs, info = env.reset()
        other_env.reset()
    start = tuple(info.get("start") or info.get("start_pos") or (base_env.agent_row, base_env.agent_col))
    goal = tuple(info.get("goal") or info.get("goal_pos") or (base_env.goal_row, base_env.goal_col))
    map_hash = info.get("map_hash") or info.get("map_fingerprint") or ""

    total_reward = 0.0
    total_energy = 0.0
    total_load = 0.0
    done = False
    terminated = False
    truncated = False

    traj = []
    actions = []
    rewards = []
    step_energy = []
    step_load = []
    expected_energy = []
    expected_load = []
    kl_curve = []
    entropies = []

    while not done:
        traj.append((base_env.agent_row, base_env.agent_col))
        mask_np = get_action_mask(env)
        mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0) if mask_np is not None else None

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, probs, entropy = logits_and_probs(agent, obs_t, mask_t, is_multi_head)

        # KL against other policy on same state
        if other_agent is not None:
            other_obs_t = obs_t
            other_mask_t = mask_t
            _, other_probs, _ = logits_and_probs(other_agent, other_obs_t, other_mask_t, other_is_multi_head)
            p = probs.detach().clamp(min=1e-8)
            q = other_probs.detach().clamp(min=1e-8)
            kl = torch.sum(p * (torch.log(p) - torch.log(q))).item()
        else:
            kl = float("nan")

        probs_det = probs.detach()
        if mask_np is not None:
            probs_np = (probs_det * torch.as_tensor(mask_np, device=device)).cpu().numpy()
        else:
            probs_np = probs_det.cpu().numpy()
        probs_np = probs_np / (probs_np.sum() + 1e-8)

        exp_e, exp_l = expected_costs(cost_wrapper, base_env, probs_np)
        expected_energy.append(exp_e)
        expected_load.append(exp_l)
        kl_curve.append(kl)
        entropies.append(float(entropy.item()))

        if deterministic:
            action = int(torch.argmax(probs).item())
        else:
            dist = Categorical(logits=logits.unsqueeze(0))
            action = int(dist.sample().item())
        obs, reward, terminated, truncated, step_info = env.step(action)

        cost_comp = step_info.get("cost_components", {}) if isinstance(step_info, dict) else {}
        s_energy = float(cost_comp.get("energy", 0.0))
        s_load = float(cost_comp.get("load", 0.0))

        total_reward += float(reward)
        total_energy += s_energy
        total_load += s_load

        actions.append(action)
        rewards.append(float(reward))
        step_energy.append(s_energy)
        step_load.append(s_load)

        done = bool(terminated or truncated)

    traj.append((base_env.agent_row, base_env.agent_col))
    budget_e, budget_l = budgets
    gap_e = total_energy - budget_e if budget_e is not None else float("nan")
    gap_l = total_load - budget_l if budget_l is not None else float("nan")
    feasible = bool((terminated and not truncated) and ((budget_e is None or total_energy <= budget_e) and (budget_l is None or total_load <= budget_l)))

    energy_map = None
    load_map = None
    if cost_wrapper is not None:
        if getattr(cost_wrapper, "_energy_map", None) is not None:
            energy_map = np.array(cost_wrapper._energy_map, copy=True)
        if getattr(cost_wrapper, "_congestion_map", None) is not None:
            load_map = np.array(cost_wrapper._congestion_map, copy=True) * getattr(cost_wrapper, "load_cost_scale", 1.0)

    return EpisodeResult(
        policy="",
        seed=info.get("seed", None) or 0,
        success=bool(terminated and not truncated),
        feasible=feasible,
        total_reward=total_reward,
        total_energy=total_energy,
        total_load=total_load,
        gap_energy=gap_e,
        gap_load=gap_l,
        ep_len=len(actions),
        map_hash=str(map_hash),
        start=start,
        goal=goal,
        traj=traj,
        actions=actions,
        rewards=rewards,
        step_energy=step_energy,
        step_load=step_load,
        expected_energy=expected_energy,
        expected_load=expected_load,
        kl_curve=kl_curve,
        action_entropy=entropies,
        energy_map=energy_map,
        load_map=load_map,
    )


def policy_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in ["total_energy", "total_load", "gap_energy", "gap_load"]:
        out[f"{col}_mean"] = df[col].mean()
        out[f"{col}_std"] = df[col].std()
        out[f"{col}_p50"] = df[col].median()
        out[f"{col}_p90"] = df[col].quantile(0.9)
        out[f"{col}_p95"] = df[col].quantile(0.95)
        out[f"{col}_gt0_pct"] = 100.0 * (df[col] > 0).mean()
    out["success_rate"] = df["success"].mean()
    out["feasible_rate"] = df["feasible"].mean()
    return out


def plot_hist_compare(df_l: pd.DataFrame, df_r: pd.DataFrame, col: str, out_path: Path, title: str):
    plt.figure(figsize=(6, 4))
    bins = 20
    plt.hist(df_l[col], bins=bins, alpha=0.6, label="lagrange", color="tab:blue")
    plt.hist(df_r[col], bins=bins, alpha=0.6, label="reward-only", color="tab:orange")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feasible_curve(df: pd.DataFrame, out_path: Path):
    df_sorted = df.sort_values("episode")
    cum = df_sorted["feasible"].expanding().mean()
    plt.figure(figsize=(6, 3))
    plt.plot(df_sorted["episode"], cum, label="feasible rate")
    plt.xlabel("episode")
    plt.ylabel("cumulative feasible")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_case(case_idx: int, seed: int, res_l: EpisodeResult, res_r: EpisodeResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    def heatmap_subplot(ax, heatmap, traj_l, traj_r, title, cmap_name):
        if heatmap is None:
            ax.set_title(title + " (no map)")
            return
        n = heatmap.shape[0]
        ax.imshow(heatmap, origin="upper", extent=(-0.5, n - 0.5, n - 0.5, -0.5), cmap=cmap_name)
        for traj, color, label in [(traj_l, "tab:blue", "lagrange"), (traj_r, "tab:orange", "reward")]:
            arr = np.asarray(traj)
            ax.plot(arr[:, 1], arr[:, 0], color=color, linewidth=2, label=label)
        ax.scatter(res_l.start[1], res_l.start[0], marker="o", color="green", s=60, label="start")
        ax.scatter(res_l.goal[1], res_l.goal[0], marker="*", color="red", s=80, label="goal")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend()

    heatmap_subplot(fig.add_subplot(gs[0, 0]), res_l.energy_map, res_l.traj, res_r.traj, f"Energy (seed={seed})", "YlOrRd")
    heatmap_subplot(fig.add_subplot(gs[0, 1]), res_l.load_map, res_l.traj, res_r.traj, "Load", "Blues")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(res_l.step_energy, label="energy lagrange", color="tab:blue")
    ax3.plot(res_r.step_energy, label="energy reward", color="tab:orange")
    ax3.plot(res_l.step_load, label="load lagrange", color="tab:blue", linestyle="--")
    ax3.plot(res_r.step_load, label="load reward", color="tab:orange", linestyle="--")
    ax3.set_title("Per-step actual cost")
    ax3.set_xlabel("t")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(res_l.expected_energy, label="E[energy]|s lagrange", color="tab:blue")
    ax4.plot(res_r.expected_energy, label="E[energy]|s reward", color="tab:orange")
    ax4.plot(res_l.expected_load, label="E[load]|s lagrange", color="tab:blue", linestyle="--")
    ax4.plot(res_r.expected_load, label="E[load]|s reward", color="tab:orange", linestyle="--")
    if res_l.kl_curve:
        ax4b = ax4.twinx()
        ax4b.plot(res_l.kl_curve, label="KL(l||r)", color="black", alpha=0.5)
        ax4b.set_ylabel("KL")
        ax4b.legend(loc="upper right")
    ax4.set_title("Expected 1-step cost & KL")
    ax4.set_xlabel("t")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")

    fig.suptitle(
        f"Case {case_idx} seed={seed} | energy {res_l.total_energy:.2f}/{res_r.total_energy:.2f}, load {res_l.total_load:.2f}/{res_r.total_load:.2f}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / f"case_{case_idx}_seed{seed}.png", dpi=220)
    plt.close(fig)


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def main():
    ap = argparse.ArgumentParser(description="Compare routes between Lagrange and reward-only policies on identical tasks")
    ap.add_argument("--lagrange_ckpt", type=str, required=True)
    ap.add_argument("--reward_only_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_episodes", type=int, default=50)
    ap.add_argument("--num_cases", type=int, default=6)
    ap.add_argument("--eval_fixed_set", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--deterministic", type=_str2bool, default=True, help="Use greedy argmax actions (set False for sampling)")
    ap.add_argument(
        "--deterministic_reward",
        type=_str2bool,
        default=None,
        help="Override deterministic flag for reward-only policy; defaults to --deterministic",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lag_cfg = load_config_from_dir(args.lagrange_ckpt)
    rew_cfg = load_config_from_dir(args.reward_only_ckpt)
    cfg = {**rew_cfg, **lag_cfg}  # prefer lagrange keys

    env0 = make_env_from_config(cfg, seed=args.seed, eval_fixed_set=args.eval_fixed_set)
    obs_dim = env0.observation_space.shape[0]
    act_dim = env0.action_space.n
    env0.close()

    lag_agent, lag_is_multi, lag_ckpt, lag_cfg_full = load_agent(args.lagrange_ckpt, obs_dim, act_dim, args.device)
    rew_agent, rew_is_multi, rew_ckpt, rew_cfg_full = load_agent(args.reward_only_ckpt, obs_dim, act_dim, args.device)

    budget_energy = lag_cfg_full.get("energy_budget") or lag_cfg.get("energy_budget")
    budget_load = lag_cfg_full.get("load_budget") or lag_cfg.get("load_budget")
    budgets = (budget_energy, budget_load)

    ep_rows: List[Dict[str, Any]] = []
    lag_results: Dict[int, EpisodeResult] = {}
    rew_results: Dict[int, EpisodeResult] = {}

    for idx in range(args.num_episodes):
        seed = args.seed + idx
        env_l = make_env_from_config(cfg, seed=seed, eval_fixed_set=args.eval_fixed_set)
        env_r = make_env_from_config(cfg, seed=seed, eval_fixed_set=args.eval_fixed_set)

        inject_obs_stats(env_l, lag_ckpt, lag_cfg_full)
        inject_obs_stats(env_r, rew_ckpt, rew_cfg_full)

        lag_res = run_episode(
            lag_agent,
            rew_agent,
            env_l,
            env_r,
            args.device,
            lag_is_multi,
            rew_is_multi,
            budgets,
            deterministic=args.deterministic,
            seed=seed,
        )
        lag_res.policy = "lagrange"
        rew_res = run_episode(
            rew_agent,
            None,
            env_r,
            env_l,
            args.device,
            rew_is_multi,
            lag_is_multi,
            budgets,
            deterministic=(args.deterministic_reward if args.deterministic_reward is not None else args.deterministic),
            seed=seed,
        )
        rew_res.policy = "reward_only"

        lag_results[seed] = lag_res
        rew_results[seed] = rew_res

        for res in [lag_res, rew_res]:
            ep_rows.append(
                {
                    "episode": idx,
                    "seed": seed,
                    "policy": res.policy,
                    "success": res.success,
                    "feasible": res.feasible,
                    "total_reward": res.total_reward,
                    "total_energy": res.total_energy,
                    "total_load": res.total_load,
                    "gap_energy": res.gap_energy,
                    "gap_load": res.gap_load,
                    "ep_len": res.ep_len,
                    "map_hash": res.map_hash,
                }
            )

        env_l.close()
        env_r.close()

    df = pd.DataFrame(ep_rows)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)

    df_l = df[df["policy"] == "lagrange"].reset_index(drop=True)
    df_r = df[df["policy"] == "reward_only"].reset_index(drop=True)

    stats = {
        "lagrange": policy_stats(df_l),
        "reward_only": policy_stats(df_r),
        "delta_mean_energy": df_l["total_energy"].mean() - df_r["total_energy"].mean(),
        "delta_mean_load": df_l["total_load"].mean() - df_r["total_load"].mean(),
        "delta_p95_energy": df_l["total_energy"].quantile(0.95) - df_r["total_energy"].quantile(0.95),
        "delta_p95_load": df_l["total_load"].quantile(0.95) - df_r["total_load"].quantile(0.95),
        "delta_feasible_rate": df_l["feasible"].mean() - df_r["feasible"].mean(),
        "budget_energy": budget_energy,
        "budget_load": budget_load,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Plots
    plot_hist_compare(df_l, df_r, "total_energy", out_dir / "energy_hist.png", "Total energy distribution")
    plot_hist_compare(df_l, df_r, "total_load", out_dir / "load_hist.png", "Total load distribution")
    plot_hist_compare(df_l, df_r, "gap_energy", out_dir / "gap_energy_hist.png", "Gap energy")
    plot_hist_compare(df_l, df_r, "gap_load", out_dir / "gap_load_hist.png", "Gap load")
    plot_feasible_curve(df_l, out_dir / "feasible_curve_lagrange.png")
    plot_feasible_curve(df_r, out_dir / "feasible_curve_reward.png")

    # Pick cases with largest load delta
    merged = df_l.merge(df_r, on="seed", suffixes=("_l", "_r"))
    merged["load_delta"] = (merged["total_load_l"] - merged["total_load_r"]).abs()
    top_cases = merged.nlargest(min(args.num_cases, len(merged)), "load_delta")

    cases_dir = out_dir / "cases"
    for i, row in enumerate(top_cases.itertuples(), 1):
        seed = int(row.seed)
        plot_case(i, seed, lag_results[seed], rew_results[seed], cases_dir)

    print(f"Saved summary to {summary_csv}")
    print(f"Saved stats to {out_dir / 'summary.json'}")
    print(f"Saved figures under {out_dir}")


if __name__ == "__main__":
    main()