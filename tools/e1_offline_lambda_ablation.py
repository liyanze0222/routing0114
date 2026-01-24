import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from networks import MultiHeadActorCritic
from utils import set_seed


@dataclass
class PPOHyperParams:
    lr: float
    clip_coef: float
    ent_coef: float
    max_grad_norm: float


def parse_args():
    parser = argparse.ArgumentParser(description="Offline lambda ablation on a fixed checkpoint")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing config and checkpoints")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--n_rollout_episodes", type=int, default=200, help="Number of episodes for collecting rollouts")
    parser.add_argument("--n_eval_episodes", type=int, default=200, help="Number of evaluation episodes per policy")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: run_dir/e1_lambda_ablation)")
    parser.add_argument("--cell_x", type=int, default=2, help="Target cell x for hit-rate")
    parser.add_argument("--cell_y", type=int, default=2, help="Target cell y for hit-rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available, falling back to cpu")
        return "cpu"
    return device_arg


def load_config(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found under {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ckpt_path(run_dir: str, ckpt_path: Optional[str]) -> str:
    if ckpt_path:
        return ckpt_path
    candidates = [
        "best_tail.pt",
        "best_fsr.pt",
        "best_feasible.pt",
        "checkpoint_final.pt",
    ]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No checkpoint found. Pass --ckpt_path explicitly.")


def _get_action_mask(env) -> Optional[np.ndarray]:
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


def inject_obs_stats(env, checkpoint, config: Dict):
    need_norm = config.get("obs_rms", False)
    stats = checkpoint.get("obs_stats")

    target = None
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, GridObsNormWrapper):
            target = cur
            break
        cur = getattr(cur, "env", None)

    if need_norm and target is None:
        raise RuntimeError("Config enables obs_rms but GridObsNormWrapper not found in env")
    if stats is not None and target is None:
        raise RuntimeError("Checkpoint carries obs_stats but env lacks GridObsNormWrapper")
    if need_norm and stats is None:
        raise RuntimeError("Config enables obs_rms but checkpoint missing obs_stats")

    if target is not None and stats is not None:
        current_dim = env.observation_space.shape[0]
        stats_dim = stats.mean.shape[0]
        if current_dim != stats_dim:
            raise RuntimeError(f"Obs dim mismatch: env {current_dim}, stats {stats_dim}")
        target.obs_rms = stats
        target.eval()
        if hasattr(target, "training"):
            target.training = False
        if hasattr(target, "norm_reward"):
            target.norm_reward = False
        print("[INFO] Obs stats injected and frozen")


def build_env_from_config(config: Dict) -> GridRoutingEnv:
    grid_size = config.get("grid_size", 8)
    max_steps_cfg = config.get("max_steps", -1)
    max_steps = None if max_steps_cfg is None or max_steps_cfg < 0 else max_steps_cfg
    env = GridRoutingEnv(
        grid_size=grid_size,
        step_penalty=config.get("step_penalty", -1.0),
        success_reward=config.get("success_reward", 10.0),
        max_steps=max_steps,
        start_goal_mode=config.get("start_goal_mode", "random"),
        start_rect=config.get("start_rect"),
        goal_rect=config.get("goal_rect"),
    )
    env = GridCostWrapper(
        env,
        energy_base=1.0,
        energy_high_cost=config.get("energy_high_cost", 3.0),
        energy_high_density=config.get("energy_high_density", 0.2),
        congestion_density=config.get("congestion_density", 0.3),
        congestion_pattern=config.get("congestion_pattern", "random"),
        load_cost_scale=config.get("load_cost_scale", 1.0),
    )
    env = GridHardWrapper(env)
    if config.get("include_congestion_obs", False):
        env = GridCongestionObsWrapper(env, patch_radius=config.get("congestion_patch_radius", 1))
    if config.get("include_energy_obs", False):
        env = GridEnergyObsWrapper(
            env,
            patch_radius=config.get("energy_patch_radius", 1),
            normalize=config.get("energy_obs_normalize", True),
        )
    if config.get("obs_rms", False):
        env = GridObsNormWrapper(env)
    return env


def compute_gae(rewards: List[float], values: List[float], dones: List[float], gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    rewards_arr = np.array(rewards, dtype=np.float32)
    values_arr = np.array(list(values) + [0.0], dtype=np.float32)
    dones_arr = np.array(dones, dtype=np.float32)
    T = len(rewards_arr)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones_arr[t]
        delta = rewards_arr[t] + gamma * values_arr[t + 1] * nonterminal - values_arr[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
    returns = advantages + values_arr[:-1]
    return advantages, returns


def normalize_advantage(adv: np.ndarray) -> np.ndarray:
    adv = adv.astype(np.float32)
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def flatten_policy_params(actor: MultiHeadActorCritic) -> torch.Tensor:
    params = list(actor.actor_backbone.parameters()) + list(actor.policy_head.parameters())
    if not params:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat([p.detach().cpu().reshape(-1) for p in params])


def approx_kl(old_logp: torch.Tensor, new_logp: torch.Tensor) -> float:
    log_ratio = new_logp - old_logp
    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
    return float(kl.item())


def rollout_once(
    actor: MultiHeadActorCritic,
    env,
    n_episodes: int,
    seed: int,
    config: Dict,
    agg_cost_normalize_by_budget: bool,
    agg_cost_w_energy: float,
    agg_cost_w_load: float,
    budgets: Dict[str, float],
    device: str,
) -> Dict[str, np.ndarray]:
    actor.eval()
    obs_list: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[float] = []
    logps: List[float] = []
    masks: List[np.ndarray] = []
    v_rewards: List[float] = []
    v_cost_energy: List[float] = []
    v_cost_load: List[float] = []
    v_cost_total: List[float] = []
    costs_energy: List[float] = []
    costs_load: List[float] = []
    cost_total: List[float] = []
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            mask = info.get("action_mask") if isinstance(info, dict) else None
            if mask is None:
                mask = _get_action_mask(env)
            mask_arr = None
            if mask is not None:
                mask_arr = np.array(mask, dtype=np.float32)
            obs_list.append(np.array(obs, copy=True))
            masks.append(np.ones(env.action_space.n, dtype=np.float32) if mask_arr is None else mask_arr)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = None
            if mask_arr is not None:
                mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, v_reward, v_costs = actor.get_action(obs_t, action_mask=mask_t)

            actions.append(int(action))
            logps.append(float(log_prob))
            v_rewards.append(float(v_reward))
            v_cost_energy.append(float(v_costs.get("energy", 0.0)))
            v_cost_load.append(float(v_costs.get("load", 0.0)))
            v_cost_total.append(float(v_costs.get("total", 0.0)))

            obs, reward, terminated, truncated, info = env.step(action)
            costs = {}
            if isinstance(info, dict):
                costs = info.get("cost_components", {})
            energy_cost = float(costs.get("energy", 0.0))
            load_cost = float(costs.get("load", 0.0))
            costs_energy.append(energy_cost)
            costs_load.append(load_cost)
            if agg_cost_normalize_by_budget:
                total_cost_step = (
                    agg_cost_w_energy * (energy_cost / max(budgets.get("energy", 1.0), 1e-8))
                    + agg_cost_w_load * (load_cost / max(budgets.get("load", 1.0), 1e-8))
                )
            else:
                total_cost_step = agg_cost_w_energy * energy_cost + agg_cost_w_load * load_cost
            cost_total.append(float(total_cost_step))

            rewards.append(float(reward))
            done_flag = bool(terminated or truncated)
            dones.append(float(done_flag))
            total_steps += 1
            done = done_flag

    rollout = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "logp": np.array(logps, dtype=np.float32),
        "action_masks": np.array(masks, dtype=np.float32),
        "v_reward": np.array(v_rewards, dtype=np.float32),
        "v_cost_energy": np.array(v_cost_energy, dtype=np.float32),
        "v_cost_load": np.array(v_cost_load, dtype=np.float32),
        "v_cost_total": np.array(v_cost_total, dtype=np.float32),
        "cost_energy": np.array(costs_energy, dtype=np.float32),
        "cost_load": np.array(costs_load, dtype=np.float32),
        "cost_total": np.array(cost_total, dtype=np.float32),
        "total_steps": total_steps,
    }
    return rollout


def single_policy_update(
    actor: MultiHeadActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    action_masks: torch.Tensor,
    adv_eff: torch.Tensor,
    hyper: PPOHyperParams,
) -> Tuple[torch.Tensor, float]:
    actor.train()
    params = list(actor.actor_backbone.parameters()) + list(actor.policy_head.parameters())
    optimizer = torch.optim.Adam(params, lr=hyper.lr)

    new_logp, entropy, _, _ = actor.evaluate_actions(obs, actions, action_mask=action_masks)
    ratio = torch.exp(new_logp - old_logp)
    ratio_clipped = torch.clamp(ratio, 1.0 - hyper.clip_coef, 1.0 + hyper.clip_coef)
    policy_loss = -torch.min(ratio * adv_eff, ratio_clipped * adv_eff).mean()
    entropy_bonus = entropy.mean()
    loss = policy_loss - hyper.ent_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, hyper.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        updated_logp, _, _, _ = actor.evaluate_actions(obs, actions, action_mask=action_masks)
    kl = approx_kl(old_logp, updated_logp)
    return updated_logp.detach(), kl


def corr_with_cost(delta_logp: np.ndarray, adv_energy: np.ndarray, adv_load: np.ndarray) -> float:
    target = -(adv_energy + adv_load)
    if target.std() < 1e-8 or delta_logp.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(delta_logp, target)[0, 1])


def evaluate_actor(
    actor: MultiHeadActorCritic,
    config: Dict,
    checkpoint,
    seeds: List[int],
    cell_x: int,
    cell_y: int,
    device: str,
) -> Dict[str, float]:
    actor.eval()
    successes = 0
    lengths: List[int] = []
    energy_means: List[float] = []
    load_means: List[float] = []
    hit_count = 0

    for seed in seeds:
        env = build_env_from_config(config)
        inject_obs_stats(env, checkpoint, config)
        obs, info = env.reset(seed=seed)
        done = False
        ep_energy = 0.0
        ep_load = 0.0
        ep_len = 0
        hit = False
        while not done:
            mask = info.get("action_mask") if isinstance(info, dict) else None
            if mask is None:
                mask = _get_action_mask(env)
            mask_arr = None
            if mask is not None:
                mask_arr = np.array(mask, dtype=np.float32)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = None
            if mask_arr is not None:
                mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = actor.get_action(obs_t, action_mask=mask_t)
            obs, reward, terminated, truncated, info = env.step(action)
            costs = {}
            if isinstance(info, dict):
                costs = info.get("cost_components", {})
            ep_energy += float(costs.get("energy", 0.0))
            ep_load += float(costs.get("load", 0.0))
            base_env = getattr(env, "unwrapped", env)
            if hasattr(base_env, "agent_row") and hasattr(base_env, "agent_col"):
                if base_env.agent_row == cell_x and base_env.agent_col == cell_y:
                    hit = True
            ep_len += 1
            done = bool(terminated or truncated)
        env.close()
        successes += 1 if (isinstance(info, dict) and info.get("success")) else 0
        lengths.append(ep_len)
        energy_means.append(ep_energy / max(1, ep_len))
        load_means.append(ep_load / max(1, ep_len))
        if hit:
            hit_count += 1

    total_eps = max(1, len(seeds))
    metrics = {
        "success_rate": successes / total_eps,
        "avg_len": float(np.mean(lengths) if lengths else 0.0),
        "mean_energy_cost": float(np.mean(energy_means) if energy_means else 0.0),
        "mean_load_cost": float(np.mean(load_means) if load_means else 0.0),
        "hit_rate": hit_count / total_eps,
    }
    return metrics


def extract_lambdas(checkpoint, metrics_path: str) -> Tuple[float, float]:
    if "lambdas" in checkpoint:
        lam = checkpoint["lambdas"]
        return float(lam.get("energy", 0.0)), float(lam.get("load", 0.0))
    if "lambda_energy" in checkpoint or "lambda_load" in checkpoint:
        return float(checkpoint.get("lambda_energy", 0.0)), float(checkpoint.get("lambda_load", 0.0))
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if records:
                last = records[-1]
                if "lambda_energy" in last or "lambda_load" in last:
                    return float(last.get("lambda_energy", 0.0)), float(last.get("lambda_load", 0.0))
        except Exception:
            pass
    raise RuntimeError("lambda_energy/lambda_load not found in checkpoint or metrics.json")


def plot_bar(save_path: str, labels: List[str], values: List[float], ylabel: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib not available, skip plotting {save_path}")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"], edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, facecolor="white")
    plt.close(fig)


def plot_eval_metrics(save_path: str, metrics_lambda: Dict[str, float], metrics_zero: Dict[str, float]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib not available, skip plotting {save_path}")
        return
    labels = ["success_rate", "avg_len", "mean_energy_cost", "mean_load_cost", "hit_rate"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
    vals_lambda = [metrics_lambda.get(k, 0.0) for k in labels]
    vals_zero = [metrics_zero.get(k, 0.0) for k in labels]
    ax.bar(x - width / 2, vals_lambda, width, label="lambda", color="#1f77b4", edgecolor="black")
    ax.bar(x + width / 2, vals_zero, width, label="lambda_zero", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "e1_lambda_ablation")
    os.makedirs(out_dir, exist_ok=True)

    config = load_config(run_dir)
    ckpt_path = resolve_ckpt_path(run_dir, args.ckpt_path)
    print(f"[INFO] Using checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    env = build_env_from_config(config)
    inject_obs_stats(env, checkpoint, config)

    obs_sample, _ = env.reset(seed=args.seed)
    obs_dim = obs_sample.shape[0] if hasattr(obs_sample, "shape") else len(obs_sample)
    act_dim = env.action_space.n

    state_dict = checkpoint.get("network_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    actor_base = MultiHeadActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=config.get("hidden_dim", 128),
        cost_names=["energy", "load"],
        cost_critic_mode=config.get("cost_critic_mode", "separate"),
        value_head_mode=config.get("value_head_mode", "standard"),
    ).to(device)
    actor_base.load_state_dict(state_dict)

    agg_cost_normalize_by_budget = config.get("agg_cost_normalize_by_budget", True)
    agg_cost_w_energy = config.get("agg_cost_w_energy", 1.0)
    agg_cost_w_load = config.get("agg_cost_w_load", 1.0)
    budgets = {
        "energy": config.get("energy_budget", 1.2),
        "load": config.get("load_budget", 0.08),
    }

    rollout = rollout_once(
        actor_base,
        env,
        n_episodes=args.n_rollout_episodes,
        seed=args.seed,
        config=config,
        agg_cost_normalize_by_budget=agg_cost_normalize_by_budget,
        agg_cost_w_energy=agg_cost_w_energy,
        agg_cost_w_load=agg_cost_w_load,
        budgets=budgets,
        device=device,
    )
    env.close()

    adv_reward, _ = compute_gae(rollout["rewards"].tolist(), rollout["v_reward"].tolist(), rollout["dones"].tolist(), config.get("gamma", 0.99), config.get("gae_lambda", 0.95))
    if rollout["cost_energy"].size == 0 or rollout["cost_load"].size == 0:
        print("[ERROR] advantage_energy or advantage_load unavailable; aborting.")
        return
    if config.get("cost_critic_mode", "separate") == "aggregated":
        adv_energy = adv_cost_total = compute_gae(rollout["cost_total"].tolist(), rollout["v_cost_total"].tolist(), rollout["dones"].tolist(), config.get("gamma", 0.99), config.get("gae_lambda", 0.95))[0]
        adv_load = np.zeros_like(adv_energy, dtype=np.float32)
    else:
        adv_energy = compute_gae(rollout["cost_energy"].tolist(), rollout["v_cost_energy"].tolist(), rollout["dones"].tolist(), config.get("gamma", 0.99), config.get("gae_lambda", 0.95))[0]
        adv_load = compute_gae(rollout["cost_load"].tolist(), rollout["v_cost_load"].tolist(), rollout["dones"].tolist(), config.get("gamma", 0.99), config.get("gae_lambda", 0.95))[0]

    if adv_energy is None or adv_load is None:
        print("[ERROR] advantage_energy or advantage_load unavailable; aborting.")
        return

    np.savez(
        os.path.join(out_dir, "rollouts_e1.npz"),
        obs=rollout["obs"],
        actions=rollout["actions"],
        old_logp=rollout["logp"],
        advantage_r=adv_reward,
        advantage_energy=adv_energy,
        advantage_load=adv_load,
        action_masks=rollout["action_masks"],
    )

    lambda_energy, lambda_load = extract_lambdas(checkpoint, os.path.join(run_dir, "metrics.json"))
    print(f"[INFO] Using lambdas: energy={lambda_energy:.4f}, load={lambda_load:.4f}")

    adv_eff_lambda = normalize_advantage(adv_reward - lambda_energy * adv_energy - lambda_load * adv_load)
    adv_eff_zero = normalize_advantage(adv_reward)

    obs_t = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(rollout["actions"], dtype=torch.int64, device=device)
    old_logp_t = torch.as_tensor(rollout["logp"], dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(rollout["action_masks"], dtype=torch.float32, device=device)

    actor_lambda = copy.deepcopy(actor_base).to(device)
    actor_zero = copy.deepcopy(actor_base).to(device)

    hyper = PPOHyperParams(
        lr=config.get("lr", 3e-4),
        clip_coef=config.get("clip_coef", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        max_grad_norm=config.get("max_grad_norm", 0.5),
    )

    adv_eff_lambda_t = torch.as_tensor(adv_eff_lambda, dtype=torch.float32, device=device)
    adv_eff_zero_t = torch.as_tensor(adv_eff_zero, dtype=torch.float32, device=device)

    updated_logp_lambda, kl_lambda = single_policy_update(
        actor_lambda, obs_t, actions_t, old_logp_t, masks_t, adv_eff_lambda_t, hyper
    )
    updated_logp_zero, kl_zero = single_policy_update(
        actor_zero, obs_t, actions_t, old_logp_t, masks_t, adv_eff_zero_t, hyper
    )

    base_params = flatten_policy_params(actor_base)
    lambda_params = flatten_policy_params(actor_lambda)
    zero_params = flatten_policy_params(actor_zero)
    base_norm = torch.linalg.norm(base_params).item() if base_params.numel() > 0 else 0.0
    delta_ratio_lambda = torch.linalg.norm(lambda_params - base_params).item() / (base_norm + 1e-8)
    delta_ratio_zero = torch.linalg.norm(zero_params - base_params).item() / (base_norm + 1e-8)

    approx_kl_lambda = kl_lambda
    approx_kl_zero = kl_zero
    approx_kl_lambda_vs_zero = approx_kl(updated_logp_lambda, updated_logp_zero)

    delta_logp_lambda = (updated_logp_lambda - old_logp_t).detach().cpu().numpy()
    delta_logp_zero = (updated_logp_zero - old_logp_t).detach().cpu().numpy()
    corr_lambda = corr_with_cost(delta_logp_lambda, adv_energy, adv_load)
    corr_zero = corr_with_cost(delta_logp_zero, adv_energy, adv_load)

    eval_seeds = [args.seed + 100000 + i for i in range(args.n_eval_episodes)]
    metrics_lambda = evaluate_actor(actor_lambda, config, checkpoint, eval_seeds, args.cell_x, args.cell_y, device)
    metrics_zero = evaluate_actor(actor_zero, config, checkpoint, eval_seeds, args.cell_x, args.cell_y, device)

    summary = {
        "run_dir": run_dir,
        "ckpt_path": ckpt_path,
        "seed": args.seed,
        "lambdas": {"energy": lambda_energy, "load": lambda_load},
        "rollout_steps": rollout["total_steps"],
        "delta_ratio": {"lambda": delta_ratio_lambda, "lambda_zero": delta_ratio_zero},
        "kl": {
            "base_to_lambda": approx_kl_lambda,
            "base_to_zero": approx_kl_zero,
            "lambda_to_zero": approx_kl_lambda_vs_zero,
        },
        "corr_delta_logp_neg_cost": {
            "lambda": corr_lambda,
            "lambda_zero": corr_zero,
        },
        "eval_metrics": {
            "lambda": metrics_lambda,
            "lambda_zero": metrics_zero,
        },
    }

    summary_path = os.path.join(out_dir, "e1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")

    plot_bar(os.path.join(out_dir, "e1_delta_ratio.png"), ["lambda", "lambda_zero"], [delta_ratio_lambda, delta_ratio_zero], "Param delta ratio")
    plot_bar(os.path.join(out_dir, "e1_kl.png"), ["base->lambda", "base->zero"], [approx_kl_lambda, approx_kl_zero], "Approx KL")
    plot_eval_metrics(os.path.join(out_dir, "e1_eval_metrics.png"), metrics_lambda, metrics_zero)


if __name__ == "__main__":
    main()
