import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

# Ensure repo root is on sys.path when run from subdir or different CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from networks import MultiHeadActorCritic


# --------------------------
# Helpers
# --------------------------

def _warn(msg: str):
    print(f"[WARN] {msg}")


def _get_cfg(cfg: Dict[str, Any], key: str, default: Any):
    if key in cfg:
        return cfg[key]
    _warn(f"config missing '{key}', using default={default}")
    return default


def _parse_rect(rect_cfg) -> Optional[Tuple[int, int, int, int]]:
    """Config may store rect as list/tuple or None."""
    if rect_cfg is None:
        return None
    if isinstance(rect_cfg, str):
        parts = [p.strip() for p in rect_cfg.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"Invalid rect string: {rect_cfg}")
        rect_cfg = [int(p) for p in parts]
    if isinstance(rect_cfg, (list, tuple)) and len(rect_cfg) == 4:
        x0, x1, y0, y1 = map(int, rect_cfg)
        return (x0, x1, y0, y1)
    raise ValueError(f"Unsupported rect format: {rect_cfg}")


def _parse_point(pt: Optional[str]) -> Optional[Tuple[int, int]]:
    if pt is None:
        return None
    parts = [p.strip() for p in str(pt).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Point must be 'x,y': {pt}")
    return (int(parts[0]), int(parts[1]))


def _make_env(cfg: Dict[str, Any], keep_maps: bool = False):
    """Recreate eval env matching training config."""
    grid_size = int(_get_cfg(cfg, "grid_size", 8))
    step_penalty = float(_get_cfg(cfg, "step_penalty", -1.0))
    success_reward = float(_get_cfg(cfg, "success_reward", 10.0))
    max_steps_raw = _get_cfg(cfg, "max_steps", -1)
    max_steps = None if max_steps_raw is None or max_steps_raw < 0 else int(max_steps_raw)

    congestion_pattern = _get_cfg(cfg, "congestion_pattern", "random")
    congestion_density = float(_get_cfg(cfg, "congestion_density", 0.3))
    energy_high_cost = float(_get_cfg(cfg, "energy_high_cost", 3.0))
    energy_high_density = float(_get_cfg(cfg, "energy_high_density", 0.2))
    load_cost_scale = float(_get_cfg(cfg, "load_cost_scale", 1.0))

    include_congestion_obs = bool(_get_cfg(cfg, "include_congestion_obs", False))
    congestion_patch_radius = int(_get_cfg(cfg, "congestion_patch_radius", 1))
    include_energy_obs = bool(_get_cfg(cfg, "include_energy_obs", False))
    energy_patch_radius = int(_get_cfg(cfg, "energy_patch_radius", 1))
    energy_obs_normalize = bool(_get_cfg(cfg, "energy_obs_normalize", True))

    start_goal_mode = _get_cfg(cfg, "start_goal_mode", "random")
    start_rect = _parse_rect(_get_cfg(cfg, "start_rect", None))
    goal_rect = _parse_rect(_get_cfg(cfg, "goal_rect", None))

    base_env = GridRoutingEnv(
        grid_size=grid_size,
        step_penalty=step_penalty,
        success_reward=success_reward,
        max_steps=max_steps,
        start_goal_mode=start_goal_mode,
        start_rect=start_rect,
        goal_rect=goal_rect,
    )
    env = GridCostWrapper(
        base_env,
        energy_base=1.0,
        energy_high_cost=energy_high_cost,
        energy_high_density=energy_high_density,
        congestion_density=congestion_density,
        congestion_pattern=congestion_pattern,
        load_cost_scale=load_cost_scale,
        keep_maps_across_episodes=keep_maps,
    )
    env = GridHardWrapper(env)
    if include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
    if include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius, normalize=energy_obs_normalize)
    return env


def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
        return ckpt["network_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def _build_policy(env, cfg: Dict[str, Any], device: torch.device):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    cost_names = list(_get_cfg(cfg, "cost_budgets", {"energy": 1.5, "load": 0.08}).keys())
    cost_critic_mode = _get_cfg(cfg, "cost_critic_mode", "separate")
    value_head_mode = _get_cfg(cfg, "value_head_mode", "standard")
    hidden_dim = int(_get_cfg(cfg, "hidden_dim", 128))

    net = MultiHeadActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        cost_names=cost_names,
        cost_critic_mode=cost_critic_mode,
        value_head_mode=value_head_mode,
    ).to(device)
    return net


def _select_action(net: MultiHeadActorCritic, obs: np.ndarray, action_mask: Optional[np.ndarray], deterministic: bool, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = None
    if action_mask is not None:
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
    with torch.no_grad():
        logits, _, _ = net.forward(obs_t, action_mask=mask_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
    return int(action.item())


def run_eval(cfg: Dict[str, Any], args, mode: str, out_dir: Path):
    device = torch.device(args.device)

    keep_maps = (mode in {"fixed_map", "fixed_map_fixed_sg"}) and args.fix_env_seed
    env = _make_env(cfg, keep_maps=keep_maps)

    net = _build_policy(env, cfg, device)
    state_dict = _load_checkpoint(Path(args.checkpoint_path), device)
    net.load_state_dict(state_dict)
    net.eval()

    energy_budget = float(_get_cfg(cfg, "energy_budget", 1.5))
    load_budget = float(_get_cfg(cfg, "load_budget", 0.08))

    records: List[Dict[str, Any]] = []

    fixed_start_pt = _parse_point(args.fixed_start)
    fixed_goal_pt = _parse_point(args.fixed_goal)
    # baseline 是否强制固定起终点仍由 CLI 控制；fixed_map 始终不固定；fixed_map_fixed_sg 固定
    force_fix_sg = args.fix_start_goal if mode != "fixed_map" else False

    captured_start = None
    captured_goal = None
    first_map_hash = None

    for i in range(args.num_episodes):
        ep_seed = args.seed_start + i
        _set_seed(ep_seed)

        override_start = None
        override_goal = None
        # fixed_map 只固定地图；fixed_map_fixed_sg 才固定起终点
        fix_sg = (mode == "fixed_map_fixed_sg") or (force_fix_sg and mode != "fixed_map")
        if fix_sg:
            if fixed_start_pt and fixed_goal_pt:
                override_start, override_goal = fixed_start_pt, fixed_goal_pt
            elif captured_start and captured_goal:
                override_start, override_goal = captured_start, captured_goal

        obs, info = env.reset(seed=ep_seed, options={
            "override_start": override_start,
            "override_goal": override_goal,
        })
        start_pos = info.get("start") or info.get("start_pos")
        goal_pos = info.get("goal") or info.get("goal_pos")
        map_hash = info.get("map_hash") or info.get("congestion_map_hash")

        if fix_sg and captured_start is None and override_start is None:
            captured_start = start_pos
            captured_goal = goal_pos

        action_mask = info.get("action_mask", None)

        ep_ret = 0.0
        ep_steps = 0
        ep_energy = 0.0
        ep_load = 0.0
        success = False

        while True:
            action = _select_action(net, obs, action_mask, args.deterministic, device)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cost_components = info.get("cost_components", {})
            ep_energy += float(cost_components.get("energy", 0.0))
            ep_load += float(cost_components.get("load", 0.0))
            ep_ret += float(reward)
            ep_steps += 1

            obs = next_obs
            action_mask = info.get("action_mask", action_mask)

            if done:
                success = bool(terminated)
                break

        steps_denom = max(ep_steps, 1)
        energy_cost = ep_energy / steps_denom
        load_cost = ep_load / steps_denom
        feasible_energy = energy_cost <= energy_budget
        feasible_load = load_cost <= load_budget
        feasible_both = feasible_energy and feasible_load and success

        # 若 step info 没有带 start/goal，则沿用 reset 采集的值
        if start_pos is None:
            start_pos = info.get("start") or info.get("start_pos")
        if goal_pos is None:
            goal_pos = info.get("goal") or info.get("goal_pos")
        map_hash = map_hash or info.get("map_hash") or info.get("congestion_map_hash")
        if first_map_hash is None:
            first_map_hash = map_hash

        records.append({
            "seed": ep_seed,
            "start": start_pos,
            "goal": goal_pos,
            "map_hash": map_hash,
            "success": success,
            "steps": ep_steps,
            "return": ep_ret,
            "energy_cost": energy_cost,
            "load_cost": load_cost,
            "feasible_energy": feasible_energy,
            "feasible_load": feasible_load,
            "feasible_both": feasible_both,
        })

    summary = summarize(records, energy_budget, load_budget)
    summary["mode"] = mode
    summary["first_map_hash"] = first_map_hash
    summary["unique_map_hashes"] = sorted(list({r.get("map_hash") for r in records if r.get("map_hash")}))

    save_csv(records, out_dir / "episodes.csv")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_scatter(records, energy_budget, load_budget, out_dir, run_tag=out_dir.name)
    plot_hists(records, out_dir, run_tag=out_dir.name)

    return summary


def _compute_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    return float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])


def summarize(records: List[Dict[str, Any]], energy_budget: float, load_budget: float) -> Dict[str, Any]:
    def stats(subset: List[Dict[str, Any]]):
        if not subset:
            return {
                "count": 0,
                "success_rate": 0.0,
                "energy_mean": None,
                "energy_std": None,
                "load_mean": None,
                "load_std": None,
                "corr": None,
                "violation_energy_rate": None,
                "violation_load_rate": None,
            }
        energy = [r["energy_cost"] for r in subset]
        load = [r["load_cost"] for r in subset]
        succ = [r["success"] for r in subset]
        success_rate = float(np.mean(succ))
        violation_energy = float(np.mean([e > energy_budget for e in energy]))
        violation_load = float(np.mean([l > load_budget for l in load]))
        return {
            "count": len(subset),
            "success_rate": success_rate,
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "load_mean": float(np.mean(load)),
            "load_std": float(np.std(load)),
            "corr": _compute_corr(energy, load),
            "violation_energy_rate": violation_energy,
            "violation_load_rate": violation_load,
        }

    all_stats = stats(records)
    success_records = [r for r in records if r["success"]]
    success_stats = stats(success_records)

    feasible_success_rate = float(np.mean([r["feasible_both"] for r in records])) if records else 0.0

    summary = {
        "all": all_stats,
        "success_only": success_stats,
        "feasible_success_rate": feasible_success_rate,
        "energy_budget": energy_budget,
        "load_budget": load_budget,
    }
    return summary


def save_csv(records: List[Dict[str, Any]], path: Path):
    import csv

    fieldnames = [
        "seed",
        "start",
        "goal",
        "map_hash",
        "success",
        "steps",
        "return",
        "energy_cost",
        "load_cost",
        "feasible_energy",
        "feasible_load",
        "feasible_both",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_scatter(records: List[Dict[str, Any]], energy_budget: float, load_budget: float, out_dir: Path, run_tag: str):
    def _scatter(subset: List[Dict[str, Any]], title_suffix: str, fname: str):
        if not subset:
            return
        x = [r["energy_cost"] for r in subset]
        y = [r["load_cost"] for r in subset]
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, alpha=0.6, s=18)
        plt.axvline(energy_budget, color="red", linestyle="--", linewidth=1, label="E budget")
        plt.axhline(load_budget, color="orange", linestyle="--", linewidth=1, label="L budget")
        plt.xlabel("Per-step energy cost")
        plt.ylabel("Per-step load cost")
        plt.title(f"{run_tag} | {title_suffix}\nE_budget={energy_budget}, L_budget={load_budget}, n={len(subset)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    _scatter(records, "All episodes", "scatter_energy_vs_load_all.png")
    success_subset = [r for r in records if r["success"]]
    _scatter(success_subset, "Success only", "scatter_energy_vs_load_success.png")


def plot_hists(records: List[Dict[str, Any]], out_dir: Path, run_tag: str):
    if not records:
        return
    energy = [r["energy_cost"] for r in records]
    load = [r["load_cost"] for r in records]
    plt.figure(figsize=(6, 4))
    plt.hist(energy, bins=20, alpha=0.7)
    plt.xlabel("Per-step energy cost")
    plt.ylabel("Count")
    plt.title(f"{run_tag} | Energy cost histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_energy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(load, bins=20, alpha=0.7)
    plt.xlabel("Per-step load cost")
    plt.ylabel("Count")
    plt.title(f"{run_tag} | Load cost histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_load.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fixed-seed episode evaluation for energy/load correlation")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.json (optional)")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deterministic", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fix_env_seed", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--fix_start_goal", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--fixed_start", type=str, default=None, help="Optional fixed start 'x,y'")
    parser.add_argument("--fixed_goal", type=str, default=None, help="Optional fixed goal 'x,y'")
    parser.add_argument("--eval_mode", type=str, default="baseline", choices=["baseline", "fixed_map", "fixed_map_fixed_sg"], help="Base mode (script仍会跑三组)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.config_path is not None:
        cfg_path = Path(args.config_path).expanduser().resolve()
    else:
        cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_tag = cfg.get("run_tag") or ckpt_path.parent.name
    base_out = Path(args.output_dir) if args.output_dir else Path("evaluate_outputs") / f"{run_tag}_eval"
    base_out.mkdir(parents=True, exist_ok=True)

    modes = ["baseline", "fixed_map", "fixed_map_fixed_sg"]
    for mode in modes:
        sub_out = base_out / mode
        sub_out.mkdir(parents=True, exist_ok=True)
        summary = run_eval(cfg, args, mode, sub_out)
        print(f"[{mode}] saved to {sub_out} (unique maps: {len(summary.get('unique_map_hashes', []))})")


if __name__ == "__main__":
    main()
