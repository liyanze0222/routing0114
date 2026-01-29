"""
Example (Windows PowerShell):
python evidence_chain_report.py ^
  --ckpt_a outputs/.../reward_only/best_tail.pt ^
  --ckpt_b outputs/.../B_hysteresis_fixed/best_tail.pt ^
  --num_seeds 300 --seed_start 0 --deterministic True ^
  --out_dir outputs/evidence_chain/std_vs_hyst_fixed
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_fixed_set import evaluate_fixed_set, load_config_from_dir, find_cost_wrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_cost_env import GridCostWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_env import GridRoutingEnv
from grid_hard_wrapper import GridHardWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from visualize_rollout import plot_traj

plt.switch_backend("Agg")


def str2bool(val: str) -> bool:
    return str(val).lower() in {"1", "true", "yes"}


def parse_rect(val):
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_env_from_config(cfg: Dict, seed: int):
    grid_size = cfg.get("grid_size", 8)
    step_penalty = cfg.get("step_penalty", -1.0)
    success_reward = cfg.get("success_reward", 20.0)
    max_steps = cfg.get("max_steps", 256)
    congestion_pattern = cfg.get("congestion_pattern", "block")
    congestion_density = cfg.get("congestion_density", 0.40)
    energy_high_density = cfg.get("energy_high_density", 0.20)
    load_threshold = cfg.get("load_threshold", 0.6)
    start_goal_mode = cfg.get("start_goal_mode", "random")
    start_rect = parse_rect(cfg.get("start_rect"))
    goal_rect = parse_rect(cfg.get("goal_rect"))

    include_congestion_obs = cfg.get("include_congestion_obs", True)
    congestion_patch_radius = cfg.get("congestion_patch_radius", 2)
    include_energy_obs = cfg.get("include_energy_obs", True)
    energy_patch_radius = cfg.get("energy_patch_radius", 2)
    obs_rms = cfg.get("obs_rms", False)

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
        randomize_maps_each_reset=False,
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


def quantiles(series: pd.Series, ps: Tuple[float, ...] = (0.5, 0.9, 0.95)) -> Dict[str, float]:
    clean = series.dropna()
    out = {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
    }
    for p in ps:
        out[f"p{int(p*100)}"] = float(clean.quantile(p))
    return out


def occupancy_distance(p_a: np.ndarray, p_b: np.ndarray) -> Tuple[float, float]:
    flat_a = np.asarray(p_a, dtype=np.float64).flatten()
    flat_b = np.asarray(p_b, dtype=np.float64).flatten()
    flat_a = flat_a / max(flat_a.sum(), 1e-12)
    flat_b = flat_b / max(flat_b.sum(), 1e-12)
    l1 = float(np.abs(flat_a - flat_b).sum())
    m = 0.5 * (flat_a + flat_b)
    js = 0.5 * (
        float(np.sum(flat_a * np.log((flat_a + 1e-12) / (m + 1e-12))))
        + float(np.sum(flat_b * np.log((flat_b + 1e-12) / (m + 1e-12))))
    )
    return l1, js


def plot_hist(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str, labels: Tuple[str, str], out_path: Path):
    plt.figure(figsize=(6, 4))
    bins = 30
    plt.hist(df_a[col], bins=bins, alpha=0.6, label=labels[0], color="tab:blue")
    plt.hist(df_b[col], bins=bins, alpha=0.6, label=labels[1], color="tab:orange")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(f"{col} distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(joined: pd.DataFrame, labels: Tuple[str, str], out_path: Path):
    plt.figure(figsize=(6, 5))
    plt.scatter(joined["agent_load_mean_a"], joined["agent_energy_mean_a"], alpha=0.6, label=labels[0], color="tab:blue")
    plt.scatter(joined["agent_load_mean_b"], joined["agent_energy_mean_b"], alpha=0.6, label=labels[1], color="tab:orange")
    plt.xlabel("load mean")
    plt.ylabel("energy mean")
    plt.title("Energy vs Load per seed")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(mat: np.ndarray, out_path: Path, title: str, cmap: str = "viridis", symmetric: bool = False):
    plt.figure(figsize=(5, 5))
    if symmetric:
        vmax = float(np.max(np.abs(mat)))
        if vmax == 0:
            vmax = 1.0
        plt.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        plt.imshow(mat, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_traj_json(path: Path) -> Dict[int, List[Tuple[int, int]]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[int, List[Tuple[int, int]]] = {}
    for item in data:
        seed = int(item.get("seed"))
        mapping[seed] = [tuple(p) for p in item.get("traj", [])]
    return mapping


def pick_representative_seeds(joined: pd.DataFrame, k: int = 8) -> List[int]:
    chosen: List[int] = []
    def add_from_series(series):
        for s in series:
            if s not in chosen:
                chosen.append(int(s))
                if len(chosen) >= k:
                    break
    mask = (~joined["success_a"]) & (joined["success_b"])
    add_from_series(joined.loc[mask, "seed"].head(k))
    add_from_series(joined.reindex(joined["d_load"].abs().sort_values(ascending=False).index)["seed"])
    add_from_series(joined.reindex(joined["d_energy"].abs().sort_values(ascending=False).index)["seed"])
    return chosen[:k]


def _soft_load_cost(raw: float, load_threshold: float) -> float:
    if load_threshold < 1.0:
        return max(0.0, (raw - load_threshold) / (1.0 - load_threshold))
    return 0.0


def _path_costs(traj: List[Tuple[int, int]], cw, load_threshold: float) -> Tuple[float, float]:
    load_sum = 0.0
    energy_sum = 0.0
    if cw is None:
        return load_sum, energy_sum
    congestion = getattr(cw, "_congestion_map", None)
    energy_map = getattr(cw, "_energy_map", None)
    for r, c in traj[1:]:
        if congestion is not None:
            load_sum += _soft_load_cost(congestion[r, c], load_threshold)
        if energy_map is not None:
            energy_sum += energy_map[r, c]
    return load_sum, energy_sum


def render_rollout(traj: List[Tuple[int, int]], cfg: Dict, seed: int, label: str, out_path: Path):
    env = make_env_from_config(cfg, seed)
    start = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    goal = (env.unwrapped.goal_row, env.unwrapped.goal_col)
    cw = find_cost_wrapper(env)
    heatmap = getattr(cw, "_congestion_map", None)
    energy_map = getattr(cw, "_energy_map", None)
    annotate_fmt = "{:.2f}"
    grid_size = getattr(env.unwrapped, "grid_size", len(traj))

    load_threshold = getattr(cw, "load_threshold", 0.6) if cw is not None else 0.6
    load_sum, energy_sum = _path_costs(traj, cw, load_threshold)
    steps = max(1, len(traj) - 1)
    footer_lines = [
        f"steps={steps} | load_sum={load_sum:.4f} (mean {load_sum/steps:.4f}) | energy_sum={energy_sum:.4f} (mean {energy_sum/steps:.4f})"
    ]
    title = f"seed={seed}, label={label}, steps={max(0, len(traj) - 1)}"
    plot_traj(
        traj_rc=traj,
        start_rc=start,
        goal_rc=goal,
        grid_size=grid_size,
        heatmap=heatmap,
        energy_map=energy_map,
        energy_threshold=0.0,
        energy_text_color="cyan",
        visit_map=None,
        title=title,
        no_color=False,
        draw_arrow=True,
        arrow_every=5,
        annotate=False,
        annotate_fmt=annotate_fmt,
        footer_lines=footer_lines,
        base_cmap="YlOrRd",
        save_path=str(out_path),
    )
    env.close()
    plt.close("all")


def summarize(joined: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, occ_l1: float, occ_js: float) -> Dict:
    feasible_col = "feasible"
    feasible_rate_a = float(df_a[feasible_col].mean()) if feasible_col in df_a.columns else float(df_a["success"].mean())
    feasible_rate_b = float(df_b[feasible_col].mean()) if feasible_col in df_b.columns else float(df_b["success"].mean())
    summary = {
        "success_rate_a": float(df_a["success"].mean()),
        "success_rate_b": float(df_b["success"].mean()),
        "feasible_rate_a": feasible_rate_a,
        "feasible_rate_b": feasible_rate_b,
        "energy_mean_a": quantiles(df_a["agent_energy_mean"]),
        "energy_mean_b": quantiles(df_b["agent_energy_mean"]),
        "load_mean_a": quantiles(df_a["agent_load_mean"]),
        "load_mean_b": quantiles(df_b["agent_load_mean"]),
        "detour": {
            "mean_a": float(df_a["detour"].mean()),
            "mean_b": float(df_b["detour"].mean()),
            "median_a": float(df_a["detour"].median()),
            "median_b": float(df_b["detour"].median()),
            "p90_a": float(df_a["detour"].quantile(0.9)),
            "p90_b": float(df_b["detour"].quantile(0.9)),
        },
        "delta_stats": {
            "d_energy_mean": float(joined["d_energy"].mean()),
            "d_load_mean": float(joined["d_load"].mean()),
            "d_detour_mean": float(joined["d_detour"].mean()),
            "b_lower_load_ratio": float((joined["d_load"] > 0).mean()),
        },
        "occupancy_L1": occ_l1,
        "occupancy_JS": occ_js,
    }
    return summary


def run(args):
    out_dir = ensure_dir(Path(args.out_dir))
    label_a, label_b = args.label_a, args.label_b

    eval_a_csv = out_dir / "eval_a.csv"
    eval_b_csv = out_dir / "eval_b.csv"
    eval_a_npz = out_dir / "eval_a_visits.npz"
    eval_b_npz = out_dir / "eval_b_visits.npz"
    eval_a_traj = out_dir / "eval_a_traj.json"
    eval_b_traj = out_dir / "eval_b_traj.json"

    df_a = evaluate_fixed_set(
        model_path=args.ckpt_a,
        num_episodes=args.num_seeds,
        seed_start=args.seed_start,
        device=args.device,
        deterministic=args.deterministic,
        out_csv=str(eval_a_csv),
        record_trajectory=True,
        out_npz=str(eval_a_npz),
        save_trajectory_json=str(eval_a_traj),
    )

    df_b = evaluate_fixed_set(
        model_path=args.ckpt_b,
        num_episodes=args.num_seeds,
        seed_start=args.seed_start,
        device=args.device,
        deterministic=args.deterministic,
        out_csv=str(eval_b_csv),
        record_trajectory=True,
        out_npz=str(eval_b_npz),
        save_trajectory_json=str(eval_b_traj),
    )

    joined = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"))
    joined["d_energy"] = joined["agent_energy_mean_a"] - joined["agent_energy_mean_b"]
    joined["d_load"] = joined["agent_load_mean_a"] - joined["agent_load_mean_b"]
    joined["d_detour"] = joined["detour_a"] - joined["detour_b"]
    joined.to_csv(out_dir / "joined.csv", index=False)

    npz_a = np.load(eval_a_npz)
    npz_b = np.load(eval_b_npz)
    visit_prob_a = npz_a["visit_prob"]
    visit_prob_b = npz_b["visit_prob"]
    occ_l1, occ_js = occupancy_distance(visit_prob_a, visit_prob_b)

    summary = summarize(joined, df_a, df_b, occ_l1, occ_js)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[INFO] summary.json written")
    print(f"occupancy L1={occ_l1:.6f}, JS={occ_js:.6f}")

    plot_hist(df_a, df_b, "agent_energy_mean", (label_a, label_b), out_dir / "hist_energy_mean.png")
    plot_hist(df_a, df_b, "agent_load_mean", (label_a, label_b), out_dir / "hist_load_mean.png")
    plot_scatter(joined, (label_a, label_b), out_dir / "scatter_energy_vs_load.png")

    plot_heatmap(visit_prob_a, out_dir / "heatmap_visit_A.png", f"Visit prob {label_a}")
    plot_heatmap(visit_prob_b, out_dir / "heatmap_visit_B.png", f"Visit prob {label_b}")
    plot_heatmap(visit_prob_b - visit_prob_a, out_dir / "heatmap_visit_diff.png", f"Visit diff {label_b}-{label_a}", cmap="coolwarm", symmetric=True)

    traj_a = load_traj_json(eval_a_traj)
    traj_b = load_traj_json(eval_b_traj)
    seeds = pick_representative_seeds(joined, k=8)
    cfg_a = load_config_from_dir(args.ckpt_a)
    cfg_b = load_config_from_dir(args.ckpt_b)
    roll_dir = ensure_dir(out_dir / "rollouts")
    for seed in seeds:
        if seed not in traj_a or seed not in traj_b:
            continue
        render_rollout(traj_a[seed], cfg_a, seed, label_a, roll_dir / f"seed{seed}_{label_a}.png")
        render_rollout(traj_b[seed], cfg_b, seed, label_b, roll_dir / f"seed{seed}_{label_b}.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evidence chain comparing two checkpoints")
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--num_seeds", type=int, default=200)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--deterministic", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="outputs/evidence_chain/default")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
