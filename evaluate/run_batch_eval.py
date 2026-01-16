import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# Ensure repo root on sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.eval_fixed_seeds import run_eval, _get_cfg, _warn  # type: ignore


def _infer_variant(run_tag: str) -> str:
    tag_upper = run_tag.upper()
    if "V2" in tag_upper:
        return "V2"
    if "V5" in tag_upper:
        return "V5"
    return "unknown"


def _find_checkpoint(run_dir: Path, checkpoint_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    primary = run_dir / checkpoint_name
    if primary.exists():
        return primary
    fallback = run_dir / "best.pt"
    if fallback.exists():
        return fallback
    pt_list = sorted(run_dir.glob("*.pt"))
    if len(pt_list) == 1:
        return pt_list[0]
    if len(pt_list) > 1:
        _warn(f"Multiple .pt found in {run_dir}, picking first: {pt_list[0].name}")
        return pt_list[0]
    return None


def _load_config(run_dir: Path, config_name: str) -> Dict:
    cfg_path = run_dir / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_quadrants(records: List[Dict], energy_budget: float, load_budget: float) -> Dict[str, int]:
    counts = {"feasible": 0, "E_only": 0, "L_only": 0, "both": 0}
    for r in records:
        e_ok = r["energy_cost"] <= energy_budget
        l_ok = r["load_cost"] <= load_budget
        if e_ok and l_ok:
            counts["feasible"] += 1
        elif (not e_ok) and l_ok:
            counts["E_only"] += 1
        elif e_ok and (not l_ok):
            counts["L_only"] += 1
        else:
            counts["both"] += 1
    return counts


def _read_episodes(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "seed": int(row["seed"]),
                "start": row.get("start"),
                "goal": row.get("goal"),
                "map_hash": row.get("map_hash"),
                "success": str(row["success"]).lower() == "true",
                "steps": int(row["steps"]),
                "return": float(row["return"]),
                "energy_cost": float(row["energy_cost"]),
                "load_cost": float(row["load_cost"]),
                "feasible_energy": str(row["feasible_energy"]).lower() == "true",
                "feasible_load": str(row["feasible_load"]).lower() == "true",
                "feasible_both": str(row["feasible_both"]).lower() == "true",
            })
    return rows


def _update_summary(summary_path: Path, cfg: Dict) -> Dict:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    episodes_path = summary_path.parent / "episodes.csv"
    records = _read_episodes(episodes_path)

    energy_budget = float(summary.get("energy_budget", _get_cfg(cfg, "energy_budget", 1.5)))
    load_budget = float(summary.get("load_budget", _get_cfg(cfg, "load_budget", 0.08)))
    load_cost_scale = float(_get_cfg(cfg, "load_cost_scale", 1.0))

    success_records = [r for r in records if r["success"]]
    quadrants_success = _compute_quadrants(success_records, energy_budget, load_budget)

    # Flatten metrics
    all_stats = summary.get("all", {})
    succ_stats = summary.get("success_only", {})

    summary_updates = {
        "success_rate": all_stats.get("success_rate"),
        "feasible_success_rate": summary.get("feasible_success_rate"),
        "violation_energy_rate": all_stats.get("violation_energy_rate"),
        "violation_load_rate": all_stats.get("violation_load_rate"),
        "mean_energy": all_stats.get("energy_mean"),
        "std_energy": all_stats.get("energy_std"),
        "mean_load": all_stats.get("load_mean"),
        "std_load": all_stats.get("load_std"),
        "corr_all": all_stats.get("corr"),
        "corr_success": succ_stats.get("corr"),
        "quadrant_counts": quadrants_success,
        "budgets": {
            "energy_budget": energy_budget,
            "load_budget": load_budget,
            "load_cost_scale": load_cost_scale,
        },
    }

    summary.update(summary_updates)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _build_args(checkpoint_path: Path, num_episodes: int, seed_start: int, deterministic: bool, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        config_path=None,
        num_episodes=num_episodes,
        seed_start=seed_start,
        output_dir=None,
        deterministic=deterministic,
        device=device,
        fix_env_seed=True,
        fix_start_goal=False,
        fixed_start=None,
        fixed_goal=None,
        eval_mode="baseline",
    )


def run_single(run_dir: Path, cfg: Dict, args, out_root: Path) -> Dict:
    run_tag = run_dir.name
    out_dir = out_root / run_tag / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_eval(cfg, args, mode="baseline", out_dir=out_dir)

    # Enrich summary with quadrants and budgets and rewrite
    summary = _update_summary(out_dir / "summary.json", cfg)

    # Build batch row
    all_stats = summary.get("all", {})
    succ_stats = summary.get("success_only", {})
    budgets = summary.get("budgets", {})
    quadr = summary.get("quadrant_counts", {})

    row = {
        "run_tag": run_tag,
        "variant": _infer_variant(run_tag),
        "energy_budget": budgets.get("energy_budget"),
        "load_budget": budgets.get("load_budget"),
        "load_cost_scale": budgets.get("load_cost_scale"),
        "success_rate": summary.get("success_rate"),
        "feasible_success_rate": summary.get("feasible_success_rate"),
        "violation_energy_rate": summary.get("violation_energy_rate"),
        "violation_load_rate": summary.get("violation_load_rate"),
        "mean_energy": summary.get("mean_energy"),
        "std_energy": summary.get("std_energy"),
        "mean_load": summary.get("mean_load"),
        "std_load": summary.get("std_load"),
        "corr_all": summary.get("corr_all"),
        "corr_success": summary.get("corr_success"),
        "quadr_feasible": quadr.get("feasible"),
        "quadr_E_only": quadr.get("E_only"),
        "quadr_L_only": quadr.get("L_only"),
        "quadr_both": quadr.get("both"),
        "success_only_count": succ_stats.get("count"),
        "total_count": all_stats.get("count"),
    }
    return row


def _discover_runs(root: Path, patterns: Optional[str]) -> List[Path]:
    if patterns:
        pats = [p.strip() for p in patterns.split(";") if p.strip()]
    else:
        pats = ["*"]
    found: List[Path] = []
    for pat in pats:
        found.extend(sorted(root.glob(pat)))
    runs = [p for p in found if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run dirs found under {root} with patterns {pats}")
    return runs


def _write_batch_summaries(rows: List[Dict], out_root: Path):
    csv_path = out_root / "batch_summary.csv"
    json_path = out_root / "batch_summary.json"
    fieldnames = list(rows[0].keys()) if rows else []

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path


def _print_table(rows: List[Dict]):
    if not rows:
        return
    headers = ["variant", "run_tag", "success_rate", "feasible_success_rate", "mean_energy", "mean_load"]
    print("\nSummary (baseline, 100 episodes):")
    print(" | ".join(headers))
    for r in rows:
        vals = [
            str(r.get("variant")),
            r.get("run_tag"),
            f"{r.get('success_rate'):.3f}" if r.get("success_rate") is not None else "",
            f"{r.get('feasible_success_rate'):.3f}" if r.get("feasible_success_rate") is not None else "",
            f"{r.get('mean_energy'):.4f}" if r.get("mean_energy") is not None else "",
            f"{r.get('mean_load'):.4f}" if r.get("mean_load") is not None else "",
        ]
        print(" | ".join(vals))


def main():
    parser = argparse.ArgumentParser(description="Batch baseline eval for multiple runs (C2C V2/V5)")
    parser.add_argument("--runs_root", type=str, required=True, help="Root dir containing run subfolders")
    parser.add_argument("--run_glob", type=str, default=None, help="Glob patterns separated by ';', e.g., 'C2C_V2_*;C2C_V5_*'")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="evaluate_outputs/batch_eval_c2c")
    parser.add_argument("--deterministic", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_name", type=str, default="best_feasible.pt")
    parser.add_argument("--config_name", type=str, default="config.json")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    runs = _discover_runs(runs_root, args.run_glob)

    batch_rows: List[Dict] = []

    for run_dir in runs:
        run_tag = run_dir.name
        ckpt_path = _find_checkpoint(run_dir, args.checkpoint_name)
        if ckpt_path is None:
            _warn(f"Skip {run_tag}: no checkpoint found")
            continue
        cfg = _load_config(run_dir, args.config_name)
        eval_args = _build_args(ckpt_path, args.num_episodes, args.seed_start, args.deterministic, args.device)
        row = run_single(run_dir, cfg, eval_args, out_root)
        batch_rows.append(row)

    if not batch_rows:
        raise RuntimeError("No runs evaluated; please check inputs")

    csv_path, json_path = _write_batch_summaries(batch_rows, out_root)
    _print_table(batch_rows)
    print(f"\nSaved batch summaries to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
