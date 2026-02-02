import argparse
import json
import os
import re
from typing import Optional, Tuple, Dict, Any, List


def parse_lams_from_dirname(name: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse lamE/lamL from dirname like ..._lamE0.5_lamL1.0"""
    mE = re.search(r"lamE([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)
    mL = re.search(r"lamL([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)
    lamE = float(mE.group(1)) if mE else None
    lamL = float(mL.group(1)) if mL else None
    return lamE, lamL


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Base directory that contains run subfolders")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--energy_budget", type=float, required=True)
    ap.add_argument("--load_budget", type=float, required=True)
    ap.add_argument(
        "--metric",
        default="best_feasible_return",
        choices=["best_feasible_return", "best_fsr_value", "best_tail_value"],
        help="Which metric in config.json->training_results to maximize",
    )
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        raise SystemExit(f"base_dir not found: {base_dir}")

    prefix = f"penalty_seed{args.seed}_EB{args.energy_budget:.2f}_LB{args.load_budget:.2f}"

    candidates = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith(prefix):
            continue
        cfg = load_json(os.path.join(full, "config.json")) or {}
        tr = safe_get(cfg, ["training_results"], {}) or {}
        score = tr.get(args.metric, None)
        if score is None:
            # fall back: if no best feasible, treat as very bad
            score = float("-inf")
        # tie-breakers
        best_fsr = tr.get("best_fsr_value", None)
        if best_fsr is None:
            best_fsr = float("-inf")
        lamE = cfg.get("initial_lambda_energy", None)
        lamL = cfg.get("initial_lambda_load", None)
        if lamE is None or lamL is None:
            pE, pL = parse_lams_from_dirname(name)
            lamE = lamE if lamE is not None else pE
            lamL = lamL if lamL is not None else pL
        candidates.append((float(score), float(best_fsr), name, lamE, lamL))

    if not candidates:
        raise SystemExit(f"No runs found with prefix: {prefix} under {base_dir}")

    # sort: primary metric desc, then best_fsr desc
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    score, best_fsr, name, lamE, lamL = candidates[0]

    out = {
        "seed": args.seed,
        "energy_budget": args.energy_budget,
        "load_budget": args.load_budget,
        "metric": args.metric,
        "best_dir": name,
        "score": score,
        "best_fsr_value": best_fsr,
        "lambdaconsidered": {
            "energy": lamE,
            "load": lamL,
        },
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # stdout: convenient for .bat parsing
    # format: lamE lamL
    print(f"{lamE} {lamL}")


if __name__ == "__main__":
    main()
