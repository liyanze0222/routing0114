import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def load_metrics(run_dir: str) -> List[Dict]:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.json not found under {run_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tail_stats(series: List[float], k: int) -> Tuple[float, float]:
    if not series:
        return 0.0, 0.0
    tail = series[-k:] if k > 0 else series
    arr = np.array(tail, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def summarize(run_dir: str, lookback: int) -> Dict[str, float]:
    data = load_metrics(run_dir)
    stats = {}

    def col(key: str) -> List[float]:
        return [d.get(key) for d in data if key in d]

    stats["gap_energy_mean"], stats["gap_energy_std"] = tail_stats(col("gap_energy"), lookback)
    stats["gap_load_mean"], stats["gap_load_std"] = tail_stats(col("gap_load"), lookback)
    stats["cost_energy_mean"], stats["cost_energy_std"] = tail_stats(col("avg_cost_energy"), lookback)
    stats["cost_load_mean"], stats["cost_load_std"] = tail_stats(col("avg_cost_load"), lookback)
    stats["fsr_mean"], stats["fsr_std"] = tail_stats(col("feasible_success_rate"), lookback)

    # 预条件 shrink 诊断（若存在）
    shrink_e = col("dual_precond_shrink_energy")
    shrink_l = col("dual_precond_shrink_load")
    if shrink_e or shrink_l:
        stats["shrink_energy_mean"], stats["shrink_energy_std"] = tail_stats(shrink_e, lookback)
        stats["shrink_load_mean"], stats["shrink_load_std"] = tail_stats(shrink_l, lookback)

    # [新增] Precond clip_hit_rate 统计
    clip_e = col("dual_precond_clip_hit_rate_energy")
    clip_l = col("dual_precond_clip_hit_rate_load")
    if clip_e or clip_l:
        stats["clip_hit_rate_energy_mean"], _ = tail_stats(clip_e, lookback)
        stats["clip_hit_rate_load_mean"], _ = tail_stats(clip_l, lookback)

    # [新增] Precond cond_approx 统计
    cond_approx = col("dual_precond_cond_approx")
    if cond_approx:
        stats["cond_approx_mean"], stats["cond_approx_std"] = tail_stats(cond_approx, lookback)

    return stats


def format_row(name: str, st: Dict[str, float]) -> str:
    shrink_e = st.get("shrink_energy_mean")
    shrink_l = st.get("shrink_load_mean")
    shrink_str = "-"
    if shrink_e is not None and shrink_l is not None:
        shrink_str = f"{shrink_e:.3f}/{shrink_l:.3f} (std {st.get('shrink_energy_std',0):.3f}/{st.get('shrink_load_std',0):.3f})"

    # [新增] clip_hit_rate 和 cond_approx 输出
    clip_str = "-"
    clip_e = st.get("clip_hit_rate_energy_mean")
    clip_l = st.get("clip_hit_rate_load_mean")
    if clip_e is not None and clip_l is not None:
        clip_str = f"E:{clip_e:.1%}/L:{clip_l:.1%}"
    
    cond_str = "-"
    cond_mean = st.get("cond_approx_mean")
    cond_std = st.get("cond_approx_std")
    if cond_mean is not None:
        cond_str = f"{cond_mean:.2f}±{cond_std:.2f}" if cond_std is not None else f"{cond_mean:.2f}"

    return (
        f"{name:10s} | gapE {st['gap_energy_mean']:.4f}±{st['gap_energy_std']:.4f} | "
        f"gapL {st['gap_load_mean']:.4f}±{st['gap_load_std']:.4f} | "
        f"costE {st['cost_energy_mean']:.4f}±{st['cost_energy_std']:.4f} | "
        f"costL {st['cost_load_mean']:.4f}±{st['cost_load_std']:.4f} | "
        f"FSR {st['fsr_mean']:.3f}±{st['fsr_std']:.3f} | "
        f"shrink {shrink_str:30s} | clip_rate {clip_str:15s} | cond {cond_str}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare dual modes over recent iterations.")
    parser.add_argument("--vanilla", required=True, help="run_dir for dual_update_mode=standard")
    parser.add_argument("--decor", required=True, help="run_dir for dual_update_mode=decorrelated")
    parser.add_argument("--precond", required=True, help="run_dir for dual_update_mode=precond")
    parser.add_argument("--lookback", type=int, default=200, help="How many recent iters to average")
    args = parser.parse_args()

    rows = []
    rows.append(("vanilla", summarize(args.vanilla, args.lookback)))
    rows.append(("decor", summarize(args.decor, args.lookback)))
    rows.append(("precond", summarize(args.precond, args.lookback)))

    print("Mode       | gapE               | gapL               | costE              | costL              | FSR           | shrink                         | clip_rate       | cond")
    print("-" * 180)
    for name, st in rows:
        print(format_row(name, st))


if __name__ == "__main__":
    main()
