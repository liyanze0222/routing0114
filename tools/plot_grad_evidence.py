import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


ITER_KEYS = ["iter", "iteration", "update", "update_idx"]
LAMBDA_E_KEYS = ["lambda_energy", "lambda_e", "lam_e", "lambda_E"]
LAMBDA_L_KEYS = ["lambda_load", "lambda_l", "lam_l", "lambda_L"]
PENALTY_ABS_KEYS = ["penalty_abs_mean", "pen_abs_mean", "penalty_abs"]
PG_TOTAL_KEYS = ["pg_loss_total", "policy_loss", "pg_loss"]
PG_R_KEYS = ["pg_loss_r_like", "pg_loss_reward_like"]
PG_P_KEYS = ["pg_loss_p_like", "pg_loss_penalty_like"]
ADV_R_STD_KEYS = ["adv_r_std"]
ADV_E_STD_KEYS = ["adv_energy_std"]
ADV_L_STD_KEYS = ["adv_load_std"]
ADV_EFF_STD_KEYS = ["adv_eff_std"]
G_C_OVER_R_KEYS = ["g_c_over_r", "gc_over_gr", "grad_c_over_r"]
COS_TOTAL_C_KEYS = ["cos_total_c", "cos_tc", "cos_total_cost"]
COS_TOTAL_R_KEYS = ["cos_total_r", "cos_tr", "cos_total_reward"]
G_R_NORM_KEYS = ["g_r_norm"]
G_C_NORM_KEYS = ["g_c_norm"]
G_T_NORM_KEYS = ["g_t_norm"]
APPROX_KL_KEYS = ["approx_kl"]
PARAM_DELTA_RATIO_KEYS = ["actor_param_delta_ratio"]


def load_metrics(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: JSON lines
    records: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


def pick_key(d: Dict, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def extract_series(records: List[Dict], keys: Sequence[str]) -> Optional[List[float]]:
    first_key = None
    for k in keys:
        if any(k in r for r in records):
            first_key = k
            break
    if first_key is None:
        return None
    vals = []
    for r in records:
        if first_key in r:
            try:
                vals.append(float(r[first_key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    return vals


def build_x(records: List[Dict]) -> (List[float], str):
    key = pick_key(records[0] if records else {}, ITER_KEYS)
    if key is None:
        # find any existing iter-like key from records
        for r in records:
            key = pick_key(r, ITER_KEYS)
            if key:
                break
    if key:
        xs = []
        for r in records:
            try:
                xs.append(float(r.get(key, np.nan)))
            except (TypeError, ValueError):
                xs.append(np.nan)
        return xs, key
    return list(range(len(records))), "index"


def smooth(series: List[float], window: int) -> List[float]:
    if window <= 1:
        return series
    arr = np.array(series, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # pad to original length for alignment (prepend)
    pad = [smoothed[0]] * (window - 1)
    return list(pad + smoothed.tolist())


def downsample(xs: List[float], ys: List[float], every_k: int) -> (List[float], List[float]):
    if every_k <= 1:
        return xs, ys
    return xs[::every_k], ys[::every_k]


def plot_lines(xs: List[float], series: Dict[str, List[float]], title: str, xlabel: str, ylabel: str, out_path: Path):
    plt.figure()
    for name, vals in series.items():
        plt.plot(xs, vals, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def maybe_plot(values: Dict[str, Optional[List[float]]], keys: Iterable[str]) -> Dict[str, List[float]]:
    usable = {}
    for k in keys:
        series = values.get(k)
        if series is not None:
            usable[k] = series
    return usable


def main():
    parser = argparse.ArgumentParser(description="Plot gradient evidence figures from metrics.json")
    parser.add_argument("--metrics_path", type=str, default="metrics.json", help="Path to metrics file (JSON or JSONL)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for plots (default: same as metrics)")
    parser.add_argument("--smooth_window", type=int, default=1, help="Smoothing window size (sliding mean), 1=disabled")
    parser.add_argument("--every_k", type=int, default=1, help="Downsample factor, plot every k points")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    records = load_metrics(metrics_path)
    if not records:
        print("[warn] empty metrics; no plots generated")
        return

    xs_raw, x_key = build_x(records)
    xlabel = x_key if x_key != "index" else "index"

    # collect series
    series_map: Dict[str, Optional[List[float]]] = {
        "lambda_energy": extract_series(records, LAMBDA_E_KEYS),
        "lambda_load": extract_series(records, LAMBDA_L_KEYS),
        "penalty_abs_mean": extract_series(records, PENALTY_ABS_KEYS),
        "pg_loss_total": extract_series(records, PG_TOTAL_KEYS),
        "pg_loss_r_like": extract_series(records, PG_R_KEYS),
        "pg_loss_p_like": extract_series(records, PG_P_KEYS),
        "adv_r_std": extract_series(records, ADV_R_STD_KEYS),
        "adv_energy_std": extract_series(records, ADV_E_STD_KEYS),
        "adv_load_std": extract_series(records, ADV_L_STD_KEYS),
        "adv_eff_std": extract_series(records, ADV_EFF_STD_KEYS),
        "g_c_over_r": extract_series(records, G_C_OVER_R_KEYS),
        "cos_total_c": extract_series(records, COS_TOTAL_C_KEYS),
        "cos_total_r": extract_series(records, COS_TOTAL_R_KEYS),
        "g_r_norm": extract_series(records, G_R_NORM_KEYS),
        "g_c_norm": extract_series(records, G_C_NORM_KEYS),
        "g_t_norm": extract_series(records, G_T_NORM_KEYS),
        "approx_kl": extract_series(records, APPROX_KL_KEYS),
        "actor_param_delta_ratio": extract_series(records, PARAM_DELTA_RATIO_KEYS),
    }

    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_series(data: List[float]) -> List[float]:
        data = smooth(data, max(1, args.smooth_window))
        xs_ds, ys_ds = downsample(xs_raw, data, max(1, args.every_k))
        return xs_ds, ys_ds

    generated: List[str] = []

    # 1) Lambdas
    lambdas = maybe_plot(series_map, ["lambda_energy", "lambda_load"])
    if lambdas:
        xs_plot, _ = downsample(xs_raw, xs_raw, max(1, args.every_k))
        fig_series = {}
        for name, vals in lambdas.items():
            x_cur, y_cur = process_series(vals)
            fig_series[name] = y_cur
            xs_plot = x_cur  # keep aligned
        out_path = out_dir / "grad_evidence_lambdas.png"
        plot_lines(xs_plot, fig_series, "Lagrange multipliers", xlabel, "lambda", out_path)
        generated.append(out_path.name)
    else:
        print("[info] skip lambdas: no lambda fields found")

    # 2) A1 losses + penalty_abs_mean (use twin axis if available)
    loss_series = maybe_plot(series_map, ["pg_loss_total", "pg_loss_r_like", "pg_loss_p_like"])
    pen_series = series_map.get("penalty_abs_mean")
    if loss_series:
        xs_plot, _ = downsample(xs_raw, xs_raw, max(1, args.every_k))
        fig_series = {}
        for name, vals in loss_series.items():
            x_cur, y_cur = process_series(vals)
            fig_series[name] = y_cur
            xs_plot = x_cur
        plt.figure()
        for name, vals in fig_series.items():
            plt.plot(xs_plot, vals, label=name)
        plt.title("Actor loss decomposition (A1)")
        plt.xlabel(xlabel)
        plt.ylabel("loss")
        plt.grid(True)
        if pen_series is not None:
            x_pen, y_pen = process_series(pen_series)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(x_pen, y_pen, label="penalty_abs_mean", color="tab:orange", linestyle="--")
            ax2.set_ylabel("penalty_abs_mean")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
        else:
            plt.legend()
        plt.tight_layout()
        out_path = out_dir / "grad_evidence_a1_losses.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        generated.append(out_path.name)
    else:
        print("[info] skip A1 losses: required fields missing")

    # 3) A1 adv std
    adv_series = maybe_plot(series_map, ["adv_r_std", "adv_energy_std", "adv_load_std", "adv_eff_std"])
    if adv_series:
        xs_plot, _ = downsample(xs_raw, xs_raw, max(1, args.every_k))
        fig_series = {}
        for name, vals in adv_series.items():
            x_cur, y_cur = process_series(vals)
            fig_series[name] = y_cur
            xs_plot = x_cur
        out_path = out_dir / "grad_evidence_adv_std.png"
        plot_lines(xs_plot, fig_series, "Adv scale / stability", xlabel, "std", out_path)
        generated.append(out_path.name)
    else:
        print("[info] skip adv std: fields missing")

    # 4) A2 gradients
    a2_any = any(series_map[k] is not None for k in ["g_c_over_r", "cos_total_c", "g_r_norm", "g_c_norm", "g_t_norm"])
    if not a2_any:
        print("[info] skip A2: gradient fields not found")
    else:
        if series_map.get("g_c_over_r") is not None:
            xs_plot, ys_plot = process_series(series_map["g_c_over_r"])
            out_path = out_dir / "grad_evidence_g_c_over_r.png"
            plot_lines(xs_plot, {"g_c_over_r": ys_plot}, "g_c_over_r", xlabel, "ratio", out_path)
            generated.append(out_path.name)

        if series_map.get("cos_total_c") is not None:
            xs_plot, ys_plot = process_series(series_map["cos_total_c"])
            out_path = out_dir / "grad_evidence_cos_total_c.png"
            plot_lines(xs_plot, {"cos_total_c": ys_plot}, "cos_total_c", xlabel, "cosine", out_path)
            generated.append(out_path.name)

        norms = maybe_plot(series_map, ["g_r_norm", "g_c_norm", "g_t_norm"])
        if norms:
            xs_plot, _ = downsample(xs_raw, xs_raw, max(1, args.every_k))
            fig_series = {}
            for name, vals in norms.items():
                x_cur, y_cur = process_series(vals)
                fig_series[name] = y_cur
                xs_plot = x_cur
            out_path = out_dir / "grad_evidence_grad_norms.png"
            plot_lines(xs_plot, fig_series, "Gradient norms (A2)", xlabel, "norm", out_path)
            generated.append(out_path.name)

    if series_map.get("approx_kl") is not None:
        xs_plot, ys_plot = process_series(series_map["approx_kl"])
        out_path = out_dir / "grad_evidence_policy_kl.png"
        plot_lines(xs_plot, {"approx_kl": ys_plot}, "Approx KL", xlabel, "approx_kl", out_path)
        generated.append(out_path.name)
    else:
        print("[info] skip approx_kl: field missing")

    if series_map.get("actor_param_delta_ratio") is not None:
        xs_plot, ys_plot = process_series(series_map["actor_param_delta_ratio"])
        out_path = out_dir / "grad_evidence_param_delta.png"
        plot_lines(
            xs_plot,
            {"actor_param_delta_ratio": ys_plot},
            "Actor param delta ratio",
            xlabel,
            "delta_ratio",
            out_path,
        )
        generated.append(out_path.name)
    else:
        print("[info] skip param delta ratio: field missing")

    if generated:
        print("[done] generated plots:")
        for name in generated:
            print(f"  - {name}")
    else:
        print("[warn] no plots were generated")


if __name__ == "__main__":
    main()
