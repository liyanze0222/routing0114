import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ITER_KEYS = ["iter", "iteration", "update", "step", "epoch"]
METRICS_BASE = [
    "approx_kl",
    "clip_frac",
    "actor_param_delta_ratio",
]
METRICS_CONSTRAINT = [
    "g_c_over_r",
    "adv_penalty_to_reward_ratio",
    "cos_total_r",
    "cos_total_c",
]


def load_metrics(path: Path) -> pd.DataFrame:
    """Load metrics from JSON array or JSONL into a DataFrame."""
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                data = json.load(f)
                df = pd.DataFrame(data)
            else:
                records = [json.loads(line) for line in f if line.strip()]
                df = pd.DataFrame(records)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load metrics from {path}: {exc}")

    if df.empty:
        raise ValueError(f"Metrics file {path} is empty")
    return df


def detect_iter_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for key in ITER_KEYS:
        if key in df.columns:
            return df.copy(), key
    df = df.copy()
    df["iter"] = np.arange(1, len(df) + 1)
    return df, "iter"


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def slice_pairs_tail(df: pd.DataFrame, x_col: str, y_col: str, n: int) -> pd.DataFrame:
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return sub
    return sub.iloc[-n:]


def slice_pairs_window(df: pd.DataFrame, x_col: str, y_col: str, iter_col: str, start: int, end: int) -> pd.DataFrame:
    mask = (df[iter_col] >= start) & (df[iter_col] <= end)
    return df.loc[mask, [x_col, y_col]].dropna()


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def plot_timeseries(
    metric: str,
    df_reward: pd.DataFrame,
    df_hyst: pd.DataFrame,
    iter_col_reward: str,
    iter_col_hyst: str,
    early_end: int,
    smooth: int,
    out_dir: Path,
    fmt: str,
) -> bool:
    if metric not in df_reward.columns or metric not in df_hyst.columns:
        warn(f"Missing metric '{metric}' in inputs; skipping timeseries plot")
        return False

    x_r = df_reward[iter_col_reward]
    y_r = smooth_series(df_reward[metric], smooth)
    x_h = df_hyst[iter_col_hyst]
    y_h = smooth_series(df_hyst[metric], smooth)

    plt.figure(figsize=(7, 4))
    plt.axvspan(1, early_end, color="gray", alpha=0.1, label=f"reward early 1-{early_end}")
    plt.plot(x_r, y_r, label="reward_only")
    plt.plot(x_h, y_h, label="hysteresis")
    plt.xlabel("iter")
    plt.ylabel(metric)
    plt.title(f"{metric} (shaded=reward early 1-{early_end})")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"timeseries_{metric}.{fmt}"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def window_slice(df: pd.DataFrame, metric: str, start: int, end: int, iter_col: str) -> pd.Series:
    mask = (df[iter_col] >= start) & (df[iter_col] <= end)
    return df.loc[mask, metric].dropna()


def tail_slice(df: pd.DataFrame, metric: str, n: int) -> pd.Series:
    return df[metric].dropna().iloc[-n:]


def boxplot_metric(
    metric: str,
    reward_early: pd.Series,
    hyst_early: pd.Series,
    reward_tail: pd.Series,
    hyst_tail: pd.Series,
    out_dir: Path,
    fmt: str,
) -> bool:
    groups = [reward_early, hyst_early, reward_tail, hyst_tail]
    labels = ["reward_early", "hyst_early", "reward_tail", "hyst_tail"]
    if any(len(g) == 0 for g in groups):
        warn(f"Not enough data for metric '{metric}' to draw box plot; skipping")
        return False
    plt.figure(figsize=(7, 4))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(metric)
    plt.tight_layout()
    out_path = out_dir / f"box_{metric}.{fmt}"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _compute_xlim(dataframes: List[pd.DataFrame], x_col: str, base: Tuple[float, float]) -> Tuple[float, float]:
    vals = []
    for df in dataframes:
        if df is not None and not df.empty:
            vals.extend([df[x_col].min(), df[x_col].max()])
    if not vals:
        return base
    return (min(base[0], float(np.nanmin(vals))), max(base[1], float(np.nanmax(vals))))


def _compute_ylim(dataframes: List[pd.DataFrame], y_col: str, bottom: float = 0.0) -> Tuple[float, float]:
    vals = []
    for df in dataframes:
        if df is not None and not df.empty:
            vals.extend([df[y_col].min(), df[y_col].max()])
    if not vals:
        return (bottom, 1.0)
    y_min = float(np.nanmin(vals))
    y_max = float(np.nanmax(vals))
    y_min = min(bottom, y_min)
    y_max = max(bottom + 1e-6, y_max)
    return (y_min, y_max * 1.05)


def scatter_tail_evidence(
    reward_tail: pd.DataFrame,
    hyst_tail: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    add_vline_one: bool = False,
    add_hline_one: bool = False,
) -> bool:
    if reward_tail.empty or hyst_tail.empty:
        warn(f"Not enough tail data for scatter ({x_col} vs {y_col}); skipping")
        return False

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(reward_tail[x_col], reward_tail[y_col], marker="o", alpha=0.35, label="reward_tail")
    plt.scatter(hyst_tail[x_col], hyst_tail[y_col], marker="x", alpha=0.35, label="hyst_tail")

    if add_vline_one:
        plt.axvline(1.0, linestyle="--")
    if add_hline_one:
        plt.axhline(1.0, linestyle="--")

    xlim = _compute_xlim([reward_tail, hyst_tail], x_col, (-1.05, 1.05))
    plt.xlim(xlim)
    ylim = _compute_ylim([reward_tail, hyst_tail], y_col, bottom=0.0)
    plt.ylim(ylim)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()

    lines = []
    for label, df in [("reward", reward_tail), ("hyst", hyst_tail)]:
        mean_x = float(np.nanmean(df[x_col])) if not df.empty else float("nan")
        mean_y = float(np.nanmean(df[y_col])) if not df.empty else float("nan")
        lines.append(f"{label}: mean {y_col}={mean_y:.3g}, mean {x_col}={mean_x:.3g}")
    plt.text(0.02, 0.98, "\n".join(lines), transform=plt.gca().transAxes, va="top", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[scatter] Saved {out_path}")
    for line in lines:
        print(f"  {line}")
    return True


def scatter_all_windows(
    windows: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
) -> bool:
    required = ["reward_early", "hyst_early", "reward_tail", "hyst_tail"]
    if any(w not in windows or windows[w].empty for w in required):
        warn(f"Not enough windowed data for scatter ({x_col} vs {y_col}); skipping")
        return False

    plt.figure(figsize=(6.4, 4.4))
    markers = {
        "reward_early": "o",
        "hyst_early": "x",
        "reward_tail": "s",
        "hyst_tail": "^",
    }

    for name in required:
        df = windows[name]
        plt.scatter(df[x_col], df[y_col], marker=markers.get(name, "o"), alpha=0.35, label=name)

    xlim = _compute_xlim([windows[k] for k in required], x_col, (-1.05, 1.05))
    plt.xlim(xlim)
    ylim = _compute_ylim([windows[k] for k in required], y_col, bottom=0.0)
    plt.ylim(ylim)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[scatter] Saved {out_path}")
    return True


def summarize_window(series: pd.Series) -> Dict[str, float]:
    arr = series.dropna().to_numpy()
    if arr.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p90": float(np.nanpercentile(arr, 90)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def collect_windows(
    df: pd.DataFrame,
    metric: str,
    iter_col: str,
    early_end: int,
    tail_n: int,
    label_prefix: str,
) -> Dict[str, pd.Series]:
    early = window_slice(df, metric, 1, early_end, iter_col)
    tail = tail_slice(df, metric, tail_n)
    return {
        f"{label_prefix}_early": early,
        f"{label_prefix}_tail": tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward-only vs hysteresis training metrics")
    parser.add_argument("--reward_json", required=True, help="Path to reward-only metrics (JSON or JSONL)")
    parser.add_argument("--hyst_json", required=True, help="Path to hysteresis metrics (JSON or JSONL)")
    parser.add_argument("--out_dir", default="outputs/metric_plots", help="Output directory")
    parser.add_argument("--early_end", type=int, default=50, help="Reward early window end (inclusive)")
    parser.add_argument("--tail_n", type=int, default=200, help="Tail window size")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling mean window for plotting")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf"], help="Figure format")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_reward = load_metrics(Path(args.reward_json))
    df_hyst = load_metrics(Path(args.hyst_json))

    df_reward = ensure_numeric(df_reward, METRICS_BASE + METRICS_CONSTRAINT)
    df_hyst = ensure_numeric(df_hyst, METRICS_BASE + METRICS_CONSTRAINT)

    df_reward, iter_reward = detect_iter_column(df_reward)
    df_hyst, iter_hyst = detect_iter_column(df_hyst)

    # Tail-only evidence plots
    cos_col = "cos_total_r"
    gc_col = "g_c_over_r"
    adv_ratio_col = "adv_penalty_to_reward_ratio"

    def have_cols(df: pd.DataFrame, cols: List[str]) -> bool:
        return all(col in df.columns for col in cols)

    if have_cols(df_reward, [cos_col, gc_col]) and have_cols(df_hyst, [cos_col, gc_col]):
        reward_tail_pairs = slice_pairs_tail(df_reward, cos_col, gc_col, args.tail_n)
        hyst_tail_pairs = slice_pairs_tail(df_hyst, cos_col, gc_col, args.tail_n)
        scatter_tail_evidence(
            reward_tail_pairs,
            hyst_tail_pairs,
            x_col=cos_col,
            y_col=gc_col,
            out_path=out_dir / f"scatter_tail_{cos_col}_vs_{gc_col}.{args.fmt}",
            title="(tail only; reward vs hyst)",
            add_vline_one=True,
            add_hline_one=True,
        )
    else:
        warn("Missing columns for cos_total_r vs g_c_over_r tail scatter")

    if have_cols(df_reward, [cos_col, adv_ratio_col]) and have_cols(df_hyst, [cos_col, adv_ratio_col]):
        reward_tail_pairs = slice_pairs_tail(df_reward, cos_col, adv_ratio_col, args.tail_n)
        hyst_tail_pairs = slice_pairs_tail(df_hyst, cos_col, adv_ratio_col, args.tail_n)
        scatter_tail_evidence(
            reward_tail_pairs,
            hyst_tail_pairs,
            x_col=cos_col,
            y_col=adv_ratio_col,
            out_path=out_dir / f"scatter_tail_{cos_col}_vs_{adv_ratio_col}.{args.fmt}",
            title="(tail only; reward vs hyst)",
            add_vline_one=True,
            add_hline_one=False,
        )
    else:
        warn("Missing columns for cos_total_r vs adv_penalty_to_reward_ratio tail scatter")

    if have_cols(df_reward, [cos_col, gc_col]) and have_cols(df_hyst, [cos_col, gc_col]):
        windows_pairs = {
            "reward_early": slice_pairs_window(df_reward, cos_col, gc_col, iter_reward, 1, args.early_end),
            "hyst_early": slice_pairs_window(df_hyst, cos_col, gc_col, iter_hyst, 1, args.early_end),
            "reward_tail": slice_pairs_tail(df_reward, cos_col, gc_col, args.tail_n),
            "hyst_tail": slice_pairs_tail(df_hyst, cos_col, gc_col, args.tail_n),
        }
        scatter_all_windows(
            windows_pairs,
            x_col=cos_col,
            y_col=gc_col,
            out_path=out_dir / f"scatter_all_windows_{cos_col}_vs_{gc_col}.{args.fmt}",
            title="(all windows; reward vs hyst)",
        )
    else:
        warn("Missing columns for all-window scatter cos_total_r vs g_c_over_r")

    made_any = False
    for metric in METRICS_BASE + METRICS_CONSTRAINT:
        ok = plot_timeseries(
            metric,
            df_reward,
            df_hyst,
            iter_reward,
            iter_hyst,
            args.early_end,
            args.smooth,
            out_dir,
            args.fmt,
        )
        made_any = made_any or ok

    summary_rows = []
    summary_dict = {}

    for metric in METRICS_BASE + METRICS_CONSTRAINT:
        if metric not in df_reward.columns or metric not in df_hyst.columns:
            warn(f"Missing metric '{metric}' for summary/box; skipping")
            continue

        windows: Dict[str, pd.Series] = {}
        windows.update(collect_windows(df_reward, metric, iter_reward, args.early_end, args.tail_n, "reward"))
        windows.update({
            "hyst_early": window_slice(df_hyst, metric, 1, args.early_end, iter_hyst),
            "hyst_tail": tail_slice(df_hyst, metric, args.tail_n),
        })

        # Box plot
        boxplot_metric(
            metric,
            windows.get("reward_early", pd.Series(dtype=float)),
            windows.get("hyst_early", pd.Series(dtype=float)),
            windows.get("reward_tail", pd.Series(dtype=float)),
            windows.get("hyst_tail", pd.Series(dtype=float)),
            out_dir,
            args.fmt,
        )

        # Summary rows
        summary_dict[metric] = {}
        for window_name, series in windows.items():
            stats = summarize_window(series)
            row = {"metric": metric, "window": window_name, **stats}
            summary_rows.append(row)
            summary_dict[metric][window_name] = stats

        # Ratios for base metrics
        if metric in METRICS_BASE:
            def safe_ratio(num: float, den: float) -> float:
                if den is None or np.isnan(den) or den == 0:
                    return float("nan")
                return num / den

            ratios = {
                "ratio_hyst_tail_over_reward_early_1_earlyEnd": {
                    "mean": safe_ratio(summary_dict[metric]["hyst_tail"]["mean"], summary_dict[metric]["reward_early"]["mean"]),
                    "median": safe_ratio(summary_dict[metric]["hyst_tail"]["median"], summary_dict[metric]["reward_early"]["median"]),
                },
                "ratio_hyst_tail_over_hyst_early_1_earlyEnd": {
                    "mean": safe_ratio(summary_dict[metric]["hyst_tail"]["mean"], summary_dict[metric]["hyst_early"]["mean"]),
                    "median": safe_ratio(summary_dict[metric]["hyst_tail"]["median"], summary_dict[metric]["hyst_early"]["median"]),
                },
            }

            for ratio_name, payload in ratios.items():
                row = {"metric": metric, "window": ratio_name, **payload}
                summary_rows.append(row)
                summary_dict[metric][ratio_name] = payload

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Print concise summary
    def mean_or_nan(metric_name: str, window: str) -> Optional[float]:
        try:
            val = summary_dict[metric_name][window]["mean"]
        except Exception:
            return None
        return val

    print("==== Summary (means) ====")
    for m in METRICS_BASE:
        r_early = mean_or_nan(m, "reward_early")
        h_early = mean_or_nan(m, "hyst_early")
        h_tail = mean_or_nan(m, "hyst_tail")
        if r_early is None or h_early is None or h_tail is None:
            continue
        print(f"{m}: reward_early={r_early:.4g}, hyst_early={h_early:.4g}, hyst_tail={h_tail:.4g}")

    print("-- Constraint metrics (hyst early vs tail) --")
    for m in METRICS_CONSTRAINT:
        h_tail = mean_or_nan(m, "hyst_tail")
        h_early = mean_or_nan(m, "hyst_early")
        if h_tail is None or h_early is None:
            continue
        tail_str = f"hyst_early={h_early:.4g}, hyst_tail={h_tail:.4g}"
        print(f"{m}: {tail_str}")

    print("Hint: if reward-only g_c_over_r is ~0 while hysteresis is nonzero, the constraint gradient influences actor updates.")


if __name__ == "__main__":
    main()
