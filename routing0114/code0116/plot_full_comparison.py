"""
plot_full_comparison.py

Plotting script for Full Spectrum Ablation Study
- Handles nested directory structure: outputs/final_benchmark/{budget_level}/{variant_tag}
- Generates comparison plots for each budget level
- Metrics: success_rate, avg_cost_energy, avg_cost_load, feasible_success_rate

Usage:
    python plot_full_comparison.py outputs/final_benchmark_20260115_123456
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Variant display names and colors
VARIANT_CONFIG = {
    "v0_proposed": {"name": "V0: Multi-Head (Proposed)", "color": "#2E7D32", "linestyle": "-", "marker": "o"},
    "v1_unconstrained": {"name": "V1: Unconstrained", "color": "#1976D2", "linestyle": "--", "marker": "s"},
    "v2_shared": {"name": "V2: Shared Critic", "color": "#F57C00", "linestyle": "-.", "marker": "^"},
    "v3_energy_only": {"name": "V3: Energy-Only", "color": "#7B1FA2", "linestyle": ":", "marker": "d"},
    "v4_load_only": {"name": "V4: Load-Only", "color": "#C2185B", "linestyle": ":", "marker": "v"},
    "v5_scalar": {"name": "V5: Scalar PPO", "color": "#D32F2F", "linestyle": "-", "marker": "x"},
}

BUDGET_NAMES = {
    "budget_1.025_stable": "Stable (L=1.025)",
    "budget_1.000_stress": "Stress (L=1.000)",
    "budget_0.900_extreme": "Extreme (L=0.900)",
}


def load_metrics(metrics_path: str) -> List[Dict]:
    """加载 metrics.json"""
    if not os.path.exists(metrics_path):
        return []
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        return []


def extract_series(data: List[Dict], key: str, default=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """提取时间序列数据"""
    iterations = []
    values = []
    
    for entry in data:
        if "iteration" in entry and key in entry:
            iterations.append(entry["iteration"])
            values.append(entry.get(key, default))
    
    return np.array(iterations), np.array(values)


def smooth_curve(values: np.ndarray, window: int = 10) -> np.ndarray:
    """移动平均平滑"""
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_budget_comparison(
    budget_dir: str,
    budget_name: str,
    output_path: str,
    smooth_window: int = 10,
):
    """为单个 budget level 生成对比图"""
    
    # 扫描所有 variant
    variants_data = {}
    
    for variant_name in os.listdir(budget_dir):
        variant_path = os.path.join(budget_dir, variant_name)
        if not os.path.isdir(variant_path):
            continue
        
        metrics_path = os.path.join(variant_path, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"[WARNING] No metrics.json in {variant_path}")
            continue
        
        data = load_metrics(metrics_path)
        if len(data) == 0:
            print(f"[WARNING] Empty metrics in {variant_path}")
            continue
        
        variants_data[variant_name] = data
        print(f"[INFO] Loaded {variant_name}: {len(data)} iterations")
    
    if len(variants_data) == 0:
        print(f"[ERROR] No valid variants found in {budget_dir}")
        return
    
    # 创建 2×2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Full Spectrum Ablation: {budget_name}", fontsize=16, fontweight='bold')
    
    # 子图 1: Success Rate
    ax = axes[0, 0]
    for variant_name, data in variants_data.items():
        if variant_name not in VARIANT_CONFIG:
            continue
        
        iters, values = extract_series(data, "success_rate", default=0.0)
        if len(iters) == 0:
            continue
        
        cfg = VARIANT_CONFIG[variant_name]
        smoothed = smooth_curve(values, smooth_window)
        ax.plot(iters[:len(smoothed)], smoothed, label=cfg["name"], 
                color=cfg["color"], linestyle=cfg["linestyle"], 
                marker=cfg["marker"], markevery=max(1, len(smoothed)//10),
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Success Rate (Higher is Better)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 子图 2: Feasible Success Rate
    ax = axes[0, 1]
    for variant_name, data in variants_data.items():
        if variant_name not in VARIANT_CONFIG:
            continue
        
        iters, values = extract_series(data, "feasible_success_rate", default=0.0)
        if len(iters) == 0:
            continue
        
        cfg = VARIANT_CONFIG[variant_name]
        smoothed = smooth_curve(values, smooth_window)
        ax.plot(iters[:len(smoothed)], smoothed, label=cfg["name"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], markevery=max(1, len(smoothed)//10),
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Feasible & Success Rate", fontsize=11)
    ax.set_title("Feasible Success Rate (Key Metric)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 子图 3: Energy Cost vs Budget
    ax = axes[1, 0]
    for variant_name, data in variants_data.items():
        if variant_name not in VARIANT_CONFIG:
            continue
        
        iters, values = extract_series(data, "avg_cost_energy", default=0.0)
        if len(iters) == 0:
            continue
        
        cfg = VARIANT_CONFIG[variant_name]
        smoothed = smooth_curve(values, smooth_window)
        ax.plot(iters[:len(smoothed)], smoothed, label=cfg["name"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], markevery=max(1, len(smoothed)//10),
                linewidth=2, markersize=6, alpha=0.8)
    
    # 添加 budget 线
    if len(variants_data) > 0:
        first_data = next(iter(variants_data.values()))
        if len(first_data) > 0 and "budget_energy" in first_data[0]:
            budget_e = first_data[0]["budget_energy"]
            ax.axhline(y=budget_e, color='red', linestyle='--', linewidth=2, 
                      label=f'Budget={budget_e:.2f}', alpha=0.7)
    
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Energy Cost (per-step)", fontsize=11)
    ax.set_title("Energy Cost vs Budget", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 子图 4: Load Cost vs Budget
    ax = axes[1, 1]
    for variant_name, data in variants_data.items():
        if variant_name not in VARIANT_CONFIG:
            continue
        
        iters, values = extract_series(data, "avg_cost_load", default=0.0)
        if len(iters) == 0:
            continue
        
        cfg = VARIANT_CONFIG[variant_name]
        smoothed = smooth_curve(values, smooth_window)
        ax.plot(iters[:len(smoothed)], smoothed, label=cfg["name"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], markevery=max(1, len(smoothed)//10),
                linewidth=2, markersize=6, alpha=0.8)
    
    # 添加 budget 线
    if len(variants_data) > 0:
        first_data = next(iter(variants_data.values()))
        if len(first_data) > 0 and "budget_load" in first_data[0]:
            budget_l = first_data[0]["budget_load"]
            ax.axhline(y=budget_l, color='red', linestyle='--', linewidth=2,
                      label=f'Budget={budget_l:.2f}', alpha=0.7)
    
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Load Cost (per-step)", fontsize=11)
    ax.set_title("Load Cost vs Budget", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def generate_summary_table(benchmark_root: str, output_path: str):
    """生成汇总表格（最终性能）"""
    
    rows = []
    
    for budget_name in os.listdir(benchmark_root):
        budget_path = os.path.join(benchmark_root, budget_name)
        if not os.path.isdir(budget_path):
            continue
        
        budget_display = BUDGET_NAMES.get(budget_name, budget_name)
        
        for variant_name in os.listdir(budget_path):
            variant_path = os.path.join(budget_path, variant_name)
            if not os.path.isdir(variant_path):
                continue
            
            metrics_path = os.path.join(variant_path, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            
            data = load_metrics(metrics_path)
            if len(data) == 0:
                continue
            
            # 取最后 10 个 iteration 的平均值
            last_n = 10
            recent_data = data[-last_n:] if len(data) >= last_n else data
            
            avg_success_rate = np.mean([d.get("success_rate", 0.0) for d in recent_data])
            avg_feas_success = np.mean([d.get("feasible_success_rate", 0.0) for d in recent_data])
            avg_cost_e = np.mean([d.get("avg_cost_energy", 0.0) for d in recent_data])
            avg_cost_l = np.mean([d.get("avg_cost_load", 0.0) for d in recent_data])
            
            variant_display = VARIANT_CONFIG.get(variant_name, {}).get("name", variant_name)
            
            rows.append({
                "Budget": budget_display,
                "Variant": variant_display,
                "Success Rate": f"{avg_success_rate:.2%}",
                "Feasible Success": f"{avg_feas_success:.2%}",
                "Avg Energy Cost": f"{avg_cost_e:.4f}",
                "Avg Load Cost": f"{avg_cost_l:.4f}",
            })
    
    # 保存为 CSV
    import csv
    with open(output_path, 'w', newline='') as f:
        if len(rows) > 0:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"[SAVED] Summary table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Full Spectrum Ablation Results")
    parser.add_argument("benchmark_root", type=str, help="Root directory (e.g., outputs/final_benchmark_20260115_123456)")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")
    
    args = parser.parse_args()
    
    benchmark_root = args.benchmark_root
    
    if not os.path.exists(benchmark_root):
        print(f"[ERROR] Directory not found: {benchmark_root}")
        return
    
    # 创建输出目录
    plot_dir = os.path.join(benchmark_root, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating Full Spectrum Comparison Plots")
    print("=" * 70)
    
    # 为每个 budget level 生成对比图
    for budget_name in sorted(os.listdir(benchmark_root)):
        budget_path = os.path.join(benchmark_root, budget_name)
        if not os.path.isdir(budget_path) or budget_name == "comparison_plots":
            continue
        
        budget_display = BUDGET_NAMES.get(budget_name, budget_name)
        output_file = os.path.join(plot_dir, f"comparison_{budget_name}.png")
        
        print(f"\n[PLOT] {budget_display}")
        plot_budget_comparison(
            budget_path,
            budget_display,
            output_file,
            smooth_window=args.smooth,
        )
    
    # 生成汇总表格
    summary_path = os.path.join(plot_dir, "summary_table.csv")
    print(f"\n[SUMMARY] Generating performance table...")
    generate_summary_table(benchmark_root, summary_path)
    
    print("\n" + "=" * 70)
    print("Plotting completed!")
    print(f"Output directory: {plot_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
