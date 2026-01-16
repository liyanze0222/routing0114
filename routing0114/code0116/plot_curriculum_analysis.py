"""
plot_curriculum_analysis.py

分析课程学习训练曲线，生成对比图表。

Usage:
    python plot_curriculum_analysis.py outputs_curriculum_extreme
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(metrics_path):
    """加载 metrics.json 文件"""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def plot_curriculum_curves(metrics_data, output_path, curriculum_iters=600):
    """
    绘制课程学习分析图表
    
    Args:
        metrics_data: 训练指标列表
        output_path: 输出图片路径
        curriculum_iters: 课程学习衰减结束的迭代次数
    """
    # 提取数据
    iterations = [d['iteration'] for d in metrics_data]
    budget_load = [d['budget_load'] for d in metrics_data]
    success_rate = [d['success_rate'] for d in metrics_data]
    feasible_success_rate = [d.get('feasible_success_rate', 0) for d in metrics_data]
    avg_cost_load = [d['avg_cost_load'] for d in metrics_data]
    lambda_load = [d['lambda_load'] for d in metrics_data]
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curriculum Learning Analysis', fontsize=16, fontweight='bold')
    
    # ========== 子图 1: Curriculum Schedule ==========
    ax1 = axes[0, 0]
    ax1.plot(iterations, budget_load, 'b-', linewidth=2.5, label='Load Budget (Curriculum)')
    ax1.axvline(curriculum_iters, color='red', linestyle='--', linewidth=2, 
                label=f'Curriculum End (iter {curriculum_iters})')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Load Budget (scaled)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('(A) Curriculum Schedule', fontsize=13, fontweight='bold')
    
    # 标注三个阶段
    ax1.axvspan(0, curriculum_iters * 0.33, alpha=0.1, color='green', label='Early (Easy)')
    ax1.axvspan(curriculum_iters * 0.33, curriculum_iters, alpha=0.1, color='orange', label='Mid (Decay)')
    ax1.axvspan(curriculum_iters, max(iterations), alpha=0.1, color='red', label='Late (Hard)')
    
    # ========== 子图 2: Success Rates ==========
    ax2 = axes[0, 1]
    ax2.plot(iterations, success_rate, 'g-', linewidth=2, label='Success Rate')
    ax2.plot(iterations, feasible_success_rate, 'r--', linewidth=2, label='Feasible Success Rate')
    ax2.axvline(curriculum_iters, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Curriculum End')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(B) Performance Metrics', fontsize=13, fontweight='bold')
    
    # ========== 子图 3: Cost vs Budget ==========
    ax3 = axes[1, 0]
    ax3.plot(iterations, avg_cost_load, 'purple', linewidth=2, label='Avg Load Cost')
    ax3.plot(iterations, budget_load, 'b--', linewidth=1.5, label='Load Budget (Target)', alpha=0.7)
    ax3.axvline(curriculum_iters, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Load Cost (scaled)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('(C) Load Cost vs Budget', fontsize=13, fontweight='bold')
    
    # 填充可行区域（cost < budget）
    ax3.fill_between(iterations, 0, budget_load, alpha=0.1, color='green', label='Feasible Region')
    
    # ========== 子图 4: Lagrange Multiplier ==========
    ax4 = axes[1, 1]
    ax4.plot(iterations, lambda_load, 'orange', linewidth=2, label='Lambda (Load)')
    ax4.axvline(curriculum_iters, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Curriculum End')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Lambda Value', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('(D) Lagrange Multiplier Dynamics', fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")


def print_phase_statistics(metrics_data, curriculum_iters=600):
    """打印三个阶段的统计信息"""
    total_iters = len(metrics_data)
    
    # 定义三个阶段
    phase1_end = int(curriculum_iters * 0.33)
    phase2_end = curriculum_iters
    
    phase1 = [d for d in metrics_data if d['iteration'] <= phase1_end]
    phase2 = [d for d in metrics_data if phase1_end < d['iteration'] <= phase2_end]
    phase3 = [d for d in metrics_data if d['iteration'] > phase2_end]
    
    print("\n" + "=" * 70)
    print("Phase Statistics")
    print("=" * 70)
    
    for phase_name, phase_data, iter_range in [
        ("Early (Easy)", phase1, f"1-{phase1_end}"),
        ("Mid (Decay)", phase2, f"{phase1_end+1}-{phase2_end}"),
        ("Late (Hard)", phase3, f"{phase2_end+1}-{total_iters}")
    ]:
        if not phase_data:
            continue
        
        avg_success = np.mean([d['success_rate'] for d in phase_data])
        avg_feasible = np.mean([d.get('feasible_success_rate', 0) for d in phase_data])
        avg_budget = np.mean([d['budget_load'] for d in phase_data])
        avg_cost = np.mean([d['avg_cost_load'] for d in phase_data])
        avg_lambda = np.mean([d['lambda_load'] for d in phase_data])
        
        print(f"\n{phase_name} (Iter {iter_range}):")
        print(f"  Avg Load Budget:     {avg_budget:.3f}")
        print(f"  Avg Load Cost:       {avg_cost:.3f}")
        print(f"  Avg Success Rate:    {avg_success:.2%}")
        print(f"  Avg Feasible Rate:   {avg_feasible:.2%}")
        print(f"  Avg Lambda (Load):   {avg_lambda:.3f}")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot curriculum learning analysis")
    parser.add_argument("run_dir", type=str, help="Path to run directory (e.g., outputs_curriculum_extreme)")
    parser.add_argument("--curriculum_iters", type=int, default=600, 
                        help="Curriculum decay end iteration (default: 600)")
    args = parser.parse_args()
    
    # 加载数据
    metrics_path = os.path.join(args.run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"❌ Error: {metrics_path} not found!")
        return
    
    print(f"Loading metrics from: {metrics_path}")
    metrics_data = load_metrics(metrics_path)
    print(f"✅ Loaded {len(metrics_data)} iterations")
    
    # 绘制图表
    output_path = os.path.join(args.run_dir, "curriculum_analysis.png")
    plot_curriculum_curves(metrics_data, output_path, args.curriculum_iters)
    
    # 打印统计
    print_phase_statistics(metrics_data, args.curriculum_iters)
    
    # 最终指标
    final_metrics = metrics_data[-1]
    print("Final Performance (Last Iteration):")
    print(f"  Success Rate:        {final_metrics['success_rate']:.2%}")
    print(f"  Feasible Success:    {final_metrics.get('feasible_success_rate', 0):.2%}")
    print(f"  Load Cost:           {final_metrics['avg_cost_load']:.4f}")
    print(f"  Load Budget:         {final_metrics['budget_load']:.4f}")
    print(f"  Lambda (Load):       {final_metrics['lambda_load']:.4f}")
    print()


if __name__ == "__main__":
    main()
