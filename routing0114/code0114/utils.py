# utils.py
"""
工具函数模块：
- set_seed: 设置随机种子
- MetricsLogger: 训练指标记录器
- plot_training_curves: 绘制训练曲线
- make_output_dir: 创建输出目录（支持 run_tag 和自动递增）
"""

import os
# 解决 Windows + Anaconda 下 OpenMP 库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from typing import Dict, List, Any, Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置全局随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(base_dir: str, run_tag: Optional[str] = None) -> str:
    """
    创建输出目录，支持 run_tag 和自动递增后缀。

    Args:
        base_dir: 基础输出目录
        run_tag: 可选的运行标签

    Returns:
        final_dir: 最终输出目录路径

    行为：
    - 如果 run_tag 为 None 或空字符串，使用 base_dir
    - 如果 run_tag 不为空，使用 os.path.join(base_dir, run_tag)
    - 如果目录已存在，自动追加递增后缀（_v2, _v3, ...）
    """
    # 构建目录名
    if run_tag:
        target_dir = os.path.join(base_dir, run_tag)
    else:
        target_dir = base_dir

    # 如果目录不存在，直接创建并返回
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    # 目录已存在，尝试添加递增后缀
    version = 2
    while True:
        versioned_dir = f"{target_dir}_v{version}"
        if not os.path.exists(versioned_dir):
            os.makedirs(versioned_dir, exist_ok=True)
            return versioned_dir
        version += 1
        # 防止无限循环
        if version > 1000:
            raise RuntimeError(f"Too many versions for directory: {target_dir}")


class MetricsLogger:
    """
    训练指标记录器。

    用法：
        logger = MetricsLogger()
        logger.log({"loss": 0.5, "reward": 10.0})
        logger.save("metrics.json")
    """

    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any]):
        """记录一条指标。"""
        self.data.append(metrics.copy())

    def get_data(self) -> List[Dict[str, Any]]:
        """获取所有记录的数据。"""
        return self.data

    def save(self, path: str):
        """保存到 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_column(self, key: str) -> List[Any]:
        """获取某一列的所有值。"""
        return [d.get(key) for d in self.data if key in d]


def plot_training_curves(
    data: List[Dict[str, Any]],
    save_path: str,
    figsize: tuple = (16, 16),
    best_iter: Optional[int] = None,
):
    """
    绘制训练曲线。

    Args:
        data: MetricsLogger.get_data() 返回的数据
        save_path: 图片保存路径
        figsize: 图片大小
        best_iter: 最优可行点的迭代数（如果提供，会在各子图中画一条绿色垂直虚线）

    生成的图包含：
    - Return / Length / Success Rate
    - Cost vs Budget (energy, load)
    - Lambda curves (energy, load)
    - Policy loss / Value loss / Entropy
    - Gap curves (energy, load) with 0 line
    - KKT residual curves (energy, load)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无需 GUI
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot")
        return

    if not data:
        print("Warning: no data to plot")
        return

    # 提取数据列
    def get_col(key):
        return [d.get(key) for d in data if key in d]

    iterations = get_col("iteration")
    avg_returns = get_col("avg_return")
    avg_lengths = get_col("avg_length")
    success_rates = get_col("success_rate")
    avg_cost_energy = get_col("avg_cost_energy")
    avg_cost_load = get_col("avg_cost_load")
    lambda_energy = get_col("lambda_energy")
    lambda_load = get_col("lambda_load")
    budget_energy = get_col("budget_energy")
    budget_load = get_col("budget_load")
    policy_loss = get_col("policy_loss")
    value_loss = get_col("value_loss")
    entropy = get_col("entropy")
    # 新增：gap 和 KKT 残差
    gap_energy = get_col("gap_energy")
    gap_load = get_col("gap_load")
    kkt_energy = get_col("kkt_energy")
    kkt_load = get_col("kkt_load")
    ema_gap_energy = get_col("ema_gap_energy")
    ema_gap_load = get_col("ema_gap_load")

    # 判断是否有 gap/KKT 数据
    has_gap_kkt = bool(gap_energy or gap_load or kkt_energy or kkt_load)

    # 根据是否有 gap/KKT 数据决定布局
    if has_gap_kkt:
        fig, axes = plt.subplots(3, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # 定义一个辅助函数，用于在子图上画 best_iter 垂直线
    def add_best_iter_line(ax):
        """在子图上添加 best_iter 垂直虚线（绿色）"""
        if best_iter is not None:
            ax.axvline(x=best_iter, color="green", linestyle="--",
                       linewidth=1.5, alpha=0.7, label=f"Best Feasible (iter={best_iter})")

    # ========== 1. Return ==========
    ax = axes[0, 0]
    if avg_returns:
        ax.plot(iterations, avg_returns, label="Avg Return", color="blue")
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Return")
        ax.set_title("Average Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ========== 2. Length & Success Rate ==========
    ax = axes[0, 1]
    if avg_lengths:
        ax.plot(iterations, avg_lengths, label="Avg Length", color="green")
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Episode Length")
        ax.set_title("Episode Length")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 在同一图上画 success rate（右轴）
        if success_rates:
            ax2 = ax.twinx()
            ax2.plot(iterations, success_rates, label="Success Rate",
                     color="orange", linestyle="--")
            ax2.set_ylabel("Success Rate")
            ax2.legend(loc="upper right")

    # ========== 3. Cost (energy) vs Budget ==========
    ax = axes[0, 2]
    if avg_cost_energy:
        ax.plot(iterations, avg_cost_energy, label="Avg Cost (energy)", color="red")
        if budget_energy:
            ax.axhline(y=budget_energy[0], color="red", linestyle="--",
                       alpha=0.7, label=f"Budget ({budget_energy[0]:.3f})")
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("Energy Cost vs Budget")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ========== 4. Cost (load) vs Budget ==========
    ax = axes[1, 0]
    if avg_cost_load:
        ax.plot(iterations, avg_cost_load, label="Avg Cost (load)", color="purple")
        if budget_load:
            ax.axhline(y=budget_load[0], color="purple", linestyle="--",
                       alpha=0.7, label=f"Budget ({budget_load[0]:.3f})")
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("Load Cost vs Budget")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ========== 5. Lambda curves ==========
    ax = axes[1, 1]
    if lambda_energy:
        ax.plot(iterations, lambda_energy, label="Lambda (energy)", color="red")
    if lambda_load:
        ax.plot(iterations, lambda_load, label="Lambda (load)", color="purple")
    add_best_iter_line(ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lambda")
    ax.set_title("Lagrange Multipliers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 6. Loss & Entropy ==========
    ax = axes[1, 2]
    if policy_loss:
        ax.plot(iterations, policy_loss, label="Policy Loss", color="blue")
    if value_loss:
        ax.plot(iterations, value_loss, label="Value Loss", color="green")
    add_best_iter_line(ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Losses")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Entropy 在右轴
    if entropy:
        ax2 = ax.twinx()
        ax2.plot(iterations, entropy, label="Entropy", color="orange", linestyle="--")
        ax2.set_ylabel("Entropy")
        ax2.legend(loc="upper right")

    # ========== 7-9. Gap 和 KKT 残差（仅当有数据时） ==========
    if has_gap_kkt:
        # ========== 7. Gap (energy & load) ==========
        ax = axes[2, 0]
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        if gap_energy:
            ax.plot(iterations[:len(gap_energy)], gap_energy,
                    label="Gap (energy)", color="red", alpha=0.7)
            if ema_gap_energy:
                ax.plot(iterations[:len(ema_gap_energy)], ema_gap_energy,
                        label="EMA Gap (energy)", color="red", linestyle="--", linewidth=2)
        if gap_load:
            ax.plot(iterations[:len(gap_load)], gap_load,
                    label="Gap (load)", color="purple", alpha=0.7)
            if ema_gap_load:
                ax.plot(iterations[:len(ema_gap_load)], ema_gap_load,
                        label="EMA Gap (load)", color="purple", linestyle="--", linewidth=2)
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gap (cost - budget)")
        ax.set_title("Constraint Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ========== 8. KKT Residual ==========
        ax = axes[2, 1]
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        if kkt_energy:
            ax.plot(iterations[:len(kkt_energy)], kkt_energy,
                    label="KKT (energy)", color="red")
        if kkt_load:
            ax.plot(iterations[:len(kkt_load)], kkt_load,
                    label="KKT (load)", color="purple")
        add_best_iter_line(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("λ × gap")
        ax.set_title("KKT Residual (λ × gap)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ========== 9. 空白或备用 ==========
        ax = axes[2, 2]
        # 绘制 gap 的滑动平均 + λ 的关系
        if gap_energy and lambda_energy:
            ax.scatter(gap_energy, lambda_energy[:len(gap_energy)],
                       c=range(len(gap_energy)), cmap="viridis", alpha=0.5, s=10, label="energy")
        if gap_load and lambda_load:
            ax.scatter(gap_load, lambda_load[:len(gap_load)],
                       c=range(len(gap_load)), cmap="plasma", alpha=0.5, s=10, label="load")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.set_xlabel("Gap")
        ax.set_ylabel("Lambda")
        ax.set_title("Gap vs Lambda (color=time)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_grid_route_viz(
    congestion_map,
    energy_map,
    traj,
    start,
    goal,
    save_path: str,
    title: str | None = None,
):
    """
    保存网格可视化：
    - 左：congestion/load heatmap + 路线
    - 右：energy heatmap + 路线
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] matplotlib import failed: {e}")
        return

    if congestion_map is None or energy_map is None:
        print("[viz] missing congestion_map/energy_map, skip.")
        return

    # traj: [(r,c), ...]
    xs = [c for r, c in traj]
    ys = [r for r, c in traj]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, grid, name in [
        (axes[0], congestion_map, "Load / Congestion"),
        (axes[1], energy_map, "Energy"),
    ]:
        im = ax.imshow(grid, origin="upper")
        ax.set_title(name)

        # route
        ax.plot(xs, ys, linewidth=2)
        ax.scatter([start[1]], [start[0]], marker="s", s=80)   # start
        ax.scatter([goal[1]], [goal[0]], marker="*", s=120)    # goal

        # cosmetics
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, linewidth=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_episode_stats(episode_infos, energy_budget, load_budget):
    """
    处理一批完成的 episode 信息，计算 Safety Gym 风格指标。
    
    Args:
        episode_infos: 包含 episode 统计信息的字典列表
        energy_budget: 能源约束预算
        load_budget: 负载约束预算
        
    Returns:
        stats: 包含统计指标的字典
    """
    success_count = 0
    total_count = len(episode_infos)
    
    if total_count == 0:
        return {}
    
    # 临时列表
    energy_costs_all = []
    load_costs_all = []
    energy_costs_success = []  # [新增] 仅成功样本
    load_costs_success = []    # [新增] 仅成功样本
    
    feasible_success_count = 0  # [新增] 真可行计数
    
    for info in episode_infos:
        # 提取基础数据
        ep_energy = info.get('episode_cost_energy', 0.0)
        ep_load = info.get('episode_cost_load', 0.0)
        is_success = info.get('success', False)  # 假设环境返回了 success 布尔值
        
        energy_costs_all.append(ep_energy)
        load_costs_all.append(ep_load)
        
        if is_success:
            success_count += 1
            energy_costs_success.append(ep_energy)
            load_costs_success.append(ep_load)
            
            # 判断"真可行"：必须成功 + 满足预算
            if ep_energy <= energy_budget and ep_load <= load_budget:
                feasible_success_count += 1
    
    # 计算统计量
    stats = {}
    
    # 基础统计
    stats['success_count'] = success_count
    stats['success_rate'] = success_count / total_count
    
    # 1. Success-Only Costs (关键改动 1.3)
    if success_count > 0:
        stats['avg_energy_success'] = np.mean(energy_costs_success)
        stats['avg_load_success'] = np.mean(load_costs_success)
        stats['std_energy_success'] = np.std(energy_costs_success)
        stats['std_load_success'] = np.std(load_costs_success)
    else:
        stats['avg_energy_success'] = float('nan')
        stats['avg_load_success'] = float('nan')
        stats['std_energy_success'] = float('nan')
        stats['std_load_success'] = float('nan')
    
    # 2. 真可行比例 (关键改动 1.2)
    stats['feasible_success_count'] = feasible_success_count
    stats['feasible_success_rate'] = feasible_success_count / total_count
    
    # 3. 全部 episode 的平均（用于对比）
    stats['avg_energy_all'] = np.mean(energy_costs_all)
    stats['avg_load_all'] = np.mean(load_costs_all)
    
    return stats
