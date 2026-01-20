import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ================= 配置区 =================
# 请填入你两个实验 metrics.json 的实际路径
PATH_EXP_A = "outputs/waterbed_effect_verification_20260120/ExpA_Squeeze_Load_L0.50/metrics.json"
PATH_EXP_B = "outputs/waterbed_effect_verification_20260120/ExpB_Squeeze_Energy_E1.20/metrics.json"

SMOOTH_WEIGHT = 0.9  # 平滑力度 (0.85 - 0.99)
LINE_WIDTH = 2.5     # 线宽
# =========================================

def load_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {path}")
        return pd.DataFrame()

def smooth(scalars, weight):
    """TensorBoard 风格的 EMA 平滑"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_paper_quality():
    # 1. 加载数据
    df_a = load_data(PATH_EXP_A)
    df_b = load_data(PATH_EXP_B)
    
    if df_a.empty or df_b.empty: return

    # 2. 数据平滑处理
    # Exp A (Squeeze Load)
    load_a = smooth(df_a['avg_cost_load'].values, SMOOTH_WEIGHT)
    energy_a = smooth(df_a['avg_cost_energy'].values, SMOOTH_WEIGHT)
    
    # Exp B (Squeeze Energy)
    load_b = smooth(df_b['avg_cost_load'].values, SMOOTH_WEIGHT)
    energy_b = smooth(df_b['avg_cost_energy'].values, SMOOTH_WEIGHT)
    
    iterations = df_a['iteration']

    # 3. 开始画图 (设置顶会风格)
    sns.set_theme(style="white", font_scale=1.2) # White 背景更适合论文打印
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左 Y 轴：Load (拥堵) ---
    color_load = "#D81B60" # 玫红色代表 Load
    ax1.set_xlabel('Training Iterations', fontsize=14)
    ax1.set_ylabel('Load Cost (Congestion)', color=color_load, fontsize=14, fontweight='bold')
    
    # 画 Exp A 的 Load (实线)
    l1, = ax1.plot(iterations, load_a, color=color_load, linewidth=LINE_WIDTH, 
             label="Exp A: Load (Tight L=0.5)")
    # 画 Exp B 的 Load (虚线)
    l2, = ax1.plot(iterations, load_b, color=color_load, linewidth=LINE_WIDTH, linestyle="--", alpha=0.6,
             label="Exp B: Load (Loose L=0.6)")
    
    ax1.tick_params(axis='y', labelcolor=color_load)
    ax1.grid(True, alpha=0.3) # 淡淡的网格

    # --- 右 Y 轴：Energy (能耗) ---
    ax2 = ax1.twinx()  # 共享 X 轴
    color_energy = "#1E88E5" # 蓝色代表 Energy
    ax2.set_ylabel('Energy Cost (Distance)', color=color_energy, fontsize=14, fontweight='bold')
    
    # 画 Exp A 的 Energy (虚线 - 因为这里 Energy 是配角)
    l3, = ax2.plot(iterations, energy_a, color=color_energy, linewidth=LINE_WIDTH, linestyle="--", alpha=0.6,
             label="Exp A: Energy (Loose E=1.35)")
    # 画 Exp B 的 Energy (实线 - 因为这里 Energy 是主角)
    l4, = ax2.plot(iterations, energy_b, color=color_energy, linewidth=LINE_WIDTH, 
             label="Exp B: Energy (Tight E=1.20)")
    
    ax2.tick_params(axis='y', labelcolor=color_energy)

    # --- 合并图例 ---
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    # 把图例放在中间上方，不遮挡曲线
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=2, frameon=False, fontsize=11)

    plt.title("The Waterbed Effect: Pareto Trade-off Verification", fontsize=16, y=1.15)
    plt.tight_layout()
    plt.savefig("waterbed_paper_plot.png", dpi=300)
    print("✅ 论文级美图已生成: waterbed_paper_plot.png")

if __name__ == "__main__":
    plot_paper_quality()