import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ================= 配置区 =================
# 1. 你的 Scheme 1 metrics.json 路径
PATH_SCHEME_1 = "outputs/final_pivot_margin_vs_risk_20260119/Scheme1_Margin_L0.50_seed0/metrics.json"

# 2. 你的 Baseline (Standard) 路径 (如果有的话，没有就留空 "")
# 如果填了，脚本会自动画对比图；没填就只画 Scheme 1 的独角戏
PATH_BASELINE = "" 

# 3. 画图参数
SMOOTH_WEIGHT = 0.95  # 平滑力度 (0.90 - 0.99)
LINE_WIDTH = 2.5      # 主线宽
RAW_ALPHA = 0.15      # 原始噪点的透明度 (很淡)
# =========================================

def load_data(path, label):
    if not path: return pd.DataFrame()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['Label'] = label
        return df
    except FileNotFoundError:
        print(f"⚠️ 找不到文件: {path}，跳过该曲线。")
        return pd.DataFrame()

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_main_results():
    # 1. 加载数据
    df_main = load_data(PATH_SCHEME_1, "Ours (Virtual Budget)")
    df_base = load_data(PATH_BASELINE, "Baseline (Standard)")
    
    # 2. 设置画板
    sns.set_theme(style="whitegrid", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义颜色
    color_ours = "#1E88E5"   # 谷歌蓝
    color_base = "#FFC107"   # 琥珀色 (如果对比的话)
    color_limit = "#D81B60"  # 警戒红

    # ==========================================
    # 子图 1: Load (拥堵控制) - 核心卖点
    # ==========================================
    ax1 = axes[0]
    ax1.set_title("Precise Load Control", fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Load Cost")
    
    # --- 画 Scheme 1 ---
    if not df_main.empty:
        raw_load = df_main['avg_cost_load'].values
        smooth_load = smooth(raw_load, SMOOTH_WEIGHT)
        iters = df_main['iteration'].values
        
        # 1. 画背景噪点 (Raw)
        ax1.plot(iters, raw_load, color=color_ours, alpha=RAW_ALPHA, linewidth=1)
        # 2. 画平滑曲线 (Smooth)
        ax1.plot(iters, smooth_load, color=color_ours, linewidth=LINE_WIDTH, label="Ours (Target 0.50)")
        
        # 3. 标注最终收敛值
        final_val = smooth_load[-1]
        ax1.text(iters[-1]+10, final_val, f"{final_val:.3f}", color=color_ours, fontweight='bold', va='center')

    # --- 画 Baseline (可选) ---
    if not df_base.empty:
        base_load_smooth = smooth(df_base['avg_cost_load'].values, SMOOTH_WEIGHT)
        ax1.plot(df_base['iteration'], base_load_smooth, color=color_base, linewidth=LINE_WIDTH, linestyle="--", label="Baseline")

    # --- 画阈值线 (关键!) ---
    # 1. 真实考核线 (0.60)
    ax1.axhline(y=0.60, color=color_limit, linestyle="-", linewidth=2, alpha=0.8, label="Test Limit (0.60)")
    # 2. 训练目标线 (0.50)
    ax1.axhline(y=0.50, color=color_ours, linestyle=":", linewidth=2, alpha=0.8, label="Train Target (0.50)")
    
    # 填充安全区 (Safety Margin)
    ax1.axhspan(0.50, 0.60, color='green', alpha=0.05, label="Safety Margin")

    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # ==========================================
    # 子图 2: Energy (能耗代价) - 诚实展示
    # ==========================================
    ax2 = axes[1]
    ax2.set_title("Energy Trade-off", fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Energy Cost")

    # --- 画 Scheme 1 ---
    if not df_main.empty:
        raw_energy = df_main['avg_cost_energy'].values
        smooth_energy = smooth(raw_energy, SMOOTH_WEIGHT)
        
        ax2.plot(iters, raw_energy, color=color_ours, alpha=RAW_ALPHA, linewidth=1)
        ax2.plot(iters, smooth_energy, color=color_ours, linewidth=LINE_WIDTH, label="Ours")
        
    # --- 画 Baseline (可选) ---
    if not df_base.empty:
        base_energy_smooth = smooth(df_base['avg_cost_energy'].values, SMOOTH_WEIGHT)
        ax2.plot(df_base['iteration'], base_energy_smooth, color=color_base, linewidth=LINE_WIDTH, linestyle="--", label="Baseline")

    # --- 画预算线 ---
    ax2.axhline(y=1.35, color=color_limit, linestyle="--", linewidth=2, alpha=0.6, label="Budget (1.35)")
    
    ax2.legend(loc='lower right')

    # ==========================================
    # 保存
    # ==========================================
    plt.tight_layout()
    sns.despine() # 去掉上方和右方的边框，显得很干净
    plt.savefig("main_result_scheme1.png", dpi=300)
    print("✅ 美化图已生成: main_result_scheme1.png")

if __name__ == "__main__":
    plot_main_results()