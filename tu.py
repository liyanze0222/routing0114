import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_maps():
    # ================= 配置参数 (来自您的 bat 文件) =================
    N = 8
    
    # Energy 参数
    E_BASE = 1.0
    E_HIGH = 3.0
    E_DENS = 0.20
    
    # Load 参数
    L_SCALE = 5.0
    # Block 模式模拟：在地图中间生成一个矩形块
    # 模拟 congestion_density 0.40 (约占 40% 面积)
    
    # 起终点 (Rect模式)
    START_RECT = (0, 1, 0, 1) # r_min, r_max, c_min, c_max
    GOAL_RECT = (6, 7, 6, 7)
    
    # ================= 生成数据 =================
    np.random.seed(42) # 固定种子方便复现

    # 1. 生成 Energy Map
    energy_map = np.full((N, N), E_BASE)
    # 随机撒点生成高耗能区
    mask = np.random.rand(N, N) < E_DENS
    energy_map[mask] = E_HIGH
    # 确保起终点没有高耗能 (通常为了公平性，可选)
    # energy_map[0:2, 0:2] = E_BASE 
    # energy_map[6:8, 6:8] = E_BASE

    # 2. 生成 Load Map (Block Pattern)
    load_map = np.zeros((N, N))
    # 创建一个大约占 40% 的块 (5x5 = 25格 ≈ 40% of 64)
    # 放在中心偏一点的位置，强迫 Agent 绕路
    r_start, r_end = 2, 6 # 行 2,3,4,5
    c_start, c_end = 2, 6 # 列 2,3,4,5
    load_map[r_start:r_end, c_start:c_end] = 1.0 # 原始值
    load_map = load_map * L_SCALE # 缩放后 Cost (0 或 5)

    # ================= 绘图 =================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 左图：Energy Map ---
    sns.heatmap(energy_map, ax=axes[0], annot=True, fmt=".1f", 
                cmap="Blues", cbar_kws={'label': 'Energy Cost'},
                linewidths=1, linecolor='gray')
    axes[0].set_title(f"Energy Map (Random High Cost)\nBase={E_BASE}, High={E_HIGH}", fontsize=14, fontweight='bold')
    
    # --- 右图：Load Map ---
    sns.heatmap(load_map, ax=axes[1], annot=True, fmt=".1f", 
                cmap="Reds", cbar_kws={'label': 'Load Cost'},
                linewidths=1, linecolor='gray')
    axes[1].set_title(f"Load Map (Block Pattern)\nScale={L_SCALE}", fontsize=14, fontweight='bold')

    # --- 标注起终点 ---
    for ax in axes:
        # 标注起点 (S) - 左上角 2x2 区域
        ax.add_patch(plt.Rectangle((0, 0), 2, 2, fill=False, edgecolor='green', lw=4, linestyle='--'))
        ax.text(0.5, 0.5, "Start", color='green', fontweight='bold', ha='center', va='center', fontsize=12)
        
        # 标注终点 (G) - 右下角 2x2 区域
        ax.add_patch(plt.Rectangle((6, 6), 2, 2, fill=False, edgecolor='blue', lw=4, linestyle='--'))
        ax.text(7, 7, "Goal", color='blue', fontweight='bold', ha='center', va='center', fontsize=12)
        
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_maps()