# 路径对比可视化工具使用说明

## 功能概述

`compare_routes_over_time.py` 用于对比同一个训练运行中不同时期 checkpoint 在相同起终点下的路径选择差异。

**核心特性：**
- ✅ 自动查找多个 checkpoint（best_fsr.pt, best_feasible.pt, best_tail.pt）
- ✅ 固定 seed 确保相同起终点
- ✅ 支持多 episode rollout（验证稳定性）
- ✅ 生成单独 + 总览对比图
- ✅ 显示每条路径的长度、能耗、负载

---

## 快速开始

### 基础用法

```bash
python compare_routes_over_time.py \
    --run_dir outputs/four_group_ablation_20260121/A_multi_critic_adaptive_seed0_v2 \
    --seed 0 \
    --episodes 3 \
    --deterministic True
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--run_dir` | str | **必需** | 包含 checkpoints 和 config.json 的运行目录 |
| `--checkpoints` | list | `["best_fsr.pt", "best_feasible.pt", "best_tail.pt"]` | 要对比的 checkpoint 文件名 |
| `--seed` | int | `0` | 环境重置种子（确保相同起终点） |
| `--episodes` | int | `3` | 每个 checkpoint rollout 的 episode 数 |
| `--deterministic` | str | `"True"` | 是否使用确定性策略（True/False） |
| `--out_name` | str | `"route_compare.png"` | 总览对比图文件名 |
| `--device` | str | `"cpu"` | PyTorch 设备 |

---

## 输出说明

### 1. 单个 Checkpoint 图
每个 checkpoint 生成一张独立图：
- **文件名**: `{checkpoint_name}_routes.png`
- **布局**: 1行2列（左：拥堵热力图+路径，右：能耗热力图+路径）
- **包含**: 同一 checkpoint 的多条 episode 轨迹（不同颜色区分）

**示例输出:**
```
outputs/.../best_fsr_routes.png
outputs/.../best_feasible_routes.png
outputs/.../best_tail_routes.png
```

### 2. 总览对比图
所有 checkpoints 的纵向对比：
- **文件名**: `route_compare.png`（或自定义 `--out_name`）
- **布局**: N行2列（N=checkpoint数量）
- **每行**: 一个 checkpoint 的路径对比（左拥堵，右能耗）

**示例:**
```
┌─────────────────────────────────────────┐
│ best_fsr.pt - Load/Congestion | Energy  │
├─────────────────────────────────────────┤
│ best_feasible.pt - Load/Conge | Energy  │
├─────────────────────────────────────────┤
│ best_tail.pt - Load/Congesti  | Energy  │
└─────────────────────────────────────────┘
```

---

## 终端输出示例

```
======================================================================
Run Directory: outputs/four_group_ablation_20260121/A_multi_critic_adaptive_seed0_v2
Found 3 checkpoint(s): ['best_fsr.pt', 'best_feasible.pt', 'best_tail.pt']
Seed: 0 | Episodes: 3 | Deterministic: True
======================================================================

--- Processing: best_fsr.pt ---
  Episode 1: ✓ len=12, mean_energy=1.350, mean_load=0.480
  Episode 2: ✓ len=12, mean_energy=1.350, mean_load=0.480
  Episode 3: ✓ len=12, mean_energy=1.350, mean_load=0.480
✅ Saved: outputs/.../best_fsr_routes.png

--- Processing: best_feasible.pt ---
  Episode 1: ✓ len=14, mean_energy=1.200, mean_load=0.450
  Episode 2: ✓ len=14, mean_energy=1.200, mean_load=0.450
  Episode 3: ✓ len=14, mean_energy=1.200, mean_load=0.450
✅ Saved: outputs/.../best_feasible_routes.png

--- Processing: best_tail.pt ---
  Episode 1: ✓ len=13, mean_energy=1.280, mean_load=0.460
  Episode 2: ✓ len=13, mean_energy=1.280, mean_load=0.460
  Episode 3: ✓ len=13, mean_energy=1.280, mean_load=0.460
✅ Saved: outputs/.../best_tail_routes.png

======================================================================
✅ All done! Output saved to:
   - Comparison: outputs/.../route_compare.png
   - best_fsr.pt: outputs/.../best_fsr_routes.png
   - best_feasible.pt: outputs/.../best_feasible_routes.png
   - best_tail.pt: outputs/.../best_tail_routes.png
======================================================================
```

---

## 高级用法

### 1. 自定义 Checkpoint 列表
```bash
python compare_routes_over_time.py \
    --run_dir outputs/my_run \
    --checkpoints checkpoint_iter100.pt checkpoint_iter500.pt checkpoint_final.pt \
    --episodes 5
```

### 2. 多种子验证（不同起终点）
```bash
# 对比不同 seed 下的路径选择
for seed in 0 1 2; do
    python compare_routes_over_time.py \
        --run_dir outputs/my_run \
        --seed $seed \
        --out_name route_compare_seed${seed}.png
done
```

### 3. 随机策略对比
```bash
python compare_routes_over_time.py \
    --run_dir outputs/my_run \
    --deterministic False \
    --episodes 10
```

---

## 依赖项

脚本复用 `visualize_rollout.py` 的函数，需要以下模块：
- `torch`
- `numpy`
- `matplotlib`
- `grid_env.py` 及相关 wrapper

---

## 常见问题

### Q: 提示 "No checkpoints found"
**A**: 确保 `--run_dir` 中包含至少一个 checkpoint 文件（默认查找 best_fsr.pt、best_feasible.pt、best_tail.pt、checkpoint_final.pt）

### Q: 不同 episode 的路径完全相同
**A**: 确认 `--deterministic True` 且 policy 已收敛。若需要多样性，使用 `--deterministic False` 或增大 `--seed` 范围。

### Q: 起终点不一致
**A**: 检查环境配置中的 `start_goal_mode`。若为 "random"，即使 seed 相同，每次 reset 可能产生不同起终点。建议使用 "rect" 模式。

### Q: 内存不足
**A**: 减少 `--episodes` 数量或使用单独的 checkpoint 文件（避免一次加载多个大模型）

---

## 实现细节

### 复用的函数（来自 visualize_rollout.py）
- `load_config_from_dir()`: 加载 config.json
- `make_env_from_config()`: 根据配置构建环境
- `inject_obs_stats()`: 冻结 observation normalization 统计
- `load_agent()`: 加载 checkpoint 到 agent
- `select_action()`: 策略采样/确定性选择
- `_get_action_mask()`: 获取合法动作掩码
- `find_cost_wrapper()`: 查找 GridCostWrapper（获取地图）

### 新增功能
- `find_checkpoints()`: 自动查找可用 checkpoint
- `rollout_single_episode()`: 执行单次 rollout 并收集统计
- `plot_single_checkpoint_routes()`: 为单个 checkpoint 绘制路径对比
- `plot_comparison_grid()`: 生成总览对比图（纵向排列）

---

## 典型应用场景

### 1. 训练过程诊断
观察不同训练阶段的路径选择偏好：
- early: 可能更激进（短路径，高 cost）
- middle: 开始平衡长度与 cost
- late: 收敛到可行路径

### 2. 约束权衡可视化
对比 best_fsr（高可行率）vs best_feasible（满足约束）vs best_tail（尾部风险）的路径差异

### 3. Policy 稳定性验证
多 episode rollout 验证策略是否稳定（deterministic=True 时应完全相同）

---

## 扩展建议

可进一步扩展为：
- 支持视频输出（逐帧动画）
- 添加交互式 Plotly 可视化
- 集成到训练脚本自动生成
- 支持多 run_dir 对比（不同算法/超参）

---

## 联系支持

若遇到问题，请检查：
1. `config.json` 是否与 checkpoint 匹配
2. 环境配置参数是否与训练时一致
3. checkpoint 文件是否完整（包含 network_state_dict 和 obs_stats）
