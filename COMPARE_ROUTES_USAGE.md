# compare_routes_over_time.py 使用说明

## 增强功能概述

已实现两个重要增强功能：

### 增强 1: 自动/手动选择"变化最大 cell"
- 自动检测 best_tail vs best_fsr 访问概率变化最大的格子
- 同时统计两个 cell 的 hit rate
- 在差分图上标注两个关注点（cell1: cyan X，cell2: white circle）

### 增强 2: Hit-rate Timeline 分析
- 绘制 hit rate 随训练迭代变化的曲线
- 支持 glob 模式查找迭代 checkpoint
- 输出 CSV 数据便于后续分析

---

## 使用示例

### 1. Visitmap 模式（增强版，自动检测 cell2）

```powershell
python compare_routes_over_time.py `
    --mode visitmap `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --topk_cells 3 `
    --deterministic True
```

**参数说明：**
- `--cell_of_interest 2,2`: 主要关注的格子（必经点候选）
- `--cell_of_interest2 auto`: 自动检测变化最大的格子
  - 也可以手动指定：`--cell_of_interest2 5,5`
  - 或禁用：`--cell_of_interest2 none`
- `--topk_cells 3`: 自动检测时报告前3个变化最大的格子

**输出：**
- `route_diag/best_fsr_visitmap.png`: best_fsr checkpoint 的访问热图
- `route_diag/best_feasible_visitmap.png`: best_feasible checkpoint 的访问热图
- `route_diag/best_tail_visitmap.png`: best_tail checkpoint 的访问热图
- `route_diag/visitmap_diff_tail_vs_fsr.png`: 差分热图（标注 cell1 和 cell2）

**控制台输出示例：**
```
--- Auto-detecting cell2 (max change cell) ---
   ✅ Auto-selected cell2: (4, 3) (delta=+0.3250)
   Top-3 cells with max |delta|:
      #1: (4, 3) -> delta=+0.3250
      #2: (3, 4) -> delta=-0.2100
      #3: (5, 2) -> delta=+0.1875

--- Re-computing stats with cell2=(4, 3) ---
  best_fsr.pt: Hit@(2,2)=85.50% | Hit@(4,3)=12.00% | Success=92.00% | AvgLen=12.3
  best_tail.pt: Hit@(2,2)=86.00% | Hit@(4,3)=45.00% | Success=93.50% | AvgLen=12.1
```

---

### 2. Timeline 模式（如果有迭代 checkpoint）

```powershell
python compare_routes_over_time.py `
    --mode timeline `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --timeline_glob "checkpoint_iter_*.pt" `
    --timeline_stride 10 `
    --timeline_episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --deterministic True
```

**参数说明：**
- `--timeline_glob "checkpoint_iter_*.pt"`: 匹配迭代 checkpoint 的 glob 模式
  - 也可以是 `"model_iter_*.pt"` 或其他命名模式
- `--timeline_stride 10`: 每隔 10 个 checkpoint 采样一个（减少计算量）
- `--timeline_episodes 200`: 每个 checkpoint rollout 多少条轨迹

**输出：**
- `route_diag/timeline_hit_lambda.png`: hit rate 随迭代变化的曲线图
  - 蓝线：Hit@cell1（主要关注点）
  - 绿线：Hit@cell2（自动检测的变化最大点）
  - 红虚线：Success Rate
- `route_diag/timeline_hit_lambda.csv`: 数值数据（便于后续分析）

**控制台输出示例：**
```
Found 50 checkpoint(s) (after stride=10)
Auto-detecting cell2 by comparing best_tail vs best_fsr...
   ✅ Auto-selected cell2: (4, 3) (delta=+0.325)

Processing: checkpoint_iter_1000.pt (iter=1000)...
   Hit@(2,2)=78.50%, Success=85.00%
Processing: checkpoint_iter_2000.pt (iter=2000)...
   Hit@(2,2)=82.00%, Success=89.50%
...

✅ Timeline data saved: route_diag/timeline_hit_lambda.csv
✅ Timeline plot saved: route_diag/timeline_hit_lambda.png
```

---

### 3. Timeline 退化模式（无迭代 checkpoint，仅用3个标准 checkpoint）

如果运行目录下没有迭代 checkpoint，timeline 模式会自动退化到使用 best_fsr/best_feasible/best_tail：

```powershell
python compare_routes_over_time.py `
    --mode timeline `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --timeline_episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --deterministic True
```

**说明：**
- 自动检测无迭代 checkpoint 时，会使用 best_fsr.pt, best_feasible.pt, best_tail.pt
- 仅绘制3个点的折线图（用于快速验证）
- 提示用户：要获得完整曲线，需在训练时开启 save_interval

---

## 输出文件结构

所有输出统一保存到 `<run_dir>/route_diag/` 目录下：

```
<run_dir>/
├── route_diag/
│   ├── best_fsr_visitmap.png          # 单个 checkpoint 热图
│   ├── best_feasible_visitmap.png
│   ├── best_tail_visitmap.png
│   ├── visitmap_diff_tail_vs_fsr.png  # 差分图（标注 cell1 和 cell2）
│   ├── timeline_hit_lambda.png        # timeline 曲线图
│   └── timeline_hit_lambda.csv        # timeline 数值数据
├── best_fsr.pt
├── best_feasible.pt
├── best_tail.pt
└── config.json
```

---

## 关键技术细节

### Cell2 自动检测算法
1. 计算 `delta = visit_prob(best_tail) - visit_prob(best_fsr)`
2. 排除 start 和 goal 格子
3. 选择 `|delta|` 最大的格子作为 cell2
4. 支持 topk 报告多个候选

### 统计方式
- **访问概率**：基于所有 episodes 计算（包括失败的）
- **Hit Rate**：某个 cell 被访问过的 episodes 比例
- **Success Rate**：到达 goal 的 episodes 比例
- **Avg Length**：仅统计成功 episodes 的平均步数

### 兼容性
- 自动处理 gym/gymnasium 的 step() 返回值差异
- 支持 rect 模式的固定起终点
- 兼容 PyTorch 2.6 的 weights_only 参数

---

## 故障排除

### 问题：所有 episodes 都走满 256 步
**原因：** grid_env.py 没有在 info 中设置 success 字段
**解决：** 已修复（在 step() 中添加 info["success"] 和 info["reached_goal"]）

### 问题：自动检测 cell2 失败
**原因：** best_tail.pt 和 best_fsr.pt 不存在，或访问概率差异太小
**解决：** 使用手动指定 `--cell_of_interest2 r,c`

### 问题：Timeline 模式找不到 checkpoint
**原因：** glob 模式不匹配
**解决：** 检查实际文件名，调整 `--timeline_glob` 参数

---

## 最小可用命令

### Visitmap + 自动 cell2
```powershell
python compare_routes_over_time.py --mode visitmap --run_dir <RUN_DIR> --episodes 200 --cell_of_interest 2,2 --cell_of_interest2 auto --deterministic True
```

### Timeline（如果有迭代 checkpoint）
```powershell
python compare_routes_over_time.py --mode timeline --run_dir <RUN_DIR> --timeline_glob "checkpoint_iter_*.pt" --timeline_stride 10 --timeline_episodes 200 --cell_of_interest 2,2 --cell_of_interest2 auto --deterministic True
```

### Timeline 退化模式（仅3个点）
```powershell
python compare_routes_over_time.py --mode timeline --run_dir <RUN_DIR> --timeline_episodes 200 --cell_of_interest 2,2 --cell_of_interest2 auto --deterministic True
```
