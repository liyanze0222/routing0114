# 固定 seeds 的 episode 评估

命令示例（一次跑 baseline / fixed_map / fixed_map_fixed_sg 三组）：

```bash
python evaluate/eval_fixed_seeds.py \
  --checkpoint_path outputs/final_safety_gym_benchmark/Benchmark_Hard_E1.35_L0.20_R20/best_feasible.pt \
  --num_episodes 100 \
  --seed_start 0 \
  --output_dir evaluate_outputs/step1_test \
  --deterministic True \
  --fix_env_seed True \
  --fix_start_goal True \
  --fixed_start 0,0 \
  --fixed_goal 7,7
```

参数说明（新增）:
- `--fix_env_seed`（默认 True）：固定地图采样；fixed_map* 组会复用首个地图。
- `--fix_start_goal`（默认 False）：允许固定起终点；在 fixed_map_fixed_sg 组中会固定（可指定 `--fixed_start/--fixed_goal`，否则沿用首次采样）。
- `--fixed_start` / `--fixed_goal`：可选，格式 `x,y`。
- `--eval_mode`：保留单组选择（默认 baseline），但脚本会自动跑三组输出到子目录。

输出：
- 每组子目录含 `episodes.csv`（含 seed/start/goal/map_hash、能耗/负载等）、`summary.json`（均值/方差、相关性、可行率、唯一地图 hash 等）。
- 散点图：`scatter_energy_vs_load_all.png` / `scatter_energy_vs_load_success.png`。
- 直方图：`hist_energy.png` / `hist_load.png`。
