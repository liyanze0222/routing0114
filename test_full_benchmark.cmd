@echo off
REM Quick test of the full benchmark pipeline (reduced iterations)

echo Testing Full Benchmark Pipeline (Quick Mode: 10 iterations)...

set OUTROOT=outputs\test_full_benchmark
if exist "%OUTROOT%" rd /s /q "%OUTROOT%"
mkdir "%OUTROOT%"

REM Test on Stable budget only, 2 variants only
echo Running V0 (Proposed)...
python train_grid_structured_lagrangian.py ^
  --total_iters 10 --batch_size 512 --log_interval 2 ^
  --grid_size 8 --max_steps 256 ^
  --energy_budget 1.35 --load_budget 1.025 --load_cost_scale 5.0 ^
  --use_lagrange True --cost_critic_mode separate ^
  --lambda_lr_energy 0.05 --lambda_lr_load 0.05 ^
  --congestion_pattern block --congestion_density 0.40 ^
  --energy_high_cost 3.0 --energy_high_density 0.20 ^
  --include_congestion_obs True --congestion_patch_radius 2 ^
  --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True ^
  --success_reward 20.0 --step_penalty -1.0 --seed 0 ^
  --output_dir "%OUTROOT%\budget_1.025_stable\v0_proposed"

echo Running V5 (Scalar)...
python train_grid_scalar.py ^
  --total_iters 10 --batch_size 512 --log_interval 2 ^
  --grid_size 8 --max_steps 256 ^
  --energy_weight 0.5 --load_weight 1.5 --load_cost_scale 5.0 ^
  --congestion_pattern block --congestion_density 0.40 ^
  --energy_high_cost 3.0 --energy_high_density 0.20 ^
  --include_congestion_obs True --congestion_patch_radius 2 ^
  --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True ^
  --success_reward 20.0 --step_penalty -1.0 --seed 0 ^
  --output_dir "%OUTROOT%\budget_1.025_stable\v5_scalar"

echo.
echo Testing plotting...
python plot_full_comparison.py "%OUTROOT%"

echo.
echo ============================================================
echo Test Complete!
echo Check outputs:
echo   - Metrics: %OUTROOT%\budget_1.025_stable\v*\metrics.json
echo   - Plots:   %OUTROOT%\comparison_plots\
echo   - Summary: %OUTROOT%\comparison_plots\summary_table.csv
echo ============================================================
pause
