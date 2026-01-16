@echo off
REM Quick test for curriculum learning (50 iterations)
setlocal EnableDelayedExpansion

python train_grid_structured_lagrangian.py ^
    --seed 0 ^
    --total_iters 50 ^
    --batch_size 1024 ^
    --minibatch_size 256 ^
    --update_epochs 5 ^
    --grid_size 8 ^
    --congestion_pattern block ^
    --include_congestion_obs true ^
    --congestion_patch_radius 2 ^
    --include_energy_obs true ^
    --energy_patch_radius 1 ^
    --energy_budget 1.35 ^
    --load_budget 999.0 ^
    --load_cost_scale 5.0 ^
    --use_curriculum true ^
    --curriculum_start_load_budget 3.0 ^
    --curriculum_end_load_budget 0.9 ^
    --curriculum_iters 40 ^
    --use_lagrange true ^
    --lambda_lr_energy 0.002 ^
    --lambda_lr_load 0.002 ^
    --cost_critic_mode separate ^
    --output_dir outputs ^
    --run_tag test_curriculum ^
    --log_interval 5

echo.
echo Test completed. Check outputs_test_curriculum/metrics.json
echo Run: python -c "import json; data=json.load(open('outputs_test_curriculum/metrics.json')); print('Iteration | Load Budget | Success Rate'); [print(f'{d[\"iteration\"]:3d}      | {d[\"budget_load\"]:.4f}      | {d[\"success_rate\"]:.2%%}') for d in data if d['iteration'] %% 10 == 0]"

endlocal
