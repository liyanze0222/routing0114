@echo off
setlocal enabledelayedexpansion

set "PY=python"
set "OUT=outputs\sat2_extreme补充_20260116"

set "E_BUDGET=1.35"
set "L_BUDGET=0.900"

:: Flattened args to avoid caret/line-continuation issues when %COMMON_ARGS% is expanded.
set "COMMON_ARGS=--grid_size 8 --max_steps 256 --congestion_pattern block --congestion_density 0.40 --success_reward 20.0 --step_penalty -1.0 --energy_high_cost 3.0 --energy_high_density 0.20 --include_congestion_obs True --congestion_patch_radius 1 --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True --total_iters 800 --batch_size 2048 --minibatch_size 256 --update_epochs 10 --hidden_dim 128 --seed 0 --lambda_gap_mode ratio --initial_lambda_energy 0 --initial_lambda_load 0 --load_cost_scale 5.0 --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --output_dir "%OUT%""

if not exist "%OUT%" mkdir "%OUT%"

REM ----------------------------
REM V3' Energy-only (keep budgets, disable load constraint)
REM ----------------------------
%PY% train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "V3p_energy_only_fixed_L%L_BUDGET%_seed0" --use_lagrange True --cost_critic_mode separate --lambda_lr_energy 0.05 --lambda_lr_load 0 --initial_lambda_load 0

REM ----------------------------
REM V4' Load-only (keep budgets, disable energy constraint)
REM ----------------------------
%PY% train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "V4p_load_only_fixed_L%L_BUDGET%_seed0" --use_lagrange True --cost_critic_mode separate --lambda_lr_energy 0 --lambda_lr_load 0.05 --initial_lambda_energy 0

REM ----------------------------
REM V5 True single-head (reward+costs share one value head)
REM NOTE: needs new arg --value_head_mode shared_all
REM ----------------------------
%PY% train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "V5_true_singlehead_dual_L%L_BUDGET%_seed0" --use_lagrange True --value_head_mode shared_all --lambda_lr_energy 0.05 --lambda_lr_load 0.05

echo Done.
endlocal
