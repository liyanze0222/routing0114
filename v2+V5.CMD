@echo off
setlocal enabledelayedexpansion

set "PY=python"
set "OUT=outputs\sat2_c2c_L0p900_20260116"

set "E_BUDGET=1.35"
set "L_BUDGET=0.900"

REM ========== Common args (keep identical to V0/V1 C2C benchmark) ==========
set "COMMON_ARGS=--grid_size 8 --max_steps 256 --congestion_pattern block --congestion_density 0.40 --success_reward 20.0 --step_penalty -1.0 --energy_high_cost 3.0 --energy_high_density 0.20 --include_congestion_obs True --congestion_patch_radius 1 --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True --total_iters 800 --batch_size 2048 --minibatch_size 256 --update_epochs 10 --hidden_dim 128 --seed 0 --lambda_gap_mode ratio --initial_lambda_energy 0 --initial_lambda_load 0 --load_cost_scale 5.0 --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --output_dir "%OUT%""

REM ========== Corner-to-Corner start/goal rectangles ==========
set "C2C_ARGS=--start_goal_mode rect --start_rect 0,1,0,1 --goal_rect 6,7,6,7"

if not exist "%OUT%" mkdir "%OUT%"

echo ============================
echo C2C Benchmark V2+V5: E=%E_BUDGET% L=%L_BUDGET% (scaled)
echo Output: %OUT%
echo ============================

REM ----------------------------
REM V2: shared cost critic (reward value still separate head)
REM ----------------------------
%PY% train_grid_structured_lagrangian.py %COMMON_ARGS% %C2C_ARGS% --run_tag "C2C_V2_sharedcost_dual_E%E_BUDGET%_L%L_BUDGET%_seed0" --use_lagrange True --cost_critic_mode shared --lambda_lr_energy 0.05 --lambda_lr_load 0.05

REM ----------------------------
REM V5: true single-head value (Vr, Ve, Vl from one head)
REM Requires: --value_head_mode shared_all implemented
REM ----------------------------
%PY% train_grid_structured_lagrangian.py %COMMON_ARGS% %C2C_ARGS% --run_tag "C2C_V5_true_singlehead_dual_E%E_BUDGET%_L%L_BUDGET%_seed0" --use_lagrange True --value_head_mode shared_all --lambda_lr_energy 0.05 --lambda_lr_load 0.05

echo Done.
endlocal
