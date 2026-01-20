@echo off
setlocal enabledelayedexpansion

set "PY=python"
set "OUT=outputs\sat2_c2c_L0p60_dual_3modes_20260116"

set "E_BUDGET=1.35"
set "L_BUDGET=0.60"

REM Dual controller knobs
set "DUAL_KNOBS= --dual_gap_ema_beta 0.10"
set "DUAL_KNOBS=!DUAL_KNOBS! --dual_deadband 0.02"
set "DUAL_KNOBS=!DUAL_KNOBS! --dual_lr_down_scale 0.20"
set "DUAL_KNOBS=!DUAL_KNOBS! --dual_corr_ema_beta 0.05"

REM Shared args (match your benchmark)
set "BASE_ARGS= --grid_size 8 --max_steps 256"
set "BASE_ARGS=!BASE_ARGS! --congestion_pattern block --congestion_density 0.40"
set "BASE_ARGS=!BASE_ARGS! --success_reward 20.0 --step_penalty -1.0"
set "BASE_ARGS=!BASE_ARGS! --energy_high_cost 3.0 --energy_high_density 0.20"
set "BASE_ARGS=!BASE_ARGS! --include_congestion_obs True --congestion_patch_radius 1"
set "BASE_ARGS=!BASE_ARGS! --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True"
set "BASE_ARGS=!BASE_ARGS! --total_iters 800 --batch_size 2048 --minibatch_size 256 --update_epochs 10"
set "BASE_ARGS=!BASE_ARGS! --hidden_dim 128 --seed 0"
set "BASE_ARGS=!BASE_ARGS! --use_lagrange True"
set "BASE_ARGS=!BASE_ARGS! --lambda_gap_mode ratio --lambda_lr_energy 0.05 --lambda_lr_load 0.05"
set "BASE_ARGS=!BASE_ARGS! --initial_lambda_energy 0 --initial_lambda_load 0"
set "BASE_ARGS=!BASE_ARGS! --cost_critic_mode separate"
set "BASE_ARGS=!BASE_ARGS! --load_cost_scale 5.0"
set "BASE_ARGS=!BASE_ARGS! --energy_budget %E_BUDGET%"
set "BASE_ARGS=!BASE_ARGS! --load_budget %L_BUDGET%"
set "BASE_ARGS=!BASE_ARGS! --start_goal_mode rect --start_rect 0,1,0,1 --goal_rect 6,7,6,7"
set "BASE_ARGS=!BASE_ARGS! --output_dir %OUT%"

set "STD_DIR=%OUT%\C2C_V0_dual_standard_E%E_BUDGET%_L%L_BUDGET%_seed0"

if not exist "%OUT%" mkdir "%OUT%"

echo ============================
echo C2C Dual 3 modes
echo E=%E_BUDGET%  L=%L_BUDGET% (scaled)
echo OUT=%OUT%
echo ============================

REM 1) standard (skip if already exists)
if exist "%STD_DIR%\metrics.json" (
	echo [skip] standard already exists at %STD_DIR%
) else (
	%PY% train_grid_structured_lagrangian.py !BASE_ARGS! --run_tag C2C_V0_dual_standard_E%E_BUDGET%_L%L_BUDGET%_seed0 --dual_update_mode standard
)

REM 2) hysteresis
%PY% train_grid_structured_lagrangian.py !BASE_ARGS! --run_tag C2C_V0_dual_hysteresis_E%E_BUDGET%_L%L_BUDGET%_seed0 --dual_update_mode hysteresis !DUAL_KNOBS!

REM 3) decorrelated
%PY% train_grid_structured_lagrangian.py !BASE_ARGS! --run_tag C2C_V0_dual_decorrelated_E%E_BUDGET%_L%L_BUDGET%_seed0 --dual_update_mode decorrelated !DUAL_KNOBS!

echo Done.
endlocal
