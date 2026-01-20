@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: FINAL PIVOT EXPERIMENTS: Margin vs Risk (Vanilla dual)
:: Baseline ablation: same config as run_final_pivot.bat Scheme1, only dual_update_mode -> standard
:: ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\final_pivot_margin_vs_risk_20260119_fixed"

:: ---- Common Env Config (C2C High Pressure) ----
set "GRID=8"
set "MAX_STEPS=256"
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"
set "SUCCESS_R=20.0"
set "STEP_P=-1.0"
set "E_HIGH_COST=3.0"
set "E_HIGH_DENS=0.20"
set "LOAD_SCALE=5.0"
set "E_BUDGET=1.35"

:: ---- Start/Goal ----
set "START_GOAL_MODE=rect"
set "START_RECT=0,1,0,1"
set "GOAL_RECT=6,7,6,7"

:: ---- Obs ----
set "INC_CONG_OBS=True"
set "CONG_R=1"
set "INC_E_OBS=True"
set "E_R=1"
set "E_NORM=True"
set "OBS_RMS=True"

:: ---- PPO / Dual Base ----
set "ITERS=800"
set "BATCH=2048"
set "MINIBATCH=256"
set "EPOCHS=10"
set "USE_LAGR=True"
set "GAP_MODE=ratio"
set "LAMBDA_LR=0.05"
set "DUAL_MODE=standard" 
:: Vanilla dual for ablation (only change)

:: ============================================================
:: EXPERIMENT 1: Virtual Budget (Scheme 1) - Vanilla Dual
:: Only change: dual_update_mode=standard, run_tag renamed
:: ============================================================
set "RUN_TAG_1=Scheme1_Margin_L0.50_seed0_fixed_vanilla"
set "L_BUDGET_MARGIN=0.50"
set "RISK_FACTOR_0=0.0"

echo.
echo [START] Scheme 1 (Vanilla dual): Train with L=%L_BUDGET_MARGIN% ...
%PY% %ENTRY% ^
  --output_dir "%OUTDIR%" ^
  --run_tag "%RUN_TAG_1%" ^
  --seed 0 ^
  --total_iters %ITERS% ^
  --batch_size %BATCH% ^
  --minibatch_size %MINIBATCH% ^
  --update_epochs %EPOCHS% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --success_reward %SUCCESS_R% ^
  --step_penalty %STEP_P% ^
  --energy_high_cost %E_HIGH_COST% ^
  --energy_high_density %E_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --load_cost_scale %LOAD_SCALE% ^
  --start_goal_mode %START_GOAL_MODE% ^
  --start_rect %START_RECT% ^
  --goal_rect %GOAL_RECT% ^
  --include_congestion_obs %INC_CONG_OBS% ^
  --congestion_patch_radius %CONG_R% ^
  --include_energy_obs %INC_E_OBS% ^
  --energy_patch_radius %E_R% ^
  --energy_obs_normalize %E_NORM% ^
  --obs_rms %OBS_RMS% ^
  --use_lagrange %USE_LAGR% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR% ^
  --lambda_lr_load %LAMBDA_LR% ^
  --dual_update_mode %DUAL_MODE% ^
  --energy_budget %E_BUDGET% ^
  --load_budget %L_BUDGET_MARGIN% ^
  --risk_factor %RISK_FACTOR_0%

echo.
echo ============================================================
echo DONE. Vanilla dual ablation saved under run_tag=%RUN_TAG_1% in %OUTDIR%
echo ============================================================
pause
