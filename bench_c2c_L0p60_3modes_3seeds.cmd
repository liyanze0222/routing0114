@echo off
setlocal enabledelayedexpansion

:: =========================
:: Multi-seed suite (C2C rect-rect high pressure)
:: 3 modes x 3 seeds = 9 runs
:: =========================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"

:: ---- Output dir (CHANGE THIS) ----
set "OUTDIR=outputs\sat2_c2c_L0p60_dual_3modes_3seeds_20260117_b"

:: ---- Task/Budget ----
set "E_BUDGET=1.35"
set "L_BUDGET=0.60"
set "LOAD_SCALE=5.0"

:: ---- Core env ----
set "GRID=8"
set "MAX_STEPS=256"
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"
set "SUCCESS_R=20.0"
set "STEP_P=-1.0"
set "E_HIGH_COST=3.0"
set "E_HIGH_DENS=0.20"

:: C2C via rect sampling (exactly matches your current config.json)
set "START_GOAL_MODE=rect"
set "START_RECT=0,1,0,1"
set "GOAL_RECT=6,7,6,7"

:: ---- Obs ----
set "INC_CONG_OBS=True"
set "CONG_R=1"
set "INC_E_OBS=True"
set "E_R=1"
set "E_NORM=True"

:: ---- PPO ----
set "ITERS=800"
set "BATCH=2048"
set "MINIBATCH=256"
set "EPOCHS=10"
set "HID=128"

:: ---- Lagrange / Dual ----
set "USE_LAGR=True"
set "GAP_MODE=ratio"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "INIT_LAMBDA_E=0.0"
set "INIT_LAMBDA_L=0.0"

:: dual anti-osc params (match your config)
set "DUAL_GAP_EMA=0.1"
set "DUAL_DEADBAND=0.02"
set "DUAL_CORR_EMA=0.05"
set "DUAL_LR_DOWN_SCALE=0.2"

:: ---- Modes / Seeds ----
set "MODES=standard hysteresis decorrelated"
set "SEEDS=0 1 2"

echo Output dir: %OUTDIR%
echo Running: %MODES%  x  seeds: %SEEDS%
echo.

for %%M in (%MODES%) do (
  for %%S in (%SEEDS%) do (
    set "RUN_TAG=C2C_V0_dual_%%M_E%E_BUDGET%_L%L_BUDGET%_seed%%S"
    echo ============================================================
    echo [RUN] mode=%%M seed=%%S tag=!RUN_TAG!
    echo ============================================================

    %PY% %ENTRY% ^
      --output_dir "%OUTDIR%" ^
      --run_tag "!RUN_TAG!" ^
      --seed %%S ^
      --total_iters %ITERS% ^
      --batch_size %BATCH% ^
      --minibatch_size %MINIBATCH% ^
      --update_epochs %EPOCHS% ^
      --hidden_dim %HID% ^
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
      --use_lagrange %USE_LAGR% ^
      --lambda_gap_mode %GAP_MODE% ^
      --lambda_lr_energy %LAMBDA_LR_E% ^
      --lambda_lr_load %LAMBDA_LR_L% ^
      --initial_lambda_energy %INIT_LAMBDA_E% ^
      --initial_lambda_load %INIT_LAMBDA_L% ^
      --dual_update_mode %%M ^
      --dual_gap_ema_beta %DUAL_GAP_EMA% ^
      --dual_deadband %DUAL_DEADBAND% ^
      --dual_corr_ema_beta %DUAL_CORR_EMA% ^
      --dual_lr_down_scale %DUAL_LR_DOWN_SCALE% ^
      --energy_budget %E_BUDGET% ^
      --load_budget %L_BUDGET%

    if errorlevel 1 (
      echo [ERROR] Run failed: mode=%%M seed=%%S
      exit /b 1
    )
  )
)

echo.
echo DONE. All runs finished under: %OUTDIR%
endlocal
