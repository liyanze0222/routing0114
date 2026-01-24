@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Re-run Group D only: Two-Critic + Aggregated Cost
REM ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\four_group_ablation_20260121"

REM -------------------- Common Env Config ---------------------
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
set "L_BUDGET=0.50"

REM -------------------- Start / Goal --------------------------
set "START_GOAL_MODE=rect"
set "START_RECT=0,1,0,1"
set "GOAL_RECT=6,7,6,7"

REM -------------------- Observations --------------------------
set "INC_CONG_OBS=True"
set "CONG_R=1"

set "INC_E_OBS=True"
set "E_R=1"
set "E_NORM=True"

set "OBS_RMS=True"

REM -------------------- PPO / Training ------------------------
set "ITERS=800"
set "BATCH=2048"
set "MINIBATCH=256"
set "EPOCHS=10"
set "SEED=0"

REM -------------------- Lagrangian / Dual Base ----------------
set "GAP_MODE=ratio"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"

REM Dual update cadence + stabilization
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "DUAL_UPDATE_MODE=standard"

REM Dual EMA parameters (explicit to prevent default pollution)
set "DUAL_GAP_BETA=0.10"
set "DUAL_CORR_BETA=0.05"

REM -------------------- Aggregated Cost 模式参数 --------------
set "AGG_W_ENERGY=1.0"
set "AGG_W_LOAD=1.0"
set "AGG_NORMALIZE=True"

echo.
echo ============================================================
echo Re-running Group D only (Two-Critic + Aggregated Cost)
echo Output Dir: %OUTDIR%
echo Seed=%SEED% ^| Iters=%ITERS% ^| Batch=%BATCH% ^| Minibatch=%MINIBATCH% ^| Epochs=%EPOCHS%
echo Budgets: Energy=%E_BUDGET% Load=%L_BUDGET% ^| LambdaMax=%LAMBDA_MAX%
echo Dual Mode: %DUAL_UPDATE_MODE% ^| UpdateFreq=%LAMBDA_UPDATE_FREQ%
echo Dual EMA: gap_beta=%DUAL_GAP_BETA% corr_beta=%DUAL_CORR_BETA%
echo ============================================================
echo.

REM ========== Group D: Two-Critic + Aggregated Cost ==========
echo.
echo [GROUP D] Two-Critic + Aggregated Cost (新增 baseline)
echo ============================================================
set "RUN_TAG=D_two_critic_aggcost_adaptive_seed%SEED%"
%PY% %ENTRY% --output_dir %OUTDIR% --seed %SEED% --total_iters %ITERS% --batch_size %BATCH% --minibatch_size %MINIBATCH% --update_epochs %EPOCHS% --grid_size %GRID% --max_steps %MAX_STEPS% --success_reward %SUCCESS_R% --step_penalty %STEP_P% --energy_high_cost %E_HIGH_COST% --energy_high_density %E_HIGH_DENS% --congestion_pattern %CONG_PATTERN% --congestion_density %CONG_DENS% --load_cost_scale %LOAD_SCALE% --start_goal_mode %START_GOAL_MODE% --start_rect %START_RECT% --goal_rect %GOAL_RECT% --include_congestion_obs %INC_CONG_OBS% --congestion_patch_radius %CONG_R% --include_energy_obs %INC_E_OBS% --energy_patch_radius %E_R% --energy_obs_normalize %E_NORM% --obs_rms %OBS_RMS% --cost_critic_mode aggregated --agg_cost_w_energy %AGG_W_ENERGY% --agg_cost_w_load %AGG_W_LOAD% --agg_cost_normalize_by_budget %AGG_NORMALIZE% --use_lagrange True --lambda_gap_mode %GAP_MODE% --lambda_lr_energy %LAMBDA_LR_E% --lambda_lr_load %LAMBDA_LR_L% --lambda_update_freq %LAMBDA_UPDATE_FREQ% --lambda_max %LAMBDA_MAX% --dual_update_mode %DUAL_UPDATE_MODE% --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --risk_factor %RISK_FACTOR% --initial_lambda_energy 0.0 --initial_lambda_load 0.0 --dual_gap_ema_beta %DUAL_GAP_BETA% --dual_corr_ema_beta %DUAL_CORR_BETA% --run_tag "%RUN_TAG%"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo DONE. Group D completed successfully!
echo Results saved: %OUTDIR%\%RUN_TAG%
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Group D run failed. Check the traceback above.
pause
exit /b 1
