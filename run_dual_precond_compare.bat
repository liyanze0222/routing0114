@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM 3-way dual comparison with fully-explicit baseline params
REM Modes: standard(vanilla) vs decorrelated vs precond
REM Goal: apples-to-apples (no implicit defaults)
REM ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\final_pivot_margin_vs_risk_20260119_fixed"

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
set "USE_LAGR=True"
set "GAP_MODE=ratio"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"

REM Dual update cadence + smoothing (fixed across all modes)
set "LAMBDA_UPDATE_FREQ=1"

REM IMPORTANT: clamp lambdas to a sane value to avoid numeric blowups
REM (use a large but finite max; 10~50 is usually plenty. Choose 10 for stability.)
set "LAMBDA_MAX=10.0"

REM decorrelated/precond path uses these EMA stats; keep explicit
set "DUAL_GAP_BETA=0.10"
set "DUAL_CORR_BETA=0.05"

REM -------------------- Preconditioner Params -----------------
REM Updated for numerical stability: lower clip, higher eps, reduced strength
set "PRECOND_EPS=0.05"
set "PRECOND_CLIP=2.0"
set "PRECOND_STRENGTH=0.3"
set "PRECOND_USE_EMA=True"

REM -------------------- Shared Arg Block ----------------------
REM Inline args (avoid tokenization issues)

echo.
echo ============================================================
echo Output Dir: %OUTDIR%
echo Seed=%SEED% ^| Iters=%ITERS% ^| Batch=%BATCH% ^| Minibatch=%MINIBATCH% ^| Epochs=%EPOCHS%
echo Budgets: Energy=%E_BUDGET% Load=%L_BUDGET% ^| LambdaMax=%LAMBDA_MAX% ^| UpdateFreq=%LAMBDA_UPDATE_FREQ%
echo Dual EMA: gap_beta=%DUAL_GAP_BETA% corr_beta=%DUAL_CORR_BETA%
echo ============================================================
echo.

REM -------------------- Run 1: Vanilla/Standard ----------------
echo [RUN] Vanilla dual (standard)
set "RUN_TAG=Scheme1_Margin_L0.50_seed0_fixed_vanilla"
%PY% %ENTRY% --output_dir %OUTDIR% --seed %SEED% --total_iters %ITERS% --batch_size %BATCH% --minibatch_size %MINIBATCH% --update_epochs %EPOCHS% --grid_size %GRID% --max_steps %MAX_STEPS% --success_reward %SUCCESS_R% --step_penalty %STEP_P% --energy_high_cost %E_HIGH_COST% --energy_high_density %E_HIGH_DENS% --congestion_pattern %CONG_PATTERN% --congestion_density %CONG_DENS% --load_cost_scale %LOAD_SCALE% --start_goal_mode %START_GOAL_MODE% --start_rect %START_RECT% --goal_rect %GOAL_RECT% --include_congestion_obs %INC_CONG_OBS% --congestion_patch_radius %CONG_R% --include_energy_obs %INC_E_OBS% --energy_patch_radius %E_R% --energy_obs_normalize %E_NORM% --obs_rms %OBS_RMS% --use_lagrange %USE_LAGR% --lambda_gap_mode %GAP_MODE% --lambda_lr_energy %LAMBDA_LR_E% --lambda_lr_load %LAMBDA_LR_L% --lambda_update_freq %LAMBDA_UPDATE_FREQ% --lambda_max %LAMBDA_MAX% --dual_gap_ema_beta %DUAL_GAP_BETA% --dual_corr_ema_beta %DUAL_CORR_BETA% --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --risk_factor %RISK_FACTOR% --run_tag "%RUN_TAG%" --dual_update_mode standard
if errorlevel 1 goto :fail

REM -------------------- Run 2: Decorrelated --------------------
echo.
echo [RUN] Decorrelated dual
set "RUN_TAG=Scheme1_Margin_L0.50_seed0_fixed_decor"
%PY% %ENTRY% --output_dir %OUTDIR% --seed %SEED% --total_iters %ITERS% --batch_size %BATCH% --minibatch_size %MINIBATCH% --update_epochs %EPOCHS% --grid_size %GRID% --max_steps %MAX_STEPS% --success_reward %SUCCESS_R% --step_penalty %STEP_P% --energy_high_cost %E_HIGH_COST% --energy_high_density %E_HIGH_DENS% --congestion_pattern %CONG_PATTERN% --congestion_density %CONG_DENS% --load_cost_scale %LOAD_SCALE% --start_goal_mode %START_GOAL_MODE% --start_rect %START_RECT% --goal_rect %GOAL_RECT% --include_congestion_obs %INC_CONG_OBS% --congestion_patch_radius %CONG_R% --include_energy_obs %INC_E_OBS% --energy_patch_radius %E_R% --energy_obs_normalize %E_NORM% --obs_rms %OBS_RMS% --use_lagrange %USE_LAGR% --lambda_gap_mode %GAP_MODE% --lambda_lr_energy %LAMBDA_LR_E% --lambda_lr_load %LAMBDA_LR_L% --lambda_update_freq %LAMBDA_UPDATE_FREQ% --lambda_max %LAMBDA_MAX% --dual_gap_ema_beta %DUAL_GAP_BETA% --dual_corr_ema_beta %DUAL_CORR_BETA% --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --risk_factor %RISK_FACTOR% --run_tag "%RUN_TAG%" --dual_update_mode decorrelated
if errorlevel 1 goto :fail

REM -------------------- Run 3: Preconditioned ------------------
echo.
echo [RUN] Preconditioned dual
set "RUN_TAG=Scheme1_Margin_L0.50_seed0_fixed_precond"
%PY% %ENTRY% --output_dir %OUTDIR% --seed %SEED% --total_iters %ITERS% --batch_size %BATCH% --minibatch_size %MINIBATCH% --update_epochs %EPOCHS% --grid_size %GRID% --max_steps %MAX_STEPS% --success_reward %SUCCESS_R% --step_penalty %STEP_P% --energy_high_cost %E_HIGH_COST% --energy_high_density %E_HIGH_DENS% --congestion_pattern %CONG_PATTERN% --congestion_density %CONG_DENS% --load_cost_scale %LOAD_SCALE% --start_goal_mode %START_GOAL_MODE% --start_rect %START_RECT% --goal_rect %GOAL_RECT% --include_congestion_obs %INC_CONG_OBS% --congestion_patch_radius %CONG_R% --include_energy_obs %INC_E_OBS% --energy_patch_radius %E_R% --energy_obs_normalize %E_NORM% --obs_rms %OBS_RMS% --use_lagrange %USE_LAGR% --lambda_gap_mode %GAP_MODE% --lambda_lr_energy %LAMBDA_LR_E% --lambda_lr_load %LAMBDA_LR_L% --lambda_update_freq %LAMBDA_UPDATE_FREQ% --lambda_max %LAMBDA_MAX% --dual_gap_ema_beta %DUAL_GAP_BETA% --dual_corr_ema_beta %DUAL_CORR_BETA% --energy_budget %E_BUDGET% --load_budget %L_BUDGET% --risk_factor %RISK_FACTOR% --run_tag "%RUN_TAG%" --dual_update_mode precond --dual_precond_eps %PRECOND_EPS% --dual_precond_clip %PRECOND_CLIP% --dual_precond_strength %PRECOND_STRENGTH% --dual_precond_use_ema_stats %PRECOND_USE_EMA%
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo DONE. Runs saved under %OUTDIR%
echo   - %OUTDIR%\%RUN_TAG%
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] One of the runs failed. Check the traceback above.
pause
exit /b 1
