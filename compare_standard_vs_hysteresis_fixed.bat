@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Standard vs Hysteresis (FIXED)
REM 
REM Groups:
REM   A. standard (baseline, no EMA)
REM   B. hysteresis_fixed (fix1+fix2)
REM 
REM Key differences:
REM   - B hysteresis now:
REM     * g_raw>0: use raw gap (immediate response to violation)
REM     * g_raw<=0: use EMA+deadband (slow decrease)
REM     * correctly uses lambda_lr_energy/load=0.05 (not 0.01)
REM ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\standard_vs_hysteresis_fixed"

REM ---------------- sweep knobs ----------------
set "SEED=0"
set "ITERS=800"
set "ENT=0.05"

REM ---------------- PPO knobs ----------------
set "LR=3e-4"
set "GAMMA=0.99"
set "GAE_LAMBDA=0.95"
set "CLIP=0.2"
set "VF_COEF=0.5"
set "ENT_COEF=%ENT%"
set "BATCH=4096"
set "MINIBATCH=256"
set "EPOCHS=10"

REM ---------------- env ----------------
set "GRID=8"
set "MAX_STEPS=256"
set "STEP_P=-1.0"
set "SUCCESS_R=10.0"
set "RANDOMIZE_MAPS=True"
set "ENERGY_HIGH_DENS=0.20"
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"
set "LOAD_TAU=0.60"
set "START_GOAL_MODE=rect"
set "START_RECT=0,1,0,1"
set "GOAL_RECT=6,7,6,7"
set "INC_CONG_OBS=True"
set "CONG_R=1"
set "INC_E_OBS=True"
set "E_R=1"
set "OBS_RMS=True"

REM ---------------- CMDP / dual ----------------
set "E_BUDGET=0.10"
set "L_BUDGET=0.05"
set "GAP_MODE=absolute"
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"

REM diagnostics
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=True"
set "GRAD_INTERVAL=1"

echo ============================================================
echo Standard vs Hysteresis (FIXED)
echo OUTDIR=%OUTDIR%
echo SEED=%SEED%  ITERS=%ITERS%  ENT=%ENT%
echo BUDGET: energy=%E_BUDGET%  load=%L_BUDGET%
echo LAMBDA_LR: energy=%LAMBDA_LR_E%  load=%LAMBDA_LR_L%
echo ============================================================
echo.

REM ---------- A: standard ----------
call :run_one "A_standard" "standard" "0.0"
if errorlevel 1 goto :fail

REM ---------- B: hysteresis (fixed) ----------
call :run_one "B_hysteresis_fixed" "hysteresis" "0.005"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo All experiments completed!
echo 
echo Key metrics to check (iter 600-800):
echo   1. lambda response: B should increase immediately when over budget
echo   2. Violation stats: B should have ~0 cases of "avg_cost^>budget but lambda stays"
echo   3. Log fields: B has new gap_abs_*, gap_ratio_* naming
echo ============================================================
pause
exit /b 0

:run_one
set "TAG=%~1"
set "DUAL_MODE=%~2"
set "DEADBAND=%~3"
set "RUN_TAG=%TAG%_ent%ENT%_seed%SEED%"

echo ------------------------------------------------------------
echo [%TAG%] dual_update_mode=%DUAL_MODE% dual_deadband=%DEADBAND%
echo run_tag=%RUN_TAG%
echo ------------------------------------------------------------

%PY% %ENTRY% ^
  --output_dir %OUTDIR% ^
  --run_tag "%RUN_TAG%" ^
  --seed %SEED% ^
  --total_iters %ITERS% ^
  --lr %LR% ^
  --gamma %GAMMA% ^
  --gae_lambda %GAE_LAMBDA% ^
  --clip_coef %CLIP% ^
  --value_coef %VF_COEF% ^
  --ent_coef %ENT_COEF% ^
  --batch_size %BATCH% ^
  --minibatch_size %MINIBATCH% ^
  --update_epochs %EPOCHS% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --success_reward %SUCCESS_R% ^
  --step_penalty %STEP_P% ^
  --energy_high_density %ENERGY_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --load_threshold %LOAD_TAU% ^
  --start_goal_mode %START_GOAL_MODE% ^
  --start_rect %START_RECT% ^
  --goal_rect %GOAL_RECT% ^
  --include_congestion_obs %INC_CONG_OBS% ^
  --congestion_patch_radius %CONG_R% ^
  --include_energy_obs %INC_E_OBS% ^
  --energy_patch_radius %E_R% ^
  --obs_rms %OBS_RMS% ^
  --randomize_maps_each_reset %RANDOMIZE_MAPS% ^
  --cost_critic_mode separate ^
  --use_lagrange True ^
  --energy_budget %E_BUDGET% ^
  --load_budget %L_BUDGET% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR_E% ^
  --lambda_lr_load %LAMBDA_LR_L% ^
  --lambda_update_freq %LAMBDA_UPDATE_FREQ% ^
  --lambda_max %LAMBDA_MAX% ^
  --dual_update_mode %DUAL_MODE% ^
  --risk_factor %RISK_FACTOR% ^
  --initial_lambda_energy 0.0 ^
  --initial_lambda_load 0.0 ^
  --dual_deadband %DEADBAND% ^
  --dual_lr_down_scale 0.20 ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_INTERVAL%

exit /b %errorlevel%

:fail
echo [ERROR] Run failed.
pause
exit /b 1
