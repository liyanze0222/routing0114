@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Verify hysteresis fix (fix1+fix2)
REM 
REM Setup:
REM   1. B_hysteresis_fixed: new code (fix1+fix2 integrated)
REM      - fix1: g_raw>0 use raw gap, g_raw<=0 use EMA
REM      - fix2: hysteresis correctly uses lambda_lr_energy/load (0.05)
REM 
REM Criteria:
REM   - iter>=600: avg_cost > budget should trigger lambda increase
REM   - "gap_ratio_* > 0 but gap_abs_*_used ~= 0" cases should drop
REM ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\hysteresis_fix_verify"

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

REM congestion: density controls area only (random=mask ratio, block=block area)
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"

REM load soft-threshold tau
set "LOAD_TAU=0.60"

REM start/goal
set "START_GOAL_MODE=rect"
set "START_RECT=0,1,0,1"
set "GOAL_RECT=6,7,6,7"

REM obs
set "INC_CONG_OBS=True"
set "CONG_R=1"
set "INC_E_OBS=True"
set "E_R=1"
set "OBS_RMS=True"

REM ---------------- CMDP / dual ----------------
set "E_BUDGET=0.10"
set "L_BUDGET=0.08"
set "GAP_MODE=absolute"

set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
REM 关键：确保 hysteresis 能吃到 0.05（改法2）
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"

REM hysteresis 参数
set "DEADBAND=0.005"
set "LR_DOWN_SCALE=0.20"
set "GAP_EMA_BETA=0.10"

REM diagnostics
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=True"
set "GRAD_INTERVAL=1"

echo ============================================================
echo Verify: hysteresis fix (fix1+fix2)
echo OUTDIR=%OUTDIR%
echo SEED=%SEED%  ITERS=%ITERS%  ENT=%ENT%
echo BUDGET: energy=%E_BUDGET%  load=%L_BUDGET%
echo LAMBDA_LR: energy=%LAMBDA_LR_E%  load=%LAMBDA_LR_L%
echo DEADBAND=%DEADBAND%  LR_DOWN_SCALE=%LR_DOWN_SCALE%
echo ============================================================
echo.

set "TAG=B_hysteresis_fixed"
set "RUN_TAG=%TAG%_ent%ENT%_seed%SEED%"

echo ------------------------------------------------------------
echo [%TAG%] Running hysteresis (fix1+fix2 integrated)
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
  --dual_update_mode hysteresis ^
  --dual_gap_ema_beta %GAP_EMA_BETA% ^
  --dual_deadband %DEADBAND% ^
  --dual_lr_down_scale %LR_DOWN_SCALE% ^
  --risk_factor %RISK_FACTOR% ^
  --initial_lambda_energy 0.0 ^
  --initial_lambda_load 0.0 ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_INTERVAL%

if errorlevel 1 (
  echo [ERROR] Run failed.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo Experiment completed!
echo 
echo Check:
echo 1. iter^>=600, review metrics.json:
echo    - When avg_cost_energy ^> 0.10, lambda_energy should increase
echo    - When avg_cost_load ^> 0.08, lambda_load should increase
echo 
echo 2. Compare old vs new log naming:
echo    Old: gap_energy_raw, gap_energy_ema, gap_energy_used
echo    New: gap_abs_energy, gap_ratio_energy, gap_abs_energy_ema, gap_abs_energy_used
echo 
echo 3. Count "gap_ratio_energy^>0 but gap_abs_energy_used~=0" cases
echo    Should drop significantly (especially iter^>=600)
echo ============================================================
pause
exit /b 0
