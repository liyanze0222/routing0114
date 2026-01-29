@echo off
setlocal enabledelayedexpansion

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\ent_sweep_20260127"

REM sweep knobs
set "ENTS=0.05"
set "ITERS=800"
set "SEED=0"

REM env
set "GRID=8"
set "MAX_STEPS=256"
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"
set "SUCCESS_R=20.0"
set "STEP_P=-1.0"
set "E_HIGH_DENS=0.20"
set "LOAD_THRESHOLD=0.6"
set "RANDOMIZE_MAPS=True"

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

REM logging
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=True"
set "GRAD_INTERVAL=1"

REM ppo
set "BATCH=2048"
set "MINIBATCH=256"
set "EPOCHS=10"

REM budgets (你自己按当前实验设)
set "E_BUDGET=0.10"
set "L_BUDGET=0.08"

REM dual/lagrange
set "GAP_MODE=ratio"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "DUAL_UPDATE_MODE=standard"
set "DUAL_LR_DOWN_SCALE=1.0"
set "DUAL_DEADBAND=0.0"
set "DUAL_GAP_BETA=0.10"
set "DUAL_CORR_BETA=0.05"

echo OUTDIR=%OUTDIR%
echo ITERS=%ITERS% SEED=%SEED%
echo ENTS=%ENTS%
echo.

for %%E in (%ENTS%) do (
  set "ENT=%%E"

  REM reward-only baseline (dual standard, risk 0)
  set "DUAL_UPDATE_MODE=standard"
  set "DUAL_LR_DOWN_SCALE=1.0"
  set "DUAL_DEADBAND=0.0"
  set "RISK_FACTOR=0.0"
  call :run_one "reward_only" "False" "!ENT!" "R_reward_only"
  if errorlevel 1 goto :fail

  REM 实验1 A1：lagrange + standard dual
  set "DUAL_UPDATE_MODE=standard"
  set "DUAL_LR_DOWN_SCALE=1.0"
  set "DUAL_DEADBAND=0.0"
  set "RISK_FACTOR=0.0"
  call :run_one "lagrange" "True" "!ENT!" "A_lagrange_std"
  if errorlevel 1 goto :fail

  REM 实验1 A2：lagrange + hysteresis dual
  set "DUAL_UPDATE_MODE=hysteresis"
  set "DUAL_LR_DOWN_SCALE=0.2"
  set "DUAL_DEADBAND=0.02"
  set "RISK_FACTOR=0.0"
  call :run_one "lagrange" "True" "!ENT!" "A_lagrange_hyst"
  if errorlevel 1 goto :fail

  REM 实验2 B2：lagrange + risk on（B1 等同 A1 已跑）
  set "DUAL_UPDATE_MODE=standard"
  set "DUAL_LR_DOWN_SCALE=1.0"
  set "DUAL_DEADBAND=0.0"
  set "RISK_FACTOR=1.0"
  call :run_one "lagrange" "True" "!ENT!" "A_lagrange_risk1"
  if errorlevel 1 goto :fail
)

echo DONE.
pause
exit /b 0

:run_one
set "MODE=%~1"
set "USE_LAGRANGE=%~2"
set "ENT=%~3"
set "TAG_PREFIX=%~4"
set "RUN_TAG=%TAG_PREFIX%_ent%ENT%_seed%SEED%"

echo ------------------------------------------------------------
echo [%MODE%] use_lagrange=%USE_LAGRANGE% ent_coef=%ENT%
echo run_tag=%RUN_TAG%
echo ------------------------------------------------------------

%PY% %ENTRY% ^
  --output_dir %OUTDIR% ^
  --seed %SEED% ^
  --total_iters %ITERS% ^
  --batch_size %BATCH% ^
  --minibatch_size %MINIBATCH% ^
  --update_epochs %EPOCHS% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --success_reward %SUCCESS_R% ^
  --step_penalty %STEP_P% ^
  --energy_high_density %E_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --load_threshold %LOAD_THRESHOLD% ^
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
  --use_lagrange %USE_LAGRANGE% ^
  --ent_coef %ENT% ^
  --energy_budget %E_BUDGET% ^
  --load_budget %L_BUDGET% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR_E% ^
  --lambda_lr_load %LAMBDA_LR_L% ^
  --lambda_update_freq %LAMBDA_UPDATE_FREQ% ^
  --lambda_max %LAMBDA_MAX% ^
  --dual_update_mode %DUAL_UPDATE_MODE% ^
  --risk_factor %RISK_FACTOR% ^
  --initial_lambda_energy 0.0 ^
  --initial_lambda_load 0.0 ^
  --dual_gap_ema_beta %DUAL_GAP_BETA% ^
  --dual_corr_ema_beta %DUAL_CORR_BETA% ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_INTERVAL% ^
  --run_tag "%RUN_TAG%"

exit /b %errorlevel%

:fail
echo [ERROR] Run failed.
pause
exit /b 1
