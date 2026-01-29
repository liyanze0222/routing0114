@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Reward-Only Baseline (lambda = 0)
REM 
REM Experiment description:
REM   - use_lagrange False (fixed lambda mode)
REM   - initial_lambda_energy 0.0 and initial_lambda_load 0.0
REM   - Agent only optimizes reward (path length), ignores constraints
REM   - For comparison with constrained (standard/hysteresis) methods
REM ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\reward_only_baseline"

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

REM ---------------- CMDP / dual (REWARD-ONLY: lambda=0) ----------------
set "E_BUDGET=0.10"
set "L_BUDGET=0.05"
set "GAP_MODE=absolute"
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "LAMBDA_LR_E=0.05"
set "LAMBDA_LR_L=0.05"
set "RISK_FACTOR=0.0"

REM reward-only: fixed lambda=0
set "USE_LAGRANGE=False"
set "INIT_LAMBDA_ENERGY=0.0"
set "INIT_LAMBDA_LOAD=0.0"

REM diagnostics
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=True"
set "GRAD_INTERVAL=1"

echo ============================================================
echo Reward-Only Baseline (lambda=0, no constraints)
echo OUTDIR=%OUTDIR%
echo SEED=%SEED%  ITERS=%ITERS%  ENT=%ENT%
echo BUDGET: energy=%E_BUDGET%  load=%L_BUDGET%
echo LAMBDA: fixed at 0.0 (reward-only)
echo ============================================================
echo.

%PY% %ENTRY% ^
  --output_dir %OUTDIR% ^
  --run_tag "reward_only_ent%ENT%_seed%SEED%" ^
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
  --use_lagrange %USE_LAGRANGE% ^
  --energy_budget %E_BUDGET% ^
  --load_budget %L_BUDGET% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR_E% ^
  --lambda_lr_load %LAMBDA_LR_L% ^
  --lambda_update_freq %LAMBDA_UPDATE_FREQ% ^
  --lambda_max %LAMBDA_MAX% ^
  --risk_factor %RISK_FACTOR% ^
  --initial_lambda_energy %INIT_LAMBDA_ENERGY% ^
  --initial_lambda_load %INIT_LAMBDA_LOAD% ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_INTERVAL%

echo.
echo ============================================================
echo Reward-Only experiment completed!
echo ============================================================
echo.
echo Notes:
echo   - lambda=0, agent ignores constraints completely
echo   - Expected: shortest path, but energy/load may exceed budget
echo   - Compare with eval_fixed_set.py or compare_routes_over_time.py
echo ============================================================
pause
