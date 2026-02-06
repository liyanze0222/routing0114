@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM E5-5: Budget-conditioned baseline
REM Train: policy_condition_on_budget=True, policy_condition_on_lambda=False
REM Eval : 3x3 budgets, no dual
REM Params aligned with run_E4_BPlus.bat
REM ==========================================================

set "SCRIPT_TRAIN=train_grid_structured_lagrangian.py"
set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E5_budgetCond_eval"
if not exist "%EVAL_OUTDIR%" mkdir "%EVAL_OUTDIR%"

REM ---- Env (align with E4) ----
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

REM ---- Train budget (center) ----
set "E_BUDGET=0.10"
set "L_BUDGET=0.05"

REM ---- PPO (align with E4) ----
set "ITERS=400"
set "BATCH=4096"
set "MB=256"
set "EPOCHS=10"
set "HIDDEN=128"
set "LR=3e-4"
set "GAMMA=0.99"
set "GAE=0.95"
set "CLIP=0.2"
set "ENT=0.05"
set "VF=0.5"
set "COST_VF=1.0"
set "MAXGN=0.5"

REM ---- Lagrange / Dual (align with E4) ----
set "GAP_MODE=absolute"
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "LAMBDA_LR=0.05"
set "RISK_FACTOR=0.0"

set "DUAL_MODE=hysteresis"
set "DUAL_GAP_EMA=0.10"
set "DUAL_CORR_EMA=0.05"
set "DUAL_DEADBAND=0.005"
set "DUAL_LR_DOWN=0.20"
set "DUAL_PRECOND_EPS=0.05"
set "DUAL_PRECOND_CLIP=2.0"
set "DUAL_PRECOND_STRENGTH=0.3"
set "DUAL_PRECOND_USE_EMA_STATS=True"

REM ---- Logging (align with E4) ----
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=False"
set "GRAD_DECOMP_INTERVAL=100"
set "LOG_INTERVAL=10"

REM ---- Switch: budget-conditioned baseline ----
set "POLICY_COND_BUDGET=True"
set "POLICY_COND_LAMBDA=False"
set "LAMBDA_OBS_CLIP=10.0"

REM ---- Eval settings ----
set "EVAL_NUM=1000"
set "EVAL_SEED_START=10000"
set "EVAL_DETERMINISTIC=True"
set "DEVICE=cpu"

set "SEED=0"
set "TAG=E5D_budgetCond_seed%SEED%"

echo ==========================================================
echo [Train Budget-Conditioned] seed=%SEED% EB=%E_BUDGET% LB=%L_BUDGET%
echo ==========================================================

python %SCRIPT_TRAIN% ^
  --output_dir "%OUTDIR%" ^
  --run_tag "%TAG%" ^
  --seed %SEED% ^
  --total_iters %ITERS% ^
  --batch_size %BATCH% ^
  --minibatch_size %MB% ^
  --update_epochs %EPOCHS% ^
  --hidden_dim %HIDDEN% ^
  --lr %LR% ^
  --gamma %GAMMA% ^
  --gae_lambda %GAE% ^
  --clip_coef %CLIP% ^
  --ent_coef %ENT% ^
  --value_coef %VF% ^
  --cost_value_coef %COST_VF% ^
  --max_grad_norm %MAXGN% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --step_penalty %STEP_P% ^
  --success_reward %SUCCESS_R% ^
  --energy_high_density %ENERGY_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --randomize_maps_each_reset %RANDOMIZE_MAPS% ^
  --load_threshold %LOAD_TAU% ^
  --start_goal_mode %START_GOAL_MODE% ^
  --start_rect %START_RECT% ^
  --goal_rect %GOAL_RECT% ^
  --include_congestion_obs %INC_CONG_OBS% ^
  --congestion_patch_radius %CONG_R% ^
  --include_energy_obs %INC_E_OBS% ^
  --energy_patch_radius %E_R% ^
  --obs_rms %OBS_RMS% ^
  --use_lagrange True ^
  --initial_lambda_energy 0.0 ^
  --initial_lambda_load 0.0 ^
  --energy_budget %E_BUDGET% ^
  --load_budget %L_BUDGET% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr %LAMBDA_LR% ^
  --lambda_lr_energy %LAMBDA_LR% ^
  --lambda_lr_load %LAMBDA_LR% ^
  --lambda_update_freq %LAMBDA_UPDATE_FREQ% ^
  --lambda_max %LAMBDA_MAX% ^
  --risk_factor %RISK_FACTOR% ^
  --dual_update_mode %DUAL_MODE% ^
  --dual_gap_ema_beta %DUAL_GAP_EMA% ^
  --dual_corr_ema_beta %DUAL_CORR_EMA% ^
  --dual_deadband %DUAL_DEADBAND% ^
  --dual_lr_down_scale %DUAL_LR_DOWN% ^
  --dual_precond_eps %DUAL_PRECOND_EPS% ^
  --dual_precond_clip %DUAL_PRECOND_CLIP% ^
  --dual_precond_strength %DUAL_PRECOND_STRENGTH% ^
  --dual_precond_use_ema_stats %DUAL_PRECOND_USE_EMA_STATS% ^
  --cost_critic_mode separate ^
  --value_head_mode standard ^
  --enable_best_checkpoint True ^
  --best_checkpoint_success_thresh 0.95 ^
  --best_window_fsr 50 ^
  --best_window_tail 50 ^
  --tail_percentile 95 ^
  --enable_early_stop False ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_DECOMP_INTERVAL% ^
  --log_interval %LOG_INTERVAL% ^
  --policy_condition_on_budget %POLICY_COND_BUDGET% ^
  --policy_condition_on_lambda %POLICY_COND_LAMBDA% ^
  --lambda_obs_clip %LAMBDA_OBS_CLIP% ^
  --save_model

REM ---- Find newest dir ----
set "RUN_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTDIR%\%TAG%*"') do (
  if not defined RUN_DIR set "RUN_DIR=%OUTDIR%\%%D"
)
set "CKPT=!RUN_DIR!\best_feasible.pt"
if not exist "!CKPT!" set "CKPT=!RUN_DIR!\checkpoint_final.pt"

echo Using ckpt: !CKPT!

REM ---- 3x3 eval (no dual) ----
for %%E in (0.05 0.10 0.15) do (
  for %%L in (0.03 0.05 0.07) do (
    python %SCRIPT_EVAL% ^
      --ckpt_path "!CKPT!" ^
      --num_seeds %EVAL_NUM% ^
      --seed_start %EVAL_SEED_START% ^
      --deterministic %EVAL_DETERMINISTIC% ^
      --device %DEVICE% ^
      --energy_budget %%E ^
      --load_budget %%L ^
      --online_dual_update False ^
      --out_csv "%EVAL_OUTDIR%\budgetCond_seed%SEED%_EB%%E_LB%%L.csv"
  )
)

echo Done.
pause
