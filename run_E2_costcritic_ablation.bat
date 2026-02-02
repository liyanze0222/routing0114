@echo off
setlocal enabledelayedexpansion

REM =======================
REM E2: Cost critic ablation
REM   A) separate (main)
REM   B) shared (shared cost head)
REM   C) aggregated (single combined constraint)
REM =======================

set "SCRIPT=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\E2_costcritic_ablation"

REM ---- Env ----
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

REM ---- Budgets ----
set "E_BUDGET=0.10"
set "L_BUDGET=0.05"

REM ---- PPO ----
set "ITERS=800"
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

REM ---- Lagrange / Dual ----
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

REM ---- aggregated params ----
set "AGG_NORM=True"
set "AGG_WE=0.5"
set "AGG_WL=0.5"

REM ---- Logging ----
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=False"
set "GRAD_DECOMP_INTERVAL=100"
set "LOG_INTERVAL=10"

for %%S in (0) do (

  REM -------- A) separate --------
  echo ==========================================================
  echo [E2-A separate] seed=%%S
  echo ==========================================================
  python %SCRIPT% ^
    --output_dir "%OUTDIR%" ^
    --run_tag "A_separate_seed%%S" ^
    --seed %%S ^
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
    --log_interval %LOG_INTERVAL%

  REM -------- B) shared --------
  echo ==========================================================
  echo [E2-B shared] seed=%%S
  echo ==========================================================
  python %SCRIPT% ^
    --output_dir "%OUTDIR%" ^
    --run_tag "B_shared_seed%%S" ^
    --seed %%S ^
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
    --cost_critic_mode shared ^
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
    --log_interval %LOG_INTERVAL%

  REM -------- C) aggregated --------
  echo ==========================================================
  echo [E2-C aggregated] seed=%%S  (wE=%AGG_WE% wL=%AGG_WL% norm=%AGG_NORM%)
  echo ==========================================================
  python %SCRIPT% ^
    --output_dir "%OUTDIR%" ^
    --run_tag "C_aggregated_seed%%S_wE%AGG_WE%_wL%AGG_WL%" ^
    --seed %%S ^
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
    --cost_critic_mode aggregated ^
    --agg_cost_normalize_by_budget %AGG_NORM% ^
    --agg_cost_w_energy %AGG_WE% ^
    --agg_cost_w_load %AGG_WL% ^
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
    --log_interval %LOG_INTERVAL%

)

echo Done.
pause
