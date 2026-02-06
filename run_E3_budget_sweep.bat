@echo off
setlocal enabledelayedexpansion

REM =======================
REM E3: Budget sweep (Lagrange hysteresis)
REM =======================

set "SCRIPT=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\E3_budget_sweep"

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

REM ---- PPO ----
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

REM ---- Logging ----
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=False"
set "GRAD_DECOMP_INTERVAL=100"
set "LOG_INTERVAL=10"

REM ---- S3: multi-seed (default 3 seeds; extend to 0 1 2 3 4 if needed) ----
set "SEEDS=0 1 2"

REM ---- S1: penalty transfer baseline (FILL THESE with E1-best @ EB=0.10/LB=0.05; NO retuning across budgets) ----
set "E1_OUTDIR=outputs\E1_penalty_fixedlambda_sweep"
set "MAIN_EB=0.10"
set "MAIN_LB=0.05"
set "PICK_BEST_PATH=%~dp0tools\pick_best_penalty.py"
set "RUN_HYST_LAGRANGE=True"
set "RUN_PENALTY_TRANSFER=True"
set "FIX_LAMBDA_E=1.0"
set "FIX_LAMBDA_L=1.0"

REM ---- S4: window/buffer consistency (adjust if your code uses a different arg name) ----
set "TRAIN_BUFFER_EPISODES=100"

for %%S in (%SEEDS%) do (
  for /f "tokens=1,2" %%A in ('python "%PICK_BEST_PATH%" --base_dir "%E1_OUTDIR%" --seed %%S --energy_budget %MAIN_EB% --load_budget %MAIN_LB% --metric best_feasible_return') do (
    set "FIX_LAMBDA_E=%%A"
    set "FIX_LAMBDA_L=%%B"
  )
  echo [E3 transfer lambdas] seed=%%S  lamE=!FIX_LAMBDA_E!  lamL=!FIX_LAMBDA_L!

  for %%E in (0.05 0.10 0.15) do (
    for %%L in (0.03 0.05 0.07) do (

      if /I "%RUN_HYST_LAGRANGE%"=="True" (
        echo ==========================================================
        echo [E3 Hyst-Lagrange] seed=%%S  EB=%%E  LB=%%L
        echo ==========================================================

        python %SCRIPT% ^
          --output_dir "%OUTDIR%" ^
          --run_tag "hyst_seed%%S_EB%%E_LB%%L" ^
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
          --energy_budget %%E ^
          --load_budget %%L ^
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
          --train_buffer_episodes %TRAIN_BUFFER_EPISODES%
      )

      if /I "%RUN_PENALTY_TRANSFER%"=="True" (
        echo ==========================================================
        echo [E3 Penalty-Transfer] seed=%%S  EB=%%E  LB=%%L  fixed_lE=!FIX_LAMBDA_E! fixed_lL=!FIX_LAMBDA_L!
        echo ==========================================================

        python %SCRIPT% ^
          --output_dir "%OUTDIR%" ^
          --run_tag "penaltyTransfer_seed%%S_EB%%E_LB%%L_lE!FIX_LAMBDA_E!_lL!FIX_LAMBDA_L!" ^
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
          --use_lagrange False ^
          --initial_lambda_energy !FIX_LAMBDA_E! ^
          --initial_lambda_load !FIX_LAMBDA_L! ^
          --energy_budget %%E ^
          --load_budget %%L ^
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
          --train_buffer_episodes %TRAIN_BUFFER_EPISODES%
      )

    )
  )
)

echo Done.
pause
