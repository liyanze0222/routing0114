@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  RUN_ALL_MERGED_FIXED.bat
REM
REM  Fixes vs the broken merged bat:
REM   (1) Subroutines were placed before MAIN WITHOUT a "goto", so the script
REM       executed :FindLatestRunDir immediately (with empty %1/%2) -> syntax errors
REM       like "此时不应有 ...".
REM   (2) Aligns args with your working bats (run_E1/run_E2/run_E3) and current
REM       argparse names in train_grid_structured_lagrangian.py:
REM       - use --lr (NOT --learning_rate)
REM       - pass True/False values for bool flags (enable_best_checkpoint, etc.)
REM       - use --cost_critic_mode {separate,shared,aggregated} (NOT --shared_critic / --aggregate_costs)
REM       - adds --train_buffer_episodes to match your bugfix (best_window_fsr <= buffer)
REM   (3) :FindLatestRunDir rewritten to avoid "dir | findstr" piping (more robust).
REM ============================================================

REM ---------- toggles ----------
set "DO_MAIN=1"
set "DO_E1_SWEEP=1"
set "DO_E1_TRANSFER=1"
set "DO_E2_ABL=1"
set "DO_E3_SWEEP=1"

set "DO_EVAL=1"
set "EVAL_NUM_SEEDS=1000"
set "EVAL_SEED_START=0"
set "EVAL_DEVICE=cpu"
set "EVAL_DETERMINISTIC=--deterministic"

REM ---------- python entrypoints ----------
set "PY=python"
set "TRAIN_PY=train_grid_structured_lagrangian.py"
set "EVAL_PY=eval_fixed_set.py"
set "PICK_PY=tools\pick_best_penalty.py"

REM ---------- seeds / iters ----------
set "MAIN_SEEDS=0 1 2 3 4 5"
set "E1_SEEDS=0 1"
set "E2_SEEDS=0 1 2 3 4 5"
set "E3_SEEDS=0 1 2"

set "KEY_EB=0.10"
set "KEY_LB=0.05"

set "TOTAL_ITERS_MAIN=800"
set "TOTAL_ITERS_E1=800"
set "TOTAL_ITERS_E2=800"
set "TOTAL_ITERS_E3=400"

REM ---------- E3 budgets (ALIGN with run_E3_budget_sweep.bat) ----------
set "E_LIST=0.05 0.10 0.15"
set "L_LIST=0.03 0.05 0.07"

REM ---------- penalty sweep grid (ALIGN with run_E1_penalty_fixedlambda_sweep.bat) ----------
set "LAM_LIST=0.1 0.5 1.0 2.0 5.0"

REM ---------- outputs ----------
set "OUT_MAIN=outputs\MAIN_keybudget"
set "OUT_E1=outputs\E1_penalty_fixedlambda_sweep"
set "OUT_E1T=outputs\E1_penalty_transfer"
set "OUT_E2=outputs\E2_costcritic_ablation"
set "OUT_E3=outputs\E3_budget_sweep"
set "EVAL_DIR=results_eval"

if not exist "%OUT_MAIN%" mkdir "%OUT_MAIN%"
if not exist "%OUT_E1%" mkdir "%OUT_E1%"
if not exist "%OUT_E1T%" mkdir "%OUT_E1T%"
if not exist "%OUT_E2%" mkdir "%OUT_E2%"
if not exist "%OUT_E3%" mkdir "%OUT_E3%"
if not exist "%EVAL_DIR%" mkdir "%EVAL_DIR%"

REM ============================================================
REM  COMMON ARGS (ALIGNED to your working bats)
REM ============================================================

REM ---- Env (match run_E1/run_E2/run_E3) ----
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

REM ---- PPO (match run_E1/run_E2/run_E3) ----
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

REM ---- Lagrange / Dual (match run_E1/run_E2/run_E3) ----
set "GAP_MODE=absolute"
set "LAMBDA_UPDATE_FREQ=1"
set "LAMBDA_MAX=10.0"
set "LAMBDA_LR=0.05"
set "RISK_FACTOR=0.0"

REM dual-mode params (used when mode requires)
set "DUAL_GAP_EMA=0.10"
set "DUAL_CORR_EMA=0.05"
set "DUAL_DEADBAND=0.005"
set "DUAL_LR_DOWN=0.20"
set "DUAL_PRECOND_EPS=0.05"
set "DUAL_PRECOND_CLIP=2.0"
set "DUAL_PRECOND_STRENGTH=0.3"
set "DUAL_PRECOND_USE_EMA_STATS=True"

REM ---- Best checkpoint / early stop (match current code + your bugfix) ----
set "ENABLE_BEST=True"
set "BEST_SUCCESS_THRESH=0.95"
set "BEST_WINDOW_FSR=50"
set "TRAIN_BUFFER_EP=100"
set "BEST_WINDOW_TAIL=50"
set "TAIL_P=95"
set "EARLY_STOP=False"

REM ---- Logging (match bats) ----
set "LOG_ACTOR_DECOMP=True"
set "LOG_ACTOR_GRAD_DECOMP=False"
set "GRAD_DECOMP_INTERVAL=100"
set "LOG_INTERVAL=10"

REM ============================================================
REM  MAIN
REM ============================================================
if "%DO_MAIN%"=="1" (
  for %%S in (%MAIN_SEEDS%) do (

    REM reward-only baseline (no penalty / no dual update)
    call :TrainOne "%OUT_MAIN%" "rewardonly_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange False --initial_lambda_energy 0.0 --initial_lambda_load 0.0 --dual_update_mode standard

    REM fixed penalty baseline (mid)
    call :TrainOne "%OUT_MAIN%" "penalty_mid_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange False --initial_lambda_energy 0.5 --initial_lambda_load 0.5 --dual_update_mode standard

    REM lagrange standard
    call :TrainOne "%OUT_MAIN%" "lagrange_std_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode standard

    REM hysteresis
    call :TrainOne "%OUT_MAIN%" "lagrange_hyst_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode hysteresis ^
      --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN%

    REM decorrelated
    call :TrainOne "%OUT_MAIN%" "lagrange_decor_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode decorrelated ^
      --dual_corr_ema_beta %DUAL_CORR_EMA%

    REM both (decorrelated + hysteresis)
    call :TrainOne "%OUT_MAIN%" "lagrange_both_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode both ^
      --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN% --dual_corr_ema_beta %DUAL_CORR_EMA%

    REM precond (optional extra)
    call :TrainOne "%OUT_MAIN%" "lagrange_precond_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode precond ^
      --dual_corr_ema_beta %DUAL_CORR_EMA% ^
      --dual_precond_eps %DUAL_PRECOND_EPS% --dual_precond_clip %DUAL_PRECOND_CLIP% --dual_precond_strength %DUAL_PRECOND_STRENGTH% --dual_precond_use_ema_stats %DUAL_PRECOND_USE_EMA_STATS%

  )
)

REM ============================================================
REM  E1: penalty grid sweep at key budget
REM ============================================================
if "%DO_E1_SWEEP%"=="1" (
  for %%S in (%E1_SEEDS%) do (
    for %%E in (%LAM_LIST%) do (
      for %%L in (%LAM_LIST%) do (
        call :TrainOne "%OUT_E1%" "penalty_seed%%S_EB%KEY_EB%_LB%KEY_LB%_lamE%%E_lamL%%L" %%S %KEY_EB% %KEY_LB% ^
          --total_iters %TOTAL_ITERS_E1% --use_lagrange False --initial_lambda_energy %%E --initial_lambda_load %%L --dual_update_mode standard
      )
    )
  )
)

REM ============================================================
REM  E1 transfer: use picked best penalty from key budget, no re-tune
REM ============================================================
if "%DO_E1_TRANSFER%"=="1" (
  if not exist "%PICK_PY%" (
    echo [ERR] "%PICK_PY%" not found. Set DO_E1_TRANSFER=0 or put the tool under tools\ .
    exit /b 1
  )

  for %%S in (%E1_SEEDS%) do (
    REM Expect pick_best_penalty.py prints: "<best_lamE> <best_lamL>" on one line
    for /f "tokens=1,2" %%A in ('%PY% %PICK_PY% --base_dir "%OUT_E1%" --seed %%S --energy_budget %KEY_EB% --load_budget %KEY_LB% --metric best_feasible_return --out_json "%OUT_E1%\best_penalty_seed%%S.json"') do (
      set "BEST_LAME=%%A"
      set "BEST_LAML=%%B"
    )
    echo [E1-TRANSFER] seed=%%S picked lamE=!BEST_LAME! lamL=!BEST_LAML!

    for %%EB in (%E_LIST%) do (
      for %%LB in (%L_LIST%) do (
        call :TrainOne "%OUT_E1T%" "penaltyXfer_seed%%S_EB%%EB_LB%%LB_lamE!BEST_LAME!_lamL!BEST_LAML!" %%S %%EB %%LB ^
          --total_iters %TOTAL_ITERS_E3% --use_lagrange False --initial_lambda_energy !BEST_LAME! --initial_lambda_load !BEST_LAML! --dual_update_mode standard
      )
    )
  )
)

REM ============================================================
REM  E2: cost critic ablation (aligned to run_E2_costcritic_ablation.bat)
REM ============================================================
if "%DO_E2_ABL%"=="1" (

  for %%S in (%E2_SEEDS%) do (

    REM A) separate (main)
    call :TrainOne "%OUT_E2%" "E2_sep_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode hysteresis ^
      --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_corr_ema_beta %DUAL_CORR_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN% ^
      --dual_precond_eps %DUAL_PRECOND_EPS% --dual_precond_clip %DUAL_PRECOND_CLIP% --dual_precond_strength %DUAL_PRECOND_STRENGTH% --dual_precond_use_ema_stats %DUAL_PRECOND_USE_EMA_STATS% ^
      --cost_critic_mode separate

    REM B) shared
    call :TrainOne "%OUT_E2%" "E2_shared_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode hysteresis ^
      --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_corr_ema_beta %DUAL_CORR_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN% ^
      --dual_precond_eps %DUAL_PRECOND_EPS% --dual_precond_clip %DUAL_PRECOND_CLIP% --dual_precond_strength %DUAL_PRECOND_STRENGTH% --dual_precond_use_ema_stats %DUAL_PRECOND_USE_EMA_STATS% ^
      --cost_critic_mode shared

  )

  REM C) aggregated weight sweep (3 settings, seeds 0-2)
  for %%S in (0 1 2) do (
    for %%W in (0.2 0.5 0.8) do (
      if "%%W"=="0.2" ( set "W_E=0.2" & set "W_L=0.8" )
      if "%%W"=="0.5" ( set "W_E=0.5" & set "W_L=0.5" )
      if "%%W"=="0.8" ( set "W_E=0.8" & set "W_L=0.2" )

      call :TrainOne "%OUT_E2%" "E2_agg_wE!W_E!_wL!W_L!_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
        --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode hysteresis ^
        --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_corr_ema_beta %DUAL_CORR_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN% ^
        --dual_precond_eps %DUAL_PRECOND_EPS% --dual_precond_clip %DUAL_PRECOND_CLIP% --dual_precond_strength %DUAL_PRECOND_STRENGTH% --dual_precond_use_ema_stats %DUAL_PRECOND_USE_EMA_STATS% ^
        --cost_critic_mode aggregated --agg_cost_normalize_by_budget True --agg_cost_w_energy !W_E! --agg_cost_w_load !W_L!
    )
  )

)

REM ============================================================
REM  E3: budget sweep (standard vs hysteresis; aligned to run_E3_budget_sweep.bat)
REM ============================================================
if "%DO_E3_SWEEP%"=="1" (
  for %%S in (%E3_SEEDS%) do (
    for %%EB in (%E_LIST%) do (
      for %%LB in (%L_LIST%) do (

        call :TrainOne "%OUT_E3%" "E3_std_seed%%S_EB%%EB_LB%%LB" %%S %%EB %%LB ^
          --total_iters %TOTAL_ITERS_E3% --use_lagrange True --dual_update_mode standard

        call :TrainOne "%OUT_E3%" "E3_hyst_seed%%S_EB%%EB_LB%%LB" %%S %%EB %%LB ^
          --total_iters %TOTAL_ITERS_E3% --use_lagrange True --dual_update_mode hysteresis ^
          --dual_gap_ema_beta %DUAL_GAP_EMA% --dual_corr_ema_beta %DUAL_CORR_EMA% --dual_deadband %DUAL_DEADBAND% --dual_lr_down_scale %DUAL_LR_DOWN%

      )
    )
  )
)

echo.
echo ALL DONE.
exit /b 0


REM ============================================================
REM  Subroutines (keep below; script exits before reaching here)
REM ============================================================

:FindLatestRunDir
REM %1=BASE_DIR  %2=TAG_PREFIX  -> sets RUN_DIR
set "RUN_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "%~1\%~2*" 2^>nul') do (
  set "RUN_DIR=%~1\%%D"
  goto :eof
)
echo [ERR] cannot find run dir. base="%~1" tag="%~2"
exit /b 1

:EvalLatest
REM %1=BASE_DIR %2=TAG_PREFIX %3=EB %4=LB
call :FindLatestRunDir "%~1" "%~2"
if errorlevel 1 exit /b 1

set "CKPT=%RUN_DIR%\best_feasible.pt"
if not exist "%CKPT%" (
  echo [ERR] best_feasible.pt not found: "%CKPT%"
  echo       Check --enable_best_checkpoint True and training logs.
  exit /b 1
)

set "OUTCSV=%EVAL_DIR%\%~2_EB%~3_LB%~4.csv"
%PY% %EVAL_PY% --ckpt_path "%CKPT%" --num_seeds %EVAL_NUM_SEEDS% --seed_start %EVAL_SEED_START% %EVAL_DETERMINISTIC% --device %EVAL_DEVICE% --energy_budget %~3 --load_budget %~4 --out_csv "%OUTCSV%"
if errorlevel 1 exit /b 1
exit /b 0

:TrainOne
REM %1=OUT_BASE %2=RUN_TAG %3=SEED %4=EB %5=LB %6+=EXTRA
set "OUT_BASE=%~1"
set "RUN_TAG=%~2"
set "SEED=%~3"
set "EB=%~4"
set "LB=%~5"
shift
shift
shift
shift
shift
set "EXTRA=%*"

echo.
echo ============================================================
echo [TRAIN] %RUN_TAG%  seed=%SEED%  EB=%EB%  LB=%LB%
echo ============================================================

%PY% %TRAIN_PY% ^
  --output_dir "%OUT_BASE%" ^
  --run_tag "%RUN_TAG%" ^
  --seed %SEED% ^
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
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr %LAMBDA_LR% ^
  --lambda_lr_energy %LAMBDA_LR% ^
  --lambda_lr_load %LAMBDA_LR% ^
  --lambda_update_freq %LAMBDA_UPDATE_FREQ% ^
  --lambda_max %LAMBDA_MAX% ^
  --risk_factor %RISK_FACTOR% ^
  --value_head_mode standard ^
  --enable_best_checkpoint %ENABLE_BEST% ^
  --best_checkpoint_success_thresh %BEST_SUCCESS_THRESH% ^
  --best_window_fsr %BEST_WINDOW_FSR% ^
  --train_buffer_episodes %TRAIN_BUFFER_EP% ^
  --best_window_tail %BEST_WINDOW_TAIL% ^
  --tail_percentile %TAIL_P% ^
  --enable_early_stop %EARLY_STOP% ^
  --log_actor_decomp %LOG_ACTOR_DECOMP% ^
  --log_actor_grad_decomp %LOG_ACTOR_GRAD_DECOMP% ^
  --grad_decomp_interval %GRAD_DECOMP_INTERVAL% ^
  --log_interval %LOG_INTERVAL% ^
  %EXTRA%

if errorlevel 1 exit /b 1

if "%DO_EVAL%"=="1" (
  call :EvalLatest "%OUT_BASE%" "%RUN_TAG%" %EB% %LB%
  if errorlevel 1 exit /b 1
)

exit /b 0
