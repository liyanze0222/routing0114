@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  One-shot runner (merged): MAIN + E1/E2/E3 + extra dual ablations
REM  Put this .bat in the SAME folder as:
REM    - train_grid_structured_lagrangian.py
REM    - eval_fixed_set.py
REM    - tools\pick_best_penalty.py
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
set "TOTAL_ITERS_E2=400"
set "TOTAL_ITERS_E3=400"

REM ---------- E3 budgets (same as your run_E3_budget_sweep.bat) ----------
set "E_LIST=0.05 0.10 0.20"
set "L_LIST=0.025 0.05 0.10"

REM ---------- penalty sweep grid (same as your E1 bat) ----------
set "LAM_LIST=0.1 0.2 0.5 1.0 2.0"

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

REM ---------- COMMON ARGS (aligned to your bats) ----------
set "ENV_ARGS=--grid_size 8 --step_penalty -1.0 --success_reward 10.0 --start_goal_mode rect --start_rect 0,1,0,1 --goal_rect 6,7,6,7 --max_steps 256 --congestion_pattern block --congestion_density 0.40 --energy_high_density 0.20 --patch_radius 2 --include_congestion_obs True --congestion_patch_radius 2 --include_energy_obs True --energy_patch_radius 2 --obs_rms True"
set "PPO_ARGS=--batch_size 2048 --learning_rate 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_coef 0.2 --ent_coef 0.01 --value_coef 0.5 --cost_value_coef 1.0 --max_grad_norm 0.5 --update_epochs 10 --hidden_dim 128"
REM Checkpoint controls aligned with train_grid_structured_lagrangian.py args
set "CKPT_ARGS=--save_model --enable_best_checkpoint True --best_window_fsr 50 --best_window_tail 100 --tail_percentile 95"
set "LOG_ARGS=--log_interval 10"

REM Skip helper labels during initial pass
goto :main

REM ============================================================
REM  helpers
REM ============================================================

:FindLatestRunDir
REM %1=BASE_DIR  %2=TAG_PREFIX  -> sets RUN_DIR
set "RUN_DIR="
for /f "delims=" %%D in ('dir /b /ad /o:-d "%~1" ^| findstr /b /c:"%~2"') do (
  set "RUN_DIR=%~1\%%D"
  goto :eof
)
echo [ERR] cannot find run dir. base="%~1" tag="%~2"
exit /b 1

:EvalLatest
REM %1=BASE_DIR %2=TAG_PREFIX %3=EB %4=LB
call :FindLatestRunDir "%~1" "%~2"
set "CKPT=%RUN_DIR%\best_feasible.pt"
if not exist "%CKPT%" (
  echo [ERR] best_feasible.pt not found: "%CKPT%"
  echo       did you forget --enable_best_checkpoint ?
  exit /b 1
)
set "OUTCSV=%EVAL_DIR%\%~2_EB%~3_LB%~4.csv"
%PY% %EVAL_PY% --ckpt_path "%CKPT%" --num_seeds %EVAL_NUM_SEEDS% --seed_start %EVAL_SEED_START% --deterministic --energy_budget %~3 --load_budget %~4 --out_csv "%OUTCSV%"
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

%PY% %TRAIN_PY% %ENV_ARGS% %PPO_ARGS% %CKPT_ARGS% %LOG_ARGS% --seed %SEED% --energy_budget %EB% --load_budget %LB% --output_dir "%OUT_BASE%" --run_tag "%RUN_TAG%" %EXTRA%
if errorlevel 1 exit /b 1

if "%DO_EVAL%"=="1" (
  call :EvalLatest "%OUT_BASE%" "%RUN_TAG%" %EB% %LB%
  if errorlevel 1 exit /b 1
)
exit /b 0

:main

REM ============================================================
REM  MAIN: key-budget multi-seed comparison
REM ============================================================
if "%DO_MAIN%"=="1" (
  for %%S in (%MAIN_SEEDS%) do (

    REM reward-only (no penalty / no dual update)
    call :TrainOne "%OUT_MAIN%" "rewardonly_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange False --initial_lambda_energy 0.0 --initial_lambda_load 0.0 --dual_update_mode standard

    REM fixed penalty (mid) baseline
    call :TrainOne "%OUT_MAIN%" "penalty_mid_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange False --initial_lambda_energy 0.5 --initial_lambda_load 0.5 --dual_update_mode standard

    REM lagrange standard
    call :TrainOne "%OUT_MAIN%" "lagrange_std_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode standard --lambda_lr 0.01 --lambda_max 10.0

    REM hysteresis
    call :TrainOne "%OUT_MAIN%" "lagrange_hyst_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode hysteresis --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5

    REM decorrelated
    call :TrainOne "%OUT_MAIN%" "lagrange_decor_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode decorrelated --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_corr_ema_beta 0.1

    REM both (hysteresis + decorrelated)
    call :TrainOne "%OUT_MAIN%" "lagrange_both_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode both --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5 --dual_corr_ema_beta 0.1

    REM precond (optional extra)
    call :TrainOne "%OUT_MAIN%" "lagrange_precond_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_MAIN% --use_lagrange True --dual_update_mode precond --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_precond_eps 1e-6 --dual_precond_beta 0.5 --dual_precond_min 0.2 --dual_precond_max 5.0 --dual_precond_use_abs True --dual_precond_clip_g 5.0

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
  for %%S in (%E1_SEEDS%) do (
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
REM  E2: cost critic ablation (aligned to your run_E2_costcritic_ablation.bat)
REM ============================================================
if "%DO_E2_ABL%"=="1" (
  for %%S in (%E2_SEEDS%) do (
    REM separate critics (default)
    call :TrainOne "%OUT_E2%" "E2_sep_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode both --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5 --dual_corr_ema_beta 0.1

    REM shared critic
    call :TrainOne "%OUT_E2%" "E2_shared_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
      --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode both --lambda_lr 0.01 --lambda_max 10.0 ^
      --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5 --dual_corr_ema_beta 0.1 ^
      --shared_critic True
  )

  REM aggregated cost sweep (wE in {0.2,0.5,0.8}, seeds 0-2 default)
  for %%S in (0 1 2) do (
    for %%W in (0.2 0.5 0.8) do (
      call :TrainOne "%OUT_E2%" "E2_agg_wE%%W_seed%%S_EB%KEY_EB%_LB%KEY_LB%" %%S %KEY_EB% %KEY_LB% ^
        --total_iters %TOTAL_ITERS_E2% --use_lagrange True --dual_update_mode both --lambda_lr 0.01 --lambda_max 10.0 ^
        --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5 --dual_corr_ema_beta 0.1 ^
        --aggregate_costs True --agg_cost_normalize_by_budget True --agg_cost_w_energy %%W --agg_cost_w_load 0.0
    )
  )
)

REM ============================================================
REM  E3: budget sweep (standard vs both; aligned to your run_E3_budget_sweep.bat)
REM ============================================================
if "%DO_E3_SWEEP%"=="1" (
  for %%S in (%E3_SEEDS%) do (
    for %%EB in (%E_LIST%) do (
      for %%LB in (%L_LIST%) do (

        call :TrainOne "%OUT_E3%" "E3_std_seed%%S_EB%%EB_LB%%LB" %%S %%EB %%LB ^
          --total_iters %TOTAL_ITERS_E3% --use_lagrange True --dual_update_mode standard --lambda_lr 0.01 --lambda_max 10.0

        call :TrainOne "%OUT_E3%" "E3_both_seed%%S_EB%%EB_LB%%LB" %%S %%EB %%LB ^
          --total_iters %TOTAL_ITERS_E3% --use_lagrange True --dual_update_mode both --lambda_lr 0.01 --lambda_max 10.0 ^
          --dual_gap_ema_beta 0.2 --dual_deadband 0.01 --dual_lr_down_scale 0.5 --dual_corr_ema_beta 0.1

      )
    )
  )
)

echo.
echo ALL DONE.
exit /b 0
