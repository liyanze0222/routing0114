@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM E5-4: Domain shift robustness
REM - randomize maps each episode
REM - sweep congestion_density and energy_high_density
REM Compare B0 vs D on representative budgets
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E5_BPlus_eval_domainShift_randomMaps_seed0"
if not exist "%EVAL_OUTDIR%" mkdir "%EVAL_OUTDIR%"

set "SEED=0"
set "TAG_BPLUS=E4B_bplus_seed%SEED%"

set "BPLUS_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTDIR%\%TAG_BPLUS%*"') do (
  if not defined BPLUS_DIR set "BPLUS_DIR=%OUTDIR%\%%D"
)
if not defined BPLUS_DIR (
  echo [ERROR] No B-Plus dir found.
  exit /b 1
)

set "BPLUS_CKPT=!BPLUS_DIR!\best_feasible.pt"
if not exist "!BPLUS_CKPT!" set "BPLUS_CKPT=!BPLUS_DIR!\checkpoint_final.pt"
if not exist "!BPLUS_CKPT!" (
  echo [ERROR] No checkpoint found.
  exit /b 1
)

echo Using ckpt: !BPLUS_CKPT!

REM ---- Eval settings ----
set "EVAL_NUM=300"
set "EVAL_SEED_START=30000"
set "DEVICE=cpu"
set "DETERMINISTIC=True"

REM ---- D config ----
set "GAPMODE=ratio"
set "LR=0.002"
set "FREQ=5"
set "DEADBAND=0.02"
set "LAMMAX=0.40"
set "OBSCLIP=0.40"
set "LAMMIN=0.20"

REM Representative budgets: tight / mid / loose
for %%E in (0.05 0.10 0.15) do (
  for %%L in (0.03 0.05 0.07) do (
    set "EB=%%E"
    set "LB=%%L"

    for %%C in (0.20 0.40 0.60) do (
      for %%H in (0.10 0.20 0.30) do (

        echo ---- EB=!EB! LB=!LB!  cong=%%C  energyD=%%H ----

        REM (B0)
        python %SCRIPT_EVAL% ^
          --ckpt_path "!BPLUS_CKPT!" ^
          --num_seeds %EVAL_NUM% ^
          --seed_start %EVAL_SEED_START% ^
          --deterministic %DETERMINISTIC% ^
          --device %DEVICE% ^
          --energy_budget !EB! ^
          --load_budget !LB! ^
          --online_dual_update False ^
          --eval_randomize_maps_each_reset True ^
          --eval_congestion_density %%C ^
          --eval_energy_high_density %%H ^
          --out_csv "%EVAL_OUTDIR%\B0_EB!EB!_LB!LB!_CD%%C_ED%%H.csv"

        REM (D)
        python %SCRIPT_EVAL% ^
          --ckpt_path "!BPLUS_CKPT!" ^
          --num_seeds %EVAL_NUM% ^
          --seed_start %EVAL_SEED_START% ^
          --deterministic %DETERMINISTIC% ^
          --device %DEVICE% ^
          --energy_budget !EB! ^
          --load_budget !LB! ^
          --online_dual_update True ^
          --eval_randomize_maps_each_reset True ^
          --eval_congestion_density %%C ^
          --eval_energy_high_density %%H ^
          --eval_lambda_gap_mode %GAPMODE% ^
          --eval_lambda_lr %LR% ^
          --eval_lambda_update_freq %FREQ% ^
          --eval_dual_deadband %DEADBAND% ^
          --eval_lambda_max %LAMMAX% ^
          --eval_lambda_obs_clip %OBSCLIP% ^
          --eval_lambda_min %LAMMIN% ^
          --dual_trace_csv "%EVAL_OUTDIR%\D_dualtrace_EB!EB!_LB!LB!_CD%%C_ED%%H.csv" ^
          --out_csv "%EVAL_OUTDIR%\D_EB!EB!_LB!LB!_CD%%C_ED%%H.csv"

      )
    )
  )
)

echo Done.
pause
