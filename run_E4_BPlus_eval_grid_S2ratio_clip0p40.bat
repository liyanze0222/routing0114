@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Full 3x3 grid eval using chosen stable online-dual config
REM Chosen config (based on your two experiments):
REM   gap_mode=ratio
REM   lr=0.002
REM   update_freq=5
REM   dual_deadband=0.02
REM   lambda_max=0.40
REM   lambda_obs_clip=0.40
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E4_BPlus_eval_grid_S2ratio_clip0p40"
if not exist "%EVAL_OUTDIR%" mkdir "%EVAL_OUTDIR%"

REM ---- Locate latest B-Plus checkpoint (seed 0) ----
set "SEED=0"
set "TAG_BPLUS=E4B_bplus_seed%SEED%"
set "BPLUS_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTDIR%\%TAG_BPLUS%*"') do (
  if not defined BPLUS_DIR set "BPLUS_DIR=%OUTDIR%\%%D"
)
if not defined BPLUS_DIR (
  echo [ERROR] No B-Plus run dir found under %OUTDIR% for tag %TAG_BPLUS%.
  exit /b 1
)
set "BPLUS_CKPT=!BPLUS_DIR!\best_feasible.pt"
if not exist "!BPLUS_CKPT!" set "BPLUS_CKPT=!BPLUS_DIR!\checkpoint_final.pt"
if not exist "!BPLUS_CKPT!" (
  echo [ERROR] No checkpoint found under !BPLUS_DIR!.
  exit /b 1
)
echo Using checkpoint: !BPLUS_CKPT!

REM ---- Eval settings ----
set "EVAL_NUM=1000"
set "EVAL_SEED_START=10000"
set "DEVICE=cpu"
set "DETERMINISTIC=True"

REM ---- Chosen stable online-dual overrides ----
set "GAPMODE=ratio"
set "LR=0.002"
set "FREQ=5"
set "DEADBAND=0.02"
set "LAMMAX=0.40"
set "OBSCLIP=0.40"

echo ==========================================================
echo [GRID EVAL] num=%EVAL_NUM% seed_start=%EVAL_SEED_START%
echo config: gap_mode=%GAPMODE% lr=%LR% freq=%FREQ% deadband=%DEADBAND% max=%LAMMAX% clip=%OBSCLIP%
echo out: %EVAL_OUTDIR%
echo ==========================================================

for %%E in (0.05 0.10 0.15) do (
  for %%L in (0.03 0.05 0.07) do (

    echo ---- EB=%%E  LB=%%L ----

    python %SCRIPT_EVAL% ^
      --ckpt_path "!BPLUS_CKPT!" ^
      --num_seeds %EVAL_NUM% ^
      --seed_start %EVAL_SEED_START% ^
      --deterministic %DETERMINISTIC% ^
      --device %DEVICE% ^
      --energy_budget %%E ^
      --load_budget %%L ^
      --online_dual_update True ^
      --eval_lambda_gap_mode %GAPMODE% ^
      --eval_lambda_lr %LR% ^
      --eval_lambda_update_freq %FREQ% ^
      --eval_dual_deadband %DEADBAND% ^
      --eval_lambda_max %LAMMAX% ^
      --eval_lambda_obs_clip %OBSCLIP% ^
      --dual_trace_csv "%EVAL_OUTDIR%\dualtrace_EB%%E_LB%%L.csv" ^
      --out_csv "%EVAL_OUTDIR%\eval_EB%%E_LB%%L.csv"
  )
)

echo Done.
pause
