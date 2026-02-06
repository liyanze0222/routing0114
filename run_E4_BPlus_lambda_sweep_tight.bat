@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Lambda sweep @ tight budget (NO online dual)
REM Purpose:
REM   Map policy response curve vs lambda (safe/effective range)
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E4_BPlus_eval_lambda_sweep_tight"
if not exist "%EVAL_OUTDIR%" mkdir "%EVAL_OUTDIR%"

REM ---- Locate latest B-Plus checkpoint (seed 0) ----
set "SEED=0"
set "TAG_BPLUS=E4B_bplus_seed%SEED%"
set "BPLUS_DIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTDIR%\%TAG_BPLUS%*"') do (
  if not defined BPLUS_DIR set "BPLUS_DIR=%OUTDIR%\%%D"
)
if not defined BPLUS_DIR (
  echo [ERROR] No B-Plus run dir found.
  exit /b 1
)
set "BPLUS_CKPT=!BPLUS_DIR!\best_feasible.pt"
if not exist "!BPLUS_CKPT!" set "BPLUS_CKPT=!BPLUS_DIR!\checkpoint_final.pt"
if not exist "!BPLUS_CKPT!" (
  echo [ERROR] No checkpoint found.
  exit /b 1
)
echo Using checkpoint: !BPLUS_CKPT!

REM ---- Tight budgets (for feasibility check only) ----
set "EB=0.05"
set "LB=0.03"

REM ---- Eval settings ----
set "EVAL_NUM=200"
set "EVAL_SEED_START=21000"
set "DEVICE=cpu"

echo ==========================================================
echo [Lambda sweep] EB=%EB%  LB=%LB%  (NO online dual)
echo ==========================================================

REM Combined sweep: lambda_energy = lambda_load = X
for %%X in (0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.80 1.00) do (
  set "LAM=%%X"
  echo [Sweep] both lambdas = !LAM!
  python %SCRIPT_EVAL% ^
    --ckpt_path "!BPLUS_CKPT!" ^
    --num_seeds %EVAL_NUM% ^
    --seed_start %EVAL_SEED_START% ^
    --deterministic True ^
    --device %DEVICE% ^
    --energy_budget %EB% ^
    --load_budget %LB% ^
    --online_dual_update False ^
    --eval_init_lambda_energy !LAM! ^
    --eval_init_lambda_load !LAM! ^
    --out_csv "%EVAL_OUTDIR%\sweep_both_lambda!LAM!_EB%EB%_LB%LB%.csv"
)

REM Energy-only sweep: lambda_load=0, lambda_energy=X
for %%X in (0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.80 1.00) do (
  set "LAM=%%X"
  echo [Sweep] energy lambda = !LAM! (load=0)
  python %SCRIPT_EVAL% ^
    --ckpt_path "!BPLUS_CKPT!" ^
    --num_seeds %EVAL_NUM% ^
    --seed_start %EVAL_SEED_START% ^
    --deterministic True ^
    --device %DEVICE% ^
    --energy_budget %EB% ^
    --load_budget %LB% ^
    --online_dual_update False ^
    --eval_init_lambda_energy !LAM! ^
    --eval_init_lambda_load 0.0 ^
    --out_csv "%EVAL_OUTDIR%\sweep_energy_lambda!LAM!_load0_EB%EB%_LB%LB%.csv"
)

REM Load-only sweep: lambda_energy=0, lambda_load=X
for %%X in (0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.80 1.00) do (
  set "LAM=%%X"
  echo [Sweep] load lambda = !LAM! (energy=0)
  python %SCRIPT_EVAL% ^
    --ckpt_path "!BPLUS_CKPT!" ^
    --num_seeds %EVAL_NUM% ^
    --seed_start %EVAL_SEED_START% ^
    --deterministic True ^
    --device %DEVICE% ^
    --energy_budget %EB% ^
    --load_budget %LB% ^
    --online_dual_update False ^
    --eval_init_lambda_energy 0.0 ^
    --eval_init_lambda_load !LAM! ^
    --out_csv "%EVAL_OUTDIR%\sweep_load_lambda!LAM!_energy0_EB%EB%_LB%LB%.csv"
)

echo Done.
pause
