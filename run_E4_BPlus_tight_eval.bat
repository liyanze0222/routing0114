@echo off
setlocal enabledelayedexpansion

REM Tight budget diagnostics for B-Plus C-group eval
REM Pulls the tight point (EB=0.05, LB=0.03) and runs online-dual variants

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E4_BPlus_eval_tight"
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

echo ==========================================================
echo [Tight eval] EB=0.05  LB=0.03
echo ==========================================================

REM ---- Tight budgets ----
set "EB=0.05"
set "LB=0.03"

REM ---- Eval settings ----
set "EVAL_NUM=200"
set "EVAL_SEED_START=20000"
set "DEVICE=cpu"

REM ===== Debug only on the tightest budget =====
REM (C0) Original: online dual + deterministic
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --dual_trace_csv "%EVAL_OUTDIR%\C0_orig_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C0_orig_EB%EB%_LB%LB%.csv"

REM (C1) Reset lambdas each episode
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --reset_lambdas_each_episode True ^
  --dual_trace_csv "%EVAL_OUTDIR%\C1_resetLambda_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C1_resetLambda_EB%EB%_LB%LB%.csv"

REM (C2) Shrink lambda lr by 10x
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_lr 0.005 ^
  --dual_trace_csv "%EVAL_OUTDIR%\C2_lr0p005_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C2_lr0p005_EB%EB%_LB%LB%.csv"

REM (C3) Clip lambda obs + max
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_obs_clip 0.5 ^
  --eval_lambda_max 1.0 ^
  --dual_trace_csv "%EVAL_OUTDIR%\C3_clip0p5_max1_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C3_clip0p5_max1_EB%EB%_LB%LB%.csv"

REM (C4) Use stochastic policy during eval
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic False ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --dual_trace_csv "%EVAL_OUTDIR%\C4_stochastic_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C4_stochastic_EB%EB%_LB%LB%.csv"

echo Done.
pause
