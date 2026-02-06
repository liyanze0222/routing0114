@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Tight budget diagnostics AFTER fixing obs_clip (Method A)
REM Goal:
REM   1) Confirm eval_lambda_obs_clip now really affects behavior
REM   2) Test if clip prevents OOD collapse under aggressive dual
REM   3) Try a couple "stable online dual" candidates
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E4_BPlus_eval_tight_postC3Fix_v2"
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

REM ---- Tight budgets ----
set "EB=0.05"
set "LB=0.03"

REM ---- Eval settings ----
set "EVAL_NUM=200"
set "EVAL_SEED_START=20000"
set "DEVICE=cpu"

echo ==========================================================
echo [Tight eval post-fix] EB=%EB%  LB=%LB%
echo Output dir: %EVAL_OUTDIR%
echo ==========================================================

REM ----------------------------------------------------------
REM (B0) No online dual (baseline for comparison)
REM ----------------------------------------------------------
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update False ^
  --out_csv "%EVAL_OUTDIR%\B0_noDual_EB%EB%_LB%LB%.csv"

REM ----------------------------------------------------------
REM (C0) Online dual original (reference)
REM ----------------------------------------------------------
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

REM ----------------------------------------------------------
REM (C3a) Online dual + obs clip to ~training range (0.30)
REM       (If this fixes timeout, it's strong evidence OOD is the trigger)
REM ----------------------------------------------------------
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_obs_clip 0.30 ^
  --dual_trace_csv "%EVAL_OUTDIR%\C3a_clip0p30_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C3a_clip0p30_EB%EB%_LB%LB%.csv"

REM ----------------------------------------------------------
REM (C3b) Online dual + obs clip 0.50 (more room, still tries to avoid OOD)
REM ----------------------------------------------------------
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_obs_clip 0.50 ^
  --dual_trace_csv "%EVAL_OUTDIR%\C3b_clip0p50_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\C3b_clip0p50_EB%EB%_LB%LB%.csv"

REM ----------------------------------------------------------
REM (S1) Stable candidate:
REM      reduce lr + slower update + larger deadband + clamp max + clip obs
REM ----------------------------------------------------------
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_lr 0.01 ^
  --eval_lambda_update_freq 5 ^
  --eval_dual_deadband 0.02 ^
  --eval_lambda_max 0.50 ^
  --eval_lambda_obs_clip 0.50 ^
  --dual_trace_csv "%EVAL_OUTDIR%\S1_lr0p01_f5_db0p02_max0p5_clip0p5_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\S1_lr0p01_f5_db0p02_max0p5_clip0p5_EB%EB%_LB%LB%.csv"

REM ----------------------------------------------------------
REM (S2) Ratio gap mode (helps one lr scale across different budgets):
REM      use smaller lr because ratio gap can be larger in tight budgets
REM ----------------------------------------------------------
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic True ^
  --device %DEVICE% ^
  --energy_budget %EB% ^
  --load_budget %LB% ^
  --online_dual_update True ^
  --eval_lambda_gap_mode ratio ^
  --eval_lambda_lr 0.002 ^
  --eval_lambda_update_freq 5 ^
  --eval_dual_deadband 0.02 ^
  --eval_lambda_max 0.50 ^
  --eval_lambda_obs_clip 0.50 ^
  --dual_trace_csv "%EVAL_OUTDIR%\S2_ratio_lr0p002_f5_db0p02_max0p5_clip0p5_dualtrace_EB%EB%_LB%LB%.csv" ^
  --out_csv "%EVAL_OUTDIR%\S2_ratio_lr0p002_f5_db0p02_max0p5_clip0p5_EB%EB%_LB%LB%.csv"

echo Done.
pause
