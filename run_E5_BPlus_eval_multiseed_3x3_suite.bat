@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM E5-1: Multi-seed replication on 3x3 budgets
REM Compare:
REM   B0: B-Plus no dual
REM   C0: B-Plus original online dual
REM   D : B-Plus stable online dual (ratio + lr/freq/deadband + max/clip + lambda_min)
REM NOTE: D uses --eval_lambda_min (needs eval_fixed_set.py patch)
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E5_BPlus_eval_multiseed_3x3_suite"
if not exist "%EVAL_OUTDIR%" mkdir "%EVAL_OUTDIR%"

REM ---- Eval settings (align with your successful bats) ----
set "EVAL_NUM=1000"
set "EVAL_SEED_START=10000"
set "DEVICE=cpu"
set "DETERMINISTIC=True"

REM ---- Stable dual config (your chosen D) ----
set "GAPMODE=ratio"
set "LR=0.002"
set "FREQ=5"
set "DEADBAND=0.02"
set "LAMMAX=0.40"
set "OBSCLIP=0.40"
set "LAMMIN=0.20"

echo ==========================================================
echo [E5-1] Multi-seed 3x3 suite
echo out: %EVAL_OUTDIR%
echo eval_num=%EVAL_NUM% seed_start=%EVAL_SEED_START%
echo D config: gap=%GAPMODE% lr=%LR% freq=%FREQ% deadband=%DEADBAND% max=%LAMMAX% clip=%OBSCLIP% min=%LAMMIN%
echo ==========================================================

for %%S in (0 1 2 3 4) do (

  echo ==========================================================
  echo [Seed %%S]
  echo ==========================================================

  set "TAG_BPLUS=E4B_bplus_seed%%S"
  set "BPLUS_DIR="
  for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTDIR%\!TAG_BPLUS!*" 2^>nul') do (
    if not defined BPLUS_DIR set "BPLUS_DIR=%OUTDIR%\%%D"
  )

  if not defined BPLUS_DIR (
    echo [WARN] No B-Plus dir found for seed %%S, skip.
  ) else (
    set "BPLUS_CKPT=!BPLUS_DIR!\best_feasible.pt"
    if not exist "!BPLUS_CKPT!" set "BPLUS_CKPT=!BPLUS_DIR!\checkpoint_final.pt"
    if not exist "!BPLUS_CKPT!" (
      echo [WARN] No checkpoint found for seed %%S, skip.
    ) else (

      set "SEED_OUT=%EVAL_OUTDIR%\seed%%S"
      if not exist "!SEED_OUT!" mkdir "!SEED_OUT!"

      echo Using ckpt: !BPLUS_CKPT!

      for %%E in (0.05 0.10 0.15) do (
        for %%L in (0.03 0.05 0.07) do (

          echo ---- EB=%%E  LB=%%L ----

          REM (B0) no dual
          python %SCRIPT_EVAL% ^
            --ckpt_path "!BPLUS_CKPT!" ^
            --num_seeds %EVAL_NUM% ^
            --seed_start %EVAL_SEED_START% ^
            --deterministic %DETERMINISTIC% ^
            --device %DEVICE% ^
            --energy_budget %%E ^
            --load_budget %%L ^
            --online_dual_update False ^
            --out_csv "!SEED_OUT!\B0_noDual_EB%%E_LB%%L.csv"

          REM (C0) original online dual
          python %SCRIPT_EVAL% ^
            --ckpt_path "!BPLUS_CKPT!" ^
            --num_seeds %EVAL_NUM% ^
            --seed_start %EVAL_SEED_START% ^
            --deterministic %DETERMINISTIC% ^
            --device %DEVICE% ^
            --energy_budget %%E ^
            --load_budget %%L ^
            --online_dual_update True ^
            --dual_trace_csv "!SEED_OUT!\C0_dualtrace_EB%%E_LB%%L.csv" ^
            --out_csv "!SEED_OUT!\C0_origDual_EB%%E_LB%%L.csv"

          REM (D) stable online dual
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
            --eval_lambda_min %LAMMIN% ^
            --dual_trace_csv "!SEED_OUT!\D_dualtrace_EB%%E_LB%%L.csv" ^
            --out_csv "!SEED_OUT!\D_stableDual_EB%%E_LB%%L.csv"
        )
      )
    )
  )
)

echo Done.
pause
