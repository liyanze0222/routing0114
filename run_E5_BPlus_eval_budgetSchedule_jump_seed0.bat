@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM E5-3: Dynamic budget jump test (single run)
REM budgets change per episode according to a CSV schedule
REM Compare: B0(no dual) vs D(stable dual)
REM ==========================================================

set "SCRIPT_EVAL=eval_fixed_set.py"

set "OUTDIR=outputs\E4_BPlus"
set "EVAL_OUTDIR=outputs\E5_BPlus_eval_budgetSchedule_jump_seed0"
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

REM ---- Create budget schedule CSV (400 episodes) ----
set "SCHED=%EVAL_OUTDIR%\budget_schedule.csv"
echo energy_budget,load_budget> "%SCHED%"

REM 1) loose 100 eps: (0.15, 0.07)
for /L %%i in (1,1,100) do echo 0.15,0.07>> "%SCHED%"

REM 2) tight 100 eps: (0.05, 0.03)
for /L %%i in (1,1,100) do echo 0.05,0.03>> "%SCHED%"

REM 3) mid 100 eps: (0.10, 0.05)
for /L %%i in (1,1,100) do echo 0.10,0.05>> "%SCHED%"

REM 4) back to loose 100 eps
for /L %%i in (1,1,100) do echo 0.15,0.07>> "%SCHED%"

echo Schedule saved: %SCHED%

REM ---- Eval settings ----
set "EVAL_NUM=400"
set "EVAL_SEED_START=50000"
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

REM (B0) no dual (budgets still affect feasibility metrics, but policy won't adapt)
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic %DETERMINISTIC% ^
  --device %DEVICE% ^
  --online_dual_update False ^
  --budget_schedule_csv "%SCHED%" ^
  --budget_schedule_mode cycle ^
  --out_csv "%EVAL_OUTDIR%\B0_noDual_budgetJump.csv"

REM (D) stable dual (should adapt lambdas across jumps)
python %SCRIPT_EVAL% ^
  --ckpt_path "!BPLUS_CKPT!" ^
  --num_seeds %EVAL_NUM% ^
  --seed_start %EVAL_SEED_START% ^
  --deterministic %DETERMINISTIC% ^
  --device %DEVICE% ^
  --online_dual_update True ^
  --budget_schedule_csv "%SCHED%" ^
  --budget_schedule_mode cycle ^
  --eval_lambda_gap_mode %GAPMODE% ^
  --eval_lambda_lr %LR% ^
  --eval_lambda_update_freq %FREQ% ^
  --eval_dual_deadband %DEADBAND% ^
  --eval_lambda_max %LAMMAX% ^
  --eval_lambda_obs_clip %OBSCLIP% ^
  --eval_lambda_min %LAMMIN% ^
  --dual_trace_csv "%EVAL_OUTDIR%\D_dualtrace_budgetJump.csv" ^
  --out_csv "%EVAL_OUTDIR%\D_stableDual_budgetJump.csv"

echo Done.
pause
