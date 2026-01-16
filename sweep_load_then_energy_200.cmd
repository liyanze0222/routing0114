@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Sweep Plan:
REM Step1: Sweep Load budgets (L) with fixed E=1.35, 200 iters
REM Step2: For a chosen L (default 0.20), sweep E=1.35 vs 1.37
REM Output: timestamped root folder, one run per subfolder
REM ============================================================

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set OUTROOT=sweep_L_then_E_%TS%
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo ===================================================
echo   SWEEP: Load budgets then Energy budgets (200 iters)
echo   Output root: %OUTROOT%
echo   Start time : %TS%
echo ===================================================

REM ======================
REM Common hyperparams
REM ======================
set GRID=8
set MAX_STEPS=256
set ITERS=200
set BS=2048
set MBS=256
set UE=10
set H=128
set SEED=0
set LOGINT=10

REM Env
set CONG_PATTERN=block
set CONG_DENS=0.40
set E_HIGH=3.0
set E_HIGH_DENS=0.20

REM Obs
set INC_CONG=True
set CONG_R=2
set INC_EN=True
set EN_R=1
set EN_NORM=True

REM Reward & step penalty
set SR=20.0
set STEP_PEN=-1.0

REM Lagrange settings
set USE_LAG=True
set GAP_MODE=ratio
set LLR_E=0.05
set LLR_L=0.05
set INIT_LE=0.0
set INIT_LL=0.0

REM Skip subroutine definition, jump to main execution
goto :MAIN

REM ------------------------------------------------------------
REM Helper label: run one config
REM call :RUN <TAG> <E_BUD> <L_BUD>
REM ------------------------------------------------------------
:RUN
set TAG=%~1
set EB=%~2
set LB=%~3

set OUTDIR=!OUTROOT!\!TAG!
if exist "!OUTDIR!" rd /s /q "!OUTDIR!"
mkdir "!OUTDIR!"

echo.
echo ---------------------------------------------------
echo [RUN] !TAG!
echo   E_budget=!EB!  L_budget=!LB!  iters=!ITERS!
echo ---------------------------------------------------

python train_grid_structured_lagrangian.py ^
  --grid_size !GRID! ^
  --max_steps !MAX_STEPS! ^
  --total_iters !ITERS! ^
  --batch_size !BS! ^
  --minibatch_size !MBS! ^
  --update_epochs !UE! ^
  --hidden_dim !H! ^
  --congestion_pattern !CONG_PATTERN! ^
  --congestion_density !CONG_DENS! ^
  --energy_high_cost !E_HIGH! ^
  --energy_high_density !E_HIGH_DENS! ^
  --include_congestion_obs !INC_CONG! ^
  --congestion_patch_radius !CONG_R! ^
  --include_energy_obs !INC_EN! ^
  --energy_patch_radius !EN_R! ^
  --energy_obs_normalize !EN_NORM! ^
  --success_reward !SR! ^
  --step_penalty !STEP_PEN! ^
  --energy_budget !EB! ^
  --load_budget !LB! ^
  --use_lagrange !USE_LAG! ^
  --lambda_gap_mode !GAP_MODE! ^
  --lambda_lr_energy !LLR_E! ^
  --lambda_lr_load !LLR_L! ^
  --initial_lambda_energy !INIT_LE! ^
  --initial_lambda_load !INIT_LL! ^
  --seed !SEED! ^
  --log_interval !LOGINT! ^
  --output_dir "!OUTDIR!"

if errorlevel 1 (
  echo [FAIL] !TAG! crashed. Continuing...
) else (
  echo [OK]   !TAG! finished.
)
goto :eof

REM ============================================================
REM Step 1: Sweep L budgets with E fixed at 1.35
REM ============================================================
:MAIN
echo.
echo ===================================================
echo [STEP 1] Sweep Load budgets (E fixed at 1.35)
echo   L in (0.205, 0.200, 0.195, 0.190)
echo ===================================================

call :RUN S1_E1p35_L0p205_T200 1.35 0.205
call :RUN S1_E1p35_L0p200_T200 1.35 0.200
call :RUN S1_E1p35_L0p195_T200 1.35 0.195
call :RUN S1_E1p35_L0p190_T200 1.35 0.190

REM ============================================================
REM Step 2: For chosen L (default = 0.200), sweep E budgets
REM You can change CHOSEN_L here if needed.
REM ============================================================
set CHOSEN_L=0.200

echo.
echo ===================================================
echo [STEP 2] Sweep Energy budgets at chosen L=!CHOSEN_L!
echo   E in (1.35, 1.37)
echo ===================================================

call :RUN S2_E1p35_L!CHOSEN_L!_T200 1.35 !CHOSEN_L!
call :RUN S2_E1p37_L!CHOSEN_L!_T200 1.37 !CHOSEN_L!

echo.
echo ===================================================
echo [DONE] Sweep finished.
echo Output root: %OUTROOT%
echo ===================================================

endlocal
exit /b 0
