@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM V5 Baseline: Scalar PPO with Hand-Tuned Weights
REM Goal: Prove Multi-Head > Scalar (Weighted Sum)
REM ============================================================

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set OUTROOT=scalar_ppo_ablation_%TS%
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo ============================================================
echo   V5 Baseline: Scalar PPO (Weighted Sum)
echo   Output root: %OUTROOT%
echo   Start time: %TS%
echo ============================================================

REM ===== 固定超参 =====
set GRID=8
set MAX_STEPS=256
set ITERS=800
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
set LOAD_SCALE=5.0

REM Obs
set INC_CONG=True
set CONG_R=2
set INC_EN=True
set EN_R=1
set EN_NORM=True

REM Reward & step penalty
set SR=20.0
set STEP_PEN=-1.0

REM Skip subroutine definition
goto :MAIN

REM ============================================================
REM Helper: run one config
REM call :RUN <TAG> <ENERGY_WEIGHT> <LOAD_WEIGHT>
REM ============================================================
:RUN
set TAG=%~1
set E_WEIGHT=%~2
set L_WEIGHT=%~3

set OUTDIR=!OUTROOT!\!TAG!
if exist "!OUTDIR!" rd /s /q "!OUTDIR!"
mkdir "!OUTDIR!"

echo.
echo ---------------------------------------------------
echo [RUN] !TAG!
echo   Energy weight (alpha): !E_WEIGHT!
echo   Load weight (beta): !L_WEIGHT!
echo   Load scale: !LOAD_SCALE!
echo ---------------------------------------------------

python train_grid_scalar.py ^
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
  --load_cost_scale !LOAD_SCALE! ^
  --include_congestion_obs !INC_CONG! ^
  --congestion_patch_radius !CONG_R! ^
  --include_energy_obs !INC_EN! ^
  --energy_patch_radius !EN_R! ^
  --energy_obs_normalize !EN_NORM! ^
  --success_reward !SR! ^
  --step_penalty !STEP_PEN! ^
  --energy_weight !E_WEIGHT! ^
  --load_weight !L_WEIGHT! ^
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
REM Main Execution: Test different weight combinations
REM ============================================================
:MAIN

echo.
echo ============================================================
echo [V5] Scalar PPO: Testing Weight Combinations
echo ============================================================

REM Weight 1: Unconstrained (pure reward)
call :RUN ScalarPPO_Unconstrained_α0_β0 0.0 0.0

REM Weight 2: Equal weights
call :RUN ScalarPPO_Equal_α0p5_β0p5 0.5 0.5

REM Weight 3: Energy focused
call :RUN ScalarPPO_EnergyFocus_α1p0_β0p2 1.0 0.2

REM Weight 4: Load focused
call :RUN ScalarPPO_LoadFocus_α0p2_β1p0 0.2 1.0

REM Weight 5: Both constrained (hand-tuned)
call :RUN ScalarPPO_BothHeavy_α0p7_β0p7 0.7 0.7

echo.
echo ============================================================
echo [DONE] Scalar PPO ablation finished.
echo Output root: %OUTROOT%
echo ============================================================

endlocal
exit /b 0
