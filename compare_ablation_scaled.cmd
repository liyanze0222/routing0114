@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM 批量对照实验：4 类 Variant × 2 个 Budget 档位
REM 核心设定：load cost × 5, load budget × 5（与 energy 同尺度）
REM ============================================================

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set OUTROOT=ablation_scaled_%TS%
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo ============================================================
echo   Ablation Study: Scaled Load Cost (×5)
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

REM Lagrange settings
set GAP_MODE=ratio
set INIT_LE=0.0
set INIT_LL=0.0

REM Energy budget (固定)
set E_BUDGET=1.35

REM Skip subroutine definition
goto :MAIN

REM ============================================================
REM Helper: run one config
REM call :RUN <TAG> <L_BUDGET> <USE_LAG> <COST_MODE> <LLR_E> <LLR_L>
REM ============================================================
:RUN
set TAG=%~1
set L_BUDGET=%~2
set USE_LAG=%~3
set COST_MODE=%~4
set LLR_E=%~5
set LLR_L=%~6

set OUTDIR=!OUTROOT!\!TAG!
if exist "!OUTDIR!" rd /s /q "!OUTDIR!"
mkdir "!OUTDIR!"

echo.
echo ---------------------------------------------------
echo [RUN] !TAG!
echo   E_budget=!E_BUDGET!  L_budget=!L_BUDGET!  load_scale=!LOAD_SCALE!
echo   use_lagrange=!USE_LAG!  cost_critic_mode=!COST_MODE!
echo   lambda_lr: E=!LLR_E!, L=!LLR_L!
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
  --load_cost_scale !LOAD_SCALE! ^
  --include_congestion_obs !INC_CONG! ^
  --congestion_patch_radius !CONG_R! ^
  --include_energy_obs !INC_EN! ^
  --energy_patch_radius !EN_R! ^
  --energy_obs_normalize !EN_NORM! ^
  --success_reward !SR! ^
  --step_penalty !STEP_PEN! ^
  --energy_budget !E_BUDGET! ^
  --load_budget !L_BUDGET! ^
  --use_lagrange !USE_LAG! ^
  --cost_critic_mode !COST_MODE! ^
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
REM Main Execution: 2 Budget Levels × 4 Variants = 8 Runs
REM ============================================================
:MAIN

echo.
echo ============================================================
echo [BUDGET 1] Stable: E=1.35, L_scaled=1.025 (≈ 0.205×5)
echo ============================================================

REM Variant 1: Unconstrained Multi-head PPO
call :RUN Stable_Unconstrained_MultiHead 1.025 False separate 0.0 0.0

REM Variant 2: Single-head PPO (shared cost head), both constraints on
call :RUN Stable_SharedCost_BothOn 1.025 True shared 0.05 0.05

REM Variant 3: Multi-head PPO, energy-only constraint
call :RUN Stable_MultiHead_EnergyOnly 1.025 True separate 0.05 0.0

REM Variant 4: Multi-head PPO, load-only constraint
call :RUN Stable_MultiHead_LoadOnly 1.025 True separate 0.0 0.05

echo.
echo ============================================================
echo [BUDGET 2] Stress: E=1.35, L_scaled=1.000 (≈ 0.200×5)
echo ============================================================

REM Variant 1: Unconstrained Multi-head PPO
call :RUN Stress_Unconstrained_MultiHead 1.000 False separate 0.0 0.0

REM Variant 2: Single-head PPO (shared cost head), both constraints on
call :RUN Stress_SharedCost_BothOn 1.000 True shared 0.05 0.05

REM Variant 3: Multi-head PPO, energy-only constraint
call :RUN Stress_MultiHead_EnergyOnly 1.000 True separate 0.05 0.0

REM Variant 4: Multi-head PPO, load-only constraint
call :RUN Stress_MultiHead_LoadOnly 1.000 True separate 0.0 0.05

echo.
echo ============================================================
echo [DONE] Ablation study finished.
echo Output root: %OUTROOT%
echo ============================================================

endlocal
exit /b 0
