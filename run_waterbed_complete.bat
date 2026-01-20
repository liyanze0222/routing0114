@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: WATERBED EFFECT VERIFICATION SUITE
:: Goal: Prove the Pareto trade-off between Load and Energy.
:: ============================================================

set "PY=python"
set "ENTRY=train_grid_structured_lagrangian.py"
set "OUTDIR=outputs\waterbed_effect_verification_20260120"

:: ---- Env Config ----
set "GRID=8"
set "MAX_STEPS=256"
set "CONG_PATTERN=block"
set "CONG_DENS=0.40"
set "SUCCESS_R=20.0"
set "STEP_P=-1.0"
set "E_HIGH_COST=3.0"
set "E_HIGH_DENS=0.20"
set "LOAD_SCALE=5.0"

:: ---- PPO / Dual Config (Decorrelated is best for precision) ----
set "ITERS=800"
set "BATCH=2048"
set "MINIBATCH=256"
set "EPOCHS=10"
set "USE_LAGR=True"
set "GAP_MODE=ratio"
set "LAMBDA_LR=0.05"
set "DUAL_MODE=decorrelated"

:: ============================================================
:: EXPERIMENT A: The Load Squeeze (左侧挤压)
:: Tighten Load to 0.50. Let Energy be loose (1.35).
:: Hypothesis: Agent detours to avoid congestion -> Energy Explodes.
:: ============================================================
set "RUN_TAG_A=ExpA_Squeeze_Load_L0.50"
set "L_BUDGET_A=0.50"
set "E_BUDGET_A=1.35"

echo.
echo [TEST A] Squeezing Load to %L_BUDGET_A%...
echo Expectation: Load OK, Energy Violation High.

%PY% %ENTRY% ^
  --output_dir "%OUTDIR%" ^
  --run_tag "%RUN_TAG_A%" ^
  --seed 0 ^
  --total_iters %ITERS% ^
  --batch_size %BATCH% ^
  --minibatch_size %MINIBATCH% ^
  --update_epochs %EPOCHS% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --success_reward %SUCCESS_R% ^
  --step_penalty %STEP_P% ^
  --energy_high_cost %E_HIGH_COST% ^
  --energy_high_density %E_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --load_cost_scale %LOAD_SCALE% ^
  --start_goal_mode "rect" ^
  --start_rect 0,1,0,1 ^
  --goal_rect 6,7,6,7 ^
  --include_congestion_obs True ^
  --congestion_patch_radius 1 ^
  --include_energy_obs True ^
  --energy_patch_radius 1 ^
  --energy_obs_normalize True ^
  --use_lagrange %USE_LAGR% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR% ^
  --lambda_lr_load %LAMBDA_LR% ^
  --dual_update_mode %DUAL_MODE% ^
  --energy_budget %E_BUDGET_A% ^
  --load_budget %L_BUDGET_A%

:: ============================================================
:: EXPERIMENT B: The Energy Squeeze (右侧挤压)
:: Tighten Energy to 1.20. Let Load be loose (0.60).
:: Hypothesis: Agent takes shortcuts through traffic -> Load Explodes.
:: ============================================================
set "RUN_TAG_B=ExpB_Squeeze_Energy_E1.20"
set "L_BUDGET_B=0.60"
set "E_BUDGET_B=1.20"

echo.
echo [TEST B] Squeezing Energy to %E_BUDGET_B%...
echo Expectation: Energy OK, Load Violation High.

%PY% %ENTRY% ^
  --output_dir "%OUTDIR%" ^
  --run_tag "%RUN_TAG_B%" ^
  --seed 0 ^
  --total_iters %ITERS% ^
  --batch_size %BATCH% ^
  --minibatch_size %MINIBATCH% ^
  --update_epochs %EPOCHS% ^
  --grid_size %GRID% ^
  --max_steps %MAX_STEPS% ^
  --success_reward %SUCCESS_R% ^
  --step_penalty %STEP_P% ^
  --energy_high_cost %E_HIGH_COST% ^
  --energy_high_density %E_HIGH_DENS% ^
  --congestion_pattern %CONG_PATTERN% ^
  --congestion_density %CONG_DENS% ^
  --load_cost_scale %LOAD_SCALE% ^
  --start_goal_mode "rect" ^
  --start_rect 0,1,0,1 ^
  --goal_rect 6,7,6,7 ^
  --include_congestion_obs True ^
  --congestion_patch_radius 1 ^
  --include_energy_obs True ^
  --energy_patch_radius 1 ^
  --energy_obs_normalize True ^
  --use_lagrange %USE_LAGR% ^
  --lambda_gap_mode %GAP_MODE% ^
  --lambda_lr_energy %LAMBDA_LR% ^
  --lambda_lr_load %LAMBDA_LR% ^
  --dual_update_mode %DUAL_MODE% ^
  --energy_budget %E_BUDGET_B% ^
  --load_budget %L_BUDGET_B%

echo.
echo ============================================================
echo DONE.
echo Please inspect metrics.json in: %OUTDIR%
echo Look for 'avg_cost_energy' vs 'avg_cost_load' in the last iteration.
echo ============================================================
pause