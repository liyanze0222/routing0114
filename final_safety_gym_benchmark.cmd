@echo off
setlocal enabledelayedexpansion

set OUTROOT=final_safety_gym_benchmark
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo ===================================================
echo   FINAL MISSION: The Twin Peaks
echo   1. Stable Run (L=0.205): Show Convergence
echo   2. Stress Run (L=0.200): Show Trade-off
echo ===================================================

:: =================================================
:: Run 1: The Stable Candidate (L=0.205)
:: Expected: High feasible rate, stable Lambda
:: =================================================
set TAG=Final_Stable_E1.35_L0.205_R20
set OUTDIR=%OUTROOT%\%TAG%

if exist "%OUTDIR%" (
    echo [SKIP] %TAG% already exists.
) else (
    echo [START] Running Stable Config...
    python train_grid_structured_lagrangian.py ^
        --grid_size 8 ^
        --max_steps 256 ^
        --total_iters 800 ^
        --batch_size 2048 ^
        --minibatch_size 256 ^
        --update_epochs 10 ^
        --hidden_dim 128 ^
        --congestion_pattern block ^
        --congestion_density 0.40 ^
        --energy_high_cost 3.0 ^
        --energy_high_density 0.20 ^
        --include_congestion_obs True ^
        --congestion_patch_radius 2 ^
        --include_energy_obs True ^
        --energy_patch_radius 1 ^
        --energy_obs_normalize True ^
        --success_reward 20.0 ^
        --energy_budget 1.35 ^
        --load_budget 0.205 ^
        --use_lagrange True ^
        --lambda_gap_mode ratio ^
        --lambda_lr_energy 0.05 ^
        --lambda_lr_load 0.05 ^
        --initial_lambda_energy 0.0 ^
        --initial_lambda_load 0.0 ^
        --seed 0 ^
        --output_dir "%OUTDIR%"
)

:: =================================================
:: Run 2: The Stress Test (L=0.200)
:: Expected: High Lambda, hovering around budget
:: =================================================
set TAG=Final_Stress_E1.35_L0.200_R20
set OUTDIR=%OUTROOT%\%TAG%

if exist "%OUTDIR%" (
    echo [SKIP] %TAG% already exists.
) else (
    echo [START] Running Stress Config...
    python train_grid_structured_lagrangian.py ^
        --grid_size 8 ^
        --max_steps 256 ^
        --total_iters 800 ^
        --batch_size 2048 ^
        --minibatch_size 256 ^
        --update_epochs 10 ^
        --hidden_dim 128 ^
        --congestion_pattern block ^
        --congestion_density 0.40 ^
        --energy_high_cost 3.0 ^
        --energy_high_density 0.20 ^
        --include_congestion_obs True ^
        --congestion_patch_radius 2 ^
        --include_energy_obs True ^
        --energy_patch_radius 1 ^
        --energy_obs_normalize True ^
        --success_reward 20.0 ^
        --energy_budget 1.35 ^
        --load_budget 0.20 ^
        --use_lagrange True ^
        --lambda_gap_mode ratio ^
        --lambda_lr_energy 0.05 ^
        --lambda_lr_load 0.05 ^
        --initial_lambda_energy 0.0 ^
        --initial_lambda_load 0.0 ^
        --seed 0 ^
        --output_dir "%OUTDIR%"
)

echo.
echo [DONE] Both runs finished.
echo Now please run: python batch_eval_all.py
pause