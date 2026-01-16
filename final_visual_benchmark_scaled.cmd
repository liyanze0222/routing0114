@echo off
setlocal enabledelayedexpansion

set OUTROOT=final_visual_benchmark_scaled
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo =========================================================
echo   FINAL VISUAL MISSION: SCALED COST (x5)
echo   Target: Show DRAMATIC load reduction curves.
echo =========================================================

:: -----------------------------------------------------------
:: Group 1: Stable Scaled (Orig L=0.205 -> New L=1.025)
:: -----------------------------------------------------------
set TAG=Scaled_Stable_E1.35_L1.025
echo [START] %TAG% ...
python train_grid_structured_lagrangian.py ^
    --grid_size 8 --max_steps 256 --total_iters 800 ^
    --batch_size 2048 --minibatch_size 256 --update_epochs 10 ^
    --hidden_dim 128 --congestion_density 0.40 --congestion_pattern block ^
    --include_congestion_obs True --include_energy_obs True ^
    --energy_obs_normalize True --success_reward 20.0 ^
    --use_lagrange True --lambda_gap_mode ratio ^
    --lambda_lr_energy 0.05 --lambda_lr_load 0.05 ^
    --energy_budget 1.35 ^
    --load_budget 1.025 ^
    --seed 0 ^
    --output_dir "%OUTROOT%\%TAG%"

:: -----------------------------------------------------------
:: Group 2: Stress Scaled (Orig L=0.200 -> New L=1.000)
:: -----------------------------------------------------------
set TAG=Scaled_Stress_E1.35_L1.000
echo [START] %TAG% ...
python train_grid_structured_lagrangian.py ^
    --grid_size 8 --max_steps 256 --total_iters 800 ^
    --batch_size 2048 --minibatch_size 256 --update_epochs 10 ^
    --hidden_dim 128 --congestion_density 0.40 --congestion_pattern block ^
    --include_congestion_obs True --include_energy_obs True ^
    --energy_obs_normalize True --success_reward 20.0 ^
    --use_lagrange True --lambda_gap_mode ratio ^
    --lambda_lr_energy 0.05 --lambda_lr_load 0.05 ^
    --energy_budget 1.35 ^
    --load_budget 1.000 ^
    --seed 0 ^
    --output_dir "%OUTROOT%\%TAG%"

:: -----------------------------------------------------------
:: Group 3: Extreme Scaled (Orig L=0.180 -> New L=0.900)
:: -----------------------------------------------------------
set TAG=Scaled_Extreme_E1.35_L0.900
echo [START] %TAG% ...
python train_grid_structured_lagrangian.py ^
    --grid_size 8 --max_steps 256 --total_iters 800 ^
    --batch_size 2048 --minibatch_size 256 --update_epochs 10 ^
    --hidden_dim 128 --congestion_density 0.40 --congestion_pattern block ^
    --include_congestion_obs True --include_energy_obs True ^
    --energy_obs_normalize True --success_reward 20.0 ^
    --use_lagrange True --lambda_gap_mode ratio ^
    --lambda_lr_energy 0.05 --lambda_lr_load 0.05 ^
    --energy_budget 1.35 ^
    --load_budget 0.900 ^
    --seed 0 ^
    --output_dir "%OUTROOT%\%TAG%"

echo [DONE] All visual experiments finished.
pause