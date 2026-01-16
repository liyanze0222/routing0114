@echo off
setlocal enabledelayedexpansion

:: =========================================================
:: MACHINE 1: MAIN BENCHMARK (Pixel-Level Reproduction)
:: =========================================================
:: 核心修复：
:: 1. [致命] 显式指定 congestion_pattern=block, density=0.4 (恢复环境结构)
:: 2. [关键] 显式指定 max_steps=256
:: 3. [关键] 显式指定 success_reward=20.0, step_penalty=-1.0
:: 4. [确认] 保留 load_cost_scale=5.0 (配合当前无硬编码的代码)
:: =========================================================

set "ENERGY_BUDGET=1.35"
set "SCALE=5.0"
set "OUT_ROOT=outputs/final_benchmark_repro"

:: 1. 环境物理参数 (必须与旧 Config 完全一致)
set "ENV_ARGS=--grid_size 8 --max_steps 256 --step_penalty -1.0 --success_reward 20.0 --congestion_pattern block --congestion_density 0.40 --energy_high_cost 3.0 --energy_high_density 0.20"

:: 2. 观测参数 (开眼)
set "OBS_ARGS=--include_congestion_obs True --congestion_patch_radius 1 --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True"

:: 3. 算法参数 (LR=0.05, Ratio Mode)
set "PPO_ARGS=--use_lagrange True --lambda_gap_mode ratio --lambda_lr_energy 0.05 --lambda_lr_load 0.05 --initial_lambda_energy 0.0 --initial_lambda_load 0.0"

:: 4. 组合所有通用参数
:: 注意：这里显式传入 load_cost_scale=5.0，因为 grid_cost_env.py 里没有硬编码
set "COMMON_ARGS=%ENV_ARGS% %OBS_ARGS% %PPO_ARGS% --energy_budget %ENERGY_BUDGET% --load_cost_scale %SCALE% --total_iters 800 --batch_size 2048 --minibatch_size 256 --update_epochs 10 --hidden_dim 128 --seed 0"

echo.
echo ==========================================================
echo STARTING MAIN BENCHMARK (Exact Reproduction Mode)
echo ==========================================================

:: 依次执行三组预算
call :RunGroup stable 1.025
call :RunGroup stress 1.000
call :RunGroup extreme 0.900

echo.
echo ==========================================================
echo MAIN BENCHMARK FINISHED!
echo ==========================================================
pause
goto :eof


:: ========== 子程序 ==========
:RunGroup
set G_NAME=%1
set L_BUDGET=%2
set CURR_OUT=%OUT_ROOT%/%G_NAME%

echo.
:: Using brackets instead of >>> to avoid accidental redirection parsing errors.
echo   [Group: %G_NAME% (Budget=%L_BUDGET%)]

:: 1. V0: Proposed (Hero)
echo       Running V0 (Proposed)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v0_proposed" --load_budget %L_BUDGET% --cost_critic_mode separate

:: 2. V1: Unconstrained (Base)
echo       Running V1 (Unconstrained)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v1_unconstrained" --load_budget %L_BUDGET% --use_lagrange False

:: 3. V2: Shared Critic (Ablation)
echo       Running V2 (Shared)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v2_shared" --load_budget %L_BUDGET% --cost_critic_mode shared

:: 4. V5: Scalar PPO (Baseline)
echo       Running V5 (Scalar)...
python train_grid_scalar.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v5_scalar" --energy_weight 0.5 --load_weight 1.5

:: 5. V3: Energy Only
echo       Running V3 (Energy Only)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v3_energy_only" --load_budget 100.0 --cost_critic_mode separate

:: 6. V4: Load Only
echo       Running V4 (Load Only)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v4_load_only" --energy_budget 100.0 --load_budget %L_BUDGET% --cost_critic_mode separate

goto :eof