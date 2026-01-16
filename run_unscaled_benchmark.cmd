@echo off
setlocal enabledelayedexpansion

:: =========================================================
:: MACHINE 2: CURRICULUM + UNSCALED (Exact Reproduction)
:: =========================================================

:: 必须同步所有环境参数，确保地图结构一致
set "ENV_ARGS=--grid_size 8 --max_steps 256 --step_penalty -1.0 --success_reward 20.0 --congestion_pattern block --congestion_density 0.40 --energy_high_cost 3.0 --energy_high_density 0.20"
set "OBS_ARGS=--include_congestion_obs True --congestion_patch_radius 1 --include_energy_obs True --energy_patch_radius 1 --energy_obs_normalize True"
:: Unscaled 模式下，LR 保持 0.05 也是合理的，因为 gap 变小了，LR*gap 的更新步长会自动适应
set "PPO_ARGS=--use_lagrange True --lambda_gap_mode ratio --lambda_lr_energy 0.05 --lambda_lr_load 0.05 --initial_lambda_energy 0.0 --initial_lambda_load 0.0"

:: ---------------------------------------------------------
:: PHASE 1: Curriculum Learning (Scale=5.0)
:: ---------------------------------------------------------
echo.
echo ==========================================================
echo [PHASE 1/2] Curriculum Learning (Block Map, Scale 5.0)
echo ==========================================================

set "E_BUDGET=1.35"
set "SCALE=5.0"
set "OUT_DIR=outputs/curriculum_experiment"
set "CURR_ARGS=--use_curriculum True --curriculum_start_load_budget 2.5 --curriculum_end_load_budget 0.9 --curriculum_iters 600"

:: Use brackets to avoid accidental redirection and flatten commands to prevent stray tokens executing.
echo   [Running V0 (Proposed) with Curriculum...]
python train_grid_structured_lagrangian.py --energy_budget %E_BUDGET% --load_cost_scale %SCALE% --total_iters 800 --seed 0 %ENV_ARGS% %OBS_ARGS% %PPO_ARGS% --run_tag "%OUT_DIR%/v0_curriculum" --cost_critic_mode separate %CURR_ARGS%

echo   [Phase 1 Completed.]

:: ---------------------------------------------------------
:: PHASE 2: Unscaled Benchmark (Scale=1.0)
:: ---------------------------------------------------------
echo.
echo ==========================================================
echo [PHASE 2/2] Unscaled Benchmark (Block Map, Scale 1.0)
echo ==========================================================

set "SCALE=1.0"
set "OUT_ROOT=outputs/final_benchmark_unscaled"
:: 注意：Unscaled 组把 load_cost_scale 强制设为 1.0，但其他 ENV 参数保持 block/0.4 不变
set "COMMON_ARGS=%ENV_ARGS% %OBS_ARGS% %PPO_ARGS% --energy_budget 1.35 --load_cost_scale 1.0 --total_iters 800 --batch_size 2048 --minibatch_size 256 --update_epochs 10 --hidden_dim 128 --seed 0"

call :RunGroup stable 0.205
call :RunGroup stress 0.200
call :RunGroup extreme 0.180

echo.
echo ==========================================================
echo ALL TASKS ON MACHINE 2 FINISHED!
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
echo   [Group: %G_NAME% (Raw Budget=%L_BUDGET%)]

:: 1. V0: Proposed
echo       Running V0 (Proposed)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v0_proposed" --load_budget %L_BUDGET% --cost_critic_mode separate

:: 2. V1: Unconstrained
echo       Running V1 (Unconstrained)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v1_unconstrained" --load_budget %L_BUDGET% --use_lagrange False

:: 3. V2: Shared Critic
echo       Running V2 (Shared)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v2_shared" --load_budget %L_BUDGET% --cost_critic_mode shared

:: 4. V5: Scalar PPO (Weight x5)
:: Unscaled 模式下，Cost 小了 5 倍，权重需要 x5 (7.5) 才能保持惩罚力度
echo       Running V5 (Scalar)...
python train_grid_scalar.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v5_scalar" --energy_weight 0.5 --load_weight 7.5

:: 5. V3: Energy Only
echo       Running V3 (Energy Only)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v3_energy_only" --load_budget 100.0 --cost_critic_mode separate

:: 6. V4: Load Only
echo       Running V4 (Load Only)...
python train_grid_structured_lagrangian.py %COMMON_ARGS% --run_tag "%CURR_OUT%/v4_load_only" --energy_budget 100.0 --load_budget %L_BUDGET% --cost_critic_mode separate

goto :eof