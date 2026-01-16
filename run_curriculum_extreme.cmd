@echo off
REM ========================================
REM Curriculum Learning - Extreme Difficulty
REM ========================================
REM
REM 创新：通过课程学习解决 "Extreme" 约束问题
REM 策略：Load Budget 从 2.5 线性衰减至 0.9（600 iters）
REM Agent：V0 (Multi-Head Separate Lagrange)
REM
REM Expected Results:
REM   - 早期（iter 1-200）: 高成功率，约束宽松
REM   - 中期（iter 200-600）: 逐步收紧约束，成功率稳定下降
REM   - 后期（iter 600-800）: 约束固定在 0.9，成功率稳定在 50-70%
REM
REM ========================================

setlocal EnableDelayedExpansion

REM 基础配置
set TOTAL_ITERS=800
set BATCH_SIZE=2048
set MINIBATCH=256
set UPDATE_EPOCHS=10
set LR=0.0003

REM 环境配置（与 Full Benchmark 保持一致）
set GRID_SIZE=8
set CONGESTION_PATTERN=block
set CONGESTION_DENSITY=0.4
set ENERGY_HIGH_DENSITY=0.2
set ENERGY_HIGH_COST=3.0

REM 观测配置
set INCLUDE_CONGESTION=true
set CONGESTION_RADIUS=2
set INCLUDE_ENERGY=true
set ENERGY_RADIUS=1

REM 约束配置
set ENERGY_BUDGET=1.35
set LOAD_COST_SCALE=5.0

REM 课程学习配置
set USE_CURRICULUM=true
set CURRICULUM_START=2.5
set CURRICULUM_END=0.9
set CURRICULUM_ITERS=600

REM Lagrangian 配置（与 V0 Proposed 保持一致）
set USE_LAGRANGE=true
set LAMBDA_LR_ENERGY=0.002
set LAMBDA_LR_LOAD=0.002

REM Cost Critic 配置
set COST_CRITIC_MODE=separate

REM 输出配置
set OUTPUT_BASE=outputs
set RUN_TAG=curriculum_extreme

REM ========================================
REM 执行训练
REM ========================================

echo.
echo ========================================
echo Curriculum Learning - Extreme Difficulty
echo ========================================
echo.
echo Configuration:
echo   - Total Iterations: %TOTAL_ITERS%
echo   - Energy Budget: %ENERGY_BUDGET% (fixed)
echo   - Load Budget: %CURRICULUM_START% -^> %CURRICULUM_END% (over %CURRICULUM_ITERS% iters)
echo   - Load Cost Scale: %LOAD_COST_SCALE%x
echo   - Lambda LR: Energy=%LAMBDA_LR_ENERGY%, Load=%LAMBDA_LR_LOAD%
echo   - Cost Critic Mode: %COST_CRITIC_MODE%
echo   - Output: %OUTPUT_BASE%_%RUN_TAG%
echo.
echo Starting training...
echo.

python train_grid_structured_lagrangian.py ^
    --seed 0 ^
    --total_iters %TOTAL_ITERS% ^
    --batch_size %BATCH_SIZE% ^
    --minibatch_size %MINIBATCH% ^
    --update_epochs %UPDATE_EPOCHS% ^
    --lr %LR% ^
    --grid_size %GRID_SIZE% ^
    --congestion_pattern %CONGESTION_PATTERN% ^
    --congestion_density %CONGESTION_DENSITY% ^
    --energy_high_density %ENERGY_HIGH_DENSITY% ^
    --energy_high_cost %ENERGY_HIGH_COST% ^
    --include_congestion_obs %INCLUDE_CONGESTION% ^
    --congestion_patch_radius %CONGESTION_RADIUS% ^
    --include_energy_obs %INCLUDE_ENERGY% ^
    --energy_patch_radius %ENERGY_RADIUS% ^
    --energy_budget %ENERGY_BUDGET% ^
    --load_budget 999.0 ^
    --load_cost_scale %LOAD_COST_SCALE% ^
    --use_curriculum %USE_CURRICULUM% ^
    --curriculum_start_load_budget %CURRICULUM_START% ^
    --curriculum_end_load_budget %CURRICULUM_END% ^
    --curriculum_iters %CURRICULUM_ITERS% ^
    --use_lagrange %USE_LAGRANGE% ^
    --lambda_lr_energy %LAMBDA_LR_ENERGY% ^
    --lambda_lr_load %LAMBDA_LR_LOAD% ^
    --initial_lambda_energy 0.0 ^
    --initial_lambda_load 0.0 ^
    --cost_critic_mode %COST_CRITIC_MODE% ^
    --output_dir %OUTPUT_BASE% ^
    --run_tag %RUN_TAG% ^
    --log_interval 10 ^
    --save_model

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Training completed successfully!
echo ========================================
echo.
echo Output directory: %OUTPUT_BASE%_%RUN_TAG%
echo.
echo Next steps:
echo   1. Check metrics.json for training curves
echo   2. Compare with baseline results from quick_lr_ratio_200/
echo   3. Plot success rate vs iteration to see curriculum effect
echo.
echo Visualization:
echo   python -c "import json; import matplotlib.pyplot as plt; import numpy as np; data=json.load(open('%OUTPUT_BASE%_%RUN_TAG%/metrics.json')); iters=[d['iteration'] for d in data]; budgets=[d['budget_load'] for d in data]; success=[d['success_rate'] for d in data]; feasible=[d.get('feasible_success_rate', 0) for d in data]; fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(14, 5)); ax1.plot(iters, budgets, 'b-', linewidth=2); ax1.set_xlabel('Iteration'); ax1.set_ylabel('Load Budget (scaled)', color='b'); ax1.tick_params(axis='y', labelcolor='b'); ax1.grid(True, alpha=0.3); ax1.set_title('Curriculum Schedule'); ax2.plot(iters, success, 'g-', label='Success Rate', linewidth=2); ax2.plot(iters, feasible, 'r--', label='Feasible Success Rate', linewidth=2); ax2.axvline(%CURRICULUM_ITERS%, color='gray', linestyle=':', label='Curriculum End'); ax2.set_xlabel('Iteration'); ax2.set_ylabel('Rate'); ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_title('Performance Metrics'); plt.tight_layout(); plt.savefig('%OUTPUT_BASE%_%RUN_TAG%/curriculum_analysis.png', dpi=150); print('Saved: %OUTPUT_BASE%_%RUN_TAG%/curriculum_analysis.png')"
echo.

endlocal
