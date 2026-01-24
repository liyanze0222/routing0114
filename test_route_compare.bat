@echo off
REM =======================================================
REM 路径对比可视化 - 快速测试脚本
REM =======================================================

echo.
echo ============================================================
echo Route Comparison Visualization Test
echo ============================================================
echo.

REM 设置参数
set "RUN_DIR=outputs\four_group_ablation_20260121\A_multi_critic_adaptive_seed0_v2"
set "SEED=0"
set "EPISODES=3"
set "DETERMINISTIC=True"

REM 检查目录是否存在
if not exist "%RUN_DIR%" (
    echo [ERROR] Run directory not found: %RUN_DIR%
    echo.
    echo Please update RUN_DIR in this script to point to a valid training output directory.
    echo Example: outputs\four_group_ablation_20260121\A_multi_critic_adaptive_seed0
    pause
    exit /b 1
)

echo Run Directory: %RUN_DIR%
echo Seed: %SEED%
echo Episodes: %EPISODES%
echo Deterministic: %DETERMINISTIC%
echo.
echo Starting comparison...
echo.

python compare_routes_over_time.py ^
    --run_dir %RUN_DIR% ^
    --seed %SEED% ^
    --episodes %EPISODES% ^
    --deterministic %DETERMINISTIC%

if errorlevel 1 (
    echo.
    echo [ERROR] Route comparison failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Check output files in: %RUN_DIR%
echo ============================================================
pause
