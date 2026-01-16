@echo off
REM Quick test for scalar PPO fix (10 iterations)
setlocal EnableDelayedExpansion

echo Testing Scalar PPO with action mask support...
echo.

python train_grid_scalar.py ^
    --seed 0 ^
    --total_iters 10 ^
    --batch_size 512 ^
    --minibatch_size 256 ^
    --update_epochs 3 ^
    --grid_size 8 ^
    --congestion_pattern block ^
    --include_congestion_obs true ^
    --congestion_patch_radius 2 ^
    --include_energy_obs true ^
    --energy_patch_radius 1 ^
    --energy_weight 0.5 ^
    --load_weight 1.5 ^
    --load_cost_scale 5.0 ^
    --output_dir outputs ^
    --run_tag test_scalar_fix ^
    --log_interval 5

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Test failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Test completed successfully!
echo ========================================
echo.
echo The ActorCritic.get_action() now correctly supports action_mask parameter.
echo Output: outputs_test_scalar_fix/
echo.

endlocal
