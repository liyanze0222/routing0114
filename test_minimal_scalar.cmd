@echo off
REM Minimal test for debugging tensor shapes
setlocal EnableDelayedExpansion

echo Minimal Scalar PPO test (5 iters, batch=256)...
echo.

python train_grid_scalar.py ^
    --seed 0 ^
    --total_iters 5 ^
    --batch_size 256 ^
    --minibatch_size 128 ^
    --update_epochs 2 ^
    --grid_size 6 ^
    --energy_weight 0.5 ^
    --load_weight 1.5 ^
    --output_dir outputs ^
    --run_tag test_minimal ^
    --log_interval 1

endlocal
