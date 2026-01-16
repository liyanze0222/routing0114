@echo off
REM Quick test to verify shared cost critic mode works

echo Testing shared cost critic mode...

python train_grid_structured_lagrangian.py ^
  --total_iters 2 ^
  --batch_size 512 ^
  --log_interval 1 ^
  --cost_critic_mode shared ^
  --use_lagrange True ^
  --lambda_lr_energy 0.05 ^
  --lambda_lr_load 0.05 ^
  --energy_budget 1.35 ^
  --load_budget 1.0 ^
  --load_cost_scale 5.0 ^
  --seed 0 ^
  --output_dir test_shared_mode

if errorlevel 1 (
  echo [FAIL] Shared mode test failed!
  pause
) else (
  echo [SUCCESS] Shared mode test passed!
  echo Testing separate mode for comparison...
  
  python train_grid_structured_lagrangian.py ^
    --total_iters 2 ^
    --batch_size 512 ^
    --log_interval 1 ^
    --cost_critic_mode separate ^
    --use_lagrange True ^
    --lambda_lr_energy 0.05 ^
    --lambda_lr_load 0.05 ^
    --energy_budget 1.35 ^
    --load_budget 1.0 ^
    --load_cost_scale 5.0 ^
    --seed 0 ^
    --output_dir test_separate_mode
  
  if errorlevel 1 (
    echo [FAIL] Separate mode test failed!
  ) else (
    echo [SUCCESS] Both modes work correctly!
  )
)

pause
