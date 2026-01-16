@echo off
REM Quick test for Scalar PPO implementation

echo Testing Scalar PPO (V5 Baseline)...

python train_grid_scalar.py ^
  --total_iters 5 ^
  --batch_size 512 ^
  --log_interval 1 ^
  --energy_weight 0.5 ^
  --load_weight 2.0 ^
  --load_cost_scale 5.0 ^
  --grid_size 8 ^
  --max_steps 256 ^
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
  --step_penalty -1.0 ^
  --seed 0 ^
  --output_dir test_scalar_ppo

if errorlevel 1 (
  echo [FAIL] Scalar PPO test failed!
) else (
  echo [SUCCESS] Scalar PPO test passed!
)

pause
