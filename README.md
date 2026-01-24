# E1 Offline Lambda Ablation

Run the offline ablation on an existing training run:

- `python tools/e1_offline_lambda_ablation.py --run_dir <run_dir> --ckpt_path <run_dir>/best_fsr.pt --device auto`
- `python tools/e1_offline_lambda_ablation.py --run_dir <run_dir> --ckpt_path <run_dir>/best_tail.pt --device auto`

Outputs under `run_dir/e1_lambda_ablation` by default:

- e1_summary.json
- e1_delta_ratio.png
- e1_kl.png
- e1_eval_metrics.png
- rollouts_e1.npz
