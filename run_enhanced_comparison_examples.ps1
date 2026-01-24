# 增强功能测试命令示例
# ===================================

# 1) Visitmap 模式 + 自动检测 cell2 (50 episodes 快速测试)
python compare_routes_over_time.py `
    --mode visitmap `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --episodes 50 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --topk_cells 3 `
    --deterministic True

# 2) Visitmap 模式 + 自动检测 cell2 (200 episodes 完整分析)
python compare_routes_over_time.py `
    --mode visitmap `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --deterministic True

# 3) Visitmap 模式 + 手动指定 cell2
python compare_routes_over_time.py `
    --mode visitmap `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 5,5 `
    --deterministic True

# 4) Timeline 模式（如果有迭代 checkpoint）
python compare_routes_over_time.py `
    --mode timeline `
    --run_dir outputs/some_run_with_iter_checkpoints `
    --timeline_glob "checkpoint_iter_*.pt" `
    --timeline_stride 10 `
    --timeline_episodes 200 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --deterministic True

# 5) Timeline 退化模式（仅用3个标准checkpoint）
python compare_routes_over_time.py `
    --mode timeline `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --timeline_episodes 100 `
    --cell_of_interest 2,2 `
    --cell_of_interest2 auto `
    --deterministic True

# 6) 原始 Routes 模式（仍然可用）
python compare_routes_over_time.py `
    --mode routes `
    --run_dir outputs/four_group_ablation_20260122/A_multi_critic_adaptive_seed0_v3 `
    --episodes 3 `
    --seed 0 `
    --deterministic True
