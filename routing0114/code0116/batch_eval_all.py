import os
import pandas as pd
import glob
import json
import sys
import torch
import numpy as np

# 确保能导入当前目录下的模块
sys.path.append(os.getcwd())

# 尝试导入评估函数
try:
    from eval_fixed_set import evaluate_fixed_set
except ImportError:
    print("Error: eval_fixed_set.py not found or evaluate_fixed_set not importable.")
    print("Please ensure eval_fixed_set.py is in the current directory.")
    sys.exit(1)

# 配置
ROOT_DIR = "final_safety_gym_benchmark"
OUTPUT_CSV = "final_benchmark_summary.csv"
NUM_EPISODES = 100

def main():
    # 查找所有 best_feasible.pt
    # 结构: final_safety_gym_benchmark/{budget_level}/{variant_tag}/best_feasible.pt (两层嵌套)
    search_path = os.path.join(ROOT_DIR, "*", "*", "best_feasible.pt")
    model_paths = glob.glob(search_path)
    
    if not model_paths:
        print(f"No 'best_feasible.pt' models found in {search_path}")
        print(f"Check if {ROOT_DIR} exists and contains training runs.")
        # 尝试查找 model_final.pt 作为备选
        print("Searching for model_final.pt instead...")
        search_path = os.path.join(ROOT_DIR, "*", "*", "model_final.pt")
        model_paths = glob.glob(search_path)
        if not model_paths:
            print("No models found. Exiting.")
            return

    summary_rows = []
    print(f"Found {len(model_paths)} models. Starting batch evaluation...")

    for model_path in model_paths:
        # 获取两层目录：budget_level/variant_tag
        variant_dir = os.path.dirname(model_path)
        budget_dir = os.path.dirname(variant_dir)
        budget_level = os.path.basename(budget_dir)
        variant_tag = os.path.basename(variant_dir)
        run_name = f"{budget_level}/{variant_tag}"
        
        print(f"\n{'='*50}")
        print(f">>> Evaluating Run: {run_name}")
        print(f"    Model: {os.path.basename(model_path)}")
        
        # 1. 尝试从 config.json 读取预算
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        e_budget = 1.35 # 默认值
        l_budget = 0.20 # 默认值
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    e_budget = config.get('energy_budget', e_budget)
                    l_budget = config.get('load_budget', l_budget)
                print(f"    Loaded Budgets: E={e_budget}, L={l_budget}")
            except Exception as e:
                print(f"    Warning: Could not read config.json: {e}. Using defaults.")
        else:
             print(f"    Warning: config.json not found. Using defaults E={e_budget}, L={l_budget}")
        
        try:
            # 2. 运行评估 (100 episodes)
            # device="cpu" 确保兼容性，如果想用 gpu 改为 "cuda"
            df = evaluate_fixed_set(model_path, num_episodes=NUM_EPISODES, seed_start=0, device="cpu")
            
            # 3. 计算指标
            # 基础指标
            success_rate = df['success'].mean()
            avg_len = df['ep_len'].mean()
            
            # 成本计算 (Per-Step Mean)
            # 注意: PPO 训练中的 avg_cost 通常是 per-step 的。
            # 这里我们计算每个 episode 的 step mean，然后取平均。
            # 防止除以 0
            ep_lengths = df['ep_len'].clip(lower=1)
            ep_energy_means = df['agent_energy_sum'] / ep_lengths
            ep_load_means = df['agent_load_sum'] / ep_lengths
            
            avg_energy_step = ep_energy_means.mean()
            avg_load_step = ep_load_means.mean()
            
            # 严格可行性 (Feasible Rate)
            # 定义：成功 且 均值成本 <= 预算
            tol = 1e-6
            is_feasible = (
                (df['success'] == True) &
                (ep_energy_means <= e_budget + tol) &
                (ep_load_means <= l_budget + tol)
            )
            feasible_rate = is_feasible.mean()
            
            # 违反率 (Violation Rate)
            violates_energy = (ep_energy_means > e_budget + tol).mean()
            violates_load = (ep_load_means > l_budget + tol).mean()

            # 4. 记录结果
            summary_rows.append({
                "Run Tag": run_name,
                "Model Type": "Best" if "best" in model_path else "Final",
                "Success Rate": f"{success_rate:.2%}",
                "Feasible Rate": f"{feasible_rate:.2%}",
                "Avg Energy": f"{avg_energy_step:.4f}",
                "Avg Load": f"{avg_load_step:.4f}",
                "Target E": e_budget,
                "Target L": l_budget,
                "Violate E %": f"{violates_energy:.2%}",
                "Violate L %": f"{violates_load:.2%}",
                "Avg Length": f"{avg_len:.2f}"
            })
            
            print(f"    -> Success: {success_rate:.2%}, Feasible: {feasible_rate:.2%}")
            
        except Exception as e:
            print(f"!!! Error evaluating {run_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存和打印汇总
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # 按 Run Tag 排序
        summary_df = summary_df.sort_values("Run Tag")
        
        print("\n" + "="*100)
        print("FINAL BATCH EVALUATION SUMMARY (Fixed Set N=100)")
        print("="*100)
        # 尝试使用 markdown 打印，如果没有 tabulate 库则直接打印
        try:
            print(summary_df.to_markdown(index=False))
        except ImportError:
            print(summary_df.to_string(index=False))
        
        summary_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSummary saved to {OUTPUT_CSV}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main()