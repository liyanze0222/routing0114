"""
验证诊断字段是否正确记录到 metrics.json

运行方式：
python verify_diagnostic_fields.py outputs/四_group_ablation_20260121/A_multi_critic_adaptive_seed0/metrics.json
"""

import json
import sys

REQUIRED_FIELDS = [
    # A. policy 是否还在动
    "approx_kl",
    # B. PPO 更新强度辅助
    "clip_frac",
    "entropy",
    # C. cost 是否真的在推动 policy（核心诊断）
    "adv_reward_abs_mean",
    "adv_penalty_abs_mean",
    "adv_penalty_to_reward_ratio",
    "adv_reward_mean",
    "adv_penalty_mean",
    # D. 分约束贡献
    "lambdaA_energy_abs_mean",
    "lambdaA_load_abs_mean",
    "lambdaA_total_abs_mean",
]

def verify_metrics(metrics_path: str):
    """验证 metrics.json 中是否包含所有必需的诊断字段"""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取 {metrics_path}: {e}")
        return False
    
    if not data or len(data) == 0:
        print(f"❌ metrics.json 为空")
        return False
    
    # 检查第一个 iteration（可能缺少诊断字段）
    first_entry = data[0]
    first_iter = first_entry.get('iteration', 0)
    
    # 检查第二个 iteration（应该稳定包含所有字段）
    if len(data) < 2:
        print(f"⚠️  只有 {len(data)} 个 iteration，建议至少运行 2 个 iteration")
        check_entry = first_entry
        check_iter = first_iter
    else:
        check_entry = data[1]
        check_iter = check_entry.get('iteration', 1)
    
    print(f"\n检查 iteration {check_iter} 的字段...")
    print("=" * 60)
    
    missing = []
    present = []
    
    for field in REQUIRED_FIELDS:
        if field in check_entry:
            value = check_entry[field]
            present.append(field)
            print(f"✓ {field:30s} = {value}")
        else:
            missing.append(field)
            print(f"✗ {field:30s} = MISSING")
    
    print("=" * 60)
    print(f"\n结果：")
    print(f"  - 必需字段总数: {len(REQUIRED_FIELDS)}")
    print(f"  - 已记录字段: {len(present)}")
    print(f"  - 缺失字段: {len(missing)}")
    
    if missing:
        print(f"\n❌ 缺失字段列表:")
        for field in missing:
            print(f"   - {field}")
        return False
    else:
        print(f"\n✓ 所有必需字段都已正确记录！")
        
        # 额外检查：rho_* 字段（若存在）
        rho_fields = [k for k in check_entry.keys() if k.startswith('rho_')]
        if rho_fields:
            print(f"\n额外字段（Safety Gym 风格）:")
            for field in rho_fields:
                print(f"  ✓ {field} = {check_entry[field]}")
        
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python verify_diagnostic_fields.py <metrics.json路径>")
        print("\n示例:")
        print("  python verify_diagnostic_fields.py outputs/four_group_ablation_20260121/A_multi_critic_adaptive_seed0/metrics.json")
        sys.exit(1)
    
    metrics_path = sys.argv[1]
    success = verify_metrics(metrics_path)
    sys.exit(0 if success else 1)
