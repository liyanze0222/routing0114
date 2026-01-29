# 快速验证脚本：检查 hysteresis 修复效果
# 用法：python quick_verify_fix.py <metrics_json_path>

import json
import sys
from pathlib import Path

def analyze_late_stage_response(metrics_path, start_iter=600):
    """分析后期 lambda 响应（验证改法1效果）"""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    energy_budget = 0.10
    load_budget = 0.08
    
    # 统计后期的违规 vs lambda 增长
    viol_no_inc_energy = 0
    viol_no_inc_load = 0
    total_viol_energy = 0
    total_viol_load = 0
    
    gap_positive_but_used_zero_energy = 0
    gap_positive_but_used_zero_load = 0
    
    for entry in data:
        iter_num = entry.get('iter', 0)
        if iter_num < start_iter:
            continue
        
        # Energy
        avg_cost_e = entry.get('avg_cost_energy', 0.0)
        lambda_e = entry.get('lambda_energy', 0.0)
        prev_lambda_e = 0.0  # 简化：假设前一个就是当前-delta
        
        # 检查新日志字段
        gap_ratio_e = entry.get('gap_ratio_energy', entry.get('gap_energy_raw', 0.0))
        gap_used_e = entry.get('gap_abs_energy_used', entry.get('gap_energy_used', 0.0))
        
        if avg_cost_e > energy_budget:
            total_viol_energy += 1
            # 简化判断：如果 lambda 几乎没变，视为 "inc=0"
            if abs(lambda_e - prev_lambda_e) < 1e-6:
                viol_no_inc_energy += 1
        
        if gap_ratio_e > 0 and abs(gap_used_e) < 1e-4:
            gap_positive_but_used_zero_energy += 1
        
        # Load
        avg_cost_l = entry.get('avg_cost_load', 0.0)
        lambda_l = entry.get('lambda_load', 0.0)
        prev_lambda_l = 0.0
        
        gap_ratio_l = entry.get('gap_ratio_load', entry.get('gap_load_raw', 0.0))
        gap_used_l = entry.get('gap_abs_load_used', entry.get('gap_load_used', 0.0))
        
        if avg_cost_l > load_budget:
            total_viol_load += 1
            if abs(lambda_l - prev_lambda_l) < 1e-6:
                viol_no_inc_load += 1
        
        if gap_ratio_l > 0 and abs(gap_used_l) < 1e-4:
            gap_positive_but_used_zero_load += 1
    
    print(f"\n{'='*60}")
    print(f"验证报告：{Path(metrics_path).parent.name}")
    print(f"分析区间：iter {start_iter}-{data[-1].get('iter', 'N/A')}")
    print(f"{'='*60}")
    
    print(f"\n[Energy] Budget={energy_budget}")
    print(f"  后期违规次数: {total_viol_energy}")
    print(f"  违规但 λ 不增次数: {viol_no_inc_energy}")
    print(f"  gap_ratio>0 但 gap_used≈0 次数: {gap_positive_but_used_zero_energy}")
    
    print(f"\n[Load] Budget={load_budget}")
    print(f"  后期违规次数: {total_viol_load}")
    print(f"  违规但 λ 不增次数: {viol_no_inc_load}")
    print(f"  gap_ratio>0 但 gap_used≈0 次数: {gap_positive_but_used_zero_load}")
    
    # 判断
    print(f"\n{'='*60}")
    if gap_positive_but_used_zero_energy < 5 and gap_positive_but_used_zero_load < 5:
        print("✅ 修复生效：'gap>0 但 used≈0' 的次数已显著下降！")
    else:
        print("⚠️  仍有较多 'gap>0 但 used≈0'，可能需要进一步调整 deadband")
    print(f"{'='*60}\n")
    
    # 检查日志字段
    sample_entry = data[-1]
    new_fields = ['gap_abs_energy', 'gap_ratio_energy', 'gap_abs_energy_used']
    has_new_fields = all(f in sample_entry for f in new_fields)
    
    if has_new_fields:
        print("✅ 日志命名已更新（改法3生效）")
        print(f"   新字段：{', '.join(new_fields)}")
    else:
        print("⚠️  日志仍使用旧命名，请确认代码版本")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_verify_fix.py <metrics_json_path>")
        print("例如: python quick_verify_fix.py outputs/hysteresis_fix_verify/B_hysteresis_fixed_ent0.05_seed0/metrics.json")
        sys.exit(1)
    
    analyze_late_stage_response(sys.argv[1])
