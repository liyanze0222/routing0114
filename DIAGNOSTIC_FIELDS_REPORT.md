# PPO 诊断字段完整性报告

## 📋 实施总结

已在 PPO 训练日志中添加完整的诊断指标，用于判断拉格朗日项是否真正影响 policy 更新。

---

## ✅ 已实现的诊断字段（14 个必需字段）

### A. Policy 是否还在动（诊断2）
- `approx_kl`: PPO policy 更新的 KL 散度近似值
  - 计算方式: `mean((exp(log_ratio) - 1) - log_ratio)`
  - 位置: 每个 minibatch 计算后取平均

### B. PPO 更新强度辅助
- `clip_frac`: PPO ratio 被 clip 的比例
  - 计算方式: `mean(|ratio - 1| > clip_coef)`
- `entropy`: Policy 分布的平均熵
  - 来源: PPO 基础 metrics，已存在

### C. Cost 是否真的在推动 Policy（核心诊断）
所有统计量基于 **actor 实际使用的 normalized advantage**（与 policy loss 完全一致）

- `adv_reward_abs_mean`: `mean(|adv_reward|)`
- `adv_penalty_abs_mean`: `mean(|penalty_adv_total|)`
- `adv_penalty_to_reward_ratio`: `penalty_abs / (reward_abs + 1e-8)`
- `adv_reward_mean`: `mean(adv_reward)` 带符号
- `adv_penalty_mean`: `mean(penalty_adv)` 带符号

**计算细节（separate 模式）:**
```python
penalty_adv_total = lambda_energy * adv_energy + lambda_load * adv_load
```

**计算细节（aggregated 模式）:**
```python
penalty_adv_total = lambda_energy * adv_cost_total
```

### D. 分约束贡献（精细诊断）
- `lambdaA_energy_abs_mean`: `mean(|lambda_energy * adv_energy|)`
- `lambdaA_load_abs_mean`: `mean(|lambda_load * adv_load|)`
- `lambdaA_total_abs_mean`: `mean(|penalty_adv_total|)`

**Aggregated 模式特殊处理:**
- `lambdaA_energy_abs_mean` = 0.0
- `lambdaA_load_abs_mean` = 0.0
- `lambdaA_total_abs_mean` = `mean(|lambda_total * adv_cost_total|)`

### E. 额外字段（已保留）
- `rho_energy`: 全局累计成本率 = `cumulative_cost_energy / total_steps`
- `rho_load`: 全局累计成本率 = `cumulative_cost_load / total_steps`

**说明:** rho_* 是 Safety Gym 风格的全局累计指标，与 C/D 的逐 batch 优势统计互补。

---

## 📍 代码实现位置

### 1. PPO Agent (`ppo_multi_agent.py`)

#### 计算位置
**函数:** `MultiCriticPPO.update()`  
**行号:** ~530-576

在完成 advantage 归一化后、进入 minibatch 循环前计算所有诊断量：

```python
# Line ~530
adv_eff = self._normalize_advantage(adv_eff)

# Advantage diagnostics (use normalized advantages aligned with actor loss)
adv_penalty_metrics: Dict[str, float] = {}
lambdaA_metrics: Dict[str, float] = {}
with torch.no_grad():
    adv_reward_norm = torch.as_tensor(
        self._normalize_advantage(adv_reward), ...
    )
    # ... 计算所有诊断量
```

#### 写入 metrics 位置
**函数:** `MultiCriticPPO.update()`  
**行号:** ~1147-1172

```python
# Line ~1147
metrics["lambda_gap_mode"] = self.cfg.lambda_gap_mode

# [新增 4] 写入新指标
metrics.update(rho_metrics)
metrics["approx_kl"] = np.mean(approx_kls) if approx_kls else 0.0
metrics["clip_frac"] = np.mean(clip_fracs) if clip_fracs else 0.0
metrics.update(adv_penalty_metrics)
metrics.update(lambdaA_metrics)

# [Sanity Check] 第一次 update 时验证
if self._iter_count == 1:
    diag_keys = [
        "approx_kl", "clip_frac", "entropy",
        "adv_reward_abs_mean", "adv_penalty_abs_mean", ...
    ]
    missing = [k for k in diag_keys if k not in metrics]
    if missing:
        print(f"[WARNING] Missing diagnostic keys: {missing}")
    else:
        print(f"[OK] All diagnostic keys present (iter={self._iter_count})")
```

### 2. 训练脚本 (`train_grid_structured_lagrangian.py`)

#### 写入 log_entry 位置
**行号:** ~1106-1131

```python
# Line ~1106
# [新增] 添加诊断指标到 log_entry（在 logger.log 之前）

# 1. Safety Gym 风格累计成本率
for key in metrics:
    if key.startswith("rho_"):
        log_entry[key] = metrics[key]

# 2. PPO 更新诊断
if "approx_kl" in metrics:
    log_entry["approx_kl"] = metrics["approx_kl"]
if "clip_frac" in metrics:
    log_entry["clip_frac"] = metrics["clip_frac"]

# 3. Advantage penalty diagnostics（核心诊断）
for key in [
    "adv_reward_abs_mean",
    "adv_penalty_abs_mean",
    "adv_penalty_to_reward_ratio",
    "adv_reward_mean",
    "adv_penalty_mean",
    "lambdaA_energy_abs_mean",
    "lambdaA_load_abs_mean",
    "lambdaA_total_abs_mean",
]:
    if key in metrics:
        log_entry[key] = metrics[key]
```

#### 最终写入 metrics.json
**行号:** ~1277

```python
logger.log(log_entry)  # 包含所有诊断字段
```

**保存位置:** ~1471
```python
metrics_path = os.path.join(output_dir, "metrics.json")
logger.save(metrics_path)
```

---

## 🔍 验证方法

### 方法1: 使用验证脚本
```bash
python verify_diagnostic_fields.py outputs/four_group_ablation_20260121/A_multi_critic_adaptive_seed0/metrics.json
```

### 方法2: 运行训练并观察输出
```bash
./run_group_a_only.bat
```

**期望输出（第一个 iteration）:**
```
[OK] All diagnostic keys present in metrics (iter=1)
```

### 方法3: 手动检查 metrics.json
打开 `outputs/.../metrics.json`，检查第 2 个 iteration 的 entry 是否包含所有 14 个必需字段。

---

## ⚠️ 重要约束（已遵守）

✅ **不改变训练逻辑**: 所有诊断量在 `torch.no_grad()` 块中计算  
✅ **不影响梯度**: 仅读取 advantage 进行统计，不修改用于 loss 的张量  
✅ **口径一致**: 使用与 actor loss 完全相同的 normalized advantage  
✅ **每 iteration 记录**: 所有字段在每次 `agent.update()` 后写入 metrics  
✅ **模式兼容**: separate 和 aggregated 模式均已适配  

---

## 📊 使用示例

### 检查拉格朗日项是否生效
```python
import json
import matplotlib.pyplot as plt

with open('outputs/.../metrics.json', 'r') as f:
    data = json.load(f)

iterations = [d['iteration'] for d in data]
ratio = [d.get('adv_penalty_to_reward_ratio', 0) for d in data]
lambda_e = [d.get('lambda_energy', 0) for d in data]

plt.plot(iterations, ratio, label='Penalty/Reward Ratio')
plt.plot(iterations, lambda_e, label='Lambda Energy', alpha=0.5)
plt.xlabel('Iteration')
plt.legend()
plt.show()
```

### 判断 policy 是否停滞
```python
approx_kl = [d.get('approx_kl', 0) for d in data]
clip_frac = [d.get('clip_frac', 0) for d in data]

# 若 approx_kl 持续 < 1e-4 且 clip_frac < 0.01，说明 policy 几乎不动
```

---

## 🎯 下一步

1. **运行训练**: `./run_group_a_only.bat`
2. **验证字段**: `python verify_diagnostic_fields.py <metrics_path>`
3. **分析结果**: 使用上述示例代码绘制诊断曲线

所有必需字段已完整实现并写入 metrics.json。
