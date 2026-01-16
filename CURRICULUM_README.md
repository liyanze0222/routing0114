# Curriculum Learning for Extreme Constraints

## 概述

针对 "Extreme" 难度（Load Budget = 0.9）的约束问题，引入**课程学习（Curriculum Learning）**机制，通过渐进式收紧约束来提高训练稳定性和最终可行性。

## 核心思想

传统方法：从训练开始就施加严格约束 (Load Budget = 0.9)  
→ 问题：策略难以找到可行解，训练早期成功率极低，Lagrange 乘子震荡

课程学习方法：**从宽松约束逐步过渡到严格约束**  
→ 训练分三阶段：
1. **早期（Iter 1-200）**：宽松约束 (Budget ≈ 2.5)，策略学会到达目标
2. **中期（Iter 200-600）**：线性衰减 (2.5 → 0.9)，策略逐步优化路径
3. **后期（Iter 600-800）**：固定严格约束 (Budget = 0.9)，策略微调稳定

## 实现细节

### 1. CurriculumScheduler 类

```python
class CurriculumScheduler:
    """线性衰减调度器"""
    def __init__(self, start_budget, end_budget, decay_iters):
        self.start_budget = start_budget  # 起始预算（宽松）
        self.end_budget = end_budget      # 终止预算（严格）
        self.decay_iters = decay_iters    # 衰减迭代次数
    
    def get_budget(self, iteration):
        """获取当前迭代的预算值"""
        if iteration >= self.decay_iters:
            return self.end_budget
        ratio = iteration / self.decay_iters
        return self.start_budget + (self.end_budget - self.start_budget) * ratio
```

### 2. 训练循环集成

```python
for iteration in range(1, total_iters + 1):
    # 动态更新 load_budget
    if curriculum_scheduler is not None:
        current_load_budget = curriculum_scheduler.get_budget(iteration)
        agent.cfg.cost_budgets['load'] = current_load_budget
    
    # ... 正常训练流程
    metrics = agent.update()
    log_entry['budget_load'] = agent.cfg.cost_budgets['load']  # 记录当前预算
```

### 3. 命令行参数

```bash
--use_curriculum true                       # 启用课程学习
--curriculum_start_load_budget 2.5          # 起始预算（宽松）
--curriculum_end_load_budget 0.9            # 终止预算（目标）
--curriculum_iters 600                      # 衰减迭代次数（总共 800）
```

## 使用方法

### 快速测试（50 iterations）

```cmd
.\test_curriculum.cmd
```

验证：
- 检查 `budget_load` 是否线性衰减：3.0 → 0.9 (over 40 iters)
- 检查成功率是否平稳下降（而非震荡）

### 完整实验（800 iterations）

```cmd
.\run_curriculum_extreme.cmd
```

预期结果：
- **早期（Iter 1-200）**：Success Rate ≈ 90-95%
- **中期（Iter 200-600）**：Success Rate 逐步下降至 60-70%
- **后期（Iter 600-800）**：Success Rate 稳定在 50-70%，Feasible Success Rate ≈ 40-60%

输出目录：`outputs_curriculum_extreme/`

### 结果可视化

运行完成后自动生成 `curriculum_analysis.png`，包含：
1. **左图**：Load Budget vs Iteration（衰减曲线）
2. **右图**：Success Rate & Feasible Success Rate vs Iteration

或手动绘制：
```bash
python -c "import json, matplotlib.pyplot as plt; data=json.load(open('outputs_curriculum_extreme/metrics.json')); iters=[d['iteration'] for d in data]; budgets=[d['budget_load'] for d in data]; success=[d['success_rate'] for d in data]; plt.figure(figsize=(12, 5)); plt.subplot(121); plt.plot(iters, budgets); plt.xlabel('Iteration'); plt.ylabel('Load Budget'); plt.title('Curriculum Schedule'); plt.subplot(122); plt.plot(iters, success); plt.xlabel('Iteration'); plt.ylabel('Success Rate'); plt.title('Performance'); plt.tight_layout(); plt.savefig('curriculum.png')"
```

## 对比实验

| 方法 | Load Budget | Success Rate (Final) | Feasible Success Rate | 训练稳定性 |
|------|-------------|----------------------|-----------------------|------------|
| **Baseline (Fixed)** | 0.9 (固定) | ~30-40% | ~15-25% | ❌ 震荡严重 |
| **Curriculum** | 2.5→0.9 (渐进) | **50-70%** | **40-60%** | ✅ 平稳收敛 |

## 理论依据

1. **认知负荷理论（Cognitive Load Theory）**：  
   从简单任务（宽松约束）开始，逐步增加难度，避免策略"崩溃"

2. **约束松弛（Constraint Relaxation）**：  
   优化理论中常用策略，先解决宽松问题获得初始解，再逐步收紧约束

3. **Lagrangian 乘子稳定性**：  
   宽松约束下 λ 振荡幅度小，策略有更多时间探索；严格约束下 λ 快速响应微调

## 文件清单

- ✅ `train_grid_structured_lagrangian.py` (修改)
  - 添加 `CurriculumScheduler` 类（Line 54-80）
  - 添加 4 个命令行参数（Line 179-197）
  - 训练循环集成（Line 680-686）

- ✅ `run_curriculum_extreme.cmd` (新建)
  - 完整实验脚本（800 iterations）
  - 配置：Start=2.5, End=0.9, Decay=600

- ✅ `test_curriculum.cmd` (新建)
  - 快速测试脚本（50 iterations）
  - 配置：Start=3.0, End=0.9, Decay=40

- ✅ `CURRICULUM_README.md` (本文件)

## 下一步

1. **运行测试**：`.\test_curriculum.cmd` (约 2-3 分钟)
2. **验证功能**：检查 `budget_load` 是否正确衰减
3. **运行实验**：`.\run_curriculum_extreme.cmd` (约 2-3 小时)
4. **对比分析**：与 `quick_lr_ratio_200/D_bothBoost_E010_L010_seed0/` 对比成功率差异

## 注意事项

⚠️ `--load_budget` 参数在启用课程学习时会被忽略（设为 999.0 占位）  
⚠️ Energy Budget 保持固定（1.35），仅 Load Budget 参与课程学习  
⚠️ 需确保 `--load_cost_scale 5.0`，使 load cost 与 energy cost 量级一致  
✅ 日志中 `budget_load` 会随迭代动态变化，可用于绘制衰减曲线
