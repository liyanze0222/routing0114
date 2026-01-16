# Full Spectrum Ablation Study

## Experiment Design

### 3 Difficulty Levels (Load Budget)
| Level | Load Budget | Energy Budget | Description |
|-------|-------------|---------------|-------------|
| **Stable** | 1.025 | 1.35 | Easy/Standard - Loose constraints |
| **Stress** | 1.000 | 1.35 | Hard/Boundary - Tight constraints |
| **Extreme** | 0.900 | 1.35 | Hero/SOTA - Very tight (proven solvable) |

### 5 Agent Variants

| Variant | Description | Config |
|---------|-------------|--------|
| **V0 (Proposed)** | Multi-Head, Separate Critic, Lagrange | `--cost_critic_mode separate --use_lagrange True` |
| **V1 (Unconstrained)** | Multi-Head, No Lagrange | `--use_lagrange False` |
| **V2 (Shared)** | Multi-Head, Shared Cost Critic | `--cost_critic_mode shared --use_lagrange True` |
| **V3 (Energy-Only)** | Multi-Head, Only Energy Constraint | `--lambda_lr_energy 0.05 --lambda_lr_load 0.0` |
| **V4 (Load-Only)** | Multi-Head, Only Load Constraint | `--lambda_lr_energy 0.0 --lambda_lr_load 0.05` |
| **V5 (Scalar)** | Single-Head Scalar PPO | `train_grid_scalar.py --energy_weight 0.5 --load_weight 1.5` |

**Note**: V3/V4 are optional and only run on **Stress** budget to save time.

## Key Settings
- **Fixed Energy Budget**: 1.35 (all experiments)
- **Load Cost Scale**: 5.0 (matches energy magnitude)
- **Iterations**: 800 per run
- **Total Runs**: 3 budgets × 4 core variants + 2 optional = **14 runs**
- **Estimated Time**: ~14-20 hours (depending on hardware)

## Usage

### 1. Quick Test (10 iterations)
```bash
.\test_full_benchmark.cmd
```

### 2. Full Benchmark (800 iterations)
```bash
.\run_full_benchmark.cmd
```

### 3. Generate Plots
```bash
python plot_full_comparison.py outputs\final_benchmark_YYYYMMDD_HHMMSS
```

## Output Structure
```
outputs/final_benchmark_YYYYMMDD_HHMMSS/
├── budget_1.025_stable/
│   ├── v0_proposed/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   └── training_curves.png
│   ├── v1_unconstrained/
│   ├── v2_shared/
│   └── v5_scalar/
├── budget_1.000_stress/
│   ├── v0_proposed/
│   ├── v1_unconstrained/
│   ├── v2_shared/
│   ├── v3_energy_only/
│   ├── v4_load_only/
│   └── v5_scalar/
├── budget_0.900_extreme/
│   ├── v0_proposed/
│   ├── v1_unconstrained/
│   ├── v2_shared/
│   └── v5_scalar/
└── comparison_plots/
    ├── comparison_budget_1.025_stable.png
    ├── comparison_budget_1.000_stress.png
    ├── comparison_budget_0.900_extreme.png
    └── summary_table.csv
```

## Key Metrics

### Primary Metrics (Goal: Maximize)
- **Feasible Success Rate**: % of episodes that reach goal AND satisfy both constraints
- **Success Rate**: % of episodes that reach goal (ignoring constraints)

### Secondary Metrics (Goal: Below Budget)
- **Avg Energy Cost**: Per-step energy consumption (target: < 1.35)
- **Avg Load Cost**: Per-step load cost (target: < budget [1.025/1.000/0.900])

## Hypothesis

**V0 (Proposed Multi-Head) should outperform all baselines:**
- ✅ Better than **V1 (Unconstrained)**: Explicitly models constraints
- ✅ Better than **V2 (Shared)**: Separate critics capture cost-specific distributions
- ✅ Better than **V5 (Scalar)**: Multi-head value functions > weighted sum
- ✅ Better than **V3/V4 (Single-Constraint)**: Balances both constraints simultaneously

## Expected Results

### Feasible Success Rate (Higher = Better)
| Budget | V0 (Proposed) | V1 (Uncon) | V2 (Shared) | V5 (Scalar) |
|--------|---------------|------------|-------------|-------------|
| Stable | ~90-95% | ~60-70% | ~85-90% | ~75-85% |
| Stress | ~70-80% | ~40-50% | ~65-75% | ~55-65% |
| Extreme | ~50-60% | ~20-30% | ~45-55% | ~35-45% |

### Key Comparisons
1. **V0 vs V5**: Multi-head value decomposition > scalar reward shaping
2. **V0 vs V2**: Separate critics > shared critic (better cost modeling)
3. **V0 vs V1**: Lagrangian constraint handling > unconstrained PPO
4. **V0 vs V3/V4**: Multi-constraint > single-constraint (on Stress budget)

## Analysis Tools

### 1. Visual Comparison
Each plot shows 4 subplots:
- Success Rate (overall task completion)
- Feasible Success Rate (key metric: success + constraints satisfied)
- Energy Cost vs Budget (with budget line)
- Load Cost vs Budget (with budget line)

### 2. Summary Table
CSV file with final performance (last 10 iterations averaged) for all variants.

## Files

### Core Training Scripts
- `train_grid_structured_lagrangian.py` - Multi-head PPO (V0/V1/V2/V3/V4)
- `train_grid_scalar.py` - Scalar PPO (V5)
- `ppo_multi_agent.py` - Multi-head PPO agent
- `ppo_scalar.py` - Scalar PPO agent

### Batch Execution
- `run_full_benchmark.cmd` - Full benchmark (14 runs × 800 iters)
- `test_full_benchmark.cmd` - Quick test (2 runs × 10 iters)

### Analysis
- `plot_full_comparison.py` - Generate comparison plots and summary table

## Notes

1. **Load Cost Scaling**: All experiments use `--load_cost_scale 5.0` to match energy magnitude
2. **Scalar PPO Weights**: V5 uses empirically tuned weights (α=0.5, β=1.5) based on previous Lagrange multipliers
3. **Reproducibility**: All runs use `--seed 0` for reproducibility
4. **Early Stopping**: KL divergence early stopping (target_kl=0.015) is enabled for training stability
