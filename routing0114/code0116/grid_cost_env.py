"""
grid_cost_env.py

为 GridRoutingEnv 添加软约束用的 cost 信号。

========== 说明 ==========

1) energy（连续非负）：
   - 每次"有效移动"产生 energy_base 能耗；
   - 无效移动（撞墙）不消耗能量。

2) load（连续值，来自拥塞图查表）：
   - 每步返回预设拥塞图的值（cell_congestion 模式）

3) invalid（0/1 事件）：
   - 若动作未改变 agent 位置，则 invalid = 1，否则 = 0；
   - 仅作为监控量，不参与多 Critic 约束。

========== 接口 ==========

- step() 返回的 info['cost_components'] 包含 {'energy': float, 'load': float, 'invalid': float}
- get_action_mask() 返回当前状态的合法动作 mask

========== 可复现性 ==========

拥塞图的采样使用底层 GridRoutingEnv 的 _rng，
确保在相同 seed 下训练曲线可复现。
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import gymnasium as gym
import numpy as np

from grid_env import GridRoutingEnv

# 支持的 congestion_pattern
CongestionPattern = Literal["random", "block"]


class GridCostWrapper(gym.Wrapper):
    """在 GridRoutingEnv 上附加 invalid / energy / load 三种 cost，并暴露 action mask。"""

    def __init__(
        self,
        env: GridRoutingEnv,
        energy_base: float = 1.0,
        energy_high_cost: float = 3.0,
        energy_high_density: float = 0.2,
        congestion_density: float = 0.3,
        congestion_pattern: CongestionPattern = "random",
        load_cost_scale: float = 1.0,
    ):
        """
        Args:
            env:                基础 GridRoutingEnv 环境实例。
            energy_base:        每次"有效移动"的基础能耗（平原区域）。
            energy_high_cost:   高功率干扰区的能耗。
            energy_high_density:
                                高能耗区域在地图中的比例（0~1）。
            congestion_density: 拥塞节点的比例（0~1）
            congestion_pattern: 拥塞图生成模式：
                                - "random": 随机分布的拥塞点（默认）
                                - "block": 在地图上生成一个连续的拥塞块
            load_cost_scale:    load cost 的缩放系数（默认 1.0，用于匹配 energy 尺度）
        """
        super().__init__(env)

        self.energy_base = energy_base
        self.energy_high_cost = energy_high_cost
        self.energy_high_density = energy_high_density
        self.congestion_density = congestion_density
        self.congestion_pattern = congestion_pattern
        self.load_cost_scale = load_cost_scale

        # 假设 GridRoutingEnv 暴露 grid_size 属性
        self.grid_size = env.grid_size
        # 拥塞图（每个 episode 重新采样）
        self._congestion_map: np.ndarray | None = None
        # 能耗图（每个 episode 重新采样）
        self._energy_map: np.ndarray | None = None

        # 向下找到真正的 GridRoutingEnv（用于 action mask）
        self._base_env = self._find_base_env(env)

    def _find_base_env(self, env):
        """向下遍历 wrapper 链，找到真正的 GridRoutingEnv（拥有 _get_action_mask 方法）。"""
        base = env
        visited = set()
        while True:
            # 找到一个有 _get_action_mask 方法的对象
            if hasattr(base, "_get_action_mask"):
                return base
            # 防止死循环
            if id(base) in visited:
                break
            visited.add(id(base))
            # gym.Wrapper 一般有 env 属性，继续往里剥
            if hasattr(base, "env"):
                base = base.env
            else:
                break
        # 找不到则返回原始 env
        return env

    # ------------------------------------------------------------------
    # 对外暴露硬约束的 action mask（Layer-1 Shielding）
    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        """
        返回当前状态下允许的动作 mask，1=允许，0=禁止。

        实现方式：
        - 调用 _base_env._get_action_mask()（已在 __init__ 中找到真正的 GridRoutingEnv）；
        - 若 _base_env 没有该方法则退化为全 1（所有动作都允许）。
        """
        if hasattr(self._base_env, "_get_action_mask"):
            mask = self._base_env._get_action_mask()
            return np.array(mask, dtype=np.float32)
        # 兜底：全部动作可用
        n = self.action_space.n
        return np.ones(n, dtype=np.float32)

    # ------------------------------------------------------------------

    def _get_rng(self) -> np.random.RandomState:
        """
        获取用于采样 congestion map 的 RNG。

        严格使用底层 GridRoutingEnv 的 _rng，确保可复现性。

        Returns:
            底层 env 的 _rng

        Raises:
            RuntimeError: 如果无法获取可复现的 RNG
        """
        base_rng = getattr(self._base_env, "_rng", None)
        if base_rng is not None and isinstance(base_rng, np.random.RandomState):
            return base_rng

        raise RuntimeError(
            "GridCostWrapper._get_rng(): 无法获取可复现的 RNG。"
            "底层 GridRoutingEnv 必须有 _rng 属性（RandomState 类型）。"
        )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """重置环境，并初始化 cost_components 信息。"""
        obs, info = self.env.reset(seed=seed, options=options)

        # 生成新的能耗图与拥塞图
        self._energy_map = self._generate_energy_map()
        # 生成新的拥塞图
        self._congestion_map = self._generate_congestion_map()

        info = {} if info is None else dict(info)
        info["cost_components"] = {"invalid": 0.0, "energy": 0.0, "load": 0.0}

        # 添加 cost 配置元信息，方便排查
        info["cost_components_meta"] = {
            "congestion_density": self.congestion_density,
            "congestion_pattern": self.congestion_pattern,
            "energy_base": self.energy_base,
            "energy_high_cost": self.energy_high_cost,
            "energy_high_density": self.energy_high_density,
            "load_cost_scale": float(self.load_cost_scale),
        }

        if self._energy_map is not None:
            energy_stats = {
                "energy_mean": float(self._energy_map.mean()),
                "energy_high_ratio": float(
                    (self._energy_map >= self.energy_high_cost).mean()
                ),
            }
            info.update(energy_stats)

        # 添加拥塞图统计信息（用于调参）
        if self._congestion_map is not None:
            info["congestion_mean"] = float(self._congestion_map.mean())
            info["congestion_mean_scaled"] = float(self._congestion_map.mean() * self.load_cost_scale)
            info["congestion_ratio"] = float((self._congestion_map > 0).mean())

        return obs, info

    def _generate_energy_map(self) -> np.ndarray:
        """生成能耗地图：绝大部分为基础值，少量高能耗区域。"""
        rng = self._get_rng()
        energy_map = np.full(
            (self.grid_size, self.grid_size), self.energy_base, dtype=np.float32
        )
        density = float(np.clip(self.energy_high_density, 0.0, 1.0))
        if density > 0.0:
            mask = rng.rand(self.grid_size, self.grid_size) < density
            energy_map[mask] = self.energy_high_cost
        return energy_map

    def _generate_congestion_map(self) -> np.ndarray:
        """
        根据 congestion_pattern 生成拥塞图。

        Returns:
            congestion_map: shape=(grid_size, grid_size) 的拥塞图
        """
        rng = self._get_rng()
        congestion_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 通过放缩控制平均拥塞程度，clip 保证范围 [0, 1]
        scale = max(0.0, min(2.0, 2.0 * float(self.congestion_density)))

        if self.congestion_pattern == "random":
            raw = rng.rand(self.grid_size, self.grid_size)
            congestion_map = np.clip(raw * scale, 0.0, 1.0).astype(np.float32)

        elif self.congestion_pattern == "block":
            total_cells = self.grid_size * self.grid_size
            target_cells = int(total_cells * self.congestion_density)

            if target_cells > 0:
                block_side = int(np.ceil(np.sqrt(target_cells)))
                block_side = min(block_side, self.grid_size)

                max_start_row = max(0, self.grid_size - block_side)
                max_start_col = max(0, self.grid_size - block_side)

                start_row = rng.randint(0, max_start_row + 1)
                start_col = rng.randint(0, max_start_col + 1)

                end_row = min(start_row + block_side, self.grid_size)
                end_col = min(start_col + block_side, self.grid_size)

                raw_block = rng.rand(end_row - start_row, end_col - start_col)
                congestion_map[start_row:end_row, start_col:end_col] = np.clip(
                    raw_block * scale, 0.0, 1.0
                )

        else:
            raw = rng.rand(self.grid_size, self.grid_size)
            congestion_map = np.clip(raw * scale, 0.0, 1.0).astype(np.float32)

        return congestion_map

    def step(self, action: int):
        """
        执行一步动作，并在 info["cost_components"] 中写入：
            - invalid: 本步是否为"无效动作"
            - energy:  本步能耗 proxy
            - load:    本步负载 cost（来自拥塞图）
        """
        # step 前的位置
        prev_r = self.env.agent_row
        prev_c = self.env.agent_col

        # 与底层环境交互
        obs, reward, terminated, truncated, info = self.env.step(action)

        # step 后的位置
        new_r = self.env.agent_row
        new_c = self.env.agent_col

        # -------------------- 1) cost_invalid --------------------
        # 若位置未改变，则视为无效动作（撞墙或环境拒绝）
        invalid = float((new_r == prev_r) and (new_c == prev_c))

        # -------------------- 2) cost_energy --------------------
        # 有效移动才产生能量消耗
        if invalid:
            energy = 0.0
        else:
            if self._energy_map is not None:
                energy = float(self._energy_map[new_r, new_c])
            else:
                energy = self.energy_base

        # -------------------- 3) cost_load --------------------
        # 返回当前节点的拥塞值，并应用缩放系数
        if self._congestion_map is not None:
            load_cost = float(self._congestion_map[new_r, new_c]) * self.load_cost_scale
        else:
            load_cost = 0.0

        # 构造输出
        info = {} if info is None else dict(info)
        info["cost_components"] = {
            "invalid": invalid,
            "energy": energy,
            "load": load_cost,
        }

        return obs, reward, terminated, truncated, info
