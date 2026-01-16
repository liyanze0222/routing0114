"""
grid_energy_obs_wrapper.py

在原始观测后拼接一个以 agent 为中心的 energy patch，帮助策略理解局部能耗分布。
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from grid_congestion_obs_wrapper import find_wrapper_with_attr
from grid_env import GridRoutingEnv


class GridEnergyObsWrapper(gym.ObservationWrapper):
    """将能耗 patch 拼接到观测末尾，可选归一化。"""

    def __init__(
        self,
        env: gym.Env,
        patch_radius: int = 1,
        normalize: bool = True,
    ):
        super().__init__(env)
        self.patch_radius = int(patch_radius)
        self.normalize = bool(normalize)
        self.patch_size = (2 * self.patch_radius + 1) ** 2
        self._norm_eps = 1e-8

        # 查找包含 energy map 的 wrapper（通常是 GridCostWrapper）
        self._energy_wrapper = find_wrapper_with_attr(env, "_energy_map")
        if self._energy_wrapper is None:
            raise RuntimeError(
                "GridEnergyObsWrapper: 无法找到包含 _energy_map 的 wrapper；"
                "请确认已包裹 GridCostWrapper。"
            )

        # 查找底层 GridRoutingEnv（获取 agent 坐标和 grid_size）
        base_env = env
        visited = set()
        self._base_env: Optional[GridRoutingEnv] = None
        while True:
            if isinstance(base_env, GridRoutingEnv):
                self._base_env = base_env
                break
            if id(base_env) in visited:
                break
            visited.add(id(base_env))
            if hasattr(base_env, "env"):
                base_env = base_env.env
            else:
                break
        if self._base_env is None:
            raise RuntimeError(
                "GridEnergyObsWrapper: 无法定位 GridRoutingEnv；"
                "请检查 wrapper 链。"
            )

        self._energy_base = getattr(self._energy_wrapper, "energy_base", None)
        self._energy_high = getattr(self._energy_wrapper, "energy_high_cost", None)

        # 更新 observation space
        orig_space = env.observation_space
        patch_low, patch_high = self._get_patch_bounds()
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([orig_space.low, patch_low]),
            high=np.concatenate([orig_space.high, patch_high]),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _get_patch_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.normalize:
            low = np.zeros(self.patch_size, dtype=np.float32)
            high = np.ones(self.patch_size, dtype=np.float32)
            return low, high
        base = self._energy_base if self._energy_base is not None else 0.0
        high_val = self._energy_high if self._energy_high is not None else base + 1.0
        if high_val <= base:
            high_val = base + 1.0
        low = np.full(self.patch_size, base, dtype=np.float32)
        high = np.full(self.patch_size, high_val, dtype=np.float32)
        return low, high

    def _resolve_bounds(self, energy_map: Optional[np.ndarray]) -> tuple[float, float]:
        base = self._energy_base
        high = self._energy_high
        if base is None and energy_map is not None:
            base = float(np.min(energy_map))
        if high is None and energy_map is not None:
            high = float(np.max(energy_map))
        if base is None:
            base = 0.0
        if high is None or high <= base:
            high = base + 1.0
        return base, high

    def _fill_value(self, energy_map: Optional[np.ndarray]) -> float:
        if self._energy_high is not None:
            return float(self._energy_high)
        if energy_map is not None:
            return float(np.max(energy_map))
        return 1.0

    def _get_energy_patch(self) -> np.ndarray:
        energy_map = getattr(self._energy_wrapper, "_energy_map", None)
        r, c = self._base_env.agent_row, self._base_env.agent_col
        grid_size = self._base_env.grid_size
        radius = self.patch_radius
        fill_value = self._fill_value(energy_map)

        values = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if (
                    energy_map is not None
                    and 0 <= nr < grid_size
                    and 0 <= nc < grid_size
                ):
                    values.append(float(energy_map[nr, nc]))
                else:
                    values.append(fill_value)

        patch = np.array(values, dtype=np.float32)

        if self.normalize:
            base, high = self._resolve_bounds(energy_map)
            scale = max(high - base, self._norm_eps)
            patch = np.clip((patch - base) / scale, 0.0, 1.0)

        return patch

    # ------------------------------------------------------------------ #
    # ObservationWrapper 接口
    # ------------------------------------------------------------------ #

    def observation(self, obs: Any) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        patch = self._get_energy_patch()
        return np.concatenate([obs, patch])
