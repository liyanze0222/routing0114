"""
grid_energy_obs_wrapper.py

在原始观测后拼接一个以 agent 为中心的 energy patch，帮助策略理解局部能耗分布。
energy_map 已经是二值 0/1，直接输出无需归一化。
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from grid_congestion_obs_wrapper import find_wrapper_with_attr
from grid_env import GridRoutingEnv


class GridEnergyObsWrapper(gym.ObservationWrapper):
    """将能耗 patch 拼接到观测末尾，energy_map 已经是 0/1 无需归一化。"""

    def __init__(
        self,
        env: gym.Env,
        patch_radius: int = 1,
        normalize: bool = False,  # 保留参数但忽略，energy_map 已是 0/1
    ):
        super().__init__(env)
        self.patch_radius = int(patch_radius)
        self.patch_size = (2 * self.patch_radius + 1) ** 2

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

        # 更新 observation space：energy_map 是 0/1，patch 范围固定 [0, 1]
        orig_space = env.observation_space
        patch_low = np.zeros(self.patch_size, dtype=np.float32)
        patch_high = np.ones(self.patch_size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([orig_space.low, patch_low]),
            high=np.concatenate([orig_space.high, patch_high]),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _get_energy_patch(self) -> np.ndarray:
        """获取以当前 agent 位置为中心的 energy patch（0/1 值）。"""
        energy_map = getattr(self._energy_wrapper, "_energy_map", None)
        r, c = self._base_env.agent_row, self._base_env.agent_col
        grid_size = self._base_env.grid_size
        radius = self.patch_radius
        # 越界位置填充 1.0（视为高能耗区域）
        fill_value = 1.0

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
        return patch

    # ------------------------------------------------------------------ #
    # ObservationWrapper 接口
    # ------------------------------------------------------------------ #

    def observation(self, obs: Any) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        patch = self._get_energy_patch()
        return np.concatenate([obs, patch])
