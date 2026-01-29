"""
GridCongestionObsWrapper

在原始观测后面拼接一个 congestion 局部 patch，使 agent 能够感知周围的拥塞情况。

功能：
- 在原始 obs（4维：位置+相对目标）后面拼接一个 congestion 局部 patch
- 默认 patch_radius=1，即 3x3=9 维，总观测维度变为 13
- patch 的值来自 GridCostWrapper 里保存的 _congestion_map
- 以 agent 当前 (row, col) 为中心取邻域；越界位置用 1.0 填充（视为拥塞）

使用示例：
    base_env = GridRoutingEnv(...)
    env = GridCostWrapper(base_env, load_mode="cell_congestion", ...)
    env = GridHardWrapper(env)
    env = GridCongestionObsWrapper(env, patch_radius=1)  # obs_dim: 4 + 9 = 13
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from grid_env import GridRoutingEnv


def find_wrapper_with_attr(env: gym.Env, attr_name: str) -> Optional[gym.Env]:
    """
    沿着 env wrapper 链向下查找，返回第一个拥有指定属性的 env。

    Args:
        env: 起始环境（通常是最外层 wrapper）
        attr_name: 要查找的属性名

    Returns:
        拥有该属性的 env，或 None（找不到时）
    """
    current = env
    visited = set()

    while True:
        if hasattr(current, attr_name):
            return current

        # 防止死循环
        if id(current) in visited:
            break
        visited.add(id(current))

        # 继续往里剥
        if hasattr(current, "env"):
            current = current.env
        else:
            break

    return None


class GridCongestionObsWrapper(gym.ObservationWrapper):
    """
    在原始观测后面拼接一个 congestion 局部 patch。

    观测格式：
        new_obs = [original_obs..., congestion_patch...]

    其中 congestion_patch 是以 agent 当前位置为中心的 (2*radius+1) x (2*radius+1) 邻域，
    展平为一维向量。越界位置用 1.0 填充（视为不可通行/拥塞）。
    """

    def __init__(self, env: gym.Env, patch_radius: int = 1):
        """
        Args:
            env: 包含 GridCostWrapper 的环境
            patch_radius: 邻域半径，默认 1（即 3x3 patch）
        """
        super().__init__(env)

        self.patch_radius = patch_radius
        self.patch_size = (2 * patch_radius + 1) ** 2  # 3x3=9, 5x5=25, etc.

        # 查找 GridCostWrapper（拥有 _congestion_map 属性）
        self._cost_wrapper = find_wrapper_with_attr(env, "_congestion_map")
        if self._cost_wrapper is None:
            raise RuntimeError(
                "GridCongestionObsWrapper: 无法找到拥有 _congestion_map 属性的 wrapper。"
                "请确保环境链中包含 GridCostWrapper 且 load_mode='cell_congestion'。"
            )

        # 查找底层 GridRoutingEnv（用于获取 agent 位置）
        base = env
        visited = set()
        self._base_env: GridRoutingEnv | None = None

        while True:
            if isinstance(base, GridRoutingEnv):
                self._base_env = base
                break
            if id(base) in visited:
                break
            visited.add(id(base))
            if hasattr(base, "env"):
                base = base.env
            else:
                break

        if self._base_env is None:
            raise RuntimeError(
                "GridCongestionObsWrapper: 无法找到底层 GridRoutingEnv。"
            )

        # 更新观测空间
        original_shape = env.observation_space.shape
        original_dim = original_shape[0] if original_shape else 0
        new_dim = original_dim + self.patch_size

        # 原始空间的 low/high
        original_low = env.observation_space.low
        original_high = env.observation_space.high

        # patch 部分的 low/high（congestion 值在 [0, 1]）
        patch_low = np.zeros(self.patch_size, dtype=np.float32)
        patch_high = np.ones(self.patch_size, dtype=np.float32)  # congestion_map 永远在 [0,1]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([original_low, patch_low]),
            high=np.concatenate([original_high, patch_high]),
            dtype=np.float32,
        )

        self._original_dim = original_dim

    def _get_congestion_patch(self) -> np.ndarray:
        """
        获取以当前 agent 位置为中心的 congestion patch。

        Returns:
            展平的 patch，shape=(patch_size,)
        """
        congestion_map = getattr(self._cost_wrapper, "_congestion_map", None)
        if congestion_map is None:
            # 如果没有 congestion map，返回全 0
            return np.zeros(self.patch_size, dtype=np.float32)

        r, c = self._base_env.agent_row, self._base_env.agent_col
        grid_size = self._base_env.grid_size
        radius = self.patch_radius

        patch = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                # 越界位置填充 1.0（视为拥塞/不可通行）
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    patch.append(float(congestion_map[nr, nc]))
                else:
                    patch.append(1.0)

        return np.array(patch, dtype=np.float32)

    def observation(self, obs: Any) -> np.ndarray:
        """
        将原始观测与 congestion patch 拼接。

        Args:
            obs: 原始观测

        Returns:
            拼接后的观测
        """
        obs = np.asarray(obs, dtype=np.float32)
        patch = self._get_congestion_patch()
        return np.concatenate([obs, patch])
