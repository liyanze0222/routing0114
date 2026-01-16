"""
GridLocalObsWrapper

将 GridRoutingEnv 的全局坐标观测转换为更"局部"的形式，用于测试
PPO 在不依赖全局图信息时的收敛表现。

局部观测格式：
    obs = [agent_row_norm, agent_col_norm, delta_row_norm, delta_col_norm]

其中：
    agent_row_norm = agent_row / (N-1)
    agent_col_norm = agent_col / (N-1)
    delta_row_norm = (goal_row - agent_row) / (N-1)
    delta_col_norm = (goal_col - agent_col) / (N-1)

这样 agent 只知道自己所在的位置和目标的相对方向，
不直接看到目标的绝对坐标。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from grid_env import GridRoutingEnv


class GridLocalObsWrapper(gym.ObservationWrapper):
    """
    将 GridRoutingEnv 的全局坐标观测转换为更"局部"的形式：

        obs = [agent_row_norm, agent_col_norm, delta_row_norm, delta_col_norm]

    其中：
        agent_row_norm = agent_row / (N-1)
        agent_col_norm = agent_col / (N-1)
        delta_row_norm = (goal_row - agent_row) / (N-1)
        delta_col_norm = (goal_col - agent_col) / (N-1)

    这样 agent 只知道自己所在的位置和目标的相对方向，
    不直接看到目标的绝对坐标。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # 向下找到最底层的 GridRoutingEnv，以便读取行列信息
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
            raise RuntimeError("GridLocalObsWrapper: cannot find underlying GridRoutingEnv.")

        N = self._base_env.grid_size
        # 新的观测空间：4 维，范围大致在 [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._denom = float(N - 1)

    def observation(self, obs: Any) -> np.ndarray:
        # 从底层 env 获取当前坐标和目标坐标
        base = self._base_env
        r_a = base.agent_row
        c_a = base.agent_col
        r_g = base.goal_row
        c_g = base.goal_col

        dr = (r_g - r_a) / self._denom
        dc = (c_g - c_a) / self._denom
        ar = r_a / self._denom
        ac = c_a / self._denom

        new_obs = np.array([ar, ac, dr, dc], dtype=np.float32)
        return new_obs
