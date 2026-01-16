"""
grid_hard_wrapper.py

为 GridRoutingEnv 提供硬约束 (Shielding)：禁止越界动作。
可包在任意外层 wrapper 外（例如 GridCostWrapper），内部会自动向下寻找真正的 GridRoutingEnv。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class GridHardWrapper(gym.Wrapper):
    """提供硬约束 action_mask，禁止非法动作（仅边界检查）。"""

    def __init__(self, env):
        super().__init__(env)
        # 向下找到真正的底层 Grid 环境（拥有 agent_row/agent_col/grid_size 等属性）
        base = env
        visited = set()
        while True:
            # 找到一个看起来像 GridRoutingEnv 的对象
            if hasattr(base, "grid_size") and hasattr(base, "agent_row"):
                break
            # 防止死循环
            if id(base) in visited:
                break
            visited.add(id(base))
            # gym.Wrapper 一般有 env 属性，继续往里剥
            if hasattr(base, "env"):
                base = base.env
            else:
                break

        self._base_env = base
        if not hasattr(self._base_env, "grid_size"):
            raise AttributeError(
                "GridHardWrapper: underlying env has no 'grid_size' attribute; "
                "请确认它最终包着的是一个 GridRoutingEnv 类环境。"
            )

        # 从底层 env 读取参数
        self.grid_size = getattr(self._base_env, "grid_size")

    def _get_agent_pos(self):
        """从底层 env 取出当前 agent 所在的 (row, col)。"""
        base = self._base_env
        if not (hasattr(base, "agent_row") and hasattr(base, "agent_col")):
            raise AttributeError(
                "GridHardWrapper: underlying env has no 'agent_row' / 'agent_col' attributes."
            )
        return int(base.agent_row), int(base.agent_col)

    def get_action_mask(self) -> np.ndarray:
        """
        生成长度 = action_space.n 的布尔 mask（True=可用动作，False=非法动作）

        动作假定为：
            0 = 上, 1 = 下, 2 = 左, 3 = 右

        优先调用底层 GridRoutingEnv._get_action_mask()，
        若底层没有该方法则 fallback 到边界检查逻辑。
        """
        # 优先使用底层 env 的 _get_action_mask
        if hasattr(self._base_env, "_get_action_mask"):
            return self._base_env._get_action_mask()

        # ====== Fallback: 边界检查逻辑 ======
        r, c = self._get_agent_pos()
        N = self.grid_size

        # 所有动作默认合法
        mask = np.ones(self.action_space.n, dtype=np.bool_)

        # 越界屏蔽
        if r == 0:
            mask[0] = False  # up
        if r == N - 1:
            mask[1] = False  # down
        if c == 0:
            mask[2] = False  # left
        if c == N - 1:
            mask[3] = False  # right

        return mask

    # ----------------------------------------------------------
    # reset() 返回 obs, info，其中 info 带上 action_mask
    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        mask = self.get_action_mask()
        if info is None:
            info = {}
        else:
            info = dict(info)
        info["action_mask"] = mask
        return obs, info

    # ----------------------------------------------------------
    # step() 同样返回 action_mask
    # ----------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = self.get_action_mask()
        if info is None:
            info = {}
        else:
            info = dict(info)
        info["action_mask"] = mask
        return obs, reward, terminated, truncated, info
