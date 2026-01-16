"""
本文件定义一个极简的 Grid 路由环境，用于验证 PPO 在标准 N*N 全连通网格上的收敛情况。

基础设定：
- 一个 N x N 的网格（默认 8x8）。
- 智能体从随机起点出发，目标是到达随机终点（起点与终点不会相同）。
- 动作空间为 4 个离散动作：0=上, 1=下, 2=左, 3=右。
- 每一步都会得到 step_penalty（默认 -1.0）。
- 当智能体到达终点时，额外得到 success_reward（默认 +10.0），并终止 episode。
- 如果超过 max_steps（默认 4*N*N），则截断 episode（truncated=True）。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridRoutingEnv(gym.Env):
    """极简 N*N 全连通 Grid 路由环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 8,
        step_penalty: float = -1.0,
        success_reward: float = 10.0,
        max_steps: int | None = None,
        start_goal_mode: str = "random",
        start_rect: tuple[int, int, int, int] | None = None,
        goal_rect: tuple[int, int, int, int] | None = None,
    ):
        """
        Args:
            grid_size: 网格大小 N（N x N）
            step_penalty: 每一步的基础惩罚（一般设为 -1）
            success_reward: 抵达目标时的额外奖励
            max_steps: episode 的最长步数（默认 4*N*N）
        """
        super().__init__()

        self.grid_size = int(grid_size)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.max_steps = (
            max_steps if max_steps is not None else 4 * self.grid_size * self.grid_size
        )

        self.start_goal_mode = start_goal_mode
        self.start_rect = start_rect
        self.goal_rect = goal_rect

        if self.start_goal_mode not in {"random", "rect"}:
            raise ValueError(f"Invalid start_goal_mode: {self.start_goal_mode}")

        # 观测：agent_row, agent_col, goal_row, goal_col，均归一化到 [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # 动作：0=上, 1=下, 2=左, 3=右
        self.action_space = spaces.Discrete(4)

        # 内部状态
        self.agent_row = 0
        self.agent_col = 0
        self.goal_row = 0
        self.goal_col = 0
        self.steps = 0

        self._rng = np.random.RandomState()

    def _sample_distinct_positions(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """随机采样两个不同坐标：起点和终点。"""
        while True:
            sr = self._rng.randint(0, self.grid_size)
            sc = self._rng.randint(0, self.grid_size)
            gr = self._rng.randint(0, self.grid_size)
            gc = self._rng.randint(0, self.grid_size)
            if (sr, sc) != (gr, gc):
                return (sr, sc), (gr, gc)

    def _sample_from_rect(self, rect: tuple[int, int, int, int]) -> Tuple[int, int]:
        """从矩形区域均匀采样一个 (row, col)。"""
        x0, x1, y0, y1 = rect
        r = self._rng.randint(x0, x1 + 1)
        c = self._rng.randint(y0, y1 + 1)
        return r, c

    def _get_obs(self) -> np.ndarray:
        """返回当前观测（归一化坐标）。"""
        gs = float(self.grid_size - 1)
        return np.array(
            [
                self.agent_row / gs,
                self.agent_col / gs,
                self.goal_row / gs,
                self.goal_col / gs,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # 确保 _rng 与 seed 绑定
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        override_start = None
        override_goal = None
        if options and isinstance(options, dict):
            override_start = options.get("override_start")
            override_goal = options.get("override_goal")

        if override_start is not None and override_goal is not None:
            self.agent_row, self.agent_col = map(int, override_start)
            self.goal_row, self.goal_col = map(int, override_goal)
            if (self.agent_row, self.agent_col) == (self.goal_row, self.goal_col):
                raise ValueError("override_start and override_goal cannot be identical")
        else:
            if self.start_goal_mode == "random":
                (self.agent_row, self.agent_col), (self.goal_row, self.goal_col) = (
                    self._sample_distinct_positions()
                )
            elif self.start_goal_mode == "rect":
                if self.start_rect is None or self.goal_rect is None:
                    raise ValueError("start_goal_mode=rect requires start_rect and goal_rect")
                self.agent_row, self.agent_col = self._sample_from_rect(self.start_rect)
                max_attempts = 1000
                attempts = 0
                while True:
                    self.goal_row, self.goal_col = self._sample_from_rect(self.goal_rect)
                    if (self.agent_row, self.agent_col) != (self.goal_row, self.goal_col):
                        break
                    attempts += 1
                    if attempts >= max_attempts:
                        raise RuntimeError("Failed to sample distinct start/goal within provided rectangles")
        self.steps = 0

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "start": (self.agent_row, self.agent_col),
            "goal": (self.goal_row, self.goal_col),
            "start_pos": (self.agent_row, self.agent_col),
            "goal_pos": (self.goal_row, self.goal_col),
            "start_goal_mode": self.start_goal_mode,
            "start_rect": self.start_rect,
            "goal_rect": self.goal_rect,
        }
        return obs, info

    def step(self, action: int):
        """执行一步动作。"""
        self.steps += 1

        nr, nc = self.agent_row, self.agent_col

        if action == 0:  # 上
            nr -= 1
        elif action == 1:  # 下
            nr += 1
        elif action == 2:  # 左
            nc -= 1
        elif action == 3:  # 右
            nc += 1
        else:
            # 非法动作，视为原地不动
            pass

        # 边界合法才移动
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            self.agent_row = nr
            self.agent_col = nc

        reward = self.step_penalty
        terminated = False
        truncated = False

        # 是否到达目标
        if self.agent_row == self.goal_row and self.agent_col == self.goal_col:
            reward += self.success_reward
            terminated = True

        # 步数上限
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info: Dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def _get_action_mask(self) -> np.ndarray:
        """
        返回当前状态下合法动作的 mask，shape=(4,)，dtype=np.bool_。
        动作定义：0=上, 1=下, 2=左, 3=右

        规则：只检查边界（防止越界）
        """
        mask = np.ones(4, dtype=np.bool_)
        r, c = self.agent_row, self.agent_col
        N = self.grid_size

        # 越界屏蔽
        if r == 0:
            mask[0] = False  # 上
        if r == N - 1:
            mask[1] = False  # 下
        if c == 0:
            mask[2] = False  # 左
        if c == N - 1:
            mask[3] = False  # 右

        return mask
