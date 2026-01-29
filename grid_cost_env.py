"""
grid_cost_env.py

为 GridRoutingEnv 添加软约束用的 cost 信号。

========== 说明 ==========

1) energy（二值 0/1）：
   - 高能耗区域 energy=1.0，普通区域 energy=0.0
   - 无效移动（撞墙）不消耗能量 (energy=0)

2) load（软阈值映射到 [0,1]）：
   - congestion_map 中的 raw 值 ∈ [0,1]
   - load = max(0, (raw - load_threshold) / (1 - load_threshold))
   - 即：低于阈值的拥塞不计入 cost，高于阈值的线性映射到 [0,1]

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
import hashlib

from grid_env import GridRoutingEnv

# 支持的 congestion_pattern
CongestionPattern = Literal["random", "block"]


class GridCostWrapper(gym.Wrapper):
    """在 GridRoutingEnv 上附加 invalid / energy / load 三种 cost，并暴露 action mask。"""

    def __init__(
        self,
        env: GridRoutingEnv,
        energy_high_density: float = 0.2,
        congestion_density: float = 0.3,
        congestion_pattern: CongestionPattern = "random",
        load_threshold: float = 0.6,
        randomize_maps_each_reset: bool = True,
        keep_maps_across_episodes: bool = False,
    ):
        """
        Args:
            env:                基础 GridRoutingEnv 环境实例。
            energy_high_density:
                                高能耗区域在地图中的比例（0~1）。
            congestion_density: 拥塞节点的比例（0~1），控制面积/块大小
            congestion_pattern: 拥塞图生成模式：
                                - "random": 随机分布的拥塞点（默认）
                                - "block": 在地图上生成一个连续的拥塞块
            load_threshold:     load cost 的软阈值 τ（默认 0.6）
                                load = max(0, (raw - τ) / (1 - τ))
            randomize_maps_each_reset:
                                是否在每次 reset 时重采样能耗/拥塞地图
            keep_maps_across_episodes:
                                是否跨 episode 保持地图不变
        """
        super().__init__(env)

        self.energy_high_density = energy_high_density
        self.congestion_density = congestion_density
        self.congestion_pattern = congestion_pattern
        self.load_threshold = load_threshold
        # True -> 每次 reset 都重采样地图；False -> 复用上一次 reset 的地图
        self.randomize_maps_each_reset = bool(randomize_maps_each_reset)
        self.keep_maps_across_episodes = keep_maps_across_episodes

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

        # 生成或复用能耗/拥塞图
        reuse_maps = (not self.randomize_maps_each_reset) or self.keep_maps_across_episodes
        if reuse_maps and self._energy_map is not None and self._congestion_map is not None:
            pass
        else:
            self._energy_map = self._generate_energy_map()
            self._congestion_map = self._generate_congestion_map()

        info = {} if info is None else dict(info)
        info["cost_components"] = {"invalid": 0.0, "energy": 0.0, "load": 0.0}

        # 添加 cost 配置元信息，方便排查
        info["cost_components_meta"] = {
            "congestion_density": self.congestion_density,
            "congestion_pattern": self.congestion_pattern,
            "energy_high_density": self.energy_high_density,
            "load_threshold": self.load_threshold,
        }

        if self._energy_map is not None:
            energy_stats = {
                "energy_mean": float(self._energy_map.mean()),
                "energy_high_ratio": float((self._energy_map == 1.0).mean()),
            }
            info.update(energy_stats)

        # 添加拥塞图统计信息（用于调参）
        if self._congestion_map is not None:
            info["congestion_mean"] = float(self._congestion_map.mean())
            info["congestion_ratio"] = float((self._congestion_map > 0).mean())

        # 添加 map hash 便于验证固定地图
        combo_hash = None
        if self._congestion_map is not None:
            info["congestion_map_hash"] = hashlib.md5(self._congestion_map.tobytes()).hexdigest()
        if self._energy_map is not None:
            info["energy_map_hash"] = hashlib.md5(self._energy_map.tobytes()).hexdigest()
        if self._congestion_map is not None and self._energy_map is not None:
            combo = self._congestion_map.tobytes() + self._energy_map.tobytes()
            combo_hash = hashlib.md5(combo).hexdigest()
            info["map_hash"] = combo_hash

        # map_fingerprint：训练侧用于判断地图是否刷新（字符串，便于 metrics 记录）
        if combo_hash is not None:
            info["map_fingerprint"] = combo_hash

        # 确保 reset 的 info 始终带上起点/终点信息，便于上层记录
        # 若底层 env 已写入，则直接复用；否则从当前状态补充
        start_pos = info.get("start") or info.get("start_pos")
        goal_pos = info.get("goal") or info.get("goal_pos")
        if start_pos is None:
            start_pos = (getattr(self.env, "agent_row", None), getattr(self.env, "agent_col", None))
            info["start"] = start_pos
            info["start_pos"] = start_pos
        if goal_pos is None:
            goal_pos = (getattr(self.env, "goal_row", None), getattr(self.env, "goal_col", None))
            info["goal"] = goal_pos
            info["goal_pos"] = goal_pos

        return obs, info

    def _generate_energy_map(self) -> np.ndarray:
        """生成能耗地图：二值 0/1，高能耗区域=1，其余=0。"""
        rng = self._get_rng()
        density = float(np.clip(self.energy_high_density, 0.0, 1.0))
        # 生成 binary mask
        mask = rng.rand(self.grid_size, self.grid_size) < density
        energy_map = mask.astype(np.float32)
        return energy_map

    def _generate_congestion_map(self) -> np.ndarray:
        """
        根据 congestion_pattern 生成拥塞图。
        
        - density 只控制面积/块大小，不再用于缩放强度
        - 拥塞值在 [0, 1] 范围内

        Returns:
            congestion_map: shape=(grid_size, grid_size) 的拥塞图，值 ∈ [0, 1]
        """
        rng = self._get_rng()
        congestion_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if self.congestion_pattern == "random":
            # random 模式：mask ~ Bernoulli(density)，mask 为 True 的 cell 赋值 Uniform(0,1)，否则 0
            mask = rng.rand(self.grid_size, self.grid_size) < self.congestion_density
            raw_values = rng.rand(self.grid_size, self.grid_size).astype(np.float32)
            congestion_map = np.where(mask, raw_values, 0.0).astype(np.float32)

        elif self.congestion_pattern == "block":
            # block 模式：按 density 计算 block 面积；block 内赋值 Uniform(0,1)，block 外 0
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

                raw_block = rng.rand(end_row - start_row, end_col - start_col).astype(np.float32)
                congestion_map[start_row:end_row, start_col:end_col] = raw_block

        else:
            # 未知模式退化到 random
            mask = rng.rand(self.grid_size, self.grid_size) < self.congestion_density
            raw_values = rng.rand(self.grid_size, self.grid_size).astype(np.float32)
            congestion_map = np.where(mask, raw_values, 0.0).astype(np.float32)

        return congestion_map

    def step(self, action: int):
        """
        执行一步动作，并在 info["cost_components"] 中写入：
            - invalid: 本步是否为"无效动作"
            - energy:  本步能耗（0 或 1）
            - load:    本步负载 cost（经过 soft-threshold 映射到 [0,1]）
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
        # 有效移动才产生能量消耗；energy_map 已经是 0/1
        if invalid:
            energy = 0.0
        else:
            if self._energy_map is not None:
                energy = float(self._energy_map[new_r, new_c])
            else:
                energy = 0.0

        # -------------------- 3) cost_load --------------------
        # soft-threshold: load = max(0, (raw - τ) / (1 - τ))
        if invalid:
            load_cost = 0.0
        elif self._congestion_map is not None:
            load_raw = float(self._congestion_map[new_r, new_c])
            tau = self.load_threshold
            if tau < 1.0:
                load_cost = max(0.0, (load_raw - tau) / (1.0 - tau))
            else:
                # tau >= 1.0 时所有 load 都被屏蔽
                load_cost = 0.0
            # clamp 到 [0, 1]
            load_cost = min(1.0, load_cost)
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
