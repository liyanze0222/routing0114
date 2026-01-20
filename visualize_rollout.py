import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from grid_env import GridRoutingEnv
from grid_cost_env import GridCostWrapper
from grid_hard_wrapper import GridHardWrapper
from grid_congestion_obs_wrapper import GridCongestionObsWrapper
from grid_energy_obs_wrapper import GridEnergyObsWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from grid_obs_norm_wrapper import GridObsNormWrapper
from networks import ActorCritic, MultiHeadActorCritic


def load_config_from_dir(ckpt_path: str) -> dict:
    """Load config.json colocated with checkpoint if present."""
    ckpt_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


def inject_obs_stats(env, checkpoint, cfg):
    need_norm = cfg.get("obs_rms", False)
    stats = checkpoint.get("obs_stats")

    # locate wrapper
    target = None
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GridObsNormWrapper):
            target = cur
            break
        cur = getattr(cur, "env", None)

    if need_norm and target is None:
        raise RuntimeError("❌ Config开启 obs_rms，但环境中未找到 GridObsNormWrapper")
    if stats is not None and target is None:
        raise RuntimeError("❌ Checkpoint 含 obs_stats，但环境未挂载 GridObsNormWrapper")
    if need_norm and stats is None:
        raise RuntimeError("❌ Config开启 obs_rms，但 checkpoint 中缺少 obs_stats")

    if target is not None and stats is not None:
        current_dim = env.observation_space.shape[0]
        stats_dim = stats.mean.shape[0]
        if current_dim != stats_dim:
            raise RuntimeError(
                f"❌ Obs Dimension Mismatch! Env: {current_dim}, Checkpoint: {stats_dim}. 请检查 Config 是否与模型匹配！"
            )
        target.obs_rms = stats
        target.eval()
        if hasattr(target, "training"):
            target.training = False
        if hasattr(target, "norm_reward"):
            target.norm_reward = False
        print("✅ Obs Stats injected & Frozen (Eval Mode).")


def _get_action_mask(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "get_action_mask"):
            try:
                return cur.get_action_mask()
            except Exception:
                return None
        cur = getattr(cur, "env", None)
    return None


def find_cost_wrapper(env):
    cur = env
    while True:
        if isinstance(cur, GridCostWrapper):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None:
            return None
        cur = nxt


def make_env_from_config(cfg: dict, seed: int):
    # 这些 key 与 eval_fixed_set.py 保持一致
    grid_size = cfg.get("grid_size", 8)
    step_penalty = cfg.get("step_penalty", -1.0)
    success_reward = cfg.get("success_reward", 20.0)
    max_steps = cfg.get("max_steps", 256)
    congestion_pattern = cfg.get("congestion_pattern", "block")
    congestion_density = cfg.get("congestion_density", 0.40)
    energy_high_cost = cfg.get("energy_high_cost", 3.0)
    energy_high_density = cfg.get("energy_high_density", 0.20)
    load_cost_scale = cfg.get("load_cost_scale", 1.0)

    include_congestion_obs = cfg.get("include_congestion_obs", True)
    congestion_patch_radius = cfg.get("congestion_patch_radius", 2)
    include_energy_obs = cfg.get("include_energy_obs", True)
    energy_patch_radius = cfg.get("energy_patch_radius", 2)
    energy_obs_normalize = cfg.get("energy_obs_normalize", True)
    obs_rms = cfg.get("obs_rms", False)

    env = GridRoutingEnv(
        grid_size=grid_size,
        step_penalty=step_penalty,
        success_reward=success_reward,
        max_steps=max_steps,
    )
    env = GridCostWrapper(
        env,
        congestion_pattern=congestion_pattern,
        congestion_density=congestion_density,
        energy_high_cost=energy_high_cost,
        energy_high_density=energy_high_density,
        load_cost_scale=load_cost_scale,
    )
    # 注意：hard wrapper 在 obs wrapper 之前（与训练/评估一致）
    env = GridHardWrapper(env)
    if include_congestion_obs:
        env = GridCongestionObsWrapper(env, patch_radius=congestion_patch_radius)
    if include_energy_obs:
        env = GridEnergyObsWrapper(env, patch_radius=energy_patch_radius, normalize=energy_obs_normalize)
    if obs_rms:
        env = GridObsNormWrapper(env)
    if energy_obs_normalize:
        env = GridObsNormWrapper(env)

    env.reset(seed=seed)
    return env


def load_agent(ckpt_path: str, obs_dim: int, act_dim: int, device: str, checkpoint=None):
    checkpoint = checkpoint or torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("network_state_dict", checkpoint.get("model_state_dict", checkpoint))

    is_multi_head = any(
        ("v_cost_heads" in k)
        or ("cost_value_heads" in k)
        or ("cost_critics" in k)
        or ("actor_backbone" in k)
        or ("reward_backbone" in k)
        or ("cost_backbone" in k)
        for k in state_dict.keys()
    )

    cfg = load_config_from_dir(ckpt_path)
    hidden_dim = cfg.get("hidden_dim", 128)

    if is_multi_head:
        agent = MultiHeadActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            cost_names=["energy", "load"],
            cost_critic_mode=cfg.get("cost_critic_mode", "separate"),
            value_head_mode=cfg.get("value_head_mode", "standard"),
        ).to(device)
    else:
        agent = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
        ).to(device)

    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, is_multi_head


@torch.no_grad()
def select_action(agent, obs_np: np.ndarray, action_mask_np, deterministic: bool, device: str):
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    mask = None
    if action_mask_np is not None:
        mask = torch.tensor(action_mask_np, dtype=torch.float32, device=device)

    if not deterministic:
        if hasattr(agent, "get_action"):
            out = agent.get_action(obs, mask)
            return int(out[0])
        raise RuntimeError("agent has no get_action()")

    if isinstance(agent, MultiHeadActorCritic):
        if obs.dim() == 1:
            obs_b = obs.unsqueeze(0)
        else:
            obs_b = obs
        logits, _, _ = agent.forward(obs_b, mask)
        logits = logits.squeeze(0)
    else:
        if obs.dim() == 1:
            obs_b = obs.unsqueeze(0)
        else:
            obs_b = obs
        logits, _ = agent.forward(obs_b)
        logits = logits.squeeze(0)
        if mask is not None:
            if mask.dim() == 2:
                logits = logits.masked_fill(mask.squeeze(0) == 0, float("-inf"))
            else:
                logits = logits.masked_fill(mask == 0, float("-inf"))

    return int(torch.argmax(logits).item())


def draw_arrows(ax, xs, ys, every: int = 5):
    for i in range(0, len(xs) - 1, every):
        ax.add_patch(
            FancyArrowPatch(
                (xs[i], ys[i]),
                (xs[i + 1], ys[i + 1]),
                arrowstyle="->",
                mutation_scale=10,
                linewidth=1.5,
                alpha=0.9,
            )
        )


def plot_traj(
    traj_rc,
    start_rc,
    goal_rc,
    grid_size,
    heatmap=None,
    visit_map=None,
    title="",
    no_color=False,
    draw_arrow=False,
    arrow_every=5,
    annotate=False,
    annotate_fmt="{:.0f}",
    save_path=None,
):
    traj_rc = np.asarray(traj_rc, dtype=np.int32)
    xs = traj_rc[:, 1].astype(np.float32)
    ys = traj_rc[:, 0].astype(np.float32)

    plt.figure()
    ax = plt.gca()

    if heatmap is not None:
        n = heatmap.shape[0]
        ax.imshow(
            heatmap,
            origin="upper",
            extent=(-0.5, n - 0.5, n - 0.5, -0.5),
            interpolation="nearest",
        )
    else:
        n = grid_size

    if visit_map is not None:
        ax.imshow(
            visit_map,
            origin="upper",
            extent=(-0.5, n - 0.5, n - 0.5, -0.5),
            interpolation="nearest",
            alpha=0.35,
        )
        plt.colorbar(ax.images[-1], ax=ax, label="visit count")

    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=1, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if len(xs) >= 2 and not no_color:
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, array=np.arange(len(segs)), linewidth=3)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="step")
        ax.autoscale()
    else:
        ax.plot(xs, ys, linewidth=2)

    if draw_arrow:
        draw_arrows(ax, xs, ys, every=max(1, arrow_every))

    ax.scatter([start_rc[1]], [start_rc[0]], marker="o", s=60, label="start")
    ax.scatter([goal_rc[1]], [goal_rc[0]], marker="*", s=80, label="goal")
    ax.scatter([xs[-1]], [ys[-1]], marker="x", s=60, label="end")

    if annotate and heatmap is not None:
        for r in range(n):
            for c in range(n):
                ax.text(
                    c,
                    r,
                    annotate_fmt.format(heatmap[r, c]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    alpha=0.75,
                )

    ax.set_xlabel("col (x)")
    ax.set_ylabel("row (y)")
    ax.grid(False)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", type=str, default="none", choices=["none", "energy", "load", "congestion"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--arrows", action="store_true")
    ap.add_argument("--arrow_every", type=int, default=5)
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    cfg = load_config_from_dir(args.ckpt_path)
    env = make_env_from_config(cfg, args.seed)
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    inject_obs_stats(env, checkpoint, cfg)
    obs, info = env.reset(seed=args.seed)

    obs_dim = obs.shape[0]
    act_dim = env.action_space.n
    agent, _ = load_agent(args.ckpt_path, obs_dim, act_dim, args.device, checkpoint)

    start_rc = info.get("start") or info.get("start_pos")
    goal_rc = info.get("goal") or info.get("goal_pos")
    if start_rc is None:
        start_rc = (env.unwrapped.agent_row, env.unwrapped.agent_col)
    if goal_rc is None:
        goal_rc = (env.unwrapped.goal_row, env.unwrapped.goal_col)

    traj = []
    rewards = []
    costs = []
    loads = []
    traj.append((env.unwrapped.agent_row, env.unwrapped.agent_col))

    terminated = False
    truncated = False
    while not (terminated or truncated):
        mask = _get_action_mask(env)
        a = select_action(agent, obs, mask, args.deterministic, args.device)
        obs, reward, terminated, truncated, info = env.step(a)

        cost_components = info.get("cost_components", {})
        load_val = cost_components.get("load", 0.0)
        rewards.append(reward)
        costs.append(cost_components)
        loads.append(load_val)
        traj.append((env.unwrapped.agent_row, env.unwrapped.agent_col))

    cw = find_cost_wrapper(env)
    heatmap = None
    map_key = args.map
    annotate_fmt = "{:.0f}"
    if map_key in ("load", "congestion"):
        if cw is not None and getattr(cw, "_congestion_map", None) is not None:
            heatmap = cw._congestion_map
            annotate_fmt = "{:.2f}"
    elif map_key == "energy":
        if cw is not None and getattr(cw, "_energy_map", None) is not None:
            heatmap = cw._energy_map
            annotate_fmt = "{:.0f}"

    n = getattr(env.unwrapped, "grid_size", heatmap.shape[0] if heatmap is not None else len(traj))
    visit = np.zeros((n, n), dtype=np.int32)
    for r, c in traj:
        if 0 <= r < n and 0 <= c < n:
            visit[r, c] += 1

    title = f"seed={args.seed}, steps={len(traj) - 1}, map={args.map}"
    plot_traj(
        traj,
        start_rc,
        goal_rc,
        grid_size=n,
        heatmap=heatmap,
        visit_map=visit,
        title=title,
        no_color=args.no_color,
        draw_arrow=args.arrows,
        arrow_every=args.arrow_every,
        annotate=args.annotate,
        annotate_fmt=annotate_fmt,
        save_path=args.save,
    )

    if loads:
        plt.figure()
        plt.plot(np.cumsum(loads))
        plt.xlabel("step")
        plt.ylabel("cumulative load cost")
        plt.title("Cumulative load along trajectory")
        plt.grid(True)
        if args.save:
            base, ext = os.path.splitext(args.save)
            plt.savefig(f"{base}_load{ext}", dpi=200, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    main()
