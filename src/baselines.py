"""
GOPT — heuristic baselines vs the trained PPO policy (apples-to-apples).

Evaluates, on the *same* seeded stream of item episodes, three online 3D-BPP solvers:
  - Random        : uniform over feasible (orientation, EMS) actions.
  - Heightmap-greedy: place into the feasible EMS with the lowest base height z0
                      (deepest-bottom packing) — the classic online-BPP heuristic.
  - PPO (GOPT)    : the trained Transformer policy, greedy argmax over masked logits.

All three drive the identical BinPackingEnv (src/data.py) with the identical reward /
utilization metric, so the mean bin space utilization is directly comparable. This is the
quantitative "does the learned policy beat the heuristic?" check the README reports.

Run:
    py -3.12 -X utf8 -m src.baselines --episodes 256                 # heuristics only
    py -3.12 -X utf8 -m src.baselines --episodes 256 --checkpoint outputs/best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data import BinPackingEnv
from src.utils import load_config


# --------------------------------------------------------------------- policies
def random_action(env: BinPackingEnv) -> int:
    """Uniform over feasible actions (orientation, EMS)."""
    return env.random_valid_action()


def heightmap_greedy_action(env: BinPackingEnv) -> int:
    """Lowest-base-height feasible placement (deepest-bottom heuristic).

    env.ems_boxes is ranked by base height z0 ascending (§III-B), so the first feasible
    (orientation, EMS) in that order is the lowest landing spot. Independent of any network.
    """
    N = env.max_ems
    mask = env.action_mask  # (2, N) bool
    best_idx, best_z = 0, None
    for i, box in enumerate(env.ems_boxes):
        z0 = box[2]
        for o in range(2):
            if mask[o, i]:
                if best_z is None or z0 < best_z:
                    best_z, best_idx = z0, o * N + i
                break  # both orientations share z0; first feasible orientation suffices
        if best_z is not None:
            break  # boxes are z0-sorted -> first feasible box is globally lowest
    return best_idx


# ------------------------------------------------------------------- evaluation
def eval_policy(action_fn, cfg, n_episodes: int, seed: int = 12345) -> tuple[float, float]:
    """Mean (utilization, items packed) over n_episodes seeded rollouts."""
    rng = np.random.default_rng(seed)
    utils, nums = [], []
    for _ in range(n_episodes):
        env = BinPackingEnv(L=cfg["env"]["L"], W=cfg["env"]["W"], H=cfg["env"]["H"],
                            max_ems=cfg["model"]["max_ems"], rng=rng)
        env.reset(); done = False; steps = 0
        while not done and steps < 300:
            action = action_fn(env)
            _, _, done, _ = env.step(action); steps += 1
        utils.append(env.utilization()); nums.append(env.num_packed)
    return float(np.mean(utils)), float(np.mean(nums))


def eval_ppo(checkpoint: str, cfg, device, n_episodes: int, seed: int = 12345):
    """Greedy rollout of the trained GOPT policy on the same seeded episode stream."""
    from src.model import GOPTModel
    from src.utils import ModelConfig
    m = cfg["model"]
    model = GOPTModel(ModelConfig(
        ems_dim=m["ems_dim"], item_dim=m["item_dim"], d_model=m["d_model"],
        n_layers=m["n_layers"], n_heads=m["n_heads"], d_ff=m["d_ff"],
        dropout=m["dropout"], norm_eps=m["norm_eps"], max_ems=m["max_ems"]))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    rng = np.random.default_rng(seed)
    utils, nums = [], []
    for _ in range(n_episodes):
        env = BinPackingEnv(L=cfg["env"]["L"], W=cfg["env"]["W"], H=cfg["env"]["H"],
                            max_ems=cfg["model"]["max_ems"], rng=rng)
        state = env.reset(); done = False; steps = 0
        while not done and steps < 300:
            ems = state[0].unsqueeze(0).to(device)
            item = state[1].unsqueeze(0).to(device)
            mask = state[2].unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(ems, item, mask)
            action = int(logits.argmax(dim=-1).item())
            state, _, done, _ = env.step(action); steps += 1
        utils.append(env.utilization()); nums.append(env.num_packed)
    return float(np.mean(utils)), float(np.mean(nums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--checkpoint", default=None, help="trained GOPT policy (optional)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    rows = []
    u, n = eval_policy(random_action, cfg, args.episodes, args.seed)
    rows.append(("Random (feasible)", u, n))
    u, n = eval_policy(heightmap_greedy_action, cfg, args.episodes, args.seed)
    rows.append(("Heightmap-greedy (lowest-EMS)", u, n))
    if args.checkpoint:
        u, n = eval_ppo(args.checkpoint, cfg, device, args.episodes, args.seed)
        rows.append(("PPO (GOPT, greedy)", u, n))

    rows.sort(key=lambda r: r[1])  # worst utilization first
    L, W, H = cfg["env"]["L"], cfg["env"]["W"], cfg["env"]["H"]
    print(f"== Online 3D-BPP {L}x{W}x{H} | {args.episodes} episodes (seed {args.seed}) ==")
    print(f"{'method':<32} {'utilization':>12} {'items':>8}")
    print("-" * 54)
    for name, util, num in rows:
        print(f"{name:<32} {util:>11.1%} {num:>8.1f}")
    print(f"{'(paper GOPT, full 40M steps)':<32} {0.761:>11.1%} {29.6:>8.1f}")


if __name__ == "__main__":
    main()
