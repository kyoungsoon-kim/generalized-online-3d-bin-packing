"""
GOPT — PPO training loop (working, vectorized).

Paper: GOPT (Xiong et al., 2024). Implements the §III-E / §IV-A on-policy PPO cycle:
  COLLECT rollouts in parallel envs -> estimate advantages with GAE -> minibatch PPO
  updates, with a linearly-decaying LR. Periodically evaluates greedy utilization.

  Educational note: this is the real loop (not a dummy step). It trains the GOPT policy on
  the BinPackingEnv (src/data.py) and reports bin space utilization vs the paper's 76.1%.
  Full reproduction needs the paper's 40M-step budget; the defaults here are a much smaller,
  laptop/GPU-runnable budget that demonstrates learning (utilization rising above random).

Usage:
  py -3.12 -m src.train --steps 400000           # ~ that many env steps
  py -3.12 -m src.train --steps 50000 --device cpu
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.distributions import Categorical

from src.data import BinPackingEnv
from src.loss import compute_gae, ppo_loss
from src.model import GOPTModel
from src.utils import ModelConfig, load_config


def build_model(cfg_dict: dict) -> tuple[GOPTModel, ModelConfig]:
    m = cfg_dict["model"]
    config = ModelConfig(
        ems_dim=m["ems_dim"], item_dim=m["item_dim"], d_model=m["d_model"],
        n_layers=m["n_layers"], n_heads=m["n_heads"], d_ff=m["d_ff"],
        dropout=m["dropout"], norm_eps=m["norm_eps"], max_ems=m["max_ems"],
    )
    return GOPTModel(config), config


def _batch_states(states, device):
    """Stack a list of (ems (N,6), item (2,3), mask (2,N)) into batched tensors."""
    ems = torch.stack([s[0] for s in states]).to(device)    # (B, N, 6)
    item = torch.stack([s[1] for s in states]).to(device)   # (B, 2, 3)
    mask = torch.stack([s[2] for s in states]).to(device)   # (B, 2, N)
    return ems, item, mask


@torch.no_grad()
def evaluate(model, cfg, device, n_episodes=64, seed=12345):
    """Greedy rollout (argmax over masked logits, §III-C testing) -> mean utilization, items."""
    model.eval()
    rng = np.random.default_rng(seed)
    utils, nums = [], []
    for _ in range(n_episodes):
        env = BinPackingEnv(L=cfg["env"]["L"], W=cfg["env"]["W"], H=cfg["env"]["H"],
                            max_ems=cfg["model"]["max_ems"], rng=rng)
        state = env.reset(); done = False; steps = 0
        while not done and steps < 300:
            ems, item, mask = _batch_states([state], device)
            logits, _ = model(ems, item, mask)
            action = int(logits.argmax(dim=-1).item())
            state, _, done, _ = env.step(action); steps += 1
        utils.append(env.utilization()); nums.append(env.num_packed)
    model.train()
    return float(np.mean(utils)), float(np.mean(nums))


def train(config_path: str, total_steps: int, device: str):
    cfg = load_config(config_path)
    t, e, mcfg = cfg["training"], cfg["env"], cfg["model"]
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    model, config = build_model(cfg)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr_start"])

    n_envs = 64           # parallel envs (paper: 128, §IV-A)
    rollout_len = 20      # steps collected per env per iteration
    batch_per_iter = n_envs * rollout_len
    n_iters = max(1, total_steps // batch_per_iter)
    # §IV-A — linearly descending LR from lr_start over the whole run.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: max(0.0, 1.0 - it / n_iters))

    rng = np.random.default_rng(cfg["seed"] if "seed" in cfg else 0)
    envs = [BinPackingEnv(L=e["L"], W=e["W"], H=e["H"], max_ems=mcfg["max_ems"], rng=rng)
            for _ in range(n_envs)]
    states = [env.reset() for env in envs]

    print(f"device={dev} | {n_iters} iters x {batch_per_iter} steps = "
          f"{n_iters * batch_per_iter} env steps (paper uses 40,000,000)")
    u0, num0 = evaluate(model, cfg, dev)
    print(f"[init ] greedy utilization = {u0:.1%}  items = {num0:.1f}")

    N = config.max_ems
    for it in range(n_iters):
        # ----- 1. COLLECT rollout (T = rollout_len) -----
        b_ems = torch.zeros(rollout_len, n_envs, N, 6, device=dev)
        b_item = torch.zeros(rollout_len, n_envs, 2, 3, device=dev)
        b_mask = torch.zeros(rollout_len, n_envs, 2, N, dtype=torch.bool, device=dev)
        b_act = torch.zeros(rollout_len, n_envs, dtype=torch.long, device=dev)
        b_logp = torch.zeros(rollout_len, n_envs, device=dev)
        b_val = torch.zeros(rollout_len, n_envs, device=dev)
        b_rew = torch.zeros(rollout_len, n_envs, device=dev)
        b_done = torch.zeros(rollout_len, n_envs, device=dev)

        for step in range(rollout_len):
            ems, item, mask = _batch_states(states, dev)
            with torch.no_grad():
                logits, value = model(ems, item, mask)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            b_ems[step], b_item[step], b_mask[step] = ems, item, mask
            b_act[step], b_logp[step], b_val[step] = action, logp, value.squeeze(-1)

            for i, env in enumerate(envs):
                ns, r, done, info = env.step(int(action[i].item()))
                b_rew[step, i] = r
                b_done[step, i] = float(done)
                states[i] = env.reset() if done else ns

        with torch.no_grad():
            ems, item, mask = _batch_states(states, dev)
            _, last_value = model(ems, item, mask)
            last_value = last_value.squeeze(-1)

        # ----- 2. GAE advantages + returns -----
        adv, ret = compute_gae(b_rew, b_val, b_done, last_value,
                               gamma=t["gamma"], gae_lambda=t["gae_lambda"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize advantages (common PPO trick)

        # flatten (T,B) -> (T*B,)
        f_ems = b_ems.reshape(-1, N, 6)
        f_item = b_item.reshape(-1, 2, 3)
        f_mask = b_mask.reshape(-1, 2, N)
        f_act = b_act.reshape(-1)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)

        # ----- 3. PPO minibatch updates -----
        n = f_act.shape[0]
        bs = t["batch_size"]
        for _ in range(4):  # PPO epochs over the collected batch
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                logits, value = model(f_ems[idx], f_item[idx], f_mask[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(f_act[idx])
                entropy = dist.entropy()
                loss, *_ = ppo_loss(
                    new_logp, f_logp[idx], f_adv[idx], value.squeeze(-1), f_ret[idx],
                    entropy, clip_ratio=t["clip_ratio"], c_value=t["c_value"],
                    c_entropy=t["c_entropy"])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # stabilize
                optimizer.step()
        scheduler.step()

        if (it + 1) % max(1, n_iters // 20) == 0 or it == n_iters - 1:
            u, num = evaluate(model, cfg, dev)
            print(f"[{it+1:4d}/{n_iters}] steps={ (it+1)*batch_per_iter:>8d}  "
                  f"util={u:.1%}  items={num:.1f}  lr={scheduler.get_last_lr()[0]:.1e}")

    uf, numf = evaluate(model, cfg, dev, n_episodes=256)
    print(f"\n[final] greedy utilization = {uf:.1%}  items = {numf:.1f}  "
          f"(paper GOPT: 76.1% / 29.6; random: ~25% / ~10)")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=400_000, help="approx total env steps")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train(args.config, args.steps, args.device)
