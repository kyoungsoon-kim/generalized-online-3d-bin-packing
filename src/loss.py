"""
GOPT — Loss Functions (PPO objective).

Paper: GOPT: Generalizable Online 3D Bin Packing via Transformer-based DRL
       (Xiong et al., 2024). Official code: https://github.com/Xiong5Heng/GOPT
Implements: §III-E (PPO clipped objective) + §IV-A (loss coefficients).

  Educational background — what PPO optimizes:
  PPO (Proximal Policy Optimization, Schulman et al. 2017) is an on-policy actor-critic
  method. The total loss has three parts:
    1. Policy (actor) loss — the clipped surrogate. It pushes up the probability of
       actions that turned out better than expected (positive advantage) and down for
       worse ones, but CLIPS how far the new policy may move from the old one so a
       single update cannot blow up the policy.
    2. Value (critic) loss — MSE between the critic's value estimate and the observed
       returns. A good critic gives low-variance advantage estimates.
    3. Entropy bonus — rewards a less-peaked action distribution, encouraging
       exploration early in training. It is SUBTRACTED from the loss (added as +entropy
       to the objective); here we add `c_entropy * (-entropy)`.

  The advantage Â_t fed in here is computed with GAE (Generalized Advantage Estimation,
  §III-E) outside this function — see `compute_gae` below.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.3,
    c_value: float = 0.5,
    c_entropy: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """§III-E, Eq.(2) — PPO clipped surrogate objective.

    §IV-A — "coefficients for value and entropy loss c_1, c_2 are 0.5 and 0.001 ...
    the clipped ratio epsilon is 0.3".

    Args (all shape (batch,)):
        new_log_probs: log π_θ(a_t | s_t) under the CURRENT policy.
        old_log_probs: log π_θ_old(a_t | s_t) under the policy that COLLECTED the data.
        advantages:    Â_t — how much better a_t was than the critic expected (GAE).
        values:        V_θ(s_t) — critic estimate for s_t.
        returns:       R_t — bootstrapped return target for the critic (= Â_t + V_old).
        entropy:       H(π_θ(· | s_t)) — policy entropy per sample.

    Returns: (total_loss, policy_loss, value_loss, entropy_loss).
    """
    # Probability ratio p_t(θ) = π_θ(a|s) / π_θ_old(a|s), computed in log-space for
    # numerical stability: exp(logπ_new − logπ_old). (§III-E, Eq.2 — "p_t(θ)").
    ratio = torch.exp(new_log_probs - old_log_probs)  # (batch,)

    # Clipped surrogate: take the MINIMUM of the unclipped and clipped terms so the
    # update is pessimistic — it never rewards moving the policy beyond [1−ε, 1+ε]·Â.
    surr1 = ratio * advantages                                            # (batch,)
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()  # negate: we MINIMIZE the loss

    # Critic regression toward the GAE returns. Low value error ⇒ low-variance advantages.
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourages exploration). Added to the objective ⇒ subtracted here.
    entropy_loss = -entropy.mean()

    # §IV-A — L = L_policy + c_1·L_value + c_2·L_entropy   (c_1=0.5, c_2=0.001).
    loss = policy_loss + c_value * value_loss + c_entropy * entropy_loss
    return loss, policy_loss, value_loss, entropy_loss


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.96,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§III-E — Generalized Advantage Estimation (Schulman et al. 2016).

    Educational note — why GAE:
    The raw advantage `R_t − V(s_t)` is unbiased but high-variance; using only the
    one-step TD error `δ_t = r_t + γV(s_{t+1}) − V(s_t)` is low-variance but biased.
    GAE interpolates between them with `λ`:  Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}.
    λ→1 ≈ Monte-Carlo (low bias), λ→0 ≈ one-step TD (low variance). The paper uses
    λ_GAE = 0.96 and γ = 1 (§IV-A — future and immediate rewards weighted equally).

    Args (time-major, shape (T, batch)):
        rewards, values, dones: per-step reward, critic value, episode-termination flag.
        last_value: V(s_T) bootstrap for the step after the last collected one. (batch,)

    Returns: (advantages, returns) each (T, batch); returns = advantages + values.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]  # zero the bootstrap across episode boundaries
        delta = rewards[t] + gamma * next_value * mask - values[t]  # TD error δ_t
        gae = delta + gamma * gae_lambda * mask * gae               # recursive GAE
        advantages[t] = gae
    returns = advantages + values  # critic target R_t = Â_t + V(s_t)
    return advantages, returns
