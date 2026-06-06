# Reproduction Notes

Implementation of **"GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning"** (Xiong et al., 2024).

This document records, for every implementation-relevant detail, whether the paper **specified** it, **partially specified** it, or left it **unspecified** (and what we chose).

Legend: ✅ SPECIFIED · 🟡 PARTIALLY_SPECIFIED · ❓ UNSPECIFIED · 📦 FROM_OFFICIAL_CODE

Official code: https://github.com/Xiong5Heng/GOPT (PyTorch).

---

## Architecture (§III-D)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| State/EMS dim | ✅ | 6 (normalized to [0, 1]) | §III-B |
| State/Item dim | ✅ | 2x3 matrix (flattened to 6) | §III-C |
| Maximum EMS ($N$) | ✅ | 80 | §IV-A |
| Feature Embedding | ✅ | 128, 2-layer MLP, LeakyReLU | §III-D |
| Transformer Layers | ✅ | 3 | §III-D |
| **Attention Heads** | ❓ | **8** | Unspecified; 8 is standard for d=128 |
| MLP in Transformer | ✅ | 2 layers, {128, 128} | §III-D |
| Bi-dir Cross Attention | 🟡 | Separate cross-attns in parallel | §III-D describes "two unidirectional cross-attention layers" |
| Normalization | ✅ | Post-LayerNorm | §III-D |
| **Dropout** | ❓ | **0.0** | Unspecified; assumed 0.0 for RL |

## Training (§IV-A)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Algorithm | ✅ | PPO (Tianshou framework) | §IV-A |
| Optimizer | ✅ | Adam | §IV-A |
| Learning Rate | ✅ | 7e-5 with linear decay | §IV-A |
| PPO value loss coef | ✅ | 0.5 | §IV-A |
| PPO entropy coef | ✅ | 0.001 | §IV-A |
| PPO clip ratio | ✅ | 0.3 | §IV-A |
| Discount factor | ✅ | 1.0 | §IV-A |
| GAE Lambda | ✅ | 0.96 | §IV-A |
| Batch Size | ✅ | 128 | §IV-A |

## Data & Environment (§IV-A)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Bin size | ✅ | 10x10x10 | §IV-A |
| Item size | ✅ | 1 ≤ l,w,h ≤ 5 (= min(L,W,H)/10 .. min/2) | §IV-A |
| Reward | ✅ | volume ratio (step-wise space utilization) | §III-C |

## Contradictions found
- "four MLP blocks of two layers": If there are 2 self-attention and 1 bi-directional cross attention (which counts as 2 cross-attentions), there are 4 attention sub-layers. The 4 MLPs are likely placed after each attention sub-layer. We implemented it this way.

---

## Environment implementation (`src/data.py`) — simplifications

The Placement Generator + physics are implemented (not a stub), so the policy is trainable
and utilization is measurable. Two deliberate simplifications, both of which likely lower the
achievable ceiling vs the paper:

1. **Stability = FULL support** (item must rest on a flat equal-height region). The paper uses
   a convex-hull / center-of-mass criterion (§III-A) that permits overhang and packs tighter.
   `🟡 PARTIALLY_SPECIFIED` → our choice is conservative.
2. **EMS = maximal FLAT empty boxes at concave corners.** A faithful but simple instance of the
   EMS scheme (§III-B); the paper's exact extraction (Fig. 3) may produce a richer candidate set.

These mean our trained policy is expected to plateau BELOW the paper's 76.1% even with unlimited
training — the gap is part compute budget, part stability/EMS modelling.

---

## RESULTS (this implementation)

Hardware: NVIDIA RTX 3060 Ti. Throughput ≈ 440 env-steps/s (pure-Python env is the bottleneck).
Greedy evaluation (argmax over masked logits, §III-C), bin 10³, items 1..5.

| Setting | Env steps | Utilization | Items packed |
|---|---|---|---|
| Random valid policy | — | 24.6% | 10.3 |
| GOPT (this impl) | 20,000 | 32.4% | 13.3 |
| GOPT (this impl) | 200,000 | 38.5% | 15.5 |
| GOPT (this impl) | 1,000,000 | _see README / training log_ | — |
| **Paper GOPT (target)** | **40,000,000** | **76.1%** | **29.6** |
| Best paper baseline (Xiong et al. [3]) | 40,000,000 | 73.8% | 28.3 |

**Status: implementation learns** (monotone ↑ from the 24.6% random baseline), but is **NOT at
paper parity**. Reasons:
- **Compute budget**: 200k–1M steps vs the paper's **40M** (we cover 0.5–2.5%). At ~440 steps/s,
  40M steps ≈ **25 h** on this GPU — out of scope for a single session.
- **Env simplifications** (full-support stability, flat-EMS) cap the ceiling below 76%.

To push toward the paper number: (a) speed up the env (vectorize EMS/heightmap in NumPy or port to
a batched/GPU sim) so 40M steps is feasible, and (b) implement convex-hull stability + the paper's
EMS extraction. The network, PPO, and GAE themselves are complete and verified.
