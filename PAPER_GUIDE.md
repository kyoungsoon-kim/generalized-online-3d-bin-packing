# Paper Guide — section by section

A learning-oriented walk through **GOPT: Generalizable Online 3D Bin Packing via
Transformer-based Deep Reinforcement Learning** (Xiong et al., 2024), tying each section
to the code in `src/`. Read alongside `notebooks/walkthrough.ipynb`.
Official code: https://github.com/Xiong5Heng/GOPT

---

## The problem in one picture (§I, §III-A)

Items arrive **one at a time** (online), and you must place each into an `L×W×H` bin
immediately, choosing a position and a 0°/90° orientation, without knowing future items.
Goal: maximize the fraction of bin volume filled. The twist GOPT solves: prior DRL
packers are trained for ONE bin size; GOPT **generalizes across bin sizes** with a single
trained policy.

Three contributions (§I):
1. **Placement Generator (PG)** — a heuristic that proposes a fixed-size set of free
   sub-spaces, controlling the action space (`src/data.py`).
2. **Packing Transformer** — self + cross attention fusing item and sub-spaces
   (`src/model.py`). This is the headline component.
3. **PPO training** of the resulting actor-critic policy (`src/loss.py`, `src/train.py`).

---

## §III-B — Placement Generator & Empty Maximal Spaces  (`src/data.py`)

### Background: why not pick any (x, y, z)?
A raw "place anywhere" action space is enormous and bin-size-dependent. Instead GOPT
represents free volume with **Empty Maximal Spaces (EMS)**: the largest empty boxes that
still fit in the bin given current stacking. The bin's occupancy is a 2-D **heightmap**
(`heightmap[x,y]` = stacked height). Corner points are found from height changes along
X/Y; an EMS grows from each corner until it hits a taller column.

### The representation
Each EMS = its front-left-bottom (FLB) vertex + the opposite vertex = a **6-D vector,
normalized to [0,1]** (so it carries no absolute bin size — the key to generalization).
The PG outputs a fixed-length sequence of `N` EMSs (padded with dummies), ranked by
height, plus a feasibility **mask** of which (orientation, EMS) pairs are legal.

---

## §III-C — The MDP  (`src/data.py`, `src/loss.py`)

- **State** `s_t = (item, bin)`: the 2×3 **item matrix** (rows = `(l,w,h)` at 0° and
  `(w,l,h)` at 90°) + the EMS sequence.
- **Action** `a_t = (orientation, EMS)`: picking one of `2·N` joint choices. Crucially
  `|A| = 2N` depends only on the EMS count, **not the bin dimensions** (§III-C).
- **Reward** `r_t = (l_t·w_t·h_t)/(L·W·H)`: the volume fraction the placed item adds. A
  **dense** reward (every step gives signal) — §IV-D shows it beats sparse terminal reward.

---

## §III-D — The Packing Transformer  (`src/model.py`)

### Background: self- vs cross-attention
- **Self-attention** lets every element of a set attend to every other element — here,
  EMSs to EMSs (and the two item rows to each other), capturing intra-set structure.
- **Cross-attention** lets one set query another. GOPT runs it **both directions**
  (EMS→item and item→EMS), answering "which sub-space suits this item, and vice versa".
  The authors frame this as **cross-modality** learning (treating {EMS} and {item} like
  two modalities, e.g. vision↔language).

### Block structure (§III-D)
Three stacked encoder blocks, each: 2 self-attention + 1 bi-directional (=2 unidirectional)
cross-attention + 4 two-layer MLPs, every sublayer wrapped in **residual + post-LayerNorm**
(`x ← Norm(x + sublayer(x))`). Embedding width `d_model = 128`.

### Actor & critic (§III-D)
- **Actor**: pass EMS and item features through MLPs, then take the dot product of each
  item-row feature with each EMS feature → a `(2, N)` **score map** of action logits;
  mask infeasible entries to −∞; softmax → action distribution.
- **Critic**: mean-pool the EMS and item sequences into one vector → MLP → scalar `V(s)`.
  *(pooling is `[UNSPECIFIED]`; mean-pool is the standard choice — see REPRODUCTION_NOTES.)*

---

## §III-E — Training with PPO  (`src/loss.py`, `src/train.py`)

### Background: PPO in one paragraph
PPO is an on-policy actor-critic method. It collects rollouts with the current policy,
estimates how much better each action was than expected (the **advantage** `Â_t`), then
nudges the policy toward better-than-expected actions — but **clips** the probability
ratio to `[1−ε, 1+ε]` so one update can't move the policy too far (stability). Three loss
terms: clipped policy loss + value (critic) MSE + entropy bonus (exploration).

### GAE (§III-E)
The advantage is computed with **Generalized Advantage Estimation**: an exponentially
weighted (`λ_GAE`) sum of one-step TD errors that trades off bias vs variance. GOPT uses
`λ_GAE = 0.96`, `γ = 1` (future and immediate rewards equally weighted). See
`loss.compute_gae`.

### Hyperparameters (§IV-A)
Adam, LR linearly decaying from `7e-5`; 1000 epochs × 40,000 steps over 128 parallel
envs; update every 640 steps; batch 128; clip `ε=0.3`; value coef `0.5`; entropy coef
`0.001`. Converges in ~6 h on an RTX 3090.

---

## §IV — What the experiments show

- GOPT beats heuristic and DRL baselines on space utilization (§IV-B).
- It **generalizes**: a policy trained on one bin size transfers to unseen bin sizes and
  unseen item sets (§IV-C) — enabled by the size-normalized EMS representation.
- Ablations (§IV-D): removing the Packing Transformer, the item representation, or the PG
  each hurts; the Transformer matters most. Step-wise reward > terminal/heuristic reward.

---

## Further reading
- Schulman et al. (2017) — *Proximal Policy Optimization* (the RL algorithm).
- Schulman et al. (2016) — *Generalized Advantage Estimation* (the advantage estimator).
- Zhao et al. (2021/2022) — CNN/constrained-MDP online 3D-BPP (a GOPT baseline lineage).
- Vaswani et al. (2017) — *Attention Is All You Need* (self/cross-attention foundations).
