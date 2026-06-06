"""
GOPT — Environment & Placement Generator (working implementation).

Paper: GOPT (Xiong et al., 2024). Implements the online 3D-BPP MDP of §III-A/B/C with a
real (if simplified) Placement Generator, so the policy can actually be trained and the
bin space utilization measured.

  Educational background — the online 3D-BPP as an MDP (§III-C):
    State  s_t = (item, bin): incoming item (2×3 matrix, two orientations) + EMS sequence.
    Action a_t = (orientation, EMS): |A| = 2·N, independent of bin size (generalization).
    Reward r_t = (l·w·h)/(L·W·H): the volume fraction the placed item adds (dense).
    Done: when no EMS is feasible for the next item (§III-C).

  Simplifications vs the paper (flagged — see REPRODUCTION_NOTES):
    • Stability = FULL support: an item may only sit on a flat region at one height
      (the paper allows the convex-hull / center-of-mass criterion, §III-A, which permits
      some overhang and packs slightly tighter). Full support is conservative but valid.
    • EMS = maximal flat empty boxes rooted at concave corners of the heightmap. This is a
      faithful, simple instance of the EMS scheme (§III-B); the paper's exact extraction
      may yield a different candidate set.
"""
from __future__ import annotations

import numpy as np
import torch


class BinPackingEnv:
    """§III-A — Online 3D Bin Packing environment (single bin, one item at a time)."""

    def __init__(self, L: int = 10, W: int = 10, H: int = 10, max_ems: int = 80,
                 item_min: int = 1, item_max: int = 5, rng: np.random.Generator | None = None):
        self.L, self.W, self.H = L, W, H
        self.max_ems = max_ems          # N — fixed EMS sequence length (§IV-A — 80)
        self.item_min = item_min        # §IV-A — RS items: 1 ≤ l,w,h ≤ 5 for a 10³ bin
        self.item_max = item_max
        self.rng = rng if rng is not None else np.random.default_rng()
        self.bin_volume = L * W * H
        self.reset()

    # -------------------------------------------------------------- item sampling
    def _sample_item(self) -> np.ndarray:
        """§IV-A — sample (l, w, h) i.i.d. uniform in [item_min, item_max] (RS dataset)."""
        return self.rng.integers(self.item_min, self.item_max + 1, size=3).astype(int)

    # -------------------------------------------------------------- EMS extraction
    def _generate_ems(self) -> list[tuple[int, int, int, int, int, int]]:
        """§III-B — produce maximal flat empty boxes rooted at concave corners.

        A column (x,y) at height z is a concave corner if its left (−X) and back (−Y)
        neighbours are walls or taller. From there we grow the largest axis-aligned box
        whose floor is FLAT at height z (full-support requirement). Each EMS is
        (x0,y0,z0, x1,y1,z1=H). Ranked by base height z (§III-B — "rank by height value").
        """
        hm = self.heightmap
        L, W, H = self.L, self.W, self.H
        boxes: list[tuple[int, int, int, int, int, int]] = []
        seen: set = set()
        for x in range(L):
            for y in range(W):
                z = int(hm[x, y])
                if z >= H:
                    continue
                left_higher = (x == 0) or (hm[x - 1, y] > z)
                back_higher = (y == 0) or (hm[x, y - 1] > z)
                if not (left_higher and back_higher):
                    continue
                # extend +X over columns that are flat at exactly height z
                x1 = x
                while x1 < L and hm[x1, y] == z:
                    x1 += 1
                # extend +Y while the whole [x, x1) strip stays flat at height z
                y1 = y
                while y1 < W and np.all(hm[x:x1, y1] == z):
                    y1 += 1
                box = (x, y, z, x1, y1, H)
                if (x1 - x) > 0 and (y1 - y) > 0 and box not in seen:
                    seen.add(box)
                    boxes.append(box)
        boxes.sort(key=lambda b: b[2])  # lowest base height first
        return boxes[: self.max_ems]

    # ----------------------------------------------------------------- state build
    def _build_state(self):
        """Return (ems_features (N,6), item_features (2,3), action_mask (2,N))."""
        self.ems_boxes = self._generate_ems()
        L, W, H = self.L, self.W, self.H

        ems_features = torch.zeros((self.max_ems, 6))
        for i, (x0, y0, z0, x1, y1, z1) in enumerate(self.ems_boxes):
            # §III-B — 6-D box (FLB + opposite vertex), normalized to [0,1] (bin-agnostic).
            ems_features[i] = torch.tensor(
                [x0 / L, y0 / W, z0 / H, x1 / L, y1 / W, z1 / H], dtype=torch.float32)

        l, w, h = self.cur_item
        # §III-C — 2×3 item matrix: row0 = (l,w,h) @0°, row1 = (w,l,h) @90°.
        # Normalized per-axis by bin dims to match the EMS [0,1] scale.
        item_features = torch.tensor(
            [[l / L, w / W, h / H], [w / L, l / W, h / H]], dtype=torch.float32)

        # §III-B feasibility mask (2 orientations × N EMSs).
        mask = torch.zeros((2, self.max_ems), dtype=torch.bool)
        orient_dims = [(l, w, h), (w, l, h)]
        for o, (li, wi, hi) in enumerate(orient_dims):
            for i, (x0, y0, z0, x1, y1, z1) in enumerate(self.ems_boxes):
                if li <= (x1 - x0) and wi <= (y1 - y0) and (z0 + hi) <= H:
                    mask[o, i] = True
        self.action_mask = mask
        return ems_features, item_features, mask

    # ----------------------------------------------------------------------- API
    def reset(self):
        self.heightmap = np.zeros((self.L, self.W), dtype=int)
        self.placed_volume = 0
        self.num_packed = 0
        self.cur_item = self._sample_item()
        return self._build_state()

    def step(self, action_idx: int):
        """Place current item per a_t=(orientation, EMS); return (state, reward, done, info).

        Flat index decodes as orientation = idx // N, ems = idx % N (matches the actor's
        (2, N) score-map flatten order in model.GOPTModel).
        """
        N = self.max_ems
        orientation = action_idx // N
        ems_idx = action_idx % N

        l, w, h = self.cur_item
        li, wi, hi = (l, w, h) if orientation == 0 else (w, l, h)

        reward = 0.0
        info = {"orientation": orientation, "ems": ems_idx, "invalid": False}

        if (not self.action_mask[orientation, ems_idx]) or ems_idx >= len(self.ems_boxes):
            # Should not happen if the policy respects the mask; guard anyway.
            info["invalid"] = True
            done = True
            return self._build_state(), reward, done, info

        x0, y0, z0, x1, y1, z1 = self.ems_boxes[ems_idx]
        # place item at the EMS FLB vertex; raise the footprint columns
        self.heightmap[x0:x0 + li, y0:y0 + wi] = z0 + hi
        vol = li * wi * hi
        self.placed_volume += vol
        self.num_packed += 1
        reward = vol / self.bin_volume  # §III-C — step-wise space-utilization gain

        # reveal next item, rebuild state; episode ends if nothing fits (§III-C)
        self.cur_item = self._sample_item()
        state = self._build_state()
        done = not bool(self.action_mask.any())
        info["utilization"] = self.placed_volume / self.bin_volume
        return state, reward, done, info

    # ------------------------------------------------------------------ helpers
    def utilization(self) -> float:
        return self.placed_volume / self.bin_volume

    def random_valid_action(self) -> int:
        """Pick a uniformly random feasible action index (for env sanity tests)."""
        valid = torch.nonzero(self.action_mask.view(-1), as_tuple=False).flatten()
        return int(valid[self.rng.integers(len(valid))]) if len(valid) else 0
