"""
GOPT — shared config utilities.

Paper: GOPT (Xiong et al., 2024). Holds the model config dataclass + YAML loader.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    """GOPT network configuration. Defaults from §III-D / §IV-A unless marked.

    The two input dims are the heart of the representation (§III-B/C):
      ems_dim = 6  -> each EMS = (FLB vertex xyz, opposite vertex xyz), normalized [0,1].
      item_dim = 3 -> one row of the 2×3 item matrix = (l, w, h) for one orientation.
    """
    ems_dim: int = 6        # §III-B — EMS is a 6-D normalized box vector
    item_dim: int = 3       # §III-C — item row (l,w,h); the 2 orientations form a 2×3 matrix
    d_model: int = 128      # §III-D — "embedding dimensions of both EMS and the item are set to 128"
    n_layers: int = 3       # §III-D — "stacking multiple (three in practice) identical encoder blocks"
    n_heads: int = 8        # [UNSPECIFIED] — head count not stated; 8 is standard for d=128
    d_ff: int = 128         # §III-D — MLP blocks of "{128, 128} neurons"
    dropout: float = 0.0    # [UNSPECIFIED] — not mentioned; default 0
    norm_eps: float = 1e-5  # [UNSPECIFIED] — PyTorch LayerNorm default
    max_ems: int = 80       # §IV-A — "maximum number of EMS is set to 80" (= sequence length N)


def load_config(path: str) -> dict:
    """Load configs/base.yaml into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
