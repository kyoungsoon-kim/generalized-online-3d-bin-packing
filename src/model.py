"""
GOPT — Model Architecture (Packing Transformer + Actor-Critic).

Paper: GOPT: Generalizable Online 3D Bin Packing via Transformer-based DRL
       (Xiong et al., 2024). Official code: https://github.com/Xiong5Heng/GOPT
Implements §III-C (item/EMS representations) and §III-D (network architecture).

  Educational background — why a Transformer here (§III-D):
  The bin is represented as a SEQUENCE of N empty sub-spaces (EMSs) plus the incoming
  item. Packing well means reasoning about (a) how the sub-spaces relate to each other
  and (b) how each sub-space fits THIS item. The authors borrow "cross-modality"
  attention (text↔vision) and treat {EMS set} and {item} as two modalities:
    • SELF-attention within EMSs / within the item rows -> intra-set structure.
    • bi-directional CROSS-attention EMS↔item -> "does this space suit this item?".
  Because the policy reads a size-normalized EMS sequence (not raw bin dimensions), one
  trained network generalizes to bins of different sizes — the paper's headline claim.

  Architecture shape conventions used throughout:
    batch=B, EMS count=N (=max_ems), item rows=2 (the two 0°/90° orientations),
    embedding width=d_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ModelConfig


class MLP(nn.Module):
    """§III-D — "two-layer linear networks with LeakyReLU activation function".

    Used everywhere GOPT needs a small learnable transform: input embeddings, the four
    per-attention feed-forward blocks, and the actor/critic heads. LeakyReLU (vs ReLU)
    keeps a small gradient for negative inputs, avoiding dead units.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU() # [UNSPECIFIED] paper mentions LeakyReLU but not if applied on final layer, assumed yes based on feature extraction usage
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """§III-D — Packing Transformer Encoder Block.
    
    "two self-attention layers, one bi-directional cross-attention layer, 
    and four MLP blocks of two layers comprising {128, 128} neurons.
    Residual connections and layer normalization (Norm) are applied after each layer."
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Self-attention layers
        self.sa_ems = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.sa_item = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        
        # Cross-attention layers (bi-directional = 2 unidirectional)
        self.ca_ems_to_item = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ca_item_to_ems = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        
        # MLPs for each of the 4 attention outputs
        self.mlp_sa_ems = MLP(cfg.d_model, cfg.d_ff, cfg.d_model)
        self.mlp_sa_item = MLP(cfg.d_model, cfg.d_ff, cfg.d_model)
        self.mlp_ca_ems = MLP(cfg.d_model, cfg.d_ff, cfg.d_model)
        self.mlp_ca_item = MLP(cfg.d_model, cfg.d_ff, cfg.d_model)
        
        # LayerNorms
        self.norm_sa_ems = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_sa_item = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_ca_ems = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_ca_item = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        self.norm_mlp_sa_ems = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_mlp_sa_item = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_mlp_ca_ems = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_mlp_ca_item = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, ems: torch.Tensor, item: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ems: (batch, N, d_model)
            item: (batch, 2, d_model)
        Returns:
            updated_ems, updated_item
        """
        # §III-D — "the self-attention layers ... establish the intrinsic connections
        # between EMSs or item dimensions". Each block is post-norm: x <- Norm(x + sublayer(x)),
        # a residual connection (keeps gradients flowing) followed by LayerNorm (stabilizes
        # activation scale). The pattern repeats for attention then its MLP.
        # 1. Self Attention — every EMS attends to every other EMS (and item rows to each other)
        # EMS self-attention
        sa_e, _ = self.sa_ems(ems, ems, ems) # (batch, N, d_model)
        ems = self.norm_sa_ems(ems + sa_e)
        mlp_e = self.mlp_sa_ems(ems)
        ems = self.norm_mlp_sa_ems(ems + mlp_e)
        
        # Item self-attention
        sa_i, _ = self.sa_item(item, item, item) # (batch, 2, d_model)
        item = self.norm_sa_item(item + sa_i)
        mlp_i = self.mlp_sa_item(item)
        item = self.norm_mlp_sa_item(item + mlp_i)
        
        # §III-D — bi-directional cross-attention "facilitates the discovery of
        # inner-relationships from one to another". In nn.MultiheadAttention(query, key,
        # value): the QUERY is updated by looking at the KEY/VALUE set. So
        #   ca_ems_to_item(query=item, key/value=ems) -> item rows enriched by the EMS set
        #   ca_item_to_ems(query=ems,  key/value=item) -> each EMS enriched by the item
        # 2. Bi-directional Cross Attention
        # EMS to Item (Item queries EMS)
        ca_i, _ = self.ca_ems_to_item(item, ems, ems) # (batch, 2, d_model)
        item_out = self.norm_ca_item(item + ca_i)
        mlp_ca_i = self.mlp_ca_item(item_out)
        item_out = self.norm_mlp_ca_item(item_out + mlp_ca_i)
        
        # Item to EMS (EMS queries Item)
        ca_e, _ = self.ca_item_to_ems(ems, item, item) # (batch, N, d_model)
        ems_out = self.norm_ca_ems(ems + ca_e)
        mlp_ca_e = self.mlp_ca_ems(ems_out)
        ems_out = self.norm_mlp_ca_ems(ems_out + mlp_ca_e)
        
        return ems_out, item_out


class PackingTransformer(nn.Module):
    """§III-D — Packing Transformer: "stacking multiple (three in practice) identical encoder blocks"."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        
    def forward(self, ems: torch.Tensor, item: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            ems, item = block(ems, item)
        return ems, item


class GOPTModel(nn.Module):
    """§III-D — Full Architecture: Packing Transformer + Actor/Critic Networks."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings
        # §III-D — "These inputs are then individually processed by Multi-Layer Perceptrons"
        self.ems_emb = MLP(cfg.ems_dim, cfg.d_model, cfg.d_model)
        self.item_emb = MLP(cfg.item_dim, cfg.d_model, cfg.d_model)
        
        self.transformer = PackingTransformer(cfg)
        
        # Actor
        # §III-D — "both the EMS and item features are processed through an MLP"
        self.actor_ems_mlp = MLP(cfg.d_model, cfg.d_model, cfg.d_model)
        self.actor_item_mlp = MLP(cfg.d_model, cfg.d_model, cfg.d_model)
        
        # Critic
        # §III-D — "critic network to estimate the expected cumulative reward"
        # [UNSPECIFIED] how exactly features are pooled for critic. Standard is mean pool.
        self.critic_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LeakyReLU(),
            nn.Linear(cfg.d_model, 1)
        )
        
    def forward(self, ems_features: torch.Tensor, item_features: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            ems_features: (batch, N, 6)
            item_features: (batch, 2, 3) 
            action_mask: (batch, 2, N) - boolean mask, True where valid
        Returns:
            action_logits: (batch, 2*N)
            state_value: (batch, 1)
        """
        batch, N, _ = ems_features.shape
        
        # 1. Embeddings
        ems_emb = self.ems_emb(ems_features)     # (batch, N, d_model)
        item_emb = self.item_emb(item_features)   # (batch, 2, d_model)
        
        # 2. Transformer
        ems_out, item_out = self.transformer(ems_emb, item_emb) # (batch, N, d), (batch, 2, d)
        
        # 3. Actor
        a_ems = self.actor_ems_mlp(ems_out)      # (batch, N, d_model)
        a_item = self.actor_item_mlp(item_out)    # (batch, 2, d_model)
        
        # §III-D — "the results are multiplied to compute a score map of actions".
        # The score of action (orientation o, EMS j) is the dot product of item-row o's
        # feature with EMS j's feature: a compatibility score. Doing this for all (o,j)
        # pairs is a batched outer product -> (B, 2, N). We scale by sqrt(d_model) (as in
        # scaled dot-product attention) to keep logit magnitudes stable.
        score_map = torch.bmm(a_item, a_ems.transpose(1, 2)) / (self.cfg.d_model ** 0.5)  # (B,2,N)

        # §III-D — "element-wise multiplication with the action mask to eliminate
        # infeasible actions". In LOGIT space, masking = set invalid entries to -inf so
        # softmax assigns them probability 0 (§III-C — "ensures the policy samples valid
        # actions"). Flatten (o,j) -> single action index in [0, 2N).
        score_map = score_map.masked_fill(~action_mask, float('-inf'))
        action_logits = score_map.view(batch, -1) # (batch, 2*N)
        
        # 4. Critic — estimates V(s), the expected future reward from this state, used by
        # PPO/GAE to compute advantages. The actor picks per-action scores, but the critic
        # needs ONE vector for the whole state, so we pool the EMS and item sequences.
        # [UNSPECIFIED] §III-D doesn't state the pooling; mean-pool is the standard choice.
        global_ems = ems_out.mean(dim=1)   # (batch, d_model)
        global_item = item_out.mean(dim=1) # (batch, d_model)
        state_rep = torch.cat([global_ems, global_item], dim=-1) # (batch, d_model*2)
        state_value = self.critic_mlp(state_rep) # (batch, 1)
        
        return action_logits, state_value
